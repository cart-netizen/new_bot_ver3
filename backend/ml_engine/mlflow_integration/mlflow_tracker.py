"""
MLflow Tracker - интеграция MLflow для experiment tracking и model registry

Функциональность:
- Automatic experiment tracking
- Parameter and metrics logging
- Model registration and versioning
- Artifact storage (models, plots, configs)
- Integration with existing ModelRegistry
"""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch
import json
import logging

from backend.core.logger import get_logger
from backend.config import settings

logger = get_logger(__name__)

# Suppress verbose MLflow, Alembic, and SQLAlchemy logs
logging.getLogger('mlflow').setLevel(logging.WARNING)
logging.getLogger('alembic').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)


class MLflowTracker:
    """
    MLflow Tracker для экспериментов и моделей

    Интеграция:
    - Tracking Server: PostgreSQL backend
    - Registry: PostgreSQL (shared with tracking)
    - Artifacts: Local filesystem

    ВАЖНО: MLflow использует PostgreSQL как единый backend store для:
    - Experiment tracking (runs, params, metrics)
    - Model Registry (versions, stages, tags)

    PostgreSQL URI берется из config.MLFLOW_TRACKING_URI
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Инициализация MLflow Tracker

        Args:
            tracking_uri: URI MLflow tracking server (default: from config.MLFLOW_TRACKING_URI)
            experiment_name: Название эксперимента (default: from config.MLFLOW_EXPERIMENT_NAME)
            artifact_location: Путь для хранения артефактов (default: from config.MLFLOW_ARTIFACT_LOCATION)
        """
        # Setup tracking URI from config
        if tracking_uri is None:
            tracking_uri = settings.MLFLOW_TRACKING_URI

        if experiment_name is None:
            experiment_name = settings.MLFLOW_EXPERIMENT_NAME

        if artifact_location is None:
            artifact_location = settings.MLFLOW_ARTIFACT_LOCATION

        # Try to use configured tracking URI, fallback to SQLite if PostgreSQL unavailable
        try:
            mlflow.set_tracking_uri(tracking_uri)
            # Test connection by trying to list experiments
            mlflow.search_experiments(max_results=1)
            self.tracking_uri = tracking_uri
            logger.info(f"✓ Connected to MLflow tracking URI: {tracking_uri}")
        except Exception as e:
            logger.warning(
                f"Failed to connect to configured MLflow tracking URI ({tracking_uri}): {e}\n"
                f"Falling back to SQLite..."
            )
            # Fallback to SQLite
            import os
            sqlite_path = os.path.join("data", "mlflow", "mlruns.db")
            os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
            tracking_uri = f"sqlite:///{sqlite_path}"
            mlflow.set_tracking_uri(tracking_uri)
            self.tracking_uri = tracking_uri
            logger.info(f"✓ Using SQLite MLflow tracking: {tracking_uri}")

        # Setup experiment
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=self.artifact_location
            )
            logger.info(f"Created new MLflow experiment: {experiment_name}")
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name}")

        mlflow.set_experiment(experiment_name)

        # MLflow client for advanced operations
        self.client = MlflowClient(tracking_uri=tracking_uri)

        # Current run tracking
        self.current_run_id: Optional[str] = None

        logger.info(
            f"MLflow Tracker initialized: "
            f"uri={tracking_uri}, "
            f"experiment={experiment_name}"
        )

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Начать новый MLflow run

        Args:
            run_name: Название run'а
            tags: Теги для run'а
            description: Описание

        Returns:
            Run ID
        """
        if self.current_run_id:
            logger.warning(f"Run {self.current_run_id} already active, ending it")
            self.end_run()

        # Default tags
        default_tags = {
            "project": "trading_bot",
            "framework": "pytorch",
            "created_at": datetime.now().isoformat()
        }

        if tags:
            default_tags.update(tags)

        # Start run
        run = mlflow.start_run(
            run_name=run_name,
            tags=default_tags,
            description=description
        )

        self.current_run_id = run.info.run_id

        logger.info(f"Started MLflow run: {self.current_run_id} ({run_name})")

        return self.current_run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Логировать параметры

        Args:
            params: Словарь параметров
        """
        if not self.current_run_id:
            logger.warning("No active run, skipping param logging")
            return

        # Flatten nested dicts
        flat_params = self._flatten_dict(params)

        # MLflow has 500 char limit for param values
        for key, value in flat_params.items():
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:497] + "..."
            mlflow.log_param(key, str_value)

        logger.debug(f"Logged {len(flat_params)} parameters")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Логировать метрики

        Args:
            metrics: Словарь метрик
            step: Номер шага (epoch/iteration)
        """
        if not self.current_run_id:
            logger.warning("No active run, skipping metrics logging")
            return

        for key, value in metrics.items():
            try:
                # Handle NaN values - MLflow doesn't accept them directly
                if isinstance(value, float) and (value != value):  # NaN check
                    value = 0.0
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                # Ignore duplicate key errors (can happen with rapid logging)
                if "UniqueViolation" in str(e) or "UNIQUE constraint" in str(e):
                    logger.debug(f"Metric {key} already logged, skipping duplicate")
                else:
                    logger.warning(f"Failed to log metric {key}: {e}")

        logger.debug(f"Logged {len(metrics)} metrics (step={step})")

    def log_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        artifacts: Optional[Dict[str, str]] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None
    ) -> str:
        """
        Логировать PyTorch модель

        Args:
            model: PyTorch модель
            model_name: Название модели
            artifacts: Дополнительные артефакты {name: path}
            signature: MLflow model signature
            input_example: Пример входных данных

        Returns:
            Model URI
        """
        if not self.current_run_id:
            logger.warning("No active run, cannot log model")
            return ""

        # Log model with error handling for duplicate metrics
        model_info = None
        try:
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example
            )
        except Exception as e:
            error_str = str(e)
            # Handle duplicate key errors from MLflow's internal metric logging
            if "UniqueViolation" in error_str or "UNIQUE constraint" in error_str:
                logger.warning(f"Duplicate metric key in MLflow, saving model without metrics: {e}")
                # Try saving model as artifact directly
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path = f"{tmpdir}/{model_name}.pt"
                    torch.save(model.state_dict(), model_path)
                    mlflow.log_artifact(model_path, artifact_path=model_name)
                logger.info(f"Model saved as artifact: {model_name}")
                return f"runs:/{self.current_run_id}/{model_name}"
            else:
                logger.error(f"Failed to log model: {e}")
                raise

        # Log additional artifacts
        if artifacts:
            for name, path in artifacts.items():
                if Path(path).exists():
                    mlflow.log_artifact(path, artifact_path=f"{model_name}/artifacts")
                else:
                    logger.warning(f"Artifact not found: {path}")

        logger.info(f"Logged model: {model_name} (uri={model_info.model_uri})")

        return model_info.model_uri

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Логировать артефакт (файл)

        Args:
            local_path: Путь к локальному файлу
            artifact_path: Путь в MLflow (опционально)
        """
        if not self.current_run_id:
            logger.warning("No active run, skipping artifact logging")
            return

        if not Path(local_path).exists():
            logger.warning(f"Artifact not found: {local_path}")
            return

        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        logger.debug(f"Logged artifact: {local_path}")

    def log_dict(self, dictionary: Dict[str, Any], filename: str) -> None:
        """
        Логировать словарь как JSON артефакт

        Args:
            dictionary: Словарь для логирования
            filename: Имя файла
        """
        if not self.current_run_id:
            logger.warning("No active run, skipping dict logging")
            return

        mlflow.log_dict(dictionary, filename)
        logger.debug(f"Logged dict as: {filename}")

    def log_figure(self, figure: Any, filename: str) -> None:
        """
        Логировать matplotlib/plotly фигуру

        Args:
            figure: Фигура для логирования
            filename: Имя файла
        """
        if not self.current_run_id:
            logger.warning("No active run, skipping figure logging")
            return

        mlflow.log_figure(figure, filename)
        logger.debug(f"Logged figure as: {filename}")

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Зарегистрировать модель в MLflow Model Registry

        Args:
            model_uri: URI модели из log_model()
            model_name: Название модели в registry
            tags: Теги модели
            description: Описание

        Returns:
            Model version
        """
        try:
            # Register model
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )

            version = result.version

            # Add description if provided
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=version,
                    description=description
                )

            logger.info(
                f"Registered model in MLflow Registry: "
                f"{model_name} v{version}"
            )

            return str(version)

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return ""

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ) -> bool:
        """
        Изменить stage модели в registry

        DEPRECATED: MLflow model registry stages are deprecated.
        This method is kept for backward compatibility but does nothing.
        Use our internal ModelRegistry instead.

        Args:
            model_name: Название модели
            version: Версия модели
            stage: Новый stage ("Staging", "Production", "Archived")

        Returns:
            True (always, for backward compatibility)
        """
        # MLflow model registry stages are deprecated since 2.9.0
        # We use our own ModelRegistry instead
        logger.debug(
            f"Skipping MLflow registry stage transition for {model_name} v{version} "
            f"(deprecated feature, using internal ModelRegistry)"
        )
        return True

    def get_model_versions(
        self,
        model_name: str,
        stages: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Получить версии модели из registry

        Args:
            model_name: Название модели
            stages: Фильтр по stages (опционально)

        Returns:
            Список версий
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")

            result = []
            for version in versions:
                if stages and version.current_stage not in stages:
                    continue

                result.append({
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "created_at": version.creation_timestamp,
                    "description": version.description
                })

            return result

        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []

    def load_model(self, model_uri: str) -> torch.nn.Module:
        """
        Загрузить модель из MLflow

        Args:
            model_uri: URI модели (runs:/run_id/model или models:/name/version)

        Returns:
            Загруженная PyTorch модель
        """
        try:
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded model from MLflow: {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def end_run(self, status: str = "FINISHED") -> None:
        """
        Завершить текущий run

        Args:
            status: Статус ("FINISHED", "FAILED", "KILLED")
        """
        if not self.current_run_id:
            logger.warning("No active run to end")
            return

        mlflow.end_run(status=status)
        logger.info(f"Ended MLflow run: {self.current_run_id} ({status})")
        self.current_run_id = None

    def search_runs(
        self,
        filter_string: Optional[str] = None,
        order_by: Optional[List[str]] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Поиск runs по фильтру

        Args:
            filter_string: Фильтр MLflow (например: "metrics.accuracy > 0.9")
            order_by: Сортировка (например: ["metrics.accuracy DESC"])
            max_results: Максимум результатов

        Returns:
            Список runs
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                order_by=order_by,
                max_results=max_results
            )

            return runs.to_dict('records')

        except Exception as e:
            logger.debug(f"Failed to search runs (MLflow may not be configured): {e}")
            return []

    def get_best_run(
        self,
        metric: str = "val_accuracy",
        order: str = "DESC"
    ) -> Optional[Dict[str, Any]]:
        """
        Получить лучший run по метрике

        Args:
            metric: Название метрики
            order: Порядок сортировки ("DESC" или "ASC")

        Returns:
            Лучший run или None
        """
        runs = self.search_runs(
            order_by=[f"metrics.{metric} {order}"],
            max_results=1
        )

        return runs[0] if runs else None

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '.'
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary

        Args:
            d: Словарь для flatten
            parent_key: Родительский ключ
            sep: Разделитель

        Returns:
            Flattened словарь
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))

        return dict(items)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.current_run_id:
            status = "FAILED" if exc_type else "FINISHED"
            self.end_run(status=status)


# Singleton instance
_mlflow_tracker_instance: Optional[MLflowTracker] = None


def get_mlflow_tracker(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "trading_bot_ml"
) -> MLflowTracker:
    """
    Получить singleton instance MLflow Tracker

    Args:
        tracking_uri: URI tracking server
        experiment_name: Название эксперимента

    Returns:
        MLflow Tracker instance
    """
    global _mlflow_tracker_instance

    if _mlflow_tracker_instance is None:
        _mlflow_tracker_instance = MLflowTracker(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name
        )

    return _mlflow_tracker_instance
