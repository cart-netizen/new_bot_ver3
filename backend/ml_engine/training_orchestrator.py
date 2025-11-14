"""
Training Orchestrator - главный скрипт для обучения моделей

Объединяет:
- MLflow Integration (tracking)
- Feature Store (data management)
- Model Training (existing trainer)
- Model Registry (versioning)
- Auto-Retraining Pipeline

Provides простой интерфейс: одна команда → обученная модель
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict, is_dataclass
import torch
import pandas as pd
import numpy as np

from backend.core.logger import get_logger
from backend.ml_engine.models.hybrid_cnn_lstm import HybridCNNLSTM, ModelConfig
from backend.ml_engine.training.model_trainer import ModelTrainer, TrainerConfig
from backend.ml_engine.training.data_loader import HistoricalDataLoader, DataConfig, TradingDataset
from backend.ml_engine.mlflow_integration.mlflow_tracker import get_mlflow_tracker
from backend.ml_engine.feature_store.feature_store import get_feature_store, FeatureMetadata
from backend.ml_engine.inference.model_registry import get_model_registry, ModelStage
from backend.ml_engine.optimization.onnx_optimizer import get_onnx_optimizer

logger = get_logger(__name__)


def config_to_dict(obj: Any) -> Dict:
    """
    Convert config object to JSON-serializable dict.

    Handles dataclasses recursively and filters out private attributes.

    Args:
        obj: Config object (dataclass or regular object)

    Returns:
        JSON-serializable dictionary
    """
    if is_dataclass(obj):
        # Use asdict for dataclasses
        return asdict(obj)
    elif isinstance(obj, dict):
        # Already a dict, process values recursively
        return {k: config_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        # Regular object with __dict__, convert to dict
        return {
            k: config_to_dict(v)
            for k, v in vars(obj).items()
            if not k.startswith('_')
        }
    else:
        # Primitive type, return as is
        return obj


class TrainingOrchestrator:
    """
    Orchestrator для полного цикла обучения модели

    Workflow:
    1. Load data from Feature Store (or legacy loader)
    2. Initialize model & trainer
    3. Train with MLflow tracking
    4. Save checkpoints
    5. Register model in Model Registry
    6. Export to ONNX (optional)
    7. Auto-promote to production (if criteria met)
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        trainer_config: Optional[TrainerConfig] = None,
        data_config: Optional[DataConfig] = None
    ):
        """
        Инициализация orchestrator

        Args:
            model_config: Конфигурация модели
            trainer_config: Конфигурация training
            data_config: Конфигурация данных
        """
        self.model_config = model_config or ModelConfig()
        self.trainer_config = trainer_config or TrainerConfig()
        self.data_config = data_config or DataConfig()

        # Components
        self.mlflow_tracker = get_mlflow_tracker(experiment_name="model_training")
        self.feature_store = get_feature_store()
        self.model_registry = get_model_registry()
        self.onnx_optimizer = get_onnx_optimizer()

        logger.info("Training Orchestrator initialized")

    async def train_model(
        self,
        model_name: str = "hybrid_cnn_lstm",
        run_name: Optional[str] = None,
        export_onnx: bool = True,
        auto_promote: bool = True,
        min_accuracy_for_promotion: float = 0.80
    ) -> Dict[str, Any]:
        """
        Полный цикл обучения модели

        Args:
            model_name: Название модели
            run_name: Название MLflow run
            export_onnx: Экспортировать в ONNX
            auto_promote: Автоматически продвинуть в production
            min_accuracy_for_promotion: Минимальная точность для promotion

        Returns:
            Результаты обучения
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or f"{model_name}_training_{timestamp}"

        logger.info(f"Starting training: {run_name}")

        # Start MLflow run
        self.mlflow_tracker.start_run(
            run_name=run_name,
            tags={
                "model_type": model_name,
                "training_mode": "manual"
            },
            description=f"Training {model_name} model"
        )

        try:
            # Step 1: Load data
            logger.info("Step 1/7: Loading training data...")
            train_loader, val_loader, test_loader = await self._load_data()

            if train_loader is None:
                raise ValueError("Failed to load training data")

            # Log data info
            train_size = self._get_dataset_size(train_loader)
            val_size = self._get_dataset_size(val_loader)
            test_size = self._get_dataset_size(test_loader)

            # Determine data source
            data_source = "feature_store" if self.data_config.use_feature_store else "legacy"

            self.mlflow_tracker.log_params({
                "data_source": data_source,
                "data_date_range_days": self.data_config.feature_store_date_range_days,
                "data_feature_store_enabled": self.data_config.use_feature_store,
                "train_samples": train_size,
                "val_samples": val_size,
                "test_samples": test_size
            })

            # Step 2: Initialize model
            logger.info("Step 2/7: Initializing model...")
            model = HybridCNNLSTM(self.model_config)

            # Log model config
            model_params = {
                f"model_{k}": v for k, v in vars(self.model_config).items()
            }
            self.mlflow_tracker.log_params(model_params)

            # Step 3: Initialize trainer
            logger.info("Step 3/7: Initializing trainer...")
            trainer = ModelTrainer(model, self.trainer_config)

            # Log trainer config
            trainer_params = {
                f"trainer_{k}": v for k, v in vars(self.trainer_config).items()
                if not k.startswith('_')
            }
            self.mlflow_tracker.log_params(trainer_params)

            # Step 4: Train model
            logger.info("Step 4/7: Training model...")
            training_history = trainer.train(
                train_loader,
                val_loader
            )

            # Extract final training metrics from history
            # training_history is a list of dicts with metrics for each epoch
            if training_history:
                final_epoch = training_history[-1]
                final_metrics = {
                    "final_train_loss": final_epoch.get("train_loss", 0),
                    "final_val_loss": final_epoch.get("val_loss", 0),
                    "final_train_accuracy": final_epoch.get("train_acc", 0),
                    "final_val_accuracy": final_epoch.get("val_acc", 0),
                    "best_val_accuracy": max([m.get("val_acc", 0) for m in training_history]),
                    "total_epochs": len(training_history)
                }
                self.mlflow_tracker.log_metrics(final_metrics)
            else:
                final_metrics = {}
                logger.warning("Training history is empty")

            # Step 5: Evaluate on test set
            logger.info("Step 5/7: Evaluating model...")
            test_metrics = {}
            if test_loader:
                test_metrics = await self._evaluate_model(model, test_loader)
                self.mlflow_tracker.log_metrics({
                    f"test_{k}": v for k, v in test_metrics.items()
                })

            # Step 6: Save and register model
            logger.info("Step 6/7: Saving and registering model...")

            # Save model locally
            model_dir = Path(self.trainer_config.checkpoint_dir) / timestamp
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / f"{model_name}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': config_to_dict(self.model_config),
                'training_history': training_history,
                'final_metrics': final_metrics,
                'test_metrics': test_metrics,
                'timestamp': timestamp
            }, model_path)

            # Save metadata (CRITICAL FIX: use config_to_dict for dataclass serialization)
            metadata = {
                'model_name': model_name,
                'version': timestamp,
                'model_config': config_to_dict(self.model_config),
                'trainer_config': config_to_dict(self.trainer_config),
                'training_history': training_history,
                'final_metrics': final_metrics,
                'test_metrics': test_metrics,
                'created_at': datetime.now().isoformat()
            }

            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Log model to MLflow
            model_uri = self.mlflow_tracker.log_model(
                model=model,
                model_name=model_name,
                artifacts={
                    "metadata": str(metadata_path)
                }
            )

            # Register in Model Registry
            model_info = await self.model_registry.register_model(
                name=model_name,
                version=timestamp,
                model_path=model_path,
                model_type="HybridCNNLSTM",
                description=f"Trained model - {run_name}",
                training_params={
                    **vars(self.model_config),
                    **final_metrics,
                    **test_metrics
                }
            )

            # Register in MLflow Registry
            mlflow_version = self.mlflow_tracker.register_model(
                model_uri=model_uri,
                model_name=model_name,
                description=f"Trained model - {run_name}"
            )

            # Step 7: Export to ONNX (optional)
            onnx_path = None
            if export_onnx:
                logger.info("Step 7/7: Exporting to ONNX...")
                onnx_path = await self._export_to_onnx(model, model_path, model_dir)

                if onnx_path:
                    self.mlflow_tracker.log_artifact(str(onnx_path), artifact_path="onnx")

            # Auto-promotion to production
            promoted = False
            test_accuracy = test_metrics.get('accuracy', 0)

            if auto_promote and test_accuracy >= min_accuracy_for_promotion:
                logger.info("Auto-promoting model to production...")
                promoted = await self.model_registry.promote_to_production(
                    model_name, timestamp
                )

                if promoted:
                    # Also promote in MLflow
                    self.mlflow_tracker.transition_model_stage(
                        model_name=model_name,
                        version=mlflow_version,
                        stage="Production"
                    )

            # End MLflow run
            self.mlflow_tracker.end_run(status="FINISHED")

            # Return results
            result = {
                "success": True,
                "model_name": model_name,
                "version": timestamp,
                "mlflow_version": mlflow_version,
                "model_path": str(model_path),
                "onnx_path": str(onnx_path) if onnx_path else None,
                "promoted_to_production": promoted,
                "final_metrics": final_metrics,
                "test_metrics": test_metrics,
                "run_name": run_name,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"Training completed successfully: "
                f"version={timestamp}, accuracy={test_accuracy:.3f}, promoted={promoted}"
            )

            return result

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            self.mlflow_tracker.end_run(status="FAILED")

            return {
                "success": False,
                "error": str(e),
                "run_name": run_name
            }

    async def _load_data(self):
        """
        Загрузить данные для обучения

        Workflow:
        1. Check if Feature Store enabled
        2. Try to load from Feature Store
           - Define date range
           - Read offline features
           - Validate schema
           - Convert to DataLoaders
        3. Fallback to legacy loader if:
           - Feature Store disabled
           - No data in Feature Store
           - Schema validation fails
           - Any exception occurs
        4. Return DataLoaders

        Returns:
            Tuple of (train_loader, val_loader, test_loader) or (None, None, None)
        """
        try:
            # Step 1: Check configuration
            if not self.data_config.use_feature_store:
                logger.info("Feature Store disabled by config, using legacy loader")
                return self._load_from_legacy()

            # Step 2: Try Feature Store
            logger.info("Attempting to load from Feature Store...")

            # 2.1: Define date range
            end_date = datetime.now()
            start_date = end_date - timedelta(
                days=self.data_config.feature_store_date_range_days
            )

            logger.info(
                f"Date range: {start_date.strftime('%Y-%m-%d')} to "
                f"{end_date.strftime('%Y-%m-%d')} "
                f"({self.data_config.feature_store_date_range_days} days)"
            )

            # 2.2: Read from Feature Store
            features_df = self.feature_store.read_offline_features(
                feature_group=self.data_config.feature_store_group,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )

            # 2.3: Check if data exists
            if features_df.empty:
                logger.warning(
                    f"No features found in Feature Store for date range "
                    f"{start_date.date()} to {end_date.date()}"
                )
                if self.data_config.fallback_to_legacy:
                    logger.info("Falling back to legacy data loader")
                    return self._load_from_legacy()
                else:
                    logger.error("Fallback disabled, no data available")
                    return None, None, None

            logger.info(f"✓ Collected {len(features_df)} samples from Feature Store")

            # 2.4: Validate schema
            from backend.ml_engine.feature_store.feature_schema import DEFAULT_SCHEMA

            logger.info("Validating DataFrame schema...")
            try:
                DEFAULT_SCHEMA.validate_dataframe(features_df, strict=False)
                logger.info("✓ Schema validation passed")
            except ValueError as e:
                logger.error(f"Schema validation failed: {e}")
                if self.data_config.fallback_to_legacy:
                    logger.info("Falling back to legacy data loader")
                    return self._load_from_legacy()
                else:
                    raise

            # 2.5: Convert to DataLoaders
            logger.info("Converting Feature Store data to DataLoaders...")

            data_loader = HistoricalDataLoader(self.data_config)

            train_loader, val_loader, test_loader = data_loader.load_from_dataframe(
                features_df=features_df,
                feature_columns=DEFAULT_SCHEMA.get_all_feature_columns(),
                label_column=DEFAULT_SCHEMA.label_column,
                timestamp_column=DEFAULT_SCHEMA.timestamp_column,
                symbol_column=DEFAULT_SCHEMA.symbol_column,
                apply_resampling=True  # Enable class balancing
            )

            # 2.6: Log success
            logger.info("✓ Successfully loaded data from Feature Store")
            logger.info(f"  • Date range: {self.data_config.feature_store_date_range_days} days")
            logger.info(f"  • Samples: {len(features_df)}")
            logger.info(f"  • Features: {len(DEFAULT_SCHEMA.get_all_feature_columns())}")
            logger.info(f"  • Class balancing: enabled")

            # Get sizes for logging
            train_size = self._get_dataset_size(train_loader)
            val_size = self._get_dataset_size(val_loader)
            test_size = self._get_dataset_size(test_loader)

            logger.info(
                f"  • Split: train={train_size}, val={val_size}, test={test_size}"
            )

            return train_loader, val_loader, test_loader

        except Exception as e:
            logger.error(f"Feature Store loading failed: {e}")
            logger.exception("Exception details:")

            if self.data_config.fallback_to_legacy:
                logger.info("Falling back to legacy data loader")
                return self._load_from_legacy()
            else:
                logger.error("Fallback disabled, re-raising exception")
                raise

    def _get_dataset_size(self, dataloader) -> int:
        """
        Safely get dataset size from DataLoader

        Args:
            dataloader: DataLoader instance or None

        Returns:
            Dataset size or 0 if unavailable
        """
        if dataloader is None:
            return 0
        try:
            return len(dataloader.dataset)  # type: ignore[arg-type]
        except (AttributeError, TypeError):
            return 0

    def _load_from_legacy(self) -> tuple:
        """
        Load data from legacy .npy files

        Returns:
            Tuple of (train_loader, val_loader, test_loader) or (None, None, None)
        """
        logger.info("Loading data from legacy .npy files...")

        data_loader = HistoricalDataLoader(self.data_config)
        train_data, val_data, test_data = data_loader.load_and_split()

        if train_data is None:
            logger.error("No training data available in legacy storage")
            return None, None, None

        # Get sizes
        train_size = self._get_dataset_size(train_data)
        val_size = self._get_dataset_size(val_data)
        test_size = self._get_dataset_size(test_data)

        logger.info(
            f"✓ Data loaded from legacy files: "
            f"train={train_size}, val={val_size}, test={test_size}"
        )

        return train_data, val_data, test_data

    async def _evaluate_model(
        self,
        model: torch.nn.Module,
        test_loader
    ) -> Dict[str, float]:
        """Оценить модель на test set"""
        try:
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for batch in test_loader:
                    sequences = batch['sequence'].to(device)
                    labels = batch['label'].to(device)

                    outputs = model(sequences)

                    # CRITICAL FIX: Handle dict output from model
                    if isinstance(outputs, dict):
                        # Model returns dict with 'direction_logits' key
                        direction_logits = outputs['direction_logits']
                        predictions = torch.argmax(direction_logits, dim=1)
                    elif isinstance(outputs, tuple):
                        # Model returns tuple (direction_logits, ...)
                        direction_logits = outputs[0]
                        predictions = torch.argmax(direction_logits, dim=1)
                    else:
                        # Model returns tensor directly
                        predictions = torch.argmax(outputs, dim=1)

                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support

            accuracy = accuracy_score(all_labels, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
            )

            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }

            logger.info(f"Test metrics: {metrics}")

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}

    async def _export_to_onnx(
        self,
        model: torch.nn.Module,
        model_path: Path,
        output_dir: Path
    ) -> Optional[Path]:
        """Экспортировать модель в ONNX"""
        try:
            # Input shape: (batch_size, sequence_length, features)
            input_shape = (
                1,
                self.model_config.sequence_length,
                self.model_config.input_features
            )

            onnx_path = output_dir / "model.onnx"

            success = await self.onnx_optimizer.export_to_onnx(
                model=model,
                model_path=model_path,
                output_path=onnx_path,
                input_shape=input_shape
            )

            if success:
                logger.info(f"ONNX export successful: {onnx_path}")
                return onnx_path
            else:
                logger.warning("ONNX export failed")
                return None

        except Exception as e:
            logger.error(f"ONNX export error: {e}")
            return None

    async def quick_train(
        self,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """
        Быстрое обучение с дефолтными параметрами

        Args:
            epochs: Количество эпох
            batch_size: Размер батча
            learning_rate: Learning rate

        Returns:
            Результаты обучения
        """
        # Update configs
        self.trainer_config.epochs = epochs
        self.trainer_config.learning_rate = learning_rate
        self.data_config.batch_size = batch_size

        # Train
        return await self.train_model(
            model_name="hybrid_cnn_lstm",
            export_onnx=True,
            auto_promote=True
        )


# Singleton instance
_training_orchestrator_instance: Optional[TrainingOrchestrator] = None


def get_training_orchestrator(
    model_config: Optional[ModelConfig] = None,
    trainer_config: Optional[TrainerConfig] = None,
    data_config: Optional[DataConfig] = None
) -> TrainingOrchestrator:
    """
    Получить singleton instance Training Orchestrator

    Args:
        model_config: Конфигурация модели
        trainer_config: Конфигурация training
        data_config: Конфигурация данных

    Returns:
        TrainingOrchestrator instance
    """
    global _training_orchestrator_instance

    if _training_orchestrator_instance is None:
        _training_orchestrator_instance = TrainingOrchestrator(
            model_config=model_config,
            trainer_config=trainer_config,
            data_config=data_config
        )

    return _training_orchestrator_instance


# CLI interface
async def main():
    """CLI interface для запуска обучения"""
    import argparse

    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")
    parser.add_argument("--no-promote", action="store_true", help="Skip auto-promotion")

    # Feature Store parameters
    parser.add_argument("--no-feature-store", action="store_true",
                       help="Skip Feature Store, use legacy .npy loader")
    parser.add_argument("--date-range", type=int, default=90,
                       help="Date range in days for Feature Store (default: 90)")
    parser.add_argument("--no-fallback", action="store_true",
                       help="Disable fallback to legacy loader if Feature Store fails")

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = TrainingOrchestrator()

    # Update configs
    orchestrator.trainer_config.epochs = args.epochs
    orchestrator.trainer_config.learning_rate = args.lr
    orchestrator.data_config.batch_size = args.batch_size

    # Feature Store configs
    orchestrator.data_config.use_feature_store = not args.no_feature_store
    orchestrator.data_config.feature_store_date_range_days = args.date_range
    orchestrator.data_config.fallback_to_legacy = not args.no_fallback

    # Train
    result = await orchestrator.train_model(
        export_onnx=not args.no_onnx,
        auto_promote=not args.no_promote
    )

    # Print results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
