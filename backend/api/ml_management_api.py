"""
ML Management API - REST API для управления ML моделями через фронтенд

Endpoints:
- POST /api/ml-management/train - Start training
- GET /api/ml-management/training/status - Training status
- GET /api/ml-management/models - List models
- POST /api/ml-management/models/{name}/{version}/promote - Promote model
- GET /api/ml-management/mlflow/runs - MLflow runs
- POST /api/ml-management/retraining/start - Start auto-retraining
- POST /api/ml-management/retraining/stop - Stop auto-retraining
- GET /api/ml-management/retraining/status - Retraining status
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import asyncio

from backend.core.logger import get_logger
from backend.ml_engine.training_orchestrator import get_training_orchestrator
from backend.ml_engine.models.hybrid_cnn_lstm import ModelConfig
from backend.ml_engine.training.model_trainer import TrainerConfig
from backend.ml_engine.training.data_loader import DataConfig
from backend.ml_engine.inference.model_registry import get_model_registry, ModelStage
from backend.ml_engine.mlflow_integration.mlflow_tracker import get_mlflow_tracker
from backend.ml_engine.auto_retraining.retraining_pipeline import (
    get_retraining_pipeline,
    RetrainingConfig,
    RetrainingTrigger
)
def _to_dict(obj: Any) -> Dict[str, Any]:
    """
    Безопасно конвертирует объект в словарь.
    Работает с dict, Pydantic v1/v2 моделями и dataclasses.
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'model_dump'):  # Pydantic v2
        return obj.model_dump()
    if hasattr(obj, 'dict'):        # Pydantic v1
        return obj.dict()
    return vars(obj)                # dataclass или обычный объект


def _convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas types to Python native types for JSON serialization.

    This fixes TypeError: 'float' object cannot be interpreted as an integer
    when FastAPI tries to serialize MLflow runs that contain numpy types.
    """
    import numpy as np
    import pandas as pd

    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/ml-management", tags=["ML Management"])


# ============================================================
# Request/Response Models
# ============================================================

class TrainingRequest(BaseModel):
    """Request для запуска обучения"""
    model_name: str = Field(default="hybrid_cnn_lstm", description="Model name")
    epochs: int = Field(default=50, ge=1, le=500, description="Number of epochs")
    batch_size: int = Field(default=64, ge=8, le=512, description="Batch size")
    learning_rate: float = Field(default=0.001, gt=0, lt=1, description="Learning rate")
    early_stopping_patience: int = Field(default=20, ge=0, le=100, description="Early stopping patience (0 to disable)")
    export_onnx: bool = Field(default=True, description="Export to ONNX")
    auto_promote: bool = Field(default=True, description="Auto-promote to production")
    min_accuracy: float = Field(default=0.80, ge=0, le=1, description="Min accuracy for promotion")

    # Data source configuration
    data_source: str = Field(default="feature_store", description="Data source: 'feature_store' or 'legacy'")
    data_path: Optional[str] = Field(default=None, description="Custom data path (optional)")

    # Optional advanced configs
    ml_model_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Model configuration (replaces deprecated 'model_config')"
    )
    trainer_config: Optional[Dict[str, Any]] = None
    data_config: Optional[Dict[str, Any]] = None


class TrainingStatusResponse(BaseModel):
    """Response статуса обучения"""
    is_training: bool
    current_job: Optional[Dict[str, Any]] = None
    last_completed: Optional[Dict[str, Any]] = None


class ModelListResponse(BaseModel):
    """Response списка моделей"""
    models: List[Dict[str, Any]]
    total: int


class RetrainingStatusResponse(BaseModel):
    """Response статуса auto-retraining"""
    is_running: bool
    config: Dict[str, Any]
    last_training_time: Optional[str] = None
    last_drift_check_time: Optional[str] = None
    last_performance_check_time: Optional[str] = None


# ============================================================
# Global State
# ============================================================

# Training job tracking
current_training_job: Optional[Dict[str, Any]] = None
training_history: List[Dict[str, Any]] = []


# ============================================================
# Training Endpoints
# ============================================================

@router.post("/train")
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Запустить обучение модели

    Args:
        request: Параметры обучения
        background_tasks: FastAPI background tasks

    Returns:
        Job ID и статус
    """
    global current_training_job

    # Check if already training
    if current_training_job and current_training_job.get("status") == "running":
        raise HTTPException(
            status_code=409,
            detail="Training already in progress"
        )

    # Create job
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    current_training_job = {
        "job_id": job_id,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "params": request.model_dump(),
        "progress": {
            "current_epoch": 0,
            "total_epochs": request.epochs,
            "current_loss": 0,
            "best_val_accuracy": 0
        }
    }

    # Start training in background
    background_tasks.add_task(
        _run_training_job,
        job_id=job_id,
        request=request
    )

    logger.info(f"Training job started: {job_id}")

    return {
        "job_id": job_id,
        "status": "started",
        "message": "Training started in background",
        "started_at": current_training_job["started_at"]
    }


@router.get("/training/status")
async def get_training_status() -> TrainingStatusResponse:
    """
    Получить статус текущего обучения

    Returns:
        Статус обучения
    """
    is_training = (
        current_training_job is not None and
        current_training_job.get("status") == "running"
    )

    last_completed = None
    if training_history:
        last_completed = training_history[-1]

    return TrainingStatusResponse(
        is_training=is_training,
        current_job=current_training_job,
        last_completed=last_completed
    )


async def _run_training_job(job_id: str, request: TrainingRequest):
    """
    Background task для обучения модели

    Args:
        job_id: ID job'а
        request: Параметры обучения
    """
    global current_training_job

    try:
        logger.info(f"Starting training job: {job_id}")

        # Create configs
        model_config = ModelConfig(**request.ml_model_config) if request.ml_model_config else ModelConfig()
        trainer_config = TrainerConfig(
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            early_stopping_patience=request.early_stopping_patience
        )
        if request.trainer_config:
            # Override with custom trainer config
            for k, v in _to_dict(request.trainer_config).items():
                setattr(trainer_config, k, v)

        # Configure data source path
        storage_path = request.data_path
        if not storage_path:
            # Use default paths based on data source
            if request.data_source == "legacy":
                storage_path = "data/ml_training"
            else:  # feature_store
                storage_path = "data/feature_store"

        logger.info(f"Training data source: {request.data_source}, path: {storage_path}")

        data_config = DataConfig(
            batch_size=request.batch_size,
            storage_path=storage_path
        )
        if request.data_config:
            # Override with custom data config
            for k, v in _to_dict(request.data_config).items():
                setattr(data_config, k, v)

        # Create orchestrator
        orchestrator = get_training_orchestrator(
            model_config=model_config,
            trainer_config=trainer_config,
            data_config=data_config
        )

        # Train
        result = await orchestrator.train_model(
            model_name=request.model_name,
            export_onnx=request.export_onnx,
            auto_promote=request.auto_promote,
            min_accuracy_for_promotion=request.min_accuracy
        )

        # Update job status
        current_training_job.update({
            "status": "completed" if result["success"] else "failed",
            "completed_at": datetime.now().isoformat(),
            "result": result
        })

        # Add to history
        training_history.append(current_training_job.copy())

        # Keep only last 10 jobs in history
        if len(training_history) > 10:
            training_history.pop(0)

        logger.info(
            f"Training job completed: {job_id}, "
            f"success={result['success']}"
        )

    except Exception as e:
        logger.error(f"Training job failed: {job_id}, error={e}", exc_info=True)

        # Provide user-friendly error messages
        error_message = str(e)
        user_friendly_message = error_message

        if "Failed to load training data" in error_message:
            user_friendly_message = (
                "No training data available. Please collect data first:\n"
                "1. Start the bot to collect live data, or\n"
                "2. Run the data collector: python -m backend.ml_engine.data_collection.data_collector\n"
                "See data/README.md for details."
            )
        elif "Insufficient data" in error_message:
            user_friendly_message = (
                f"{error_message}\n"
                "Let the bot run longer to collect more samples. "
                "Recommended: 10,000+ samples for good performance."
            )
        elif "No such file or directory" in error_message or "not found" in error_message.lower():
            user_friendly_message = (
                "Training data directory not found. Please collect data first. "
                "See data/README.md for instructions."
            )

        current_training_job.update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": user_friendly_message,
            "raw_error": error_message
        })

        training_history.append(current_training_job.copy())


# ============================================================
# Model Management Endpoints
# ============================================================

@router.get("/models")
async def list_models(
    stage: Optional[str] = None
) -> ModelListResponse:
    """
    Список всех моделей

    Args:
        stage: Фильтр по stage (production, staging, etc.)

    Returns:
        Список моделей
    """
    try:
        registry = get_model_registry()

        # Get all models
        models = []
        for name in ["hybrid_cnn_lstm"]:  # Add more model names as needed
            try:
                # list_models returns List[ModelInfo]
                model_infos = await registry.list_models(name)

                for model_info in model_infos:
                    metadata = model_info.metadata
                    # Convert stage to lowercase for frontend
                    stage_value = metadata.stage.value if hasattr(metadata.stage, "value") else str(metadata.stage)
                    stage_lower = stage_value.lower()

                    model_data = {
                        "name": metadata.name,
                        "version": metadata.version,
                        "stage": stage_lower,
                        "created_at": metadata.created_at.isoformat() if hasattr(metadata.created_at, "isoformat") else str(metadata.created_at),
                        "description": metadata.description or "",
                        "metrics": metadata.metrics or {}
                    }

                    # Filter by stage if specified
                    stage_value = metadata.stage.value if hasattr(metadata.stage, "value") else str(metadata.stage)
                    if stage is None or stage_value.lower() == stage.lower():
                        models.append(model_data)
            except Exception as e:
                logger.warning(f"Failed to get versions for model {name}: {e}")
                # Continue to next model

        return ModelListResponse(
            models=models,
            total=len(models)
        )

    except Exception as e:
        logger.error(f"Failed to list models: {e}", exc_info=True)
        # Return empty list instead of 500 error
        return ModelListResponse(models=[], total=0)


@router.post("/models/{name}/{version}/promote")
async def promote_model(
    name: str,
    version: str,
    stage: str = "production"
) -> Dict[str, Any]:
    """
    Продвинуть модель к указанному stage

    Args:
        name: Название модели
        version: Версия модели
        stage: Целевой stage

    Returns:
        Результат операции
    """
    try:
        registry = get_model_registry()

        # Map stage string to ModelStage enum
        stage_map = {
            "production": ModelStage.PRODUCTION,
            "staging": ModelStage.STAGING,
            "archived": ModelStage.ARCHIVED
        }

        target_stage = stage_map.get(stage.lower())
        if not target_stage:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stage: {stage}"
            )

        # Promote
        if target_stage == ModelStage.PRODUCTION:
            success = await registry.promote_to_production(name, version)
        else:
            success = await registry.set_model_stage(name, version, target_stage)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {name} v{version}"
            )

        # Also promote in MLflow if available
        try:
            mlflow_tracker = get_mlflow_tracker()
            mlflow_tracker.transition_model_stage(
                model_name=name,
                version=version,
                stage=stage.capitalize()
            )
        except:
            pass  # MLflow promotion is optional

        return {
            "success": True,
            "model_name": name,
            "version": version,
            "new_stage": stage
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{name}/{version}/download")
async def download_model(name: str, version: str):
    """
    Скачать файл модели (.h5)

    Args:
        name: Название модели
        version: Версия модели

    Returns:
        FileResponse с файлом модели
    """
    try:
        registry = get_model_registry()
        model_info = await registry.get_model(name, version)

        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {name} v{version}"
            )

        # Path to model file
        model_file = Path(model_info.model_path)

        if not model_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {model_file}"
            )

        # Return file for download
        return FileResponse(
            path=str(model_file),
            filename=f"{name}_{version}.h5",
            media_type="application/octet-stream"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# MLflow Integration Endpoints
# ============================================================

@router.get("/mlflow/runs")
async def get_mlflow_runs(
    limit: int = 10
) -> Dict[str, Any]:
    """
    Получить список MLflow runs

    Args:
        limit: Максимум результатов

    Returns:
        Список runs
    """
    try:
        mlflow_tracker = get_mlflow_tracker()

        runs = mlflow_tracker.search_runs(
            order_by=["start_time DESC"],
            max_results=limit
        )

        # Convert numpy types to Python types for JSON serialization
        return {
            "runs": _convert_numpy_types(runs),
            "total": len(runs)
        }

    except Exception as e:
        logger.debug(f"Failed to get MLflow runs (MLflow may not be configured): {e}")
        # Return empty list instead of error when MLflow is not available
        return {"runs": [], "total": 0}


@router.get("/mlflow/best-run")
async def get_best_run(
    metric: str = "val_accuracy"
) -> Dict[str, Any]:
    """
    Получить лучший run по метрике

    Args:
        metric: Название метрики

    Returns:
        Лучший run
    """
    try:
        mlflow_tracker = get_mlflow_tracker()

        best_run = mlflow_tracker.get_best_run(metric=metric, order="DESC")

        if not best_run:
            # Return None instead of 404 error when no runs exist
            return {"run": None, "message": "No runs found"}

        # Convert numpy types to Python types for JSON serialization
        return _convert_numpy_types(best_run)

    except HTTPException:
        raise
    except Exception as e:
        logger.debug(f"Failed to get best run (MLflow may not be configured): {e}")
        # Return None instead of error when MLflow is not available
        return {"run": None, "message": "MLflow not available"}


# ============================================================
# Auto-Retraining Endpoints
# ============================================================

@router.post("/retraining/start")
async def start_retraining_pipeline(
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Запустить auto-retraining pipeline

    Args:
        config: Конфигурация (опционально)

    Returns:
        Статус запуска
    """
    try:
        # Create config
        if config:
            retraining_config = RetrainingConfig(**config)
        else:
            retraining_config = RetrainingConfig()

        # Get pipeline
        pipeline = get_retraining_pipeline(config=retraining_config)

        # Start
        await pipeline.start()

        return {
            "success": True,
            "message": "Auto-retraining pipeline started",
            "config": vars(retraining_config)
        }

    except Exception as e:
        logger.error(f"Failed to start retraining pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retraining/stop")
async def stop_retraining_pipeline() -> Dict[str, Any]:
    """
    Остановить auto-retraining pipeline

    Returns:
        Статус остановки
    """
    try:
        pipeline = get_retraining_pipeline()
        await pipeline.stop()

        return {
            "success": True,
            "message": "Auto-retraining pipeline stopped"
        }

    except Exception as e:
        logger.error(f"Failed to stop retraining pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retraining/status")
async def get_retraining_status() -> RetrainingStatusResponse:
    """
    Получить статус auto-retraining pipeline

    Returns:
        Статус pipeline
    """
    try:
        pipeline = get_retraining_pipeline()

        # Safely convert config to dict
        try:
            from dataclasses import asdict, is_dataclass
            if is_dataclass(pipeline.config):
                config_dict = asdict(pipeline.config)
            else:
                config_dict = vars(pipeline.config) if hasattr(pipeline.config, '__dict__') else {}
        except Exception as e:
            logger.warning(f"Failed to convert config to dict: {e}")
            config_dict = {}

        return RetrainingStatusResponse(
            is_running=pipeline.is_running,
            config=config_dict,
            last_training_time=pipeline.last_training_time.isoformat() if pipeline.last_training_time else None,
            last_drift_check_time=pipeline.last_drift_check_time.isoformat() if pipeline.last_drift_check_time else None,
            last_performance_check_time=pipeline.last_performance_check_time.isoformat() if pipeline.last_performance_check_time else None
        )

    except Exception as e:
        logger.debug(f"Failed to get retraining status (pipeline may not be initialized): {e}")
        # Return default status instead of error when pipeline is not available
        return RetrainingStatusResponse(
            is_running=False,
            config={},
            last_training_time=None,
            last_drift_check_time=None,
            last_performance_check_time=None
        )


@router.post("/retraining/trigger")
async def trigger_retraining(
    trigger: str = "manual",
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Вручную запустить retraining

    Args:
        trigger: Причина запуска
        background_tasks: Background tasks

    Returns:
        Результат запуска
    """
    try:
        pipeline = get_retraining_pipeline()

        # Map trigger string to enum
        trigger_map = {
            "manual": RetrainingTrigger.MANUAL,
            "drift": RetrainingTrigger.DRIFT_DETECTED,
            "performance": RetrainingTrigger.PERFORMANCE_DROP,
            "scheduled": RetrainingTrigger.SCHEDULED
        }

        trigger_enum = trigger_map.get(trigger.lower(), RetrainingTrigger.MANUAL)

        # Trigger retraining (in background if possible)
        if background_tasks:
            # Use a wrapper to satisfy type checker
            async def _trigger():
                await pipeline.trigger_retraining(trigger=trigger_enum)

            background_tasks.add_task(_trigger)

            return {
                "success": True,
                "message": "Retraining triggered in background",
                "trigger": trigger
            }
        else:
            result = await pipeline.trigger_retraining(trigger=trigger_enum)
            return result

    except Exception as e:
        logger.error(f"Failed to trigger retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Health Check
# ============================================================

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check для ML management API

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "ml_management",
        "timestamp": datetime.now().isoformat()
    }
