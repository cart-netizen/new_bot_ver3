"""
Hyperparameter Optimization API - REST API для управления оптимизацией гиперпараметров.

Endpoints:
- POST /api/hyperopt/start - Start optimization
- POST /api/hyperopt/stop - Stop optimization
- POST /api/hyperopt/resume - Resume optimization
- GET /api/hyperopt/status - Get current status
- GET /api/hyperopt/results - Get optimization results
- GET /api/hyperopt/history - Get optimization history
- GET /api/hyperopt/best-params - Get best parameters found
- DELETE /api/hyperopt/clear - Clear optimization data

Файл: backend/api/hyperopt_api.py
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from enum import Enum
import asyncio
import json
import os
import signal
import traceback
import threading  # Added for stop flag

from backend.core.logger import get_logger
from backend.api.websocket_manager import get_websocket_manager

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/hyperopt", tags=["Hyperparameter Optimization"])

# Log API initialization
logger.info("=" * 60)
logger.info("Hyperparameter Optimization API initialized")
logger.info("=" * 60)


# ============================================================
# Enums & Request/Response Models
# ============================================================

class OptimizationMode(str, Enum):
    """Режимы оптимизации."""
    FULL = "full"              # Полная оптимизация всех групп
    QUICK = "quick"            # Быстрая оптимизация (только Learning Rate)
    GROUP = "group"            # Оптимизация конкретной группы
    FINE_TUNE = "fine_tune"    # Тонкая настройка


class ParameterGroup(str, Enum):
    """Группы параметров."""
    LEARNING_RATE = "learning_rate"
    REGULARIZATION = "regularization"
    AUGMENTATION = "augmentation"
    SCHEDULER = "scheduler"
    CLASS_BALANCE = "class_balance"
    TRIPLE_BARRIER = "triple_barrier"


class OptimizationRequest(BaseModel):
    """Request для запуска оптимизации."""
    mode: OptimizationMode = Field(
        default=OptimizationMode.FULL,
        description="Режим оптимизации"
    )
    target_group: Optional[ParameterGroup] = Field(
        default=None,
        description="Целевая группа (для mode=group)"
    )
    epochs_per_trial: int = Field(
        default=4, ge=1, le=20,
        description="Эпох на один trial (4 эпохи * 12 мин = 48 мин)"
    )
    max_trials_per_group: int = Field(
        default=15, ge=5, le=50,
        description="Максимум trials на группу параметров"
    )
    max_total_hours: float = Field(
        default=24.0, gt=0, le=168,
        description="Максимальное время оптимизации (часы)"
    )
    primary_metric: str = Field(
        default="val_f1",
        description="Основная метрика для оптимизации"
    )
    study_name: str = Field(
        default="ml_hyperopt",
        description="Имя study для сохранения"
    )
    use_mlflow: bool = Field(
        default=True,
        description="Использовать MLflow для трекинга"
    )
    seed: int = Field(
        default=42,
        description="Random seed для воспроизводимости"
    )


class OptimizationStatusResponse(BaseModel):
    """Response статуса оптимизации."""
    is_running: bool
    can_resume: bool
    current_job: Optional[Dict[str, Any]] = None
    progress: Optional[Dict[str, Any]] = None
    estimated_time_remaining: Optional[str] = None


class OptimizationResultsResponse(BaseModel):
    """Response результатов оптимизации."""
    best_params: Dict[str, Any]
    best_value: float
    total_trials: int
    completed_groups: List[str]
    group_results: Dict[str, Any]
    data_paths: Dict[str, str]


# ============================================================
# Global State
# ============================================================

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Текущая задача оптимизации
current_optimization: Optional[Dict[str, Any]] = None
optimization_task: Optional[asyncio.Task] = None
optimization_history: List[Dict[str, Any]] = []

# CRITICAL: Stop flag for graceful shutdown
# asyncio.Task.cancel() does NOT stop threads in ThreadPoolExecutor!
# This Event is checked by the optimizer to stop training
stop_event: threading.Event = threading.Event()

# Paths - use absolute paths based on project root
HYPEROPT_DATA_PATH = PROJECT_ROOT / "data" / "hyperopt"
HYPEROPT_RESULTS_PATH = HYPEROPT_DATA_PATH / "results.json"
HYPEROPT_STATE_PATH = HYPEROPT_DATA_PATH / "state.json"

logger.info(f"HYPEROPT API: Project root: {PROJECT_ROOT}")
logger.info(f"HYPEROPT API: Data path: {HYPEROPT_DATA_PATH}")


# ============================================================
# Helper Functions
# ============================================================

def _save_state(state: Dict[str, Any]):
    """Сохранить состояние оптимизации для возможности продолжения."""
    HYPEROPT_DATA_PATH.mkdir(parents=True, exist_ok=True)
    with open(HYPEROPT_STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def _load_state() -> Optional[Dict[str, Any]]:
    """Загрузить сохранённое состояние."""
    if HYPEROPT_STATE_PATH.exists():
        try:
            with open(HYPEROPT_STATE_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    return None


def _can_resume() -> bool:
    """Проверить, можно ли продолжить оптимизацию."""
    state = _load_state()
    if state and state.get("status") in ["paused", "interrupted"]:
        return True
    # Также проверяем наличие Optuna studies
    db_files = list(HYPEROPT_DATA_PATH.glob("*.db"))
    return len(db_files) > 0


def _get_data_paths() -> Dict[str, str]:
    """Получить пути к данным."""
    return {
        "training_data": "data/feature_store или data/ml_training",
        "results_output": str(HYPEROPT_DATA_PATH),
        "best_params": str(HYPEROPT_DATA_PATH / "best_params.json"),
        "optuna_studies": str(HYPEROPT_DATA_PATH / "*.db"),
        "mlflow_tracking": "mlruns/"
    }


def _estimate_time(
    mode: OptimizationMode,
    epochs_per_trial: int,
    max_trials_per_group: int,
    minutes_per_epoch: float = 12.0
) -> Dict[str, Any]:
    """Оценить время выполнения."""
    minutes_per_trial = epochs_per_trial * minutes_per_epoch

    if mode == OptimizationMode.QUICK:
        n_groups = 1
    elif mode == OptimizationMode.GROUP:
        n_groups = 1
    elif mode == OptimizationMode.FINE_TUNE:
        n_groups = 2
    else:  # FULL
        n_groups = 6

    total_trials = n_groups * max_trials_per_group
    # С учётом pruning (~40% trials будут прерваны рано)
    effective_trials = total_trials * 0.7
    total_minutes = effective_trials * minutes_per_trial

    return {
        "estimated_hours": round(total_minutes / 60, 1),
        "total_trials": total_trials,
        "effective_trials": int(effective_trials),
        "minutes_per_trial": minutes_per_trial,
        "groups_to_optimize": n_groups
    }


# ============================================================
# Background Task
# ============================================================

async def _run_optimization(request: OptimizationRequest, is_resume: bool = False):
    """
    Background task для выполнения оптимизации.

    Args:
        request: Конфигурация оптимизации
        is_resume: True если это возобновление после паузы
    """
    global current_optimization, stop_event

    try:
        logger.info("=" * 60)
        logger.info(f"HYPEROPT: Starting background optimization task {'(RESUME)' if is_resume else ''}")
        logger.info(f"HYPEROPT: Mode={request.mode}, Study={request.study_name}")
        logger.info(f"HYPEROPT: Epochs/trial={request.epochs_per_trial}, Max trials/group={request.max_trials_per_group}")
        logger.info(f"HYPEROPT: Max hours={request.max_total_hours}, Metric={request.primary_metric}")
        logger.info("=" * 60)

        # Импортируем оптимизатор с детальным логированием
        logger.info("HYPEROPT: Importing hyperparameter_optimizer module...")
        try:
            from backend.ml_engine.hyperparameter_optimizer import (
                HyperparameterOptimizer,
                OptimizationConfig,
                OptimizationMode as OptimizerMode,
                ParameterGroup as OptimizerParamGroup
            )
            logger.info("HYPEROPT: Import successful")
        except ImportError as ie:
            logger.error(f"HYPEROPT: Import failed: {ie}")
            logger.error(f"HYPEROPT: Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to import hyperparameter_optimizer: {ie}")

        # Создаём конфигурацию
        logger.info("HYPEROPT: Creating OptimizationConfig...")
        config = OptimizationConfig(
            study_name=request.study_name,
            storage_path=str(HYPEROPT_DATA_PATH),
            epochs_per_trial=request.epochs_per_trial,
            max_trials_per_group=request.max_trials_per_group,
            max_total_time_hours=request.max_total_hours,
            primary_metric=request.primary_metric,
            optimization_direction="minimize" if request.primary_metric == "val_loss" else "maximize",
            use_mlflow=request.use_mlflow,
            seed=request.seed
        )
        logger.info(f"HYPEROPT: Config created: {config}")

        # Создаём оптимизатор
        # CRITICAL: Pass stop_event so optimizer can check for stop requests
        logger.info("HYPEROPT: Creating HyperparameterOptimizer instance...")
        optimizer = HyperparameterOptimizer(config=config, stop_event=stop_event)
        logger.info("HYPEROPT: Optimizer created successfully (with stop_event)")

        # Обновляем статус
        current_optimization["status"] = "running"
        current_optimization["started_at"] = datetime.now().isoformat()
        _save_state(current_optimization)
        logger.info("HYPEROPT: State saved, status=running")

        # WebSocket broadcast: started
        try:
            ws_manager = get_websocket_manager()
            await ws_manager.broadcast_hyperopt_started(
                job_id=current_optimization.get("job_id", "unknown"),
                mode=request.mode.value,
                total_trials_estimate=request.max_trials_per_group * 6,  # 6 groups
                time_estimate=current_optimization.get("time_estimate", {})
            )
        except Exception as ws_error:
            logger.debug(f"WebSocket broadcast error: {ws_error}")

        # Определяем режим
        # CRITICAL: If resuming, use RESUME mode to restore previous state
        if is_resume:
            opt_mode = OptimizerMode.RESUME
            logger.info(f"HYPEROPT: Using RESUME mode to restore previous progress")
        else:
            mode_map = {
                OptimizationMode.FULL: OptimizerMode.FULL,
                OptimizationMode.QUICK: OptimizerMode.QUICK,
                OptimizationMode.GROUP: OptimizerMode.GROUP,
                OptimizationMode.FINE_TUNE: OptimizerMode.FINE_TUNE
            }
            opt_mode = mode_map.get(request.mode, OptimizerMode.FULL)
            logger.info(f"HYPEROPT: Mapped mode: {request.mode} -> {opt_mode}")

        # Целевая группа
        target_group = None
        if request.target_group:
            group_map = {
                ParameterGroup.LEARNING_RATE: OptimizerParamGroup.LEARNING_RATE,
                ParameterGroup.REGULARIZATION: OptimizerParamGroup.REGULARIZATION,
                ParameterGroup.AUGMENTATION: OptimizerParamGroup.AUGMENTATION,
                ParameterGroup.SCHEDULER: OptimizerParamGroup.SCHEDULER,
                ParameterGroup.CLASS_BALANCE: OptimizerParamGroup.CLASS_BALANCE,
                ParameterGroup.TRIPLE_BARRIER: OptimizerParamGroup.TRIPLE_BARRIER
            }
            target_group = group_map.get(request.target_group)
            logger.info(f"HYPEROPT: Target group: {request.target_group} -> {target_group}")

        # Resume path for RESUME mode
        resume_from = str(HYPEROPT_DATA_PATH) if is_resume else None

        # CRITICAL: Run optimization in a thread pool to avoid blocking event loop
        # This allows FastAPI to respond to status requests while training runs
        logger.info("HYPEROPT: Starting optimizer.optimize() in thread pool...")

        import concurrent.futures
        import threading
        loop = asyncio.get_event_loop()

        # Lock для потокобезопасного обновления current_optimization
        progress_lock = threading.Lock()

        def progress_callback(progress_data: dict):
            """Callback для обновления прогресса из optimizer."""
            nonlocal current_optimization
            with progress_lock:
                if current_optimization:
                    current_optimization["progress"]["current_group"] = progress_data.get("current_group")
                    current_optimization["progress"]["current_trial"] = progress_data.get("current_trial", 0)
                    current_optimization["progress"]["best_value"] = progress_data.get("best_value")
                    current_optimization["progress"]["best_params"] = progress_data.get("best_params", {})
                    current_optimization["progress"]["current_params"] = progress_data.get("current_params", {})
                    # Сохраняем состояние периодически (каждые 5 trials)
                    if progress_data.get("current_trial", 0) % 5 == 0:
                        _save_state(current_optimization)

        def run_sync_optimization():
            """Wrapper to run async optimization in sync context."""
            import asyncio
            # Create new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

            # CRITICAL: Apply nest_asyncio to allow nested run_until_complete calls
            # This is needed because Optuna's objective function is sync but calls async training
            try:
                import nest_asyncio
                nest_asyncio.apply(new_loop)
            except ImportError:
                logger.warning("nest_asyncio not installed - hyperopt may fail with 'event loop already running'")

            try:
                return new_loop.run_until_complete(
                    optimizer.optimize(
                        mode=opt_mode,
                        target_group=target_group,
                        resume_from=resume_from,
                        progress_callback=progress_callback# Pass resume path for RESUME mode
                    )
                )
            finally:
                new_loop.close()

        # Run in thread pool executor
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            results = await loop.run_in_executor(executor, run_sync_optimization)

        logger.info(f"HYPEROPT: Optimization finished, results keys: {results.keys() if results else 'None'}")

        # Обновляем статус
        current_optimization["status"] = "completed"
        current_optimization["completed_at"] = datetime.now().isoformat()
        current_optimization["results"] = results
        _save_state(current_optimization)

        # Добавляем в историю
        optimization_history.append(current_optimization.copy())

        logger.info("=" * 60)
        logger.info(f"HYPEROPT: Optimization COMPLETED successfully")
        logger.info(f"HYPEROPT: Best value={results.get('best_value', 'N/A')}")
        logger.info("=" * 60)

        # WebSocket broadcast: completed
        try:
            ws_manager = get_websocket_manager()
            started_at = datetime.fromisoformat(current_optimization.get("started_at", datetime.now().isoformat()))
            elapsed = str(datetime.now() - started_at)
            await ws_manager.broadcast_hyperopt_completed(
                job_id=current_optimization.get("job_id", "unknown"),
                best_params=results.get("best_params", {}),
                best_value=results.get("best_value", 0.0),
                total_trials=results.get("total_trials", 0),
                elapsed_time=elapsed
            )
        except Exception as ws_error:
            logger.debug(f"WebSocket broadcast error: {ws_error}")

    except asyncio.CancelledError:
        logger.warning("HYPEROPT: Optimization CANCELLED by user")
        if current_optimization:
            current_optimization["status"] = "paused"
            current_optimization["paused_at"] = datetime.now().isoformat()
            _save_state(current_optimization)
        raise

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"HYPEROPT: Optimization FAILED with error: {e}")
        logger.error(f"HYPEROPT: Error type: {type(e).__name__}")
        logger.error(f"HYPEROPT: Traceback:\n{traceback.format_exc()}")
        logger.error("=" * 60)

        if current_optimization:
            current_optimization["status"] = "failed"
            current_optimization["error"] = str(e)
            current_optimization["error_traceback"] = traceback.format_exc()
            current_optimization["failed_at"] = datetime.now().isoformat()
            _save_state(current_optimization)

            # WebSocket broadcast: failed
            try:
                ws_manager = get_websocket_manager()
                await ws_manager.broadcast_hyperopt_failed(
                    job_id=current_optimization.get("job_id", "unknown"),
                    error=str(e),
                    error_type=type(e).__name__
                )
            except Exception:
                pass  # Ignore WebSocket errors during failure handling


# ============================================================
# API Endpoints
# ============================================================

@router.post("/start")
async def start_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Запустить оптимизацию гиперпараметров.

    Args:
        request: Параметры оптимизации

    Returns:
        Job ID и информация о запуске
    """
    global current_optimization, optimization_task, stop_event

    logger.info("=" * 60)
    logger.info("HYPEROPT API: /start endpoint called")
    logger.info(f"HYPEROPT API: Request: mode={request.mode}, epochs={request.epochs_per_trial}")
    logger.info("=" * 60)

    # CRITICAL: Clear the stop flag before starting new optimization
    stop_event.clear()
    logger.info("HYPEROPT API: Stop flag CLEARED")

    try:
        # Проверяем, не запущена ли уже оптимизация
        if current_optimization and current_optimization.get("status") == "running":
            logger.warning("HYPEROPT API: Optimization already running, rejecting request")
            raise HTTPException(
                status_code=409,
                detail="Оптимизация уже выполняется. Остановите текущую или дождитесь завершения."
            )

        # Оценка времени
        time_estimate = _estimate_time(
            request.mode,
            request.epochs_per_trial,
            request.max_trials_per_group
        )
        logger.info(f"HYPEROPT API: Time estimate: {time_estimate}")

        # Создаём job
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"HYPEROPT API: Created job_id={job_id}")

        current_optimization = {
            "job_id": job_id,
            "status": "starting",
            "mode": request.mode.value,
            "target_group": request.target_group.value if request.target_group else None,
            "config": request.model_dump(),
            "time_estimate": time_estimate,
            "data_paths": _get_data_paths(),
            "created_at": datetime.now().isoformat(),
            "progress": {
                "current_group": None,
                "current_trial": 0,
                "total_trials_estimated": time_estimate["total_trials"],
                "groups_completed": [],
                "best_value": None,
                "best_params": {}
            }
        }

        # Создаём директорию для данных
        HYPEROPT_DATA_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(f"HYPEROPT API: Data path ensured: {HYPEROPT_DATA_PATH}")

        _save_state(current_optimization)
        logger.info("HYPEROPT API: Initial state saved")

        # Запускаем в background
        logger.info("HYPEROPT API: Creating background task for optimization...")
        optimization_task = asyncio.create_task(_run_optimization(request))
        logger.info("HYPEROPT API: Background task created and started")

        logger.info(f"HYPEROPT API: Optimization started successfully: job_id={job_id}, mode={request.mode}")

        return {
            "success": True,
            "job_id": job_id,
            "status": "started",
            "message": f"Оптимизация запущена в режиме '{request.mode.value}'",
            "time_estimate": time_estimate,
            "data_paths": _get_data_paths()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HYPEROPT API: Failed to start optimization: {e}")
        logger.error(f"HYPEROPT API: Traceback:\n{traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "message": f"Ошибка запуска оптимизации: {e}"
        }


@router.post("/stop")
async def stop_optimization() -> Dict[str, Any]:
    """
    Остановить текущую оптимизацию.

    Состояние сохраняется и можно продолжить позже через /resume.
    """
    global current_optimization, optimization_task, stop_event

    if not current_optimization or current_optimization.get("status") not in ["running", "starting", "resuming"]:
        raise HTTPException(
            status_code=400,
            detail="Нет активной оптимизации для остановки"
        )

    logger.info("=" * 60)
    logger.info("HYPEROPT API: STOP requested by user")
    logger.info("=" * 60)

    # CRITICAL: Set the stop flag FIRST
    # This is checked by the optimizer in its callback
    stop_event.set()
    logger.info("HYPEROPT API: Stop flag SET - optimizer will stop after current trial")

    # Also cancel the asyncio task (for cleanup)
    if optimization_task:
        optimization_task.cancel()
        try:
            # Wait briefly for cancellation, but don't block forever
            await asyncio.wait_for(asyncio.shield(optimization_task), timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception as e:
            logger.warning(f"Error during task cancellation: {e}")

    current_optimization["status"] = "paused"
    current_optimization["paused_at"] = datetime.now().isoformat()
    _save_state(current_optimization)

    logger.info(f"HYPEROPT API: Optimization stopped: job_id={current_optimization.get('job_id')}")

    return {
        "success": True,
        "status": "paused",
        "message": "Остановка запрошена. Оптимизация остановится после текущего trial.",
        "can_resume": True,
        "progress": current_optimization.get("progress", {})
    }


@router.post("/resume")
async def resume_optimization(
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Продолжить приостановленную оптимизацию.
    """
    global current_optimization, optimization_task, stop_event

    if current_optimization and current_optimization.get("status") == "running":
        raise HTTPException(
            status_code=409,
            detail="Оптимизация уже выполняется"
        )

    # CRITICAL: Clear the stop flag before resuming
    # Without this, the optimizer would immediately see the stop flag!
    stop_event.clear()
    logger.info("HYPEROPT API: Stop flag CLEARED for resume")

    # Загружаем сохранённое состояние
    saved_state = _load_state()
    if not saved_state:
        raise HTTPException(
            status_code=404,
            detail="Нет сохранённого состояния для продолжения. Запустите новую оптимизацию."
        )

    if saved_state.get("status") not in ["paused", "interrupted"]:
        raise HTTPException(
            status_code=400,
            detail=f"Невозможно продолжить оптимизацию со статусом '{saved_state.get('status')}'"
        )

    # Восстанавливаем конфигурацию
    config = saved_state.get("config", {})

    # Создаём новый request на основе сохранённой конфигурации
    request = OptimizationRequest(
        mode=OptimizationMode(config.get("mode", "full")),
        target_group=ParameterGroup(config.get("target_group")) if config.get("target_group") else None,
        epochs_per_trial=config.get("epochs_per_trial", 4),
        max_trials_per_group=config.get("max_trials_per_group", 15),
        max_total_hours=config.get("max_total_hours", 24.0),
        primary_metric=config.get("primary_metric", "val_f1"),
        study_name=config.get("study_name", "ml_hyperopt"),
        use_mlflow=config.get("use_mlflow", True),
        seed=config.get("seed", 42)
    )

    current_optimization = saved_state
    current_optimization["status"] = "resuming"
    current_optimization["resumed_at"] = datetime.now().isoformat()

    # CRITICAL: Pass is_resume=True to restore previous progress
    # This will:
    # 1. Use RESUME mode in optimizer
    # 2. Load fixed_params, best_params, best_value from results.json
    # 3. Skip already completed groups
    optimization_task = asyncio.create_task(_run_optimization(request, is_resume=True))

    logger.info(f"HYPEROPT API: Optimization RESUMED: job_id={saved_state.get('job_id')}")

    return {
        "success": True,
        "status": "resumed",
        "message": "Оптимизация продолжена с сохранённого состояния",
        "job_id": saved_state.get("job_id"),
        "progress": saved_state.get("progress", {})
    }


@router.get("/status")
async def get_optimization_status() -> Dict[str, Any]:
    """
    Получить текущий статус оптимизации.

    Returns dict with fields expected by frontend:
    - is_running, can_resume, current_mode, current_group
    - trials_completed, total_trials, best_metric
    - elapsed_time, estimated_remaining
    - current_trial_params, results_path, data_source_path
    """
    global current_optimization

    logger.debug("HYPEROPT API: /status endpoint called")

    is_running = (
        current_optimization is not None and
        current_optimization.get("status") in ["running", "starting", "resuming"]
    )

    can_resume = _can_resume()

    # Дефолтные значения
    result = {
        "is_running": is_running,
        "can_resume": can_resume,
        "current_mode": None,
        "current_group": None,
        "trials_completed": 0,
        "total_trials": 0,
        "best_metric": None,
        "elapsed_time": None,
        "estimated_remaining": None,
        "current_trial_params": None,
        "results_path": str(HYPEROPT_DATA_PATH),
        "data_source_path": "data/feature_store/"
    }

    if current_optimization:
        progress = current_optimization.get("progress", {})
        time_estimate = current_optimization.get("time_estimate", {})

        result["current_mode"] = current_optimization.get("mode")
        result["current_group"] = progress.get("current_group")
        result["trials_completed"] = progress.get("current_trial", 0)
        result["total_trials"] = time_estimate.get("total_trials", 0)
        result["best_metric"] = progress.get("best_value")
        result["current_trial_params"] = progress.get("current_params")

        # Если есть ошибка, добавляем её
        if current_optimization.get("error"):
            result["error"] = current_optimization.get("error")
            result["error_traceback"] = current_optimization.get("error_traceback")

        # Расчёт времени
        if current_optimization.get("started_at"):
            try:
                started = datetime.fromisoformat(current_optimization["started_at"])
                elapsed = datetime.now() - started
                hours = int(elapsed.total_seconds() // 3600)
                minutes = int((elapsed.total_seconds() % 3600) // 60)
                result["elapsed_time"] = f"{hours}:{minutes:02d}"
            except Exception:
                pass

        # Оценка оставшегося времени
        if is_running:
            current_trial = progress.get("current_trial", 0)
            total_trials = time_estimate.get("effective_trials", 0)

            if total_trials > 0 and current_trial > 0:
                remaining_trials = total_trials - current_trial
                minutes_per_trial = time_estimate.get("minutes_per_trial", 48)
                remaining_minutes = remaining_trials * minutes_per_trial

                hours = int(remaining_minutes // 60)
                minutes = int(remaining_minutes % 60)
                result["estimated_remaining"] = f"{hours}ч {minutes}м"

    return result


@router.get("/results")
async def get_optimization_results() -> OptimizationResultsResponse:
    """
    Получить результаты оптимизации.
    """
    # Пробуем загрузить из файла
    results_path = HYPEROPT_DATA_PATH / "results.json"
    best_params_path = HYPEROPT_DATA_PATH / "best_params.json"

    best_params = {}
    best_value = 0.0
    total_trials = 0
    completed_groups = []
    group_results = {}

    if results_path.exists():
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
                best_params = results.get("best_params", {})
                best_value = results.get("best_value", 0.0)
                group_results = results.get("group_results", {})
                completed_groups = list(group_results.keys())

                # Считаем total trials
                for gr in group_results.values():
                    total_trials += gr.get("n_trials", 0)
        except Exception as e:
            logger.warning(f"Failed to load results: {e}")

    # Также проверяем текущую оптимизацию
    if current_optimization and current_optimization.get("results"):
        res = current_optimization["results"]
        best_params = res.get("best_params", best_params)
        best_value = res.get("best_value", best_value)

    return OptimizationResultsResponse(
        best_params=best_params,
        best_value=best_value,
        total_trials=total_trials,
        completed_groups=completed_groups,
        group_results=group_results,
        data_paths=_get_data_paths()
    )


@router.get("/best-params")
async def get_best_params() -> Dict[str, Any]:
    """
    Получить лучшие найденные параметры.

    Returns:
        {success: bool, best_params: dict, best_value: float}
    """
    logger.debug("HYPEROPT API: /best-params endpoint called")

    best_params = {}
    best_value = None

    # Пробуем загрузить из файла
    best_params_path = HYPEROPT_DATA_PATH / "best_params.json"
    if best_params_path.exists():
        try:
            with open(best_params_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    best_params = data.get("params", data)
                    best_value = data.get("value")
                    logger.info(f"HYPEROPT API: Loaded best params from file: {len(best_params)} params")
        except Exception as e:
            logger.warning(f"HYPEROPT API: Failed to load best params from file: {e}")

    # Fallback на текущие результаты
    if not best_params and current_optimization:
        results = current_optimization.get("results", {})
        if results:
            best_params = results.get("best_params", {})
            best_value = results.get("best_value")
            logger.info(f"HYPEROPT API: Using best params from current optimization")

        # Также проверяем progress
        progress = current_optimization.get("progress", {})
        if progress.get("best_params"):
            best_params = progress.get("best_params", best_params)
            best_value = progress.get("best_value", best_value)

    if best_params:
        return {
            "success": True,
            "best_params": best_params,
            "best_value": best_value
        }

    return {
        "success": False,
        "best_params": None,
        "message": "Нет сохранённых лучших параметров. Запустите оптимизацию."
    }


@router.get("/history")
async def get_optimization_history(limit: int = 50) -> Dict[str, Any]:
    """
    Получить историю оптимизаций (trials).

    Args:
        limit: Максимум записей для возврата

    Returns:
        {trials: list, total: int}
    """
    logger.debug(f"HYPEROPT API: /history endpoint called, limit={limit}")

    trials = []

    # Пробуем загрузить из файла истории
    history_path = HYPEROPT_DATA_PATH / "trial_history.json"
    if history_path.exists():
        try:
            with open(history_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    trials = data
                elif isinstance(data, dict) and "trials" in data:
                    trials = data["trials"]
            logger.info(f"HYPEROPT API: Loaded {len(trials)} trials from history file")
        except Exception as e:
            logger.warning(f"HYPEROPT API: Failed to load trial history: {e}")

    # Также добавляем данные из текущей оптимизации
    if current_optimization:
        progress = current_optimization.get("progress", {})
        current_trials = progress.get("trials", [])
        if current_trials:
            # Добавляем триалы из текущего запуска, избегая дубликатов
            existing_ids = {t.get("trial_id") for t in trials}
            for trial in current_trials:
                if trial.get("trial_id") not in existing_ids:
                    trials.append(trial)

    # Сортируем по trial_id (новые сверху)
    trials.sort(key=lambda x: x.get("trial_id", 0), reverse=True)

    # Ограничиваем
    limited_trials = trials[:limit]

    return {
        "trials": limited_trials,
        "total": len(trials)
    }


@router.delete("/clear")
async def clear_optimization_data() -> Dict[str, Any]:
    """
    Очистить все данные оптимизации.

    ВНИМАНИЕ: Удаляет все результаты и состояния!
    """
    global current_optimization, optimization_task

    if current_optimization and current_optimization.get("status") == "running":
        raise HTTPException(
            status_code=409,
            detail="Невозможно очистить данные во время выполнения оптимизации. Сначала остановите."
        )

    # Удаляем файлы
    import shutil
    if HYPEROPT_DATA_PATH.exists():
        shutil.rmtree(HYPEROPT_DATA_PATH)
        HYPEROPT_DATA_PATH.mkdir(parents=True, exist_ok=True)

    current_optimization = None
    optimization_history.clear()

    return {
        "success": True,
        "message": "Все данные оптимизации удалены"
    }


@router.get("/parameter-groups")
async def get_parameter_groups() -> Dict[str, Any]:
    """
    Получить информацию о группах параметров.
    """
    return {
        "groups": [
            {
                "id": "learning_rate",
                "name": "Learning Rate",
                "name_ru": "Скорость обучения",
                "description_ru": "Оптимизация learning_rate, weight_decay, batch_size. Наиболее важная группа (~40% влияния).",
                "parameters": ["learning_rate", "weight_decay", "batch_size"],
                "importance": 0.9,
                "estimated_trials": 15,
                "estimated_hours": 12
            },
            {
                "id": "regularization",
                "name": "Regularization",
                "name_ru": "Регуляризация",
                "description_ru": "Оптимизация dropout, label_smoothing, focal_gamma. Предотвращает переобучение (~25% влияния).",
                "parameters": ["dropout", "label_smoothing", "focal_gamma"],
                "importance": 0.7,
                "estimated_trials": 15,
                "estimated_hours": 12
            },
            {
                "id": "class_balance",
                "name": "Class Balance",
                "name_ru": "Балансировка классов",
                "description_ru": "Оптимизация Focal Loss, class weights, oversampling. Для несбалансированных данных (~15% влияния).",
                "parameters": ["use_focal_loss", "use_class_weights", "use_oversampling", "oversample_ratio"],
                "importance": 0.6,
                "estimated_trials": 10,
                "estimated_hours": 8
            },
            {
                "id": "augmentation",
                "name": "Data Augmentation",
                "name_ru": "Аугментация данных",
                "description_ru": "Оптимизация gaussian_noise_std, use_augmentation. Увеличивает разнообразие данных (~10% влияния).",
                "parameters": ["use_augmentation", "gaussian_noise_std"],
                "importance": 0.5,
                "estimated_trials": 10,
                "estimated_hours": 8
            },
            {
                "id": "scheduler",
                "name": "LR Scheduler",
                "name_ru": "Планировщик LR",
                "description_ru": "Оптимизация scheduler_T_0, scheduler_T_mult. Управление изменением learning rate (~5% влияния).",
                "parameters": ["scheduler_T_0", "scheduler_T_mult"],
                "importance": 0.4,
                "estimated_trials": 8,
                "estimated_hours": 6
            },
            {
                "id": "triple_barrier",
                "name": "Triple Barrier",
                "name_ru": "Triple Barrier",
                "description_ru": "Оптимизация параметров разметки: Take Profit, Stop Loss, Max Holding Period (~5% влияния).",
                "parameters": ["tb_tp_multiplier", "tb_sl_multiplier", "tb_max_holding_period"],
                "importance": 0.4,
                "estimated_trials": 10,
                "estimated_hours": 8
            }
        ],
        "optimization_order": [
            "learning_rate",
            "regularization",
            "class_balance",
            "augmentation",
            "scheduler",
            "triple_barrier"
        ]
    }


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check для API."""
    return {
        "status": "healthy",
        "service": "hyperopt",
        "timestamp": datetime.now().isoformat()
    }
