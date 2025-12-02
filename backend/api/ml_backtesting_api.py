"""
ML Backtesting Management API - REST API для бэктестинга ML моделей

Endpoints:
- POST /api/ml-backtesting/runs - Create and start ML backtest
- GET /api/ml-backtesting/runs - List ML backtests with filtering
- GET /api/ml-backtesting/runs/{id} - Get ML backtest details
- GET /api/ml-backtesting/runs/{id}/predictions - Get predictions with confidence
- GET /api/ml-backtesting/runs/{id}/confusion-matrix - Get confusion matrix
- GET /api/ml-backtesting/runs/{id}/periods - Get walk-forward periods
- POST /api/ml-backtesting/runs/{id}/cancel - Cancel running backtest
- DELETE /api/ml-backtesting/runs/{id} - Delete backtest
- GET /api/ml-backtesting/models - List available models
- GET /api/ml-backtesting/statistics - Get aggregate statistics
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import UUID
import asyncio
import traceback
import os
from pathlib import Path

from backend.core.logger import get_logger
from backend.database.models import MLBacktestStatus

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/ml-backtesting", tags=["ML Backtesting"])


# ============================================================
# Request/Response Models
# ============================================================

class CreateMLBacktestRequest(BaseModel):
    """Request для создания ML бэктеста"""
    name: str = Field(..., min_length=1, max_length=200, description="Название бэктеста")
    description: Optional[str] = Field(None, max_length=1000, description="Описание")

    # Модель
    model_checkpoint: str = Field(..., description="Путь к .pt файлу или MLflow URI")
    model_version: Optional[str] = Field(None, description="Версия модели")

    # Данные
    data_source: str = Field(default="holdout", description="Источник данных: holdout, custom, feature_store")
    symbol: Optional[str] = Field(None, description="Торговая пара (для custom)")
    start_date: Optional[datetime] = Field(None, description="Дата начала (для custom)")
    end_date: Optional[datetime] = Field(None, description="Дата окончания (для custom)")
    holdout_set_id: Optional[str] = Field(None, description="ID holdout set")

    # Walk-forward
    use_walk_forward: bool = Field(default=True, description="Использовать walk-forward")
    n_periods: int = Field(default=5, ge=2, le=20, description="Количество периодов")
    retrain_each_period: bool = Field(default=False, description="Переобучать каждый период")

    # Trading simulation
    initial_capital: float = Field(default=10000.0, gt=0, description="Начальный капитал")
    position_size: float = Field(default=0.1, gt=0, le=1, description="Размер позиции (доля)")
    commission: float = Field(default=0.001, ge=0, description="Комиссия")
    slippage: float = Field(default=0.0005, ge=0, description="Проскальзывание")

    # Confidence filtering
    use_confidence_filter: bool = Field(default=True, description="Фильтр по confidence")
    min_confidence: float = Field(default=0.6, ge=0.5, le=0.99, description="Минимальный confidence")
    confidence_mode: str = Field(default="threshold", description="Режим: threshold, dynamic, percentile")

    # Inference
    sequence_length: int = Field(default=60, ge=10, le=200, description="Длина последовательности")
    batch_size: int = Field(default=128, ge=16, le=512, description="Размер батча")
    device: str = Field(default="auto", description="Device: auto, cuda, cpu")

    @validator('data_source')
    def validate_data_source(cls, v):
        valid = ["holdout", "custom", "feature_store"]
        if v not in valid:
            raise ValueError(f"data_source must be one of: {valid}")
        return v

    @validator('confidence_mode')
    def validate_confidence_mode(cls, v):
        valid = ["threshold", "dynamic", "percentile"]
        if v not in valid:
            raise ValueError(f"confidence_mode must be one of: {valid}")
        return v


class MLBacktestRunResponse(BaseModel):
    """Response с информацией о ML бэктесте"""
    id: str
    name: str
    description: Optional[str]
    status: str

    # Модель
    model_checkpoint: str
    model_version: Optional[str]
    model_architecture: Optional[str]

    # Данные
    data_source: str
    symbol: Optional[str]
    start_date: Optional[datetime]
    end_date: Optional[datetime]

    # Classification metrics
    total_samples: Optional[int]
    accuracy: Optional[float]
    precision_macro: Optional[float]
    recall_macro: Optional[float]
    f1_macro: Optional[float]
    precision_per_class: Optional[Dict[str, float]]
    recall_per_class: Optional[Dict[str, float]]
    f1_per_class: Optional[Dict[str, float]]
    confusion_matrix: Optional[List[List[int]]]

    # Trading metrics
    total_trades: Optional[int]
    winning_trades: Optional[int]
    losing_trades: Optional[int]
    win_rate: Optional[float]
    total_pnl: Optional[float]
    total_pnl_percent: Optional[float]
    max_drawdown: Optional[float]
    sharpe_ratio: Optional[float]
    profit_factor: Optional[float]
    final_capital: Optional[float]

    # Walk-forward
    period_results: Optional[List[Dict[str, Any]]]

    # Meta
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    progress_pct: Optional[float]
    error_message: Optional[str]


class MLBacktestListResponse(BaseModel):
    """Response списка ML бэктестов"""
    runs: List[MLBacktestRunResponse]
    total: int
    page: int
    page_size: int


class PredictionResponse(BaseModel):
    """Response одного предсказания"""
    timestamp: datetime
    sequence: int
    predicted_class: int
    actual_class: int
    confidence: float
    prob_sell: Optional[float]
    prob_hold: Optional[float]
    prob_buy: Optional[float]
    trade_executed: bool
    trade_pnl: Optional[float]
    period: Optional[int]


class ConfusionMatrixResponse(BaseModel):
    """Response confusion matrix"""
    matrix: List[List[int]]
    labels: List[str]
    normalized: Optional[List[List[float]]]
    total_samples: int
    correct_predictions: int
    accuracy: float


class PeriodResultResponse(BaseModel):
    """Response результата одного периода walk-forward"""
    period: int
    start_idx: int
    end_idx: int
    samples: int
    accuracy: float
    f1_macro: float
    precision_macro: Optional[float]
    recall_macro: Optional[float]
    pnl_percent: Optional[float]
    win_rate: Optional[float]
    class_distribution: Dict[str, int]


class ModelInfoResponse(BaseModel):
    """Response информации о доступной модели"""
    checkpoint_path: str
    version: Optional[str]
    architecture: Optional[str]
    created_at: Optional[datetime]
    metrics: Optional[Dict[str, float]]
    stage: Optional[str]  # staging, production, archived


class StatisticsResponse(BaseModel):
    """Response агрегированной статистики"""
    total_backtests: int
    completed_backtests: int
    running_backtests: int
    failed_backtests: int
    avg_accuracy: float
    avg_f1_macro: float
    avg_sharpe_ratio: float
    avg_win_rate: float
    best_backtest: Optional[Dict[str, Any]]
    worst_backtest: Optional[Dict[str, Any]]


# ============================================================
# In-memory storage (replace with repository later)
# ============================================================

# Temporary in-memory storage for demo
ml_backtest_runs: Dict[str, Dict[str, Any]] = {}
running_backtests: Dict[str, Any] = {}


# ============================================================
# Helper Functions
# ============================================================

def get_class_name(class_id: int) -> str:
    """Convert class ID to name."""
    return {0: 'SELL', 1: 'HOLD', 2: 'BUY'}.get(class_id, 'UNKNOWN')


# ============================================================
# Endpoints
# ============================================================

@router.post("/runs")
async def create_ml_backtest(
    request: CreateMLBacktestRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Создать и запустить новый ML бэктест

    Args:
        request: Параметры бэктеста
        background_tasks: FastAPI background tasks

    Returns:
        ID и статус созданного бэктеста
    """
    import uuid

    logger.info(f"Creating ML backtest: {request.name}")

    try:
        # Validate model checkpoint exists
        if not request.model_checkpoint.startswith("mlflow:"):
            checkpoint_path = Path(request.model_checkpoint)
            if not checkpoint_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Model checkpoint not found: {request.model_checkpoint}"
                )

        # Generate ID
        backtest_id = str(uuid.uuid4())

        # Create run record
        run_data = {
            "id": backtest_id,
            "name": request.name,
            "description": request.description,
            "status": MLBacktestStatus.PENDING.value,
            "model_checkpoint": request.model_checkpoint,
            "model_version": request.model_version,
            "model_architecture": None,
            "data_source": request.data_source,
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "holdout_set_id": request.holdout_set_id,
            "use_walk_forward": request.use_walk_forward,
            "n_periods": request.n_periods,
            "retrain_each_period": request.retrain_each_period,
            "initial_capital": request.initial_capital,
            "position_size": request.position_size,
            "commission": request.commission,
            "slippage": request.slippage,
            "use_confidence_filter": request.use_confidence_filter,
            "min_confidence": request.min_confidence,
            "confidence_mode": request.confidence_mode,
            "sequence_length": request.sequence_length,
            "batch_size": request.batch_size,
            "device": request.device,
            "created_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
            "progress_pct": 0.0,
            "total_samples": None,
            "accuracy": None,
            "precision_per_class": None,
            "recall_per_class": None,
            "f1_per_class": None,
            "confusion_matrix": None,
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": None,
            "total_pnl": None,
            "total_pnl_percent": None,
            "max_drawdown": None,
            "sharpe_ratio": None,
            "profit_factor": None,
            "final_capital": None,
            "period_results": None,
            "predictions": [],
            "error_message": None
        }

        ml_backtest_runs[backtest_id] = run_data

        logger.info(f"ML Backtest created: {backtest_id}")

        # Start background job
        background_tasks.add_task(_run_ml_backtest_job, backtest_id, request)

        return {
            "id": backtest_id,
            "name": request.name,
            "status": "pending",
            "message": "ML Backtest created and started in background",
            "created_at": run_data["created_at"].isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating ML backtest: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs")
async def list_ml_backtests(
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size")
) -> MLBacktestListResponse:
    """
    Получить список ML бэктестов

    Args:
        status: Фильтр по статусу
        page: Номер страницы
        page_size: Размер страницы

    Returns:
        Список бэктестов
    """
    try:
        # Filter runs
        runs = list(ml_backtest_runs.values())

        if status:
            runs = [r for r in runs if r["status"] == status]

        # Sort by created_at descending
        runs.sort(key=lambda x: x["created_at"], reverse=True)

        # Pagination
        total = len(runs)
        offset = (page - 1) * page_size
        runs = runs[offset:offset + page_size]

        # Convert to response
        run_responses = []
        for run in runs:
            run_responses.append(MLBacktestRunResponse(
                id=run["id"],
                name=run["name"],
                description=run.get("description"),
                status=run["status"],
                model_checkpoint=run["model_checkpoint"],
                model_version=run.get("model_version"),
                model_architecture=run.get("model_architecture"),
                data_source=run["data_source"],
                symbol=run.get("symbol"),
                start_date=run.get("start_date"),
                end_date=run.get("end_date"),
                total_samples=run.get("total_samples"),
                accuracy=run.get("accuracy"),
                precision_macro=run.get("precision_macro"),
                recall_macro=run.get("recall_macro"),
                f1_macro=run.get("f1_macro"),
                precision_per_class=run.get("precision_per_class"),
                recall_per_class=run.get("recall_per_class"),
                f1_per_class=run.get("f1_per_class"),
                confusion_matrix=run.get("confusion_matrix"),
                total_trades=run.get("total_trades"),
                winning_trades=run.get("winning_trades"),
                losing_trades=run.get("losing_trades"),
                win_rate=run.get("win_rate"),
                total_pnl=run.get("total_pnl"),
                total_pnl_percent=run.get("total_pnl_percent"),
                max_drawdown=run.get("max_drawdown"),
                sharpe_ratio=run.get("sharpe_ratio"),
                profit_factor=run.get("profit_factor"),
                final_capital=run.get("final_capital"),
                period_results=run.get("period_results"),
                created_at=run["created_at"],
                started_at=run.get("started_at"),
                completed_at=run.get("completed_at"),
                duration_seconds=run.get("duration_seconds"),
                progress_pct=run.get("progress_pct"),
                error_message=run.get("error_message")
            ))

        return MLBacktestListResponse(
            runs=run_responses,
            total=total,
            page=page,
            page_size=page_size
        )

    except Exception as e:
        logger.error(f"Error listing ML backtests: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{backtest_id}")
async def get_ml_backtest(
    backtest_id: str,
    include_predictions: bool = Query(False, description="Include predictions")
) -> Dict[str, Any]:
    """
    Получить детальную информацию о ML бэктесте

    Args:
        backtest_id: ID бэктеста
        include_predictions: Включить предсказания

    Returns:
        Детальная информация
    """
    try:
        if backtest_id not in ml_backtest_runs:
            raise HTTPException(status_code=404, detail=f"ML Backtest {backtest_id} not found")

        run = ml_backtest_runs[backtest_id]

        response = {
            "id": run["id"],
            "name": run["name"],
            "description": run.get("description"),
            "status": run["status"],
            "model_checkpoint": run["model_checkpoint"],
            "model_version": run.get("model_version"),
            "model_architecture": run.get("model_architecture"),
            "data_source": run["data_source"],
            "symbol": run.get("symbol"),
            "start_date": run.get("start_date").isoformat() if run.get("start_date") else None,
            "end_date": run.get("end_date").isoformat() if run.get("end_date") else None,
            "use_walk_forward": run.get("use_walk_forward"),
            "n_periods": run.get("n_periods"),
            "initial_capital": run.get("initial_capital"),
            "position_size": run.get("position_size"),
            "commission": run.get("commission"),
            "slippage": run.get("slippage"),
            "use_confidence_filter": run.get("use_confidence_filter"),
            "min_confidence": run.get("min_confidence"),
            "sequence_length": run.get("sequence_length"),
            "batch_size": run.get("batch_size"),
            "device": run.get("device"),
            "total_samples": run.get("total_samples"),
            "accuracy": run.get("accuracy"),
            "precision_macro": run.get("precision_macro"),
            "recall_macro": run.get("recall_macro"),
            "f1_macro": run.get("f1_macro"),
            "precision_per_class": run.get("precision_per_class"),
            "recall_per_class": run.get("recall_per_class"),
            "f1_per_class": run.get("f1_per_class"),
            "support_per_class": run.get("support_per_class"),
            "confusion_matrix": run.get("confusion_matrix"),
            "total_trades": run.get("total_trades"),
            "winning_trades": run.get("winning_trades"),
            "losing_trades": run.get("losing_trades"),
            "win_rate": run.get("win_rate"),
            "total_pnl": run.get("total_pnl"),
            "total_pnl_percent": run.get("total_pnl_percent"),
            "max_drawdown": run.get("max_drawdown"),
            "sharpe_ratio": run.get("sharpe_ratio"),
            "profit_factor": run.get("profit_factor"),
            "final_capital": run.get("final_capital"),
            "period_results": run.get("period_results"),
            "created_at": run["created_at"].isoformat(),
            "started_at": run.get("started_at").isoformat() if run.get("started_at") else None,
            "completed_at": run.get("completed_at").isoformat() if run.get("completed_at") else None,
            "duration_seconds": run.get("duration_seconds"),
            "progress_pct": run.get("progress_pct"),
            "error_message": run.get("error_message")
        }

        if include_predictions:
            response["predictions"] = run.get("predictions", [])

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ML backtest {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{backtest_id}/predictions")
async def get_ml_backtest_predictions(
    backtest_id: str,
    limit: int = Query(1000, ge=1, le=10000, description="Max predictions"),
    period: Optional[int] = Query(None, description="Filter by period")
) -> Dict[str, Any]:
    """
    Получить предсказания ML бэктеста

    Args:
        backtest_id: ID бэктеста
        limit: Максимальное количество
        period: Фильтр по периоду

    Returns:
        Список предсказаний
    """
    try:
        if backtest_id not in ml_backtest_runs:
            raise HTTPException(status_code=404, detail=f"ML Backtest {backtest_id} not found")

        run = ml_backtest_runs[backtest_id]
        predictions = run.get("predictions", [])

        # Filter by period if specified
        if period is not None:
            predictions = [p for p in predictions if p.get("period") == period]

        # Limit
        predictions = predictions[:limit]

        return {
            "backtest_id": backtest_id,
            "predictions": predictions,
            "total": len(predictions)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting predictions for {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{backtest_id}/confusion-matrix")
async def get_confusion_matrix(backtest_id: str) -> ConfusionMatrixResponse:
    """
    Получить confusion matrix ML бэктеста

    Args:
        backtest_id: ID бэктеста

    Returns:
        Confusion matrix с нормализацией
    """
    try:
        if backtest_id not in ml_backtest_runs:
            raise HTTPException(status_code=404, detail=f"ML Backtest {backtest_id} not found")

        run = ml_backtest_runs[backtest_id]
        matrix = run.get("confusion_matrix")

        if matrix is None:
            raise HTTPException(status_code=400, detail="Confusion matrix not available yet")

        # Calculate normalized matrix
        import numpy as np
        matrix_np = np.array(matrix)
        row_sums = matrix_np.sum(axis=1, keepdims=True)
        normalized = (matrix_np / row_sums.clip(min=1)).tolist()

        total_samples = int(matrix_np.sum())
        correct_predictions = int(np.trace(matrix_np))
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        return ConfusionMatrixResponse(
            matrix=matrix,
            labels=["SELL", "HOLD", "BUY"],
            normalized=normalized,
            total_samples=total_samples,
            correct_predictions=correct_predictions,
            accuracy=accuracy
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting confusion matrix for {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{backtest_id}/periods")
async def get_walk_forward_periods(backtest_id: str) -> Dict[str, Any]:
    """
    Получить результаты walk-forward периодов

    Args:
        backtest_id: ID бэктеста

    Returns:
        Результаты по периодам
    """
    try:
        if backtest_id not in ml_backtest_runs:
            raise HTTPException(status_code=404, detail=f"ML Backtest {backtest_id} not found")

        run = ml_backtest_runs[backtest_id]
        period_results = run.get("period_results")

        if period_results is None:
            raise HTTPException(status_code=400, detail="Period results not available yet")

        return {
            "backtest_id": backtest_id,
            "use_walk_forward": run.get("use_walk_forward"),
            "n_periods": run.get("n_periods"),
            "periods": period_results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting periods for {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/runs/{backtest_id}/cancel")
async def cancel_ml_backtest(backtest_id: str) -> Dict[str, Any]:
    """
    Отменить выполняющийся ML бэктест

    Args:
        backtest_id: ID бэктеста

    Returns:
        Результат операции
    """
    try:
        if backtest_id not in ml_backtest_runs:
            raise HTTPException(status_code=404, detail=f"ML Backtest {backtest_id} not found")

        run = ml_backtest_runs[backtest_id]

        if run["status"] not in ["pending", "running"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel backtest in status {run['status']}"
            )

        run["status"] = MLBacktestStatus.CANCELLED.value
        run["completed_at"] = datetime.now()

        if backtest_id in running_backtests:
            running_backtests[backtest_id]["cancelled"] = True

        logger.info(f"ML Backtest cancelled: {backtest_id}")

        return {
            "success": True,
            "backtest_id": backtest_id,
            "message": "ML Backtest cancelled"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling ML backtest {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/runs/{backtest_id}")
async def delete_ml_backtest(backtest_id: str) -> Dict[str, Any]:
    """
    Удалить ML бэктест

    Args:
        backtest_id: ID бэктеста

    Returns:
        Результат операции
    """
    try:
        if backtest_id not in ml_backtest_runs:
            raise HTTPException(status_code=404, detail=f"ML Backtest {backtest_id} not found")

        run = ml_backtest_runs[backtest_id]

        if run["status"] == "running":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete running backtest. Cancel it first."
            )

        del ml_backtest_runs[backtest_id]

        logger.info(f"ML Backtest deleted: {backtest_id}")

        return {
            "success": True,
            "backtest_id": backtest_id,
            "message": "ML Backtest deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting ML backtest {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_available_models() -> Dict[str, Any]:
    """
    Получить список доступных моделей для бэктестинга

    Returns:
        Список моделей
    """
    try:
        models = []

        # Scan for .pt files in common directories
        model_dirs = [
            "data/models",
            "models",
            "checkpoints",
            "data/ml_models"
        ]

        for dir_path in model_dirs:
            path = Path(dir_path)
            if path.exists():
                for pt_file in path.glob("**/*.pt"):
                    models.append({
                        "checkpoint_path": str(pt_file),
                        "version": None,
                        "architecture": "Unknown",
                        "created_at": datetime.fromtimestamp(pt_file.stat().st_mtime).isoformat(),
                        "stage": "local"
                    })

        # TODO: Add MLflow integration to list registered models

        return {
            "models": models,
            "total": len(models)
        }

    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics() -> StatisticsResponse:
    """
    Получить агрегированную статистику по ML бэктестам

    Returns:
        Статистика
    """
    try:
        runs = list(ml_backtest_runs.values())

        total = len(runs)
        completed = len([r for r in runs if r["status"] == "completed"])
        running = len([r for r in runs if r["status"] == "running"])
        failed = len([r for r in runs if r["status"] == "failed"])

        # Calculate averages for completed runs
        completed_runs = [r for r in runs if r["status"] == "completed"]

        avg_accuracy = 0
        avg_f1 = 0
        avg_sharpe = 0
        avg_win_rate = 0

        if completed_runs:
            accuracies = [r["accuracy"] for r in completed_runs if r.get("accuracy")]
            f1s = [r["f1_macro"] for r in completed_runs if r.get("f1_macro")]
            sharpes = [r["sharpe_ratio"] for r in completed_runs if r.get("sharpe_ratio")]
            win_rates = [r["win_rate"] for r in completed_runs if r.get("win_rate")]

            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)
            if f1s:
                avg_f1 = sum(f1s) / len(f1s)
            if sharpes:
                avg_sharpe = sum(sharpes) / len(sharpes)
            if win_rates:
                avg_win_rate = sum(win_rates) / len(win_rates)

        # Find best and worst
        best_backtest = None
        worst_backtest = None

        if completed_runs:
            sorted_by_sharpe = sorted(
                [r for r in completed_runs if r.get("sharpe_ratio") is not None],
                key=lambda x: x["sharpe_ratio"],
                reverse=True
            )
            if sorted_by_sharpe:
                best_backtest = {
                    "id": sorted_by_sharpe[0]["id"],
                    "name": sorted_by_sharpe[0]["name"],
                    "sharpe_ratio": sorted_by_sharpe[0]["sharpe_ratio"],
                    "accuracy": sorted_by_sharpe[0].get("accuracy")
                }
                worst_backtest = {
                    "id": sorted_by_sharpe[-1]["id"],
                    "name": sorted_by_sharpe[-1]["name"],
                    "sharpe_ratio": sorted_by_sharpe[-1]["sharpe_ratio"],
                    "accuracy": sorted_by_sharpe[-1].get("accuracy")
                }

        return StatisticsResponse(
            total_backtests=total,
            completed_backtests=completed,
            running_backtests=running,
            failed_backtests=failed,
            avg_accuracy=avg_accuracy,
            avg_f1_macro=avg_f1,
            avg_sharpe_ratio=avg_sharpe,
            avg_win_rate=avg_win_rate,
            best_backtest=best_backtest,
            worst_backtest=worst_backtest
        )

    except Exception as e:
        logger.error(f"Error getting statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check для ML backtesting API

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "ml-backtesting",
        "timestamp": datetime.now().isoformat(),
        "running_backtests": len([b for b in running_backtests.values() if not b.get("cancelled")])
    }


# ============================================================
# Background Job
# ============================================================

async def _run_ml_backtest_job(backtest_id: str, config: CreateMLBacktestRequest):
    """
    Background task для выполнения ML бэктеста

    Args:
        backtest_id: ID бэктеста
        config: Конфигурация
    """
    running_backtests[backtest_id] = {"cancelled": False, "started_at": datetime.now()}

    try:
        logger.info(f"Starting ML backtest: {backtest_id}")

        run = ml_backtest_runs[backtest_id]
        run["status"] = MLBacktestStatus.RUNNING.value
        run["started_at"] = datetime.now()

        # Import here to avoid circular imports
        from backend.ml_engine.backtesting.backtest_evaluator import (
            BacktestEvaluator, BacktestConfig
        )
        import torch
        import numpy as np

        # Load model
        logger.info(f"Loading model from {config.model_checkpoint}")

        checkpoint = torch.load(config.model_checkpoint, map_location='cpu')

        # Get model config and create model
        model_config_dict = checkpoint.get('model_config', {})
        run["model_architecture"] = model_config_dict.get('architecture', 'HybridCNNLSTMv2')

        from backend.ml_engine.models.hybrid_cnn_lstm_v2 import HybridCNNLSTMv2, ModelConfigV2
        model_cfg = ModelConfigV2(**model_config_dict)
        model = HybridCNNLSTMv2(model_cfg)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Create BacktestConfig
        bt_config = BacktestConfig(
            sequence_length=config.sequence_length,
            batch_size=config.batch_size,
            initial_capital=config.initial_capital,
            position_size=config.position_size,
            commission=config.commission,
            slippage=config.slippage,
            min_confidence=config.min_confidence,
            use_confidence_filter=config.use_confidence_filter,
            n_periods=config.n_periods,
            retrain_each_period=config.retrain_each_period
        )

        # Determine device
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create evaluator
        evaluator = BacktestEvaluator(model, bt_config, device)

        # Load test data
        # For now, try to load from a default holdout file
        data_path = None
        if config.data_source == "holdout":
            # Look for holdout data
            holdout_paths = [
                "data/holdout/test_data.npz",
                "data/ml_data/holdout_set.npz",
                "data/test_data.npz"
            ]
            for path in holdout_paths:
                if Path(path).exists():
                    data_path = path
                    break

            if data_path is None:
                raise ValueError("No holdout data found. Please prepare test data first.")

        elif config.data_source == "custom":
            # Custom data loading - implement as needed
            raise ValueError("Custom data source not implemented yet")

        logger.info(f"Loading data from {data_path}")
        data = np.load(data_path, allow_pickle=True)

        X = data['X']
        y = data['y']
        timestamps = data.get('timestamps', None)
        prices = data.get('prices', None)

        run["total_samples"] = len(X)
        run["progress_pct"] = 10.0

        # Check for cancellation
        if running_backtests[backtest_id].get("cancelled"):
            logger.info(f"ML Backtest cancelled: {backtest_id}")
            return

        # Run backtest
        if config.use_walk_forward:
            logger.info("Running walk-forward backtest")
            results = evaluator.run_walk_forward_backtest(X, y, timestamps, prices)
        else:
            logger.info("Running simple backtest")
            results = evaluator.run_backtest(X, y, timestamps, prices)

        # Check for cancellation
        if running_backtests[backtest_id].get("cancelled"):
            logger.info(f"ML Backtest cancelled: {backtest_id}")
            return

        run["progress_pct"] = 80.0

        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix
        cm = sk_confusion_matrix(results.actuals, results.predictions, labels=[0, 1, 2])
        run["confusion_matrix"] = cm.tolist()

        # Store results
        run["accuracy"] = results.accuracy
        run["precision_per_class"] = results.precision
        run["recall_per_class"] = results.recall
        run["f1_per_class"] = results.f1

        # Calculate macro averages
        run["precision_macro"] = np.mean(list(results.precision.values()))
        run["recall_macro"] = np.mean(list(results.recall.values()))
        run["f1_macro"] = np.mean(list(results.f1.values()))

        # Trading metrics
        run["total_trades"] = results.total_trades
        run["winning_trades"] = results.winning_trades
        run["losing_trades"] = results.losing_trades
        run["win_rate"] = results.win_rate
        run["total_pnl"] = results.total_pnl
        run["total_pnl_percent"] = results.total_pnl_percent
        run["max_drawdown"] = results.max_drawdown
        run["sharpe_ratio"] = results.sharpe_ratio
        run["profit_factor"] = results.profit_factor
        run["final_capital"] = config.initial_capital + results.total_pnl

        # Walk-forward period results
        if results.period_results:
            run["period_results"] = results.period_results

        # Store predictions (limited)
        predictions_list = []
        for i, (pred, actual, conf) in enumerate(zip(
            results.predictions[:5000],
            results.actuals[:5000],
            results.confidences[:5000]
        )):
            predictions_list.append({
                "sequence": i,
                "timestamp": datetime.now().isoformat(),  # Replace with actual timestamp if available
                "predicted_class": int(pred),
                "actual_class": int(actual),
                "confidence": float(conf),
                "period": None
            })
        run["predictions"] = predictions_list

        # Complete
        run["status"] = MLBacktestStatus.COMPLETED.value
        run["completed_at"] = datetime.now()
        run["duration_seconds"] = (run["completed_at"] - run["started_at"]).total_seconds()
        run["progress_pct"] = 100.0

        logger.info(
            f"ML Backtest completed: {backtest_id}, "
            f"accuracy={run['accuracy']:.2%}, "
            f"f1_macro={run['f1_macro']:.2%}, "
            f"sharpe={run['sharpe_ratio']:.2f}"
        )

    except Exception as e:
        logger.error(f"Error in ML backtest {backtest_id}: {e}", exc_info=True)

        if backtest_id in ml_backtest_runs:
            run = ml_backtest_runs[backtest_id]
            run["status"] = MLBacktestStatus.FAILED.value
            run["error_message"] = str(e)
            run["completed_at"] = datetime.now()

    finally:
        if backtest_id in running_backtests:
            del running_backtests[backtest_id]
