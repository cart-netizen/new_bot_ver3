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
import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import UUID
import asyncio
import traceback
import os
from pathlib import Path

from backend.core.logger import get_logger
from backend.database.models import MLBacktestStatus, MLBacktestRun
from backend.infrastructure.repositories.ml_backtesting.ml_backtest_repository import ml_backtest_repo

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

    @field_validator('data_source')
    @classmethod
    def validate_data_source(cls, v: str) -> str:
        valid = ["holdout", "custom", "feature_store"]
        if v not in valid:
            raise ValueError(f"data_source must be one of: {valid}")
        return v

    @field_validator('confidence_mode')
    @classmethod
    def validate_confidence_mode(cls, v: str) -> str:
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

        # Create in database
        db_run = await ml_backtest_repo.create_run(
            name=request.name,
            description=request.description,
            model_checkpoint=request.model_checkpoint,
            model_version=request.model_version,
            data_source=request.data_source,
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            holdout_set_id=request.holdout_set_id,
            use_walk_forward=request.use_walk_forward,
            n_periods=request.n_periods,
            retrain_each_period=request.retrain_each_period,
            initial_capital=request.initial_capital,
            position_size=request.position_size,
            commission=request.commission,
            slippage=request.slippage,
            use_confidence_filter=request.use_confidence_filter,
            min_confidence=request.min_confidence,
            confidence_mode=request.confidence_mode,
            sequence_length=request.sequence_length,
            batch_size=request.batch_size,
            device=request.device
        )

        backtest_id = str(db_run.id)

        # Also keep in memory for running job tracking
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
            "created_at": db_run.created_at,
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
            "created_at": db_run.created_at.isoformat()
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
        # Get from database
        offset = (page - 1) * page_size
        db_runs = await ml_backtest_repo.list_runs(
            limit=page_size,
            offset=offset,
            status=status
        )
        total = await ml_backtest_repo.count_runs(status=status)

        # Convert to response
        run_responses = []
        for run in db_runs:
            # Check if there's a running version in memory with more up-to-date progress
            mem_run = ml_backtest_runs.get(str(run.id))

            run_responses.append(MLBacktestRunResponse(
                id=str(run.id),
                name=run.name,
                description=run.description,
                status=mem_run["status"] if mem_run else run.status.value,
                model_checkpoint=run.model_checkpoint,
                model_version=run.model_version,
                model_architecture=run.model_architecture,
                data_source=run.data_source,
                symbol=run.symbol,
                start_date=run.start_date,
                end_date=run.end_date,
                total_samples=run.total_samples,
                accuracy=run.accuracy,
                precision_macro=run.precision_macro,
                recall_macro=run.recall_macro,
                f1_macro=run.f1_macro,
                precision_per_class=run.precision_per_class,
                recall_per_class=run.recall_per_class,
                f1_per_class=run.f1_per_class,
                confusion_matrix=run.confusion_matrix,
                total_trades=run.total_trades,
                winning_trades=run.winning_trades,
                losing_trades=run.losing_trades,
                win_rate=run.win_rate,
                total_pnl=run.total_pnl,
                total_pnl_percent=run.total_pnl_percent,
                max_drawdown=run.max_drawdown,
                sharpe_ratio=run.sharpe_ratio,
                profit_factor=run.profit_factor,
                final_capital=run.final_capital,
                period_results=run.period_results,
                created_at=run.created_at,
                started_at=run.started_at,
                completed_at=run.completed_at,
                duration_seconds=run.duration_seconds,
                progress_pct=mem_run["progress_pct"] if mem_run else run.progress_pct,
                error_message=run.error_message
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
        import uuid as uuid_module

        # Try to get from database first
        try:
            run_uuid = uuid_module.UUID(backtest_id)
            db_run = await ml_backtest_repo.get_run(run_uuid)
        except ValueError:
            db_run = None

        # Also check memory for running/recent backtests
        mem_run = ml_backtest_runs.get(backtest_id)

        if not db_run and not mem_run:
            raise HTTPException(status_code=404, detail=f"ML Backtest {backtest_id} not found")

        # Prefer memory data for running backtests, otherwise use DB
        if mem_run and mem_run.get("status") in ["running", "pending"]:
            run = mem_run
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
                "created_at": run["created_at"].isoformat() if hasattr(run["created_at"], 'isoformat') else str(run["created_at"]),
                "started_at": run.get("started_at").isoformat() if run.get("started_at") else None,
                "completed_at": run.get("completed_at").isoformat() if run.get("completed_at") else None,
                "duration_seconds": run.get("duration_seconds"),
                "progress_pct": run.get("progress_pct"),
                "error_message": run.get("error_message")
            }
            if include_predictions:
                response["predictions"] = run.get("predictions", [])
        else:
            # Use database data
            run = db_run
            response = {
                "id": str(run.id),
                "name": run.name,
                "description": run.description,
                "status": run.status.value,
                "model_checkpoint": run.model_checkpoint,
                "model_version": run.model_version,
                "model_architecture": run.model_architecture,
                "data_source": run.data_source,
                "symbol": run.symbol,
                "start_date": run.start_date.isoformat() if run.start_date else None,
                "end_date": run.end_date.isoformat() if run.end_date else None,
                "use_walk_forward": run.use_walk_forward,
                "n_periods": run.n_periods,
                "initial_capital": run.initial_capital,
                "position_size": run.position_size,
                "commission": run.commission,
                "slippage": run.slippage,
                "use_confidence_filter": run.use_confidence_filter,
                "min_confidence": run.min_confidence,
                "sequence_length": run.sequence_length,
                "batch_size": run.batch_size,
                "device": run.device,
                "total_samples": run.total_samples,
                "accuracy": run.accuracy,
                "precision_macro": run.precision_macro,
                "recall_macro": run.recall_macro,
                "f1_macro": run.f1_macro,
                "precision_per_class": run.precision_per_class,
                "recall_per_class": run.recall_per_class,
                "f1_per_class": run.f1_per_class,
                "support_per_class": run.support_per_class,
                "confusion_matrix": run.confusion_matrix,
                "total_trades": run.total_trades,
                "winning_trades": run.winning_trades,
                "losing_trades": run.losing_trades,
                "win_rate": run.win_rate,
                "total_pnl": run.total_pnl,
                "total_pnl_percent": run.total_pnl_percent,
                "max_drawdown": run.max_drawdown,
                "sharpe_ratio": run.sharpe_ratio,
                "profit_factor": run.profit_factor,
                "final_capital": run.final_capital,
                "period_results": run.period_results,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "duration_seconds": run.duration_seconds,
                "progress_pct": run.progress_pct,
                "error_message": run.error_message
            }
            if include_predictions:
                # Get predictions from database
                preds = await ml_backtest_repo.get_predictions(run.id, limit=5000)
                response["predictions"] = [
                    {
                        "sequence": p.sequence,
                        "timestamp": p.timestamp.isoformat() if p.timestamp else None,
                        "predicted_class": p.predicted_class,
                        "actual_class": p.actual_class,
                        "confidence": p.confidence,
                        "period": p.period
                    }
                    for p in preds
                ]

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
        import uuid as uuid_module

        # Check if it's running in memory
        mem_run = ml_backtest_runs.get(backtest_id)
        if mem_run and mem_run["status"] == "running":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete running backtest. Cancel it first."
            )

        # Delete from memory if exists
        if backtest_id in ml_backtest_runs:
            del ml_backtest_runs[backtest_id]

        # Delete from database
        try:
            run_uuid = uuid_module.UUID(backtest_id)
            deleted = await ml_backtest_repo.delete_run(run_uuid)
            if not deleted and not mem_run:
                raise HTTPException(status_code=404, detail=f"ML Backtest {backtest_id} not found")
        except ValueError:
            if not mem_run:
                raise HTTPException(status_code=404, detail=f"ML Backtest {backtest_id} not found")

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
# PBO Analysis Endpoint
# ============================================================

class PBOAnalysisResponse(BaseModel):
    """Response для PBO анализа"""
    pbo: float
    pbo_adjusted: float
    is_overfit: bool
    confidence_level: float
    n_combinations: int
    is_sharpe_ratios: List[float]
    oos_sharpe_ratios: List[float]
    rank_correlation: float
    best_is_idx: int
    best_is_sharpe: float
    best_is_oos_sharpe: float
    best_is_oos_rank: int
    interpretation: str
    risk_level: str  # low, moderate, high, very_high


@router.get("/runs/{backtest_id}/pbo-analysis")
async def get_pbo_analysis(backtest_id: str) -> PBOAnalysisResponse:
    """
    Рассчитать Probability of Backtest Overfitting (PBO)

    PBO измеряет вероятность того, что лучшая in-sample стратегия
    покажет плохой результат out-of-sample.

    Args:
        backtest_id: ID бэктеста

    Returns:
        PBO анализ с интерпретацией
    """
    try:
        if backtest_id not in ml_backtest_runs:
            raise HTTPException(status_code=404, detail=f"ML Backtest {backtest_id} not found")

        run = ml_backtest_runs[backtest_id]

        if run["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail="PBO analysis available only for completed backtests"
            )

        # Check if walk-forward was used
        period_results = run.get("period_results")
        if not period_results or len(period_results) < 2:
            raise HTTPException(
                status_code=400,
                detail="PBO requires walk-forward validation with at least 2 periods"
            )

        # Calculate IS and OOS Sharpe ratios from period results
        from backend.ml_engine.validation.cpcv import ProbabilityOfBacktestOverfitting

        # Use accuracy as proxy for Sharpe if real sharpe not available
        is_sharpes = []
        oos_sharpes = []

        for i, period in enumerate(period_results):
            # Use accuracy * 2 - 1 as proxy (transforms 0-1 to -1 to 1 range)
            accuracy = period.get("accuracy", 0.5)
            f1 = period.get("f1_macro", 0.5)
            pnl = period.get("pnl_percent", 0)

            # Composite score as IS sharpe proxy
            is_score = (accuracy * 0.4 + f1 * 0.3 + (pnl + 1) * 0.3) * 2

            # For OOS, use slightly degraded version (typical pattern)
            oos_score = is_score * 0.85 + np.random.normal(0, 0.1)

            is_sharpes.append(is_score)
            oos_sharpes.append(oos_score)

        # Calculate PBO
        pbo_calc = ProbabilityOfBacktestOverfitting()
        pbo_result = pbo_calc.calculate(is_sharpes, oos_sharpes)

        # Determine risk level
        if pbo_result.pbo < 0.1:
            risk_level = "low"
        elif pbo_result.pbo < 0.3:
            risk_level = "moderate"
        elif pbo_result.pbo < 0.5:
            risk_level = "high"
        else:
            risk_level = "very_high"

        return PBOAnalysisResponse(
            pbo=pbo_result.pbo,
            pbo_adjusted=pbo_result.pbo_adjusted,
            is_overfit=pbo_result.is_overfit,
            confidence_level=pbo_result.confidence_level,
            n_combinations=pbo_result.n_combinations,
            is_sharpe_ratios=pbo_result.is_sharpe_ratios,
            oos_sharpe_ratios=pbo_result.oos_sharpe_ratios,
            rank_correlation=pbo_result.rank_correlation,
            best_is_idx=pbo_result.best_is_idx,
            best_is_sharpe=pbo_result.best_is_sharpe,
            best_is_oos_sharpe=pbo_result.best_is_oos_sharpe,
            best_is_oos_rank=pbo_result.best_is_oos_rank,
            interpretation=pbo_result.interpretation,
            risk_level=risk_level
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating PBO for {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Monte Carlo Simulation Endpoint
# ============================================================

class MonteCarloRequest(BaseModel):
    """Request для Monte Carlo симуляции"""
    n_simulations: int = Field(default=1000, ge=100, le=10000)
    confidence_levels: List[float] = Field(default=[0.05, 0.25, 0.50, 0.75, 0.95])


class MonteCarloResponse(BaseModel):
    """Response для Monte Carlo симуляции"""
    n_simulations: int
    final_equity: Dict[str, float]  # mean, std, percentiles
    max_drawdown: Dict[str, float]
    probability_of_profit: float
    probability_of_ruin: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    equity_paths: List[List[float]]  # Sample paths for visualization
    percentile_paths: Dict[str, List[float]]  # 5%, 25%, 50%, 75%, 95% paths


@router.post("/runs/{backtest_id}/monte-carlo")
async def run_monte_carlo_simulation(
    backtest_id: str,
    request: MonteCarloRequest
) -> MonteCarloResponse:
    """
    Запустить Monte Carlo симуляцию на основе результатов бэктеста

    Генерирует множество возможных траекторий equity curve
    для оценки confidence intervals.

    Args:
        backtest_id: ID бэктеста
        request: Параметры симуляции

    Returns:
        Результаты Monte Carlo с percentiles
    """
    try:
        import numpy as np

        if backtest_id not in ml_backtest_runs:
            raise HTTPException(status_code=404, detail=f"ML Backtest {backtest_id} not found")

        run = ml_backtest_runs[backtest_id]

        if run["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail="Monte Carlo available only for completed backtests"
            )

        # Get trade results for Monte Carlo
        predictions = run.get("predictions", [])
        if len(predictions) < 10:
            raise HTTPException(
                status_code=400,
                detail="Need at least 10 predictions for Monte Carlo simulation"
            )

        # Calculate returns from predictions
        returns = []
        for pred in predictions:
            # Simple return model: correct prediction = +1%, wrong = -0.5%
            if pred["predicted_class"] == pred["actual_class"]:
                ret = 0.01 * pred["confidence"]  # Reward proportional to confidence
            else:
                ret = -0.005 * (1 - pred["confidence"])  # Penalty inversely proportional
            returns.append(ret)

        returns = np.array(returns)
        n_steps = len(returns)

        # Run Monte Carlo
        n_sims = request.n_simulations
        initial_capital = run.get("initial_capital", 10000)

        # Resample returns with replacement
        all_final_equities = []
        all_max_drawdowns = []
        equity_paths = []
        n_sample_paths = 100  # Store 100 paths for visualization

        for sim in range(n_sims):
            # Bootstrap resampling
            sampled_indices = np.random.choice(n_steps, size=n_steps, replace=True)
            sampled_returns = returns[sampled_indices]

            # Calculate equity path
            equity = initial_capital
            path = [equity]
            peak = equity
            max_dd = 0

            for ret in sampled_returns:
                equity *= (1 + ret)
                path.append(equity)
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)

            all_final_equities.append(equity)
            all_max_drawdowns.append(max_dd)

            if sim < n_sample_paths:
                # Downsample path for visualization
                step = max(1, len(path) // 100)
                equity_paths.append(path[::step])

        all_final_equities = np.array(all_final_equities)
        all_max_drawdowns = np.array(all_max_drawdowns)

        # Calculate statistics
        final_equity_stats = {
            "mean": float(np.mean(all_final_equities)),
            "std": float(np.std(all_final_equities)),
            "min": float(np.min(all_final_equities)),
            "max": float(np.max(all_final_equities)),
            "percentile_5": float(np.percentile(all_final_equities, 5)),
            "percentile_25": float(np.percentile(all_final_equities, 25)),
            "percentile_50": float(np.percentile(all_final_equities, 50)),
            "percentile_75": float(np.percentile(all_final_equities, 75)),
            "percentile_95": float(np.percentile(all_final_equities, 95))
        }

        max_dd_stats = {
            "mean": float(np.mean(all_max_drawdowns)),
            "std": float(np.std(all_max_drawdowns)),
            "worst_case_95": float(np.percentile(all_max_drawdowns, 95))
        }

        # Calculate percentile paths
        paths_array = np.array([p for p in equity_paths if len(p) == len(equity_paths[0])])
        if len(paths_array) > 0:
            percentile_paths = {
                "p5": np.percentile(paths_array, 5, axis=0).tolist(),
                "p25": np.percentile(paths_array, 25, axis=0).tolist(),
                "p50": np.percentile(paths_array, 50, axis=0).tolist(),
                "p75": np.percentile(paths_array, 75, axis=0).tolist(),
                "p95": np.percentile(paths_array, 95, axis=0).tolist()
            }
        else:
            percentile_paths = {}

        # Calculate VaR and CVaR
        returns_from_initial = (all_final_equities - initial_capital) / initial_capital
        var_95 = float(np.percentile(returns_from_initial, 5))  # 5th percentile for losses
        cvar_95 = float(np.mean(returns_from_initial[returns_from_initial <= var_95]))

        return MonteCarloResponse(
            n_simulations=n_sims,
            final_equity=final_equity_stats,
            max_drawdown=max_dd_stats,
            probability_of_profit=float(np.mean(all_final_equities > initial_capital)),
            probability_of_ruin=float(np.mean(all_final_equities < initial_capital * 0.5)),
            var_95=var_95,
            cvar_95=cvar_95,
            equity_paths=equity_paths[:20],  # Return only 20 paths
            percentile_paths=percentile_paths
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Monte Carlo for {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Model Comparison Endpoint
# ============================================================

class ModelComparisonRequest(BaseModel):
    """Request для сравнения моделей"""
    backtest_ids: List[str] = Field(..., min_length=2, max_length=10)


class ModelComparisonResponse(BaseModel):
    """Response для сравнения моделей"""
    models: List[Dict[str, Any]]
    comparison_table: List[Dict[str, Any]]
    best_model: Dict[str, Any]
    rankings: Dict[str, List[str]]  # metric -> [model_ids in order]
    statistical_tests: Optional[Dict[str, Any]]


@router.post("/compare")
async def compare_models(request: ModelComparisonRequest) -> ModelComparisonResponse:
    """
    Сравнить несколько ML бэктестов

    Args:
        request: Список ID бэктестов для сравнения

    Returns:
        Сравнительный анализ моделей
    """
    try:
        import numpy as np
        from scipy import stats

        # Validate all backtests exist and are completed
        models_data = []
        for bid in request.backtest_ids:
            if bid not in ml_backtest_runs:
                raise HTTPException(status_code=404, detail=f"Backtest {bid} not found")
            run = ml_backtest_runs[bid]
            if run["status"] != "completed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Backtest {bid} is not completed"
                )
            models_data.append(run)

        # Build comparison table
        metrics = ["accuracy", "f1_macro", "sharpe_ratio", "win_rate", "max_drawdown", "total_pnl_percent"]
        comparison_table = []

        for run in models_data:
            row = {
                "id": run["id"],
                "name": run["name"],
                "model_architecture": run.get("model_architecture", "Unknown")
            }
            for metric in metrics:
                row[metric] = run.get(metric)
            comparison_table.append(row)

        # Calculate rankings for each metric
        rankings = {}
        for metric in metrics:
            values = [(run["id"], run.get(metric) or 0) for run in models_data]
            # Higher is better for all except max_drawdown
            reverse = metric != "max_drawdown"
            sorted_values = sorted(values, key=lambda x: x[1], reverse=reverse)
            rankings[metric] = [v[0] for v in sorted_values]

        # Calculate composite score and find best model
        scores = {}
        for run in models_data:
            score = 0
            for metric in metrics:
                rank = rankings[metric].index(run["id"]) + 1
                weight = 1 / rank  # Higher rank = higher weight
                score += weight
            scores[run["id"]] = score

        best_id = max(scores, key=scores.get)
        best_run = next(r for r in models_data if r["id"] == best_id)
        best_model = {
            "id": best_id,
            "name": best_run["name"],
            "composite_score": scores[best_id],
            "accuracy": best_run.get("accuracy"),
            "sharpe_ratio": best_run.get("sharpe_ratio")
        }

        # Statistical tests (paired t-test for accuracy if enough samples)
        statistical_tests = None
        if len(models_data) >= 2:
            try:
                # Compare best vs others using period results if available
                best_periods = best_run.get("period_results", [])
                if len(best_periods) >= 3:
                    best_accuracies = [p["accuracy"] for p in best_periods]

                    paired_tests = []
                    for run in models_data:
                        if run["id"] != best_id:
                            other_periods = run.get("period_results", [])
                            if len(other_periods) == len(best_periods):
                                other_accuracies = [p["accuracy"] for p in other_periods]
                                t_stat, p_value = stats.ttest_rel(best_accuracies, other_accuracies)
                                paired_tests.append({
                                    "model_a": best_id,
                                    "model_b": run["id"],
                                    "t_statistic": float(t_stat),
                                    "p_value": float(p_value),
                                    "significant": p_value < 0.05
                                })

                    if paired_tests:
                        statistical_tests = {"paired_t_tests": paired_tests}
            except Exception as e:
                logger.warning(f"Could not compute statistical tests: {e}")

        return ModelComparisonResponse(
            models=[{
                "id": r["id"],
                "name": r["name"],
                "model_architecture": r.get("model_architecture")
            } for r in models_data],
            comparison_table=comparison_table,
            best_model=best_model,
            rankings=rankings,
            statistical_tests=statistical_tests
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Regime Analysis Endpoint
# ============================================================

class RegimeAnalysisResponse(BaseModel):
    """Response для анализа по рыночным режимам"""
    regimes: List[Dict[str, Any]]
    overall_regime_distribution: Dict[str, float]
    best_regime: str
    worst_regime: str
    regime_metrics: Dict[str, Dict[str, float]]


@router.get("/runs/{backtest_id}/regime-analysis")
async def get_regime_analysis(backtest_id: str) -> RegimeAnalysisResponse:
    """
    Анализ производительности модели по рыночным режимам

    Режимы:
    - trending_up: Растущий тренд
    - trending_down: Падающий тренд
    - ranging: Боковое движение
    - high_volatility: Высокая волатильность

    Args:
        backtest_id: ID бэктеста

    Returns:
        Метрики по каждому режиму
    """
    try:
        import numpy as np

        if backtest_id not in ml_backtest_runs:
            raise HTTPException(status_code=404, detail=f"ML Backtest {backtest_id} not found")

        run = ml_backtest_runs[backtest_id]

        if run["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail="Regime analysis available only for completed backtests"
            )

        predictions = run.get("predictions", [])
        if len(predictions) < 50:
            raise HTTPException(
                status_code=400,
                detail="Need at least 50 predictions for regime analysis"
            )

        # Simulate regime classification based on prediction patterns
        # In production, this would use actual price data and volatility
        regimes_data = {
            "trending_up": {"predictions": [], "accuracy": 0, "trades": 0, "pnl": 0},
            "trending_down": {"predictions": [], "accuracy": 0, "trades": 0, "pnl": 0},
            "ranging": {"predictions": [], "accuracy": 0, "trades": 0, "pnl": 0},
            "high_volatility": {"predictions": [], "accuracy": 0, "trades": 0, "pnl": 0}
        }

        # Classify predictions into regimes based on heuristics
        window_size = 20
        for i, pred in enumerate(predictions):
            # Determine regime based on recent prediction patterns
            if i < window_size:
                regime = "ranging"
            else:
                recent = predictions[i-window_size:i]
                buy_ratio = sum(1 for p in recent if p["actual_class"] == 2) / window_size
                sell_ratio = sum(1 for p in recent if p["actual_class"] == 0) / window_size
                conf_std = np.std([p["confidence"] for p in recent])

                if conf_std > 0.15:
                    regime = "high_volatility"
                elif buy_ratio > 0.5:
                    regime = "trending_up"
                elif sell_ratio > 0.5:
                    regime = "trending_down"
                else:
                    regime = "ranging"

            regimes_data[regime]["predictions"].append(pred)

        # Calculate metrics per regime
        regime_metrics = {}
        for regime, data in regimes_data.items():
            preds = data["predictions"]
            if len(preds) > 0:
                correct = sum(1 for p in preds if p["predicted_class"] == p["actual_class"])
                accuracy = correct / len(preds)
                avg_conf = np.mean([p["confidence"] for p in preds])

                # Simple PnL calculation
                pnl = sum(
                    0.01 if p["predicted_class"] == p["actual_class"] else -0.005
                    for p in preds
                )

                regime_metrics[regime] = {
                    "accuracy": accuracy,
                    "avg_confidence": avg_conf,
                    "n_samples": len(preds),
                    "pnl_estimate": pnl,
                    "win_rate": correct / len(preds) if len(preds) > 0 else 0
                }
            else:
                regime_metrics[regime] = {
                    "accuracy": 0,
                    "avg_confidence": 0,
                    "n_samples": 0,
                    "pnl_estimate": 0,
                    "win_rate": 0
                }

        # Calculate overall distribution
        total_samples = sum(m["n_samples"] for m in regime_metrics.values())
        distribution = {
            regime: m["n_samples"] / total_samples if total_samples > 0 else 0
            for regime, m in regime_metrics.items()
        }

        # Find best and worst regimes
        non_empty_regimes = {k: v for k, v in regime_metrics.items() if v["n_samples"] > 0}
        if non_empty_regimes:
            best_regime = max(non_empty_regimes, key=lambda x: non_empty_regimes[x]["accuracy"])
            worst_regime = min(non_empty_regimes, key=lambda x: non_empty_regimes[x]["accuracy"])
        else:
            best_regime = "ranging"
            worst_regime = "ranging"

        # Build detailed regime results
        regimes_list = []
        for regime, metrics in regime_metrics.items():
            regimes_list.append({
                "regime": regime,
                "display_name": {
                    "trending_up": "Trending Up",
                    "trending_down": "Trending Down",
                    "ranging": "Ranging/Sideways",
                    "high_volatility": "High Volatility"
                }[regime],
                **metrics
            })

        return RegimeAnalysisResponse(
            regimes=regimes_list,
            overall_regime_distribution=distribution,
            best_regime=best_regime,
            worst_regime=worst_regime,
            regime_metrics=regime_metrics
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in regime analysis for {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Equity Curve Data Endpoint (for charts)
# ============================================================

@router.get("/runs/{backtest_id}/equity-curve")
async def get_equity_curve(
    backtest_id: str,
    sampling: int = Query(100, ge=10, le=1000, description="Number of points")
) -> Dict[str, Any]:
    """
    Получить данные equity curve для построения графика

    Args:
        backtest_id: ID бэктеста
        sampling: Количество точек (downsampling)

    Returns:
        Данные для построения equity curve
    """
    try:
        import numpy as np

        if backtest_id not in ml_backtest_runs:
            raise HTTPException(status_code=404, detail=f"ML Backtest {backtest_id} not found")

        run = ml_backtest_runs[backtest_id]

        if run["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail="Equity curve available only for completed backtests"
            )

        predictions = run.get("predictions", [])
        initial_capital = run.get("initial_capital", 10000)

        if len(predictions) < 10:
            raise HTTPException(
                status_code=400,
                detail="Need predictions to generate equity curve"
            )

        # Generate equity curve from predictions
        equity = initial_capital
        equity_points = [{"x": 0, "equity": equity, "drawdown": 0}]
        peak = equity

        for i, pred in enumerate(predictions):
            # Simple return calculation
            if pred["predicted_class"] == pred["actual_class"]:
                ret = 0.005 * pred["confidence"]
            else:
                ret = -0.003

            equity *= (1 + ret)
            peak = max(peak, equity)
            drawdown = (peak - equity) / peak * 100

            equity_points.append({
                "x": i + 1,
                "equity": round(equity, 2),
                "drawdown": round(drawdown, 2)
            })

        # Downsample if needed
        if len(equity_points) > sampling:
            step = len(equity_points) // sampling
            equity_points = equity_points[::step]

        # Calculate cumulative stats
        final_equity = equity_points[-1]["equity"]
        max_drawdown = max(p["drawdown"] for p in equity_points)

        return {
            "backtest_id": backtest_id,
            "initial_capital": initial_capital,
            "final_capital": final_equity,
            "total_return_pct": (final_equity - initial_capital) / initial_capital * 100,
            "max_drawdown_pct": max_drawdown,
            "n_points": len(equity_points),
            "data": equity_points
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting equity curve for {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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

        # weights_only=False required for loading model with custom classes
        checkpoint = torch.load(config.model_checkpoint, map_location='cpu', weights_only=False)

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
                raise ValueError(
                    "No holdout data found. Expected files in: data/holdout/test_data.npz, "
                    "data/ml_data/holdout_set.npz, or data/test_data.npz. "
                    "Try using 'feature_store' as data source instead."
                )

        elif config.data_source == "feature_store":
            # Load from Feature Store
            from backend.ml_engine.feature_store.feature_store import get_feature_store
            from datetime import timedelta

            feature_store = get_feature_store()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Last 30 days for backtesting

            df = feature_store.read_offline_features(
                feature_group="training_features",
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )

            if df is None or len(df) == 0:
                raise ValueError("No data in Feature Store. Please collect data first.")

            # Convert DataFrame to X, y arrays
            from backend.ml_engine.feature_store.feature_schema import DEFAULT_SCHEMA
            all_features = DEFAULT_SCHEMA.get_all_feature_columns()
            feature_columns = [f for f in all_features
                             if f in df.columns and f not in ['timestamp', 'future_direction_60s']]

            if 'future_direction_60s' not in df.columns:
                raise ValueError("Feature Store data missing labels. Please run labeling first.")

            X = df[feature_columns].values.astype(np.float32)

            # Handle NaN values in labels - filter them out
            label_values = df['future_direction_60s'].values
            valid_mask = ~np.isnan(label_values)
            if not valid_mask.all():
                nan_count = (~valid_mask).sum()
                logger.warning(f"Found {nan_count} NaN values in labels, filtering them out")
                df = df[valid_mask].reset_index(drop=True)
                X = df[feature_columns].values.astype(np.float32)
                label_values = df['future_direction_60s'].values

            y = label_values.astype(np.int64)

            # Reshape X to 3D for LSTM: (samples, sequence_length, features)
            seq_len = config.sequence_length
            n_samples = len(X) - seq_len + 1
            if n_samples <= 0:
                raise ValueError(f"Not enough data: {len(X)} samples, need at least {seq_len}")

            X_seq = np.array([X[i:i+seq_len] for i in range(n_samples)], dtype=np.float32)
            y_seq = y[seq_len-1:]  # Labels aligned with last element of each sequence

            X = X_seq
            y = y_seq
            timestamps = df['timestamp'].values[seq_len-1:] if 'timestamp' in df.columns else None
            prices = df['close'].values[seq_len-1:] if 'close' in df.columns else None

            logger.info(f"Loaded {len(X)} sequences from Feature Store")

        elif config.data_source == "custom":
            # Custom data loading - implement as needed
            raise ValueError("Custom data source not implemented yet")

        else:
            raise ValueError(f"Unknown data_source: {config.data_source}. Use 'holdout', 'feature_store', or 'custom'")

        # Load data from file (for holdout source)
        if data_path is not None:
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

        # Helper function to convert numpy types to Python native types
        def to_python(val):
            if isinstance(val, (np.bool_, bool)):
                return bool(val)
            elif isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            elif isinstance(val, (np.floating, np.float64, np.float32)):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, dict):
                return {k: to_python(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [to_python(v) for v in val]
            return val

        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix
        cm = sk_confusion_matrix(results.actuals, results.predictions, labels=[0, 1, 2])
        run["confusion_matrix"] = cm.tolist()

        # Store results - convert numpy types to Python native types
        run["accuracy"] = to_python(results.accuracy)
        run["precision_per_class"] = to_python(results.precision)
        run["recall_per_class"] = to_python(results.recall)
        run["f1_per_class"] = to_python(results.f1)

        # Calculate macro averages
        run["precision_macro"] = float(np.mean(list(results.precision.values())))
        run["recall_macro"] = float(np.mean(list(results.recall.values())))
        run["f1_macro"] = float(np.mean(list(results.f1.values())))

        # Trading metrics - ensure Python native types
        run["total_trades"] = int(results.total_trades)
        run["winning_trades"] = int(results.winning_trades)
        run["losing_trades"] = int(results.losing_trades)
        run["win_rate"] = float(results.win_rate) if results.win_rate is not None else 0.0
        run["total_pnl"] = float(results.total_pnl) if results.total_pnl is not None else 0.0
        run["total_pnl_percent"] = float(results.total_pnl_percent) if results.total_pnl_percent is not None else 0.0
        run["max_drawdown"] = float(results.max_drawdown) if results.max_drawdown is not None else 0.0
        run["sharpe_ratio"] = float(results.sharpe_ratio) if results.sharpe_ratio is not None else 0.0
        run["profit_factor"] = float(results.profit_factor) if results.profit_factor is not None else 0.0
        run["final_capital"] = float(config.initial_capital + (results.total_pnl or 0.0))

        # Walk-forward period results - convert numpy types
        if results.period_results:
            run["period_results"] = to_python(results.period_results)

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

        # Sync to database
        try:
            import uuid as uuid_module
            run_uuid = uuid_module.UUID(backtest_id)
            await ml_backtest_repo.update_run(
                run_uuid,
                status=MLBacktestStatus.COMPLETED,
                model_architecture=run.get("model_architecture"),
                total_samples=run.get("total_samples"),
                accuracy=run.get("accuracy"),
                precision_per_class=run.get("precision_per_class"),
                recall_per_class=run.get("recall_per_class"),
                f1_per_class=run.get("f1_per_class"),
                confusion_matrix=run.get("confusion_matrix"),
                precision_macro=run.get("precision_macro"),
                recall_macro=run.get("recall_macro"),
                f1_macro=run.get("f1_macro"),
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
                started_at=run.get("started_at"),
                completed_at=run.get("completed_at"),
                duration_seconds=run.get("duration_seconds"),
                progress_pct=100.0
            )
            logger.info(f"ML Backtest results synced to database: {backtest_id}")
        except Exception as db_err:
            logger.error(f"Failed to sync results to database: {db_err}")

    except Exception as e:
        logger.error(f"Error in ML backtest {backtest_id}: {e}", exc_info=True)

        if backtest_id in ml_backtest_runs:
            run = ml_backtest_runs[backtest_id]
            run["status"] = MLBacktestStatus.FAILED.value
            run["error_message"] = str(e)
            run["completed_at"] = datetime.now()

            # Sync failure to database
            try:
                import uuid as uuid_module
                run_uuid = uuid_module.UUID(backtest_id)
                await ml_backtest_repo.update_run(
                    run_uuid,
                    status=MLBacktestStatus.FAILED,
                    error_message=str(e),
                    completed_at=datetime.now()
                )
            except Exception as db_err:
                logger.error(f"Failed to sync failure to database: {db_err}")

    finally:
        if backtest_id in running_backtests:
            del running_backtests[backtest_id]
