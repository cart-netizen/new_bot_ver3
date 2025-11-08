"""
Backtesting Management API - REST API для управления бэктестами через фронтенд

Endpoints:
- POST /api/backtesting/runs - Create and start backtest
- GET /api/backtesting/runs - List backtests with filtering
- GET /api/backtesting/runs/{id} - Get backtest details
- GET /api/backtesting/runs/{id}/trades - Get backtest trades
- GET /api/backtesting/runs/{id}/equity-curve - Get equity curve
- POST /api/backtesting/runs/{id}/cancel - Cancel running backtest
- DELETE /api/backtesting/runs/{id} - Delete backtest
- GET /api/backtesting/statistics - Get aggregate statistics
- GET /api/backtesting/config/defaults - Get default configuration values
- POST /api/backtesting/config/validate - Validate configuration
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import UUID
import asyncio

from backend.core.logger import get_logger
from backend.infrastructure.repositories.backtesting.backtest_repository import BacktestRepository
from backend.backtesting.models import (
    BacktestConfig,
    ExchangeConfig,
    StrategyConfig,
    RiskConfig,
    SlippageModel
)
from backend.backtesting.core.backtesting_engine import BacktestingEngine
from backend.backtesting.core.data_handler import HistoricalDataHandler
from backend.backtesting.core.simulated_exchange import SimulatedExchange
from backend.database.models import BacktestStatus, OrderSide

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/backtesting", tags=["Backtesting"])


# ============================================================
# Request/Response Models
# ============================================================

class CreateBacktestRequest(BaseModel):
    """Request для создания бэктеста"""
    name: str = Field(..., min_length=1, max_length=200, description="Название бэктеста")
    description: Optional[str] = Field(None, max_length=1000, description="Описание бэктеста")

    # Основные параметры
    symbol: str = Field(default="BTCUSDT", description="Торговая пара")
    start_date: datetime = Field(..., description="Начальная дата бэктеста")
    end_date: datetime = Field(..., description="Конечная дата бэктеста")
    initial_capital: float = Field(default=10000.0, gt=0, description="Начальный капитал (USDT)")
    candle_interval: str = Field(default="1", description="Интервал свечей (1, 5, 15, 60)")

    # Конфигурация биржи
    commission_rate: float = Field(default=0.0006, ge=0, le=0.1, description="Комиссия (0.06%)")
    maker_commission: Optional[float] = Field(default=0.0002, ge=0, le=0.1, description="Maker комиссия")
    taker_commission: Optional[float] = Field(default=0.0006, ge=0, le=0.1, description="Taker комиссия")
    slippage_model: str = Field(default="fixed", description="Модель slippage (fixed, volume_based, percentage)")
    slippage_pct: float = Field(default=0.01, ge=0, le=1.0, description="Процент slippage")
    simulate_latency: bool = Field(default=False, description="Симулировать задержку")

    # Конфигурация стратегий
    enabled_strategies: List[str] = Field(
        default=["momentum", "sar_wave", "supertrend", "volume_profile"],
        description="Включенные стратегии"
    )
    strategy_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Параметры стратегий")
    consensus_mode: str = Field(default="weighted", description="Режим консенсуса (weighted, majority, unanimous)")
    min_strategies_for_signal: int = Field(default=2, ge=1, description="Минимум стратегий для сигнала")
    min_consensus_confidence: float = Field(default=0.6, ge=0, le=1, description="Минимальная уверенность консенсуса")
    strategy_weights: Dict[str, float] = Field(default_factory=dict, description="Веса стратегий")

    # Конфигурация риск-менеджмента
    position_size_pct: float = Field(default=10.0, ge=0.1, le=100, description="Размер позиции (% капитала)")
    position_size_mode: str = Field(default="percentage", description="Режим размера позиции")
    max_open_positions: int = Field(default=3, ge=1, le=20, description="Максимум открытых позиций")
    stop_loss_pct: float = Field(default=2.0, ge=0.1, le=50, description="Stop Loss (%)")
    take_profit_pct: float = Field(default=4.0, ge=0.1, le=100, description="Take Profit (%)")
    use_trailing_stop: bool = Field(default=True, description="Использовать trailing stop")
    trailing_stop_activation_pct: float = Field(default=1.0, ge=0, description="Активация trailing stop (%)")
    trailing_stop_distance_pct: float = Field(default=0.5, ge=0, description="Расстояние trailing stop (%)")
    risk_per_trade_pct: float = Field(default=1.0, ge=0.1, le=10, description="Риск на сделку (%)")

    # Дополнительные параметры
    use_orderbook_data: bool = Field(default=False, description="Использовать данные orderbook")
    warmup_period_bars: int = Field(default=100, ge=0, description="Период прогрева индикаторов (свечей)")
    verbose: bool = Field(default=False, description="Подробное логирование")

    # Phase 1: OrderBook & Market Trades
    orderbook_num_levels: Optional[int] = Field(default=20, ge=10, le=50, description="Количество уровней в orderbook (10-50)")
    orderbook_base_spread_bps: Optional[float] = Field(default=2.0, ge=0.1, le=50, description="Базовый спред в basis points")
    use_market_trades: Optional[bool] = Field(default=False, description="Генерировать синтетические market trades")
    trades_per_volume_unit: Optional[int] = Field(default=100, ge=10, le=1000, description="Количество trades на единицу объема")

    # Phase 2: ML Model Integration
    use_ml_model: Optional[bool] = Field(default=False, description="Использовать ML модель для предсказаний")
    ml_server_url: Optional[str] = Field(default="http://localhost:8001", description="URL ML сервера")
    ml_model_name: Optional[str] = Field(default=None, description="Имя ML модели (опционально)")
    ml_model_version: Optional[str] = Field(default=None, description="Версия ML модели (опционально)")

    # Phase 3: Performance Optimization
    use_cache: Optional[bool] = Field(default=True, description="Использовать кэширование данных")
    skip_orderbook_generation_every_n: Optional[int] = Field(default=None, ge=1, description="Генерировать orderbook каждые N свечей")
    skip_trades_generation_every_n: Optional[int] = Field(default=None, ge=1, description="Генерировать trades каждые N свечей")
    log_trades: Optional[bool] = Field(default=False, description="Логировать каждую сделку")

    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        """Валидация дат"""
        if v > datetime.now():
            raise ValueError("Дата не может быть в будущем")
        return v

    @validator('end_date')
    def validate_date_range(cls, v, values):
        """Валидация диапазона дат"""
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("end_date должна быть больше start_date")
        return v

    @validator('slippage_model')
    def validate_slippage_model(cls, v):
        """Валидация модели slippage"""
        valid_models = ["fixed", "volume_based", "percentage"]
        if v not in valid_models:
            raise ValueError(f"Недопустимая модель slippage. Используйте: {', '.join(valid_models)}")
        return v


class BacktestRunResponse(BaseModel):
    """Response с информацией о бэктесте"""
    id: str
    name: str
    description: Optional[str]
    status: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: Optional[float]
    total_pnl: Optional[float]
    total_pnl_pct: Optional[float]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    progress: Optional[float]
    error_message: Optional[str]
    metrics: Optional[Dict[str, Any]]


class BacktestListResponse(BaseModel):
    """Response списка бэктестов"""
    runs: List[BacktestRunResponse]
    total: int
    page: int
    page_size: int


class TradeResponse(BaseModel):
    """Response одной сделки"""
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    duration_seconds: float
    exit_reason: str


class EquityPointResponse(BaseModel):
    """Response точки на кривой доходности"""
    timestamp: datetime
    equity: float
    cash: float
    positions_value: float
    drawdown: float
    drawdown_pct: float
    total_return: float
    total_return_pct: float
    open_positions_count: int


class StatisticsResponse(BaseModel):
    """Response агрегированной статистики"""
    total_backtests: int
    completed_backtests: int
    running_backtests: int
    failed_backtests: int
    avg_total_return_pct: float
    avg_sharpe_ratio: float
    avg_max_drawdown_pct: float
    avg_win_rate_pct: float
    best_backtest: Optional[Dict[str, Any]]
    worst_backtest: Optional[Dict[str, Any]]


# ============================================================
# Global State
# ============================================================

# Repository instance
repository = BacktestRepository()

# Background task tracking
running_backtests: Dict[str, Any] = {}


# ============================================================
# Backtest Management Endpoints
# ============================================================

@router.post("/runs")
async def create_backtest(
    request: CreateBacktestRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Создать и запустить новый бэктест

    Args:
        request: Параметры бэктеста
        background_tasks: FastAPI background tasks

    Returns:
        ID и статус созданного бэктеста
    """
    logger.info(f"📊 Создание бэктеста: {request.name} ({request.symbol})")

    try:
        # Создать конфигурацию
        exchange_config = ExchangeConfig(
            commission_rate=request.commission_rate,
            maker_commission=request.maker_commission,
            taker_commission=request.taker_commission,
            slippage_model=SlippageModel(request.slippage_model),
            slippage_pct=request.slippage_pct,
            simulate_latency=request.simulate_latency
        )

        strategy_config = StrategyConfig(
            enabled_strategies=request.enabled_strategies,
            strategy_params=request.strategy_params,
            consensus_mode=request.consensus_mode,
            min_strategies_for_signal=request.min_strategies_for_signal,
            min_consensus_confidence=request.min_consensus_confidence,
            strategy_weights=request.strategy_weights
        )

        risk_config = RiskConfig(
            position_size_pct=request.position_size_pct,
            position_size_mode=request.position_size_mode,
            max_open_positions=request.max_open_positions,
            stop_loss_pct=request.stop_loss_pct,
            take_profit_pct=request.take_profit_pct,
            use_trailing_stop=request.use_trailing_stop,
            trailing_stop_activation_pct=request.trailing_stop_activation_pct,
            trailing_stop_distance_pct=request.trailing_stop_distance_pct,
            risk_per_trade_pct=request.risk_per_trade_pct
        )

        backtest_config = BacktestConfig(
            name=request.name,
            description=request.description,
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            exchange_config=exchange_config,
            strategy_config=strategy_config,
            risk_config=risk_config,
            candle_interval=request.candle_interval,
            use_orderbook_data=request.use_orderbook_data,
            warmup_period_bars=request.warmup_period_bars,
            verbose=request.verbose
        )

        # Сохранить в БД
        backtest_run = await repository.create_run(
            name=request.name,
            description=request.description,
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            exchange_config=exchange_config.__dict__,
            strategies_config={
                "enabled_strategies": strategy_config.enabled_strategies,
                "strategy_params": strategy_config.strategy_params,
                "consensus_mode": strategy_config.consensus_mode,
                "min_strategies_for_signal": strategy_config.min_strategies_for_signal,
                "min_consensus_confidence": strategy_config.min_consensus_confidence,
                "strategy_weights": strategy_config.strategy_weights
            },
            risk_config=risk_config.__dict__
        )

        backtest_id = str(backtest_run.id)

        logger.info(f"✅ Бэктест создан: {backtest_id}")

        # Запустить в background
        background_tasks.add_task(
            _run_backtest_job,
            backtest_id=backtest_id,
            config=backtest_config
        )

        return {
            "id": backtest_id,
            "name": request.name,
            "status": "pending",
            "message": "Бэктест создан и запущен в фоновом режиме",
            "created_at": backtest_run.created_at.isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Ошибка создания бэктеста: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка создания бэктеста: {str(e)}")


@router.get("/runs")
async def list_backtests(
    status: Optional[str] = Query(None, description="Фильтр по статусу"),
    symbol: Optional[str] = Query(None, description="Фильтр по символу"),
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(20, ge=1, le=100, description="Размер страницы")
) -> BacktestListResponse:
    """
    Получить список бэктестов с фильтрацией и пагинацией

    Args:
        status: Фильтр по статусу (pending, running, completed, failed, cancelled)
        symbol: Фильтр по торговой паре
        page: Номер страницы
        page_size: Размер страницы

    Returns:
        Список бэктестов
    """
    try:
        # Валидация статуса
        if status and status not in ["pending", "running", "completed", "failed", "cancelled"]:
            raise HTTPException(status_code=400, detail="Недопустимый статус")

        # Конвертировать page в offset
        offset = (page - 1) * page_size

        # Получить список из БД
        runs = await repository.list_runs(
            status=BacktestStatus(status) if status else None,
            symbol=symbol,
            limit=page_size,
            offset=offset
        )

        # Преобразовать в response
        run_responses = []
        for run in runs:
            run_responses.append(BacktestRunResponse(
                id=str(run.id),
                name=run.name,
                description=run.description,
                status=run.status.value,
                symbol=run.symbol,
                start_date=run.start_date,
                end_date=run.end_date,
                initial_capital=run.initial_capital,
                final_capital=run.final_capital,
                total_pnl=run.total_pnl,
                total_pnl_pct=run.total_pnl_pct,
                created_at=run.created_at,
                started_at=run.started_at,
                completed_at=run.completed_at,
                progress=run.progress_pct,
                error_message=run.error_message,
                metrics=run.metrics
            ))

        # Подсчитать общее количество
        total = await repository.count_runs(status=BacktestStatus(status) if status else None, symbol=symbol)

        return BacktestListResponse(
            runs=run_responses,
            total=total,
            page=page,
            page_size=page_size
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения списка бэктестов: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{backtest_id}")
async def get_backtest(
    backtest_id: UUID,
    include_trades: bool = Query(False, description="Включить сделки"),
    include_equity: bool = Query(False, description="Включить equity curve")
) -> Dict[str, Any]:
    """
    Получить детальную информацию о бэктесте

    Args:
        backtest_id: ID бэктеста
        include_trades: Включить сделки в ответ
        include_equity: Включить equity curve в ответ

    Returns:
        Детальная информация о бэктесте
    """
    try:
        # Получить из БД
        run = await repository.get_by_id(
            backtest_id,
            include_trades=include_trades,
            include_equity=include_equity
        )

        if not run:
            raise HTTPException(status_code=404, detail=f"Бэктест {backtest_id} не найден")

        # Базовая информация
        response = {
            "id": str(run.id),
            "name": run.name,
            "description": run.description,
            "status": run.status.value,
            "symbol": run.symbol,
            "start_date": run.start_date.isoformat(),
            "end_date": run.end_date.isoformat(),
            "initial_capital": run.initial_capital,
            "final_capital": run.final_capital,
            "total_pnl": run.total_pnl,
            "total_pnl_pct": run.total_pnl_pct,
            "created_at": run.created_at.isoformat(),
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "progress": run.progress_pct,
            "error_message": run.error_message,
            "exchange_config": run.exchange_config,
            "strategies_config": run.strategies_config,
            "risk_config": run.risk_config,
            "metrics": run.metrics
        }

        # Добавить сделки если запрошены
        if include_trades and run.trades:
            response["trades"] = [
                {
                    "symbol": trade.symbol,
                    "side": trade.side.value,
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "quantity": trade.quantity,
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                    "commission": trade.commission,
                    "duration_seconds": trade.duration_seconds,
                    "exit_reason": trade.exit_reason
                }
                for trade in run.trades
            ]

        # Добавить equity curve если запрошена
        if include_equity and run.equity_curve:
            response["equity_curve"] = [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "equity": point.equity,
                    "cash": point.cash,
                    "positions_value": point.positions_value,
                    "drawdown": point.drawdown,
                    "drawdown_pct": point.drawdown_pct,
                    "total_return": point.total_return,
                    "total_return_pct": point.total_return_pct,
                    "open_positions_count": point.open_positions_count
                }
                for point in run.equity_curve
            ]

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения бэктеста {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{backtest_id}/trades")
async def get_backtest_trades(
    backtest_id: UUID,
    limit: int = Query(100, ge=1, le=1000, description="Максимум сделок")
) -> Dict[str, Any]:
    """
    Получить сделки бэктеста

    Args:
        backtest_id: ID бэктеста
        limit: Максимальное количество сделок

    Returns:
        Список сделок
    """
    try:
        # Получить сделки из БД
        trades = await repository.get_trades(backtest_id, limit=limit)

        return {
            "backtest_id": str(backtest_id),
            "trades": [
                {
                    "symbol": trade.symbol,
                    "side": trade.side.value,
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "quantity": trade.quantity,
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                    "commission": trade.commission,
                    "duration_seconds": trade.duration_seconds,
                    "exit_reason": trade.exit_reason,
                    "max_favorable_excursion": trade.max_favorable_excursion,
                    "max_adverse_excursion": trade.max_adverse_excursion
                }
                for trade in trades
            ],
            "total": len(trades)
        }

    except Exception as e:
        logger.error(f"Ошибка получения сделок бэктеста {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{backtest_id}/equity-curve")
async def get_equity_curve(
    backtest_id: UUID,
    sampling_interval_minutes: int = Query(60, ge=1, description="Интервал выборки (минуты)")
) -> Dict[str, Any]:
    """
    Получить equity curve бэктеста

    Args:
        backtest_id: ID бэктеста
        sampling_interval_minutes: Интервал выборки точек (минуты)

    Returns:
        Equity curve
    """
    try:
        # Получить equity curve из БД
        equity_points = await repository.get_equity_curve(backtest_id)

        return {
            "backtest_id": str(backtest_id),
            "equity_curve": [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "sequence": point.sequence,
                    "equity": point.equity,
                    "cash": point.cash,
                    "positions_value": point.positions_value,
                    "drawdown": point.drawdown,
                    "drawdown_pct": point.drawdown_pct,
                    "total_return": point.total_return,
                    "total_return_pct": point.total_return_pct,
                    "open_positions_count": point.open_positions_count
                }
                for point in equity_points
            ],
            "total_points": len(equity_points)
        }

    except Exception as e:
        logger.error(f"Ошибка получения equity curve бэктеста {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/runs/{backtest_id}/cancel")
async def cancel_backtest(backtest_id: UUID) -> Dict[str, Any]:
    """
    Отменить выполняющийся бэктест

    Args:
        backtest_id: ID бэктеста

    Returns:
        Результат операции
    """
    try:
        # Проверить существование
        run = await repository.get_by_id(backtest_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Бэктест {backtest_id} не найден")

        # Проверить статус
        if run.status not in [BacktestStatus.PENDING, BacktestStatus.RUNNING]:
            raise HTTPException(
                status_code=400,
                detail=f"Нельзя отменить бэктест в статусе {run.status.value}"
            )

        # Обновить статус
        await repository.update_status(backtest_id, BacktestStatus.CANCELLED)

        # Остановить background task если запущен
        if str(backtest_id) in running_backtests:
            running_backtests[str(backtest_id)]["cancelled"] = True

        logger.info(f"🚫 Бэктест отменен: {backtest_id}")

        return {
            "success": True,
            "backtest_id": str(backtest_id),
            "message": "Бэктест отменен"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка отмены бэктеста {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/runs/{backtest_id}")
async def delete_backtest(backtest_id: UUID) -> Dict[str, Any]:
    """
    Удалить бэктест

    Args:
        backtest_id: ID бэктеста

    Returns:
        Результат операции
    """
    try:
        # Проверить существование
        run = await repository.get_by_id(backtest_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Бэктест {backtest_id} не найден")

        # Нельзя удалить запущенный бэктест
        if run.status == BacktestStatus.RUNNING:
            raise HTTPException(
                status_code=400,
                detail="Нельзя удалить выполняющийся бэктест. Сначала отмените его."
            )

        # Удалить из БД
        await repository.delete_run(backtest_id)

        logger.info(f"🗑️ Бэктест удален: {backtest_id}")

        return {
            "success": True,
            "backtest_id": str(backtest_id),
            "message": "Бэктест удален"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка удаления бэктеста {backtest_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics() -> StatisticsResponse:
    """
    Получить агрегированную статистику по всем бэктестам

    Returns:
        Статистика
    """
    try:
        stats = await repository.get_statistics()

        return StatisticsResponse(
            total_backtests=stats["total_backtests"],
            completed_backtests=stats["completed_backtests"],
            running_backtests=stats["running_backtests"],
            failed_backtests=stats["failed_backtests"],
            avg_total_return_pct=stats["avg_total_return_pct"],
            avg_sharpe_ratio=stats["avg_sharpe_ratio"],
            avg_max_drawdown_pct=stats["avg_max_drawdown_pct"],
            avg_win_rate_pct=stats["avg_win_rate_pct"],
            best_backtest=stats.get("best_backtest"),
            worst_backtest=stats.get("worst_backtest")
        )

    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Configuration Management
# ============================================================

@router.get("/config/defaults")
async def get_default_config() -> Dict[str, Any]:
    """
    Получить дефолтные настройки для конфигурации бэктеста

    Returns:
        Дефолтные значения для всех параметров конфигурации
    """
    try:
        defaults = {
            # Основные настройки
            "symbol": "BTCUSDT",
            "candle_interval": "1m",
            "initial_capital": 10000.0,
            "warmup_period_bars": 100,

            # Биржа
            "commission_rate": 0.001,  # 0.1%
            "maker_commission": 0.0002,  # 0.02%
            "taker_commission": 0.001,  # 0.1%
            "slippage_model": "fixed",
            "slippage_pct": 0.1,  # 0.1%
            "simulate_latency": False,

            # Стратегии и консенсус
            "enabled_strategies": ["momentum", "sar_wave", "supertrend", "volume_profile"],
            "consensus_mode": "weighted",
            "min_strategies_for_signal": 2,
            "min_consensus_confidence": 0.5,
            "strategy_weights": {},
            "strategy_params": {},

            # Риск-менеджмент
            "position_size_pct": 10.0,
            "position_size_mode": "fixed_percent",
            "max_open_positions": 3,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0,
            "use_trailing_stop": False,
            "trailing_stop_activation_pct": 2.0,
            "trailing_stop_distance_pct": 1.0,
            "risk_per_trade_pct": 1.0,

            # OrderBook & Market Trades (Phase 1)
            "use_orderbook_data": False,
            "orderbook_num_levels": 20,
            "orderbook_base_spread_bps": 2.0,
            "use_market_trades": False,
            "trades_per_volume_unit": 100,

            # ML Model (Phase 2)
            "use_ml_model": False,
            "ml_server_url": "http://localhost:8001",
            "ml_model_name": None,
            "ml_model_version": None,

            # Optimization (Phase 3)
            "use_cache": True,
            "skip_orderbook_generation_every_n": None,
            "skip_trades_generation_every_n": None,

            # Отладка
            "verbose": False,
            "log_trades": False
        }

        return defaults

    except Exception as e:
        logger.error(f"Ошибка получения дефолтных настроек: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/validate")
async def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Валидировать конфигурацию бэктеста

    Args:
        config: Конфигурация для валидации

    Returns:
        Результат валидации с ошибками (если есть)
    """
    try:
        errors = []

        # Валидация обязательных полей
        if not config.get("name"):
            errors.append("Название теста обязательно")

        if not config.get("start_date"):
            errors.append("Дата начала обязательна")

        if not config.get("end_date"):
            errors.append("Дата окончания обязательна")

        # Валидация дат
        if config.get("start_date") and config.get("end_date"):
            try:
                start = datetime.fromisoformat(config["start_date"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(config["end_date"].replace("Z", "+00:00"))

                if start >= end:
                    errors.append("Дата окончания должна быть позже даты начала")

                if start > datetime.now():
                    errors.append("Дата начала не может быть в будущем")

                if end > datetime.now():
                    errors.append("Дата окончания не может быть в будущем")

            except (ValueError, TypeError) as e:
                errors.append(f"Некорректный формат даты: {str(e)}")

        # Валидация числовых значений
        if config.get("initial_capital") and config["initial_capital"] <= 0:
            errors.append("Начальный капитал должен быть больше 0")

        if config.get("position_size_pct"):
            if config["position_size_pct"] < 0.1 or config["position_size_pct"] > 100:
                errors.append("Размер позиции должен быть от 0.1% до 100%")

        if config.get("max_open_positions"):
            if config["max_open_positions"] < 1 or config["max_open_positions"] > 20:
                errors.append("Максимум открытых позиций должен быть от 1 до 20")

        if config.get("stop_loss_pct"):
            if config["stop_loss_pct"] < 0.1 or config["stop_loss_pct"] > 50:
                errors.append("Stop Loss должен быть от 0.1% до 50%")

        if config.get("take_profit_pct"):
            if config["take_profit_pct"] < 0.1 or config["take_profit_pct"] > 100:
                errors.append("Take Profit должен быть от 0.1% до 100%")

        if config.get("min_consensus_confidence"):
            if config["min_consensus_confidence"] < 0 or config["min_consensus_confidence"] > 1:
                errors.append("Уверенность консенсуса должна быть от 0 до 1")

        if config.get("min_strategies_for_signal"):
            if config["min_strategies_for_signal"] < 1:
                errors.append("Минимум стратегий для сигнала должен быть >= 1")

        # Валидация enum полей
        valid_slippage_models = ["fixed", "volume_based", "percentage"]
        if config.get("slippage_model") and config["slippage_model"] not in valid_slippage_models:
            errors.append(f"Модель проскальзывания должна быть одной из: {', '.join(valid_slippage_models)}")

        valid_consensus_modes = ["weighted", "majority", "unanimous"]
        if config.get("consensus_mode") and config["consensus_mode"] not in valid_consensus_modes:
            errors.append(f"Режим консенсуса должен быть одним из: {', '.join(valid_consensus_modes)}")

        valid_position_modes = ["fixed_percent", "risk_based", "volatility_adjusted"]
        if config.get("position_size_mode") and config["position_size_mode"] not in valid_position_modes:
            errors.append(f"Режим расчета позиции должен быть одним из: {', '.join(valid_position_modes)}")

        # Валидация OrderBook настроек
        if config.get("orderbook_num_levels"):
            if config["orderbook_num_levels"] < 10 or config["orderbook_num_levels"] > 50:
                errors.append("Количество уровней OrderBook должно быть от 10 до 50")

        if config.get("orderbook_base_spread_bps"):
            if config["orderbook_base_spread_bps"] < 0.1 or config["orderbook_base_spread_bps"] > 50:
                errors.append("Базовый спред должен быть от 0.1 до 50 bps")

        # Валидация ML настроек
        if config.get("use_ml_model") and not config.get("ml_server_url"):
            errors.append("При использовании ML модели необходимо указать URL сервера")

        # Результат
        if errors:
            return {
                "valid": False,
                "errors": errors
            }
        else:
            return {
                "valid": True,
                "errors": []
            }

    except Exception as e:
        logger.error(f"Ошибка валидации конфигурации: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Health Check
# ============================================================

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check для backtesting API

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "backtesting",
        "timestamp": datetime.now().isoformat(),
        "running_backtests": len([b for b in running_backtests.values() if not b.get("cancelled")])
    }


# ============================================================
# Background Job
# ============================================================

async def _run_backtest_job(backtest_id: str, config: BacktestConfig):
    """
    Background task для выполнения бэктеста

    Args:
        backtest_id: ID бэктеста (строка)
        config: Конфигурация бэктеста
    """
    # Конвертировать ID в UUID
    backtest_uuid = UUID(backtest_id)

    # Добавить в tracking
    running_backtests[backtest_id] = {
        "cancelled": False,
        "started_at": datetime.now()
    }

    try:
        logger.info(f"🚀 Запуск бэктеста: {backtest_id}")

        # Обновить статус на RUNNING
        await repository.update_status(backtest_uuid, BacktestStatus.RUNNING, progress_pct=0.0)

        # Создать компоненты
        data_handler = HistoricalDataHandler()
        simulated_exchange = SimulatedExchange(config.exchange_config)

        engine = BacktestingEngine(
            config=config,
            data_handler=data_handler,
            simulated_exchange=simulated_exchange
        )

        # Запустить бэктест
        result = await engine.run()

        # Проверка на отмену
        if running_backtests[backtest_id].get("cancelled"):
            logger.info(f"⏹️ Бэктест отменен: {backtest_id}")
            return

        # Сохранить результаты
        await repository.update_results(
            backtest_uuid,
            final_capital=result.final_capital,
            total_pnl=result.total_pnl,
            total_pnl_pct=result.total_pnl_pct,
            total_trades=len(result.trades),
            winning_trades=len([t for t in result.trades if t.pnl > 0]),
            losing_trades=len([t for t in result.trades if t.pnl < 0]),
            metrics=result.metrics.to_dict()
        )

        # Сохранить сделки
        for trade in result.trades:
            await repository.create_trade(
                backtest_run_id=backtest_uuid,
                symbol=trade.symbol,
                side=OrderSide(trade.side),  # Преобразовать строку в enum
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                pnl=trade.pnl,
                pnl_pct=trade.pnl_pct,
                commission=trade.commission,
                exit_reason=trade.exit_reason,
                max_favorable_excursion=trade.max_favorable_excursion,
                max_adverse_excursion=trade.max_adverse_excursion
            )

        # Сохранить equity curve
        for point in result.equity_curve:
            # Вычислить peak_equity из equity и drawdown
            peak_equity = point.equity + point.drawdown

            await repository.create_equity_point(
                backtest_run_id=backtest_uuid,
                timestamp=point.timestamp,
                sequence=point.sequence,
                equity=point.equity,
                cash=point.cash,
                positions_value=point.positions_value,
                peak_equity=peak_equity,  # Вычисленное значение
                drawdown=point.drawdown,
                drawdown_pct=point.drawdown_pct,
                total_return=point.total_return,
                total_return_pct=point.total_return_pct,
                open_positions_count=point.open_positions_count
            )

        # Обновить статус на COMPLETED
        await repository.update_status(backtest_uuid, BacktestStatus.COMPLETED, progress_pct=100.0)

        logger.info(
            f"✅ Бэктест завершен: {backtest_id}, "
            f"PnL={result.total_pnl:.2f} ({result.total_pnl_pct:.2f}%), "
            f"Sharpe={result.metrics.sharpe_ratio:.2f}"
        )

    except Exception as e:
        logger.error(f"❌ Ошибка выполнения бэктеста {backtest_id}: {e}", exc_info=True)

        # Обновить статус на FAILED
        await repository.update_status(
            backtest_uuid,
            BacktestStatus.FAILED,
            error_message=str(e)
        )

    finally:
        # Удалить из tracking
        if backtest_id in running_backtests:
            del running_backtests[backtest_id]
