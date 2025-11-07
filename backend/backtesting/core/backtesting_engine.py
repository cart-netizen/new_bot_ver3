"""
Backtesting Engine - главный движок event-driven бэктестинга.

Архитектура:
- Event-driven: избегает look-ahead bias
- Portfolio state tracking
- Real-time equity curve
- Complete trade history
- Performance metrics calculation

Flow:
1. Load historical data (candles, orderbook)
2. Initialize portfolio state
3. Event loop: for each time step
   - Update market data
   - Run strategies
   - Process signals
   - Execute orders
   - Update portfolio
   - Record equity
4. Calculate final metrics
5. Save results to database
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import uuid

from backend.core.logger import get_logger
from backend.strategy.candle_manager import Candle, CandleManager
from backend.strategies.strategy_manager import ExtendedStrategyManager, ExtendedStrategyManagerConfig
from backend.models.signal import TradingSignal, SignalType
from backend.models.orderbook import OrderBookSnapshot, OrderBookMetrics
from backend.database.models import OrderSide, OrderType

from backend.backtesting.models import (
    BacktestConfig,
    BacktestResult,
    TradeResult,
    EquityPoint,
    PerformanceMetrics
)
from backend.backtesting.core.data_handler import HistoricalDataHandler
from backend.backtesting.core.simulated_exchange import SimulatedExchange, SimulatedOrder

logger = get_logger(__name__)


@dataclass
class Position:
    """Открытая позиция в портфеле."""
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Tracking
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    max_favorable_excursion: float = 0.0  # MFE
    max_adverse_excursion: float = 0.0    # MAE

    # Entry context
    entry_signal: Optional[Dict] = None


@dataclass
class Portfolio:
    """Состояние портфеля бэктеста."""
    initial_capital: float
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    # Tracking
    equity_history: List[EquityPoint] = field(default_factory=list)
    peak_equity: float = 0.0

    def __post_init__(self):
        self.cash = self.initial_capital
        self.peak_equity = self.initial_capital

    @property
    def equity(self) -> float:
        """Текущий капитал (cash + positions value)."""
        positions_value = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    @property
    def positions_value(self) -> float:
        """Стоимость открытых позиций."""
        return sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )

    @property
    def total_return(self) -> float:
        """Абсолютная доходность."""
        return self.equity - self.initial_capital

    @property
    def total_return_pct(self) -> float:
        """Процентная доходность."""
        return (self.equity / self.initial_capital - 1) * 100

    def update_peak_equity(self):
        """Обновить пиковый капитал."""
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    @property
    def current_drawdown(self) -> float:
        """Текущая просадка."""
        if self.peak_equity == 0:
            return 0.0
        return self.peak_equity - self.equity

    @property
    def current_drawdown_pct(self) -> float:
        """Текущая просадка в процентах."""
        if self.peak_equity == 0:
            return 0.0
        return (self.current_drawdown / self.peak_equity) * 100


class BacktestingEngine:
    """
    Главный движок бэктестинга.

    Event-driven архитектура:
    - Обрабатывает исторические данные последовательно
    - Избегает look-ahead bias
    - Реалистичная симуляция биржи
    - Полное отслеживание портфеля
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_handler: HistoricalDataHandler,
        simulated_exchange: SimulatedExchange,
        strategy_manager: Optional[ExtendedStrategyManager] = None
    ):
        """
        Инициализация backtesting engine.

        Args:
            config: Конфигурация бэктеста
            data_handler: Handler для загрузки исторических данных
            simulated_exchange: Симулятор биржи
            strategy_manager: Менеджер стратегий (если None - создаст по конфигу)
        """
        self.config = config
        self.data_handler = data_handler
        self.exchange = simulated_exchange

        # Strategy Manager
        if strategy_manager is None:
            # Создать из конфигурации
            self.strategy_manager = self._create_strategy_manager_from_config()
        else:
            self.strategy_manager = strategy_manager

        # Portfolio state
        self.portfolio = Portfolio(initial_capital=config.initial_capital)

        # Candle Manager для хранения истории свечей
        self.candle_manager = CandleManager(
            max_candles=1000,  # Достаточно для индикаторов
            interval_minutes=self._parse_interval(config.candle_interval)
        )

        # Trade tracking
        self.closed_trades: List[TradeResult] = []

        # Current market state
        self.current_time: Optional[datetime] = None
        self.current_price: float = 0.0
        self.current_orderbook: Optional[OrderBookSnapshot] = None

        # Progress tracking
        self.total_candles = 0
        self.processed_candles = 0

        logger.info(
            f"BacktestingEngine инициализирован: {config.symbol} "
            f"{config.start_date.strftime('%Y-%m-%d')} → {config.end_date.strftime('%Y-%m-%d')}"
        )

    def _create_strategy_manager_from_config(self) -> ExtendedStrategyManager:
        """Создать Strategy Manager из конфигурации."""
        strategy_config = ExtendedStrategyManagerConfig(
            consensus_mode=self.config.strategy_config.consensus_mode,
            min_strategies_for_signal=self.config.strategy_config.min_strategies_for_signal,
            min_consensus_confidence=self.config.strategy_config.min_consensus_confidence
        )

        # TODO: Настроить параметры стратегий из config.strategy_config.strategy_params

        return ExtendedStrategyManager(strategy_config)

    async def run(self) -> BacktestResult:
        """
        Запустить бэктест.

        Returns:
            BacktestResult с полными результатами
        """
        backtest_id = str(uuid.uuid4())
        started_at = datetime.now()

        logger.info(f"🚀 Запуск бэктеста: {self.config.name} (ID: {backtest_id})")

        try:
            # 1. Загрузка исторических данных
            logger.info("📊 Загрузка исторических данных...")
            candles = await self.data_handler.get_candles(
                symbol=self.config.symbol,
                start=self.config.start_date,
                end=self.config.end_date,
                interval=self.config.candle_interval
            )

            if not candles:
                raise ValueError("Не удалось загрузить исторические данные")

            logger.info(f"✅ Загружено {len(candles)} свечей")

            # 2. Валидация данных
            logger.info("🔍 Валидация качества данных...")
            quality_report = await self.data_handler.validate_data_quality(
                candles,
                interval_minutes=self._parse_interval(self.config.candle_interval)
            )

            if not quality_report.is_valid:
                logger.warning(
                    f"⚠️ Качество данных низкое (score: {quality_report.quality_score:.1f}): "
                    f"{', '.join(quality_report.issues)}"
                )

            # 3. Warmup period (прогрев индикаторов)
            logger.info(f"🔥 Warmup period: {self.config.warmup_period_bars} свечей")
            warmup_candles = candles[:self.config.warmup_period_bars]
            for candle in warmup_candles:
                self.candle_manager.update_candle(candle)

            # 4. Main backtest loop
            logger.info("🔄 Запуск main event loop...")
            self.total_candles = len(candles) - self.config.warmup_period_bars
            test_candles = candles[self.config.warmup_period_bars:]

            for candle in test_candles:
                await self._process_candle(candle)
                self.processed_candles += 1

                # Log progress every 100 candles
                if self.processed_candles % 100 == 0:
                    progress = (self.processed_candles / self.total_candles) * 100
                    logger.info(
                        f"⏳ Progress: {progress:.1f}% ({self.processed_candles}/{self.total_candles}), "
                        f"Equity: ${self.portfolio.equity:.2f}, "
                        f"Open positions: {len(self.portfolio.positions)}"
                    )

            # 5. Close all remaining positions
            logger.info("🔚 Закрытие всех оставшихся позиций...")
            await self._close_all_positions(reason="END_OF_BACKTEST")

            # 6. Calculate final metrics
            logger.info("📈 Расчет финальных метрик...")
            metrics = self._calculate_performance_metrics()

            # 7. Create result
            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()

            result = BacktestResult(
                backtest_id=backtest_id,
                config=self.config,
                final_capital=self.portfolio.equity,
                total_pnl=self.portfolio.total_return,
                total_pnl_pct=self.portfolio.total_return_pct,
                metrics=metrics,
                trades=self.closed_trades,
                equity_curve=self.portfolio.equity_history,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                success=True
            )

            logger.info(
                f"✅ Бэктест завершен! "
                f"Final Capital: ${result.final_capital:.2f}, "
                f"PnL: ${result.total_pnl:.2f} ({result.total_pnl_pct:.2f}%), "
                f"Trades: {len(self.closed_trades)}, "
                f"Duration: {duration:.1f}s"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Ошибка выполнения бэктеста: {e}", exc_info=True)

            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()

            # Return failed result
            return BacktestResult(
                backtest_id=backtest_id,
                config=self.config,
                final_capital=self.portfolio.equity,
                total_pnl=self.portfolio.total_return,
                total_pnl_pct=self.portfolio.total_return_pct,
                metrics=PerformanceMetrics(
                    total_return=self.portfolio.total_return,
                    total_return_pct=self.portfolio.total_return_pct,
                    annual_return_pct=0.0
                ),
                trades=self.closed_trades,
                equity_curve=self.portfolio.equity_history,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                success=False,
                error_message=str(e)
            )

    async def _process_candle(self, candle: Candle):
        """
        Обработать одну свечу (один тик времени).

        Steps:
        1. Update candle manager
        2. Update market state
        3. Process limit orders (simulated exchange)
        4. Update open positions
        5. Check stop loss / take profit
        6. Run strategies
        7. Process signals
        8. Record equity
        """
        # 1. Update candle manager
        self.candle_manager.update_candle(candle)

        # 2. Update current market state
        self.current_time = datetime.fromtimestamp(candle.timestamp / 1000)
        self.current_price = candle.close

        # 3. Process limit orders
        await self.exchange.process_tick(
            current_time=self.current_time,
            current_price=self.current_price,
            orderbook=self.current_orderbook
        )

        # 4. Update positions
        self._update_positions(self.current_price)

        # 5. Check stop loss / take profit
        await self._check_stop_loss_take_profit()

        # 6. Run strategies (only if we can open new positions)
        if len(self.portfolio.positions) < self.config.risk_config.max_open_positions:
            signal = await self._run_strategies()

            # 7. Process signal
            if signal:
                await self._process_signal(signal)

        # 8. Record equity (periodically)
        if self.processed_candles % 60 == 0:  # Every 60 candles
            self._record_equity_point()

    async def _run_strategies(self) -> Optional[TradingSignal]:
        """Запустить стратегии и получить сигнал."""
        candles = self.candle_manager.get_candles()

        if len(candles) < 20:
            # Недостаточно данных для стратегий
            return None

        # Run strategy manager
        consensus = await asyncio.to_thread(
            self.strategy_manager.analyze_with_consensus,
            symbol=self.config.symbol,
            candles=candles,
            current_price=self.current_price,
            orderbook=self.current_orderbook,
            metrics=None,  # TODO: Calculate OrderBookMetrics if needed
            sr_levels=None,
            volume_profile=None,
            ml_prediction=None
        )

        if consensus:
            return consensus.final_signal

        return None

    async def _process_signal(self, signal: TradingSignal):
        """Обработать торговый сигнал."""
        # Check if we already have a position for this symbol
        if signal.symbol in self.portfolio.positions:
            logger.debug(f"Позиция по {signal.symbol} уже открыта, пропуск сигнала")
            return

        # Calculate position size
        position_size_usdt = self.portfolio.cash * (self.config.risk_config.position_size_pct / 100)
        quantity = position_size_usdt / self.current_price

        # Determine side
        side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL

        # Place order
        order = await self.exchange.place_order(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            current_price=self.current_price,
            orderbook=self.current_orderbook
        )

        # If filled, create position
        if order.status.value == "filled":
            await self._open_position(order, signal)

    async def _open_position(self, order: SimulatedOrder, signal: TradingSignal):
        """Открыть позицию после исполнения ордера."""
        # Calculate stop loss and take profit
        entry_price = order.average_fill_price

        if order.side == OrderSide.BUY:
            stop_loss = entry_price * (1 - self.config.risk_config.stop_loss_pct / 100)
            take_profit = entry_price * (1 + self.config.risk_config.take_profit_pct / 100)
        else:
            stop_loss = entry_price * (1 + self.config.risk_config.stop_loss_pct / 100)
            take_profit = entry_price * (1 - self.config.risk_config.take_profit_pct / 100)

        # Create position
        position = Position(
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            entry_price=entry_price,
            entry_time=self.current_time,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=self.current_price,
            entry_signal={
                'signal_type': signal.signal_type.value,
                'source': signal.source.value,
                'confidence': signal.confidence,
                'reason': signal.reason
            }
        )

        # Update portfolio
        self.portfolio.positions[order.symbol] = position
        self.portfolio.cash -= (order.filled_quantity * entry_price + order.commission)

        logger.info(
            f"✅ Позиция открыта: {order.side.value} {order.filled_quantity} {order.symbol} @ "
            f"{entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}"
        )

    def _update_positions(self, current_price: float):
        """Обновить состояние всех открытых позиций."""
        for symbol, position in self.portfolio.positions.items():
            position.current_price = current_price

            # Calculate unrealized PnL
            if position.side == OrderSide.BUY:
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity

            # Track MFE and MAE
            if position.unrealized_pnl > position.max_favorable_excursion:
                position.max_favorable_excursion = position.unrealized_pnl

            if position.unrealized_pnl < position.max_adverse_excursion:
                position.max_adverse_excursion = position.unrealized_pnl

    async def _check_stop_loss_take_profit(self):
        """Проверить stop loss и take profit для всех позиций."""
        positions_to_close = []

        for symbol, position in self.portfolio.positions.items():
            should_close = False
            exit_reason = None

            if position.side == OrderSide.BUY:
                # Long position
                if position.current_price <= position.stop_loss:
                    should_close = True
                    exit_reason = "SL"
                elif position.current_price >= position.take_profit:
                    should_close = True
                    exit_reason = "TP"
            else:
                # Short position
                if position.current_price >= position.stop_loss:
                    should_close = True
                    exit_reason = "SL"
                elif position.current_price <= position.take_profit:
                    should_close = True
                    exit_reason = "TP"

            if should_close:
                positions_to_close.append((symbol, exit_reason))

        # Close positions
        for symbol, exit_reason in positions_to_close:
            await self._close_position(symbol, exit_reason)

    async def _close_position(self, symbol: str, exit_reason: str):
        """Закрыть позицию."""
        if symbol not in self.portfolio.positions:
            return

        position = self.portfolio.positions[symbol]

        # Place closing order (opposite side)
        close_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY

        order = await self.exchange.place_order(
            symbol=symbol,
            side=close_side,
            quantity=position.quantity,
            order_type=OrderType.MARKET,
            current_price=self.current_price,
            orderbook=self.current_orderbook
        )

        if order.status.value == "filled":
            # Calculate PnL
            exit_price = order.average_fill_price

            if position.side == OrderSide.BUY:
                pnl = (exit_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - exit_price) * position.quantity

            pnl -= order.commission  # Subtract commission
            pnl_pct = (pnl / (position.entry_price * position.quantity)) * 100

            # Create trade result
            duration = (self.current_time - position.entry_time).total_seconds()

            trade = TradeResult(
                symbol=symbol,
                side=position.side.value,
                entry_time=position.entry_time,
                exit_time=self.current_time,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=order.commission,
                duration_seconds=duration,
                exit_reason=exit_reason,
                max_favorable_excursion=position.max_favorable_excursion,
                max_adverse_excursion=position.max_adverse_excursion,
                entry_signal=position.entry_signal,
                exit_signal=None
            )

            self.closed_trades.append(trade)

            # Update portfolio
            self.portfolio.cash += (position.quantity * exit_price - order.commission)
            del self.portfolio.positions[symbol]

            logger.info(
                f"🔴 Позиция закрыта: {position.side.value} {position.quantity} {symbol} @ "
                f"{exit_price:.2f}, PnL: ${pnl:.2f} ({pnl_pct:+.2f}%), Reason: {exit_reason}"
            )

    async def _close_all_positions(self, reason: str = "END_OF_BACKTEST"):
        """Закрыть все оставшиеся позиции."""
        symbols = list(self.portfolio.positions.keys())
        for symbol in symbols:
            await self._close_position(symbol, reason)

    def _record_equity_point(self):
        """Записать точку equity curve."""
        self.portfolio.update_peak_equity()

        point = EquityPoint(
            timestamp=self.current_time,
            sequence=len(self.portfolio.equity_history),
            equity=self.portfolio.equity,
            cash=self.portfolio.cash,
            positions_value=self.portfolio.positions_value,
            drawdown=self.portfolio.current_drawdown,
            drawdown_pct=self.portfolio.current_drawdown_pct,
            total_return=self.portfolio.total_return,
            total_return_pct=self.portfolio.total_return_pct,
            open_positions_count=len(self.portfolio.positions)
        )

        self.portfolio.equity_history.append(point)

    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Расчет финальных метрик производительности."""
        # Basic metrics
        total_trades = len(self.closed_trades)
        winning_trades = len([t for t in self.closed_trades if t.pnl > 0])
        losing_trades = len([t for t in self.closed_trades if t.pnl < 0])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        # PnL statistics
        winning_pnls = [t.pnl for t in self.closed_trades if t.pnl > 0]
        losing_pnls = [t.pnl for t in self.closed_trades if t.pnl < 0]

        avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0.0
        avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0.0

        largest_win = max(winning_pnls) if winning_pnls else 0.0
        largest_loss = min(losing_pnls) if losing_pnls else 0.0

        # Profit factor
        total_wins = sum(winning_pnls)
        total_losses = abs(sum(losing_pnls))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # TODO: Calculate advanced metrics (Sharpe, Sortino, etc.)
        # This will be implemented in Performance Analyzer

        return PerformanceMetrics(
            total_return=self.portfolio.total_return,
            total_return_pct=self.portfolio.total_return_pct,
            annual_return_pct=0.0,  # TODO: Calculate annualized return
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            max_drawdown_pct=0.0,  # TODO: Calculate from equity curve
        )

    def _parse_interval(self, interval: str) -> int:
        """Parse interval string to minutes."""
        # Simple implementation
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval == '1':
            return 1
        else:
            return 1  # Default to 1 minute
