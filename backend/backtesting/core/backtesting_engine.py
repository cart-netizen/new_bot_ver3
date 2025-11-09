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
from backend.backtesting.core.orderbook_data_handler import OrderBookDataHandler, OrderBookSimulationConfig
from backend.backtesting.core.trade_data_handler import TradeDataHandler, TradeSimulationConfig
from backend.backtesting.metrics.advanced_metrics import AdvancedMetricsCalculator, TradeData, EquityData
from backend.models.market_data import MarketTrade

# ML Engine интеграция (НОВОЕ в Фазе 2)
from backend.ml_engine.features.feature_pipeline import FeaturePipeline, FeatureVector
from backend.ml_engine.inference.model_client import ModelClient

# Оптимизация (НОВОЕ в Фазе 3)
from backend.backtesting.core.data_cache import DataCache

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
    cash: float = 0.0  # Будет установлен в __post_init__
    positions: Dict[str, Position] = field(default_factory=dict)

    # Tracking
    equity_history: List[EquityPoint] = field(default_factory=list)
    peak_equity: float = 0.0

    # FIX: Memory leak prevention - limit equity history size
    max_equity_points: int = 5000  # Maximum points before downsampling
    downsample_to: int = 2500  # Downsample to this many points

    def __post_init__(self):
        if self.cash == 0.0:  # Если не задан вручную
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

        # OrderBook and Trades handlers (НОВОЕ в Фазе 1)
        if config.use_orderbook_data:
            orderbook_config = OrderBookSimulationConfig(
                num_levels=config.orderbook_num_levels,
                base_spread_bps=config.orderbook_base_spread_bps
            )
            self.orderbook_handler = OrderBookDataHandler(orderbook_config)
            logger.info("✅ OrderBook симулятор включен")
        else:
            self.orderbook_handler = None

        if config.use_market_trades:
            trade_config = TradeSimulationConfig(
                trades_per_volume_unit=config.trades_per_volume_unit
            )
            self.trade_handler = TradeDataHandler(trade_config)
            logger.info("✅ Market Trades симулятор включен")
        else:
            self.trade_handler = None

        # ML Model интеграция (НОВОЕ в Фазе 2)
        self.feature_pipeline: Optional[FeaturePipeline] = None
        self.ml_client: Optional[ModelClient] = None

        if config.use_ml_model:
            # Инициализация feature pipeline
            self.feature_pipeline = FeaturePipeline(
                symbol=config.symbol,
                normalize=True,  # Включить нормализацию признаков
                cache_enabled=False,  # Не использовать Redis cache в бэктесте
                trade_manager=None  # Не нужен в бэктесте
            )

            # Инициализация ML client (будет подключаться к серверу при первом predict)
            self.ml_client = ModelClient(server_url=config.ml_server_url)

            logger.info(
                f"✅ ML Model интеграция включена: server={config.ml_server_url}, "
                f"model={config.ml_model_name or 'default'}"
            )
        else:
            logger.info("ℹ️ ML Model интеграция отключена")

        # Data Cache (НОВОЕ в Фазе 3)
        self.data_cache = DataCache() if config.use_cache else None
        if self.data_cache:
            logger.info("✅ Data Cache включен для ускорения повторных запусков")

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
            symbol=config.symbol,
            timeframe=f"{config.candle_interval}m",  # "1m", "5m", etc.
            max_candles=1000  # Достаточно для индикаторов
        )

        # Trade tracking
        self.closed_trades: List[TradeResult] = []

        # Current market state
        self.current_time: Optional[datetime] = None
        self.current_price: float = 0.0
        self.current_orderbook: Optional[OrderBookSnapshot] = None
        self.current_market_trades: List[MarketTrade] = []  # НОВОЕ

        # Исторические данные (НОВОЕ)
        self.historical_orderbooks: List[OrderBookSnapshot] = []
        self.historical_trades: List[MarketTrade] = []

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

        # Создаем менеджер
        strategy_manager = ExtendedStrategyManager(strategy_config)

        # Настроить параметры стратегий из config.strategy_config.strategy_params
        if self.config.strategy_config.strategy_params:
            for strategy_name, params in self.config.strategy_config.strategy_params.items():
                if strategy_name in strategy_manager.all_strategies:
                    strategy_instance = strategy_manager.all_strategies[strategy_name]

                    # Обновляем параметры config объекта стратегии
                    if hasattr(strategy_instance, 'config'):
                        for param_name, param_value in params.items():
                            if hasattr(strategy_instance.config, param_name):
                                setattr(strategy_instance.config, param_name, param_value)
                                logger.debug(
                                    f"Установлен параметр {strategy_name}.{param_name} = {param_value}"
                                )
                            else:
                                logger.warning(
                                    f"Параметр {param_name} не найден в {strategy_name}.config"
                                )
                    else:
                        logger.warning(f"Стратегия {strategy_name} не имеет config объекта")
                else:
                    logger.warning(f"Стратегия {strategy_name} не найдена в менеджере")

        return strategy_manager

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

            # 1.5. Генерация OrderBook и Market Trades (ОБНОВЛЕНО в Фазе 3 - с кэшированием)
            if self.config.use_orderbook_data and self.orderbook_handler:
                # Попытка загрузить из кэша
                orderbook_config_params = {
                    "num_levels": self.config.orderbook_num_levels,
                    "base_spread_bps": self.config.orderbook_base_spread_bps
                }

                if self.data_cache:
                    self.historical_orderbooks = self.data_cache.load_orderbooks(
                        symbol=self.config.symbol,
                        start_date=self.config.start_date,
                        end_date=self.config.end_date,
                        interval=self.config.candle_interval,
                        config_params=orderbook_config_params
                    )

                if not self.historical_orderbooks:
                    # Генерируем если не нашли в кэше
                    logger.info("📚 Генерация orderbook snapshots из свечей...")
                    self.historical_orderbooks = self.orderbook_handler.generate_orderbook_sequence(
                        candles, self.config.symbol
                    )
                    logger.info(f"✅ Сгенерировано {len(self.historical_orderbooks)} orderbook snapshots")

                    # Сохраняем в кэш для следующих запусков
                    if self.data_cache:
                        self.data_cache.save_orderbooks(
                            self.historical_orderbooks,
                            symbol=self.config.symbol,
                            start_date=self.config.start_date,
                            end_date=self.config.end_date,
                            interval=self.config.candle_interval,
                            config_params=orderbook_config_params
                        )

            if self.config.use_market_trades and self.trade_handler:
                # Попытка загрузить из кэша
                trades_config_params = {
                    "trades_per_volume_unit": self.config.trades_per_volume_unit
                }

                if self.data_cache:
                    self.historical_trades = self.data_cache.load_market_trades(
                        symbol=self.config.symbol,
                        start_date=self.config.start_date,
                        end_date=self.config.end_date,
                        interval=self.config.candle_interval,
                        config_params=trades_config_params
                    )

                if not self.historical_trades:
                    # Генерируем если не нашли в кэше
                    logger.info("💱 Генерация market trades из свечей...")
                    self.historical_trades = self.trade_handler.generate_trades_from_candles(
                        candles, self.config.symbol
                    )
                    logger.info(f"✅ Сгенерировано {len(self.historical_trades)} market trades")

                    # Сохраняем в кэш для следующих запусков
                    if self.data_cache:
                        self.data_cache.save_market_trades(
                            self.historical_trades,
                            symbol=self.config.symbol,
                            start_date=self.config.start_date,
                            end_date=self.config.end_date,
                            interval=self.config.candle_interval,
                            config_params=trades_config_params
                        )

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
                # Преобразуем Candle в список для CandleManager
                candle_data = [candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume]
                await self.candle_manager.update_candle(candle_data, is_closed=True)

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

        finally:
            # Cleanup ML client connection
            if self.ml_client:
                await self.ml_client.cleanup()
                logger.info("ML client connection closed")

    async def _process_candle(self, candle: Candle):
        """
        Обработать одну свечу (один тик времени).

        Steps:
        1. Update candle manager
        2. Update market state (включая orderbook и trades)
        3. Process limit orders (simulated exchange)
        4. Update open positions
        5. Check stop loss / take profit
        6. Run strategies
        7. Process signals
        8. Record equity
        """
        # 1. Update candle manager
        candle_data = [candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume]
        await self.candle_manager.update_candle(candle_data, is_closed=True)

        # 2. Update current market state
        self.current_time = datetime.fromtimestamp(candle.timestamp / 1000)
        self.current_price = candle.close

        # 2.1. Update current orderbook (НОВОЕ в Фазе 1)
        if self.historical_orderbooks:
            # Найти orderbook для текущей свечи
            matching_orderbook = next(
                (ob for ob in self.historical_orderbooks if ob.timestamp == candle.timestamp),
                None
            )
            if matching_orderbook:
                self.current_orderbook = matching_orderbook

        # 2.2. Update current market trades (НОВОЕ в Фазе 1)
        if self.historical_trades:
            # Найти trades для текущей свечи (60 секунд окно)
            candle_end_time = candle.timestamp + 60000  # +60 секунд
            self.current_market_trades = [
                t for t in self.historical_trades
                if candle.timestamp <= t.timestamp < candle_end_time
            ]

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
            consensus_result = await self._run_strategies()

            # 7. Process signal (передаем полный consensus)
            if consensus_result:
                await self._process_signal(consensus_result)

        # 8. Record equity (periodically)
        if self.processed_candles % 60 == 0:  # Every 60 candles
            self._record_equity_point()

    async def _run_strategies(self):
        """
        Запустить стратегии и получить консенсус.

        Returns:
            ConsensusSignal или None
        """
        candles = self.candle_manager.get_candles()

        if len(candles) < 20:
            # Недостаточно данных для стратегий
            return None

        # Calculate OrderBookMetrics if orderbook available
        orderbook_metrics = None
        if self.current_orderbook:
            orderbook_metrics = self._calculate_orderbook_metrics(self.current_orderbook)

        # Calculate ML prediction if model available (НОВОЕ в Фазе 2)
        ml_prediction = None
        if self.ml_client and self.feature_pipeline:
            ml_prediction = await self._get_ml_prediction(
                candles, self.current_orderbook, orderbook_metrics
            )

        # Run strategy manager - возвращаем полный ConsensusSignal (ОБНОВЛЕНО в Фазе 3)
        consensus = await asyncio.to_thread(
            self.strategy_manager.analyze_with_consensus,
            symbol=self.config.symbol,
            candles=candles,
            current_price=self.current_price,
            orderbook=self.current_orderbook,
            metrics=orderbook_metrics,
            sr_levels=None,
            volume_profile=None,
            ml_prediction=ml_prediction,  # Передаем ML prediction
            market_trades=self.current_market_trades  # Передаем market trades
        )

        # Возвращаем полный consensus объект, а не только final_signal
        return consensus

    async def _get_ml_prediction(
        self,
        candles: List[Candle],
        orderbook: Optional[OrderBookSnapshot],
        orderbook_metrics: Optional[OrderBookMetrics]
    ) -> Optional[Dict]:
        """
        Получить ML prediction из модели (НОВОЕ в Фазе 2).

        Args:
            candles: История свечей
            orderbook: Текущий orderbook snapshot
            orderbook_metrics: Метрики orderbook

        Returns:
            Dict с prediction или None при ошибке/отсутствии модели
        """
        if not self.feature_pipeline or not self.ml_client:
            return None

        if not orderbook:
            # ML модель требует orderbook данные
            logger.debug("ML prediction пропущен: orderbook отсутствует")
            return None

        try:
            # 1. Извлечь признаки через FeaturePipeline
            prev_orderbook = None
            prev_candle = candles[-2] if len(candles) > 1 else None

            feature_vector: FeatureVector = await self.feature_pipeline.extract_features(
                orderbook_snapshot=orderbook,
                candles=candles,
                prev_orderbook=prev_orderbook,
                prev_candle=prev_candle
            )

            # 2. Преобразовать в numpy array
            features = feature_vector.to_array()

            # 3. Получить prediction от ML сервера
            prediction = await self.ml_client.predict(
                symbol=self.config.symbol,
                features=features,
                model_name=self.config.ml_model_name,
                model_version=self.config.ml_model_version
            )

            if prediction:
                logger.debug(
                    f"ML prediction получен: direction={prediction.get('direction')}, "
                    f"confidence={prediction.get('confidence', 0):.3f}"
                )

            return prediction

        except Exception as e:
            logger.warning(f"Ошибка получения ML prediction: {e}")
            return None

    def _calculate_orderbook_metrics(self, orderbook: OrderBookSnapshot) -> OrderBookMetrics:
        """
        Рассчитать метрики из OrderBookSnapshot.

        Args:
            orderbook: Snapshot стакана ордеров

        Returns:
            OrderBookMetrics с рассчитанными метриками
        """
        # Basic prices
        best_bid = orderbook.best_bid
        best_ask = orderbook.best_ask
        spread = orderbook.spread
        mid_price = orderbook.mid_price

        # Volume calculations
        total_bid_volume = sum(qty for _, qty in orderbook.bids)
        total_ask_volume = sum(qty for _, qty in orderbook.asks)

        bid_volume_depth_5 = sum(qty for _, qty in orderbook.bids[:5])
        ask_volume_depth_5 = sum(qty for _, qty in orderbook.asks[:5])

        bid_volume_depth_10 = sum(qty for _, qty in orderbook.bids[:10])
        ask_volume_depth_10 = sum(qty for _, qty in orderbook.asks[:10])

        # Imbalance calculations
        total_volume = total_bid_volume + total_ask_volume
        imbalance = bid_volume_depth_5 / (bid_volume_depth_5 + ask_volume_depth_5) if (bid_volume_depth_5 + ask_volume_depth_5) > 0 else 0.5

        depth_5_total = bid_volume_depth_5 + ask_volume_depth_5
        imbalance_depth_5 = bid_volume_depth_5 / depth_5_total if depth_5_total > 0 else 0.5

        depth_10_total = bid_volume_depth_10 + ask_volume_depth_10
        imbalance_depth_10 = bid_volume_depth_10 / depth_10_total if depth_10_total > 0 else 0.5

        # VWAP calculations
        def calculate_vwap(levels: List[Tuple[float, float]], depth: int = 5) -> Optional[float]:
            """Calculate Volume-Weighted Average Price for N levels."""
            if not levels:
                return None
            top_levels = levels[:depth]
            total_value = sum(price * qty for price, qty in top_levels)
            total_volume = sum(qty for _, qty in top_levels)
            return total_value / total_volume if total_volume > 0 else None

        vwap_bid = calculate_vwap(orderbook.bids, 5)
        vwap_ask = calculate_vwap(orderbook.asks, 5)
        vwmp = (vwap_bid + vwap_ask) / 2 if vwap_bid and vwap_ask else mid_price

        # Cluster detection (largest volume level)
        largest_bid = max(orderbook.bids, key=lambda x: x[1], default=None) if orderbook.bids else None
        largest_ask = max(orderbook.asks, key=lambda x: x[1], default=None) if orderbook.asks else None

        largest_bid_cluster_price = largest_bid[0] if largest_bid else None
        largest_bid_cluster_volume = largest_bid[1] if largest_bid else 0.0

        largest_ask_cluster_price = largest_ask[0] if largest_ask else None
        largest_ask_cluster_volume = largest_ask[1] if largest_ask else 0.0

        return OrderBookMetrics(
            symbol=orderbook.symbol,
            timestamp=orderbook.timestamp,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            mid_price=mid_price,
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume,
            bid_volume_depth_5=bid_volume_depth_5,
            ask_volume_depth_5=ask_volume_depth_5,
            bid_volume_depth_10=bid_volume_depth_10,
            ask_volume_depth_10=ask_volume_depth_10,
            imbalance=imbalance,
            imbalance_depth_5=imbalance_depth_5,
            imbalance_depth_10=imbalance_depth_10,
            vwap_bid=vwap_bid,
            vwap_ask=vwap_ask,
            vwmp=vwmp,
            largest_bid_cluster_price=largest_bid_cluster_price,
            largest_bid_cluster_volume=largest_bid_cluster_volume,
            largest_ask_cluster_price=largest_ask_cluster_price,
            largest_ask_cluster_volume=largest_ask_cluster_volume
        )

    async def _process_signal(self, consensus):
        """
        Обработать консенсус-сигнал (ОБНОВЛЕНО в Фазе 3).

        Args:
            consensus: ConsensusSignal объект с полной информацией о консенсусе
        """
        signal = consensus.final_signal

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

        # If filled, create position (передаем полный consensus)
        if order.status.value == "filled":
            await self._open_position(order, signal, consensus)

    async def _open_position(self, order: SimulatedOrder, signal: TradingSignal, consensus=None):
        """
        Открыть позицию после исполнения ордера (ОБНОВЛЕНО в Фазе 3).

        Args:
            order: Исполненный ордер
            signal: Trading signal
            consensus: ConsensusSignal с полной информацией о консенсусе
        """
        # Calculate stop loss and take profit
        entry_price = order.average_fill_price

        if order.side == OrderSide.BUY:
            stop_loss = entry_price * (1 - self.config.risk_config.stop_loss_pct / 100)
            take_profit = entry_price * (1 + self.config.risk_config.take_profit_pct / 100)
        else:
            stop_loss = entry_price * (1 + self.config.risk_config.stop_loss_pct / 100)
            take_profit = entry_price * (1 - self.config.risk_config.take_profit_pct / 100)

        # Извлечь информацию о консенсусе (НОВОЕ)
        consensus_info = None
        if consensus:
            # Разделяем стратегии по направлениям
            strategies_buy = []
            strategies_sell = []

            for result in consensus.strategy_results:
                if result.signal and result.signal.signal_type == SignalType.BUY:
                    strategies_buy.append(result.strategy_name)
                elif result.signal and result.signal.signal_type == SignalType.SELL:
                    strategies_sell.append(result.strategy_name)

            consensus_info = {
                'mode': self.config.strategy_config.consensus_mode,
                'strategies_voted': consensus.contributing_strategies,
                'strategies_buy': strategies_buy,
                'strategies_sell': strategies_sell,
                'agreement_count': consensus.agreement_count,
                'disagreement_count': consensus.disagreement_count,
                'consensus_confidence': consensus.consensus_confidence,
                'final_confidence': signal.confidence,
                'candle_strategies': consensus.candle_strategies_count,
                'orderbook_strategies': consensus.orderbook_strategies_count,
                'hybrid_strategies': consensus.hybrid_strategies_count
            }

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
                'reason': signal.reason,
                'consensus_info': consensus_info  # НОВОЕ: Добавляем consensus info
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

            # Create trade result (ОБНОВЛЕНО в Фазе 3 - добавлен consensus_info)
            duration = (self.current_time - position.entry_time).total_seconds()

            # Извлекаем consensus_info из entry_signal
            consensus_info = None
            if position.entry_signal and 'consensus_info' in position.entry_signal:
                consensus_info = position.entry_signal['consensus_info']

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
                exit_signal=None,
                consensus_info=consensus_info  # НОВОЕ: Сохраняем consensus info
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

    def _downsample_equity_history(self):
        """
        Downsample equity history to prevent memory overflow.
        Keeps every 2nd point when limit is reached.
        """
        history = self.portfolio.equity_history
        if len(history) <= self.portfolio.max_equity_points:
            return

        # Keep every 2nd point to reduce to target size
        logger.info(
            f"📉 Downsampling equity history: {len(history)} -> "
            f"{self.portfolio.downsample_to} points"
        )

        # Calculate step to achieve target size
        step = len(history) // self.portfolio.downsample_to

        # Keep evenly spaced points
        downsampled = history[::step]

        # Always keep the last point (most recent)
        if history[-1] not in downsampled:
            downsampled.append(history[-1])

        # Update sequence numbers
        for i, point in enumerate(downsampled):
            point.sequence = i

        self.portfolio.equity_history = downsampled[:self.portfolio.downsample_to]

        logger.info(
            f"✅ Equity history downsampled to {len(self.portfolio.equity_history)} points"
        )

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

        # FIX: Downsample if history is too large to prevent memory leak
        self._downsample_equity_history()

    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Расчет финальных метрик производительности с использованием AdvancedMetricsCalculator."""
        import numpy as np
        from scipy import stats

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

        # Trade duration
        if self.closed_trades:
            avg_duration = np.mean([t.duration_seconds for t in self.closed_trades]) / 60  # minutes
        else:
            avg_duration = 0.0

        # Drawdown metrics from equity curve
        max_dd = 0.0
        max_dd_pct = 0.0
        max_dd_duration_days = 0.0

        if self.portfolio.equity_history:
            drawdowns = [ep.drawdown for ep in self.portfolio.equity_history]
            drawdowns_pct = [ep.drawdown_pct for ep in self.portfolio.equity_history]

            if drawdowns:
                max_dd = max(drawdowns)
                max_dd_pct = max(drawdowns_pct)

                # Drawdown duration
                in_drawdown = False
                dd_start = None
                dd_durations = []

                for i, ep in enumerate(self.portfolio.equity_history):
                    if ep.drawdown_pct > 0 and not in_drawdown:
                        in_drawdown = True
                        dd_start = ep.timestamp
                    elif ep.drawdown_pct == 0 and in_drawdown:
                        in_drawdown = False
                        if dd_start:
                            duration = (ep.timestamp - dd_start).total_seconds() / (24 * 3600)
                            dd_durations.append(duration)

                if dd_durations:
                    max_dd_duration_days = max(dd_durations)

        # Calculate basic risk-adjusted metrics
        sharpe_ratio = 0.0
        volatility_annual_pct = 0.0
        stability = 0.0

        if len(self.portfolio.equity_history) > 2:
            returns = []
            for i in range(1, len(self.portfolio.equity_history)):
                prev_equity = self.portfolio.equity_history[i-1].equity
                curr_equity = self.portfolio.equity_history[i].equity
                if prev_equity > 0:
                    ret = (curr_equity - prev_equity) / prev_equity
                    returns.append(ret)

            if returns:
                returns_array = np.array(returns)

                # Volatility
                volatility_annual_pct = np.std(returns) * np.sqrt(252) * 100

                # Sharpe Ratio (assuming 0% risk-free rate)
                if volatility_annual_pct > 0:
                    mean_return = np.mean(returns) * 252  # Annualized
                    sharpe_ratio = mean_return / (volatility_annual_pct / 100)

                # VaR and CVaR calculation (95% confidence level)
                # VaR_95: Value at Risk - maximum expected loss in worst 5% of cases
                # CVaR_95: Conditional VaR - average loss in worst 5% of cases (Expected Shortfall)
                var_95 = np.percentile(returns_array, 5)  # 5th percentile (worst 5%)

                # CVaR: mean of all returns that are less than or equal to VaR
                tail_losses = returns_array[returns_array <= var_95]
                cvar_95 = np.mean(tail_losses) if len(tail_losses) > 0 else var_95

                # Convert to percentage and make positive for easier interpretation
                # (e.g., VaR=2.5% means max loss is 2.5% with 95% confidence)
                var_95_pct = abs(var_95 * 100)
                cvar_95_pct = abs(cvar_95 * 100)
            else:
                var_95_pct = 0.0
                cvar_95_pct = 0.0
        else:
            var_95_pct = 0.0
            cvar_95_pct = 0.0

        # Calculate stability (R-squared of equity curve) only if we have equity history
        if len(self.portfolio.equity_history) > 1:
            x = np.arange(len(self.portfolio.equity_history))
            y = np.array([ep.equity for ep in self.portfolio.equity_history])
            if len(x) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                stability = r_value ** 2  # R-squared

        # Prepare data for Advanced Metrics Calculator
        trade_data = [
            TradeData(
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                pnl=t.pnl,
                pnl_pct=t.pnl_pct,
                duration_seconds=t.duration_seconds
            )
            for t in self.closed_trades
        ]

        equity_data = [
            EquityData(
                timestamp=ep.timestamp,
                equity=ep.equity,
                drawdown=ep.drawdown,
                drawdown_pct=ep.drawdown_pct,
                total_return_pct=ep.total_return_pct
            )
            for ep in self.portfolio.equity_history
        ]

        # Calculate advanced metrics
        advanced_calc = AdvancedMetricsCalculator(risk_free_rate=0.0)

        if trade_data and equity_data:
            advanced_metrics = advanced_calc.calculate(
                trades=trade_data,
                equity_curve=equity_data,
                initial_capital=self.config.initial_capital,
                final_capital=self.portfolio.equity,
                total_pnl=self.portfolio.total_return,
                max_drawdown=max_dd,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
        else:
            # Default values if no trades
            from backend.backtesting.metrics.advanced_metrics import AdvancedMetrics
            advanced_metrics = AdvancedMetrics(
                sortino_ratio=0.0, calmar_ratio=0.0, omega_ratio=0.0,
                profit_factor=0.0, expectancy=0.0, kelly_criterion=0.0, monthly_win_rate=0.0,
                avg_drawdown=0.0, avg_drawdown_pct=0.0, avg_drawdown_duration_days=0.0,
                max_drawdown_duration_days=0.0, recovery_factor=0.0, ulcer_index=0.0,
                win_loss_ratio=0.0, avg_win=0.0, avg_loss=0.0, largest_win=0.0, largest_loss=0.0,
                consecutive_wins_max=0, consecutive_losses_max=0,
                market_exposure_pct=0.0, avg_trade_duration_hours=0.0,
                returns_skewness=0.0, returns_kurtosis=0.0, tail_ratio=1.0
            )

        # Monthly returns
        monthly_pnl = {}
        for trade in self.closed_trades:
            month_key = trade.exit_time.strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += trade.pnl

        monthly_returns = list(monthly_pnl.values())

        # Combine all metrics
        return PerformanceMetrics(
            # Returns
            total_return=self.portfolio.total_return,
            total_return_pct=self.portfolio.total_return_pct,
            annual_return_pct=self.portfolio.total_return_pct * (365 / max(1, (self.config.end_date - self.config.start_date).days)),
            monthly_returns=monthly_returns,

            # Risk-Adjusted
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=advanced_metrics.sortino_ratio,
            calmar_ratio=advanced_metrics.calmar_ratio,
            volatility_annual_pct=volatility_annual_pct,

            # Drawdown
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_duration_days=max_dd_duration_days,
            avg_drawdown=advanced_metrics.avg_drawdown,
            avg_drawdown_pct=advanced_metrics.avg_drawdown_pct,
            avg_drawdown_duration_days=advanced_metrics.avg_drawdown_duration_days,

            # Trade Statistics
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate,
            profit_factor=advanced_metrics.profit_factor,
            avg_win=advanced_metrics.avg_win,
            avg_loss=advanced_metrics.avg_loss,
            largest_win=advanced_metrics.largest_win,
            largest_loss=advanced_metrics.largest_loss,
            avg_trade_duration_minutes=avg_duration,

            # Advanced Metrics
            omega_ratio=advanced_metrics.omega_ratio,
            tail_ratio=advanced_metrics.tail_ratio,
            var_95=var_95_pct,  # Value at Risk (95% confidence) - max expected loss in worst 5%
            cvar_95=cvar_95_pct,  # Conditional VaR (95%) - average loss in worst 5%

            # Quality
            stability=stability,

            # Extended Consistency
            expectancy=advanced_metrics.expectancy,
            kelly_criterion=advanced_metrics.kelly_criterion,
            monthly_win_rate=advanced_metrics.monthly_win_rate,
            win_loss_ratio=advanced_metrics.win_loss_ratio,
            consecutive_wins_max=advanced_metrics.consecutive_wins_max,
            consecutive_losses_max=advanced_metrics.consecutive_losses_max,

            # Extended Drawdown
            recovery_factor=advanced_metrics.recovery_factor,
            ulcer_index=advanced_metrics.ulcer_index,

            # Market Exposure
            market_exposure_pct=advanced_metrics.market_exposure_pct,
            avg_trade_duration_hours=advanced_metrics.avg_trade_duration_hours,

            # Distribution
            returns_skewness=advanced_metrics.returns_skewness,
            returns_kurtosis=advanced_metrics.returns_kurtosis
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
