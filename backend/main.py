"""
Главный файл приложения.
Точка входа и контроллер торгового бота.
"""

import asyncio
import os
import signal
import time
import traceback
from datetime import datetime
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import WebSocket, WebSocketDisconnect

from config import settings
from core.dynamic_symbols import DynamicSymbolsManager
from core.logger import get_logger, setup_logging
from core.exceptions import log_exception, OrderBookSyncError, OrderBookError
from core.trace_context import trace_operation
from database.connection import db_manager
from domain.services.fsm_registry import fsm_registry
from exchange.rest_client import rest_client
from exchange.websocket_manager import BybitWebSocketManager
from infrastructure.repositories.position_repository import position_repository
from infrastructure.resilience.recovery_service import recovery_service
from ml_engine.detection.layering_detector import LayeringConfig, LayeringDetector
from ml_engine.detection.spoofing_detector import SpoofingConfig, SpoofingDetector
from ml_engine.detection.sr_level_detector import SRLevelConfig, SRLevelDetector
from ml_engine.integration.ml_signal_validator import ValidationConfig, MLSignalValidator
from ml_engine.monitoring.drift_detector import DriftDetector
# from models.signal import TradingSignal, SignalType, SignalStrength, SignalSource
from screener.screener_manager import ScreenerManager
from strategies.adaptive import AdaptiveConsensusManager, WeightOptimizerConfig, OptimizationMethod, \
  RegimeDetectorConfig, PerformanceTrackerConfig, AdaptiveConsensusConfig
from strategies.strategy_manager import ExtendedStrategyManagerConfig, ExtendedStrategyManager
from strategy.candle_manager import CandleManager
from strategy.correlation_manager import correlation_manager
from strategy.daily_loss_killer import daily_loss_killer
from strategy.orderbook_manager import OrderBookManager
from strategy.analyzer import MarketAnalyzer
from strategy.position_monitor import PositionMonitor
from strategy.reversal_detector import reversal_detector
from strategy.risk_manager_ml_enhanced import RiskManagerMLEnhanced
from strategy.risk_models import ReversalSignal
from strategy.strategy_engine import StrategyEngine
from strategy.risk_manager import RiskManager
from execution.execution_manager import ExecutionManager
from strategy.trailing_stop_manager import trailing_stop_manager
from utils.balance_tracker import balance_tracker
from utils.constants import BotStatus
from api.websocket import manager as ws_manager, handle_websocket_messages
from tasks.cleanup_tasks import cleanup_tasks
from utils.helpers import safe_enum_value
# ML FEATURE PIPELINE - НОВОЕ
from ml_engine.features import (
    MultiSymbolFeaturePipeline,
    FeatureVector
)
from ml_engine.data_collection import MLDataCollector  # НОВОЕ

# Фаза 2: Adaptive Consensus
from strategies.adaptive import (
    AdaptiveConsensusManager,
    AdaptiveConsensusConfig,

    WeightOptimizerConfig
)

# Фаза 3: Multi-Timeframe
from strategies.mtf import (
    MultiTimeframeManager,
    MTFManagerConfig,
    MultiTimeframeConfig,
    AlignmentConfig,
    SynthesizerConfig,
    SynthesisMode,
    Timeframe
)

# Фаза 4: Integrated Engine
from engine.integrated_analysis_engine import (
    IntegratedAnalysisEngine,
    IntegratedAnalysisConfig,
    AnalysisMode
)

from models.signal import TradingSignal, SignalType, SignalStrength, SignalSource

# Сохраняем оригинальный __post_init__
_original_tradingsignal_post_init = TradingSignal.__post_init__


def _patched_tradingsignal_post_init(self):
  """
  Патч для TradingSignal который автоматически конвертирует строки в Enum.

  Это исправляет проблему когда signal_type/strength/source приходят как строки,
  но код ожидает Enum и пытается использовать .value
  """
  # Вызываем оригинальный __post_init__
  _original_tradingsignal_post_init(self)

  # Конвертируем строки в Enum если нужно
  if isinstance(self.signal_type, str):
    try:
      self.signal_type = SignalType(self.signal_type)
    except (ValueError, KeyError):
      # Если не можем сконвертировать, оставляем как есть
      pass

  if isinstance(self.strength, str):
    try:
      self.strength = SignalStrength(self.strength)
    except (ValueError, KeyError):
      pass

  if isinstance(self.source, str):
    try:
      self.source = SignalSource(self.source)
    except (ValueError, KeyError):
      pass


# Применяем патч
TradingSignal.__post_init__ = _patched_tradingsignal_post_init

print("✓ TradingSignal патч применен - все .value будут работать корректно")

original_post_init = TradingSignal.__post_init__


def patched_post_init(self):
  original_post_init(self)
  if isinstance(self.signal_type, str):
    self.signal_type = SignalType(self.signal_type)
  if isinstance(self.strength, str):
    self.strength = SignalStrength(self.strength)
  if isinstance(self.source, str):
    self.source = SignalSource(self.source)


TradingSignal.__post_init__ = patched_post_init

# Настройка логирования
setup_logging()
logger = get_logger(__name__)



class BotController:
  """Главный контроллер торгового бота."""

  def __init__(self):
    """
    Инициализация контроллера с поддержкой всех фаз.

    АРХИТЕКТУРА:
    - Базовые компоненты (WebSocket, OrderBook, Candles)
    - Strategy Manager (Фаза 1)
    - Adaptive Consensus (Фаза 2)
    - MTF Manager (Фаза 3)
    - Integrated Engine (Фаза 4)
    - ML Components
    - Execution & Risk Management
    """
    self.status = BotStatus.STOPPED
    self.symbols = settings.get_trading_pairs_list()
    self.initialized = False

    # ==================== БАЗОВЫЕ КОМПОНЕНТЫ ====================
    self.websocket_manager: Optional[BybitWebSocketManager] = None
    self.orderbook_managers: Dict[str, OrderBookManager] = {}
    self.candle_managers: Dict[str, CandleManager] = {}
    self.market_analyzer: Optional[MarketAnalyzer] = None
    self.strategy_engine: Optional[StrategyEngine] = None
    self.risk_manager: Optional[RiskManager] = None
    self.execution_manager: Optional[ExecutionManager] = None
    self.balance_tracker = balance_tracker

    # ==================== ML КОМПОНЕНТЫ ====================
    self.ml_feature_pipeline: Optional[MultiSymbolFeaturePipeline] = None
    self.ml_data_collector: Optional[MLDataCollector] = None
    self.latest_features: Dict[str, FeatureVector] = {}

    # ==================== ФАЗА 1: EXTENDED STRATEGY MANAGER ====================
    self.strategy_manager: Optional[ExtendedStrategyManager] = None

    # Флаги для включения/отключения компонентов
    self.enable_orderbook_strategies = settings.ENABLE_ORDERBOOK_STRATEGIES if hasattr(settings,
                                                                                       'ENABLE_ORDERBOOK_STRATEGIES') else True
    self.enable_adaptive_consensus = settings.ENABLE_ADAPTIVE_CONSENSUS if hasattr(settings,
                                                                                   'ENABLE_ADAPTIVE_CONSENSUS') else True
    self.enable_mtf_analysis = settings.ENABLE_MTF_ANALYSIS if hasattr(settings, 'ENABLE_MTF_ANALYSIS') else True
    self.enable_ml_validation = settings.ENABLE_ML_VALIDATION if hasattr(settings, 'ENABLE_ML_VALIDATION') else True
    self.enable_paper_trading = settings.PAPER_TRADING if hasattr(settings, 'PAPER_TRADING') else False

    # ==================== ФАЗА 2: ADAPTIVE CONSENSUS ====================
    self.adaptive_consensus: Optional[AdaptiveConsensusManager] = None

    # ==================== ФАЗА 3: MULTI-TIMEFRAME ====================
    self.mtf_manager: Optional[MultiTimeframeManager] = None

    # ==================== ФАЗА 4: INTEGRATED ENGINE ====================
    self.integrated_engine: Optional[IntegratedAnalysisEngine] = None

    # ==================== ML SIGNAL VALIDATOR ====================
    # Создаём конфигурацию для ML Validator
    logger.info("🤖 Создание ML Signal Validator...")
    try:
      ml_validator_config = ValidationConfig(
        model_server_url=settings.ML_SERVER_URL,
        model_version="latest",
        request_timeout=5.0,
        health_check_enabled=True,
        health_check_interval=30,
        health_check_timeout=2.0,
        min_ml_confidence=settings.ML_MIN_CONFIDENCE,
        confidence_boost_factor=1.2,
        confidence_penalty_factor=0.7,
        ml_weight=settings.ML_WEIGHT,
        strategy_weight=settings.STRATEGY_WEIGHT,
        use_fallback_on_error=True,
        fallback_to_strategy=True,
        cache_predictions=True,
        cache_ttl_seconds=30,
        enable_mae_prediction=True,
        enable_manipulation_detection=True,
        enable_regime_detection=True,
        enable_feature_quality_check=True
      )
      self.ml_validator = MLSignalValidator(config=ml_validator_config)
      logger.info(f"✓ ML Signal Validator создан: server={settings.ML_SERVER_URL}")
    except Exception as e:
      logger.warning(f"⚠️ ML Signal Validator creation failed: {e}. Продолжаем без ML валидации.")
      self.ml_validator = None

    # ==================== DETECTION SYSTEMS ====================
    # Drift Detector
    self.drift_detector = DriftDetector(
      window_size=10000,
      baseline_window_size=50000,
      drift_threshold=0.1
    )

    # Spoofing Detector
    spoofing_config = SpoofingConfig(
      large_order_threshold_usdt=50000.0,
      suspicious_ttl_seconds=10.0,
      cancel_rate_threshold=0.7
    )
    self.spoofing_detector = SpoofingDetector(spoofing_config)

    # Layering Detector
    layering_config = LayeringConfig(
      min_orders_in_layer=3,
      max_price_spread_pct=0.005,
      min_layer_volume_usdt=30000.0
    )
    self.layering_detector = LayeringDetector(layering_config)

    # S/R Level Detector
    sr_config = SRLevelConfig(
      min_touches=2,
      lookback_candles=200,
      max_age_hours=24
    )
    self.sr_detector = SRLevelDetector(sr_config)

    # ==================== ЗАДАЧИ ====================
    self.websocket_task: Optional[asyncio.Task] = None
    self.analysis_task: Optional[asyncio.Task] = None
    self.candle_update_task: Optional[asyncio.Task] = None
    self.ml_stats_task: Optional[asyncio.Task] = None
    self.screener_broadcast_task: Optional[asyncio.Task] = None
    self.symbols_refresh_task: Optional[asyncio.Task] = None
    self.correlation_update_task: Optional[asyncio.Task] = None
    self.position_monitor_task: Optional[asyncio.Task] = None

    # ==================== ДРУГИЕ КОМПОНЕНТЫ ====================
    self.screener_manager: Optional[ScreenerManager] = None
    self.dynamic_symbols_manager: Optional[DynamicSymbolsManager] = None
    self.position_monitor: Optional[PositionMonitor] = None
    self.weight_optimization_task: Optional[asyncio.Task] = None
    self.mtf_update_task: Optional[asyncio.Task] = None
    self.running = False

    logger.info("✅ BotController инициализирован с поддержкой Фаз 1-4")

  async def initialize(self):
    """Инициализация всех компонентов бота."""
    try:
      logger.info("=" * 80)
      logger.info("ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ БОТА (ML-ENHANCED)")
      logger.info("=" * 80)

      initialization_start = time.time()

      # Инициализируем REST клиент
      await rest_client.initialize()
      logger.info("✓ REST клиент инициализирован")

      # Инициализируем анализатор рынка (пока без символов)
      self.market_analyzer = MarketAnalyzer()
      logger.info("✓ Анализатор рынка инициализирован")

      # Проверяем подключение к бирже
      server_time = await rest_client.get_server_time()
      logger.info(f"✓ Подключение к Bybit успешно. Серверное время: {server_time}")

      # ===== SCREENER MANAGER - СРАЗУ инициализируем =====
      if settings.SCREENER_ENABLED:
        logger.info("Инициализация Screener Manager...")
        self.screener_manager = ScreenerManager()
        logger.info("✓ Screener Manager инициализирован")

      # ===== DYNAMIC SYMBOLS - Инициализируем менеджер =====
      if settings.DYNAMIC_SYMBOLS_ENABLED:
        logger.info("Инициализация Dynamic Symbols Manager...")
        self.dynamic_symbols_manager = DynamicSymbolsManager(
          min_volume=settings.DYNAMIC_MIN_VOLUME,
          max_volume_pairs=settings.DYNAMIC_MAX_VOLUME_PAIRS,
          top_gainers=settings.DYNAMIC_TOP_GAINERS,
          top_losers=settings.DYNAMIC_TOP_LOSERS
        )
        logger.info("✓ Dynamic Symbols Manager инициализирован")


      # ===== ML DATA COLLECTOR =====
      self.ml_data_collector = MLDataCollector(
        storage_path="../data/ml_training",
        max_samples_per_file=10000
      )
      await self.ml_data_collector.initialize()
      logger.info("✓ ML Data Collector инициализирован")

      # ========== ЭТАП 5: STRATEGY MANAGER (ФАЗА 1) ==========
      logger.info("🎯 [5/10] Инициализация ExtendedStrategyManager (Фаза 1)...")

      from strategies.strategy_manager import StrategyPriority

      # Конфигурация Extended Strategy Manager
      strategy_config = ExtendedStrategyManagerConfig(
        consensus_mode="weighted",  # weighted / majority / unanimous
        min_strategies_for_signal=2,
        min_consensus_confidence=0.6,

        # Веса CANDLE стратегий
        candle_strategy_weights={
          'momentum': 0.20,
          'sar_wave': 0.15,
          'supertrend': 0.20,
          'volume_profile': 0.15
        },

        # Веса ORDERBOOK стратегий
        orderbook_strategy_weights={
          'imbalance': 0.10,
          'volume_flow': 0.10,
          'liquidity_zone': 0.10
        } if self.enable_orderbook_strategies else {},

        # Веса HYBRID стратегий
        hybrid_strategy_weights={
          'smart_money': 0.15
        } if self.enable_orderbook_strategies else {},

        # Приоритеты стратегий
        strategy_priorities={
          'momentum': StrategyPriority.HIGH,
          'supertrend': StrategyPriority.HIGH,
          'liquidity_zone': StrategyPriority.HIGH,
          'smart_money': StrategyPriority.HIGH,
          'sar_wave': StrategyPriority.MEDIUM,
          'volume_profile': StrategyPriority.MEDIUM,
          'imbalance': StrategyPriority.MEDIUM,
          'volume_flow': StrategyPriority.MEDIUM
        },

        # Включение типов стратегий
        enable_orderbook_strategies=self.enable_orderbook_strategies,
        enable_hybrid_strategies=self.enable_orderbook_strategies
      )

      self.strategy_manager = ExtendedStrategyManager(strategy_config)
      logger.info("✅ ExtendedStrategyManager инициализирован")
      logger.info(f"📊 Активные стратегии: {list(self.strategy_manager.get_all_strategy_names())}")

      # ========== ЭТАП 6: ADAPTIVE CONSENSUS (ФАЗА 2) ==========
      if self.enable_adaptive_consensus:
        logger.info("🔄 [6/10] Инициализация Adaptive Consensus Manager (Фаза 2)...")

        try:
          adaptive_config = AdaptiveConsensusConfig(
            # Enable/disable компонентов
            enable_performance_tracking=True,
            enable_regime_detection=True,
            enable_weight_optimization=True,

            # Performance Tracker Config
            performance_tracker_config=PerformanceTrackerConfig(
              data_dir="data/strategy_performance",
              enable_persistence=True,
              short_term_hours=24,
              medium_term_days=7,
              long_term_days=30,
              min_signals_for_metrics=settings.ADAPTIVE_MIN_SIGNALS_FOR_EVALUATION if hasattr(settings,
                                                                                              'ADAPTIVE_MIN_SIGNALS_FOR_EVALUATION') else 20,
              min_closed_signals_for_metrics=10
            ),

            # Regime Detector Config
            regime_detector_config=RegimeDetectorConfig(
              adx_strong_threshold=25.0,
              adx_weak_threshold=15.0,
              update_frequency_seconds=300  # 5 минут
            ),

            # Weight Optimizer Config
            weight_optimizer_config=WeightOptimizerConfig(
              optimization_method=OptimizationMethod.HYBRID,  # Performance + Regime
              min_weight=0.05,
              max_weight=0.40,
              update_frequency_seconds=settings.ADAPTIVE_WEIGHT_UPDATE_FREQUENCY_SECONDS if hasattr(settings,
                                                                                                    'ADAPTIVE_WEIGHT_UPDATE_FREQUENCY_SECONDS') else 21600,
              regime_weight_blend=0.6,  # 60% performance, 40% regime
              min_signals_for_optimization=30
            ),

            # Consensus Config
            consensus_mode="adaptive_weighted",
            min_consensus_confidence=0.6,
            conflict_resolution_mode="performance_priority",
            enable_quality_metrics=True,
            min_consensus_quality=0.6
          )

          self.adaptive_consensus = AdaptiveConsensusManager(
            config=adaptive_config,
            strategy_manager=self.strategy_manager
          )

          logger.info("✅ Adaptive Consensus Manager инициализирован")

        except Exception as e:
          logger.error(f"❌ Ошибка инициализации Adaptive Consensus: {e}")
          logger.warning("⚠️ Продолжаем без Adaptive Consensus")
          self.adaptive_consensus = None
      else:
        logger.info("ℹ️ [6/10] Adaptive Consensus отключен в настройках")

      # ========== ЭТАП 7: MTF MANAGER (ФАЗА 3) ==========
      if self.enable_mtf_analysis:
        logger.info("⏱️ [7/10] Инициализация Multi-Timeframe Manager (Фаза 3)...")

        try:
          # Парсинг таймфреймов из настроек
          mtf_active_tfs = settings.MTF_ACTIVE_TIMEFRAMES if hasattr(settings,
                                                                     'MTF_ACTIVE_TIMEFRAMES') else "1m,5m,15m,1h"
          mtf_primary_tf = settings.MTF_PRIMARY_TIMEFRAME if hasattr(settings, 'MTF_PRIMARY_TIMEFRAME') else "1h"
          mtf_execution_tf = settings.MTF_EXECUTION_TIMEFRAME if hasattr(settings, 'MTF_EXECUTION_TIMEFRAME') else "1m"
          mtf_synthesis_mode = settings.MTF_SYNTHESIS_MODE if hasattr(settings, 'MTF_SYNTHESIS_MODE') else "top_down"
          mtf_min_quality = settings.MTF_MIN_QUALITY if hasattr(settings, 'MTF_MIN_QUALITY') else 0.60
          mtf_staggered_interval = settings.MTF_STAGGERED_UPDATE_INTERVAL if hasattr(settings,
                                                                                     'MTF_STAGGERED_UPDATE_INTERVAL') else 5

          active_tfs_str = mtf_active_tfs.split(',')
          active_timeframes = [Timeframe(tf.strip()) for tf in active_tfs_str]
          primary_tf = Timeframe(mtf_primary_tf)
          execution_tf = Timeframe(mtf_execution_tf)

          logger.info(f"📊 MTF Таймфреймы: {[tf.value for tf in active_timeframes]}")
          logger.info(f"🎯 Primary TF: {primary_tf.value}, Execution TF: {execution_tf.value}")

          # Конфигурация MTF Manager
          mtf_config = MTFManagerConfig(
            enabled=True,

            # Coordinator Config
            coordinator_config=MultiTimeframeConfig(
              active_timeframes=active_timeframes,
              primary_timeframe=primary_tf,
              execution_timeframe=execution_tf,
              enable_caching=True,
              staggered_update_interval=mtf_staggered_interval,
              enable_validation=True
            ),

            # Aligner Config
            aligner_config=AlignmentConfig(
              htf_weight=0.50,  # Higher Timeframe weight
              mtf_weight=0.30,  # Medium Timeframe weight
              ltf_weight=0.20,  # Lower Timeframe weight
              min_alignment_score=0.65,
              enable_confluence_detection=True,
              min_confluence_zones=1,
              enable_divergence_detection=True
            ),

            # Synthesizer Config
            synthesizer_config=SynthesizerConfig(
              synthesis_mode=SynthesisMode(mtf_synthesis_mode),
              min_signal_quality=mtf_min_quality,
              enable_dynamic_sizing=True,
              position_size_multiplier_range=(0.3, 1.5),
              enable_smart_sl=True,
              default_risk_reward_ratio=2.0
            ),

            # Quality Control
            min_quality_threshold=mtf_min_quality,
            enable_quality_scoring=True,

            # Fallback
            fallback_to_single_tf=True,
            min_timeframes_for_signal=2
          )

          self.mtf_manager = MultiTimeframeManager(
            strategy_manager=self.strategy_manager,
            config=mtf_config
          )

          # Инициализация символов в MTF Manager
          for symbol in self.symbols:
            await self.mtf_manager.initialize_symbol(symbol)
            logger.info(f"✅ {symbol}: MTF Manager инициализирован")

          logger.info("✅ Multi-Timeframe Manager инициализирован")

        except Exception as e:
          logger.error(f"❌ Ошибка инициализации MTF Manager: {e}")
          logger.warning("⚠️ Продолжаем без MTF Analysis")
          self.mtf_manager = None
      else:
        logger.info("ℹ️ [7/10] Multi-Timeframe Analysis отключен в настройках")

      # ========== ЭТАП 8: INTEGRATED ENGINE (ФАЗА 4) ==========
      logger.info("🎯 [8/10] Инициализация Integrated Analysis Engine (Фаза 4)...")

      try:
        integrated_mode = settings.INTEGRATED_ANALYSIS_MODE if hasattr(settings,
                                                                       'INTEGRATED_ANALYSIS_MODE') else "hybrid"
        hybrid_mtf_priority = settings.HYBRID_MTF_PRIORITY if hasattr(settings, 'HYBRID_MTF_PRIORITY') else 0.6
        hybrid_min_agreement = settings.HYBRID_MIN_AGREEMENT if hasattr(settings, 'HYBRID_MIN_AGREEMENT') else True
        hybrid_conflict_resolution = settings.HYBRID_CONFLICT_RESOLUTION if hasattr(settings,
                                                                                    'HYBRID_CONFLICT_RESOLUTION') else "highest_quality"
        min_combined_quality = settings.MIN_COMBINED_QUALITY if hasattr(settings, 'MIN_COMBINED_QUALITY') else 0.65

        integrated_config = IntegratedAnalysisConfig(
          # Режим анализа
          analysis_mode=AnalysisMode(integrated_mode),

          # Доступность компонентов
          enable_adaptive_consensus=(self.adaptive_consensus is not None),
          enable_mtf_analysis=(self.mtf_manager is not None),

          # Hybrid режим настройки
          hybrid_mtf_priority=hybrid_mtf_priority,
          hybrid_min_agreement=hybrid_min_agreement,
          hybrid_conflict_resolution=hybrid_conflict_resolution,

          # Quality control
          min_combined_quality=min_combined_quality,
          enable_quality_scoring=True,

          # Fallback
          fallback_to_single_tf=True,
          fallback_to_basic_consensus=True
        )

        self.integrated_engine = IntegratedAnalysisEngine(integrated_config)

        # Инициализация символов в Integrated Engine
        for symbol in self.symbols:
          await self.integrated_engine.initialize_symbol(symbol)
          logger.info(f"✅ {symbol}: Integrated Engine инициализирован")

        logger.info("✅ Integrated Analysis Engine инициализирован")
        logger.info(f"📊 Режим анализа: {integrated_mode}")

      except Exception as e:
        logger.error(f"❌ Критическая ошибка инициализации Integrated Engine: {e}")
        raise  # Критическая ошибка - прерываем инициализацию


      # Инициализируем базовую стратегию
      self.strategy_engine = StrategyEngine()
      logger.info("✓ Торговая стратегия инициализирована")

      # # Передаем список торгуемых символов
      # await correlation_manager.initialize(self.symbols)


      logger.info("=" * 80)
      logger.info("БАЗОВЫЕ КОМПОНЕНТЫ ИНИЦИАЛИЗИРОВАНЫ (БЕЗ WEBSOCKET)")
      logger.info("=" * 80)
      self.initialized = True
      self.startup_timestamp = datetime.now()

      initialization_time = time.time() - initialization_start
      logger.info("=" * 80)
      logger.info(f"✅ ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА за {initialization_time:.2f}с")
      logger.info("=" * 80)
      logger.info(f"📊 Компоненты инициализированы:")
      logger.info(f"   - Базовые сервисы: ✅")
      logger.info(f"   - Market Data Managers: ✅ ({len(self.symbols)} пар)")
      logger.info(f"   - Strategy Manager: ✅")
      logger.info(f"   - Adaptive Consensus: {'✅' if self.adaptive_consensus else '❌'}")
      logger.info(f"   - MTF Manager: {'✅' if self.mtf_manager else '❌'}")
      logger.info(f"   - Integrated Engine: ✅")
      logger.info(f"   - ML Components: {'✅' if self.ml_validator else '⚠️'}")
      logger.info(f"   - Execution & Risk: ⏳ (в start())")
      logger.info("=" * 80)

    except Exception as e:
      logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА ИНИЦИАЛИЗАЦИИ: {e}")
      logger.error(traceback.format_exc())
      log_exception(logger, e, "Инициализация бота")

      # Cleanup частично инициализированных компонентов
      await self._cleanup_on_error()

      raise RuntimeError(f"Не удалось инициализировать BotController: {e}") from e

  async def start(self):
    """Запуск бота с правильной последовательностью инициализации.
    ПОСЛЕДОВАТЕЛЬНОСТЬ:
    1. ML Signal Validator - инициализация HTTP сессии
    2. Risk Manager - получение баланса и инициализация
    3. Execution Manager - создание и запуск
    4. Balance Tracker - запуск
    5. Daily Loss Killer - запуск
    6. Screener Manager (опционально) - запуск
    7. Dynamic Symbols (опционально) - выбор пар
    8. Correlation Manager - инициализация
    9. ML Feature Pipeline - создание для финальных символов
    10. OrderBook/Candle Managers - создание для финальных символов
    11. Market Analyzer - добавление символов
    12. Position Monitor - создание
    13. WebSocket Manager - создание и подключение
    14. Historical Candles - загрузка
    15. Analysis Loop - запуск
    16. Position Monitor - запуск
    17. Вспомогательные задачи - запуск

    """
    if self.status == BotStatus.RUNNING:
      logger.warning("Бот уже запущен")
      return

    try:
      self.status = BotStatus.STARTING
      logger.info("=" * 80)
      logger.info("ЗАПУСК ТОРГОВОГО БОТА (ML-ENHANCED)")
      logger.info("=" * 80)

      # ========== 1. ML SIGNAL VALIDATOR - ИНИЦИАЛИЗАЦИЯ ==========
      # ВАЖНО: Инициализируем HTTP сессию и health check
      if self.ml_validator:
        logger.info("🤖 Инициализация ML Signal Validator...")
        try:
          await self.ml_validator.initialize()
          logger.info("✅ ML Signal Validator инициализирован")
        except Exception as e:
          logger.error(
            f"❌ Ошибка инициализации ML Validator: {e}. "
            f"ML validator будет недоступен."
          )
          # Не останавливаем бота, просто логируем
      else:
        logger.warning("⚠️ ML Signal Validator не создан, пропускаем инициализацию")

      # ========== 2. RISK MANAGER - ИНИЦИАЛИЗАЦИЯ ==========

      # Инициализация риск-менеджера с реальным балансом
      await self._initialize_risk_manager()

      # ========== 3. EXECUTION MANAGER - СОЗДАНИЕ И ЗАПУСК ==========

      self.execution_manager = ExecutionManager(self.risk_manager)
      logger.info("✓ Менеджер исполнения инициализирован")

      # Запускаем менеджер исполнения
      await self.execution_manager.start()
      logger.info("✓ Менеджер исполнения запущен")

      # ========== 4. BALANCE TRACKER - ЗАПУСК ==========

      # Запускаем трекер баланса
      await self.balance_tracker.start()
      logger.info("✓ Трекер баланса запущен")

      # ========== 5. DAILY LOSS KILLER - ЗАПУСК ===========
      await daily_loss_killer.start()
      logger.info("✓ Daily Loss Killer запущен")

      # ========== 6. SCREENER MANAGER (ОПЦИОНАЛЬНО) - ЗАПУСК ==========
      if self.screener_manager:
        logger.info("Запуск Screener Manager...")
        await self.screener_manager.start()

        # Запускаем broadcast задачу
        self.screener_broadcast_task = asyncio.create_task(
          self._screener_broadcast_loop()
        )
        logger.info("Ожидание первой загрузки пар от screener...")
        await asyncio.sleep(6)  # Даем время на загрузку данных

        logger.info("✓ Screener Manager запущен")

        # ========== 7. DYNAMIC SYMBOLS (ОПЦИОНАЛЬНО) - ВЫБОР ПАР ==========
        if settings.DYNAMIC_SYMBOLS_ENABLED and self.dynamic_symbols_manager:
          logger.info("Динамический отбор торговых пар...")

          # Получаем данные от screener
          screener_pairs = self.screener_manager.get_all_pairs()

          # Отбираем по критериям
          self.symbols = self.dynamic_symbols_manager.select_symbols(screener_pairs)

          logger.info(f"✓ Динамически отобрано {len(self.symbols)} пар для мониторинга")
        else:
          # Fallback на статический список
          self.symbols = settings.get_trading_pairs_list()
          logger.info(f"✓ Используется статический список: {len(self.symbols)} пар")
      else:
        # Если screener выключен - статический список
        self.symbols = settings.get_trading_pairs_list()
        logger.info(f"✓ Screener отключен, статический список: {len(self.symbols)} пар")

      # ========== 8. CORRELATION MANAGER - ИНИЦИАЛИЗАЦИЯ ==========

      logger.info("=" * 80)
      logger.info("ИНИЦИАЛИЗАЦИЯ CORRELATION MANAGER")
      logger.info("=" * 80)

      await correlation_manager.initialize(self.symbols)

      logger.info(
        f"✓ CorrelationManager инициализирован для {len(self.symbols)} символов: "
        f"групп={len(correlation_manager.group_manager.groups)}, "
        f"покрыто={len(correlation_manager.group_manager.symbol_to_group)} символов"
      )


      # ========== 9. ML FEATURE PIPELINE - СОЗДАНИЕ ДЛЯ ФИНАЛЬНЫХ СИМВОЛОВ ==========
      logger.info("Создание ML Feature Pipeline...")
      self.ml_feature_pipeline = MultiSymbolFeaturePipeline(
        symbols=self.symbols,  # ← Правильные динамические символы!
        normalize=True,
        cache_enabled=True
      )
      logger.info(f"✓ ML Feature Pipeline создан для {len(self.symbols)} символов")

      # ========== 10. ORDERBOOK/CANDLE MANAGERS - СОЗДАНИЕ ДЛЯ ФИНАЛЬНЫХ ПАР ==========
      logger.info(f"Создание менеджеров стакана для {len(self.symbols)} пар...")
      for symbol in self.symbols:
        self.orderbook_managers[symbol] = OrderBookManager(symbol)
      logger.info(f"✓ Создано {len(self.orderbook_managers)} менеджеров стакана")

      # ===== Создаем менеджеры свечей для ФИНАЛЬНЫХ пар =====
      logger.info(f"Создание менеджеров свечей для {len(self.symbols)} пар...")
      for symbol in self.symbols:
        self.candle_managers[symbol] = CandleManager(
          symbol=symbol,
          timeframe="1m",
          max_candles=200
        )
      logger.info(f"✓ Создано {len(self.candle_managers)} менеджеров свечей")

      # ========== 11. MARKET ANALYZER - ДОБАВЛЕНИЕ СИМВОЛОВ ==========
      for symbol in self.symbols:
        self.market_analyzer.add_symbol(symbol)
      logger.info(f"✓ {len(self.symbols)} символов добавлено в анализатор")

      # ========== 12. POSITION MONITOR - СОЗДАНИЕ ==========

      # НОВОЕ: Создание Position Monitor (ПОСЛЕ создания всех менеджеров)
      # ВАЖНО: Создаем ПОСЛЕ того, как все зависимости готовы:
      # - risk_manager ✓ (создан в начале start)
      # - execution_manager ✓ (создан в начале start)
      # - orderbook_managers ✓ (созданы выше)
      # - candle_managers ✓ (созданы выше)
      logger.info("Создание Position Monitor...")

      # Проверка зависимостей
      if not self.risk_manager:
        raise RuntimeError("RiskManager не инициализирован")
      if not self.execution_manager:
        raise RuntimeError("ExecutionManager не инициализирован")
      if not self.orderbook_managers:
        raise RuntimeError("OrderBookManagers не созданы")
      if not self.candle_managers:
        raise RuntimeError("CandleManagers не созданы")

      self.position_monitor = PositionMonitor(
        risk_manager=self.risk_manager,
        candle_managers=self.candle_managers,
        orderbook_managers=self.orderbook_managers,
        execution_manager=self.execution_manager
      )

      logger.info(
        f"✓ Position Monitor создан с {len(self.candle_managers)} "
        f"candle managers и {len(self.orderbook_managers)} orderbook managers"
      )

      # ========== 13. WEBSOCKET MANAGER - СОЗДАНИЕ И ПОДКЛЮЧЕНИЕ ==========

      logger.info("Создание WebSocket Manager...")
      logger.info(f"Символы для WebSocket: {self.symbols[:5]}..." if len(
        self.symbols) > 5 else f"Символы для WebSocket: {self.symbols}")

      self.websocket_manager = BybitWebSocketManager(
        symbols=self.symbols,  # ← Правильные динамические символы!
        on_message=self._handle_orderbook_message
      )
      logger.info("✓ WebSocket менеджер создан с правильными символами")

      # ========== 14. HISTORICAL CANDLES - ЗАГРУЗКА ==========

      await self._load_historical_candles()
      logger.info("✓ Исторические свечи загружены")

      # ========== 15. WEBSOCKET CONNECTIONS - ЗАПУСК ==========

      self.websocket_task = asyncio.create_task(
        self.websocket_manager.start()
      )
      logger.info("✓ WebSocket соединения запущены")

      # ========== 16. CANDLE UPDATE LOOP - ЗАПУСК ==========

      self.candle_update_task = asyncio.create_task(
        self._candle_update_loop()
      )
      logger.info("✓ Цикл обновления свечей запущен")

      # ========== 17. ML STATS LOOP - ЗАПУСК ==========

      self.ml_stats_task = asyncio.create_task(
        self._ml_stats_loop()
      )

      # ========== 18. ANALYSIS LOOP - ЗАПУСК ==========

      self.analysis_task = asyncio.create_task(
        self._analysis_loop_ml_enhanced()
      )
      logger.info("✓ Цикл анализа (ML-Enhanced) запущен")

      # ========== 19. POSITION MONITOR - ЗАПУСК ==========

      # ========== ЗАПУСК POSITION MONITOR ==========
      # ВАЖНО: Запускаем ПОСЛЕ analysis_task, так как:
      # 1. analysis_loop генерирует сигналы
      # 2. execution_manager открывает позиции
      # 3. position_monitor мониторит открытые позиции

      if self.position_monitor:
        await self.position_monitor.start()
        logger.info("✓ Position Monitor запущен")

      # ========== 20. FSM CLEANUP TASK - ЗАПУСК ==========

      asyncio.create_task(fsm_cleanup_task())
      logger.info("✓ FSM Cleanup Task запланирован")

      # ========== 21. SYMBOLS REFRESH (ОПЦИОНАЛЬНО) - ЗАПУСК ==========
      if settings.DYNAMIC_SYMBOLS_ENABLED and self.dynamic_symbols_manager:
        logger.info("Запуск задачи обновления списка пар...")
        self.symbols_refresh_task = asyncio.create_task(
          self._symbols_refresh_loop()
        )
        logger.info("✓ Задача обновления списка пар запущена")

      # ========== 22. CORRELATION UPDATE - ЗАПУСК ==========
      if correlation_manager.enabled:
        logger.info("Запуск периодического обновления корреляций...")
        self.correlation_update_task = asyncio.create_task(
          self._correlation_update_loop()
        )
        logger.info("✓ Correlation update task запущен")

      logger.info("✓ Запущено периодическое обновление корреляций")

      # ========== 23. TRAILING STOP MANAGER - ЗАПУСК ==========

      logger.info("Запуск Trailing Stop Manager...")
      await trailing_stop_manager.start()

      # ========== 24. ЗАПУСК ADAPTIVE WEIGHT OPTIMIZATION ==========

      # Периодическая оптимизация весов стратегий
      self.weight_optimization_task = asyncio.create_task(
        self._weight_optimization_loop(),
        name="weight_optimization"
      )
      logger.info("✅ Adaptive Weight Optimization запущен")

      # ========== 25. ЗАПУСК MTF UPDATES ==========

      # Staggered обновления таймфреймов
      self.mtf_update_task = asyncio.create_task(
        self._mtf_update_loop(),
        name="mtf_updates"
      )

      # Уведомляем фронтенд
      from api.websocket import broadcast_bot_status
      await broadcast_bot_status("running", {
        "symbols": self.symbols,
        "integrated_mode": True,
        "adaptive_consensus_enabled": self.adaptive_consensus is not None,
        "mtf_enabled": self.mtf_manager is not None,
        "ml_enabled": True,
        "position_monitor_enabled": self.position_monitor.enabled if self.position_monitor else False,
        "message": "Бот успешно запущен с ML поддержкой"
      })

      self.status = BotStatus.RUNNING
      logger.info("=" * 80)
      logger.info("БОТ УСПЕШНО ЗАПУЩЕН (ML-READY)")
      logger.info("=" * 80)

    except Exception as e:
      self.status = BotStatus.ERROR
      logger.error(f"Ошибка запуска бота: {e}")
      log_exception(logger, e, "Запуск бота")
      raise

  async def _symbols_refresh_loop(self):
    """
    Цикл обновления списка торговых пар.
    Запускается каждые DYNAMIC_REFRESH_INTERVAL секунд.
    """
    interval = settings.DYNAMIC_REFRESH_INTERVAL
    logger.info(f"Запущен symbols refresh loop (интервал: {interval}s)")

    # Даем время на стабилизацию
    await asyncio.sleep(interval)

    while self.status == BotStatus.RUNNING:
      try:
        logger.info("=" * 60)
        logger.info("ОБНОВЛЕНИЕ СПИСКА ТОРГОВЫХ ПАР")

        # Получаем актуальные данные от screener
        screener_pairs = self.screener_manager.get_all_pairs()

        # Отбираем по критериям
        new_symbols = self.dynamic_symbols_manager.select_symbols(screener_pairs)

        # Определяем изменения
        changes = self.dynamic_symbols_manager.get_changes(new_symbols)
        added = changes['added']
        removed = changes['removed']

        if not added and not removed:
          logger.info("✓ Список пар не изменился")
        else:
          logger.info(f"Изменения: +{len(added)} -{len(removed)}")

          # Добавляем новые пары
          for symbol in added:
            logger.info(f"  + Добавление пары: {symbol}")
            self.orderbook_managers[symbol] = OrderBookManager(symbol)
            self.candle_managers[symbol] = CandleManager(symbol, "1m", 200)
            self.market_analyzer.add_symbol(symbol)

          # Удаляем старые пары
          for symbol in removed:
            logger.info(f"  - Удаление пары: {symbol}")
            if symbol in self.orderbook_managers:
              del self.orderbook_managers[symbol]
            if symbol in self.candle_managers:
              del self.candle_managers[symbol]

          # Обновляем список
          self.symbols = new_symbols

          # Пересоздаем WebSocket соединения
          logger.info("Перезапуск WebSocket с новым списком пар...")
          if self.websocket_task:
            self.websocket_task.cancel()
            try:
              await self.websocket_task
            except asyncio.CancelledError:
              pass

          # Пересоздаем WebSocket менеджер
          self.websocket_manager = BybitWebSocketManager(
            symbols=self.symbols,
            on_message=self._handle_orderbook_message
          )
          self.websocket_task = asyncio.create_task(
            self.websocket_manager.start()
          )
          logger.info("✓ WebSocket перезапущен")

        logger.info("=" * 60)
        await asyncio.sleep(interval)

      except asyncio.CancelledError:
        logger.info("Symbols refresh loop остановлен")
        break
      except Exception as e:
        logger.error(f"Ошибка в symbols refresh loop: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        await asyncio.sleep(interval)


  async def _load_historical_candles(self):
    """Загрузка исторических свечей для всех символов."""
    logger.info("Загрузка исторических свечей...")

    for symbol in self.symbols:
      try:
        # ИСПРАВЛЕНО: get_kline (единственное число!)
        candles_data = await rest_client.get_kline(
          symbol=symbol,
          interval="1",  # 1 минута
          limit=200
        )

        # Добавляем в CandleManager
        candle_manager = self.candle_managers[symbol]
        await candle_manager.load_historical_data(candles_data)

        logger.debug(
          f"{symbol} | Загружено {len(candles_data)} исторических свечей"
        )

      except Exception as e:
        logger.warning(f"{symbol} | Ошибка загрузки свечей: {e}")

  async def _candle_update_loop(self):
    """Цикл обновления свечей (каждую минуту)."""
    logger.info("Запущен цикл обновления свечей")

    while self.status == BotStatus.RUNNING:
      try:
        for symbol in self.symbols:
          try:
            # Получаем последнюю свечу
            candles_data = await rest_client.get_kline(
              symbol=symbol,
              interval="1",
              limit=2  # Последние 2 свечи (закрытая + текущая)
            )

            if candles_data and len(candles_data) >= 2:
              candle_manager = self.candle_managers[symbol]

              # Обновляем закрытую свечу
              closed_candle = candles_data[-2]
              await candle_manager.update_candle(closed_candle, is_closed=True)

              # Обновляем текущую свечу
              current_candle = candles_data[-1]
              await candle_manager.update_candle(current_candle, is_closed=False)

          except Exception as e:
            logger.error(f"{symbol} | Ошибка обновления свечи: {e}")

        # Обновляем каждые 5 секунд
        await asyncio.sleep(5)

      except asyncio.CancelledError:
        logger.info("Цикл обновления свечей отменен")
        break
      except Exception as e:
        logger.error(f"Ошибка в цикле обновления свечей: {e}")
        await asyncio.sleep(10)

  async def _analysis_loop_ml_enhanced(self):
    """
    Продвинутый цикл анализа с ML и опциональными детекторами.

    Workflow:
    1. Получить данные (orderbook, candles)
    2. [OPTIONAL] Проверка детекторов манипуляций
    3. [OPTIONAL] Обновление S/R детектора
    4. Извлечение ML признаков
    5. [OPTIONAL] Strategy Manager consensus ИЛИ базовая генерация сигналов
    6. [OPTIONAL] ML валидация сигнала
    7. [OPTIONAL] S/R контекст
    8. Исполнение сигнала
    9. [OPTIONAL] Drift monitoring
    10. Сбор данных для ML обучения
    """
    # КРИТИЧНО: Импорты в начале функции для использования во всех блоках
    from models.signal import TradingSignal, SignalType, SignalStrength, SignalSource
    from datetime import datetime

    logger.info("🔄 Запущен продвинутый analysis loop (ML-Enhanced)")

    # Проверяем какие компоненты доступны
    has_spoofing_detector = hasattr(self, 'spoofing_detector') and self.spoofing_detector
    has_layering_detector = hasattr(self, 'layering_detector') and self.layering_detector
    has_sr_detector = hasattr(self, 'sr_detector') and self.sr_detector
    has_strategy_manager = hasattr(self, 'strategy_manager') and self.strategy_manager
    has_ml_validator = hasattr(self, 'ml_validator') and self.ml_validator
    has_drift_detector = hasattr(self, 'drift_detector') and self.drift_detector

    logger.info(
      f"📊 Доступные компоненты: "
      f"Spoofing={has_spoofing_detector}, "
      f"Layering={has_layering_detector}, "
      f"S/R={has_sr_detector}, "
      f"StrategyManager={has_strategy_manager}, "
      f"MLValidator={has_ml_validator}, "
      f"Drift={has_drift_detector}"
    )

    while self.status == BotStatus.RUNNING:
      try:
        # Ждем пока все WebSocket соединения установятся
        if not self.websocket_manager.is_all_connected():
          await asyncio.sleep(1)
          continue

        # Анализируем каждую пару
        for symbol in self.symbols:
          try:
            # ==================== 1. ПОЛУЧЕНИЕ ДАННЫХ ====================
            manager = self.orderbook_managers[symbol]
            candle_manager = self.candle_managers[symbol]

            # Пропускаем если нет данных
            if not manager.snapshot_received:
              continue

            # Получаем снимок стакана
            snapshot = manager.get_snapshot()
            if not snapshot:
              continue

            # Получаем свечи
            candles = candle_manager.get_candles()
            if not candles or len(candles) < 50:
              continue

            current_price = snapshot.mid_price
            if not current_price:
              continue

            # ==================== BROADCAST ORDERBOOK (КРИТИЧНО ДЛЯ ФРОНТЕНДА) ====================
            try:
              from api.websocket import broadcast_orderbook_update
              await broadcast_orderbook_update(symbol, snapshot.to_dict())
            except Exception as e:
              logger.error(f"{symbol} | Ошибка broadcast orderbook: {e}")

            # ==================== 2. ДЕТЕКТОРЫ МАНИПУЛЯЦИЙ (OPTIONAL) ====================
            manipulation_detected = False
            manipulation_details = []

            if has_spoofing_detector:
              try:
                self.spoofing_detector.update(snapshot)
                has_spoofing = self.spoofing_detector.is_spoofing_active(
                  symbol,
                  time_window_seconds=60
                )
                if has_spoofing:
                  manipulation_detected = True
                  manipulation_details.append("spoofing")
              except Exception as e:
                logger.error(f"{symbol} | Ошибка spoofing detector: {e}")

            if has_layering_detector:
              try:
                self.layering_detector.update(snapshot)
                has_layering = self.layering_detector.is_layering_active(
                  symbol,
                  time_window_seconds=60
                )
                if has_layering:
                  manipulation_detected = True
                  manipulation_details.append("layering")
              except Exception as e:
                logger.error(f"{symbol} | Ошибка layering detector: {e}")

            if manipulation_detected:
              logger.warning(
                f"⚠️  МАНИПУЛЯЦИИ [{symbol}]: "
                f"{', '.join(manipulation_details)} - "
                f"ТОРГОВЛЯ ЗАБЛОКИРОВАНА (признаки извлекаются)"
              )
              # НЕ делаем continue! Продолжаем извлечение признаков

            # ==================== 3. S/R ДЕТЕКТОР (OPTIONAL) ====================
            sr_levels = None
            if has_sr_detector:
              try:
                self.sr_detector.update_candles(symbol, candles)
                sr_levels = self.sr_detector.detect_levels(symbol)
              except Exception as e:
                logger.error(f"{symbol} | Ошибка S/R detector: {e}")

            # ==================== 4. ТРАДИЦИОННЫЙ АНАЛИЗ ====================
            # ПРАВИЛЬНО: передаём OrderBookManager, НЕ OrderBookSnapshot
            metrics = self.market_analyzer.analyze_symbol(symbol, manager)

            # ==================== BROADCAST METRICS (КРИТИЧНО ДЛЯ ФРОНТЕНДА) ====================
            try:
              from api.websocket import broadcast_metrics_update
              await broadcast_metrics_update(symbol, metrics.to_dict())
            except Exception as e:
              logger.error(f"{symbol} | Ошибка broadcast metrics: {e}")

            # Шаг 4a: Multi-Timeframe Analysis
            if config.enable_mtf:
              mtf_signal = await mtf_manager.analyze_symbol(
                symbol=symbol,
                orderbook=orderbook,
                metrics=metrics
              )

              if mtf_signal:
                # Используем MTF сигнал вместо single-TF
                final_signal = mtf_signal.signal

                # Модифицируем risk parameters
                position_size *= mtf_signal.recommended_position_size_multiplier
                stop_loss_price = mtf_signal.recommended_stop_loss_price
                take_profit_price = mtf_signal.recommended_take_profit_price

                # Quality check
                if mtf_signal.signal_quality < config.min_mtf_quality:
                  logger.info("MTF signal quality too low, skipping")
                  continue

                if mtf_signal.risk_level == "EXTREME":
                  logger.warning("EXTREME risk, skipping trade")
                  continue
              else:
                # Fallback к single-TF analysis
                logger.debug("No MTF signal, using single-TF")

            # ==================== 5. ML ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ====================
            feature_vector = None
            try:
              feature_vector = await self.ml_feature_pipeline.extract_features_single(
                symbol=symbol,
                orderbook_snapshot=snapshot,
                candles=candles
              )

              if feature_vector:
                logger.debug(
                  f"{symbol} | Извлечено {feature_vector.feature_count} ML признаков"
                )
            except Exception as e:
              logger.error(f"{symbol} | Ошибка извлечения признаков: {e}")

            # ==================== 6. ГЕНЕРАЦИЯ СИГНАЛОВ ====================
            signal = None
            consensus_info = None

            # БЛОКИРОВКА: Пропускаем генерацию сигналов если обнаружены манипуляции
            if manipulation_detected:
              logger.debug(
                f"{symbol} | Генерация сигналов пропущена из-за манипуляций: "
                f"{', '.join(manipulation_details)}"
              )
            else:
              # РЕЖИМ 1: Strategy Manager с Consensus (если доступен)
              if has_strategy_manager:
                try:
                  sr_levels = None
                  if has_sr_detector:
                    sr_levels = self.sr_detector.detect_levels(symbol)

                  # Получаем Volume Profile если доступен
                  volume_profile_data = None
                  if 'volume_profile' in self.strategy_manager.candle_strategies:
                    vp_strategy = self.strategy_manager.candle_strategies['volume_profile']

                    if symbol in vp_strategy.profiles:
                      profile = vp_strategy.profiles[symbol]
                      volume_profile_data = {
                        'poc_price': profile.poc_price,
                        'poc_volume': profile.poc_volume,
                        'value_area_high': profile.value_area_high,
                        'value_area_low': profile.value_area_low,
                        'hvn_nodes': [
                          {'price': node.price, 'volume': node.volume, 'strength': node.strength}
                          for node in profile.hvn_nodes
                        ],
                        'lvn_nodes': [
                          {'price': node.price, 'volume': node.volume, 'strength': node.strength}
                          for node in profile.lvn_nodes
                        ]
                      }

                  # Получаем ML предсказание если доступно
                  ml_prediction = None
                  if has_ml_validator and feature_vector:
                    try:
                      validation = await self.ml_validator.validate_signal(
                        symbol=symbol,
                        signal=None,  # Пока нет сигнала
                        features=feature_vector
                      )
                      ml_prediction = {
                        'confidence': validation.ml_confidence,
                        'prediction': 'bullish' if validation.should_trade else 'bearish'
                      }
                    except Exception as e:
                      logger.error(f"{symbol} | ML prediction error: {e}")

                  if self.adaptive_consensus_manager:
                    # ===== ADAPTIVE CONSENSUS РЕЖИМ =====
                    consensus = self.adaptive_consensus_manager.build_adaptive_consensus(
                      symbol=symbol,
                      candles=candles,
                      current_price=current_price,
                      orderbook=snapshot,
                      metrics=metrics,
                      sr_levels=sr_levels if has_sr_detector else None,
                      volume_profile=volume_profile_data,
                      ml_prediction=ml_prediction
                    )

                    if consensus:
                      logger.info(
                        f"✅ ADAPTIVE CONSENSUS [{symbol}]: "
                        f"{safe_enum_value(consensus.final_signal.signal_type)}, "
                        f"confidence={consensus.consensus_confidence:.2f}, "
                        f"quality={consensus.final_signal.metadata.get('consensus_quality', 0.0):.2f}"
                      )

                  else:
                    # ===== СТАНДАРТНЫЙ CONSENSUS РЕЖИМ =====
                    consensus = self.strategy_manager.analyze_with_consensus(
                      symbol=symbol,
                      candles=candles,
                      current_price=current_price,
                      orderbook=snapshot,
                      metrics=metrics,
                      sr_levels=sr_levels if has_sr_detector else None,
                      volume_profile=volume_profile_data,
                      ml_prediction=ml_prediction
                    )


                  # Проверяем что consensus не None и имеет нужные атрибуты
                  if consensus and hasattr(consensus, 'final_signal') and consensus.final_signal:
                    # Безопасное получение атрибутов с fallback значениями
                    contributing_strategies = getattr(consensus, 'contributing_strategies', [])
                    total_strategies = getattr(consensus, 'total_strategies', len(contributing_strategies))
                    agreement_count = getattr(consensus, 'agreement_count', len(contributing_strategies))
                    final_confidence = getattr(consensus, 'final_confidence', 0.7)

                    consensus_info = {
                      'signal_type': consensus.final_signal,
                      'strategies': contributing_strategies,
                      'agreement': f"{agreement_count}/{total_strategies}",
                      'confidence': final_confidence
                    }

                    # Создаём сигнал из consensus (импорты уже в начале функции)
                    # ИСПРАВЛЕНИЕ: final_signal это SignalType, не TradingSignal
                    final_signal_type = consensus.final_signal

                    # Если final_signal это строка, конвертируем в SignalType
                    if isinstance(final_signal_type, str):
                      final_signal_type = SignalType(final_signal_type)

                    signal = TradingSignal(
                      symbol=symbol,
                      signal_type=final_signal_type,  # ИСПРАВЛЕНО: используем final_signal_type
                      source=SignalSource.STRATEGY,  # Изменится на ML_VALIDATED после валидации
                      strength=(
                        SignalStrength.STRONG
                        if final_confidence > 0.7
                        else SignalStrength.MEDIUM
                      ),
                      price=current_price,
                      confidence=final_confidence,
                      timestamp=int(datetime.now().timestamp() * 1000),
                      reason=f"Consensus ({len(contributing_strategies)} strategies)",
                      metadata={
                        'consensus_strategies': contributing_strategies,
                        'consensus_agreement': consensus_info['agreement']
                      }
                    )

                    logger.info(
                      f"🎯 Strategy Manager Consensus [{symbol}]: "
                      f"{safe_enum_value(signal.signal_type)}, "  
                      f"confidence={final_confidence:.2f}, "
                      f"strategies={contributing_strategies}"
                    )
                except Exception as e:
                  logger.error(f"{symbol} | Ошибка Strategy Manager: {e}", exc_info=True)

              # РЕЖИМ 2: Базовая генерация сигналов (fallback)
              if not signal:
                try:
                  signal = self.strategy_engine.analyze_and_generate_signal(
                    symbol=symbol,
                    metrics=metrics,
                    features=feature_vector
                  )

                  if signal:
                    logger.debug(
                      f"🎯 Базовый сигнал [{symbol}]: "
                      f"{safe_enum_value(signal.signal_type)}, "
                      f"confidence={signal.confidence:.2f}"
                    )
                except Exception as e:
                  logger.error(f"{symbol} | Ошибка генерации сигнала: {e}", exc_info=True)

            # Если сигнала нет - пропускаем
            if not signal:
              # Всё равно собираем данные для ML
              if feature_vector and self.ml_data_collector:
                try:
                  await self.ml_data_collector.collect_sample(
                    symbol=symbol,
                    feature_vector=feature_vector,
                    orderbook_snapshot=snapshot,
                    market_metrics=metrics,
                    executed_signal=None
                  )
                except Exception as e:
                  logger.error(f"{symbol} | Ошибка сбора ML данных: {e}")
              continue

            # ==================== 7. ML ВАЛИДАЦИЯ (OPTIONAL) ====================
            if has_ml_validator and feature_vector and signal:
              try:
                validation_result = await self.ml_validator.validate(
                  signal,
                  feature_vector
                )

                # Проверяем результат валидации
                if validation_result.validated:
                  # ========================================
                  # ИСПРАВЛЕНИЕ: Проверяем fallback режим
                  # ========================================
                  is_fallback = validation_result.used_fallback

                  # 1. Меняем source на ML_VALIDATED
                  signal.source = SignalSource.ML_VALIDATED

                  # 2. Обновляем confidence
                  # ИСПРАВЛЕНИЕ: В fallback режиме НЕ понижаем confidence
                  if is_fallback:
                    # Используем оригинальный confidence из стратегии
                    signal.confidence = validation_result.final_confidence
                    logger.info(
                      f"🔄 ML Fallback режим [{symbol}]: "
                      f"используем стратегию confidence={signal.confidence:.2f}"
                    )
                  else:
                    # ML доступна - используем ML confidence
                    signal.confidence = validation_result.final_confidence
                    logger.info(
                      f"🤖 ML валидация [{symbol}]: "
                      f"ML confidence={validation_result.ml_confidence:.2f}, "
                      f"final={signal.confidence:.2f}"
                    )

                  # 3. Пересчитываем strength на основе final confidence
                  # ИСПРАВЛЕНИЕ: Используем более мягкие пороги для fallback
                  if is_fallback:
                    # В fallback режиме используем пороги стратегии
                    if signal.confidence > 0.7:
                      signal.strength = SignalStrength.STRONG
                    elif signal.confidence > 0.5:
                      signal.strength = SignalStrength.MEDIUM
                    else:
                      signal.strength = SignalStrength.WEAK
                  else:
                    # С ML используем стандартные пороги
                    if validation_result.final_confidence > 0.8:
                      signal.strength = SignalStrength.STRONG
                    elif validation_result.final_confidence > 0.6:
                      signal.strength = SignalStrength.MEDIUM
                    else:
                      signal.strength = SignalStrength.WEAK

                  # 4. Добавляем ML метаданные
                  if not signal.metadata:
                    signal.metadata = {}

                  signal.metadata['ml_validated'] = True
                  signal.metadata['ml_fallback'] = is_fallback
                  signal.metadata['ml_direction'] = validation_result.ml_direction
                  signal.metadata['ml_confidence'] = validation_result.ml_confidence

                  # Дополнительные метрики
                  if validation_result.predicted_mae:
                    signal.metadata['predicted_mae'] = validation_result.predicted_mae
                  if validation_result.manipulation_risk:
                    signal.metadata['manipulation_risk'] = validation_result.manipulation_risk
                  if validation_result.market_regime:
                    signal.metadata['market_regime'] = validation_result.market_regime.value
                  if validation_result.feature_quality:
                    signal.metadata['feature_quality'] = validation_result.feature_quality

                  logger.info(
                    f"✅ Сигнал подтвержден ML Validator [{symbol}]: "
                    f"source=ML_VALIDATED, "
                    f"strength={safe_enum_value(signal.strength)}, "
                    f"final_confidence={signal.confidence:.2f}, "
                    f"fallback={is_fallback}"
                  )
                else:
                  # Валидация не прошла
                  logger.warning(
                    f"❌ ML Validator отклонил сигнал [{symbol}]: "
                    f"reason={validation_result.reason}"
                  )
                  signal = None  # Отклоняем сигнал

              except Exception as e:
                logger.error(f"{symbol} | Ошибка ML Validator: {e}", exc_info=True)
                # В случае ошибки оставляем сигнал как есть (fallback)
                logger.info(
                  f"⚠️ ML Validator error, используем сигнал стратегии [{symbol}]"
                )

            # ==================== 8. S/R КОНТЕКСТ (OPTIONAL) ====================
            sr_context = []
            if has_sr_detector and sr_levels and signal:
              try:
                nearest_levels = self.sr_detector.get_nearest_levels(
                  symbol,
                  current_price,
                  max_distance_pct=0.02
                )

                if nearest_levels.get("support"):
                  support = nearest_levels["support"]
                  sr_context.append(
                    f"Support: ${support.price:.2f} "
                    f"(strength={support.strength:.2f})"
                  )

                if nearest_levels.get("resistance"):
                  resistance = nearest_levels["resistance"]
                  sr_context.append(
                    f"Resistance: ${resistance.price:.2f} "
                    f"(strength={resistance.strength:.2f})"
                  )

                if sr_context:
                  if not signal.metadata:
                    signal.metadata = {}
                  signal.metadata['sr_context'] = sr_context
              except Exception as e:
                logger.error(f"{symbol} | Ошибка S/R context: {e}")

            # ==================== 9. ФИНАЛЬНЫЙ ЛОГ И ИСПОЛНЕНИЕ ====================
            # ИСПРАВЛЕНИЕ: Проверяем что signal существует и является TradingSignal
            if signal:
              try:
                # КРИТИЧНО: Проверяем тип объекта signal перед использованием
                if not isinstance(signal, TradingSignal):
                  logger.error(
                    f"{symbol} | КРИТИЧЕСКАЯ ОШИБКА: signal имеет неправильный тип: {type(signal)}. "
                    f"Ожидается TradingSignal. Пропускаем исполнение."
                  )
                  continue

                # Формируем лог с проверками атрибутов
                log_parts = [
                  f"🎯 ФИНАЛЬНЫЙ СИГНАЛ [{symbol}]:",
                  f"{safe_enum_value(signal.signal_type)}",
                  f"confidence={signal.confidence:.2f}",
                  f"strength={safe_enum_value(signal.strength)}"
                ]

                if consensus_info:
                  log_parts.append(
                    f"strategies={consensus_info['strategies']}"
                  )

                if signal.metadata and signal.metadata.get('ml_validated'):
                  log_parts.append("ML_VALIDATED")

                if sr_context:
                  log_parts.append(f"SR: {', '.join(sr_context)}")

                logger.info(" | ".join(log_parts))

                # Отправляем на исполнение
                await self.execution_manager.submit_signal(signal)

                # Уведомляем фронтенд
                try:
                  # Конвертируем TradingSignal в dict ПЕРЕД broadcast
                  signal_dict = signal.to_dict()

                  # КРИТИЧНО: Конвертируем все Enum в строки
                  if 'signal_type' in signal_dict and hasattr(signal_dict['signal_type'], 'value'):
                    signal_dict['signal_type'] = signal_dict['signal_type'].value

                  if 'strength' in signal_dict and hasattr(signal_dict['strength'], 'value'):
                    signal_dict['strength'] = signal_dict['strength'].value

                  if 'source' in signal_dict and hasattr(signal_dict['source'], 'value'):
                    signal_dict['source'] = signal_dict['source'].value

                  logger.debug(
                    f"{symbol} | Подготовлен signal_dict для broadcast: "
                    f"type={type(signal_dict)}, "
                    f"signal_type={signal_dict.get('signal_type')}"
                  )

                  from api.websocket import broadcast_signal
                  await broadcast_signal(signal_dict)

                except Exception as e:
                  logger.error(
                    f"{symbol} | Ошибка broadcast_signal: {e}. "
                    f"signal_type={type(getattr(signal, 'signal_type', None))}, "
                    f"strength={type(getattr(signal, 'strength', None))}, "
                    f"source={type(getattr(signal, 'source', None))}",
                    exc_info=True
                  )

              except AttributeError as e:
                logger.error(
                  f"{symbol} | AttributeError при обработке сигнала: {e}. "
                  f"Тип signal: {type(signal)}, "
                  f"Атрибуты: {dir(signal) if signal else 'None'}",
                  exc_info=True
                )
                continue
              except Exception as e:
                logger.error(
                  f"{symbol} | Ошибка исполнения сигнала: {e}",
                  exc_info=True
                )
                continue

            # ==================== 10. DRIFT MONITORING (OPTIONAL) ====================
            if has_drift_detector and feature_vector and signal:
              try:
                # Конвертируем SignalType enum в int для drift detector
                # SignalType.BUY -> 1, SignalType.SELL -> 2, SignalType.HOLD -> 0
                signal_type_value = safe_enum_value(signal.signal_type)  # Получаем строку "BUY", "SELL", "HOLD"
                signal_type_map = {
                  "BUY": 1,
                  "SELL": 2,
                  "HOLD": 0
                }
                prediction_int = signal_type_map.get(
                  signal_type_value,
                  0
                )

                self.drift_detector.add_observation(
                  features=feature_vector.to_array(),
                  prediction=prediction_int,
                  label=None  # Label будет установлен позже
                )

                # Периодическая проверка drift
                if self.drift_detector.should_check_drift():
                  drift_metrics = self.drift_detector.check_drift()

                  if drift_metrics and drift_metrics.drift_detected:
                    logger.warning(
                      f"⚠️  MODEL DRIFT ОБНАРУЖЕН:\n"
                      f"   Severity: {drift_metrics.severity}\n"
                      f"   Feature drift: {drift_metrics.feature_drift_score:.4f}\n"
                      f"   Prediction drift: {drift_metrics.prediction_drift_score:.4f}\n"
                      f"   Recommendation: {drift_metrics.recommendation}"
                    )

                    # Сохраняем drift history
                    try:
                      self.drift_detector.save_drift_history(
                        f"logs/drift_history_{symbol}.json"
                      )
                    except Exception as e:
                      logger.error(f"Ошибка сохранения drift history: {e}")
              except Exception as e:
                logger.error(f"{symbol} | Ошибка drift monitoring: {e}")

            # ==================== 11. СБОР ДАННЫХ ДЛЯ ML ОБУЧЕНИЯ ====================
            if feature_vector and self.ml_data_collector:
              try:
                await self.ml_data_collector.collect_sample(
                  symbol=symbol,
                  feature_vector=feature_vector,
                  orderbook_snapshot=snapshot,
                  market_metrics=metrics,
                  executed_signal={
                    "type": safe_enum_value(signal.signal_type),  # Получаем строковое значение enum
                    "confidence": signal.confidence,
                    "strength": safe_enum_value(signal.strength),  # Тоже enum
                  } if signal else None
                )
              except Exception as e:
                logger.error(f"{symbol} | Ошибка сбора ML данных: {e}")

            # Логирование статистики по стратегиям

            stats = self.strategy_manager.get_statistics()

            logger.info(
              f"Strategy Manager Stats: "
              f"total_analyses={stats['total_analyses']}, "
              f"signals={stats['signals_generated']}, "
              f"consensus_rate={stats['consensus_rate']:.2%}"
            )

            # Статистика по стратегиям
            for strategy_name, strategy_stats in stats['strategies'].items():
              logger.debug(f"[{strategy_name}] {strategy_stats}")

          except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}", exc_info=True)
            log_exception(logger, e, f"Анализ {symbol}")

        # Пауза между циклами
        await asyncio.sleep(0.5)  # 500ms

      except asyncio.CancelledError:
        logger.info("Цикл анализа отменен")
        break
      except Exception as e:
        logger.error(f"Критическая ошибка в цикле анализа: {e}", exc_info=True)
        log_exception(logger, e, "Цикл анализа")
        await asyncio.sleep(1)


  async def stop(self):
    """Остановка бота."""
    if self.status == BotStatus.STOPPED:
      logger.warning("Бот уже остановлен")
      return

    try:
      self.status = BotStatus.STOPPING
      logger.info("=" * 80)
      logger.info("ОСТАНОВКА ТОРГОВОГО БОТА")
      logger.info("=" * 80)

      # Останавливаем задачи
      tasks_to_cancel = []

      # ===== SCREENER MANAGER (НОВОЕ) =====
      if self.screener_broadcast_task:
        self.screener_broadcast_task.cancel()

      if self.screener_manager:
        logger.info("Остановка Screener Manager...")
        await self.screener_manager.stop()
        logger.info("✓ Screener Manager остановлен")

      if self.analysis_task:
        tasks_to_cancel.append(self.analysis_task)

      if self.candle_update_task:  # НОВОЕ
        tasks_to_cancel.append(self.candle_update_task)

      if self.websocket_task:
        tasks_to_cancel.append(self.websocket_task)

      for task in tasks_to_cancel:
        task.cancel()
        try:
          await task
        except asyncio.CancelledError:
          pass

      # ===== НОВОЕ: Финализация ML Data Collector =====
      if self.ml_data_collector:
        await self.ml_data_collector.finalize()
        logger.info("✓ ML Data Collector финализирован")

      # Останавливаем остальные компоненты
      if self.websocket_manager:
        await self.websocket_manager.stop()
        logger.info("✓ WebSocket соединения остановлены")

      if self.execution_manager:
        await self.execution_manager.stop()
        logger.info("✓ Менеджер исполнения остановлен")

      # ========== Останавливаем Daily Loss Killer ==========
      await daily_loss_killer.stop()
      logger.info("✓ Daily Loss Killer остановлен")

      if self.balance_tracker:
        await self.balance_tracker.stop()
        logger.info("✓ Трекер баланса остановлен")

      # ========== Остановка обновления корреляций ==========
      if self.correlation_update_task:
        self.correlation_update_task.cancel()
        try:
          await self.correlation_update_task
        except asyncio.CancelledError:
          pass

      if self.symbols_refresh_task:
        self.symbols_refresh_task.cancel()
        try:
          await self.symbols_refresh_task
        except asyncio.CancelledError:
          pass
        logger.info("✓ Symbols refresh task остановлен")

      # ============================================
      # ML SIGNAL VALIDATOR - Остановка
      # ============================================
      # КРИТИЧЕСКИ ВАЖНО: Используем cleanup() вместо stop()
      if hasattr(self, 'ml_validator') and self.ml_validator:
        try:
          logger.info("🤖 Останавливаем ML Signal Validator...")
          await self.ml_validator.cleanup()  # ← ИСПРАВЛЕНО: cleanup() вместо stop()
          logger.info("✅ ML Signal Validator остановлен")
        except Exception as e:
          logger.error(f"❌ Ошибка при остановке ML validator: {e}")

      # ==========================================
      # ОСТАНОВКА TRAILING STOP MANAGER
      # ==========================================
      logger.info("Остановка Trailing Stop Manager...")
      await trailing_stop_manager.stop()

      # Остановка Position Monitor
      if self.position_monitor:
        await self.position_monitor.stop()
        logger.info("✓ Position Monitor остановлен")


      self.status = BotStatus.STOPPED
      logger.info("=" * 80)
      logger.info("БОТ УСПЕШНО ОСТАНОВЛЕН")
      logger.info("=" * 80)

      # Уведомляем фронтенд
      from api.websocket import broadcast_bot_status
      await broadcast_bot_status("stopped", {
        "message": "Бот успешно остановлен"
      })

    except Exception as e:
      self.status = BotStatus.ERROR
      logger.error(f"Ошибка остановки бота: {e}")
      log_exception(logger, e, "Остановка бота")
      raise

  async def _correlation_update_loop(self):
    """
    Периодическое обновление корреляций.

    Запускается раз в день для пересчета корреляционных групп
    при изменении списка торговых пар.
    """
    logger.info("Запущен цикл обновления корреляций (каждые 24 часа)")

    while self.running:
      try:
        # Ждем 24 часа
        await asyncio.sleep(24 * 3600)

        if not self.running:
          break

        logger.info("Время обновления корреляций...")

        # Если символы изменились - пересчитываем корреляции
        current_symbols = set(self.symbols)
        registered_symbols = set(correlation_manager.group_manager.symbol_to_group.keys())

        if current_symbols != registered_symbols:
          logger.warning(
            f"⚠️ Список символов изменился! "
            f"Старые: {len(registered_symbols)}, Новые: {len(current_symbols)}"
          )

          # Пересчитываем корреляции для новых символов
          await correlation_manager.update_correlations(list(current_symbols))

          logger.info("✓ Корреляции пересчитаны для обновленного списка символов")
        else:
          # Просто обновляем существующие корреляции
          await correlation_manager.update_correlations(self.symbols)
          logger.info("✓ Корреляции обновлены")

      except asyncio.CancelledError:
        logger.info("Задача обновления корреляций отменена")
        break
      except Exception as e:
        logger.error(f"Ошибка в цикле обновления корреляций: {e}", exc_info=True)
        # Продолжаем работу даже при ошибке
        await asyncio.sleep(3600)  # Повторная попытка через 1 час

  async def _handle_reversal_signal(
        self,
        symbol: str,
        reversal: ReversalSignal,
        position: Dict
    ):
      """
      Обработка сигнала разворота.

      Args:
          symbol: Торговая пара
          reversal: Сигнал разворота
          position: Информация о позиции из RiskManager
      """
      try:
        if reversal.suggested_action == "close_position":
          logger.warning(
            f"{symbol} | 🚨 CRITICAL REVERSAL DETECTED | "
            f"Strength: {reversal.strength.value} | "
            f"Confidence: {reversal.confidence:.2%} | "
            f"Reason: {reversal.reason}"
          )

          if reversal_detector.auto_action:
            logger.warning(
              f"{symbol} | AUTO-CLOSING position due to critical reversal"
            )

            # Находим position_id в БД
            position_in_db = await position_repository.find_open_by_symbol(symbol)

            if position_in_db:
              current_price = position.get('entry_price', 0) * 1.01  # Fallback

              # Или получаем из OrderBook Manager
              orderbook_manager = self.orderbook_managers.get(symbol)
              if orderbook_manager:
                snapshot = orderbook_manager.get_snapshot()
                if snapshot and snapshot.mid_price:
                  current_price = snapshot.mid_price

              # Закрываем позицию через ExecutionManager
              await self.execution_manager.close_position(
                position_id=str(position_in_db.id),
                exit_price=current_price,
                exit_reason=f"Critical reversal: {reversal.reason}",
                exit_signal={
                  "type": "reversal",
                  "strength": reversal.strength.value,
                  "indicators": reversal.indicators_confirming,
                  "confidence": reversal.confidence
                }
              )

              logger.info(
                f"{symbol} | ✓ Position closed due to critical reversal"
              )
            else:
              logger.error(
                f"{symbol} | Position found in RiskManager but not in DB!"
              )
          else:
            logger.warning(
              f"{symbol} | ⚠️ MANUAL INTERVENTION REQUIRED | "
              f"Auto-action disabled - please close position manually"
            )

        elif reversal.suggested_action == "reduce_size":
          logger.warning(
            f"{symbol} | 🔶 STRONG REVERSAL | "
            f"Strength: {reversal.strength.value} | "
            f"Suggestion: Reduce position size by 50%"
          )

          # TODO: Реализовать частичное закрытие позиции
          # Требуется добавить метод partial_close в ExecutionManager
          logger.info(
            f"{symbol} | Partial close not yet implemented - "
            f"consider manual reduction"
          )

        elif reversal.suggested_action == "tighten_sl":
          logger.warning(
            f"{symbol} | 🔸 MODERATE REVERSAL | "
            f"Strength: {reversal.strength.value} | "
            f"Suggestion: Tighten stop loss"
          )

          # TODO: Реализовать динамическое обновление SL
          # Требуется добавить метод update_stop_loss в ExecutionManager
          logger.info(
            f"{symbol} | Stop loss update not yet implemented - "
            f"consider manual adjustment"
          )

        else:
          logger.debug(
            f"{symbol} | Weak reversal detected, no action required"
          )

      except Exception as e:
        logger.error(
          f"{symbol} | Error handling reversal signal: {e}",
          exc_info=True
        )

  async def _handle_orderbook_message(self, data: Dict[str, Any]):
    """
    Обработка сообщения о стакане от WebSocket.

    Args:
        data: Данные от WebSocket
    """
    try:
      topic = data.get("topic", "")
      message_type = data.get("type", "")
      message_data = data.get("data", {})

      # Извлекаем символ из топика
      if "orderbook" in topic:
        parts = topic.split(".")
        if len(parts) >= 3:
          symbol = parts[2]

          if symbol not in self.orderbook_managers:
            logger.warning(f"Получены данные для неизвестного символа: {symbol}")
            return

          manager = self.orderbook_managers[symbol]

          if message_type == "snapshot":
            logger.info(f"{symbol} | Получен snapshot стакана")
            manager.apply_snapshot(message_data)
            logger.info(
              f"{symbol} | Snapshot применен: "
              f"{len(manager.bids)} bids, {len(manager.asks)} asks"
            )

          elif message_type == "delta":
            if not manager.snapshot_received:
              logger.debug(
                f"{symbol} | Delta получена до snapshot, пропускаем"
              )
              return

            manager.apply_delta(message_data)
            logger.debug(f"{symbol} | Delta применена")
          else:
            logger.warning(f"{symbol} | Неизвестный тип сообщения: {message_type}")

    except Exception as e:
      logger.error(f"Ошибка обработки сообщения стакана: {e}")
      if not isinstance(e, (OrderBookSyncError, OrderBookError)):
        log_exception(logger, e, "Обработка сообщения стакана")

  def get_status(self) -> Dict[str, Any]:
    """Получение статуса бота с расширенной ML аналитикой."""

    # ========================================
    # СУЩЕСТВУЮЩАЯ ЛОГИКА (БЕЗ ИЗМЕНЕНИЙ)
    # ========================================

    ws_status: Dict[Any, Any] = {}
    if self.websocket_manager:
      ws_status = self.websocket_manager.get_connection_statuses()

    # ===== СУЩЕСТВУЮЩАЯ ML статистика =====
    ml_status: Dict[str, Any] = {
      "features_extracted": len(self.latest_features),
      "data_collected_samples": (
        self.ml_data_collector.get_statistics()
        if self.ml_data_collector else {}
      )
    }

    # ========================================
    # РАСШИРЕНИЕ ml_status НОВЫМИ МЕТРИКАМИ
    # ========================================

    # Добавляем статус ML интеграции
    try:
      ml_status["ml_integration_enabled"] = getattr(
        settings, 'ML_RISK_INTEGRATION_ENABLED', False
      )
    except Exception:
      ml_status["ml_integration_enabled"] = False

    # ML Validator статистика
    if hasattr(self, 'ml_validator') and self.ml_validator:
      try:
        validator_stats = self.ml_validator.get_statistics()
        ml_status["validator"] = {
          "total_validations": validator_stats.get("total_validations", 0),
          "ml_success_count": validator_stats.get("ml_success_count", 0),
          "fallback_count": validator_stats.get("fallback_count", 0),
          "agreement_count": validator_stats.get("agreement_count", 0),
          "ml_server_available": validator_stats.get("ml_server_available", False),
          "success_rate": validator_stats.get("success_rate", 0.0),
          "agreement_rate": validator_stats.get("agreement_rate", 0.0),
          "fallback_rate": validator_stats.get("fallback_rate", 0.0),
          # Расширенные метрики
          "avg_mae": validator_stats.get("avg_mae"),
          "avg_manipulation_risk": validator_stats.get("avg_manipulation_risk", 0.0)
        }
      except Exception as e:
        logger.debug(f"Cannot get ML validator stats: {e}")
        ml_status["validator"] = {"status": "unavailable"}
    else:
      ml_status["validator"] = {"status": "not_initialized"}

    # ML-Enhanced Risk Manager статистика
    if (
        hasattr(self, 'risk_manager') and
        hasattr(self.risk_manager, 'get_ml_stats')
    ):
      try:
        ml_risk_stats = self.risk_manager.get_ml_stats()
        ml_status["risk_manager"] = {
          "ml_enabled": ml_risk_stats.get("ml_enabled", False),
          "total_validations": ml_risk_stats.get("total_validations", 0),
          "ml_used": ml_risk_stats.get("ml_used", 0),
          "ml_rejected": ml_risk_stats.get("ml_rejected", 0),
          "fallback_used": ml_risk_stats.get("fallback_used", 0),
          "ml_usage_rate": ml_risk_stats.get("ml_usage_rate", 0.0),
          "ml_rejection_rate": ml_risk_stats.get("ml_rejection_rate", 0.0)
        }
      except Exception as e:
        logger.debug(f"Cannot get ML risk manager stats: {e}")
        ml_status["risk_manager"] = {"status": "unavailable"}
    else:
      ml_status["risk_manager"] = {"status": "standard_mode"}

    # Feature Pipeline статистика
    if hasattr(self, 'ml_feature_pipeline') and self.ml_feature_pipeline:
      try:
        symbols_with_features = list(self.latest_features.keys()) if hasattr(self, 'latest_features') else []
        ml_status["feature_pipeline"] = {
          "active": True,
          "symbols_count": len(symbols_with_features),
          "recent_symbols": symbols_with_features[:10]
        }
      except Exception as e:
        logger.debug(f"Cannot get feature pipeline stats: {e}")
        ml_status["feature_pipeline"] = {"active": False}
    else:
      ml_status["feature_pipeline"] = {"active": False}

    # ========================================
    # БАЗОВЫЙ RETURN (СУЩЕСТВУЮЩАЯ СТРУКТУРА)
    # ========================================

    status_dict: Dict[str, Any] = {
      "status": self.status.value,
      "symbols": self.symbols,
      "ml_enabled": True,  # СУЩЕСТВУЮЩЕЕ
      "ml_status": ml_status,  # РАСШИРЕННОЕ
      "websocket_connections": ws_status,
      "orderbook_managers": {
        symbol: manager.get_stats()
        for symbol, manager in self.orderbook_managers.items()
      },
      "execution_stats": (
        self.execution_manager.get_statistics()
        if self.execution_manager else {}
      ),
    }

    # ========================================
    # ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ (НОВЫЕ КЛЮЧИ)
    # ========================================

    # Risk Manager metrics
    if hasattr(self, 'risk_manager') and self.risk_manager:
      try:
        status_dict["risk_metrics"] = self.risk_manager.metrics.to_dict()
        status_dict["active_positions"] = len(self.risk_manager.open_positions)
        status_dict["open_positions_list"] = list(
          self.risk_manager.open_positions.keys()
        )
      except Exception as e:
        logger.debug(f"Cannot get risk metrics: {e}")

    # Balance Tracker
    try:
      from utils.balance_tracker import balance_tracker
      balance_stats = balance_tracker.get_stats()
      status_dict["balance"] = {
        "current": balance_stats.get("current_balance", 0.0),
        "initial": balance_stats.get("initial_balance", 0.0),
        "total_pnl": balance_stats.get("total_pnl", 0.0),
        "total_pnl_percentage": balance_stats.get("total_pnl_percentage", 0.0)
      }
    except Exception as e:
      logger.debug(f"Cannot get balance stats: {e}")

    # Daily Loss Killer
    try:
      from strategy.daily_loss_killer import daily_loss_killer
      dlk_stats = daily_loss_killer.get_statistics()
      status_dict["daily_loss_killer"] = {
        "trading_allowed": dlk_stats.get("is_allowed", True),
        "daily_pnl": dlk_stats.get("daily_pnl", 0.0),
        "daily_loss_percent": dlk_stats.get("daily_loss_percent", 0.0),
        "max_loss_percent": dlk_stats.get("max_daily_loss_percent", 0.0)
      }
    except Exception as e:
      logger.debug(f"Cannot get daily loss killer stats: {e}")

    # Correlation Manager
    try:
      from strategy.correlation_manager import correlation_manager
      corr_stats = correlation_manager.get_statistics()
      status_dict["correlation_stats"] = {
        "total_groups": corr_stats.get("total_groups", 0),
        "total_symbols": corr_stats.get("total_symbols", 0),
        "active_positions": corr_stats.get("active_positions", 0)
      }
    except Exception as e:
      logger.debug(f"Cannot get correlation stats: {e}")

    # Position Monitor (если есть)
    if hasattr(self, 'position_monitor') and self.position_monitor:
      try:
        status_dict["position_monitor"] = self.position_monitor.get_statistics()
      except Exception as e:
        logger.debug(f"Cannot get position monitor stats: {e}")

    # Timestamp
    status_dict["timestamp"] = datetime.now().isoformat()

    return status_dict

  async def _ml_stats_loop(self):
    """
    Периодический вывод статистики сбора ML данных.

    Выводит:
    - Общую статистику (всего семплов, файлов)
    - Детальную статистику по каждому символу
    """
    logger.info("Запущен цикл мониторинга ML статистики")

    while True:
      try:
        await asyncio.sleep(300)  # Каждые 5 минут

        if self.ml_data_collector:
          stats = self.ml_data_collector.get_statistics()

          # ===== ИСПРАВЛЕНИЕ: Выводим общую статистику =====
          logger.info(
            f"ML Stats | ОБЩАЯ: "
            f"всего_семплов={stats['total_samples_collected']:,}, "
            f"файлов={stats['files_written']}, "
            f"итераций={stats['iteration_counter']}, "
            f"интервал={stats['collection_interval']}"
          )

          # ===== ИСПРАВЛЕНИЕ: Итерируемся по stats["symbols"], а не stats =====
          symbol_stats = stats.get("symbols", {})

          if not symbol_stats:
            logger.info("ML Stats | Нет данных по символам")
          else:
            for symbol, stat in symbol_stats.items():
              # ===== ИСПРАВЛЕНИЕ: Используем правильные ключи =====
              logger.info(
                f"ML Stats | {symbol}: "
                f"samples={stat['total_samples']:,}, "
                f"batch={stat['current_batch']}, "  # ← НЕ 'batches_saved'
                f"buffer={stat['buffer_size']}/{self.ml_data_collector.max_samples_per_file}"
              )

      except asyncio.CancelledError:
        logger.info("ML stats loop остановлен (CancelledError)")
        break
      except Exception as e:
        logger.error(f"Ошибка в ML stats loop: {e}")
        # Логируем полный traceback для диагностики
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")

  async def _screener_broadcast_loop(self):
    """
    Цикл рассылки данных скринера через WebSocket.
    Отправляет обновления каждые N секунд.
    """
    from api.websocket import broadcast_screener_update

    interval = settings.SCREENER_BROADCAST_INTERVAL
    logger.info(f"Запущен screener broadcast loop (интервал: {interval}s)")

    while self.status == BotStatus.RUNNING:
      try:
        if self.screener_manager:
          pairs = self.screener_manager.get_all_pairs()
          await broadcast_screener_update(pairs)

        await asyncio.sleep(interval)

      except asyncio.CancelledError:
        break
      except Exception as e:
        logger.error(f"Ошибка в screener broadcast loop: {e}")
        await asyncio.sleep(interval)

  # ============================================================================
  # BACKGROUND TASK: Weight Optimization Loop
  # ============================================================================

  async def _weight_optimization_loop(self):
    """
    Фоновый цикл оптимизации весов стратегий (Adaptive Consensus).

    Частота: Каждые 6 часов (по умолчанию)
    """
    logger.info("🔄 Weight Optimization Loop started")

    if not self.adaptive_consensus:
      logger.warning("⚠️ Adaptive Consensus не инициализирован, loop остановлен")
      return

    error_count = 0
    max_errors = 5

    while self.status == BotStatus.RUNNING:
      try:
        # Оптимизация весов для каждого символа
        for symbol in self.symbols:
          try:
            update_result = await self.adaptive_consensus.optimize_weights(symbol)

            if update_result:
              logger.info(
                f"⚖️ [{symbol}] Веса обновлены: "
                f"изменено {update_result['strategies_updated']} стратегий"
              )
              self.stats['adaptive_weight_updates'] += 1

          except Exception as e:
            logger.error(f"❌ Ошибка оптимизации весов для {symbol}: {e}")

        # Reset error counter
        error_count = 0

        # Интервал обновления (по умолчанию 6 часов)
        await asyncio.sleep(settings.ADAPTIVE_WEIGHT_UPDATE_FREQUENCY_SECONDS)

      except Exception as e:
        error_count += 1
        logger.error(f"❌ Ошибка в Weight Optimization Loop: {e}")

        if error_count >= max_errors:
          logger.critical(f"🚨 Weight Optimization Loop: превышен лимит ошибок")
          break

        await asyncio.sleep(3600)  # 1 hour

    logger.warning("⚠️ Weight Optimization Loop остановлен")

  # ============================================================================
  # BACKGROUND TASK: MTF Update Loop
  # ============================================================================

  async def _mtf_update_loop(self):
    """
    Фоновый цикл staggered обновления MTF таймфреймов.

    Функции:
    - Обновление свечей на разных таймфреймах
    - Staggered updates (не все TF одновременно)
    - Валидация данных
    """
    logger.info("🔄 MTF Update Loop started")

    if not self.mtf_manager:
      logger.warning("⚠️ MTF Manager не инициализирован, loop остановлен")
      return

    error_count = 0
    max_errors = 10

    while self.status == BotStatus.RUNNING:
      try:
        # Обновление таймфреймов для всех символов
        for symbol in self.symbols:
          try:
            await self.mtf_manager.update_all_timeframes(symbol)
          except Exception as e:
            logger.error(f"❌ Ошибка MTF update для {symbol}: {e}")

        # Reset error counter
        error_count = 0

        # Staggered interval (небольшая задержка между обновлениями)
        await asyncio.sleep(settings.MTF_STAGGERED_UPDATE_INTERVAL)

      except Exception as e:
        error_count += 1
        logger.error(f"❌ Ошибка в MTF Update Loop: {e}")

        if error_count >= max_errors:
          logger.critical(f"🚨 MTF Update Loop: превышен лимит ошибок")
          break

        await asyncio.sleep(60)

    logger.warning("⚠️ MTF Update Loop остановлен")

  # async def _initialize_risk_manager(self):
  #   """Инициализация Risk Manager."""
  #   # Создаём без баланса
  #   self.risk_manager = RiskManager(default_leverage=settings.DEFAULT_LEVERAGE)
  #
  #   # Получаем реальный баланс
  #   try:
  #     balance_data = await rest_client.get_wallet_balance()
  #     real_balance = balance_tracker._calculate_total_balance(balance_data)
  #
  #     # ИСПОЛЬЗУЕМ update_available_balance
  #     self.risk_manager.update_available_balance(real_balance)
  #
  #     logger.info(f"✓ Risk Manager обновлён балансом: {real_balance:.2f} USDT")
  #   except Exception as e:
  #     logger.error(f"Ошибка получения баланса: {e}")


  async def _initialize_risk_manager(self):
    """
    Инициализация Risk Manager с правильным балансом.

    ЛОГИКА:
    - Если ML_RISK_INTEGRATION_ENABLED=True → RiskManagerMLEnhanced
    - Если ML_RISK_INTEGRATION_ENABLED=False → обычный RiskManager
    - При ml_validator=None → RiskManagerMLEnhanced работает в fallback режиме
    """
    logger.info("=" * 80)
    logger.info("ИНИЦИАЛИЗАЦИЯ RISK MANAGER")
    logger.info("=" * 80)

    try:
        # Получаем реальный баланс
        balance_data = await rest_client.get_wallet_balance()
        real_balance = balance_tracker._calculate_total_balance(balance_data)

        logger.info(f"✓ Получен баланс с биржи: {real_balance:.2f} USDT")

        # ========================================
        # УСЛОВНАЯ ИНИЦИАЛИЗАЦИЯ RISK MANAGER
        # ========================================

        # Проверяем, включена ли ML интеграция
        ml_enabled = settings.ML_RISK_INTEGRATION_ENABLED

        if ml_enabled:
            # ========================================
            # ML-ENHANCED RISK MANAGER
            # ========================================
            logger.info("📊 Создание ML-Enhanced Risk Manager...")

            # Проверяем доступность ml_validator
            ml_validator_available = (
                hasattr(self, 'ml_validator') and
                self.ml_validator is not None
            )

            if ml_validator_available:
                logger.info(
                    f"✓ ML Validator доступен, будет использован для валидации"
                )
            else:
                logger.warning(
                    f"⚠️ ML Validator недоступен, Risk Manager будет работать "
                    f"в fallback режиме (как обычный RiskManager)"
                )

            # Создаем ML-Enhanced Risk Manager
            # ВАЖНО: Даже если ml_validator=None, он будет работать в fallback
            self.risk_manager = RiskManagerMLEnhanced(
                ml_validator=self.ml_validator if ml_validator_available else None,
                default_leverage=settings.DEFAULT_LEVERAGE,
                initial_balance=real_balance
            )

            logger.info(
                f"✅ ML-Enhanced Risk Manager инициализирован: "
                f"leverage={settings.DEFAULT_LEVERAGE}x, "
                f"balance=${real_balance:.2f}, "
                f"ml_validator={'enabled' if ml_validator_available else 'disabled (fallback)'}"
            )

        else:
            # ========================================
            # ОБЫЧНЫЙ RISK MANAGER (БЕЗ ML)
            # ========================================
            logger.info("📊 Создание обычного Risk Manager (ML отключен)...")

            self.risk_manager = RiskManager(
                default_leverage=settings.DEFAULT_LEVERAGE,
                initial_balance=real_balance
            )

            logger.info(
                f"✅ Risk Manager инициализирован: "
                f"leverage={settings.DEFAULT_LEVERAGE}x, "
                f"balance=${real_balance:.2f}, "
                f"mode=standard (без ML)"
            )

        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Ошибка инициализации Risk Manager: {e}", exc_info=True)
        raise

  async def _cleanup_on_error(self):
    """Cleanup частично инициализированных компонентов при ошибке."""
    logger.warning("⚠️ Выполняется cleanup после ошибки инициализации...")

    try:
      # Закрываем WebSocket соединения
      if self.websocket_manager:
        try:
          await self.websocket_manager.stop()
        except Exception as e:
          logger.error(f"Ошибка при cleanup WebSocket: {e}")

      # Закрываем ML Validator
      if hasattr(self, 'ml_validator') and self.ml_validator:
        try:
          await self.ml_validator.cleanup()
        except Exception as e:
          logger.error(f"Ошибка при cleanup ML Validator: {e}")

      logger.info("✓ Cleanup завершен")

    except Exception as e:
      logger.error(f"Ошибка в процессе cleanup: {e}")

# Глобальный контроллер бота
bot_controller: Optional[BotController] = None


@asynccontextmanager
async def lifespan(app):
  """
  Управление жизненным циклом приложения.

  Args:
      app: FastAPI приложение
  """
  global bot_controller

  # Startup
  logger.info("Запуск приложения")
  try:

    with trace_operation("app_startup"):
      # 1. Инициализация базы данных
      logger.info("→ Инициализация базы данных...")
      await db_manager.initialize()
      logger.info("✓ База данных подключена")

      # 2. Recovery & Reconciliation (если включено)
      if settings.ENABLE_AUTO_RECOVERY:
        logger.info("Запуск автоматического восстановления...")

        recovery_result = await recovery_service.recover_from_crash()

        if recovery_result["recovered"]:
          logger.info("✓ Автоматическое восстановление завершено успешно")

          # Логируем детали
          if recovery_result["hanging_orders"]:
            logger.warning(
              f"⚠ Обнаружено {len(recovery_result['hanging_orders'])} "
              f"зависших ордеров - требуется внимание!"
            )

          logger.info(
            f"FSM восстановлено: "
            f"{recovery_result['fsm_restored']['orders']} ордеров, "
            f"{recovery_result['fsm_restored']['positions']} позиций"
          )
        else:
          logger.error("✗ Ошибка автоматического восстановления")
          if "error" in recovery_result:
            logger.error(f"Детали: {recovery_result['error']}")
      else:
        logger.info("Автоматическое восстановление отключено в конфигурации")

      # Создаем и инициализируем контроллер
      bot_controller = BotController()
      await bot_controller.initialize()

      await cleanup_tasks.start()

    logger.info("=" * 80)
    logger.info("✓ ПРИЛОЖЕНИЕ ГОТОВО К РАБОТЕ")
    logger.info("=" * 80)

    yield

  except Exception as e:
    logger.error(f"Критическая ошибка при запуске: {e}")
    log_exception(logger, e, "Запуск приложения")
    raise

  finally:
    # Shutdown
    logger.info("Остановка приложения")

    # if bot_controller:
    #   if bot_controller.status == BotStatus.RUNNING:
    #     await bot_controller.stop()
    #
    #   # Закрываем REST клиент
    #   await rest_client.close()
    with trace_operation("app_shutdown"):
      if bot_controller:
        await bot_controller.stop()

      await rest_client.close()
      await db_manager.close()

      await cleanup_tasks.stop()

    logger.info("Приложение остановлено")

async def fsm_cleanup_task():
  """
  Background task для периодической очистки терминальных FSM.
  Освобождает память от завершенных FSM.
  """
  logger.info("FSM Cleanup Task запущен")

  while True:
    try:
      # Ждем 30 минут
      await asyncio.sleep(1800)

      logger.info("Запуск очистки терминальных FSM...")

      # Очищаем терминальные FSM
      cleared = fsm_registry.clear_terminal_fsms()

      logger.info(
        f"Очистка завершена: "
        f"ордеров - {cleared['orders_cleared']}, "
        f"позиций - {cleared['positions_cleared']}"
      )

      # Логируем статистику
      stats = fsm_registry.get_stats()
      logger.info(
        f"FSM Registry статистика: "
        f"ордеров - {stats['total_order_fsms']}, "
        f"позиций - {stats['total_position_fsms']}"
      )

    except Exception as e:
      logger.error(f"Ошибка в FSM cleanup task: {e}", exc_info=True)
      # Продолжаем работу даже при ошибке
      await asyncio.sleep(60)

# Импортируем FastAPI приложение и добавляем lifespan
from api.app import app

app.router.lifespan_context = lifespan

# Регистрируем роутеры
from api.routes import auth_router, bot_router, data_router, trading_router, monitoring_router, screener_router, \
  adaptive_router

app.include_router(auth_router)
app.include_router(bot_router)
app.include_router(data_router)
app.include_router(trading_router)
app.include_router(monitoring_router)
app.include_router(screener_router)
app.include_router(adaptive_router)
# WebSocket эндпоинт
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
  """
  WebSocket эндпоинт для фронтенда.

  Args:
      websocket: WebSocket соединение
  """
  await ws_manager.connect(websocket)

  try:
    await handle_websocket_messages(websocket)
  except WebSocketDisconnect:
    logger.info("WebSocket клиент отключен")
  except Exception as e:
    logger.error(f"Ошибка WebSocket: {e}")
  finally:
    ws_manager.disconnect(websocket)


def handle_shutdown_signal(signum, frame):
  """
  Обработчик сигналов завершения.

  Args:
      signum: Номер сигнала
      frame: Фрейм
  """
  logger.info(f"Получен сигнал завершения: {signum}")
  # Uvicorn обработает остановку автоматически


# Регистрируем обработчики сигналов
signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)

if __name__ == "__main__":
  """Точка входа при запуске напрямую."""

  logger.info("=" * 80)
  logger.info(f"Запуск {settings.APP_NAME} v{settings.APP_VERSION}")
  logger.info(f"Режим: {settings.BYBIT_MODE.upper()}")
  logger.info(f"Хост: {settings.API_HOST}:{settings.API_PORT}")
  logger.info("=" * 80)

  # Запускаем Uvicorn сервер
  uvicorn.run(
    "main:app",
    host=settings.API_HOST,
    port=settings.API_PORT,
    reload=settings.DEBUG,
    log_level=settings.LOG_LEVEL.lower(),
  )