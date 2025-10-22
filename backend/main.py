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
from strategy.analyzer import MarketAnalyzer, OrderBookAnalyzer
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
    self.orderbook_analyzer: Optional[OrderBookAnalyzer] = None
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
      logger.info(f"📊 Активные стратегии: {list(self.strategy_manager.all_strategies.keys())}")

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

            ),

            # Aligner Config
            aligner_config=AlignmentConfig(
              timeframe_weights={  # ✅ Вместо htf_weight, mtf_weight, ltf_weight
                Timeframe.H1: 0.50,
                Timeframe.M15: 0.30,
                Timeframe.M5: 0.15,
                Timeframe.M1: 0.05
              },  # Lower Timeframe weight
              min_alignment_score=0.65,
              confluence_price_tolerance_percent=0.5,
              min_timeframes_for_confluence=1,  # ✅ Вместо min_confluence_zones
              allow_trend_counter_signals=False
            ),

            # Synthesizer Config
            synthesizer_config=SynthesizerConfig(
              mode=SynthesisMode(mtf_synthesis_mode),  # ✅ mode, НЕ synthesis_mode
              min_signal_quality=mtf_min_quality,  # ✅ Вместо min_quality_threshold
              min_timeframes_required=2,  # ✅ Вместо min_timeframes_for_signal
              enable_dynamic_position_sizing=True,
              max_position_multiplier=1.5,  # ✅ Вместо position_size_multiplier_range
              min_position_multiplier=0.3,  # ✅
              use_higher_tf_for_stops=True,  # ✅ Вместо enable_smart_sl
              atr_multiplier_for_stops=2.0  # ✅ Вместо default_risk_reward_ratio

            ),

            # Quality Control


            # Fallback
            fallback_to_single_tf=True,

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


          # Fallback

        )

        self.integrated_engine = IntegratedAnalysisEngine(integrated_config)

        # Инициализация символов в Integrated Engine
        for symbol in self.symbols:
          await self.integrated_engine.initialize_symbol(symbol)
          logger.info(f"✅ {symbol}: Integrated Engine инициализирован")

        logger.info("✅ Integrated Analysis Engine инициализирован")
        logger.info(f"📊 Режим анализа: {integrated_mode}")

        # ========== STATISTICS & DIAGNOSTICS ==========
        self.stats = {
          'signals_generated': 0,
          'signals_executed': 0,
          'orders_placed': 0,
          'positions_opened': 0,
          'positions_closed': 0,
          'total_pnl': 0.0,
          'consensus_achieved': 0,
          'consensus_failed': 0,
          'mtf_signals': 0,
          'adaptive_weight_updates': 0,
          'ml_validations': 0,
          'analysis_cycles': 0,
          'errors': 0,
          'warnings': 0
        }

        # ========== CONFIGURATION SNAPSHOT ==========
        # Сохраняем ключевые настройки для диагностики
        self.config_snapshot = {
          'trading_pairs': settings.TRADING_PAIRS,
          'default_leverage': settings.DEFAULT_LEVERAGE,
          'analysis_interval': settings.ANALYSIS_INTERVAL,
          'candle_limit': settings.CANDLE_LIMIT,
          'enable_orderbook_strategies': self.enable_orderbook_strategies,
          'enable_adaptive_consensus': self.enable_adaptive_consensus,
          'enable_mtf_analysis': self.enable_mtf_analysis,
          'enable_ml_validation': self.enable_ml_validation,
          'paper_trading': self.enable_paper_trading,
          'mtf_timeframes': settings.MTF_ACTIVE_TIMEFRAMES if self.enable_mtf_analysis else None,
          'mtf_synthesis_mode': settings.MTF_SYNTHESIS_MODE if self.enable_mtf_analysis else None,
          'integrated_analysis_mode': settings.INTEGRATED_ANALYSIS_MODE,
          'timestamp': datetime.now().isoformat()
        }


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
    === ФИНАЛЬНАЯ ОПТИМАЛЬНАЯ РЕАЛИЗАЦИЯ ===

    Главный цикл анализа рынка с полной интеграцией всех модулей системы.

    АРХИТЕКТУРА ИНТЕГРАЦИИ:
    ═══════════════════════════════════════════════════════════════════════

    Фаза 1: OrderBook-Aware Strategies
    ├── ExtendedStrategyManager
    ├── CANDLE strategies (momentum, sar_wave, supertrend, volume_profile)
    ├── ORDERBOOK strategies (imbalance, volume_flow, liquidity_zone)
    └── HYBRID strategies (smart_money)

    Фаза 2: Adaptive Consensus Management
    ├── StrategyPerformanceTracker - мониторинг эффективности
    ├── MarketRegimeDetector - определение рыночного режима
    ├── WeightOptimizer - динамическая оптимизация весов
    └── Continuous learning через signal outcomes

    Фаза 3: Multi-Timeframe Analysis
    ├── TimeframeCoordinator - управление свечами на 4+ TF
    ├── TimeframeAnalyzer - анализ каждого TF независимо
    ├── TimeframeAligner - проверка alignment и confluence
    └── TimeframeSignalSynthesizer - синтез финального сигнала

    Фаза 4: Integrated Analysis Engine
    ├── Объединение всех фаз
    ├── 4 режима: SINGLE_TF_ONLY, MTF_ONLY, HYBRID, ADAPTIVE
    ├── Intelligent fallback механизмы
    └── Comprehensive quality control

    ═══════════════════════════════════════════════════════════════════════

    WORKFLOW (Per Symbol):
    ┌────────────────────────────────────────────────────────────────────┐
    │ 1. Market Data Collection                                          │
    │    ├─ OrderBook Snapshot                                           │
    │    ├─ Candles (Single TF + MTF if enabled)                        │
    │    ├─ OrderBook Metrics                                            │
    │    └─ Market Metrics                                               │
    ├────────────────────────────────────────────────────────────────────┤
    │ 2. Manipulation Detection (если включено)                          │
    │    ├─ Spoofing Detector                                            │
    │    └─ Layering Detector                                            │
    ├────────────────────────────────────────────────────────────────────┤
    │ 3. S/R Levels Detection & Update (если включено)                   │
    ├────────────────────────────────────────────────────────────────────┤
    │ 4. ML Feature Extraction                                           │
    │    └─ 110+ признаков из всех источников                            │
    ├────────────────────────────────────────────────────────────────────┤
    │ 5. 🎯 INTEGRATED ANALYSIS (ЯДРО СИСТЕМЫ)                          │
    │    └─ IntegratedEngine.analyze()                                   │
    │       ├─ Single-TF Analysis (Фаза 1 + Фаза 2)                     │
    │       │   ├─ ExtendedStrategyManager                               │
    │       │   └─ AdaptiveConsensusManager (если включен)               │
    │       ├─ MTF Analysis (Фаза 3, если включен)                      │
    │       │   └─ MultiTimeframeManager                                 │
    │       └─ Signal Synthesis (Фаза 4)                                 │
    │           ├─ Conflict Resolution                                   │
    │           ├─ Quality Scoring                                       │
    │           └─ Risk Assessment                                       │
    ├────────────────────────────────────────────────────────────────────┤
    │ 6. ML Validation финального сигнала (если включено)                │
    ├────────────────────────────────────────────────────────────────────┤
    │ 7. Quality & Risk Checks                                           │
    ├────────────────────────────────────────────────────────────────────┤
    │ 8. Signal Metadata Enrichment                                      │
    │    └─ S/R context, contributing strategies, timestamps             │
    ├────────────────────────────────────────────────────────────────────┤
    │ 9. Execution Submission                                            │
    │    └─ ExecutionManager.submit_signal()                             │
    ├────────────────────────────────────────────────────────────────────┤
    │ 10. Drift Monitoring (если включено)                               │
    ├────────────────────────────────────────────────────────────────────┤
    │ 11. ML Data Collection для обучения                                │
    ├────────────────────────────────────────────────────────────────────┤
    │ 12. Real-time Broadcasting к UI (если включено)                    │
    └────────────────────────────────────────────────────────────────────┘

    ERROR HANDLING:
    - Per-symbol error counter с автоматическим skip
    - Fallback механизмы на каждом уровне
    - Критические алерты при превышении лимитов
    - Graceful degradation (работа даже при отключенных модулях)

    PERFORMANCE:
    - Асинхронная обработка символов
    - Intelligent caching
    - Performance tracking и warning при превышении порогов
    - Периодическая статистика (каждые 100 циклов)

    ПРИМЕЧАНИЕ:
    - Функция спроектирована для работы даже при частично отключенных модулях
    - Feature flags позволяют гибко управлять компонентами
    - Все критические операции имеют try-catch обработку
    """
    from models.signal import TradingSignal, SignalType, SignalStrength, SignalSource
    from datetime import datetime
    import traceback

    # ========================================================================
    # БЛОК 1: ИНИЦИАЛИЗАЦИЯ И ПОДГОТОВКА
    # ========================================================================

    logger.info("=" * 80)
    logger.info("🚀 ANALYSIS LOOP ЗАПУЩЕН (ФИНАЛЬНАЯ РЕАЛИЗАЦИЯ)")
    logger.info("=" * 80)
    logger.info(f"📊 Режим анализа: {settings.INTEGRATED_ANALYSIS_MODE}")
    logger.info(f"⏱️ Интервал анализа: {settings.ANALYSIS_INTERVAL}с")
    logger.info(
      f"📈 Торговые пары: {len(self.symbols)} ({', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''})")

    # Проверка доступности компонентов
    has_strategy_manager = self.strategy_manager is not None
    has_adaptive_consensus = self.adaptive_consensus is not None
    has_mtf_manager = self.mtf_manager is not None
    has_integrated_engine = self.integrated_engine is not None
    has_ml_validator = self.ml_validator is not None
    has_ml_feature_pipeline = self.ml_feature_pipeline is not None
    has_ml_data_collector = self.ml_data_collector is not None
    has_sr_detector = self.sr_detector is not None
    has_spoofing_detector = hasattr(self, 'spoofing_detector') and self.spoofing_detector
    has_layering_detector = hasattr(self, 'layering_detector') and self.layering_detector
    has_drift_detector = hasattr(self, 'drift_detector') and self.drift_detector

    logger.info("📦 Статус компонентов:")
    logger.info(f"   ├─ Strategy Manager: {'✅' if has_strategy_manager else '❌'}")
    logger.info(f"   ├─ Adaptive Consensus: {'✅' if has_adaptive_consensus else '❌'}")
    logger.info(f"   ├─ MTF Manager: {'✅' if has_mtf_manager else '❌'}")
    logger.info(f"   ├─ Integrated Engine: {'✅' if has_integrated_engine else '❌'}")
    logger.info(f"   ├─ ML Validator: {'✅' if has_ml_validator else '❌'}")
    logger.info(f"   ├─ ML Feature Pipeline: {'✅' if has_ml_feature_pipeline else '❌'}")
    logger.info(f"   ├─ ML Data Collector: {'✅' if has_ml_data_collector else '❌'}")
    logger.info(f"   ├─ S/R Detector: {'✅' if has_sr_detector else '❌'}")
    logger.info(f"   ├─ Spoofing Detector: {'✅' if has_spoofing_detector else '❌'}")
    logger.info(f"   ├─ Layering Detector: {'✅' if has_layering_detector else '❌'}")
    logger.info(f"   └─ Drift Detector: {'✅' if has_drift_detector else '❌'}")
    logger.info("=" * 80)

    # КРИТИЧЕСКАЯ ПРОВЕРКА: IntegratedEngine обязателен
    if not has_integrated_engine:
      logger.critical(
        "🚨 КРИТИЧЕСКАЯ ОШИБКА: IntegratedEngine не инициализирован! "
        "Analysis loop не может работать без него."
      )
      if settings.ENABLE_CRITICAL_ALERTS:
        await self._send_critical_alert(
          "IntegratedEngine отсутствует",
          "Analysis loop остановлен из-за отсутствия критического компонента"
        )
      return

    # Инициализация счетчиков и статистики
    error_count = {}  # Счетчик ошибок по символам
    max_consecutive_errors = 5  # Максимум последовательных ошибок перед skip
    cycle_number = 0

    # Инициализация статистики (если еще не инициализирована)
    if not hasattr(self, 'stats') or not self.stats:
      self.stats = {
        'analysis_cycles': 0,
        'signals_generated': 0,
        'signals_executed': 0,
        'orders_placed': 0,
        'positions_opened': 0,
        'positions_closed': 0,
        'total_pnl': 0.0,
        'consensus_achieved': 0,
        'consensus_failed': 0,
        'adaptive_weight_updates': 0,
        'mtf_signals': 0,
        'ml_validations': 0,
        'ml_data_collected': 0,
        'manipulations_detected': 0,
        'drift_detections': 0,
        'warnings': 0,
        'errors': 0
      }

    logger.info("✅ Analysis Loop инициализирован и готов к работе")

    # ========================================================================
    # БЛОК 2: ГЛАВНЫЙ ЦИКЛ АНАЛИЗА
    # ========================================================================

    while self.status == BotStatus.RUNNING:
      cycle_start = time.time()
      cycle_number += 1

      try:
        # async with self.analysis_lock:

        # Ждем пока все WebSocket соединения установятся
        if not self.websocket_manager.is_all_connected():
          await asyncio.sleep(1)
          continue

        # Анализируем каждую пару
        for symbol in self.symbols:
          symbol_start = time.time()

          # Инициализация error counter для символа
          if symbol not in error_count:
            error_count[symbol] = 0

          # Пропуск символа если слишком много ошибок подряд
          if error_count[symbol] >= max_consecutive_errors:
            if cycle_number % 10 == 0:  # Лог каждые 10 циклов
              logger.warning(
                f"⚠️ [{symbol}] Пропуск анализа: {error_count[symbol]} "
                f"последовательных ошибок (лимит: {max_consecutive_errors})"
              )
            continue

          try:
            # ============================================================
            # ШАГ 1: ПОЛУЧЕНИЕ MARKET DATA
            # ============================================================

            ob_manager = self.orderbook_managers[symbol]
            candle_manager = self.candle_managers[symbol]

            # Пропускаем если нет данных
            if not ob_manager.snapshot_received:
              logger.debug(f"[{symbol}] OrderBook Manager не найден, пропускаем")
              continue

            # Получаем снимок стакана
            orderbook_snapshot  = ob_manager.get_snapshot()
            if not orderbook_snapshot :
              logger.debug(f"[{symbol}] OrderBook не готов или невалиден, пропускаем")
              continue

            # Получаем свечи
            candles = candle_manager.get_candles()
            if not candles or len(candles) < 50:
              logger.debug(
                f"[{symbol}] Недостаточно свечей: "
                f"{len(candles)}/{settings.MIN_CANDLES_FOR_ANALYSIS}"
              )
              continue

            current_price = orderbook_snapshot.mid_price
            if not current_price:
              continue

            # 1.3 OrderBook Metrics
            # orderbook_metrics = self.market_analyzer.analyze_symbol(symbol, ob_manager)
            #
            # # 1.4 Market Metrics
            # market_metrics = self.market_analyzer.analyze_symbol(
            #   symbol=symbol,
            #   candles=candles,
            #   orderbook=orderbook_snapshot
            # )

            orderbook_metrics = self.orderbook_analyzer.analyze(orderbook_snapshot)

            # 1.4 Market Metrics
            market_metrics = self.market_analyzer.analyze_symbol(
              symbol,
              ob_manager
            )

            logger.debug(
              f"[{symbol}] Market Data: "
              f"price={current_price:.2f}, "
              f"candles={len(candles)}, "
              f"spread={orderbook_metrics.spread_bps:.2f}bps, "
              f"imbalance={orderbook_metrics.imbalance:.3f}, "
              f"volatility={market_metrics.volatility:.4f}"
            )
            # ==================== BROADCAST ORDERBOOK (КРИТИЧНО ДЛЯ ФРОНТЕНДА) ====================
            # try:
            #   from api.websocket import broadcast_orderbook_update
            #   await broadcast_orderbook_update(symbol, orderbook_snapshot.to_dict())
            # except Exception as e:
            #   logger.error(f"{symbol} | Ошибка broadcast orderbook: {e}")

            # ============================================================
            # ШАГ 2: ДЕТЕКЦИЯ МАНИПУЛЯЦИЙ (опционально)
            # ============================================================

            manipulation_detected = False
            manipulation_types = []

            # 2.1 Spoofing Detection
            if has_spoofing_detector:
              try:
                self.spoofing_detector.update(orderbook_snapshot)
                has_spoofing = self.spoofing_detector.is_spoofing_active(
                  symbol,
                  time_window_seconds=60
                )

                if has_spoofing:
                  manipulation_detected = True
                  manipulation_types.append("spoofing")

              except Exception as e:
                logger.error(f"[{symbol}] Ошибка Spoofing Detector: {e}")

            # 2.2 Layering Detection
            if has_layering_detector:
              try:
                self.layering_detector.update(orderbook_snapshot)
                has_layering = self.layering_detector.is_layering_active(
                  symbol,
                  time_window_seconds=60
                )

                if has_layering:
                  manipulation_detected = True
                  manipulation_types.append("layering")

              except Exception as e:
                logger.error(f"[{symbol}] Ошибка Layering Detector: {e}")

            # Блокировка торговли при манипуляциях
            if manipulation_detected:
              logger.warning(
                f"⚠️ [{symbol}] МАНИПУЛЯЦИИ ОБНАРУЖЕНЫ: "
                f"{', '.join(manipulation_types).upper()} - "
                f"ТОРГОВЛЯ ЗАБЛОКИРОВАНА"
              )
              self.stats['manipulations_detected'] += 1

              # Продолжаем извлечение признаков для data collection,
              # но НЕ генерируем торговые сигналы
              # (skip будет после извлечения признаков)

            # ============================================================
            # ШАГ 3: S/R LEVELS DETECTION & UPDATE (опционально)
            # ============================================================

            sr_levels = None
            if has_sr_detector:
              try:
                # Обновляем S/R детектор свежими свечами
                self.sr_detector.update_candles(symbol, candles)

                # Детектируем уровни
                sr_levels = self.sr_detector.detect_levels(symbol)

                if sr_levels:
                  logger.debug(
                    f"[{symbol}] S/R Levels: "
                    f"{len(sr_levels.get('support', []))} supports, "
                    f"{len(sr_levels.get('resistance', []))} resistances"
                  )

              except Exception as e:
                logger.error(f"[{symbol}] Ошибка S/R Detector: {e}")

            # ==================== 4. ТРАДИЦИОННЫЙ АНАЛИЗ ====================
            # ПРАВИЛЬНО: передаём OrderBookManager, НЕ OrderBookSnapshot
            # metrics = self.market_analyzer.analyze_symbol(symbol, ob_manager)

            # ==================== BROADCAST METRICS (КРИТИЧНО ДЛЯ ФРОНТЕНДА) ====================
            # try:
            #   from api.websocket import broadcast_metrics_update
            #   await broadcast_metrics_update(symbol, metrics.to_dict())
            # except Exception as e:
            #   logger.error(f"{symbol} | Ошибка broadcast metrics: {e}")

            # ============================================================
            # ШАГ 4: ML FEATURE EXTRACTION
            # ============================================================

            feature_vector = None
            ml_prediction = None

            if has_ml_feature_pipeline:
              try:
                # Извлекаем признаки из всех доступных источников
                feature_vector = await self.ml_feature_pipeline.extract_features(
                  symbol=symbol,
                  candles=candles,
                  orderbook_snapshot=orderbook_snapshot,
                  orderbook_metrics=orderbook_metrics,
                  market_metrics=market_metrics,
                  sr_levels=sr_levels  # Может быть None
                )

                if feature_vector:
                  logger.debug(
                    f"[{symbol}] Извлечено {feature_vector.feature_count} ML признаков, "
                    f"качество: {feature_vector.quality_score:.3f}"
                  )

                  # Опционально: получаем ML prediction (если нужно для обогащения)
                  # НЕ используем для блокировки торговли, только для метаданных
                  if self.ml_validator and not manipulation_detected:
                    try:
                      ml_prediction = await self.ml_validator.predict(
                        feature_vector=feature_vector
                      )

                      if ml_prediction:
                        logger.debug(
                          f"[{symbol}] ML Prediction: "
                          f"direction={ml_prediction.get('prediction')}, "
                          f"confidence={ml_prediction.get('confidence', 0):.3f}"
                        )
                    except Exception as e:
                      logger.error(f"[{symbol}] Ошибка ML Prediction: {e}")
                else:
                  logger.warning(f"[{symbol}] Feature extraction вернул None")

              except Exception as e:
                logger.error(f"[{symbol}] Ошибка ML Feature Extraction: {e}")
                logger.debug(traceback.format_exc())

            # Блокировка торговли если были манипуляции
            # (но продолжаем для data collection)
            if manipulation_detected:
              # Переходим к ML Data Collection (ШАГ 11)
              # НЕ генерируем сигнал
              logger.debug(f"[{symbol}] Пропускаем analysis из-за манипуляций")
              # Jump to ML Data Collection...
              # (код ниже будет пропущен через continue в конце этого блока)

            # ============================================================
            # ШАГ 5: 🎯 INTEGRATED ANALYSIS (ЯДРО СИСТЕМЫ)
            # ============================================================

            integrated_signal = None

            if not manipulation_detected:  # Анализируем только если нет манипуляций
              try:
                logger.debug(f"[{symbol}] Запуск IntegratedEngine.analyze()...")

                # Вызываем IntegratedEngine для полного анализа
                integrated_signal = await self.integrated_engine.analyze(
                  symbol=symbol,
                  candles=candles,
                  current_price=current_price,
                  orderbook=orderbook_snapshot,
                  metrics=orderbook_metrics
                )

                if integrated_signal:
                  # ========================================================
                  # ОБРАБОТКА INTEGRATED SIGNAL
                  # ========================================================

                  logger.info(
                    f"🎯 [{symbol}] IntegratedSignal получен: "
                    f"type={integrated_signal.final_signal.signal_type.value}, "
                    f"mode={integrated_signal.source_analysis_mode.value}, "
                    f"quality={integrated_signal.combined_quality_score:.3f}, "
                    f"confidence={integrated_signal.combined_confidence:.3f}"
                  )

                  # Детальное логирование (если включено)
                  if settings.VERBOSE_SIGNAL_LOGGING:
                    self._log_integrated_signal(symbol, integrated_signal)

                  # Извлекаем финальный сигнал
                  final_signal = integrated_signal.final_signal

                  # ========================================================
                  # ШАГ 6: ENRICHMENT SIGNAL METADATA
                  # ========================================================

                  # Инициализация metadata если нужно
                  if not final_signal.metadata:
                    final_signal.metadata = {}

                  # Добавляем integrated analysis метаданные
                  final_signal.metadata.update({
                    # Integrated Analysis Info
                    'integrated_analysis': True,
                    'analysis_mode': integrated_signal.source_analysis_mode.value,
                    'combined_quality': integrated_signal.combined_quality_score,
                    'combined_confidence': integrated_signal.combined_confidence,

                    # Source tracking
                    'single_tf_used': integrated_signal.used_single_tf,
                    'mtf_used': integrated_signal.used_mtf,

                    # Risk parameters
                    'position_multiplier': integrated_signal.recommended_position_multiplier,
                    'risk_level': integrated_signal.risk_level,

                    # Timestamps
                    'signal_timestamp': int(time.time() * 1000),
                    'analysis_timestamp': integrated_signal.analysis_timestamp,
                    'analysis_duration_ms': integrated_signal.analysis_duration_ms,
                    'cycle_number': cycle_number,

                    # Market context
                    'current_price': current_price,
                    'orderbook_imbalance': orderbook_metrics.imbalance,
                    'spread_bps': orderbook_metrics.spread_bps,
                    'market_volatility': market_metrics.volatility if market_metrics else None
                  })

                  # Single-TF Consensus Info
                  if integrated_signal.single_tf_consensus:
                    consensus = integrated_signal.single_tf_consensus
                    final_signal.metadata['single_tf_consensus'] = {
                      'mode': consensus.consensus_mode,
                      'confidence': consensus.consensus_confidence,
                      'agreement_count': consensus.agreement_count,
                      'disagreement_count': consensus.disagreement_count
                    }

                    # Contributing strategies для Performance Tracker
                    contributing_strategies = consensus.contributing_strategies
                    final_signal.metadata['contributing_strategies'] = contributing_strategies

                  # MTF Signal Info
                  if integrated_signal.mtf_signal:
                    mtf = integrated_signal.mtf_signal
                    final_signal.metadata['mtf_signal'] = {
                      'quality': mtf.signal_quality,
                      'risk_level': mtf.risk_level,
                      'alignment_score': mtf.alignment_score,
                      'confluence_detected': mtf.confluence_detected,
                      'divergence_type': mtf.divergence_type,
                      'recommended_position_multiplier': mtf.recommended_position_size_multiplier
                    }

                    if mtf.warnings:
                      final_signal.metadata['mtf_warnings'] = mtf.warnings

                  # Adaptive weights (если доступно)
                  if integrated_signal.adaptive_weights:
                    final_signal.metadata['adaptive_weights'] = integrated_signal.adaptive_weights

                  # Market regime (если доступно)
                  if integrated_signal.market_regime:
                    final_signal.metadata['market_regime'] = integrated_signal.market_regime

                  # ML Prediction (если было)
                  if ml_prediction:
                    final_signal.metadata['ml_prediction'] = {
                      'direction': ml_prediction.get('prediction'),
                      'confidence': ml_prediction.get('confidence')
                    }

                  # Warnings от engine
                  if integrated_signal.warnings:
                    final_signal.metadata['engine_warnings'] = integrated_signal.warnings

                  # ========================================================
                  # ШАГ 7: ML VALIDATION ФИНАЛЬНОГО СИГНАЛА
                  # ========================================================

                  ml_should_trade = True  # По умолчанию разрешаем
                  ml_validation_confidence = None

                  if has_ml_validator and feature_vector:
                    try:
                      logger.debug(f"[{symbol}] Запуск ML Validation...")

                      # ML Validator проверяет финальный сигнал
                      validation_result = await self.ml_validator.validate_signal(
                        symbol=symbol,
                        signal=final_signal,
                        features=feature_vector
                      )

                      ml_should_trade = validation_result.should_trade
                      ml_validation_confidence = validation_result.ml_confidence

                      # Добавляем ML validation метаданные
                      final_signal.metadata.update({
                        'ml_validated': True,
                        'ml_should_trade': ml_should_trade,
                        'ml_validation_confidence': ml_validation_confidence,
                        'ml_validation_reason': validation_result.reason if not ml_should_trade else None
                      })

                      # Логирование результата
                      if ml_should_trade:
                        logger.info(
                          f"✅ [{symbol}] ML Validation: APPROVED "
                          f"(confidence={ml_validation_confidence:.3f})"
                        )
                      else:
                        logger.warning(
                          f"❌ [{symbol}] ML Validation: REJECTED "
                          f"(reason={validation_result.reason})"
                        )

                      self.stats['ml_validations'] += 1

                      # Отклоняем сигнал если ML не одобрил
                      if not ml_should_trade:
                        logger.info(f"⛔ [{symbol}] Сигнал отклонен ML Validator")
                        integrated_signal = None  # Отменяем сигнал
                        continue  # Следующий символ

                    except Exception as e:
                      logger.error(f"[{symbol}] Ошибка ML Validation: {e}")
                      logger.debug(traceback.format_exc())
                      # Продолжаем без ML validation

                  # ========================================================
                  # ШАГ 8: QUALITY & RISK CHECKS
                  # ========================================================

                  # 8.1 Проверка минимального качества
                  if integrated_signal.combined_quality_score < settings.MIN_COMBINED_QUALITY:
                    logger.info(
                      f"⚠️ [{symbol}] Низкое качество сигнала: "
                      f"{integrated_signal.combined_quality_score:.3f} < "
                      f"{settings.MIN_COMBINED_QUALITY}, пропускаем"
                    )
                    self.stats['warnings'] += 1
                    continue  # Следующий символ

                  # 8.2 Проверка confidence
                  if integrated_signal.combined_confidence < settings.MIN_SIGNAL_CONFIDENCE:
                    logger.info(
                      f"⚠️ [{symbol}] Низкая уверенность сигнала: "
                      f"{integrated_signal.combined_confidence:.3f} < "
                      f"{settings.MIN_SIGNAL_CONFIDENCE}, пропускаем"
                    )
                    self.stats['warnings'] += 1
                    continue  # Следующий символ

                  # 8.3 Проверка EXTREME риска
                  if integrated_signal.risk_level == "EXTREME":
                    logger.warning(
                      f"🚨 [{symbol}] EXTREME RISK детектирован, пропускаем"
                    )
                    self.stats['warnings'] += 1
                    continue  # Следующий символ

                  # ========================================================
                  # ШАГ 9: S/R CONTEXT ENRICHMENT (опционально)
                  # ========================================================

                  if has_sr_detector and sr_levels:
                    try:
                      # Получаем ближайшие S/R уровни
                      nearest_levels = self.sr_detector.get_nearest_levels(
                        symbol=symbol,
                        current_price=current_price,
                        max_distance_pct=0.02  # 2% от цены
                      )

                      sr_context = []

                      # Support
                      if nearest_levels.get("support"):
                        support = nearest_levels["support"]
                        sr_context.append(
                          f"Support: ${support.price:.2f} "
                          f"(strength={support.strength:.2f}, "
                          f"distance={abs(current_price - support.price) / current_price * 100:.2f}%)"
                        )

                      # Resistance
                      if nearest_levels.get("resistance"):
                        resistance = nearest_levels["resistance"]
                        sr_context.append(
                          f"Resistance: ${resistance.price:.2f} "
                          f"(strength={resistance.strength:.2f}, "
                          f"distance={abs(resistance.price - current_price) / current_price * 100:.2f}%)"
                        )

                      if sr_context:
                        final_signal.metadata['sr_context'] = sr_context
                        logger.debug(
                          f"[{symbol}] S/R Context: {' | '.join(sr_context)}"
                        )

                    except Exception as e:
                      logger.error(f"[{symbol}] Ошибка S/R Context: {e}")

                  # ========================================================
                  # ШАГ 10: EXECUTION SUBMISSION
                  # ========================================================

                  try:
                    logger.info(
                      f"📤 [{symbol}] Отправка сигнала на исполнение: "
                      f"{final_signal.signal_type.value} @ {final_signal.entry_price:.2f}"
                    )

                    # Отправляем сигнал в ExecutionManager
                    submission_result = await self.execution_manager.submit_signal(
                      signal=final_signal
                    )

                    if submission_result.success:
                      logger.info(
                        f"✅ [{symbol}] Сигнал принят ExecutionManager: "
                        f"order_id={submission_result.order_id or 'pending'}"
                      )
                      self.stats['signals_executed'] += 1
                    else:
                      logger.warning(
                        f"⚠️ [{symbol}] Сигнал отклонен ExecutionManager: "
                        f"{submission_result.reason}"
                      )
                      self.stats['warnings'] += 1

                  except Exception as e:
                    logger.error(f"❌ [{symbol}] Ошибка submission: {e}")
                    logger.debug(traceback.format_exc())

                  # Обновление статистики
                  self.stats['signals_generated'] += 1

                  if integrated_signal.used_mtf:
                    self.stats['mtf_signals'] += 1

                else:
                  # Сигнал не сгенерирован
                  logger.debug(
                    f"[{symbol}] IntegratedEngine не вернул сигнал "
                    f"(консенсус не достигнут или низкое качество)"
                  )

              except Exception as e:
                logger.error(
                  f"❌ [{symbol}] Ошибка IntegratedEngine.analyze(): {e}"
                )
                logger.error(traceback.format_exc())
                error_count[symbol] += 1
                continue  # Следующий символ

            # ============================================================
            # ШАГ 11: DRIFT MONITORING (опционально)
            # ============================================================

            if has_drift_detector and feature_vector:
              try:
                drift_detected = self.drift_detector.detect_drift(
                  feature_vector=feature_vector
                )

                if drift_detected:
                  logger.warning(
                    f"🔔 [{symbol}] Model Drift обнаружен! "
                    f"Требуется переобучение модели."
                  )
                  self.stats['drift_detections'] += 1

                  # Опционально: отправить алерт
                  if settings.ENABLE_DRIFT_ALERTS:
                    await self._send_drift_alert(symbol)

              except Exception as e:
                logger.error(f"[{symbol}] Ошибка Drift Detection: {e}")

            # ============================================================
            # ШАГ 12: ML DATA COLLECTION (для обучения)
            # ============================================================

            if has_ml_data_collector and feature_vector:
              try:
                # Проверяем нужно ли собирать данные
                if self.ml_data_collector.should_collect():
                  # Подготовка sample
                  sample_data = {
                    'symbol': symbol,
                    'timestamp': int(time.time() * 1000),
                    'features': feature_vector,
                    'price': current_price,
                    'orderbook_snapshot': {
                      'best_bid': orderbook_snapshot.best_bid_price,
                      'best_ask': orderbook_snapshot.best_ask_price,
                      'mid_price': orderbook_snapshot.mid_price,
                      'spread': orderbook_snapshot.spread,
                      'imbalance': orderbook_metrics.imbalance
                    },
                    'market_metrics': {
                      'volatility': market_metrics.volatility if market_metrics else None,
                      'volume': market_metrics.volume if market_metrics else None,
                      'momentum': market_metrics.momentum if market_metrics else None
                    }
                  }

                  # Если был сгенерирован сигнал - добавляем его
                  if integrated_signal:
                    sample_data['signal'] = {
                      'type': integrated_signal.final_signal.signal_type.value,
                      'confidence': integrated_signal.combined_confidence,
                      'quality': integrated_signal.combined_quality_score,
                      'entry_price': integrated_signal.final_signal.entry_price,
                      'stop_loss': integrated_signal.final_signal.stop_loss,
                      'take_profit': integrated_signal.final_signal.take_profit,
                      'source_mode': integrated_signal.source_analysis_mode.value
                    }

                  # Сохранение sample
                  await self.ml_data_collector.collect_sample(sample_data)

                  self.stats['ml_data_collected'] += 1
                  logger.debug(f"[{symbol}] ML Data sample собран")

              except Exception as e:
                logger.error(f"[{symbol}] Ошибка ML Data Collection: {e}")

            # ============================================================
            # ШАГ 13: REAL-TIME BROADCASTING (опционально)
            # ============================================================

            try:
              # Broadcast OrderBook Update
              from api.websocket import broadcast_orderbook_update
              await broadcast_orderbook_update(
                symbol=symbol,
                orderbook=orderbook_snapshot.to_dict()
              )

              # Broadcast Metrics Update
              from api.websocket import broadcast_metrics_update
              await broadcast_metrics_update(
                symbol=symbol,
                metrics=market_metrics.to_dict()
              )

              # Broadcast Signal (если был)
              if integrated_signal:
                from api.websocket import broadcast_signal_update
                await broadcast_signal_update(
                  symbol=symbol,
                  signal=integrated_signal.final_signal.to_dict()
                )

            except Exception as e:
              # Broadcasting errors не критичны
              logger.debug(f"[{symbol}] Ошибка broadcasting: {e}")

            # ============================================================
            # УСПЕШНОЕ ЗАВЕРШЕНИЕ АНАЛИЗА СИМВОЛА
            # ============================================================

            # Сброс error counter при успехе
            error_count[symbol] = 0

            # Время выполнения для символа
            symbol_elapsed = time.time() - symbol_start

            if symbol_elapsed > settings.ANALYSIS_WARNING_THRESHOLD:
              logger.warning(
                f"⏱️ [{symbol}] Анализ занял {symbol_elapsed:.2f}с "
                f"(> {settings.ANALYSIS_WARNING_THRESHOLD}с)"
              )
            else:
              logger.debug(
                f"[{symbol}] Анализ завершен за {symbol_elapsed:.2f}с"
              )

          except Exception as e:
            # Обработка ошибки для конкретного символа
            error_count[symbol] += 1

            logger.error(
              f"❌ [{symbol}] Ошибка в analysis loop "
              f"(#{error_count[symbol]}/{max_consecutive_errors}): {e}"
            )
            logger.debug(traceback.format_exc())

            self.stats['errors'] += 1

            # Проверка превышения лимита ошибок
            if error_count[symbol] >= max_consecutive_errors:
              logger.critical(
                f"🚨 [{symbol}] Достигнут лимит последовательных ошибок "
                f"({max_consecutive_errors}), символ будет пропущен до рестарта"
              )

              # Отправка критического алерта
              if settings.ENABLE_CRITICAL_ALERTS:
                await self._send_critical_alert(
                  f"[{symbol}] Множественные ошибки в analysis loop",
                  f"Символ пропущен после {max_consecutive_errors} ошибок подряд"
                )

            continue  # Следующий символ

            # Конец цикла по символам

            # ================================================================
            # ОБНОВЛЕНИЕ СТАТИСТИКИ ЦИКЛА
            # ================================================================

            self.stats['analysis_cycles'] += 1

            # Периодическое логирование статистики (каждые 100 циклов)
            if cycle_number % 100 == 0:
              self._log_analysis_statistics()

            # Расчет времени выполнения цикла
          cycle_elapsed = time.time() - cycle_start

          # Warning если цикл занял слишком много времени
          if cycle_elapsed > settings.ANALYSIS_INTERVAL:
            logger.warning(
              f"⏱️ Цикл анализа #{cycle_number} занял {cycle_elapsed:.2f}с "
              f"(> интервал {settings.ANALYSIS_INTERVAL}с)"
            )

          # Ожидание до следующего цикла
          sleep_time = max(0, settings.ANALYSIS_INTERVAL - cycle_elapsed)
          if sleep_time > 0:
            await asyncio.sleep(sleep_time)

      except asyncio.CancelledError:
        # Graceful shutdown
        logger.info("🛑 Analysis Loop получил CancelledError, завершаем...")
        break

      except Exception as e:
        # Критическая ошибка в главном цикле
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА в главном analysis loop: {e}")
        logger.error(traceback.format_exc())

        self.stats['errors'] += 1

        # Отправка критического алерта
        if settings.ENABLE_CRITICAL_ALERTS:
          await self._send_critical_alert(
            "Критическая ошибка в главном цикле",
            f"Error: {str(e)}"
          )

        # Небольшая задержка перед следующей попыткой
        await asyncio.sleep(5)

      # ========================================================================
      # ЗАВЕРШЕНИЕ LOOP
      # ========================================================================

      logger.warning("⚠️ Analysis Loop остановлен")
      logger.info("=" * 80)
      logger.info("📊 ФИНАЛЬНАЯ СТАТИСТИКА РАБОТЫ")
      logger.info("=" * 80)
      logger.info(f"   ├─ Циклов анализа: {self.stats['analysis_cycles']}")
      logger.info(f"   ├─ Сигналов сгенерировано: {self.stats['signals_generated']}")
      logger.info(f"   ├─ Сигналов выполнено: {self.stats['signals_executed']}")
      logger.info(f"   ├─ Ордеров размещено: {self.stats['orders_placed']}")
      logger.info(f"   ├─ Позиций открыто: {self.stats['positions_opened']}")
      logger.info(f"   ├─ Позиций закрыто: {self.stats['positions_closed']}")
      logger.info(f"   ├─ Общий PnL: {self.stats['total_pnl']:.2f} USDT")
      logger.info(f"   ├─ MTF сигналов: {self.stats['mtf_signals']}")
      logger.info(f"   ├─ ML валидаций: {self.stats['ml_validations']}")
      logger.info(f"   ├─ ML данных собрано: {self.stats['ml_data_collected']}")
      logger.info(f"   ├─ Манипуляций обнаружено: {self.stats['manipulations_detected']}")
      logger.info(f"   ├─ Drift детекций: {self.stats['drift_detections']}")
      logger.info(f"   ├─ Предупреждений: {self.stats['warnings']}")
      logger.info(f"   └─ Ошибок: {self.stats['errors']}")
      logger.info("=" * 80)


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

    def _log_integrated_signal(self, symbol: str, integrated_signal):
      """
      Детальное логирование интегрированного сигнала.

      Args:
          symbol: Торговая пара
          integrated_signal: IntegratedSignal объект
      """
      signal = integrated_signal.final_signal

      logger.info("=" * 80)
      logger.info(f"🎯 INTEGRATED SIGNAL: {symbol}")
      logger.info("=" * 80)

      # ===== ОСНОВНАЯ ИНФОРМАЦИЯ =====
      logger.info(f"📊 Тип сигнала: {signal.signal_type.value}")
      logger.info(f"💯 Combined Confidence: {integrated_signal.combined_confidence:.3f}")
      logger.info(f"⭐ Combined Quality: {integrated_signal.combined_quality_score:.3f}")
      logger.info(f"📈 Entry Price: ${signal.entry_price:.2f}")
      logger.info(f"🛡️ Stop Loss: ${signal.stop_loss:.2f} ({signal.stop_loss_pct:.2f}%)")
      logger.info(f"🎯 Take Profit: ${signal.take_profit:.2f} ({signal.take_profit_pct:.2f}%)")
      logger.info(f"💰 Position Multiplier: {integrated_signal.recommended_position_multiplier:.2f}x")
      logger.info(f"⚠️ Risk Level: {integrated_signal.risk_level}")

      # ===== ИСТОЧНИК АНАЛИЗА =====
      logger.info("-" * 80)
      logger.info("🔧 ANALYSIS SOURCE:")
      logger.info(f"   ├─ Analysis Mode: {integrated_signal.source_analysis_mode.value}")
      logger.info(f"   ├─ Single-TF: {'✅ USED' if integrated_signal.used_single_tf else '❌ NOT USED'}")
      logger.info(f"   └─ MTF: {'✅ USED' if integrated_signal.used_mtf else '❌ NOT USED'}")

      # ===== SINGLE-TF CONSENSUS =====
      if integrated_signal.single_tf_consensus:
        consensus = integrated_signal.single_tf_consensus
        logger.info("-" * 80)
        logger.info("🔸 SINGLE-TF CONSENSUS:")
        logger.info(f"   ├─ Consensus Mode: {consensus.consensus_mode}")
        logger.info(f"   ├─ Consensus Confidence: {consensus.consensus_confidence:.3f}")
        logger.info(f"   ├─ Agreement: {consensus.agreement_count} strategies")
        logger.info(f"   ├─ Disagreement: {consensus.disagreement_count} strategies")
        logger.info(f"   └─ Contributing Strategies:")
        for strategy in consensus.contributing_strategies:
          logger.info(f"       └─ {strategy}")

      # ===== MTF SIGNAL =====
      if integrated_signal.mtf_signal:
        mtf = integrated_signal.mtf_signal
        logger.info("-" * 80)
        logger.info("🔹 MTF SIGNAL:")
        logger.info(f"   ├─ Signal Quality: {mtf.signal_quality:.3f}")
        logger.info(f"   ├─ Risk Level: {mtf.risk_level}")
        logger.info(f"   ├─ Alignment Score: {mtf.alignment_score:.3f}")
        logger.info(f"   ├─ Confluence Detected: {'✅ YES' if mtf.confluence_detected else '❌ NO'}")
        logger.info(f"   ├─ Recommended Position Multiplier: {mtf.recommended_position_size_multiplier:.2f}x")

        if mtf.divergence_type:
          logger.info(f"   ├─ Divergence Type: {mtf.divergence_type}")

        if mtf.warnings:
          logger.info("   └─ MTF Warnings:")
          for warning in mtf.warnings:
            logger.info(f"       ⚠️ {warning}")

      # ===== ADAPTIVE WEIGHTS =====
      if integrated_signal.adaptive_weights:
        logger.info("-" * 80)
        logger.info("⚖️ ADAPTIVE WEIGHTS:")
        for strategy, weight in integrated_signal.adaptive_weights.items():
          logger.info(f"   ├─ {strategy}: {weight:.3f}")

      # ===== MARKET REGIME =====
      if integrated_signal.market_regime:
        logger.info("-" * 80)
        logger.info(f"📊 Market Regime: {integrated_signal.market_regime}")

      # ===== WARNINGS =====
      if integrated_signal.warnings:
        logger.info("-" * 80)
        logger.info("⚠️ WARNINGS:")
        for warning in integrated_signal.warnings:
          logger.info(f"   └─ {warning}")

      # ===== ANALYSIS PERFORMANCE =====
      logger.info("-" * 80)
      logger.info("⏱️ PERFORMANCE:")
      logger.info(f"   ├─ Analysis Duration: {integrated_signal.analysis_duration_ms:.2f}ms")
      logger.info(f"   └─ Analysis Timestamp: {integrated_signal.analysis_timestamp}")

      logger.info("=" * 80)

  def _log_analysis_statistics(self):
    """
    Периодическое логирование статистики работы analysis loop.
    """
    logger.info("=" * 80)
    logger.info("📊 ANALYSIS LOOP STATISTICS")
    logger.info("=" * 80)

    # ===== ОСНОВНЫЕ МЕТРИКИ =====
    logger.info("🔄 CYCLES & SIGNALS:")
    logger.info(f"   ├─ Analysis Cycles: {self.stats['analysis_cycles']}")
    logger.info(f"   ├─ Signals Generated: {self.stats['signals_generated']}")
    logger.info(f"   ├─ Signals Executed: {self.stats['signals_executed']}")
    logger.info(
      f"   └─ Execution Rate: {self.stats['signals_executed'] / max(self.stats['signals_generated'], 1) * 100:.1f}%")

    # ===== TRADING ACTIVITY =====
    logger.info("💰 TRADING ACTIVITY:")
    logger.info(f"   ├─ Orders Placed: {self.stats['orders_placed']}")
    logger.info(f"   ├─ Positions Opened: {self.stats['positions_opened']}")
    logger.info(f"   ├─ Positions Closed: {self.stats['positions_closed']}")
    logger.info(f"   └─ Total PnL: {self.stats['total_pnl']:.2f} USDT")

    # ===== ADAPTIVE CONSENSUS =====
    if self.adaptive_consensus:
      logger.info("🔄 ADAPTIVE CONSENSUS:")
      logger.info(f"   ├─ Consensus Achieved: {self.stats['consensus_achieved']}")
      logger.info(f"   ├─ Consensus Failed: {self.stats['consensus_failed']}")
      logger.info(f"   ├─ Weight Updates: {self.stats['adaptive_weight_updates']}")

      consensus_rate = self.stats['consensus_achieved'] / max(
        self.stats['consensus_achieved'] + self.stats['consensus_failed'], 1
      ) * 100
      logger.info(f"   └─ Consensus Rate: {consensus_rate:.1f}%")

    # ===== MTF ANALYSIS =====
    if self.mtf_manager:
      logger.info("⏱️ MULTI-TIMEFRAME:")
      logger.info(f"   ├─ MTF Signals: {self.stats['mtf_signals']}")
      mtf_rate = self.stats['mtf_signals'] / max(self.stats['signals_generated'], 1) * 100
      logger.info(f"   └─ MTF Signal Rate: {mtf_rate:.1f}%")

    # ===== ML COMPONENTS =====
    if self.ml_validator:
      logger.info("🤖 ML COMPONENTS:")
      logger.info(f"   ├─ ML Validations: {self.stats['ml_validations']}")
      logger.info(f"   ├─ ML Data Collected: {self.stats['ml_data_collected']}")
      logger.info(f"   ├─ Drift Detections: {self.stats['drift_detections']}")
      logger.info(f"   └─ Manipulations Detected: {self.stats['manipulations_detected']}")

    # ===== ERRORS & WARNINGS =====
    logger.info("⚠️ ISSUES:")
    logger.info(f"   ├─ Warnings: {self.stats['warnings']}")
    logger.info(f"   └─ Errors: {self.stats['errors']}")

    # ===== COMPONENT STATISTICS =====
    if self.integrated_engine:
      logger.info("-" * 80)
      logger.info("🎯 INTEGRATED ENGINE STATS:")
      engine_stats = self.integrated_engine.get_statistics()
      for key, value in engine_stats.items():
        logger.info(f"   ├─ {key}: {value}")

    if self.adaptive_consensus:
      logger.info("-" * 80)
      logger.info("🔄 ADAPTIVE CONSENSUS STATS:")
      adaptive_stats = self.adaptive_consensus.get_statistics()
      for key, value in adaptive_stats.items():
        logger.info(f"   ├─ {key}: {value}")

    if self.mtf_manager:
      logger.info("-" * 80)
      logger.info("⏱️ MTF MANAGER STATS:")
      mtf_stats = self.mtf_manager.get_statistics()
      for key, value in mtf_stats.items():
        logger.info(f"   ├─ {key}: {value}")

    logger.info("=" * 80)

  async def _send_critical_alert(self, title: str, message: str):
    """
    Отправка критического алерта.

    Args:
        title: Заголовок алерта
        message: Сообщение
    """
    try:
      logger.critical(f"🚨 CRITICAL ALERT: {title}")
      logger.critical(f"   Message: {message}")

      # Здесь можно добавить отправку в Telegram, Discord, Email и т.д.
      # Например:
      # if self.telegram_notifier:
      #     await self.telegram_notifier.send_critical_alert(title, message)

      # Или запись в специальную таблицу критических событий
      # if self.alert_repository:
      #     await self.alert_repository.create_critical_alert(
      #         title=title,
      #         message=message,
      #         timestamp=datetime.now()
      #     )

    except Exception as e:
      logger.error(f"Ошибка отправки критического алерта: {e}")

  async def _send_drift_alert(self, symbol: str):
    """
    Отправка алерта о model drift.

    Args:
        symbol: Торговая пара
    """
    try:
      logger.warning(f"🔔 DRIFT ALERT: {symbol}")
      logger.warning("   Model drift обнаружен, рекомендуется переобучение")

      # Здесь можно добавить отправку алерта
      # if self.telegram_notifier:
      #     await self.telegram_notifier.send_drift_alert(symbol)

    except Exception as e:
      logger.error(f"Ошибка отправки drift алерта: {e}")

  def _handle_symbol_error(self, symbol: str, error: Exception, error_count: dict):
    """
    Обработка ошибки для конкретного символа.

    Args:
        symbol: Торговая пара
        error: Exception
        error_count: Словарь счетчиков ошибок
    """
    error_count[symbol] = error_count.get(symbol, 0) + 1

    logger.error(
      f"❌ [{symbol}] Ошибка анализа "
      f"(#{error_count[symbol]}/{settings.MAX_CONSECUTIVE_ERRORS}): {error}"
    )
    logger.debug(traceback.format_exc())

    self.stats['errors'] += 1

    # Проверка превышения лимита
    if error_count[symbol] >= settings.MAX_CONSECUTIVE_ERRORS:
      logger.critical(
        f"🚨 [{symbol}] Достигнут лимит последовательных ошибок "
        f"({settings.MAX_CONSECUTIVE_ERRORS}), символ будет пропущен"
      )

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