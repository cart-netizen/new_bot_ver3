"""
Главный файл приложения.
Точка входа и контроллер торгового бота.
"""

import asyncio
import os
import signal
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
from infrastructure.resilience.recovery_service import recovery_service
from ml_engine.detection.layering_detector import LayeringConfig, LayeringDetector
from ml_engine.detection.spoofing_detector import SpoofingConfig, SpoofingDetector
from ml_engine.detection.sr_level_detector import SRLevelConfig, SRLevelDetector
from ml_engine.integration.ml_signal_validator import ValidationConfig, MLSignalValidator
from ml_engine.monitoring.drift_detector import DriftDetector
from models.signal import TradingSignal, SignalType, SignalStrength, SignalSource
from screener.screener_manager import ScreenerManager
from strategies.strategy_manager import StrategyManagerConfig, StrategyManager
from strategy.candle_manager import CandleManager
from strategy.orderbook_manager import OrderBookManager
from strategy.analyzer import MarketAnalyzer
from strategy.strategy_engine import StrategyEngine
from strategy.risk_manager import RiskManager
from execution.execution_manager import ExecutionManager
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
    """Инициализация контроллера."""
    self.status = BotStatus.STOPPED
    self.symbols = settings.get_trading_pairs_list()

    # Существующие компоненты
    self.websocket_manager: Optional[BybitWebSocketManager] = None
    self.orderbook_managers: Dict[str, OrderBookManager] = {}
    self.market_analyzer: Optional[MarketAnalyzer] = None
    self.strategy_engine: Optional[StrategyEngine] = None
    self.risk_manager: Optional[RiskManager] = None
    self.execution_manager: Optional[ExecutionManager] = None
    self.balance_tracker = balance_tracker

    # ===== НОВЫЕ ML КОМПОНЕНТЫ =====
    self.candle_managers: Dict[str, CandleManager] = {}
    self.ml_feature_pipeline: Optional[MultiSymbolFeaturePipeline] = None
    self.ml_data_collector: Optional[MLDataCollector] = None

    # Хранение последних признаков для каждого символа
    self.latest_features: Dict[str, FeatureVector] = {}

    # Задачи
    self.websocket_task: Optional[asyncio.Task] = None
    self.analysis_task: Optional[asyncio.Task] = None
    self.candle_update_task: Optional[asyncio.Task] = None  # НОВОЕ

    self.ml_stats_task: Optional[asyncio.Task] = None

    # ML Signal Validator
    # ml_validator_config = ValidationConfig(
    #   model_server_url=os.getenv('ML_SERVER_URL', 'http://localhost:8001'),
    #   min_ml_confidence=float(os.getenv('ML_MIN_CONFIDENCE', '0.6')),
    #   ml_weight=float(os.getenv('ML_WEIGHT', '0.6')),
    #   strategy_weight=float(os.getenv('STRATEGY_WEIGHT', '0.4'))
    # )
    # self.ml_validator = MLSignalValidator(ml_validator_config)

    ml_config = ValidationConfig(
      model_server_url=settings.ML_SERVER_URL,
      min_ml_confidence=settings.ML_MIN_CONFIDENCE,
      ml_weight=settings.ML_WEIGHT,
      strategy_weight=settings.STRATEGY_WEIGHT,
      health_check_enabled=True,
      health_check_interval=30,
    )

    self.ml_validator = MLSignalValidator(config=ml_config)

    # Drift Detector
    self.drift_detector = DriftDetector(
      window_size=10000,
      baseline_window_size=50000,
      drift_threshold=0.1
    )

    # ==================== DETECTION SYSTEMS ====================

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

    # ==================== STRATEGY MANAGER ====================

    strategy_manager_config = StrategyManagerConfig(
      consensus_mode=os.getenv('CONSENSUS_MODE', 'weighted'),
      min_strategies_for_signal=int(os.getenv('MIN_STRATEGIES', '2')),
      min_consensus_confidence=float(os.getenv('MIN_CONSENSUS_CONFIDENCE', '0.6'))
    )
    self.strategy_manager = StrategyManager(strategy_manager_config)

    # ===== SCREENER MANAGER (НОВОЕ) =====
    self.screener_manager: Optional[ScreenerManager] = None
    self.screener_broadcast_task: Optional[asyncio.Task] = None
    # self.screener_manager = ScreenerManager()

    self.dynamic_symbols_manager: Optional[DynamicSymbolsManager] = None
    self.symbols_refresh_task: Optional[asyncio.Task] = None

    self.running = False

    logger.info("Инициализирован контроллер бота с ML поддержкой")

  async def initialize(self):
    """Инициализация всех компонентов бота."""
    try:
      logger.info("=" * 80)
      logger.info("ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ БОТА (ML-ENHANCED)")
      logger.info("=" * 80)

      # Инициализируем REST клиент
      await rest_client.initialize()
      logger.info("✓ REST клиент инициализирован")

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

      # ===== ВАЖНО: НЕ создаем WebSocket Manager и ML Pipeline здесь! =====
      # self.symbols пока не определены
      # Эти компоненты будут созданы в start() после выбора пар
      # WebSocket Manager - зависит от символов
      # ML Feature Pipeline - зависит от символов

      # ===== ML DATA COLLECTOR =====
      self.ml_data_collector = MLDataCollector(
        storage_path="../data/ml_training",
        max_samples_per_file=10000
      )
      await self.ml_data_collector.initialize()
      logger.info("✓ ML Data Collector инициализирован")

      # Инициализируем анализатор рынка (пока без символов)
      self.market_analyzer = MarketAnalyzer()
      logger.info("✓ Анализатор рынка инициализирован")

      # Инициализируем стратегию
      self.strategy_engine = StrategyEngine()
      logger.info("✓ Торговая стратегия инициализирована")

      # # Инициализируем риск-менеджер
      # self.risk_manager = RiskManager(default_leverage=settings.DEFAULT_LEVERAGE)
      # logger.info("✓ Риск-менеджер инициализирован")

      # Инициализируем менеджер исполнения


      logger.info("=" * 80)
      logger.info("БАЗОВЫЕ КОМПОНЕНТЫ ИНИЦИАЛИЗИРОВАНЫ (БЕЗ WEBSOCKET)")
      logger.info("=" * 80)

    except Exception as e:
      logger.error(f"Ошибка инициализации бота: {e}")
      log_exception(logger, e, "Инициализация бота")
      raise

  async def start(self):
    """Запуск бота с правильной последовательностью инициализации."""
    if self.status == BotStatus.RUNNING:
      logger.warning("Бот уже запущен")
      return

    try:
      self.status = BotStatus.STARTING
      logger.info("=" * 80)
      logger.info("ЗАПУСК ТОРГОВОГО БОТА (ML-ENHANCED)")
      logger.info("=" * 80)

      # Инициализация риск-менеджера с реальным балансом
      await self._initialize_risk_manager()

      self.execution_manager = ExecutionManager(self.risk_manager)
      logger.info("✓ Менеджер исполнения инициализирован")

      # Запускаем менеджер исполнения
      await self.execution_manager.start()
      logger.info("✓ Менеджер исполнения запущен")

      # Запускаем трекер баланса
      await self.balance_tracker.start()
      logger.info("✓ Трекер баланса запущен")

      # Инициализация ML Validator
      await self.ml_validator.initialize()
      logger.info("✅ ML Signal Validator инициализирован")

      # ===== SCREENER MANAGER - Запускаем =====
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

        # ===== DYNAMIC SYMBOLS - Выбираем финальный список пар =====
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

      # ===== КРИТИЧЕСКИ ВАЖНО: СОЗДАЕМ ML Feature Pipeline ЗДЕСЬ =====
      logger.info("Создание ML Feature Pipeline...")
      self.ml_feature_pipeline = MultiSymbolFeaturePipeline(
        symbols=self.symbols,  # ← Правильные динамические символы!
        normalize=True,
        cache_enabled=True
      )
      logger.info(f"✓ ML Feature Pipeline создан для {len(self.symbols)} символов")

      # ===== Создаем менеджеры стакана для ФИНАЛЬНЫХ пар =====
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

      # ===== Добавляем символы в анализатор =====
      for symbol in self.symbols:
        self.market_analyzer.add_symbol(symbol)
      logger.info(f"✓ {len(self.symbols)} символов добавлено в анализатор")

      # ===== ТЕПЕРЬ создаем WebSocket Manager с ПРАВИЛЬНЫМИ символами =====
      logger.info("Создание WebSocket Manager...")
      logger.info(f"Символы для WebSocket: {self.symbols[:5]}..." if len(
        self.symbols) > 5 else f"Символы для WebSocket: {self.symbols}")

      self.websocket_manager = BybitWebSocketManager(
        symbols=self.symbols,  # ← Правильные динамические символы!
        on_message=self._handle_orderbook_message
      )
      logger.info("✓ WebSocket менеджер создан с правильными символами")

      # ===== Загружаем исторические свечи =====
      await self._load_historical_candles()
      logger.info("✓ Исторические свечи загружены")

      # ===== Запускаем WebSocket соединения =====
      self.websocket_task = asyncio.create_task(
        self.websocket_manager.start()
      )
      logger.info("✓ WebSocket соединения запущены")

      # ===== Запускаем остальные задачи =====
      self.candle_update_task = asyncio.create_task(
        self._candle_update_loop()
      )
      logger.info("✓ Цикл обновления свечей запущен")

      self.ml_stats_task = asyncio.create_task(
        self._ml_stats_loop()
      )

      self.analysis_task = asyncio.create_task(
        self._analysis_loop_ml_enhanced()
      )
      logger.info("✓ Цикл анализа (ML-Enhanced) запущен")

      asyncio.create_task(fsm_cleanup_task())
      logger.info("✓ FSM Cleanup Task запланирован")

      # ===== Запускаем задачу обновления списка пар =====
      if settings.DYNAMIC_SYMBOLS_ENABLED and self.dynamic_symbols_manager:
        logger.info("Запуск задачи обновления списка пар...")
        self.symbols_refresh_task = asyncio.create_task(
          self._symbols_refresh_loop()
        )
        logger.info("✓ Задача обновления списка пар запущена")

      # Уведомляем фронтенд
      from api.websocket import broadcast_bot_status
      await broadcast_bot_status("running", {
        "symbols": self.symbols,
        "ml_enabled": True,
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
                  consensus = self.strategy_manager.analyze_with_consensus(
                    symbol,
                    candles,
                    current_price
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
                      f"{safe_enum_value(signal.signal_type)}, "  # ИСПРАВЛЕНО: signal.signal_type.value
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
                      f"{signal.signal_type.value}, "
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
                # Передаём весь объект TradingSignal, а не только signal_type
                validation_result = await self.ml_validator.validate(
                  signal,
                  feature_vector
                )

                if not validation_result.validated:
                  logger.info(
                    f"❌ Сигнал отклонен ML Validator [{symbol}]: "
                    f"{validation_result.reason}"
                  )
                  signal = None  # Сбрасываем сигнал, но продолжаем обработку
                else:
                  # ===== ОБНОВЛЯЕМ СИГНАЛ С ML ВАЛИДАЦИЕЙ =====
                  # 1. Меняем source на ML_VALIDATED
                  signal.source = SignalSource.ML_VALIDATED

                  # 2. Обновляем confidence и strength
                  signal.confidence = validation_result.final_confidence

                  # 3. Пересчитываем strength на основе ML confidence
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
                  signal.metadata['ml_direction'] = validation_result.ml_direction
                  signal.metadata['ml_confidence'] = validation_result.ml_confidence

                  logger.info(
                    f"✅ Сигнал подтвержден ML Validator [{symbol}]: "
                    f"source=ML_VALIDATED, "
                    f"strength={signal.strength.value}, "
                    f"final_confidence={validation_result.final_confidence:.2f}"
                  )
              except Exception as e:
                logger.error(f"{symbol} | Ошибка ML Validator: {e}", exc_info=True)

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
                  f"strength={signal.strength.value}"
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
                  from api.websocket import broadcast_signal
                  await broadcast_signal(signal.to_dict())
                except Exception as e:
                  logger.error(f"{symbol} | Ошибка broadcast_signal: {e}")

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
                signal_type_value = signal.signal_type.value  # Получаем строку "BUY", "SELL", "HOLD"
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
                    "type": signal.signal_type.value,  # Получаем строковое значение enum
                    "confidence": signal.confidence,
                    "strength": signal.strength.value,  # Тоже enum
                  } if signal else None
                )
              except Exception as e:
                logger.error(f"{symbol} | Ошибка сбора ML данных: {e}")

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

      if self.balance_tracker:
        await self.balance_tracker.stop()
        logger.info("✓ Трекер баланса остановлен")

      if self.symbols_refresh_task:
        self.symbols_refresh_task.cancel()
        try:
          await self.symbols_refresh_task
        except asyncio.CancelledError:
          pass
        logger.info("✓ Symbols refresh task остановлен")

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

  def get_status(self) -> dict:
    """Получение статуса бота."""
    ws_status = {}
    if self.websocket_manager:
      ws_status = self.websocket_manager.get_connection_statuses()

    # ===== НОВОЕ: Добавляем ML статистику =====
    ml_status = {
      "features_extracted": len(self.latest_features),
      "data_collected_samples": (
          self.ml_data_collector.get_statistics()
          if self.ml_data_collector else {}
      )
    }

    return {
      "status": self.status.value,
      "symbols": self.symbols,
      "ml_enabled": True,  # НОВОЕ
      "ml_status": ml_status,  # НОВОЕ
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

  async def _initialize_risk_manager(self):
    """Инициализация Risk Manager."""
    # Создаём без баланса
    self.risk_manager = RiskManager(default_leverage=settings.DEFAULT_LEVERAGE)

    # Получаем реальный баланс
    try:
      balance_data = await rest_client.get_wallet_balance()
      real_balance = balance_tracker._calculate_total_balance(balance_data)

      # ИСПОЛЬЗУЕМ update_available_balance
      self.risk_manager.update_available_balance(real_balance)

      logger.info(f"✓ Risk Manager обновлён балансом: {real_balance:.2f} USDT")
    except Exception as e:
      logger.error(f"Ошибка получения баланса: {e}")


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
from api.routes import auth_router, bot_router, data_router, trading_router, monitoring_router, screener_router

app.include_router(auth_router)
app.include_router(bot_router)
app.include_router(data_router)
app.include_router(trading_router)
app.include_router(monitoring_router)
app.include_router(screener_router)

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