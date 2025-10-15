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
from core.logger import setup_logging, get_logger
from core.exceptions import log_exception, OrderBookSyncError, OrderBookError
from core.screener_processor import ScreenerProcessor
from core.ticker_websocket import ScreenerTickerManager
from core.trace_context import trace_operation
from database.connection import db_manager
from domain.services.fsm_registry import fsm_registry
from exchange.rest_client import rest_client
from exchange.websocket_manager import BybitWebSocketManager
from infrastructure.resilience.recovery_service import recovery_service
from models.signal import TradingSignal, SignalStrength, SignalSource
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

# ML FEATURE PIPELINE - НОВОЕ
from ml_engine.features import (

    FeatureVector
)
from ml_engine.data_collection import MLDataCollector  # НОВОЕ

# ==================== ML INFRASTRUCTURE ====================
from ml_engine.integration.ml_signal_validator import (
    MLSignalValidator, ValidationConfig
)
from ml_engine.monitoring.drift_detector import DriftDetector
from ml_engine.features import MultiSymbolFeaturePipeline

# ==================== DETECTION SYSTEMS ====================
from ml_engine.detection.spoofing_detector import (
    SpoofingDetector, SpoofingConfig
)
from ml_engine.detection.layering_detector import (
    LayeringDetector, LayeringConfig
)
from ml_engine.detection.sr_level_detector import (
    SRLevelDetector, SRLevelConfig
)

# ==================== ADVANCED STRATEGIES ====================
from strategies.strategy_manager import (
    StrategyManager, StrategyManagerConfig
)

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
    ml_validator_config = ValidationConfig(
      model_server_url=os.getenv('ML_SERVER_URL', 'http://localhost:8001'),
      min_ml_confidence=float(os.getenv('ML_MIN_CONFIDENCE', '0.6')),
      ml_weight=float(os.getenv('ML_WEIGHT', '0.6')),
      strategy_weight=float(os.getenv('STRATEGY_WEIGHT', '0.4'))
    )
    self.ml_validator = MLSignalValidator(ml_validator_config)

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

      # Создаем менеджеры стакана для каждой пары
      for symbol in self.symbols:
        self.orderbook_managers[symbol] = OrderBookManager(symbol)
      logger.info(f"✓ Создано {len(self.orderbook_managers)} менеджеров стакана")

      # ===== НОВОЕ: Создаем менеджеры свечей =====
      for symbol in self.symbols:
        self.candle_managers[symbol] = CandleManager(
            symbol=symbol,
            timeframe="1m",  # Основной таймфрейм
            max_candles=200  # Достаточно для индикаторов
        )
      logger.info(f"✓ Создано {len(self.candle_managers)} менеджеров свечей")

      # ===== НОВОЕ: Инициализируем ML Feature Pipeline =====
      self.ml_feature_pipeline = MultiSymbolFeaturePipeline(
          symbols=self.symbols,
          normalize=True,
          cache_enabled=True
      )
      logger.info("✓ ML Feature Pipeline инициализирован")

      # ===== НОВОЕ: Инициализируем ML Data Collector =====
      self.ml_data_collector = MLDataCollector(
          storage_path="../data/ml_training",
          max_samples_per_file=10000
      )
      await self.ml_data_collector.initialize()
      logger.info("✓ ML Data Collector инициализирован")

      # Инициализируем анализатор рынка
      self.market_analyzer = MarketAnalyzer()
      for symbol in self.symbols:
        self.market_analyzer.add_symbol(symbol)
      logger.info("✓ Анализатор рынка инициализирован")

      # Инициализируем стратегию
      self.strategy_engine = StrategyEngine()
      logger.info("✓ Торговая стратегия инициализирована")

      # Инициализируем риск-менеджер
      self.risk_manager = RiskManager()
      logger.info("✓ Риск-менеджер инициализирован")

      # Инициализируем менеджер исполнения
      self.execution_manager = ExecutionManager(self.risk_manager)
      logger.info("✓ Менеджер исполнения инициализирован")

      # Создаем WebSocket менеджер
      self.websocket_manager = BybitWebSocketManager(
        symbols=self.symbols,
        on_message=self._handle_orderbook_message
      )
      logger.info("✓ WebSocket менеджер создан")

      logger.info("=" * 80)
      logger.info("ВСЕ КОМПОНЕНТЫ УСПЕШНО ИНИЦИАЛИЗИРОВАНЫ (ML-READY)")
      logger.info("=" * 80)

    except Exception as e:
      logger.error(f"Ошибка инициализации бота: {e}")
      log_exception(logger, e, "Инициализация бота")
      raise

  async def start(self):
    """Запуск бота."""
    if self.status == BotStatus.RUNNING:
      logger.warning("Бот уже запущен")
      return

    try:
      self.status = BotStatus.STARTING
      logger.info("=" * 80)
      logger.info("ЗАПУСК ТОРГОВОГО БОТА (ML-ENHANCED)")
      logger.info("=" * 80)

      # Запускаем менеджер исполнения
      await self.execution_manager.start()
      logger.info("✓ Менеджер исполнения запущен")

      # Запускаем трекер баланса
      await self.balance_tracker.start()
      logger.info("✓ Трекер баланса запущен")

      # Инициализация ML Validator
      await self.ml_validator.initialize()
      logger.info("✅ ML Signal Validator инициализирован")

      # ===== НОВОЕ: Загружаем исторические свечи =====
      await self._load_historical_candles()
      logger.info("✓ Исторические свечи загружены")

      # Запускаем WebSocket соединения
      self.websocket_task = asyncio.create_task(
        self.websocket_manager.start()
      )
      logger.info("✓ WebSocket соединения запущены")

      # ===== НОВОЕ: Запускаем задачу обновления свечей =====
      self.candle_update_task = asyncio.create_task(
        self._candle_update_loop()
      )
      logger.info("✓ Цикл обновления свечей запущен")

      self.ml_stats_task = asyncio.create_task(
        self._ml_stats_loop()
      )

      # Запускаем задачу анализа (теперь с ML)
      self.analysis_task = asyncio.create_task(
        self._analysis_loop_ml_enhanced()
      )
      logger.info("✓ Цикл анализа (ML-Enhanced) запущен")

      # Запускаем background task для очистки FSM
      asyncio.create_task(fsm_cleanup_task())
      logger.info("✓ FSM Cleanup Task запланирован")

      self.status = BotStatus.RUNNING
      logger.info("=" * 80)
      logger.info("БОТ УСПЕШНО ЗАПУЩЕН (ML-READY)")
      logger.info("=" * 80)

      # Уведомляем фронтенд
      from api.websocket import broadcast_bot_status
      await broadcast_bot_status("running", {
        "symbols": self.symbols,
        "ml_enabled": True,
        "message": "Бот успешно запущен с ML поддержкой"
      })

    except Exception as e:
      self.status = BotStatus.ERROR
      logger.error(f"Ошибка запуска бота: {e}")
      log_exception(logger, e, "Запуск бота")
      raise

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
    Продвинутый цикл анализа с ML, детекторами и мульти-стратегией.

    Workflow:
    1. Получить данные (orderbook, candles)
    2. Обновить детекторы манипуляций
    3. Обновить S/R детектор
    4. Проверить активные манипуляции (блокируем торговлю)
    5. Извлечь ML признаки
    6. Запустить Strategy Manager (все стратегии + consensus)
    7. ML валидация consensus сигнала
    8. Проверить близость к S/R уровням
    9. Финальная фильтрация и исполнение
    10. Обновить drift detector
    """
    logger.info("🔄 Запущен продвинутый analysis loop")

    while self.running:
        try:
            for symbol in self.symbols:
                # ==================== 1. ПОЛУЧЕНИЕ ДАННЫХ ====================
                orderbook = self.orderbook_managers[symbol].get_snapshot()
                candles = self.candle_managers[symbol].get_candles()

                if not orderbook or len(candles) < 50:
                    continue

                current_price = orderbook.mid_price
                if not current_price:
                    continue

                # ==================== 2. ДЕТЕКТОРЫ МАНИПУЛЯЦИЙ ====================
                self.spoofing_detector.update(orderbook)
                self.layering_detector.update(orderbook)

                # ==================== 3. S/R ДЕТЕКТОР ====================
                self.sr_detector.update_candles(symbol, candles)
                sr_levels = self.sr_detector.detect_levels(symbol)

                # ==================== 4. ПРОВЕРКА МАНИПУЛЯЦИЙ ====================
                has_spoofing = self.spoofing_detector.is_spoofing_active(
                    symbol,
                    time_window_seconds=60
                )
                has_layering = self.layering_detector.is_layering_active(
                    symbol,
                    time_window_seconds=60
                )

                if has_spoofing or has_layering:
                    logger.warning(
                        f"⚠️  МАНИПУЛЯЦИИ [{symbol}]: "
                        f"spoofing={has_spoofing}, layering={has_layering} - "
                        f"ТОРГОВЛЯ ЗАБЛОКИРОВАНА"
                    )
                    continue  # Пропускаем этот символ

                # ==================== 5. ML ПРИЗНАКИ ====================
                feature_vector = await self.ml_feature_pipeline.extract_features_single(
                  symbol=symbol,
                  orderbook_snapshot=orderbook,
                  candles=candles
                )

                if not feature_vector:
                  continue  # Пропускаем если не удалось извлечь признаки

                # ==================== 6. STRATEGY MANAGER (CONSENSUS) ====================
                consensus = self.strategy_manager.analyze_with_consensus(
                    symbol,
                    candles,
                    current_price
                )

                if not consensus:
                    continue  # Нет consensus сигнала

                # ==================== 7. ML ВАЛИДАЦИЯ ====================
                validation_result = await self.ml_validator.validate_signal(
                    consensus.final_signal,
                    feature_vector
                )

                if not validation_result.validated:
                    logger.info(
                        f"❌ Consensus сигнал отклонен ML [{symbol}]: "
                        f"{validation_result.reason}"
                    )
                    continue

                logger.info(
                    f"✅ Consensus сигнал подтвержден ML [{symbol}]: "
                    f"{validation_result.final_signal_type.value}, "
                    f"confidence={validation_result.final_confidence:.2f}"
                )

                # ==================== 8. S/R КОНТЕКСТ ====================
                nearest_levels = self.sr_detector.get_nearest_levels(
                    symbol,
                    current_price,
                    max_distance_pct=0.02
                )

                sr_context = []
                if nearest_levels["support"]:
                    sr_context.append(
                        f"Support: ${nearest_levels['support'].price:.2f} "
                        f"(strength={nearest_levels['support'].strength:.2f})"
                    )

                if nearest_levels["resistance"]:
                    sr_context.append(
                        f"Resistance: ${nearest_levels['resistance'].price:.2f} "
                        f"(strength={nearest_levels['resistance'].strength:.2f})"
                    )

                # ==================== 9. ФИНАЛЬНЫЙ СИГНАЛ ====================
                final_signal = TradingSignal(
                  symbol=symbol,
                  signal_type=validation_result.final_signal_type,
                  source=SignalSource.ML_VALIDATED,
                  strength=(
                    SignalStrength.STRONG
                    if validation_result.final_confidence > 0.8
                    else SignalStrength.MEDIUM
                  ),
                  price=current_price,
                  confidence=validation_result.final_confidence,
                  timestamp=int(datetime.now().timestamp() * 1000),
                  reason=(
                        f"Consensus ({len(consensus.contributing_strategies)} strategies) + "
                        f"ML validated | {validation_result.reason}"
                    ),
                    metadata={
                        'consensus_strategies': consensus.contributing_strategies,
                        'consensus_agreement': f"{consensus.agreement_count}/{consensus.agreement_count + consensus.disagreement_count}",
                        'ml_direction': validation_result.ml_direction,
                        'ml_confidence': validation_result.ml_confidence,
                        'sr_context': sr_context,
                        'spoofing_clear': not has_spoofing,
                        'layering_clear': not has_layering
                    }
                )

                # ==================== 10. ИСПОЛНЕНИЕ ====================
                logger.info(
                    f"🎯 ФИНАЛЬНЫЙ СИГНАЛ [{symbol}]: "
                    f"{final_signal.signal_type.value}, "
                    f"confidence={final_signal.confidence:.2f}, "
                    f"strategies={consensus.contributing_strategies}, "
                    f"SR context: {', '.join(sr_context) if sr_context else 'None'}"
                )

                # Отправляем на исполнение
                await self.execution_manager.submit_signal(final_signal)

                # ==================== 11. DRIFT MONITORING ====================
                # Конвертируем ml_direction в int (BUY=1, HOLD=0, SELL=2)
                ml_direction_map = {"BUY": 1, "HOLD": 0, "SELL": 2}
                prediction_int = ml_direction_map.get(
                  validation_result.ml_direction,
                  0
                ) if validation_result.ml_direction else 0

                self.drift_detector.add_observation(
                  features=feature_vector.to_array(),
                  prediction=prediction_int,
                  label=None
                )

                # Периодическая проверка drift (каждые 24 часа)
                if self.drift_detector.should_check_drift():
                    drift_metrics = self.drift_detector.check_drift()

                    if drift_metrics and drift_metrics.drift_detected:
                        logger.warning(
                            f"⚠️  MODEL DRIFT ОБНАРУЖЕН:\n"
                            f"   Severity: {drift_metrics.severity}\n"
                            f"   Feature drift: {drift_metrics.feature_drift_score:.4f}\n"
                            f"   Prediction drift: {drift_metrics.prediction_drift_score:.4f}\n"
                            f"   Accuracy drop: {drift_metrics.accuracy_drop:.4f}\n"
                            f"   Recommendation: {drift_metrics.recommendation}"
                        )

                        # Сохраняем drift history
                        self.drift_detector.save_drift_history(
                            f"logs/drift_history_{symbol}.json"
                        )

            # Пауза между итерациями
            await asyncio.sleep(0.5)  # 500ms цикл

        except Exception as e:
            logger.error(f"Ошибка в analysis loop: {e}", exc_info=True)
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


# Глобальный контроллер бота
bot_controller: Optional[BotController] = None


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

@asynccontextmanager
async def lifespan(app):
  """
  Управление жизненным циклом приложения.

  Args:
      app: FastAPI приложение
  """
  global bot_controller, screener_processor, screener_ticker_manager

  # Startup
  logger.info("Запуск приложения")
  try:

    with trace_operation("app_startup"):
      # 1. Инициализация базы данных
      logger.info("→ Инициализация базы данных...")
      await db_manager.initialize()
      logger.info("✓ База данных подключена")

      # ===== НОВОЕ: ИНИЦИАЛИЗАЦИЯ SCREENER =====
      if settings.SCREENER_ENABLED:
        logger.info("=" * 80)
        logger.info("ИНИЦИАЛИЗАЦИЯ SCREENER")
        logger.info("=" * 80)

        # ШАГ 1: Создаем процессор скринера
        screener_processor = ScreenerProcessor(
          min_volume=settings.SCREENER_MIN_VOLUME
        )
        logger.info("✓ ScreenerProcessor создан")

        # ШАГ 2: Создаем менеджер тикеров
        screener_ticker_manager = ScreenerTickerManager(screener_processor)
        logger.info("✓ ScreenerTickerManager создан")

        # ШАГ 3: Запускаем менеджер тикеров (WebSocket подключение)
        await screener_ticker_manager.start()
        logger.info("✓ ScreenerTickerManager запущен")

        # ШАГ 4: ТОЛЬКО ТЕПЕРЬ запускаем фоновые задачи
        asyncio.create_task(screener_broadcast_task())
        logger.info("✓ Screener broadcast task запущен")

        asyncio.create_task(screener_stats_task())
        logger.info("✓ Screener stats task запущен")

        logger.info("=" * 80)
        logger.info("✅ SCREENER ПОЛНОСТЬЮ ИНИЦИАЛИЗИРОВАН")
        logger.info("=" * 80)
      else:
        logger.info("⚠️  Screener отключен в конфигурации")

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
      if screener_ticker_manager:
        await screener_ticker_manager.stop()
        logger.info("✓ ScreenerTickerManager остановлен")

      if bot_controller:
        await bot_controller.stop()

      await rest_client.close()
      await db_manager.close()

      await cleanup_tasks.stop()

    logger.info("Приложение остановлено")


# Импортируем FastAPI приложение и добавляем lifespan
from api.app import app

app.router.lifespan_context = lifespan

# Регистрируем роутеры
from api.routes import auth_router, bot_router, data_router, trading_router, monitoring_router, ml_router, \
  detection_router, strategies_router, screener_router, orders_router

app.include_router(auth_router)
app.include_router(bot_router)
app.include_router(data_router)
app.include_router(trading_router)
app.include_router(monitoring_router)
app.include_router(ml_router)
app.include_router(detection_router)
app.include_router(strategies_router)

app.include_router(screener_router)

app.include_router(orders_router)
# WebSocket эндпоинт
# @app.websocket("/ws")
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

@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    """WebSocket эндпоинт для фронтенда."""
    await websocket_endpoint(websocket)


# ==================== SCREENER COMPONENTS ====================

# Глобальные компоненты скринера
screener_processor: Optional[ScreenerProcessor] = None
screener_ticker_manager: Optional[ScreenerTickerManager] = None


# ==================== SCREENER BROADCAST TASK ====================

async def screener_broadcast_task():
  """
  Фоновая задача для broadcast данных скринера через WebSocket.

  Периодически:
  - Получает актуальные данные от ScreenerProcessor
  - Отправляет их всем подключенным клиентам через WebSocket
  - Логирует статистику

  Интервал обновления настраивается через SCREENER_BROADCAST_INTERVAL.
  """
  global screener_processor

  logger.info("=" * 80)
  logger.info("ЗАПУСК SCREENER BROADCAST TASK")
  logger.info(f"Интервал: {settings.SCREENER_BROADCAST_INTERVAL} сек")
  logger.info(f"Min Volume: {settings.SCREENER_MIN_VOLUME:,.0f} USDT")
  logger.info("=" * 80)

  iteration = 0
  last_stats_log = 0

  # Ждем инициализации процессора
  while screener_processor is None:
    logger.warning("Ожидание инициализации screener_processor...")
    await asyncio.sleep(1)

  logger.info("✓ screener_processor инициализирован, начинаем broadcast")

  while True:
    try:
      iteration += 1
      current_time = time.time()

      # Проверка на None (на всякий случай)
      if screener_processor is None:
        logger.error("screener_processor стал None! Прерываем задачу.")
        break

      # Получаем данные от процессора
      screener_data = screener_processor.get_screener_data()

      # Логируем статистику каждые N секунд
      if current_time - last_stats_log >= settings.SCREENER_STATS_LOG_INTERVAL:
        stats = screener_processor.get_statistics()
        logger.info(
          f"📊 Screener Stats: {stats['total_pairs']} пар, "
          f"{stats['active_pairs']} активных, "
          f"broadcast #{iteration}"
        )
        last_stats_log = current_time

      # Broadcast через WebSocket
      from api.websocket import manager

      if manager.active_connections:
        await manager.broadcast(screener_data, authenticated_only=False)

        # Детальное логирование для первых 5 итераций
        if iteration <= 5:
          logger.debug(
            f"Broadcast #{iteration}: {len(screener_data['pairs'])} пар → "
            f"{len(manager.active_connections)} клиентов"
          )
      else:
        # Логируем только раз в минуту если нет клиентов
        if iteration % 30 == 1:
          logger.debug("Нет активных WebSocket подключений для broadcast")

    except Exception as e:
      logger.error(f"Ошибка в screener_broadcast_task (итерация {iteration}): {e}")
      logger.error(traceback.format_exc())

    # Ждем до следующей итерации
    await asyncio.sleep(settings.SCREENER_BROADCAST_INTERVAL)


async def screener_stats_task():
  """
  Фоновая задача для периодического логирования статистики скринера.

  Логирует:
  - Статистику ScreenerProcessor
  - Статистику ScreenerTickerManager
  - Статистику WebSocket соединений
  """
  global screener_processor, screener_ticker_manager

  logger.info("Запуск screener_stats_task")

  # Ждем инициализации компонентов
  while screener_processor is None or screener_ticker_manager is None:
    logger.warning("Ожидание инициализации компонентов screener...")
    await asyncio.sleep(1)

  logger.info("✓ Компоненты screener инициализированы, начинаем логирование статистики")

  while True:
    try:
      await asyncio.sleep(settings.SCREENER_STATS_LOG_INTERVAL)

      # Проверка на None
      if screener_processor is None or screener_ticker_manager is None:
        logger.error("Компоненты screener стали None! Прерываем задачу.")
        break

      # Статистика процессора
      proc_stats = screener_processor.get_statistics()
      logger.info(
        f"ScreenerProcessor: {proc_stats['total_pairs']} всего, "
        f"{proc_stats['active_pairs']} активных, "
        f"min_volume={proc_stats['min_volume']:,.0f}"
      )

      # Статистика менеджера тикеров
      mgr_stats = screener_ticker_manager.get_statistics()

      if mgr_stats.get('websocket_stats'):
        ws_stats = mgr_stats['websocket_stats']
        logger.info(
          f"TickerWebSocket: {ws_stats['tickers_processed']} тикеров обработано, "
          f"{ws_stats['messages_received']} сообщений, "
          f"{ws_stats['errors_count']} ошибок, "
          f"connected={ws_stats['is_connected']}"
        )

    except Exception as e:
      logger.error(f"Ошибка в screener_stats_task: {e}")
      logger.error(traceback.format_exc())


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