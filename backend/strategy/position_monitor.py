"""
Position Monitor - Dedicated мониторинг открытых позиций в реальном времени.

ВОЗМОЖНОСТИ:
1. Real-time отслеживание всех открытых позиций
2. Обновление текущей цены и unrealized PnL
3. Проверка Reversal Detector для каждой позиции
4. Проверка Trailing Stop условий
5. Проверка достижения Stop Loss / Take Profit
6. Автоматическое закрытие при критических условиях

ИНТЕГРАЦИЯ:
- Запускается как отдельная asyncio задача
- Работает параллельно с analysis_loop
- Проверка каждые 1-2 секунды
- Отслеживает ТОЛЬКО открытые позиции
"""
import asyncio
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from decimal import Decimal



from core.logger import get_logger
from config import settings
from database.models import PositionStatus
from infrastructure.repositories.position_repository import position_repository
from exchange.rest_client import rest_client
from strategy.reversal_detector import reversal_detector
from strategy.risk_manager import RiskManager
from strategy.trailing_stop_manager import trailing_stop_manager
from models.signal import SignalType
from ml_engine.features.candle_feature_extractor import Candle
import numpy as np
logger = get_logger(__name__)


class PositionMonitor:
  """
  Dedicated мониторинг открытых позиций.

  Отслеживает каждую открытую позицию каждые 1-2 секунды:
  - Обновляет текущую цену
  - Рассчитывает unrealized PnL
  - Проверяет Reversal Detector
  - Проверяет Trailing Stop
  - Проверяет SL/TP условия
  """

  def __init__(
      self,
      risk_manager: RiskManager,
      candle_managers: Dict,
      orderbook_managers: Dict,
      execution_manager
  ):
    """
    Инициализация Position Monitor.

    Args:
        risk_manager: RiskManager instance
        candle_managers: Dict[symbol, CandleManager]
        orderbook_managers: Dict[symbol, OrderBookManager]
        execution_manager: ExecutionManager instance
    """
    self.risk_manager = risk_manager
    self.candle_managers = candle_managers
    self.orderbook_managers = orderbook_managers
    self.execution_manager = execution_manager

    # Настройки
    self.enabled = settings.POSITION_MONITOR_ENABLED
    self.check_interval = settings.POSITION_MONITOR_INTERVAL
    self.enable_reversal_check = settings.POSITION_MONITOR_REVERSAL_CHECK
    self.enable_trailing_stop = settings.POSITION_MONITOR_TRAILING_STOP
    self.enable_sltp_check = settings.POSITION_MONITOR_SLTP_CHECK

    # Статус
    self.is_running = False
    self.monitor_task: Optional[asyncio.Task] = None

    # Метрики
    self.total_checks = 0
    self.reversal_detections = 0
    self.trailing_stop_updates = 0
    self.sltp_triggers = 0

    logger.info(
      f"PositionMonitor initialized: "
      f"enabled={self.enabled}, "
      f"interval={self.check_interval}s, "
      f"reversal={self.enable_reversal_check}, "
      f"trailing={self.enable_trailing_stop}, "
      f"sltp={self.enable_sltp_check}"
    )

  async def start(self):
    """Запуск мониторинга."""
    if not self.enabled:
      logger.info("PositionMonitor disabled, skipping start")
      return

    if self.is_running:
      logger.warning("PositionMonitor already running")
      return

    self.is_running = True
    self.monitor_task = asyncio.create_task(self._monitoring_loop())
    logger.info("✓ PositionMonitor started")

  async def stop(self):
    """Остановка мониторинга."""
    if not self.is_running:
      return

    self.is_running = False

    if self.monitor_task:
      self.monitor_task.cancel()
      try:
        await self.monitor_task
      except asyncio.CancelledError:
        pass

    logger.info("PositionMonitor stopped")

  async def _monitoring_loop(self):
    """
    Основной цикл мониторинга.

    Выполняется каждые check_interval секунд для ВСЕХ открытых позиций.
    """
    logger.info("PositionMonitor loop started")

    while self.is_running:
      try:
        # Получаем все открытые позиции из RiskManager
        open_positions = self.risk_manager.get_all_positions()

        if not open_positions:
          logger.debug("No open positions to monitor")
          await asyncio.sleep(self.check_interval)
          continue

        logger.debug(
          f"Monitoring {len(open_positions)} position(s): "
          f"{list(open_positions.keys())}"
        )

        # Мониторим каждую позицию
        for symbol, position_info in open_positions.items():
          try:
            await self._monitor_single_position(symbol, position_info)
          except Exception as e:
            logger.error(
              f"{symbol} | Error monitoring position: {e}",
              exc_info=True
            )
            continue

        self.total_checks += 1

        # Пауза до следующей проверки
        await asyncio.sleep(self.check_interval)

      except asyncio.CancelledError:
        logger.info("Monitoring loop cancelled")
        break
      except Exception as e:
        logger.error(f"Error in monitoring loop: {e}", exc_info=True)
        await asyncio.sleep(self.check_interval)

    logger.info("PositionMonitor loop stopped")

  async def _monitor_single_position(self, symbol: str, position_info: Dict):
    """
    Мониторинг одной позиции.

    Args:
        symbol: Торговая пара
        position_info: Информация из RiskManager
    """
    # ===== ШАГ 1: ПОЛУЧЕНИЕ ТЕКУЩЕЙ ЦЕНЫ =====
    current_price = await self._get_current_price(symbol)

    if not current_price:
      logger.warning(f"{symbol} | Cannot get current price, skipping")
      return

    # ===== ШАГ 2: ПОЛУЧЕНИЕ ПОЗИЦИИ ИЗ БД =====
    position = await position_repository.find_open_by_symbol(symbol)

    if not position:
      logger.warning(
        f"{symbol} | Position in RiskManager but not in DB, "
        f"possible desync"
      )
      return

    position_id = str(position.id)

    # ===== ШАГ 3: ОБНОВЛЕНИЕ ТЕКУЩЕЙ ЦЕНЫ И PNL =====
    await self._update_position_price(position_id, current_price)

    # Рассчитываем PnL
    entry_price = position.entry_price
    quantity = position.quantity
    side = position.side.value

    unrealized_pnl = self._calculate_pnl(
      entry_price=entry_price,
      current_price=current_price,
      quantity=quantity,
      side=side
    )

    pnl_percent = (unrealized_pnl / (entry_price * quantity)) * 100

    logger.debug(
      f"{symbol} | Price: {current_price:.8f} | "
      f"PnL: ${unrealized_pnl:.2f} ({pnl_percent:+.2f}%)"
    )

    # ===== ШАГ 4: ПРОВЕРКА REVERSAL DETECTOR =====
    if self.enable_reversal_check:
      reversal_triggered = await self._check_reversal(
        symbol=symbol,
        position=position,
        position_info=position_info,
        current_price=current_price
      )

      if reversal_triggered:
        self.reversal_detections += 1
        # Если reversal critical и auto_action=True, позиция уже закрыта
        return

    # ===== ШАГ 5: ПРОВЕРКА TRAILING STOP =====
    if self.enable_trailing_stop:
      trailing_updated = await self._check_trailing_stop(
        symbol=symbol,
        position=position,
        current_price=current_price
      )

      if trailing_updated:
        self.trailing_stop_updates += 1

    # ===== ШАГ 6: ПРОВЕРКА SL/TP =====
    if self.enable_sltp_check:
      sltp_triggered = await self._check_stop_loss_take_profit(
        symbol=symbol,
        position=position,
        current_price=current_price,
        unrealized_pnl=unrealized_pnl
      )

      if sltp_triggered:
        self.sltp_triggers += 1
        return  # Позиция закрыта

  async def _get_current_price(self, symbol: str) -> Optional[float]:
    """
    Получение текущей цены из OrderBook или REST API.

    Args:
        symbol: Торговая пара

    Returns:
        Текущая цена или None
    """
    # Попытка 1: Из OrderBook Manager (самое быстрое)
    orderbook_manager = self.orderbook_managers.get(symbol)
    if orderbook_manager:
      snapshot = orderbook_manager.get_snapshot()
      if snapshot and snapshot.mid_price:
        return snapshot.mid_price

    # Попытка 2: Из REST API ticker
    try:
      ticker = await rest_client.get_ticker(symbol)
      if ticker and 'last_price' in ticker:
        return float(ticker['last_price'])
    except Exception as e:
      logger.error(f"{symbol} | Error fetching ticker: {e}")

    return None

  async def _update_position_price(self, position_id: str, current_price: float):
    """
    Обновление текущей цены позиции в БД.

    Args:
        position_id: ID позиции
        current_price: Текущая цена
    """
    try:
      await position_repository.update_current_price(
        position_id=position_id,
        current_price=current_price
      )
    except Exception as e:
      logger.error(
        f"Error updating position price {position_id}: {e}",
        exc_info=False
      )

  def _calculate_pnl(
      self,
      entry_price: float,
      current_price: float,
      quantity: float,
      side: str
  ) -> float:
    """
    Расчет unrealized PnL.

    Args:
        entry_price: Цена входа
        current_price: Текущая цена
        quantity: Количество
        side: Сторона (BUY/SELL)

    Returns:
        Unrealized PnL в USDT
    """
    if side == "BUY":
      pnl = (current_price - entry_price) * quantity
    else:  # SELL
      pnl = (entry_price - current_price) * quantity

    return pnl

  async def _check_reversal(
      self,
      symbol: str,
      position,
      position_info: Dict,
      current_price: float
  ) -> bool:
    """
    Проверка Reversal Detector для позиции.

    Args:
        symbol: Торговая пара
        position: Position model из БД
        position_info: Информация из RiskManager
        current_price: Текущая цена

    Returns:
        True если reversal обнаружен и обработан
    """
    try:
      # Получаем данные для анализа
      candle_manager = self.candle_managers.get(symbol)
      orderbook_manager = self.orderbook_managers.get(symbol)

      if not candle_manager:
        logger.debug(f"{symbol} | CandleManager not available")
        return False

      candles = candle_manager.get_candles()

      if len(candles) < 50:
        logger.debug(f"{symbol} | Insufficient candles: {len(candles)}")
        return False

      # Определяем тренд позиции
      side = position_info.get('side', 'BUY')
      current_trend = SignalType.BUY if side == 'BUY' else SignalType.SELL

      # Извлекаем индикаторы из свечей (простой расчет RSI/MACD)
      indicators = self._calculate_indicators(candles)

      # Метрики стакана
      orderbook_metrics = None
      if orderbook_manager:
        snapshot = orderbook_manager.get_snapshot()
        if snapshot:
          orderbook_metrics = {
            'imbalance': snapshot.imbalance
          }

      # Проверяем разворот
      reversal = reversal_detector.detect_reversal(
        symbol=symbol,
        candles=candles,
        current_trend=current_trend,
        indicators=indicators,
        orderbook_metrics=orderbook_metrics
      )

      if not reversal:
        return False

      # Reversal обнаружен!
      logger.warning(
        f"{symbol} | 🔄 REVERSAL in Position Monitor | "
        f"Strength: {reversal.strength.value} | "
        f"Confidence: {reversal.confidence:.2%} | "
        f"Action: {reversal.suggested_action}"
      )

      # Обработка reversal
      position_id = str(position.id)

      if reversal.suggested_action == "close_position":
        if reversal_detector.auto_action:
          logger.warning(
            f"{symbol} | CRITICAL REVERSAL - AUTO-CLOSING position"
          )

          await self.execution_manager.close_position(
            position_id=position_id,
            exit_reason=f"Critical reversal detected: {reversal.reason}",
            exit_signal={
              "type": "reversal",
              "strength": reversal.strength.value,
              "indicators": reversal.indicators_confirming,
              "confidence": reversal.confidence
            }
          )

          return True
        else:
          logger.warning(
            f"{symbol} | CRITICAL REVERSAL - Manual action required"
          )

      elif reversal.suggested_action == "reduce_size":
        logger.warning(
          f"{symbol} | STRONG REVERSAL - Consider reducing 50%"
        )
        # TODO: Реализовать partial close

      elif reversal.suggested_action == "tighten_sl":
        logger.warning(
          f"{symbol} | MODERATE REVERSAL - Consider tightening SL"
        )
        # TODO: Реализовать dynamic SL update

      return False

    except Exception as e:
      logger.error(f"{symbol} | Error checking reversal: {e}", exc_info=True)
      return False

  def _calculate_indicators(self, candles: List[Candle]) -> Dict:
    """
    Быстрый расчет индикаторов для Reversal Detector.

    Args:
        candles: История свечей

    Returns:
        Dict с индикаторами (rsi, macd, macd_signal)
    """


    if len(candles) < 50:
      return {}

    closes = np.array([c.close for c in candles])

    # Простой RSI (14 периодов)
    rsi_values = []
    period = 14

    if len(closes) >= period + 1:
      for i in range(period, len(closes)):
        window = closes[i - period:i + 1]
        gains = np.maximum(np.diff(window), 0)
        losses = np.abs(np.minimum(np.diff(window), 0))

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        if avg_loss == 0:
          rsi = 100
        else:
          rs = avg_gain / avg_loss
          rsi = 100 - (100 / (1 + rs))

        rsi_values.append(rsi)

    # Простой MACD (12, 26, 9)
    ema_12 = self._calculate_ema(closes, 12)
    ema_26 = self._calculate_ema(closes, 26)

    macd_line = ema_12 - ema_26
    macd_signal = self._calculate_ema(macd_line, 9)

    return {
      'rsi': rsi_values if rsi_values else None,
      'macd': macd_line.tolist() if len(macd_line) > 0 else None,
      'macd_signal': macd_signal.tolist() if len(macd_signal) > 0 else None
    }

  def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
    """Расчет EMA."""
    import numpy as np

    if len(data) < period:
      return np.array([])

    alpha = 2 / (period + 1)
    ema = np.zeros(len(data))
    ema[0] = data[0]

    for i in range(1, len(data)):
      ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

    return ema

  async def _check_trailing_stop(
        self,
        symbol: str,
        position,
        current_price: float
    ) -> bool:
      """
      Проверка и обновление trailing stop для позиции.

      Интегрирован с TrailingStopManager для автоматического
      подтягивания stop loss за ценой.

        Trailing manager автоматически:
      1. Проверяет достижение порога активации (1.5% прибыли)
      2. Активирует trailing stop
      3. Отслеживает highest_price (для long) или lowest_price (для short)
      4. Рассчитывает новый stop loss на расстоянии 0.8% от пика
      5. Обновляет SL в БД и на бирже

      Args:
          symbol: Торговая пара
          position: Объект позиции из БД
          current_price: Текущая цена

      Returns:
          bool: True если trailing stop был обновлен
      """
      try:
        # Обновляем текущую цену в trailing stop manager
        trailing_stop_manager.update_position_price(symbol, current_price)

        # Получаем статус trailing stop
        status = trailing_stop_manager.get_trailing_status(symbol)

        if not status:
          # Позиция не зарегистрирована в trailing manager
          logger.debug(f"{symbol} | Trailing stop не зарегистрирован")
          return False

        # Логируем статус если trailing активен
        if status['is_active']:
          logger.debug(
            f"{symbol} | Trailing stop активен: "
            f"SL=${status['current_stop_loss']:.2f}, "
            f"Distance={status['trailing_distance']:.2%}"
          )

        return False  # Trailing manager обновляет автоматически

      except Exception as e:
        logger.error(
          f"{symbol} | Ошибка проверки trailing stop: {e}",
          exc_info=True
        )
        return False

  async def _check_stop_loss_take_profit(
      self,
      symbol: str,
      position,
      current_price: float,
      unrealized_pnl: float
  ) -> bool:
    """
    Проверка достижения Stop Loss или Take Profit.

    Args:
        symbol: Торговая пара
        position: Position model
        current_price: Текущая цена
        unrealized_pnl: Текущий PnL

    Returns:
        True если SL/TP сработал и позиция закрыта
    """
    stop_loss = position.stop_loss
    take_profit = position.take_profit
    side = position.side.value

    # Проверка Stop Loss
    if stop_loss:
      sl_triggered = False

      if side == "BUY" and current_price <= stop_loss:
        sl_triggered = True
      elif side == "SELL" and current_price >= stop_loss:
        sl_triggered = True

      if sl_triggered:
        logger.warning(
          f"{symbol} | 🛑 STOP LOSS TRIGGERED | "
          f"Price: {current_price:.8f} | SL: {stop_loss:.8f}"
        )

        await self.execution_manager.close_position(
          position_id=str(position.id),
          exit_reason=f"Stop Loss triggered at {current_price:.8f}",
          exit_signal={"type": "stop_loss", "price": current_price}
        )

        return True

    # Проверка Take Profit
    if take_profit:
      tp_triggered = False

      if side == "BUY" and current_price >= take_profit:
        tp_triggered = True
      elif side == "SELL" and current_price <= take_profit:
        tp_triggered = True

      if tp_triggered:
        logger.info(
          f"{symbol} | 🎯 TAKE PROFIT TRIGGERED | "
          f"Price: {current_price:.8f} | TP: {take_profit:.8f}"
        )

        await self.execution_manager.close_position(
          position_id=str(position.id),
          exit_reason=f"Take Profit triggered at {current_price:.8f}",
          exit_signal={"type": "take_profit", "price": current_price}
        )

        return True

    return False

  def get_statistics(self) -> Dict:
    """Получение статистики работы монитора."""
    return {
      "enabled": self.enabled,
      "is_running": self.is_running,
      "check_interval": self.check_interval,
      "total_checks": self.total_checks,
      "reversal_detections": self.reversal_detections,
      "trailing_stop_updates": self.trailing_stop_updates,
      "sltp_triggers": self.sltp_triggers,
      "monitored_positions": len(self.risk_manager.get_all_positions())
    }


# Глобальный экземпляр (инициализируется в main.py)
position_monitor: Optional[PositionMonitor] = None