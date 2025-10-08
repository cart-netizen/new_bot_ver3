"""
Модуль исполнения торговых ордеров.
Обрабатывает торговые сигналы и размещает ордера на бирже.
"""

import asyncio
from typing import Optional, Dict, List
from collections import deque

from core.logger import get_logger
from core.exceptions import ExecutionError, OrderExecutionError
from models.signal import TradingSignal, SignalType
from models.market_data import OrderSide, OrderType, TimeInForce
from exchange.rest_client import rest_client
from strategy.risk_manager import RiskManager
from utils.helpers import get_timestamp_ms, round_price, round_quantity

logger = get_logger(__name__)


class ExecutionManager:
  """Менеджер исполнения торговых ордеров."""

  def __init__(self, risk_manager: RiskManager):
    """
    Инициализация менеджера исполнения.

    Args:
        risk_manager: Менеджер рисков
    """
    self.risk_manager = risk_manager
    self.rest_client = rest_client

    # Очередь сигналов для исполнения
    self.signal_queue: asyncio.Queue = asyncio.Queue()

    # История исполнения
    self.execution_history: deque = deque(maxlen=1000)

    # Флаг работы
    self.is_running = False
    self.execution_task: Optional[asyncio.Task] = None

    # Статистика
    self.stats = {
      "total_signals": 0,
      "executed_orders": 0,
      "rejected_orders": 0,
      "failed_orders": 0,
    }

    logger.info("Инициализирован менеджер исполнения")

  async def start(self):
    """Запуск менеджера исполнения."""
    if self.is_running:
      logger.warning("Менеджер исполнения уже запущен")
      return

    self.is_running = True
    logger.info("Запуск менеджера исполнения")

    # Запускаем задачу обработки очереди
    self.execution_task = asyncio.create_task(self._process_queue())

  async def stop(self):
    """Остановка менеджера исполнения."""
    if not self.is_running:
      logger.warning("Менеджер исполнения уже остановлен")
      return

    logger.info("Остановка менеджера исполнения")
    self.is_running = False

    # Отменяем задачу обработки
    if self.execution_task and not self.execution_task.done():
      self.execution_task.cancel()
      try:
        await self.execution_task
      except asyncio.CancelledError:
        pass

  async def submit_signal(self, signal: TradingSignal):
    """
    Отправка сигнала на исполнение.

    Args:
        signal: Торговый сигнал
    """
    await self.signal_queue.put(signal)
    self.stats["total_signals"] += 1
    logger.debug(f"{signal.symbol} | Сигнал добавлен в очередь исполнения")

  async def _process_queue(self):
    """Обработка очереди сигналов."""
    logger.info("Запущена обработка очереди исполнения")

    while self.is_running:
      try:
        # Получаем сигнал из очереди с таймаутом
        try:
          signal = await asyncio.wait_for(
            self.signal_queue.get(),
            timeout=1.0
          )
        except asyncio.TimeoutError:
          continue

        # Обрабатываем сигнал
        await self._execute_signal(signal)

      except Exception as e:
        logger.error(f"Ошибка обработки очереди исполнения: {e}")
        await asyncio.sleep(1)

  async def _execute_signal(self, signal: TradingSignal):
    """
    Исполнение торгового сигнала.

    Args:
        signal: Торговый сигнал
    """
    logger.info(
      f"{signal.symbol} | Исполнение сигнала: "
      f"{signal.signal_type.value} @ {signal.price:.8f}"
    )

    try:
      # Получаем доступный баланс (ЗАГЛУШКА)
      available_balance = 10000.0  # USDT

      # Рассчитываем размер позиции
      position_size_usdt = self.risk_manager.calculate_position_size(
        signal,
        available_balance
      )

      # Валидируем сигнал
      is_valid, rejection_reason = self.risk_manager.validate_signal(
        signal,
        position_size_usdt
      )

      if not is_valid:
        logger.warning(
          f"{signal.symbol} | Сигнал отклонен: {rejection_reason}"
        )
        self.stats["rejected_orders"] += 1
        self._add_to_history(signal, "rejected", rejection_reason)
        return

      # Исполняем ордер
      result = await self._place_order(signal, position_size_usdt)

      if result["success"]:
        # Регистрируем открытую позицию
        self.risk_manager.register_position_opened(
          symbol=signal.symbol,
          side=signal.signal_type,
          size_usdt=position_size_usdt,
          entry_price=signal.price
        )

        # Обновляем сигнал
        signal.executed = True
        signal.execution_price = signal.price
        signal.execution_timestamp = get_timestamp_ms()

        self.stats["executed_orders"] += 1
        self._add_to_history(signal, "executed", result["order_id"])

        logger.info(
          f"{signal.symbol} | Ордер успешно исполнен: "
          f"order_id={result['order_id']}"
        )
      else:
        self.stats["failed_orders"] += 1
        self._add_to_history(signal, "failed", result["error"])
        logger.error(
          f"{signal.symbol} | Ошибка исполнения ордера: {result['error']}"
        )

    except Exception as e:
      self.stats["failed_orders"] += 1
      self._add_to_history(signal, "failed", str(e))
      logger.error(f"{signal.symbol} | Ошибка исполнения сигнала: {e}")
      raise ExecutionError(f"Failed to execute signal: {str(e)}")

  async def _place_order(
      self,
      signal: TradingSignal,
      position_size_usdt: float
  ) -> Dict:
    """
    Размещение ордера на бирже.

    Args:
        signal: Торговый сигнал
        position_size_usdt: Размер позиции в USDT

    Returns:
        Dict: Результат размещения ордера
    """
    try:
      # Конвертируем тип сигнала в сторону ордера
      side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL

      # Рассчитываем количество базового актива
      quantity = position_size_usdt / signal.price
      quantity = round_quantity(quantity, decimals=6)

      # Для скальпинга используем Market ордер
      order_type = OrderType.MARKET

      logger.info(
        f"{signal.symbol} | Размещение ордера: "
        f"{side.value} {order_type.value} qty={quantity:.6f}"
      )

      # ЗАГЛУШКА: Вызываем REST API (не размещает реальный ордер)
      response = await self.rest_client.place_order(
        symbol=signal.symbol,
        side=side.value,
        order_type=order_type.value,
        quantity=quantity,
        price=None,  # Market ордер не требует цены
        time_in_force=TimeInForce.IOC.value  # Immediate or Cancel для скальпинга
      )

      return {
        "success": True,
        "order_id": response["result"]["orderId"],
        "quantity": quantity
      }

    except Exception as e:
      logger.error(f"{signal.symbol} | Ошибка размещения ордера: {e}")
      return {
        "success": False,
        "error": str(e)
      }

  def _add_to_history(self, signal: TradingSignal, status: str, details: str):
    """
    Добавление записи в историю исполнения.

    Args:
        signal: Торговый сигнал
        status: Статус исполнения
        details: Детали
    """
    self.execution_history.append({
      "timestamp": get_timestamp_ms(),
      "symbol": signal.symbol,
      "signal_type": signal.signal_type.value,
      "price": signal.price,
      "status": status,
      "details": details
    })

  def get_execution_history(self, limit: Optional[int] = None) -> List[Dict]:
    """
    Получение истории исполнения.

    Args:
        limit: Максимальное количество записей

    Returns:
        List[Dict]: История исполнения
    """
    history = list(self.execution_history)

    if limit:
      history = history[-limit:]

    return history

  def get_statistics(self) -> Dict:
    """
    Получение статистики исполнения.

    Returns:
        Dict: Статистика
    """
    return {
      **self.stats,
      "queue_size": self.signal_queue.qsize(),
      "success_rate": (
        (self.stats["executed_orders"] / self.stats["total_signals"] * 100)
        if self.stats["total_signals"] > 0 else 0
      ),
      "rejection_rate": (
        (self.stats["rejected_orders"] / self.stats["total_signals"] * 100)
        if self.stats["total_signals"] > 0 else 0
      ),
    }