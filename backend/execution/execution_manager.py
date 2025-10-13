"""
Модуль исполнения торговых ордеров.
Обрабатывает торговые сигналы и размещает ордера на бирже.
"""

import asyncio
from typing import Optional, Dict, List
from collections import deque

from core.logger import get_logger
from core.exceptions import ExecutionError, OrderExecutionError
from core.trace_context import trace_operation
from database.models import AuditAction, OrderStatus, PositionStatus
from domain.services.fsm_registry import fsm_registry
from domain.services.idempotency_service import idempotency_service
from domain.state_machines.order_fsm import OrderStateMachine
from domain.state_machines.position_fsm import PositionStateMachine
from infrastructure.repositories.audit_repository import audit_repository
from infrastructure.repositories.order_repository import order_repository
from infrastructure.repositories.position_repository import position_repository
from infrastructure.resilience.circuit_breaker import circuit_breaker_manager
from infrastructure.resilience.rate_limiter import rate_limited
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

    # Circuit breakers для API
    self.order_breaker = circuit_breaker_manager.get_breaker(
      name="order_placement",
      failure_threshold=5,
      cooldown_seconds=60
    )

    logger.info("ExecutionManager инициализирован с полной защитой")

    logger.info("Инициализирован менеджер исполнения")

  async def open_position(
        self,
        symbol: str,
        side: str,  # "Buy" или "Sell"
        entry_price: float,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        entry_signal: Optional[dict] = None,
        entry_market_data: Optional[dict] = None,
        entry_indicators: Optional[dict] = None,
        entry_reason: Optional[str] = None,
    ) -> Optional[dict]:
      """
      Открытие позиции с полной интеграцией FSM Registry.

      Этот метод:
      1. Создает позицию в БД со статусом OPENING
      2. Создает и регистрирует FSM в Registry
      3. Подтверждает открытие (OPENING -> OPEN)
      4. Регистрирует в Risk Manager
      5. Логирует в Audit
      6. Возвращает результат для отслеживания

      Args:
          symbol: Торговая пара (например, "BTCUSDT")
          side: Сторона позиции "Buy" (Long) или "Sell" (Short)
          entry_price: Цена входа
          quantity: Количество базового актива
          stop_loss: Уровень Stop Loss (опционально)
          take_profit: Уровень Take Profit (опционально)
          entry_signal: Данные сигнала на вход (для анализа)
          entry_market_data: Рыночные данные при входе
          entry_indicators: Показатели индикаторов при входе
          entry_reason: Причина открытия позиции

      Returns:
          Optional[dict]: Результат с position_id или None при ошибке

      """
      with trace_operation("open_position", symbol=symbol, side=side):
        logger.info(
          f"→ Открытие позиции: {symbol} {side} | "
          f"Количество: {quantity} @ {entry_price}"
        )

        try:
          # ========================================
          # 1. СОЗДАНИЕ ПОЗИЦИИ В БД (OPENING)
          # ========================================

          order_side = OrderSide.BUY if side == "Buy" else OrderSide.SELL

          position = await position_repository.create(
            symbol=symbol,
            side=order_side,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_signal=entry_signal,
            entry_market_data=entry_market_data,
            entry_indicators=entry_indicators,
            entry_reason=entry_reason or f"{side} position opened"
          )

          position_id = str(position.id)

          logger.info(
            f"✓ Позиция создана в БД: {position_id} | "
            f"Статус: {position.status.value}"
          )

          # ========================================
          # 2. СОЗДАНИЕ И РЕГИСТРАЦИЯ FSM
          # ========================================

          position_fsm = PositionStateMachine(
            position_id=position_id,
            initial_state=PositionStatus.OPENING
          )

          # Регистрируем FSM в глобальном реестре
          fsm_registry.register_position_fsm(position_id, position_fsm)

          logger.debug(
            f"✓ FSM зарегистрирована для позиции {position_id} | "
            f"Доступные переходы: {position_fsm.get_available_transitions()}"
          )

          # ========================================
          # 3. ПОДТВЕРЖДЕНИЕ ОТКРЫТИЯ (OPENING -> OPEN)
          # ========================================

          # Используем триггер FSM напрямую
          position_fsm.confirm_open()  # OPENING -> OPEN через триггер

          # Обновляем статус в БД
          await position_repository.update_status(
            position_id=position_id,
            new_status=PositionStatus.OPEN
          )

          logger.info(
            f"✓ Позиция подтверждена через FSM: {position_id} | "
            f"Новый статус: {position_fsm.current_status.value}"
          )

          # ========================================
          # 4. РЕГИСТРАЦИЯ В RISK MANAGER
          # ========================================

          # Рассчитываем размер позиции в USDT
          position_size_usdt = quantity * entry_price

          # Конвертируем в SignalType для risk manager
          from models.signal import SignalType
          signal_type = SignalType.BUY if side == "Buy" else SignalType.SELL

          self.risk_manager.register_position_opened(
            symbol=symbol,
            side=signal_type,
            size_usdt=position_size_usdt,
            entry_price=entry_price
          )

          logger.info(
            f"✓ Позиция зарегистрирована в Risk Manager | "
            f"Размер: {position_size_usdt:.2f} USDT"
          )

          # ========================================
          # 5. АУДИТ ЛОГИРОВАНИЕ
          # ========================================

          await audit_repository.log(
            action=AuditAction.POSITION_OPEN,
            entity_type="Position",
            entity_id=position_id,
            new_value={
              "symbol": symbol,
              "side": side,
              "quantity": quantity,
              "entry_price": entry_price,
              "stop_loss": stop_loss,
              "take_profit": take_profit,
              "size_usdt": position_size_usdt,
              "status": PositionStatus.OPEN.value
            },
            reason=entry_reason or f"{side} position opened",
            success=True,
            context={
              "entry_signal": entry_signal,
              "entry_market_data": entry_market_data,
              "entry_indicators": entry_indicators
            }
          )

          logger.info(f"✓ Аудит записан для позиции {position_id}")

          # ========================================
          # 6. ВОЗВРАТ РЕЗУЛЬТАТА
          # ========================================

          result = {
            "position_id": position_id,
            "status": PositionStatus.OPEN.value,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "entry_price": entry_price,
            "size_usdt": position_size_usdt,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "opened_at": position.opened_at.isoformat()
          }

          logger.info(
            f"✓✓✓ ПОЗИЦИЯ УСПЕШНО ОТКРЫТА ✓✓✓\n"
            f"  Position ID: {position_id}\n"
            f"  Symbol: {symbol}\n"
            f"  Side: {side}\n"
            f"  Entry Price: {entry_price}\n"
            f"  Quantity: {quantity}\n"
            f"  Size: {position_size_usdt:.2f} USDT"
          )

          return result

        except Exception as e:
          logger.error(
            f"✗ Ошибка открытия позиции {symbol} {side}: {e}",
            exc_info=True
          )

          # Если FSM была создана, попытаемся её отменить
          if 'position_fsm' in locals() and 'position_id' in locals():
            try:
              # Используем триггер abort для перехода OPENING -> CLOSED
              position_fsm.abort()

              await position_repository.update_status(
                position_id=position_id,
                new_status=PositionStatus.CLOSED,
                exit_reason=f"Opening aborted due to error: {str(e)}"
              )

              # Удаляем FSM из реестра
              fsm_registry.unregister_position_fsm(position_id)

              logger.info(f"FSM отменена для позиции {position_id}")

            except Exception as cleanup_error:
              logger.error(
                f"Ошибка при очистке FSM: {cleanup_error}",
                exc_info=True
              )

          # Логируем ошибку в audit
          await audit_repository.log(
            action=AuditAction.POSITION_OPEN,
            entity_type="Position",
            entity_id=position_id if 'position_id' in locals() else "unknown",
            success=False,
            error_message=str(e),
            reason=f"Failed to open {side} position for {symbol}"
          )

          return None

  # ========================================
  # ТАКЖЕ ДОБАВИТЬ МЕТОД ЗАКРЫТИЯ ПОЗИЦИИ
  # ========================================

  async def close_position(
      self,
      position_id: str,
      exit_price: float,
      exit_signal: Optional[dict] = None,
      exit_market_data: Optional[dict] = None,
      exit_indicators: Optional[dict] = None,
      exit_reason: str = "Position closed"
  ) -> Optional[dict]:
    """
    Закрытие позиции с валидацией через FSM.

    Этот метод:
    1. Получает FSM из Registry (или восстанавливает из БД)
    2. Валидирует возможность закрытия
    3. Выполняет двухшаговое закрытие (OPEN -> CLOSING -> CLOSED)
    4. Обновляет БД
    5. Удаляет FSM из Registry
    6. Обновляет Risk Manager
    7. Логирует в Audit

    Args:
        position_id: ID позиции для закрытия
        exit_price: Цена выхода
        exit_signal: Сигнал на выход (опционально)
        exit_market_data: Рыночные данные при выходе
        exit_indicators: Индикаторы при выходе
        exit_reason: Причина закрытия

    Returns:
        Optional[dict]: Результат с realized_pnl или None при ошибке
    """
    with trace_operation("close_position", position_id=position_id):
      logger.info(f"→ Закрытие позиции: {position_id} @ {exit_price}")

      try:
        # ========================================
        # 1. ПОЛУЧЕНИЕ ПОЗИЦИИ ИЗ БД
        # ========================================

        position = await position_repository.get_by_id(position_id)

        if not position:
          logger.error(f"Позиция {position_id} не найдена в БД")
          return None

        # ========================================
        # 2. ПОЛУЧЕНИЕ ИЛИ ВОССТАНОВЛЕНИЕ FSM
        # ========================================

        position_fsm = fsm_registry.get_position_fsm(position_id)

        if not position_fsm:
          logger.warning(
            f"FSM не найдена для позиции {position_id}, "
            f"восстанавливаем из БД"
          )

          # Восстанавливаем FSM из текущего состояния в БД
          position_fsm = PositionStateMachine(
            position_id=position_id,
            initial_state=position.status
          )

          # Регистрируем восстановленную FSM
          fsm_registry.register_position_fsm(position_id, position_fsm)

          logger.info(
            f"FSM восстановлена из БД для позиции {position_id} | "
            f"Статус: {position_fsm.current_status.value}"
          )

        # ========================================
        # 3. ВАЛИДАЦИЯ ВОЗМОЖНОСТИ ЗАКРЫТИЯ
        # ========================================

        if not position_fsm.can_transition_to(PositionStatus.CLOSING):
          logger.error(
            f"Невозможно закрыть позицию {position_id} | "
            f"Текущий статус: {position_fsm.current_status.value} | "
            f"Доступные переходы: {position_fsm.get_available_transitions()}"
          )
          return None

        logger.debug(f"✓ Валидация закрытия прошла для позиции {position_id}")

        # ========================================
        # 4. ДВУХШАГОВОЕ ЗАКРЫТИЕ ЧЕРЕЗ FSM
        # ========================================

        # Шаг 1: OPEN -> CLOSING
        position_fsm.start_closing()

        await position_repository.update_status(
          position_id=position_id,
          new_status=PositionStatus.CLOSING
        )

        logger.info(
          f"✓ Позиция переведена в CLOSING: {position_id} | "
          f"FSM статус: {position_fsm.current_status.value}"
        )

        # Здесь можно добавить логику размещения ордера на закрытие
        # await self._place_close_order(position, exit_price)

        # Шаг 2: CLOSING -> CLOSED
        position_fsm.confirm_close()

        await position_repository.update_status(
          position_id=position_id,
          new_status=PositionStatus.CLOSED,
          exit_price=exit_price,
          exit_signal=exit_signal,
          exit_market_data=exit_market_data,
          exit_indicators=exit_indicators,
          exit_reason=exit_reason
        )

        logger.info(
          f"✓ Позиция переведена в CLOSED: {position_id} | "
          f"FSM статус: {position_fsm.current_status.value}"
        )

        # ========================================
        # 5. УДАЛЕНИЕ FSM ИЗ REGISTRY
        # ========================================

        # Позиция закрыта, FSM больше не нужна
        fsm_registry.unregister_position_fsm(position_id)

        logger.debug(f"✓ FSM удалена из Registry для позиции {position_id}")

        # ========================================
        # 6. ОБНОВЛЕНИЕ RISK MANAGER
        # ========================================

        self.risk_manager.register_position_closed(position.symbol)

        logger.info(
          f"✓ Позиция удалена из Risk Manager: {position.symbol}"
        )

        # ========================================
        # 7. ПОЛУЧЕНИЕ ОБНОВЛЕННОЙ ПОЗИЦИИ ДЛЯ PNL
        # ========================================

        # Обновляем объект позиции из БД
        updated_position = await position_repository.get_by_id(position_id)

        realized_pnl = updated_position.realized_pnl or 0.0
        duration = (
          (updated_position.closed_at - updated_position.opened_at).total_seconds()
          if updated_position.closed_at and updated_position.opened_at
          else 0
        )

        # ========================================
        # 8. АУДИТ ЛОГИРОВАНИЕ
        # ========================================

        await audit_repository.log(
          action=AuditAction.POSITION_CLOSE,
          entity_type="Position",
          entity_id=position_id,
          new_value={
            "exit_price": exit_price,
            "realized_pnl": realized_pnl,
            "duration_seconds": duration,
            "status": PositionStatus.CLOSED.value
          },
          reason=exit_reason,
          success=True,
          context={
            "exit_signal": exit_signal,
            "exit_market_data": exit_market_data,
            "exit_indicators": exit_indicators
          }
        )

        logger.info(f"✓ Аудит записан для закрытия позиции {position_id}")

        # ========================================
        # 9. ВОЗВРАТ РЕЗУЛЬТАТА
        # ========================================

        result = {
          "position_id": position_id,
          "status": PositionStatus.CLOSED.value,
          "symbol": updated_position.symbol,
          "exit_price": exit_price,
          "realized_pnl": realized_pnl,
          "duration_seconds": duration,
          "closed_at": updated_position.closed_at.isoformat() if updated_position.closed_at else None
        }

        logger.info(
          f"✓✓✓ ПОЗИЦИЯ УСПЕШНО ЗАКРЫТА ✓✓✓\n"
          f"  Position ID: {position_id}\n"
          f"  Symbol: {updated_position.symbol}\n"
          f"  Exit Price: {exit_price}\n"
          f"  Realized PnL: {realized_pnl:.2f} USDT\n"
          f"  Duration: {duration:.0f}s"
        )

        return result

      except Exception as e:
        logger.error(
          f"✗ Ошибка закрытия позиции {position_id}: {e}",
          exc_info=True
        )

        # Логируем ошибку в audit
        await audit_repository.log(
          action=AuditAction.POSITION_CLOSE,
          entity_type="Position",
          entity_id=position_id,
          success=False,
          error_message=str(e),
          reason=f"Failed to close position: {exit_reason}"
        )

        return None

  @rate_limited("order_placement", tokens=1, max_wait=5.0)
  async def place_order(
      self,
      symbol: str,
      side: str,
      quantity: float,
      price: Optional[float] = None,
      signal_data: Optional[dict] = None,
      market_data: Optional[dict] = None,
      indicators: Optional[dict] = None,
      reason: Optional[str] = None
  ) -> Optional[dict]:
    """Размещение ордера с FSM регистрацией."""

    # 1. Генерация client_order_id (существующий код)
    client_order_id = idempotency_service.generate_client_order_id(
      symbol=symbol,
      side=side,
      quantity=quantity,
      price=price
    )

    # 2. Проверка идемпотентности (существующий код)
    cached = await idempotency_service.check_operation(
      operation="place_order",
      params={...}
    )
    if cached:
      return cached

    try:
      # 3. Создание ордера в БД (существующий код)
      order = await order_repository.create(
        client_order_id=client_order_id,
        symbol=symbol,
        side=OrderSide.BUY if side == "Buy" else OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        price=price,
        signal_data=signal_data,
        market_data=market_data,
        indicators=indicators,
        reason=reason
      )

      # ========================================
      # ⭐ НОВОЕ: Создание и регистрация FSM
      # ========================================
      order_fsm = OrderStateMachine(
        order_id=client_order_id,
        initial_state=OrderStatus.PENDING
      )
      fsm_registry.register_order_fsm(client_order_id, order_fsm)

      logger.debug(f"FSM зарегистрирована для ордера {client_order_id}")
      # ========================================

      # 4. Размещение на бирже (существующий код)
      exchange_result = await self._place_on_exchange(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        client_order_id=client_order_id
      )

      # ========================================
      # ⭐ ОБНОВЛЕНО: Обновление через FSM
      # ========================================
      exchange_order_id = exchange_result.get("orderId")

      # Проверяем возможность перехода
      if order_fsm.can_transition_to(OrderStatus.PLACED):
        order_fsm.update_status(OrderStatus.PLACED)

        await order_repository.update_status(
          client_order_id=client_order_id,
          new_status=OrderStatus.PLACED,
          exchange_order_id=exchange_order_id
        )

        logger.info(
          f"✓ Ордер размещен и обновлен через FSM: "
          f"{client_order_id} -> {exchange_order_id}"
        )
      else:
        logger.error(
          f"Невозможный переход FSM для ордера {client_order_id}: "
          f"{order_fsm.current_status} -> PLACED"
        )
      # ========================================

      # 5. Аудит и кэширование (существующий код)
      await audit_repository.log(...)
      await idempotency_service.save_operation_result(...)

      return {
        "client_order_id": client_order_id,
        "exchange_order_id": exchange_order_id,
        "status": "placed"
      }

    except Exception as e:
      logger.error(f"Ошибка размещения ордера: {e}", exc_info=True)

      # ========================================
      # ⭐ НОВОЕ: Обновление FSM при ошибке
      # ========================================
      if 'order_fsm' in locals():
        order_fsm.update_status(OrderStatus.FAILED)
      # ========================================

      return None

  async def cancel_order(self, client_order_id: str, symbol: str) -> bool:
    """Отмена ордера с валидацией FSM."""

    # Получаем FSM
    order_fsm = fsm_registry.get_order_fsm(client_order_id)

    if not order_fsm:
      logger.warning(f"FSM не найдена для ордера {client_order_id}")

      # Восстанавливаем из БД
      order = await order_repository.get_by_client_order_id(client_order_id)
      if order:
        order_fsm = OrderStateMachine(client_order_id, order.status)
        fsm_registry.register_order_fsm(client_order_id, order_fsm)
      else:
        logger.error(f"Ордер {client_order_id} не найден в БД")
        return False

    # Проверяем возможность отмены
    if not order_fsm.can_transition_to(OrderStatus.CANCELLED):
      logger.error(
        f"Невозможно отменить ордер {client_order_id} "
        f"в статусе {order_fsm.current_status}"
      )
      return False

    try:
      # Отменяем на бирже
      await rest_client.cancel_order(
        symbol=symbol,
        order_id=client_order_id
      )

      # Обновляем через FSM
      order_fsm.update_status(OrderStatus.CANCELLED)

      await order_repository.update_status(
        client_order_id=client_order_id,
        new_status=OrderStatus.CANCELLED
      )

      # Удаляем FSM (терминальное состояние)
      fsm_registry.unregister_order_fsm(client_order_id)

      logger.info(f"✓ Ордер отменен: {client_order_id}")
      return True

    except Exception as e:
      logger.error(f"Ошибка отмены ордера: {e}", exc_info=True)
      return False

  async def _place_on_exchange(
      self,
      symbol: str,
      side: str,
      quantity: float,
      price: Optional[float],
      client_order_id: str
  ) -> Optional[dict]:
    """
    Внутренний метод размещения на бирже.

    Args:
        symbol: Торговая пара
        side: Сторона (Buy/Sell)
        quantity: Количество
        price: Цена (для лимитных ордеров, None для рыночных)
        client_order_id: Client Order ID для отслеживания

    Returns:
        Optional[dict]: Результат размещения или None при ошибке
    """
    # Определяем тип ордера
    order_type = "Limit" if price else "Market"

    logger.debug(
      f"Размещение на бирже: {symbol} {side} {order_type} "
      f"qty={quantity} price={price} client_id={client_order_id}"
    )

    try:
      # Передаем все необходимые параметры, включая order_type
      response = await rest_client.place_order(
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        client_order_id=client_order_id
      )

      # Извлекаем результат из ответа
      result = response.get("result", {})

      # Проверка успешности размещения
      if not result or "orderId" not in result:
        logger.error(f"Некорректный ответ от биржи: {response}")
        return None

      logger.info(
        f"✓ Ордер успешно размещен на бирже: "
        f"exchange_order_id={result['orderId']} "
        f"client_order_id={result.get('orderLinkId', 'N/A')}"
      )

      return result

    except Exception as e:
      logger.error(f"Ошибка размещения ордера на бирже: {e}", exc_info=True)
      return None

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