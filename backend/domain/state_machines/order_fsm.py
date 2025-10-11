"""
FSM для жизненного цикла ордера.
Контроль переходов между состояниями с валидацией.
"""

from typing import Optional, Dict, Any
from transitions import Machine
from datetime import datetime

from core.logger import get_logger
from database.models import OrderStatus

logger = get_logger(__name__)


class OrderStateMachine:
  """
  Машина состояний для ордера.
  Управляет переходами между состояниями с валидацией.
  """

  # Определение состояний
  states = [
    OrderStatus.PENDING,
    OrderStatus.PLACED,
    OrderStatus.PARTIALLY_FILLED,
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
    OrderStatus.FAILED,
  ]

  def __init__(self, order_id: str, initial_state: OrderStatus = OrderStatus.PENDING):
    """
    Инициализация FSM для ордера.

    Args:
        order_id: ID ордера
        initial_state: Начальное состояние
    """
    self.order_id = order_id
    self.current_status = initial_state
    self.transition_history = []

    # Создаем машину состояний
    self.machine = Machine(
      model=self,
      states=OrderStateMachine.states,
      initial=initial_state,
      auto_transitions=False,
      send_event=True,
    )

    # Определяем разрешенные переходы
    self._setup_transitions()

    logger.debug(f"Order FSM создана: {order_id} в состоянии {initial_state}")

  def _setup_transitions(self):
    """Настройка разрешенных переходов между состояниями."""

    # PENDING -> PLACED
    self.machine.add_transition(
      trigger="place",
      source=OrderStatus.PENDING,
      dest=OrderStatus.PLACED,
      before="_log_transition",
      after="_on_placed",
    )

    # PENDING -> REJECTED
    self.machine.add_transition(
      trigger="reject",
      source=OrderStatus.PENDING,
      dest=OrderStatus.REJECTED,
      before="_log_transition",
      after="_on_rejected",
    )

    # PENDING -> FAILED
    self.machine.add_transition(
      trigger="fail",
      source=OrderStatus.PENDING,
      dest=OrderStatus.FAILED,
      before="_log_transition",
    )

    # PLACED -> PARTIALLY_FILLED
    self.machine.add_transition(
      trigger="partial_fill",
      source=OrderStatus.PLACED,
      dest=OrderStatus.PARTIALLY_FILLED,
      before="_log_transition",
      after="_on_partial_fill",
    )

    # PLACED -> FILLED
    self.machine.add_transition(
      trigger="fill",
      source=OrderStatus.PLACED,
      dest=OrderStatus.FILLED,
      before="_log_transition",
      after="_on_filled",
    )

    # PLACED -> CANCELLED
    self.machine.add_transition(
      trigger="cancel",
      source=OrderStatus.PLACED,
      dest=OrderStatus.CANCELLED,
      before="_log_transition",
      after="_on_cancelled",
    )

    # PARTIALLY_FILLED -> FILLED
    self.machine.add_transition(
      trigger="fill",
      source=OrderStatus.PARTIALLY_FILLED,
      dest=OrderStatus.FILLED,
      before="_log_transition",
      after="_on_filled",
    )

    # PARTIALLY_FILLED -> CANCELLED (частичная отмена)
    self.machine.add_transition(
      trigger="cancel",
      source=OrderStatus.PARTIALLY_FILLED,
      dest=OrderStatus.CANCELLED,
      before="_log_transition",
      after="_on_cancelled",
    )

  def _log_transition(self, event_data):
    """Логирование перехода состояния."""
    transition = {
      "from": self.current_status.value if hasattr(self.current_status, 'value') else str(self.current_status),
      "to": event_data.transition.dest.value if hasattr(event_data.transition.dest, 'value') else str(
        event_data.transition.dest),
      "trigger": event_data.event.name,
      "timestamp": datetime.utcnow().isoformat(),
    }

    self.transition_history.append(transition)

    logger.info(
      f"Order {self.order_id} переход: "
      f"{transition['from']} -> {transition['to']} "
      f"(trigger: {transition['trigger']})"
    )

  def _on_placed(self, event_data):
    """Callback при размещении ордера."""
    logger.debug(f"Order {self.order_id} успешно размещен на бирже")

  def _on_partial_fill(self, event_data):
    """Callback при частичном исполнении."""
    logger.debug(f"Order {self.order_id} частично исполнен")

  def _on_filled(self, event_data):
    """Callback при полном исполнении."""
    logger.info(f"Order {self.order_id} полностью исполнен")

  def _on_cancelled(self, event_data):
    """Callback при отмене."""
    logger.info(f"Order {self.order_id} отменен")

  def _on_rejected(self, event_data):
    """Callback при отклонении."""
    logger.warning(f"Order {self.order_id} отклонен биржей")

  def can_transition_to(self, target_state: OrderStatus) -> bool:
    """
    Проверка возможности перехода в состояние.

    Args:
        target_state: Целевое состояние

    Returns:
        bool: True если переход возможен
    """
    # Получаем все возможные переходы из текущего состояния
    current = self.current_status.value if hasattr(self.current_status, 'value') else str(self.current_status)

    for transition in self.machine.get_transitions(source=current):
      dest = transition.dest.value if hasattr(transition.dest, 'value') else str(transition.dest)
      if dest == target_state.value:
        return True

    return False

  def get_available_transitions(self) -> list:
    """
    Получение доступных переходов.

    Returns:
        list: Список доступных триггеров
    """
    current = self.current_status.value if hasattr(self.current_status, 'value') else str(self.current_status)
    transitions = self.machine.get_transitions(source=current)
    return [t.trigger for t in transitions]

  def is_terminal_state(self) -> bool:
    """
    Проверка финального состояния.

    Returns:
        bool: True если состояние финальное
    """
    terminal_states = [
      OrderStatus.FILLED,
      OrderStatus.CANCELLED,
      OrderStatus.REJECTED,
      OrderStatus.FAILED,
    ]
    return self.current_status in terminal_states

  def update_status(self, new_status: OrderStatus) -> bool:
    """
    Обновление статуса через FSM.

    Args:
        new_status: Новый статус

    Returns:
        bool: True если переход успешен
    """
    if self.current_status == new_status:
      logger.debug(f"Order {self.order_id} уже в состоянии {new_status}")
      return True

    # Проверяем возможность перехода
    if not self.can_transition_to(new_status):
      logger.error(
        f"Невозможный переход Order {self.order_id}: "
        f"{self.current_status} -> {new_status}"
      )
      return False

    # Выполняем переход
    try:
      # Определяем триггер по целевому состоянию
      trigger_map = {
        OrderStatus.PLACED: "place",
        OrderStatus.PARTIALLY_FILLED: "partial_fill",
        OrderStatus.FILLED: "fill",
        OrderStatus.CANCELLED: "cancel",
        OrderStatus.REJECTED: "reject",
        OrderStatus.FAILED: "fail",
      }

      trigger = trigger_map.get(new_status)
      if trigger and hasattr(self, trigger):
        getattr(self, trigger)()
        self.current_status = new_status
        return True
      else:
        logger.error(f"Триггер для {new_status} не найден")
        return False

    except Exception as e:
      logger.error(f"Ошибка перехода Order {self.order_id}: {e}")
      return False

  def get_transition_history(self) -> list:
    """
    Получение истории переходов.

    Returns:
        list: История переходов
    """
    return self.transition_history.copy()