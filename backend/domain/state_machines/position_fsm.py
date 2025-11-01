"""
FSM для жизненного цикла позиции.
Управление состояниями позиции с валидацией.
"""

from transitions import Machine
from datetime import datetime

from backend.core.logger import get_logger
from backend.core.exceptions import ExecutionError
from backend.database.models import PositionStatus

logger = get_logger(__name__)


class InvalidStateTransitionError(ExecutionError):
    """Ошибка невалидного перехода состояния."""
    def __init__(self, entity_id: str, from_state: str, to_state: str):
        message = f"Invalid transition for {entity_id}: {from_state} -> {to_state}"
        details = {
            "entity_id": entity_id,
            "from_state": from_state,
            "to_state": to_state
        }
        super().__init__(message, details)


class PositionStateMachine:
    """
    Машина состояний для позиции.
    Контролирует жизненный цикл от открытия до закрытия.
    """

    # Определение состояний
    states = [
        PositionStatus.OPENING,
        PositionStatus.OPEN,
        PositionStatus.CLOSING,
        PositionStatus.CLOSED,
    ]

    def __init__(self, position_id: str, initial_state: PositionStatus = PositionStatus.OPENING):
        """
        Инициализация FSM для позиции.

        Args:
            position_id: ID позиции
            initial_state: Начальное состояние
        """
        self.position_id = position_id
        self.current_status = initial_state
        self.transition_history = []

        # Создаем машину состояний
        self.machine = Machine(
            model=self,
            states=PositionStateMachine.states,
            initial=initial_state,
            auto_transitions=False,
            send_event=True,
        )

        # Настраиваем переходы
        self._setup_transitions()

        logger.debug(f"Position FSM создана: {position_id} в состоянии {initial_state}")

    def _setup_transitions(self):
        """Настройка разрешенных переходов между состояниями."""

        # OPENING -> OPEN (ордер на вход исполнен)
        self.machine.add_transition(
            trigger="confirm_open",
            source=PositionStatus.OPENING,
            dest=PositionStatus.OPEN,
            before="_log_transition",
            after="_on_opened",
        )

        # OPEN -> CLOSING (начинаем закрытие)
        self.machine.add_transition(
            trigger="start_close",
            source=PositionStatus.OPEN,
            dest=PositionStatus.CLOSING,
            before="_log_transition",
            after="_on_closing",
        )

        # CLOSING -> CLOSED (ордер на выход исполнен)
        self.machine.add_transition(
            trigger="confirm_close",
            source=PositionStatus.CLOSING,
            dest=PositionStatus.CLOSED,
            before="_log_transition",
            after="_on_closed",
        )

        # OPENING -> CLOSED (не удалось открыть, сразу закрываем)
        self.machine.add_transition(
            trigger="abort",
            source=PositionStatus.OPENING,
            dest=PositionStatus.CLOSED,
            before="_log_transition",
            after="_on_aborted",
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
            f"Position {self.position_id} переход: "
            f"{transition['from']} -> {transition['to']} "
            f"(trigger: {transition['trigger']})"
        )

    def _on_opened(self, event_data):
        """Callback при открытии позиции."""
        logger.info(f"Position {self.position_id} успешно открыта")

    def _on_closing(self, event_data):
        """Callback при начале закрытия."""
        logger.info(f"Position {self.position_id} начинает закрываться")

    def _on_closed(self, event_data):
        """Callback при закрытии позиции."""
        logger.info(f"Position {self.position_id} закрыта")

    def _on_aborted(self, event_data):
        """Callback при отмене открытия."""
        logger.warning(f"Position {self.position_id} не открыта, прервана")

    def can_transition_to(self, target_state: PositionStatus) -> bool:
        """
        Проверка возможности перехода в состояние.

        Args:
            target_state: Целевое состояние

        Returns:
            bool: True если переход возможен
        """
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
        return self.current_status == PositionStatus.CLOSED

    def is_active(self) -> bool:
        """
        Проверка активности позиции.

        Returns:
            bool: True если позиция активна
        """
        return self.current_status == PositionStatus.OPEN

    def update_status(self, new_status: PositionStatus, raise_on_invalid: bool = True) -> bool:
        """
        Обновление статуса через FSM.

        Args:
            new_status: Новый статус
            raise_on_invalid: Бросать исключение при невалидном переходе

        Returns:
            bool: True если переход успешен

        Raises:
            InvalidStateTransitionError: Если переход невозможен и raise_on_invalid=True
        """
        if self.current_status == new_status:
            logger.debug(f"Position {self.position_id} уже в состоянии {new_status}")
            return True

        # Проверяем возможность перехода
        if not self.can_transition_to(new_status):
            error_msg = (
                f"Невозможный переход Position {self.position_id}: "
                f"{self.current_status} -> {new_status}"
            )
            logger.error(error_msg)

            if raise_on_invalid:
                raise InvalidStateTransitionError(
                    self.position_id,
                    str(self.current_status),
                    str(new_status)
                )
            return False

        # Выполняем переход
        try:
            # Определяем триггер по целевому состоянию
            trigger_map = {
                PositionStatus.OPEN: "confirm_open",
                PositionStatus.CLOSING: "start_close",
                PositionStatus.CLOSED: "confirm_close",
            }

            trigger = trigger_map.get(new_status)
            if trigger and hasattr(self, trigger):
                getattr(self, trigger)()
                self.current_status = new_status
                return True
            else:
                error_msg = f"Триггер для {new_status} не найден"
                logger.error(error_msg)
                if raise_on_invalid:
                    raise ExecutionError(error_msg)
                return False

        except Exception as e:
            logger.error(f"Ошибка перехода Position {self.position_id}: {e}")
            if raise_on_invalid:
                raise
            return False

    def get_transition_history(self) -> list:
        """
        Получение истории переходов.

        Returns:
            list: История переходов
        """
        return self.transition_history.copy()