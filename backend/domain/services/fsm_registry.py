"""
FSM Registry - Глобальный реестр машин состояний.

Централизованное управление FSM для ордеров и позиций.
Обеспечивает доступ к state machines после восстановления системы.
"""

from typing import Dict, Optional, List
from datetime import datetime

from backend.core.logger import get_logger
from backend.domain.state_machines.order_fsm import OrderStateMachine
from backend.domain.state_machines.position_fsm import PositionStateMachine

logger = get_logger(__name__)


class FSMRegistry:
  """
  Глобальный реестр FSM для ордеров и позиций.

  Обеспечивает:
  - Централизованное хранение FSM в памяти
  - Доступ к FSM по идентификаторам
  - Автоматическую очистку завершенных FSM
  - Статистику активных state machines
  """

  def __init__(self):
    """Инициализация реестра."""
    self._order_fsms: Dict[str, OrderStateMachine] = {}
    self._position_fsms: Dict[str, PositionStateMachine] = {}
    self._initialized_at = datetime.utcnow()

    logger.info("FSM Registry инициализирован")

  # ==================== ORDER FSM ====================

  def register_order_fsm(
      self,
      client_order_id: str,
      fsm: OrderStateMachine
  ) -> None:
    """
    Регистрация FSM ордера в реестре.

    Args:
        client_order_id: Уникальный идентификатор ордера
        fsm: Экземпляр OrderStateMachine

    Raises:
        ValueError: Если FSM с таким ID уже зарегистрирована
    """
    if client_order_id in self._order_fsms:
      logger.warning(
        f"FSM для ордера {client_order_id} уже зарегистрирована, "
        f"перезаписываем"
      )

    self._order_fsms[client_order_id] = fsm

    logger.debug(
      f"FSM зарегистрирована для ордера: {client_order_id} | "
      f"Текущий статус: {fsm.current_status} | "
      f"Доступные переходы: {fsm.get_available_transitions()}"
    )

  def get_order_fsm(self, client_order_id: str) -> Optional[OrderStateMachine]:
    """
    Получение FSM ордера из реестра.

    Args:
        client_order_id: Идентификатор ордера

    Returns:
        Optional[OrderStateMachine]: FSM ордера или None если не найдена
    """
    fsm = self._order_fsms.get(client_order_id)

    if not fsm:
      logger.debug(f"FSM для ордера {client_order_id} не найдена в реестре")

    return fsm

  def unregister_order_fsm(self, client_order_id: str) -> bool:
    """
    Удаление FSM ордера из реестра.

    Используется при завершении жизненного цикла ордера
    (Filled, Cancelled, Rejected, Failed).

    Args:
        client_order_id: Идентификатор ордера

    Returns:
        bool: True если FSM была удалена, False если не существовала
    """
    if client_order_id in self._order_fsms:
      fsm = self._order_fsms[client_order_id]

      # Проверяем что FSM в терминальном состоянии
      if not fsm.is_terminal_state():
        logger.warning(
          f"Попытка удалить FSM ордера {client_order_id} "
          f"в нетерминальном состоянии {fsm.current_status}"
        )

      del self._order_fsms[client_order_id]
      logger.debug(f"FSM удалена для ордера: {client_order_id}")
      return True

    logger.debug(f"FSM для ордера {client_order_id} не существует в реестре")
    return False

  def get_all_order_fsms(self) -> Dict[str, OrderStateMachine]:
    """
    Получение всех зарегистрированных FSM ордеров.

    Returns:
        Dict[str, OrderStateMachine]: Словарь всех FSM ордеров
    """
    return self._order_fsms.copy()

  def get_order_ids_by_status(self, status_filter: List[str]) -> List[str]:
    """
    Получение ID ордеров по статусам.

    Args:
        status_filter: Список статусов для фильтрации

    Returns:
        List[str]: Список client_order_id
    """
    filtered_ids = [
      client_order_id
      for client_order_id, fsm in self._order_fsms.items()
      if str(fsm.current_status.value) in status_filter
    ]

    logger.debug(
      f"Найдено {len(filtered_ids)} ордеров со статусами {status_filter}"
    )

    return filtered_ids

  # ==================== POSITION FSM ====================

  def register_position_fsm(
      self,
      position_id: str,
      fsm: PositionStateMachine
  ) -> None:
    """
    Регистрация FSM позиции в реестре.

    Args:
        position_id: Уникальный идентификатор позиции
        fsm: Экземпляр PositionStateMachine
    """
    if position_id in self._position_fsms:
      logger.warning(
        f"FSM для позиции {position_id} уже зарегистрирована, "
        f"перезаписываем"
      )

    self._position_fsms[position_id] = fsm

    logger.debug(
      f"FSM зарегистрирована для позиции: {position_id} | "
      f"Текущий статус: {fsm.current_status} | "
      f"Активна: {fsm.is_active()}"
    )

  def get_position_fsm(self, position_id: str) -> Optional[PositionStateMachine]:
    """
    Получение FSM позиции из реестра.

    Args:
        position_id: Идентификатор позиции

    Returns:
        Optional[PositionStateMachine]: FSM позиции или None если не найдена
    """
    fsm = self._position_fsms.get(position_id)

    if not fsm:
      logger.debug(f"FSM для позиции {position_id} не найдена в реестре")

    return fsm

  def unregister_position_fsm(self, position_id: str) -> bool:
    """
    Удаление FSM позиции из реестра.

    Используется при закрытии позиции.

    Args:
        position_id: Идентификатор позиции

    Returns:
        bool: True если FSM была удалена, False если не существовала
    """
    if position_id in self._position_fsms:
      fsm = self._position_fsms[position_id]

      # Проверяем что позиция закрыта
      if not fsm.is_terminal_state():
        logger.warning(
          f"Попытка удалить FSM позиции {position_id} "
          f"в нетерминальном состоянии {fsm.current_status}"
        )

      del self._position_fsms[position_id]
      logger.debug(f"FSM удалена для позиции: {position_id}")
      return True

    logger.debug(f"FSM для позиции {position_id} не существует в реестре")
    return False

  def get_all_position_fsms(self) -> Dict[str, PositionStateMachine]:
    """
    Получение всех зарегистрированных FSM позиций.

    Returns:
        Dict[str, PositionStateMachine]: Словарь всех FSM позиций
    """
    return self._position_fsms.copy()

  def get_active_position_ids(self) -> List[str]:
    """
    Получение ID активных позиций (статус OPEN).

    Returns:
        List[str]: Список position_id активных позиций
    """
    active_ids = [
      position_id
      for position_id, fsm in self._position_fsms.items()
      if fsm.is_active()
    ]

    logger.debug(f"Найдено {len(active_ids)} активных позиций")

    return active_ids

  # ==================== УТИЛИТЫ ====================

  def get_stats(self) -> Dict[str, any]:
    """
    Статистика реестра.

    Returns:
        Dict: Статистика по FSM
    """
    # Подсчет ордеров по статусам
    order_stats = {}
    for fsm in self._order_fsms.values():
      status = str(fsm.current_status.value)
      order_stats[status] = order_stats.get(status, 0) + 1

    # Подсчет позиций по статусам
    position_stats = {}
    for fsm in self._position_fsms.values():
      status = str(fsm.current_status.value)
      position_stats[status] = position_stats.get(status, 0) + 1

    stats = {
      "initialized_at": self._initialized_at.isoformat(),
      "total_order_fsms": len(self._order_fsms),
      "total_position_fsms": len(self._position_fsms),
      "order_fsms_by_status": order_stats,
      "position_fsms_by_status": position_stats,
    }

    return stats

  def clear_terminal_fsms(self) -> Dict[str, int]:
    """
    Очистка всех FSM в терминальных состояниях.

    Освобождает память от завершенных FSM.

    Returns:
        Dict[str, int]: Количество очищенных FSM по типам
    """
    logger.info("Начало очистки терминальных FSM...")

    # Очистка ордеров в терминальных состояниях
    terminal_orders = [
      client_order_id
      for client_order_id, fsm in self._order_fsms.items()
      if fsm.is_terminal_state()
    ]

    for client_order_id in terminal_orders:
      del self._order_fsms[client_order_id]

    # Очистка закрытых позиций
    closed_positions = [
      position_id
      for position_id, fsm in self._position_fsms.items()
      if fsm.is_terminal_state()
    ]

    for position_id in closed_positions:
      del self._position_fsms[position_id]

    cleared = {
      "orders_cleared": len(terminal_orders),
      "positions_cleared": len(closed_positions),
    }

    logger.info(
      f"✓ Очистка завершена: "
      f"ордеров - {cleared['orders_cleared']}, "
      f"позиций - {cleared['positions_cleared']}"
    )

    return cleared

  def clear_all(self) -> None:
    """
    Полная очистка всех FSM.

    ВНИМАНИЕ: Используется только для тестов или экстренного рестарта!
    """
    orders_count = len(self._order_fsms)
    positions_count = len(self._position_fsms)

    self._order_fsms.clear()
    self._position_fsms.clear()

    logger.warning(
      f"FSM Registry полностью очищен! "
      f"Удалено: {orders_count} ордеров, {positions_count} позиций"
    )

  def __repr__(self) -> str:
    """Строковое представление реестра."""
    return (
      f"<FSMRegistry orders={len(self._order_fsms)} "
      f"positions={len(self._position_fsms)}>"
    )


# ==================== ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР ====================

fsm_registry = FSMRegistry()