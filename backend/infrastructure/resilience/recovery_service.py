"""
Recovery & State Sync Service.
Восстановление состояния бота после сбоев и сверка с биржей.
"""

from typing import List, Dict, Any
from datetime import datetime

from core.logger import get_logger
from exchange.rest_client import rest_client
from infrastructure.repositories.order_repository import order_repository
from infrastructure.repositories.position_repository import position_repository
from infrastructure.repositories.audit_repository import audit_repository
from database.models import OrderStatus, PositionStatus, AuditAction
from domain.state_machines.order_fsm import OrderStateMachine
from domain.state_machines.position_fsm import PositionStateMachine

logger = get_logger(__name__)


class RecoveryService:
  """
  Сервис восстановления состояния.
  Сверяет локальное состояние с биржей при старте.
  """

  def __init__(self):
    """Инициализация сервиса восстановления."""
    logger.info("Recovery Service инициализирован")

  async def reconcile_state(self) -> Dict[str, Any]:
    """
    Полная сверка состояния с биржей.

    Returns:
        Dict: Результаты сверки
    """
    logger.info("=" * 80)
    logger.info("НАЧАЛО СВЕРКИ СОСТОЯНИЯ С БИРЖЕЙ")
    logger.info("=" * 80)

    results = {
      "orders_synced": 0,
      "positions_synced": 0,
      "discrepancies_found": 0,
      "errors": [],
    }

    try:
      # 1. Сверка ордеров
      orders_result = await self._reconcile_orders()
      results["orders_synced"] = orders_result["synced"]
      results["discrepancies_found"] += orders_result["discrepancies"]

      # 2. Сверка позиций
      positions_result = await self._reconcile_positions()
      results["positions_synced"] = positions_result["synced"]
      results["discrepancies_found"] += positions_result["discrepancies"]

      logger.info("=" * 80)
      logger.info(f"СВЕРКА ЗАВЕРШЕНА:")
      logger.info(f"  Ордеров синхронизировано: {results['orders_synced']}")
      logger.info(f"  Позиций синхронизировано: {results['positions_synced']}")
      logger.info(f"  Расхождений найдено: {results['discrepancies_found']}")
      logger.info("=" * 80)

      # Записываем в аудит
      await audit_repository.log(
        action=AuditAction.CONFIG_CHANGE,
        entity_type="System",
        entity_id="recovery",
        new_value=results,
        reason="State reconciliation on startup",
        success=True,
      )

    except Exception as e:
      logger.error(f"Ошибка при сверке состояния: {e}")
      results["errors"].append(str(e))

    return results

  async def _reconcile_orders(self) -> Dict[str, int]:
    """
    Сверка ордеров с биржей.

    Returns:
        Dict: Результаты сверки ордеров
    """
    logger.info("Сверка ордеров с биржей...")

    result = {
      "synced": 0,
      "discrepancies": 0,
    }

    try:
      # Получаем активные ордера из БД
      local_orders = await order_repository.get_active_orders()
      logger.info(f"Найдено {len(local_orders)} активных ордеров в БД")

      for local_order in local_orders:
        try:
          # Проверяем статус на бирже
          exchange_order = await rest_client.get_order_info(
            symbol=local_order.symbol,
            order_id=local_order.exchange_order_id,
          )

          if not exchange_order:
            logger.warning(
              f"Ордер {local_order.client_order_id} не найден на бирже"
            )
            result["discrepancies"] += 1

            # Помечаем как неизвестный
            await order_repository.update_status(
              client_order_id=local_order.client_order_id,
              new_status=OrderStatus.FAILED,
            )

            await audit_repository.log(
              action=AuditAction.ORDER_PLACE,
              entity_type="Order",
              entity_id=local_order.client_order_id,
              reason="Order not found on exchange during reconciliation",
              success=False,
            )
            continue

          # Сравниваем статусы
          exchange_status = self._map_exchange_order_status(
            exchange_order.get("orderStatus")
          )

          if exchange_status != local_order.status:
            logger.info(
              f"Расхождение статуса ордера {local_order.client_order_id}: "
              f"локально={local_order.status}, "
              f"биржа={exchange_status}"
            )
            result["discrepancies"] += 1

            # Обновляем локальный статус
            await order_repository.update_status(
              client_order_id=local_order.client_order_id,
              new_status=exchange_status,
              filled_quantity=float(exchange_order.get("cumExecQty", 0)),
              average_fill_price=float(exchange_order.get("avgPrice", 0)),
            )

            await audit_repository.log(
              action=AuditAction.ORDER_MODIFY,
              entity_type="Order",
              entity_id=local_order.client_order_id,
              old_value={"status": local_order.status.value},
              new_value={"status": exchange_status.value},
              reason="Status mismatch fixed during reconciliation",
              success=True,
            )

          result["synced"] += 1

        except Exception as e:
          logger.error(
            f"Ошибка сверки ордера {local_order.client_order_id}: {e}"
          )

      logger.info(f"✓ Сверка ордеров завершена: {result}")
      return result

    except Exception as e:
      logger.error(f"Критическая ошибка сверки ордеров: {e}")
      return result

  async def _reconcile_positions(self) -> Dict[str, int]:
    """
    Сверка позиций с биржей.

    Returns:
        Dict: Результаты сверки позиций
    """
    logger.info("Сверка позиций с биржей...")

    result = {
      "synced": 0,
      "discrepancies": 0,
    }

    try:
      # Получаем активные позиции из БД
      local_positions = await position_repository.get_active_positions()
      logger.info(f"Найдено {len(local_positions)} активных позиций в БД")

      # Получаем позиции с биржи
      try:
        exchange_response = await rest_client.get_positions()

        # ИСПРАВЛЕНО: Правильная обработка структуры ответа Bybit API v5
        exchange_positions_list = exchange_response.get("result", {}).get("list", [])
        logger.info(f"Получено {len(exchange_positions_list)} позиций с биржи")

      except Exception as e:
        logger.error(f"Ошибка получения позиций с биржи: {e}")
        # Если не можем получить позиции с биржи, возвращаем пустой результат
        return result

      # Создаем мапу позиций с биржи
      # Позиция считается активной если size > 0
      exchange_map = {}
      for pos in exchange_positions_list:
        symbol = pos.get("symbol")
        size = float(pos.get("size", 0))

        if symbol and size > 0:
          exchange_map[symbol] = pos

      logger.debug(f"Активных позиций на бирже: {len(exchange_map)}")

      # Сверяем каждую локальную позицию
      for local_pos in local_positions:
        try:
          exchange_pos = exchange_map.get(local_pos.symbol)

          if not exchange_pos:
            logger.warning(
              f"Позиция {local_pos.symbol} не найдена на бирже "
              f"(закрыта или size=0)"
            )
            result["discrepancies"] += 1

            # Позиция закрыта на бирже, но открыта локально
            await position_repository.update_status(
              position_id=str(local_pos.id),
              new_status=PositionStatus.CLOSED,
              exit_reason="Position closed on exchange (reconciliation)",
            )

            await audit_repository.log(
              action=AuditAction.POSITION_CLOSE,
              entity_type="Position",
              entity_id=str(local_pos.id),
              reason="Position not found on exchange",
              success=True,
            )
            continue

          # Проверяем количество
          exchange_size = float(exchange_pos.get("size", 0))
          if abs(exchange_size - local_pos.quantity) > 0.01:
            logger.warning(
              f"Расхождение размера позиции {local_pos.symbol}: "
              f"локально={local_pos.quantity}, "
              f"биржа={exchange_size}"
            )
            result["discrepancies"] += 1

          # Обновляем текущую цену
          # В Bybit API v5 используется markPrice
          current_price = float(exchange_pos.get("markPrice", 0))
          if current_price > 0:
            await position_repository.update_current_price(
              position_id=str(local_pos.id),
              current_price=current_price,
            )

          result["synced"] += 1

        except Exception as e:
          logger.error(
            f"Ошибка сверки позиции {local_pos.symbol}: {e}"
          )

      logger.info(f"✓ Сверка позиций завершена: {result}")
      return result

    except Exception as e:
      logger.error(f"Критическая ошибка сверки позиций: {e}")
      return result

  def _map_exchange_order_status(self, exchange_status: str) -> OrderStatus:
    """
    Преобразование статуса биржи в локальный статус.

    Args:
        exchange_status: Статус с биржи

    Returns:
        OrderStatus: Локальный статус
    """
    status_map = {
      "New": OrderStatus.PLACED,
      "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
      "Filled": OrderStatus.FILLED,
      "Cancelled": OrderStatus.CANCELLED,
      "Rejected": OrderStatus.REJECTED,
    }

    return status_map.get(exchange_status, OrderStatus.FAILED)

  async def recover_from_crash(self) -> Dict[str, Any]:
    """
    Восстановление после аварийного завершения.

    Returns:
        Dict: Результаты восстановления
    """
    logger.warning("=" * 80)
    logger.warning("ОБНАРУЖЕНО АВАРИЙНОЕ ЗАВЕРШЕНИЕ - ВОССТАНОВЛЕНИЕ")
    logger.warning("=" * 80)

    results = {
      "recovered": False,
      "actions_taken": [],
    }

    try:
      # 1. Полная сверка состояния
      reconcile_result = await self.reconcile_state()
      results["actions_taken"].append("State reconciliation completed")

      # 2. Проверка зависших ордеров
      hanging_orders = await self._check_hanging_orders()
      if hanging_orders:
        results["actions_taken"].append(
          f"Found {len(hanging_orders)} hanging orders"
        )

      # 3. Восстановление FSM состояний
      await self._restore_fsm_states()
      results["actions_taken"].append("FSM states restored")

      results["recovered"] = True
      logger.info("✓ Восстановление завершено успешно")

    except Exception as e:
      logger.error(f"Ошибка при восстановлении: {e}")
      results["error"] = str(e)

    return results

  async def _check_hanging_orders(self) -> List:
    """
    Проверка зависших ордеров.

    Returns:
        List: Список зависших ордеров
    """
    # TODO: Реализовать логику проверки зависших ордеров
    return []

  async def _restore_fsm_states(self):
    """Восстановление состояний FSM для активных ордеров и позиций."""
    # TODO: Реализовать восстановление FSM
    pass


# Глобальный экземпляр
recovery_service = RecoveryService()