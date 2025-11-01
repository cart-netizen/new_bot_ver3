"""
Admin API Routes - Управление и диагностика.

НОВЫЕ ENDPOINTS:
- POST /admin/sync-order/{client_order_id} - синхронизация одного ордера
- POST /admin/fix-hanging-orders - исправление всех зависших ордеров
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
from pydantic import BaseModel

from backend.core.logger import get_logger
from backend.infrastructure.resilience.recovery_service import recovery_service
from backend.infrastructure.repositories.order_repository import order_repository
from backend.exchange.rest_client import rest_client

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


class SyncOrderResponse(BaseModel):
  """Ответ на запрос синхронизации ордера."""
  status: str
  order_id: str
  old_status: str
  new_status: str
  message: str
  position_created: bool = False


class FixHangingOrdersResponse(BaseModel):
  """Ответ на запрос исправления зависших ордеров."""
  status: str
  total_found: int
  fixed: int
  failed: int
  positions_created: int
  details: list


@router.post("/sync-order/{client_order_id}", response_model=SyncOrderResponse)
async def sync_order_with_exchange(client_order_id: str) -> SyncOrderResponse:
  """
  Синхронизировать статус одного ордера с биржей.

  Запрашивает актуальный статус ордера с биржи и обновляет локальное состояние.
  Если ордер исполнен - создаёт позицию.

  Args:
      client_order_id: Client Order ID ордера для синхронизации

  Returns:
      SyncOrderResponse: Результат синхронизации

  Raises:
      HTTPException: Если ордер не найден или произошла ошибка
  """
  try:
    logger.info(f"Запрос на синхронизацию ордера: {client_order_id}")

    # Получаем ордер из БД
    order = await order_repository.get_by_client_order_id(client_order_id)

    if not order:
      raise HTTPException(
        status_code=404,
        detail=f"Ордер {client_order_id} не найден в БД"
      )

    old_status = order.status.value

    # Запрашиваем актуальный статус с биржи
    exchange_order = await rest_client.get_order_by_id(
      order.symbol,
      order.exchange_order_id
    )

    if not exchange_order:
      raise HTTPException(
        status_code=404,
        detail=f"Ордер не найден на бирже (Exchange Order ID: {order.exchange_order_id})"
      )

    exchange_status = exchange_order.get("orderStatus", "")
    position_created = False

    # Синхронизируем статус
    if exchange_status == "Filled" and order.status.value != "Filled":
      await recovery_service._sync_filled_order(order, exchange_order)
      position_created = True
      message = f"Ордер синхронизирован: {old_status} → Filled, позиция создана"

    elif exchange_status == "Cancelled" and order.status.value != "Cancelled":
      await recovery_service._sync_cancelled_order(order, exchange_order)
      message = f"Ордер синхронизирован: {old_status} → Cancelled"

    else:
      message = f"Статус актуален: {order.status.value}"

    logger.info(f"✓ Синхронизация завершена: {message}")

    return SyncOrderResponse(
      status="success",
      order_id=client_order_id,
      old_status=old_status,
      new_status=order.status.value,
      message=message,
      position_created=position_created
    )

  except HTTPException:
    raise
  except Exception as e:
    logger.error(f"Ошибка синхронизации ордера {client_order_id}: {e}", exc_info=True)
    raise HTTPException(
      status_code=500,
      detail=f"Ошибка синхронизации: {str(e)}"
    )


@router.post("/fix-hanging-orders", response_model=FixHangingOrdersResponse)
async def fix_all_hanging_orders(background_tasks: BackgroundTasks) -> FixHangingOrdersResponse:
  """
  Проверить и исправить все зависшие ордера.

  Выполняет полный цикл:
  1. Проверка зависших ордеров (_check_hanging_orders)
  2. Автоматическое исправление (_fix_hanging_orders)

  Returns:
      FixHangingOrdersResponse: Статистика исправлений

  Raises:
      HTTPException: Если произошла ошибка
  """
  try:
    logger.info("Запрос на исправление всех зависших ордеров")

    # Шаг 1: Проверка зависших ордеров
    hanging_orders = await recovery_service._check_hanging_orders()

    if not hanging_orders:
      return FixHangingOrdersResponse(
        status="success",
        total_found=0,
        fixed=0,
        failed=0,
        positions_created=0,
        details=[]
      )

    # Шаг 2: Исправление
    fix_stats = await recovery_service._fix_hanging_orders(hanging_orders)

    # Формируем детали
    details = []
    for order in hanging_orders:
      details.append({
        "client_order_id": order["client_order_id"],
        "symbol": order["symbol"],
        "issue_type": order["issue"]["type"],
        "issue_reason": order["issue"]["reason"]
      })

    logger.info(
      f"✓ Исправление завершено: {fix_stats['fixed']}/{fix_stats['total']} успешно"
    )

    return FixHangingOrdersResponse(
      status="success",
      total_found=fix_stats["total"],
      fixed=fix_stats["fixed"],
      failed=fix_stats["failed"],
      positions_created=fix_stats["positions_created"],
      details=details
    )

  except Exception as e:
    logger.error(f"Ошибка исправления зависших ордеров: {e}", exc_info=True)
    raise HTTPException(
      status_code=500,
      detail=f"Ошибка исправления: {str(e)}"
    )


@router.get("/hanging-orders")
async def get_hanging_orders():
  """
  Получить список зависших ордеров без исправления.

  Только проверка, без изменений в БД.

  Returns:
      Dict: Список зависших ордеров
  """
  try:
    hanging_orders = await recovery_service._check_hanging_orders()

    return {
      "status": "success",
      "count": len(hanging_orders),
      "orders": hanging_orders
    }

  except Exception as e:
    logger.error(f"Ошибка получения зависших ордеров: {e}", exc_info=True)
    raise HTTPException(
      status_code=500,
      detail=f"Ошибка: {str(e)}"
    )


@router.post("/force-recovery")
async def force_full_recovery():
  """
  Принудительный запуск полного цикла восстановления.

  Эквивалент recover_from_crash(), но вызывается вручную.

  Returns:
      Dict: Результаты восстановления
  """
  try:
    logger.info("Запрос на принудительное восстановление")

    recovery_result = await recovery_service.recover_from_crash()

    return {
      "status": "success" if recovery_result["recovered"] else "failed",
      "result": recovery_result
    }

  except Exception as e:
    logger.error(f"Ошибка принудительного восстановления: {e}", exc_info=True)
    raise HTTPException(
      status_code=500,
      detail=f"Ошибка восстановления: {str(e)}"
    )