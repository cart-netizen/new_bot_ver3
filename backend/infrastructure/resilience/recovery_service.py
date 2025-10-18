"""
Recovery & State Sync Service - ПОЛНАЯ РЕАЛИЗАЦИЯ.

Восстановление состояния бота после сбоев и сверка с биржей.
Включает обнаружение зависших ордеров и восстановление FSM.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from core.logger import get_logger
from config import settings
from exchange.rest_client import rest_client
from infrastructure.repositories.order_repository import order_repository
from infrastructure.repositories.position_repository import position_repository
from infrastructure.repositories.audit_repository import audit_repository
from database.models import OrderStatus, PositionStatus, AuditAction, OrderSide
from domain.state_machines.order_fsm import OrderStateMachine
from domain.state_machines.position_fsm import PositionStateMachine
from domain.services.fsm_registry import fsm_registry

logger = get_logger(__name__)


class RecoveryService:
    """
    Сервис восстановления состояния.

    Основные функции:
    - Сверка локального состояния с биржей
    - Обнаружение зависших ордеров
    - Восстановление FSM после рестарта
    - Синхронизация расхождений
    """

    def __init__(self):
        """Инициализация сервиса восстановления."""
        # Параметры из конфигурации
        self.hanging_order_timeout_minutes = getattr(
            settings,
            'HANGING_ORDER_TIMEOUT_MINUTES',
            30
        )
        self.enable_auto_recovery = getattr(
            settings,
            'ENABLE_AUTO_RECOVERY',
            True
        )
        # ===== НОВЫЕ ПАРАМЕТРЫ =====
        self.auto_fix_hanging_orders = getattr(
            settings,
            'AUTO_FIX_HANGING_ORDERS',
            True
        )
        self.auto_create_positions = getattr(
            settings,
            'AUTO_CREATE_POSITIONS_FROM_FILLED',
            True
        )

        logger.info(
            f"Recovery Service инициализирован | "
            f"Timeout для зависших ордеров: {self.hanging_order_timeout_minutes} мин | "
            f"Авто-восстановление: {self.enable_auto_recovery} | "
            f"Авто-исправление зависших: {self.auto_fix_hanging_orders} | "
            f"Авто-создание позиций: {self.auto_create_positions}"
        )

    # ==================== ПУБЛИЧНЫЕ МЕТОДЫ ====================

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
            logger.error(f"Ошибка при сверке состояния: {e}", exc_info=True)
            results["errors"].append(str(e))

        return results

    async def recover_from_crash(self) -> Dict[str, Any]:
        """
        Полное восстановление после краша/рестарта.

        ОБНОВЛЕНО: Теперь включает автоматическое исправление зависших ордеров!

        Выполняет следующие шаги:
        1. Сверка состояния с биржей (reconcile_state)
        2. Проверка зависших ордеров (_check_hanging_orders)
        3. **НОВОЕ:** Автоматическое исправление зависших ордеров (_fix_hanging_orders)
        4. Восстановление FSM состояний (_restore_fsm_states)

        Returns:
            Dict: Результаты восстановления
        """
        logger.info("=" * 80)
        logger.info("НАЧАЛО ПРОЦЕССА ВОССТАНОВЛЕНИЯ ПОСЛЕ КРАША")
        logger.info("=" * 80)

        results = {
            "recovered": False,
            "actions_taken": [],
            "hanging_orders": [],
            "hanging_orders_fixed": {},  # НОВОЕ
            "fsm_restored": {"orders": 0, "positions": 0},
            "error": None
        }

        try:
            # ==================== 1. СВЕРКА СОСТОЯНИЯ ====================
            logger.info("Полная сверка состояния с биржей...")
            reconcile_result = await self.reconcile_state()
            results["actions_taken"].append("State reconciliation completed")
            logger.info("✓ Шаг 1/4: Сверка состояния завершена")

            # ==================== 2. ПРОВЕРКА ЗАВИСШИХ ОРДЕРОВ ====================
            hanging_orders = await self._check_hanging_orders()
            results["hanging_orders"] = hanging_orders

            if hanging_orders:
                results["actions_taken"].append(
                    f"Found {len(hanging_orders)} hanging orders"
                )
                logger.warning(f"⚠ Обнаружено {len(hanging_orders)} зависших ордеров")
            else:
                logger.info("✓ Зависших ордеров не обнаружено")

            logger.info("✓ Шаг 2/4: Проверка зависших ордеров завершена")

            # ==================== 3. АВТОМАТИЧЕСКОЕ ИСПРАВЛЕНИЕ (НОВОЕ!) ====================
            if hanging_orders:
                if self.auto_fix_hanging_orders:
                    logger.info("Запуск автоматического исправления зависших ордеров...")
                    fix_stats = await self._fix_hanging_orders(hanging_orders)
                    results["hanging_orders_fixed"] = fix_stats
                    results["actions_taken"].append(
                        f"Fixed {fix_stats['fixed']} hanging orders, "
                        f"created {fix_stats['positions_created']} positions"
                    )

                    if fix_stats["fixed"] > 0:
                        logger.info(
                            f"✓ Исправлено {fix_stats['fixed']}/{fix_stats['total']} "
                            f"зависших ордеров"
                        )
                    if fix_stats["failed"] > 0:
                        logger.warning(
                            f"⚠ Не удалось исправить {fix_stats['failed']} ордеров"
                        )
                else:
                    logger.warning(
                        f"⚠ Автоматическое исправление отключено (AUTO_FIX_HANGING_ORDERS=false). "
                        f"Найдено {len(hanging_orders)} зависших ордеров - требуется ручное исправление!"
                    )
                    results["actions_taken"].append(
                        f"Auto-fix disabled: {len(hanging_orders)} hanging orders require manual attention"
                    )
            else:
                logger.info("✓ Исправление не требуется (нет зависших ордеров)")

            logger.info("✓ Шаг 3/4: Исправление зависших ордеров завершено")

            # ==================== 4. ВОССТАНОВЛЕНИЕ FSM ====================
            fsm_result = await self._restore_fsm_states()
            results["fsm_restored"] = fsm_result
            results["actions_taken"].append(
                f"FSM states restored: {fsm_result['orders']} orders, "
                f"{fsm_result['positions']} positions"
            )
            logger.info("✓ Шаг 4/4: FSM состояния восстановлены")

            results["recovered"] = True

            logger.info("=" * 80)
            logger.info("✓ ВОССТАНОВЛЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Ошибка при восстановлении: {e}", exc_info=True)
            results["error"] = str(e)
            results["recovered"] = False

        return results

    async def _fix_hanging_orders(self, hanging_orders: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Автоматическое исправление зависших ордеров.

        Синхронизирует статусы ордеров с реальным состоянием на бирже.
        """
        logger.info("=" * 80)
        logger.info("НАЧАЛО АВТОМАТИЧЕСКОГО ИСПРАВЛЕНИЯ ЗАВИСШИХ ОРДЕРОВ")
        logger.info("=" * 80)

        stats = {
            "total": len(hanging_orders),
            "fixed": 0,
            "failed": 0,
            "positions_created": 0,
            "orders_cancelled": 0
        }

        if not hanging_orders:
            logger.info("✓ Нет зависших ордеров для исправления")
            return stats

        logger.info(f"Обработка {len(hanging_orders)} зависших ордеров...")

        for hanging_order in hanging_orders:
            try:
                issue_type = hanging_order["issue"]["type"]
                client_order_id = hanging_order["client_order_id"]

                logger.info(f"Исправление ордера {client_order_id} (тип: {issue_type})")

                # Получаем ордер из БД
                order = await order_repository.get_by_client_order_id(client_order_id)

                if not order:
                    logger.error(f"Ордер {client_order_id} не найден в БД")
                    stats["failed"] += 1
                    continue

                # Обработка по типу проблемы
                if issue_type == "status_mismatch":
                    exchange_data = hanging_order["issue"].get("exchange_data", {})

                    if exchange_data.get("status") == "Filled":
                        await self._sync_filled_order(order, exchange_data)
                        stats["fixed"] += 1
                        stats["positions_created"] += 1
                        logger.info(f"✓ {client_order_id}: Placed → Filled")

                    elif exchange_data.get("status") == "Cancelled":
                        await self._sync_cancelled_order(order, exchange_data)
                        stats["fixed"] += 1
                        stats["orders_cancelled"] += 1
                        logger.info(f"✓ {client_order_id}: Placed → Cancelled")

                elif issue_type == "missing_from_exchange":
                    await self._handle_missing_order(order)
                    stats["fixed"] += 1
                    stats["orders_cancelled"] += 1
                    logger.info(f"✓ {client_order_id}: отменён (не найден)")

                elif issue_type == "timeout_in_status":
                    await self._check_and_fix_timeout_order(order)
                    stats["fixed"] += 1
                    logger.info(f"✓ {client_order_id}: проверен")

            except Exception as e:
                logger.error(f"Ошибка исправления {hanging_order.get('client_order_id')}: {e}", exc_info=True)
                stats["failed"] += 1

        logger.info("=" * 80)
        logger.info(f"ИСПРАВЛЕНИЕ ЗАВЕРШЕНО: {stats['fixed']}/{stats['total']}")
        logger.info("=" * 80)

        return stats

    # ==================== ПРОВЕРКА ЗАВИСШИХ ОРДЕРОВ ====================

    async def _check_hanging_orders(self) -> List[Dict[str, Any]]:
        """
        Проверка зависших ордеров.

        Критерии "зависших" ордеров:
        1. Локально активен, но на бирже в финальном статусе
        2. Локально активен, но не найден на бирже
        3. Слишком долго в промежуточном статусе (> HANGING_ORDER_TIMEOUT)

        Returns:
            List[Dict]: Список зависших ордеров с деталями проблемы
        """
        logger.info("=" * 80)
        logger.info("НАЧАЛО ПРОВЕРКИ ЗАВИСШИХ ОРДЕРОВ")
        logger.info("=" * 80)

        hanging_orders = []

        try:
            # 1. Получаем все активные ордера из БД
            local_active_orders = await order_repository.get_active_orders()
            logger.info(f"Найдено {len(local_active_orders)} активных ордеров в БД")

            if not local_active_orders:
                logger.info("✓ Активных ордеров не найдено, проверка не требуется")
                return hanging_orders

            # 2. Получаем активные ордера с биржи для всех уникальных символов
            symbols = list(set(order.symbol for order in local_active_orders))
            logger.info(f"Проверяем {len(symbols)} уникальных символов: {symbols}")

            exchange_orders_map = {}

            for symbol in symbols:
                try:
                    response = await rest_client.get_open_orders(symbol=symbol)
                    exchange_orders_list = response.get("result", {}).get("list", [])

                    # Создаем мапу для быстрого поиска по orderLinkId
                    for ex_order in exchange_orders_list:
                        order_link_id = ex_order.get("orderLinkId")
                        if order_link_id:
                            exchange_orders_map[order_link_id] = ex_order

                    logger.debug(
                        f"Получено {len(exchange_orders_list)} активных ордеров "
                        f"с биржи для {symbol}"
                    )

                except Exception as e:
                    logger.error(
                        f"Ошибка получения ордеров с биржи для {symbol}: {e}",
                        exc_info=True
                    )
                    # Пропускаем этот символ, но продолжаем проверку остальных
                    continue

            logger.info(f"Всего активных ордеров на бирже: {len(exchange_orders_map)}")

            # 3. Проверяем каждый локальный ордер
            current_time = datetime.utcnow()
            timeout_threshold = timedelta(minutes=self.hanging_order_timeout_minutes)

            for local_order in local_active_orders:
                issue_detected = None

                # ========================================
                # ПРОВЕРКА 1: Ордер не найден в активных на бирже
                # ========================================

                if local_order.client_order_id not in exchange_orders_map:
                    logger.debug(
                        f"Ордер {local_order.client_order_id} не найден в активных, "
                        f"ищем в истории..."
                    )

                    # Дополнительная проверка: возможно ордер в истории
                    try:
                        order_info = await rest_client.get_order_info(
                            symbol=local_order.symbol,
                            order_link_id=local_order.client_order_id
                        )

                        if order_info:
                            # Ордер найден в истории - проверяем статус
                            exchange_status_str = order_info.get("orderStatus")
                            exchange_status = self._map_exchange_status(exchange_status_str)

                            logger.debug(
                                f"Ордер {local_order.client_order_id} найден в истории | "
                                f"Статус на бирже: {exchange_status_str}"
                            )

                            # Если на бирже завершен, а локально активен - это проблема
                            if exchange_status in [
                                OrderStatus.FILLED,
                                OrderStatus.CANCELLED,
                                OrderStatus.REJECTED,
                                OrderStatus.FAILED
                            ]:
                                issue_detected = {
                                    "type": "status_mismatch",
                                    "reason": (
                                        f"Ордер локально активен ({local_order.status.value}), "
                                        f"но на бирже завершен ({exchange_status.value})"
                                    ),
                                    "local_status": local_order.status.value,
                                    "exchange_status": exchange_status.value,
                                    "exchange_data": {
                                        "orderId": order_info.get("orderId"),
                                        "cumExecQty": order_info.get("cumExecQty"),
                                        "avgPrice": order_info.get("avgPrice"),
                                        "updatedTime": order_info.get("updatedTime")
                                    }
                                }
                        else:
                            # Ордер вообще не найден на бирже (ни активный, ни в истории)
                            logger.warning(
                                f"Ордер {local_order.client_order_id} не найден на бирже!"
                            )

                            issue_detected = {
                                "type": "not_found_on_exchange",
                                "reason": "Ордер не найден на бирже (ни в активных, ни в истории)",
                                "possible_causes": [
                                    "Отменен вручную через UI биржи",
                                    "Системный сбой при размещении",
                                    "Ордер был отклонен биржей без уведомления"
                                ],
                                "local_status": local_order.status.value
                            }

                    except Exception as e:
                        logger.warning(
                            f"Не удалось проверить ордер {local_order.client_order_id} "
                            f"в истории: {e}"
                        )

                        issue_detected = {
                            "type": "verification_failed",
                            "reason": f"Не удалось проверить статус на бирже: {str(e)}",
                            "local_status": local_order.status.value
                        }

                # ========================================
                # ПРОВЕРКА 2: Ордер слишком долго в промежуточном статусе
                # ========================================

                if not issue_detected and local_order.status in [
                    OrderStatus.PENDING,
                    OrderStatus.PLACED
                ]:
                    time_in_status = current_time - local_order.updated_at

                    if time_in_status > timeout_threshold:
                        minutes_stuck = time_in_status.total_seconds() / 60

                        logger.warning(
                            f"Ордер {local_order.client_order_id} зависший по таймауту | "
                            f"Время в статусе {local_order.status.value}: {minutes_stuck:.1f} мин"
                        )

                        issue_detected = {
                            "type": "timeout_in_status",
                            "reason": (
                                f"Ордер {minutes_stuck:.1f} минут в статусе "
                                f"{local_order.status.value} (порог: {self.hanging_order_timeout_minutes} мин)"
                            ),
                            "minutes_stuck": round(minutes_stuck, 2),
                            "current_status": local_order.status.value,
                            "threshold_minutes": self.hanging_order_timeout_minutes,
                            "updated_at": local_order.updated_at.isoformat()
                        }

                # ========================================
                # ЕСЛИ ОБНАРУЖЕНА ПРОБЛЕМА - ДОБАВЛЯЕМ В СПИСОК
                # ========================================

                if issue_detected:
                    hanging_order_info = {
                        "order_id": str(local_order.id),
                        "client_order_id": local_order.client_order_id,
                        "exchange_order_id": local_order.exchange_order_id,
                        "symbol": local_order.symbol,
                        "side": local_order.side.value,
                        "order_type": local_order.order_type.value,
                        "quantity": local_order.quantity,
                        "price": local_order.price,
                        "local_status": local_order.status.value,
                        "created_at": local_order.created_at.isoformat(),
                        "updated_at": local_order.updated_at.isoformat(),
                        "issue": issue_detected
                    }

                    hanging_orders.append(hanging_order_info)

                    logger.error(
                        f"⚠ ЗАВИСШИЙ ОРДЕР ОБНАРУЖЕН ⚠\n"
                        f"  Client Order ID: {local_order.client_order_id}\n"
                        f"  Символ: {local_order.symbol}\n"
                        f"  Локальный статус: {local_order.status.value}\n"
                        f"  Тип проблемы: {issue_detected['type']}\n"
                        f"  Причина: {issue_detected['reason']}"
                    )

                    # Логируем в audit для последующего анализа
                    await audit_repository.log(
                        action=AuditAction.ORDER_MODIFY,
                        entity_type="Order",
                        entity_id=local_order.client_order_id,
                        old_value={"status": local_order.status.value},
                        new_value={"issue": issue_detected},
                        reason="Hanging order detected during recovery check",
                        success=True,
                        context=hanging_order_info
                    )

            # ========================================
            # ФИНАЛЬНОЕ ЛОГИРОВАНИЕ
            # ========================================

            logger.info("=" * 80)
            if hanging_orders:
                logger.warning(
                    f"⚠ ПРОВЕРКА ЗАВЕРШЕНА: НАЙДЕНО {len(hanging_orders)} ЗАВИСШИХ ОРДЕРОВ"
                )

                # Группируем по типам проблем
                issues_by_type = {}
                for order in hanging_orders:
                    issue_type = order["issue"]["type"]
                    issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1

                logger.warning("Распределение по типам проблем:")
                for issue_type, count in issues_by_type.items():
                    logger.warning(f"  - {issue_type}: {count}")
            else:
                logger.info("✓ ПРОВЕРКА ЗАВЕРШЕНА: ЗАВИСШИХ ОРДЕРОВ НЕ ОБНАРУЖЕНО")

            logger.info("=" * 80)

            return hanging_orders

        except Exception as e:
            logger.error(
                f"Критическая ошибка при проверке зависших ордеров: {e}",
                exc_info=True
            )
            return hanging_orders

    async def _sync_filled_order(self, order, exchange_data: dict):
        """Синхронизировать исполненный ордер."""
        from database.models import Order

        order.status = OrderStatus.FILLED
        order.filled_quantity = float(exchange_data.get("cumExecQty", 0))
        order.average_fill_price = float(exchange_data.get("avgPrice", 0))

        filled_timestamp = int(exchange_data.get("updatedTime", 0))
        if filled_timestamp > 0:
            order.filled_at = datetime.fromtimestamp(filled_timestamp / 1000)
        else:
            order.filled_at = datetime.now()

        await order_repository.update(order)

        logger.info(
            f"Ордер {order.client_order_id} обновлён: "
            f"qty={order.filled_quantity}, price={order.average_fill_price:.8f}"
        )

        # Создаём позицию если её нет
        if self.auto_create_positions:
            existing = await position_repository.get_by_symbol_and_status(
                order.symbol,
                [PositionStatus.OPENING, PositionStatus.OPEN]
            )

            if not existing:
                await self._create_position_from_filled_order(order)
            else:
                logger.warning(f"Позиция по {order.symbol} уже существует")

        # Audit log
        await audit_repository.log(
            action=AuditAction.ORDER_MODIFY,
            entity_type="Order",
            entity_id=order.client_order_id,
            old_value={"status": "Placed"},
            new_value={
                "status": "Filled",
                "filled_quantity": order.filled_quantity,
                "average_fill_price": order.average_fill_price
            },
            reason="Auto-sync hanging order",
            success=True
        )

    async def _create_position_from_filled_order(self, order):
        """Создать позицию из исполненного ордера."""
        from database.models import Position

        position = Position(
            symbol=order.symbol,
            side=order.side,
            status=PositionStatus.OPEN,
            quantity=order.filled_quantity,
            entry_price=order.average_fill_price,
            current_price=order.average_fill_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            opened_at=order.filled_at or datetime.now(),
            entry_reason=f"Recovery: {order.client_order_id}",
            entry_signal=order.signal_data
        )

        saved_position = await position_repository.create(position)

        order.position_id = saved_position.id
        await order_repository.update(order)

        logger.info(
            f"✓ Позиция создана: {order.side.value} "
            f"{order.filled_quantity} @ {order.average_fill_price:.8f}"
        )

        await audit_repository.log(
            action=AuditAction.POSITION_OPEN,
            entity_type="Position",
            entity_id=str(saved_position.id),
            new_value={
                "symbol": position.symbol,
                "side": position.side.value,
                "quantity": position.quantity
            },
            reason="Auto-created from hanging order",
            success=True
        )

        return saved_position

    async def _sync_cancelled_order(self, order, exchange_data: dict):
        """Синхронизировать отменённый ордер."""
        order.status = OrderStatus.CANCELLED

        cancelled_timestamp = int(exchange_data.get("updatedTime", 0))
        if cancelled_timestamp > 0:
            order.cancelled_at = datetime.fromtimestamp(cancelled_timestamp / 1000)
        else:
            order.cancelled_at = datetime.now()

        await order_repository.update(order)
        logger.info(f"Ордер {order.client_order_id} синхронизирован: Cancelled")

        await audit_repository.log(
            action=AuditAction.ORDER_MODIFY,
            entity_type="Order",
            entity_id=order.client_order_id,
            old_value={"status": "Placed"},
            new_value={"status": "Cancelled"},
            reason="Auto-sync cancelled order",
            success=True
        )

    async def _handle_missing_order(self, order):
        """Обработать ордер, не найденный на бирже."""
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now()

        if not order.metadata_json:
            order.metadata_json = {}
        order.metadata_json["cancellation_reason"] = "missing_from_exchange"

        await order_repository.update(order)
        logger.warning(f"Ордер {order.client_order_id} отменён (не найден на бирже)")

        await audit_repository.log(
            action=AuditAction.ORDER_MODIFY,
            entity_type="Order",
            entity_id=order.client_order_id,
            new_value={"status": "Cancelled", "reason": "missing"},
            reason="Order not found on exchange",
            success=True
        )

    async def _check_and_fix_timeout_order(self, order):
        """Проверить и исправить ордер с таймаутом."""
        try:
            exchange_order = await rest_client.get_order_by_id(
                order.symbol,
                order.exchange_order_id
            )

            if not exchange_order:
                await self._handle_missing_order(order)
                return

            status = exchange_order.get("orderStatus", "")

            if status == "Filled":
                await self._sync_filled_order(order, exchange_order)
            elif status in ["Cancelled", "Rejected"]:
                await self._sync_cancelled_order(order, exchange_order)
            else:
                logger.info(f"Ордер {order.client_order_id} всё ещё активен")

        except Exception as e:
            logger.error(f"Ошибка проверки таймаут ордера: {e}", exc_info=True)

    # ==================== ВОССТАНОВЛЕНИЕ FSM ====================

    async def _restore_fsm_states(self) -> Dict[str, int]:
        """
        Восстановление состояний FSM для активных ордеров и позиций.

        После рестарта бота FSM существуют только в памяти и требуют
        восстановления из текущих состояний в БД для корректной валидации
        будущих переходов.

        Восстанавливаем:
        1. OrderStateMachine для всех активных ордеров
        2. PositionStateMachine для всех активных позиций

        Returns:
            Dict[str, int]: Количество восстановленных FSM по типам
        """
        logger.info("=" * 80)
        logger.info("НАЧАЛО ВОССТАНОВЛЕНИЯ FSM СОСТОЯНИЙ")
        logger.info("=" * 80)

        restored_orders = 0
        restored_positions = 0

        try:
            # ============================================================
            # 1. ВОССТАНОВЛЕНИЕ FSM ДЛЯ ОРДЕРОВ
            # ============================================================

            logger.info("Шаг 1/2: Восстановление FSM ордеров...")

            # Получаем все активные ордера из БД
            active_orders = await order_repository.get_active_orders()
            logger.info(f"Найдено {len(active_orders)} активных ордеров для восстановления")

            for order in active_orders:
                try:
                    # Создаем FSM с текущим состоянием из БД
                    order_fsm = OrderStateMachine(
                        order_id=order.client_order_id,
                        initial_state=order.status
                    )

                    # Восстанавливаем историю переходов если она сохранена в metadata
                    if order.metadata_json and "transition_history" in order.metadata_json:
                        order_fsm.transition_history = order.metadata_json["transition_history"]
                        logger.debug(
                            f"Восстановлена история переходов для ордера {order.client_order_id}: "
                            f"{len(order_fsm.transition_history)} переходов"
                        )

                    # Регистрируем FSM в глобальном реестре
                    fsm_registry.register_order_fsm(order.client_order_id, order_fsm)

                    restored_orders += 1

                    logger.debug(
                        f"✓ FSM восстановлена для ордера {order.client_order_id} | "
                        f"Символ: {order.symbol} | "
                        f"Статус: {order.status.value} | "
                        f"Доступные переходы: {order_fsm.get_available_transitions()}"
                    )

                except Exception as e:
                    logger.error(
                        f"Ошибка восстановления FSM для ордера {order.client_order_id}: {e}",
                        exc_info=True
                    )
                    # Продолжаем с остальными ордерами
                    continue

            logger.info(
                f"✓ FSM ордеров восстановлены: {restored_orders}/{len(active_orders)}"
            )

            # ============================================================
            # 2. ВОССТАНОВЛЕНИЕ FSM ДЛЯ ПОЗИЦИЙ
            # ============================================================

            logger.info("Шаг 2/2: Восстановление FSM позиций...")

            # Получаем все активные позиции из БД
            active_positions = await position_repository.get_active_positions()
            logger.info(f"Найдено {len(active_positions)} активных позиций для восстановления")

            for position in active_positions:
                try:
                    # Создаем FSM с текущим состоянием из БД
                    position_fsm = PositionStateMachine(
                        position_id=str(position.id),
                        initial_state=position.status
                    )

                    # Восстанавливаем историю переходов если сохранена
                    if position.metadata_json and "transition_history" in position.metadata_json:
                        position_fsm.transition_history = position.metadata_json["transition_history"]
                        logger.debug(
                            f"Восстановлена история переходов для позиции {position.id}: "
                            f"{len(position_fsm.transition_history)} переходов"
                        )

                    # Регистрируем FSM в глобальном реестре
                    fsm_registry.register_position_fsm(str(position.id), position_fsm)

                    restored_positions += 1

                    logger.debug(
                        f"✓ FSM восстановлена для позиции {position.id} | "
                        f"Символ: {position.symbol} | "
                        f"Сторона: {position.side.value} | "
                        f"Статус: {position.status.value} | "
                        f"Активна: {position_fsm.is_active()}"
                    )

                except Exception as e:
                    logger.error(
                        f"Ошибка восстановления FSM для позиции {position.id}: {e}",
                        exc_info=True
                    )
                    continue

            logger.info(
                f"✓ FSM позиций восстановлены: {restored_positions}/{len(active_positions)}"
            )

            # ============================================================
            # 3. ФИНАЛЬНОЕ ЛОГИРОВАНИЕ И АУДИТ
            # ============================================================

            result = {
                "orders": restored_orders,
                "positions": restored_positions
            }

            logger.info("=" * 80)
            logger.info("ВОССТАНОВЛЕНИЕ FSM ЗАВЕРШЕНО:")
            logger.info(f"  Ордеров восстановлено: {restored_orders}/{len(active_orders)}")
            logger.info(f"  Позиций восстановлено: {restored_positions}/{len(active_positions)}")
            logger.info("=" * 80)

            # Получаем статистику реестра
            registry_stats = fsm_registry.get_stats()
            logger.info("Статистика FSM Registry:")
            logger.info(f"  Всего FSM ордеров: {registry_stats['total_order_fsms']}")
            logger.info(f"  Всего FSM позиций: {registry_stats['total_position_fsms']}")
            logger.info(f"  Ордера по статусам: {registry_stats['order_fsms_by_status']}")
            logger.info(f"  Позиции по статусам: {registry_stats['position_fsms_by_status']}")

            # Логируем в audit
            await audit_repository.log(
                action=AuditAction.CONFIG_CHANGE,
                entity_type="System",
                entity_id="fsm_recovery",
                new_value={
                    "orders_restored": restored_orders,
                    "positions_restored": restored_positions,
                    "total_active_orders": len(active_orders),
                    "total_active_positions": len(active_positions),
                    "registry_stats": registry_stats
                },
                reason="FSM state restoration after system restart",
                success=True
            )

            return result

        except Exception as e:
            logger.error(
                f"Критическая ошибка при восстановлении FSM: {e}",
                exc_info=True
            )

            # Логируем ошибку в audit
            await audit_repository.log(
                action=AuditAction.CONFIG_CHANGE,
                entity_type="System",
                entity_id="fsm_recovery",
                new_value={
                    "error": str(e),
                    "orders_restored": restored_orders,
                    "positions_restored": restored_positions
                },
                reason="FSM restoration failed",
                success=False
            )

            raise

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================

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

            if not local_orders:
                logger.info("Активных ордеров нет, сверка не требуется")
                return result

            # Получаем ордера с биржи
            symbols = list(set(order.symbol for order in local_orders))

            for symbol in symbols:
                try:
                    response = await rest_client.get_open_orders(symbol=symbol)
                    exchange_orders = response.get("result", {}).get("list", [])

                    # Создаем мапу для быстрого поиска
                    exchange_map = {
                        order.get("orderLinkId"): order
                        for order in exchange_orders
                        if order.get("orderLinkId")
                    }

                    logger.debug(
                        f"Получено {len(exchange_orders)} ордеров с биржи для {symbol}"
                    )

                    # Сверяем каждый локальный ордер
                    for local_order in [o for o in local_orders if o.symbol == symbol]:
                        exchange_order = exchange_map.get(local_order.client_order_id)

                        if exchange_order:
                            # Ордер найден - проверяем статус
                            exchange_status_str = exchange_order.get("orderStatus")
                            exchange_status = self._map_exchange_status(exchange_status_str)

                            if local_order.status != exchange_status:
                                # Расхождение обнаружено
                                result["discrepancies"] += 1

                                logger.warning(
                                    f"Расхождение статуса ордера {local_order.client_order_id}: "
                                    f"локально={local_order.status.value}, "
                                    f"биржа={exchange_status.value}"
                                )

                                # Обновляем локальный статус
                                exchange_order_id = exchange_order.get("orderId")
                                cum_exec_qty = exchange_order.get("cumExecQty", "0")
                                avg_price = exchange_order.get("avgPrice", "0")

                                try:
                                    filled_qty = float(cum_exec_qty) if cum_exec_qty and cum_exec_qty != "" else 0.0
                                except (ValueError, TypeError):
                                    logger.warning(f"Некорректное значение cumExecQty: {cum_exec_qty}")
                                    filled_qty = 0.0

                                try:
                                    avg_price_value = float(avg_price) if avg_price and avg_price != "" else 0.0
                                except (ValueError, TypeError):
                                    logger.warning(f"Некорректное значение avgPrice: {avg_price}")
                                    avg_price_value = 0.0

                                await order_repository.update_status(
                                    client_order_id=local_order.client_order_id,
                                    new_status=exchange_status,
                                    exchange_order_id=exchange_order_id,
                                    filled_quantity=filled_qty,
                                    average_fill_price=avg_price_value if avg_price_value > 0 else None,
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
                    logger.error(f"Ошибка сверки ордеров для {symbol}: {e}", exc_info=True)
                    continue

            logger.info(f"✓ Сверка ордеров завершена: {result}")
            return result

        except Exception as e:
            logger.error(f"Критическая ошибка сверки ордеров: {e}", exc_info=True)
            return result

    # async def _reconcile_positions(self) -> Dict[str, int]:
    #     """
    #     Сверка позиций с биржей.
    #
    #     Returns:
    #         Dict: Результаты сверки позиций
    #     """
    #     logger.info("Сверка позиций с биржей...")
    #
    #     result = {
    #         "synced": 0,
    #         "discrepancies": 0,
    #     }
    #
    #     try:
    #         # Получаем активные позиции из БД
    #         local_positions = await position_repository.get_active_positions()
    #         logger.info(f"Найдено {len(local_positions)} активных позиций в БД")
    #
    #         # Получаем позиции с биржи
    #         try:
    #             exchange_response = await rest_client.get_positions()
    #             exchange_positions_list = exchange_response.get("result", {}).get("list", [])
    #             logger.info(f"Получено {len(exchange_positions_list)} позиций с биржи")
    #
    #         except Exception as e:
    #             logger.error(f"Ошибка получения позиций с биржи: {e}", exc_info=True)
    #             return result
    #
    #         # Создаем мапу позиций с биржи (только активные с size > 0)
    #         exchange_map = {}
    #         for pos in exchange_positions_list:
    #             symbol = pos.get("symbol")
    #             size = float(pos.get("size", 0))
    #
    #             if symbol and size > 0:
    #                 exchange_map[symbol] = pos
    #
    #         # Сверяем каждую локальную позицию
    #         for local_position in local_positions:
    #             exchange_position = exchange_map.get(local_position.symbol)
    #
    #             if exchange_position:
    #                 # Позиция найдена на бирже
    #                 exchange_size = float(exchange_position.get("size", 0))
    #                 exchange_side = exchange_position.get("side", "")
    #
    #                 # Проверяем расхождения
    #                 if abs(local_position.quantity - exchange_size) > 0.001:
    #                     result["discrepancies"] += 1
    #
    #                     logger.warning(
    #                         f"Расхождение количества позиции {local_position.symbol}: "
    #                         f"локально={local_position.quantity}, "
    #                         f"биржа={exchange_size}"
    #                     )
    #
    #             result["synced"] += 1
    #
    #         logger.info(f"✓ Сверка позиций завершена: {result}")
    #         return result
    #
    #     except Exception as e:
    #         logger.error(f"Критическая ошибка сверки позиций: {e}", exc_info=True)
    #         return result

    async def _reconcile_positions(self) -> Dict[str, int]:
        """
        Сверка позиций с биржей.

        ЛОГИКА:
        1. Получаем активные позиции из БД (статусы: OPENING, OPEN)
        2. Получаем позиции с биржи (size > 0)
        3. Сверяем три сценария:
           - Позиция ЕСТЬ в БД И на бирже → проверяем количество
           - Позиция ЕСТЬ в БД, но НЕТ на бирже → закрываем (призрак)
           - Позиция НЕТ в БД, но ЕСТЬ на бирже → импортируем

        Returns:
            Dict: Результаты сверки позиций
        """
        logger.info("Сверка позиций с биржей...")

        result = {
            "synced": 0,
            "discrepancies": 0,
            "ghost_positions_closed": 0,
            "positions_imported": 0,
        }

        try:
            # ==========================================
            # ШАГ 1: ПОЛУЧЕНИЕ ЛОКАЛЬНЫХ ПОЗИЦИЙ
            # ==========================================
            local_positions = await position_repository.get_active_positions()
            logger.info(f"Найдено {len(local_positions)} активных позиций в БД")

            # ==========================================
            # ШАГ 2: ПОЛУЧЕНИЕ ПОЗИЦИЙ С БИРЖИ
            # ==========================================
            try:
                exchange_response = await rest_client.get_positions()
                exchange_positions_list = exchange_response.get("result", {}).get("list", [])
                logger.info(f"Получено {len(exchange_positions_list)} позиций с биржи")

            except Exception as e:
                logger.error(f"Ошибка получения позиций с биржи: {e}", exc_info=True)

                # Если не удалось получить позиции с биржи, но есть локальные позиции
                # считаем это критической ситуацией
                if local_positions:
                    logger.error(
                        f"❌ КРИТИЧНО: Не удалось получить позиции с биржи, "
                        f"но в БД есть {len(local_positions)} активных позиций!"
                    )

                return result

            # ==========================================
            # ШАГ 3: СОЗДАНИЕ МАПЫ ПОЗИЦИЙ С БИРЖИ
            # ==========================================
            # Только активные позиции с size > 0
            exchange_map = {}
            for pos in exchange_positions_list:
                symbol = pos.get("symbol")
                size = float(pos.get("size", 0))

                if symbol and size > 0:
                    exchange_map[symbol] = pos

            logger.debug(f"Создана мапа позиций с биржи: {len(exchange_map)} активных")

            # ==========================================
            # ШАГ 4: СВЕРКА ЛОКАЛЬНЫХ ПОЗИЦИЙ
            # ==========================================
            local_symbols = set()  # Множество символов в БД

            for local_position in local_positions:
                symbol = local_position.symbol
                position_id = str(local_position.id)
                local_symbols.add(symbol)  # Запоминаем символ

                exchange_position = exchange_map.get(symbol)

                if exchange_position:
                    # ==========================================
                    # СЦЕНАРИЙ 1: ПОЗИЦИЯ ЕСТЬ НА БИРЖЕ
                    # ==========================================
                    exchange_size = float(exchange_position.get("size", 0))
                    exchange_side = exchange_position.get("side", "")

                    # Проверяем расхождения в количестве
                    if abs(local_position.quantity - exchange_size) > 0.001:
                        result["discrepancies"] += 1

                        logger.warning(
                            f"⚠ Расхождение количества позиции {symbol}: "
                            f"локально={local_position.quantity:.8f}, "
                            f"биржа={exchange_size:.8f}"
                        )

                        # ОПЦИОНАЛЬНО: Обновляем количество в БД
                        # await position_repository.update(
                        #     position_id=position_id,
                        #     quantity=exchange_size
                        # )

                    result["synced"] += 1
                    logger.debug(f"✓ Позиция {symbol} синхронизирована")

                else:
                    # ==========================================
                    # СЦЕНАРИЙ 2: ПОЗИЦИЯ НЕТ НА БИРЖЕ (ПРИЗРАК)
                    # ==========================================
                    result["discrepancies"] += 1
                    result["ghost_positions_closed"] += 1

                    logger.error(
                        f"❌ ПРИЗРАЧНАЯ ПОЗИЦИЯ ОБНАРУЖЕНА ❌\n"
                        f"  Symbol: {symbol}\n"
                        f"  Position ID: {position_id}\n"
                        f"  Статус в БД: {local_position.status.value}\n"
                        f"  Quantity: {local_position.quantity}\n"
                        f"  Entry Price: {local_position.entry_price}\n"
                        f"  Opened At: {local_position.opened_at}\n"
                        f"  ПРОБЛЕМА: Позиция есть в БД, но НЕТ на бирже!"
                    )

                    # ==========================================
                    # АВТОМАТИЧЕСКОЕ ЗАКРЫТИЕ ПРИЗРАЧНОЙ ПОЗИЦИИ
                    # ==========================================
                    try:
                        # Получаем FSM для позиции
                        position_fsm = fsm_registry.get_position_fsm(position_id)

                        if not position_fsm:
                            # Создаем FSM если не существует
                            position_fsm = PositionStateMachine(
                                position_id=position_id,
                                initial_state=local_position.status
                            )
                            fsm_registry.register_position_fsm(position_id, position_fsm)

                        # Закрываем позицию в зависимости от статуса
                        if local_position.status == PositionStatus.OPENING:
                            # OPENING -> CLOSED (используем abort)
                            position_fsm.abort()  # type: ignore[attr-defined]

                            await position_repository.update_status(
                                position_id=position_id,
                                new_status=PositionStatus.CLOSED,
                                exit_reason="Ghost position aborted by RecoveryService (never opened on exchange)"
                            )

                            logger.info(
                                f"✓ Призрачная позиция {symbol} прервана: {position_id}"
                            )

                        elif local_position.status == PositionStatus.OPEN:
                            # ✅ ИСПРАВЛЕНО: Прямой переход OPEN -> CLOSING -> CLOSED
                            # Проверяем доступность триггера start_close
                            if hasattr(position_fsm, 'start_close'):
                                # OPEN -> CLOSING
                                position_fsm.start_close()  # type: ignore[attr-defined]

                                await position_repository.update_status(
                                    position_id=position_id,
                                    new_status=PositionStatus.CLOSING
                                )

                                # CLOSING -> CLOSED
                                position_fsm.confirm_close()  # type: ignore[attr-defined]

                                await position_repository.update_status(
                                    position_id=position_id,
                                    new_status=PositionStatus.CLOSED,
                                    exit_price=local_position.current_price or local_position.entry_price,
                                    exit_reason="Ghost position closed by RecoveryService (not found on exchange)"
                                )

                                logger.info(
                                    f"✓ Призрачная позиция {symbol} закрыта: {position_id}"
                                )
                            else:
                                # Fallback: Прямое обновление статуса без FSM
                                logger.warning(
                                    f"⚠ FSM не поддерживает start_close, обновляем статус напрямую"
                                )

                                await position_repository.update_status(
                                    position_id=position_id,
                                    new_status=PositionStatus.CLOSED,
                                    exit_price=local_position.current_price or local_position.entry_price,
                                    exit_reason="Ghost position closed by RecoveryService (not found on exchange, direct update)"
                                )

                                logger.info(
                                    f"✓ Призрачная позиция {symbol} закрыта напрямую: {position_id}"
                                )

                        else:
                            logger.warning(
                                f"⚠ Невозможно автоматически закрыть позицию {symbol} | "
                                f"Статус: {local_position.status.value} | "
                                f"Требуется ручное вмешательство"
                            )

                        # Удаляем FSM из registry
                        fsm_registry.unregister_position_fsm(position_id)

                        # Логируем в audit
                        await audit_repository.log(
                            action=AuditAction.POSITION_CLOSE,
                            entity_type="Position",
                            entity_id=position_id,
                            old_value={"status": local_position.status.value},
                            new_value={"status": "CLOSED"},
                            reason="Ghost position detected and closed during reconciliation",
                            success=True,
                            context={
                                "symbol": symbol,
                                "local_quantity": local_position.quantity,
                                "exchange_found": False
                            }
                        )

                    except Exception as close_error:
                        logger.error(
                            f"Ошибка закрытия призрачной позиции {symbol}: {close_error}",
                            exc_info=True
                        )

                        # Логируем ошибку в audit
                        await audit_repository.log(
                            action=AuditAction.POSITION_CLOSE,
                            entity_type="Position",
                            entity_id=position_id,
                            old_value={"status": local_position.status.value},
                            new_value={"error": str(close_error)},
                            reason="Failed to close ghost position during reconciliation",
                            success=False,
                            error_message=str(close_error)
                        )

            # ==========================================
            # ШАГ 5: ИМПОРТ ПОЗИЦИЙ С БИРЖИ (НЕТ В БД)
            # ==========================================
            # Находим позиции, которые есть на бирже, но НЕТ в БД
            exchange_only_symbols = set(exchange_map.keys()) - local_symbols

            if exchange_only_symbols:
                logger.info(
                    f"📥 Найдено {len(exchange_only_symbols)} позиций на бирже, "
                    f"которых НЕТ в БД: {list(exchange_only_symbols)}"
                )

                for symbol in exchange_only_symbols:
                    exchange_position = exchange_map[symbol]

                    try:
                        # Извлекаем данные с биржи
                        size = float(exchange_position.get("size", 0))
                        side_str = exchange_position.get("side", "")
                        entry_price = float(exchange_position.get("avgPrice", 0))
                        current_price = float(exchange_position.get("markPrice", entry_price))
                        unrealized_pnl = float(exchange_position.get("unrealisedPnl", 0))
                        leverage = int(exchange_position.get("leverage", 10))

                        # Конвертируем side
                        side = OrderSide.BUY if side_str == "Buy" else OrderSide.SELL

                        logger.info(
                            f"📥 Импорт позиции с биржи:\n"
                            f"   Symbol: {symbol}\n"
                            f"   Side: {side_str}\n"
                            f"   Size: {size}\n"
                            f"   Entry Price: {entry_price}\n"
                            f"   Current Price: {current_price}\n"
                            f"   Unrealized PnL: {unrealized_pnl}\n"
                            f"   Leverage: {leverage}x"
                        )

                        # Создаем позицию в БД
                        imported_position = await position_repository.create(
                            symbol=symbol,
                            side=side,
                            quantity=size,
                            entry_price=entry_price,
                            entry_reason=f"Position imported from exchange (manual trade or external system)"
                        )

                        imported_position_id = str(imported_position.id)

                        # Обновляем текущую цену и PnL
                        await position_repository.update_price(
                            position_id=imported_position_id,
                            current_price=current_price
                        )

                        # Обновляем статус на OPEN (т.к. позиция уже открыта)
                        await position_repository.update_status(
                            position_id=imported_position_id,
                            new_status=PositionStatus.OPEN
                        )

                        # Сохраняем метаданные
                        await position_repository.update_metadata(
                            position_id=imported_position_id,
                            metadata={
                                "imported_from_exchange": True,
                                "imported_at": datetime.utcnow().isoformat(),
                                "leverage": leverage,
                                "unrealized_pnl": unrealized_pnl
                            }
                        )

                        # Создаем FSM для импортированной позиции
                        position_fsm = PositionStateMachine(
                            position_id=imported_position_id,
                            initial_state=PositionStatus.OPEN  # Сразу OPEN
                        )
                        fsm_registry.register_position_fsm(imported_position_id, position_fsm)

                        # Логируем в audit
                        await audit_repository.log(
                            action=AuditAction.POSITION_OPEN,
                            entity_type="Position",
                            entity_id=imported_position_id,
                            new_value={
                                "symbol": symbol,
                                "side": side_str,
                                "quantity": size,
                                "entry_price": entry_price,
                                "imported": True
                            },
                            reason="Position imported from exchange during reconciliation",
                            success=True
                        )

                        result["synced"] += 1
                        result["positions_imported"] += 1

                        logger.info(
                            f"✓ Позиция {symbol} импортирована в БД: {imported_position_id}"
                        )

                    except Exception as import_error:
                        logger.error(
                            f"❌ Ошибка импорта позиции {symbol}: {import_error}",
                            exc_info=True
                        )

                        # Логируем ошибку в audit
                        await audit_repository.log(
                            action=AuditAction.POSITION_OPEN,
                            entity_type="Position",
                            entity_id="IMPORT_FAILED",
                            new_value={
                                "symbol": symbol,
                                "error": str(import_error)
                            },
                            reason="Failed to import position from exchange",
                            success=False,
                            error_message=str(import_error)
                        )

            # ==========================================
            # ШАГ 6: ИТОГОВОЕ ЛОГИРОВАНИЕ
            # ==========================================
            logger.info(
                f"✓ Сверка позиций завершена: "
                f"синхронизировано={result['synced']}, "
                f"расхождений={result['discrepancies']}, "
                f"призрачных закрыто={result['ghost_positions_closed']}, "
                f"импортировано={result['positions_imported']}"
            )

            if result["ghost_positions_closed"] > 0:
                logger.warning(
                    f"⚠⚠⚠ ВНИМАНИЕ: Обнаружено и закрыто {result['ghost_positions_closed']} "
                    f"призрачных позиций! Проверьте логи!"
                )

            if result["positions_imported"] > 0:
                logger.info(
                    f"📥 Импортировано {result['positions_imported']} позиций с биржи. "
                    f"Бот теперь управляет ими."
                )

            return result

        except Exception as e:
            logger.error(f"Критическая ошибка сверки позиций: {e}", exc_info=True)
            return result

    def _map_exchange_status(self, exchange_status: str) -> OrderStatus:
        """
        Маппинг статуса с биржи на локальный статус.

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


# ==================== ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР ====================

recovery_service = RecoveryService()