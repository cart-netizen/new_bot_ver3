"""
Исправленный ExecutionManager с полной интеграцией open_position/close_position.

КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ:
1. Получение реального баланса через balance_tracker
2. Правильное использование триггеров PositionStateMachine
3. _execute_signal интегрирован с open_position
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
from strategy.signal_deduplicator import signal_deduplicator
from utils.balance_tracker import balance_tracker  # ИМПОРТ balance_tracker
from utils.helpers import get_timestamp_ms, round_price, round_quantity

logger = get_logger(__name__)


class ExecutionManager:
    """Менеджер исполнения торговых ордеров с полным управлением позициями."""

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

        logger.info("ExecutionManager инициализирован с полной интеграцией позиций")

    # ==================== ПУБЛИЧНЫЕ МЕТОДЫ ====================

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

    # ==================== УПРАВЛЕНИЕ ПОЗИЦИЯМИ ====================

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
        Открытие позиции с РЕАЛЬНЫМ размещением ордера на Bybit.

        КРИТИЧЕСКИЕ ИЗМЕНЕНИЯ:
        1. Ордер размещается на бирже ПЕРВЫМ шагом
        2. Только после успешного размещения создаётся запись в БД
        3. Exchange order_id сохраняется в metadata
        4. Rollback при ошибке
        """
        with trace_operation("open_position", symbol=symbol, side=side):
            logger.info(
                f"→ Открытие позиции: {symbol} {side} | "
                f"Количество: {quantity} @ {entry_price}"
            )

            position_id = None
            exchange_order_id = None

            try:
                # ==========================================
                # ШАГ 0: ГЕНЕРАЦИЯ CLIENT ORDER ID
                # ==========================================
                client_order_id = idempotency_service.generate_idempotency_key(
                    operation="place_order",
                    params={
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "timestamp": get_timestamp_ms()
                    }
                )

                # Проверка идемпотентности
                existing_result = await idempotency_service.check_idempotency(
                    operation="place_order",
                    params={"symbol": symbol, "side": side, "quantity": quantity}
                )

                if existing_result:
                    logger.warning(
                        f"⚠️ Обнаружен дубликат операции: {symbol} {side}"
                    )
                    return existing_result

                # ==========================================
                # ШАГ 1: РАЗМЕЩЕНИЕ ОРДЕРА НА BYBIT
                # ==========================================
                logger.info(
                    f"📤 Размещение MARKET ордера на Bybit: {symbol} {side} {quantity}"
                )

                try:
                    # КРИТИЧЕСКИЙ ВЫЗОВ К BYBIT API
                    bybit_response = await self.rest_client.place_order(
                        symbol=symbol,
                        side=side,
                        order_type="Market",
                        quantity=quantity,
                        price=None,  # Market order
                        time_in_force="GTC",
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        client_order_id=client_order_id
                    )

                    # Извлечение данных
                    result_data = bybit_response.get("result", {})
                    exchange_order_id = result_data.get("orderId")
                    order_link_id = result_data.get("orderLinkId")

                    if not exchange_order_id:
                        raise OrderExecutionError(
                            f"Bybit не вернул orderId: {bybit_response}"
                        )

                    logger.info(
                        f"✅ Ордер размещён на Bybit: "
                        f"exchange_order_id={exchange_order_id}, "
                        f"client_order_id={order_link_id}"
                    )

                    # Сохранение для идемпотентности
                    await idempotency_service.save_operation_result(
                        operation="place_order",
                        params={"symbol": symbol, "side": side, "quantity": quantity},
                        result={
                            "exchange_order_id": exchange_order_id,
                            "client_order_id": order_link_id,
                            "timestamp": get_timestamp_ms()
                        },
                        ttl_minutes=60
                    )

                except Exception as order_error:
                    logger.error(
                        f"❌ ОШИБКА размещения ордера на Bybit: {order_error}"
                    )
                    self.stats["failed_orders"] += 1

                    await audit_repository.log(
                        action=AuditAction.POSITION_OPEN,
                        entity_type="Position",
                        entity_id="FAILED",
                        new_value={
                            "symbol": symbol,
                            "side": side,
                            "quantity": quantity,
                            "error": str(order_error)
                        },
                        reason=f"Failed to place order: {str(order_error)}",
                        success=False,
                        error_message=str(order_error)
                    )

                    return None

                # ==========================================
                # ШАГ 2: СОЗДАНИЕ ПОЗИЦИИ В БД
                # ==========================================
                logger.info(f"💾 Создание позиции в БД после успешного размещения")

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
                    entry_reason=entry_reason or f"{side} position opened",
                    # ВАЖНО: Сохраняем exchange_order_id
                    metadata_json={
                        "exchange_order_id": exchange_order_id,
                        "client_order_id": client_order_id,
                        "order_placed_at": get_timestamp_ms()
                    }
                )

                position_id = str(position.id)

                logger.info(
                    f"✓ Позиция создана в БД: {position_id} | "
                    f"Статус: {position.status.value} | "
                    f"Exchange Order: {exchange_order_id}"
                )

                # ==========================================
                # ШАГ 3-6: FSM, Risk Manager, Audit
                # ==========================================
                # ... остальной код без изменений ...

                position_fsm = PositionStateMachine(
                    position_id=position_id,
                    initial_state=PositionStatus.OPENING
                )

                fsm_registry.register_position_fsm(position_id, position_fsm)
                position_fsm.confirm_open()  # type: ignore

                await position_repository.update_status(
                    position_id=position_id,
                    new_status=PositionStatus.OPEN
                )

                position_size_usdt = quantity * entry_price
                signal_type = SignalType.BUY if side == "Buy" else SignalType.SELL

                self.risk_manager.register_position_opened(
                    symbol=symbol,
                    side=signal_type,
                    size_usdt=position_size_usdt,
                    entry_price=entry_price,
                    leverage=10
                )

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
                        "exchange_order_id": exchange_order_id
                    },
                    reason=entry_reason or "Position opened",
                    success=True,
                    context={
                        "entry_signal": entry_signal,
                        "entry_market_data": entry_market_data,
                        "entry_indicators": entry_indicators
                    }
                )

                logger.info(
                    f"✓✓✓ ПОЗИЦИЯ УСПЕШНО ОТКРЫТА ✓✓✓\n"
                    f"  Position ID: {position_id}\n"
                    f"  Exchange Order ID: {exchange_order_id}\n"
                    f"  Symbol: {symbol}\n"
                    f"  Side: {side}\n"
                    f"  Entry Price: {entry_price}\n"
                    f"  Quantity: {quantity}\n"
                    f"  Size: {position_size_usdt:.2f} USDT"
                )

                return {
                    "position_id": position_id,
                    "exchange_order_id": exchange_order_id,
                    "client_order_id": client_order_id,
                    "status": "success"
                }

            except Exception as e:
                logger.error(f"❌ Критическая ошибка open_position: {e}")

                # Если позиция создана в БД, но что-то пошло не так далее - откатываем
                if position_id:
                    try:
                        await position_repository.update_status(
                            position_id=position_id,
                            new_status=PositionStatus.FAILED
                        )
                        logger.warning(f"Позиция {position_id} помечена как FAILED")
                    except:
                        pass

                raise ExecutionError(f"Failed to open position: {str(e)}")

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

        Args:
            position_id: ID позиции для закрытия
            exit_price: Цена выхода
            exit_signal: Сигнал на выход
            exit_market_data: Рыночные данные при выходе
            exit_indicators: Индикаторы при выходе
            exit_reason: Причина закрытия

        Returns:
            Optional[dict]: Результат с realized_pnl или None при ошибке
        """
        with trace_operation("close_position", position_id=position_id):
            logger.info(f"→ Закрытие позиции: {position_id} @ {exit_price}")

            try:
                # 1. ПОЛУЧЕНИЕ ПОЗИЦИИ ИЗ БД
                position = await position_repository.get_by_id(position_id)

                if not position:
                    logger.error(f"Позиция {position_id} не найдена в БД")
                    return None

                symbol = position.symbol

                # 2. ПОЛУЧЕНИЕ ИЛИ ВОССТАНОВЛЕНИЕ FSM
                position_fsm = fsm_registry.get_position_fsm(position_id)

                if not position_fsm:
                    logger.warning(
                        f"FSM не найдена для позиции {position_id}, "
                        f"восстанавливаем из БД"
                    )

                    position_fsm = PositionStateMachine(
                        position_id=position_id,
                        initial_state=position.status
                    )

                    fsm_registry.register_position_fsm(position_id, position_fsm)

                    logger.info(
                        f"FSM восстановлена для позиции {position_id} | "
                        f"Статус: {position_fsm.current_status.value}"
                    )

                # 3. ВАЛИДАЦИЯ ВОЗМОЖНОСТИ ЗАКРЫТИЯ
                if not position_fsm.can_transition_to(PositionStatus.CLOSING):
                    logger.error(
                        f"Невозможно закрыть позицию {position_id} | "
                        f"Текущий статус: {position_fsm.current_status.value} | "
                        f"Доступные переходы: {position_fsm.get_available_transitions()}"
                    )
                    return None

                logger.debug(f"✓ Валидация закрытия прошла для позиции {position_id}")

                # 4. ДВУХШАГОВОЕ ЗАКРЫТИЕ ЧЕРЕЗ FSM
                # Триггеры создаются динамически библиотекой transitions

                # Шаг 1: OPEN -> CLOSING
                position_fsm.start_close()  # type: ignore[attr-defined]

                await position_repository.update_status(
                    position_id=position_id,
                    new_status=PositionStatus.CLOSING
                )

                logger.info(
                    f"✓ Позиция переведена в CLOSING: {position_id} | "
                    f"FSM статус: {position_fsm.current_status.value}"
                )

                # Шаг 2: CLOSING -> CLOSED
                position_fsm.confirm_close()  # type: ignore[attr-defined]

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

                # 5. УДАЛЕНИЕ FSM ИЗ REGISTRY
                fsm_registry.unregister_position_fsm(position_id)

                logger.debug(f"✓ FSM удалена из Registry для позиции {position_id}")

                # 6. ОБНОВЛЕНИЕ RISK MANAGER
                self.risk_manager.register_position_closed(position.symbol)

                logger.info(
                    f"✓ Позиция удалена из Risk Manager: {position.symbol}"
                )

                # 7. ПОЛУЧЕНИЕ ОБНОВЛЕННОЙ ПОЗИЦИИ ДЛЯ PNL
                updated_position = await position_repository.get_by_id(position_id)

                realized_pnl = updated_position.realized_pnl or 0.0
                duration = (
                    (updated_position.closed_at - updated_position.opened_at).total_seconds()
                    if updated_position.closed_at and updated_position.opened_at
                    else 0
                )

                # 8. AUDIT LOGGING
                await audit_repository.log(
                    action=AuditAction.POSITION_CLOSE,
                    entity_type="Position",
                    entity_id=position_id,
                    new_value={
                        "exit_price": exit_price,
                        "realized_pnl": realized_pnl
                    },
                    reason=exit_reason,
                    success=True,
                    context={
                        "exit_signal": exit_signal,
                        "exit_market_data": exit_market_data,
                        "exit_indicators": exit_indicators
                    }
                )
                # ============================================
                # НОВЫЙ ШАГ: ОЧИСТКА ИСТОРИИ СИГНАЛОВ
                # ============================================
                from strategy.signal_deduplicator import signal_deduplicator

                signal_deduplicator.clear_symbol(symbol)
                logger.info(
                    f"{symbol} | История сигналов очищена после закрытия позиции"
                )

                # 9. ВОЗВРАТ РЕЗУЛЬТАТА
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

                await audit_repository.log(
                    action=AuditAction.POSITION_CLOSE,
                    entity_type="Position",
                    entity_id=position_id,
                    success=False,
                    error_message=str(e),
                    reason=f"Failed to close position: {exit_reason}"
                )

                return None

    # ==================== ПРИВАТНЫЕ МЕТОДЫ ====================

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
        Исполнение торгового сигнала с использованием open_position.

        ✅ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Используется open_position вместо _place_order

        Args:
            signal: Торговый сигнал
        """
        # ============================================
        # ШАГ 0.0: ПРОВЕРКА ЛИМИТА ПОЗИЦИЙ (CIRCUIT BREAKER)
        # ============================================
        # КРИТИЧНО: Проверяем ДО дедупликации, ДО баланса, ДО всего!
        current_positions = self.risk_manager.metrics.open_positions_count
        max_positions = self.risk_manager.limits.max_open_positions

        if current_positions >= max_positions:
            logger.warning(
                f"🛑 CIRCUIT BREAKER: Достигнут лимит позиций {current_positions}/{max_positions}. "
                f"Открытые пары: {list(self.risk_manager.open_positions.keys())}. "
                f"Сигнал {signal.symbol} отклонён БЕЗ обработки."
            )
            self.stats["rejected_orders"] += 1
            return

        # Проверка: уже есть позиция по этой паре?
        if signal.symbol in self.risk_manager.open_positions:
            logger.warning(
                f"⚠️ CIRCUIT BREAKER: По паре {signal.symbol} уже открыта позиция. "
                f"Сигнал отклонён."
            )
            self.stats["rejected_orders"] += 1
            return

        logger.debug(
            f"{signal.symbol} | ✓ Проверка лимита: {current_positions}/{max_positions} "
            f"(после открытия будет {current_positions + 1}/{max_positions})"
        )

        # ==========================================
        # ШАГ 0: ДЕДУПЛИКАЦИЯ СИГНАЛА
        # ==========================================
        should_process, block_reason = signal_deduplicator.should_process_signal(signal)

        if not should_process:
            logger.info(
                f"{signal.symbol} | ⏭️ Сигнал пропущен (дубликат): {block_reason}"
            )
            self.stats["rejected_orders"] += 1
            return

        logger.info(
            f"{signal.symbol} | Исполнение сигнала: "
            f"{signal.signal_type.value} @ {signal.price:.8f}"
        )

        try:
            # 1. СТРОГАЯ ПРОВЕРКА БАЛАНСА
            # ✅ КРИТИЧНО: Fallback недопустим - только реальный баланс
            available_balance = balance_tracker.get_current_balance()

            if available_balance is None:
                error_msg = (
                    f"КРИТИЧЕСКАЯ ОШИБКА: Баланс недоступен для {signal.symbol}. "
                    f"Невозможно определить доступность средств для открытия позиции."
                )
                logger.error(error_msg)

                # Отклоняем сигнал из-за недоступности баланса
                self.stats["rejected_orders"] += 1
                self._add_to_history(signal, "rejected", "Balance unavailable")

                # Уведомляем через audit
                await audit_repository.log(
                    action=AuditAction.POSITION_OPEN,
                    entity_type="Signal",
                    entity_id=signal.symbol,
                    success=False,
                    error_message=error_msg,
                    reason="Balance check failed - balance unavailable"
                )

                return

            logger.debug(f"Доступный баланс: ${available_balance:.2f} USDT")

            # 2. РАСЧЕТ РАЗМЕРА ПОЗИЦИИ
            position_size_usdt = self.risk_manager.calculate_position_size(
                signal,
                available_balance
            )

            # 3. ВАЛИДАЦИЯ СИГНАЛА
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

            # 4. ОТКРЫТИЕ ПОЗИЦИИ
            # ✅ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Используем open_position

            # Конвертируем тип сигнала в сторону
            side = "Buy" if signal.signal_type == SignalType.BUY else "Sell"

            # Рассчитываем количество
            quantity = position_size_usdt / signal.price
            quantity = round_quantity(quantity, decimals=6)

            # Рассчитываем SL/TP
            if side == "Buy":
                stop_loss = signal.price * 0.98  # -2%
                take_profit = signal.price * 1.05  # +5%
            else:
                stop_loss = signal.price * 1.02  # +2%
                take_profit = signal.price * 0.95  # -5%

            # Открываем позицию через новый метод
            result = await self.open_position(
                symbol=signal.symbol,
                side=side,
                entry_price=signal.price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_signal={
                    "type": signal.signal_type.value,
                    "source": signal.source.value,
                    "strength": signal.strength.value,
                    "confidence": signal.confidence
                },
                entry_market_data={
                    "price": signal.price,
                    "timestamp": signal.timestamp
                },
                entry_indicators=signal.metadata.get("indicators", {}),
                entry_reason=signal.reason
            )

            if result:
                # Обновляем сигнал
                signal.executed = True
                signal.execution_price = signal.price
                signal.execution_timestamp = get_timestamp_ms()

                self.stats["executed_orders"] += 1
                self._add_to_history(signal, "executed", result["position_id"])

                logger.info(
                    f"{signal.symbol} | Позиция успешно открыта: "
                    f"position_id={result['position_id']}"
                )
            else:
                self.stats["failed_orders"] += 1
                self._add_to_history(signal, "failed", "Failed to open position")
                logger.error(
                    f"{signal.symbol} | Ошибка открытия позиции"
                )

        except Exception as e:
            self.stats["failed_orders"] += 1
            self._add_to_history(signal, "failed", str(e))
            logger.error(f"{signal.symbol} | Ошибка исполнения сигнала: {e}")
            raise ExecutionError(f"Failed to execute signal: {str(e)}")

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

    # ==================== СТАТИСТИКА ====================

    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Получение истории исполнения."""
        history = list(self.execution_history)

        if limit:
            history = history[-limit:]

        return history

    def get_statistics(self) -> Dict:
        """Получение статистики исполнения."""
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