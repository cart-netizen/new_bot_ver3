"""
Исправленный ExecutionManager с полной интеграцией open_position/close_position.

КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ:
1. Получение реального баланса через balance_tracker
2. Правильное использование триггеров PositionStateMachine
3. _execute_signal интегрирован с open_position
"""

import asyncio
from decimal import Decimal
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
from utils.balance_tracker import balance_tracker
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

        # Кеш информации об инструментах
        self.instruments_cache: Dict[str, dict] = {}
        self.cache_ttl = 3600  # 1 час

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
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        entry_signal: Optional[dict] = None,
        entry_market_data: Optional[dict] = None,
        entry_indicators: Optional[dict] = None,
        entry_reason: Optional[str] = None
    ) -> Optional[dict]:
        """
        Открытие позиции с полным управлением жизненным циклом.

        ИЗМЕНЕНИЯ:
        1. Создаем Position БЕЗ metadata_json
        2. Привязываем Order к Position через update_position_link()
        3. Опционально обновляем metadata_json через update_metadata()

        Args:
            symbol: Торговая пара
            side: Сторона ("Buy" или "Sell")
            quantity: Количество
            entry_price: Цена входа
            stop_loss: Stop Loss (опционально)
            take_profit: Take Profit (опционально)
            entry_signal: Сигнал на вход
            entry_market_data: Рыночные данные при входе
            entry_indicators: Индикаторы при входе
            entry_reason: Причина открытия

        Returns:
            Optional[dict]: Результат с position_id и exchange_order_id или None
        """
        with trace_operation("open_position", symbol=symbol, side=side):
            logger.info(
                f"→ Открытие позиции: {symbol} {side} {quantity} @ {entry_price}"
            )

            position_id = None

            try:
                # ==========================================
                # ШАГ 1: РАЗМЕЩЕНИЕ ОРДЕРА НА БИРЖЕ
                # ==========================================
                logger.info(f"📡 Размещение ордера на бирже: {symbol} {side}")

                # Генерируем client_order_id
                client_order_id = idempotency_service.generate_client_order_id(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=entry_price
                )

                logger.debug(f"Client Order ID: {client_order_id}")

                # Размещаем ордер на бирже
                try:
                    order_response = await rest_client.place_order(
                        symbol=symbol,
                        side=side,
                        order_type="Market",  # или "Limit" в зависимости от стратегии
                        quantity=quantity,
                        price=entry_price if side == "Limit" else None,
                        client_order_id=client_order_id
                    )

                    result = order_response.get("result", {})
                    exchange_order_id = result.get("orderId")
                    order_link_id = result.get("orderLinkId")

                    if not exchange_order_id:
                        raise OrderExecutionError("Exchange не вернул orderId")

                    logger.info(
                        f"✓ Ордер размещен на бирже:\n"
                        f"  Exchange Order ID: {exchange_order_id}\n"
                        f"  Order Link ID: {order_link_id}"
                    )

                    # Проверка корректности orderLinkId
                    if order_link_id != client_order_id:
                        logger.warning(
                            f"⚠ orderLinkId не совпадает с client_order_id!\n"
                            f"  Ожидалось: {client_order_id}\n"
                            f"  Получено: {order_link_id}"
                        )

                except Exception as order_error:
                    logger.error(
                        f"❌ Ошибка размещения ордера на бирже: {order_error}",
                        exc_info=True
                    )

                    # Аудит неудачного размещения
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

                # ✅ ИСПРАВЛЕНО: Убран metadata_json из create()
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
                    # ❌ УБРАНО: metadata_json больше не передается здесь
                )

                position_id = str(position.id)

                logger.info(
                    f"✓ Позиция создана в БД: {position_id} | "
                    f"Статус: {position.status.value}"
                )

                # ==========================================
                # ШАГ 2.5: СОЗДАНИЕ ORDER В БД (НОВОЕ!)
                # ==========================================
                logger.info(f"📝 Создание Order в БД")

                try:
                    order = await order_repository.create(
                        client_order_id=client_order_id,
                        symbol=symbol,
                        side=OrderSide.BUY if side == "Buy" else OrderSide.SELL,
                        order_type=OrderType.MARKET,  # или LIMIT в зависимости от типа
                        quantity=quantity,
                        price=entry_price,
                        signal_data=entry_signal,
                        market_data=entry_market_data,
                        indicators=entry_indicators,
                        reason=entry_reason or f"{side} market order",
                        position_id=position_id  # Сразу привязываем к позиции
                    )

                    # Обновляем статус Order на PLACED (ордер уже на бирже)
                    await order_repository.update_status(
                        client_order_id=client_order_id,
                        new_status=OrderStatus.PLACED,
                        exchange_order_id=exchange_order_id
                    )

                    logger.info(
                        f"✓ Order создан в БД: {client_order_id} | "
                        f"Exchange ID: {exchange_order_id}"
                    )

                except Exception as order_create_error:
                    logger.error(
                        f"⚠ Ошибка создания Order в БД: {order_create_error}",
                        exc_info=True
                    )
                    # Продолжаем, т.к. позиция уже создана на бирже

                # ==========================================
                # ШАГ 3: ОБНОВЛЕНИЕ ПРИВЯЗКИ (ЕСЛИ НУЖНО)
                # ==========================================
                # Примечание: Order уже создан с position_id в Шаге 2.5
                # Этот шаг оставлен для совместимости, но может быть пропущен
                logger.debug(f"🔗 Order уже привязан к Position при создании")

                # Опциональная дополнительная проверка привязки
                # link_success = await order_repository.update_position_link(
                #     client_order_id=client_order_id,
                #     position_id=position_id
                # )
                #
                # if not link_success:
                #     logger.warning(
                #         f"⚠ Не удалось обновить привязку Order {client_order_id} "
                #         f"к Position {position_id}"
                #     )

                # ==========================================
                # ШАГ 4: ОБНОВЛЕНИЕ METADATA (ОПЦИОНАЛЬНО)
                # ==========================================
                logger.debug(f"📝 Обновление metadata позиции (справочная информация)")

                # ✅ ДОБАВЛЕНО: Опционально сохраняем справочную информацию
                metadata_success = await position_repository.update_metadata(
                    position_id=position_id,
                    metadata={
                        "exchange_order_id": exchange_order_id,  # Справочно
                        "client_order_id": client_order_id,  # Справочно
                        "order_placed_at": get_timestamp_ms(),
                        "order_link_id": order_link_id
                    }
                )

                if not metadata_success:
                    logger.warning(
                        f"⚠ Не удалось обновить metadata для позиции {position_id}"
                    )
                else:
                    logger.debug(f"✓ Metadata обновлена для позиции {position_id}")

                # ==========================================
                # ШАГ 5: FSM ДЛЯ ПОЗИЦИИ
                # ==========================================
                logger.info(f"🔄 Инициализация FSM для позиции")

                # Создаем FSM с начальным статусом OPENING
                position_fsm = PositionStateMachine(
                    position_id=position_id,
                    initial_state=PositionStatus.OPENING
                )

                # Регистрируем в глобальном реестре
                fsm_registry.register_position_fsm(position_id, position_fsm)

                logger.debug(
                    f"✓ FSM зарегистрирована для позиции {position_id} | "
                    f"Статус: {position_fsm.current_status.value}"
                )

                # Переход OPENING -> OPEN через триггер
                position_fsm.confirm_open()  # type: ignore[attr-defined]

                await position_repository.update_status(
                    position_id=position_id,
                    new_status=PositionStatus.OPEN
                )

                logger.info(
                    f"✓ Позиция переведена в OPEN: {position_id} | "
                    f"FSM статус: {position_fsm.current_status.value}"
                )

                # ==========================================
                # ШАГ 6: РЕГИСТРАЦИЯ В RISK MANAGER
                # ==========================================
                logger.info(f"📊 Регистрация позиции в RiskManager")

                position_size_usdt = quantity * entry_price

                # ✅ ИСПРАВЛЕНО: Конвертируем str -> SignalType
                signal_type = SignalType.BUY if side == "Buy" else SignalType.SELL

                self.risk_manager.register_position_opened(
                    symbol=symbol,
                    side=signal_type,  # ✅ Передаем SignalType вместо str
                    size_usdt=position_size_usdt,
                    entry_price=entry_price,
                    leverage=10  # Опционально: можно добавить в параметры метода
                )

                logger.info(f"✓ Позиция зарегистрирована в RiskManager")

                # ==========================================
                # ШАГ 7: АУДИТ УСПЕШНОГО ОТКРЫТИЯ
                # ==========================================
                await audit_repository.log(
                    action=AuditAction.POSITION_OPEN,
                    entity_type="Position",
                    entity_id=position_id,
                    new_value={
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "entry_price": entry_price,
                        "exchange_order_id": exchange_order_id,
                        "position_size_usdt": position_size_usdt,
                        "entry_signal": entry_signal,
                        "entry_market_data": entry_market_data,
                        "entry_indicators": entry_indicators
                    },
                    reason=entry_reason or f"{side} position opened",
                    success=True
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
                logger.error(f"❌ Критическая ошибка open_position: {e}", exc_info=True)

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

                # 5. РАСЧЕТ REALIZED PNL
                if position.side == OrderSide.BUY:
                    realized_pnl = (exit_price - position.entry_price) * position.quantity
                else:
                    realized_pnl = (position.entry_price - exit_price) * position.quantity

                logger.info(f"💰 Realized PnL: {realized_pnl:.2f} USDT")

                # 6. УДАЛЕНИЕ ИЗ RISK MANAGER
                # ✅ ИСПРАВЛЕНО: Убран аргумент realized_pnl
                self.risk_manager.register_position_closed(symbol=symbol)

                # 7. АУДИТ
                await audit_repository.log(
                    action=AuditAction.POSITION_CLOSE,
                    entity_type="Position",
                    entity_id=position_id,
                    new_value={
                        "exit_price": exit_price,
                        "realized_pnl": realized_pnl,
                        "exit_reason": exit_reason
                    },
                    success=True
                )

                logger.info(f"✓✓✓ ПОЗИЦИЯ УСПЕШНО ЗАКРЫТА ✓✓✓")

                return {
                    "position_id": position_id,
                    "realized_pnl": realized_pnl,
                    "status": "success"
                }

            except Exception as e:
                logger.error(f"❌ Ошибка закрытия позиции: {e}", exc_info=True)
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
        Исполнение торгового сигнала.

        ИСПРАВЛЕНИЯ:
        1. Добавлена валидация и округление quantity
        2. Добавлена проверка минимального размера ордера (5 USDT)
        3. Улучшено логирование всех расчетов

        Args:
            signal: Торговый сигнал
        """
        # ============================================
        # ШАГ 0.0: ПРОВЕРКА ЛИМИТА ПОЗИЦИЙ
        # ============================================
        current_positions = self.risk_manager.metrics.open_positions_count
        max_positions = self.risk_manager.limits.max_open_positions

        if current_positions >= max_positions:
            logger.warning(
                f"🛑 CIRCUIT BREAKER: Достигнут лимит позиций {current_positions}/{max_positions}. "
                f"Сигнал {signal.symbol} отклонён."
            )
            self.stats["rejected_orders"] += 1
            return

        if signal.symbol in self.risk_manager.open_positions:
            logger.warning(
                f"⚠️ CIRCUIT BREAKER: По паре {signal.symbol} уже открыта позиция. Сигнал отклонён."
            )
            self.stats["rejected_orders"] += 1
            return

        # ==========================================
        # ШАГ 0.1: ДЕДУПЛИКАЦИЯ СИГНАЛА
        # ==========================================
        should_process, block_reason = signal_deduplicator.should_process_signal(signal)

        if not should_process:
            logger.info(
                f"{signal.symbol} | ⏭️ Сигнал пропущен (дубликат): {block_reason}"
            )
            self.stats["rejected_orders"] += 1
            return

        logger.info(
            f"{signal.symbol} | Исполнение сигнала: {signal.signal_type.value} @ {signal.price:.8f}"
        )

        try:
            # ==========================================
            # ШАГ 1: ПОЛУЧЕНИЕ ИНФОРМАЦИИ ОБ ИНСТРУМЕНТЕ
            # ==========================================
            instrument_info = await self._get_instrument_info(signal.symbol)

            if not instrument_info:
                error_msg = f"Не удалось получить информацию об инструменте {signal.symbol}"
                logger.error(f"{signal.symbol} | {error_msg}")
                self.stats["failed_orders"] += 1
                return

            # ==========================================
            # ШАГ 2: ПРОВЕРКА БАЛАНСА
            # ==========================================
            available_balance = balance_tracker.get_current_balance()

            if available_balance is None or available_balance <= 0:
                error_msg = (
                    f"КРИТИЧЕСКАЯ ОШИБКА: Баланс недоступен для {signal.symbol}. "
                    f"Невозможно открыть позицию."
                )
                logger.error(error_msg)
                self.stats["failed_orders"] += 1
                return

            logger.info(
                f"{signal.symbol} | Доступный баланс: {available_balance:.2f} USDT"
            )

            # ==========================================
            # ШАГ 3: РАСЧЕТ РАЗМЕРА ПОЗИЦИИ
            # ==========================================
            # Используем текущую цену сигнала
            entry_price = signal.price

            # ИСПРАВЛЕНИЕ: Правильная сигнатура метода calculate_position_size
            # Метод принимает: (signal: TradingSignal, available_balance: float, leverage: Optional[int])
            # и возвращает ТОЛЬКО position_size_usdt (float), а не tuple!

            # Расчет через risk_manager (учитывает leverage)
            raw_position_size_usdt = self.risk_manager.calculate_position_size(
                signal=signal,
                available_balance=available_balance,
                leverage=self.risk_manager.limits.default_leverage
            )

            # Рассчитываем quantity из position_size
            raw_quantity = raw_position_size_usdt / entry_price

            logger.info(
                f"{signal.symbol} | Расчет позиции: "
                f"баланс={available_balance:.2f} USDT, "
                f"leverage={self.risk_manager.limits.default_leverage}x, "
                f"размер={raw_position_size_usdt:.2f} USDT, "
                f"raw_quantity={raw_quantity:.8f}"
            )

            # ==========================================
            # ШАГ 4: ВАЛИДАЦИЯ И ОКРУГЛЕНИЕ QUANTITY
            # ==========================================
            validated_quantity = self._validate_and_round_quantity(
                symbol=signal.symbol,
                quantity=raw_quantity,
                price=entry_price,
                instrument_info=instrument_info
            )

            if validated_quantity is None:
                error_msg = (
                    f"Quantity {raw_quantity:.8f} не прошло валидацию. "
                    f"Ордер отклонен."
                )
                logger.error(f"{signal.symbol} | {error_msg}")
                self.stats["failed_orders"] += 1
                return

            # Финальная проверка notional value
            final_notional = validated_quantity * entry_price
            min_notional = instrument_info["minNotionalValue"]

            if final_notional < min_notional:
                error_msg = (
                    f"Финальный размер ордера {final_notional:.2f} USDT < минимума {min_notional} USDT. "
                    f"Ордер отклонен (недостаточно средств для минимального ордера)."
                )
                logger.error(f"{signal.symbol} | {error_msg}")
                self.stats["failed_orders"] += 1
                return

            logger.info(
                f"{signal.symbol} | ✅ Финальные параметры ордера: "
                f"quantity={validated_quantity:.8f}, "
                f"notional={final_notional:.2f} USDT"
            )

            # ==========================================
            # ШАГ 5: РАСЧЕТ STOP LOSS И TAKE PROFIT
            # ==========================================
            stop_loss_pct = 0.02  # 2%
            take_profit_pct = 0.04  # 4%

            if signal.signal_type == SignalType.BUY:
                side = "Buy"
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
            else:
                side = "Sell"
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)

            # ==========================================
            # ШАГ 6: ОТКРЫТИЕ ПОЗИЦИИ
            # ==========================================
            result = await self.open_position(
                symbol=signal.symbol,
                side=side,
                entry_price=entry_price,
                quantity=validated_quantity,  # Используем валидированное quantity
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_signal=signal.to_dict(),
                entry_reason=f"Signal: {signal.signal_type.value}",
            )

            if result:
                self.stats["executed_orders"] += 1
                logger.info(
                    f"{signal.symbol} | ✅ Позиция успешно открыта: "
                    f"{side} {validated_quantity:.8f} @ {entry_price:.8f}"
                )
            else:
                self.stats["failed_orders"] += 1
                logger.error(f"{signal.symbol} | ❌ Не удалось открыть позицию")

        except Exception as e:
            logger.error(
                f"{signal.symbol} | ❌ Критическая ошибка исполнения сигнала: {e}",
                exc_info=True
            )
            self.stats["failed_orders"] += 1

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

        # ==================== НОВЫЕ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================

    async def _get_instrument_info(self, symbol: str) -> Optional[dict]:
            """
            Получение информации об инструменте с кешированием.

            Args:
                symbol: Торговая пара

            Returns:
                dict: Информация об инструменте или None при ошибке
            """
            # Проверка кеша
            if symbol in self.instruments_cache:
                cached = self.instruments_cache[symbol]
                cache_age = get_timestamp_ms() - cached.get("cached_at", 0)

                if cache_age < self.cache_ttl * 1000:
                    logger.debug(f"{symbol} | Использование кешированной информации об инструменте")
                    return cached

            # Запрос информации с Bybit
            try:
                logger.debug(f"{symbol} | Запрос информации об инструменте с Bybit")

                response = await self.rest_client.get_instruments_info(
                    symbol=symbol
                )

                if not response or not isinstance(response, list) or len(response) == 0:
                    logger.error(f"{symbol} | Некорректный ответ от Bybit: {response}")
                    return None

                # response это уже List[Dict], берем первый элемент
                instrument_info_raw = response[0]

                if not instrument_info_raw:
                    logger.error(f"{symbol} | Инструмент не найден на Bybit")
                    return None

                lot_size_filter = instrument_info_raw.get("lotSizeFilter", {})

                # Извлечение критических параметров
                info = {
                    "symbol": symbol,
                    "qtyStep": float(lot_size_filter.get("qtyStep", 0.001)),
                    "minOrderQty": float(lot_size_filter.get("minOrderQty", 0.001)),
                    "maxOrderQty": float(lot_size_filter.get("maxOrderQty", 100000)),
                    "minNotionalValue": float(lot_size_filter.get("minNotionalValue", 5)),  # Минимум 5 USDT
                    "cached_at": get_timestamp_ms()
                }

                # Кеширование
                self.instruments_cache[symbol] = info

                logger.info(
                    f"{symbol} | Информация об инструменте получена: "
                    f"qtyStep={info['qtyStep']}, minOrderQty={info['minOrderQty']}, "
                    f"minNotionalValue={info['minNotionalValue']}"
                )

                return info

            except Exception as e:
                logger.error(f"{symbol} | Ошибка получения информации об инструменте: {e}")
                return None

    def _validate_and_round_quantity(
            self,
            symbol: str,
            quantity: float,
            price: float,
            instrument_info: dict
        ) -> Optional[float]:
            """
            Валидация и округление quantity согласно правилам инструмента.

            Args:
                symbol: Торговая пара
                quantity: Исходное количество
                price: Текущая цена
                instrument_info: Информация об инструменте

            Returns:
                float: Округленное quantity или None при ошибке
            """
            qty_step = instrument_info["qtyStep"]
            min_order_qty = instrument_info["minOrderQty"]
            max_order_qty = instrument_info["maxOrderQty"]
            min_notional = instrument_info["minNotionalValue"]

            logger.debug(
                f"{symbol} | Валидация quantity: "
                f"raw={quantity:.8f}, price={price:.8f}, "
                f"qtyStep={qty_step}, minQty={min_order_qty}, minNotional={min_notional}"
            )

            # Округление quantity до qtyStep (вниз)
            decimal_qty = Decimal(str(quantity))
            decimal_step = Decimal(str(qty_step))

            rounded_qty = float((decimal_qty // decimal_step) * decimal_step)

            logger.debug(f"{symbol} | После округления по qtyStep: {rounded_qty:.8f}")

            # Проверка минимального quantity
            if rounded_qty < min_order_qty:
                logger.warning(
                    f"{symbol} | Quantity {rounded_qty:.8f} < minOrderQty {min_order_qty}. "
                    f"Увеличение до минимума."
                )
                rounded_qty = min_order_qty

            # Проверка максимального quantity
            if rounded_qty > max_order_qty:
                logger.error(
                    f"{symbol} | Quantity {rounded_qty:.8f} > maxOrderQty {max_order_qty}. "
                    f"Ордер отклонен."
                )
                return None

            # Проверка минимального размера ордера в USDT (notional value)
            notional_value = rounded_qty * price

            if notional_value < min_notional:
                logger.warning(
                    f"{symbol} | Размер ордера {notional_value:.2f} USDT < минимума {min_notional} USDT. "
                    f"Увеличение quantity до минимального размера."
                )

                # Пересчет quantity для достижения минимального notional
                required_qty = min_notional / price

                # Округление до qtyStep (вверх)
                decimal_required = Decimal(str(required_qty))
                rounded_qty = float(((decimal_required // decimal_step) + 1) * decimal_step)

                # Повторная проверка notional после округления
                new_notional = rounded_qty * price

                if new_notional < min_notional:
                    # Добавляем еще один шаг для гарантии
                    rounded_qty += qty_step
                    new_notional = rounded_qty * price

                logger.info(
                    f"{symbol} | Quantity скорректировано: {quantity:.8f} → {rounded_qty:.8f} "
                    f"(notional: {notional_value:.2f} → {new_notional:.2f} USDT)"
                )

            logger.info(
                f"{symbol} | ✅ Quantity прошло валидацию: {rounded_qty:.8f} "
                f"(размер ордера: {rounded_qty * price:.2f} USDT)"
            )

            return rounded_qty