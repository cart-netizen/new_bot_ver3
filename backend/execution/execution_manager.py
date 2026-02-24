"""
Исправленный ExecutionManager с полной интеграцией open_position/close_position.

КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ:
1. Получение реального баланса через balance_tracker
2. Правильное использование триггеров PositionStateMachine
3. _execute_signal интегрирован с open_position
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, List
from collections import deque

from backend.config import settings
from backend.core.logger import get_logger
from backend.core.exceptions import ExecutionError, OrderExecutionError
from backend.core.trace_context import trace_operation
from backend.database.models import AuditAction, OrderStatus, PositionStatus
from backend.domain.services.fsm_registry import fsm_registry
from backend.domain.services.idempotency_service import idempotency_service
from backend.domain.state_machines.order_fsm import OrderStateMachine
from backend.domain.state_machines.position_fsm import PositionStateMachine
from backend.infrastructure.repositories.audit_repository import audit_repository
from backend.infrastructure.repositories.order_repository import order_repository
from backend.infrastructure.repositories.position_repository import position_repository
from backend.infrastructure.resilience.circuit_breaker import circuit_breaker_manager
from backend.infrastructure.resilience.rate_limiter import rate_limited
from backend.models.signal import TradingSignal, SignalType
from backend.models.market_data import OrderSide, OrderType, TimeInForce
from backend.exchange.rest_client import rest_client
from backend.strategies.adaptive import adaptive_consensus_manager, AdaptiveConsensusManager

from backend.strategy.risk_manager import RiskManager
from backend.strategy.risk_models import MarketRegime, MLRiskAdjustments
from backend.strategy.signal_deduplicator import signal_deduplicator
from backend.strategy.sltp_calculator import sltp_calculator
from backend.strategy.trailing_stop_manager import trailing_stop_manager
from backend.utils.balance_tracker import balance_tracker
from backend.utils.helpers import get_timestamp_ms, round_price, round_quantity, safe_enum_value
from backend.core.trade_reporter import trade_reporter, PositionEvent, PositionEventType

logger = get_logger(__name__)

@dataclass
class SubmissionResult:
    """Результат отправки сигнала на исполнение."""
    success: bool
    reason: str
    order_id: Optional[str] = None
    symbol: Optional[str] = None

class ExecutionManager:
    """Менеджер исполнения торговых ордеров с полным управлением позициями."""

    def __init__(self, risk_manager: RiskManager, adaptive_consensus_manager: Optional[AdaptiveConsensusManager] = None):
        """
        Инициализация менеджера исполнения.

        Args:
            risk_manager: Менеджер рисков
        """
        self.risk_manager = risk_manager
        self.adaptive_consensus_manager = adaptive_consensus_manager
        self.rest_client = rest_client

        # Очередь сигналов для исполнения
        # self.signal_queue: asyncio.Queue = asyncio.Queue()

        # История исполнения
        self.execution_history: deque = deque(maxlen=1000)

        # Кеш информации об инструментах
        self.instruments_cache: Dict[str, dict] = {}
        self.cache_ttl = 3600  # 1 час

        # Флаг работы
        self.is_running = False
        # self.execution_task: Optional[asyncio.Task] = None

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
        # logger.info("Запуск менеджера исполнения")
        #
        # # Запускаем задачу обработки очереди
        # self.execution_task = asyncio.create_task(self._process_queue())
        logger.info("ExecutionManager запущен (режим немедленного исполнения)")


    async def stop(self):
        """Остановка менеджера исполнения."""
        if not self.is_running:
            logger.warning("Менеджер исполнения уже остановлен")
            return

        logger.info("Остановка ExecutionManager")
        self.is_running = False

        # # Отменяем задачу обработки
        # if self.execution_task and not self.execution_task.done():
        #     self.execution_task.cancel()
        #     try:
        #         await self.execution_task
        #     except asyncio.CancelledError:
        #         pass

    async def submit_signal(self, signal: TradingSignal) -> SubmissionResult:
        """
        Немедленное исполнение сигнала с проверкой лимитов.

        ИЗМЕНЕНИЯ:
        - Убрана очередь - сигналы исполняются немедленно
        - Проверка актуальности сигнала (возраст < 60 секунд)
        - Синхронизация позиций с биржей перед проверкой лимитов
        - Проверка лимита открытых позиций
        - Немедленное исполнение или отклонение

        Args:
            signal: Торговый сигнал

        Returns:
            SubmissionResult: Результат немедленного исполнения

        Args:
            signal: Торговый сигнал
        """
        try:
            # await self.signal_queue.put(signal)
            self.stats["total_signals"] += 1
            # logger.debug(f"{signal.symbol} | Сигнал добавлен в очередь исполнения")

            # ==========================================
            # ШАГ 1: ПРОВЕРКА АКТУАЛЬНОСТИ СИГНАЛА
            # ==========================================
            if not signal.is_valid:
                logger.warning(
                    f"{signal.symbol} | ⏰ Сигнал устарел: "
                    f"возраст={signal.age_seconds:.1f}s (лимит 60s)"
                )
                self.stats["rejected_orders"] += 1
                return SubmissionResult(
                    success=False,
                    reason=f"Signal expired (age: {signal.age_seconds:.1f}s)",
                    symbol=signal.symbol
                )

            # ==========================================
            # ШАГ 2: СИНХРОНИЗАЦИЯ ПОЗИЦИЙ С БИРЖЕЙ
            # ==========================================
            await self._sync_positions_with_exchange()

            # ==========================================
            # ШАГ 3: ПРОВЕРКА ЛИМИТА ПОЗИЦИЙ
            # ==========================================
            current_positions = self.risk_manager.metrics.open_positions_count
            max_positions = self.risk_manager.limits.max_open_positions

            if current_positions >= max_positions:
                logger.warning(
                    f"{signal.symbol} | 🛑 Лимит позиций достигнут: "
                    f"{current_positions}/{max_positions}"
                )
                self.stats["rejected_orders"] += 1
                return SubmissionResult(
                    success=False,
                    reason=f"Position limit reached ({current_positions}/{max_positions})",
                    symbol=signal.symbol
                )

            # Проверка дубликата по символу
            if signal.symbol in self.risk_manager.open_positions:
                logger.warning(
                    f"{signal.symbol} | 🛑 Позиция по этому символу уже открыта"
                )
                self.stats["rejected_orders"] += 1
                return SubmissionResult(
                    success=False,
                    reason="Position already exists for this symbol",
                    symbol=signal.symbol
                )

            # ==========================================
            # ШАГ 4: НЕМЕДЛЕННОЕ ИСПОЛНЕНИЕ
            # ==========================================
            logger.info(
                f"{signal.symbol} | ✅ Немедленное исполнение сигнала "
                f"(age={signal.age_seconds:.1f}s, positions={current_positions}/{max_positions})"
            )

            # Исполняем сигнал немедленно в текущем контексте
            await self._execute_signal(signal)

            return SubmissionResult(
                success=True,
                reason="Signal executed immediately",
                symbol=signal.symbol
            )

        except Exception as e:
            logger.error(
                f"{signal.symbol} | ❌ Ошибка при исполнении сигнала: {e}",
                exc_info=True
            )
            self.stats["failed_orders"] += 1
            return SubmissionResult(
                success=False,
                reason=f"Execution error: {str(e)}",
                symbol=signal.symbol
)

    async def _sync_positions_with_exchange(self):
        """
        Синхронизация локальных позиций с реальным состоянием на бирже.

        Критично для случаев когда:
        - Позиции закрыты вручную через биржу
        - Произошел рестарт бота
        - WebSocket пропустил событие
        """
        try:
            logger.debug("🔄 Синхронизация позиций с биржей...")

            # Запрашиваем актуальные позиции с биржи
            response = await rest_client.get_positions()

            # ✅ ДОБАВЛЕНО: Детальное логирование для отладки
            logger.debug(f"🔍 Тип ответа от get_positions: {type(response)}")
            logger.debug(f"🔍 Содержимое ответа: {response}")

            # ✅ ИСПРАВЛЕНО: Правильная обработка ответа от Bybit API
            # Bybit API возвращает: {"result": {"list": [...]}}
            if not response:
                logger.debug("Нет открытых позиций на бирже (пустой ответ)")
                exchange_positions = []
            elif isinstance(response, dict):
                # Если response это dict с результатом
                result = response.get("result", {})
                if isinstance(result, dict):
                    exchange_positions = result.get("list", [])
                elif isinstance(result, list):
                    exchange_positions = result
                else:
                    logger.warning(f"⚠️ Неожиданный формат result: {type(result)}")
                    exchange_positions = []
            elif isinstance(response, list):
                # Если response уже список позиций
                exchange_positions = response
            else:
                logger.warning(f"⚠️ Неожиданный тип ответа: {type(response)}")
                exchange_positions = []

            # ✅ ДОБАВЛЕНО: Логирование после парсинга
            logger.debug(f"📊 Распарсено позиций: {len(exchange_positions)}")

            if not exchange_positions:
                logger.debug("Нет открытых позиций на бирже")
                # Очищаем локальный стейт если на бирже пусто
                if self.risk_manager.open_positions:
                    logger.warning(
                        f"⚠️ Локально {len(self.risk_manager.open_positions)} позиций, "
                        f"на бирже 0 → очищаем локальный стейт"
                    )
                    self.risk_manager.open_positions.clear()
                    self.risk_manager.metrics.open_positions_count = 0
                    self.risk_manager.metrics.total_exposure_usdt = 0.0
                return

            # ✅ ИСПРАВЛЕНО: Безопасная обработка позиций с проверкой типов
            exchange_symbols = set()

            for pos in exchange_positions:
                # Проверяем что pos это dict
                if not isinstance(pos, dict):
                    logger.warning(f"⚠️ Позиция не dict: {type(pos)} = {pos}")
                    continue

                symbol = pos.get("symbol")
                size = pos.get("size", "0")

                # Безопасное преобразование size в float
                try:
                    size_float = float(size)
                except (ValueError, TypeError):
                    logger.warning(f"⚠️ Некорректный size для {symbol}: {size}")
                    continue

                if size_float > 0:
                    exchange_symbols.add(symbol)
                    logger.debug(f"  Позиция на бирже: {symbol}, size={size_float}")

            # Получаем символы из локального стейта
            local_symbols = set(self.risk_manager.open_positions.keys())

            logger.info(
                f"📊 Сравнение: локально={len(local_symbols)}, "
                f"на бирже={len(exchange_symbols)}"
            )

            # Находим расхождения
            missing_locally = exchange_symbols - local_symbols  # На бирже есть, локально нет
            missing_on_exchange = local_symbols - exchange_symbols  # Локально есть, на бирже нет

            # Удаляем локальные позиции которых нет на бирже
            for symbol in missing_on_exchange:
                logger.warning(
                    f"⚠️ Позиция {symbol} закрыта на бирже, удаляем из локального стейта"
                )
                self.risk_manager.register_position_closed(symbol)

            # Добавляем позиции которые есть на бирже но нет локально
            for symbol in missing_locally:
                # Находим данные позиции
                pos_data = next(
                    (p for p in exchange_positions
                     if isinstance(p, dict) and p.get("symbol") == symbol),
                    None
                )

                if pos_data:
                    try:
                        size = float(pos_data.get("size", 0))
                        entry_price = float(pos_data.get("avgPrice", 0))
                        side_str = pos_data.get("side", "Buy")

                        # Конвертируем side в SignalType
                        side = SignalType.BUY if side_str == "Buy" else SignalType.SELL

                        logger.warning(
                            f"⚠️ Позиция {symbol} найдена на бирже, добавляем в локальный стейт"
                        )

                        self.risk_manager.register_position_opened(
                            symbol=symbol,
                            side=side,
                            size_usdt=size * entry_price,
                            entry_price=entry_price,
                            leverage=10
                        )
                    except (ValueError, TypeError, KeyError) as e:
                        logger.error(f"❌ Ошибка добавления позиции {symbol}: {e}")

            logger.debug(
                f"✓ Синхронизация завершена: "
                f"локально={len(self.risk_manager.open_positions)}, "
                f"на бирже={len(exchange_symbols)}"
            )

        except Exception as e:
            logger.error(f"❌ Ошибка синхронизации позиций: {e}", exc_info=True)
            # Не падаем, продолжаем работу с текущим стейтом

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
            # ==========================================
            # ШАГ 0: ВАЛИДАЦИЯ РАЗМЕРА ОРДЕРА
            # ==========================================
            notional_value = quantity * entry_price
            min_order_value = settings.MIN_ORDER_SIZE_USDT

            if notional_value < min_order_value:
                logger.error(
                    f"→ Открытие позиции: {symbol} {side} {quantity} @ {entry_price} "
                f"(notional: {notional_value:.2f} USDT)"
            )
                return None
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
                # ==========================================
                # ШАГ 1.5: УСТАНОВКА LEVERAGE
                # ==========================================
                # Получаем leverage из risk_manager
                leverage = self.risk_manager.limits.default_leverage
                leverage_str = str(leverage)

                try:
                    # Пытаемся установить leverage для символа
                    await rest_client.set_leverage(
                        symbol=symbol,
                        buy_leverage=leverage_str,
                        sell_leverage=leverage_str
                    )
                    logger.info(f"✓ Leverage установлен для {symbol}: {leverage}x")
                except Exception as leverage_error:
                    # Если не удалось установить - пытаемся с меньшим leverage
                    error_msg = str(leverage_error)

                    # Код 110043 - "leverage not modified" означает что leverage уже установлен
                    # Это НОРМАЛЬНАЯ ситуация, не требующая действий
                    if "110043" in error_msg or "leverage not modified" in error_msg:
                        logger.debug(f"✓ Leverage для {symbol} уже установлен на {leverage}x")
                        # Продолжаем работу - это не ошибка

                    # Проверяем если это ошибка превышения максимального leverage
                    elif "maxLeverage" in error_msg or "110013" in error_msg:
                        # Извлекаем максимальное плечо из ошибки
                        # Формат: "cannot set leverage [2500] gt maxLeverage [2000]"
                        import re
                        max_lev_match = re.search(r'maxLeverage \[(\d+)\]', error_msg)
                        if max_lev_match:
                            max_leverage_api = int(max_lev_match.group(1))
                            # API формат: 2000 = 20.00x, конвертируем
                            max_leverage = max_leverage_api // 100

                            logger.warning(
                                f"⚠️ Leverage {leverage}x превышает максимум {max_leverage}x для {symbol}. "
                                f"Использую максимально допустимое плечо {max_leverage}x"
                            )

                            # Устанавливаем максимально допустимое плечо
                            try:
                                await rest_client.set_leverage(
                                    symbol=symbol,
                                    buy_leverage=str(max_leverage),
                                    sell_leverage=str(max_leverage)
                                )
                                logger.info(f"✓ Leverage установлен для {symbol}: {max_leverage}x (максимум)")
                                leverage = max_leverage  # Обновляем для использования далее
                            except Exception as e2:
                                logger.error(f"❌ Не удалось установить leverage {max_leverage}x: {e2}")
                                # Продолжаем с дефолтным leverage биржи
                        else:
                            logger.warning(f"⚠️ Не удалось извлечь max leverage из ошибки: {error_msg}")
                    else:
                        # Другие ошибки leverage - логируем и продолжаем
                        logger.warning(f"⚠️ Не удалось установить leverage для {symbol}: {leverage_error}")
                # Размещаем ордер на бирже
                logger.info(
                    f"📊 Параметры TP/SL для {symbol}:\n"
                    f"  Entry Price:  {entry_price}\n"
                    f"  Stop Loss:    {stop_loss}\n"
                    f"  Take Profit:  {take_profit}\n"
                    f"  Leverage:     {leverage}x"
                )


                try:
                    order_response = await rest_client.place_order(
                        symbol=symbol,
                        side=side,
                        order_type="Market",  # или "Limit" в зависимости от стратегии
                        quantity=quantity,
                        price=entry_price if side == "Limit" else None,
                        stop_loss=stop_loss,  # ✅ ДОБАВЛЕНО
                        take_profit=take_profit,  # ✅ ДОБАВЛЕНО
                        client_order_id=client_order_id
                    )

                    result = order_response.get("result", {})
                    exchange_order_id = result.get("orderId")
                    order_link_id = result.get("orderLinkId")

                    logger.info(
                        f"✓ Ордер размещен на бирже С TP/SL:\n"
                        f"  Exchange Order ID: {result.get('orderId')}\n"
                        f"  Stop Loss:   {stop_loss}\n"
                        f"  Take Profit: {take_profit}"
                    )

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
                # ШАГ 2.6: ЗАПИСЬ ДЕТАЛЬНОГО ОТЧЁТА (trades.log)
                # ==========================================
                try:
                    # entry_signal - это dict из signal.to_dict()
                    # strategy_results и ml_validation_result находятся внутри metadata
                    signal_data = entry_signal if entry_signal else {}
                    metadata = signal_data.get('metadata', {})
                    strategy_results = metadata.get('strategy_results')
                    ml_validation_result = metadata.get('ml_validation_result')

                    # Создаём и записываем отчёт
                    trade_report = trade_reporter.create_report_from_dict(
                        signal_dict=signal_data,
                        strategy_results=strategy_results,
                        ml_validation_result=ml_validation_result,
                        sl_price=stop_loss,
                        tp_price=take_profit,
                        position_size=float(quantity) * entry_price
                    )

                    trade_reporter.log_trade(trade_report)

                    logger.info(f"📝 Детальный отчёт записан в trades.log")

                except Exception as report_error:
                    logger.warning(f"⚠ Ошибка записи отчёта: {report_error}")
                    # Не критично - продолжаем

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
                # ШАГ 6: РЕГИСТРАЦИЯ В TRAILING STOP MANAGER
                # ==========================================
                trailing_stop_manager.register_position_opened(
                    symbol=symbol,
                    position_id=str(position.id),
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    side=order_side
                )

                logger.debug(
                    f"Позиция {symbol} зарегистрирована в Trailing Stop Manager"
                )

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

                # ==========================================
                # ЛОГИРОВАНИЕ В trades.log - POSITION_CLOSED
                # ==========================================
                pnl_percent = ((exit_price - position.entry_price) / position.entry_price) * 100 if position.side == OrderSide.BUY else ((position.entry_price - exit_price) / position.entry_price) * 100
                trade_reporter.log_position_event(PositionEvent(
                    event_type=PositionEventType.POSITION_CLOSED,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    position_id=position_id,
                    side=position.side.value,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    quantity=position.quantity,
                    realized_pnl=realized_pnl,
                    unrealized_pnl_percent=pnl_percent,
                    exit_reason=exit_reason,
                    reason=f"PnL: ${realized_pnl:+.2f} ({pnl_percent:+.2f}%)"
                ))

                is_win = realized_pnl > 0
                # ===== Записываем результат для Adaptive Risk =====
                self.risk_manager.record_trade_result(
                    is_win=is_win,
                    pnl=realized_pnl
                )

                logger.info(
                    f"{position.symbol} | Trade result recorded: "
                    f"win={is_win}, pnl={realized_pnl:.2f} USDT"
                )


                # 6. УДАЛЕНИЕ ИЗ RISK MANAGER
                # ✅ ИСПРАВЛЕНО: Убран аргумент realized_pnl
                self.risk_manager.register_position_closed(symbol=symbol)

                # ==========================================
                # ШАГ 6.5: УДАЛЕНИЕ ИЗ TRAILING STOP MANAGER
                # ==========================================
                trailing_stop_manager.register_position_closed(symbol)

                logger.debug(
                    f"Позиция {symbol} удалена из Trailing Stop Manager"
                )

                # ========================================
                # ✅ ШАГ 6.5: ИНТЕГРАЦИЯ С ADAPTIVE CONSENSUS MANAGER
                # ========================================
                if self.adaptive_consensus_manager:
                    try:
                        # Получаем метаданные из позиции
                        contributing_strategies = position.metadata.get('contributing_strategies', [])
                        signal_timestamp = position.metadata.get('signal_timestamp')

                        # Логируем полученные метаданные
                        logger.debug(
                            f"📊 Position metadata: "
                            f"strategies={contributing_strategies}, "
                            f"signal_ts={signal_timestamp}"
                        )

                        # Проверяем наличие необходимых данных
                        if contributing_strategies and signal_timestamp:
                            # Текущий timestamp выхода
                            exit_timestamp = int(datetime.now().timestamp() * 1000)

                            # Вызываем метод записи результата
                            self.adaptive_consensus_manager.record_signal_outcome(
                                symbol=position.symbol,
                                signal_timestamp=signal_timestamp,
                                contributing_strategies=contributing_strategies,
                                exit_price=exit_price,
                                exit_timestamp=exit_timestamp,
                                pnl_usdt=realized_pnl
                            )

                            logger.info(
                                f"📊 Performance recorded for Adaptive Consensus: "
                                f"{position.symbol}, "
                                f"PnL={realized_pnl:+.2f} USDT, "
                                f"strategies={', '.join(contributing_strategies)}, "
                                f"hold_time={(exit_timestamp - signal_timestamp) / 1000:.0f}s"
                            )
                        else:
                            # Логируем отсутствие данных
                            if not contributing_strategies:
                                logger.debug(
                                    f"📊 No contributing_strategies in position metadata for {symbol}"
                                )
                            if not signal_timestamp:
                                logger.debug(
                                    f"📊 No signal_timestamp in position metadata for {symbol}"
                                )

                    except Exception as e:
                        # Не падаем, если запись в adaptive consensus не удалась
                        logger.error(
                            f"❌ Ошибка при записи результата в Adaptive Consensus: {e}",
                            exc_info=True
                        )

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

    async def partial_close_position(
        self,
        position_id: str,
        close_percentage: float,
        exit_price: float,
        exit_reason: str = "Partial close"
    ) -> Optional[dict]:
        """
        Частичное закрытие позиции.

        Args:
            position_id: ID позиции
            close_percentage: Процент закрытия (0.0 - 1.0, например 0.5 для 50%)
            exit_price: Цена закрытия
            exit_reason: Причина частичного закрытия

        Returns:
            Dict: {
                'position_id': str,
                'closed_quantity': float,
                'remaining_quantity': float,
                'partial_pnl': float,
                'status': 'success'
            }
        """
        with trace_operation("partial_close_position", position_id=position_id):
            logger.info(
                f"→ Частичное закрытие позиции: {position_id} @ {close_percentage:.0%}"
            )

            try:
                # ==========================================
                # ШАГ 1: ПОЛУЧЕНИЕ ПОЗИЦИИ
                # ==========================================
                position = await position_repository.get_by_id(position_id)

                if not position:
                    logger.error(f"Позиция {position_id} не найдена")
                    return None

                symbol = position.symbol

                # ==========================================
                # ШАГ 2: ВАЛИДАЦИЯ
                # ==========================================
                if position.status != PositionStatus.OPEN:
                    logger.error(
                        f"Позиция {position_id} не в статусе OPEN "
                        f"(текущий: {position.status.value})"
                    )
                    return None

                if not (0.0 < close_percentage < 1.0):
                    logger.error(
                        f"Некорректный close_percentage: {close_percentage}. "
                        f"Должен быть между 0 и 1"
                    )
                    return None

                # ==========================================
                # ШАГ 3: РАСЧЕТ КОЛИЧЕСТВА ДЛЯ ЗАКРЫТИЯ
                # ==========================================
                close_quantity_raw = position.quantity * close_percentage

                # Получаем информацию об инструменте для округления
                instrument_info = await self._get_instrument_info(symbol)
                if not instrument_info:
                    logger.error(f"Не удалось получить instrument info для {symbol}")
                    return None

                # Округляем quantity
                close_quantity = self._validate_and_round_quantity(
                    symbol=symbol,
                    quantity=close_quantity_raw,
                    price=exit_price,
                    instrument_info=instrument_info
                )

                if close_quantity is None:
                    logger.error(f"Не удалось валидировать quantity для {symbol}")
                    return None

                logger.info(
                    f"{symbol} | Closing {close_quantity:.8f} "
                    f"({close_percentage:.0%} of {position.quantity:.8f})"
                )

                # ==========================================
                # ШАГ 4: РАЗМЕЩЕНИЕ ЗАКРЫВАЮЩЕГО ОРДЕРА
                # ==========================================
                # Определяем противоположную сторону
                if position.side == OrderSide.BUY:
                    close_side = "Sell"  # Закрываем long
                else:
                    close_side = "Buy"   # Закрываем short

                logger.info(
                    f"📡 Размещение закрывающего ордера: "
                    f"{symbol} {close_side} {close_quantity:.8f}"
                )

                # Генерируем client_order_id
                client_order_id = idempotency_service.generate_client_order_id(
                    symbol=symbol,
                    side=close_side,
                    quantity=close_quantity,
                    price=exit_price
                )

                # Размещаем ордер на бирже
                order_response = await rest_client.place_order(
                    symbol=symbol,
                    side=close_side,
                    order_type="Market",
                    quantity=close_quantity,
                    client_order_id=client_order_id
                )

                result = order_response.get("result", {})
                exchange_order_id = result.get("orderId")

                if not exchange_order_id:
                    raise OrderExecutionError("Биржа не вернула orderId")

                logger.info(
                    f"✓ Закрывающий ордер размещен: {exchange_order_id}"
                )

                # ==========================================
                # ШАГ 5: РАСЧЕТ PARTIAL PNL
                # ==========================================
                if position.side == OrderSide.BUY:
                    partial_pnl = (exit_price - position.entry_price) * close_quantity
                else:
                    partial_pnl = (position.entry_price - exit_price) * close_quantity

                logger.info(f"💰 Partial PnL: ${partial_pnl:+.2f}")

                # ==========================================
                # ЛОГИРОВАНИЕ В trades.log - PARTIAL_CLOSE
                # ==========================================
                pnl_percent = ((exit_price - position.entry_price) / position.entry_price) * 100 if position.side == OrderSide.BUY else ((position.entry_price - exit_price) / position.entry_price) * 100
                trade_reporter.log_position_event(PositionEvent(
                    event_type=PositionEventType.PARTIAL_CLOSE,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    position_id=position_id,
                    side=position.side.value,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    quantity=close_quantity,
                    close_percentage=close_percentage,
                    realized_pnl=partial_pnl,
                    unrealized_pnl_percent=pnl_percent,
                    exit_reason=exit_reason,
                    reason=f"Closed {close_percentage:.0%}, PnL: ${partial_pnl:+.2f}"
                ))

                # ==========================================
                # ШАГ 6: ОБНОВЛЕНИЕ ПОЗИЦИИ В БД
                # ==========================================
                remaining_quantity = position.quantity - close_quantity

                # Получаем текущую metadata
                current_metadata = position.metadata or {}

                # Добавляем информацию о частичном закрытии
                if 'partial_closes' not in current_metadata:
                    current_metadata['partial_closes'] = []

                current_metadata['partial_closes'].append({
                    'timestamp': int(datetime.now().timestamp() * 1000),
                    'close_percentage': close_percentage,
                    'closed_quantity': close_quantity,
                    'exit_price': exit_price,
                    'partial_pnl': partial_pnl,
                    'exchange_order_id': exchange_order_id,
                    'reason': exit_reason
                })

                # Обновляем quantity в БД
                from backend.database.database import db_manager
                from sqlalchemy import update
                from backend.database.models import Position

                async with db_manager.session() as session:
                    stmt = (
                        update(Position)
                        .where(Position.id == position_id)
                        .values(
                            quantity=remaining_quantity,
                            metadata_json=current_metadata,
                            updated_at=datetime.utcnow()
                        )
                    )
                    await session.execute(stmt)
                    await session.commit()

                logger.info(
                    f"✓ Позиция обновлена: quantity {position.quantity:.8f} → "
                    f"{remaining_quantity:.8f}"
                )

                # ==========================================
                # ШАГ 7: ОБНОВЛЕНИЕ RISK MANAGER
                # ==========================================
                # Уменьшаем exposure
                closed_value = close_quantity * exit_price

                if symbol in self.risk_manager.open_positions:
                    position_data = self.risk_manager.open_positions[symbol]
                    current_exposure = position_data.get("size_usdt", 0)
                    new_exposure = current_exposure - closed_value

                    if new_exposure > 0:
                        position_data["size_usdt"] = new_exposure
                        logger.debug(
                            f"RiskManager exposure updated: "
                            f"{symbol} ${current_exposure:.2f} → ${new_exposure:.2f}"
                        )

                # Обновляем total exposure
                self.risk_manager.metrics.total_exposure_usdt -= closed_value

                # ==========================================
                # ШАГ 8: ЗАПИСЬ РЕЗУЛЬТАТА ДЛЯ ADAPTIVE RISK
                # ==========================================
                is_win = partial_pnl > 0
                self.risk_manager.record_trade_result(
                    is_win=is_win,
                    pnl=partial_pnl
                )

                logger.info(
                    f"{symbol} | Partial close result recorded: "
                    f"win={is_win}, pnl={partial_pnl:.2f} USDT"
                )

                # ==========================================
                # ШАГ 9: AUDIT LOG
                # ==========================================
                await audit_repository.log(
                    action=AuditAction.POSITION_UPDATE,
                    entity_type="Position",
                    entity_id=position_id,
                    new_value={
                        "action": "partial_close",
                        "close_percentage": close_percentage,
                        "closed_quantity": close_quantity,
                        "remaining_quantity": remaining_quantity,
                        "exit_price": exit_price,
                        "partial_pnl": partial_pnl,
                        "exchange_order_id": exchange_order_id
                    },
                    reason=exit_reason,
                    success=True
                )

                logger.info(
                    f"✓✓✓ PARTIAL CLOSE COMPLETED ✓✓✓\n"
                    f"  Position ID: {position_id}\n"
                    f"  Symbol: {symbol}\n"
                    f"  Closed: {close_quantity:.8f} ({close_percentage:.0%})\n"
                    f"  Remaining: {remaining_quantity:.8f}\n"
                    f"  Exit Price: ${exit_price:.2f}\n"
                    f"  Partial PnL: ${partial_pnl:+.2f}"
                )

                return {
                    'position_id': position_id,
                    'closed_quantity': close_quantity,
                    'remaining_quantity': remaining_quantity,
                    'partial_pnl': partial_pnl,
                    'exchange_order_id': exchange_order_id,
                    'status': 'success'
                }

            except Exception as e:
                logger.error(
                    f"❌ Ошибка partial close позиции: {e}",
                    exc_info=True
                )
                return None

    async def update_stop_loss(
        self,
        position_id: str,
        new_stop_loss: float,
        reason: str = "Manual update"
    ) -> Optional[dict]:
        """
        Обновление Stop Loss для открытой позиции.

        Args:
            position_id: ID позиции
            new_stop_loss: Новый уровень Stop Loss
            reason: Причина обновления

        Returns:
            Dict: {
                'position_id': str,
                'symbol': str,
                'old_stop_loss': float,
                'new_stop_loss': float,
                'status': 'success'
            }
        """
        with trace_operation("update_stop_loss", position_id=position_id):
            logger.info(
                f"→ Обновление SL для позиции: {position_id} → ${new_stop_loss:.2f}"
            )

            try:
                # ==========================================
                # ШАГ 1: ПОЛУЧЕНИЕ ПОЗИЦИИ
                # ==========================================
                position = await position_repository.get_by_id(position_id)

                if not position:
                    logger.error(f"Позиция {position_id} не найдена")
                    return None

                symbol = position.symbol
                old_stop_loss = position.stop_loss

                # ==========================================
                # ШАГ 2: ВАЛИДАЦИЯ СТАТУСА
                # ==========================================
                if position.status != PositionStatus.OPEN:
                    logger.error(
                        f"Позиция {position_id} не в статусе OPEN "
                        f"(текущий: {position.status.value})"
                    )
                    return None

                # ==========================================
                # ШАГ 3: ВАЛИДАЦИЯ SL LOGIC
                # ==========================================
                # Получаем текущую цену
                current_price = None
                try:
                    ticker = await rest_client.get_ticker(symbol=symbol)
                    result = ticker.get("result", {})
                    if isinstance(result, dict):
                        ticker_list = result.get("list", [])
                        if ticker_list:
                            current_price = float(ticker_list[0].get("lastPrice", 0))
                except Exception as e:
                    logger.warning(f"Не удалось получить текущую цену: {e}")

                # Валидация для LONG
                if position.side == OrderSide.BUY:
                    if current_price and new_stop_loss >= current_price:
                        logger.error(
                            f"Некорректный SL для LONG: "
                            f"new_sl={new_stop_loss:.2f} >= current_price={current_price:.2f}"
                        )
                        return None

                    if new_stop_loss >= position.entry_price:
                        logger.warning(
                            f"SL выше entry price для LONG: "
                            f"new_sl={new_stop_loss:.2f} >= entry={position.entry_price:.2f} "
                            f"(будет в прибыли)"
                        )

                # Валидация для SHORT
                else:
                    if current_price and new_stop_loss <= current_price:
                        logger.error(
                            f"Некорректный SL для SHORT: "
                            f"new_sl={new_stop_loss:.2f} <= current_price={current_price:.2f}"
                        )
                        return None

                    if new_stop_loss <= position.entry_price:
                        logger.warning(
                            f"SL ниже entry price для SHORT: "
                            f"new_sl={new_stop_loss:.2f} <= entry={position.entry_price:.2f} "
                            f"(будет в прибыли)"
                        )

                logger.debug(f"{symbol} | SL validation passed ✓")

                # ==========================================
                # ШАГ 4: ОБНОВЛЕНИЕ SL НА БИРЖЕ
                # ==========================================
                logger.info(f"📡 Обновление SL на бирже для {symbol}")

                try:
                    update_response = await rest_client.set_trading_stop(
                        symbol=symbol,
                        stop_loss=new_stop_loss,
                        take_profit=None  # Не трогаем TP
                    )

                    logger.info(
                        f"✓ SL обновлен на бирже: {symbol} | "
                        f"${old_stop_loss:.2f} → ${new_stop_loss:.2f}"
                    )

                except Exception as exchange_error:
                    logger.error(
                        f"❌ Ошибка обновления SL на бирже: {exchange_error}",
                        exc_info=True
                    )
                    return None

                # ==========================================
                # ШАГ 5: ОБНОВЛЕНИЕ SL В БД
                # ==========================================
                success = await position_repository.update_stop_loss(
                    position_id=position_id,
                    new_stop_loss=new_stop_loss
                )

                if not success:
                    logger.error(f"Не удалось обновить SL в БД для {position_id}")
                    # SL обновлен на бирже, но не в БД - это не критично
                else:
                    logger.info(f"✓ SL обновлен в БД: {position_id}")

                # ==========================================
                # ШАГ 6: ОБНОВЛЕНИЕ В TRAILING STOP MANAGER
                # ==========================================
                try:
                    if symbol in trailing_stop_manager.tracked_positions:
                        trailing_stop_manager.tracked_positions[symbol]['stop_loss'] = new_stop_loss
                        logger.debug(f"TrailingStopManager updated for {symbol}")
                except Exception as tsm_error:
                    logger.warning(f"Не удалось обновить TrailingStopManager: {tsm_error}")

                # ==========================================
                # ШАГ 7: AUDIT LOG
                # ==========================================
                await audit_repository.log(
                    action=AuditAction.POSITION_UPDATE,
                    entity_type="Position",
                    entity_id=position_id,
                    old_value={'stop_loss': old_stop_loss},
                    new_value={'stop_loss': new_stop_loss},
                    reason=reason,
                    success=True
                )

                logger.info(
                    f"✓✓✓ STOP LOSS UPDATED ✓✓✓\n"
                    f"  Position ID: {position_id}\n"
                    f"  Symbol: {symbol}\n"
                    f"  Old SL: ${old_stop_loss:.2f}\n"
                    f"  New SL: ${new_stop_loss:.2f}\n"
                    f"  Reason: {reason}"
                )

                # ==========================================
                # ЛОГИРОВАНИЕ В trades.log - STOP_LOSS_UPDATED
                # ==========================================
                trade_reporter.log_position_event(PositionEvent(
                    event_type=PositionEventType.STOP_LOSS_UPDATED,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    position_id=position_id,
                    side=position.side.value,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    old_stop_loss=old_stop_loss,
                    new_stop_loss=new_stop_loss,
                    reason=reason
                ))

                return {
                    'position_id': position_id,
                    'symbol': symbol,
                    'old_stop_loss': old_stop_loss,
                    'new_stop_loss': new_stop_loss,
                    'status': 'success'
                }

            except Exception as e:
                logger.error(
                    f"❌ Ошибка обновления SL: {e}",
                    exc_info=True
                )
                return None

    # ==================== ПРИВАТНЫЕ МЕТОДЫ ====================

    # async def _process_queue(self):
    #     """Обработка очереди сигналов."""
    #     logger.info("Запущена обработка очереди исполнения")
    #
    #     while self.is_running:
    #         try:
    #             # Получаем сигнал из очереди с таймаутом
    #             try:
    #                 signal = await asyncio.wait_for(
    #                     self.signal_queue.get(),
    #                     timeout=1.0
    #                 )
    #             except asyncio.TimeoutError:
    #                 continue
    #
    #             # Обрабатываем сигнал
    #             await self._execute_signal(signal)
    #
    #         except Exception as e:
    #             logger.error(f"Ошибка обработки очереди исполнения: {e}")
    #             await asyncio.sleep(1)

    # async def _process_queue(self):
    #     """
    #     Обработка очереди сигналов на исполнение.
    #
    #     ИСПРАВЛЕНИЯ:
    #     - Проверка типа signal перед обработкой
    #     - Обработка некорректных объектов в очереди
    #     - Детальное логирование для диагностики
    #     """
    #     logger.info("Запущена обработка очереди исполнения")
    #
    #     while self.is_running:
    #         try:
    #             # ==========================================
    #             # ШАГ 1: ПОЛУЧЕНИЕ СИГНАЛА ИЗ ОЧЕРЕДИ
    #             # ==========================================
    #             try:
    #                 signal = await asyncio.wait_for(
    #                     self.signal_queue.get(),
    #                     timeout=1.0
    #                 )
    #             except asyncio.TimeoutError:
    #                 # Таймаут - это нормально, просто продолжаем ждать
    #                 continue
    #
    #             # ==========================================
    #             # ШАГ 2: КРИТИЧЕСКАЯ ВАЛИДАЦИЯ ТИПА
    #             # ==========================================
    #             if signal is None:
    #                 logger.warning("Получен None из очереди, пропускаем")
    #                 continue
    #
    #             # КРИТИЧНО: Проверяем что это TradingSignal
    #             if not isinstance(signal, TradingSignal):
    #                 logger.error(
    #                     f"❌ КРИТИЧЕСКАЯ ОШИБКА: Неверный тип объекта в очереди! "
    #                     f"Ожидался: TradingSignal, "
    #                     f"Получен: {type(signal).__name__}"
    #                 )
    #
    #                 # Пытаемся вывести содержимое для диагностики
    #                 try:
    #                     logger.error(f"Содержимое объекта: {signal}")
    #                 except Exception as e:
    #                     logger.error(f"Не удалось вывести содержимое: {e}")
    #
    #                 # Пропускаем некорректный объект
    #                 continue
    #
    #             # ==========================================
    #             # ШАГ 3: ВАЛИДАЦИЯ ОБЯЗАТЕЛЬНЫХ АТРИБУТОВ
    #             # ==========================================
    #             try:
    #                 # Проверяем наличие критических атрибутов
    #                 required_attrs = ['symbol', 'signal_type', 'price']
    #                 missing_attrs = [attr for attr in required_attrs if not hasattr(signal, attr)]
    #
    #                 if missing_attrs:
    #                     logger.error(
    #                         f"❌ TradingSignal не содержит обязательных атрибутов: {missing_attrs}. "
    #                         f"Пропускаем сигнал."
    #                     )
    #                     continue
    #
    #                 # Проверяем что signal_type это Enum, а не строка
    #                 if hasattr(signal.signal_type, 'value'):
    #                     signal_type_value = signal.signal_type.value
    #                 else:
    #                     signal_type_value = str(signal.signal_type)
    #
    #                 logger.debug(
    #                     f"✓ Валидный сигнал получен: "
    #                     f"symbol={signal.symbol}, "
    #                     f"type={signal_type_value}, "
    #                     f"price={signal.price:.8f}"
    #                 )
    #
    #             except Exception as e:
    #                 logger.error(
    #                     f"❌ Ошибка валидации атрибутов сигнала: {e}",
    #                     exc_info=True
    #                 )
    #                 continue
    #
    #             # ==========================================
    #             # ШАГ 4: ОБРАБОТКА СИГНАЛА
    #             # ==========================================
    #             try:
    #                 await self._execute_signal(signal)
    #             except Exception as e:
    #                 logger.error(
    #                     f"❌ Ошибка исполнения сигнала {signal.symbol}: {e}",
    #                     exc_info=True
    #                 )
    #                 # Продолжаем обработку следующих сигналов
    #
    #         except Exception as e:
    #             logger.error(
    #                 f"❌ Критическая ошибка в цикле обработки очереди: {e}",
    #                 exc_info=True
    #             )
    #             # Небольшая задержка для предотвращения зацикливания при критических ошибках
    #             await asyncio.sleep(1)
    #
    #     logger.info("Обработка очереди исполнения остановлена")

    async def _execute_signal(self, signal: TradingSignal):
        """
        Исполнение торгового сигнала с ML-enhanced risk management.

        ИСПРАВЛЕННАЯ ВЕРСИЯ:
        - SL/TP рассчитывается ТОЛЬКО в validate_signal_ml_enhanced
        - Fallback расчет SL/TP если ML недоступна
        - Все .value заменены на safe_enum_value()
        - Проверка лимитов позиций перенесена в submit_signal()



        Pipeline:
        0. Дедупликация сигнала

        1. Получение информации об инструменте
        2. Проверка баланса
        2.5. Извлечение ML features
        3. Валидация signal_type
        4. ML-enhanced validation (рассчитывает SL/TP внутри)
        4.1. Fallback расчет SL/TP (если ML недоступна)
        5. Расчет размера позиции
        6. Валидация и округление quantity
        7. Открытие позиции

        Args:
            signal: Торговый сигнал для исполнения
        """
        # ============================================
        # ШАГ 0.0: ПРОВЕРКА ЛИМИТА ПОЗИЦИЙ
        # ============================================
        await self._sync_positions_with_exchange()

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
            f"{signal.symbol} | Исполнение сигнала: {safe_enum_value(signal.signal_type)} @ {signal.price:.8f}"
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
            # ШАГ 2.5: ИЗВЛЕЧЕНИЕ ML FEATURES
            # ==========================================
            feature_vector = None

            # Попытка 1: Из метаданных сигнала
            if signal.metadata and 'ml_features' in signal.metadata:
                feature_vector = signal.metadata['ml_features']
                logger.debug(f"{signal.symbol} | ML features из signal metadata")

            # Попытка 2: Из bot_controller cache
            if not feature_vector:
                try:
                    from main import bot_controller
                    if hasattr(bot_controller, 'latest_features'):
                        feature_vector = bot_controller.latest_features.get(signal.symbol)
                        if feature_vector:
                            logger.debug(
                                f"{signal.symbol} | ML features из bot_controller cache"
                            )
                except Exception as e:
                    logger.debug(
                        f"{signal.symbol} | Не удалось получить cached features: {e}"
                    )

            # Попытка 3: Извлечь on-the-fly
            if not feature_vector:
                try:
                    from main import bot_controller

                    if (hasattr(bot_controller, 'ml_feature_pipeline') and
                        hasattr(bot_controller, 'orderbook_managers') and
                        hasattr(bot_controller, 'candle_managers')):

                        pipeline = bot_controller.ml_feature_pipeline
                        orderbook_manager = bot_controller.orderbook_managers.get(signal.symbol)
                        if not orderbook_manager:
                            raise ValueError(f"OrderBook manager для {signal.symbol} не найден")

                        orderbook_snapshot = orderbook_manager.get_snapshot()
                        if not orderbook_snapshot:
                            raise ValueError(f"OrderBook snapshot для {signal.symbol} недоступен")

                        candle_manager = bot_controller.candle_managers.get(signal.symbol)
                        if not candle_manager:
                            raise ValueError(f"Candle manager для {signal.symbol} не найден")

                        candles = candle_manager.get_candles()
                        if not candles or len(candles) == 0:
                            raise ValueError(f"Candles для {signal.symbol} недоступны")

                        feature_vector = await pipeline.extract_features_single(
                            symbol=signal.symbol,
                            orderbook_snapshot=orderbook_snapshot,
                            candles=candles
                        )

                        if feature_vector:
                            logger.debug(
                                f"{signal.symbol} | ML features извлечены on-the-fly: "
                                f"{feature_vector.feature_count} признаков"
                            )

                except Exception as e:
                    logger.debug(
                        f"{signal.symbol} | Failed to extract ML features on-the-fly: {e}"
                    )

            if not feature_vector:
                logger.debug(
                    f"{signal.symbol} | ML features недоступны, будет использован fallback"
                )

            # ==========================================
            # ШАГ 3: ВАЛИДАЦИЯ SIGNAL_TYPE И ОПРЕДЕЛЕНИЕ SIDE
            # ==========================================
            if signal.signal_type == SignalType.HOLD:
                logger.info(
                    f"{signal.symbol} | HOLD сигнал - не требует исполнения ордера"
                )
                return

            if signal.signal_type not in [SignalType.BUY, SignalType.SELL]:
                logger.warning(
                    f"{signal.symbol} | Неизвестный signal_type: {safe_enum_value(signal.signal_type)}, "
                    f"пропускаем исполнение"
                )
                self.stats["rejected_orders"] += 1
                return

            # Определение side для API биржи
            if signal.signal_type == SignalType.BUY:
                side = "Buy"
            elif signal.signal_type == SignalType.SELL:
                side = "Sell"
            else:
                logger.error(
                    f"{signal.symbol} | Недопустимый signal_type: {safe_enum_value(signal.signal_type)}"
                )
                self.stats["failed_orders"] += 1
                return

            logger.debug(f"{signal.symbol} | Side: {side}")

            # ==========================================
            # ШАГ 4: ПРОВЕРКА MTF PRE-CALCULATED ПАРАМЕТРОВ
            # ==========================================
            # CRITICAL FIX: Избегаем дублирования расчета SL/TP
            # Если сигнал от MTF - используем pre-calculated параметры
            # Если сигнал от single-TF - используем ML-enhanced validation
            # ==========================================

            ml_adjustments = None
            stop_loss = None
            take_profit = None
            entry_price = signal.price
            mtf_params_used = False

            # Проверяем наличие MTF pre-calculated параметров
            if signal.metadata and signal.metadata.get('has_mtf_risk_params'):
                # ========================================
                # ИСПОЛЬЗУЕМ MTF PRE-CALCULATED SL/TP
                # ========================================
                stop_loss = signal.metadata.get('mtf_recommended_stop_loss')
                take_profit = signal.metadata.get('mtf_recommended_take_profit')
                mtf_reliability = signal.metadata.get('mtf_reliability_score', 0.0)
                mtf_risk_level = signal.metadata.get('mtf_risk_level', 'UNKNOWN')
                mtf_quality = signal.metadata.get('mtf_signal_quality', 0.0)

                if stop_loss is not None and take_profit is not None:
                    mtf_params_used = True

                    logger.info(
                        f"{signal.symbol} | ✅ Используем MTF pre-calculated SL/TP | "
                        f"SL=${stop_loss:.2f}, "
                        f"TP=${take_profit:.2f}, "
                        f"R/R={(abs(take_profit - entry_price) / abs(entry_price - stop_loss)):.2f}, "
                        f"reliability={mtf_reliability:.3f}, "
                        f"risk_level={mtf_risk_level}, "
                        f"quality={mtf_quality:.3f}"
                    )

                    # Создаем ml_adjustments для совместимости с последующим кодом
                    # (хотя это не ML, но используем ту же структуру)
                    # Ограничиваем position_size_multiplier в допустимом диапазоне [0.5, 2.5]
                    raw_multiplier = signal.metadata.get('mtf_position_multiplier', 1.0)
                    clamped_multiplier = max(0.5, min(2.5, raw_multiplier))

                    ml_adjustments = MLRiskAdjustments(
                        position_size_multiplier=clamped_multiplier,
                        stop_loss_price=stop_loss,
                        take_profit_price=take_profit,
                        ml_confidence=mtf_reliability,  # Используем reliability как confidence
                        expected_return=(take_profit - entry_price) / entry_price,
                        market_regime=MarketRegime.MILD_TREND,  # Default, можно улучшить
                        manipulation_risk_score=0.0,
                        feature_quality=mtf_quality,
                        allow_entry=True,
                        rejection_reason=None
                    )
                else:
                    logger.warning(
                        f"{signal.symbol} | MTF params flag present but SL/TP not found, "
                        f"falling back to ML-enhanced validation"
                    )

            # ==========================================
            # ШАГ 4.1: ML-ENHANCED VALIDATION (FALLBACK)
            # Используется только если MTF параметры НЕ доступны
            # ==========================================
            if not mtf_params_used and hasattr(self.risk_manager, 'validate_signal_ml_enhanced') and feature_vector:
                try:
                    logger.debug(f"{signal.symbol} | Используем ML-enhanced validation")

                    # ML validation рассчитает SL/TP внутри
                    is_valid_ml, reason_ml, ml_adjustments = await self.risk_manager.validate_signal_ml_enhanced(
                        signal=signal,
                        balance=available_balance,
                        feature_vector=feature_vector
                    )

                    if not is_valid_ml:
                        logger.warning(
                            f"{signal.symbol} | ❌ ML-enhanced validation FAILED: {reason_ml}"
                        )
                        self.stats["rejected_orders"] += 1
                        return

                    # ========================================
                    # ИЗВЛЕКАЕМ SL/TP ИЗ ML_ADJUSTMENTS
                    # ========================================
                    stop_loss = ml_adjustments.stop_loss_price
                    take_profit = ml_adjustments.take_profit_price

                    logger.info(
                        f"{signal.symbol} | ✅ ML-enhanced validation PASSED | "
                        f"ML conf={ml_adjustments.ml_confidence:.2f}, "
                        f"SL=${stop_loss:.2f}, "
                        f"TP=${take_profit:.2f}, "
                        f"R/R={(abs(take_profit - entry_price) / abs(entry_price - stop_loss)):.2f}, "
                        f"Size mult={ml_adjustments.position_size_multiplier:.2f}x"
                    )

                except Exception as e:
                    logger.error(
                        f"{signal.symbol} | ML-enhanced validation error: {e}, "
                        f"falling back to standard SL/TP calculation",
                        exc_info=True
                    )
                    ml_adjustments = None
                    stop_loss = None
                    take_profit = None

            # ==========================================
            # ШАГ 4.1: FALLBACK РАСЧЕТ SL/TP
            # (если ML validation недоступна или произошла ошибка)
            # ==========================================
            if stop_loss is None or take_profit is None:
                # Проверяем, есть ли ML данные из signal validation (main.py)
                has_ml_metadata = (signal.metadata and
                                  signal.metadata.get('ml_validation_result') is not None)
                logger.info(
                    f"{signal.symbol} | ML-enhanced SL/TP не рассчитан, "
                    f"используем fallback (ML metadata: {'есть' if has_ml_metadata else 'нет'})"
                )

                try:
                    # Получаем дополнительные данные для расчета
                    atr = signal.metadata.get('atr') if signal.metadata else None

                    ml_sltp_data = None
                    # ml_validation_result хранится в metadata как dict, а не как атрибут signal
                    ml_validation_dict = signal.metadata.get('ml_validation_result') if signal.metadata else None
                    if ml_validation_dict:
                        ml_sltp_data = {
                            'predicted_mae': ml_validation_dict.get('predicted_mae') or 0.012,
                            'predicted_return': ml_validation_dict.get('ml_expected_return') or 0.0,
                            'confidence': ml_validation_dict.get('final_confidence') or 0.0
                        }
                        logger.debug(
                            f"{signal.symbol} | ML данные для SL/TP: "
                            f"mae={ml_sltp_data['predicted_mae']:.4f}, "
                            f"return={ml_sltp_data['predicted_return']:.4f}, "
                            f"confidence={ml_sltp_data['confidence']:.4f}"
                        )

                    market_regime_str = signal.metadata.get('market_regime') if signal.metadata else None
                    market_regime = None
                    if market_regime_str:
                        try:
                            if isinstance(market_regime_str, str):
                                market_regime = MarketRegime(market_regime_str)
                            else:
                                market_regime = market_regime_str
                        except (ValueError, AttributeError):
                            pass

                    # Расчет через UnifiedSLTPCalculator
                    logger.info(
                        f"{signal.symbol} | Fallback расчет SL/TP: "
                        f"entry=${entry_price:.2f}, "
                        f"has_atr={atr is not None}, "
                        f"has_regime={market_regime is not None}"
                    )

                    sltp_calc = sltp_calculator.calculate(
                        signal=signal,
                        entry_price=entry_price,
                        ml_result=ml_sltp_data,
                        atr=atr,
                        market_regime=market_regime
                    )

                    stop_loss = sltp_calc.stop_loss
                    take_profit = sltp_calc.take_profit

                    logger.info(
                        f"{signal.symbol} | Fallback SL/TP рассчитаны: "
                        f"method={sltp_calc.calculation_method}, "
                        f"SL=${stop_loss:.2f}, "
                        f"TP=${take_profit:.2f}, "
                        f"R/R={sltp_calc.risk_reward_ratio:.2f}"
                    )

                except Exception as e:
                    logger.error(
                        f"{signal.symbol} | Ошибка fallback расчета SL/TP: {e}",
                        exc_info=True
                    )
                    self.stats["failed_orders"] += 1
                    return

            # ==========================================
            # ВАЛИДАЦИЯ РАССЧИТАННЫХ SL/TP
            # ==========================================
            if side == "Buy":
                if stop_loss >= entry_price:
                    logger.error(
                        f"{signal.symbol} | ОШИБКА: SL для long должен быть < entry! "
                        f"SL={stop_loss:.2f}, entry={entry_price:.2f}"
                    )
                    self.stats["failed_orders"] += 1
                    return

                if take_profit <= entry_price:
                    logger.error(
                        f"{signal.symbol} | ОШИБКА: TP для long должен быть > entry! "
                        f"TP={take_profit:.2f}, entry={entry_price:.2f}"
                    )
                    self.stats["failed_orders"] += 1
                    return

            else:  # side == "Sell"
                if stop_loss <= entry_price:
                    logger.error(
                        f"{signal.symbol} | ОШИБКА: SL для short должен быть > entry! "
                        f"SL={stop_loss:.2f}, entry={entry_price:.2f}"
                    )
                    self.stats["failed_orders"] += 1
                    return

                if take_profit >= entry_price:
                    logger.error(
                        f"{signal.symbol} | ОШИБКА: TP для short должен быть < entry! "
                        f"TP={take_profit:.2f}, entry={entry_price:.2f}"
                    )
                    self.stats["failed_orders"] += 1
                    return

            logger.debug(f"{signal.symbol} | SL/TP validation passed ✓")

            # ==========================================
            # ШАГ 5: РАСЧЕТ РАЗМЕРА ПОЗИЦИИ
            # ==========================================
            try:
                # Получаем дополнительные данные
                current_volatility = None
                atr = signal.metadata.get('atr') if signal.metadata else None
                if atr:
                    current_volatility = atr / entry_price

                ml_confidence = None
                if ml_adjustments:
                    ml_confidence = ml_adjustments.ml_confidence

                # Рассчитываем базовый размер позиции
                raw_position_size_usdt = self.risk_manager.calculate_position_size(
                    signal=signal,
                    available_balance=available_balance,
                    stop_loss_price=stop_loss,
                    leverage=self.risk_manager.limits.default_leverage,
                    current_volatility=current_volatility,
                    ml_confidence=ml_confidence
                )

                # Применяем ML adjustments (если есть)
                if ml_adjustments and ml_adjustments.position_size_multiplier:
                    ml_adjusted_size = raw_position_size_usdt * ml_adjustments.position_size_multiplier
                    max_size = available_balance * 0.05
                    final_position_size_usdt = min(ml_adjusted_size, max_size)

                    logger.info(
                        f"{signal.symbol} | 📊 ML position sizing: "
                        f"base=${raw_position_size_usdt:.2f} × "
                        f"{ml_adjustments.position_size_multiplier:.2f} = "
                        f"${ml_adjusted_size:.2f} → "
                        f"capped at ${final_position_size_usdt:.2f}"
                    )
                else:
                    final_position_size_usdt = raw_position_size_usdt
                    logger.debug(
                        f"{signal.symbol} | Standard sizing: ${final_position_size_usdt:.2f}"
                    )

                # Рассчитываем quantity
                raw_quantity = final_position_size_usdt / entry_price

                logger.info(
                    f"{signal.symbol} | Расчет позиции: "
                    f"баланс={available_balance:.2f} USDT, "
                    f"leverage={self.risk_manager.limits.default_leverage}x, "
                    f"размер={final_position_size_usdt:.2f} USDT, "
                    f"raw_quantity={raw_quantity:.8f}"
                )

            except Exception as e:
                logger.error(
                    f"{signal.symbol} | Ошибка расчета размера позиции: {e}",
                    exc_info=True
                )
                self.stats["failed_orders"] += 1
                return

            # ==========================================
            # ШАГ 6: ВАЛИДАЦИЯ И ОКРУГЛЕНИЕ QUANTITY
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
                f"notional={final_notional:.2f} USDT, "
                f"ML={'ENABLED' if ml_adjustments else 'DISABLED'}"
            )

            # ==========================================
            # ШАГ 7: ОТКРЫТИЕ ПОЗИЦИИ
            # ==========================================
            # Подготовка entry_signal с ML метаданными
            entry_signal_dict = signal.to_dict()

            # Добавляем ML метаданные (если есть)
            if ml_adjustments:
                entry_signal_dict.update({
                    'ml_enhanced': True,
                    'ml_confidence': ml_adjustments.ml_confidence,
                    'ml_expected_return': ml_adjustments.expected_return,
                    'ml_position_multiplier': ml_adjustments.position_size_multiplier,
                    'ml_market_regime': ml_adjustments.market_regime.value if ml_adjustments.market_regime else None,
                    'final_position_size_usdt': final_notional
                })
            else:
                entry_signal_dict['ml_enhanced'] = False

            result = await self.open_position(
                symbol=signal.symbol,
                side=side,
                entry_price=entry_price,
                quantity=validated_quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_signal=entry_signal_dict,
                entry_reason=f"Signal: {safe_enum_value(signal.signal_type)}",
            )

            if result:
                self.stats["executed_orders"] += 1
                logger.info(
                    f"{signal.symbol} | ✅ Позиция успешно открыта: "
                    f"{side} {validated_quantity:.8f} @ {entry_price:.8f}, "
                    f"SL={stop_loss:.2f}, TP={take_profit:.2f}, "
                    f"ML={'ENABLED' if ml_adjustments else 'DISABLED'}"
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
            "signal_type": safe_enum_value(signal.signal_type),
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
            # "queue_size": self.signal_queue.qsize(),
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