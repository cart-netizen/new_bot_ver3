"""
ПОЛНЫЙ ПРИМЕР ТОРГОВОГО ПОТОКА С ИНТЕГРАЦИЕЙ ВСЕХ КОМПОНЕНТОВ ФАЗЫ 0.

Демонстрирует:
- FSM для Orders и Positions с ПРАВИЛЬНЫМ использованием триггеров
- Idempotency Service
- Circuit Breaker
- Rate Limiting
- Trace Context
- Repositories с версионированием
- Полный аудит
"""

import sys
from pathlib import Path

# Определяем путь к backend директории
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

print(f"[INIT] Добавлен путь в sys.path: {backend_path}")
print(f"[INIT] Путь существует: {backend_path.exists()}")

import asyncio
from datetime import datetime
from typing import Optional

from core.logger import setup_logging, get_logger
from core.trace_context import trace_operation, TraceContext
from database.connection import db_manager
from database.models import OrderSide, OrderType, OrderStatus, PositionStatus, AuditAction
from domain.state_machines.order_fsm import OrderStateMachine
from domain.state_machines.position_fsm import PositionStateMachine
from domain.services.idempotency_service import idempotency_service
from infrastructure.repositories.order_repository import order_repository
from infrastructure.repositories.position_repository import position_repository
from infrastructure.repositories.audit_repository import audit_repository
from infrastructure.resilience.circuit_breaker import circuit_breaker_manager
from infrastructure.resilience.rate_limiter import rate_limiter

# Настройка логирования
setup_logging()
logger = get_logger(__name__)


class ComprehensiveTradingExample:
    """
    Полный пример торгового потока с использованием всех компонентов.
    """

    def __init__(self):
        """Инициализация."""
        self.circuit_breaker = circuit_breaker_manager.get_breaker(
            name="trading_example",
            failure_threshold=3,
            cooldown_seconds=30
        )

    async def execute_full_trade_cycle(self, symbol: str, side: str) -> dict:
        """
        Полный цикл сделки: от анализа до закрытия позиции.

        Args:
            symbol: Торговая пара
            side: Сторона (Buy/Sell)

        Returns:
            dict: Результаты выполнения
        """
        trace_id = TraceContext.generate_trace_id()
        TraceContext.set_trace_id(trace_id)

        with trace_operation(
            "full_trade_cycle",
            symbol=symbol,
            side=side
        ):
            results = {
                "trade_id": trace_id,
                "symbol": symbol,
                "side": side,
                "steps": [],
                "success": False,
            }

            try:
                # =====================================================
                # ШАГ 1: АНАЛИЗ РЫНКА И ГЕНЕРАЦИЯ СИГНАЛА
                # =====================================================
                logger.info(f"[{symbol}] → ШАГ 1: Анализ рынка")

                market_data = await self._analyze_market(symbol)
                signal_data = await self._generate_signal(symbol, side, market_data)
                indicators = await self._calculate_indicators(symbol)

                results["steps"].append({
                    "step": 1,
                    "name": "market_analysis",
                    "status": "completed",
                    "data": {
                        "imbalance": market_data["imbalance"],
                        "signal_strength": signal_data["strength"]
                    }
                })

                # Проверка сигнала
                if signal_data["strength"] < 0.7:
                    logger.warning(f"[{symbol}] Сигнал слабый: {signal_data['strength']}")
                    results["reason"] = "Weak signal"
                    return results

                # =====================================================
                # ШАГ 2: РАЗМЕЩЕНИЕ ОРДЕРА С ИДЕМПОТЕНТНОСТЬЮ
                # =====================================================
                logger.info(f"[{symbol}] → ШАГ 2: Размещение ордера")

                order_result = await self._place_order_with_protection(
                    symbol=symbol,
                    side=side,
                    quantity=0.001,
                    price=market_data["price"],
                    signal_data=signal_data,
                    market_data=market_data,
                    indicators=indicators,
                    reason=f"Strong {side} signal from momentum strategy"
                )

                if not order_result:
                    results["steps"].append({
                        "step": 2,
                        "name": "place_order",
                        "status": "failed"
                    })
                    return results

                results["order_id"] = order_result["client_order_id"]
                results["steps"].append({
                    "step": 2,
                    "name": "place_order",
                    "status": "completed",
                    "order_id": order_result["client_order_id"]
                })

                # =====================================================
                # ШАГ 3: СОЗДАНИЕ И ОТСЛЕЖИВАНИЕ ПОЗИЦИИ
                # =====================================================
                logger.info(f"[{symbol}] → ШАГ 3: Создание позиции")

                position_result = await self._create_and_track_position(
                    symbol=symbol,
                    side=side,
                    order_id=order_result["client_order_id"],
                    entry_price=market_data["price"],
                    quantity=0.001,
                    signal_data=signal_data,
                    market_data=market_data,
                    indicators=indicators
                )

                if not position_result:
                    results["steps"].append({
                        "step": 3,
                        "name": "create_position",
                        "status": "failed"
                    })
                    return results

                results["position_id"] = position_result["position_id"]
                results["steps"].append({
                    "step": 3,
                    "name": "create_position",
                    "status": "completed",
                    "position_id": position_result["position_id"]
                })

                # =====================================================
                # ШАГ 4: МОНИТОРИНГ ПОЗИЦИИ
                # =====================================================
                logger.info(f"[{symbol}] → ШАГ 4: Мониторинг позиции")

                monitoring_result = await self._monitor_position(
                    position_id=position_result["position_id"],
                    symbol=symbol,
                    duration_seconds=10
                )

                results["steps"].append({
                    "step": 4,
                    "name": "monitor_position",
                    "status": "completed",
                    "pnl": monitoring_result["unrealized_pnl"]
                })

                # =====================================================
                # ШАГ 5: ЗАКРЫТИЕ ПОЗИЦИИ
                # =====================================================
                logger.info(f"[{symbol}] → ШАГ 5: Закрытие позиции")

                exit_data = await self._analyze_exit_conditions(symbol)

                close_result = await self._close_position_with_audit(
                    position_id=position_result["position_id"],
                    symbol=symbol,
                    exit_signal=exit_data["signal"],
                    exit_market_data=exit_data["market_data"],
                    exit_indicators=exit_data["indicators"],
                    reason=exit_data["reason"]
                )

                results["steps"].append({
                    "step": 5,
                    "name": "close_position",
                    "status": "completed",
                    "realized_pnl": close_result["realized_pnl"]
                })

                # =====================================================
                # ФИНАЛ: АНАЛИЗ РЕЗУЛЬТАТОВ
                # =====================================================
                results["success"] = True
                results["realized_pnl"] = close_result["realized_pnl"]
                results["total_duration"] = close_result["duration"]

                logger.info(
                    f"✓ [{symbol}] Полный цикл сделки завершен | "
                    f"PnL: {close_result['realized_pnl']:.2f}"
                )

                return results

            except Exception as e:
                logger.error(f"Критическая ошибка в торговом цикле: {e}", exc_info=True)
                results["error"] = str(e)
                return results

    async def _analyze_market(self, symbol: str) -> dict:
        """Анализ рынка (имитация)."""
        with trace_operation("analyze_market", symbol=symbol):
            await asyncio.sleep(0.1)
            return {
                "price": 50000.0,
                "imbalance": 0.75,
                "spread": 0.5,
                "volume_bid": 1000.0,
                "volume_ask": 800.0,
            }

    async def _generate_signal(self, symbol: str, side: str, market_data: dict) -> dict:
        """Генерация торгового сигнала."""
        with trace_operation("generate_signal", symbol=symbol, side=side):
            await asyncio.sleep(0.05)
            return {
                "type": "momentum",
                "strength": 0.85,
                "direction": side,
                "confidence": 0.9,
            }

    async def _calculate_indicators(self, symbol: str) -> dict:
        """Расчет индикаторов."""
        with trace_operation("calculate_indicators", symbol=symbol):
            await asyncio.sleep(0.05)
            return {
                "rsi": 55,
                "macd": 0.05,
                "ema_20": 49900.0,
                "ema_50": 49500.0,
            }

    async def _place_order_with_protection(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        signal_data: dict,
        market_data: dict,
        indicators: dict,
        reason: str
    ) -> Optional[dict]:
        """Размещение ордера с полной защитой."""
        with trace_operation("place_order_protected", symbol=symbol):
            try:
                # 1. Rate limiting
                allowed = await rate_limiter.acquire("order_placement", tokens=1)
                if not allowed:
                    logger.error("Rate limit достигнут")
                    return None

                # 2. Генерация Client Order ID
                client_order_id = idempotency_service.generate_client_order_id(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price
                )

                # 3. Проверка идемпотентности
                params = {"symbol": symbol, "side": side, "qty": quantity}
                cached = await idempotency_service.check_idempotency("place_order", params)

                if cached:
                    logger.info(f"Возврат кэшированного результата: {client_order_id}")
                    return cached["result"]

                # 4. Создание ордера в БД
                order_side = OrderSide.BUY if side == "Buy" else OrderSide.SELL

                order = await order_repository.create(
                    client_order_id=client_order_id,
                    symbol=symbol,
                    side=order_side,
                    order_type=OrderType.LIMIT,
                    quantity=quantity,
                    price=price,
                    signal_data=signal_data,
                    market_data=market_data,
                    indicators=indicators,
                    reason=reason
                )

                # 5. FSM для ордера - используем триггер напрямую!
                fsm = OrderStateMachine(client_order_id, OrderStatus.PENDING)

                # 6. Размещение через Circuit Breaker
                exchange_result = await self.circuit_breaker.call_async(
                    self._simulate_exchange_placement,
                    client_order_id
                )

                if not exchange_result:
                    # Переход в FAILED через триггер
                    fsm.fail()
                    await order_repository.update_status(
                        client_order_id, OrderStatus.FAILED
                    )
                    return None

                # 7. Обновление статуса - используем триггер FSM!
                fsm.place()  # ✅ PENDING -> PLACED через триггер

                await order_repository.update_status(
                    client_order_id=client_order_id,
                    new_status=OrderStatus.PLACED,
                    exchange_order_id=exchange_result["order_id"]
                )

                # 8. Аудит
                await audit_repository.log(
                    action=AuditAction.ORDER_PLACE,
                    entity_type="Order",
                    entity_id=client_order_id,
                    new_value={
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "price": price
                    },
                    reason=reason,
                    trace_id=TraceContext.get_trace_id(),
                    success=True,
                    context={
                        "signal_data": signal_data,
                        "indicators": indicators
                    }
                )

                # 9. Сохранение для идемпотентности
                result = {
                    "client_order_id": client_order_id,
                    "exchange_order_id": exchange_result["order_id"],
                    "status": "placed"
                }

                await idempotency_service.save_operation_result(
                    "place_order", params, result, success=True
                )

                logger.info(f"✓ Ордер размещен: {client_order_id}")
                return result

            except Exception as e:
                logger.error(f"Ошибка размещения ордера: {e}")
                return None

    async def _simulate_exchange_placement(self, client_order_id: str) -> dict:
        """Имитация размещения на бирже."""
        await asyncio.sleep(0.2)
        return {"order_id": f"exchange_{client_order_id[:8]}"}

    async def _create_and_track_position(
        self,
        symbol: str,
        side: str,
        order_id: str,
        entry_price: float,
        quantity: float,
        signal_data: dict,
        market_data: dict,
        indicators: dict
    ) -> Optional[dict]:
        """Создание и отслеживание позиции."""
        with trace_operation("create_position", symbol=symbol):
            try:
                order_side = OrderSide.BUY if side == "Buy" else OrderSide.SELL

                # Создание позиции в БД
                position = await position_repository.create(
                    symbol=symbol,
                    side=order_side,
                    quantity=quantity,
                    entry_price=entry_price,
                    stop_loss=entry_price * 0.98,
                    take_profit=entry_price * 1.05,
                    entry_signal=signal_data,
                    entry_market_data=market_data,
                    entry_indicators=indicators,
                    entry_reason=f"Entry based on {signal_data['type']} strategy"
                )

                # FSM для позиции - используем триггер напрямую!
                fsm = PositionStateMachine(str(position.id), PositionStatus.OPENING)

                # Подтверждение открытия через триггер
                fsm.confirm_open()  # ✅ OPENING -> OPEN через триггер

                await position_repository.update_status(
                    position_id=str(position.id),
                    new_status=PositionStatus.OPEN
                )

                # Аудит
                await audit_repository.log(
                    action=AuditAction.POSITION_OPEN,
                    entity_type="Position",
                    entity_id=str(position.id),
                    new_value={
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "entry_price": entry_price
                    },
                    trace_id=TraceContext.get_trace_id(),
                    success=True
                )

                logger.info(f"✓ Позиция открыта: {position.id}")
                return {"position_id": str(position.id)}

            except Exception as e:
                logger.error(f"Ошибка создания позиции: {e}")
                return None

    async def _monitor_position(
        self,
        position_id: str,
        symbol: str,
        duration_seconds: int
    ) -> dict:
        """Мониторинг позиции."""
        with trace_operation("monitor_position", position_id=position_id):
            logger.info(f"Мониторинг позиции {position_id} в течение {duration_seconds}s")

            for i in range(duration_seconds):
                await asyncio.sleep(1)

                # Обновляем текущую цену
                current_price = 50000.0 + (i * 10)
                await position_repository.update_current_price(
                    position_id=position_id,
                    current_price=current_price
                )

            # Получаем финальное состояние
            position = await position_repository.get_by_id(position_id)

            return {
                "unrealized_pnl": position.unrealized_pnl,
                "current_price": position.current_price
            }

    async def _analyze_exit_conditions(self, symbol: str) -> dict:
        """Анализ условий выхода."""
        await asyncio.sleep(0.1)
        return {
            "signal": {"type": "take_profit", "strength": 0.9},
            "market_data": {"price": 50100.0, "imbalance": 0.4},
            "indicators": {"rsi": 70, "macd": -0.02},
            "reason": "Take profit target reached"
        }

    async def _close_position_with_audit(
        self,
        position_id: str,
        symbol: str,
        exit_signal: dict,
        exit_market_data: dict,
        exit_indicators: dict,
        reason: str
    ) -> dict:
        """Закрытие позиции с полным аудитом."""
        with trace_operation("close_position", position_id=position_id):
            # Получаем позицию
            position = await position_repository.get_by_id(position_id)

            # FSM - используем триггеры напрямую!
            fsm = PositionStateMachine(position_id, PositionStatus.OPEN)

            # Двухшаговое закрытие: OPEN -> CLOSING -> CLOSED
            fsm.start_close()  # ✅ OPEN -> CLOSING
            fsm.confirm_close()  # ✅ CLOSING -> CLOSED

            # Закрываем в БД
            exit_price = exit_market_data["price"]

            await position_repository.update_status(
                position_id=position_id,
                new_status=PositionStatus.CLOSED,
                exit_price=exit_price,
                exit_signal=exit_signal,
                exit_market_data=exit_market_data,
                exit_indicators=exit_indicators,
                exit_reason=reason
            )

            # Получаем обновленную позицию
            position = await position_repository.get_by_id(position_id)

            # Аудит
            await audit_repository.log(
                action=AuditAction.POSITION_CLOSE,
                entity_type="Position",
                entity_id=position_id,
                new_value={
                    "exit_price": exit_price,
                    "realized_pnl": position.realized_pnl
                },
                reason=reason,
                trace_id=TraceContext.get_trace_id(),
                success=True,
                context={
                    "exit_signal": exit_signal,
                    "exit_indicators": exit_indicators
                }
            )

            logger.info(f"✓ Позиция закрыта: {position_id} | PnL: {position.realized_pnl:.2f}")

            duration = (position.closed_at - position.opened_at).total_seconds()

            return {
                "realized_pnl": position.realized_pnl,
                "duration": duration
            }


async def main():
    """Главная функция для запуска примера."""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE TRADING FLOW EXAMPLE")
    logger.info("=" * 80)

    # Инициализация БД
    await db_manager.initialize()

    try:
        # Создаем экземпляр
        example = ComprehensiveTradingExample()

        # Выполняем полный торговый цикл
        result = await example.execute_full_trade_cycle(
            symbol="BTCUSDT",
            side="Buy"
        )

        # Результаты
        logger.info("=" * 80)
        logger.info("РЕЗУЛЬТАТЫ")
        logger.info("=" * 80)
        logger.info(f"Trade ID: {result.get('trade_id')}")
        logger.info(f"Success: {result.get('success')}")
        logger.info(f"Steps completed: {len(result.get('steps', []))}")

        if result.get('success'):
            logger.info(f"Realized PnL: {result.get('realized_pnl', 0):.2f}")
            logger.info(f"Duration: {result.get('total_duration', 0):.2f}s")

        logger.info("=" * 80)

    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())