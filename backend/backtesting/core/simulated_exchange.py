"""
Simulated Exchange - реалистичная эмуляция биржи для бэктестинга.

Функции:
- Исполнение market и limit ордеров
- Симуляция slippage (проскальзывание)
- Комиссии (maker/taker fees)
- Partial fills (частичное исполнение)
- Order rejection (отклонение ордеров)
- Latency simulation (задержка исполнения)

Best Practices:
- Realistic slippage models
- Volume-based execution
- No look-ahead bias
"""

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import uuid

from backend.core.logger import get_logger
from backend.models.orderbook import OrderBookSnapshot
from backend.database.models import OrderSide, OrderType
from backend.backtesting.models import ExchangeConfig, SlippageModel

logger = get_logger(__name__)


class OrderStatus(str, Enum):
    """Статус ордера в симуляции."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class SimulatedOrder:
    """Симулированный ордер."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]  # None для market orders

    # Execution state
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0

    # Timestamps
    created_at: datetime = None
    filled_at: Optional[datetime] = None

    # Metadata
    slippage_applied: float = 0.0  # Процент проскальзывания
    reject_reason: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class SimulatedExchange:
    """
    Эмулятор биржи для бэктестинга.

    Реалистично симулирует:
    1. Order Execution - исполнение ордеров
    2. Slippage - проскальзывание цены
    3. Commissions - комиссии maker/taker
    4. Partial Fills - частичное исполнение
    5. Rejections - отклонение ордеров
    6. Latency - задержка исполнения
    """

    def __init__(self, config: ExchangeConfig):
        """
        Инициализация симулятора биржи.

        Args:
            config: Конфигурация exchange simulation
        """
        self.config = config

        # Order tracking
        self.open_orders: Dict[str, SimulatedOrder] = {}
        self.filled_orders: List[SimulatedOrder] = []
        self.rejected_orders: List[SimulatedOrder] = []

        # Statistics
        self.total_orders = 0
        self.total_filled = 0
        self.total_rejected = 0
        self.total_commission_paid = 0.0

        logger.info(
            f"SimulatedExchange инициализирован: "
            f"commission={config.commission_rate*100:.2f}%, "
            f"slippage_model={config.slippage_model.value}"
        )

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        current_price: float = None,
        orderbook: Optional[OrderBookSnapshot] = None
    ) -> SimulatedOrder:
        """
        Разместить ордер на симулированной бирже.

        Args:
            symbol: Торговая пара
            side: BUY или SELL
            quantity: Объем
            order_type: MARKET или LIMIT
            price: Цена (для limit orders)
            current_price: Текущая рыночная цена
            orderbook: Снимок стакана (для volume-based slippage)

        Returns:
            SimulatedOrder с результатом исполнения
        """
        self.total_orders += 1

        # Создать ордер
        order = SimulatedOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price
        )

        logger.debug(
            f"📝 Размещение ордера: {side.value} {quantity} {symbol} @ "
            f"{'MARKET' if order_type == OrderType.MARKET else f'{price}'}"
        )

        # 1. Проверка на отклонение (random rejection)
        if self._should_reject_order():
            order.status = OrderStatus.REJECTED
            order.reject_reason = "Random rejection (simulated exchange error)"
            self.rejected_orders.append(order)
            self.total_rejected += 1
            logger.warning(f"❌ Ордер отклонен: {order.reject_reason}")
            return order

        # 2. Симуляция задержки (latency)
        if self.config.simulate_latency:
            await self._simulate_latency()

        # 3. Исполнение ордера
        if order_type == OrderType.MARKET:
            await self._execute_market_order(order, current_price, orderbook)
        else:
            # Limit orders добавляются в open_orders и исполняются через process_tick
            self.open_orders[order.order_id] = order
            logger.debug(f"🕐 Limit ордер добавлен в очередь: {order.order_id}")

        return order

    async def _execute_market_order(
        self,
        order: SimulatedOrder,
        current_price: float,
        orderbook: Optional[OrderBookSnapshot]
    ):
        """
        Исполнить market ордер немедленно.

        Market orders исполняются по лучшей доступной цене с учетом:
        - Slippage (проскальзывание)
        - Commission (комиссия)
        - Available liquidity (если есть orderbook)
        """
        # Расчет execution price с учетом slippage
        execution_price, slippage_pct = self._calculate_execution_price(
            base_price=current_price,
            side=order.side,
            quantity=order.quantity,
            orderbook=orderbook
        )

        # Полное исполнение (в реальности может быть partial fill)
        order.filled_quantity = order.quantity
        order.average_fill_price = execution_price
        order.slippage_applied = slippage_pct

        # Расчет комиссии (taker fee для market orders)
        commission_rate = self.config.taker_commission or self.config.commission_rate
        order.commission = order.filled_quantity * execution_price * commission_rate

        # Обновление статуса
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()

        # Добавить в filled orders
        self.filled_orders.append(order)
        self.total_filled += 1
        self.total_commission_paid += order.commission

        logger.info(
            f"✅ Market ордер исполнен: {order.side.value} {order.filled_quantity} @ "
            f"{execution_price:.2f} (slippage: {slippage_pct:.3f}%, "
            f"commission: ${order.commission:.2f})"
        )

    def _calculate_execution_price(
        self,
        base_price: float,
        side: OrderSide,
        quantity: float,
        orderbook: Optional[OrderBookSnapshot]
    ) -> Tuple[float, float]:
        """
        Расчет цены исполнения с учетом slippage.

        Returns:
            (execution_price, slippage_pct)
        """
        if self.config.slippage_model == SlippageModel.FIXED:
            # Фиксированное проскальзывание
            slippage_pct = self.config.slippage_pct

            if side == OrderSide.BUY:
                # BUY: цена выше (хуже для покупателя)
                execution_price = base_price * (1 + slippage_pct / 100)
            else:
                # SELL: цена ниже (хуже для продавца)
                execution_price = base_price * (1 - slippage_pct / 100)

            return execution_price, slippage_pct

        elif self.config.slippage_model == SlippageModel.VOLUME_BASED:
            # Slippage зависит от объема в стакане
            if not orderbook:
                # Fallback to fixed slippage
                return self._calculate_execution_price(
                    base_price, side, quantity, None
                )

            # Рассчитываем weighted average price на основе orderbook
            execution_price = self._calculate_vwap_from_orderbook(
                orderbook, side, quantity
            )

            slippage_pct = abs((execution_price - base_price) / base_price) * 100

            return execution_price, slippage_pct

        elif self.config.slippage_model == SlippageModel.PERCENTAGE:
            # Процент от цены
            slippage_pct = self.config.slippage_pct

            if side == OrderSide.BUY:
                execution_price = base_price * (1 + slippage_pct / 100)
            else:
                execution_price = base_price * (1 - slippage_pct / 100)

            return execution_price, slippage_pct

        else:
            # Default: no slippage
            return base_price, 0.0

    def _calculate_vwap_from_orderbook(
        self,
        orderbook: OrderBookSnapshot,
        side: OrderSide,
        quantity: float
    ) -> float:
        """
        Расчет VWAP (Volume-Weighted Average Price) из orderbook.

        Симулирует реальное исполнение через уровни стакана.
        """
        if side == OrderSide.BUY:
            # BUY: едим ask levels
            levels = orderbook.asks[:10]  # Топ 10 уровней
        else:
            # SELL: едим bid levels
            levels = orderbook.bids[:10]

        if not levels:
            # Нет данных orderbook
            return orderbook.best_ask if side == OrderSide.BUY else orderbook.best_bid

        remaining_quantity = quantity
        total_cost = 0.0
        total_quantity = 0.0

        for level in levels:
            # level is a tuple (price, quantity)
            level_price, level_quantity = level

            # Сколько можем взять с этого уровня
            take_quantity = min(remaining_quantity, level_quantity)

            total_cost += take_quantity * level_price
            total_quantity += take_quantity
            remaining_quantity -= take_quantity

            if remaining_quantity <= 0:
                break

        # Если недостаточно ликвидности, добавляем penalty
        if remaining_quantity > 0:
            # Худшая цена + penalty
            worst_price, _ = levels[-1]
            penalty = 0.1  # 0.1% penalty за недостаток ликвидности

            if side == OrderSide.BUY:
                penalty_price = worst_price * (1 + penalty / 100)
            else:
                penalty_price = worst_price * (1 - penalty / 100)

            total_cost += remaining_quantity * penalty_price
            total_quantity += remaining_quantity

        vwap = total_cost / total_quantity if total_quantity > 0 else levels[0].price
        return vwap

    async def process_tick(
        self,
        current_time: datetime,
        current_price: float,
        orderbook: Optional[OrderBookSnapshot] = None
    ):
        """
        Обработать тик - проверить limit orders на исполнение.

        Args:
            current_time: Текущее время симуляции
            current_price: Текущая цена
            orderbook: Снимок стакана
        """
        if not self.open_orders:
            return

        filled_order_ids = []

        for order_id, order in self.open_orders.items():
            # Проверка условий исполнения limit order
            should_fill = False

            if order.side == OrderSide.BUY:
                # BUY limit: исполняется когда цена падает до limit price или ниже
                if current_price <= order.price:
                    should_fill = True
            else:
                # SELL limit: исполняется когда цена растет до limit price или выше
                if current_price >= order.price:
                    should_fill = True

            if should_fill:
                # Исполнить limit order
                execution_price = order.price  # Limit orders исполняются по limit price

                order.filled_quantity = order.quantity
                order.average_fill_price = execution_price
                order.slippage_applied = 0.0  # Limit orders без slippage

                # Комиссия (maker fee для limit orders)
                commission_rate = self.config.maker_commission or self.config.commission_rate
                order.commission = order.filled_quantity * execution_price * commission_rate

                # Обновление статуса
                order.status = OrderStatus.FILLED
                order.filled_at = current_time

                self.filled_orders.append(order)
                self.total_filled += 1
                self.total_commission_paid += order.commission

                filled_order_ids.append(order_id)

                logger.info(
                    f"✅ Limit ордер исполнен: {order.side.value} {order.filled_quantity} @ "
                    f"{execution_price:.2f} (maker fee: ${order.commission:.2f})"
                )

        # Удалить исполненные ордера из open_orders
        for order_id in filled_order_ids:
            del self.open_orders[order_id]

    async def cancel_order(self, order_id: str) -> bool:
        """
        Отменить открытый ордер.

        Returns:
            True если ордер был отменен, False если не найден
        """
        if order_id not in self.open_orders:
            return False

        order = self.open_orders[order_id]
        order.status = OrderStatus.CANCELLED

        del self.open_orders[order_id]

        logger.debug(f"🚫 Ордер отменен: {order_id}")
        return True

    def _should_reject_order(self) -> bool:
        """Проверка на случайное отклонение ордера."""
        if self.config.order_reject_probability == 0:
            return False

        return random.random() < self.config.order_reject_probability

    async def _simulate_latency(self):
        """Симуляция задержки исполнения."""
        # Нормальное распределение задержки
        latency_ms = max(
            0.0,
            random.gauss(
                self.config.latency_mean_ms,
                self.config.latency_std_ms
            )
        )

        await asyncio.sleep(latency_ms / 1000.0)

    def get_statistics(self) -> Dict:
        """Получить статистику работы симулятора."""
        success_rate = (
            self.total_filled / self.total_orders * 100
            if self.total_orders > 0 else 0
        )

        return {
            'total_orders': self.total_orders,
            'total_filled': self.total_filled,
            'total_rejected': self.total_rejected,
            'open_orders': len(self.open_orders),
            'success_rate_pct': success_rate,
            'total_commission_paid': self.total_commission_paid,
            'avg_commission_per_order': (
                self.total_commission_paid / self.total_filled
                if self.total_filled > 0 else 0
            )
        }

    def reset(self):
        """Сброс состояния для нового бэктеста."""
        self.open_orders.clear()
        self.filled_orders.clear()
        self.rejected_orders.clear()
        self.total_orders = 0
        self.total_filled = 0
        self.total_rejected = 0
        self.total_commission_paid = 0.0

        logger.info("SimulatedExchange сброшен")
