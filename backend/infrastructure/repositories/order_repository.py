"""
Order Repository.
CRUD операции для ордеров с версионированием.
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from core.logger import get_logger
from database.connection import db_manager
from database.models import Order, OrderStatus, OrderSide, OrderType
from core.exceptions import DatabaseError

logger = get_logger(__name__)


class OrderRepository:
    """Repository для работы с ордерами."""

    async def create(
        self,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        signal_data: Optional[dict] = None,
        market_data: Optional[dict] = None,
        indicators: Optional[dict] = None,
        reason: Optional[str] = None,
        position_id: Optional[str] = None,
    ) -> Order:
        """
        Создание нового ордера.

        Args:
            client_order_id: Уникальный ID клиента
            symbol: Торговая пара
            side: Сторона (Buy/Sell)
            order_type: Тип ордера
            quantity: Количество
            price: Цена (для limit)
            stop_loss: Stop Loss
            take_profit: Take Profit
            signal_data: Данные сигнала
            market_data: Рыночные данные
            indicators: Показатели индикаторов
            reason: Причина создания
            position_id: ID позиции

        Returns:
            Order: Созданный ордер

        Raises:
            DatabaseError: При ошибке создания
        """
        try:
            async with db_manager.session() as session:
                order = Order(
                    client_order_id=client_order_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    status=OrderStatus.PENDING,
                    quantity=quantity,
                    price=price,
                    filled_quantity=0.0,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_data=signal_data,
                    market_data=market_data,
                    indicators=indicators,
                    reason=reason,
                    position_id=position_id,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    version=1,
                )

                session.add(order)
                await session.commit()
                await session.refresh(order)

                logger.info(
                    f"Создан ордер: {client_order_id} | "
                    f"{symbol} {side.value} {quantity} @ {price or 'MARKET'}"
                )

                return order

        except IntegrityError as e:
            logger.error(f"Ордер {client_order_id} уже существует: {e}")
            raise DatabaseError(f"Duplicate order: {client_order_id}")
        except Exception as e:
            logger.error(f"Ошибка создания ордера {client_order_id}: {e}")
            raise DatabaseError(f"Failed to create order: {str(e)}")

    async def get_by_client_order_id(self, client_order_id: str) -> Optional[Order]:
        """
        Получение ордера по client_order_id.

        Args:
            client_order_id: Client Order ID

        Returns:
            Optional[Order]: Ордер или None
        """
        try:
            async with db_manager.session() as session:
                stmt = select(Order).where(Order.client_order_id == client_order_id)
                result = await session.execute(stmt)
                order = result.scalar_one_or_none()

                if order:
                    logger.debug(f"Найден ордер: {client_order_id}")
                else:
                    logger.debug(f"Ордер не найден: {client_order_id}")

                return order

        except Exception as e:
            logger.error(f"Ошибка получения ордера {client_order_id}: {e}")
            return None

    async def update_status(
        self,
        client_order_id: str,
        new_status: OrderStatus,
        exchange_order_id: Optional[str] = None,
        filled_quantity: Optional[float] = None,
        average_fill_price: Optional[float] = None,
    ) -> bool:
        """
        Обновление статуса ордера с версионированием.

        Args:
            client_order_id: Client Order ID
            new_status: Новый статус
            exchange_order_id: ID от биржи
            filled_quantity: Исполненное количество
            average_fill_price: Средняя цена исполнения

        Returns:
            bool: True если обновлено успешно
        """
        try:
            async with db_manager.session() as session:
                # Получаем текущую версию
                order = await self.get_by_client_order_id(client_order_id)
                if not order:
                    logger.error(f"Ордер {client_order_id} не найден для обновления")
                    return False

                current_version = order.version

                # Подготавливаем обновление
                update_data = {
                    "status": new_status,
                    "updated_at": datetime.utcnow(),
                    "version": current_version + 1,
                }

                if exchange_order_id:
                    update_data["exchange_order_id"] = exchange_order_id

                if filled_quantity is not None:
                    update_data["filled_quantity"] = filled_quantity

                if average_fill_price is not None:
                    update_data["average_fill_price"] = average_fill_price

                # Обновляем временные метки
                if new_status == OrderStatus.PLACED:
                    update_data["placed_at"] = datetime.utcnow()
                elif new_status == OrderStatus.FILLED:
                    update_data["filled_at"] = datetime.utcnow()
                elif new_status == OrderStatus.CANCELLED:
                    update_data["cancelled_at"] = datetime.utcnow()

                # Обновляем с проверкой версии (optimistic locking)
                stmt = (
                    update(Order)
                    .where(
                        Order.client_order_id == client_order_id,
                        Order.version == current_version,
                    )
                    .values(**update_data)
                )

                result = await session.execute(stmt)
                await session.commit()

                if result.rowcount == 0:
                    logger.warning(
                        f"Конфликт версий при обновлении ордера {client_order_id}"
                    )
                    return False

                logger.info(
                    f"Обновлен статус ордера {client_order_id}: {new_status.value}"
                )
                return True

        except Exception as e:
            logger.error(f"Ошибка обновления статуса ордера {client_order_id}: {e}")
            return False

    async def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Получение активных ордеров.

        Args:
            symbol: Фильтр по символу (опционально)

        Returns:
            List[Order]: Список активных ордеров
        """
        try:
            async with db_manager.session() as session:
                stmt = select(Order).where(
                    Order.status.in_([
                        OrderStatus.PENDING,
                        OrderStatus.PLACED,
                        OrderStatus.PARTIALLY_FILLED,
                    ])
                )

                if symbol:
                    stmt = stmt.where(Order.symbol == symbol)

                stmt = stmt.order_by(Order.created_at.desc())

                result = await session.execute(stmt)
                orders = result.scalars().all()

                logger.debug(f"Найдено {len(orders)} активных ордеров")
                return list(orders)

        except Exception as e:
            logger.error(f"Ошибка получения активных ордеров: {e}")
            return []

    async def get_recent_orders(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Order]:
        """
        Получение недавних ордеров.

        Args:
            symbol: Фильтр по символу
            limit: Количество ордеров

        Returns:
            List[Order]: Список ордеров
        """
        try:
            async with db_manager.session() as session:
                stmt = select(Order)

                if symbol:
                    stmt = stmt.where(Order.symbol == symbol)

                stmt = stmt.order_by(Order.created_at.desc()).limit(limit)

                result = await session.execute(stmt)
                orders = result.scalars().all()

                return list(orders)

        except Exception as e:
            logger.error(f"Ошибка получения недавних ордеров: {e}")
            return []


# Глобальный экземпляр
order_repository = OrderRepository()