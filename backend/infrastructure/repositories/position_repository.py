"""
Position Repository.
CRUD операции для позиций с полным контекстом.
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy import select, update

from core.logger import get_logger
from core.trace_context import trace_operation
from database.connection import db_manager
from database.models import Position, PositionStatus, OrderSide
from core.exceptions import DatabaseError

logger = get_logger(__name__)


class PositionRepository:
  """Repository для работы с позициями."""

  async def create(
      self,
      symbol: str,
      side: OrderSide,
      quantity: float,
      entry_price: float,
      stop_loss: Optional[float] = None,
      take_profit: Optional[float] = None,
      entry_signal: Optional[dict] = None,
      entry_market_data: Optional[dict] = None,
      entry_indicators: Optional[dict] = None,
      entry_reason: Optional[str] = None,
  ) -> Position:
    """
    Создание новой позиции.

    Args:
        symbol: Торговая пара
        side: Сторона (Buy/Sell)
        quantity: Количество
        entry_price: Цена входа
        stop_loss: Stop Loss
        take_profit: Take Profit
        entry_signal: Сигнал на вход
        entry_market_data: Рыночные данные при входе
        entry_indicators: Индикаторы при входе
        entry_reason: Причина открытия

    Returns:
        Position: Созданная позиция
    """
    try:
      async with db_manager.session() as session:
        position = Position(
          symbol=symbol,
          side=side,
          status=PositionStatus.OPENING,
          quantity=quantity,
          entry_price=entry_price,
          current_price=entry_price,
          stop_loss=stop_loss,
          take_profit=take_profit,
          entry_signal=entry_signal,
          entry_market_data=entry_market_data,
          entry_indicators=entry_indicators,
          entry_reason=entry_reason,
          opened_at=datetime.utcnow(),
          updated_at=datetime.utcnow(),
          version=1,
        )

        session.add(position)
        await session.commit()
        await session.refresh(position)

        logger.info(
          f"Создана позиция: {position.id} | "
          f"{symbol} {side.value} {quantity} @ {entry_price}"
        )

        return position

    except Exception as e:
      logger.error(f"Ошибка создания позиции: {e}")
      raise DatabaseError(f"Failed to create position: {str(e)}")

  async def get_by_id(self, position_id: str) -> Optional[Position]:
    """
    Получение позиции по ID.

    Args:
        position_id: ID позиции

    Returns:
        Optional[Position]: Позиция или None
    """
    try:
      async with db_manager.session() as session:
        stmt = select(Position).where(Position.id == position_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    except Exception as e:
      logger.error(f"Ошибка получения позиции {position_id}: {e}")
      return None

  async def update_status(
        self,
        position_id: str,
        new_status: PositionStatus,
        exit_price: Optional[float] = None,
        exit_signal: Optional[dict] = None,
        exit_market_data: Optional[dict] = None,
        exit_indicators: Optional[dict] = None,
        exit_reason: Optional[str] = None,
    ) -> bool:
      """
      Обновление статуса позиции.

      Args:
          position_id: ID позиции
          new_status: Новый статус
          exit_price: Цена выхода
          exit_signal: Сигнал на выход
          exit_market_data: Рыночные данные при выходе
          exit_indicators: Индикаторы при выходе
          exit_reason: Причина закрытия

      Returns:
          bool: True если обновлено
      """
      try:
        async with db_manager.session() as session:
          position = await self.get_by_id(position_id)
          if not position:
            return False

          update_data = {
            "status": new_status,
            "updated_at": datetime.utcnow(),
          }

          if exit_price is not None:
            update_data["exit_price"] = exit_price

          if exit_signal is not None:
            update_data["exit_signal"] = exit_signal

          if exit_market_data is not None:
            update_data["exit_market_data"] = exit_market_data

          if exit_indicators is not None:
            update_data["exit_indicators"] = exit_indicators

          if exit_reason is not None:
            update_data["exit_reason"] = exit_reason

          if new_status == PositionStatus.CLOSED:
            update_data["closed_at"] = datetime.utcnow()

            # Расчет realized PnL
            if exit_price is not None:
              if position.side == OrderSide.BUY:
                realized_pnl = (exit_price - position.entry_price) * position.quantity
              else:
                realized_pnl = (position.entry_price - exit_price) * position.quantity

              update_data["realized_pnl"] = realized_pnl

          stmt = (
            update(Position)
            .where(Position.id == position_id)
            .values(**update_data)
          )

          await session.execute(stmt)
          await session.commit()
          return True

      except Exception as e:
        logger.error(f"Ошибка обновления статуса позиции {position_id}: {e}")
        return False

  async def update_metadata(
      self,
      position_id: str,
      metadata: dict
  ) -> bool:
    """
    Обновить metadata_json позиции.

    Этот метод используется для сохранения справочной информации,
    которая НЕ является критичной для работы системы, но полезна
    для анализа и отладки.

    Примеры данных в metadata:
    - exchange_order_id (справочно, основной источник - Order.exchange_order_id)
    - client_order_id (справочно)
    - order_placed_at (timestamp размещения ордера)
    - transition_history (история FSM переходов)
    - debug_info (отладочная информация)

    Args:
        position_id: Position ID (UUID в строковом формате)
        metadata: Словарь с метаданными

    Returns:
        bool: True если обновлено успешно

    Raises:
        DatabaseError: При критической ошибке БД
    """
    try:
      async with db_manager.session() as session:
        # Проверяем существование позиции
        position = await self.get_by_id(position_id)
        if not position:
          logger.error(
            f"Позиция {position_id} не найдена для обновления metadata"
          )
          return False

        # Обновляем metadata
        stmt = (
          update(Position)
          .where(Position.id == position_id)
          .values(
            metadata_json=metadata,
            updated_at=datetime.utcnow()
          )
        )

        result = await session.execute(stmt)
        await session.commit()

        if result.rowcount == 0:
          logger.warning(
            f"Не удалось обновить metadata для позиции {position_id}"
          )
          return False

        logger.debug(
          f"✓ Metadata обновлена для позиции {position_id}: "
          f"{len(metadata)} полей"
        )
        return True

    except Exception as e:
      logger.error(
        f"Ошибка обновления metadata позиции {position_id}: {e}",
        exc_info=True
      )
      return False

  async def update_price(
        self,
        position_id: str,
        current_price: float
    ) -> bool:
      """
      Обновление текущей цены позиции.

      Args:
          position_id: ID позиции
          current_price: Текущая цена

      Returns:
          bool: True если обновлено
      """
      try:
        async with db_manager.session() as session:
          position = await self.get_by_id(position_id)
          if not position:
            return False

          # Расчет unrealized PnL
          if position.side == OrderSide.BUY:
            unrealized_pnl = (current_price - position.entry_price) * position.quantity
          else:
            unrealized_pnl = (position.entry_price - current_price) * position.quantity

          stmt = (
            update(Position)
            .where(Position.id == position_id)
            .values(
              current_price=current_price,
              unrealized_pnl=unrealized_pnl,
              updated_at=datetime.utcnow(),
            )
          )

          await session.execute(stmt)
          await session.commit()
          return True

      except Exception as e:
        logger.error(f"Ошибка обновления цены позиции {position_id}: {e}")
        return False

  async def update_current_price(
      self,
      position_id: str,
      current_price: float,
  ) -> bool:
    """
    Обновление текущей цены и unrealized PnL.

    Args:
        position_id: ID позиции
        current_price: Текущая цена

    Returns:
        bool: True если обновлено
    """
    try:
      async with db_manager.session() as session:
        position = await self.get_by_id(position_id)
        if not position:
          return False

        # Расчет unrealized PnL
        if position.side == OrderSide.BUY:
          unrealized_pnl = (current_price - position.entry_price) * position.quantity
        else:
          unrealized_pnl = (position.entry_price - current_price) * position.quantity

        stmt = (
          update(Position)
          .where(Position.id == position_id)
          .values(
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            updated_at=datetime.utcnow(),
          )
        )

        await session.execute(stmt)
        await session.commit()
        return True

    except Exception as e:
      logger.error(f"Ошибка обновления цены позиции {position_id}: {e}")
      return False

  async def get_active_positions(self, symbol: Optional[str] = None) -> List[Position]:
    """
    Получение активных позиций.

    Args:
        symbol: Фильтр по символу

    Returns:
        List[Position]: Список активных позиций
    """
    try:
      async with db_manager.session() as session:
        stmt = select(Position).where(
          Position.status.in_([PositionStatus.OPENING, PositionStatus.OPEN])
        )

        if symbol:
          stmt = stmt.where(Position.symbol == symbol)

        stmt = stmt.order_by(Position.opened_at.desc())

        result = await session.execute(stmt)
        positions = result.scalars().all()

        logger.debug(f"Найдено {len(positions)} активных позиций")
        return list(positions)

    except Exception as e:
      logger.error(f"Ошибка получения активных позиций: {e}")
      return []

  async def find_open_by_symbol(self, symbol: str) -> Optional[Position]:
    """
    Найти открытую позицию по символу.

    Args:
        symbol: Торговая пара

    Returns:
        Position если найдена, иначе None
    """
    with trace_operation("find_open_position_by_symbol", symbol=symbol):
      try:
        async with db_manager.get_session() as session:
          result = await session.execute(
            select(Position)
            .where(
              Position.symbol == symbol,
              Position.status == PositionStatus.OPEN
            )
            .order_by(Position.opened_at.desc())
            .limit(1)
          )

          position = result.scalar_one_or_none()

          if position:
            logger.debug(
              f"{symbol} | Found open position: {position.id}"
            )
          else:
            logger.debug(
              f"{symbol} | No open position found in DB"
            )

          return position

      except Exception as e:
        logger.error(
          f"Error finding open position for {symbol}: {e}",
          exc_info=True
        )
        return None

  async def update_stop_loss(
        self,
        position_id: str,
        new_stop_loss: float
    ) -> bool:
      """
      Обновление Stop Loss для позиции.

      Args:
          position_id: ID позиции
          new_stop_loss: Новый уровень Stop Loss

      Returns:
          bool: True если обновлено успешно
      """
      try:
        async with db_manager.session() as session:
          stmt = (
            update(Position)
            .where(Position.id == position_id)
            .values(
              stop_loss=new_stop_loss,
              updated_at=datetime.utcnow()
            )
          )

          result = await session.execute(stmt)
          await session.commit()

          if result.rowcount > 0:
            logger.debug(
              f"Stop Loss для позиции {position_id} обновлен: "
              f"новый SL=${new_stop_loss:.2f}"
            )
            return True
          else:
            logger.warning(
              f"Не удалось обновить Stop Loss для позиции {position_id}: "
              f"позиция не найдена"
            )
            return False

      except Exception as e:
        logger.error(
          f"Ошибка обновления Stop Loss для позиции {position_id}: {e}",
          exc_info=True
        )
        return False

  async def update_take_profit(
      self,
      position_id: str,
      new_take_profit: float
  ) -> bool:
    """
    Обновление Take Profit для позиции.

    Args:
        position_id: ID позиции
        new_take_profit: Новый уровень Take Profit

    Returns:
        bool: True если обновлено успешно
    """
    try:
      async with db_manager.session() as session:
        stmt = (
          update(Position)
          .where(Position.id == position_id)
          .values(
            take_profit=new_take_profit,
            updated_at=datetime.utcnow()
          )
        )

        result = await session.execute(stmt)
        await session.commit()

        if result.rowcount > 0:
          logger.debug(
            f"Take Profit для позиции {position_id} обновлен: "
            f"новый TP=${new_take_profit:.2f}"
          )
          return True
        else:
          logger.warning(
            f"Не удалось обновить Take Profit для позиции {position_id}: "
            f"позиция не найдена"
          )
          return False

    except Exception as e:
      logger.error(
        f"Ошибка обновления Take Profit для позиции {position_id}: {e}",
        exc_info=True
      )
      return False

  async def get_by_status(self, status: PositionStatus) -> List[Position]:
    """
    Получение позиций по статусу.

    Args:
        status: Статус позиции

    Returns:
        List[Position]: Список позиций
    """
    try:
      async with db_manager.session() as session:
        stmt = (
          select(Position)
          .where(Position.status == status)
          .order_by(Position.opened_at.desc())
        )

        result = await session.execute(stmt)
        positions = result.scalars().all()

        logger.debug(f"Найдено {len(positions)} позиций со статусом {status.value}")
        return list(positions)

    except Exception as e:
      logger.error(f"Ошибка получения позиций по статусу {status.value}: {e}")
      return []

# Глобальный экземпляр
position_repository = PositionRepository()