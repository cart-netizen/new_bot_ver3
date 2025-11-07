"""
Backtest Repository - CRUD операции для бэктестов.

Работа с таблицами:
- backtest_runs
- backtest_trades
- backtest_equity
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import select, update, delete, func, and_
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import IntegrityError
import uuid

from backend.core.logger import get_logger
from backend.database.connection import db_manager
from backend.database.models import (
    BacktestRun,
    BacktestTrade,
    BacktestEquity,
    BacktestStatus,
    OrderSide
)
from backend.core.exceptions import DatabaseError

logger = get_logger(__name__)


class BacktestRepository:
    """Repository для работы с бэктестами."""

    # ==================== BacktestRun CRUD ====================

    async def create_run(
        self,
        name: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        strategies_config: Dict,
        risk_config: Dict,
        exchange_config: Optional[Dict] = None,
        description: Optional[str] = None
    ) -> BacktestRun:
        """
        Создать новый backtest run.

        Args:
            name: Название бэктеста
            symbol: Торговая пара
            start_date: Дата начала
            end_date: Дата окончания
            initial_capital: Начальный капитал
            strategies_config: Конфигурация стратегий (JSONB)
            risk_config: Конфигурация risk management (JSONB)
            exchange_config: Конфигурация биржи (JSONB)
            description: Описание

        Returns:
            BacktestRun: Созданный backtest run

        Raises:
            DatabaseError: При ошибке БД
        """
        try:
            async with db_manager.session() as session:
                backtest = BacktestRun(
                    id=uuid.uuid4(),
                    name=name,
                    description=description,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    strategies_config=strategies_config,
                    risk_config=risk_config,
                    exchange_config=exchange_config or {},
                    status=BacktestStatus.PENDING,
                    progress_pct=0.0,
                    created_at=datetime.utcnow()
                )

                session.add(backtest)
                await session.commit()
                await session.refresh(backtest)

                logger.info(
                    f"Создан backtest run: {backtest.id} | {name} | "
                    f"{symbol} {start_date.date()} → {end_date.date()}"
                )

                return backtest

        except Exception as e:
            logger.error(f"Ошибка создания backtest run: {e}", exc_info=True)
            raise DatabaseError(f"Failed to create backtest run: {str(e)}")

    async def get_by_id(
        self,
        backtest_id: uuid.UUID,
        include_trades: bool = False,
        include_equity: bool = False
    ) -> Optional[BacktestRun]:
        """
        Получить backtest run по ID.

        Args:
            backtest_id: ID бэктеста
            include_trades: Загрузить связанные trades
            include_equity: Загрузить equity curve

        Returns:
            BacktestRun или None
        """
        try:
            async with db_manager.session() as session:
                query = select(BacktestRun).where(BacktestRun.id == backtest_id)

                # Eager loading
                if include_trades:
                    query = query.options(selectinload(BacktestRun.trades))
                if include_equity:
                    query = query.options(selectinload(BacktestRun.equity_curve))

                result = await session.execute(query)
                backtest = result.scalar_one_or_none()

                return backtest

        except Exception as e:
            logger.error(f"Ошибка получения backtest {backtest_id}: {e}")
            raise DatabaseError(f"Failed to get backtest: {str(e)}")

    async def list_runs(
        self,
        symbol: Optional[str] = None,
        status: Optional[BacktestStatus] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "created_at",
        order_desc: bool = True
    ) -> List[BacktestRun]:
        """
        Список backtest runs с фильтрацией.

        Args:
            symbol: Фильтр по символу
            status: Фильтр по статусу
            limit: Лимит результатов
            offset: Offset для пагинации
            order_by: Поле для сортировки
            order_desc: Обратная сортировка

        Returns:
            Список BacktestRun
        """
        try:
            async with db_manager.session() as session:
                query = select(BacktestRun)

                # Filters
                if symbol:
                    query = query.where(BacktestRun.symbol == symbol)
                if status:
                    query = query.where(BacktestRun.status == status)

                # Ordering
                order_column = getattr(BacktestRun, order_by, BacktestRun.created_at)
                if order_desc:
                    query = query.order_by(order_column.desc())
                else:
                    query = query.order_by(order_column.asc())

                # Pagination
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                runs = result.scalars().all()

                return list(runs)

        except Exception as e:
            logger.error(f"Ошибка получения списка backtests: {e}")
            raise DatabaseError(f"Failed to list backtests: {str(e)}")

    async def count_runs(
        self,
        symbol: Optional[str] = None,
        status: Optional[BacktestStatus] = None
    ) -> int:
        """
        Подсчитать количество backtest runs с фильтрацией.

        Args:
            symbol: Фильтр по символу
            status: Фильтр по статусу

        Returns:
            Количество runs
        """
        try:
            async with db_manager.session() as session:
                query = select(func.count(BacktestRun.id))

                # Filters
                if symbol:
                    query = query.where(BacktestRun.symbol == symbol)
                if status:
                    query = query.where(BacktestRun.status == status)

                result = await session.execute(query)
                count = result.scalar()

                return count or 0

        except Exception as e:
            logger.error(f"Ошибка подсчета backtests: {e}")
            raise DatabaseError(f"Failed to count backtests: {str(e)}")

    async def update_status(
        self,
        backtest_id: uuid.UUID,
        status: BacktestStatus,
        progress_pct: Optional[float] = None,
        current_date: Optional[datetime] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None
    ) -> bool:
        """
        Обновить статус backtest run.

        Args:
            backtest_id: ID бэктеста
            status: Новый статус
            progress_pct: Процент выполнения
            current_date: Текущая дата в симуляции
            error_message: Сообщение об ошибке
            error_traceback: Traceback ошибки

        Returns:
            True если обновлено успешно
        """
        try:
            async with db_manager.session() as session:
                update_data = {'status': status}

                if progress_pct is not None:
                    update_data['progress_pct'] = progress_pct
                if current_date is not None:
                    update_data['current_date'] = current_date
                if error_message is not None:
                    update_data['error_message'] = error_message
                if error_traceback is not None:
                    update_data['error_traceback'] = error_traceback

                # Set timestamps based on status
                if status == BacktestStatus.RUNNING and 'started_at' not in update_data:
                    update_data['started_at'] = datetime.utcnow()
                elif status in [BacktestStatus.COMPLETED, BacktestStatus.FAILED, BacktestStatus.CANCELLED]:
                    update_data['completed_at'] = datetime.utcnow()

                stmt = (
                    update(BacktestRun)
                    .where(BacktestRun.id == backtest_id)
                    .values(**update_data)
                )

                await session.execute(stmt)
                await session.commit()

                logger.debug(f"Обновлен статус backtest {backtest_id}: {status.value}")
                return True

        except Exception as e:
            logger.error(f"Ошибка обновления статуса backtest {backtest_id}: {e}")
            raise DatabaseError(f"Failed to update backtest status: {str(e)}")

    async def update_results(
        self,
        backtest_id: uuid.UUID,
        final_capital: float,
        total_pnl: float,
        total_pnl_pct: float,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Обновить результаты backtest run.

        Args:
            backtest_id: ID бэктеста
            final_capital: Финальный капитал
            total_pnl: Общий PnL
            total_pnl_pct: Общий PnL в процентах
            total_trades: Всего сделок
            winning_trades: Выигрышных сделок
            losing_trades: Проигрышных сделок
            metrics: Детальные метрики (JSONB)

        Returns:
            True если обновлено успешно
        """
        try:
            async with db_manager.session() as session:
                stmt = (
                    update(BacktestRun)
                    .where(BacktestRun.id == backtest_id)
                    .values(
                        final_capital=final_capital,
                        total_pnl=total_pnl,
                        total_pnl_pct=total_pnl_pct,
                        total_trades=total_trades,
                        winning_trades=winning_trades,
                        losing_trades=losing_trades,
                        metrics=metrics
                    )
                )

                await session.execute(stmt)
                await session.commit()

                logger.info(
                    f"Обновлены результаты backtest {backtest_id}: "
                    f"PnL={total_pnl:.2f} ({total_pnl_pct:.2f}%), Trades={total_trades}"
                )
                return True

        except Exception as e:
            logger.error(f"Ошибка обновления результатов backtest {backtest_id}: {e}")
            raise DatabaseError(f"Failed to update backtest results: {str(e)}")

    async def delete_run(self, backtest_id: uuid.UUID) -> bool:
        """
        Удалить backtest run (CASCADE удалит trades и equity).

        Args:
            backtest_id: ID бэктеста

        Returns:
            True если удалено успешно
        """
        try:
            async with db_manager.session() as session:
                stmt = delete(BacktestRun).where(BacktestRun.id == backtest_id)
                result = await session.execute(stmt)
                await session.commit()

                deleted = result.rowcount > 0
                if deleted:
                    logger.info(f"Удален backtest run {backtest_id}")
                else:
                    logger.warning(f"Backtest run {backtest_id} не найден")

                return deleted

        except Exception as e:
            logger.error(f"Ошибка удаления backtest {backtest_id}: {e}")
            raise DatabaseError(f"Failed to delete backtest: {str(e)}")

    # ==================== BacktestTrade CRUD ====================

    async def create_trade(
        self,
        backtest_run_id: uuid.UUID,
        symbol: str,
        side: OrderSide,
        entry_time: datetime,
        entry_price: float,
        quantity: float,
        exit_time: Optional[datetime] = None,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        commission: float = 0.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        entry_signal: Optional[Dict] = None,
        exit_reason: Optional[str] = None,
        exit_signal: Optional[Dict] = None,
        max_favorable_excursion: Optional[float] = None,
        max_adverse_excursion: Optional[float] = None,
        duration_seconds: Optional[float] = None,
        entry_market_data: Optional[Dict] = None,
        exit_market_data: Optional[Dict] = None
    ) -> BacktestTrade:
        """
        Создать запись trade.

        Args:
            backtest_run_id: ID родительского backtest run
            symbol: Торговая пара
            side: Сторона (Buy/Sell)
            entry_time: Время входа
            entry_price: Цена входа
            quantity: Объем
            exit_time: Время выхода
            exit_price: Цена выхода
            pnl: PnL
            pnl_pct: PnL в процентах
            commission: Комиссия
            stop_loss: Stop Loss
            take_profit: Take Profit
            entry_signal: Данные сигнала входа
            exit_reason: Причина выхода
            exit_signal: Данные сигнала выхода
            max_favorable_excursion: MFE
            max_adverse_excursion: MAE
            duration_seconds: Длительность в секундах
            entry_market_data: Рыночные данные при входе
            exit_market_data: Рыночные данные при выходе

        Returns:
            BacktestTrade
        """
        try:
            async with db_manager.session() as session:
                trade = BacktestTrade(
                    id=uuid.uuid4(),
                    backtest_run_id=backtest_run_id,
                    symbol=symbol,
                    side=side,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    quantity=quantity,
                    exit_time=exit_time,
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    commission=commission,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    entry_signal=entry_signal,
                    exit_reason=exit_reason,
                    exit_signal=exit_signal,
                    max_favorable_excursion=max_favorable_excursion,
                    max_adverse_excursion=max_adverse_excursion,
                    duration_seconds=duration_seconds,
                    entry_market_data=entry_market_data,
                    exit_market_data=exit_market_data
                )

                session.add(trade)
                await session.commit()
                await session.refresh(trade)

                return trade

        except Exception as e:
            logger.error(f"Ошибка создания trade: {e}")
            raise DatabaseError(f"Failed to create trade: {str(e)}")

    async def get_trades(
        self,
        backtest_run_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0
    ) -> List[BacktestTrade]:
        """
        Получить trades для backtest run.

        Args:
            backtest_run_id: ID бэктеста
            limit: Лимит результатов
            offset: Offset для пагинации

        Returns:
            Список BacktestTrade
        """
        try:
            async with db_manager.session() as session:
                query = (
                    select(BacktestTrade)
                    .where(BacktestTrade.backtest_run_id == backtest_run_id)
                    .order_by(BacktestTrade.entry_time.asc())
                    .limit(limit)
                    .offset(offset)
                )

                result = await session.execute(query)
                trades = result.scalars().all()

                return list(trades)

        except Exception as e:
            logger.error(f"Ошибка получения trades: {e}")
            raise DatabaseError(f"Failed to get trades: {str(e)}")

    # ==================== BacktestEquity CRUD ====================

    async def create_equity_point(
        self,
        backtest_run_id: uuid.UUID,
        timestamp: datetime,
        sequence: int,
        equity: float,
        cash: float,
        positions_value: float,
        peak_equity: float,
        drawdown: float,
        drawdown_pct: float,
        total_return: float,
        total_return_pct: float,
        open_positions_count: int
    ) -> BacktestEquity:
        """
        Создать точку equity curve.

        Args:
            backtest_run_id: ID бэктеста
            timestamp: Временная метка
            sequence: Порядковый номер
            equity: Капитал
            cash: Свободные средства
            positions_value: Стоимость позиций
            peak_equity: Пиковый капитал
            drawdown: Абсолютная просадка
            drawdown_pct: Процентная просадка
            total_return: Абсолютная доходность
            total_return_pct: Процентная доходность
            open_positions_count: Количество открытых позиций

        Returns:
            BacktestEquity
        """
        try:
            async with db_manager.session() as session:
                equity_point = BacktestEquity(
                    id=uuid.uuid4(),
                    backtest_run_id=backtest_run_id,
                    timestamp=timestamp,
                    sequence=sequence,
                    equity=equity,
                    cash=cash,
                    positions_value=positions_value,
                    peak_equity=peak_equity,
                    drawdown=drawdown,
                    drawdown_pct=drawdown_pct,
                    total_return=total_return,
                    total_return_pct=total_return_pct,
                    open_positions_count=open_positions_count
                )

                session.add(equity_point)
                await session.commit()
                await session.refresh(equity_point)

                return equity_point

        except Exception as e:
            logger.error(f"Ошибка создания equity point: {e}")
            raise DatabaseError(f"Failed to create equity point: {str(e)}")

    async def get_equity_curve(
        self,
        backtest_run_id: uuid.UUID
    ) -> List[BacktestEquity]:
        """
        Получить полную equity curve для backtest run.

        Args:
            backtest_run_id: ID бэктеста

        Returns:
            Список BacktestEquity, отсортированный по sequence
        """
        try:
            async with db_manager.session() as session:
                query = (
                    select(BacktestEquity)
                    .where(BacktestEquity.backtest_run_id == backtest_run_id)
                    .order_by(BacktestEquity.sequence.asc())
                )

                result = await session.execute(query)
                equity_points = result.scalars().all()

                return list(equity_points)

        except Exception as e:
            logger.error(f"Ошибка получения equity curve: {e}")
            raise DatabaseError(f"Failed to get equity curve: {str(e)}")

    # ==================== Statistics ====================

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Получить общую статистику по бэктестам.

        Returns:
            Dict с статистикой
        """
        try:
            async with db_manager.session() as session:
                # Total runs
                total_query = select(func.count(BacktestRun.id))
                total_result = await session.execute(total_query)
                total_runs = total_result.scalar() or 0

                # Runs by status
                status_query = (
                    select(
                        BacktestRun.status,
                        func.count(BacktestRun.id)
                    )
                    .group_by(BacktestRun.status)
                )
                status_result = await session.execute(status_query)
                runs_by_status = {row[0].value: row[1] for row in status_result}

                # Recent runs
                recent_query = (
                    select(BacktestRun)
                    .order_by(BacktestRun.created_at.desc())
                    .limit(10)
                )
                recent_result = await session.execute(recent_query)
                recent_runs = recent_result.scalars().all()

                return {
                    'total_runs': total_runs,
                    'runs_by_status': runs_by_status,
                    'recent_runs': [
                        {
                            'id': str(run.id),
                            'name': run.name,
                            'symbol': run.symbol,
                            'status': run.status.value,
                            'created_at': run.created_at.isoformat() if run.created_at else None,
                            'total_pnl_pct': run.total_pnl_pct
                        }
                        for run in recent_runs
                    ]
                }

        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            raise DatabaseError(f"Failed to get statistics: {str(e)}")


# Singleton instance
backtest_repository = BacktestRepository()

__all__ = ['BacktestRepository', 'backtest_repository']
