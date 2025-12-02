"""
ML Backtest Repository - CRUD операции для ML бэктестов.

Работа с таблицами:
- ml_backtest_runs
- ml_backtest_predictions
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import select, update, delete, func
from sqlalchemy.orm import selectinload
import uuid

from backend.core.logger import get_logger
from backend.database.connection import db_manager
from backend.database.models import (
    MLBacktestRun,
    MLBacktestPrediction,
    MLBacktestStatus
)
from backend.core.exceptions import DatabaseError

logger = get_logger(__name__)


class MLBacktestRepository:
    """Repository для работы с ML бэктестами."""

    # ==================== MLBacktestRun CRUD ====================

    async def create_run(
        self,
        name: str,
        model_checkpoint: str,
        data_source: str,
        description: Optional[str] = None,
        model_version: Optional[str] = None,
        model_architecture: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        holdout_set_id: Optional[str] = None,
        use_walk_forward: bool = True,
        n_periods: int = 5,
        retrain_each_period: bool = False,
        initial_capital: float = 10000.0,
        position_size: float = 0.1,
        commission: float = 0.001,
        slippage: float = 0.0005,
        use_confidence_filter: bool = True,
        min_confidence: float = 0.6,
        confidence_mode: str = 'threshold',
        sequence_length: int = 60,
        batch_size: int = 128,
        device: str = 'auto'
    ) -> MLBacktestRun:
        """Создать новый ML backtest run."""
        try:
            async with db_manager.session() as session:
                backtest = MLBacktestRun(
                    id=uuid.uuid4(),
                    name=name,
                    description=description,
                    model_checkpoint=model_checkpoint,
                    model_version=model_version,
                    model_architecture=model_architecture,
                    data_source=data_source,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    holdout_set_id=holdout_set_id,
                    use_walk_forward=use_walk_forward,
                    n_periods=n_periods,
                    retrain_each_period=retrain_each_period,
                    initial_capital=initial_capital,
                    position_size=position_size,
                    commission=commission,
                    slippage=slippage,
                    use_confidence_filter=use_confidence_filter,
                    min_confidence=min_confidence,
                    confidence_mode=confidence_mode,
                    sequence_length=sequence_length,
                    batch_size=batch_size,
                    device=device,
                    status=MLBacktestStatus.PENDING,
                    progress_pct=0.0,
                    created_at=datetime.utcnow()
                )

                session.add(backtest)
                await session.commit()
                await session.refresh(backtest)

                logger.info(f"Создан ML backtest run: {backtest.id} | {name}")
                return backtest

        except Exception as e:
            logger.error(f"Ошибка создания ML backtest run: {e}", exc_info=True)
            raise DatabaseError(f"Failed to create ML backtest run: {str(e)}")

    async def get_run(self, run_id: uuid.UUID) -> Optional[MLBacktestRun]:
        """Получить ML backtest run по ID."""
        try:
            async with db_manager.session() as session:
                result = await session.execute(
                    select(MLBacktestRun).where(MLBacktestRun.id == run_id)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Ошибка получения ML backtest run {run_id}: {e}")
            raise DatabaseError(f"Failed to get ML backtest run: {str(e)}")

    async def list_runs(
        self,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[MLBacktestRun]:
        """Получить список ML backtest runs."""
        try:
            async with db_manager.session() as session:
                query = select(MLBacktestRun)

                if status:
                    query = query.where(MLBacktestRun.status == status)

                query = query.order_by(MLBacktestRun.created_at.desc())
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Ошибка получения списка ML backtests: {e}")
            raise DatabaseError(f"Failed to list ML backtest runs: {str(e)}")

    async def update_run(
        self,
        run_id: uuid.UUID,
        **updates
    ) -> Optional[MLBacktestRun]:
        """Обновить ML backtest run."""
        try:
            async with db_manager.session() as session:
                result = await session.execute(
                    select(MLBacktestRun).where(MLBacktestRun.id == run_id)
                )
                backtest = result.scalar_one_or_none()

                if not backtest:
                    return None

                for key, value in updates.items():
                    if hasattr(backtest, key):
                        setattr(backtest, key, value)

                await session.commit()
                await session.refresh(backtest)
                return backtest
        except Exception as e:
            logger.error(f"Ошибка обновления ML backtest run {run_id}: {e}")
            raise DatabaseError(f"Failed to update ML backtest run: {str(e)}")

    async def delete_run(self, run_id: uuid.UUID) -> bool:
        """Удалить ML backtest run."""
        try:
            async with db_manager.session() as session:
                result = await session.execute(
                    delete(MLBacktestRun).where(MLBacktestRun.id == run_id)
                )
                await session.commit()
                return result.rowcount > 0
        except Exception as e:
            logger.error(f"Ошибка удаления ML backtest run {run_id}: {e}")
            raise DatabaseError(f"Failed to delete ML backtest run: {str(e)}")

    async def count_runs(self, status: Optional[str] = None) -> int:
        """Подсчитать количество ML backtest runs."""
        try:
            async with db_manager.session() as session:
                query = select(func.count(MLBacktestRun.id))
                if status:
                    query = query.where(MLBacktestRun.status == status)
                result = await session.execute(query)
                return result.scalar() or 0
        except Exception as e:
            logger.error(f"Ошибка подсчёта ML backtests: {e}")
            return 0

    # ==================== Predictions ====================

    async def add_predictions(
        self,
        run_id: uuid.UUID,
        predictions: List[Dict[str, Any]]
    ) -> int:
        """Добавить предсказания для бэктеста."""
        try:
            async with db_manager.session() as session:
                for pred in predictions:
                    prediction = MLBacktestPrediction(
                        id=uuid.uuid4(),
                        backtest_run_id=run_id,
                        sequence=pred.get('sequence', 0),
                        timestamp=pred.get('timestamp'),
                        predicted_class=pred.get('predicted_class'),
                        actual_class=pred.get('actual_class'),
                        confidence=pred.get('confidence'),
                        period=pred.get('period')
                    )
                    session.add(prediction)

                await session.commit()
                return len(predictions)
        except Exception as e:
            logger.error(f"Ошибка добавления предсказаний: {e}")
            raise DatabaseError(f"Failed to add predictions: {str(e)}")

    async def get_predictions(
        self,
        run_id: uuid.UUID,
        limit: int = 1000,
        period: Optional[int] = None
    ) -> List[MLBacktestPrediction]:
        """Получить предсказания бэктеста."""
        try:
            async with db_manager.session() as session:
                query = select(MLBacktestPrediction).where(
                    MLBacktestPrediction.backtest_run_id == run_id
                )

                if period is not None:
                    query = query.where(MLBacktestPrediction.period == period)

                query = query.order_by(MLBacktestPrediction.sequence)
                query = query.limit(limit)

                result = await session.execute(query)
                return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Ошибка получения предсказаний: {e}")
            raise DatabaseError(f"Failed to get predictions: {str(e)}")


# Global repository instance
ml_backtest_repo = MLBacktestRepository()
