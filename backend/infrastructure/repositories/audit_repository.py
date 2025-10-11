"""
Audit Repository.
Неизменяемый аудит всех критических операций.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy import select

from core.logger import get_logger
from database.connection import db_manager
from database.models import AuditLog, AuditAction

logger = get_logger(__name__)


class AuditRepository:
  """Repository для аудит-логов."""

  async def log(
      self,
      action: AuditAction,
      entity_type: str,
      entity_id: str,
      old_value: Optional[Dict[str, Any]] = None,
      new_value: Optional[Dict[str, Any]] = None,
      changes: Optional[Dict[str, Any]] = None,
      reason: Optional[str] = None,
      user_id: Optional[str] = None,
      trace_id: Optional[str] = None,
      success: bool = True,
      error_message: Optional[str] = None,
      context: Optional[Dict[str, Any]] = None,
  ) -> bool:
    """
    Запись аудит-лога.

    Args:
        action: Тип действия
        entity_type: Тип сущности (Order, Position, etc.)
        entity_id: ID сущности
        old_value: Старое значение
        new_value: Новое значение
        changes: Изменения
        reason: Причина действия
        user_id: ID пользователя
        trace_id: ID трассировки
        success: Успешность операции
        error_message: Сообщение об ошибке
        context: Дополнительный контекст

    Returns:
        bool: True если записано успешно
    """
    try:
      async with db_manager.session() as session:
        audit_entry = AuditLog(
          action=action,
          entity_type=entity_type,
          entity_id=entity_id,
          old_value=old_value,
          new_value=new_value,
          changes=changes,
          reason=reason,
          user_id=user_id,
          trace_id=trace_id,
          success=success,
          error_message=error_message,
          context=context,
          timestamp=datetime.utcnow(),
        )

        session.add(audit_entry)
        await session.commit()

        logger.debug(
          f"Аудит записан: {action.value} | "
          f"{entity_type}:{entity_id} | "
          f"success={success}"
        )

        return True

    except Exception as e:
      logger.error(f"Ошибка записи аудита: {e}")
      # Аудит не должен ломать основную логику
      return False

  async def get_entity_history(
      self,
      entity_type: str,
      entity_id: str,
      limit: int = 100,
  ) -> List[AuditLog]:
    """
    Получение истории изменений сущности.

    Args:
        entity_type: Тип сущности
        entity_id: ID сущности
        limit: Количество записей

    Returns:
        List[AuditLog]: История изменений
    """
    try:
      async with db_manager.session() as session:
        stmt = (
          select(AuditLog)
          .where(
            AuditLog.entity_type == entity_type,
            AuditLog.entity_id == entity_id,
          )
          .order_by(AuditLog.timestamp.desc())
          .limit(limit)
        )

        result = await session.execute(stmt)
        logs = result.scalars().all()

        return list(logs)

    except Exception as e:
      logger.error(f"Ошибка получения истории аудита: {e}")
      return []

  async def get_recent_logs(
      self,
      action: Optional[AuditAction] = None,
      hours: int = 24,
      limit: int = 1000,
  ) -> List[AuditLog]:
    """
    Получение недавних логов.

    Args:
        action: Фильтр по типу действия
        hours: За последние N часов
        limit: Количество записей

    Returns:
        List[AuditLog]: Список логов
    """
    try:
      async with db_manager.session() as session:
        since = datetime.utcnow() - timedelta(hours=hours)

        stmt = select(AuditLog).where(AuditLog.timestamp >= since)

        if action:
          stmt = stmt.where(AuditLog.action == action)

        stmt = stmt.order_by(AuditLog.timestamp.desc()).limit(limit)

        result = await session.execute(stmt)
        logs = result.scalars().all()

        return list(logs)

    except Exception as e:
      logger.error(f"Ошибка получения недавних логов: {e}")
      return []

  async def get_failed_operations(
      self,
      hours: int = 24,
      limit: int = 100,
  ) -> List[AuditLog]:
    """
    Получение неудачных операций.

    Args:
        hours: За последние N часов
        limit: Количество записей

    Returns:
        List[AuditLog]: Список неудачных операций
    """
    try:
      async with db_manager.session() as session:
        since = datetime.utcnow() - timedelta(hours=hours)

        stmt = (
          select(AuditLog)
          .where(
            AuditLog.timestamp >= since,
            AuditLog.success == False,
          )
          .order_by(AuditLog.timestamp.desc())
          .limit(limit)
        )

        result = await session.execute(stmt)
        logs = result.scalars().all()

        logger.debug(f"Найдено {len(logs)} неудачных операций за {hours}ч")
        return list(logs)

    except Exception as e:
      logger.error(f"Ошибка получения неудачных операций: {e}")
      return []


# Глобальный экземпляр
audit_repository = AuditRepository()