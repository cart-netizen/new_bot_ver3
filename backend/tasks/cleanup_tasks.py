"""
Периодические задачи очистки.
"""

import asyncio
from datetime import datetime, timedelta

from backend.core.logger import get_logger
from backend.domain.services.idempotency_service import idempotency_service
from backend.infrastructure.resilience.recovery_service import recovery_service

logger = get_logger(__name__)


class CleanupTasks:
  """Периодические задачи очистки и обслуживания."""

  def __init__(self):
    """Инициализация."""
    self.running = False
    self.tasks = []

  async def start(self):
    """Запуск периодических задач."""
    self.running = True

    # Cleanup idempotency cache каждый час
    self.tasks.append(
      asyncio.create_task(self._cleanup_idempotency_loop())
    )

    # Reconciliation каждый час (если включено)
    from backend.config import settings
    if settings.AUTO_RECONCILE_ON_STARTUP:
      self.tasks.append(
        asyncio.create_task(self._reconciliation_loop())
      )

    logger.info("Cleanup tasks запущены")

  async def stop(self):
    """Остановка задач."""
    self.running = False

    for task in self.tasks:
      task.cancel()

    await asyncio.gather(*self.tasks, return_exceptions=True)
    logger.info("Cleanup tasks остановлены")

  async def _cleanup_idempotency_loop(self):
    """Периодическая очистка idempotency cache."""
    while self.running:
      try:
        await asyncio.sleep(3600)  # Каждый час

        deleted = await idempotency_service.cleanup_expired()
        if deleted > 0:
          logger.info(f"Очищено {deleted} истекших idempotency записей")

      except asyncio.CancelledError:
        break
      except Exception as e:
        logger.error(f"Ошибка cleanup idempotency: {e}")

  async def _reconciliation_loop(self):
    """Периодическая сверка состояния."""
    from backend.config import settings
    interval_minutes = settings.RECONCILE_INTERVAL_MINUTES

    while self.running:
      try:
        await asyncio.sleep(interval_minutes * 60)

        logger.info("→ Периодическая reconciliation...")
        result = await recovery_service.reconcile_state()
        logger.info(f"✓ Reconciliation завершена: {result}")

      except asyncio.CancelledError:
        break
      except Exception as e:
        logger.error(f"Ошибка reconciliation: {e}")


# Глобальный экземпляр
cleanup_tasks = CleanupTasks()