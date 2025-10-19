"""
Сервис уведомлений (Email + Telegram).

ТЕКУЩАЯ ВЕРСИЯ: Заглушки для будущей реализации.
Вся логика логируется через logger с соответствующим уровнем.

Путь: backend/services/notification_service.py
"""
from typing import Optional, Dict
from datetime import datetime

from core.logger import get_logger
from config import settings

logger = get_logger(__name__)


class NotificationService:
  """
  Централизованный сервис уведомлений.

  Поддерживает три типа сообщений:
  - CRITICAL: Emergency shutdown, Daily Loss Killer
  - WARNING: Daily loss warning, корреляционные конфликты
  - INFO: Общие информационные сообщения
  """

  def __init__(self):
    """Инициализация сервиса."""
    self.email_enabled = settings.NOTIFICATION_EMAIL_ENABLED
    self.telegram_enabled = settings.NOTIFICATION_TELEGRAM_ENABLED

    logger.info(
      f"NotificationService инициализирован: "
      f"email={self.email_enabled}, telegram={self.telegram_enabled}"
    )

  async def send_critical_alert(
      self,
      title: str,
      message: str,
      context: Optional[Dict] = None
  ):
    """
    Отправка критического алерта.

    Используется для:
    - Daily Loss Killer emergency shutdown
    - Критические ошибки риск-менеджмента
    - Детекция подозрительной активности

    Args:
        title: Заголовок алерта
        message: Текст сообщения
        context: Дополнительный контекст (dict)
    """
    timestamp = datetime.now().isoformat()

    logger.critical("=" * 80)
    logger.critical(f"🚨 CRITICAL ALERT: {title}")
    logger.critical(f"Время: {timestamp}")
    logger.critical(f"Сообщение: {message}")

    if context:
      logger.critical("Контекст:")
      for key, value in context.items():
        logger.critical(f"  • {key}: {value}")

    logger.critical("=" * 80)

    # TODO: Реальная отправка
    if self.email_enabled:
      await self._send_email_stub(title, message, context)

    if self.telegram_enabled:
      await self._send_telegram_stub(title, message, context)

  async def send_warning(
      self,
      title: str,
      message: str,
      context: Optional[Dict] = None
  ):
    """
    Отправка предупреждения.

    Используется для:
    - Приближение к daily loss limit
    - Превышение корреляционных лимитов
    - Детекция потенциальных проблем

    Args:
        title: Заголовок предупреждения
        message: Текст сообщения
        context: Дополнительный контекст
    """
    timestamp = datetime.now().isoformat()

    logger.warning("=" * 60)
    logger.warning(f"⚠️ WARNING: {title}")
    logger.warning(f"Время: {timestamp}")
    logger.warning(f"Сообщение: {message}")

    if context:
      logger.warning("Контекст:")
      for key, value in context.items():
        logger.warning(f"  • {key}: {value}")

    logger.warning("=" * 60)

    # TODO: Реальная отправка
    if self.email_enabled:
      await self._send_email_stub(title, message, context)

    if self.telegram_enabled:
      await self._send_telegram_stub(title, message, context)

  async def send_info(
      self,
      title: str,
      message: str
  ):
    """
    Отправка информационного сообщения.

    Используется для:
    - Ежедневные отчеты
    - Статистика по сделкам
    - Общие уведомления

    Args:
        title: Заголовок
        message: Текст сообщения
    """
    logger.info(f"ℹ️ INFO: {title} - {message}")

    # TODO: Реальная отправка для важных info сообщений

  async def _send_email_stub(
      self,
      title: str,
      message: str,
      context: Optional[Dict]
  ):
    """
    Заглушка Email отправки.

    TODO: Реализовать с использованием:
    - SMTP для отправки
    - Шаблоны HTML писем
    - Приоритизация критических алертов
    """
    logger.info(
      f"[EMAIL STUB] Would send email: "
      f"title='{title}', message='{message}'"
    )

  async def _send_telegram_stub(
      self,
      title: str,
      message: str,
      context: Optional[Dict]
  ):
    """
    Заглушка Telegram отправки.

    TODO: Реализовать с использованием:
    - python-telegram-bot или aiogram
    - Форматирование сообщений в Markdown
    - Inline кнопки для критических алертов (Emergency Stop)
    """
    logger.info(
      f"[TELEGRAM STUB] Would send telegram: "
      f"title='{title}', message='{message}'"
    )


# Глобальный экземпляр
notification_service = NotificationService()