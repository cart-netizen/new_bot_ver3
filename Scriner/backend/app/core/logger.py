# backend/app/core/logger.py

import sys
from loguru import logger
from ..config import settings # <-- Исправлено


def setup_logger():
  """
  Настраивает глобальный логгер для всего приложения.

  Использует уровень логирования из файла конфигурации.
  Формат логов включает время, уровень, модуль, функцию и сообщение.
  - DEBUG уровень для детальной отладки.
  - INFO и выше для продакшена.
  """
  # Удаляем стандартный обработчик, чтобы избежать дублирования
  logger.remove()

  # Формат вывода логов
  log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
  )

  # Добавляем обработчик в stdout с заданным уровнем и форматом
  logger.add(
    sys.stdout,
    level=settings.log_level.upper(),
    format=log_format,
    colorize=True,
    backtrace=True,  # Полезно для отладки, показывает полный стектрейс
    diagnose=True  # Добавляет доп. информацию для отладки исключений
  )

  # Можно также добавить обработчик для записи в файл
  # logger.add(
  #     "logs/screener_{time}.log",
  #     level="INFO",
  #     rotation="10 MB", # Ротация логов при достижении 10 MB
  #     retention="7 days", # Хранить логи за последние 7 дней
  #     format=log_format,
  #     encoding="utf-8"
  # )

  logger.info("Система логирования успешно настроена.")
  logger.debug(f"Установлен уровень логирования: {settings.log_level}")


# Экспортируем настроенный логгер для использования в других модулях
# Это гарантирует, что все части приложения используют один и тот же экземпляр логгера
log = logger