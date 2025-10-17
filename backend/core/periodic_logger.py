"""
Periodic Logger - утилита для ограничения частоты логов.

backend/core/periodic_logger.py
"""

from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class LogRecord:
  """Запись о последнем логе."""
  count: int = 0
  last_logged: Optional[datetime] = None


class PeriodicLogger:
  """
  Менеджер периодического логирования.

  Позволяет логировать сообщения только каждые N раз,
  чтобы избежать засорения логов повторяющимися сообщениями.
  """

  def __init__(self):
    """Инициализация."""
    # Хранилище счётчиков по ключам
    self.records: Dict[str, LogRecord] = {}

  def should_log(
      self,
      key: str,
      every_n: int = 500,
      first_n: int = 1
  ) -> tuple[bool, int]:
    """
    Проверка, нужно ли логировать.

    Args:
        key: Уникальный ключ для группировки логов
        every_n: Логировать каждые N раз
        first_n: Логировать первые N раз

    Returns:
        tuple[bool, int]: (нужно_логировать, текущий_счётчик)
    """
    if key not in self.records:
      self.records[key] = LogRecord()

    record = self.records[key]
    record.count += 1

    # Логируем первые N раз
    if record.count <= first_n:
      record.last_logged = datetime.now()
      return True, record.count

    # Логируем каждые every_n раз
    if record.count % every_n == 0:
      record.last_logged = datetime.now()
      return True, record.count

    return False, record.count

  def should_log_with_cooldown(
      self,
      key: str,
      cooldown_seconds: int = 60
  ) -> tuple[bool, Optional[float]]:
    """
    Проверка с cooldown периодом.

    Args:
        key: Уникальный ключ
        cooldown_seconds: Минимальное время между логами (сек)

    Returns:
        tuple[bool, Optional[float]]: (нужно_логировать, время_с_последнего)
    """
    if key not in self.records:
      self.records[key] = LogRecord()

    record = self.records[key]
    record.count += 1

    # Первый лог - всегда разрешаем
    if record.last_logged is None:
      record.last_logged = datetime.now()
      return True, None

    # Проверяем cooldown
    time_since = (datetime.now() - record.last_logged).total_seconds()

    if time_since >= cooldown_seconds:
      record.last_logged = datetime.now()
      return True, time_since

    return False, time_since

  def reset(self, key: str):
    """Сбросить счётчик для ключа."""
    if key in self.records:
      del self.records[key]

  def get_count(self, key: str) -> int:
    """Получить текущий счётчик."""
    return self.records.get(key, LogRecord()).count


# Глобальный экземпляр
periodic_logger = PeriodicLogger()