"""
Вспомогательные функции и утилиты.
"""

import time
import asyncio
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union
from functools import wraps
from datetime import datetime

from backend.core.logger import get_logger
from backend.models.signal import SignalType, SignalStrength

logger = get_logger(__name__)

T = TypeVar('T')


def get_timestamp_ms() -> int:
  """
  Получение текущего timestamp в миллисекундах.

  Returns:
      int: Unix timestamp в миллисекундах
  """
  return int(time.time() * 1000)


def get_timestamp_s() -> int:
  """
  Получение текущего timestamp в секундах.

  Returns:
      int: Unix timestamp в секундах
  """
  return int(time.time())


def format_timestamp(timestamp_ms: int) -> str:
  """
  Форматирование timestamp в читаемый формат.

  Args:
      timestamp_ms: Timestamp в миллисекундах

  Returns:
      str: Отформатированная дата и время
  """
  dt = datetime.fromtimestamp(timestamp_ms / 1000)
  return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def round_price(price: float, decimals: int = 8) -> float:
  """
  Округление цены до заданного количества знаков.

  Args:
      price: Цена
      decimals: Количество знаков после запятой

  Returns:
      float: Округленная цена
  """
  return round(price, decimals)


def round_quantity(quantity: float, decimals: int = 8) -> float:
  """
  Округление количества до заданного количества знаков.

  Args:
      quantity: Количество
      decimals: Количество знаков после запятой

  Returns:
      float: Округленное количество
  """
  return round(quantity, decimals)


def truncate_float(value: float, decimals: int = 8) -> float:
  """
  Обрезание float до заданного количества знаков (без округления).

  Args:
      value: Значение
      decimals: Количество знаков после запятой

  Returns:
      float: Обрезанное значение
  """
  multiplier = 10 ** decimals
  return int(value * multiplier) / multiplier


def calculate_percentage(value: float, total: float) -> float:
  """
  Расчет процента от общего значения.

  Args:
      value: Значение
      total: Общее значение

  Returns:
      float: Процент (0-100)
  """
  if total == 0:
    return 0.0
  return (value / total) * 100


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
  """
  Безопасное деление с обработкой деления на ноль.

  Args:
      numerator: Числитель
      denominator: Знаменатель
      default: Значение по умолчанию при делении на ноль

  Returns:
      float: Результат деления или значение по умолчанию
  """
  if denominator == 0:
    return default
  return numerator / denominator


def clamp(value: float, min_value: float, max_value: float) -> float:
  """
  Ограничение значения в заданном диапазоне.

  Args:
      value: Значение
      min_value: Минимальное значение
      max_value: Максимальное значение

  Returns:
      float: Ограниченное значение
  """
  return max(min_value, min(value, max_value))


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
  """
  Декоратор для повторных попыток выполнения асинхронной функции.

  Args:
      max_attempts: Максимальное количество попыток
      delay: Начальная задержка между попытками (секунды)
      backoff: Множитель увеличения задержки
      exceptions: Типы исключений для обработки
  """

  def decorator(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs):
      current_delay = delay
      last_exception = None

      for attempt in range(1, max_attempts + 1):
        try:
          return await func(*args, **kwargs)
        except exceptions as e:
          last_exception = e

          # Проверяем на ошибки, которые не требуют повторных попыток
          error_msg = str(e)

          # Код 110043 "leverage not modified" - leverage уже установлен, это ОК
          if "110043" in error_msg or "leverage not modified" in error_msg:
            logger.debug(
              f"{func.__name__}: Leverage уже установлен (код 110043), "
              "повторные попытки не требуются"
            )
            raise last_exception  # Пробрасываем сразу, без retry

          if attempt < max_attempts:
            logger.warning(
              f"Попытка {attempt}/{max_attempts} не удалась для {func.__name__}: {e}. "
              f"Повтор через {current_delay}с"
            )
            await asyncio.sleep(current_delay)
            current_delay *= backoff
          else:
            logger.error(
              f"Все {max_attempts} попытки исчерпаны для {func.__name__}: {e}"
            )

      raise last_exception

    return wrapper

  return decorator


def retry_sync(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
  """
  Декоратор для повторных попыток выполнения синхронной функции.

  Args:
      max_attempts: Максимальное количество попыток
      delay: Начальная задержка между попытками (секунды)
      backoff: Множитель увеличения задержки
      exceptions: Типы исключений для обработки
  """

  def decorator(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
      current_delay = delay
      last_exception = None

      for attempt in range(1, max_attempts + 1):
        try:
          return func(*args, **kwargs)
        except exceptions as e:
          last_exception = e
          # Проверяем на ошибки, которые не требуют повторных попыток
          error_msg = str(e)

          # Код 110043 "leverage not modified" - leverage уже установлен, это ОК
          if "110043" in error_msg or "leverage not modified" in error_msg:
            logger.debug(
              f"{func.__name__}: Leverage уже установлен (код 110043), "
              "повторные попытки не требуются"
            )
            raise last_exception  # Пробрасываем сразу, без retry
          if attempt < max_attempts:
            logger.warning(
              f"Попытка {attempt}/{max_attempts} не удалась для {func.__name__}: {e}. "
              f"Повтор через {current_delay}с"
            )
            time.sleep(current_delay)
            current_delay *= backoff
          else:
            logger.error(
              f"Все {max_attempts} попытки исчерпаны для {func.__name__}: {e}"
            )

      raise last_exception

    return wrapper

  return decorator


async def wait_with_timeout(
    coro,
    timeout: float,
    timeout_message: str = "Операция превысила таймаут"
):
  """
  Выполнение корутины с таймаутом.

  Args:
      coro: Корутина для выполнения
      timeout: Таймаут в секундах
      timeout_message: Сообщение при таймауте

  Returns:
      Результат корутины

  Raises:
      asyncio.TimeoutError: Если превышен таймаут
  """
  try:
    return await asyncio.wait_for(coro, timeout=timeout)
  except asyncio.TimeoutError:
    logger.error(f"{timeout_message} ({timeout}с)")
    raise asyncio.TimeoutError(timeout_message)


def chunk_list(lst: list, chunk_size: int) -> list:
  """
  Разбиение списка на части заданного размера.

  Args:
      lst: Исходный список
      chunk_size: Размер части

  Returns:
      list: Список частей
  """
  return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(dict1: dict, dict2: dict) -> dict:
  """
  Глубокое слияние двух словарей.

  Args:
      dict1: Первый словарь
      dict2: Второй словарь

  Returns:
      dict: Объединенный словарь
  """
  result = dict1.copy()
  for key, value in dict2.items():
    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
      result[key] = merge_dicts(result[key], value)
    else:
      result[key] = value
  return result


def validate_symbol(symbol: str) -> bool:
  """
  Валидация формата торговой пары.

  Args:
      symbol: Торговая пара

  Returns:
      bool: True если формат корректный
  """
  return symbol.endswith("USDT") and len(symbol) >= 5


def sanitize_string(s: str, max_length: int = 100) -> str:
  """
  Очистка и обрезка строки для безопасного логирования.

  Args:
      s: Исходная строка
      max_length: Максимальная длина

  Returns:
      str: Очищенная строка
  """
  if not s:
    return ""

  # Обрезаем
  if len(s) > max_length:
    s = s[:max_length] + "..."

  # Удаляем управляющие символы
  s = ''.join(char for char in s if char.isprintable())

  return s

def safe_enum_value(obj: Union[Enum, str, Any]) -> str:
    """
    Безопасное получение строкового значения из Enum или строки.

    Args:
        obj: Enum или строка

    Returns:
        str: Строковое значение

    Examples:
        >>> safe_enum_value(SignalType.BUY)
        'BUY'
        >>> safe_enum_value("BUY")
        'BUY'
        >>> safe_enum_value(SignalStrength.WEAK)
        'WEAK'
    """
    if isinstance(obj, Enum):
      return obj.value
    elif isinstance(obj, str):
      return obj
    else:
      # Для любого другого типа пытаемся преобразовать в строку
      return str(obj)



class ExponentialBackoff:
  """Класс для реализации экспоненциального отката."""

  def __init__(
      self,
      initial_delay: float = 1.0,
      max_delay: float = 60.0,
      multiplier: float = 2.0
  ):
    """
    Инициализация.

    Args:
        initial_delay: Начальная задержка
        max_delay: Максимальная задержка
        multiplier: Множитель увеличения
    """
    self.initial_delay = initial_delay
    self.max_delay = max_delay
    self.multiplier = multiplier
    self.current_delay = initial_delay
    self.attempts = 0

  def get_delay(self) -> float:
    """Получение текущей задержки."""
    return self.current_delay

  def increment(self) -> float:
    """
    Увеличение задержки для следующей попытки.

    Returns:
        float: Новая задержка
    """
    self.attempts += 1
    delay = self.current_delay
    self.current_delay = min(self.current_delay * self.multiplier, self.max_delay)
    return delay

  def reset(self):
    """Сброс задержки к начальному значению."""
    self.current_delay = self.initial_delay
    self.attempts = 0

