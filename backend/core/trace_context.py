"""
Trace Context для распространения Trace ID через систему.
Позволяет связывать логи распределенных операций.
"""

import uuid
import contextvars
from typing import Optional, Dict, Any
from contextlib import contextmanager
from datetime import datetime

from core.logger import get_logger

logger = get_logger(__name__)

# Context var для хранения trace ID
_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
  'trace_id', default=None
)

_correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
  'correlation_id', default=None
)

_trace_metadata_var: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
  'trace_metadata', default={}
)


class TraceContext:
  """
  Управление Trace ID для распределенной трассировки.
  Автоматически добавляет trace_id ко всем логам.
  """

  @staticmethod
  def get_trace_id() -> Optional[str]:
    """
    Получение текущего Trace ID.

    Returns:
        Optional[str]: Trace ID или None
    """
    return _trace_id_var.get()

  @staticmethod
  def get_correlation_id() -> Optional[str]:
    """
    Получение текущего Correlation ID.

    Returns:
        Optional[str]: Correlation ID или None
    """
    return _correlation_id_var.get()

  @staticmethod
  def set_trace_id(trace_id: str):
    """
    Установка Trace ID в текущий контекст.

    Args:
        trace_id: Trace ID
    """
    _trace_id_var.set(trace_id)

  @staticmethod
  def set_correlation_id(correlation_id: str):
    """
    Установка Correlation ID в текущий контекст.

    Args:
        correlation_id: Correlation ID
    """
    _correlation_id_var.set(correlation_id)

  @staticmethod
  def generate_trace_id() -> str:
    """
    Генерация нового Trace ID.

    Returns:
        str: Новый Trace ID
    """
    return str(uuid.uuid4())

  @staticmethod
  def set_metadata(key: str, value: Any):
    """
    Добавление метаданных к trace.

    Args:
        key: Ключ метаданных
        value: Значение
    """
    metadata = _trace_metadata_var.get().copy()
    metadata[key] = value
    _trace_metadata_var.set(metadata)

  @staticmethod
  def get_metadata() -> Dict[str, Any]:
    """
    Получение метаданных trace.

    Returns:
        Dict: Метаданные
    """
    return _trace_metadata_var.get().copy()

  @staticmethod
  def clear():
    """Очистка контекста трассировки."""
    _trace_id_var.set(None)
    _correlation_id_var.set(None)
    _trace_metadata_var.set({})


@contextmanager
def trace_operation(operation_name: str, **metadata):
  """
  Context manager для трассировки операции.

  Args:
      operation_name: Название операции
      **metadata: Дополнительные метаданные

  Example:
      with trace_operation("place_order", symbol="BTCUSDT"):
          # Все логи внутри будут иметь trace_id
          await place_order(...)
  """
  # Генерируем или наследуем trace_id
  trace_id = TraceContext.get_trace_id()
  if not trace_id:
    trace_id = TraceContext.generate_trace_id()
    TraceContext.set_trace_id(trace_id)
    is_new_trace = True
  else:
    is_new_trace = False

  # Устанавливаем метаданные
  TraceContext.set_metadata("operation", operation_name)
  TraceContext.set_metadata("started_at", datetime.utcnow().isoformat())

  for key, value in metadata.items():
    TraceContext.set_metadata(key, value)

  logger.info(
    f"→ Начало операции: {operation_name}",
    trace_id=trace_id,
    operation=operation_name,
    **metadata
  )

  start_time = datetime.utcnow()

  try:
    yield trace_id

    # Успешное завершение
    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
    logger.info(
      f"✓ Операция завершена: {operation_name} ({duration_ms:.2f}ms)",
      trace_id=trace_id,
      operation=operation_name,
      duration_ms=duration_ms,
      success=True,
    )

  except Exception as e:
    # Ошибка
    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
    logger.error(
      f"✗ Ошибка операции: {operation_name} ({duration_ms:.2f}ms): {e}",
      trace_id=trace_id,
      operation=operation_name,
      duration_ms=duration_ms,
      success=False,
      error=str(e),
    )
    raise

  finally:
    # Очищаем контекст только если это новый trace
    if is_new_trace:
      TraceContext.clear()


def with_trace(operation_name: str):
  """
  Декоратор для автоматической трассировки функции.

  Args:
      operation_name: Название операции

  Example:
      @with_trace("calculate_metrics")
      async def calculate_metrics(symbol: str):
          ...
  """

  def decorator(func):
    async def async_wrapper(*args, **kwargs):
      with trace_operation(operation_name):
        return await func(*args, **kwargs)

    def sync_wrapper(*args, **kwargs):
      with trace_operation(operation_name):
        return func(*args, **kwargs)

    # Определяем асинхронная функция или нет
    import asyncio
    if asyncio.iscoroutinefunction(func):
      return async_wrapper
    else:
      return sync_wrapper

  return decorator


# Monkey-patch logger для автоматического добавления trace_id
def inject_trace_id_to_logger():
  """
  Расширение logger для автоматического добавления trace_id.
  """
  original_log = logger._log

  def patched_log(level, msg, *args, **kwargs):
    # Добавляем trace_id если он есть
    trace_id = TraceContext.get_trace_id()
    if trace_id:
      kwargs['trace_id'] = trace_id

    # Добавляем correlation_id если он есть
    correlation_id = TraceContext.get_correlation_id()
    if correlation_id:
      kwargs['correlation_id'] = correlation_id

    return original_log(level, msg, *args, **kwargs)

  logger._log = patched_log


# Применяем патч при импорте
inject_trace_id_to_logger()