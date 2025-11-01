"""
Circuit Breaker Pattern.
Защита от каскадных сбоев при вызовах API биржи.
"""

import time
from typing import Callable, Any, Optional
from enum import Enum
from functools import wraps
from datetime import datetime, timedelta

from backend.core.logger import get_logger
from backend.core.exceptions import CircuitBreakerError

logger = get_logger(__name__)


class CircuitState(str, Enum):
  """Состояния Circuit Breaker."""
  CLOSED = "closed"  # Нормальная работа
  OPEN = "open"  # Предохранитель сработал, блокируем вызовы
  HALF_OPEN = "half_open"  # Пробуем восстановиться


class CircuitBreaker:
  """
  Circuit Breaker для защиты от каскадных сбоев.

  Паттерн:
  - CLOSED: Нормальная работа, вызовы проходят
  - OPEN: После N ошибок блокируем вызовы на cooldown период
  - HALF_OPEN: Пробуем восстановиться, пропускаем 1 вызов
  """

  def __init__(
      self,
      name: str,
      failure_threshold: int = 5,
      success_threshold: int = 2,
      timeout_seconds: int = 60,
      cooldown_seconds: int = 60,
  ):
    """
    Инициализация Circuit Breaker.

    Args:
        name: Имя предохранителя
        failure_threshold: Количество ошибок для срабатывания
        success_threshold: Количество успехов для восстановления
        timeout_seconds: Таймаут ожидания ответа
        cooldown_seconds: Период охлаждения после срабатывания
    """
    self.name = name
    self.failure_threshold = failure_threshold
    self.success_threshold = success_threshold
    self.timeout_seconds = timeout_seconds
    self.cooldown_seconds = cooldown_seconds

    # Состояние
    self.state = CircuitState.CLOSED
    self.failure_count = 0
    self.success_count = 0
    self.last_failure_time: Optional[float] = None
    self.opened_at: Optional[datetime] = None

    logger.info(
      f"Circuit Breaker '{name}' инициализирован: "
      f"failure_threshold={failure_threshold}, "
      f"cooldown={cooldown_seconds}s"
    )

  def call(self, func: Callable, *args, **kwargs) -> Any:
    """
    Выполнение вызова через предохранитель.

    Args:
        func: Функция для вызова
        *args: Позиционные аргументы
        **kwargs: Именованные аргументы

    Returns:
        Any: Результат вызова функции

    Raises:
        CircuitBreakerError: Если предохранитель сработал
    """
    # Проверяем состояние
    if self.state == CircuitState.OPEN:
      if self._should_attempt_reset():
        self._transition_to_half_open()
      else:
        # Блокируем вызов
        logger.warning(
          f"Circuit Breaker '{self.name}' OPEN, "
          f"вызов заблокирован"
        )
        raise CircuitBreakerError(
          f"Circuit breaker '{self.name}' is OPEN. "
          f"Too many failures, please wait."
        )

    # Выполняем вызов
    try:
      logger.debug(f"Circuit Breaker '{self.name}': выполнение вызова")
      result = func(*args, **kwargs)

      # Успех
      self._on_success()
      return result

    except Exception as e:
      # Ошибка
      self._on_failure()
      logger.error(
        f"Circuit Breaker '{self.name}' зафиксировал ошибку: {e}"
      )
      raise

  async def call_async(self, func: Callable, *args, **kwargs) -> Any:
    """
    Асинхронная версия вызова через предохранитель.

    Args:
        func: Асинхронная функция
        *args: Позиционные аргументы
        **kwargs: Именованные аргументы

    Returns:
        Any: Результат вызова функции
    """
    # Проверяем состояние
    if self.state == CircuitState.OPEN:
      if self._should_attempt_reset():
        self._transition_to_half_open()
      else:
        logger.warning(
          f"Circuit Breaker '{self.name}' OPEN, "
          f"вызов заблокирован"
        )
        raise CircuitBreakerError(
          f"Circuit breaker '{self.name}' is OPEN"
        )

    # Выполняем вызов
    try:
      logger.debug(f"Circuit Breaker '{self.name}': выполнение async вызова")
      result = await func(*args, **kwargs)

      # Успех
      self._on_success()
      return result

    except Exception as e:
      # Ошибка
      self._on_failure()
      logger.error(
        f"Circuit Breaker '{self.name}' зафиксировал ошибку: {e}"
      )
      raise

  def _on_success(self):
    """Обработка успешного вызова."""
    self.failure_count = 0
    self.last_failure_time = None

    if self.state == CircuitState.HALF_OPEN:
      self.success_count += 1
      logger.debug(
        f"Circuit Breaker '{self.name}' HALF_OPEN: "
        f"успех {self.success_count}/{self.success_threshold}"
      )

      if self.success_count >= self.success_threshold:
        self._transition_to_closed()

  def _on_failure(self):
    """Обработка ошибки вызова."""
    self.failure_count += 1
    self.last_failure_time = time.time()

    logger.warning(
      f"Circuit Breaker '{self.name}': "
      f"ошибка {self.failure_count}/{self.failure_threshold}"
    )

    if self.failure_count >= self.failure_threshold:
      self._transition_to_open()

  def _transition_to_open(self):
    """Переход в состояние OPEN (сработал)."""
    if self.state != CircuitState.OPEN:
      self.state = CircuitState.OPEN
      self.opened_at = datetime.utcnow()

      logger.error(
        f"🔴 Circuit Breaker '{self.name}' ОТКРЫТ! "
        f"Достигнут порог ошибок ({self.failure_threshold}). "
        f"Вызовы блокируются на {self.cooldown_seconds}s"
      )

  def _transition_to_half_open(self):
    """Переход в состояние HALF_OPEN (пробуем восстановиться)."""
    if self.state != CircuitState.HALF_OPEN:
      self.state = CircuitState.HALF_OPEN
      self.success_count = 0

      logger.info(
        f"🟡 Circuit Breaker '{self.name}' -> HALF_OPEN. "
        f"Пробуем восстановиться..."
      )

  def _transition_to_closed(self):
    """Переход в состояние CLOSED (восстановлено)."""
    if self.state != CircuitState.CLOSED:
      self.state = CircuitState.CLOSED
      self.failure_count = 0
      self.success_count = 0
      self.opened_at = None

      logger.info(
        f"🟢 Circuit Breaker '{self.name}' -> CLOSED. "
        f"Восстановление успешно!"
      )

  def _should_attempt_reset(self) -> bool:
    """
    Проверка, можно ли попробовать восстановиться.

    Returns:
        bool: True если прошел cooldown период
    """
    if not self.opened_at:
      return False

    elapsed = (datetime.utcnow() - self.opened_at).total_seconds()
    return elapsed >= self.cooldown_seconds

  def reset(self):
    """Ручной сброс предохранителя."""
    logger.info(f"Circuit Breaker '{self.name}' сброшен вручную")
    self._transition_to_closed()

  def get_status(self) -> dict:
    """
    Получение статуса предохранителя.

    Returns:
        dict: Статус предохранителя
    """
    return {
      "name": self.name,
      "state": self.state.value,
      "failure_count": self.failure_count,
      "success_count": self.success_count,
      "failure_threshold": self.failure_threshold,
      "cooldown_seconds": self.cooldown_seconds,
      "opened_at": self.opened_at.isoformat() if self.opened_at else None,
    }


class CircuitBreakerManager:
  """Менеджер для управления несколькими Circuit Breakers."""

  def __init__(self):
    """Инициализация менеджера."""
    self.breakers: dict[str, CircuitBreaker] = {}
    logger.info("Circuit Breaker Manager инициализирован")

  def get_breaker(
      self,
      name: str,
      failure_threshold: int = 5,
      cooldown_seconds: int = 60,
  ) -> CircuitBreaker:
    """
    Получение или создание Circuit Breaker.

    Args:
        name: Имя предохранителя
        failure_threshold: Порог ошибок
        cooldown_seconds: Период охлаждения

    Returns:
        CircuitBreaker: Экземпляр предохранителя
    """
    if name not in self.breakers:
      self.breakers[name] = CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        cooldown_seconds=cooldown_seconds,
      )
      logger.debug(f"Создан Circuit Breaker: {name}")

    return self.breakers[name]

  def get_all_status(self) -> dict:
    """
    Получение статуса всех предохранителей.

    Returns:
        dict: Статусы всех предохранителей
    """
    return {
      name: breaker.get_status()
      for name, breaker in self.breakers.items()
    }

  def reset_all(self):
    """Сброс всех предохранителей."""
    for breaker in self.breakers.values():
      breaker.reset()
    logger.info("Все Circuit Breakers сброшены")


# Глобальный менеджер
circuit_breaker_manager = CircuitBreakerManager()