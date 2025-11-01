"""
Advanced Rate Limiter с Token Bucket алгоритмом.
Динамическая адаптация к лимитам биржи.
"""

import time
import asyncio
from typing import Optional, Dict
from dataclasses import dataclass
from datetime import datetime

from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
  """Конфигурация rate limit для эндпоинта."""
  max_tokens: int  # Максимум токенов в bucket
  refill_rate: float  # Токенов в секунду
  name: str


class TokenBucket:
  """
  Token Bucket алгоритм для rate limiting.
  Более гибкий чем простой счетчик запросов.
  """

  def __init__(
      self,
      max_tokens: int,
      refill_rate: float,
      name: str = "default",
  ):
    """
    Инициализация Token Bucket.

    Args:
        max_tokens: Максимальное количество токенов
        refill_rate: Скорость пополнения (токенов в секунду)
        name: Имя bucket
    """
    self.max_tokens = max_tokens
    self.refill_rate = refill_rate
    self.name = name

    self.tokens = float(max_tokens)
    self.last_refill = time.time()

    logger.debug(
      f"TokenBucket '{name}' создан: "
      f"max_tokens={max_tokens}, refill_rate={refill_rate}/s"
    )

  def _refill(self):
    """Пополнение токенов на основе прошедшего времени."""
    now = time.time()
    elapsed = now - self.last_refill

    # Добавляем токены
    tokens_to_add = elapsed * self.refill_rate
    self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)

    self.last_refill = now

  def consume(self, tokens: int = 1) -> bool:
    """
    Попытка потребить токены.

    Args:
        tokens: Количество токенов

    Returns:
        bool: True если токены доступны
    """
    self._refill()

    if self.tokens >= tokens:
      self.tokens -= tokens
      return True

    return False

  async def consume_async(self, tokens: int = 1, max_wait: float = 10.0) -> bool:
    """
    Асинхронное потребление токенов с ожиданием.

    Args:
        tokens: Количество токенов
        max_wait: Максимальное время ожидания (сек)

    Returns:
        bool: True если токены получены
    """
    start_time = time.time()

    while True:
      if self.consume(tokens):
        return True

      # Проверяем таймаут
      elapsed = time.time() - start_time
      if elapsed >= max_wait:
        logger.warning(
          f"TokenBucket '{self.name}': timeout при ожидании токенов"
        )
        return False

      # Вычисляем время ожидания
      wait_time = tokens / self.refill_rate
      wait_time = min(wait_time, max_wait - elapsed)

      if wait_time > 0:
        await asyncio.sleep(wait_time)

  def get_available_tokens(self) -> float:
    """
    Получение количества доступных токенов.

    Returns:
        float: Доступные токены
    """
    self._refill()
    return self.tokens

  def get_wait_time(self, tokens: int = 1) -> float:
    """
    Расчет времени ожидания для токенов.

    Args:
        tokens: Количество токенов

    Returns:
        float: Время ожидания в секундах
    """
    self._refill()

    if self.tokens >= tokens:
      return 0.0

    needed_tokens = tokens - self.tokens
    return needed_tokens / self.refill_rate

  def get_status(self) -> Dict:
    """
    Получение статуса bucket.

    Returns:
        Dict: Статус
    """
    self._refill()
    return {
      "name": self.name,
      "tokens": self.tokens,
      "max_tokens": self.max_tokens,
      "refill_rate": self.refill_rate,
      "utilization": (1 - self.tokens / self.max_tokens) * 100,
    }


class RateLimiterManager:
  """
  Менеджер Rate Limiters для разных эндпоинтов биржи.
  Автоматически адаптируется к лимитам.
  """

  def __init__(self):
    """Инициализация менеджера."""
    self.buckets: Dict[str, TokenBucket] = {}

    # Лимиты Bybit API
    self._initialize_bybit_limits()

    logger.info("RateLimiterManager инициализирован")

  def _initialize_bybit_limits(self):
    """
    Инициализация лимитов для Bybit API.
    Основано на документации Bybit.
    """
    # REST API лимиты
    self.add_bucket(
      name="rest_public",
      max_tokens=120,
      refill_rate=120 / 60,  # 120 запросов в минуту
    )

    self.add_bucket(
      name="rest_private",
      max_tokens=120,
      refill_rate=120 / 60,  # 120 запросов в минуту
    )

    self.add_bucket(
      name="rest_trade",
      max_tokens=100,
      refill_rate=100 / 60,  # 100 запросов в минуту для торговли
    )

    # WebSocket connection лимиты
    self.add_bucket(
      name="websocket_connections",
      max_tokens=10,
      refill_rate=1 / 60,  # Медленное пополнение
    )

    # Order placement лимит
    self.add_bucket(
      name="order_placement",
      max_tokens=50,
      refill_rate=50 / 60,  # 50 ордеров в минуту
    )

    logger.info("✓ Bybit rate limits инициализированы")

  def add_bucket(
      self,
      name: str,
      max_tokens: int,
      refill_rate: float,
  ) -> TokenBucket:
    """
    Добавление нового bucket.

    Args:
        name: Имя bucket
        max_tokens: Максимум токенов
        refill_rate: Скорость пополнения

    Returns:
        TokenBucket: Созданный bucket
    """
    if name in self.buckets:
      logger.warning(f"TokenBucket '{name}' уже существует, переопределяем")

    bucket = TokenBucket(
      max_tokens=max_tokens,
      refill_rate=refill_rate,
      name=name,
    )

    self.buckets[name] = bucket
    return bucket

  def get_bucket(self, name: str) -> Optional[TokenBucket]:
    """
    Получение bucket по имени.

    Args:
        name: Имя bucket

    Returns:
        Optional[TokenBucket]: Bucket или None
    """
    return self.buckets.get(name)

  async def acquire(
      self,
      bucket_name: str,
      tokens: int = 1,
      max_wait: float = 10.0,
  ) -> bool:
    """
    Получение разрешения на выполнение операции.

    Args:
        bucket_name: Имя bucket
        tokens: Количество токенов
        max_wait: Максимальное время ожидания

    Returns:
        bool: True если разрешение получено
    """
    bucket = self.get_bucket(bucket_name)
    if not bucket:
      logger.warning(f"TokenBucket '{bucket_name}' не найден, создаем default")
      bucket = self.add_bucket(
        name=bucket_name,
        max_tokens=100,
        refill_rate=100 / 60,
      )

    success = await bucket.consume_async(tokens, max_wait)

    if not success:
      logger.warning(
        f"Rate limit достигнут для '{bucket_name}', "
        f"запрос отклонен после {max_wait}s ожидания"
      )

    return success

  def check_availability(self, bucket_name: str, tokens: int = 1) -> bool:
    """
    Проверка доступности токенов без потребления.

    Args:
        bucket_name: Имя bucket
        tokens: Количество токенов

    Returns:
        bool: True если токены доступны
    """
    bucket = self.get_bucket(bucket_name)
    if not bucket:
      return True

    return bucket.get_available_tokens() >= tokens

  def get_wait_time(self, bucket_name: str, tokens: int = 1) -> float:
    """
    Расчет времени ожидания.

    Args:
        bucket_name: Имя bucket
        tokens: Количество токенов

    Returns:
        float: Время ожидания в секундах
    """
    bucket = self.get_bucket(bucket_name)
    if not bucket:
      return 0.0

    return bucket.get_wait_time(tokens)

  def get_all_status(self) -> Dict[str, Dict]:
    """
    Получение статуса всех buckets.

    Returns:
        Dict: Статусы
    """
    return {
      name: bucket.get_status()
      for name, bucket in self.buckets.items()
    }

  def reset_bucket(self, name: str):
    """
    Сброс bucket (восстановление всех токенов).

    Args:
        name: Имя bucket
    """
    bucket = self.get_bucket(name)
    if bucket:
      bucket.tokens = float(bucket.max_tokens)
      bucket.last_refill = time.time()
      logger.info(f"TokenBucket '{name}' сброшен")


# Глобальный менеджер
rate_limiter = RateLimiterManager()


# Декоратор для автоматического rate limiting
def rate_limited(bucket_name: str, tokens: int = 1, max_wait: float = 10.0):
  """
  Декоратор для автоматического применения rate limiting.

  Args:
      bucket_name: Имя bucket
      tokens: Количество токенов
      max_wait: Максимальное время ожидания

  Example:
      @rate_limited("rest_trade", tokens=1)
      async def place_order(...):
          ...
  """

  def decorator(func):
    async def wrapper(*args, **kwargs):
      # Получаем разрешение
      allowed = await rate_limiter.acquire(
        bucket_name=bucket_name,
        tokens=tokens,
        max_wait=max_wait,
      )

      if not allowed:
        raise Exception(
          f"Rate limit exceeded for '{bucket_name}', "
          f"max wait time {max_wait}s reached"
        )

      # Выполняем функцию
      return await func(*args, **kwargs)

    return wrapper

  return decorator