"""
Idempotency Service.
Предотвращение дублирования операций и генерация уникальных ID.
"""

import uuid
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from core.logger import get_logger
from database.connection import db_manager
from database.models import IdempotencyCache
from sqlalchemy import select, delete

logger = get_logger(__name__)


class IdempotencyService:
  """
  Сервис идемпотентности операций.
  Гарантирует, что повторные запросы не приведут к дублированию.
  """

  def __init__(self, default_ttl_minutes: int = 60):
    """
    Инициализация сервиса.

    Args:
        default_ttl_minutes: TTL кэша по умолчанию в минутах
    """
    self.default_ttl_minutes = default_ttl_minutes
    logger.info(f"Idempotency Service инициализирован (TTL: {default_ttl_minutes}m)")

  def generate_client_order_id(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> str:
      """
      Генерация уникального Client Order ID (максимум 45 символов для Bybit).

      Формат: SYMBOL_Side_YYMMDDHHMMSS_UUID
      Пример: BTCUSDT_Buy_251018112220_a1b2c3d4
      Длина: ~35-40 символов (в зависимости от длины символа)

      КРИТИЧНО: Bybit ограничивает orderLinkId до 45 символов!
      """
      # ИСПРАВЛЕНИЕ: Укороченный timestamp (12 символов вместо 20)
      timestamp = datetime.utcnow().strftime("%y%m%d%H%M%S")  # YYMMDDHHmmss = 12 символов

      # UUID остается 8 символов
      unique_part = uuid.uuid4().hex[:8]

      # Укороченный Side (B/S вместо Buy/Sell)
      side_short = side[0]  # "B" или "S"

      # Формат: SYMBOL_S_YYMMDDHHMMSS_UUID
      # Пример: BTCUSDT_B_251018112220_a1b2c3d4
      # Длина для BTCUSDT: 7 + 1 + 1 + 1 + 12 + 1 + 8 = 31 символ ✅
      # Длина для 1000RATSUSDT: 13 + 1 + 1 + 1 + 12 + 1 + 8 = 37 символов ✅
      client_order_id = f"{symbol}_{side_short}_{timestamp}_{unique_part}"

      logger.debug(
        f"Сгенерирован Client Order ID: {client_order_id} "
        f"(длина: {len(client_order_id)} символов)"
      )

      # ПРОВЕРКА: Если длина все еще > 45, обрезаем symbol
      if len(client_order_id) > 45:
        # Обрезаем symbol до нужной длины
        max_symbol_len = 45 - (1 + 1 + 1 + 12 + 1 + 8)  # = 21 символ для symbol
        symbol_short = symbol[:max_symbol_len]
        client_order_id = f"{symbol_short}_{side_short}_{timestamp}_{unique_part}"
        logger.warning(
          f"Client Order ID обрезан до 45 символов: {client_order_id}"
        )

      return client_order_id

  def generate_idempotency_key(
      self,
      operation: str,
      params: Dict[str, Any],
  ) -> str:
    """
    Генерация ключа идемпотентности на основе операции и параметров.

    Args:
        operation: Тип операции (например, "place_order")
        params: Параметры операции

    Returns:
        str: Хэш-ключ идемпотентности
    """
    # Сортируем параметры для детерминированного хэша
    sorted_params = sorted(params.items())
    params_str = str(sorted_params)

    # Создаем хэш
    hash_input = f"{operation}:{params_str}"
    idempotency_key = hashlib.sha256(hash_input.encode()).hexdigest()

    logger.debug(f"Сгенерирован ключ идемпотентности: {idempotency_key[:16]}...")
    return idempotency_key

  async def check_idempotency(
      self,
      operation: str,
      params: Dict[str, Any],
  ) -> Optional[Dict[str, Any]]:
    """
    Проверка идемпотентности операции.

    Args:
        operation: Тип операции
        params: Параметры операции

    Returns:
        Optional[Dict]: Результат предыдущей операции или None
    """
    idempotency_key = self.generate_idempotency_key(operation, params)

    try:
      async with db_manager.session() as session:
        # Ищем существующую запись
        stmt = select(IdempotencyCache).where(
          IdempotencyCache.idempotency_key == idempotency_key,
          IdempotencyCache.expires_at > datetime.utcnow()
        )
        result = await session.execute(stmt)
        cache_entry = result.scalar_one_or_none()

        if cache_entry:
          logger.info(
            f"Найдена идемпотентная операция: {operation}, "
            f"ключ: {idempotency_key[:16]}..."
          )
          return {
            "success": cache_entry.success,
            "result": cache_entry.result,
            "error": cache_entry.error,
          }

        return None

    except Exception as e:
      logger.error(f"Ошибка проверки идемпотентности: {e}")
      # В случае ошибки БД, возвращаем None (операция продолжится)
      return None

  async def save_operation_result(
      self,
      operation: str,
      params: Dict[str, Any],
      result: Optional[Dict[str, Any]] = None,
      success: bool = True,
      error: Optional[str] = None,
      ttl_minutes: Optional[int] = None,
  ) -> bool:
    """
    Сохранение результата операции для идемпотентности.

    Args:
        operation: Тип операции
        params: Параметры операции
        result: Результат операции
        success: Успешность операции
        error: Сообщение об ошибке
        ttl_minutes: TTL в минутах

    Returns:
        bool: True если сохранено успешно
    """
    idempotency_key = self.generate_idempotency_key(operation, params)
    ttl = ttl_minutes or self.default_ttl_minutes

    try:
      async with db_manager.session() as session:
        # Создаем запись
        cache_entry = IdempotencyCache(
          idempotency_key=idempotency_key,
          operation=operation,
          result=result,
          success=success,
          error=error,
          created_at=datetime.utcnow(),
          expires_at=datetime.utcnow() + timedelta(minutes=ttl),
        )

        session.add(cache_entry)
        await session.commit()

        logger.debug(
          f"Результат операции сохранен: {operation}, "
          f"success={success}, TTL={ttl}m"
        )
        return True

    except Exception as e:
      logger.error(f"Ошибка сохранения результата операции: {e}")
      return False

  async def cleanup_expired(self) -> int:
    """
    Очистка истекших записей идемпотентности.

    Returns:
        int: Количество удаленных записей
    """
    try:
      async with db_manager.session() as session:
        stmt = delete(IdempotencyCache).where(
          IdempotencyCache.expires_at < datetime.utcnow()
        )
        result = await session.execute(stmt)
        await session.commit()

        deleted_count = result.rowcount
        if deleted_count > 0:
          logger.info(f"Очищено {deleted_count} истекших записей идемпотентности")

        return deleted_count

    except Exception as e:
      logger.error(f"Ошибка очистки идемпотентности: {e}")
      return 0

  async def idempotent_operation(
      self,
      operation: str,
      params: Dict[str, Any],
      executor,
      ttl_minutes: Optional[int] = None,
  ) -> Dict[str, Any]:
    """
    Выполнение идемпотентной операции.

    Args:
        operation: Тип операции
        params: Параметры операции
        executor: Функция-исполнитель (async callable)
        ttl_minutes: TTL результата

    Returns:
        Dict: Результат операции
    """
    # Проверяем идемпотентность
    cached_result = await self.check_idempotency(operation, params)
    if cached_result is not None:
      logger.info(f"Возврат кэшированного результата для {operation}")
      return cached_result

    # Выполняем операцию
    try:
      logger.debug(f"Выполнение операции: {operation}")
      result = await executor()

      # Сохраняем результат
      await self.save_operation_result(
        operation=operation,
        params=params,
        result=result,
        success=True,
        ttl_minutes=ttl_minutes,
      )

      return {
        "success": True,
        "result": result,
        "error": None,
      }

    except Exception as e:
      error_msg = str(e)
      logger.error(f"Ошибка выполнения операции {operation}: {error_msg}")

      # Сохраняем ошибку
      await self.save_operation_result(
        operation=operation,
        params=params,
        result=None,
        success=False,
        error=error_msg,
        ttl_minutes=ttl_minutes,
      )

      return {
        "success": False,
        "result": None,
        "error": error_msg,
      }


# Глобальный экземпляр
idempotency_service = IdempotencyService()