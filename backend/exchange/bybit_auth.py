"""
Модуль аутентификации для Bybit API.
Обеспечивает HMAC SHA256 подписание запросов.
"""
import json
import time
import hmac
import hashlib
from typing import Dict, Optional
from urllib.parse import urlencode

from config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class BybitAuthenticator:
  """Класс для аутентификации запросов к Bybit V5 API."""

  def __init__(self, api_key: str, api_secret: str):
    """
    Инициализация аутентификатора.

    Args:
        api_key: API ключ Bybit
        api_secret: API секрет Bybit
    """
    # Очищаем от возможных пробелов
    self.api_key = api_key.strip()
    self.api_secret = api_secret.strip()

    logger.info(f"Инициализирован аутентификатор Bybit (ключ: {self.api_key[:8]}...)")
    logger.debug(f"Длина API ключа: {len(self.api_key)}")
    logger.debug(f"Длина API secret: {len(self.api_secret)}")

  def generate_signature(
      self,
      timestamp: int,
      method: str,
      params: Dict,
      recv_window: int = 5000
  ) -> str:
    """
    Генерация HMAC SHA256 подписи для запроса согласно документации Bybit V5.

    Формат подписи:
    - GET:  timestamp + api_key + recv_window + queryString
    - POST: timestamp + api_key + recv_window + jsonBody

    Args:
        timestamp: Unix timestamp в миллисекундах
        method: HTTP метод (GET или POST)
        params: Параметры запроса
        recv_window: Окно приема запроса в миллисекундах

    Returns:
        str: Hex строка подписи
    """
    # Формируем параметры в зависимости от метода
    if method == "GET":
      # Для GET: сортированный query string
      if params:
        # Сортируем по ключам и формируем query string
        param_str = urlencode(sorted(params.items()))
      else:
        param_str = ""
    else:  # POST, PUT, DELETE
      # Для POST: JSON строка БЕЗ пробелов
      if params:
        # КРИТИЧНО: Используем те же параметры сериализации, что и в _request
        param_str = json.dumps(params, separators=(',', ':'), sort_keys=True)
      else:
        param_str = ""

    # Собираем строку для подписи согласно документации V5
    sign_str = f"{timestamp}{self.api_key}{recv_window}{param_str}"

    # УЛУЧШЕННОЕ ЛОГИРОВАНИЕ
    logger.debug("=" * 80)
    logger.debug(f"Создание подписи ({method}):")
    logger.debug(f"  Timestamp: {timestamp}")
    logger.debug(f"  API Key: {self.api_key}")
    logger.debug(f"  Recv Window: {recv_window}")

    # Для POST показываем отсортированные ключи
    if method == "POST" and params:
      sorted_keys = sorted(params.keys())
      logger.debug(f"  Sorted keys: {sorted_keys}")
      logger.debug(f"  Param String (first 200 chars): {param_str[:200]}")
      logger.debug(f"  Param String (full): {param_str}")
    else:
      logger.debug(f"  Param String: {param_str[:100]}...")

    logger.debug(f"  Sign String (first 150 chars): {sign_str[:150]}...")
    logger.debug("=" * 80)

    # Генерируем HMAC SHA256
    signature = hmac.new(
      self.api_secret.encode('utf-8'),
      sign_str.encode('utf-8'),
      hashlib.sha256
    ).hexdigest()

    logger.debug(f"  Generated Signature: {signature}")
    logger.debug("=" * 80)

    return signature

  def get_headers(
      self,
      timestamp: int,
      signature: str,
      recv_window: int = 5000
  ) -> Dict[str, str]:
    """
    Получение заголовков для аутентифицированного запроса.

    Args:
        timestamp: Unix timestamp в миллисекундах
        signature: Подпись запроса
        recv_window: Окно приема запроса в миллисекундах

    Returns:
        Dict[str, str]: Заголовки запроса
    """
    headers = {
      "X-BAPI-API-KEY": self.api_key,
      "X-BAPI-SIGN": signature,
      "X-BAPI-TIMESTAMP": str(timestamp),
      "X-BAPI-RECV-WINDOW": str(recv_window),
      "Content-Type": "application/json"
    }

    # НЕ добавляем X-BAPI-SIGN-TYPE для V5 API (это для V3)

    return headers

  def prepare_request(
      self,
      method: str,
      params: Optional[Dict] = None,
      recv_window: int = 5000
  ) -> Dict:
    """
    Подготовка аутентифицированного запроса.

    Args:
        method: HTTP метод (GET, POST, etc.)
        params: Параметры запроса
        recv_window: Окно приема запроса в миллисекундах

    Returns:
        Dict: Словарь с заголовками и параметрами
    """
    if params is None:
      params = {}

    # Генерируем timestamp
    timestamp = int(time.time() * 1000)

    # Генерируем подпись с учетом метода
    signature = self.generate_signature(timestamp, method, params, recv_window)

    # Формируем заголовки
    headers = self.get_headers(timestamp, signature, recv_window)

    return {
      "headers": headers,
      "params": params,
      "timestamp": timestamp
    }

  @staticmethod
  def get_server_time_offset() -> int:
    """
    Получение разницы между локальным и серверным временем.
    Используется для синхронизации при проблемах с временем.

    Returns:
        int: Offset в миллисекундах
    """
    # Эта функция будет реализована в REST клиенте
    # Здесь возвращаем 0 как заглушку
    return 0

  def adjust_timestamp(self, timestamp: int) -> int:
    """
    Корректировка timestamp с учетом offset.

    Args:
        timestamp: Локальный timestamp

    Returns:
        int: Скорректированный timestamp
    """
    offset = self.get_server_time_offset()
    return timestamp + offset


def create_authenticator() -> BybitAuthenticator:
  """
  Создание аутентификатора с учетом текущего режима работы.

  Returns:
      BybitAuthenticator: Инстанс аутентификатора
  """
  api_key, api_secret = settings.get_bybit_credentials()

  logger.info(f"Создан аутентификатор для режима: {settings.BYBIT_MODE}")
  logger.info(f"API ключ: {api_key[:8]}... (длина: {len(api_key)})")
  logger.info(f"API secret: {'*' * len(api_secret)} (длина: {len(api_secret)})")

  return BybitAuthenticator(api_key, api_secret)


# Глобальный экземпляр аутентификатора
authenticator = create_authenticator()