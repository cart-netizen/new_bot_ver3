"""
Модуль аутентификации для Bybit API.
Обеспечивает HMAC SHA256 подписание запросов.
"""

import time
import hmac
import hashlib
from typing import Dict, Optional
from urllib.parse import urlencode

from backend.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class BybitAuthenticator:
  """Класс для аутентификации запросов к Bybit API."""

  def __init__(self, api_key: str, api_secret: str):
    """
    Инициализация аутентификатора.

    Args:
        api_key: API ключ Bybit
        api_secret: API секрет Bybit
    """
    self.api_key = api_key
    self.api_secret = api_secret
    logger.info(f"Инициализирован аутентификатор Bybit (ключ: {api_key[:8]}...)")

  def generate_signature(
      self,
      timestamp: int,
      params: Dict,
      recv_window: int = 5000
  ) -> str:
    """
    Генерация HMAC SHA256 подписи для запроса.

    Args:
        timestamp: Unix timestamp в миллисекундах
        params: Параметры запроса
        recv_window: Окно приема запроса в миллисекундах

    Returns:
        str: Hex строка подписи
    """
    # Формируем строку для подписи
    param_str = urlencode(sorted(params.items()))
    sign_str = f"{timestamp}{self.api_key}{recv_window}{param_str}"

    # Генерируем HMAC SHA256
    signature = hmac.new(
      self.api_secret.encode('utf-8'),
      sign_str.encode('utf-8'),
      hashlib.sha256
    ).hexdigest()

    logger.debug(f"Сгенерирована подпись для запроса: {signature[:16]}...")
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
    return {
      "X-BAPI-API-KEY": self.api_key,
      "X-BAPI-SIGN": signature,
      "X-BAPI-SIGN-TYPE": "2",
      "X-BAPI-TIMESTAMP": str(timestamp),
      "X-BAPI-RECV-WINDOW": str(recv_window),
      "Content-Type": "application/json"
    }

  def prepare_request(
      self,
      params: Optional[Dict] = None,
      recv_window: int = 5000
  ) -> Dict:
    """
    Подготовка аутентифицированного запроса.

    Args:
        params: Параметры запроса
        recv_window: Окно приема запроса в миллисекундах

    Returns:
        Dict: Словарь с заголовками и параметрами
    """
    if params is None:
      params = {}

    # Генерируем timestamp
    timestamp = int(time.time() * 1000)

    # Генерируем подпись
    signature = self.generate_signature(timestamp, params, recv_window)

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
  logger.info(f"API ключ: {api_key[:8]}...")

  return BybitAuthenticator(api_key, api_secret)


# Глобальный экземпляр аутентификатора
authenticator = create_authenticator()