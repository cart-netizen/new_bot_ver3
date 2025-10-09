"""
REST API клиент для Bybit.
Обеспечивает взаимодействие с REST API биржи для получения данных и размещения ордеров.
"""

import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime

from config import settings
from core.logger import get_logger
from core.exceptions import ExchangeAPIError, RateLimitError
from exchange.bybit_auth import authenticator
from utils.constants import BybitEndpoints, BybitAPIPaths, BybitCategory
from utils.helpers import get_timestamp_ms, retry_async

logger = get_logger(__name__)


class BybitRESTClient:
  """Клиент для работы с REST API Bybit."""

  def __init__(self):
    """Инициализация REST клиента."""
    self.base_url = (
      BybitEndpoints.TESTNET_REST
      if settings.BYBIT_MODE == "testnet"
      else BybitEndpoints.MAINNET_REST
    )
    self.session: Optional[aiohttp.ClientSession] = None
    self.authenticator = authenticator

    logger.info(f"Инициализирован REST клиент Bybit: {self.base_url}")

  async def __aenter__(self):
    """Асинхронный вход в контекст."""
    await self.initialize()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Асинхронный выход из контекста."""
    await self.close()

  async def initialize(self):
    """Инициализация HTTP сессии."""
    if self.session is None:
      timeout = aiohttp.ClientTimeout(total=30)
      self.session = aiohttp.ClientSession(timeout=timeout)
      logger.info("HTTP сессия инициализирована")

  async def close(self):
    """Закрытие HTTP сессии."""
    if self.session:
      await self.session.close()
      self.session = None
      logger.info("HTTP сессия закрыта")

  async def _request(
      self,
      method: str,
      path: str,
      params: Optional[Dict] = None,
      authenticated: bool = False
  ) -> Dict[str, Any]:
    """
    Выполнение HTTP запроса.

    Args:
        method: HTTP метод (GET, POST, etc.)
        path: Путь к эндпоинту
        params: Параметры запроса
        authenticated: Требуется ли аутентификация

    Returns:
        Dict: Ответ от API

    Raises:
        ExchangeAPIError: При ошибке API
        RateLimitError: При превышении лимита запросов
    """
    if self.session is None:
      await self.initialize()

    url = f"{self.base_url}{path}"
    headers = {"Content-Type": "application/json"}

    # Добавляем аутентификацию если требуется
    if authenticated:
      auth_data = self.authenticator.prepare_request(params or {})
      headers.update(auth_data["headers"])
      params = auth_data["params"]

    logger.debug(f"{method} запрос: {url}")
    if params:
      logger.debug(f"Параметры: {params}")

    try:
      async with self.session.request(
          method,
          url,
          params=params if method == "GET" else None,
          json=params if method == "POST" else None,
          headers=headers
      ) as response:
        data = await response.json()

        # Проверяем код ответа
        if response.status == 429:
          logger.warning("Превышен лимит запросов к API")
          raise RateLimitError("Превышен лимит запросов к Bybit API")

        # Проверяем статус в ответе Bybit
        if data.get("retCode") != 0:
          error_msg = data.get("retMsg", "Unknown error")
          logger.error(f"Ошибка API Bybit: {error_msg}")
          raise ExchangeAPIError(
            f"Bybit API error: {error_msg}",
            status_code=data.get("retCode"),
            response=data
          )

        logger.debug(f"Успешный ответ от {path}")
        return data

    except aiohttp.ClientError as e:
      logger.error(f"Ошибка HTTP запроса к {url}: {e}")
      raise ExchangeAPIError(f"HTTP request failed: {str(e)}")

  # ===== ПУБЛИЧНЫЕ МЕТОДЫ =====

  @retry_async(max_attempts=3, delay=1.0)
  async def get_server_time(self) -> int:
    """
    Получение серверного времени Bybit.

    Returns:
        int: Timestamp сервера в миллисекундах
    """
    logger.info("Запрос серверного времени")
    response = await self._request("GET", BybitAPIPaths.SERVER_TIME)
    server_time = int(response["result"]["timeSecond"]) * 1000
    logger.info(f"Серверное время: {datetime.fromtimestamp(server_time / 1000)}")
    return server_time

  @retry_async(max_attempts=3, delay=1.0)
  async def get_orderbook(self, symbol: str, limit: int = 50) -> Dict:
    """
    Получение стакана ордеров для символа.

    Args:
        symbol: Торговая пара
        limit: Количество уровней (1, 50, 200, 500)

    Returns:
        Dict: Данные стакана
    """
    logger.debug(f"Запрос стакана для {symbol}, лимит: {limit}")
    params = {
      "category": BybitCategory.SPOT.value,
      "symbol": symbol,
      "limit": limit
    }
    response = await self._request("GET", BybitAPIPaths.ORDERBOOK, params)
    return response["result"]

  @retry_async(max_attempts=3, delay=1.0)
  async def get_tickers(self, symbol: Optional[str] = None) -> List[Dict]:
    """
    Получение тикеров.

    Args:
        symbol: Торговая пара (опционально, для всех если не указано)

    Returns:
        List[Dict]: Список тикеров
    """
    logger.debug(f"Запрос тикеров{f' для {symbol}' if symbol else ''}")
    params = {"category": BybitCategory.SPOT.value}
    if symbol:
      params["symbol"] = symbol
    response = await self._request("GET", BybitAPIPaths.TICKERS, params)
    return response["result"]["list"]

  @retry_async(max_attempts=3, delay=1.0)
  async def get_instruments_info(self, symbol: Optional[str] = None) -> List[Dict]:
    """
    Получение информации о торговых парах.

    Args:
        symbol: Торговая пара (опционально)

    Returns:
        List[Dict]: Информация о парах
    """
    logger.debug(f"Запрос информации о парах{f' для {symbol}' if symbol else ''}")
    params = {"category": BybitCategory.SPOT.value}
    if symbol:
      params["symbol"] = symbol
    response = await self._request("GET", BybitAPIPaths.INSTRUMENTS_INFO, params)
    return response["result"]["list"]

  # ===== ПРИВАТНЫЕ МЕТОДЫ (ЗАГЛУШКИ) =====

  async def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict:
    """
    Получение баланса кошелька (ЗАГЛУШКА).

    Args:
        account_type: Тип аккаунта

    Returns:
        Dict: Данные баланса
    """
    logger.info("Запрос баланса кошелька")

    # ЗАГЛУШКА: возвращаем тестовые данные
    logger.warning("ЗАГЛУШКА: Возвращаем тестовый баланс")
    return {
      "result": {
        "list": [{
          "coin": [{
            "coin": "USDT",
            "walletBalance": "10000.0",
            "availableToWithdraw": "10000.0"
          }]
        }]
      }
    }

  async def place_order(
      self,
      symbol: str,
      side: str,
      order_type: str,
      quantity: float,
      price: Optional[float] = None,
      time_in_force: str = "GTC"
  ) -> Dict:
    """
    Размещение ордера (ЗАГЛУШКА).

    Args:
        symbol: Торговая пара
        side: Сторона (Buy/Sell)
        order_type: Тип ордера (Market/Limit)
        quantity: Количество
        price: Цена (для лимитных ордеров)
        time_in_force: Time in force

    Returns:
        Dict: Результат размещения ордера
    """
    logger.info(
      f"Размещение ордера: {symbol} {side} {order_type} "
      f"qty={quantity} price={price}"
    )

    # ЗАГЛУШКА: только логируем, не размещаем реальный ордер
    logger.warning("ЗАГЛУШКА: Ордер не размещается на бирже")
    return {
      "result": {
        "orderId": f"mock_order_{get_timestamp_ms()}",
        "orderLinkId": "",
        "symbol": symbol,
        "side": side,
        "orderType": order_type,
        "price": str(price) if price else "",
        "qty": str(quantity),
        "timeInForce": time_in_force,
        "orderStatus": "New",
        "createdTime": str(get_timestamp_ms())
      }
    }

  async def cancel_order(self, symbol: str, order_id: str) -> Dict:
    """
    Отмена ордера (ЗАГЛУШКА).

    Args:
        symbol: Торговая пара
        order_id: ID ордера

    Returns:
        Dict: Результат отмены
    """
    logger.info(f"Отмена ордера: {symbol} order_id={order_id}")

    # ЗАГЛУШКА
    logger.warning("ЗАГЛУШКА: Ордер не отменяется")
    return {
      "result": {
        "orderId": order_id,
        "orderLinkId": "",
        "symbol": symbol,
        "status": "Cancelled"
      }
    }

  async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
    """
    Получение открытых ордеров (ЗАГЛУШКА).

    Args:
        symbol: Торговая пара (опционально)

    Returns:
        List[Dict]: Список открытых ордеров
    """
    logger.info(f"Запрос открытых ордеров{f' для {symbol}' if symbol else ''}")

    # ЗАГЛУШКА
    logger.warning("ЗАГЛУШКА: Возвращаем пустой список ордеров")
    return {"result": {"list": []}}

  async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
    """
    Получение позиций (ЗАГЛУШКА).

    Args:
        symbol: Торговая пара (опционально)

    Returns:
        List[Dict]: Список позиций
    """
    logger.info(f"Запрос позиций{f' для {symbol}' if symbol else ''}")

    # ЗАГЛУШКА
    logger.warning("ЗАГЛУШКА: Возвращаем пустой список позиций")
    return {"result": {"list": []}}


# Глобальный экземпляр клиента
rest_client = BybitRESTClient()