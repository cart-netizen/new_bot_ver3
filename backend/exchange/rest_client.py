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

  # ===== ПРИВАТНЫЕ МЕТОДЫ (ЗАГЛУШКИ) =====

  async def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict:
    """
    Получение баланса кошелька.

    Args:
        account_type: Тип аккаунта (UNIFIED для UTA)

    Returns:
        Dict: Данные баланса

    Raises:
        ValueError: Если API ключи не настроены
    """
    logger.info("Запрос баланса кошелька")

    # Проверяем наличие API ключей
    if not settings.BYBIT_API_KEY or not settings.BYBIT_API_SECRET:
      error_msg = (
        "API ключи Bybit не настроены. "
        "Установите BYBIT_API_KEY и BYBIT_API_SECRET в файле .env"
      )
      logger.error(error_msg)
      raise ValueError(error_msg)

    # Реальный запрос к API
    params = {"accountType": account_type}
    response = await self._request(
      "GET",
      BybitAPIPaths.GET_WALLET_BALANCE,
      params,
      authenticated=True
    )

    logger.info(f"Баланс получен успешно для аккаунта {account_type}")
    return response

  async def place_order(
      self,
      symbol: str,
      side: str,
      order_type: str,
      quantity: float,
      price: Optional[float] = None,
      time_in_force: str = "GTC",
      stop_loss: Optional[float] = None,
      take_profit: Optional[float] = None
  ) -> Dict:
    """
    Размещение ордера.

    Args:
        symbol: Торговая пара
        side: Сторона (Buy/Sell)
        order_type: Тип ордера (Market/Limit)
        quantity: Количество
        price: Цена (для лимитных ордеров)
        time_in_force: Time in force
        stop_loss: Уровень стоп-лосса (опционально)
        take_profit: Уровень тейк-профита (опционально)

    Returns:
        Dict: Результат размещения ордера

    Raises:
        ValueError: Если API ключи не настроены
    """
    logger.info(
      f"Размещение ордера: {symbol} {side} {order_type} "
      f"qty={quantity} price={price} SL={stop_loss} TP={take_profit}"
    )

    # Проверяем наличие API ключей
    if not settings.BYBIT_API_KEY or not settings.BYBIT_API_SECRET:
      error_msg = (
        "API ключи Bybit не настроены. "
        "Установите BYBIT_API_KEY и BYBIT_API_SECRET в файле .env"
      )
      logger.error(error_msg)
      raise ValueError(error_msg)

    # Реальный запрос к API для фьючерсов
    params = {
      "category": BybitCategory.LINEAR.value,  # LINEAR для фьючерсов
      "symbol": symbol,
      "side": side,
      "orderType": order_type,
      "qty": str(quantity),
      "timeInForce": time_in_force
    }

    if price:
      params["price"] = str(price)

    if stop_loss:
      params["stopLoss"] = str(stop_loss)

    if take_profit:
      params["takeProfit"] = str(take_profit)

    response = await self._request(
      "POST",
      BybitAPIPaths.PLACE_ORDER,
      params,
      authenticated=True
    )

    logger.info(f"Ордер размещен успешно: order_id={response['result']['orderId']}")
    return response

  async def cancel_order(self, symbol: str, order_id: str) -> Dict:
    """
    Отмена ордера.

    Args:
        symbol: Торговая пара
        order_id: ID ордера

    Returns:
        Dict: Результат отмены

    Raises:
        ValueError: Если API ключи не настроены
    """
    logger.info(f"Отмена ордера: {symbol} order_id={order_id}")

    # Проверяем наличие API ключей
    if not settings.BYBIT_API_KEY or not settings.BYBIT_API_SECRET:
      error_msg = (
        "API ключи Bybit не настроены. "
        "Установите BYBIT_API_KEY и BYBIT_API_SECRET в файле .env"
      )
      logger.error(error_msg)
      raise ValueError(error_msg)

    # Реальный запрос к API
    params = {
      "category": BybitCategory.LINEAR.value,
      "symbol": symbol,
      "orderId": order_id
    }

    response = await self._request(
      "POST",
      BybitAPIPaths.CANCEL_ORDER,
      params,
      authenticated=True
    )

    logger.info(f"Ордер отменен успешно: order_id={order_id}")
    return response

  async def get_open_orders(self, symbol: Optional[str] = None) -> Dict:
    """
    Получение открытых ордеров.

    Args:
        symbol: Торговая пара (опционально)

    Returns:
        Dict: Список открытых ордеров

    Raises:
        ValueError: Если API ключи не настроены
    """
    logger.info(f"Запрос открытых ордеров{f' для {symbol}' if symbol else ''}")

    # Проверяем наличие API ключей
    if not settings.BYBIT_API_KEY or not settings.BYBIT_API_SECRET:
      error_msg = (
        "API ключи Bybit не настроены. "
        "Установите BYBIT_API_KEY и BYBIT_API_SECRET в файле .env"
      )
      logger.error(error_msg)
      raise ValueError(error_msg)

    # Реальный запрос к API
    params = {"category": BybitCategory.LINEAR.value}
    if symbol:
      params["symbol"] = symbol

    response = await self._request(
      "GET",
      BybitAPIPaths.GET_OPEN_ORDERS,
      params,
      authenticated=True
    )

    order_count = len(response.get("result", {}).get("list", []))
    logger.info(f"Получено {order_count} открытых ордеров")
    return response

  async def get_positions(self, symbol: Optional[str] = None) -> Dict:
    """
    Получение позиций.

    Args:
        symbol: Торговая пара (опционально)

    Returns:
        Dict: Список позиций

    Raises:
        ValueError: Если API ключи не настроены
    """
    logger.info(f"Запрос позиций{f' для {symbol}' if symbol else ''}")

    # Проверяем наличие API ключей
    if not settings.BYBIT_API_KEY or not settings.BYBIT_API_SECRET:
      error_msg = (
        "API ключи Bybit не настроены. "
        "Установите BYBIT_API_KEY и BYBIT_API_SECRET в файле .env"
      )
      logger.error(error_msg)
      raise ValueError(error_msg)

    # Реальный запрос к API
    params = {"category": BybitCategory.LINEAR.value}
    if symbol:
      params["symbol"] = symbol

    response = await self._request(
      "GET",
      BybitAPIPaths.GET_POSITIONS,
      params,
      authenticated=True
    )

    position_count = len(response.get("result", {}).get("list", []))
    logger.info(f"Получено {position_count} позиций")
    return response

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
      "category": BybitCategory.LINEAR.value,  # LINEAR для фьючерсов
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
    params = {"category": BybitCategory.LINEAR.value}  # LINEAR для фьючерсов
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
    params = {"category": BybitCategory.LINEAR.value}  # LINEAR для фьючерсов
    if symbol:
      params["symbol"] = symbol
    response = await self._request("GET", BybitAPIPaths.INSTRUMENTS_INFO, params)
    return response["result"]["list"]

  @retry_async(max_attempts=3, delay=1.0)
  async def get_api_key_info(self) -> Dict:
    """
    Получение информации об API ключе.

    Returns:
        Dict: Информация об API ключе

    Raises:
        ValueError: Если API ключи не настроены
    """
    logger.info("Запрос информации об API ключе")

    # Проверяем наличие API ключей
    if not settings.BYBIT_API_KEY or not settings.BYBIT_API_SECRET:
      error_msg = (
        "API ключи Bybit не настроены. "
        "Установите BYBIT_API_KEY и BYBIT_API_SECRET в файле .env"
      )
      logger.error(error_msg)
      raise ValueError(error_msg)

    response = await self._request(
      "GET",
      BybitAPIPaths.GET_API_KEY_INFO,
      authenticated=True
    )

    logger.info("Информация об API ключе получена успешно")
    return response

  @retry_async(max_attempts=3, delay=1.0)
  async def set_leverage(self, symbol: str, buy_leverage: str, sell_leverage: str) -> Dict:
    """
    Установка кредитного плеча для символа.

    Args:
        symbol: Торговая пара
        buy_leverage: Плечо для покупок (1-100)
        sell_leverage: Плечо для продаж (1-100)

    Returns:
        Dict: Результат установки плеча

    Raises:
        ValueError: Если API ключи не настроены
    """
    logger.info(f"Установка плеча для {symbol}: Buy={buy_leverage}x, Sell={sell_leverage}x")

    # Проверяем наличие API ключей
    if not settings.BYBIT_API_KEY or not settings.BYBIT_API_SECRET:
      error_msg = (
        "API ключи Bybit не настроены. "
        "Установите BYBIT_API_KEY и BYBIT_API_SECRET в файле .env"
      )
      logger.error(error_msg)
      raise ValueError(error_msg)

    params = {
      "category": BybitCategory.LINEAR.value,
      "symbol": symbol,
      "buyLeverage": buy_leverage,
      "sellLeverage": sell_leverage
    }

    response = await self._request(
      "POST",
      BybitAPIPaths.SET_LEVERAGE,
      params,
      authenticated=True
    )

    logger.info(f"Плечо успешно установлено для {symbol}")
    return response

  @retry_async(max_attempts=3, delay=1.0)
  async def get_recent_trades(self, symbol: str, limit: int = 60) -> List[Dict]:
    """
    Получение последних сделок.

    Args:
        symbol: Торговая пара
        limit: Количество сделок (макс 1000)

    Returns:
        List[Dict]: Список последних сделок
    """
    logger.debug(f"Запрос последних сделок для {symbol}, лимит: {limit}")
    params = {
      "category": BybitCategory.LINEAR.value,  # LINEAR для фьючерсов
      "symbol": symbol,
      "limit": limit
    }

    response = await self._request("GET", BybitAPIPaths.RECENT_TRADES, params)

    trade_count = len(response["result"]["list"])
    logger.debug(f"Получено {trade_count} последних сделок для {symbol}")
    return response["result"]["list"]

  @retry_async(max_attempts=3, delay=1.0)
  async def get_kline(
      self,
      symbol: str,
      interval: str,
      limit: int = 200,
      start: Optional[int] = None,
      end: Optional[int] = None
  ) -> List[Dict]:
    """
    Получение свечных данных (kline).

    Args:
        symbol: Торговая пара
        interval: Интервал свечи (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
        limit: Количество свечей (макс 1000)
        start: Timestamp начала (опционально)
        end: Timestamp окончания (опционально)

    Returns:
        List[Dict]: Список свечей
    """
    logger.debug(f"Запрос kline для {symbol}, интервал: {interval}, лимит: {limit}")
    params = {
      "category": BybitCategory.LINEAR.value,  # LINEAR для фьючерсов
      "symbol": symbol,
      "interval": interval,
      "limit": limit
    }

    if start:
      params["start"] = start
    if end:
      params["end"] = end

    response = await self._request("GET", BybitAPIPaths.KLINE, params)

    kline_count = len(response["result"]["list"])
    logger.debug(f"Получено {kline_count} свечей для {symbol}")
    return response["result"]["list"]

# Глобальный экземпляр клиента
rest_client = BybitRESTClient()