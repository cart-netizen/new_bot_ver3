"""
REST API клиент для Bybit.
Обеспечивает взаимодействие с REST API биржи для получения данных и размещения ордеров.
"""

import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime

from backend.config import settings
from backend.core.logger import get_logger
from backend.core.exceptions import ExchangeAPIError, RateLimitError
from backend.exchange.bybit_auth import authenticator
from backend.utils.constants import BybitEndpoints, BybitAPIPaths, BybitCategory
from backend.utils.helpers import get_timestamp_ms, retry_async

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

  def _validate_api_credentials(self) -> None:
    """
    Проверка наличия API ключей для текущего режима работы.

    Raises:
        ValueError: Если API ключи не настроены
    """
    try:
      api_key, api_secret = settings.get_bybit_credentials()
      if not api_key or not api_secret:
        raise ValueError("Пустые API ключи")
    except ValueError as e:
      error_msg = f"API ключи Bybit не настроены для режима {settings.BYBIT_MODE}: {e}"
      logger.error(error_msg)
      raise ValueError(error_msg)

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

      # Настройки для стабильности на Windows
      # - ttl_dns_cache: Кэшируем DNS на 300 секунд чтобы уменьшить DNS запросы
      # - limit: Ограничиваем количество соединений
      # - force_close: Закрываем соединения после использования (избегаем проблем keep-alive на Windows)
      connector = aiohttp.TCPConnector(
          ttl_dns_cache=300,
          limit=100,
          force_close=True,
          enable_cleanup_closed=True
      )

      self.session = aiohttp.ClientSession(
          timeout=timeout,
          connector=connector
      )
      logger.info("HTTP сессия инициализирована (с DNS кэшированием)")

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

    # Подготовка аутентифицированного запроса
    if authenticated:
      auth_data = self.authenticator.prepare_request(
        method=method,
        params=params or {}
      )
      headers.update(auth_data["headers"])
      params = auth_data["params"]

    logger.debug(f"{method} запрос: {url}")
    if params:
      logger.debug(f"Параметры: {params}")

    # КРИТИЧНОЕ ИСПРАВЛЕНИЕ: Сериализуем параметры вручную с сортировкой
    # чтобы порядок совпадал с подписью как для GET, так и для POST
    request_body = None
    query_params = None

    if method == "GET":
      # Для GET формируем отсортированный query string вручную
      if params:
        from urllib.parse import urlencode
        # ВАЖНО: Сортируем параметры так же, как в bybit_auth.py
        query_params = urlencode(sorted(params.items()))
        logger.debug(f"Query string (sorted): {query_params}")
    else:
      # Для POST/PUT/DELETE сериализуем JSON вручную с сортировкой ключей
      if params:
        import json
        # ВАЖНО: Используем те же параметры сериализации, что и в bybit_auth.py
        request_body = json.dumps(params, separators=(',', ':'), sort_keys=True)
        logger.debug(f"Request body (sorted): {request_body[:200]}...")

    # Формируем финальный URL для GET запросов
    if method == "GET" and query_params:
      # Добавляем query string к URL вручную
      final_url = f"{url}?{query_params}"
    else:
      final_url = url

    try:
      async with self.session.request(
          method,
          url,
          params=query_params,
          data=request_body,  # ← ИЗМЕНЕНО: Используем data вместо json
          headers=headers
      ) as response:
        # Сохраняем статус до чтения body (для обработки ошибок)
        status = response.status

        # Пытаемся получить JSON
        try:
          data = await response.json()
        except:
          # Если не JSON - получаем текст
          text = await response.text()
          logger.error(f"Не удалось распарсить JSON. Ответ: {text[:500]}")
          raise ExchangeAPIError(
            f"Invalid JSON response: {text[:200]}",
            status_code=status
          )

        # MEMORY FIX: Явно освобождаем response для предотвращения утечки CIMultiDict
        # CIMultiDict/CIMultiDictProxy накапливались с 35K до 296K за 11 часов
        await response.release()

        # Логируем статус
        logger.debug(f"HTTP Status: {status}")

        # Проверяем код ответа HTTP
        if status == 401:
          logger.error("401 Unauthorized - проблема с аутентификацией")
          raise ExchangeAPIError(
            "Unauthorized: Invalid API key or signature",
            status_code=401,
            response=data
          )

        if status == 429:
          logger.warning("Превышен лимит запросов к API")
          raise RateLimitError("Превышен лимит запросов к Bybit API")

        # Проверяем статус в ответе Bybit
        ret_code = data.get("retCode")
        if ret_code != 0:
          error_msg = data.get("retMsg", "Unknown error")
          logger.error(f"Ошибка API Bybit: код={ret_code}, сообщение={error_msg}")

          # ДОБАВЛЕНО: Детальное логирование для ошибок подписи
          if ret_code == 10004:
            logger.error("=" * 80)
            logger.error("ОШИБКА ПОДПИСИ API (retCode=10004)")
            logger.error("=" * 80)
            logger.error(f"URL: {url}")
            logger.error(f"Method: {method}")
            logger.error(f"Headers: {headers}")
            logger.error(f"Request body: {request_body}")
            logger.error(f"Response: {data}")
            logger.error("=" * 80)

          raise ExchangeAPIError(
            f"Bybit API error: {error_msg}",
            status_code=ret_code,
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
    self._validate_api_credentials()

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
      take_profit: Optional[float] = None,
      client_order_id: Optional[str] = None,
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
    if client_order_id:
      logger.info(f"Client Order ID: {client_order_id}")

    logger.info(
      f"Размещение ордера: {symbol} {side} {order_type} "
      f"qty={quantity} price={price} SL={stop_loss} TP={take_profit}"
    )

    # Проверяем наличие API ключей
    self._validate_api_credentials()

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

    if client_order_id:
      params["orderLinkId"] = client_order_id
      logger.debug(f"Установлен orderLinkId: {client_order_id}")

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
    self._validate_api_credentials()

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
    self._validate_api_credentials()

    # Реальный запрос к API
    # ИСПРАВЛЕНО: Добавлен settleCoin - обязательный параметр если symbol не указан
    params = {
      "category": BybitCategory.LINEAR.value,
      "settleCoin": "USDT"  # ← ОБЯЗАТЕЛЬНЫЙ параметр для LINEAR без symbol
    }
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

  @retry_async(max_attempts=3, delay=1.0)
  async def get_order_info(
      self,
      symbol: str,
      order_id: Optional[str] = None,
      order_link_id: Optional[str] = None
  ) -> Optional[Dict]:
    """
    Получение информации о конкретном ордере.

    Args:
        symbol: Торговая пара
        order_id: ID ордера от биржи (опционально)
        order_link_id: Client Order ID (опционально)

    Returns:
        Optional[Dict]: Информация об ордере или None если не найден

    Raises:
        ValueError: Если API ключи не настроены или не указан ни один ID
    """
    # Проверяем наличие API ключей
    self._validate_api_credentials()

    # Должен быть указан хотя бы один ID
    if not order_id and not order_link_id:
      raise ValueError("Необходимо указать order_id или order_link_id")

    logger.debug(
      f"Запрос информации об ордере: symbol={symbol}, "
      f"order_id={order_id}, order_link_id={order_link_id}"
    )

    # Параметры запроса
    params = {
      "category": BybitCategory.LINEAR.value,
      "symbol": symbol
    }

    if order_id:
      params["orderId"] = order_id
    elif order_link_id:
      params["orderLinkId"] = order_link_id

    try:
      # Пробуем найти в активных ордерах
      response = await self._request(
        "GET",
        BybitAPIPaths.GET_OPEN_ORDERS,
        params,
        authenticated=True
      )

      orders_list = response.get("result", {}).get("list", [])

      if orders_list:
        # Найден активный ордер
        logger.debug(f"Ордер найден в активных: {orders_list[0].get('orderId')}")
        return orders_list[0]

      # Если не найден в активных, ищем в истории
      logger.debug("Ордер не найден в активных, ищем в истории...")

      response = await self._request(
        "GET",
        BybitAPIPaths.GET_ORDER_HISTORY,
        params,
        authenticated=True
      )

      orders_list = response.get("result", {}).get("list", [])

      if orders_list:
        # Найден в истории
        logger.debug(f"Ордер найден в истории: {orders_list[0].get('orderId')}")
        return orders_list[0]

      # Ордер не найден нигде
      logger.warning(
        f"Ордер не найден: symbol={symbol}, "
        f"order_id={order_id}, order_link_id={order_link_id}"
      )
      return None

    except Exception as e:
      logger.error(f"Ошибка получения информации об ордере: {e}")
      # Возвращаем None вместо выброса исключения
      # так как ордер может быть просто не найден
      return None

  @retry_async(max_attempts=3, delay=1.0)
  async def get_order_history(
      self,
      symbol: Optional[str] = None,
      limit: int = 50
  ) -> Dict:
    """
    Получение истории ордеров.

    Args:
        symbol: Торговая пара (опционально)
        limit: Количество ордеров (макс 50)

    Returns:
        Dict: История ордеров

    Raises:
        ValueError: Если API ключи не настроены
    """
    logger.info(f"Запрос истории ордеров{f' для {symbol}' if symbol else ''}")

    # Проверяем наличие API ключей
    self._validate_api_credentials()

    # Параметры запроса
    params = {
      "category": BybitCategory.LINEAR.value,
      "limit": limit
    }

    if symbol:
      params["symbol"] = symbol

    response = await self._request(
      "GET",
      BybitAPIPaths.GET_ORDER_HISTORY,
      params,
      authenticated=True
    )

    order_count = len(response.get("result", {}).get("list", []))
    logger.info(f"Получено {order_count} ордеров из истории")
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
    self._validate_api_credentials()

    # ИСПРАВЛЕНО: Добавлен параметр settleCoin для категории LINEAR
    params = {
      "category": BybitCategory.LINEAR.value,
      "settleCoin": "USDT"  # ← ОБЯЗАТЕЛЬНЫЙ параметр для LINEAR
    }

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
    self._validate_api_credentials()

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
    self._validate_api_credentials()

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

  @retry_async(max_attempts=3, delay=1.0)
  async def set_trading_stop(
      self,
      symbol: str,
      stop_loss: Optional[float] = None,
      take_profit: Optional[float] = None,
      position_idx: int = 0  # 0 для One-Way Mode
  ) -> Dict:
    """
    Установка/обновление Stop Loss и Take Profit для открытой позиции.

    Args:
        symbol: Торговая пара
        stop_loss: Новый уровень Stop Loss (опционально)
        take_profit: Новый уровень Take Profit (опционально)
        position_idx: Индекс позиции (0 для One-Way Mode)

    Returns:
        Dict: Результат обновления

    Raises:
        ValueError: Если API ключи не настроены или оба параметра None
    """
    if stop_loss is None and take_profit is None:
      raise ValueError("Необходимо указать хотя бы один параметр: stop_loss или take_profit")

    logger.info(
      f"Обновление Trading Stop для {symbol}: "
      f"SL={stop_loss if stop_loss else 'не изменяется'}, "
      f"TP={take_profit if take_profit else 'не изменяется'}"
    )

    # Проверка наличия API ключей
    self._validate_api_credentials()

    # Параметры запроса
    params = {
      "category": BybitCategory.LINEAR.value,
      "symbol": symbol,
      "positionIdx": position_idx
    }

    if stop_loss is not None:
      params["stopLoss"] = str(stop_loss)

    if take_profit is not None:
      params["takeProfit"] = str(take_profit)

    response = await self._request(
      "POST",
      BybitAPIPaths.SET_TRADING_STOP,
      params,
      authenticated=True
    )

    logger.info(
      f"Trading Stop для {symbol} обновлен успешно: "
      f"SL={stop_loss}, TP={take_profit}"
    )
    return response

# Глобальный экземпляр клиента
rest_client = BybitRESTClient()