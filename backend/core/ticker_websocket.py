# backend/core/ticker_websocket.py
"""
Bybit Ticker WebSocket Manager - получение тикеров для скринера.

Функционал:
- Подключение к Bybit WebSocket v5 (tickers channel)
- Получение данных всех USDT пар
- Автоматическое переподключение
- Передача данных в ScreenerProcessor

Оптимизации:
- Использование одного соединения для всех пар
- Эффективная обработка ping/pong
- Graceful shutdown
"""

import asyncio
import json
import websockets
from typing import Optional, Callable, Awaitable
from datetime import datetime

from core.logger import get_logger
from config import settings

logger = get_logger(__name__)


class BybitTickerWebSocket:
  """
  WebSocket клиент для получения тикеров от Bybit v5.

  Подключается к публичному WebSocket endpoint и подписывается
  на канал tickers.* для получения данных всех торговых пар.
  """

  def __init__(
      self,
      on_ticker_callback: Optional[Callable[[dict], Awaitable[None]]] = None
  ):
    """
    Инициализация WebSocket клиента.

    Args:
        on_ticker_callback: Async callback для обработки тикеров
    """
    self.ws_url = settings.BYBIT_WS_URL  # Из конфига
    self.on_ticker_callback = on_ticker_callback

    # Состояние
    self.ws: Optional[websockets.WebSocketClientProtocol] = None
    self.is_running = False
    self.reconnect_delay = 5  # секунд
    self.ping_interval = 20  # секунд

    # Статистика
    self.messages_received = 0
    self.tickers_processed = 0
    self.errors_count = 0
    self.last_message_ts = 0

    logger.info("Инициализирован BybitTickerWebSocket")

  async def start(self):
    """Запуск WebSocket соединения с автоматическим переподключением."""
    self.is_running = True

    logger.info("=" * 80)
    logger.info("ЗАПУСК BYBIT TICKER WEBSOCKET")
    logger.info("=" * 80)

    while self.is_running:
      try:
        await self._connect_and_listen()
      except Exception as e:
        self.errors_count += 1
        logger.error(f"Ошибка в WebSocket соединении: {e}")

        if self.is_running:
          logger.info(f"Переподключение через {self.reconnect_delay} секунд...")
          await asyncio.sleep(self.reconnect_delay)

  async def stop(self):
    """Остановка WebSocket соединения."""
    logger.info("Остановка BybitTickerWebSocket...")
    self.is_running = False

    if self.ws:
      await self.ws.close()
      self.ws = None

    logger.info("BybitTickerWebSocket остановлен")

  async def _connect_and_listen(self):
    """Подключение и прослушивание сообщений."""
    logger.info(f"Подключение к Bybit WebSocket: {self.ws_url}")

    async with websockets.connect(self.ws_url, ping_interval=None) as ws:
      self.ws = ws
      logger.info("✓ Подключение к Bybit WebSocket установлено")

      # Подписываемся на канал tickers
      await self._subscribe_tickers()

      # Запускаем задачу ping
      ping_task = asyncio.create_task(self._ping_loop())

      try:
        # Основной цикл получения сообщений
        async for message in ws:
          await self._handle_message(message)
      except websockets.exceptions.ConnectionClosed as e:
        logger.warning(f"WebSocket соединение закрыто: {e}")
      finally:
        ping_task.cancel()
        try:
          await ping_task
        except asyncio.CancelledError:
          pass

  async def _subscribe_tickers(self):
    """
    Подписка на канал tickers для всех USDT пар.

    Bybit v5 позволяет подписаться на tickers.* для получения
    данных всех торговых пар.
    """
    subscribe_message = {
      "op": "subscribe",
      "args": ["tickers.BTCUSDT"]  # Подписываемся на все USDT пары через паттерн
    }

    # Отправляем подписку
    await self.ws.send(json.dumps(subscribe_message))
    logger.info("Отправлен запрос подписки на tickers.*")

    # Ждем подтверждения
    try:
      response = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
      response_data = json.loads(response)

      if response_data.get("op") == "subscribe":
        if response_data.get("success"):
          logger.info("✅ Подписка на tickers подтверждена")
        else:
          logger.error(f"❌ Ошибка подписки: {response_data}")
      else:
        logger.debug(f"Получен ответ: {response_data.get('op', 'unknown')}")

    except asyncio.TimeoutError:
      logger.warning("Таймаут ожидания подтверждения подписки")
    except Exception as e:
      logger.error(f"Ошибка при подписке: {e}")

  async def _ping_loop(self):
    """Периодическая отправка ping для поддержания соединения."""
    try:
      while self.is_running and self.ws:
        await asyncio.sleep(self.ping_interval)

        if self.ws and not self.ws.closed:
          ping_message = {"op": "ping"}
          await self.ws.send(json.dumps(ping_message))
          logger.debug("Отправлен ping")

    except asyncio.CancelledError:
      pass
    except Exception as e:
      logger.error(f"Ошибка в ping loop: {e}")

  async def _handle_message(self, message: str):
    """
    Обработка входящего сообщения от Bybit.

    Args:
        message: JSON строка с данными
    """
    try:
      self.messages_received += 1
      self.last_message_ts = int(datetime.now().timestamp())

      data = json.loads(message)

      # Обработка ping/pong
      if data.get("op") == "ping":
        pong_message = {"op": "pong", "req_id": data.get("req_id")}
        await self.ws.send(json.dumps(pong_message))
        logger.debug("Ответили на ping → pong")
        return

      # Обработка данных тикера
      if "topic" in data and "tickers" in data["topic"]:
        ticker_data = data.get("data", {})

        if ticker_data.get("symbol") and ticker_data.get("lastPrice"):
          self.tickers_processed += 1

          # Логируем первые 5 тикеров для отладки
          if self.tickers_processed <= 5:
            logger.debug(f"Получен тикер: {ticker_data.get('symbol')} @ "
                         f"{ticker_data.get('lastPrice')}")

          # Каждые 100 тикеров логируем статистику
          if self.tickers_processed % 100 == 0:
            logger.info(f"Обработано {self.tickers_processed} тикеров")

          # Вызываем callback
          if self.on_ticker_callback:
            await self.on_ticker_callback(ticker_data)

      # Игнорируем подтверждения подписки и прочее
      elif data.get("op") in ["subscribe", "pong"]:
        pass
      else:
        # Неизвестный тип сообщения
        if self.messages_received <= 10:
          logger.debug(f"Неизвестное сообщение: {data.get('op', 'unknown')}")

    except json.JSONDecodeError as e:
      self.errors_count += 1
      logger.error(f"Ошибка парсинга JSON: {e}")
    except Exception as e:
      self.errors_count += 1
      logger.error(f"Ошибка обработки сообщения: {e}")

  def get_statistics(self) -> dict:
    """
    Получение статистики работы WebSocket.

    Returns:
        Словарь со статистикой
    """
    return {
      "is_running": self.is_running,
      "is_connected": self.ws is not None and not self.ws.close,
      "messages_received": self.messages_received,
      "tickers_processed": self.tickers_processed,
      "errors_count": self.errors_count,
      "last_message_ts": self.last_message_ts,
      "uptime_seconds": int(datetime.now().timestamp()) - self.last_message_ts
      if self.last_message_ts else 0,
    }


class ScreenerTickerManager:
  """
  Менеджер тикеров для скринера.

  Координирует работу WebSocket и ScreenerProcessor,
  обеспечивая передачу данных и управление жизненным циклом.
  """

  def __init__(self, screener_processor):
    """
    Инициализация менеджера.

    Args:
        screener_processor: Экземпляр ScreenerProcessor
    """
    self.screener_processor = screener_processor
    self.ticker_ws: Optional[BybitTickerWebSocket] = None
    self.is_running = False

    logger.info("Инициализирован ScreenerTickerManager")

  async def _on_ticker_update(self, ticker_data: dict):
    """
    Callback для обработки тикеров от WebSocket.

    Args:
        ticker_data: Данные тикера от Bybit
    """
    try:
      # Передаем данные в процессор
      self.screener_processor.update_from_ticker(ticker_data)
    except Exception as e:
      logger.error(f"Ошибка обработки тикера в процессоре: {e}")

  async def start(self):
    """Запуск менеджера тикеров."""
    if self.is_running:
      logger.warning("ScreenerTickerManager уже запущен")
      return

    self.is_running = True
    logger.info("Запуск ScreenerTickerManager...")

    # Создаем и запускаем WebSocket
    self.ticker_ws = BybitTickerWebSocket(
      on_ticker_callback=self._on_ticker_update
    )

    # Запускаем в фоновой задаче
    asyncio.create_task(self.ticker_ws.start())

    logger.info("✓ ScreenerTickerManager запущен")

  async def stop(self):
    """Остановка менеджера тикеров."""
    if not self.is_running:
      return

    logger.info("Остановка ScreenerTickerManager...")
    self.is_running = False

    if self.ticker_ws:
      await self.ticker_ws.stop()
      self.ticker_ws = None

    logger.info("✓ ScreenerTickerManager остановлен")

  def get_statistics(self) -> dict:
    """
    Получение статистики менеджера.

    Returns:
        Словарь со статистикой
    """
    stats = {
      "is_running": self.is_running,
      "processor_stats": self.screener_processor.get_statistics(),
    }

    if self.ticker_ws:
      stats["websocket_stats"] = self.ticker_ws.get_statistics()

    return stats