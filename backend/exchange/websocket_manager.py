"""
WebSocket менеджер для Bybit.
Обеспечивает подключение к WebSocket API биржи и управление подписками.
"""

import asyncio
import json
from typing import Dict, List, Callable, Any
import websockets

from config import settings
from core.logger import get_logger
from core.exceptions import (
  WebSocketConnectionError,
  WebSocketDisconnectedError,
  WebSocketTimeoutError
)
from utils.constants import BybitEndpoints, BybitWSTopics, ConnectionStatus
from utils.helpers import ExponentialBackoff, get_timestamp_ms

logger = get_logger(__name__)


class BybitWebSocketManager:
  """Менеджер WebSocket подключений к Bybit."""

  def __init__(self, symbols: List[str], on_message: Callable):
    """
    Инициализация WebSocket менеджера.

    Args:
        symbols: Список торговых пар для отслеживания
        on_message: Callback функция для обработки сообщений
    """
    self.symbols = symbols
    self.on_message = on_message

    # WebSocket URL
    self.ws_url = (
      BybitEndpoints.TESTNET_WS_PUBLIC
      if settings.BYBIT_MODE == "testnet"
      else BybitEndpoints.MAINNET_WS_PUBLIC
    )

    # Состояние подключений
    self.websockets: Dict[int, websockets.WebSocketClientProtocol] = {}
    self.connection_status: Dict[int, ConnectionStatus] = {}
    self.subscribed_topics: Dict[int, List[str]] = {}

    # Управление переподключением
    self.reconnect_tasks: Dict[int, asyncio.Task] = {}
    self.backoff: Dict[int, ExponentialBackoff] = {}

    # Флаг остановки
    self.is_running = False

    # Разбиваем символы на группы (максимум 10 на соединение)
    self.symbol_groups = self._split_symbols_into_groups()

    logger.info(
      f"Инициализирован WebSocket менеджер для {len(self.symbols)} пар "
      f"({len(self.symbol_groups)} соединений)"
    )
    logger.info(f"WebSocket URL: {self.ws_url}")

  def _split_symbols_into_groups(self) -> List[List[str]]:
    """
    Разбиение символов на группы по максимальному количеству на соединение.

    Returns:
        List[List[str]]: Список групп символов
    """
    max_per_connection = settings.MAX_PAIRS_PER_CONNECTION
    groups = []

    for i in range(0, len(self.symbols), max_per_connection):
      group = self.symbols[i:i + max_per_connection]
      groups.append(group)

    logger.debug(f"Символы разбиты на {len(groups)} групп")
    return groups

  async def start(self):
    """Запуск всех WebSocket соединений."""
    if self.is_running:
      logger.warning("WebSocket менеджер уже запущен")
      return

    self.is_running = True
    logger.info("Запуск WebSocket менеджера")

    # Создаем задачи для каждой группы символов
    tasks = []
    for connection_id, symbol_group in enumerate(self.symbol_groups):
      task = asyncio.create_task(
        self._manage_connection(connection_id, symbol_group)
      )
      tasks.append(task)

    # Ждем завершения всех задач
    try:
      await asyncio.gather(*tasks)
    except Exception as e:
      logger.error(f"Ошибка в WebSocket менеджере: {e}")
      raise

  async def stop(self):
    """Остановка всех WebSocket соединений."""
    if not self.is_running:
      logger.warning("WebSocket менеджер уже остановлен")
      return

    logger.info("Остановка WebSocket менеджера")
    self.is_running = False

    # Отменяем задачи переподключения
    for task in self.reconnect_tasks.values():
      if not task.done():
        task.cancel()

    # Закрываем все WebSocket соединения
    for connection_id, ws in self.websockets.items():
      try:
        await ws.close()
        logger.info(f"WebSocket соединение {connection_id} закрыто")
      except Exception as e:
        logger.error(f"Ошибка закрытия соединения {connection_id}: {e}")

    self.websockets.clear()
    self.connection_status.clear()
    self.subscribed_topics.clear()

  async def _manage_connection(self, connection_id: int, symbols: List[str]):
    """
    Управление отдельным WebSocket соединением.

    Args:
        connection_id: ID соединения
        symbols: Список символов для этого соединения
    """
    logger.info(f"Управление соединением {connection_id} для {len(symbols)} пар")

    # Инициализируем backoff для этого соединения
    self.backoff[connection_id] = ExponentialBackoff(
      initial_delay=settings.WS_RECONNECT_TIMEOUT,
      max_delay=60.0
    )

    while self.is_running:
      try:
        await self._connect_and_subscribe(connection_id, symbols)
      except Exception as e:
        logger.error(f"Ошибка соединения {connection_id}: {e}")

        if self.is_running:
          delay = self.backoff[connection_id].increment()
          logger.info(
            f"Переподключение соединения {connection_id} через {delay:.1f}с"
          )
          await asyncio.sleep(delay)
        else:
          break

  async def _connect_and_subscribe(self, connection_id: int, symbols: List[str]):
    """
    Подключение к WebSocket и подписка на топики.

    Args:
        connection_id: ID соединения
        symbols: Список символов для подписки
    """
    self.connection_status[connection_id] = ConnectionStatus.CONNECTING
    logger.info(f"Подключение к WebSocket {connection_id}...")

    try:
      async with websockets.connect(
          self.ws_url,
          ping_interval=settings.WS_PING_INTERVAL,
          ping_timeout=30,
          close_timeout=10
      ) as websocket:
        self.websockets[connection_id] = websocket
        self.connection_status[connection_id] = ConnectionStatus.CONNECTED
        logger.info(f"WebSocket соединение {connection_id} установлено")

        # Сбрасываем backoff при успешном подключении
        self.backoff[connection_id].reset()

        # Подписываемся на топики
        await self._subscribe_to_topics(connection_id, symbols)

        # Обрабатываем входящие сообщения
        await self._handle_messages(connection_id, websocket)

    except websockets.exceptions.ConnectionClosed as e:
      logger.warning(f"WebSocket соединение {connection_id} закрыто: {e}")
      self.connection_status[connection_id] = ConnectionStatus.DISCONNECTED
      raise WebSocketDisconnectedError(f"Connection {connection_id} closed")

    except asyncio.TimeoutError:
      logger.error(f"Таймаут WebSocket соединения {connection_id}")
      self.connection_status[connection_id] = ConnectionStatus.ERROR
      raise WebSocketTimeoutError(f"Connection {connection_id} timeout")

    except Exception as e:
      logger.error(f"Ошибка WebSocket соединения {connection_id}: {e}")
      self.connection_status[connection_id] = ConnectionStatus.ERROR
      raise WebSocketConnectionError(str(e))

  async def _subscribe_to_topics(self, connection_id: int, symbols: List[str]):
    """
    Подписка на топики для символов.

    Args:
        connection_id: ID соединения
        symbols: Список символов
    """
    websocket = self.websockets[connection_id]
    topics = []

    # Формируем список топиков для подписки
    for symbol in symbols:
      topic = BybitWSTopics.get_orderbook_topic(
        symbol,
        settings.ORDERBOOK_DEPTH
      )
      topics.append(topic)

    # Отправляем запрос подписки
    subscribe_message = {
      "op": "subscribe",
      "args": topics
    }

    logger.info(f"Подписка на топики для соединения {connection_id}: {topics}")
    await websocket.send(json.dumps(subscribe_message))

    # Сохраняем подписанные топики
    self.subscribed_topics[connection_id] = topics

    # ИСПРАВЛЕНИЕ: Bybit V5 может прислать либо подтверждение, либо сразу snapshot
    # Ждем ответа и правильно его обрабатываем
    try:
      response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
      response_data = json.loads(response)

      # Проверяем тип ответа
      if response_data.get("success"):
        # Подтверждение подписки
        logger.info(f"Подписка соединения {connection_id} подтверждена")

      elif response_data.get("op") == "subscribe" and response_data.get("success"):
        # Альтернативный формат подтверждения
        logger.info(f"Подписка соединения {connection_id} подтверждена (alt format)")

      elif "topic" in response_data and "type" in response_data:
        # Bybit сразу прислал данные (snapshot или delta)!
        message_type = response_data.get("type")
        logger.info(
          f"Подписка соединения {connection_id} активна - "
          f"получено первое сообщение: {message_type}"
        )

        # КРИТИЧНО: Передаем первое сообщение в обработчик!
        await self._process_message(connection_id, response_data)

      else:
        # Неожиданный формат ответа - логируем для анализа
        logger.warning(
          f"Подписка соединения {connection_id}: "
          f"неожиданный формат ответа: {response_data.get('op', 'unknown')}"
        )

    except asyncio.TimeoutError:
      logger.warning(
        f"Таймаут ожидания подтверждения подписки {connection_id}. "
        f"Продолжаем работу - данные могут прийти позже."
      )
    except json.JSONDecodeError as e:
      logger.error(f"Ошибка парсинга ответа подписки {connection_id}: {e}")
    except Exception as e:
      logger.error(f"Неожиданная ошибка при подписке {connection_id}: {e}")

  async def _handle_messages(
      self,
      connection_id: int,
      websocket: websockets.WebSocketClientProtocol
  ):
    """
    Обработка входящих WebSocket сообщений.

    Args:
        connection_id: ID соединения
        websocket: WebSocket соединение
    """
    logger.info(f"Начало обработки сообщений для соединения {connection_id}")

    try:
      async for message in websocket:
        try:
          data = json.loads(message)

          # Обрабатываем ping-pong
          if data.get("op") == "ping":
            pong_message = {"op": "pong", "args": [get_timestamp_ms()]}
            await websocket.send(json.dumps(pong_message))
            logger.debug(f"Отправлен pong для соединения {connection_id}")
            continue

          # Пропускаем подтверждения подписки (они уже обработаны)
          if data.get("op") == "subscribe":
            logger.debug(f"Получено подтверждение подписки: {data.get('success')}")
            continue

          # Пропускаем pong ответы
          if data.get("op") == "pong":
            continue

          # Обрабатываем данные стакана
          if "topic" in data and "data" in data:
            await self._process_message(connection_id, data)
          else:
            # Логируем неизвестные сообщения для отладки
            logger.debug(f"Пропущено сообщение: op={data.get('op')}, keys={list(data.keys())}")

        except json.JSONDecodeError as e:
          logger.error(f"Ошибка парсинга JSON для соединения {connection_id}: {e}")
        except Exception as e:
          logger.error(f"Ошибка обработки сообщения {connection_id}: {e}")
          # НЕ прерываем цикл - продолжаем обработку других сообщений

    except websockets.exceptions.ConnectionClosed:
      logger.warning(f"Соединение {connection_id} закрыто во время обработки сообщений")
      raise

  async def _process_message(self, connection_id: int, data: Dict[str, Any]):
    """
    Обработка отдельного сообщения.

    Args:
        connection_id: ID соединения
        data: Данные сообщения
    """
    try:
      topic = data.get("topic", "")
      message_type = data.get("type", "")

      # Дополнительные поля для логирования
      timestamp = data.get("ts", "N/A")

      logger.debug(
        f"Conn {connection_id} | Topic: {topic} | "
        f"Type: {message_type} | TS: {timestamp}"
      )

      # Передаем сообщение в callback (в main.py)
      if self.on_message:
        await self.on_message(data)

    except Exception as e:
      logger.error(f"Ошибка обработки сообщения conn {connection_id}: {e}")

  def get_connection_statuses(self) -> Dict[int, str]:
    """
    Получение статусов всех соединений.

    Returns:
        Dict[int, str]: Словарь со статусами соединений
    """
    return {
      conn_id: status.value
      for conn_id, status in self.connection_status.items()
    }

  def is_all_connected(self) -> bool:
    """
    Проверка, что все соединения установлены.

    Returns:
        bool: True если все соединения активны
    """
    if not self.connection_status:
      return False

    return all(
      status == ConnectionStatus.CONNECTED
      for status in self.connection_status.values()
    )