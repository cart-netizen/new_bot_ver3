"""
WebSocket менеджер для Bybit с улучшенным логированием и обработкой ошибок.
ИСПРАВЛЕННАЯ ВЕРСИЯ - без ошибок RuntimeError и AttributeError
Заменяет backend/exchange/websocket_manager.py
"""

import asyncio
import json
from typing import Dict, List, Callable, Any
import websockets
import traceback

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
  """Менеджер WebSocket подключений к Bybit с улучшенной диагностикой."""

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

    logger.info("=" * 80)
    logger.info(
      f"Инициализирован WebSocket менеджер для {len(self.symbols)} пар "
      f"({len(self.symbol_groups)} соединений)"
    )
    logger.info(f"WebSocket URL: {self.ws_url}")
    logger.info(f"Группы символов: {self.symbol_groups}")
    logger.info("=" * 80)

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

    logger.debug(f"Символы разбиты на {len(groups)} групп по {max_per_connection} пар")
    return groups

  async def start(self):
    """Запуск всех WebSocket соединений."""
    if self.is_running:
      logger.warning("WebSocket менеджер уже запущен")
      return

    self.is_running = True
    logger.info("=" * 80)
    logger.info("ЗАПУСК WEBSOCKET МЕНЕДЖЕРА")
    logger.info("=" * 80)

    # Создаем задачи для каждой группы символов
    tasks = []
    for connection_id, symbol_group in enumerate(self.symbol_groups):
      logger.info(
        f"Создание задачи для соединения {connection_id}: "
        f"{len(symbol_group)} пар {symbol_group}"
      )
      task = asyncio.create_task(
        self._manage_connection(connection_id, symbol_group)
      )
      tasks.append(task)

    logger.info(f"Создано {len(tasks)} задач для управления соединениями")

    # Ждем завершения всех задач
    try:
      await asyncio.gather(*tasks)
    except Exception as e:
      logger.error(f"Критическая ошибка в WebSocket менеджере: {e}")
      logger.error(traceback.format_exc())
      raise

  async def stop(self):
    """Остановка всех WebSocket соединений."""
    if not self.is_running:
      logger.warning("WebSocket менеджер уже остановлен")
      return

    logger.info("=" * 80)
    logger.info("ОСТАНОВКА WEBSOCKET МЕНЕДЖЕРА")
    logger.info("=" * 80)
    self.is_running = False

    # Отменяем задачи переподключения
    for connection_id, task in list(self.reconnect_tasks.items()):
      if not task.done():
        logger.info(f"Отмена задачи переподключения {connection_id}")
        task.cancel()

    # ИСПРАВЛЕНИЕ: Создаем копию словаря для итерации
    websockets_copy = dict(self.websockets)

    # Закрываем все WebSocket соединения
    for connection_id, ws in websockets_copy.items():
      try:
        logger.info(f"Закрытие WebSocket соединения {connection_id}...")
        await ws.close()
        logger.info(f"✓ WebSocket соединение {connection_id} закрыто")
      except Exception as e:
        logger.error(f"Ошибка закрытия соединения {connection_id}: {e}")

    self.websockets.clear()
    self.connection_status.clear()
    self.subscribed_topics.clear()
    logger.info("Все соединения закрыты")

  def is_all_connected(self) -> bool:
    """
    Проверка, что все соединения установлены.

    Returns:
        bool: True если все соединения активны
    """
    expected_connections = len(self.symbol_groups)
    active_connections = sum(
      1 for status in self.connection_status.values()
      if status == ConnectionStatus.CONNECTED
    )

    result = active_connections == expected_connections

    if not result:
      logger.debug(
        f"Не все соединения активны: {active_connections}/{expected_connections}"
      )

    return result

  def get_connection_count(self) -> int:
    """
    Получение количества активных соединений.

    Returns:
        int: Количество активных соединений
    """
    return sum(
      1 for status in self.connection_status.values()
      if status == ConnectionStatus.CONNECTED
    )

  async def _manage_connection(self, connection_id: int, symbols: List[str]):
    """
    Управление отдельным WebSocket соединением.

    Args:
        connection_id: ID соединения
        symbols: Список символов для этого соединения
    """
    logger.info("=" * 80)
    logger.info(f"УПРАВЛЕНИЕ СОЕДИНЕНИЕМ {connection_id}")
    logger.info(f"Символы: {symbols}")
    logger.info("=" * 80)

    # Инициализируем backoff для этого соединения
    self.backoff[connection_id] = ExponentialBackoff(
      initial_delay=settings.WS_RECONNECT_TIMEOUT,
      max_delay=60.0
    )

    attempt_number = 0

    while self.is_running:
      attempt_number += 1
      logger.info(
        f"Попытка подключения #{attempt_number} для соединения {connection_id}"
      )

      try:
        await self._connect_and_subscribe(connection_id, symbols)

        # Если мы здесь, значит соединение было закрыто корректно
        logger.info(f"Соединение {connection_id} завершено корректно")

      except WebSocketDisconnectedError as e:
        logger.warning(f"Соединение {connection_id} разорвано: {e}")

      except WebSocketTimeoutError as e:
        logger.error(f"Таймаут соединения {connection_id}: {e}")

      except WebSocketConnectionError as e:
        logger.error(f"Ошибка подключения {connection_id}: {e}")

      except Exception as e:
        logger.error(
          f"Неожиданная ошибка соединения {connection_id}: "
          f"{type(e).__name__}: {e}"
        )
        logger.error(traceback.format_exc())

      # Переподключение
      if self.is_running:
        delay = self.backoff[connection_id].increment()
        logger.info(
          f"Переподключение соединения {connection_id} через {delay:.1f}с "
          f"(попытка #{attempt_number + 1})"
        )
        await asyncio.sleep(delay)
      else:
        logger.info(f"Остановка управления соединением {connection_id}")
        break

  async def _connect_and_subscribe(self, connection_id: int, symbols: List[str]):
    """
    Подключение к WebSocket и подписка на топики.

    Args:
        connection_id: ID соединения
        symbols: Список символов для подписки
    """
    self.connection_status[connection_id] = ConnectionStatus.CONNECTING
    logger.info(f"[{connection_id}] Подключение к WebSocket: {self.ws_url}")
    logger.info(f"[{connection_id}] Параметры: ping_interval={settings.WS_PING_INTERVAL}s")

    try:
      # КРИТИЧЕСКОЕ УЛУЧШЕНИЕ: Добавляем подробное логирование
      logger.debug(f"[{connection_id}] Вызов websockets.connect()...")

      async with websockets.connect(
          self.ws_url,
          ping_interval=settings.WS_PING_INTERVAL,
          ping_timeout=30,
          close_timeout=10
      ) as websocket:
        # ВАЖНО: Если мы здесь, значит соединение установлено
        self.websockets[connection_id] = websocket
        self.connection_status[connection_id] = ConnectionStatus.CONNECTED

        logger.info("=" * 80)
        logger.info(f"[{connection_id}] ✅ WEBSOCKET СОЕДИНЕНИЕ УСТАНОВЛЕНО")
        logger.info(f"[{connection_id}] Состояние: {websocket.state}")
        logger.info(f"[{connection_id}] Локальный адрес: {websocket.local_address}")
        logger.info(f"[{connection_id}] Удаленный адрес: {websocket.remote_address}")
        logger.info("=" * 80)

        # Сбрасываем backoff при успешном подключении
        self.backoff[connection_id].reset()

        # Подписываемся на топики
        await self._subscribe_to_topics(connection_id, symbols)

        # Обрабатываем входящие сообщения
        logger.info(f"[{connection_id}] Переход к обработке сообщений...")
        await self._handle_messages(connection_id, websocket)

    except websockets.exceptions.ConnectionClosed as e:
      logger.warning(
        f"[{connection_id}] WebSocket соединение закрыто: "
        f"code={e.code}, reason={e.reason}"
      )
      self.connection_status[connection_id] = ConnectionStatus.DISCONNECTED
      raise WebSocketDisconnectedError(
        f"Connection {connection_id} closed: {e.code} - {e.reason}"
      )

    except asyncio.TimeoutError:
      logger.error(f"[{connection_id}] Таймаут WebSocket соединения")
      self.connection_status[connection_id] = ConnectionStatus.ERROR
      raise WebSocketTimeoutError(f"Connection {connection_id} timeout")

    except websockets.exceptions.InvalidURI as e:
      logger.error(f"[{connection_id}] Неверный URI: {self.ws_url}")
      logger.error(f"[{connection_id}] Ошибка: {e}")
      self.connection_status[connection_id] = ConnectionStatus.ERROR
      raise WebSocketConnectionError(f"Invalid URI: {e}")

    except Exception as e:
      logger.error(
        f"[{connection_id}] Критическая ошибка WebSocket: "
        f"{type(e).__name__}: {e}"
      )
      logger.error(traceback.format_exc())
      self.connection_status[connection_id] = ConnectionStatus.ERROR
      raise WebSocketConnectionError(f"Unexpected error: {e}")

    finally:
      # Очищаем состояние при выходе
      if connection_id in self.websockets:
        del self.websockets[connection_id]
      logger.info(f"[{connection_id}] Выход из _connect_and_subscribe")

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

    logger.info(f"[{connection_id}] Подписка на {len(topics)} топиков:")
    for i, topic in enumerate(topics, 1):
      logger.info(f"[{connection_id}]   {i}. {topic}")

    logger.debug(
      f"[{connection_id}] Отправка: {json.dumps(subscribe_message, indent=2)}"
    )

    await websocket.send(json.dumps(subscribe_message))
    logger.info(f"[{connection_id}] ✓ Запрос подписки отправлен")

    # Сохраняем подписанные топики
    self.subscribed_topics[connection_id] = topics

    # Ждем первого ответа (может быть snapshot или подтверждение)
    try:
      logger.info(f"[{connection_id}] Ожидание ответа от сервера (timeout=5s)...")
      response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
      response_data = json.loads(response)

      logger.info(f"[{connection_id}] Получен ответ от сервера:")
      logger.debug(f"[{connection_id}] {json.dumps(response_data, indent=2)[:500]}...")

      # Проверяем тип ответа
      if response_data.get("op") == "subscribe":
        success = response_data.get("success")
        if success:
          logger.info(f"[{connection_id}] ✅ Подписка подтверждена сервером")
        else:
          logger.warning(
            f"[{connection_id}] ⚠️  Подписка НЕ подтверждена: {response_data}"
          )

      elif "topic" in response_data and "data" in response_data:
        logger.info(
          f"[{connection_id}] 📊 Получен snapshot стакана: "
          f"topic={response_data.get('topic')}, "
          f"type={response_data.get('type')}"
        )
        # Обрабатываем snapshot через callback
        if self.on_message:
          await self.on_message(response_data)

      else:
        logger.warning(
          f"[{connection_id}] Получен неожиданный ответ: "
          f"keys={list(response_data.keys())}"
        )

    except asyncio.TimeoutError:
      logger.warning(
        f"[{connection_id}] Таймаут ожидания ответа на подписку. "
        f"Продолжаем работу - данные могут прийти позже."
      )

    except json.JSONDecodeError as e:
      logger.error(f"[{connection_id}] Ошибка парсинга ответа подписки: {e}")

    except Exception as e:
      logger.error(
        f"[{connection_id}] Неожиданная ошибка при подписке: "
        f"{type(e).__name__}: {e}"
      )
      logger.error(traceback.format_exc())

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
    logger.info("=" * 80)
    logger.info(f"[{connection_id}] НАЧАЛО ОБРАБОТКИ СООБЩЕНИЙ")
    logger.info("=" * 80)

    message_count = 0
    ping_count = 0
    data_count = 0

    try:
      async for message in websocket:
        message_count += 1

        try:
          data = json.loads(message)

          # Обрабатываем ping-pong
          if data.get("op") == "ping":
            ping_count += 1
            pong_message = {"op": "pong", "args": [get_timestamp_ms()]}
            await websocket.send(json.dumps(pong_message))
            logger.debug(
              f"[{connection_id}] 🏓 Ping #{ping_count} получен и обработан"
            )
            continue

          # Пропускаем подтверждения подписки (они уже обработаны)
          if data.get("op") == "subscribe":
            logger.debug(
              f"[{connection_id}] Подтверждение подписки: "
              f"success={data.get('success')}"
            )
            continue

          # Пропускаем pong ответы
          if data.get("op") == "pong":
            logger.debug(f"[{connection_id}] Pong получен")
            continue

          # Обрабатываем данные стакана
          if "topic" in data and "data" in data:
            data_count += 1

            if data_count <= 5:
              # Логируем первые 5 сообщений подробно
              logger.info(
                f"[{connection_id}] 📊 Данные стакана #{data_count}: "
                f"topic={data.get('topic')}, type={data.get('type')}, "
                f"ts={data.get('ts')}"
              )
            elif data_count % 10000 == 0:
              # Каждое 100-е сообщение
              logger.info(
                f"[{connection_id}] 📊 Обработано {data_count} сообщений стакана"
              )

            await self._process_message(connection_id, data)

          else:
            # Логируем неизвестные сообщения для отладки
            logger.debug(
              f"[{connection_id}] Пропущено сообщение: "
              f"op={data.get('op')}, keys={list(data.keys())}"
            )

        except json.JSONDecodeError as e:
          logger.error(
            f"[{connection_id}] Ошибка парсинга JSON (сообщение #{message_count}): {e}"
          )
          logger.debug(f"[{connection_id}] Сырое сообщение: {message[:500]}...")

        except Exception as e:
          logger.error(
            f"[{connection_id}] Ошибка обработки сообщения #{message_count}: "
            f"{type(e).__name__}: {e}"
          )
          logger.error(traceback.format_exc())
          # НЕ прерываем цикл - продолжаем обработку других сообщений

    except websockets.exceptions.ConnectionClosed as e:
      logger.warning(
        f"[{connection_id}] Соединение закрыто во время обработки сообщений: "
        f"code={e.code}, reason={e.reason}"
      )
      logger.info(
        f"[{connection_id}] Статистика перед закрытием: "
        f"всего={message_count}, ping={ping_count}, данные={data_count}"
      )
      raise

    except Exception as e:
      logger.error(
        f"[{connection_id}] Критическая ошибка в цикле обработки сообщений: "
        f"{type(e).__name__}: {e}"
      )
      logger.error(traceback.format_exc())
      raise

    finally:
      logger.info("=" * 80)
      logger.info(f"[{connection_id}] ЗАВЕРШЕНИЕ ОБРАБОТКИ СООБЩЕНИЙ")
      logger.info(f"[{connection_id}] Обработано сообщений: {message_count}")
      logger.info(f"[{connection_id}] Ping/pong обменов: {ping_count}")
      logger.info(f"[{connection_id}] Полученных данных: {data_count}")
      logger.info("=" * 80)

  async def _process_message(self, connection_id: int, data: Dict[str, Any]):
    """
    Обработка отдельного сообщения.

    Args:
        connection_id: ID соединения
        data: Данные сообщения
    """
    try:
      # Передаем сообщение в callback (в main.py)
      if self.on_message:
        await self.on_message(data)

    except Exception as e:
      logger.error(
        f"[{connection_id}] Ошибка в callback обработки сообщения: "
        f"{type(e).__name__}: {e}"
      )
      logger.error(traceback.format_exc())

  def get_connection_statuses(self) -> Dict[int, str]:
    """
    Получение статусов всех соединений.

    Returns:
        Dict[int, str]: Статусы соединений
    """
    return {
      conn_id: status.value
      for conn_id, status in self.connection_status.items()
    }

  def get_stats(self) -> Dict[str, Any]:
    """
    Получение статистики WebSocket менеджера.

    Returns:
        Dict: Статистика
    """
    return {
      "is_running": self.is_running,
      "total_symbols": len(self.symbols),
      "connection_groups": len(self.symbol_groups),
      "active_connections": len(self.websockets),
      "connection_statuses": self.get_connection_statuses(),
      "subscribed_topics": {
        conn_id: len(topics)
        for conn_id, topics in self.subscribed_topics.items()
      }
    }