"""
WebSocket эндпоинт для фронтенда.
Обеспечивает реалтайм обновления данных для пользовательского интерфейса.
"""

import asyncio
import json
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect, Depends
from datetime import datetime

from core.logger import get_logger
from core.auth import TokenManager, InvalidTokenError, TokenExpiredError

logger = get_logger(__name__)


class ConnectionManager:
  """Менеджер WebSocket подключений к фронтенду."""

  def __init__(self):
    """Инициализация менеджера подключений."""
    self.active_connections: Set[WebSocket] = set()
    self.authenticated_connections: Dict[WebSocket, str] = {}
    logger.info("Инициализирован менеджер WebSocket подключений")

  async def connect(self, websocket: WebSocket) -> bool:
    """
    Подключение нового клиента.

    Args:
        websocket: WebSocket соединение

    Returns:
        bool: True если подключение успешно
    """
    await websocket.accept()
    self.active_connections.add(websocket)
    logger.info(f"Новое WebSocket подключение. Всего: {len(self.active_connections)}")
    return True

  def disconnect(self, websocket: WebSocket):
    """
    Отключение клиента.

    Args:
        websocket: WebSocket соединение
    """
    self.active_connections.discard(websocket)
    self.authenticated_connections.pop(websocket, None)
    logger.info(f"WebSocket отключен. Осталось: {len(self.active_connections)}")

  def authenticate(self, websocket: WebSocket, user_id: str):
    """
    Аутентификация соединения.

    Args:
        websocket: WebSocket соединение
        user_id: ID пользователя
    """
    self.authenticated_connections[websocket] = user_id
    logger.info(f"WebSocket аутентифицирован для пользователя: {user_id}")

  def is_authenticated(self, websocket: WebSocket) -> bool:
    """
    Проверка аутентификации.

    Args:
        websocket: WebSocket соединение

    Returns:
        bool: True если аутентифицирован
    """
    return websocket in self.authenticated_connections

  async def send_personal_message(self, message: dict, websocket: WebSocket):
    """
    Отправка сообщения конкретному клиенту.

    Args:
        message: Сообщение
        websocket: WebSocket соединение
    """
    try:
      await websocket.send_json(message)
    except Exception as e:
      logger.error(f"Ошибка отправки сообщения: {e}")
      self.disconnect(websocket)

  async def broadcast(self, message: dict, authenticated_only: bool = True):
    """
    Рассылка сообщения всем подключенным клиентам.

    Args:
        message: Сообщение
        authenticated_only: Только аутентифицированным клиентам
    """
    # Добавляем timestamp
    message["timestamp"] = datetime.now().isoformat()

    if authenticated_only:
      connections = [
        ws for ws in self.active_connections
        if ws in self.authenticated_connections
      ]
    else:
      connections = list(self.active_connections)

    # Отправляем сообщения асинхронно
    disconnected = []
    for connection in connections:
      try:
        await connection.send_json(message)
      except Exception as e:
        logger.error(f"Ошибка рассылки сообщения: {e}")
        disconnected.append(connection)

    # Удаляем отключенные соединения
    for ws in disconnected:
      self.disconnect(ws)

  def get_stats(self) -> Dict:
    """
    Получение статистики подключений.

    Returns:
        Dict: Статистика
    """
    return {
      "total_connections": len(self.active_connections),
      "authenticated_connections": len(self.authenticated_connections),
    }


# Глобальный менеджер подключений
manager = ConnectionManager()


async def handle_websocket_messages(websocket: WebSocket):
  """
  Обработка входящих WebSocket сообщений.

  Args:
      websocket: WebSocket соединение
  """
  try:
    while True:
      # Получаем сообщение от клиента
      data = await websocket.receive_text()
      message = json.loads(data)

      message_type = message.get("type")

      logger.debug(f"Получено WebSocket сообщение: type={message_type}, keys={list(message.keys())}")

      # Обрабатываем аутентификацию
      if message_type == "authenticate":
        token = message.get("token")

        if not token:
          await manager.send_personal_message({
            "type": "error",
            "message": "Токен не предоставлен"
          }, websocket)
          continue

        try:
          # Проверяем токен
          payload = TokenManager.verify_token(token)
          user_id = payload.get("sub")

          # Аутентифицируем соединение
          manager.authenticate(websocket, user_id)

          await manager.send_personal_message({
            "type": "authenticated",
            "user_id": user_id,
            "message": "Успешная аутентификация"
          }, websocket)

          logger.info(f"WebSocket успешно аутентифицирован: {user_id}")

        except (TokenExpiredError, InvalidTokenError) as e:
          await manager.send_personal_message({
            "type": "error",
            "message": str(e)
          }, websocket)

      # Обрабатываем ping
      elif message_type == "ping":
        await manager.send_personal_message({
          "type": "pong",
          "timestamp": datetime.now().isoformat()
        }, websocket)

      # Обрабатываем запрос подписки на обновления
      elif message_type == "subscribe":
        if not manager.is_authenticated(websocket):
          await manager.send_personal_message({
            "type": "error",
            "message": "Требуется аутентификация"
          }, websocket)
          continue

        channels = message.get("channels", [])
        await manager.send_personal_message({
          "type": "subscribed",
          "channels": channels,
          "message": f"Подписка на каналы: {', '.join(channels)}"
        }, websocket)

      # Неизвестный тип сообщения
      else:
        logger.warning(f"Неизвестный тип сообщения: {message_type}")

  except WebSocketDisconnect:
    logger.info("WebSocket клиент отключился")
  except Exception as e:
    logger.error(f"Ошибка обработки WebSocket сообщения: {e}")


async def broadcast_bot_status(status: str, details: dict = None):
  """
  Рассылка статуса бота всем клиентам.

  Args:
      status: Статус бота
      details: Дополнительные детали
  """
  message = {
    "type": "bot_status",
    "status": status,
    "details": details or {}
  }
  await manager.broadcast(message)


async def broadcast_metrics_update(symbol: str, metrics: dict):
  """
  Рассылка обновления метрик.

  Args:
      symbol: Торговая пара
      metrics: Метрики
  """
  message = {
    "type": "metrics_update",
    "symbol": symbol,
    "metrics": metrics
  }
  await manager.broadcast(message)


async def broadcast_signal(signal: dict):
  """
  Рассылка нового торгового сигнала.

  Args:
      signal: Торговый сигнал
  """
  message = {
    "type": "trading_signal",
    "signal": signal
  }
  await manager.broadcast(message)


async def broadcast_orderbook_update(symbol: str, orderbook: dict):
  """
  Рассылка обновления стакана.

  Args:
      symbol: Торговая пара
      orderbook: Данные стакана
  """
  message = {
    "type": "orderbook_update",
    "symbol": symbol,
    "orderbook": orderbook
  }
  await manager.broadcast(message)


async def broadcast_execution_update(execution: dict):
  """
  Рассылка обновления исполнения.

  Args:
      execution: Данные исполнения
  """
  message = {
    "type": "execution_update",
    "execution": execution
  }
  await manager.broadcast(message)


async def broadcast_error(error_message: str, error_type: str = "general"):
  """
  Рассылка ошибки.

  Args:
      error_message: Сообщение об ошибке
      error_type: Тип ошибки
  """
  message = {
    "type": "error",
    "error_type": error_type,
    "message": error_message
  }
  await manager.broadcast(message)

async def broadcast_screener_update(pairs: list):
    """
    Рассылка обновления данных скринера.

    Args:
        pairs: Список торговых пар с данными
    """
    message = {
      "type": "screener_update",
      "pairs": pairs,
      "timestamp": int(datetime.now().timestamp() * 1000)
    }
    await manager.broadcast(message, authenticated_only=True)