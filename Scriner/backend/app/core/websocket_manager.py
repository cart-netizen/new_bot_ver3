# backend/app/core/websocket_manager.py

import asyncio
from typing import List, Dict, Any
from fastapi import WebSocket

from .logger import log


class ConnectionManager:
  """
  Управляет активными WebSocket-соединениями с клиентами (фронтендом).
  Обеспечивает централизованную рассылку данных.
  """

  def __init__(self):
    self.active_connections: List[WebSocket] = []
    self.loop: asyncio.AbstractEventLoop | None = None  # Сохраняем цикл событий FastAPI
    log.info("Менеджер соединений инициализирован.")

  def set_loop(self, loop: asyncio.AbstractEventLoop):
    """Устанавливает основной цикл событий."""
    self.loop = loop

  async def connect(self, websocket: WebSocket):
    """
    Регистрирует новое WebSocket-соединение.
    """
    await websocket.accept()
    self.active_connections.append(websocket)
    log.info(
      f"Новое подключение: {websocket.client.host}:{websocket.client.port}. Всего клиентов: {len(self.active_connections)}")

  def disconnect(self, websocket: WebSocket):
    """
    Удаляет закрытое соединение из списка активных.
    """
    if websocket in self.active_connections:
      self.active_connections.remove(websocket)
      log.info(
        f"Клиент отключился: {websocket.client.host}:{websocket.client.port}. Осталось клиентов: {len(self.active_connections)}")

  async def broadcast(self, data: Dict[str, Any]):
    """
    Отправляет данные всем подключенным клиентам.
    """
    if not self.active_connections:
      return

    log.debug(f"Начало рассылки данных {len(self.active_connections)} клиентам.")

    # ДОБАВИТЬ ЭТУ СТРОКУ ДЛЯ ДИАГНОСТИКИ:
    log.debug(f"Размер данных: {len(str(data))} символов, пар: {len(data.get('pairs', []))}")

    # Используем `asyncio.gather` для параллельной отправки
    results = await asyncio.gather(
      *(connection.send_json(data) for connection in self.active_connections),
      return_exceptions=True
    )

    # Обрабатываем отключившихся клиентов
    disconnected_connections = []
    for i, result in enumerate(results):
      if isinstance(result, Exception):
        log.warning(f"Ошибка отправки данных клиенту: {result}")
        disconnected_connections.append(self.active_connections[i])

    for connection in disconnected_connections:
      self.disconnect(connection)


# Создаем единственный экземпляр менеджера для всего приложения
manager = ConnectionManager()
