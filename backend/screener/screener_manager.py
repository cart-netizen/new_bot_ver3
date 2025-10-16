# backend/screener/screener_manager.py
"""
Менеджер скринера торговых пар.
Подключается к WebSocket Bybit, получает тикеры всех пар,
фильтрует по объему и рассылает данные клиентам.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from core.logger import get_logger
from models.screener import ScreenerPairData
from config import settings

logger = get_logger(__name__)


class ScreenerManager:
  """Менеджер скринера торговых пар."""

  def __init__(self):
    """Инициализация менеджера скринера."""
    self.pairs: Dict[str, ScreenerPairData] = {}
    self.ws_url = settings.BYBIT_WS_URL
    self.min_volume = settings.SCREENER_MIN_VOLUME
    self.max_pairs = settings.SCREENER_MAX_PAIRS

    self.is_running = False
    self.ws_connection: Optional[Any] = None  # Тип Any для websocket

    # Задачи
    self.ws_task: Optional[asyncio.Task] = None
    self.cleanup_task: Optional[asyncio.Task] = None

    logger.info(
      f"Инициализирован ScreenerManager: "
      f"min_volume={self.min_volume:,.0f}, max_pairs={self.max_pairs}"
    )

  async def start(self):
    """Запуск скринера."""
    if self.is_running:
      logger.warning("ScreenerManager уже запущен")
      return

    self.is_running = True
    logger.info("Запуск ScreenerManager...")

    # Запускаем задачи
    self.ws_task = asyncio.create_task(self._websocket_loop())
    self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    logger.info("✓ ScreenerManager запущен")

  async def stop(self):
    """Остановка скринера."""
    if not self.is_running:
      return

    logger.info("Остановка ScreenerManager...")
    self.is_running = False

    # Отменяем задачи
    if self.ws_task:
      self.ws_task.cancel()
    if self.cleanup_task:
      self.cleanup_task.cancel()

    # Закрываем WebSocket
    if self.ws_connection:
      try:
        await self.ws_connection.close()
      except Exception as e:
        logger.error(f"Ошибка закрытия WebSocket: {e}")

    logger.info("✓ ScreenerManager остановлен")

  async def _websocket_loop(self):
    """Цикл WebSocket подключения."""
    # Импортируем websockets здесь, чтобы избежать проблем с типами
    try:
      import websockets
    except ImportError:
      logger.error("Библиотека websockets не установлена. Установите: pip install websockets")
      return

    while self.is_running:
      try:
        logger.info(f"Подключение к WebSocket: {self.ws_url}")

        async with websockets.connect(self.ws_url) as ws:
          self.ws_connection = ws

          # Подписываемся на все тикеры
          subscribe_msg = {
            "op": "subscribe",
            "args": ["tickers.linear"]  # Все линейные (USDT) фьючерсы
          }
          await ws.send(json.dumps(subscribe_msg))
          logger.info("✓ Подписка на тикеры отправлена")

          # Обрабатываем сообщения
          async for message in ws:
            if not self.is_running:
              break

            await self._handle_message(message)

      except Exception as e:
        logger.error(f"Ошибка WebSocket: {e}")
        if self.is_running:
          await asyncio.sleep(5)  # Переподключение через 5 сек

  async def _handle_message(self, message: str):
    """Обработка сообщения от WebSocket."""
    try:
      data = json.loads(message)

      # Пропускаем служебные сообщения
      if data.get("op") in ["subscribe", "pong"]:
        return

      topic = data.get("topic")
      if not topic or not topic.startswith("tickers"):
        return

      # Обрабатываем тикеры
      ticker_data = data.get("data")
      if not ticker_data:
        return

      symbol = ticker_data.get("symbol")
      if not symbol or not symbol.endswith("USDT"):
        return  # Только USDT пары

      # Проверяем объем
      volume_24h = float(ticker_data.get("turnover24h", 0))
      if volume_24h < self.min_volume:
        return  # Фильтруем по объему

      # Обновляем или создаем пару
      if symbol in self.pairs:
        self.pairs[symbol].update_from_ticker(ticker_data)
      else:
        # Проверяем лимит пар
        if len(self.pairs) >= self.max_pairs:
          return

        pair = ScreenerPairData(symbol=symbol)
        pair.update_from_ticker(ticker_data)
        self.pairs[symbol] = pair

        logger.debug(
          f"Добавлена новая пара: {symbol}, "
          f"объем={volume_24h:,.0f} USDT"
        )

    except Exception as e:
      logger.error(f"Ошибка обработки сообщения: {e}")

  async def _cleanup_loop(self):
    """Цикл очистки неактивных пар."""
    while self.is_running:
      try:
        await asyncio.sleep(settings.SCREENER_CLEANUP_INTERVAL)

        current_time = int(datetime.now().timestamp() * 1000)
        ttl_ms = settings.SCREENER_INACTIVE_TTL * 1000

        # Находим неактивные пары
        inactive_symbols = [
          symbol for symbol, pair in self.pairs.items()
          if (current_time - pair.last_update) > ttl_ms
        ]

        # Удаляем
        for symbol in inactive_symbols:
          del self.pairs[symbol]
          logger.debug(f"Удалена неактивная пара: {symbol}")

        if inactive_symbols:
          logger.info(
            f"Очищено неактивных пар: {len(inactive_symbols)}, "
            f"осталось: {len(self.pairs)}"
          )

      except Exception as e:
        logger.error(f"Ошибка очистки: {e}")

  def get_all_pairs(self) -> List[Dict]:
    """Получить все пары для API."""
    return [pair.to_dict() for pair in self.pairs.values()]

  def get_pair(self, symbol: str) -> Optional[Dict]:
    """Получить конкретную пару."""
    pair = self.pairs.get(symbol)
    return pair.to_dict() if pair else None

  def toggle_selection(self, symbol: str) -> bool:
    """Переключить выбор пары для графиков."""
    if symbol in self.pairs:
      self.pairs[symbol].is_selected = not self.pairs[symbol].is_selected
      logger.info(
        f"Пара {symbol} {'выбрана' if self.pairs[symbol].is_selected else 'снята'}"
      )
      return True
    return False

  def get_selected_pairs(self) -> List[str]:
    """Получить список выбранных пар."""
    return [
      symbol for symbol, pair in self.pairs.items()
      if pair.is_selected
    ]

  def get_stats(self) -> Dict:
    """Получить статистику скринера."""
    return {
      "total_pairs": len(self.pairs),
      "selected_pairs": len(self.get_selected_pairs()),
      "is_running": self.is_running,
      "min_volume": self.min_volume,
      "max_pairs": self.max_pairs,
    }