# backend/screener/screener_manager.py
"""
Менеджер скринера торговых пар.
ИСПРАВЛЕН: Использует REST API вместо WebSocket для получения тикеров.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime

from backend.core.logger import get_logger
from backend.models.screener import ScreenerPairData
from backend.config import settings

logger = get_logger(__name__)


class ScreenerManager:
  """Менеджер скринера торговых пар."""

  def __init__(self):
    """Инициализация менеджера скринера."""
    self.pairs: Dict[str, ScreenerPairData] = {}

    # REST API endpoint для тикеров
    self.api_url = (
      "https://api-testnet.bybit.com"
      if settings.BYBIT_MODE == "testnet"
      else "https://api.bybit.com"
    )

    self.min_volume = settings.SCREENER_MIN_VOLUME
    self.max_pairs = settings.SCREENER_MAX_PAIRS

    self.is_running = False
    self.session: Optional[aiohttp.ClientSession] = None

    # Задачи
    self.fetch_task: Optional[asyncio.Task] = None
    self.cleanup_task: Optional[asyncio.Task] = None

    logger.info(
      f"Инициализирован ScreenerManager: "
      f"min_volume={self.min_volume:,.0f}, max_pairs={self.max_pairs}"
    )
    logger.info(f"API URL: {self.api_url}")

  async def start(self):
    """Запуск скринера."""
    if self.is_running:
      logger.warning("ScreenerManager уже запущен")
      return

    self.is_running = True
    logger.info("Запуск ScreenerManager...")

    # Создаем HTTP сессию
    self.session = aiohttp.ClientSession()

    # Запускаем задачи
    self.fetch_task = asyncio.create_task(self._fetch_loop())
    self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    logger.info("✓ ScreenerManager запущен")

  async def stop(self):
    """Остановка скринера."""
    if not self.is_running:
      return

    logger.info("Остановка ScreenerManager...")
    self.is_running = False

    # Отменяем задачи
    if self.fetch_task:
      self.fetch_task.cancel()
    if self.cleanup_task:
      self.cleanup_task.cancel()

    # Закрываем HTTP сессию
    if self.session:
      try:
        await self.session.close()
      except Exception as e:
        logger.error(f"Ошибка закрытия HTTP сессии: {e}")

    logger.info("✓ ScreenerManager остановлен")

  async def _fetch_loop(self):
    """Цикл получения данных через REST API."""
    logger.info("Запуск fetch loop")

    # Первая загрузка сразу
    await self._fetch_tickers()

    # Затем каждые 5 секунд
    while self.is_running:
      try:
        await asyncio.sleep(5)
        await self._fetch_tickers()
      except asyncio.CancelledError:
        break
      except Exception as e:
        logger.error(f"Ошибка в fetch loop: {e}")
        await asyncio.sleep(5)

  async def _fetch_tickers(self):
    """Получение тикеров через REST API."""
    if not self.session:
      return

    try:
      url = f"{self.api_url}/v5/market/tickers"
      params = {"category": "linear"}  # Все USDT фьючерсы

      async with self.session.get(url, params=params, timeout=10) as response:
        if response.status != 200:
          logger.error(f"API вернул статус {response.status}")
          return

        data = await response.json()

        if data.get("retCode") != 0:
          logger.error(f"API ошибка: {data.get('retMsg')}")
          return

        tickers = data.get("result", {}).get("list", [])

        if not tickers:
          logger.warning("API вернул пустой список тикеров")
          return

        # Обрабатываем тикеры
        processed = 0
        added = 0

        for ticker in tickers:
          symbol = ticker.get("symbol", "")

          # Только USDT пары
          if not symbol.endswith("USDT"):
            continue

          # Проверяем объем
          volume_24h = float(ticker.get("turnover24h", 0))
          if volume_24h < self.min_volume:
            continue

          # Проверяем лимит пар
          if symbol not in self.pairs and len(self.pairs) >= self.max_pairs:
            continue

          # Обновляем или создаем пару
          if symbol in self.pairs:
            self.pairs[symbol].update_from_ticker(ticker)
          else:
            pair = ScreenerPairData(symbol=symbol)
            pair.update_from_ticker(ticker)
            self.pairs[symbol] = pair
            added += 1
            logger.debug(
              f"Добавлена пара: {symbol}, "
              f"цена={pair.last_price:.2f}, "
              f"объем={volume_24h:,.0f}"
            )

          processed += 1

        logger.info(
          f"Обработано тикеров: {processed}, "
          f"добавлено новых: {added}, "
          f"всего пар: {len(self.pairs)}"
        )

    except asyncio.TimeoutError:
      logger.error("Таймаут при запросе к API")
    except Exception as e:
      logger.error(f"Ошибка получения тикеров: {e}")
      import traceback
      logger.error(f"Traceback:\n{traceback.format_exc()}")

  async def _cleanup_loop(self):
    """Цикл очистки неактивных пар и старой истории."""
    logger.info("Запуск cleanup loop")

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

        # Удаляем неактивные пары
        for symbol in inactive_symbols:
          del self.pairs[symbol]
          logger.debug(f"Удалена неактивная пара: {symbol}")

        if inactive_symbols:
          logger.info(
            f"Очищено неактивных пар: {len(inactive_symbols)}, "
            f"осталось: {len(self.pairs)}"
          )

        # Очищаем старую историю цен для всех активных пар
        for pair in self.pairs.values():
          pair.cleanup_old_history()

        logger.debug(f"Очищена старая история для {len(self.pairs)} пар")

      except asyncio.CancelledError:
        break
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