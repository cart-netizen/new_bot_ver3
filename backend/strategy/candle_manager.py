"""
Менеджер свечей для торгового бота.
Хранит и управляет историей свечей для ML feature extraction.

Файл: backend/strategy/candle_manager.py
"""

from typing import List, Dict, Optional, Any
from collections import deque
from datetime import datetime

from core.logger import get_logger
from ml_engine.features.candle_feature_extractor import Candle

logger = get_logger(__name__)


class CandleManager:
  """
  Менеджер для хранения и обновления свечей.
  Используется для ML feature extraction.
  """

  def __init__(
      self,
      symbol: str,
      timeframe: str = "1m",
      max_candles: int = 200
  ):
    """
    Инициализация менеджера свечей.

    Args:
        symbol: Торговая пара
        timeframe: Таймфрейм свечей (1m, 5m, 15m, etc.)
        max_candles: Максимальное количество свечей в истории
    """
    self.symbol = symbol
    self.timeframe = timeframe
    self.max_candles = max_candles

    # История свечей (используем deque для эффективности)
    self.candles: deque[Candle] = deque(maxlen=max_candles)

    # Текущая (незакрытая) свеча
    self.current_candle: Optional[Candle] = None

    # Статистика
    self.total_candles_processed = 0
    self.last_update_timestamp: Optional[int] = None

    logger.info(
      f"CandleManager инициализирован для {symbol}, "
      f"timeframe={timeframe}, max_candles={max_candles}"
    )

  async def load_historical_data(self, candles_data: List[Dict[str, Any]]):
    """
    Загрузка исторических свечей.

    Args:
        candles_data: Список свечей от API биржи
            Формат: [
                [timestamp, open, high, low, close, volume, turnover],
                ...
            ]
    """
    try:
      loaded_count = 0

      for candle_data in candles_data:
        # Парсим данные свечи
        # Bybit возвращает: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
        if isinstance(candle_data, list) and len(candle_data) >= 6:
          timestamp = int(candle_data[0])
          open_price = float(candle_data[1])
          high_price = float(candle_data[2])
          low_price = float(candle_data[3])
          close_price = float(candle_data[4])
          volume = float(candle_data[5])

          candle = Candle(
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
          )

          self.candles.append(candle)
          loaded_count += 1

        elif isinstance(candle_data, dict):
          # Альтернативный формат (если API возвращает dict)
          candle = Candle(
            timestamp=int(candle_data.get("timestamp", 0)),
            open=float(candle_data.get("open", 0)),
            high=float(candle_data.get("high", 0)),
            low=float(candle_data.get("low", 0)),
            close=float(candle_data.get("close", 0)),
            volume=float(candle_data.get("volume", 0))
          )

          self.candles.append(candle)
          loaded_count += 1

      self.total_candles_processed += loaded_count

      logger.info(
        f"{self.symbol} | Загружено {loaded_count} исторических свечей, "
        f"всего в истории: {len(self.candles)}"
      )

    except Exception as e:
      logger.error(f"{self.symbol} | Ошибка загрузки исторических свечей: {e}")
      raise

  async def update_candle(
      self,
      candle_data: Any,
      is_closed: bool = False
  ):
    """
    Обновление свечи (закрытой или текущей).

    Args:
        candle_data: Данные свечи от API
        is_closed: Флаг закрытой свечи
    """
    try:
      # Парсим данные
      if isinstance(candle_data, list) and len(candle_data) >= 6:
        timestamp = int(candle_data[0])
        open_price = float(candle_data[1])
        high_price = float(candle_data[2])
        low_price = float(candle_data[3])
        close_price = float(candle_data[4])
        volume = float(candle_data[5])
      elif isinstance(candle_data, dict):
        timestamp = int(candle_data.get("timestamp", 0))
        open_price = float(candle_data.get("open", 0))
        high_price = float(candle_data.get("high", 0))
        low_price = float(candle_data.get("low", 0))
        close_price = float(candle_data.get("close", 0))
        volume = float(candle_data.get("volume", 0))
      else:
        logger.warning(f"{self.symbol} | Неподдерживаемый формат данных свечи")
        return

      candle = Candle(
        timestamp=timestamp,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume
      )

      if is_closed:
        # Закрытая свеча - добавляем в историю
        # Проверяем, не дубликат ли это
        if self.candles and self.candles[-1].timestamp == timestamp:
          # Обновляем последнюю свечу (может быть апдейт)
          self.candles[-1] = candle
          logger.debug(f"{self.symbol} | Обновлена закрытая свеча")
        else:
          # Новая свеча
          self.candles.append(candle)
          self.total_candles_processed += 1
          logger.debug(
            f"{self.symbol} | Добавлена новая свеча, "
            f"всего: {len(self.candles)}"
          )
      else:
        # Текущая (незакрытая) свеча
        self.current_candle = candle
        logger.debug(f"{self.symbol} | Обновлена текущая свеча")

      self.last_update_timestamp = timestamp

    except Exception as e:
      logger.error(f"{self.symbol} | Ошибка обновления свечи: {e}")

  def get_candles(self, count: Optional[int] = None) -> List[Candle]:
    """
    Получение списка свечей.

    Args:
        count: Количество последних свечей (None = все)

    Returns:
        List[Candle]: Список свечей
    """
    if count is None:
      return list(self.candles)
    else:
      return list(self.candles)[-count:]

  def get_latest_candle(self) -> Optional[Candle]:
    """
    Получение последней закрытой свечи.

    Returns:
        Optional[Candle]: Последняя свеча или None
    """
    return self.candles[-1] if self.candles else None

  def get_current_candle(self) -> Optional[Candle]:
    """
    Получение текущей (незакрытой) свечи.

    Returns:
        Optional[Candle]: Текущая свеча или None
    """
    return self.current_candle

  def get_candles_count(self) -> int:
    """
    Получение количества свечей в истории.

    Returns:
        int: Количество свечей
    """
    return len(self.candles)

  def get_statistics(self) -> Dict[str, Any]:
    """
    Получение статистики менеджера.

    Returns:
        Dict: Статистика
    """
    latest_candle = self.get_latest_candle()

    return {
      "symbol": self.symbol,
      "timeframe": self.timeframe,
      "candles_count": len(self.candles),
      "max_candles": self.max_candles,
      "total_processed": self.total_candles_processed,
      "last_update_timestamp": self.last_update_timestamp,
      "latest_candle": {
        "timestamp": latest_candle.timestamp,
        "close": latest_candle.close,
        "volume": latest_candle.volume
      } if latest_candle else None,
      "current_candle": {
        "timestamp": self.current_candle.timestamp,
        "close": self.current_candle.close
      } if self.current_candle else None
    }

  def clear_history(self):
    """Очистка истории свечей."""
    self.candles.clear()
    self.current_candle = None
    logger.info(f"{self.symbol} | История свечей очищена")

  def is_ready_for_ml(self, min_candles: int = 50) -> bool:
    """
    Проверка готовности для ML (достаточно ли свечей).

    Args:
        min_candles: Минимальное количество свечей

    Returns:
        bool: True если готов для ML
    """
    return len(self.candles) >= min_candles