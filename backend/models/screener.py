# backend/models/screener.py
"""
Модели данных для скринера торговых пар.
Хранит информацию о торговых парах для отображения в списке.
ОБНОВЛЕНО: Добавлена история цен для расчета динамики по всем интервалам.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
from collections import deque


@dataclass
class PriceSnapshot:
  """Снимок цены в определенный момент времени."""
  price: float
  timestamp: int  # Миллисекунды


@dataclass
class ScreenerPairData:
  """Данные торговой пары для скринера."""

  symbol: str  # Торговая пара (BTCUSDT)
  last_price: float = 0.0  # Текущая цена
  volume_24h: float = 0.0  # Объем за 24ч в USDT
  price_change_24h_percent: float = 0.0  # Изменение цены за 24ч (%)
  high_24h: float = 0.0  # Максимум за 24ч
  low_24h: float = 0.0  # Минимум за 24ч

  # Все таймфреймы для изменения цены
  price_change_1m: Optional[float] = None
  price_change_2m: Optional[float] = None
  price_change_5m: Optional[float] = None
  price_change_15m: Optional[float] = None
  price_change_30m: Optional[float] = None
  price_change_1h: Optional[float] = None
  price_change_4h: Optional[float] = None
  price_change_8h: Optional[float] = None
  price_change_12h: Optional[float] = None
  price_change_24h: Optional[float] = None

  # Метаданные
  last_update: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
  is_selected: bool = False  # Выбрана ли для графиков

  # История цен (MEMORY FIX: хранится в памяти, ограничено 2 часами вместо 24)
  # deque для эффективного добавления/удаления с обоих концов
  # 2 часа * 60 мин = 120 записей (вместо 1440)
  price_history: deque = field(default_factory=lambda: deque(maxlen=120))

  # MEMORY FIX: Timestamp последнего добавленного snapshot (для throttling)
  last_snapshot_time: int = 0

  def to_dict(self) -> Dict:
    """Преобразование в словарь для API."""
    return {
      "symbol": self.symbol,
      "last_price": self.last_price,
      "volume_24h": self.volume_24h,
      "price_change_24h_percent": self.price_change_24h_percent,
      "high_24h": self.high_24h,
      "low_24h": self.low_24h,
      "price_change_1m": self.price_change_1m,
      "price_change_2m": self.price_change_2m,
      "price_change_5m": self.price_change_5m,
      "price_change_15m": self.price_change_15m,
      "price_change_30m": self.price_change_30m,
      "price_change_1h": self.price_change_1h,
      "price_change_4h": self.price_change_4h,
      "price_change_8h": self.price_change_8h,
      "price_change_12h": self.price_change_12h,
      "price_change_24h": self.price_change_24h,
      "last_update": self.last_update,
      "is_selected": self.is_selected,
    }

  def update_from_ticker(self, ticker_data: Dict):
    """Обновление данных из тикера Bybit."""
    current_price = float(ticker_data.get("lastPrice", self.last_price))
    current_time = int(datetime.now().timestamp() * 1000)

    # Обновляем базовые данные
    self.last_price = current_price
    self.volume_24h = float(ticker_data.get("turnover24h", self.volume_24h))
    self.price_change_24h_percent = float(ticker_data.get("price24hPcnt", 0)) * 100
    self.high_24h = float(ticker_data.get("highPrice24h", self.high_24h))
    self.low_24h = float(ticker_data.get("lowPrice24h", self.low_24h))
    self.last_update = current_time

    # MEMORY FIX: Добавляем snapshot только раз в минуту (throttling)
    # Вместо каждые 5 секунд, сохраняем раз в 60 секунд
    time_since_last_snapshot = current_time - self.last_snapshot_time
    if time_since_last_snapshot >= 60_000 or self.last_snapshot_time == 0:
      self.price_history.append(PriceSnapshot(price=current_price, timestamp=current_time))
      self.last_snapshot_time = current_time

      # Рассчитываем процентные изменения для всех интервалов
      self._calculate_price_changes()

  def _calculate_price_changes(self):
    """
    Рассчитывает процентные изменения цены для всех временных интервалов.
    """
    if len(self.price_history) < 2:
      return

    current_time = self.last_update
    current_price = self.last_price

    # Определяем интервалы в миллисекундах
    intervals = {
      '1m': 1 * 60 * 1000,
      '2m': 2 * 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '30m': 30 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      '4h': 4 * 60 * 60 * 1000,
      '8h': 8 * 60 * 60 * 1000,
      '12h': 12 * 60 * 60 * 1000,
      '24h': 24 * 60 * 60 * 1000,
    }

    # Рассчитываем для каждого интервала
    for interval_name, interval_ms in intervals.items():
      target_time = current_time - interval_ms
      old_price = self._find_closest_price(target_time)

      if old_price is not None and old_price > 0:
        percent_change = ((current_price - old_price) / old_price) * 100

        # Присваиваем значение соответствующему полю
        if interval_name == '1m':
          self.price_change_1m = percent_change
        elif interval_name == '2m':
          self.price_change_2m = percent_change
        elif interval_name == '5m':
          self.price_change_5m = percent_change
        elif interval_name == '15m':
          self.price_change_15m = percent_change
        elif interval_name == '30m':
          self.price_change_30m = percent_change
        elif interval_name == '1h':
          self.price_change_1h = percent_change
        elif interval_name == '4h':
          self.price_change_4h = percent_change
        elif interval_name == '8h':
          self.price_change_8h = percent_change
        elif interval_name == '12h':
          self.price_change_12h = percent_change
        elif interval_name == '24h':
          self.price_change_24h = percent_change

  def _find_closest_price(self, target_timestamp: int) -> Optional[float]:
    """
    Находит ближайшую цену к указанному времени в истории.

    Args:
        target_timestamp: Целевое время в миллисекундах

    Returns:
        Цена или None, если история пустая
    """
    if not self.price_history:
      return None

    # Ищем ближайший снимок (бинарный поиск не нужен для deque, делаем линейный)
    closest_snapshot = None
    min_diff = float('inf')

    for snapshot in self.price_history:
      diff = abs(snapshot.timestamp - target_timestamp)
      if diff < min_diff:
        min_diff = diff
        closest_snapshot = snapshot

    # Возвращаем цену, если разница не более 2 минут (допустимая погрешность)
    if closest_snapshot and min_diff <= 2 * 60 * 1000:
      return closest_snapshot.price

    return None

  def cleanup_old_history(self):
    """
    Очищает историю цен старше 2 часов (MEMORY FIX: 24ч → 2ч).
    Вызывается периодически для экономии памяти.
    """
    if not self.price_history:
      return

    current_time = int(datetime.now().timestamp() * 1000)
    cutoff_time = current_time - (2 * 60 * 60 * 1000)  # MEMORY FIX: 2 часа назад (было 24)

    # Удаляем старые снимки с начала deque
    while self.price_history and self.price_history[0].timestamp < cutoff_time:
      self.price_history.popleft()
