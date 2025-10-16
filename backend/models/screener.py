# backend/models/screener.py
"""
Модели данных для скринера торговых пар.
Хранит информацию о торговых парах для отображения в списке.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
from datetime import datetime


@dataclass
class ScreenerPairData:
  """Данные торговой пары для скринера."""

  symbol: str  # Торговая пара (BTCUSDT)
  last_price: float = 0.0  # Текущая цена
  volume_24h: float = 0.0  # Объем за 24ч в USDT
  price_change_24h_percent: float = 0.0  # Изменение цены за 24ч (%)
  high_24h: float = 0.0  # Максимум за 24ч
  low_24h: float = 0.0  # Минимум за 24ч

  # Таймфреймы для изменения цены
  price_change_5m: Optional[float] = None
  price_change_15m: Optional[float] = None
  price_change_1h: Optional[float] = None
  price_change_4h: Optional[float] = None

  # Метаданные
  last_update: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
  is_selected: bool = False  # Выбрана ли для графиков

  def to_dict(self) -> Dict:
    """Преобразование в словарь для API."""
    return {
      "symbol": self.symbol,
      "last_price": self.last_price,
      "volume_24h": self.volume_24h,
      "price_change_24h_percent": self.price_change_24h_percent,
      "high_24h": self.high_24h,
      "low_24h": self.low_24h,
      "price_change_5m": self.price_change_5m,
      "price_change_15m": self.price_change_15m,
      "price_change_1h": self.price_change_1h,
      "price_change_4h": self.price_change_4h,
      "last_update": self.last_update,
      "is_selected": self.is_selected,
    }

  def update_from_ticker(self, ticker_data: Dict):
    """Обновление данных из тикера Bybit."""
    self.last_price = float(ticker_data.get("lastPrice", self.last_price))
    self.volume_24h = float(ticker_data.get("turnover24h", self.volume_24h))
    self.price_change_24h_percent = float(ticker_data.get("price24hPcnt", 0)) * 100
    self.high_24h = float(ticker_data.get("highPrice24h", self.high_24h))
    self.low_24h = float(ticker_data.get("lowPrice24h", self.low_24h))
    self.last_update = int(datetime.now().timestamp() * 1000)