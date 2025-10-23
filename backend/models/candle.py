from dataclasses import dataclass


@dataclass
class Candle:
  """Модель свечи (OHLCV)"""
  timestamp: int
  open: float
  high: float
  low: float
  close: float
  volume: float

  @property
  def typical_price(self) -> float:
    """Typical price для VWAP: (H+L+C)/3"""
    return (self.high + self.low + self.close) / 3

  @property
  def range(self) -> float:
    """Диапазон свечи (high - low)"""
    return self.high - self.low

  @property
  def body_size(self) -> float:
    """Размер тела свечи"""
    return abs(self.close - self.open)

  @property
  def is_bullish(self) -> bool:
    """Бычья свеча (закрытие выше открытия)"""
    return self.close > self.open

  @property
  def is_bearish(self) -> bool:
    """Медвежья свеча (закрытие ниже открытия)"""
    return self.close < self.open