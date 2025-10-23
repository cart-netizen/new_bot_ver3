"""
Модели данных для стакана ордеров (OrderBook).
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OrderBookLevel:
  """Модель одного уровня стакана."""

  price: float
  quantity: float

  def __repr__(self) -> str:
    return f"Level(price={self.price:.8f}, qty={self.quantity:.8f})"


@dataclass
class OrderBookSnapshot:
  """Модель снимка стакана ордеров."""

  symbol: str
  bids: List[Tuple[float, float]]  # [(price, quantity), ...]
  asks: List[Tuple[float, float]]  # [(price, quantity), ...]
  timestamp: int
  update_id: Optional[int] = None
  sequence_id: Optional[int] = None

  def __post_init__(self):
    """Валидация и сортировка данных после инициализации."""
    # Сортируем bids по убыванию цены (лучшие предложения первыми)
    self.bids = sorted(self.bids, key=lambda x: x[0], reverse=True)
    # Сортируем asks по возрастанию цены (лучшие предложения первыми)
    self.asks = sorted(self.asks, key=lambda x: x[0])

  @property
  def best_bid(self) -> Optional[float]:
    """Лучшая цена покупки."""
    return self.bids[0][0] if self.bids else None

  @property
  def best_ask(self) -> Optional[float]:
    """Лучшая цена продажи."""
    return self.asks[0][0] if self.asks else None

  @property
  def spread(self) -> Optional[float]:
    """Спред между лучшими ценами."""
    if self.best_bid and self.best_ask:
      return self.best_ask - self.best_bid
    return None

  @property
  def mid_price(self) -> Optional[float]:
    """Средняя цена между лучшими bid и ask."""
    if self.best_bid and self.best_ask:
      return (self.best_bid + self.best_ask) / 2
    return None

  def to_dict(self) -> dict:
    """Преобразование в словарь для API."""
    return {
      "symbol": self.symbol,
      "bids": [[price, qty] for price, qty in self.bids[:20]],  # Топ 20
      "asks": [[price, qty] for price, qty in self.asks[:20]],  # Топ 20
      "timestamp": self.timestamp,
      "best_bid": self.best_bid,
      "best_ask": self.best_ask,
      "spread": self.spread,
      "mid_price": self.mid_price,
      "update_id": self.update_id,
      "sequence_id": self.sequence_id
    }


@dataclass
class OrderBookDelta:
  """Модель дельта-обновления стакана."""

  symbol: str
  bids_update: List[Tuple[float, float]]  # [(price, quantity), ...]
  asks_update: List[Tuple[float, float]]  # [(price, quantity), ...]
  timestamp: int
  update_id: Optional[int] = None
  sequence_id: Optional[int] = None


@dataclass
class OrderBookMetrics:
  """Модель метрик стакана для анализа."""

  symbol: str
  timestamp: int

  # Основные метрики
  best_bid: Optional[float] = None
  best_ask: Optional[float] = None
  spread: Optional[float] = None
  mid_price: Optional[float] = None

  # Объемные метрики
  total_bid_volume: float = 0.0
  total_ask_volume: float = 0.0
  bid_volume_depth_5: float = 0.0  # Объем на 5 уровнях
  ask_volume_depth_5: float = 0.0  # Объем на 5 уровнях
  bid_volume_depth_10: float = 0.0  # Объем на 10 уровнях
  ask_volume_depth_10: float = 0.0  # Объем на 10 уровнях

  # Дисбаланс
  imbalance: float = 0.5  # 0.0 (только продажа) - 0.5 (баланс) - 1.0 (только покупка)
  imbalance_depth_5: float = 0.5
  imbalance_depth_10: float = 0.5

  # Средневзвешенная цена
  vwap_bid: Optional[float] = None  # Volume Weighted Average Price для bid
  vwap_ask: Optional[float] = None  # Volume Weighted Average Price для ask
  vwmp: Optional[float] = None  # Volume Weighted Mid Price

  # Кластеры
  largest_bid_cluster_price: Optional[float] = None
  largest_bid_cluster_volume: float = 0.0
  largest_ask_cluster_price: Optional[float] = None
  largest_ask_cluster_volume: float = 0.0

  @property
  def spread_bps(self) -> Optional[float]:
    """
    Спред в базисных пунктах (basis points).

    Returns:
        float: Спред в bps (1 bp = 0.01%)
        None: Если невозможно рассчитать
    """
    if self.spread and self.mid_price and self.mid_price > 0:
      return (self.spread / self.mid_price) * 10000
    return None

  @property
  def spread_percentage(self) -> Optional[float]:
    """
    Спред в процентах.

    Returns:
        float: Спред в % от mid_price
        None: Если невозможно рассчитать
    """
    if self.spread and self.mid_price and self.mid_price > 0:
      return (self.spread / self.mid_price) * 100
    return None

  def to_dict(self) -> dict:
    """Преобразование в словарь для API."""
    return {
      "symbol": self.symbol,
      "timestamp": self.timestamp,
      "datetime": datetime.fromtimestamp(self.timestamp / 1000).isoformat(),
      "prices": {
        "best_bid": self.best_bid,
        "best_ask": self.best_ask,
        "spread": self.spread,
        "mid_price": self.mid_price,
      },
      "volumes": {
        "total_bid": self.total_bid_volume,
        "total_ask": self.total_ask_volume,
        "bid_depth_5": self.bid_volume_depth_5,
        "ask_depth_5": self.ask_volume_depth_5,
        "bid_depth_10": self.bid_volume_depth_10,
        "ask_depth_10": self.ask_volume_depth_10,
      },
      "imbalance": {
        "overall": self.imbalance,
        "depth_5": self.imbalance_depth_5,
        "depth_10": self.imbalance_depth_10,
      },
      "vwap": {
        "bid": self.vwap_bid,
        "ask": self.vwap_ask,
        "mid": self.vwmp,
      },
      "clusters": {
        "largest_bid": {
          "price": self.largest_bid_cluster_price,
          "volume": self.largest_bid_cluster_volume,
        },
        "largest_ask": {
          "price": self.largest_ask_cluster_price,
          "volume": self.largest_ask_cluster_volume,
        }
      }
    }