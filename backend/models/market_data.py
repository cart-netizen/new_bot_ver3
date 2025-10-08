"""
Модели данных для рыночной информации.
"""

from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
  """Сторона ордера."""
  BUY = "Buy"
  SELL = "Sell"


class OrderType(str, Enum):
  """Тип ордера."""
  MARKET = "Market"
  LIMIT = "Limit"


class OrderStatus(str, Enum):
  """Статус ордера."""
  NEW = "New"
  PARTIALLY_FILLED = "PartiallyFilled"
  FILLED = "Filled"
  CANCELLED = "Cancelled"
  REJECTED = "Rejected"


class TimeInForce(str, Enum):
  """Time in Force для ордера."""
  GTC = "GTC"  # Good Till Cancel
  IOC = "IOC"  # Immediate or Cancel
  FOK = "FOK"  # Fill or Kill
  POST_ONLY = "PostOnly"


@dataclass
class Balance:
  """Модель баланса счета."""

  asset: str
  free: float  # Доступный баланс
  locked: float  # Заблокированный баланс
  total: float  # Общий баланс

  def to_dict(self) -> dict:
    """Преобразование в словарь для API."""
    return {
      "asset": self.asset,
      "free": self.free,
      "locked": self.locked,
      "total": self.total,
    }


@dataclass
class Position:
  """Модель позиции."""

  symbol: str
  side: OrderSide
  size: float
  entry_price: float
  current_price: float
  unrealized_pnl: float
  realized_pnl: float
  leverage: int
  margin: float
  timestamp: int

  @property
  def pnl_percentage(self) -> float:
    """Процент прибыли/убытка."""
    if self.entry_price > 0:
      return ((self.current_price - self.entry_price) / self.entry_price) * 100
    return 0.0

  def to_dict(self) -> dict:
    """Преобразование в словарь для API."""
    return {
      "symbol": self.symbol,
      "side": self.side.value,
      "size": self.size,
      "entry_price": self.entry_price,
      "current_price": self.current_price,
      "unrealized_pnl": self.unrealized_pnl,
      "realized_pnl": self.realized_pnl,
      "pnl_percentage": self.pnl_percentage,
      "leverage": self.leverage,
      "margin": self.margin,
      "timestamp": self.timestamp,
      "datetime": datetime.fromtimestamp(self.timestamp / 1000).isoformat(),
    }


@dataclass
class Order:
  """Модель ордера."""

  order_id: str
  symbol: str
  side: OrderSide
  order_type: OrderType
  price: float
  quantity: float
  filled_quantity: float
  status: OrderStatus
  time_in_force: TimeInForce
  created_at: int
  updated_at: int

  @property
  def is_active(self) -> bool:
    """Проверка, активен ли ордер."""
    return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]

  @property
  def fill_percentage(self) -> float:
    """Процент заполнения ордера."""
    if self.quantity > 0:
      return (self.filled_quantity / self.quantity) * 100
    return 0.0

  def to_dict(self) -> dict:
    """Преобразование в словарь для API."""
    return {
      "order_id": self.order_id,
      "symbol": self.symbol,
      "side": self.side.value,
      "order_type": self.order_type.value,
      "price": self.price,
      "quantity": self.quantity,
      "filled_quantity": self.filled_quantity,
      "fill_percentage": self.fill_percentage,
      "status": self.status.value,
      "time_in_force": self.time_in_force.value,
      "is_active": self.is_active,
      "created_at": self.created_at,
      "updated_at": self.updated_at,
      "created_datetime": datetime.fromtimestamp(self.created_at / 1000).isoformat(),
      "updated_datetime": datetime.fromtimestamp(self.updated_at / 1000).isoformat(),
    }


@dataclass
class Trade:
  """Модель сделки."""

  trade_id: str
  order_id: str
  symbol: str
  side: OrderSide
  price: float
  quantity: float
  commission: float
  commission_asset: str
  timestamp: int

  @property
  def value(self) -> float:
    """Стоимость сделки."""
    return self.price * self.quantity

  def to_dict(self) -> dict:
    """Преобразование в словарь для API."""
    return {
      "trade_id": self.trade_id,
      "order_id": self.order_id,
      "symbol": self.symbol,
      "side": self.side.value,
      "price": self.price,
      "quantity": self.quantity,
      "value": self.value,
      "commission": self.commission,
      "commission_asset": self.commission_asset,
      "timestamp": self.timestamp,
      "datetime": datetime.fromtimestamp(self.timestamp / 1000).isoformat(),
    }


@dataclass
class TradingPairInfo:
  """Информация о торговой паре."""

  symbol: str
  base_asset: str
  quote_asset: str
  status: str
  min_order_qty: float
  max_order_qty: float
  min_order_value: float
  price_precision: int
  quantity_precision: int

  def to_dict(self) -> dict:
    """Преобразование в словарь для API."""
    return {
      "symbol": self.symbol,
      "base_asset": self.base_asset,
      "quote_asset": self.quote_asset,
      "status": self.status,
      "limits": {
        "min_order_qty": self.min_order_qty,
        "max_order_qty": self.max_order_qty,
        "min_order_value": self.min_order_value,
      },
      "precision": {
        "price": self.price_precision,
        "quantity": self.quantity_precision,
      }
    }


@dataclass
class MarketStats:
  """Рыночная статистика для пары."""

  symbol: str
  last_price: float
  high_24h: float
  low_24h: float
  volume_24h: float
  quote_volume_24h: float
  price_change_24h: float
  price_change_percent_24h: float
  timestamp: int

  def to_dict(self) -> dict:
    """Преобразование в словарь для API."""
    return {
      "symbol": self.symbol,
      "last_price": self.last_price,
      "24h": {
        "high": self.high_24h,
        "low": self.low_24h,
        "volume": self.volume_24h,
        "quote_volume": self.quote_volume_24h,
        "price_change": self.price_change_24h,
        "price_change_percent": self.price_change_percent_24h,
      },
      "timestamp": self.timestamp,
      "datetime": datetime.fromtimestamp(self.timestamp / 1000).isoformat(),
    }