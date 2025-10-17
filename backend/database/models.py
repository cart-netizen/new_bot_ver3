"""
SQLAlchemy модели для БД.
Полные модели с версионированием и аудитом.
"""

from datetime import datetime
from decimal import Decimal
from sqlalchemy import (
  Column, String, Float, Integer, Boolean, DateTime,
  Text, Enum, JSON, Index, UniqueConstraint, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid
import enum

from database.connection import Base


# ==================== ENUMS ====================

class OrderSide(str, enum.Enum):
  """Сторона ордера."""
  BUY = "Buy"
  SELL = "Sell"


class OrderType(str, enum.Enum):
  """Тип ордера."""
  MARKET = "Market"
  LIMIT = "Limit"


class OrderStatus(str, enum.Enum):
  """Статус ордера."""
  PENDING = "Pending"
  PLACED = "Placed"
  PARTIALLY_FILLED = "PartiallyFilled"
  FILLED = "Filled"
  CANCELLED = "Cancelled"
  REJECTED = "Rejected"
  FAILED = "Failed"


class PositionStatus(str, enum.Enum):
  """Статус позиции."""
  OPENING = "Opening"
  OPEN = "Open"
  CLOSING = "Closing"
  CLOSED = "Closed"
  FAILED = "Failed"


class AuditAction(str, enum.Enum):
  """Тип действия в аудите."""
  ORDER_PLACE = "order_place"
  ORDER_CANCEL = "order_cancel"
  ORDER_MODIFY = "order_modify"
  POSITION_OPEN = "position_open"
  POSITION_CLOSE = "position_close"
  POSITION_MODIFY = "position_modify"
  BALANCE_UPDATE = "balance_update"
  CONFIG_CHANGE = "config_change"


# ==================== MODELS ====================

class Order(Base):
  """Модель ордера с полным жизненным циклом."""

  __tablename__ = "orders"

  # Идентификаторы
  id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
  client_order_id = Column(String(100), unique=True, nullable=False, index=True)
  exchange_order_id = Column(String(100), unique=True, nullable=True, index=True)

  # Основные данные
  symbol = Column(String(20), nullable=False, index=True)
  side = Column(Enum(OrderSide), nullable=False)
  order_type = Column(Enum(OrderType), nullable=False)
  status = Column(Enum(OrderStatus), nullable=False, index=True)

  # Объемы и цены
  quantity = Column(Float, nullable=False)
  price = Column(Float, nullable=True)
  filled_quantity = Column(Float, default=0.0)
  average_fill_price = Column(Float, nullable=True)

  # Stop Loss / Take Profit
  stop_loss = Column(Float, nullable=True)
  take_profit = Column(Float, nullable=True)

  # Временные метки
  created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
  placed_at = Column(DateTime, nullable=True)
  filled_at = Column(DateTime, nullable=True)
  cancelled_at = Column(DateTime, nullable=True)
  updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

  # Контекст сделки (для анализа)
  signal_data = Column(JSONB, nullable=True)  # Данные сигнала
  market_data = Column(JSONB, nullable=True)  # Данные рынка на момент ордера
  indicators = Column(JSONB, nullable=True)  # Показатели индикаторов
  reason = Column(Text, nullable=True)  # Причина открытия/закрытия

  # Связь с позицией
  position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"), nullable=True)
  position = relationship("Position", back_populates="orders")

  # Версионирование
  version = Column(Integer, default=1, nullable=False)

  # Метаданные
  metadata_json = Column(JSONB, nullable=True)

  __table_args__ = (
    Index("idx_orders_symbol_status", "symbol", "status"),
    Index("idx_orders_created_at", "created_at"),
  )

  def __repr__(self):
    return f"<Order {self.client_order_id} {self.symbol} {self.side} {self.status}>"


class Position(Base):
  """Модель позиции с полным контекстом."""

  __tablename__ = "positions"

  # Идентификаторы
  id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
  symbol = Column(String(20), nullable=False, index=True)

  # Основные данные
  side = Column(Enum(OrderSide), nullable=False)
  status = Column(Enum(PositionStatus), nullable=False, index=True)

  # Объемы
  quantity = Column(Float, nullable=False)

  # Цены
  entry_price = Column(Float, nullable=False)
  exit_price = Column(Float, nullable=True)
  current_price = Column(Float, nullable=True)

  # Стопы
  stop_loss = Column(Float, nullable=True)
  take_profit = Column(Float, nullable=True)
  trailing_stop = Column(Float, nullable=True)

  # PnL
  unrealized_pnl = Column(Float, default=0.0)
  realized_pnl = Column(Float, nullable=True)

  # Временные метки
  opened_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
  closed_at = Column(DateTime, nullable=True)
  updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

  # Контекст открытия/закрытия
  entry_signal = Column(JSONB, nullable=True)  # Сигнал на вход
  exit_signal = Column(JSONB, nullable=True)  # Сигнал на выход
  entry_market_data = Column(JSONB, nullable=True)  # Рынок при входе
  exit_market_data = Column(JSONB, nullable=True)  # Рынок при выходе
  entry_indicators = Column(JSONB, nullable=True)  # Индикаторы при входе
  exit_indicators = Column(JSONB, nullable=True)  # Индикаторы при выходе
  entry_reason = Column(Text, nullable=True)  # Причина открытия
  exit_reason = Column(Text, nullable=True)  # Причина закрытия

  # Связи
  orders = relationship("Order", back_populates="position")

  # Версионирование
  version = Column(Integer, default=1, nullable=False)

  # Метаданные
  metadata_json = Column(JSONB, nullable=True)

  __table_args__ = (
    Index("idx_positions_symbol_status", "symbol", "status"),
    Index("idx_positions_opened_at", "opened_at"),
  )

  def __repr__(self):
    return f"<Position {self.symbol} {self.side} {self.status}>"


class Trade(Base):
  """История сделок (fills)."""

  __tablename__ = "trades"

  id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

  # Связи
  order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id"), nullable=False)
  position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"), nullable=True)

  # Данные сделки
  symbol = Column(String(20), nullable=False, index=True)
  side = Column(Enum(OrderSide), nullable=False)
  quantity = Column(Float, nullable=False)
  price = Column(Float, nullable=False)
  commission = Column(Float, default=0.0)

  # Временная метка
  executed_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

  # Exchange данные
  exchange_trade_id = Column(String(100), nullable=True)

  # Метаданные
  metadata_json = Column(JSONB, nullable=True)

  __table_args__ = (
    Index("idx_trades_symbol_executed", "symbol", "executed_at"),
  )


class AuditLog(Base):
  """Неизменяемый аудит всех операций."""

  __tablename__ = "audit_logs"

  id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

  # Контекст операции
  action = Column(Enum(AuditAction), nullable=False, index=True)
  entity_type = Column(String(50), nullable=False)
  entity_id = Column(String(100), nullable=False)

  # Данные операции
  old_value = Column(JSONB, nullable=True)
  new_value = Column(JSONB, nullable=True)
  changes = Column(JSONB, nullable=True)

  # Контекст выполнения
  reason = Column(Text, nullable=True)
  user_id = Column(String(100), nullable=True)
  trace_id = Column(String(100), nullable=True, index=True)

  # Результат
  success = Column(Boolean, nullable=False)
  error_message = Column(Text, nullable=True)

  # Временная метка (неизменяемая)
  timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

  # Дополнительный контекст
  context = Column(JSONB, nullable=True)

  __table_args__ = (
    Index("idx_audit_action_timestamp", "action", "timestamp"),
    Index("idx_audit_entity", "entity_type", "entity_id"),
  )


class IdempotencyCache(Base):
  """Кэш для идемпотентности операций."""

  __tablename__ = "idempotency_cache"

  id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

  # Ключ идемпотентности
  idempotency_key = Column(String(200), unique=True, nullable=False, index=True)

  # Операция
  operation = Column(String(100), nullable=False)

  # Результат
  result = Column(JSONB, nullable=True)
  success = Column(Boolean, nullable=False)
  error = Column(Text, nullable=True)

  # Временные метки
  created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
  expires_at = Column(DateTime, nullable=False, index=True)

  __table_args__ = (
    Index("idx_idempotency_key_created", "idempotency_key", "created_at"),
  )


class MarketDataSnapshot(Base):
  """Снимки рыночных данных (TimescaleDB hypertable)."""

  __tablename__ = "market_data_snapshots"

  id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

  # Данные
  symbol = Column(String(20), nullable=False, index=True)
  timestamp = Column(DateTime, nullable=False, index=True)

  # Цены
  best_bid = Column(Float, nullable=True)
  best_ask = Column(Float, nullable=True)
  mid_price = Column(Float, nullable=True)

  # Объемы
  bid_volume = Column(Float, nullable=True)
  ask_volume = Column(Float, nullable=True)

  # Метрики
  imbalance = Column(Float, nullable=True)
  spread = Column(Float, nullable=True)

  # Полные данные
  orderbook = Column(JSONB, nullable=True)
  metrics = Column(JSONB, nullable=True)

  __table_args__ = (
    Index("idx_market_symbol_timestamp", "symbol", "timestamp"),
  )