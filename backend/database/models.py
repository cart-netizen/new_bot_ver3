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

from backend.database.connection import Base


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
  SYSTEM = "system"  # Системные события (daily reset, startup, shutdown)
  EMERGENCY_SHUTDOWN = "emergency_shutdown"  # Emergency shutdown из Daily Loss Killer

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
  # action = Column(Enum(AuditAction), nullable=False, index=True)
  # FIXED: Используем values_callable чтобы SQLAlchemy передавал .value (lowercase), а не .name (uppercase)
  action = Column(
    Enum(AuditAction, values_callable=lambda x: [e.value for e in x], name='auditaction'),
    nullable=False,
    index=True
  )
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


class LayeringPattern(Base):
  """
  Historical layering patterns for ML detection.

  Stores detected layering/spoofing patterns with behavioral fingerprints
  for pattern matching and blacklist management.
  """

  __tablename__ = "layering_patterns"

  # Identifiers
  id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
  pattern_id = Column(String(64), unique=True, nullable=False, index=True)

  # Timestamps
  first_seen = Column(DateTime, nullable=False, index=True)
  last_seen = Column(DateTime, nullable=False, index=True)
  occurrence_count = Column(Integer, default=1, nullable=False)

  # Fingerprint features (behavioral signature)
  avg_layer_count = Column(Float, nullable=False)
  avg_cancellation_rate = Column(Float, nullable=False)
  avg_volume_btc = Column(Float, nullable=False)
  avg_placement_duration = Column(Float, nullable=False)
  typical_spread_pct = Column(Float, nullable=False)
  typical_order_count = Column(Integer, nullable=False)
  spoofing_execution_ratio = Column(Float, nullable=True)

  # Temporal patterns (JSONB for array storage)
  time_of_day_pattern = Column(JSONB, nullable=True)  # [0-23] active hours
  avg_lifetime_seconds = Column(Float, nullable=False)

  # Computed hash for fast matching
  fingerprint_hash = Column(String(64), nullable=False, index=True)

  # Metadata
  symbols = Column(JSONB, nullable=False)  # List of symbols
  success_rate = Column(Float, default=0.0, nullable=False)  # How often manipulation succeeded
  avg_price_impact_bps = Column(Float, default=0.0, nullable=False)
  avg_confidence = Column(Float, nullable=False)

  # Risk assessment
  risk_level = Column(String(20), nullable=False)  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
  blacklist = Column(Boolean, default=False, nullable=False, index=True)

  # Notes
  notes = Column(Text, nullable=True)

  # Audit
  created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
  updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

  __table_args__ = (
    Index("idx_layering_fingerprint_hash", "fingerprint_hash"),
    Index("idx_layering_blacklist", "blacklist"),
    Index("idx_layering_last_seen", "last_seen"),
    Index("idx_layering_occurrence_count", "occurrence_count"),
  )

  def __repr__(self):
    return f"<LayeringPattern {self.pattern_id} blacklist={self.blacklist} occurrences={self.occurrence_count}>"


# ==================== BACKTESTING MODELS ====================

class BacktestStatus(str, enum.Enum):
  """Статус бэктеста."""
  PENDING = "pending"
  RUNNING = "running"
  COMPLETED = "completed"
  FAILED = "failed"
  CANCELLED = "cancelled"


class BacktestRun(Base):
  """
  История запусков бэктестов.

  Хранит конфигурацию, параметры и результаты каждого бэктеста.
  """

  __tablename__ = "backtest_runs"

  # Идентификаторы
  id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
  name = Column(String(200), nullable=False)  # Название теста
  description = Column(Text, nullable=True)  # Описание

  # Параметры бэктеста
  symbol = Column(String(20), nullable=False, index=True)
  start_date = Column(DateTime, nullable=False, index=True)
  end_date = Column(DateTime, nullable=False, index=True)
  initial_capital = Column(Float, nullable=False)

  # Конфигурация стратегий (JSONB для гибкости)
  strategies_config = Column(JSONB, nullable=False)
  # Пример: {
  #   "enabled_strategies": ["momentum", "sar_wave"],
  #   "strategy_params": {
  #     "momentum": {"period": 14, "threshold": 0.7},
  #     "sar_wave": {"acceleration": 0.02, "maximum": 0.2}
  #   },
  #   "consensus_mode": "weighted",
  #   "min_strategies_for_signal": 2
  # }

  # Risk Management конфигурация
  risk_config = Column(JSONB, nullable=False)
  # Пример: {
  #   "position_size_pct": 10.0,
  #   "max_open_positions": 3,
  #   "stop_loss_pct": 2.0,
  #   "take_profit_pct": 4.0,
  #   "use_trailing_stop": true
  # }

  # Exchange simulation конфигурация
  exchange_config = Column(JSONB, nullable=True)
  # Пример: {
  #   "commission_rate": 0.001,
  #   "slippage_model": "fixed",
  #   "slippage_pct": 0.01,
  #   "simulate_latency": false
  # }

  # Статус выполнения
  status = Column(
    Enum(BacktestStatus, values_callable=lambda x: [e.value for e in x], name='backteststatus'),
    nullable=False,
    default=BacktestStatus.PENDING,
    index=True
  )

  # Прогресс выполнения
  progress_pct = Column(Float, default=0.0)  # 0-100%
  current_date = Column(DateTime, nullable=True)  # Текущая дата в симуляции

  # Результаты - базовые
  total_trades = Column(Integer, default=0)
  winning_trades = Column(Integer, default=0)
  losing_trades = Column(Integer, default=0)
  final_capital = Column(Float, nullable=True)
  total_pnl = Column(Float, nullable=True)
  total_pnl_pct = Column(Float, nullable=True)

  # Результаты - детальные метрики (JSONB)
  metrics = Column(JSONB, nullable=True)
  # Пример: {
  #   "returns": {
  #     "total_return_pct": 45.2,
  #     "annual_return_pct": 18.5,
  #     "monthly_returns": [...]
  #   },
  #   "risk": {
  #     "sharpe_ratio": 1.85,
  #     "sortino_ratio": 2.1,
  #     "calmar_ratio": 1.2,
  #     "volatility_annual": 15.3
  #   },
  #   "drawdown": {
  #     "max_drawdown_pct": 12.5,
  #     "max_drawdown_duration_days": 45,
  #     "avg_drawdown_pct": 5.2
  #   },
  #   "trade_stats": {
  #     "win_rate_pct": 55.0,
  #     "profit_factor": 1.8,
  #     "avg_win": 120.5,
  #     "avg_loss": -65.3,
  #     "largest_win": 450.0,
  #     "largest_loss": -180.0
  #   },
  #   "advanced": {
  #     "omega_ratio": 1.45,
  #     "tail_ratio": 1.2,
  #     "var_95": -8.5,
  #     "cvar_95": -12.3
  #   }
  # }

  # Временные метки
  created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
  started_at = Column(DateTime, nullable=True)
  completed_at = Column(DateTime, nullable=True)
  duration_seconds = Column(Float, nullable=True)

  # Ошибки (если failed)
  error_message = Column(Text, nullable=True)
  error_traceback = Column(Text, nullable=True)

  # Метаданные
  metadata_json = Column(JSONB, nullable=True)

  # Связи
  trades = relationship("BacktestTrade", back_populates="backtest_run", cascade="all, delete-orphan")
  equity_curve = relationship("BacktestEquity", back_populates="backtest_run", cascade="all, delete-orphan")

  __table_args__ = (
    Index("idx_backtest_runs_symbol_status", "symbol", "status"),
    Index("idx_backtest_runs_created_at", "created_at"),
    Index("idx_backtest_runs_start_end", "start_date", "end_date"),
  )

  def __repr__(self):
    return f"<BacktestRun {self.name} {self.symbol} {self.status.value}>"


class BacktestTrade(Base):
  """
  Сделки бэктеста.

  Хранит все трейды выполненные во время бэктеста.
  """

  __tablename__ = "backtest_trades"

  # Идентификаторы
  id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
  backtest_run_id = Column(UUID(as_uuid=True), ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False, index=True)

  # Основные данные трейда
  symbol = Column(String(20), nullable=False)
  side = Column(Enum(OrderSide), nullable=False)

  # Время входа и выхода
  entry_time = Column(DateTime, nullable=False, index=True)
  exit_time = Column(DateTime, nullable=True)

  # Цены
  entry_price = Column(Float, nullable=False)
  exit_price = Column(Float, nullable=True)

  # Объем
  quantity = Column(Float, nullable=False)

  # Stop Loss / Take Profit
  stop_loss = Column(Float, nullable=True)
  take_profit = Column(Float, nullable=True)

  # PnL
  pnl = Column(Float, nullable=True)  # Абсолютный PnL в USDT
  pnl_pct = Column(Float, nullable=True)  # PnL в процентах
  commission = Column(Float, default=0.0)  # Комиссии

  # Контекст входа
  entry_signal = Column(JSONB, nullable=True)  # Данные сигнала входа
  # Пример: {
  #   "signal_type": "BUY",
  #   "source": "CONSENSUS",
  #   "confidence": 0.85,
  #   "contributing_strategies": ["momentum", "sar_wave"],
  #   "reason": "Strong uptrend momentum"
  # }

  # Причина выхода
  exit_reason = Column(String(100), nullable=True)  # "TP", "SL", "SIGNAL", "TIMEOUT", "END_OF_BACKTEST"
  exit_signal = Column(JSONB, nullable=True)  # Данные сигнала выхода (если есть)

  # Метрики трейда
  max_favorable_excursion = Column(Float, nullable=True)  # MFE - максимальная прибыль во время позиции
  max_adverse_excursion = Column(Float, nullable=True)   # MAE - максимальный убыток во время позиции
  duration_seconds = Column(Float, nullable=True)  # Длительность позиции

  # Рыночные условия при входе
  entry_market_data = Column(JSONB, nullable=True)
  # Пример: {
  #   "price": 42500.0,
  #   "volume_24h": 1500000000,
  #   "volatility": 2.5,
  #   "trend": "bullish"
  # }

  # Рыночные условия при выходе
  exit_market_data = Column(JSONB, nullable=True)

  # Метаданные
  metadata_json = Column(JSONB, nullable=True)

  # Связь
  backtest_run = relationship("BacktestRun", back_populates="trades")

  __table_args__ = (
    Index("idx_backtest_trades_run_time", "backtest_run_id", "entry_time"),
    Index("idx_backtest_trades_symbol", "symbol"),
  )

  def __repr__(self):
    return f"<BacktestTrade {self.symbol} {self.side.value} PnL={self.pnl}>"


class BacktestEquity(Base):
  """
  Кривая доходности (equity curve) бэктеста.

  Хранит снимки состояния капитала через определенные интервалы времени.
  Используется для построения графиков и расчета просадок.
  """

  __tablename__ = "backtest_equity"

  # Идентификаторы
  id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
  backtest_run_id = Column(UUID(as_uuid=True), ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False, index=True)

  # Временная метка
  timestamp = Column(DateTime, nullable=False, index=True)

  # Порядковый номер (для быстрой сортировки)
  sequence = Column(Integer, nullable=False)

  # Состояние капитала
  equity = Column(Float, nullable=False)  # Текущий капитал (USDT)
  cash = Column(Float, nullable=False)  # Свободные средства
  positions_value = Column(Float, default=0.0)  # Стоимость открытых позиций

  # Просадка
  peak_equity = Column(Float, nullable=False)  # Пиковый капитал до этого момента
  drawdown = Column(Float, default=0.0)  # Абсолютная просадка
  drawdown_pct = Column(Float, default=0.0)  # Процентная просадка

  # Доходность
  total_return = Column(Float, default=0.0)  # Абсолютная доходность
  total_return_pct = Column(Float, default=0.0)  # Процентная доходность

  # Количество открытых позиций
  open_positions_count = Column(Integer, default=0)

  # Связь
  backtest_run = relationship("BacktestRun", back_populates="equity_curve")

  __table_args__ = (
    Index("idx_backtest_equity_run_time", "backtest_run_id", "timestamp"),
    Index("idx_backtest_equity_sequence", "backtest_run_id", "sequence"),
  )

  def __repr__(self):
    return f"<BacktestEquity equity={self.equity} drawdown={self.drawdown_pct}%>"