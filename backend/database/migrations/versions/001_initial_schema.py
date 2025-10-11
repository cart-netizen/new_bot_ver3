"""
Initial database schema.

Revision ID: 001_initial
Create Date: 2025-01-11
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
  """Создание начальной схемы БД."""

  # Создаем расширение TimescaleDB
  op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

  # Создаем ENUM типы
  op.execute("""
        CREATE TYPE orderside AS ENUM ('Buy', 'Sell');
        CREATE TYPE ordertype AS ENUM ('Market', 'Limit');
        CREATE TYPE orderstatus AS ENUM (
            'Pending', 'Placed', 'PartiallyFilled', 
            'Filled', 'Cancelled', 'Rejected', 'Failed'
        );
        CREATE TYPE positionstatus AS ENUM (
            'Opening', 'Open', 'Closing', 'Closed'
        );
        CREATE TYPE auditaction AS ENUM (
            'order_place', 'order_cancel', 'order_modify',
            'position_open', 'position_close', 'position_modify',
            'balance_update', 'config_change'
        );
    """)

  # Таблица Positions
  op.create_table(
    'positions',
    sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column('symbol', sa.String(20), nullable=False, index=True),
    sa.Column('side', postgresql.ENUM('Buy', 'Sell', name='orderside'), nullable=False),
    sa.Column('status', postgresql.ENUM('Opening', 'Open', 'Closing', 'Closed', name='positionstatus'), nullable=False,
              index=True),
    sa.Column('quantity', sa.Float(), nullable=False),
    sa.Column('entry_price', sa.Float(), nullable=False),
    sa.Column('exit_price', sa.Float(), nullable=True),
    sa.Column('current_price', sa.Float(), nullable=True),
    sa.Column('stop_loss', sa.Float(), nullable=True),
    sa.Column('take_profit', sa.Float(), nullable=True),
    sa.Column('trailing_stop', sa.Float(), nullable=True),
    sa.Column('unrealized_pnl', sa.Float(), default=0.0),
    sa.Column('realized_pnl', sa.Float(), nullable=True),
    sa.Column('opened_at', sa.DateTime(), nullable=False, index=True),
    sa.Column('closed_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.Column('entry_signal', postgresql.JSONB(), nullable=True),
    sa.Column('exit_signal', postgresql.JSONB(), nullable=True),
    sa.Column('entry_market_data', postgresql.JSONB(), nullable=True),
    sa.Column('exit_market_data', postgresql.JSONB(), nullable=True),
    sa.Column('entry_indicators', postgresql.JSONB(), nullable=True),
    sa.Column('exit_indicators', postgresql.JSONB(), nullable=True),
    sa.Column('entry_reason', sa.Text(), nullable=True),
    sa.Column('exit_reason', sa.Text(), nullable=True),
    sa.Column('version', sa.Integer(), default=1, nullable=False),
    sa.Column('metadata_json', postgresql.JSONB(), nullable=True),
  )

  # Индексы для positions
  op.create_index('idx_positions_symbol_status', 'positions', ['symbol', 'status'])
  op.create_index('idx_positions_opened_at', 'positions', ['opened_at'])

  # Таблица Orders
  op.create_table(
    'orders',
    sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column('client_order_id', sa.String(100), unique=True, nullable=False, index=True),
    sa.Column('exchange_order_id', sa.String(100), unique=True, nullable=True, index=True),
    sa.Column('symbol', sa.String(20), nullable=False, index=True),
    sa.Column('side', postgresql.ENUM('Buy', 'Sell', name='orderside'), nullable=False),
    sa.Column('order_type', postgresql.ENUM('Market', 'Limit', name='ordertype'), nullable=False),
    sa.Column('status',
              postgresql.ENUM('Pending', 'Placed', 'PartiallyFilled', 'Filled', 'Cancelled', 'Rejected', 'Failed',
                              name='orderstatus'), nullable=False, index=True),
    sa.Column('quantity', sa.Float(), nullable=False),
    sa.Column('price', sa.Float(), nullable=True),
    sa.Column('filled_quantity', sa.Float(), default=0.0),
    sa.Column('average_fill_price', sa.Float(), nullable=True),
    sa.Column('stop_loss', sa.Float(), nullable=True),
    sa.Column('take_profit', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False, index=True),
    sa.Column('placed_at', sa.DateTime(), nullable=True),
    sa.Column('filled_at', sa.DateTime(), nullable=True),
    sa.Column('cancelled_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.Column('signal_data', postgresql.JSONB(), nullable=True),
    sa.Column('market_data', postgresql.JSONB(), nullable=True),
    sa.Column('indicators', postgresql.JSONB(), nullable=True),
    sa.Column('reason', sa.Text(), nullable=True),
    sa.Column('position_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('positions.id'), nullable=True),
    sa.Column('version', sa.Integer(), default=1, nullable=False),
    sa.Column('metadata_json', postgresql.JSONB(), nullable=True),
  )

  # Индексы для orders
  op.create_index('idx_orders_symbol_status', 'orders', ['symbol', 'status'])
  op.create_index('idx_orders_created_at', 'orders', ['created_at'])

  # Таблица Trades
  op.create_table(
    'trades',
    sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column('order_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('orders.id'), nullable=False),
    sa.Column('position_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('positions.id'), nullable=True),
    sa.Column('symbol', sa.String(20), nullable=False, index=True),
    sa.Column('side', postgresql.ENUM('Buy', 'Sell', name='orderside'), nullable=False),
    sa.Column('quantity', sa.Float(), nullable=False),
    sa.Column('price', sa.Float(), nullable=False),
    sa.Column('commission', sa.Float(), default=0.0),
    sa.Column('executed_at', sa.DateTime(), nullable=False, index=True),
    sa.Column('exchange_trade_id', sa.String(100), nullable=True),
    sa.Column('metadata_json', postgresql.JSONB(), nullable=True),
  )

  # Индекс для trades
  op.create_index('idx_trades_symbol_executed', 'trades', ['symbol', 'executed_at'])

  # Таблица Audit Logs
  op.create_table(
    'audit_logs',
    sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column('action',
              postgresql.ENUM('order_place', 'order_cancel', 'order_modify', 'position_open', 'position_close',
                              'position_modify', 'balance_update', 'config_change', name='auditaction'), nullable=False,
              index=True),
    sa.Column('entity_type', sa.String(50), nullable=False),
    sa.Column('entity_id', sa.String(100), nullable=False),
    sa.Column('old_value', postgresql.JSONB(), nullable=True),
    sa.Column('new_value', postgresql.JSONB(), nullable=True),
    sa.Column('changes', postgresql.JSONB(), nullable=True),
    sa.Column('reason', sa.Text(), nullable=True),
    sa.Column('user_id', sa.String(100), nullable=True),
    sa.Column('trace_id', sa.String(100), nullable=True, index=True),
    sa.Column('success', sa.Boolean(), nullable=False),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=False, index=True),
    sa.Column('context', postgresql.JSONB(), nullable=True),
  )

  # Индексы для audit_logs
  op.create_index('idx_audit_action_timestamp', 'audit_logs', ['action', 'timestamp'])
  op.create_index('idx_audit_entity', 'audit_logs', ['entity_type', 'entity_id'])

  # Таблица Idempotency Cache
  op.create_table(
    'idempotency_cache',
    sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column('idempotency_key', sa.String(200), unique=True, nullable=False, index=True),
    sa.Column('operation', sa.String(100), nullable=False),
    sa.Column('result', postgresql.JSONB(), nullable=True),
    sa.Column('success', sa.Boolean(), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False, index=True),
    sa.Column('expires_at', sa.DateTime(), nullable=False, index=True),
  )

  # Индекс для idempotency_cache
  op.create_index('idx_idempotency_key_created', 'idempotency_cache', ['idempotency_key', 'created_at'])

  # Таблица Market Data Snapshots (TimescaleDB hypertable)
  op.create_table(
    'market_data_snapshots',
    sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column('symbol', sa.String(20), nullable=False, index=True),
    sa.Column('timestamp', sa.DateTime(), nullable=False, index=True),
    sa.Column('best_bid', sa.Float(), nullable=True),
    sa.Column('best_ask', sa.Float(), nullable=True),
    sa.Column('mid_price', sa.Float(), nullable=True),
    sa.Column('bid_volume', sa.Float(), nullable=True),
    sa.Column('ask_volume', sa.Float(), nullable=True),
    sa.Column('imbalance', sa.Float(), nullable=True),
    sa.Column('spread', sa.Float(), nullable=True),
    sa.Column('orderbook', postgresql.JSONB(), nullable=True),
    sa.Column('metrics', postgresql.JSONB(), nullable=True),
  )

  # Индекс для market_data_snapshots
  op.create_index('idx_market_symbol_timestamp', 'market_data_snapshots', ['symbol', 'timestamp'])

  # Преобразуем market_data_snapshots в TimescaleDB hypertable
  op.execute("""
        SELECT create_hypertable(
            'market_data_snapshots',
            'timestamp',
            if_not_exists => TRUE,
            migrate_data => TRUE
        );
    """)

  # Устанавливаем retention policy (автоудаление старых данных через 30 дней)
  op.execute("""
        SELECT add_retention_policy(
            'market_data_snapshots',
            INTERVAL '30 days',
            if_not_exists => TRUE
        );
    """)


def downgrade() -> None:
  """Откат миграции."""

  # Удаляем таблицы
  op.drop_table('market_data_snapshots')
  op.drop_table('idempotency_cache')
  op.drop_table('audit_logs')
  op.drop_table('trades')
  op.drop_table('orders')
  op.drop_table('positions')

  # Удаляем ENUM типы
  op.execute("DROP TYPE IF EXISTS auditaction CASCADE")
  op.execute("DROP TYPE IF EXISTS positionstatus CASCADE")
  op.execute("DROP TYPE IF EXISTS orderstatus CASCADE")
  op.execute("DROP TYPE IF EXISTS ordertype CASCADE")
  op.execute("DROP TYPE IF EXISTS orderside CASCADE")