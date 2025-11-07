"""
Добавление таблиц для системы бэктестинга.

Revision ID: 004_add_backtesting_tables
Create Date: 2025-01-15
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

# revision identifiers
revision = '004_add_backtesting_tables'
down_revision = '003_add_layering_patterns'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Создание таблиц для бэктестинга."""

    # ========== Создание enum для статусов бэктеста ==========
    # Используем DO блок с обработкой исключения для идемпотентности
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE backteststatus AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # ========== Таблица backtest_runs ==========
    op.create_table(
        'backtest_runs',
        # Идентификаторы
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text, nullable=True),

        # Параметры бэктеста
        sa.Column('symbol', sa.String(20), nullable=False, index=True),
        sa.Column('start_date', sa.DateTime, nullable=False, index=True),
        sa.Column('end_date', sa.DateTime, nullable=False, index=True),
        sa.Column('initial_capital', sa.Float, nullable=False),

        # Конфигурации (JSONB)
        sa.Column('strategies_config', JSONB, nullable=False),
        sa.Column('risk_config', JSONB, nullable=False),
        sa.Column('exchange_config', JSONB, nullable=True),

        # Статус выполнения
        sa.Column('status', sa.Enum('pending', 'running', 'completed', 'failed', 'cancelled',
                                     name='backteststatus', create_type=False),
                  nullable=False, server_default='pending', index=True),
        sa.Column('progress_pct', sa.Float, default=0.0),
        sa.Column('current_date', sa.DateTime, nullable=True),

        # Результаты - базовые
        sa.Column('total_trades', sa.Integer, default=0),
        sa.Column('winning_trades', sa.Integer, default=0),
        sa.Column('losing_trades', sa.Integer, default=0),
        sa.Column('final_capital', sa.Float, nullable=True),
        sa.Column('total_pnl', sa.Float, nullable=True),
        sa.Column('total_pnl_pct', sa.Float, nullable=True),

        # Результаты - детальные метрики
        sa.Column('metrics', JSONB, nullable=True),

        # Временные метки
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('now()'), index=True),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
        sa.Column('duration_seconds', sa.Float, nullable=True),

        # Ошибки
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('error_traceback', sa.Text, nullable=True),

        # Метаданные
        sa.Column('metadata_json', JSONB, nullable=True)
    )

    # Индексы для backtest_runs
    op.create_index('idx_backtest_runs_symbol_status', 'backtest_runs', ['symbol', 'status'])
    op.create_index('idx_backtest_runs_created_at', 'backtest_runs', ['created_at'])
    op.create_index('idx_backtest_runs_start_end', 'backtest_runs', ['start_date', 'end_date'])

    # ========== Таблица backtest_trades ==========
    op.create_table(
        'backtest_trades',
        # Идентификаторы
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('backtest_run_id', UUID(as_uuid=True),
                  sa.ForeignKey('backtest_runs.id', ondelete='CASCADE'),
                  nullable=False, index=True),

        # Основные данные трейда
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', sa.Enum('Buy', 'Sell', name='orderside'), nullable=False),

        # Время входа и выхода
        sa.Column('entry_time', sa.DateTime, nullable=False, index=True),
        sa.Column('exit_time', sa.DateTime, nullable=True),

        # Цены
        sa.Column('entry_price', sa.Float, nullable=False),
        sa.Column('exit_price', sa.Float, nullable=True),

        # Объем
        sa.Column('quantity', sa.Float, nullable=False),

        # Stop Loss / Take Profit
        sa.Column('stop_loss', sa.Float, nullable=True),
        sa.Column('take_profit', sa.Float, nullable=True),

        # PnL
        sa.Column('pnl', sa.Float, nullable=True),
        sa.Column('pnl_pct', sa.Float, nullable=True),
        sa.Column('commission', sa.Float, default=0.0),

        # Контекст входа/выхода
        sa.Column('entry_signal', JSONB, nullable=True),
        sa.Column('exit_reason', sa.String(100), nullable=True),
        sa.Column('exit_signal', JSONB, nullable=True),

        # Метрики трейда
        sa.Column('max_favorable_excursion', sa.Float, nullable=True),
        sa.Column('max_adverse_excursion', sa.Float, nullable=True),
        sa.Column('duration_seconds', sa.Float, nullable=True),

        # Рыночные условия
        sa.Column('entry_market_data', JSONB, nullable=True),
        sa.Column('exit_market_data', JSONB, nullable=True),

        # Метаданные
        sa.Column('metadata_json', JSONB, nullable=True)
    )

    # Индексы для backtest_trades
    op.create_index('idx_backtest_trades_run_time', 'backtest_trades',
                    ['backtest_run_id', 'entry_time'])
    op.create_index('idx_backtest_trades_symbol', 'backtest_trades', ['symbol'])

    # ========== Таблица backtest_equity ==========
    op.create_table(
        'backtest_equity',
        # Идентификаторы
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('backtest_run_id', UUID(as_uuid=True),
                  sa.ForeignKey('backtest_runs.id', ondelete='CASCADE'),
                  nullable=False, index=True),

        # Временная метка
        sa.Column('timestamp', sa.DateTime, nullable=False, index=True),
        sa.Column('sequence', sa.Integer, nullable=False),

        # Состояние капитала
        sa.Column('equity', sa.Float, nullable=False),
        sa.Column('cash', sa.Float, nullable=False),
        sa.Column('positions_value', sa.Float, default=0.0),

        # Просадка
        sa.Column('peak_equity', sa.Float, nullable=False),
        sa.Column('drawdown', sa.Float, default=0.0),
        sa.Column('drawdown_pct', sa.Float, default=0.0),

        # Доходность
        sa.Column('total_return', sa.Float, default=0.0),
        sa.Column('total_return_pct', sa.Float, default=0.0),

        # Количество открытых позиций
        sa.Column('open_positions_count', sa.Integer, default=0)
    )

    # Индексы для backtest_equity
    op.create_index('idx_backtest_equity_run_time', 'backtest_equity',
                    ['backtest_run_id', 'timestamp'])
    op.create_index('idx_backtest_equity_sequence', 'backtest_equity',
                    ['backtest_run_id', 'sequence'])

    # ========== Конвертируем таблицу backtest_equity в TimescaleDB hypertable ==========
    # Это позволит эффективно хранить временные ряды equity curve
    try:
        op.execute("""
            SELECT create_hypertable('backtest_equity', 'timestamp',
                                    chunk_time_interval => INTERVAL '7 days',
                                    if_not_exists => TRUE);
        """)
    except Exception as e:
        # Если TimescaleDB не установлен, просто пропускаем
        print(f"Warning: Could not create hypertable for backtest_equity: {e}")


def downgrade() -> None:
    """Удаление таблиц бэктестинга."""

    # Drop indices first
    op.drop_index('idx_backtest_equity_sequence', table_name='backtest_equity')
    op.drop_index('idx_backtest_equity_run_time', table_name='backtest_equity')
    op.drop_index('idx_backtest_trades_symbol', table_name='backtest_trades')
    op.drop_index('idx_backtest_trades_run_time', table_name='backtest_trades')
    op.drop_index('idx_backtest_runs_start_end', table_name='backtest_runs')
    op.drop_index('idx_backtest_runs_created_at', table_name='backtest_runs')
    op.drop_index('idx_backtest_runs_symbol_status', table_name='backtest_runs')

    # Drop tables (CASCADE will handle foreign keys)
    op.drop_table('backtest_equity')
    op.drop_table('backtest_trades')
    op.drop_table('backtest_runs')

    # Drop enum type
    op.execute("DROP TYPE IF EXISTS backteststatus")
