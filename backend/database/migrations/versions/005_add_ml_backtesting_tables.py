"""
Добавление таблиц для ML бэктестинга.

Создает таблицы ml_backtest_runs и ml_backtest_predictions
для хранения результатов бэктестов ML моделей.

Revision ID: 005_add_ml_backtesting_tables
Create Date: 2025-12-02
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, ENUM
import uuid

# revision identifiers
revision = '005_add_ml_backtesting_tables'
down_revision = '004_add_backtesting_tables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Создание таблиц для ML бэктестинга."""

    # ========== Создание enum для статусов ML бэктеста ==========
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE mlbackteststatus AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    ml_backtest_status_enum = ENUM('pending', 'running', 'completed', 'failed', 'cancelled',
                                    name='mlbackteststatus', create_type=False)

    # ========== Таблица ml_backtest_runs ==========
    op.create_table(
        'ml_backtest_runs',

        # Идентификаторы
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text, nullable=True),

        # Модель
        sa.Column('model_checkpoint', sa.String(500), nullable=False),
        sa.Column('model_version', sa.String(100), nullable=True),
        sa.Column('model_architecture', sa.String(100), nullable=True),
        sa.Column('model_config', JSONB, nullable=True),

        # Данные
        sa.Column('data_source', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=True, index=True),
        sa.Column('start_date', sa.DateTime, nullable=True, index=True),
        sa.Column('end_date', sa.DateTime, nullable=True, index=True),
        sa.Column('holdout_set_id', sa.String(100), nullable=True),

        # Walk-forward настройки
        sa.Column('use_walk_forward', sa.Boolean, default=True),
        sa.Column('n_periods', sa.Integer, default=5),
        sa.Column('retrain_each_period', sa.Boolean, default=False),

        # Trading simulation настройки
        sa.Column('initial_capital', sa.Float, default=10000.0),
        sa.Column('position_size', sa.Float, default=0.1),
        sa.Column('commission', sa.Float, default=0.001),
        sa.Column('slippage', sa.Float, default=0.0005),

        # Confidence filtering
        sa.Column('use_confidence_filter', sa.Boolean, default=True),
        sa.Column('min_confidence', sa.Float, default=0.6),
        sa.Column('confidence_mode', sa.String(50), default='threshold'),

        # Inference настройки
        sa.Column('sequence_length', sa.Integer, default=60),
        sa.Column('batch_size', sa.Integer, default=128),
        sa.Column('device', sa.String(20), default='auto'),

        # Статус
        sa.Column('status', ml_backtest_status_enum,
                  nullable=False, server_default='pending', index=True),
        sa.Column('progress_pct', sa.Float, default=0.0),

        # === Classification Results ===
        sa.Column('total_samples', sa.Integer, nullable=True),
        sa.Column('accuracy', sa.Float, nullable=True),

        # Per-class metrics (JSON)
        sa.Column('precision_per_class', JSONB, nullable=True),
        sa.Column('recall_per_class', JSONB, nullable=True),
        sa.Column('f1_per_class', JSONB, nullable=True),
        sa.Column('support_per_class', JSONB, nullable=True),

        # Confusion matrix
        sa.Column('confusion_matrix', JSONB, nullable=True),

        # Macro/weighted averages
        sa.Column('precision_macro', sa.Float, nullable=True),
        sa.Column('recall_macro', sa.Float, nullable=True),
        sa.Column('f1_macro', sa.Float, nullable=True),

        # === Trading Results ===
        sa.Column('total_trades', sa.Integer, default=0),
        sa.Column('winning_trades', sa.Integer, default=0),
        sa.Column('losing_trades', sa.Integer, default=0),
        sa.Column('win_rate', sa.Float, nullable=True),
        sa.Column('total_pnl', sa.Float, nullable=True),
        sa.Column('total_pnl_percent', sa.Float, nullable=True),
        sa.Column('max_drawdown', sa.Float, nullable=True),
        sa.Column('sharpe_ratio', sa.Float, nullable=True),
        sa.Column('profit_factor', sa.Float, nullable=True),
        sa.Column('final_capital', sa.Float, nullable=True),

        # === Walk-Forward Results ===
        sa.Column('period_results', JSONB, nullable=True),

        # === Advanced Metrics ===
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

    # Индексы для ml_backtest_runs
    op.create_index('idx_ml_backtest_runs_status', 'ml_backtest_runs', ['status'])
    op.create_index('idx_ml_backtest_runs_created_at', 'ml_backtest_runs', ['created_at'])
    op.create_index('idx_ml_backtest_runs_model', 'ml_backtest_runs', ['model_checkpoint'])

    # ========== Таблица ml_backtest_predictions ==========
    op.create_table(
        'ml_backtest_predictions',

        # Идентификаторы
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('backtest_run_id', UUID(as_uuid=True),
                  sa.ForeignKey('ml_backtest_runs.id', ondelete='CASCADE'),
                  nullable=False, index=True),

        # Временная метка
        sa.Column('timestamp', sa.DateTime, nullable=False, index=True),
        sa.Column('sequence', sa.Integer, nullable=False),

        # Предсказание
        sa.Column('predicted_class', sa.Integer, nullable=False),
        sa.Column('actual_class', sa.Integer, nullable=False),
        sa.Column('confidence', sa.Float, nullable=False),

        # Probabilities per class
        sa.Column('prob_sell', sa.Float, nullable=True),
        sa.Column('prob_hold', sa.Float, nullable=True),
        sa.Column('prob_buy', sa.Float, nullable=True),

        # Трейд на этом предсказании
        sa.Column('trade_executed', sa.Boolean, default=False),
        sa.Column('trade_pnl', sa.Float, nullable=True),

        # Period для walk-forward
        sa.Column('period', sa.Integer, nullable=True)
    )

    # Индексы для ml_backtest_predictions
    op.create_index('idx_ml_backtest_predictions_run_seq', 'ml_backtest_predictions',
                    ['backtest_run_id', 'sequence'])
    op.create_index('idx_ml_backtest_predictions_period', 'ml_backtest_predictions',
                    ['backtest_run_id', 'period'])


def downgrade() -> None:
    """Удаление таблиц ML бэктестинга."""

    # Drop indices first
    op.drop_index('idx_ml_backtest_predictions_period', table_name='ml_backtest_predictions')
    op.drop_index('idx_ml_backtest_predictions_run_seq', table_name='ml_backtest_predictions')
    op.drop_index('idx_ml_backtest_runs_model', table_name='ml_backtest_runs')
    op.drop_index('idx_ml_backtest_runs_created_at', table_name='ml_backtest_runs')
    op.drop_index('idx_ml_backtest_runs_status', table_name='ml_backtest_runs')

    # Drop tables
    op.drop_table('ml_backtest_predictions')
    op.drop_table('ml_backtest_runs')

    # Drop enum type
    op.execute("DROP TYPE IF EXISTS mlbackteststatus")
