"""
Добавление таблицы layering_patterns для ML detection.

Revision ID: 003_add_layering_patterns
Create Date: 2025-11-01
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

# revision identifiers
revision = '003_add_layering_patterns'
down_revision = '002_add_emergency_shutdown'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Создание таблицы layering_patterns."""

    op.create_table(
        'layering_patterns',
        # Identifiers
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('pattern_id', sa.String(64), unique=True, nullable=False, index=True),

        # Timestamps
        sa.Column('first_seen', sa.DateTime, nullable=False, index=True),
        sa.Column('last_seen', sa.DateTime, nullable=False, index=True),
        sa.Column('occurrence_count', sa.Integer, default=1, nullable=False),

        # Fingerprint features (behavioral signature)
        sa.Column('avg_layer_count', sa.Float, nullable=False),
        sa.Column('avg_cancellation_rate', sa.Float, nullable=False),
        sa.Column('avg_volume_btc', sa.Float, nullable=False),
        sa.Column('avg_placement_duration', sa.Float, nullable=False),
        sa.Column('typical_spread_pct', sa.Float, nullable=False),
        sa.Column('typical_order_count', sa.Integer, nullable=False),
        sa.Column('spoofing_execution_ratio', sa.Float, nullable=True),

        # Temporal patterns (JSONB for array storage)
        sa.Column('time_of_day_pattern', JSONB, nullable=True),  # [0-23] active hours
        sa.Column('avg_lifetime_seconds', sa.Float, nullable=False),

        # Computed hash for fast matching
        sa.Column('fingerprint_hash', sa.String(64), nullable=False, index=True),

        # Metadata
        sa.Column('symbols', JSONB, nullable=False),  # List of symbols
        sa.Column('success_rate', sa.Float, default=0.0, nullable=False),
        sa.Column('avg_price_impact_bps', sa.Float, default=0.0, nullable=False),
        sa.Column('avg_confidence', sa.Float, nullable=False),

        # Risk assessment
        sa.Column('risk_level', sa.String(20), nullable=False),
        sa.Column('blacklist', sa.Boolean, default=False, nullable=False, index=True),

        # Notes
        sa.Column('notes', sa.Text, nullable=True),

        # Audit
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.text('now()'), onupdate=sa.text('now()'))
    )

    # Create indices
    op.create_index('idx_layering_fingerprint_hash', 'layering_patterns', ['fingerprint_hash'])
    op.create_index('idx_layering_blacklist', 'layering_patterns', ['blacklist'])
    op.create_index('idx_layering_last_seen', 'layering_patterns', ['last_seen'])
    op.create_index('idx_layering_occurrence_count', 'layering_patterns', ['occurrence_count'])


def downgrade() -> None:
    """Удаление таблицы layering_patterns."""

    # Drop indices first
    op.drop_index('idx_layering_occurrence_count', table_name='layering_patterns')
    op.drop_index('idx_layering_last_seen', table_name='layering_patterns')
    op.drop_index('idx_layering_blacklist', table_name='layering_patterns')
    op.drop_index('idx_layering_fingerprint_hash', table_name='layering_patterns')

    # Drop table
    op.drop_table('layering_patterns')
