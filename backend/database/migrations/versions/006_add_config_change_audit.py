"""
Добавление config_change в AuditAction enum.

Исправление для случаев, когда база была создана без этого значения.

Revision ID: 006_add_config_change_audit
Create Date: 2026-01-08
"""

from alembic import op

# revision identifiers
revision = '006_add_config_change_audit'
down_revision = '005_add_ml_backtesting_tables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Добавление config_change в AuditAction enum если отсутствует."""
    op.execute("ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'config_change'")


def downgrade() -> None:
    """
    Откат миграции.

    В PostgreSQL нельзя удалить значение из enum без пересоздания типа.
    """
    pass
