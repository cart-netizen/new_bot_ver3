"""
Добавление всех недостающих значений в AuditAction enum.

Исправление для случаев, когда база была создана без некоторых значений.

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
    """Добавление всех недостающих значений в AuditAction enum."""
    # Все значения из Python enum - добавляем IF NOT EXISTS для идемпотентности
    audit_actions = [
        'order_place',
        'order_cancel',
        'order_modify',
        'position_open',
        'position_close',
        'position_modify',
        'balance_update',
        'config_change',
        'system',
        'emergency_shutdown'
    ]

    for action in audit_actions:
        op.execute(f"ALTER TYPE auditaction ADD VALUE IF NOT EXISTS '{action}'")


def downgrade() -> None:
    """
    Откат миграции.

    В PostgreSQL нельзя удалить значение из enum без пересоздания типа.
    """
    pass
