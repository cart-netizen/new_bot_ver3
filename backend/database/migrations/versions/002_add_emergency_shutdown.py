"""
Добавление новых типов в AuditAction enum.

Revision ID: 002_add_emergency_shutdown
Create Date: 2025-01-XX
"""

from alembic import op

# revision identifiers
revision = '002_add_emergency_shutdown'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Добавление новых значений в AuditAction enum."""

    # ✅ ИСПРАВЛЕНО: Разделено на отдельные команды для asyncpg
    # asyncpg не поддерживает несколько команд в одном prepared statement
    op.execute("ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'system'")
    op.execute("ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'emergency_shutdown'")


def downgrade() -> None:
    """
    Откат миграции.

    ВНИМАНИЕ: Удаление значений из enum в PostgreSQL требует пересоздания типа.
    Рекомендуется не использовать downgrade для production.
    """
    # В PostgreSQL нельзя просто удалить значение из enum
    # Требуется пересоздание типа, что сложно и опасно
    # Для production лучше не делать downgrade
    pass