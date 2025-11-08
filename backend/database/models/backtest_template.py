"""Database model для шаблонов бэктеста"""

from sqlalchemy import Column, String, DateTime, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

from backend.database.connection import Base


# from backend.database.database import Base


class BacktestTemplate(Base):
    """Шаблон конфигурации бэктеста"""

    __tablename__ = "backtest_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # JSON конфигурация
    config = Column(JSON, nullable=False)

    # Теги для категоризации
    tags = Column(JSON, nullable=True)  # ["scalping", "swing", "conservative"]

    # Метаданные
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # User ID (опционально, для multi-user support)
    user_id = Column(String(100), nullable=True, index=True)

    # Публичный шаблон или приватный
    is_public = Column(String(10), default="private", nullable=False)  # "public" | "private"

    # Использование
    usage_count = Column(String(10), default="0", nullable=False)

    def __repr__(self):
        return f"<BacktestTemplate(id={self.id}, name='{self.name}')>"
