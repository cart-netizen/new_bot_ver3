"""
Модуль подключения к базе данных.
Асинхронное подключение к PostgreSQL + TimescaleDB.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
  create_async_engine,
  AsyncSession,
  async_sessionmaker,
  AsyncEngine
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from contextlib import asynccontextmanager

from config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# Base для моделей
Base = declarative_base()


class DatabaseManager:
  """Менеджер подключения к базе данных."""

  def __init__(self):
    """Инициализация менеджера БД."""
    self.engine: AsyncEngine | None = None
    self.session_factory: async_sessionmaker | None = None
    self._is_initialized = False

  async def initialize(self):
    """Инициализация подключения к БД."""
    if self._is_initialized:
      logger.warning("Database уже инициализирована")
      return

    try:
      logger.info("Инициализация подключения к PostgreSQL...")

      # Создаем движок
      self.engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        poolclass=NullPool if settings.DEBUG else None,
      )

      # Создаем фабрику сессий
      self.session_factory = async_sessionmaker(
        self.engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
      )

      # Проверяем подключение
      async with self.engine.begin() as conn:
        await conn.execute("SELECT 1")

      logger.info("✓ База данных подключена успешно")
      self._is_initialized = True

    except Exception as e:
      logger.error(f"Ошибка подключения к БД: {e}")
      raise

  async def create_tables(self):
    """Создание таблиц в БД."""
    if not self._is_initialized:
      raise RuntimeError("Database не инициализирована")

    try:
      logger.info("Создание таблиц в БД...")

      async with self.engine.begin() as conn:
        # Создаем расширение TimescaleDB
        await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

        # Создаем все таблицы
        await conn.run_sync(Base.metadata.create_all)

      logger.info("✓ Таблицы созданы успешно")

    except Exception as e:
      logger.error(f"Ошибка создания таблиц: {e}")
      raise

  async def close(self):
    """Закрытие подключения к БД."""
    if self.engine:
      logger.info("Закрытие подключения к БД...")
      await self.engine.dispose()
      self._is_initialized = False
      logger.info("✓ БД отключена")

  @asynccontextmanager
  async def session(self) -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager для работы с сессией.

    Yields:
        AsyncSession: Сессия БД
    """
    if not self._is_initialized:
      raise RuntimeError("Database не инициализирована")

    session = self.session_factory()
    try:
      yield session
      await session.commit()
    except Exception:
      await session.rollback()
      raise
    finally:
      await session.close()


# Глобальный экземпляр
db_manager = DatabaseManager()