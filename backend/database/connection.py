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
from sqlalchemy import text
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

    async def initialize(self, max_retries: int = 3, retry_delay: int = 2):
        """
        Инициализация подключения к БД.

        Args:
            max_retries: Максимальное количество попыток подключения
            retry_delay: Задержка между попытками в секундах
        """
        if self._is_initialized:
            logger.warning("Database уже инициализирована")
            return

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Инициализация подключения к PostgreSQL (попытка {attempt}/{max_retries})...")

                # Параметры для create_async_engine
                engine_kwargs = {
                    "echo": settings.DEBUG,
                    "pool_pre_ping": True,
                }

                # Добавляем pool параметры только если НЕ используется NullPool
                if settings.DEBUG:
                    # В DEBUG режиме используем NullPool (без пула)
                    engine_kwargs["poolclass"] = NullPool
                else:
                    # В production используем пул с настройками
                    engine_kwargs["pool_size"] = 10
                    engine_kwargs["max_overflow"] = 20

                # Создаем движок
                self.engine = create_async_engine(
                    settings.DATABASE_URL,
                    **engine_kwargs
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
                    await conn.execute(text("SELECT 1"))

                logger.info("✓ База данных подключена успешно")
                self._is_initialized = True
                return

            except OSError as e:
                # Ошибки подключения (10061 - Connection refused)
                last_error = e
                if "10061" in str(e) or "Connection refused" in str(e):
                    logger.error(
                        f"❌ Не удалось подключиться к PostgreSQL (попытка {attempt}/{max_retries}): "
                        f"База данных не запущена или недоступна на {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'localhost:5432'}"
                    )
                    logger.error(
                        "РЕШЕНИЕ: Убедитесь, что PostgreSQL установлен и запущен. "
                        "Windows: проверьте службу PostgreSQL в services.msc"
                    )
                else:
                    logger.error(f"❌ Ошибка сети при подключении к БД: {e}")

                if attempt < max_retries:
                    logger.info(f"Повторная попытка через {retry_delay} секунд...")
                    import asyncio
                    await asyncio.sleep(retry_delay)

            except Exception as e:
                # Другие ошибки
                last_error = e
                logger.error(f"❌ Ошибка подключения к БД (попытка {attempt}/{max_retries}): {e}")
                logger.error(f"Тип ошибки: {type(e).__name__}")

                if attempt < max_retries:
                    logger.info(f"Повторная попытка через {retry_delay} секунд...")
                    import asyncio
                    await asyncio.sleep(retry_delay)

        # Если все попытки исчерпаны
        logger.error("=" * 80)
        logger.error("❌ НЕ УДАЛОСЬ ПОДКЛЮЧИТЬСЯ К БАЗЕ ДАННЫХ")
        logger.error("=" * 80)
        logger.error("Проверьте следующее:")
        logger.error("1. PostgreSQL установлен и запущен")
        logger.error("2. Настройки подключения в .env файле корректны")
        logger.error(f"   Текущий DATABASE_URL: {settings.DATABASE_URL}")
        logger.error("3. Пользователь и база данных созданы:")
        logger.error("   CREATE USER trading_bot WITH PASSWORD 'robocop';")
        logger.error("   CREATE DATABASE trading_bot OWNER trading_bot;")
        logger.error("=" * 80)
        raise last_error

    async def create_tables(self):
        """Создание таблиц в БД."""
        if not self._is_initialized:
            raise RuntimeError("Database не инициализирована")

        try:
            logger.info("Создание таблиц в БД...")

            async with self.engine.begin() as conn:
                # Создаем расширение TimescaleDB
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))

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