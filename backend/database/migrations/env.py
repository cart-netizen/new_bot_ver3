"""
Alembic environment configuration.
Настройка миграций для асинхронной работы с PostgreSQL.
"""

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent.parent.parent  # new_bot_ver3/
sys.path.insert(0, str(project_root))

# ВАЖНО: Импортируем ВСЕ модели до использования Base.metadata
from backend.database.models import (
    Order,
    Position,
    Trade,
    AuditLog,
    IdempotencyCache,
    MarketDataSnapshot,
    BacktestRun,
    BacktestTrade,
    BacktestEquity,
    BacktestStatus
)
from backend.database.connection import Base
from backend.config import settings

# Alembic Config object
config = context.config

# Устанавливаем DATABASE_URL из настроек
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Интерпретация конфига для логирования
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# MetaData для автогенерации миграций
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Запуск миграций в 'offline' режиме.

    Генерирует SQL без подключения к БД.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    Выполнение миграций с подключением.

    Args:
        connection: Подключение к БД
    """
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Асинхронное выполнение миграций."""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = settings.DATABASE_URL

    # ВАЖНО: Добавляем настройки для Windows
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={
            "server_settings": {"application_name": "alembic"},
            "command_timeout": 60,
        },
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Запуск миграций в 'online' режиме.

    Подключается к БД и выполняет миграции.
    """
    # ВАЖНО: Настройка event loop для Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(run_async_migrations())
    finally:
        loop.close()


# Определяем режим работы
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()