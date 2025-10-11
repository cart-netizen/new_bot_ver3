"""
Alembic environment configuration.
Настройка миграций для асинхронной работы с PostgreSQL.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Импортируем модели и конфигурацию
import sys
from pathlib import Path

# Добавляем backend в путь
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from database.connection import Base
from database.models import *  # Импортируем все модели
from config import settings

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

  connectable = async_engine_from_config(
    configuration,
    prefix="sqlalchemy.",
    poolclass=pool.NullPool,
  )

  async with connectable.connect() as connection:
    await connection.run_sync(do_run_migrations)

  await connectable.dispose()


def run_migrations_online() -> None:
  """
  Запуск миграций в 'online' режиме.

  Подключается к БД и выполняет миграции.
  """
  asyncio.run(run_async_migrations())


# Определяем режим работы
if context.is_offline_mode():
  run_migrations_offline()
else:
  run_migrations_online()