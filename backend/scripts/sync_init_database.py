"""
Синхронный скрипт инициализации базы данных.
Альтернативный способ создания таблиц через синхронное подключение.
"""

import sys
from pathlib import Path

# Добавляем backend в путь
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import psycopg2
from psycopg2 import sql

# Импортируем модели ПЕРЕД использованием Base
from backend.database.models import (
  Order,
  Position,
  Trade,
  AuditLog,
  IdempotencyCache,
  MarketDataSnapshot
)
from backend.database.connection import Base
from backend.core.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def get_sync_database_url():
  """Получаем синхронный URL для подключения к БД."""
  # Преобразуем асинхронный URL в синхронный
  from backend.config import settings
  url = settings.DATABASE_URL
  # Заменяем postgresql+asyncpg на postgresql+psycopg2
  return url.replace("postgresql+asyncpg", "postgresql+psycopg2")


def create_database_if_not_exists():
  """Создаем базу данных, если она не существует."""
  from backend.config import settings

  # Парсим DATABASE_URL для получения параметров
  import re
  pattern = r'postgresql\+\w+://(\w+):([^@]+)@([^:]+):(\d+)/(\w+)'
  match = re.match(pattern, settings.DATABASE_URL)

  if not match:
    logger.error("Не удалось распарсить DATABASE_URL")
    return False

  user, password, host, port, dbname = match.groups()

  # Подключаемся к PostgreSQL (к базе postgres)
  try:
    conn = psycopg2.connect(
      host=host,
      port=port,
      user=user,
      password=password,
      database='postgres'
    )
    conn.autocommit = True
    cursor = conn.cursor()

    # Проверяем существование БД
    cursor.execute(
      "SELECT 1 FROM pg_database WHERE datname = %s",
      (dbname,)
    )

    if cursor.fetchone() is None:
      # Создаем БД
      cursor.execute(sql.SQL("CREATE DATABASE {}").format(
        sql.Identifier(dbname)
      ))
      logger.info(f"✓ База данных '{dbname}' создана")
    else:
      logger.info(f"✓ База данных '{dbname}' уже существует")

    cursor.close()
    conn.close()
    return True

  except Exception as e:
    logger.error(f"Ошибка при создании БД: {e}")
    return False


def init_database():
  """Инициализация базы данных через синхронное подключение."""
  logger.info("=" * 80)
  logger.info("СИНХРОННАЯ ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ")
  logger.info("=" * 80)

  # Создаем БД если не существует
  if not create_database_if_not_exists():
    sys.exit(1)

  try:
    # Создаем синхронный engine
    sync_url = get_sync_database_url()
    engine = create_engine(
      sync_url,
      echo=True,
      pool_pre_ping=True
    )

    # Проверяем подключение
    with engine.connect() as conn:
      result = conn.execute(text("SELECT 1"))
      logger.info("✓ Подключение к БД установлено")

      # Создаем расширение TimescaleDB
      try:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
        conn.commit()
        logger.info("✓ TimescaleDB расширение установлено")
      except Exception as e:
        logger.warning(f"TimescaleDB: {e}")

    # Создаем все таблицы
    Base.metadata.create_all(bind=engine)
    logger.info("✓ Таблицы созданы")

    # Проверяем созданные таблицы
    with engine.connect() as conn:
      result = conn.execute(text("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                ORDER BY tablename
            """))
      tables = result.fetchall()

      if tables:
        logger.info("Созданные таблицы:")
        for table in tables:
          logger.info(f"  - {table[0]}")
      else:
        logger.error("⚠ Таблицы не были созданы!")

    logger.info("=" * 80)
    logger.info("БАЗА ДАННЫХ УСПЕШНО ИНИЦИАЛИЗИРОВАНА")
    logger.info("=" * 80)

    # Создаем таблицу миграций Alembic
    with engine.connect() as conn:
      conn.execute(text("""
                CREATE TABLE IF NOT EXISTS alembic_version (
                    version_num VARCHAR(32) NOT NULL,
                    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
                )
            """))
      conn.commit()
      logger.info("✓ Таблица alembic_version создана")

    engine.dispose()

  except Exception as e:
    logger.error(f"Ошибка инициализации: {e}", exc_info=True)
    sys.exit(1)


def drop_all_tables():
  """Удаление всех таблиц."""
  logger.warning("=" * 80)
  logger.warning("УДАЛЕНИЕ ВСЕХ ТАБЛИЦ")
  logger.warning("=" * 80)

  confirmation = input("Введите 'YES' для подтверждения: ")
  if confirmation != "YES":
    logger.info("Операция отменена")
    return

  try:
    sync_url = get_sync_database_url()
    engine = create_engine(sync_url)

    # Удаляем все таблицы
    Base.metadata.drop_all(bind=engine)
    logger.info("✓ Все таблицы удалены")

    # Удаляем таблицу миграций
    with engine.connect() as conn:
      conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
      conn.commit()
      logger.info("✓ Таблица alembic_version удалена")

    engine.dispose()

  except Exception as e:
    logger.error(f"Ошибка: {e}", exc_info=True)


def check_database():
  """Проверка состояния БД."""
  logger.info("Проверка базы данных...")

  try:
    sync_url = get_sync_database_url()
    engine = create_engine(sync_url)

    with engine.connect() as conn:
      # Версия PostgreSQL
      result = conn.execute(text("SELECT version()"))
      version = result.scalar()
      logger.info(f"PostgreSQL: {version[:50]}...")

      # TimescaleDB
      result = conn.execute(text("""
                SELECT extname, extversion 
                FROM pg_extension 
                WHERE extname = 'timescaledb'
            """))
      ts = result.fetchone()
      if ts:
        logger.info(f"TimescaleDB: v{ts[1]}")

      # Таблицы
      result = conn.execute(text("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                ORDER BY tablename
            """))
      tables = result.fetchall()

      if tables:
        logger.info(f"Найдено таблиц: {len(tables)}")
        for table in tables:
          # Считаем записи
          count_result = conn.execute(
            text(f"SELECT COUNT(*) FROM {table[0]}")
          )
          count = count_result.scalar()
          logger.info(f"  - {table[0]} ({count} записей)")
      else:
        logger.warning("Таблицы не найдены!")

      # Проверяем миграции Alembic
      result = conn.execute(text("""
                SELECT version_num 
                FROM alembic_version
            """))
      migration = result.fetchone()
      if migration:
        logger.info(f"Текущая миграция: {migration[0]}")
      else:
        logger.info("Миграции не применены")

    engine.dispose()
    logger.info("✓ Проверка завершена")

  except Exception as e:
    logger.error(f"Ошибка: {e}", exc_info=True)


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Синхронное управление БД")
  parser.add_argument(
    "command",
    choices=["init", "check", "drop"],
    help="Команда для выполнения"
  )

  args = parser.parse_args()

  if args.command == "init":
    init_database()
  elif args.command == "check":
    check_database()
  elif args.command == "drop":
    drop_all_tables()