"""
Диагностический скрипт для проверки и исправления проблем с БД.
"""

import sys
import asyncio
from pathlib import Path
import traceback

# Добавляем backend в путь
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from backend.core.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def check_imports():
  """Проверка импорта всех необходимых модулей."""
  logger.info("Проверка импортов...")

  modules = {
    "sqlalchemy": "SQLAlchemy",
    "asyncpg": "AsyncPG",
    "psycopg2": "Psycopg2",
    "alembic": "Alembic"
  }

  success = True
  for module, name in modules.items():
    try:
      __import__(module)
      logger.info(f"  ✓ {name} установлен")
    except ImportError:
      logger.error(f"  ✗ {name} НЕ установлен! Установите: pip install {module}")
      success = False

  return success


def check_database_url():
  """Проверка настроек подключения к БД."""
  logger.info("Проверка DATABASE_URL...")

  try:
    from backend.config import settings
    url = settings.DATABASE_URL
    logger.info(f"  URL: {url[:30]}...")

    # Проверяем формат URL
    import re
    pattern = r'postgresql\+(\w+)://(\w+):([^@]+)@([^:]+):(\d+)/(\w+)'
    match = re.match(pattern, url)

    if match:
      driver, user, _, host, port, dbname = match.groups()
      logger.info(f"  ✓ Драйвер: {driver}")
      logger.info(f"  ✓ Пользователь: {user}")
      logger.info(f"  ✓ Хост: {host}")
      logger.info(f"  ✓ Порт: {port}")
      logger.info(f"  ✓ База данных: {dbname}")
      return True
    else:
      logger.error("  ✗ Неверный формат DATABASE_URL!")
      return False

  except Exception as e:
    logger.error(f"  ✗ Ошибка: {e}")
    return False


async def check_async_connection():
  """Проверка асинхронного подключения."""
  logger.info("Проверка асинхронного подключения...")

  try:
    from backend.database.connection import db_manager

    await db_manager.initialize()

    async with db_manager.session() as session:
      result = await session.execute("SELECT 1")
      if result.scalar() == 1:
        logger.info("  ✓ Асинхронное подключение работает")
        await db_manager.close()
        return True
  except Exception as e:
    logger.error(f"  ✗ Ошибка асинхронного подключения: {e}")
    logger.debug(traceback.format_exc())

  return False


def check_sync_connection():
  """Проверка синхронного подключения."""
  logger.info("Проверка синхронного подключения...")

  try:
    from backend.config import settings
    import psycopg2
    import re

    # Парсим URL
    pattern = r'postgresql\+\w+://(\w+):([^@]+)@([^:]+):(\d+)/(\w+)'
    match = re.match(pattern, settings.DATABASE_URL)

    if not match:
      logger.error("  ✗ Не удалось распарсить DATABASE_URL")
      return False

    user, password, host, port, dbname = match.groups()

    # Пробуем подключиться
    conn = psycopg2.connect(
      host=host,
      port=port,
      user=user,
      password=password,
      database=dbname
    )

    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()

    if result[0] == 1:
      logger.info("  ✓ Синхронное подключение работает")

    cursor.close()
    conn.close()
    return True

  except psycopg2.OperationalError as e:
    logger.error(f"  ✗ Ошибка подключения к БД: {e}")
    logger.info("    Проверьте, что PostgreSQL запущен и доступен")
    return False
  except Exception as e:
    logger.error(f"  ✗ Ошибка: {e}")
    return False


def check_models():
  """Проверка импорта моделей."""
  logger.info("Проверка моделей...")

  try:
    from backend.database.models import (
      Order,
      Position,
      Trade,
      AuditLog,
      IdempotencyCache,
      MarketDataSnapshot
    )
    from backend.database.connection import Base

    # Проверяем, что модели зарегистрированы
    tables = Base.metadata.tables
    expected_tables = [
      'orders',
      'positions',
      'trades',
      'audit_logs',
      'idempotency_cache',
      'market_data_snapshots'
    ]

    for table in expected_tables:
      if table in tables:
        logger.info(f"  ✓ Модель {table} зарегистрирована")
      else:
        logger.error(f"  ✗ Модель {table} НЕ найдена!")

    return len(tables) >= len(expected_tables)

  except Exception as e:
    logger.error(f"  ✗ Ошибка импорта моделей: {e}")
    return False


def diagnose():
  """Полная диагностика системы БД."""
  logger.info("=" * 80)
  logger.info("ДИАГНОСТИКА СИСТЕМЫ БАЗЫ ДАННЫХ")
  logger.info("=" * 80)

  results = {
    "Импорты": check_imports(),
    "DATABASE_URL": check_database_url(),
    "Модели": check_models(),
    "Синхронное подключение": check_sync_connection(),
  }

  # Проверка асинхронного подключения
  if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)

  try:
    results["Асинхронное подключение"] = loop.run_until_complete(
      check_async_connection()
    )
  finally:
    loop.close()

  # Итоговый результат
  logger.info("=" * 80)
  logger.info("РЕЗУЛЬТАТЫ ДИАГНОСТИКИ:")
  logger.info("=" * 80)

  all_ok = True
  for name, status in results.items():
    status_str = "✓ OK" if status else "✗ ОШИБКА"
    logger.info(f"  {name}: {status_str}")
    if not status:
      all_ok = False

  logger.info("=" * 80)

  if all_ok:
    logger.info("✓ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
    logger.info("")
    logger.info("Теперь вы можете:")
    logger.info("  1. Инициализировать БД: python backend/scripts/sync_init_database.py init")
    logger.info("  2. Или использовать async версию: python backend/scripts/init_database.py init")
    logger.info("  3. Применить миграции: cd backend && alembic upgrade head")
  else:
    logger.error("✗ ОБНАРУЖЕНЫ ПРОБЛЕМЫ!")
    logger.info("")
    logger.info("Рекомендации:")
    logger.info("  1. Проверьте, что PostgreSQL запущен")
    logger.info("  2. Проверьте правильность DATABASE_URL в .env")
    logger.info("  3. Установите недостающие зависимости: pip install -r requirements.txt")
    logger.info("  4. Используйте синхронную версию инициализации: python backend/scripts/sync_init_database.py init")


if __name__ == "__main__":
  diagnose()