"""
Скрипт инициализации базы данных.
Создание таблиц и запуск миграций.
"""

import asyncio
import sys
from pathlib import Path

# Добавляем backend в путь
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# ВАЖНО: Импортируем модели ДО инициализации БД
# Это необходимо, чтобы SQLAlchemy знал о всех таблицах
from backend.database.models import (
    Order,
    Position,
    Trade,
    AuditLog,
    IdempotencyCache,
    MarketDataSnapshot
)

from backend.database.connection import db_manager
from backend.core.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def init_database():
    """Инициализация базы данных."""
    logger.info("=" * 80)
    logger.info("ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ")
    logger.info("=" * 80)

    try:
        # Инициализируем подключение
        await db_manager.initialize()
        logger.info("✓ Подключение к БД установлено")

        # Создаем таблицы
        await db_manager.create_tables()
        logger.info("✓ Таблицы созданы")

        # Проверяем созданные таблицы
        async with db_manager.session() as session:
            # Проверяем наличие таблиц
            result = await session.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                ORDER BY tablename
            """)
            tables = result.fetchall()

            if tables:
                logger.info("Созданные таблицы:")
                for table in tables:
                    logger.info(f"  - {table[0]}")
            else:
                logger.warning("⚠ Таблицы не найдены в базе данных!")

        logger.info("=" * 80)
        logger.info("БАЗА ДАННЫХ УСПЕШНО ИНИЦИАЛИЗИРОВАНА")
        logger.info("=" * 80)

        # Для запуска Alembic миграций используйте:
        logger.info("")
        logger.info("Для управления миграциями используйте Alembic:")
        logger.info("  alembic upgrade head     # Применить все миграции")
        logger.info("  alembic downgrade -1     # Откатить последнюю миграцию")
        logger.info("  alembic current          # Текущая версия")
        logger.info("  alembic history          # История миграций")
        logger.info("")

    except Exception as e:
        logger.error(f"Ошибка инициализации БД: {e}", exc_info=True)
        sys.exit(1)

    finally:
        await db_manager.close()


async def check_connection():
    """Проверка подключения к БД."""
    logger.info("Проверка подключения к базе данных...")

    try:
        await db_manager.initialize()

        async with db_manager.session() as session:
            # Версия PostgreSQL
            result = await session.execute("SELECT version()")
            version = result.scalar()
            logger.info(f"✓ PostgreSQL версия: {version}")

            # Проверяем TimescaleDB
            result = await session.execute(
                "SELECT extname, extversion FROM pg_extension WHERE extname = 'timescaledb'"
            )
            timescale = result.fetchone()
            if timescale:
                logger.info(f"✓ TimescaleDB версия: {timescale[1]}")
            else:
                logger.warning("⚠ TimescaleDB не установлен")

            # Проверяем существующие таблицы
            result = await session.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename
            """)
            tables = result.fetchall()

            if tables:
                logger.info(f"✓ Найдено таблиц: {len(tables)}")
                for table in tables:
                    logger.info(f"  - {table[0]}")
            else:
                logger.info("⚠ Таблицы не найдены (БД пуста)")

        logger.info("✓ Подключение работает корректно")

    except Exception as e:
        logger.error(f"Ошибка подключения: {e}", exc_info=True)
        sys.exit(1)

    finally:
        await db_manager.close()


async def reset_database():
    """ОПАСНО: Полная очистка базы данных."""
    logger.warning("=" * 80)
    logger.warning("ВНИМАНИЕ: УДАЛЕНИЕ ВСЕХ ДАННЫХ!")
    logger.warning("=" * 80)

    confirmation = input("Введите 'YES' для подтверждения удаления: ")
    if confirmation != "YES":
        logger.info("Операция отменена")
        return

    try:
        # ВАЖНО: Импортируем модели
        from backend.database.models import (
            Order,
            Position,
            Trade,
            AuditLog,
            IdempotencyCache,
            MarketDataSnapshot
        )

        await db_manager.initialize()

        async with db_manager.session() as session:
            # Удаляем все таблицы
            await session.execute("DROP SCHEMA public CASCADE")
            await session.execute("CREATE SCHEMA public")
            await session.execute("GRANT ALL ON SCHEMA public TO public")

            logger.info("✓ Все таблицы удалены")

        # Создаем заново
        await db_manager.create_tables()
        logger.info("✓ Таблицы созданы заново")

        logger.info("=" * 80)
        logger.info("БАЗА ДАННЫХ ОЧИЩЕНА И ПЕРЕСОЗДАНА")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Ошибка при reset: {e}", exc_info=True)
        sys.exit(1)

    finally:
        await db_manager.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Управление базой данных")
    parser.add_argument(
        "command",
        choices=["init", "check", "reset"],
        help="Команда для выполнения",
    )

    args = parser.parse_args()

    # Настройка event loop для Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if args.command == "init":
        asyncio.run(init_database())
    elif args.command == "check":
        asyncio.run(check_connection())
    elif args.command == "reset":
        asyncio.run(reset_database())