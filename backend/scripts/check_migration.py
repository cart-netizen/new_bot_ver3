"""
Скрипт для проверки статуса миграций.

Проверяет:
1. Текущую версию миграции в БД
2. Существование таблицы layering_patterns
3. Структуру таблицы
4. Индексы
"""

import asyncio
import sys
from pathlib import Path

# Добавляем backend в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.connection import db_manager
from sqlalchemy import text
from backend.core.logger import get_logger

logger = get_logger(__name__)


async def check_migration_status():
    """Проверка статуса миграций."""

    print("=" * 80)
    print("🔍 ПРОВЕРКА СТАТУСА МИГРАЦИЙ")
    print("=" * 80)

    try:
        # Инициализация БД
        print("\n1️⃣ Подключение к PostgreSQL...")
        await db_manager.initialize()
        print("   ✅ Подключение успешно")

        async with db_manager.session() as session:
            # Проверка 1: Alembic version
            print("\n2️⃣ Проверка текущей версии Alembic...")
            try:
                result = await session.execute(
                    text("SELECT version_num FROM alembic_version")
                )
                current_version = result.scalar()
                print(f"   ✅ Текущая версия: {current_version}")

                # Ожидаемая версия
                expected = "003_add_layering_patterns"
                if current_version == expected:
                    print(f"   🎉 Миграция {expected} ПРИМЕНЕНА!")
                elif current_version and int(current_version.split('_')[0]) >= 3:
                    print(f"   ✅ Миграция применена (версия >= 003)")
                else:
                    print(f"   ⚠️  Текущая версия: {current_version}, ожидается: {expected}")
                    print(f"   💡 Запустите: alembic upgrade head")
            except Exception as e:
                print(f"   ❌ Таблица alembic_version не найдена: {e}")
                print(f"   💡 Возможно миграции еще не применялись")

            # Проверка 2: Таблица layering_patterns
            print("\n3️⃣ Проверка таблицы layering_patterns...")
            result = await session.execute(text(
                """SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'layering_patterns'
                )"""
            ))
            table_exists = result.scalar()

            if table_exists:
                print("   ✅ Таблица layering_patterns существует")

                # Проверка 3: Колонки таблицы
                print("\n4️⃣ Проверка структуры таблицы...")
                result = await session.execute(text(
                    """SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'layering_patterns'
                    ORDER BY ordinal_position"""
                ))
                columns = result.fetchall()
                print(f"   ✅ Найдено {len(columns)} колонок:")

                key_columns = [
                    'id', 'pattern_id', 'fingerprint_hash',
                    'avg_layer_count', 'avg_cancellation_rate',
                    'blacklist', 'risk_level'
                ]

                for col in columns:
                    col_name = col[0]
                    if col_name in key_columns:
                        nullable = "NULL" if col[2] == "YES" else "NOT NULL"
                        print(f"      - {col_name}: {col[1]} ({nullable})")

                # Проверка 4: Индексы
                print("\n5️⃣ Проверка индексов...")
                result = await session.execute(text(
                    """SELECT indexname
                    FROM pg_indexes
                    WHERE tablename = 'layering_patterns'
                    ORDER BY indexname"""
                ))
                indexes = result.fetchall()
                print(f"   ✅ Найдено {len(indexes)} индексов:")
                for idx in indexes:
                    print(f"      - {idx[0]}")

                # Проверка 5: Количество записей
                print("\n6️⃣ Проверка данных...")
                result = await session.execute(
                    text("SELECT COUNT(*) FROM layering_patterns")
                )
                count = result.scalar()
                print(f"   ℹ️  Записей в таблице: {count}")

                if count > 0:
                    # Показать примеры
                    result = await session.execute(text(
                        """SELECT pattern_id, occurrence_count, blacklist, risk_level
                        FROM layering_patterns
                        ORDER BY occurrence_count DESC
                        LIMIT 3"""
                    ))
                    patterns = result.fetchall()
                    print(f"\n   📊 Топ-3 паттерна:")
                    for p in patterns:
                        print(f"      - ID: {p[0][:12]}..., occurrences: {p[1]}, "
                              f"blacklist: {p[2]}, risk: {p[3]}")

            else:
                print("   ❌ Таблица layering_patterns НЕ СУЩЕСТВУЕТ")
                print("\n   💡 Возможные причины:")
                print("      1. Миграция еще не применена")
                print("      2. PostgreSQL не запущен при старте бота")
                print("      3. Ошибка при применении миграции")
                print("\n   🔧 Решение:")
                print("      cd backend")
                print("      alembic upgrade head")

        await db_manager.close()

        print("\n" + "=" * 80)
        print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        print("\n💡 Убедитесь что:")
        print("   1. PostgreSQL запущен")
        print("   2. Настройки DATABASE_URL корректны в .env")
        print("   3. База данных создана")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(check_migration_status())
