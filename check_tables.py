import asyncio
from backend.database.db_manager import db_manager
from backend.config import settings
from sqlalchemy import text

async def check_tables():
    # Инициализировать базу данных
    await db_manager.init()

    try:
        async with db_manager.session() as session:
            # Проверить существование таблиц
            result = await session.execute(text("""
                SELECT tablename
                FROM pg_tables
                WHERE tablename LIKE 'backtest%'
                ORDER BY tablename;
            """))
            tables = result.fetchall()

            if not tables:
                print("❌ Таблицы backtesting не найдены!")
                return

            print("✅ Созданные таблицы backtesting:")
            for table in tables:
                print(f"   - {table[0]}")

            # Подсчитать колонки в каждой таблице
            print("\n📊 Структура таблиц:")
            for table in tables:
                result = await session.execute(text(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = '{table[0]}'
                    ORDER BY ordinal_position;
                """))
                columns = result.fetchall()
                print(f"\n   {table[0]} ({len(columns)} колонок):")
                for col in columns[:5]:  # Показываем первые 5 колонок
                    nullable = "NULL" if col[2] == "YES" else "NOT NULL"
                    print(f"     - {col[0]}: {col[1]} ({nullable})")
                if len(columns) > 5:
                    print(f"     ... и еще {len(columns) - 5} колонок")

            # Проверить индексы
            print("\n🔍 Индексы:")
            for table in tables:
                result = await session.execute(text(f"""
                    SELECT indexname
                    FROM pg_indexes
                    WHERE tablename = '{table[0]}'
                    AND indexname NOT LIKE '%pkey%';
                """))
                indexes = result.fetchall()
                if indexes:
                    print(f"   {table[0]}:")
                    for idx in indexes:
                        print(f"     - {idx[0]}")

            # Проверить hypertable
            print("\n⏰ Проверка TimescaleDB hypertable:")
            try:
                result = await session.execute(text("""
                    SELECT hypertable_name, num_dimensions, num_chunks
                    FROM timescaledb_information.hypertables
                    WHERE hypertable_name = 'backtest_equity';
                """))
                hypertable = result.fetchone()
                if hypertable:
                    print(f"   ✅ backtest_equity является TimescaleDB hypertable")
                    print(f"      - Dimensions: {hypertable[1]}")
                    print(f"      - Chunks: {hypertable[2]}")
                else:
                    print(f"   ⚠️  backtest_equity НЕ является hypertable")
            except Exception as e:
                print(f"   ⚠️  TimescaleDB не установлен или не активирован: {e}")

            # Проверить ENUM типы
            print("\n📝 ENUM типы:")
            result = await session.execute(text("""
                SELECT t.typname, string_agg(e.enumlabel, ', ' ORDER BY e.enumsortorder) as values
                FROM pg_type t
                JOIN pg_enum e ON t.oid = e.enumtypid
                WHERE t.typname = 'backteststatus'
                GROUP BY t.typname;
            """))
            enum_type = result.fetchone()
            if enum_type:
                print(f"   - {enum_type[0]}: {enum_type[1]}")

            print("\n✅ Все таблицы backtesting успешно созданы!")

    finally:
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(check_tables())
