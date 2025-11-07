import asyncio
from backend.database.connection import db_manager
from sqlalchemy import text


async def check_tables():
  async with db_manager.session() as session:
    # Проверить существование таблиц
    result = await session.execute(text("""
            SELECT tablename 
            FROM pg_tables 
            WHERE tablename LIKE 'backtest%'
            ORDER BY tablename;
        """))
    tables = result.fetchall()
    print("Созданные таблицы backtesting:")
    for table in tables:
      print(f"  - {table[0]}")

    # Подсчитать колонки в каждой таблице
    for table in tables:
      result = await session.execute(text(f"""
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_name = '{table[0]}';
            """))
      count = result.scalar()
      print(f"\nТаблица {table[0]} имеет {count} колонок")

    # Проверить hypertable
    try:
      result = await session.execute(text("""
                SELECT hypertable_name, num_dimensions 
                FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'backtest_equity';
            """))
      hypertable = result.fetchone()
      if hypertable:
        print(f"\n✅ backtest_equity является TimescaleDB hypertable")
      else:
        print(f"\n⚠️ backtest_equity НЕ является hypertable")
    except:
      print(f"\n⚠️ TimescaleDB не установлен или не активирован")


if __name__ == "__main__":
  asyncio.run(check_tables())