"""
Утилита для безопасной очистки тестовых и неудачных ордеров из БД.
Использование: python cleanup_test_orders.py [опция]
"""
import asyncio
import sys
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Определяем путь к backend директории
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))
if sys.platform == 'win32':
  asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from backend.database.connection import db_manager
from backend.database.models import OrderStatus
from sqlalchemy import select, delete, func
from backend.core.logger import get_logger

logger = get_logger(__name__)


class OrderCleanupService:
  """Сервис очистки ордеров"""

  async def get_orders_stats(self):
    """Получить статистику ордеров"""
    print("=" * 80)
    print("СТАТИСТИКА ОРДЕРОВ В БД")
    print("=" * 80)

    async with db_manager.session() as session:
      # Общее количество
      from backend.database.models import Order

      stmt = select(func.count()).select_from(Order)
      result = await session.execute(stmt)
      total = result.scalar()

      print(f"Всего ордеров: {total}")

      # По статусам
      stmt = select(
        Order.status,
        func.count().label('count')
      ).group_by(Order.status)

      result = await session.execute(stmt)
      stats = result.all()

      print("\nРаспределение по статусам:")
      for status, count in stats:
        print(f"  {status.value}: {count}")

      # Без exchange_order_id
      stmt = select(func.count()).select_from(Order).where(
        Order.exchange_order_id.is_(None)
      )
      result = await session.execute(stmt)
      without_exchange = result.scalar()

      print(f"\nБез exchange_order_id: {without_exchange}")

      # Тестовые
      stmt = select(func.count()).select_from(Order).where(
        Order.reason.ilike('%test%') | Order.reason.ilike('%integration%')
      )
      result = await session.execute(stmt)
      test_orders = result.scalar()

      print(f"Тестовые ордера (по reason): {test_orders}")

      print("=" * 80)

  async def preview_pending_orphans(self):
    """Предпросмотр PENDING ордеров без exchange_order_id"""
    print("\nPENDING ордера без exchange_order_id:")
    print("-" * 80)

    async with db_manager.session() as session:
      from backend.database.models import Order

      stmt = select(Order).where(
        Order.status == OrderStatus.PENDING,
        Order.exchange_order_id.is_(None)
      ).order_by(Order.created_at.desc())

      result = await session.execute(stmt)
      orders = result.scalars().all()

      if not orders:
        print("  Нет ордеров для удаления")
        return 0

      for order in orders:
        print(
          f"  {order.client_order_id} | {order.symbol} | "
          f"{order.created_at} | {order.reason or 'N/A'}"
        )

      return len(orders)

  async def preview_failed_orders(self):
    """Предпросмотр FAILED ордеров"""
    print("\nFAILED ордера:")
    print("-" * 80)

    async with db_manager.session() as session:
      from backend.database.models import Order

      stmt = select(Order).where(
        Order.status == OrderStatus.FAILED
      ).order_by(Order.created_at.desc())

      result = await session.execute(stmt)
      orders = result.scalars().all()

      if not orders:
        print("  Нет ордеров для удаления")
        return 0

      for order in orders:
        print(
          f"  {order.client_order_id} | {order.symbol} | "
          f"{order.created_at} | {order.reason or 'N/A'}"
        )

      return len(orders)

  async def preview_test_orders(self):
    """Предпросмотр тестовых ордеров"""
    print("\nТестовые ордера (по reason):")
    print("-" * 80)

    async with db_manager.session() as session:
      from backend.database.models import Order

      stmt = select(Order).where(
        Order.reason.ilike('%test%') | Order.reason.ilike('%integration%')
      ).order_by(Order.created_at.desc())

      result = await session.execute(stmt)
      orders = result.scalars().all()

      if not orders:
        print("  Нет ордеров для удаления")
        return 0

      for order in orders:
        print(
          f"  {order.client_order_id} | {order.symbol} | "
          f"{order.status.value} | {order.created_at} | {order.reason}"
        )

      return len(orders)

  async def delete_pending_orphans(self):
    """Удалить PENDING ордера без exchange_order_id"""
    count = await self.preview_pending_orphans()

    if count == 0:
      return 0

    confirm = input(f"\nУдалить {count} PENDING ордеров? (yes/no): ")
    if confirm.lower() != 'yes':
      print("Отменено")
      return 0

    async with db_manager.session() as session:
      from backend.database.models import Order

      stmt = delete(Order).where(
        Order.status == OrderStatus.PENDING,
        Order.exchange_order_id.is_(None)
      )

      result = await session.execute(stmt)
      await session.commit()

      deleted = result.rowcount
      print(f"✓ Удалено {deleted} PENDING ордеров")
      return deleted

  async def delete_failed_orders(self):
    """Удалить FAILED ордера"""
    count = await self.preview_failed_orders()

    if count == 0:
      return 0

    confirm = input(f"\nУдалить {count} FAILED ордеров? (yes/no): ")
    if confirm.lower() != 'yes':
      print("Отменено")
      return 0

    async with db_manager.session() as session:
      from backend.database.models import Order

      stmt = delete(Order).where(
        Order.status == OrderStatus.FAILED
      )

      result = await session.execute(stmt)
      await session.commit()

      deleted = result.rowcount
      print(f"✓ Удалено {deleted} FAILED ордеров")
      return deleted

  async def delete_test_orders(self):
    """Удалить тестовые ордера"""
    count = await self.preview_test_orders()

    if count == 0:
      return 0

    confirm = input(f"\nУдалить {count} тестовых ордеров? (yes/no): ")
    if confirm.lower() != 'yes':
      print("Отменено")
      return 0

    async with db_manager.session() as session:
      from backend.database.models import Order

      stmt = delete(Order).where(
        Order.reason.ilike('%test%') | Order.reason.ilike('%integration%')
      )

      result = await session.execute(stmt)
      await session.commit()

      deleted = result.rowcount
      print(f"✓ Удалено {deleted} тестовых ордеров")
      return deleted

  async def delete_all_orders(self):
    """ОПАСНО: Удалить ВСЕ ордера"""
    print("=" * 80)
    print("ВНИМАНИЕ: ЭТО УДАЛИТ ВСЕ ОРДЕРА ИЗ БД!")
    print("=" * 80)

    await self.get_orders_stats()

    confirm = input("\nВы УВЕРЕНЫ? Введите 'DELETE ALL' для подтверждения: ")
    if confirm != 'DELETE ALL':
      print("Отменено")
      return 0

    async with db_manager.session() as session:
      from backend.database.models import Order

      stmt = delete(Order)
      result = await session.execute(stmt)
      await session.commit()

      deleted = result.rowcount
      print(f"✓ Удалено {deleted} ордеров (ВСЕ)")
      return deleted


async def main():
  """Главная функция"""
  service = OrderCleanupService()

  await db_manager.initialize()

  try:
    # Показываем меню
    print("\n" + "=" * 80)
    print("УТИЛИТА ОЧИСТКИ ОРДЕРОВ")
    print("=" * 80)
    print("\nВыберите действие:")
    print("  1. Показать статистику")
    print("  2. Удалить PENDING ордера без exchange_order_id")
    print("  3. Удалить FAILED ордера")
    print("  4. Удалить тестовые ордера (по reason)")
    print("  5. ОПАСНО: Удалить ВСЕ ордера")
    print("  0. Выход")
    print("=" * 80)

    choice = input("\nВведите номер: ")

    if choice == "1":
      await service.get_orders_stats()

    elif choice == "2":
      await service.delete_pending_orphans()
      print("\n")
      await service.get_orders_stats()

    elif choice == "3":
      await service.delete_failed_orders()
      print("\n")
      await service.get_orders_stats()

    elif choice == "4":
      await service.delete_test_orders()
      print("\n")
      await service.get_orders_stats()

    elif choice == "5":
      await service.delete_all_orders()
      print("\n")
      await service.get_orders_stats()

    elif choice == "0":
      print("Выход")

    else:
      print("Неверный выбор")

  finally:
    await db_manager.close()


if __name__ == "__main__":
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)

  try:
    loop.run_until_complete(main())
  finally:
    try:
      pending = asyncio.all_tasks(loop)
      for task in pending:
        task.cancel()
      loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception:
      pass
    finally:
      loop.close()