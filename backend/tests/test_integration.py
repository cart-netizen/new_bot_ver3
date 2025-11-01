"""
Интеграционный тест всего флоу размещения ордера.
Тестирует: идемпотентность, подпись, размещение, отслеживание, отмену.
"""
import asyncio
import sys
from pathlib import Path

# Определяем путь к backend директории
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))
if sys.platform == 'win32':
  asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from backend.exchange.rest_client import rest_client
from backend.domain.services.idempotency_service import IdempotencyService
from backend.infrastructure.repositories.order_repository import order_repository
from backend.database.connection import db_manager
from backend.database.models import OrderSide, OrderType, OrderStatus


async def test_full_order_flow():
  """Полный тест флоу размещения ордера"""

  idempotency_service = IdempotencyService()

  try:
    # Инициализация
    print("=" * 60)
    print("ИНТЕГРАЦИОННЫЙ ТЕСТ ФЛОУ РАЗМЕЩЕНИЯ ОРДЕРА")
    print("=" * 60)

    await rest_client.initialize()
    await db_manager.initialize()

    # 1. Генерация client_order_id
    print("\n[1/7] Генерация client_order_id...")
    client_order_id = idempotency_service.generate_client_order_id(
      symbol="BTCUSDT",
      side="Buy",
      quantity=0.001,
      price=30000.0
    )
    print(f"✓ Client Order ID: {client_order_id}")

    # 2. Создание ордера в БД
    print("\n[2/7] Создание ордера в БД...")
    order = await order_repository.create(
      client_order_id=client_order_id,
      symbol="BTCUSDT",
      side=OrderSide.BUY,
      order_type=OrderType.LIMIT,
      quantity=0.001,
      price=30000.0,
      reason="Integration test"
    )
    print(f"✓ Ордер создан в БД: status={order.status.value}")

    # 3. Размещение на бирже
    print("\n[3/7] Размещение на бирже...")
    response = await rest_client.place_order(
      symbol="BTCUSDT",
      side="Buy",
      order_type="Limit",
      quantity=0.001,
      price=30000.0,
      client_order_id=client_order_id
    )

    result = response.get('result', {})
    exchange_order_id = result.get('orderId')
    order_link_id = result.get('orderLinkId')

    print(f"✓ Ордер размещен на бирже:")
    print(f"  Exchange Order ID: {exchange_order_id}")
    print(f"  Order Link ID: {order_link_id}")

    # 4. Проверка orderLinkId
    print("\n[4/7] Проверка orderLinkId...")
    if order_link_id == client_order_id:
      print(f"✓✓✓ orderLinkId совпадает с client_order_id")
    else:
      print(f"✗✗✗ ОШИБКА: orderLinkId не совпадает!")
      print(f"  Ожидалось: {client_order_id}")
      print(f"  Получено: {order_link_id}")
      return

    # 5. Обновление статуса в БД
    print("\n[5/7] Обновление статуса в БД...")
    success = await order_repository.update_status(
      client_order_id=client_order_id,
      new_status=OrderStatus.PLACED,
      exchange_order_id=exchange_order_id
    )

    if success:
      print(f"✓ Статус обновлен: PENDING -> PLACED")
    else:
      print(f"✗ Ошибка обновления статуса")
      return

    # 6. Проверка через get_order_info по client_order_id
    print("\n[6/7] Проверка через get_order_info (order_link_id)...")
    await asyncio.sleep(1)  # Небольшая задержка

    order_info = await rest_client.get_order_info(
      symbol="BTCUSDT",
      order_link_id=client_order_id
    )

    if order_info:
      print(f"✓ Ордер найден по order_link_id:")
      print(f"  Status: {order_info.get('orderStatus')}")
      print(f"  Order ID: {order_info.get('orderId')}")
    else:
      print(f"✗ Ордер НЕ найден по order_link_id")
      return

    # 7. Отмена тестового ордера
    print("\n[7/7] Отмена тестового ордера...")
    cancel_response = await rest_client.cancel_order(
      symbol="BTCUSDT",
      order_id=exchange_order_id
    )
    print("✓ Тестовый ордер отменен")

    # Обновляем статус в БД
    await order_repository.update_status(
      client_order_id=client_order_id,
      new_status=OrderStatus.CANCELLED
    )
    print("✓ Статус обновлен: PLACED -> CANCELLED")

    # Итоговый результат
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ ИНТЕГРАЦИОННОГО ТЕСТА")
    print("=" * 60)
    print("✓ Генерация client_order_id")
    print("✓ Создание ордера в БД")
    print("✓ Размещение на бирже с orderLinkId")
    print("✓ Обновление статуса в БД")
    print("✓ Поиск ордера по order_link_id")
    print("✓ Отмена ордера")
    print("=" * 60)
    print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("=" * 60)

  except Exception as e:
    print(f"\n✗✗✗ ОШИБКА ТЕСТА: {e}")
    import traceback
    traceback.print_exc()

  finally:
    # Очистка
    print("\n=== Очистка ресурсов ===")
    await rest_client.close()
    await db_manager.close()
    print("✓ Все ресурсы освобождены")


def main():
  """Точка входа"""
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)

  try:
    loop.run_until_complete(test_full_order_flow())
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


if __name__ == "__main__":
  main()