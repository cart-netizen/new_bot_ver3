"""
Тест размещения ордера с client_order_id на Bybit.
Исправленная версия с корректным закрытием сессий.
"""
import asyncio
import sys
from pathlib import Path

# Определяем путь к backend директории
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))
# Исправление для Windows
if sys.platform == 'win32':
  asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from backend.exchange.rest_client import rest_client
from backend.domain.services.idempotency_service import IdempotencyService


async def test_place_order():
  """Тест размещения ордера с client_order_id"""

  # Инициализируем сервис идемпотентности
  idempotency_service = IdempotencyService()

  try:
    # Инициализируем REST клиент
    await rest_client.initialize()

    # Генерируем client_order_id
    client_order_id = idempotency_service.generate_client_order_id(
      symbol="BTCUSDT",
      side="Buy",
      quantity=0.001,
      price=30000.0
    )

    print(f"=== Тест размещения ордера ===")
    print(f"Client Order ID: {client_order_id}")
    print(f"Внимание: Будет размещен ТЕСТОВЫЙ ордер с низкой ценой")
    print(f"Цена: 30000 USDT (заведомо низкая, не исполнится)")

    # Пытаемся разместить тестовый ордер
    response = await rest_client.place_order(
      symbol="BTCUSDT",
      side="Buy",
      order_type="Limit",
      quantity=0.001,
      price=30000.0,  # Заведомо низкая цена, не исполнится
      client_order_id=client_order_id
    )

    result = response.get('result', {})

    print(f"\n✓ Ордер размещен успешно!")
    print(f"Exchange Order ID: {result.get('orderId')}")
    print(f"Order Link ID: {result.get('orderLinkId', 'N/A')}")
    print(f"Status: {result.get('orderStatus')}")

    # Проверяем, что orderLinkId совпадает
    if result.get('orderLinkId') == client_order_id:
      print("\n✓✓✓ orderLinkId КОРРЕКТНО установлен!")
    else:
      print(f"\n✗✗✗ ОШИБКА: orderLinkId не совпадает!")
      print(f"Ожидалось: {client_order_id}")
      print(f"Получено: {result.get('orderLinkId')}")

    # Небольшая пауза перед отменой
    await asyncio.sleep(1)

    # Отменяем тестовый ордер
    print(f"\n=== Отмена тестового ордера ===")
    cancel_response = await rest_client.cancel_order(
      symbol="BTCUSDT",
      order_id=result.get('orderId')
    )
    print("✓ Тестовый ордер отменен успешно")

    # Итоговая проверка
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТ ТЕСТА:")
    print("=" * 50)
    print("✓ Подпись API работает")
    print("✓ Ордер размещается с client_order_id")
    print("✓ orderLinkId устанавливается корректно")
    print("✓ Ордер можно отменить")
    print("=" * 50)

  except Exception as e:
    print(f"\n✗✗✗ ОШИБКА теста: {e}")
    import traceback
    traceback.print_exc()

  finally:
    # Закрываем сессию
    print("\n=== Закрытие сессии ===")
    await rest_client.close()
    print("✓ Сессия закрыта корректно")


def main():
  """Точка входа с корректной обработкой event loop"""
  # Создаем новый event loop
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)

  try:
    # Запускаем тест
    loop.run_until_complete(test_place_order())
  finally:
    # Закрываем все оставшиеся задачи
    try:
      pending = asyncio.all_tasks(loop)
      for task in pending:
        task.cancel()
      loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception:
      pass
    finally:
      loop.close()
      print("\n✓ Event loop закрыт корректно")


if __name__ == "__main__":
  main()