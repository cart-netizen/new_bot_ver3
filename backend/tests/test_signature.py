"""
Тест генерации подписи и запросов к Bybit API.
Исправленная версия с корректным закрытием сессий.
"""

import sys
from pathlib import Path

# Определяем путь к backend директории
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

import asyncio
import asyncio
import sys

# Исправление для Windows
if sys.platform == 'win32':
  asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from exchange.bybit_auth import authenticator
from exchange.rest_client import rest_client


async def test_signature():
  """Тест генерации подписи и запроса к API"""

  try:
    # Тест 1: GET запрос
    print("=== Тест GET запроса ===")
    params = {"category": "linear", "symbol": "BTCUSDT"}
    auth_data = authenticator.prepare_request("GET", params)
    print(f"Подпись сгенерирована: {auth_data['headers']['X-BAPI-SIGN'][:16]}...")

    # Тест 2: POST запрос
    print("\n=== Тест POST запроса ===")
    params = {
      "category": "linear",
      "symbol": "BTCUSDT",
      "side": "Buy",
      "orderType": "Limit",
      "qty": "0.001",
      "price": "50000"
    }
    auth_data = authenticator.prepare_request("POST", params)
    print(f"Подпись сгенерирована: {auth_data['headers']['X-BAPI-SIGN'][:16]}...")

    # Тест 3: Реальный запрос к API
    print("\n=== Тест реального запроса ===")
    try:
      # Инициализируем сессию явно
      await rest_client.initialize()

      server_time = await rest_client.get_server_time()
      print(f"✓ Успешный запрос! Серверное время: {server_time}")
      print(f"✓ Подпись API работает корректно")

    except Exception as e:
      print(f"✗ Ошибка запроса: {e}")

  finally:
    # КРИТИЧНО: Закрываем сессию явно
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
    loop.run_until_complete(test_signature())
  finally:
    # Закрываем все оставшиеся задачи
    try:
      # Отменяем все pending задачи
      pending = asyncio.all_tasks(loop)
      for task in pending:
        task.cancel()

      # Даем задачам время на отмену
      loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception:
      pass
    finally:
      # Закрываем event loop
      loop.close()
      print("\n✓ Event loop закрыт корректно")


if __name__ == "__main__":
  main()