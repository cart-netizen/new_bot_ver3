"""
Тест WebSocket эндпоинта бэкенда.
Проверяет, работает ли /ws эндпоинт на бэкенде.
"""

import asyncio
import websockets
import json

BACKEND_WS_URL = "ws://localhost:8000/ws"


async def test_websocket_connection():
  """Тест подключения к WebSocket эндпоинту бэкенда."""
  print("=" * 80)
  print("🧪 ТЕСТ WEBSOCKET ЭНДПОИНТА БЭКЕНДА")
  print("=" * 80)
  print(f"URL: {BACKEND_WS_URL}")
  print()

  try:
    print("⏳ Попытка подключения...")

    # Пробуем подключиться
    async with websockets.connect(BACKEND_WS_URL) as websocket:
      print("✅ Подключение установлено!")
      print(f"   Состояние: {websocket.state}")
      print()

      # Получаем токен (для теста используем тестовый пароль)
      # В реальности нужно сначала получить токен через /auth/login
      test_token = "test_token_123"  # Это заглушка

      # Отправляем аутентификацию
      auth_message = {
        "type": "authenticate",
        "token": test_token
      }

      print(f"📤 Отправка аутентификации:")
      print(f"   {json.dumps(auth_message, indent=2)}")
      print()

      await websocket.send(json.dumps(auth_message))
      print("✅ Сообщение отправлено")
      print()

      # Ждем ответ
      print("⏳ Ожидание ответа от сервера...")
      try:
        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        print("✅ Получен ответ от сервера:")
        print()

        response_data = json.loads(response)
        print(json.dumps(response_data, indent=2))
        print()

        # Проверяем тип ответа
        if response_data.get("type") == "error":
          print("⚠️  Сервер вернул ошибку:")
          print(f"   {response_data.get('message')}")
        elif response_data.get("type") == "authenticated":
          print("✅ Аутентификация успешна!")
        else:
          print(f"ℹ️  Тип ответа: {response_data.get('type')}")

      except asyncio.TimeoutError:
        print("⏰ Таймаут ожидания ответа")

      # Пробуем отправить ping
      print()
      print("📤 Отправка ping...")
      ping_message = {"type": "ping"}
      await websocket.send(json.dumps(ping_message))

      try:
        pong = await asyncio.wait_for(websocket.recv(), timeout=2.0)
        print("✅ Получен pong:")
        print(f"   {pong}")
      except asyncio.TimeoutError:
        print("⏰ Нет ответа на ping")

  except websockets.exceptions.InvalidURI:
    print("❌ ОШИБКА: Неверный URI")
    print(f"   Проверьте URL: {BACKEND_WS_URL}")

  except websockets.exceptions.WebSocketException as e:
    print(f"❌ ОШИБКА WebSocket: {e}")
    print()
    print("💡 ВОЗМОЖНЫЕ ПРИЧИНЫ:")
    print("   1. Бэкенд не запущен")
    print("   2. WebSocket эндпоинт не зарегистрирован")
    print("   3. Проблема с CORS")
    print("   4. Порт 8000 занят другим процессом")

  except ConnectionRefusedError:
    print("❌ ОШИБКА: Соединение отклонено")
    print()
    print("💡 ПРИЧИНА:")
    print("   Бэкенд не запущен или не слушает на порту 8000")
    print()
    print("📝 РЕШЕНИЕ:")
    print("   1. Запустите бэкенд:")
    print("      cd backend")
    print("      python main.py")
    print("   2. Убедитесь, что видите: 'Uvicorn running on http://0.0.0.0:8000'")

  except Exception as e:
    print(f"❌ НЕОЖИДАННАЯ ОШИБКА: {type(e).__name__}: {e}")
    import traceback
    print()
    print("Traceback:")
    traceback.print_exc()

  print()
  print("=" * 80)


async def test_http_endpoint():
  """Тест HTTP эндпоинта для проверки, что сервер работает."""
  print()
  print("🌐 ТЕСТ HTTP ЭНДПОИНТА")
  print("=" * 80)

  import aiohttp

  http_url = "http://localhost:8000/health"
  print(f"URL: {http_url}")
  print()

  try:
    print("⏳ Отправка GET запроса...")
    async with aiohttp.ClientSession() as session:
      async with session.get(http_url, timeout=5.0) as response:
        print(f"✅ Ответ получен: HTTP {response.status}")

        if response.status == 200:
          data = await response.json()
          print(f"   {json.dumps(data, indent=2)}")
          print()
          print("✅ HTTP сервер работает!")
        else:
          print(f"⚠️  Неожиданный статус: {response.status}")

  except aiohttp.ClientConnectorError:
    print("❌ ОШИБКА: Не удалось подключиться к HTTP серверу")
    print()
    print("💡 ПРИЧИНА:")
    print("   Бэкенд не запущен")

  except Exception as e:
    print(f"❌ ОШИБКА: {type(e).__name__}: {e}")

  print("=" * 80)


async def main():
  """Запуск всех тестов."""
  # Сначала проверяем HTTP
  await test_http_endpoint()

  print()
  print()

  # Потом WebSocket
  await test_websocket_connection()

  print()
  print("✅ Тестирование завершено")


if __name__ == "__main__":
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    print("\n⚠️  Прервано пользователем")