"""
Тест полного процесса аутентификации и WebSocket подключения.
Эмулирует поведение фронтенда.
"""

import asyncio
import aiohttp
import websockets
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Настройки
BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"
PASSWORD = os.getenv("APP_PASSWORD", "robocop89")


async def test_full_flow():
  """Полный тест: логин → получение токена → WebSocket подключение."""

  print("=" * 80)
  print("🔐 ТЕСТ ПОЛНОГО ПРОЦЕССА АУТЕНТИФИКАЦИИ")
  print("=" * 80)
  print()

  # ШАГ 1: Логин
  print("📝 ШАГ 1: Логин через /auth/login")
  print("-" * 80)

  token = None

  try:
    async with aiohttp.ClientSession() as session:
      login_url = f"{BACKEND_URL}/auth/login"
      login_data = {"password": PASSWORD}

      print(f"URL: {login_url}")
      print(f"Данные: {json.dumps(login_data, indent=2)}")
      print()
      print("⏳ Отправка запроса...")

      async with session.post(login_url, json=login_data) as response:
        print(f"✅ Ответ получен: HTTP {response.status}")
        print()

        if response.status == 200:
          data = await response.json()
          token = data.get("access_token")

          print("✅ ЛОГИН УСПЕШЕН!")
          print(f"   Token получен: {token[:20]}...{token[-20:]}")
          print()

        elif response.status == 401:
          error_data = await response.json()
          print("❌ ОШИБКА АУТЕНТИФИКАЦИИ!")
          print(f"   {error_data.get('detail', 'Неизвестная ошибка')}")
          print()
          print("💡 ПРИЧИНА:")
          print(f"   Пароль в .env: {PASSWORD}")
          print("   Проверьте, что это правильный пароль")
          return

        else:
          print(f"⚠️  Неожиданный статус: {response.status}")
          text = await response.text()
          print(f"   Ответ: {text[:200]}")
          return

  except aiohttp.ClientConnectorError:
    print("❌ ОШИБКА: Не удалось подключиться к бэкенду")
    print()
    print("💡 РЕШЕНИЕ:")
    print("   1. Убедитесь, что бэкенд запущен:")
    print("      cd backend")
    print("      python main.py")
    return

  except Exception as e:
    print(f"❌ ОШИБКА: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    return

  # ШАГ 2: Проверка токена
  print("🔍 ШАГ 2: Проверка токена через /auth/verify")
  print("-" * 80)

  try:
    async with aiohttp.ClientSession() as session:
      verify_url = f"{BACKEND_URL}/auth/verify"
      headers = {"Authorization": f"Bearer {token}"}

      print(f"URL: {verify_url}")
      print(f"Header: Authorization: Bearer {token[:20]}...")
      print()
      print("⏳ Отправка запроса...")

      async with session.get(verify_url, headers=headers) as response:
        print(f"✅ Ответ получен: HTTP {response.status}")
        print()

        if response.status == 200:
          data = await response.json()
          print("✅ ТОКЕН ВАЛИДНЫЙ!")
          print(f"   User ID: {data.get('user_id')}")
          print()
        else:
          print(f"⚠️  Токен невалидный: {response.status}")
          text = await response.text()
          print(f"   {text}")
          return

  except Exception as e:
    print(f"❌ ОШИБКА: {type(e).__name__}: {e}")
    return

  # ШАГ 3: WebSocket подключение
  print("🔌 ШАГ 3: WebSocket подключение")
  print("-" * 80)

  try:
    print(f"URL: {WS_URL}")
    print()
    print("⏳ Подключение...")

    async with websockets.connect(WS_URL) as websocket:
      print("✅ WebSocket соединение установлено!")
      print()

      # Отправляем аутентификацию
      auth_message = {
        "type": "authenticate",
        "token": token
      }

      print("📤 Отправка аутентификации с ВАЛИДНЫМ токеном...")
      await websocket.send(json.dumps(auth_message))
      print("✅ Сообщение отправлено")
      print()

      # Ждем ответ
      print("⏳ Ожидание ответа...")
      try:
        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        response_data = json.loads(response)

        print("✅ Получен ответ:")
        print(json.dumps(response_data, indent=2, ensure_ascii=False))
        print()

        if response_data.get("type") == "authenticated":
          print("🎉 АУТЕНТИФИКАЦИЯ В WEBSOCKET УСПЕШНА!")
          print()

          # Тестируем ping
          print("📤 Отправка ping...")
          ping_message = {"type": "ping"}
          await websocket.send(json.dumps(ping_message))

          pong = await asyncio.wait_for(websocket.recv(), timeout=2.0)
          pong_data = json.loads(pong)
          print("✅ Получен pong:")
          print(json.dumps(pong_data, indent=2, ensure_ascii=False))
          print()

          # Подписываемся на обновления
          print("📤 Подписка на обновления...")
          subscribe_message = {
            "type": "subscribe",
            "channels": ["orderbook", "signals", "bot_status"]
          }
          await websocket.send(json.dumps(subscribe_message))

          sub_response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
          sub_data = json.loads(sub_response)
          print("✅ Ответ на подписку:")
          print(json.dumps(sub_data, indent=2, ensure_ascii=False))
          print()

          print("=" * 80)
          print("🎉 ВСЁ РАБОТАЕТ ИДЕАЛЬНО!")
          print("=" * 80)
          print()
          print("✅ Логин работает")
          print("✅ Токен валидный")
          print("✅ WebSocket подключается")
          print("✅ Аутентификация в WebSocket работает")
          print("✅ Ping/pong работает")
          print("✅ Подписка работает")
          print()
          print("💡 ВЫВОД:")
          print("   Проблема НЕ на бэкенде!")
          print("   Проблема в логике фронтенда:")
          print("   - Фронтенд не получает токен перед подключением к WS")
          print("   - Или не передает его правильно в WebSocket сервис")

        elif response_data.get("type") == "error":
          print("⚠️  Сервер вернул ошибку:")
          print(f"   {response_data.get('message')}")

      except asyncio.TimeoutError:
        print("⏰ Таймаут ожидания ответа")

  except Exception as e:
    print(f"❌ ОШИБКА WebSocket: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

  print()
  print("=" * 80)


async def test_auth_endpoint():
  """Быстрый тест эндпоинта аутентификации."""
  print()
  print("🔐 БЫСТРЫЙ ТЕСТ /auth/login")
  print("=" * 80)

  try:
    async with aiohttp.ClientSession() as session:
      url = f"{BACKEND_URL}/auth/login"

      # Тест с правильным паролем
      print(f"Тест с паролем: {PASSWORD}")
      async with session.post(url, json={"password": PASSWORD}) as response:
        if response.status == 200:
          data = await response.json()
          print(f"✅ Логин успешен! Token получен.")
        else:
          print(f"❌ Ошибка: HTTP {response.status}")
          print(await response.text())

  except Exception as e:
    print(f"❌ ОШИБКА: {e}")

  print("=" * 80)


if __name__ == "__main__":
  print()
  print("🚀 ЗАПУСК ТЕСТОВ АУТЕНТИФИКАЦИИ")
  print()

  try:
    # Сначала быстрый тест
    asyncio.run(test_auth_endpoint())

    print()
    print()

    # Потом полный тест
    asyncio.run(test_full_flow())

  except KeyboardInterrupt:
    print("\n⚠️  Прервано пользователем")