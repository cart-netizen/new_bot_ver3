"""
Тест подключения к Bybit API с подробной диагностикой.
"""

import asyncio
import hmac
import hashlib
import time
import aiohttp
import os
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

# Настройки из .env
BYBIT_MODE = os.getenv("BYBIT_MODE", "testnet")
API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")

BASE_URL = (
  "https://api-testnet.bybit.com" if BYBIT_MODE == "testnet"
  else "https://api.bybit.com"
)

print("=" * 80)
print("🔧 BYBIT API KEY TEST")
print("=" * 80)
print(f"Режим: {BYBIT_MODE}")
print(f"URL: {BASE_URL}")
print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:] if len(API_KEY) > 14 else ''}")
print(f"API Secret: {'*' * len(API_SECRET)}")
print("=" * 80)


def create_signature(timestamp: str, api_key: str, recv_window: str, params: str) -> str:
  """
  Создание подписи для Bybit API V5.

  Формат: timestamp + api_key + recv_window + queryString
  """
  sign_string = f"{timestamp}{api_key}{recv_window}{params}"

  print("\n📝 Создание подписи:")
  print(f"  Timestamp: {timestamp}")
  print(f"  API Key: {api_key[:10]}...")
  print(f"  Recv Window: {recv_window}")
  print(f"  Params: {params}")
  print(f"  Sign String: {sign_string[:50]}...")

  signature = hmac.new(
    API_SECRET.encode('utf-8'),
    sign_string.encode('utf-8'),
    hashlib.sha256
  ).hexdigest()

  print(f"  Signature: {signature[:20]}...")

  return signature


async def test_server_time():
  """Тест публичного эндпоинта (без аутентификации)"""
  print("\n" + "=" * 80)
  print("📡 ТЕСТ 1: Проверка подключения к серверу (публичный эндпоинт)")
  print("=" * 80)

  url = f"{BASE_URL}/v5/market/time"

  try:
    async with aiohttp.ClientSession() as session:
      async with session.get(url) as response:
        status = response.status
        data = await response.json()

        print(f"  Status Code: {status}")
        print(f"  Response: {data}")

        if status == 200 and data.get("retCode") == 0:
          server_time = data.get("result", {}).get("timeSecond", "N/A")
          print(f"  ✅ Сервер доступен! Server time: {server_time}")
          return True
        else:
          print(f"  ❌ Сервер вернул ошибку")
          return False

  except Exception as e:
    print(f"  ❌ Ошибка подключения: {e}")
    return False


async def test_api_key():
  """Тест приватного эндпоинта (с аутентификацией)"""
  print("\n" + "=" * 80)
  print("🔐 ТЕСТ 2: Проверка API ключа (приватный эндпоинт)")
  print("=" * 80)

  if not API_KEY or not API_SECRET:
    print("  ❌ API ключи не настроены в .env файле!")
    print("  Убедитесь что .env содержит:")
    print("    BYBIT_API_KEY=ваш_ключ")
    print("    BYBIT_API_SECRET=ваш_секрет")
    return False

  # Параметры запроса
  timestamp = str(int(time.time() * 1000))
  recv_window = "5000"

  # Query string для GET запроса
  query_params = "accountType=UNIFIED"

  # Создаем подпись
  signature = create_signature(timestamp, API_KEY, recv_window, query_params)

  # Заголовки
  headers = {
    "X-BAPI-API-KEY": API_KEY,
    "X-BAPI-SIGN": signature,
    "X-BAPI-TIMESTAMP": timestamp,
    "X-BAPI-RECV-WINDOW": recv_window,
    "Content-Type": "application/json"
  }

  # URL
  url = f"{BASE_URL}/v5/account/wallet-balance"

  print("\n📤 Отправка запроса:")
  print(f"  URL: {url}")
  print(f"  Params: {query_params}")
  print(f"  Headers:")
  for key, value in headers.items():
    if key == "X-BAPI-SIGN":
      print(f"    {key}: {value[:20]}...")
    else:
      print(f"    {key}: {value}")

  try:
    async with aiohttp.ClientSession() as session:
      async with session.get(
          url,
          params={"accountType": "UNIFIED"},
          headers=headers
      ) as response:
        status = response.status

        # Пытаемся получить JSON ответ
        try:
          data = await response.json()
        except:
          text = await response.text()
          data = None
          print(f"\n📥 Ответ сервера:")
          print(f"  Status Code: {status}")
          print(f"  Raw Text: {text[:200]}")

        print(f"\n📥 Ответ сервера:")
        print(f"  Status Code: {status}")

        if data:
          print(f"  Response: {data}")

          ret_code = data.get("retCode")
          ret_msg = data.get("retMsg", "")

          if ret_code == 0:
            print(f"\n  ✅ API ключ работает!")
            print(f"  Режим: {BYBIT_MODE}")

            # Показываем баланс
            result = data.get("result", {})
            wallet_list = result.get("list", [])
            if wallet_list:
              print(f"\n  💰 Баланс:")
              for wallet in wallet_list:
                coins = wallet.get("coin", [])
                for coin in coins:
                  coin_name = coin.get("coin", "N/A")
                  balance = coin.get("walletBalance", "0")
                  print(f"    {coin_name}: {balance}")

            return True
          else:
            print(f"\n  ❌ Ошибка API:")
            print(f"    Код: {ret_code}")
            print(f"    Сообщение: {ret_msg}")

            # Диагностика распространенных ошибок
            if ret_code == 10003:
              print(f"\n  💡 Диагностика ошибки 10003 (Invalid API key):")
              print(f"    1. Убедитесь что используете ключ от {BYBIT_MODE}")
              print(f"    2. Проверьте что ключ скопирован полностью")
              print(f"    3. Убедитесь что ключ активен в личном кабинете")

            elif ret_code == 10004:
              print(f"\n  💡 Диагностика ошибки 10004 (Invalid signature):")
              print(f"    1. Проверьте что API Secret скопирован правильно")
              print(f"    2. Проверьте что нет лишних пробелов в .env")

            elif ret_code == 10005:
              print(f"\n  💡 Диагностика ошибки 10005 (Permission denied):")
              print(f"    1. Проверьте права API ключа:")
              print(f"       ✅ Read-Write")
              print(f"       ✅ Contract Trading")

            return False
        else:
          print(f"  ❌ Не удалось получить JSON ответ")
          print(f"  Status Code: {status}")

          if status == 401:
            print(f"\n  💡 Ошибка 401 (Unauthorized):")
            print(f"    Проблема с аутентификацией. Возможные причины:")
            print(
              f"    1. API ключ от {BYBIT_MODE}, а вы используете {'mainnet' if BYBIT_MODE == 'testnet' else 'testnet'}")
            print(f"    2. API Secret неверный")
            print(f"    3. Проблема с созданием подписи")
            print(f"    4. IP адрес не в whitelist (если настроено)")

          return False

  except Exception as e:
    print(f"  ❌ Ошибка запроса: {e}")
    import traceback
    traceback.print_exc()
    return False


async def main():
  """Запуск всех тестов"""

  # Тест 1: Проверка подключения к серверу
  server_ok = await test_server_time()

  if not server_ok:
    print("\n❌ Не удалось подключиться к Bybit серверу")
    print("Проверьте ваше интернет соединение")
    return

  # Тест 2: Проверка API ключа
  api_ok = await test_api_key()

  # Итоговый результат
  print("\n" + "=" * 80)
  print("📊 ИТОГИ ТЕСТИРОВАНИЯ")
  print("=" * 80)
  print(f"  Сервер доступен: {'✅' if server_ok else '❌'}")
  print(f"  API ключ работает: {'✅' if api_ok else '❌'}")
  print("=" * 80)

  if not api_ok:
    print("\n💡 РЕКОМЕНДАЦИИ:")
    print(f"  1. Зайдите на https://{BYBIT_MODE}.bybit.com")
    print(f"  2. Создайте НОВЫЙ API ключ с правами:")
    print(f"     - Read-Write")
    print(f"     - Contract Trading")
    print(f"  3. Скопируйте ключи в .env файл:")
    print(f"     BYBIT_MODE={BYBIT_MODE}")
    print(f"     BYBIT_API_KEY=ваш_новый_ключ")
    print(f"     BYBIT_API_SECRET=ваш_новый_секрет")
    print(f"  4. Запустите тест снова: python test_api_key.py")


if __name__ == "__main__":
  asyncio.run(main())