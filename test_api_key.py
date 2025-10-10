"""
Расширенная диагностика API ключей Bybit с проверкой .env файла.
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
BYBIT_MODE = os.getenv("BYBIT_MODE", "mainnet")
API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")

BASE_URL = (
    "https://api-testnet.bybit.com" if BYBIT_MODE == "testnet"
    else "https://api.bybit.com"
)

print("=" * 80)
print("🔧 BYBIT API KEY DIAGNOSTIC TEST")
print("=" * 80)


def check_env_file():
    """Проверка корректности .env файла"""
    print("\n📋 ШАГ 1: Проверка .env файла")
    print("-" * 80)

    issues = []

    # Проверка режима
    print(f"✓ BYBIT_MODE: {BYBIT_MODE}")
    if BYBIT_MODE not in ["testnet", "mainnet"]:
        issues.append("⚠️  BYBIT_MODE должен быть 'testnet' или 'mainnet'")

    # Проверка API_KEY
    if not API_KEY:
        issues.append("❌ BYBIT_API_KEY не задан в .env")
    else:
        key_len = len(API_KEY)
        has_spaces = API_KEY != API_KEY.strip()
        has_newlines = '\n' in API_KEY or '\r' in API_KEY

        print(f"✓ BYBIT_API_KEY присутствует")
        print(f"  Длина: {key_len} символов")
        print(f"  Первые 10: {API_KEY[:10]}")
        print(f"  Последние 4: {API_KEY[-4:]}")

        if key_len < 15:
            issues.append(f"⚠️  API_KEY слишком короткий ({key_len} символов). Ожидается 20+")

        if has_spaces:
            issues.append("⚠️  API_KEY содержит пробелы в начале/конце!")
            print(f"  ВНИМАНИЕ: Обнаружены пробелы! Реальная длина: {len(API_KEY.strip())}")

        if has_newlines:
            issues.append("⚠️  API_KEY содержит переносы строк!")

        # Проверка на кавычки
        if API_KEY.startswith('"') or API_KEY.startswith("'"):
            issues.append("⚠️  API_KEY содержит кавычки! Удалите их из .env")

    # Проверка API_SECRET
    if not API_SECRET:
        issues.append("❌ BYBIT_API_SECRET не задан в .env")
    else:
        secret_len = len(API_SECRET)
        has_spaces = API_SECRET != API_SECRET.strip()
        has_newlines = '\n' in API_SECRET or '\r' in API_SECRET

        print(f"✓ BYBIT_API_SECRET присутствует")
        print(f"  Длина: {secret_len} символов")
        print(f"  Первые 10: {API_SECRET[:10]}")
        print(f"  Последние 4: {API_SECRET[-4:]}")

        if secret_len < 15:
            issues.append(f"⚠️  API_SECRET слишком короткий ({secret_len} символов). Ожидается 20+")

        if has_spaces:
            issues.append("⚠️  API_SECRET содержит пробелы в начале/конце!")
            print(f"  ВНИМАНИЕ: Обнаружены пробелы! Реальная длина: {len(API_SECRET.strip())}")

        if has_newlines:
            issues.append("⚠️  API_SECRET содержит переносы строк!")

        # Проверка на кавычки
        if API_SECRET.startswith('"') or API_SECRET.startswith("'"):
            issues.append("⚠️  API_SECRET содержит кавычки! Удалите их из .env")

    # Вывод проблем
    if issues:
        print("\n🚨 ОБНАРУЖЕНЫ ПРОБЛЕМЫ:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ .env файл в порядке")
        return True


def create_signature_v5(timestamp: str, api_key: str, recv_window: str, query_string: str) -> str:
    """
    Создание подписи для Bybit V5 API (ПРАВИЛЬНЫЙ метод).

    Формат: timestamp + api_key + recv_window + queryString
    """
    # ВАЖНО: Очищаем от пробелов и переносов строк
    api_key_clean = api_key.strip()

    # Собираем строку для подписи
    param_str = f"{timestamp}{api_key_clean}{recv_window}{query_string}"

    print("\n📝 Создание подписи (V5 метод):")
    print(f"  Timestamp: {timestamp}")
    print(f"  API Key (clean): {api_key_clean}")
    print(f"  API Key length: {len(api_key_clean)}")
    print(f"  Recv Window: {recv_window}")
    print(f"  Query String: {query_string}")
    print(f"  Param String: {param_str}")

    # ВАЖНО: Очищаем секрет от пробелов
    api_secret_clean = API_SECRET.strip()

    # Создаем HMAC SHA256 подпись
    signature = hmac.new(
        api_secret_clean.encode('utf-8'),
        param_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    print(f"  Signature: {signature}")

    return signature


async def test_public_endpoint():
    """Тест публичного эндпоинта (без аутентификации)"""
    print("\n" + "=" * 80)
    print("📡 ШАГ 2: Тест публичного эндпоинта")
    print("-" * 80)

    url = f"{BASE_URL}/v5/market/time"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                status = response.status
                data = await response.json()

                print(f"URL: {url}")
                print(f"Status: {status}")

                if status == 200 and data.get("retCode") == 0:
                    server_time = data.get("result", {}).get("timeSecond", "N/A")
                    print(f"✅ Сервер доступен! Time: {server_time}")
                    return True
                else:
                    print(f"❌ Ошибка сервера: {data}")
                    return False

    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False


async def test_authenticated_endpoint():
    """Тест приватного эндпоинта (с аутентификацией)"""
    print("\n" + "=" * 80)
    print("🔐 ШАГ 3: Тест приватного эндпоинта")
    print("-" * 80)

    if not API_KEY or not API_SECRET:
        print("❌ API ключи не настроены!")
        return False

    # Очищаем ключи от пробелов и переносов
    api_key_clean = API_KEY.strip()
    api_secret_clean = API_SECRET.strip()

    # Параметры
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    query_string = "accountType=UNIFIED"

    # Создаем подпись
    signature = create_signature_v5(timestamp, api_key_clean, recv_window, query_string)

    # Заголовки (используем ОЧИЩЕННЫЙ ключ)
    headers = {
        "X-BAPI-API-KEY": api_key_clean,
        "X-BAPI-SIGN": signature,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json"
    }

    url = f"{BASE_URL}/v5/account/wallet-balance"

    print(f"\n📤 Запрос:")
    print(f"  URL: {url}")
    print(f"  Query: {query_string}")
    print(f"  Headers:")
    print(f"    X-BAPI-API-KEY: {api_key_clean[:10]}...{api_key_clean[-4:]} (len={len(api_key_clean)})")
    print(f"    X-BAPI-SIGN: {signature[:20]}...")
    print(f"    X-BAPI-TIMESTAMP: {timestamp}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params={"accountType": "UNIFIED"},
                headers=headers,
                timeout=10
            ) as response:
                status = response.status

                print(f"\n📥 Ответ:")
                print(f"  Status: {status}")

                # Получаем ответ
                try:
                    data = await response.json()
                    print(f"  JSON: {data}")
                except:
                    text = await response.text()
                    print(f"  Text: {text[:500]}")
                    data = None

                if data and data.get("retCode") == 0:
                    print(f"\n✅ Успех! API ключ работает!")

                    # Показываем баланс
                    result = data.get("result", {})
                    wallet_list = result.get("list", [])
                    if wallet_list:
                        print(f"\n💰 Баланс:")
                        for wallet in wallet_list:
                            account_type = wallet.get("accountType", "N/A")
                            print(f"  Account Type: {account_type}")
                            coins = wallet.get("coin", [])
                            for coin in coins:
                                coin_name = coin.get("coin", "N/A")
                                balance = coin.get("walletBalance", "0")
                                available = coin.get("availableToWithdraw", "0")
                                if float(balance) > 0:
                                    print(f"    {coin_name}: {balance} (доступно: {available})")

                    return True

                elif data:
                    ret_code = data.get("retCode")
                    ret_msg = data.get("retMsg", "")

                    print(f"\n❌ Ошибка API:")
                    print(f"  Код: {ret_code}")
                    print(f"  Сообщение: {ret_msg}")

                    # Специфичная диагностика
                    if ret_code == 10003:
                        print(f"\n💡 Код 10003 = Invalid API key")
                        print(f"  Возможные причины:")
                        print(f"  1. Ключ от другого режима (у вас {BYBIT_MODE})")
                        print(f"  2. Ключ скопирован с ошибкой")
                        print(f"  3. Ключ деактивирован в личном кабинете")

                    elif ret_code == 10004:
                        print(f"\n💡 Код 10004 = Invalid signature")
                        print(f"  Возможные причины:")
                        print(f"  1. API Secret скопирован с ошибкой")
                        print(f"  2. Лишние пробелы в .env файле")
                        print(f"  3. Кавычки в значениях .env")

                    return False
                else:
                    print(f"❌ Не удалось распарсить ответ")
                    return False

    except Exception as e:
        print(f"❌ Ошибка запроса: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Главная функция"""

    # Шаг 1: Проверка .env
    env_ok = check_env_file()

    if not env_ok:
        print("\n" + "=" * 80)
        print("❌ КРИТИЧЕСКИЕ ОШИБКИ В .env ФАЙЛЕ")
        print("=" * 80)
        print("\n📝 Исправьте .env файл:")
        print(f"  1. Откройте файл: backend/.env")
        print(f"  2. Убедитесь что формат правильный:")
        print(f"")
        print(f"     BYBIT_MODE=testnet")
        print(f"     BYBIT_API_KEY=ваш_ключ_без_пробелов_и_кавычек")
        print(f"     BYBIT_API_SECRET=ваш_секрет_без_пробелов_и_кавычек")
        print(f"")
        print(f"  3. Убедитесь что:")
        print(f"     - НЕТ пробелов до/после '='")
        print(f"     - НЕТ кавычек вокруг значений")
        print(f"     - НЕТ лишних пробелов в ключах")
        print(f"")
        return

    # Шаг 2: Тест публичного API
    public_ok = await test_public_endpoint()

    if not public_ok:
        print("\n❌ Сервер Bybit недоступен")
        print("Проверьте интернет соединение")
        return

    # Шаг 3: Тест приватного API
    auth_ok = await test_authenticated_endpoint()

    # Итоги
    print("\n" + "=" * 80)
    print("📊 ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("=" * 80)
    print(f"  .env файл: {'✅' if env_ok else '❌'}")
    print(f"  Сервер доступен: {'✅' if public_ok else '❌'}")
    print(f"  API ключ работает: {'✅' if auth_ok else '❌'}")
    print("=" * 80)

    if not auth_ok:
        print("\n💡 СЛЕДУЮЩИЕ ШАГИ:")
        print(f"  1. Зайдите на: https://{BYBIT_MODE}.bybit.com")
        print(f"  2. API Management → Create New Key")
        print(f"  3. Включите разрешения:")
        print(f"     ✅ Read-Write")
        print(f"     ✅ Contract Trading")
        print(f"  4. Скопируйте ключи БЕЗ пробелов и кавычек")
        print(f"  5. Обновите .env файл")
        print(f"  6. Запустите: python test_api_key.py")


if __name__ == "__main__":
    asyncio.run(main())