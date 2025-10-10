"""
Проверка точных значений из .env файла
"""

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")

print("=" * 80)
print("🔍 ПРОВЕРКА ЗНАЧЕНИЙ ИЗ .env")
print("=" * 80)

print("\nAPI_KEY:")
print(f"  Значение: {API_KEY}")
print(f"  Длина: {len(API_KEY)}")
print(f"  Есть пробелы в начале: {API_KEY != API_KEY.lstrip()}")
print(f"  Есть пробелы в конце: {API_KEY != API_KEY.rstrip()}")
print(f"  Есть переносы строк: {repr(API_KEY)}")

print("\nAPI_SECRET:")
print(f"  Значение: {API_SECRET}")
print(f"  Длина: {len(API_SECRET)}")
print(f"  Есть пробелы в начале: {API_SECRET != API_SECRET.lstrip()}")
print(f"  Есть пробелы в конце: {API_SECRET != API_SECRET.rstrip()}")
print(f"  Есть переносы строк: {repr(API_SECRET)}")

print("\n" + "=" * 80)
print("ОЖИДАЕМЫЕ ЗНАЧЕНИЯ (со скриншота):")
print("=" * 80)

expected_key = "M3607daMPXdu6q170A"
expected_secret = "js7gGVSAmbhemPUyCduYwzZPHGfey600GQkr"

print(f"\nОжидаемый API_KEY: {expected_key}")
print(f"Ваш API_KEY:       {API_KEY}")
print(f"Совпадает: {API_KEY.strip() == expected_key}")

print(f"\nОжидаемый API_SECRET: {expected_secret}")
print(f"Ваш API_SECRET:       {API_SECRET}")
print(f"Совпадает: {API_SECRET.strip() == expected_secret}")

# Посимвольное сравнение
if API_KEY.strip() != expected_key:
    print("\n🚨 API_KEY НЕ СОВПАДАЕТ!")
    print("Посимвольная разница:")
    for i, (a, b) in enumerate(zip(API_KEY.strip(), expected_key)):
        if a != b:
            print(f"  Позиция {i}: ваш='{a}' ожидаемый='{b}'")

if API_SECRET.strip() != expected_secret:
    print("\n🚨 API_SECRET НЕ СОВПАДАЕТ!")
    print("Посимвольная разница:")
    for i, (a, b) in enumerate(zip(API_SECRET.strip(), expected_secret)):
        if a != b:
            print(f"  Позиция {i}: ваш='{a}' ожидаемый='{b}'")

print("\n" + "=" * 80)
print("💡 РЕКОМЕНДАЦИИ:")
print("=" * 80)

if API_KEY.strip() == expected_key and API_SECRET.strip() == expected_secret:
    print("\n✅ Ключи в .env совпадают со скриншотом!")
    print("\nВозможные причины ошибки 401:")
    print("  1. Ключ еще не активирован (подождите 1-2 минуты)")
    print("  2. Нужно пересоздать ключ с другими настройками")
    print("  3. Проблема на стороне Bybit testnet")
else:
    print("\n❌ Ключи НЕ совпадают!")
    print("\nИсправьте .env файл:")
    print(f"\nПравильное содержимое .env:")
    print(f"BYBIT_MODE=testnet")
    print(f"BYBIT_API_KEY={expected_key}")
    print(f"BYBIT_API_SECRET={expected_secret}")