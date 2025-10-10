#!/usr/bin/env python3
"""
Скрипт для проверки откуда загружается .env файл.
Запустите из папки backend: python check_env_location.py
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("🔍 ПРОВЕРКА РАСПОЛОЖЕНИЯ .env ФАЙЛА")
print("=" * 80)

# 1. Текущая рабочая директория
cwd = Path.cwd()
print(f"\n1️⃣ Текущая рабочая директория (откуда запущен скрипт):")
print(f"   {cwd}")

# 2. Директория этого скрипта
script_dir = Path(__file__).parent.resolve()
print(f"\n2️⃣ Директория этого скрипта:")
print(f"   {script_dir}")

# 3. Где находится config.py
try:
  import config

  config_file = Path(config.__file__).resolve()
  config_dir = config_file.parent
  print(f"\n3️⃣ Директория config.py:")
  print(f"   {config_dir}")
except ImportError:
  print(f"\n3️⃣ Директория config.py:")
  print(f"   ⚠️  Не удалось импортировать config.py")
  config_dir = None

# 4. Проверка существования .env файлов
print(f"\n4️⃣ Проверка существования .env файлов:")

locations_to_check = [
  ("Текущая директория", cwd / ".env"),
  ("Директория скрипта", script_dir / ".env"),
]

if config_dir:
  locations_to_check.append(("Директория config.py", config_dir / ".env"))

# Добавляем родительскую директорию (корень проекта)
parent_dir = cwd.parent
locations_to_check.append(("Родительская директория (корень проекта)", parent_dir / ".env"))

found_env_files = []

for name, path in locations_to_check:
  exists = path.exists()
  status = "✅ НАЙДЕН" if exists else "❌ НЕ НАЙДЕН"
  print(f"\n   {status}")
  print(f"   Путь: {path}")
  print(f"   Расположение: {name}")

  if exists:
    found_env_files.append((name, path))
    # Показываем первые несколько строк
    try:
      with open(path, 'r') as f:
        lines = f.readlines()[:3]
        print(f"   Первые строки:")
        for line in lines:
          print(f"      {line.rstrip()}")
    except Exception as e:
      print(f"   ⚠️  Ошибка чтения: {e}")

# 5. Откуда load_dotenv() загрузит .env
print(f"\n5️⃣ Откуда load_dotenv() загрузит .env:")
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
if dotenv_path:
  print(f"   ✅ Найден: {dotenv_path}")
else:
  print(f"   ❌ НЕ найден")

# 6. Загрузка и проверка переменных
print(f"\n6️⃣ Загрузка переменных окружения:")
load_dotenv()

test_vars = [
  "BYBIT_MODE",
  "BYBIT_API_KEY",
  "BYBIT_API_SECRET",
  "APP_PASSWORD",
  "SECRET_KEY",
]

print(f"\n   Переменные из окружения:")
for var in test_vars:
  value = os.getenv(var, "")
  if value:
    # Скрываем секреты
    if "SECRET" in var or "PASSWORD" in var or "KEY" in var:
      display = f"{value[:10]}...{value[-4:]}" if len(value) > 14 else "***"
    else:
      display = value
    print(f"      {var}: {display} ✅")
  else:
    print(f"      {var}: (не задано) ❌")

# 7. Рекомендации
print(f"\n7️⃣ РЕКОМЕНДАЦИИ:")
print("=" * 80)

if len(found_env_files) == 0:
  print("\n❌ ФАЙЛ .env НЕ НАЙДЕН НИ В ОДНОЙ ИЗ ПРОВЕРЕННЫХ ДИРЕКТОРИЙ!")
  print("\n📝 Создайте файл .env:")
  print(f"   Рекомендуемое расположение: {script_dir / '.env'}")
  print(f"\n   Скопируйте .env.example:")
  print(f"   cp .env.example .env")

elif len(found_env_files) == 1:
  name, path = found_env_files[0]
  print(f"\n✅ Найден один .env файл:")
  print(f"   Расположение: {name}")
  print(f"   Путь: {path}")
  print(f"\n   Это правильное расположение!")

else:
  print(f"\n⚠️  НАЙДЕНО НЕСКОЛЬКО .env ФАЙЛОВ:")
  for name, path in found_env_files:
    print(f"   - {name}: {path}")

  print(f"\n   load_dotenv() использует первый найденный:")
  print(f"   {dotenv_path}")

  print(f"\n   🔧 Рекомендация:")
  print(f"   Удалите лишние .env файлы и оставьте только один:")
  print(f"   {script_dir / '.env'}")

# 8. Команда для запуска
print(f"\n8️⃣ КАК ПРАВИЛЬНО ЗАПУСКАТЬ:")
print("=" * 80)

if script_dir.name == "backend":
  print(f"\n   ✅ ПРАВИЛЬНО (из папки backend):")
  print(f"   cd backend")
  print(f"   python main.py")
  print(f"\n   Файл .env должен быть: backend/.env")

  print(f"\n   ❌ НЕПРАВИЛЬНО (из корня проекта):")
  print(f"   python backend/main.py")
  print(f"   (будет искать .env в корне, а не в backend/)")
else:
  print(f"\n   Скрипт запущен из: {script_dir}")

print("\n" + "=" * 80)
print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
print("=" * 80)