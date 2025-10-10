#!/usr/bin/env python3
"""
Комплексная диагностика бэкенда.
Запустите: python diagnose_backend.py
"""

import asyncio
import sys
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("🔍 ДИАГНОСТИКА БЭКЕНДА")
print("=" * 80)


async def main():
  issues = []

  # 1. Проверка .env файла
  print("\n1️⃣ Проверка .env файла...")
  from dotenv import load_dotenv, find_dotenv
  import os

  env_path = find_dotenv()
  if env_path:
    print(f"   ✅ .env найден: {env_path}")
    load_dotenv()

    # Проверяем критические переменные
    critical_vars = {
      "SECRET_KEY": os.getenv("SECRET_KEY", ""),
      "APP_PASSWORD": os.getenv("APP_PASSWORD", ""),
      "BYBIT_MODE": os.getenv("BYBIT_MODE", ""),
      "BYBIT_API_KEY": os.getenv("BYBIT_API_KEY", ""),
      "BYBIT_API_SECRET": os.getenv("BYBIT_API_SECRET", ""),
    }

    for var, value in critical_vars.items():
      if not value:
        print(f"   ❌ {var}: НЕ ЗАДАНО")
        issues.append(f"{var} не задано в .env")
      else:
        display = f"{value[:10]}...{value[-4:]}" if len(value) > 14 else "***"
        if "KEY" in var or "SECRET" in var or "PASSWORD" in var:
          print(f"   ✅ {var}: {display}")
        else:
          print(f"   ✅ {var}: {value}")
  else:
    print("   ❌ .env файл НЕ найден!")
    issues.append(".env файл не найден")
    return issues

  # 2. Проверка config
  print("\n2️⃣ Проверка config.py...")
  try:
    from config import settings
    print(f"   ✅ Config загружен")
    print(f"   Режим Bybit: {settings.BYBIT_MODE}")
    print(f"   Торговые пары: {settings.TRADING_PAIRS}")
    print(f"   API Host: {settings.API_HOST}:{settings.API_PORT}")
  except Exception as e:
    print(f"   ❌ Ошибка загрузки config: {e}")
    issues.append(f"Ошибка config: {e}")
    return issues

  # 3. Проверка REST клиента
  print("\n3️⃣ Проверка REST клиента...")
  try:
    from exchange.rest_client import rest_client

    # Проверяем инициализацию
    if not rest_client.api_key or not rest_client.api_secret:
      print(f"   ⚠️  REST клиент НЕ инициализирован (API ключи пустые)")
      print(f"   Это нормально если вы еще не настроили API ключи")
      print(f"   Страница Account работать не будет")
    else:
      print(f"   ✅ REST клиент инициализирован")

      # Пробуем получить серверное время
      try:
        await rest_client.initialize()
        server_time = await rest_client.get_server_time()
        print(f"   ✅ Подключение к Bybit работает")
        print(f"   Серверное время: {server_time}")
      except Exception as e:
        print(f"   ❌ Ошибка подключения к Bybit: {e}")
        issues.append(f"Bybit подключение: {e}")
  except Exception as e:
    print(f"   ❌ Ошибка загрузки REST клиента: {e}")
    issues.append(f"REST клиент: {e}")

  # 4. Проверка BotController
  print("\n4️⃣ Проверка BotController...")
  try:
    from main import bot_controller

    if bot_controller is None:
      print(f"   ⚠️  bot_controller = None (бэкенд не запущен)")
      print(f"   Запустите: python main.py")
      issues.append("bot_controller не инициализирован")
    else:
      print(f"   ✅ BotController инициализирован")
      print(f"   Статус: {bot_controller.status}")

      if bot_controller.status.value != "running":
        print(f"   ⚠️  Бот НЕ запущен")
        print(f"   Запустите бота через UI или API")
  except ImportError:
    print(f"   ⚠️  Не удалось импортировать main.py")
    print(f"   Это нормально если бэкенд не запущен")
  except Exception as e:
    print(f"   ❌ Ошибка проверки BotController: {e}")

  # 5. Проверка WebSocket manager
  print("\n5️⃣ Проверка WebSocket...")
  try:
    from api.websocket import manager as ws_manager

    stats = ws_manager.get_stats()
    print(f"   ✅ WebSocket manager инициализирован")
    print(f"   Всего подключений: {stats['total_connections']}")
    print(f"   Аутентифицированных: {stats['authenticated_connections']}")

    if stats['total_connections'] == 0:
      print(f"   ⚠️  Нет активных WebSocket подключений")
      print(f"   Откройте фронтенд в браузере: http://localhost:5173")
  except Exception as e:
    print(f"   ❌ Ошибка WebSocket manager: {e}")
    issues.append(f"WebSocket: {e}")

  # 6. Проверка Balance Tracker
  print("\n6️⃣ Проверка Balance Tracker...")
  try:
    from utils.balance_tracker import balance_tracker

    print(f"   ✅ Balance Tracker загружен")
    print(f"   Запущен: {balance_tracker.is_running}")

    history = balance_tracker.get_history("24h")
    print(f"   Записей в истории: {len(history)}")

    if len(history) == 0:
      print(f"   ⚠️  История баланса пустая")
      print(f"   Подождите 1-2 минуты после запуска бота")
  except Exception as e:
    print(f"   ❌ Ошибка Balance Tracker: {e}")
    issues.append(f"Balance Tracker: {e}")

  return issues


# Запуск
try:
  issues = asyncio.run(main())

  # Итоги
  print("\n" + "=" * 80)
  print("📊 ИТОГИ ДИАГНОСТИКИ")
  print("=" * 80)

  if not issues:
    print("\n✅ Все проверки пройдены!")
    print("\nЕсли фронтенд все еще показывает 'Отключено':")
    print("  1. Перезапустите фронтенд: npm run dev")
    print("  2. Обновите страницу в браузере (Ctrl+F5)")
    print("  3. Проверьте консоль браузера (F12)")
  else:
    print(f"\n❌ Найдено проблем: {len(issues)}")
    for i, issue in enumerate(issues, 1):
      print(f"  {i}. {issue}")

    print("\n🔧 РЕКОМЕНДАЦИИ:")

    if any("API ключи" in issue or "BYBIT_API_KEY" in issue for issue in issues):
      print("\n  📝 Настройте API ключи в .env:")
      print("     BYBIT_API_KEY=ваш_ключ")
      print("     BYBIT_API_SECRET=ваш_секрет")

    if any(".env" in issue for issue in issues):
      print("\n  📝 Создайте файл .env:")
      print("     cp .env.example .env")

    if any("bot_controller" in issue for issue in issues):
      print("\n  🚀 Запустите бэкенд:")
      print("     python main.py")

    if any("WebSocket" in issue for issue in issues):
      print("\n  🌐 Откройте фронтенд:")
      print("     http://localhost:5173")

  print("\n" + "=" * 80)

except KeyboardInterrupt:
  print("\n\n⚠️  Диагностика прервана")
except Exception as e:
  print(f"\n\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
  import traceback

  traceback.print_exc()