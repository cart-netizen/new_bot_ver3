#!/usr/bin/env python3
"""
Скрипт полной диагностики ML интеграции.
Проверяет все компоненты перед запуском бота.

Запуск:
    python check_ml_integration_health.py
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime

# Добавляем backend в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))


def print_section(title):
  """Печать секции с разделителем."""
  print("\n" + "=" * 70)
  print(title)
  print("=" * 70)


def print_result(name, success, message=""):
  """Печать результата проверки."""
  status = "✓" if success else "✗"
  status_text = "PASS" if success else "FAIL"
  color = "\033[92m" if success else "\033[91m"
  reset = "\033[0m"

  print(f"{color}{status} {name}: {status_text}{reset}")
  if message:
    print(f"  {message}")


async def check_imports():
  """Проверка импорта всех модулей."""
  print_section("1. ПРОВЕРКА ИМПОРТОВ")
  results = []

  # CandleManager
  try:
    from backend.strategy.candle_manager import CandleManager
    print_result("CandleManager", True)
    results.append(("CandleManager", True))
  except Exception as e:
    print_result("CandleManager", False, str(e))
    results.append(("CandleManager", False))

  # MLDataCollector
  try:
    from backend.ml_engine.data_collection import MLDataCollector
    print_result("MLDataCollector", True)
    results.append(("MLDataCollector", True))
  except Exception as e:
    print_result("MLDataCollector", False, str(e))
    results.append(("MLDataCollector", False))

  # FeaturePipeline
  try:
    from backend.ml_engine.features import MultiSymbolFeaturePipeline
    print_result("MultiSymbolFeaturePipeline", True)
    results.append(("FeaturePipeline", True))
  except Exception as e:
    print_result("FeaturePipeline", False, str(e))
    results.append(("FeaturePipeline", False))

  # Candle Feature Extractor
  try:
    from backend.ml_engine.features.candle_feature_extractor import (
      CandleFeatureExtractor,
      Candle
    )
    print_result("CandleFeatureExtractor", True)

    # Проверка property typical_price
    test_candle = Candle(
      timestamp=1234567890000,
      open=50000.0,
      high=50100.0,
      low=49900.0,
      close=50050.0,
      volume=1.5
    )

    tp = test_candle.typical_price  # Должно работать!
    print(f"    typical_price property работает: {tp:.2f}")
    results.append(("CandleFeatureExtractor", True))

  except Exception as e:
    print_result("CandleFeatureExtractor", False, str(e))
    results.append(("CandleFeatureExtractor", False))

  # REST Client
  try:
    from backend.exchange.rest_client import rest_client
    print_result("REST Client", True)
    results.append(("REST Client Import", True))
  except Exception as e:
    print_result("REST Client", False, str(e))
    results.append(("REST Client Import", False))

  return results


async def check_rest_api():
  """Проверка REST API Bybit."""
  print_section("2. ПРОВЕРКА REST API")
  results = []

  try:
    from backend.exchange.rest_client import rest_client

    # Инициализация
    await rest_client.initialize()
    print_result("REST Client Initialize", True)

    # Server Time (публичный endpoint)
    try:
      server_time = await rest_client.get_server_time()
      print_result("get_server_time()", True, f"Time: {server_time}")
      results.append(("Server Time", True))
    except Exception as e:
      print_result("get_server_time()", False, str(e))
      results.append(("Server Time", False))

    # Kline (ПРАВИЛЬНЫЙ метод - единственное число!)
    try:
      candles = await rest_client.get_kline("BTCUSDT", "1", limit=5)
      print_result(
        "get_kline()",
        True,
        f"{len(candles)} свечей получено"
      )

      # Проверка формата
      if candles and len(candles) > 0:
        print(f"    Формат свечи: {candles[0]}")

        # Проверка что это list с 7 элементами
        if isinstance(candles[0], list) and len(candles[0]) >= 6:
          print(f"    ✓ Формат корректный [timestamp, O, H, L, C, V, ...]")
        else:
          print(f"    ⚠ Неожиданный формат")

      results.append(("get_kline", True))

    except Exception as e:
      print_result("get_kline()", False, str(e))
      results.append(("get_kline", False))

    await rest_client.close()

  except Exception as e:
    print_result("REST API", False, f"Общая ошибка: {e}")
    results.append(("REST API", False))

  return results


def check_directories():
  """Проверка структуры директорий."""
  print_section("3. ПРОВЕРКА ДИРЕКТОРИЙ")
  results = []

  required_dirs = [
    "backend/strategy",
    "backend/ml_engine/data_collection",
    "backend/ml_engine/features",
    "data/ml_training",
    "logs"
  ]

  required_files = [
    "backend/main.py",
    "backend/strategy/candle_manager.py",
    "backend/ml_engine/data_collection/__init__.py",
    "backend/ml_engine/data_collection/ml_data_collector.py",
    "backend/ml_engine/features/candle_feature_extractor.py",
    "backend/ml_engine/features/feature_pipeline.py"
  ]

  # Проверка директорий
  print("\nДиректории:")
  for dir_path in required_dirs:
    exists = Path(dir_path).exists()
    print_result(dir_path, exists)
    results.append((dir_path, exists))

  # Проверка файлов
  print("\nФайлы:")
  for file_path in required_files:
    exists = Path(file_path).exists()
    print_result(file_path, exists)
    results.append((file_path, exists))

  return results


async def check_candle_manager():
  """Проверка CandleManager."""
  print_section("4. ПРОВЕРКА CANDLE MANAGER")
  results = []

  try:
    from backend.strategy.candle_manager import CandleManager
    from backend.ml_engine.features.candle_feature_extractor import Candle

    # Создание manager
    manager = CandleManager("TEST", timeframe="1m", max_candles=100)
    print_result("CandleManager создан", True)

    # Тест добавления свечи (формат Bybit: [timestamp, O, H, L, C, V])
    test_data = [1234567890000, 50000.0, 50100.0, 49900.0, 50050.0, 1.5]
    await manager.update_candle(test_data, is_closed=True)

    count = manager.get_candles_count()
    print_result("update_candle()", True, f"Добавлено свечей: {count}")

    # Проверка получения свечей
    candles = manager.get_candles()
    if candles and len(candles) > 0:
      print_result("get_candles()", True, f"Получено: {len(candles)} свечей")
      print(f"    Последняя свеча: close={candles[-1].close}")
    else:
      print_result("get_candles()", False, "Нет свечей")

    # Проверка статистики
    stats = manager.get_statistics()
    print(f"    Статистика: {stats['candles_count']} свечей в истории")

    results.append(("CandleManager", True))

  except Exception as e:
    print_result("CandleManager", False, str(e))
    import traceback
    traceback.print_exc()
    results.append(("CandleManager", False))

  return results


async def check_ml_data_collector():
  """Проверка MLDataCollector."""
  print_section("5. ПРОВЕРКА ML DATA COLLECTOR")
  results = []

  try:
    from backend.ml_engine.data_collection import MLDataCollector

    # Создание collector
    test_path = Path("data/ml_training_test")
    if test_path.exists():
      import shutil
      shutil.rmtree(test_path)

    collector = MLDataCollector(
      storage_path=str(test_path),
      max_samples_per_file=100,
      collection_interval=1
    )

    await collector.initialize()
    print_result("MLDataCollector создан", True)
    print(f"    Storage path: {collector.storage_path}")

    # Проверка директории создана
    if test_path.exists():
      print_result("Директория создана", True, str(test_path))
    else:
      print_result("Директория создана", False)

    # Проверка статистики
    stats = collector.get_statistics()
    print(f"    Статистика: {stats}")

    results.append(("MLDataCollector", True))

    # Cleanup
    if test_path.exists():
      import shutil
      shutil.rmtree(test_path)

  except Exception as e:
    print_result("MLDataCollector", False, str(e))
    import traceback
    traceback.print_exc()
    results.append(("MLDataCollector", False))

  return results


async def check_feature_extraction():
  """Проверка извлечения признаков."""
  print_section("6. ПРОВЕРКА FEATURE EXTRACTION")
  results = []

  try:
    from backend.ml_engine.features.candle_feature_extractor import (
      CandleFeatureExtractor,
      Candle
    )
    from backend.ml_engine.features import FeaturePipeline
    from backend.models.orderbook import OrderBookSnapshot
    import numpy as np

    # Создание тестовых данных
    candles = []
    base_time = int(datetime.now().timestamp() * 1000)

    for i in range(60):
      price = 50000.0 + np.sin(i / 5) * 100
      candles.append(Candle(
        timestamp=base_time - (60 - i) * 60000,
        open=price,
        high=price + np.random.uniform(10, 30),
        low=price - np.random.uniform(10, 30),
        close=price + np.random.uniform(-20, 20),
        volume=np.random.uniform(0.5, 2.0)
      ))

    print(f"Создано {len(candles)} тестовых свечей")

    # Тест CandleFeatureExtractor
    candle_extractor = CandleFeatureExtractor("TEST")
    candle_features = candle_extractor.extract(candles[-1], candles[-2])

    print_result(
      "CandleFeatureExtractor",
      True,
      f"Извлечено {len(candle_features.to_array())} признаков"
    )

    # Тест FeaturePipeline
    orderbook = OrderBookSnapshot(
      symbol="TEST",
      bids=[(50000.0, 1.5), (49999.0, 2.0)],
      asks=[(50001.0, 1.2), (50002.0, 1.8)],
      timestamp=base_time
    )

    pipeline = FeaturePipeline("TEST", normalize=False)
    feature_vector = await pipeline.extract_features(
      orderbook_snapshot=orderbook,
      candles=candles
    )

    print_result(
      "FeaturePipeline",
      True,
      f"Извлечено {feature_vector.feature_count} признаков"
    )

    # Проверка массива
    features_array = feature_vector.to_array()
    print(f"    Shape: {features_array.shape}")
    print(f"    No NaN: {not np.any(np.isnan(features_array))}")
    print(f"    No Inf: {not np.any(np.isinf(features_array))}")

    results.append(("Feature Extraction", True))

  except Exception as e:
    print_result("Feature Extraction", False, str(e))
    import traceback
    traceback.print_exc()
    results.append(("Feature Extraction", False))

  return results


async def main():
  """Главная функция."""
  print("\n" + "=" * 70)
  print("ML INTEGRATION HEALTH CHECK")
  print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
  print("=" * 70)

  all_results = []

  # Запуск всех проверок
  all_results.extend(await check_imports())
  all_results.extend(await check_rest_api())
  all_results.extend(check_directories())
  all_results.extend(await check_candle_manager())
  all_results.extend(await check_ml_data_collector())
  all_results.extend(await check_feature_extraction())

  # Итоги
  print_section("ИТОГИ ПРОВЕРКИ")

  passed = sum(1 for _, result in all_results if result)
  total = len(all_results)

  for name, result in all_results:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status} | {name}")

  print(f"\nПройдено: {passed}/{total}")
  print(f"Процент успеха: {passed / total * 100:.1f}%")

  if passed == total:
    print("\n" + "=" * 70)
    print("🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
    print("=" * 70)
    print("\nСистема готова к запуску:")
    print("  python backend/main.py")
    return 0
  else:
    print("\n" + "=" * 70)
    print("⚠️  ОБНАРУЖЕНЫ ПРОБЛЕМЫ")
    print("=" * 70)
    print("\nИсправьте ошибки выше перед запуском бота.")
    print("См. ML_INTEGRATION_TROUBLESHOOTING.md для решений.")
    return 1


if __name__ == "__main__":
  try:
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
  except KeyboardInterrupt:
    print("\n\nПроверка прервана пользователем")
    sys.exit(1)
  except Exception as e:
    print(f"\n\nКритическая ошибка: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)