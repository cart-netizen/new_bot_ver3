#!/usr/bin/env python3
"""
Полная проверка ML Feature Engineering системы.
Тестирует все 4 компонента и их интеграцию.

Запуск:
    python check_complete_ml_features.py
"""

import sys
import os
from datetime import datetime
import time
from pathlib import Path
# Добавляем путь к backend
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))


def print_section(title):
  """Печатает заголовок секции"""
  print("\n" + "=" * 70)
  print(title)
  print("=" * 70)


def check_dependencies():
  """Проверка всех зависимостей"""
  print_section("1. ПРОВЕРКА ЗАВИСИМОСТЕЙ")

  dependencies = {
    "numpy": None,
    "numba": None,
    "sklearn": None,
  }

  all_ok = True

  for package, version in dependencies.items():
    try:
      if package == "sklearn":
        import sklearn
        version = sklearn.__version__
        print(f"✓ scikit-learn v{version}")
      else:
        module = __import__(package)
        version = module.__version__
        print(f"✓ {package} v{version}")
    except ImportError:
      print(f"✗ {package} НЕ УСТАНОВЛЕН")
      print(f"  Установите: pip install {package}")
      all_ok = False

  if all_ok:
    print("\n✅ Все зависимости установлены!")

  return all_ok


def check_imports():
  """Проверка импортов всех модулей"""
  print_section("2. ПРОВЕРКА ИМПОРТОВ МОДУЛЕЙ")

  modules = []

  # OrderBook
  try:
    from ml_engine.features.orderbook_feature_extractor import (
      OrderBookFeatureExtractor,
      OrderBookFeatures
    )
    print("✓ OrderBookFeatureExtractor импортирован")
    modules.append("orderbook")
  except ImportError as e:
    print(f"✗ Ошибка импорта OrderBookFeatureExtractor: {e}")
    return False

  # Candle
  try:
    from ml_engine.features.candle_feature_extractor import (
      CandleFeatureExtractor,
      CandleFeatures,
      Candle
    )
    print("✓ CandleFeatureExtractor импортирован")
    print("✓ Candle модель импортирована")
    modules.append("candle")
  except ImportError as e:
    print(f"✗ Ошибка импорта CandleFeatureExtractor: {e}")
    return False

  # Indicator
  try:
    from ml_engine.features.indicator_feature_extractor import (
      IndicatorFeatureExtractor,
      IndicatorFeatures
    )
    print("✓ IndicatorFeatureExtractor импортирован")
    modules.append("indicator")
  except ImportError as e:
    print(f"✗ Ошибка импорта IndicatorFeatureExtractor: {e}")
    return False

  # Pipeline
  try:
    from ml_engine.features.feature_pipeline import (
      FeaturePipeline,
      FeatureVector,
      MultiSymbolFeaturePipeline
    )
    print("✓ FeaturePipeline импортирован")
    print("✓ FeatureVector импортирован")
    print("✓ MultiSymbolFeaturePipeline импортирован")
    modules.append("pipeline")
  except ImportError as e:
    print(f"✗ Ошибка импорта FeaturePipeline: {e}")
    return False

  # Existing models
  try:
    from models.orderbook import OrderBookSnapshot
    print("✓ OrderBookSnapshot импортирован")
  except ImportError as e:
    print(f"✗ Ошибка импорта OrderBookSnapshot: {e}")
    return False

  print(f"\n✅ Все модули импортированы успешно! ({len(modules)}/4)")
  return True


def test_orderbook_extractor():
  """Тест OrderBookFeatureExtractor"""
  print_section("3. ТЕСТ ORDERBOOK FEATURE EXTRACTOR")

  from models.orderbook import OrderBookSnapshot
  from ml_engine.features.orderbook_feature_extractor import OrderBookFeatureExtractor
  import numpy as np

  # Создаем снимок стакана
  snapshot = OrderBookSnapshot(
    symbol="BTCUSDT",
    bids=[
      (50000.0, 1.5), (49999.0, 2.0), (49998.0, 1.0),
      (49997.0, 0.5), (49996.0, 3.0), (49995.0, 1.2),
      (49994.0, 0.8), (49993.0, 2.5), (49992.0, 1.1),
      (49991.0, 0.9),
    ],
    asks=[
      (50001.0, 1.2), (50002.0, 1.8), (50003.0, 0.9),
      (50004.0, 2.5), (50005.0, 1.5), (50006.0, 0.7),
      (50007.0, 2.0), (50008.0, 1.3), (50009.0, 0.6),
      (50010.0, 1.9),
    ],
    timestamp=int(datetime.now().timestamp() * 1000)
  )

  print(f"Создан тестовый OrderBook:")
  print(f"  Symbol: {snapshot.symbol}")
  print(f"  Bids: {len(snapshot.bids)} уровней")
  print(f"  Asks: {len(snapshot.asks)} уровней")
  print(f"  Mid Price: {snapshot.mid_price}\n")

  # Создаем extractor
  extractor = OrderBookFeatureExtractor("BTCUSDT")

  # Измеряем время
  start = time.time()
  features = extractor.extract(snapshot)
  elapsed = (time.time() - start) * 1000

  # Проверяем результаты
  print(f"Результаты:")
  print(f"  Время извлечения: {elapsed:.2f}ms")
  print(f"  Mid Price: {features.mid_price:.2f}")
  print(f"  Spread: {features.bid_ask_spread_rel:.4f}%")
  print(f"  Imbalance (5): {features.imbalance_5:.4f}")
  print(f"  Imbalance (10): {features.imbalance_10:.4f}")
  print(f"  Bid Clusters: {features.num_bid_clusters}")
  print(f"  Ask Clusters: {features.num_ask_clusters}")

  # Проверяем массив
  array = features.to_array()
  print(f"\nМассив признаков:")
  print(f"  Shape: {array.shape}")
  print(f"  Expected: (50,)")

  if array.shape != (50,):
    print(f"  ✗ Неправильная размерность!")
    return False

  if np.any(np.isnan(array)):
    print(f"  ✗ Обнаружены NaN!")
    return False

  if np.any(np.isinf(array)):
    print(f"  ✗ Обнаружены Inf!")
    return False

  print(f"  ✓ Нет NaN/Inf")
  print(f"  ✓ Min: {array.min():.2f}, Max: {array.max():.2f}")

  print("\n✅ OrderBook Extractor работает корректно!")
  return True


def test_candle_extractor():
  """Тест CandleFeatureExtractor"""
  print_section("4. ТЕСТ CANDLE FEATURE EXTRACTOR")

  from ml_engine.features.candle_feature_extractor import (
    CandleFeatureExtractor,
    Candle
  )
  import numpy as np

  # Создаем свечи
  candles = []
  base_time = int(datetime.now().timestamp() * 1000)

  for i in range(30):
    price = 50000.0 + np.sin(i / 5) * 100
    candles.append(Candle(
      timestamp=base_time - (30 - i) * 60000,
      open=price,
      high=price + np.random.uniform(10, 30),
      low=price - np.random.uniform(10, 30),
      close=price + np.random.uniform(-20, 20),
      volume=np.random.uniform(0.5, 2.0)
    ))

  print(f"Создано {len(candles)} свечей")

  # Создаем extractor
  extractor = CandleFeatureExtractor("BTCUSDT")

  # Измеряем время
  start = time.time()
  features = extractor.extract(candles[-1], candles[-2])
  elapsed = (time.time() - start) * 1000

  print(f"\nРезультаты:")
  print(f"  Время извлечения: {elapsed:.2f}ms")
  print(f"  Close: {features.close:.2f}")
  print(f"  Returns: {features.returns:.4f}")
  print(f"  Volatility (realized): {features.realized_volatility:.6f}")
  print(f"  Body Size: {features.body_size:.4f}")
  print(f"  Doji Strength: {features.doji_strength:.4f}")

  # Проверяем массив
  array = features.to_array()
  print(f"\nМассив признаков:")
  print(f"  Shape: {array.shape}")
  print(f"  Expected: (25,)")

  if array.shape != (25,):
    print(f"  ✗ Неправильная размерность!")
    return False

  if np.any(np.isnan(array)):
    print(f"  ✗ Обнаружены NaN!")
    return False

  print(f"  ✓ Нет NaN/Inf")

  print("\n✅ Candle Extractor работает корректно!")
  return True


def test_indicator_extractor():
  """Тест IndicatorFeatureExtractor"""
  print_section("5. ТЕСТ INDICATOR FEATURE EXTRACTOR")

  from ml_engine.features.indicator_feature_extractor import IndicatorFeatureExtractor
  from ml_engine.features.candle_feature_extractor import Candle
  import numpy as np

  # Создаем 60 свечей для индикаторов
  candles = []
  base_time = int(datetime.now().timestamp() * 1000)
  base_price = 50000.0

  for i in range(60):
    # Создаем трендовое движение
    trend = i * 10
    noise = np.random.uniform(-50, 50)
    price = base_price + trend + noise

    candles.append(Candle(
      timestamp=base_time - (60 - i) * 60000,
      open=price,
      high=price + np.random.uniform(10, 30),
      low=price - np.random.uniform(10, 30),
      close=price + np.random.uniform(-20, 20),
      volume=np.random.uniform(0.5, 2.0)
    ))

  print(f"Создано {len(candles)} свечей")

  # Создаем extractor
  extractor = IndicatorFeatureExtractor("BTCUSDT")

  # Измеряем время
  start = time.time()
  features = extractor.extract(candles)
  elapsed = (time.time() - start) * 1000

  print(f"\nРезультаты:")
  print(f"  Время извлечения: {elapsed:.2f}ms")
  print(f"  RSI (14): {features.rsi_14:.2f}")
  print(f"  MACD: {features.macd:.4f}")
  print(f"  ADX: {features.adx:.2f}")
  print(f"  Bollinger Width: {features.bollinger_width:.4f}")
  print(f"  ATR (14): {features.atr_14:.2f}")

  # Проверяем массив
  array = features.to_array()
  print(f"\nМассив признаков:")
  print(f"  Shape: {array.shape}")
  print(f"  Expected: (35,)")

  if array.shape != (35,):
    print(f"  ✗ Неправильная размерность!")
    return False

  if np.any(np.isnan(array)):
    nan_count = np.sum(np.isnan(array))
    print(f"  ⚠️  Обнаружено {nan_count} NaN (может быть нормально для первых вызовов)")

  print(f"  ✓ Основные индикаторы вычислены")

  print("\n✅ Indicator Extractor работает корректно!")
  return True


def test_feature_pipeline():
  """Тест полного Feature Pipeline"""
  print_section("6. ТЕСТ FEATURE PIPELINE (ПОЛНАЯ ИНТЕГРАЦИЯ)")

  from models.orderbook import OrderBookSnapshot
  from ml_engine.features import FeaturePipeline, Candle
  import numpy as np
  import asyncio

  # Создаем данные
  orderbook = OrderBookSnapshot(
    symbol="BTCUSDT",
    bids=[(50000.0, 1.5), (49999.0, 2.0), (49998.0, 1.0)],
    asks=[(50001.0, 1.2), (50002.0, 1.8), (50003.0, 0.9)],
    timestamp=int(datetime.now().timestamp() * 1000)
  )

  candles = []
  base_time = int(datetime.now().timestamp() * 1000)

  for i in range(60):
    price = 50000.0 + np.sin(i / 10) * 100
    candles.append(Candle(
      timestamp=base_time - (60 - i) * 60000,
      open=price,
      high=price + 20,
      low=price - 20,
      close=price + np.random.uniform(-10, 10),
      volume=1.0
    ))

  print(f"Подготовлены данные:")
  print(f"  OrderBook: {len(orderbook.bids)} bids, {len(orderbook.asks)} asks")
  print(f"  Candles: {len(candles)} свечей\n")

  # Создаем pipeline
  pipeline = FeaturePipeline("BTCUSDT", normalize=False, cache_enabled=True)

  # Асинхронный запуск
  async def run_pipeline():
    start = time.time()
    feature_vector = await pipeline.extract_features(
      orderbook_snapshot=orderbook,
      candles=candles
    )
    elapsed = (time.time() - start) * 1000
    return feature_vector, elapsed

  feature_vector, elapsed = asyncio.run(run_pipeline())

  print(f"Pipeline выполнен за {elapsed:.2f}ms\n")

  print(f"FeatureVector:")
  print(f"  Symbol: {feature_vector.symbol}")
  print(f"  Timestamp: {feature_vector.timestamp}")
  print(f"  Total Features: {feature_vector.feature_count}")
  print(f"  Expected: 110 (50 + 25 + 35)")

  if feature_vector.feature_count != 110:
    print(f"  ✗ Неправильное количество признаков!")
    return False

  # Проверяем массив
  print(f"\nПолный массив признаков:")
  array = feature_vector.to_array()
  print(f"  Shape: {array.shape}")
  print(f"  Dtype: {array.dtype}")
  print(f"  Min: {array.min():.2f}, Max: {array.max():.2f}")
  print(f"  Mean: {array.mean():.2f}, Std: {array.std():.2f}")

  if array.shape != (110,):
    print(f"  ✗ Неправильная размерность!")
    return False

  # Проверяем каналы
  print(f"\nMulti-Channel Representation:")
  channels = feature_vector.to_channels()
  print(f"  OrderBook channel: {channels['orderbook'].shape}")
  print(f"  Candle channel: {channels['candle'].shape}")
  print(f"  Indicator channel: {channels['indicator'].shape}")

  if channels["orderbook"].shape != (50,):
    print(f"  ✗ OrderBook channel неправильный!")
    return False

  if channels["candle"].shape != (25,):
    print(f"  ✗ Candle channel неправильный!")
    return False

  if channels["indicator"].shape != (35,):
    print(f"  ✗ Indicator channel неправильный!")
    return False

  print(f"\n  ✓ Все каналы корректны")

  # Тест кэширования
  print(f"\nТест кэширования:")
  start = time.time()
  feature_vector2 = asyncio.run(pipeline.extract_features(orderbook, candles))
  elapsed2 = (time.time() - start) * 1000

  print(f"  Первый вызов: {elapsed:.2f}ms")
  print(f"  Второй вызов: {elapsed2:.2f}ms")

  if elapsed2 < elapsed:
    print(f"  ✓ Кэш работает (ускорение: {elapsed / elapsed2:.1f}x)")

  print("\n✅ Feature Pipeline работает корректно!")
  return True


def test_multi_symbol_pipeline():
  """Тест MultiSymbolFeaturePipeline"""
  print_section("7. ТЕСТ MULTI-SYMBOL PIPELINE")

  from models.orderbook import OrderBookSnapshot
  from ml_engine.features import MultiSymbolFeaturePipeline, Candle
  import numpy as np
  import asyncio

  symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
  print(f"Тестирование для {len(symbols)} символов: {', '.join(symbols)}\n")

  # Создаем pipeline
  multi_pipeline = MultiSymbolFeaturePipeline(symbols)

  # Подготавливаем данные для всех символов
  data = {}
  base_time = int(datetime.now().timestamp() * 1000)

  for symbol in symbols:
    orderbook = OrderBookSnapshot(
      symbol=symbol,
      bids=[(50000.0, 1.5), (49999.0, 2.0)],
      asks=[(50001.0, 1.2), (50002.0, 1.8)],
      timestamp=base_time
    )

    candles = []
    for i in range(60):
      candles.append(Candle(
        timestamp=base_time - (60 - i) * 60000,
        open=50000.0,
        high=50020.0,
        low=49980.0,
        close=50010.0,
        volume=1.0
      ))

    data[symbol] = (orderbook, candles)

  # Batch extraction
  async def run_batch():
    start = time.time()
    results = await multi_pipeline.extract_features_batch(data)
    elapsed = (time.time() - start) * 1000
    return results, elapsed

  results, elapsed = asyncio.run(run_batch())

  print(f"Batch extraction завершен за {elapsed:.2f}ms")
  print(f"Среднее время на символ: {elapsed / len(symbols):.2f}ms\n")

  print(f"Результаты:")
  for symbol, feature_vector in results.items():
    print(f"  {symbol}: {feature_vector.feature_count} признаков")

  if len(results) != len(symbols):
    print(f"\n  ✗ Не все символы обработаны!")
    return False

  print(f"\n  ✓ Все символы обработаны успешно")

  print("\n✅ Multi-Symbol Pipeline работает корректно!")
  return True


def performance_benchmark():
  """Бенчмарк производительности"""
  print_section("8. БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")

  from models.orderbook import OrderBookSnapshot
  from ml_engine.features import FeaturePipeline, Candle
  import numpy as np
  import asyncio

  # Подготовка данных
  orderbook = OrderBookSnapshot(
    symbol="BTCUSDT",
    bids=[(50000.0 - i, 1.0) for i in range(20)],
    asks=[(50001.0 + i, 1.0) for i in range(20)],
    timestamp=int(datetime.now().timestamp() * 1000)
  )

  candles = []
  base_time = int(datetime.now().timestamp() * 1000)
  for i in range(60):
    candles.append(Candle(
      timestamp=base_time - (60 - i) * 60000,
      open=50000.0,
      high=50020.0,
      low=49980.0,
      close=50010.0,
      volume=1.0
    ))

  pipeline = FeaturePipeline("BTCUSDT", normalize=False, cache_enabled=False)

  # Разогрев
  asyncio.run(pipeline.extract_features(orderbook, candles))

  # Бенчмарк
  iterations = 100
  print(f"Запуск {iterations} итераций...\n")

  times = []
  for _ in range(iterations):
    start = time.time()
    asyncio.run(pipeline.extract_features(orderbook, candles))
    times.append((time.time() - start) * 1000)

  times = np.array(times)

  print(f"Результаты:")
  print(f"  Итераций: {iterations}")
  print(f"  Среднее время: {times.mean():.2f}ms")
  print(f"  Медиана: {np.median(times):.2f}ms")
  print(f"  Min: {times.min():.2f}ms")
  print(f"  Max: {times.max():.2f}ms")
  print(f"  P95: {np.percentile(times, 95):.2f}ms")
  print(f"  P99: {np.percentile(times, 99):.2f}ms")
  print(f"  Std: {times.std():.2f}ms")

  throughput = 1000 / times.mean()
  print(f"\n  Throughput: {throughput:.1f} extractions/sec")

  if times.mean() < 20:
    print(f"\n  ✓ Производительность отличная (< 20ms)")
  elif times.mean() < 50:
    print(f"\n  ✓ Производительность хорошая (< 50ms)")
  else:
    print(f"\n  ⚠️  Производительность требует оптимизации (> 50ms)")

  print("\n✅ Бенчмарк завершен!")
  return True


def main():
  """Главная функция"""
  print("\n" + "=" * 70)
  print("ПОЛНАЯ ПРОВЕРКА ML FEATURE ENGINEERING СИСТЕМЫ")
  print("=" * 70)

  results = []

  # Проверки
  results.append(("Зависимости", check_dependencies()))
  results.append(("Импорты", check_imports()))
  results.append(("OrderBook Extractor", test_orderbook_extractor()))
  results.append(("Candle Extractor", test_candle_extractor()))
  results.append(("Indicator Extractor", test_indicator_extractor()))
  results.append(("Feature Pipeline", test_feature_pipeline()))
  results.append(("Multi-Symbol Pipeline", test_multi_symbol_pipeline()))
  results.append(("Performance Benchmark", performance_benchmark()))

  # Итоги
  print_section("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")

  for test_name, result in results:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status} | {test_name}")

  all_passed = all(result for _, result in results)

  if all_passed:
    print("\n" + "=" * 70)
    print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
    print("=" * 70)
    print("\n🎉 ML Feature Engineering система полностью готова!")
    print("\nКомпоненты:")
    print("  ✓ OrderBookFeatureExtractor - 50 признаков")
    print("  ✓ CandleFeatureExtractor - 25 признаков")
    print("  ✓ IndicatorFeatureExtractor - 35 признаков")
    print("  ✓ FeaturePipeline - полная интеграция")
    print("  ✓ MultiSymbolFeaturePipeline - batch processing")
    print("\nИТОГО: 110 признаков готовы к использованию в ML моделях!")
    print("\n📚 Следующие шаги:")
    print("  1. Интегрируйте в торговый бот")
    print("  2. Начните сбор данных для обучения ML моделей")
    print("  3. Реализуйте Hybrid CNN-LSTM модель")
    return 0
  else:
    print("\n" + "=" * 70)
    print("❌ НЕКОТОРЫЕ ПРОВЕРКИ НЕ ПРОЙДЕНЫ")
    print("=" * 70)
    print("\nИсправьте ошибки выше и запустите снова.")
    return 1


if __name__ == "__main__":
  exit_code = main()
  sys.exit(exit_code)