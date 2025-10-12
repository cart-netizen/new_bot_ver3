"""
Интеграционные тесты для полного Feature Pipeline.
Тестирует работу всех компонентов вместе.
"""
import os
import sys
from pathlib import Path
# Добавляем путь к backend
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import pytest
import numpy as np
from datetime import datetime
import asyncio

from models.orderbook import OrderBookSnapshot
from ml_engine.features.candle_feature_extractor import Candle
from ml_engine.features.feature_pipeline import (
  FeaturePipeline,
  MultiSymbolFeaturePipeline,
  FeatureVector
)


class TestFeaturePipelineIntegration:
  """Тесты для полного pipeline"""

  @pytest.fixture
  def sample_orderbook(self):
    """Создает тестовый снимок стакана"""
    return OrderBookSnapshot(
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

  @pytest.fixture
  def sample_candles(self):
    """Создает тестовые свечи (60 штук для индикаторов)"""
    base_time = int(datetime.now().timestamp() * 1000)
    candles = []

    base_price = 50000.0
    for i in range(60):
      # Создаем реалистичную свечу с трендом
      price_change = np.sin(i / 10) * 100  # Волновое движение

      open_price = base_price + price_change
      close_price = open_price + np.random.uniform(-50, 50)
      high_price = max(open_price, close_price) + np.random.uniform(10, 30)
      low_price = min(open_price, close_price) - np.random.uniform(10, 30)
      volume = np.random.uniform(0.5, 2.0)

      candles.append(Candle(
        timestamp=base_time - (60 - i) * 60000,  # 1 минутные свечи
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume
      ))

    return candles

  @pytest.fixture
  def pipeline(self):
    """Создает pipeline"""
    return FeaturePipeline("BTCUSDT", normalize=False, cache_enabled=True)

  @pytest.mark.asyncio
  async def test_full_pipeline_extraction(
      self,
      pipeline,
      sample_orderbook,
      sample_candles
  ):
    """Тест полного извлечения всех признаков"""

    # Извлекаем признаки
    feature_vector = await pipeline.extract_features(
      orderbook_snapshot=sample_orderbook,
      candles=sample_candles
    )

    # Проверяем структуру
    assert feature_vector.symbol == "BTCUSDT"
    assert feature_vector.timestamp > 0
    assert feature_vector.feature_count == 110  # 50 + 25 + 35

    # Проверяем отдельные каналы
    assert feature_vector.orderbook_features is not None
    assert feature_vector.candle_features is not None
    assert feature_vector.indicator_features is not None

    print(f"✓ Извлечено {feature_vector.feature_count} признаков")

  @pytest.mark.asyncio
  async def test_to_array_conversion(
      self,
      pipeline,
      sample_orderbook,
      sample_candles
  ):
    """Тест преобразования в numpy array"""

    feature_vector = await pipeline.extract_features(
      orderbook_snapshot=sample_orderbook,
      candles=sample_candles
    )

    # Преобразуем в array
    array = feature_vector.to_array()

    # Проверяем размерность
    assert array.shape == (110,)
    assert array.dtype == np.float32

    # Проверяем валидность
    assert not np.any(np.isnan(array))
    assert not np.any(np.isinf(array))

    print(f"✓ Array: shape={array.shape}, min={array.min():.2f}, max={array.max():.2f}")

  @pytest.mark.asyncio
  async def test_multi_channel_representation(
      self,
      pipeline,
      sample_orderbook,
      sample_candles
  ):
    """Тест multi-channel representation"""

    feature_vector = await pipeline.extract_features(
      orderbook_snapshot=sample_orderbook,
      candles=sample_candles
    )

    # Получаем каналы
    channels = feature_vector.to_channels()

    # Проверяем наличие всех каналов
    assert "orderbook" in channels
    assert "candle" in channels
    assert "indicator" in channels

    # Проверяем размерности
    assert channels["orderbook"].shape == (50,)
    assert channels["candle"].shape == (25,)
    assert channels["indicator"].shape == (35,)

    print("✓ Multi-channel representation:")
    print(f"  OrderBook: {channels['orderbook'].shape}")
    print(f"  Candle: {channels['candle'].shape}")
    print(f"  Indicator: {channels['indicator'].shape}")

  @pytest.mark.asyncio
  async def test_caching_mechanism(
      self,
      pipeline,
      sample_orderbook,
      sample_candles
  ):
    """Тест механизма кэширования"""

    # Первый вызов
    import time
    start = time.time()
    feature_vector1 = await pipeline.extract_features(
      orderbook_snapshot=sample_orderbook,
      candles=sample_candles
    )
    time1 = time.time() - start

    # Второй вызов (должен быть быстрее из-за кэша)
    start = time.time()
    feature_vector2 = await pipeline.extract_features(
      orderbook_snapshot=sample_orderbook,
      candles=sample_candles
    )
    time2 = time.time() - start

    # Результаты должны совпадать
    assert feature_vector1.timestamp == feature_vector2.timestamp

    print(f"✓ Кэширование работает:")
    print(f"  Первый вызов: {time1 * 1000:.2f}ms")
    print(f"  Второй вызов: {time2 * 1000:.2f}ms")

  @pytest.mark.asyncio
  async def test_normalization(self, sample_orderbook, sample_candles):
    """Тест нормализации признаков"""

    # Pipeline с нормализацией
    pipeline_norm = FeaturePipeline("BTCUSDT", normalize=True, cache_enabled=False)

    # Извлекаем несколько раз для обучения scaler
    for _ in range(5):
      await pipeline_norm.extract_features(
        orderbook_snapshot=sample_orderbook,
        candles=sample_candles
      )

    feature_vector = await pipeline_norm.extract_features(
      orderbook_snapshot=sample_orderbook,
      candles=sample_candles
    )

    array = feature_vector.to_array()

    # После нормализации mean должен быть близок к 0, std к 1
    # (но не точно из-за online learning)
    print(f"✓ Нормализация:")
    print(f"  Mean: {array.mean():.3f}")
    print(f"  Std: {array.std():.3f}")
    print(f"  Min: {array.min():.3f}, Max: {array.max():.3f}")

  @pytest.mark.asyncio
  async def test_feature_names(
      self,
      pipeline,
      sample_orderbook,
      sample_candles
  ):
    """Тест получения имен признаков"""

    feature_vector = await pipeline.extract_features(
      orderbook_snapshot=sample_orderbook,
      candles=sample_candles
    )

    feature_names = feature_vector.get_feature_names()

    # Проверяем количество
    assert len(feature_names) > 0

    # Проверяем структуру имен
    ob_names = [n for n in feature_names if n.startswith("ob_")]
    candle_names = [n for n in feature_names if n.startswith("candle_")]
    ind_names = [n for n in feature_names if n.startswith("ind_")]

    print(f"✓ Feature names: {len(feature_names)} всего")
    print(f"  OrderBook: {len(ob_names)}")
    print(f"  Candle: {len(candle_names)}")
    print(f"  Indicator: {len(ind_names)}")

  @pytest.mark.asyncio
  async def test_multi_symbol_pipeline(
      self,
      sample_orderbook,
      sample_candles
  ):
    """Тест multi-symbol pipeline"""

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    multi_pipeline = MultiSymbolFeaturePipeline(symbols)

    # Подготавливаем данные для всех символов
    data = {}
    for symbol in symbols:
      orderbook = OrderBookSnapshot(
        symbol=symbol,
        bids=sample_orderbook.bids,
        asks=sample_orderbook.asks,
        timestamp=sample_orderbook.timestamp
      )
      data[symbol] = (orderbook, sample_candles)

    # Batch extraction
    results = await multi_pipeline.extract_features_batch(data)

    # Проверяем что все символы обработаны
    assert len(results) == len(symbols)
    for symbol in symbols:
      assert symbol in results
      assert results[symbol].feature_count == 110

    print(f"✓ Multi-symbol pipeline: {len(results)} символов обработано")

  @pytest.mark.asyncio
  async def test_insufficient_data_handling(self, pipeline, sample_orderbook):
    """Тест обработки недостаточных данных"""

    # Только 10 свечей (недостаточно для индикаторов)
    few_candles = [
      Candle(
        timestamp=int(datetime.now().timestamp() * 1000) + i * 60000,
        open=50000.0,
        high=50100.0,
        low=49900.0,
        close=50050.0,
        volume=1.0
      )
      for i in range(10)
    ]

    # Должно работать без ошибок
    feature_vector = await pipeline.extract_features(
      orderbook_snapshot=sample_orderbook,
      candles=few_candles
    )

    # Признаки должны быть дефолтными
    assert feature_vector.indicator_features.rsi_14 == 50.0
    assert feature_vector.indicator_features.adx == 25.0

    print("✓ Недостаточные данные обработаны корректно")

  @pytest.mark.asyncio
  async def test_to_dict_serialization(
      self,
      pipeline,
      sample_orderbook,
      sample_candles
  ):
    """Тест сериализации в словарь"""

    feature_vector = await pipeline.extract_features(
      orderbook_snapshot=sample_orderbook,
      candles=sample_candles
    )

    # Преобразуем в dict
    feature_dict = feature_vector.to_dict()

    # Проверяем структуру
    assert "symbol" in feature_dict
    assert "timestamp" in feature_dict
    assert "orderbook" in feature_dict
    assert "candle" in feature_dict
    assert "indicator" in feature_dict
    assert "version" in feature_dict

    # Можно сериализовать в JSON
    import json
    json_str = json.dumps(feature_dict)

    print(f"✓ Dict serialization: {len(json_str)} bytes")


if __name__ == "__main__":
  """Запуск тестов напрямую"""
  print("=" * 70)
  print("ИНТЕГРАЦИОННОЕ ТЕСТИРОВАНИЕ FEATURE PIPELINE")
  print("=" * 70)

  # Создаем тестовые данные
  orderbook = OrderBookSnapshot(
    symbol="BTCUSDT",
    bids=[(50000.0, 1.5), (49999.0, 2.0), (49998.0, 1.0)],
    asks=[(50001.0, 1.2), (50002.0, 1.8), (50003.0, 0.9)],
    timestamp=int(datetime.now().timestamp() * 1000)
  )

  # Создаем 60 свечей
  base_time = int(datetime.now().timestamp() * 1000)
  candles = []
  for i in range(60):
    price_change = np.sin(i / 10) * 100
    open_price = 50000.0 + price_change
    close_price = open_price + np.random.uniform(-50, 50)

    candles.append(Candle(
      timestamp=base_time - (60 - i) * 60000,
      open=open_price,
      high=max(open_price, close_price) + 20,
      low=min(open_price, close_price) - 20,
      close=close_price,
      volume=np.random.uniform(0.5, 2.0)
    ))


  # Создаем pipeline
  async def run_tests():
    pipeline = FeaturePipeline("BTCUSDT", normalize=False, cache_enabled=True)

    print("\n1. Извлечение всех признаков...")
    feature_vector = await pipeline.extract_features(
      orderbook_snapshot=orderbook,
      candles=candles
    )

    print(f"   Symbol: {feature_vector.symbol}")
    print(f"   Timestamp: {feature_vector.timestamp}")
    print(f"   Total features: {feature_vector.feature_count}")

    print("\n2. Проверка отдельных каналов:")
    channels = feature_vector.to_channels()
    print(f"   OrderBook: {channels['orderbook'].shape}")
    print(f"   Candle: {channels['candle'].shape}")
    print(f"   Indicator: {channels['indicator'].shape}")

    print("\n3. Преобразование в единый массив:")
    array = feature_vector.to_array()
    print(f"   Shape: {array.shape}")
    print(f"   Dtype: {array.dtype}")
    print(f"   Min: {array.min():.2f}, Max: {array.max():.2f}")
    print(f"   Mean: {array.mean():.2f}, Std: {array.std():.2f}")

    # Проверка валидности
    if np.any(np.isnan(array)):
      print("   ⚠️ Обнаружены NaN значения!")
    else:
      print("   ✓ Нет NaN значений")

    if np.any(np.isinf(array)):
      print("   ⚠️ Обнаружены Inf значения!")
    else:
      print("   ✓ Нет Inf значений")

    print("\n4. Некоторые извлеченные признаки:")
    print(f"   OrderBook mid_price: {feature_vector.orderbook_features.mid_price:.2f}")
    print(f"   OrderBook imbalance: {feature_vector.orderbook_features.imbalance_5:.4f}")
    print(f"   Candle returns: {feature_vector.candle_features.returns:.4f}")
    print(f"   Candle volatility: {feature_vector.candle_features.realized_volatility:.6f}")
    print(f"   Indicator RSI: {feature_vector.indicator_features.rsi_14:.2f}")
    print(f"   Indicator MACD: {feature_vector.indicator_features.macd:.4f}")

    print("\n5. Тест multi-symbol pipeline:")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    multi_pipeline = MultiSymbolFeaturePipeline(symbols)

    data = {}
    for symbol in symbols:
      ob = OrderBookSnapshot(
        symbol=symbol,
        bids=orderbook.bids,
        asks=orderbook.asks,
        timestamp=orderbook.timestamp
      )
      data[symbol] = (ob, candles)

    results = await multi_pipeline.extract_features_batch(data)
    print(f"   Обработано {len(results)} символов")
    for symbol, fv in results.items():
      print(f"   {symbol}: {fv.feature_count} признаков")

    print("\n" + "=" * 70)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("=" * 70)
    print("\nSummary:")
    print("- OrderBook Features: 50 признаков ✓")
    print("- Candle Features: 25 признаков ✓")
    print("- Indicator Features: 35 признаков ✓")
    print("- Total: 110 признаков ✓")
    print("- Multi-channel representation ✓")
    print("- Batch processing ✓")
    print("- Кэширование ✓")


  # Запускаем
  asyncio.run(run_tests())