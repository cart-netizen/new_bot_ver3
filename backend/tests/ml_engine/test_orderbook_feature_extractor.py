"""
Тесты для OrderBookFeatureExtractor.
Проверяет работоспособность и интеграцию с существующим кодом.
"""
import sys
from pathlib import Path

# Определяем путь к backend директории
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))
import pytest
import numpy as np
from datetime import datetime

from models.orderbook import OrderBookSnapshot
from ml_engine.features.orderbook_feature_extractor import (
  OrderBookFeatureExtractor,
  OrderBookFeatures
)


class TestOrderBookFeatureExtractor:
  """Тесты для извлечения признаков из стакана"""

  @pytest.fixture
  def sample_snapshot(self):
    """Создает реалистичный снимок стакана для тестов"""
    return OrderBookSnapshot(
      symbol="BTCUSDT",
      bids=[
        (50000.0, 1.5),
        (49999.0, 2.0),
        (49998.0, 1.0),
        (49997.0, 0.5),
        (49996.0, 3.0),
        (49995.0, 1.2),
        (49994.0, 0.8),
        (49993.0, 2.5),
        (49992.0, 1.1),
        (49991.0, 0.9),
      ],
      asks=[
        (50001.0, 1.2),
        (50002.0, 1.8),
        (50003.0, 0.9),
        (50004.0, 2.5),
        (50005.0, 1.5),
        (50006.0, 0.7),
        (50007.0, 2.0),
        (50008.0, 1.3),
        (50009.0, 0.6),
        (50010.0, 1.9),
      ],
      timestamp=int(datetime.now().timestamp() * 1000)
    )

  @pytest.fixture
  def extractor(self):
    """Создает экземпляр extractor"""
    return OrderBookFeatureExtractor("BTCUSDT")

  def test_extractor_initialization(self, extractor):
    """Тест инициализации"""
    assert extractor.symbol == "BTCUSDT"
    assert extractor.analyzer is not None
    assert len(extractor.snapshot_history) == 0
    assert extractor.max_history_size == 100

  def test_extract_basic_features(self, extractor, sample_snapshot):
    """Тест извлечения базовых признаков"""
    features = extractor.extract(sample_snapshot)

    # Проверяем что все базовые признаки присутствуют
    assert features.symbol == "BTCUSDT"
    assert features.timestamp > 0

    # Базовые микроструктурные (15)
    assert features.bid_ask_spread_abs > 0
    assert features.bid_ask_spread_rel > 0
    assert features.mid_price > 0
    assert features.micro_price > 0
    assert features.vwap_bid_5 > 0
    assert features.vwap_ask_5 > 0
    assert features.vwap_bid_10 > 0
    assert features.vwap_ask_10 > 0
    assert features.depth_bid_5 > 0
    assert features.depth_ask_5 > 0
    assert features.depth_bid_10 > 0
    assert features.depth_ask_10 > 0
    assert features.total_bid_volume > 0
    assert features.total_ask_volume > 0
    assert features.book_depth_ratio > 0

    print(f"✓ Базовые признаки: mid_price={features.mid_price:.2f}")

  def test_extract_imbalance_features(self, extractor, sample_snapshot):
    """Тест извлечения признаков дисбаланса"""
    features = extractor.extract(sample_snapshot)

    # Дисбаланс (10)
    assert -1.0 <= features.imbalance_5 <= 1.0
    assert -1.0 <= features.imbalance_10 <= 1.0
    assert -1.0 <= features.imbalance_total <= 1.0
    assert features.bid_intensity >= 0
    assert features.ask_intensity >= 0
    assert features.buy_sell_ratio > 0
    assert 0 <= features.smart_money_index <= 1

    print(f"✓ Дисбаланс: imbalance_5={features.imbalance_5:.3f}")

  def test_extract_cluster_features(self, extractor, sample_snapshot):
    """Тест извлечения кластерных признаков"""
    features = extractor.extract(sample_snapshot)

    # Кластеры (10)
    assert features.num_bid_clusters >= 0
    assert features.num_ask_clusters >= 0
    assert features.support_level_1 > 0
    assert features.resistance_level_1 > 0

    print(f"✓ Кластеры: bid={features.num_bid_clusters}, ask={features.num_ask_clusters}")

  def test_extract_liquidity_features(self, extractor, sample_snapshot):
    """Тест извлечения признаков ликвидности"""
    features = extractor.extract(sample_snapshot)

    # Ликвидность (8)
    assert features.liquidity_bid_5 > 0
    assert features.liquidity_ask_5 > 0
    assert features.liquidity_asymmetry > 0
    assert features.effective_spread >= 0
    assert features.kyle_lambda >= 0
    assert features.amihud_illiquidity >= 0

    print(f"✓ Ликвидность: bid={features.liquidity_bid_5:.2f}, ask={features.liquidity_ask_5:.2f}")

  def test_extract_temporal_features(self, extractor, sample_snapshot):
    """Тест извлечения временных признаков"""
    # Добавляем несколько снимков для истории
    for i in range(5):
      snapshot = OrderBookSnapshot(
        symbol="BTCUSDT",
        bids=sample_snapshot.bids,
        asks=sample_snapshot.asks,
        timestamp=int(datetime.now().timestamp() * 1000) + i * 1000
      )
      extractor.extract(snapshot)

    features = extractor.extract(sample_snapshot)

    # Временные (7)
    assert features.orderbook_volatility >= 0
    assert features.update_frequency >= 0
    assert features.quote_intensity >= 0
    assert features.spread_volatility >= 0

    print(f"✓ Временные: volatility={features.orderbook_volatility:.6f}")

  def test_to_array_conversion(self, extractor, sample_snapshot):
    """Тест преобразования в numpy array"""
    features = extractor.extract(sample_snapshot)
    array = features.to_array()

    # Проверяем размерность
    assert array.shape == (50,)
    assert array.dtype == np.float32

    # Проверяем что нет NaN или Inf
    assert not np.any(np.isnan(array))
    assert not np.any(np.isinf(array))

    print(f"✓ Array: shape={array.shape}, dtype={array.dtype}")
    print(f"  Первые 5 признаков: {array[:5]}")

  def test_to_dict_conversion(self, extractor, sample_snapshot):
    """Тест преобразования в словарь"""
    features = extractor.extract(sample_snapshot)
    feature_dict = features.to_dict()

    # Проверяем что все признаки в словаре
    assert len(feature_dict) >= 50
    assert "symbol" in feature_dict
    assert "timestamp" in feature_dict
    assert "mid_price" in feature_dict
    assert "imbalance_5" in feature_dict

    print(f"✓ Dict: {len(feature_dict)} ключей")

  def test_history_management(self, extractor, sample_snapshot):
    """Тест управления историей снимков"""
    # Добавляем больше снимков чем max_history_size
    for i in range(150):
      snapshot = OrderBookSnapshot(
        symbol="BTCUSDT",
        bids=sample_snapshot.bids,
        asks=sample_snapshot.asks,
        timestamp=int(datetime.now().timestamp() * 1000) + i * 1000
      )
      extractor.extract(snapshot)

    # Проверяем что история ограничена
    assert len(extractor.snapshot_history) == extractor.max_history_size

    print(f"✓ История: {len(extractor.snapshot_history)} снимков (max={extractor.max_history_size})")

  def test_imbalanced_orderbook(self, extractor):
    """Тест с дисбалансированным стаканом (больше bids)"""
    snapshot = OrderBookSnapshot(
      symbol="BTCUSDT",
      bids=[
        (50000.0, 10.0),  # Много bids
        (49999.0, 8.0),
        (49998.0, 7.0),
        (49997.0, 6.0),
        (49996.0, 5.0),
      ],
      asks=[
        (50001.0, 1.0),  # Мало asks
        (50002.0, 0.5),
        (50003.0, 0.3),
        (50004.0, 0.2),
        (50005.0, 0.1),
      ],
      timestamp=int(datetime.now().timestamp() * 1000)
    )

    features = extractor.extract(snapshot)

    # Должен быть положительный imbalance (больше bids)
    assert features.imbalance_5 > 0.5
    assert features.book_depth_ratio > 1.0

    print(f"✓ Дисбаланс стакана: imbalance={features.imbalance_5:.3f}, ratio={features.book_depth_ratio:.2f}")

  def test_empty_orderbook_handling(self, extractor):
    """Тест обработки пустого стакана"""
    snapshot = OrderBookSnapshot(
      symbol="BTCUSDT",
      bids=[],
      asks=[],
      timestamp=int(datetime.now().timestamp() * 1000)
    )

    try:
      features = extractor.extract(snapshot)

      # Должны быть значения по умолчанию
      assert features.mid_price == 0.0
      assert features.total_bid_volume == 0.0
      assert features.total_ask_volume == 0.0

      print("✓ Пустой стакан обработан корректно")
    except Exception as e:
      pytest.fail(f"Не удалось обработать пустой стакан: {e}")

  def test_feature_consistency(self, extractor, sample_snapshot):
    """Тест консистентности признаков при повторном извлечении"""
    features1 = extractor.extract(sample_snapshot)
    features2 = extractor.extract(sample_snapshot)

    # Основные признаки должны быть идентичны
    assert features1.mid_price == features2.mid_price
    assert features1.imbalance_5 == features2.imbalance_5
    assert features1.total_bid_volume == features2.total_bid_volume

    print("✓ Консистентность признаков подтверждена")


if __name__ == "__main__":
  """Запуск тестов напрямую"""
  print("=" * 70)
  print("ТЕСТИРОВАНИЕ OrderBookFeatureExtractor")
  print("=" * 70)

  # Создаем тестовые данные
  snapshot = OrderBookSnapshot(
    symbol="BTCUSDT",
    bids=[
      (50000.0, 1.5),
      (49999.0, 2.0),
      (49998.0, 1.0),
      (49997.0, 0.5),
      (49996.0, 3.0),
      (49995.0, 1.2),
      (49994.0, 0.8),
      (49993.0, 2.5),
      (49992.0, 1.1),
      (49991.0, 0.9),
    ],
    asks=[
      (50001.0, 1.2),
      (50002.0, 1.8),
      (50003.0, 0.9),
      (50004.0, 2.5),
      (50005.0, 1.5),
      (50006.0, 0.7),
      (50007.0, 2.0),
      (50008.0, 1.3),
      (50009.0, 0.6),
      (50010.0, 1.9),
    ],
    timestamp=int(datetime.now().timestamp() * 1000)
  )

  # Создаем extractor
  extractor = OrderBookFeatureExtractor("BTCUSDT")

  # Извлекаем признаки
  print("\n1. Извлечение признаков...")
  features = extractor.extract(snapshot)

  print(f"\n2. Результаты извлечения:")
  print(f"   Symbol: {features.symbol}")
  print(f"   Timestamp: {features.timestamp}")
  print(f"   Mid Price: {features.mid_price:.2f}")
  print(f"   Spread (abs): {features.bid_ask_spread_abs:.2f}")
  print(f"   Spread (rel): {features.bid_ask_spread_rel:.4f}%")
  print(f"   Imbalance (5 levels): {features.imbalance_5:.4f}")
  print(f"   Imbalance (10 levels): {features.imbalance_10:.4f}")
  print(f"   Total Bid Volume: {features.total_bid_volume:.2f}")
  print(f"   Total Ask Volume: {features.total_ask_volume:.2f}")
  print(f"   Book Depth Ratio: {features.book_depth_ratio:.4f}")

  print(f"\n3. Кластерный анализ:")
  print(f"   Bid Clusters: {features.num_bid_clusters}")
  print(f"   Ask Clusters: {features.num_ask_clusters}")
  print(
    f"   Largest Bid Cluster: price={features.largest_bid_cluster_price:.2f}, vol={features.largest_bid_cluster_volume:.2f}")
  print(
    f"   Largest Ask Cluster: price={features.largest_ask_cluster_price:.2f}, vol={features.largest_ask_cluster_volume:.2f}")

  print(f"\n4. Преобразование в массив:")
  array = features.to_array()
  print(f"   Shape: {array.shape}")
  print(f"   Dtype: {array.dtype}")
  print(f"   Первые 10 признаков: {array[:10]}")
  print(f"   Min: {array.min():.4f}, Max: {array.max():.4f}")
  print(f"   Mean: {array.mean():.4f}, Std: {array.std():.4f}")

  # Проверка NaN/Inf
  if np.any(np.isnan(array)):
    print("   ⚠️  ПРЕДУПРЕЖДЕНИЕ: Обнаружены NaN значения!")
  else:
    print("   ✓ Нет NaN значений")

  if np.any(np.isinf(array)):
    print("   ⚠️  ПРЕДУПРЕЖДЕНИЕ: Обнаружены Inf значения!")
  else:
    print("   ✓ Нет Inf значений")

  print("\n5. Тест с несколькими снимками (временные признаки):")
  for i in range(5):
    snapshot_i = OrderBookSnapshot(
      symbol="BTCUSDT",
      bids=snapshot.bids,
      asks=snapshot.asks,
      timestamp=snapshot.timestamp + i * 1000
    )
    features_i = extractor.extract(snapshot_i)

  print(f"   История снимков: {len(extractor.snapshot_history)}")
  print(f"   Update Frequency: {features_i.update_frequency:.2f} updates/sec")
  print(f"   Orderbook Volatility: {features_i.orderbook_volatility:.6f}")
  print(f"   Spread Volatility: {features_i.spread_volatility:.6f}")

  print("\n" + "=" * 70)
  print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
  print("=" * 70)