"""
Тесты для MLDataCollector - проверка сохранения семплов.

Файл: backend/tests/ml_engine/test_ml_data_collection.py
"""

import pytest
import asyncio
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from ml_engine.data_collection import MLDataCollector
from ml_engine.features import FeatureVector, FeaturePipeline, Candle
from models.orderbook import OrderBookSnapshot, OrderBookMetrics


class TestMLDataCollector:
    """Тесты для MLDataCollector."""

    @pytest.fixture
    def temp_storage(self):
        """Создание временной директории для тестов."""
        temp_dir = tempfile.mkdtemp(prefix="ml_test_")
        yield temp_dir
        # Cleanup после теста
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def ml_collector(self, temp_storage):
        """Создание MLDataCollector с временным хранилищем."""
        collector = MLDataCollector(
            storage_path=temp_storage,
            max_samples_per_file=10,  # Малое значение для быстрого сохранения
            collection_interval=1      # Собираем каждую итерацию
        )
        return collector

    @pytest.fixture
    def sample_orderbook(self):
        """Создание тестового снимка стакана."""
        return OrderBookSnapshot(
            symbol="TESTUSDT",
            bids=[
                (50000.0, 1.5),
                (49999.0, 2.0),
                (49998.0, 1.8),
            ],
            asks=[
                (50001.0, 1.2),
                (50002.0, 1.6),
                (50003.0, 1.4),
            ],
            timestamp=1760254272035
        )

    @pytest.fixture
    def sample_metrics(self):
        """Создание тестовых метрик."""
        return OrderBookMetrics(
            symbol="TESTUSDT",
            timestamp=1760254272035,
            best_bid=50000.0,
            best_ask=50001.0,
            spread=1.0,
            mid_price=50000.5,
            total_bid_volume=5.3,
            total_ask_volume=4.2,
            imbalance=0.558
        )

    @pytest.fixture
    async def sample_feature_vector(self, sample_orderbook):
        """Создание тестового вектора признаков."""
        # Создаем свечи для индикаторов
        candles = []
        base_time = 1760254272035
        for i in range(60):
            candles.append(Candle(
                timestamp=base_time - (60 - i) * 60000,
                open=50000.0 + i * 10,
                high=50100.0 + i * 10,
                low=49900.0 + i * 10,
                close=50050.0 + i * 10,
                volume=1.0 + i * 0.1
            ))

        # Создаем pipeline и извлекаем признаки
        pipeline = FeaturePipeline("TESTUSDT", normalize=False)
        feature_vector = await pipeline.extract_features(
            orderbook_snapshot=sample_orderbook,
            candles=candles
        )
        return feature_vector

    @pytest.fixture
    def sample_signal(self):
        """Создание тестового сигнала."""
        return {
            "type": "buy",
            "confidence": 0.85,
            "strength": "strong"
        }

    @pytest.mark.asyncio
    async def test_collector_initialization(self, ml_collector, temp_storage):
        """Тест инициализации коллектора."""
        await ml_collector.initialize()

        # Проверяем что директория создана
        assert Path(temp_storage).exists()
        assert Path(temp_storage).is_dir()

        print("✓ Коллектор инициализирован успешно")

    @pytest.mark.asyncio
    async def test_should_collect(self, ml_collector):
        """Тест логики интервала сбора."""
        # Первая итерация - должно быть True
        assert ml_collector.should_collect() == True

        # Следующие итерации зависят от collection_interval
        # У нас interval=1, поэтому каждая итерация = True
        assert ml_collector.should_collect() == True

        print("✓ Логика интервала сбора работает корректно")

    @pytest.mark.asyncio
    async def test_collect_single_sample(
        self,
        ml_collector,
        sample_orderbook,
        sample_metrics,
        sample_feature_vector,
        sample_signal
    ):
        """Тест сбора одного семпла."""
        await ml_collector.initialize()

        # Собираем один семпл
        await ml_collector.collect_sample(
            symbol="TESTUSDT",
            feature_vector=sample_feature_vector,
            orderbook_snapshot=sample_orderbook,
            market_metrics=sample_metrics,
            executed_signal=sample_signal
        )

        # Проверяем счетчики
        assert ml_collector.sample_counts["TESTUSDT"] == 1
        assert ml_collector.total_samples_collected == 1
        assert len(ml_collector.feature_buffers["TESTUSDT"]) == 1
        assert len(ml_collector.label_buffers["TESTUSDT"]) == 1
        assert len(ml_collector.metadata_buffers["TESTUSDT"]) == 1

        print("✓ Семпл успешно добавлен в буферы")

    @pytest.mark.asyncio
    async def test_batch_save_trigger(
        self,
        ml_collector,
        sample_orderbook,
        sample_metrics,
        sample_feature_vector,
        sample_signal,
        temp_storage
    ):
        """Тест автоматического сохранения batch при достижении лимита."""
        await ml_collector.initialize()

        # Собираем семплы до достижения max_samples_per_file (10)
        for i in range(12):  # Больше чем лимит
            await ml_collector.collect_sample(
                symbol="TESTUSDT",
                feature_vector=sample_feature_vector,
                orderbook_snapshot=sample_orderbook,
                market_metrics=sample_metrics,
                executed_signal=sample_signal
            )

        # После 10 семплов должен был сохраниться batch
        symbol_dir = Path(temp_storage) / "TESTUSDT"
        features_dir = symbol_dir / "features"
        labels_dir = symbol_dir / "labels"
        metadata_dir = symbol_dir / "metadata"

        # Проверяем что директории созданы
        assert features_dir.exists(), "Директория features не создана"
        assert labels_dir.exists(), "Директория labels не создана"
        assert metadata_dir.exists(), "Директория metadata не создана"

        # Проверяем что файлы созданы
        feature_files = list(features_dir.glob("*.npy"))
        label_files = list(labels_dir.glob("*.json"))
        metadata_files = list(metadata_dir.glob("*.json"))

        assert len(feature_files) >= 1, "Features файл не создан"
        assert len(label_files) >= 1, "Labels файл не создан"
        assert len(metadata_files) >= 1, "Metadata файл не создан"

        # Буферы должны быть очищены после сохранения
        # Осталось только 2 семпла в буфере (12 - 10)
        assert len(ml_collector.feature_buffers["TESTUSDT"]) == 2

        print("✓ Batch автоматически сохранен при достижении лимита")

    @pytest.mark.asyncio
    async def test_saved_features_format(
        self,
        ml_collector,
        sample_orderbook,
        sample_metrics,
        sample_feature_vector,
        sample_signal,
        temp_storage
    ):
        """Тест формата и содержимого сохраненных features."""
        await ml_collector.initialize()

        # Собираем 10 семплов для сохранения
        for i in range(10):
            await ml_collector.collect_sample(
                symbol="TESTUSDT",
                feature_vector=sample_feature_vector,
                orderbook_snapshot=sample_orderbook,
                market_metrics=sample_metrics,
                executed_signal=sample_signal
            )

        # Читаем сохраненный features файл
        features_dir = Path(temp_storage) / "TESTUSDT" / "features"
        feature_file = list(features_dir.glob("*.npy"))[0]
        features_array = np.load(feature_file)

        # Проверяем размерность
        assert features_array.shape == (10, 110), \
            f"Неправильная размерность: {features_array.shape}"

        # Проверяем тип данных
        assert features_array.dtype == np.float32, \
            f"Неправильный тип данных: {features_array.dtype}"

        # Проверяем что нет NaN
        assert not np.any(np.isnan(features_array)), \
            "Обнаружены NaN значения в features"

        # Проверяем что нет Inf
        assert not np.any(np.isinf(features_array)), \
            "Обнаружены Inf значения в features"

        print(f"✓ Features сохранены корректно: shape={features_array.shape}")
        print(f"  - Min: {features_array.min():.4f}")
        print(f"  - Max: {features_array.max():.4f}")
        print(f"  - Mean: {features_array.mean():.4f}")

    @pytest.mark.asyncio
    async def test_saved_labels_content(
        self,
        ml_collector,
        sample_orderbook,
        sample_metrics,
        sample_feature_vector,
        sample_signal,
        temp_storage
    ):
        """Тест содержимого сохраненных labels."""
        await ml_collector.initialize()

        # Собираем 10 семплов
        for i in range(10):
            await ml_collector.collect_sample(
                symbol="TESTUSDT",
                feature_vector=sample_feature_vector,
                orderbook_snapshot=sample_orderbook,
                market_metrics=sample_metrics,
                executed_signal=sample_signal
            )

        # Читаем сохраненный labels файл
        labels_dir = Path(temp_storage) / "TESTUSDT" / "labels"
        label_file = list(labels_dir.glob("*.json"))[0]

        with open(label_file, 'r') as f:
            labels = json.load(f)

        # Проверяем количество
        assert len(labels) == 10, f"Неправильное количество labels: {len(labels)}"

        # Проверяем структуру первого label
        first_label = labels[0]

        # Проверяем обязательные поля
        required_fields = [
            "future_direction_10s",
            "future_direction_30s",
            "future_direction_60s",
            "future_movement_10s",
            "future_movement_30s",
            "future_movement_60s",
            "current_mid_price",
            "current_imbalance",
            "signal_type",
            "signal_confidence",
            "signal_strength"
        ]

        for field in required_fields:
            assert field in first_label, f"Отсутствует поле: {field}"

        # Проверяем значения
        assert first_label["current_mid_price"] == sample_orderbook.mid_price
        assert first_label["current_imbalance"] == sample_metrics.imbalance
        assert first_label["signal_type"] == "buy"
        assert first_label["signal_confidence"] == 0.85
        assert first_label["signal_strength"] == "strong"

        # Проверяем что future поля None (еще не рассчитаны)
        assert first_label["future_direction_10s"] is None
        assert first_label["future_movement_10s"] is None

        print("✓ Labels сохранены с правильной структурой:")
        print(f"  - Все обязательные поля присутствуют")
        print(f"  - Signal: {first_label['signal_type']} (conf={first_label['signal_confidence']})")
        print(f"  - Price: {first_label['current_mid_price']}")
        print(f"  - Imbalance: {first_label['current_imbalance']:.4f}")

    @pytest.mark.asyncio
    async def test_saved_metadata_content(
        self,
        ml_collector,
        sample_orderbook,
        sample_metrics,
        sample_feature_vector,
        sample_signal,
        temp_storage
    ):
        """Тест содержимого сохраненных metadata."""
        await ml_collector.initialize()

        # Собираем 10 семплов
        for i in range(10):
            await ml_collector.collect_sample(
                symbol="TESTUSDT",
                feature_vector=sample_feature_vector,
                orderbook_snapshot=sample_orderbook,
                market_metrics=sample_metrics,
                executed_signal=sample_signal
            )

        # Читаем сохраненный metadata файл
        metadata_dir = Path(temp_storage) / "TESTUSDT" / "metadata"
        metadata_file = list(metadata_dir.glob("*.json"))[0]

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Проверяем batch_info
        assert "batch_info" in metadata
        batch_info = metadata["batch_info"]

        assert batch_info["symbol"] == "TESTUSDT"
        assert batch_info["batch_number"] == 1
        assert batch_info["sample_count"] == 10
        assert "timestamp" in batch_info
        assert batch_info["feature_shape"] == [10, 110]

        # Проверяем samples
        assert "samples" in metadata
        samples = metadata["samples"]
        assert len(samples) == 10

        # Проверяем структуру первого sample
        first_sample = samples[0]

        required_fields = [
            "timestamp",
            "symbol",
            "mid_price",
            "spread",
            "imbalance",
            "signal_type",
            "signal_confidence",
            "signal_strength",
            "feature_count"
        ]

        for field in required_fields:
            assert field in first_sample, f"Отсутствует поле в metadata: {field}"

        # Проверяем значения
        assert first_sample["symbol"] == "TESTUSDT"
        assert first_sample["timestamp"] == sample_orderbook.timestamp
        assert first_sample["mid_price"] == sample_orderbook.mid_price
        assert first_sample["spread"] == sample_orderbook.spread
        assert first_sample["imbalance"] == sample_metrics.imbalance
        assert first_sample["signal_type"] == "buy"
        assert first_sample["signal_confidence"] == 0.85
        assert first_sample["signal_strength"] == "strong"
        assert first_sample["feature_count"] == 110

        print("✓ Metadata сохранены с правильной структурой:")
        print(f"  - Batch info корректен")
        print(f"  - Samples содержат все обязательные поля")
        print(f"  - Timestamp: {first_sample['timestamp']}")
        print(f"  - Feature count: {first_sample['feature_count']}")

    @pytest.mark.asyncio
    async def test_finalize_saves_remaining_buffer(
        self,
        ml_collector,
        sample_orderbook,
        sample_metrics,
        sample_feature_vector,
        sample_signal,
        temp_storage
    ):
        """Тест что finalize() сохраняет оставшиеся данные в буфере."""
        await ml_collector.initialize()

        # Собираем только 5 семплов (меньше чем max_samples_per_file=10)
        for i in range(5):
            await ml_collector.collect_sample(
                symbol="TESTUSDT",
                feature_vector=sample_feature_vector,
                orderbook_snapshot=sample_orderbook,
                market_metrics=sample_metrics,
                executed_signal=sample_signal
            )

        # Буфер должен содержать 5 семплов
        assert len(ml_collector.feature_buffers["TESTUSDT"]) == 5

        # Вызываем finalize()
        await ml_collector.finalize()

        # Проверяем что данные сохранены
        features_dir = Path(temp_storage) / "TESTUSDT" / "features"
        feature_files = list(features_dir.glob("*.npy"))

        assert len(feature_files) == 1, "Batch не сохранен при finalize()"

        # Проверяем количество семплов в файле
        features_array = np.load(feature_files[0])
        assert features_array.shape[0] == 5, \
            f"Неправильное количество семплов: {features_array.shape[0]}"

        # Буфер должен быть очищен
        assert len(ml_collector.feature_buffers["TESTUSDT"]) == 0

        print("✓ Finalize() корректно сохранил оставшиеся данные")

    @pytest.mark.asyncio
    async def test_statistics(
        self,
        ml_collector,
        sample_orderbook,
        sample_metrics,
        sample_feature_vector,
        sample_signal
    ):
        """Тест получения статистики."""
        await ml_collector.initialize()

        # Собираем семплы
        for i in range(15):  # Больше чем max_samples_per_file
            await ml_collector.collect_sample(
                symbol="TESTUSDT",
                feature_vector=sample_feature_vector,
                orderbook_snapshot=sample_orderbook,
                market_metrics=sample_metrics,
                executed_signal=sample_signal
            )

        # Получаем статистику
        stats = ml_collector.get_statistics()

        # Проверяем общую статистику
        assert stats["total_samples_collected"] == 15
        assert stats["files_written"] >= 3  # features, labels, metadata
        assert stats["collection_interval"] == 1

        # Проверяем статистику по символу
        assert "symbols" in stats
        assert "TESTUSDT" in stats["symbols"]

        symbol_stats = stats["symbols"]["TESTUSDT"]
        assert symbol_stats["total_samples"] == 15
        assert symbol_stats["current_batch"] == 2  # Первый batch сохранен
        assert symbol_stats["buffer_size"] == 5  # 15 - 10 = 5

        print("✓ Статистика рассчитывается корректно:")
        print(f"  - Total samples: {stats['total_samples_collected']}")
        print(f"  - Files written: {stats['files_written']}")
        print(f"  - Buffer size: {symbol_stats['buffer_size']}")

    @pytest.mark.asyncio
    async def test_multiple_symbols(
        self,
        ml_collector,
        sample_orderbook,
        sample_metrics,
        sample_feature_vector,
        sample_signal,
        temp_storage
    ):
        """Тест сбора данных для нескольких символов."""
        await ml_collector.initialize()

        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

        # Собираем данные для каждого символа
        for symbol in symbols:
            for i in range(5):
                # Копируем данные с правильным символом
                orderbook = OrderBookSnapshot(
                    symbol=symbol,
                    bids=sample_orderbook.bids,
                    asks=sample_orderbook.asks,
                    timestamp=sample_orderbook.timestamp
                )
                metrics = OrderBookMetrics(
                    symbol=symbol,
                    timestamp=sample_metrics.timestamp,
                    best_bid=sample_metrics.best_bid,
                    best_ask=sample_metrics.best_ask,
                    spread=sample_metrics.spread,
                    mid_price=sample_metrics.mid_price,
                    imbalance=sample_metrics.imbalance
                )

                await ml_collector.collect_sample(
                    symbol=symbol,
                    feature_vector=sample_feature_vector,
                    orderbook_snapshot=orderbook,
                    market_metrics=metrics,
                    executed_signal=sample_signal
                )

        # Проверяем что данные собраны для всех символов
        assert len(ml_collector.feature_buffers) == 3
        assert len(ml_collector.sample_counts) == 3

        for symbol in symbols:
            assert ml_collector.sample_counts[symbol] == 5

        print("✓ Данные корректно собираются для нескольких символов")


def run_tests():
    """Запуск всех тестов."""
    print("\n" + "="*70)
    print("ЗАПУСК ТЕСТОВ ML DATA COLLECTOR")
    print("="*70)

    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s"  # Показывать print() выводы
    ])


if __name__ == "__main__":
    run_tests()