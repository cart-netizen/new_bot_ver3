"""
Тесты для MLDataCollector - проверка сохранения семплов.

Файл: backend/tests/ml_engine/test_ml_data_collection.py

Описание:
    Комплексные тесты для проверки корректности работы системы сбора
    и сохранения ML данных. Тестирует инициализацию, сбор данных,
    автоматическое сохранение батчей и форматы выходных файлов.
"""
import sys
from pathlib import Path

# Определяем путь к backend директории
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

import pytest
import json
import numpy as np
from datetime import datetime
import tempfile
import shutil
from typing import List

from ml_engine.data_collection import MLDataCollector
from ml_engine.features import FeatureVector, FeaturePipeline
from ml_engine.features.candle_feature_extractor import Candle
from models.orderbook import OrderBookSnapshot, OrderBookMetrics


# Устанавливаем маркер для всех тестов в этом модуле
pytestmark = pytest.mark.asyncio


class TestMLDataCollector:
    """
    Тестовый класс для проверки MLDataCollector.

    Покрывает следующие сценарии:
    - Инициализация коллектора
    - Логика интервала сбора
    - Сбор одиночных семплов
    - Автоматическое сохранение при достижении лимита
    - Форматы сохраненных файлов (features, labels, metadata)
    - Финализация и сохранение остаточных данных
    - Статистика сбора
    - Работа с множественными символами
    """

    # ============================================================================
    # FIXTURES - Подготовка тестовых данных
    # ============================================================================

    @pytest.fixture
    def temp_storage(self) -> str:
        """
        Создание временной директории для хранения тестовых данных.

        Returns:
            str: Путь к временной директории
        """
        temp_dir = tempfile.mkdtemp(prefix="ml_test_")
        print(f"\n[FIXTURE] Создана временная директория: {temp_dir}")
        yield temp_dir

        # Cleanup после теста
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[FIXTURE] Удалена временная директория: {temp_dir}")
        except Exception as e:
            print(f"[FIXTURE] Ошибка при удалении директории: {e}")

    @pytest.fixture
    def ml_collector(self, temp_storage: str) -> MLDataCollector:
        """
        Создание экземпляра MLDataCollector с тестовыми параметрами.

        Args:
            temp_storage: Путь к временной директории

        Returns:
            MLDataCollector: Настроенный коллектор для тестов
        """
        collector = MLDataCollector(
            storage_path=temp_storage,
            max_samples_per_file=10,  # Малое значение для быстрого тестирования
            collection_interval=1      # Собираем каждую итерацию
        )
        print(f"[FIXTURE] Создан MLDataCollector с max_samples={10}")
        return collector

    @pytest.fixture
    def sample_orderbook(self) -> OrderBookSnapshot:
        """
        Создание тестового снимка стакана ордеров.

        Returns:
            OrderBookSnapshot: Снимок стакана для тестов
        """
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
    def sample_metrics(self) -> OrderBookMetrics:
        """
        Создание тестовых рыночных метрик.

        Returns:
            OrderBookMetrics: Метрики для тестов
        """
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
    def sample_candles(self) -> List[Candle]:
        """
        Создание тестовых свечей для расчета индикаторов.

        Returns:
            List[Candle]: Список из 60 свечей
        """
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

        return candles

    @pytest.fixture
    async def sample_feature_vector(
        self,
        sample_orderbook: OrderBookSnapshot,
        sample_candles: List[Candle]
    ) -> FeatureVector:
        """
        Создание тестового вектора признаков.

        Важно: Это async fixture, так как извлечение признаков асинхронно.

        Args:
            sample_orderbook: Снимок стакана
            sample_candles: Список свечей

        Returns:
            FeatureVector: Вектор признаков для ML
        """
        # Создаем pipeline и извлекаем признаки
        pipeline = FeaturePipeline("TESTUSDT", normalize=False)

        feature_vector = await pipeline.extract_features(
            orderbook_snapshot=sample_orderbook,
            candles=sample_candles
        )

        print(f"[FIXTURE] Создан FeatureVector с {feature_vector.feature_count} признаками")
        return feature_vector

    @pytest.fixture
    def sample_signal(self) -> dict:
        """
        Создание тестового торгового сигнала.

        Returns:
            dict: Словарь с данными сигнала
        """
        return {
            "type": "buy",
            "confidence": 0.85,
            "strength": "strong"
        }

    # ============================================================================
    # ТЕСТЫ - Основная функциональность
    # ============================================================================

    async def test_collector_initialization(
        self,
        ml_collector: MLDataCollector,
        temp_storage: str
    ):
        """
        Тест 1: Проверка корректной инициализации коллектора.

        Проверяет:
        - Создание директорий для хранения
        - Инициализация внутренних структур данных
        """
        print("\n" + "="*70)
        print("ТЕСТ 1: Инициализация MLDataCollector")
        print("="*70)

        await ml_collector.initialize()

        # Проверяем что директория создана
        storage_path = Path(temp_storage)
        assert storage_path.exists(), "Директория хранения не создана"
        assert storage_path.is_dir(), "Путь хранения не является директорией"

        print("✓ Коллектор инициализирован успешно")
        print(f"  - Директория хранения: {temp_storage}")

    async def test_should_collect(self, ml_collector: MLDataCollector):
        """
        Тест 2: Проверка логики интервала сбора данных.

        Проверяет:
        - Корректность работы collection_interval
        - Счетчик итераций
        """
        print("\n" + "="*70)
        print("ТЕСТ 2: Логика интервала сбора")
        print("="*70)

        # Первая итерация - должно быть True (счетчик = 0)
        should_collect_1 = ml_collector.should_collect()
        assert should_collect_1 == True, "Первый вызов должен возвращать True"
        print(f"✓ Первая итерация: should_collect = {should_collect_1}")

        # Следующие итерации зависят от collection_interval
        # У нас interval=1, поэтому каждая итерация должна возвращать True
        should_collect_2 = ml_collector.should_collect()
        assert should_collect_2 == True, "Со interval=1 каждая итерация должна собирать"
        print(f"✓ Вторая итерация: should_collect = {should_collect_2}")

        print("✓ Логика интервала сбора работает корректно")

    async def test_collect_single_sample(
        self,
        ml_collector: MLDataCollector,
        sample_orderbook: OrderBookSnapshot,
        sample_metrics: OrderBookMetrics,
        sample_feature_vector: FeatureVector,
        sample_signal: dict
    ):
        """
        Тест 3: Проверка сбора одного семпла данных.

        Проверяет:
        - Добавление данных в буферы
        - Обновление счетчиков
        - Корректность внутренних структур
        """
        print("\n" + "="*70)
        print("ТЕСТ 3: Сбор одиночного семпла")
        print("="*70)

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
        assert ml_collector.sample_counts["TESTUSDT"] == 1, "Счетчик семплов неверный"
        assert ml_collector.total_samples_collected == 1, "Общий счетчик неверный"

        # Проверяем буферы
        assert len(ml_collector.feature_buffers["TESTUSDT"]) == 1, "Feature buffer неверный"
        assert len(ml_collector.label_buffers["TESTUSDT"]) == 1, "Label buffer неверный"
        assert len(ml_collector.metadata_buffers["TESTUSDT"]) == 1, "Metadata buffer неверный"

        print("✓ Семпл успешно добавлен в буферы")
        print(f"  - Счетчик семплов: {ml_collector.sample_counts['TESTUSDT']}")
        print(f"  - Размер feature buffer: {len(ml_collector.feature_buffers['TESTUSDT'])}")
        print(f"  - Размер label buffer: {len(ml_collector.label_buffers['TESTUSDT'])}")
        print(f"  - Размер metadata buffer: {len(ml_collector.metadata_buffers['TESTUSDT'])}")

    async def test_batch_save_trigger(
        self,
        ml_collector: MLDataCollector,
        sample_orderbook: OrderBookSnapshot,
        sample_metrics: OrderBookMetrics,
        sample_feature_vector: FeatureVector,
        sample_signal: dict,
        temp_storage: str
    ):
        """
        Тест 4: Проверка автоматического сохранения при достижении лимита.

        Проверяет:
        - Триггер сохранения при max_samples_per_file
        - Создание файлов
        - Очистку буферов после сохранения
        """
        print("\n" + "="*70)
        print("ТЕСТ 4: Автоматическое сохранение batch")
        print("="*70)

        await ml_collector.initialize()

        # Собираем семплы до достижения max_samples_per_file (10) и чуть больше
        num_samples = 12
        print(f"Собираем {num_samples} семплов (лимит = 10)...")

        for i in range(num_samples):
            await ml_collector.collect_sample(
                symbol="TESTUSDT",
                feature_vector=sample_feature_vector,
                orderbook_snapshot=sample_orderbook,
                market_metrics=sample_metrics,
                executed_signal=sample_signal
            )
            if (i + 1) % 5 == 0:
                print(f"  - Собрано {i + 1} семплов...")

        # После 10 семплов должен был сохраниться batch
        symbol_dir = Path(temp_storage) / "TESTUSDT"
        features_dir = symbol_dir / "features"
        labels_dir = symbol_dir / "labels"
        metadata_dir = symbol_dir / "metadata"

        # Проверяем что директории созданы
        assert features_dir.exists(), "Директория features не создана"
        assert labels_dir.exists(), "Директория labels не создана"
        assert metadata_dir.exists(), "Директория metadata не создана"
        print("✓ Все директории созданы")

        # Проверяем что файлы созданы
        feature_files = list(features_dir.glob("*.npy"))
        label_files = list(labels_dir.glob("*.json"))
        metadata_files = list(metadata_dir.glob("*.json"))

        assert len(feature_files) >= 1, "Features файл не создан"
        assert len(label_files) >= 1, "Labels файл не создан"
        assert len(metadata_files) >= 1, "Metadata файл не создан"

        print(f"✓ Файлы созданы:")
        print(f"  - Features: {len(feature_files)} файл(ов)")
        print(f"  - Labels: {len(label_files)} файл(ов)")
        print(f"  - Metadata: {len(metadata_files)} файл(ов)")

        # Буферы должны быть частично очищены после сохранения
        # Осталось только 2 семпла в буфере (12 - 10)
        buffer_size = len(ml_collector.feature_buffers["TESTUSDT"])
        assert buffer_size == 2, f"Ожидалось 2 семпла в буфере, получено {buffer_size}"

        print(f"✓ Буферы корректно очищены, осталось: {buffer_size} семплов")
        print("✓ Batch автоматически сохранен при достижении лимита")

    async def test_saved_features_format(
        self,
        ml_collector: MLDataCollector,
        sample_orderbook: OrderBookSnapshot,
        sample_metrics: OrderBookMetrics,
        sample_feature_vector: FeatureVector,
        sample_signal: dict,
        temp_storage: str
    ):
        """
        Тест 5: Проверка формата сохраненных features.

        Проверяет:
        - Корректность размерности numpy array
        - Тип данных (float32)
        - Отсутствие NaN и Inf значений
        - Диапазон значений
        """
        print("\n" + "="*70)
        print("ТЕСТ 5: Формат сохраненных features")
        print("="*70)

        await ml_collector.initialize()

        # Собираем 10 семплов для сохранения
        num_samples = 10
        print(f"Собираем {num_samples} семплов...")

        for i in range(num_samples):
            await ml_collector.collect_sample(
                symbol="TESTUSDT",
                feature_vector=sample_feature_vector,
                orderbook_snapshot=sample_orderbook,
                market_metrics=sample_metrics,
                executed_signal=sample_signal
            )

        # Читаем сохраненный features файл
        features_dir = Path(temp_storage) / "TESTUSDT" / "features"
        feature_files = list(features_dir.glob("*.npy"))

        assert len(feature_files) > 0, "Features файл не найден"
        feature_file = feature_files[0]
        print(f"✓ Найден файл: {feature_file.name}")

        features_array = np.load(feature_file)

        # Проверяем размерность
        expected_shape = (10, 110)  # 10 семплов, 110 признаков
        assert features_array.shape == expected_shape, \
            f"Неправильная размерность: {features_array.shape}, ожидалось {expected_shape}"
        print(f"✓ Размерность корректна: {features_array.shape}")

        # Проверяем тип данных
        assert features_array.dtype == np.float32, \
            f"Неправильный тип данных: {features_array.dtype}, ожидалось float32"
        print(f"✓ Тип данных корректен: {features_array.dtype}")

        # Проверяем что нет NaN
        nan_count = np.sum(np.isnan(features_array))
        assert nan_count == 0, f"Обнаружены NaN значения: {nan_count}"
        print(f"✓ NaN значений не обнаружено")

        # Проверяем что нет Inf
        inf_count = np.sum(np.isinf(features_array))
        assert inf_count == 0, f"Обнаружены Inf значения: {inf_count}"
        print(f"✓ Inf значений не обнаружено")

        # Статистика
        print(f"\nСтатистика features:")
        print(f"  - Min: {features_array.min():.4f}")
        print(f"  - Max: {features_array.max():.4f}")
        print(f"  - Mean: {features_array.mean():.4f}")
        print(f"  - Std: {features_array.std():.4f}")

        print("✓ Features сохранены в корректном формате")

    async def test_saved_labels_content(
        self,
        ml_collector: MLDataCollector,
        sample_orderbook: OrderBookSnapshot,
        sample_metrics: OrderBookMetrics,
        sample_feature_vector: FeatureVector,
        sample_signal: dict,
        temp_storage: str
    ):
        """
        Тест 6: Проверка содержимого сохраненных labels.

        Проверяет:
        - Структуру JSON файла
        - Наличие всех обязательных полей
        - Корректность значений
        """
        print("\n" + "="*70)
        print("ТЕСТ 6: Содержимое сохраненных labels")
        print("="*70)

        await ml_collector.initialize()

        # Собираем 10 семплов
        num_samples = 10
        print(f"Собираем {num_samples} семплов...")

        for i in range(num_samples):
            await ml_collector.collect_sample(
                symbol="TESTUSDT",
                feature_vector=sample_feature_vector,
                orderbook_snapshot=sample_orderbook,
                market_metrics=sample_metrics,
                executed_signal=sample_signal
            )

        # Читаем сохраненный labels файл
        labels_dir = Path(temp_storage) / "TESTUSDT" / "labels"
        label_files = list(labels_dir.glob("*.json"))

        assert len(label_files) > 0, "Labels файл не найден"
        label_file = label_files[0]
        print(f"✓ Найден файл: {label_file.name}")

        with open(label_file, 'r') as f:
            labels = json.load(f)

        # Проверяем количество
        assert len(labels) == num_samples, \
            f"Неправильное количество labels: {len(labels)}, ожидалось {num_samples}"
        print(f"✓ Количество labels корректно: {len(labels)}")

        # Проверяем структуру первого label
        first_label = labels[0]
        print("\nПроверка структуры label...")

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
            print(f"  ✓ Поле '{field}' присутствует")

        # Проверяем значения
        assert first_label["current_mid_price"] == sample_orderbook.mid_price, \
            "Неверное значение mid_price"
        assert first_label["current_imbalance"] == sample_metrics.imbalance, \
            "Неверное значение imbalance"
        assert first_label["signal_type"] == "buy", "Неверный тип сигнала"
        assert first_label["signal_confidence"] == 0.85, "Неверная уверенность сигнала"
        assert first_label["signal_strength"] == "strong", "Неверная сила сигнала"

        print("\n✓ Все значения корректны:")
        print(f"  - Signal: {first_label['signal_type']} (conf={first_label['signal_confidence']})")
        print(f"  - Price: {first_label['current_mid_price']}")
        print(f"  - Imbalance: {first_label['current_imbalance']:.4f}")

        # Проверяем что future поля None (еще не рассчитаны)
        assert first_label["future_direction_10s"] is None, "future_direction должно быть None"
        assert first_label["future_movement_10s"] is None, "future_movement должно быть None"
        print("✓ Future поля корректно установлены в None (еще не рассчитаны)")

        print("✓ Labels сохранены с правильной структурой")

    async def test_saved_metadata_content(
        self,
        ml_collector: MLDataCollector,
        sample_orderbook: OrderBookSnapshot,
        sample_metrics: OrderBookMetrics,
        sample_feature_vector: FeatureVector,
        sample_signal: dict,
        temp_storage: str
    ):
        """
        Тест 7: Проверка содержимого сохраненных metadata.

        Проверяет:
        - Информацию о batch
        - Детали каждого семпла
        - Временные метки
        """
        print("\n" + "="*70)
        print("ТЕСТ 7: Содержимое сохраненных metadata")
        print("="*70)

        await ml_collector.initialize()

        # Собираем 10 семплов
        num_samples = 10
        print(f"Собираем {num_samples} семплов...")

        for i in range(num_samples):
            await ml_collector.collect_sample(
                symbol="TESTUSDT",
                feature_vector=sample_feature_vector,
                orderbook_snapshot=sample_orderbook,
                market_metrics=sample_metrics,
                executed_signal=sample_signal
            )

        # Читаем сохраненный metadata файл
        metadata_dir = Path(temp_storage) / "TESTUSDT" / "metadata"
        metadata_files = list(metadata_dir.glob("*.json"))

        assert len(metadata_files) > 0, "Metadata файл не найден"
        metadata_file = metadata_files[0]
        print(f"✓ Найден файл: {metadata_file.name}")

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Проверяем batch_info
        assert "batch_info" in metadata, "Отсутствует batch_info"
        batch_info = metadata["batch_info"]
        print("\n✓ batch_info присутствует")

        # Проверяем поля batch_info
        assert batch_info["symbol"] == "TESTUSDT", "Неверный symbol в batch_info"
        assert batch_info["batch_number"] == 1, "Неверный batch_number"
        assert batch_info["sample_count"] == num_samples, "Неверный sample_count"
        assert "timestamp" in batch_info, "Отсутствует timestamp"
        assert batch_info["feature_shape"] == [num_samples, 110], "Неверный feature_shape"

        print(f"  - Symbol: {batch_info['symbol']}")
        print(f"  - Batch number: {batch_info['batch_number']}")
        print(f"  - Sample count: {batch_info['sample_count']}")
        print(f"  - Feature shape: {batch_info['feature_shape']}")

        # Проверяем samples
        assert "samples" in metadata, "Отсутствует samples"
        samples = metadata["samples"]
        assert len(samples) == num_samples, f"Неверное количество samples: {len(samples)}"
        print(f"\n✓ Samples присутствуют: {len(samples)} записей")

        # Проверяем структуру первого sample
        first_sample = samples[0]
        print("\nПроверка структуры sample...")

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
            assert field in first_sample, f"Отсутствует поле: {field}"
            print(f"  ✓ Поле '{field}' присутствует")

        # Проверяем значения
        assert first_sample["symbol"] == "TESTUSDT", "Неверный symbol"
        assert first_sample["timestamp"] == sample_orderbook.timestamp, "Неверный timestamp"
        assert first_sample["mid_price"] == sample_orderbook.mid_price, "Неверный mid_price"
        assert first_sample["spread"] == sample_orderbook.spread, "Неверный spread"
        assert first_sample["imbalance"] == sample_metrics.imbalance, "Неверный imbalance"
        assert first_sample["signal_type"] == "buy", "Неверный signal_type"
        assert first_sample["signal_confidence"] == 0.85, "Неверный signal_confidence"
        assert first_sample["signal_strength"] == "strong", "Неверный signal_strength"
        assert first_sample["feature_count"] == 110, "Неверный feature_count"

        print("\n✓ Все значения корректны:")
        print(f"  - Timestamp: {first_sample['timestamp']}")
        print(f"  - Mid price: {first_sample['mid_price']}")
        print(f"  - Spread: {first_sample['spread']}")
        print(f"  - Imbalance: {first_sample['imbalance']:.4f}")
        print(f"  - Feature count: {first_sample['feature_count']}")

        print("✓ Metadata сохранены с правильной структурой")

    async def test_finalize_saves_remaining_buffer(
        self,
        ml_collector: MLDataCollector,
        sample_orderbook: OrderBookSnapshot,
        sample_metrics: OrderBookMetrics,
        sample_feature_vector: FeatureVector,
        sample_signal: dict,
        temp_storage: str
    ):
        """
        Тест 8: Проверка что finalize() сохраняет остаточные данные.

        Проверяет:
        - Сохранение данных меньше max_samples_per_file
        - Очистку буферов после финализации
        """
        print("\n" + "="*70)
        print("ТЕСТ 8: Финализация и сохранение остаточных данных")
        print("="*70)

        await ml_collector.initialize()

        # Собираем только 5 семплов (меньше чем max_samples_per_file=10)
        num_samples = 5
        print(f"Собираем {num_samples} семплов (меньше лимита 10)...")

        for i in range(num_samples):
            await ml_collector.collect_sample(
                symbol="TESTUSDT",
                feature_vector=sample_feature_vector,
                orderbook_snapshot=sample_orderbook,
                market_metrics=sample_metrics,
                executed_signal=sample_signal
            )

        # Буфер должен содержать 5 семплов
        buffer_size_before = len(ml_collector.feature_buffers["TESTUSDT"])
        assert buffer_size_before == num_samples, \
            f"Ожидалось {num_samples} семплов в буфере, получено {buffer_size_before}"
        print(f"✓ Буфер содержит {buffer_size_before} семплов перед финализацией")

        # Вызываем finalize()
        print("Вызываем finalize()...")
        await ml_collector.finalize()

        # Проверяем что данные сохранены
        features_dir = Path(temp_storage) / "TESTUSDT" / "features"
        feature_files = list(features_dir.glob("*.npy"))

        assert len(feature_files) == 1, "Batch не сохранен при finalize()"
        print(f"✓ Создан файл: {feature_files[0].name}")

        # Проверяем количество семплов в файле
        features_array = np.load(feature_files[0])
        assert features_array.shape[0] == num_samples, \
            f"Неправильное количество семплов: {features_array.shape[0]}, ожидалось {num_samples}"
        print(f"✓ Файл содержит {features_array.shape[0]} семплов")

        # Буфер должен быть очищен
        buffer_size_after = len(ml_collector.feature_buffers["TESTUSDT"])
        assert buffer_size_after == 0, \
            f"Буфер не очищен после finalize(), содержит {buffer_size_after} семплов"
        print(f"✓ Буфер очищен после финализации")

        print("✓ Finalize() корректно сохранил остаточные данные")

    async def test_statistics(
        self,
        ml_collector: MLDataCollector,
        sample_orderbook: OrderBookSnapshot,
        sample_metrics: OrderBookMetrics,
        sample_feature_vector: FeatureVector,
        sample_signal: dict
    ):
        """
        Тест 9: Проверка статистики сбора данных.

        Проверяет:
        - Общие счетчики
        - Статистику по символам
        - Счетчики файлов и батчей
        """
        print("\n" + "="*70)
        print("ТЕСТ 9: Статистика сбора данных")
        print("="*70)

        await ml_collector.initialize()

        # Собираем 15 семплов (больше чем max_samples_per_file)
        num_samples = 15
        print(f"Собираем {num_samples} семплов...")

        for i in range(num_samples):
            await ml_collector.collect_sample(
                symbol="TESTUSDT",
                feature_vector=sample_feature_vector,
                orderbook_snapshot=sample_orderbook,
                market_metrics=sample_metrics,
                executed_signal=sample_signal
            )
            if (i + 1) % 5 == 0:
                print(f"  - Собрано {i + 1} семплов...")

        # Получаем статистику
        stats = ml_collector.get_statistics()
        print("\nПолученная статистика:")

        # Проверяем общую статистику
        assert stats["total_samples_collected"] == num_samples, \
            f"Неверное total_samples_collected: {stats['total_samples_collected']}"
        print(f"✓ Total samples: {stats['total_samples_collected']}")

        assert stats["files_written"] >= 3, \
            f"Неверное files_written: {stats['files_written']}"
        print(f"✓ Files written: {stats['files_written']}")

        assert stats["collection_interval"] == 1, "Неверный collection_interval"
        print(f"✓ Collection interval: {stats['collection_interval']}")

        # Проверяем статистику по символу
        assert "symbols" in stats, "Отсутствует symbols в статистике"
        assert "TESTUSDT" in stats["symbols"], "Отсутствует TESTUSDT в символах"

        symbol_stats = stats["symbols"]["TESTUSDT"]
        print(f"\nСтатистика по TESTUSDT:")

        assert symbol_stats["total_samples"] == num_samples, \
            f"Неверное total_samples: {symbol_stats['total_samples']}"
        print(f"  - Total samples: {symbol_stats['total_samples']}")

        assert symbol_stats["current_batch"] == 2, \
            f"Неверное current_batch: {symbol_stats['current_batch']}"
        print(f"  - Current batch: {symbol_stats['current_batch']}")

        # 15 семплов - 10 сохранено = 5 в буфере
        assert symbol_stats["buffer_size"] == 5, \
            f"Неверное buffer_size: {symbol_stats['buffer_size']}"
        print(f"  - Buffer size: {symbol_stats['buffer_size']}")

        print("✓ Статистика рассчитывается корректно")

    async def test_multiple_symbols(
        self,
        ml_collector: MLDataCollector,
        sample_orderbook: OrderBookSnapshot,
        sample_metrics: OrderBookMetrics,
        sample_feature_vector: FeatureVector,
        sample_signal: dict,
        temp_storage: str
    ):
        """
        Тест 10: Проверка работы с несколькими символами одновременно.

        Проверяет:
        - Независимость буферов для разных символов
        - Корректность счетчиков
        - Изоляцию данных между символами
        """
        print("\n" + "="*70)
        print("ТЕСТ 10: Работа с несколькими символами")
        print("="*70)

        await ml_collector.initialize()

        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        num_samples_per_symbol = 5

        print(f"Собираем данные для {len(symbols)} символов...")

        # Собираем данные для каждого символа
        for symbol in symbols:
            print(f"\nСбор данных для {symbol}...")
            for i in range(num_samples_per_symbol):
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
                    total_bid_volume=sample_metrics.total_bid_volume,
                    total_ask_volume=sample_metrics.total_ask_volume,
                    imbalance=sample_metrics.imbalance
                )

                await ml_collector.collect_sample(
                    symbol=symbol,
                    feature_vector=sample_feature_vector,
                    orderbook_snapshot=orderbook,
                    market_metrics=metrics,
                    executed_signal=sample_signal
                )

            print(f"  ✓ Собрано {num_samples_per_symbol} семплов для {symbol}")

        # Проверяем что данные собраны для всех символов
        assert len(ml_collector.feature_buffers) == len(symbols), \
            f"Неверное количество буферов: {len(ml_collector.feature_buffers)}"
        print(f"\n✓ Создано {len(ml_collector.feature_buffers)} буферов для {len(symbols)} символов")

        assert len(ml_collector.sample_counts) == len(symbols), \
            f"Неверное количество счетчиков: {len(ml_collector.sample_counts)}"

        # Проверяем счетчики для каждого символа
        print("\nПроверка счетчиков:")
        for symbol in symbols:
            count = ml_collector.sample_counts[symbol]
            assert count == num_samples_per_symbol, \
                f"Неверное количество семплов для {symbol}: {count}"
            print(f"  ✓ {symbol}: {count} семплов")

        # Проверяем изоляцию данных - каждый символ должен иметь свои директории
        print("\nПроверка изоляции директорий:")
        for symbol in symbols:
            symbol_dir = Path(temp_storage) / symbol
            assert symbol_dir.exists(), f"Директория для {symbol} не создана"
            print(f"  ✓ Директория {symbol} создана")

        print("\n✓ Данные корректно собираются для нескольких символов")
        print("✓ Буферы изолированы и работают независимо")


# ============================================================================
# RUNNER - Запуск тестов
# ============================================================================

def run_tests():
    """
    Запуск всех тестов с конфигурацией pytest.

    Использует:
    - Verbose режим для детального вывода
    - Short traceback для компактности
    - Вывод print() для отслеживания прогресса
    """
    print("\n" + "="*70)
    print("ЗАПУСК ТЕСТОВ ML DATA COLLECTOR")
    print("="*70)
    print("Тестовый набор: Проверка сбора и сохранения ML данных")
    print("Количество тестов: 10")
    print("="*70 + "\n")

    exit_code = pytest.main([
        __file__,
        "-v",              # Verbose - детальный вывод
        "--tb=short",      # Short traceback
        "-s",              # Показывать print() выводы
        "--color=yes",     # Цветной вывод
        "-p", "no:warnings"  # Отключить предупреждения для чистого вывода
    ])

    print("\n" + "="*70)
    if exit_code == 0:
        print("✓✓✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО ✓✓✓")
    else:
        print("✗✗✗ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ ✗✗✗")
    print("="*70 + "\n")

    return exit_code


if __name__ == "__main__":
    run_tests()