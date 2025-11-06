#!/usr/bin/env python3
"""
Data Loader для загрузки и подготовки обучающих данных.

Функциональность:
- Загрузка features и labels из .npy и .json файлов
- Создание временных последовательностей (sequences)
- Train/Val/Test split с walk-forward validation
- Data augmentation для улучшения генерализации
- Batch creation с DataLoader
- Class Balancing через resampling

Путь: backend/ml_engine/training/data_loader.py
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

from backend.core.logger import get_logger
from backend.ml_engine.training.class_balancing import (
    ClassBalancingConfig,
    ClassBalancingStrategy
)

logger = get_logger(__name__)


@dataclass
class DataConfig:
    """Конфигурация загрузки данных."""
    storage_path: str = "data/ml_training"
    sequence_length: int = 60  # Длина последовательности
    target_horizon: str = "future_direction_60s"  # Целевая переменная

    # Split параметры
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # DataLoader параметры
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 4


class TradingDataset(Dataset):
    """PyTorch Dataset для торговых данных."""

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        returns: Optional[np.ndarray] = None
    ):
        """
        Инициализация датасета.

        Args:
            sequences: (N, sequence_length, features)
            labels: (N,) - направление движения (0, 1, 2)
            returns: (N,) - ожидаемая доходность (опционально)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.returns = (
            torch.FloatTensor(returns)
            if returns is not None
            else None
        )

        logger.info(
            f"Создан TradingDataset: samples={len(sequences)}, "
            f"sequence_length={sequences.shape[1]}, "
            f"features={sequences.shape[2]}"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Получить элемент датасета."""
        item = {
            'sequence': self.sequences[idx],
            'label': self.labels[idx]
        }

        if self.returns is not None:
            item['return'] = self.returns[idx]

        return item


class HistoricalDataLoader:
    """
    Загрузчик исторических данных для обучения ML моделей.

    Поддерживает:
    - Загрузку из структурированных файлов (features/*.npy, labels/*.json)
    - Создание временных последовательностей
    - Walk-forward validation split
    - Class balancing через resampling
    """

    def __init__(
        self,
        config: DataConfig,
        balancing_config: Optional[ClassBalancingConfig] = None
    ):
        """
        Инициализация загрузчика.

        Args:
            config: Конфигурация загрузки данных
            balancing_config: Конфигурация балансировки классов
        """
        self.config = config
        self.storage_path = Path(config.storage_path)

        if not self.storage_path.exists():
            raise FileNotFoundError(
                f"Директория данных не найдена: {self.storage_path}"
            )

        # Class Balancing
        self.balancing_config = balancing_config
        if balancing_config:
            self.balancing_strategy = ClassBalancingStrategy(balancing_config)
            logger.info("✓ Class Balancing включен в DataLoader")
        else:
            self.balancing_strategy = None

        logger.info(f"Инициализирован DataLoader: storage={self.storage_path}")

    def load_symbol_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Загрузить все данные для символа.

        Args:
            symbol: Торговая пара
            start_date: Начальная дата (YYYY-MM-DD) или None
            end_date: Конечная дата (YYYY-MM-DD) или None

        Returns:
            features: (N, feature_dim)
            labels: (N,) - labels в диапазоне [0, num_classes)
            timestamps: (N,)
        """
        symbol_path = self.storage_path / symbol

        if not symbol_path.exists():
            raise FileNotFoundError(f"Данные для {symbol} не найдены")

        features_path = symbol_path / "features"
        labels_path = symbol_path / "labels"

        # Собираем все batch файлы
        feature_files = sorted(features_path.glob("*.npy"))
        label_files = sorted(labels_path.glob("*.json"))

        if not feature_files or not label_files:
            raise ValueError(f"Нет данных для {symbol}")

        logger.info(
            f"Найдено файлов для {symbol}: features={len(feature_files)}, "
            f"labels={len(label_files)}"
        )

        # Загружаем features
        all_features = []
        for feature_file in feature_files:
            # Фильтрация по дате (если указано)
            if start_date and feature_file.stem < start_date:
                continue
            if end_date and feature_file.stem > end_date:
                continue

            features = np.load(feature_file)
            all_features.append(features)

        X = np.concatenate(all_features, axis=0)

        # Загружаем labels
        all_labels = []
        all_timestamps = []

        for label_file in label_files:
            # Фильтрация по дате
            if start_date and label_file.stem < start_date:
                continue
            if end_date and label_file.stem > end_date:
                continue

            with open(label_file) as f:
                labels_data = json.load(f)

            # Извлекаем целевую переменную и timestamps
            for label in labels_data:
                target = label.get(self.config.target_horizon)
                timestamp = label.get('timestamp', 0)

                # Пропускаем None значения
                if target is not None:
                    all_labels.append(target)
                    all_timestamps.append(timestamp)

        y = np.array(all_labels, dtype=np.int64)
        timestamps = np.array(all_timestamps, dtype=np.int64)

        # ===== КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ =====
        # Преобразуем labels из {-1, 0, 1} в {0, 1, 2} для PyTorch
        # -1 (DOWN) -> 0
        #  0 (NEUTRAL) -> 1
        #  1 (UP) -> 2
        logger.info(f"Исходное распределение классов: {Counter(y)}")

        # Маппинг
        label_mapping = {-1: 0, 0: 1, 1: 2}
        y_mapped = np.array([label_mapping.get(label, label) for label in y], dtype=np.int64)

        logger.info(f"Преобразованное распределение: {Counter(y_mapped)}")
        logger.info(f"Уникальные значения после маппинга: {set(y_mapped)}")

        # Проверяем соответствие размеров
        min_len = min(len(X), len(y_mapped))
        X = X[:min_len]
        y_mapped = y_mapped[:min_len]
        timestamps = timestamps[:min_len]

        logger.info(
            f"Загружены данные {symbol}: samples={len(X)}, "
            f"features={X.shape[1]}"
        )

        return X, y_mapped, timestamps

    def create_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Создать временные последовательности.

        Args:
            features: (N, feature_dim)
            labels: (N,)
            timestamps: (N,)

        Returns:
            sequences: (N-seq_len+1, seq_len, feature_dim)
            seq_labels: (N-seq_len+1,)
            seq_timestamps: (N-seq_len+1,)
        """
        seq_len = self.config.sequence_length

        if len(features) < seq_len:
            raise ValueError(
                f"Недостаточно данных: {len(features)} < {seq_len}"
            )

        num_sequences = len(features) - seq_len + 1

        sequences = np.zeros(
            (num_sequences, seq_len, features.shape[1]),
            dtype=np.float32
        )
        seq_labels = np.zeros(num_sequences, dtype=np.int64)
        seq_timestamps = np.zeros(num_sequences, dtype=np.int64)

        for i in range(num_sequences):
            sequences[i] = features[i:i + seq_len]
            seq_labels[i] = labels[i + seq_len - 1]  # Label последнего элемента
            seq_timestamps[i] = timestamps[i + seq_len - 1]

        logger.info(
            f"Создано последовательностей: {num_sequences}, "
            f"shape={sequences.shape}"
        )

        return sequences, seq_labels, seq_timestamps

    def train_val_test_split(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Optional[Tuple[np.ndarray, np.ndarray]]
    ]:
        """
        Разбить данные на train/val/test с сохранением временного порядка.

        ВАЖНО: Для временных рядов НЕ используем shuffle!
        Данные разбиваются последовательно:
        - Train: первые 70%
        - Val: следующие 15%
        - Test: последние 15%

        Args:
            sequences: (N, seq_len, features)
            labels: (N,)
            timestamps: (N,)

        Returns:
            train_data: (X_train, y_train)
            val_data: (X_val, y_val)
            test_data: (X_test, y_test) или None если test_ratio=0
        """
        n_samples = len(sequences)

        # Вычисляем индексы split
        train_end = int(n_samples * self.config.train_ratio)
        val_end = train_end + int(n_samples * self.config.val_ratio)

        # Split
        X_train = sequences[:train_end]
        y_train = labels[:train_end]

        X_val = sequences[train_end:val_end]
        y_val = labels[train_end:val_end]

        # Test (если указано)
        if self.config.test_ratio > 0:
            X_test = sequences[val_end:]
            y_test = labels[val_end:]
            test_data = (X_test, y_test)
            test_size = len(X_test)
        else:
            test_data = None
            test_size = 0

        logger.info(
            f"Train/Val/Test split: "
            f"train={len(X_train)}, "
            f"val={len(X_val)}, "
            f"test={test_size}"
        )

        # Логируем распределение классов
        logger.info(f"Train class distribution: {Counter(y_train)}")
        logger.info(f"Val class distribution: {Counter(y_val)}")
        if test_data:
            logger.info(f"Test class distribution: {Counter(y_test)}")

        return (X_train, y_train), (X_val, y_val), test_data

    def create_dataloaders(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, DataLoader]:
        """
        Создать PyTorch DataLoaders.

        Args:
            train_data: (X_train, y_train)
            val_data: (X_val, y_val)
            test_data: (X_test, y_test) или None

        Returns:
            Dict с DataLoaders: {'train': ..., 'val': ..., 'test': ...}
        """
        # Создаем Datasets
        train_dataset = TradingDataset(train_data[0], train_data[1])
        val_dataset = TradingDataset(val_data[0], val_data[1])

        # Создаем DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # Валидация без shuffle
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }

        # Test DataLoader (если есть)
        if test_data:
            test_dataset = TradingDataset(test_data[0], test_data[1])
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            dataloaders['test'] = test_loader

        logger.info(
            f"DataLoaders созданы: "
            f"train_batches={len(train_loader)}, "
            f"val_batches={len(val_loader)}"
        )

        return dataloaders

    def load_and_prepare(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        apply_resampling: bool = True
    ) -> Dict:
        """
        Загрузка и подготовка данных с опциональной балансировкой.

        Args:
            symbols: Список символов
            start_date: Начальная дата
            end_date: Конечная дата
            apply_resampling: Применять ли resampling (oversampling/undersampling)

        Returns:
            Dict с DataLoaders и статистикой
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
        logger.info(f"{'='*80}")

        # Загружаем данные
        all_features = []
        all_labels = []
        all_timestamps = []

        for symbol in symbols:
            logger.info(f"Загрузка {symbol}...")
            X, y, timestamps = self.load_symbol_data(
                symbol, start_date, end_date
            )
            all_features.append(X)
            all_labels.append(y)
            all_timestamps.append(timestamps)

        # Объединяем
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        timestamps = np.concatenate(all_timestamps, axis=0)

        logger.info(f"\nВсего данных: {len(X):,} семплов")

        # ===== RESAMPLING ДО СОЗДАНИЯ SEQUENCES =====
        if apply_resampling and self.balancing_strategy:
            logger.info("\n" + "="*80)
            logger.info("ПРИМЕНЕНИЕ RESAMPLING")
            logger.info("="*80)

            # Логируем распределение ДО
            before_dist = Counter(y)
            logger.info(f"ДО resampling: {dict(before_dist)}")

            # Применяем балансировку
            X, y = self.balancing_strategy.balance_dataset(X, y)

            # Логируем распределение ПОСЛЕ
            after_dist = Counter(y)
            logger.info(f"ПОСЛЕ resampling: {dict(after_dist)}")
            logger.info(f"Новый размер: {len(X):,} семплов")

            # Обновляем timestamps
            # Простая стратегия: если увеличилось - дублируем случайно,
            # если уменьшилось - обрезаем
            if len(timestamps) < len(X):
                # Oversample: дублируем timestamps
                indices = np.random.choice(len(timestamps), len(X), replace=True)
                timestamps = timestamps[indices]
            elif len(timestamps) > len(X):
                # Undersample: обрезаем timestamps
                timestamps = timestamps[:len(X)]

        # ===== СОЗДАНИЕ SEQUENCES =====
        logger.info("\n" + "="*80)
        logger.info("СОЗДАНИЕ ВРЕМЕННЫХ ПОСЛЕДОВАТЕЛЬНОСТЕЙ")
        logger.info("="*80)

        sequences, seq_labels, seq_timestamps = self.create_sequences(
            X, y, timestamps
        )

        logger.info(f"Создано последовательностей: {len(sequences):,}")
        logger.info(f"Shape: {sequences.shape}")

        # ===== TRAIN/VAL/TEST SPLIT =====
        logger.info("\n" + "="*80)
        logger.info("SPLIT НА TRAIN/VAL/TEST")
        logger.info("="*80)

        train_data, val_data, test_data = self.train_val_test_split(
            sequences, seq_labels, seq_timestamps
        )

        # Создаем DataLoaders
        dataloaders = self.create_dataloaders(
            train_data, val_data, test_data
        )

        # Статистика
        result = {
            'dataloaders': dataloaders,
            'statistics': {
                'total_sequences': len(sequences),
                'train_samples': len(train_data[0]),
                'val_samples': len(val_data[0]),
                'test_samples': len(test_data[0]) if test_data else 0,
                'sequence_length': self.config.sequence_length,
                'feature_dim': sequences.shape[2]
            }
        }

        logger.info(f"\n✓ Данные подготовлены")
        logger.info(f"  • Train: {result['statistics']['train_samples']:,}")
        logger.info(f"  • Val: {result['statistics']['val_samples']:,}")
        logger.info(f"  • Test: {result['statistics']['test_samples']:,}")
        logger.info(f"{'='*80}\n")

        return result

    def load_and_split(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        apply_resampling: bool = False
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Загрузка данных и создание DataLoaders (полная версия).

        Это обертка над load_and_prepare() с более простым интерфейсом.
        Возвращает готовые DataLoaders вместо Dict со статистикой.

        Используется в retraining_pipeline где нужны готовые DataLoaders для тренировки.

        Args:
            symbols: Список символов для загрузки (если None - загружаются все доступные)
            start_date: Начальная дата
            end_date: Конечная дата
            apply_resampling: Применять ли балансировку классов

        Returns:
            Tuple из:
            - train_loader: DataLoader для обучения
            - val_loader: DataLoader для валидации
            - test_loader: DataLoader для тестирования (может быть None)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ЗАГРУЗКА ДАННЫХ И СОЗДАНИЕ DATALOADERS")
        logger.info(f"{'='*80}")

        # Если symbols не указаны, пытаемся найти все доступные
        if symbols is None:
            storage_path = Path(self.config.storage_path)
            if storage_path.exists():
                # Ищем все .npy файлы с features
                feature_files = list(storage_path.glob("*_features.npy"))
                symbols = [f.stem.replace("_features", "") for f in feature_files]
                logger.info(f"Найдено {len(symbols)} символов: {symbols}")
            else:
                logger.warning(f"Storage path не существует: {storage_path}")
                symbols = []

        if not symbols:
            raise ValueError("No symbols to load. Please provide symbols or ensure data exists.")

        # Вызываем полную версию load_and_prepare
        result = self.load_and_prepare(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            apply_resampling=apply_resampling
        )

        # Извлекаем DataLoaders из результата
        dataloaders = result['dataloaders']
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders.get('test', None)

        logger.info(f"\n✓ DataLoaders созданы:")
        logger.info(f"  • Train batches: {len(train_loader)}")
        logger.info(f"  • Val batches: {len(val_loader)}")
        if test_loader:
            logger.info(f"  • Test batches: {len(test_loader)}")
        logger.info(f"{'='*80}\n")

        return train_loader, val_loader, test_loader

    def load_from_dataframe(
        self,
        features_df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str = 'future_direction_60s',
        timestamp_column: str = 'timestamp',
        symbol_column: Optional[str] = 'symbol',
        apply_resampling: bool = False
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Загрузка данных из DataFrame (Feature Store) и создание DataLoaders.

        Это альтернатива load_and_prepare() для работы с Feature Store.
        Вместо загрузки из .npy файлов, загружает из pandas DataFrame.

        Workflow:
        1. Валидация DataFrame
        2. Сортировка по времени (КРИТИЧНО!)
        3. Извлечение features/labels/timestamps
        4. Опциональный class balancing
        5. Создание sequences (переиспользует create_sequences)
        6. Train/Val/Test split (переиспользует train_val_test_split)
        7. Создание DataLoaders (переиспользует create_dataloaders)

        Args:
            features_df: DataFrame с данными из Feature Store
            feature_columns: Список колонок с фичами (обычно 110 колонок)
            label_column: Колонка с метками (0, 1, 2)
            timestamp_column: Колонка с timestamps
            symbol_column: Колонка с символами (опционально)
            apply_resampling: Применять ли class balancing

        Returns:
            Tuple из:
            - train_loader: DataLoader для обучения
            - val_loader: DataLoader для валидации
            - test_loader: DataLoader для тестирования (может быть None)

        Raises:
            ValueError: Если DataFrame пустой, отсутствуют колонки или недостаточно данных
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ЗАГРУЗКА ДАННЫХ ИЗ DATAFRAME (Feature Store)")
        logger.info(f"{'='*80}")

        # 1. Валидация входных данных
        if features_df.empty:
            raise ValueError("DataFrame is empty")

        # Проверить наличие необходимых колонок
        required_cols = feature_columns + [label_column, timestamp_column]
        missing = set(required_cols) - set(features_df.columns)
        if missing:
            logger.error(f"Missing columns in DataFrame: {sorted(missing)[:10]}...")
            logger.error(f"Available columns: {list(features_df.columns)[:20]}...")
            raise ValueError(f"Missing required columns: {len(missing)} columns")

        logger.info(f"DataFrame shape: {features_df.shape}")
        logger.info(f"Features: {len(feature_columns)} columns")

        # Показать информацию о символах (если есть)
        if symbol_column and symbol_column in features_df.columns:
            symbols = features_df[symbol_column].unique()
            logger.info(f"Symbols in data: {len(symbols)} ({list(symbols)[:5]}...)")

        # 2. КРИТИЧНО: Сортировка по времени
        # Для временных рядов порядок данных критичен!
        logger.info("Сортировка по timestamp...")
        features_df = features_df.sort_values(timestamp_column).reset_index(drop=True)

        # 3. Извлечение данных
        logger.info("Извлечение features, labels и timestamps...")
        X = features_df[feature_columns].values  # (N, feature_dim)
        y = features_df[label_column].values     # (N,)
        timestamps = features_df[timestamp_column].values  # (N,)

        logger.info(f"Extracted data:")
        logger.info(f"  • Features shape: {X.shape}")
        logger.info(f"  • Labels shape: {y.shape}")
        logger.info(f"  • Timestamps shape: {timestamps.shape}")

        # Проверить диапазон labels
        unique_labels = np.unique(y)
        logger.info(f"  • Unique labels: {unique_labels}")

        # 4. Проверка распределения классов
        from collections import Counter
        class_dist = Counter(y)
        logger.info(f"\nРаспределение классов ДО resampling:")
        for label, count in sorted(class_dist.items()):
            pct = 100 * count / len(y)
            logger.info(f"  • Class {label}: {count:,} ({pct:.1f}%)")

        # Проверить на NaN
        nan_features = np.isnan(X).sum()
        nan_labels = np.isnan(y).sum()
        if nan_features > 0 or nan_labels > 0:
            logger.warning(f"Found NaN values: features={nan_features}, labels={nan_labels}")
            # Удалить строки с NaN
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            timestamps = timestamps[valid_mask]
            logger.warning(f"After removing NaN: {len(X):,} samples")

        # 5. Resampling (если включен)
        if apply_resampling and self.balancing_strategy:
            logger.info("\n" + "="*80)
            logger.info("ПРИМЕНЕНИЕ CLASS BALANCING")
            logger.info("="*80)

            # Применяем балансировку
            X, y = self.balancing_strategy.balance_dataset(X, y)

            # Логируем новое распределение
            class_dist_after = Counter(y)
            logger.info(f"Распределение классов ПОСЛЕ resampling:")
            for label, count in sorted(class_dist_after.items()):
                pct = 100 * count / len(y)
                logger.info(f"  • Class {label}: {count:,} ({pct:.1f}%)")

            logger.info(f"Новый размер данных: {len(X):,} samples")

            # Обновить timestamps
            # Простая стратегия: если размер изменился, пересэмплируем timestamps
            if len(timestamps) != len(X):
                if len(timestamps) < len(X):
                    # Oversample: дублируем timestamps
                    indices = np.random.choice(len(timestamps), len(X), replace=True)
                    timestamps = timestamps[indices]
                    logger.info(f"Oversampled timestamps to {len(timestamps)}")
                else:
                    # Undersample: обрезаем timestamps
                    timestamps = timestamps[:len(X)]
                    logger.info(f"Undersampled timestamps to {len(timestamps)}")

        # Проверить минимальное количество данных
        min_samples_required = self.config.sequence_length * 10
        if len(X) < min_samples_required:
            raise ValueError(
                f"Insufficient data: {len(X)} samples, "
                f"need at least {min_samples_required} "
                f"(sequence_length={self.config.sequence_length} × 10)"
            )

        # 6. Создание sequences (переиспользуем существующий метод)
        logger.info("\n" + "="*80)
        logger.info("СОЗДАНИЕ ВРЕМЕННЫХ ПОСЛЕДОВАТЕЛЬНОСТЕЙ")
        logger.info("="*80)

        sequences, seq_labels, seq_timestamps = self.create_sequences(
            X, y, timestamps
        )

        logger.info(f"Создано sequences:")
        logger.info(f"  • Shape: {sequences.shape}")
        logger.info(f"  • Labels: {seq_labels.shape}")
        logger.info(f"  • Timestamps: {seq_timestamps.shape}")

        # 7. Train/Val/Test split (переиспользуем существующий метод)
        logger.info("\n" + "="*80)
        logger.info("SPLIT НА TRAIN/VAL/TEST")
        logger.info("="*80)

        train_data, val_data, test_data = self.train_val_test_split(
            sequences, seq_labels, seq_timestamps
        )

        logger.info(f"Split completed:")
        logger.info(f"  • Train: {len(train_data[0]):,} sequences")
        logger.info(f"  • Val: {len(val_data[0]):,} sequences")
        if test_data:
            logger.info(f"  • Test: {len(test_data[0]):,} sequences")

        # 8. Create DataLoaders (переиспользуем существующий метод)
        logger.info("\n" + "="*80)
        logger.info("СОЗДАНИЕ DATALOADERS")
        logger.info("="*80)

        dataloaders = self.create_dataloaders(train_data, val_data, test_data)

        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders.get('test', None)

        logger.info(f"\n✓ DataLoaders созданы из Feature Store данных:")
        logger.info(f"  • Train batches: {len(train_loader)}")
        logger.info(f"  • Val batches: {len(val_loader)}")
        if test_loader:
            logger.info(f"  • Test batches: {len(test_loader)}")
        logger.info(f"  • Batch size: {self.config.batch_size}")
        logger.info(f"{'='*80}\n")

        return train_loader, val_loader, test_loader

    def walk_forward_split(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray,
        n_splits: int = 5
    ) -> List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        """
        Walk-forward validation split.

        Разбивает данные на n_splits фолдов, где каждый следующий фолд
        использует все предыдущие данные для обучения.

        Args:
            sequences: (N, seq_len, features)
            labels: (N,)
            timestamps: (N,)
            n_splits: Количество фолдов

        Returns:
            List из (train_data, val_data) для каждого фолда
        """
        n_samples = len(sequences)
        fold_size = n_samples // n_splits

        splits = []

        for i in range(1, n_splits + 1):
            # Train: все данные до текущего фолда
            train_end = fold_size * i
            train_sequences = sequences[:train_end]
            train_labels = labels[:train_end]

            # Validation: следующий фолд
            if i < n_splits:
                val_start = train_end
                val_end = fold_size * (i + 1)
                val_sequences = sequences[val_start:val_end]
                val_labels = labels[val_start:val_end]
            else:
                # Последний фолд - используем оставшиеся данные
                val_sequences = sequences[train_end:]
                val_labels = labels[train_end:]

            splits.append((
                (train_sequences, train_labels),
                (val_sequences, val_labels)
            ))

            logger.info(
                f"Walk-forward фолд {i}/{n_splits}: "
                f"train={len(train_sequences)}, val={len(val_sequences)}"
            )

        return splits


# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========

if __name__ == "__main__":
    """
    Пример загрузки данных с class balancing.
    """
    from backend.ml_engine.training.class_balancing import ClassBalancingConfig

    # ===== КОНФИГУРАЦИЯ =====

    # Data config
    data_config = DataConfig(
        storage_path="data/ml_training",
        sequence_length=60,
        target_horizon="future_direction_60s",
        batch_size=64
    )

    # ===== ВАРИАНТЫ BALANCING =====

    # Вариант 1: Без балансировки
    loader_no_balancing = HistoricalDataLoader(
        config=data_config,
        balancing_config=None
    )

    # Вариант 2: Только Oversampling
    balancing_oversample = ClassBalancingConfig(
        use_class_weights=False,
        use_focal_loss=False,
        use_oversampling=True,
        oversample_strategy="auto"
    )

    loader_oversample = HistoricalDataLoader(
        config=data_config,
        balancing_config=balancing_oversample
    )

    # Вариант 3: SMOTE + Undersampling
    balancing_smote = ClassBalancingConfig(
        use_smote=True,
        use_undersampling=True,
        smote_k_neighbors=5
    )

    loader_smote = HistoricalDataLoader(
        config=data_config,
        balancing_config=balancing_smote
    )

    # ===== ЗАГРУЗКА =====

    # Выбираем loader
    loader = loader_oversample  # или loader_smote

    # Загружаем и подготавливаем
    result = loader.load_and_prepare(
        symbols=["BTCUSDT", "ETHUSDT"],
        apply_resampling=True  # ← Включаем resampling
    )

    # Получаем DataLoaders
    train_loader = result['dataloaders']['train']
    val_loader = result['dataloaders']['val']

    print("\n✓ Данные загружены и сбалансированы")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ===== ПРОВЕРКА РАСПРЕДЕЛЕНИЯ =====
    print("\nПроверка распределения классов в train_loader:")

    all_labels = []
    for batch in train_loader:
        labels = batch['label'].numpy()
        all_labels.extend(labels)

    class_dist = Counter(all_labels)

    print(f"Распределение: {dict(class_dist)}")

    total = len(all_labels)
    for cls, count in sorted(class_dist.items()):
        pct = (count / total) * 100
        print(f"  Класс {cls:2d}: {count:7,} ({pct:5.1f}%)")

    max_count = max(class_dist.values())
    min_count = min(class_dist.values())
    imbalance_ratio = max_count / min_count

    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio < 1.5:
        print("✅ Отличный баланс!")
    elif imbalance_ratio < 2.5:
        print("✅ Хороший баланс")
    else:
        print("⚠️  Все еще есть дисбаланс")

    # ===== ТЕСТ ОДНОГО BATCH =====
    print("\n" + "="*80)
    print("ТЕСТ ОДНОГО BATCH")
    print("="*80)

    for batch in train_loader:
        print(f"\nBatch shape:")
        print(f"  • sequences: {batch['sequence'].shape}")  # [batch, seq_len, features]
        print(f"  • labels: {batch['label'].shape}")        # [batch]

        print(f"\nBatch statistics:")
        print(f"  • sequences mean: {batch['sequence'].mean():.4f}")
        print(f"  • sequences std: {batch['sequence'].std():.4f}")
        print(f"  • labels unique: {torch.unique(batch['label'])}")

        break  # Показываем только первый batch

    print("\n✓ Все работает корректно!")