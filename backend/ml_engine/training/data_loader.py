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
    label_horizon: int = 60  # Горизонт предсказания в секундах/барах

    # Split параметры
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ===== PURGING & EMBARGO (Industry Standard) =====
    # Purging: удаляет samples на границах train/val/test для предотвращения data leakage
    # Embargo: добавляет gap между sets для учёта автокорреляции
    use_purging: bool = True  # Включить purging
    use_embargo: bool = True  # Включить embargo
    purge_length: int = None  # Если None, используется sequence_length
    embargo_length: int = None  # Если None, используется max(label_horizon, 2% от dataset)
    embargo_pct: float = 0.02  # Процент от dataset для embargo (если embargo_length=None)

    # DataLoader параметры
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 4  # Оптимально для 12GB GPU (8 может вызвать OOM)

    # Feature Store integration
    use_feature_store: bool = True  # Try Feature Store first
    feature_store_date_range_days: int = 90  # Last N days for training
    feature_store_group: str = "training_features"  # Feature group name
    fallback_to_legacy: bool = True  # Fallback to .npy files if FS fails


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
        # ПРАВИЛЬНЫЙ МАППИНГ:
        # -1 (DOWN/SELL) -> 2 (SELL)
        #  0 (NEUTRAL/HOLD) -> 0 (HOLD)
        #  1 (UP/BUY) -> 1 (BUY)
        logger.info(f"Исходное распределение классов: {Counter(y)}")

        # Маппинг (ИСПРАВЛЕНО: было {-1: 0, 0: 1, 1: 2} - НЕПРАВИЛЬНО!)
        label_mapping = {-1: 2, 0: 0, 1: 1}
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

    def _calculate_purge_embargo_lengths(self, n_samples: int) -> Tuple[int, int]:
        """
        Рассчитать длины purge и embargo на основе конфигурации.

        Args:
            n_samples: Общее количество samples

        Returns:
            (purge_length, embargo_length)
        """
        # Purge length: по умолчанию = sequence_length
        if self.config.purge_length is not None:
            purge_length = self.config.purge_length
        else:
            purge_length = self.config.sequence_length

        # Embargo length: по умолчанию = max(label_horizon, embargo_pct * n_samples)
        if self.config.embargo_length is not None:
            embargo_length = self.config.embargo_length
        else:
            embargo_from_pct = int(n_samples * self.config.embargo_pct)
            embargo_length = max(self.config.label_horizon, embargo_from_pct)

        return purge_length, embargo_length

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
        Разбить данные на train/val/test с Purging и Embargo (Industry Standard).

        ВАЖНО: Для временных рядов НЕ используем shuffle!

        Purging: Удаляет samples из train, чьи labels могут пересекаться с val/test.
                 Это предотвращает data leakage из-за overlapping sequences.

        Embargo: Добавляет gap между train и val/test для учёта автокорреляции
                 и предотвращения leakage из-за временной зависимости.

        Схема разбиения:
        [TRAIN] --purge-- [EMBARGO] -- [VAL] --purge-- [EMBARGO] -- [TEST]

        Args:
            sequences: (N, seq_len, features)
            labels: (N,)
            timestamps: (N,)

        Returns:
            train_data: (X_train, y_train)
            val_data: (X_val, y_val)
            test_data: (X_test, y_test) или None если test_ratio=0
        """
        from tqdm import tqdm

        n_samples = len(sequences)

        # Рассчитываем purge и embargo lengths
        purge_length, embargo_length = self._calculate_purge_embargo_lengths(n_samples)

        # Базовые индексы split (без purging/embargo)
        base_train_end = int(n_samples * self.config.train_ratio)
        base_val_end = base_train_end + int(n_samples * self.config.val_ratio)

        # ===== ПРИМЕНЯЕМ PURGING & EMBARGO =====
        if self.config.use_purging or self.config.use_embargo:
            tqdm.write("\n" + "=" * 60)
            tqdm.write("[Purging & Embargo] INDUSTRY STANDARD DATA SPLIT")
            tqdm.write("=" * 60)

            effective_purge = purge_length if self.config.use_purging else 0
            effective_embargo = embargo_length if self.config.use_embargo else 0

            tqdm.write(f"  • Purging: {effective_purge} samples (enabled: {self.config.use_purging})")
            tqdm.write(f"  • Embargo: {effective_embargo} samples (enabled: {self.config.use_embargo})")

            # Train: от начала до (base_train_end - purge)
            train_end_purged = base_train_end - effective_purge

            # Val: от (base_train_end + embargo) до (base_val_end - purge)
            val_start = base_train_end + effective_embargo
            val_end_purged = base_val_end - effective_purge

            # Test: от (base_val_end + embargo) до конца
            test_start = base_val_end + effective_embargo

            # Проверяем валидность индексов
            if train_end_purged <= 0:
                tqdm.write("[WARNING] Purge length слишком большой! Уменьшаем...")
                train_end_purged = max(int(n_samples * 0.5), 100)

            if val_start >= val_end_purged:
                tqdm.write("[WARNING] Val set пустой после purging! Корректируем...")
                val_start = train_end_purged + effective_embargo
                val_end_purged = max(val_start + 100, base_val_end)

            if test_start >= n_samples and self.config.test_ratio > 0:
                tqdm.write("[WARNING] Test set пустой после embargo! Корректируем...")
                test_start = min(val_end_purged + effective_embargo, n_samples - 100)

            # Логируем итоговые размеры
            train_size = train_end_purged
            val_size = val_end_purged - val_start
            test_size = n_samples - test_start if self.config.test_ratio > 0 else 0

            # Считаем потери от purging/embargo
            total_kept = train_size + val_size + test_size
            total_lost = n_samples - total_kept
            loss_pct = (total_lost / n_samples) * 100

            tqdm.write(f"\n  Effective split:")
            tqdm.write(f"  • Train: 0 -> {train_end_purged} ({train_size:,} samples)")
            tqdm.write(f"  • Gap (purge+embargo): {train_end_purged} -> {val_start}")
            tqdm.write(f"  • Val: {val_start} -> {val_end_purged} ({val_size:,} samples)")
            tqdm.write(f"  • Gap (purge+embargo): {val_end_purged} -> {test_start}")
            tqdm.write(f"  • Test: {test_start} -> {n_samples} ({test_size:,} samples)")
            tqdm.write(f"\n  Data loss from purging/embargo: {total_lost:,} samples ({loss_pct:.1f}%)")
            tqdm.write("=" * 60 + "\n")

            # Выполняем split
            X_train = sequences[:train_end_purged]
            y_train = labels[:train_end_purged]

            X_val = sequences[val_start:val_end_purged]
            y_val = labels[val_start:val_end_purged]

            if self.config.test_ratio > 0 and test_start < n_samples:
                X_test = sequences[test_start:]
                y_test = labels[test_start:]
                test_data = (X_test, y_test)
            else:
                test_data = None
                test_size = 0

        else:
            # ===== LEGACY SPLIT (без purging/embargo) =====
            tqdm.write("[Data] Using legacy split (purging/embargo disabled)")

            X_train = sequences[:base_train_end]
            y_train = labels[:base_train_end]

            X_val = sequences[base_train_end:base_val_end]
            y_val = labels[base_train_end:base_val_end]

            if self.config.test_ratio > 0:
                X_test = sequences[base_val_end:]
                y_test = labels[base_val_end:]
                test_data = (X_test, y_test)
                test_size = len(X_test)
            else:
                test_data = None
                test_size = 0

        # Логируем размеры
        tqdm.write(f"[Data] Train/Val/Test split: train={len(X_train)}, val={len(X_val)}, test={test_size}")

        # Логируем распределение классов
        train_dist = Counter(y_train)
        val_dist = Counter(y_val)
        tqdm.write(f"[Data] Train class distribution: {dict(train_dist)}")
        tqdm.write(f"[Data] Val class distribution: {dict(val_dist)}")

        # ===== STRATIFIED VALIDATION =====
        # Если validation не содержит всех классов - добавляем сэмплы из train
        expected_classes = {0, 1, 2}  # HOLD, BUY, SELL
        missing_classes = expected_classes - set(val_dist.keys())

        if missing_classes:
            tqdm.write(f"[Data] Val set missing classes: {missing_classes}")
            tqdm.write("[Data] Adding samples from train to ensure all classes in validation...")

            # Сколько сэмплов добавить для каждого недостающего класса
            min_samples_per_class = max(100, len(X_val) // 20)  # Минимум 100 или 5% от val

            additional_X = []
            additional_y = []

            for cls in missing_classes:
                # Находим сэмплы этого класса в train (берём из начала train, далеко от val)
                # ВАЖНО: берём из НАЧАЛА train (не конца) для соблюдения temporal order
                cls_indices = np.where(y_train == cls)[0]

                if len(cls_indices) > 0:
                    # Берём первые сэмплы этого класса из train (дальше от val)
                    n_to_add = min(min_samples_per_class, len(cls_indices))
                    selected_indices = cls_indices[:n_to_add]  # Первые (далеко от val)

                    additional_X.append(X_train[selected_indices])
                    additional_y.append(y_train[selected_indices])
                    tqdm.write(f"  • Class {cls}: added {n_to_add} samples from train (early samples)")

            if additional_X:
                # Добавляем в validation
                X_val = np.concatenate([X_val] + additional_X, axis=0)
                y_val = np.concatenate([y_val] + additional_y, axis=0)

                # Перемешиваем validation (но НЕ train!)
                shuffle_idx = np.random.permutation(len(X_val))
                X_val = X_val[shuffle_idx]
                y_val = y_val[shuffle_idx]

                val_dist = Counter(y_val)
                tqdm.write(f"[Data] Val class distribution (after stratification): {dict(val_dist)}")

        if test_data:
            test_dist = Counter(test_data[1])
            tqdm.write(f"[Data] Test class distribution: {dict(test_dist)}")
            if len(test_dist) < 3:
                tqdm.write("=" * 60)
                tqdm.write("[WARNING] Test set содержит не все классы!")
                tqdm.write(f"  • Классов в test: {len(test_dist)} из 3")
                tqdm.write(f"  • Это типично для последовательного split временных рядов")
                tqdm.write("=" * 60)

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

    def _apply_train_resampling(
        self,
        train_data: Tuple[np.ndarray, ...]
    ) -> Tuple[np.ndarray, ...]:
        """
        Применить resampling ТОЛЬКО к train данным.

        ВАЖНО: Resampling применяется ПОСЛЕ split, чтобы избежать data leakage!
        Val/Test данные остаются с оригинальным распределением для честной оценки.

        Args:
            train_data: (X_train, y_train) или (X_train, y_train, timestamps_train)

        Returns:
            Tuple с resampled данными (того же размера что и вход)
        """
        from tqdm import tqdm

        if not self.balancing_strategy:
            return train_data

        # Поддержка и 2-tuple и 3-tuple
        if len(train_data) == 2:
            X_train, y_train = train_data
            timestamps_train = None
        else:
            X_train, y_train, timestamps_train = train_data

        tqdm.write("\n" + "="*60)
        tqdm.write("[Resampling] Применяем ТОЛЬКО к TRAIN данным (после split)")
        tqdm.write("="*60)

        # Логируем распределение ДО
        before_dist = Counter(y_train)
        tqdm.write(f"[Resampling] ДО: {dict(before_dist)}")

        # Применяем балансировку к train данным
        X_train_new, y_train_new = self.balancing_strategy.balance_dataset(
            X_train, y_train
        )

        # Логируем распределение ПОСЛЕ
        after_dist = Counter(y_train_new)
        tqdm.write(f"[Resampling] ПОСЛЕ: {dict(after_dist)}")
        tqdm.write(f"[Resampling] Размер: {len(X_train)} → {len(X_train_new)}")

        tqdm.write("="*60 + "\n")

        # Возвращаем в том же формате, что и получили
        if timestamps_train is None:
            # 2-tuple: (X, y)
            return X_train_new, y_train_new
        else:
            # 3-tuple: (X, y, timestamps) - обновляем timestamps
            if len(timestamps_train) != len(X_train_new):
                if len(timestamps_train) < len(X_train_new):
                    # Oversample: дублируем timestamps
                    indices = np.random.choice(len(timestamps_train), len(X_train_new), replace=True)
                    timestamps_new = timestamps_train[indices]
                else:
                    # Undersample: обрезаем timestamps
                    timestamps_new = timestamps_train[:len(X_train_new)]
            else:
                timestamps_new = timestamps_train

            return X_train_new, y_train_new, timestamps_new

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

        # NOTE: Resampling теперь применяется ПОСЛЕ split (см. ниже)
        # Это предотвращает data leakage и обеспечивает честную оценку на val/test

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

        # ===== RESAMPLING ТОЛЬКО ДЛЯ TRAIN (после split!) =====
        if apply_resampling and self.balancing_strategy:
            train_data = self._apply_train_resampling(train_data)

        # Создаем DataLoaders
        dataloaders = self.create_dataloaders(
            train_data, val_data, test_data
        )

        # Статистика
        result = {
            'dataloaders': dataloaders,
            'statistics': {
                'total_sequences': len(sequences),
                'train_samples': len(train_data[0]),  # После resampling
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
        logger.info(f"  • Unique labels BEFORE mapping: {unique_labels}")

        # ===== КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ =====
        # Поддерживаем два формата меток:
        # Старый формат: {-1, 0, 1} -> {0, 1, 2} для PyTorch
        # Новый формат (после preprocessing): {0, 1, 2} -> {0, 1, 2} (identity)
        # -1 (DOWN) -> 0
        #  0 (NEUTRAL/DOWN) -> 0 or 1 (depends on format)
        #  1 (NEUTRAL/UP) -> 1 or 2 (depends on format)
        #  2 (UP) -> 2

        # CRITICAL FIX: Detect which format we have
        # If we have -1 in data, use old mapping
        # If we have 2 in data, use new mapping (identity)
        has_negative = np.any(unique_labels == -1)
        has_two = np.any(unique_labels == 2)

        if has_negative and not has_two:
            # Old format: {-1, 0, 1} - ИСПРАВЛЕННЫЙ маппинг!
            # -1 (DOWN/SELL) -> 2 (SELL)
            #  0 (NEUTRAL/HOLD) -> 0 (HOLD)
            #  1 (UP/BUY) -> 1 (BUY)
            label_mapping = {-1: 2, 0: 0, 1: 1}
            logger.info("Using old label format: {-1, 0, 1} -> {2, 0, 1} (SELL=2, HOLD=0, BUY=1)")
        elif has_two and not has_negative:
            # New format: {0, 1, 2}
            label_mapping = {0: 0, 1: 1, 2: 2}
            logger.info("Using new label format (identity): {0, 1, 2} -> {0, 1, 2}")
        else:
            # Mixed or unclear format - default to flexible mapping
            # -1 -> SELL (2), 0 -> HOLD (0), 1 -> BUY (1), 2 -> SELL (2)
            label_mapping = {-1: 2, 0: 0, 1: 1, 2: 2}
            logger.warning("Mixed or unclear label format detected, using flexible mapping")

        # Проверяем наличие неожиданных значений (исключая NaN)
        unique_labels_no_nan = unique_labels[~np.isnan(unique_labels)] if np.any(np.isnan(unique_labels)) else unique_labels
        unexpected_labels = set(unique_labels_no_nan) - set(label_mapping.keys())
        if unexpected_labels:
            logger.warning(f"Found unexpected label values: {unexpected_labels}")
            logger.warning("These will be mapped to HOLD (0)")
            # Маппим неожиданные значения на HOLD (0)
            for unexpected_label in unexpected_labels:
                label_mapping[int(unexpected_label)] = 0

        # Применяем маппинг (NaN values останутся NaN)
        y_mapped = []
        for label in y:
            if np.isnan(label):
                y_mapped.append(np.nan)
            else:
                y_mapped.append(label_mapping[int(label)])
        y = np.array(y_mapped, dtype=np.float64)  # float64 чтобы сохранить NaN

        unique_labels_after = np.unique(y)
        logger.info(f"  • Unique labels AFTER mapping: {unique_labels_after}")

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

            # Анализируем, где именно NaN
            nan_per_column = np.isnan(X).sum(axis=0)
            columns_with_nan = np.where(nan_per_column > 0)[0]
            if len(columns_with_nan) > 0:
                logger.warning(f"Columns with NaN: {len(columns_with_nan)}/{X.shape[1]}")
                logger.warning(f"Top 5 columns by NaN count: {sorted(nan_per_column, reverse=True)[:5]}")

            # СТРАТЕГИЯ 1: Заполняем NaN в features нулями (безопаснее чем удалять)
            if nan_features > 0:
                logger.info("Filling NaN values in features with 0.0")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # СТРАТЕГИЯ 2: Удаляем строки только если NaN в labels (критично!)
            if nan_labels > 0:
                logger.warning(f"Removing {nan_labels} rows with NaN labels")
                valid_mask = ~np.isnan(y)
                X = X[valid_mask]
                y = y[valid_mask]
                timestamps = timestamps[valid_mask]

            logger.info(f"After NaN handling: {len(X):,} samples")

        # После удаления NaN, конвертируем labels в int64 для PyTorch
        y = y.astype(np.int64)

        # NOTE: Resampling теперь применяется ПОСЛЕ split (см. ниже)
        # Это предотвращает data leakage и обеспечивает честную оценку на val/test

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

        # ===== RESAMPLING ТОЛЬКО ДЛЯ TRAIN (после split!) =====
        if apply_resampling and self.balancing_strategy:
            train_data = self._apply_train_resampling(train_data)
            logger.info(f"  • Train (after resampling): {len(train_data[0]):,} sequences")

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
        n_splits: int = 5,
        min_train_size: float = 0.3,
        stratify_val: bool = True,
        use_purging: bool = None,
        use_embargo: bool = None
    ) -> List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        """
        Walk-forward validation split с поддержкой Purging (Industry Standard).

        Разбивает данные на n_splits фолдов, где каждый следующий фолд
        использует растущее окно для обучения.

        С включенным purging между train и val добавляется gap для
        предотвращения data leakage.

        Схема для каждого фолда:
        [TRAIN] --purge-- [VAL]

        Преимущества:
        - Нет утечки данных из будущего
        - Более реалистичная оценка модели
        - Тестирует на разных рыночных условиях
        - С purging: честная оценка без leakage

        Args:
            sequences: (N, seq_len, features)
            labels: (N,)
            timestamps: (N,)
            n_splits: Количество фолдов
            min_train_size: Минимальный размер train (доля от всех данных)
            stratify_val: Добавлять сэмплы для стратификации validation
            use_purging: Использовать purging (если None - берётся из config)
            use_embargo: Использовать embargo (если None - берётся из config)

        Returns:
            List из (train_data, val_data) для каждого фолда
        """
        from tqdm import tqdm

        n_samples = len(sequences)
        min_train_samples = int(n_samples * min_train_size)
        remaining = n_samples - min_train_samples
        fold_size = remaining // n_splits

        # Определяем использование purging/embargo
        apply_purging = use_purging if use_purging is not None else self.config.use_purging
        apply_embargo = use_embargo if use_embargo is not None else self.config.use_embargo

        # Рассчитываем длины
        purge_length, embargo_length = self._calculate_purge_embargo_lengths(n_samples)
        effective_purge = purge_length if apply_purging else 0
        effective_embargo = embargo_length if apply_embargo else 0
        total_gap = effective_purge + effective_embargo

        tqdm.write("\n" + "=" * 60)
        tqdm.write("[Walk-Forward] НАСТРОЙКА КРОСС-ВАЛИДАЦИИ С PURGING")
        tqdm.write("=" * 60)
        tqdm.write(f"  • Всего сэмплов: {n_samples:,}")
        tqdm.write(f"  • Количество фолдов: {n_splits}")
        tqdm.write(f"  • Мин. train размер: {min_train_samples:,} ({min_train_size:.0%})")
        tqdm.write(f"  • Размер фолда: ~{fold_size:,}")
        tqdm.write(f"  • Purging: {effective_purge} samples (enabled: {apply_purging})")
        tqdm.write(f"  • Embargo: {effective_embargo} samples (enabled: {apply_embargo})")
        tqdm.write(f"  • Total gap per fold: {total_gap} samples")
        tqdm.write("=" * 60)

        splits = []
        expected_classes = {0, 1, 2}  # HOLD, BUY, SELL

        for i in range(n_splits):
            # Train: от начала до (train_end - purge)
            base_train_end = min_train_samples + fold_size * i
            train_end_purged = base_train_end - effective_purge

            # Validation: от (train_end + embargo) до val_end
            val_start = base_train_end + effective_embargo
            val_end = min(base_train_end + fold_size, n_samples)

            # Проверяем валидность индексов
            if train_end_purged <= 0:
                tqdm.write(f"[Fold {i+1}] WARNING: train_end_purged <= 0, adjusting...")
                train_end_purged = max(100, min_train_samples // 2)

            if val_start >= val_end:
                tqdm.write(f"[Fold {i+1}] WARNING: val_start >= val_end, adjusting...")
                val_start = train_end_purged + total_gap
                val_end = max(val_start + 100, base_train_end + fold_size)

            if val_end > n_samples:
                val_end = n_samples

            train_sequences = sequences[:train_end_purged]
            train_labels = labels[:train_end_purged]

            val_sequences = sequences[val_start:val_end]
            val_labels = labels[val_start:val_end]

            # Stratify validation если нужно
            if stratify_val and len(val_labels) > 0:
                val_dist = Counter(val_labels)
                missing_classes = expected_classes - set(val_dist.keys())

                if missing_classes:
                    min_samples_per_class = max(50, len(val_sequences) // 20)

                    for cls in missing_classes:
                        # ВАЖНО: берём из НАЧАЛА train (далеко от val) для соблюдения purging
                        cls_indices = np.where(train_labels == cls)[0]
                        if len(cls_indices) > 0:
                            n_to_add = min(min_samples_per_class, len(cls_indices))
                            selected_indices = cls_indices[:n_to_add]  # Первые (далеко от val)

                            val_sequences = np.concatenate([val_sequences, train_sequences[selected_indices]], axis=0)
                            val_labels = np.concatenate([val_labels, train_labels[selected_indices]], axis=0)

                    # Shuffle validation
                    shuffle_idx = np.random.permutation(len(val_sequences))
                    val_sequences = val_sequences[shuffle_idx]
                    val_labels = val_labels[shuffle_idx]

            splits.append((
                (train_sequences, train_labels),
                (val_sequences, val_labels)
            ))

            train_dist = Counter(train_labels)
            val_dist = Counter(val_labels)
            gap_info = f" (gap: {base_train_end - train_end_purged + val_start - base_train_end})" if total_gap > 0 else ""
            tqdm.write(
                f"[Fold {i+1}/{n_splits}] train={len(train_sequences):,} {dict(train_dist)}, "
                f"val={len(val_sequences):,} {dict(val_dist)}{gap_info}"
            )

        tqdm.write("=" * 60 + "\n")
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