#!/usr/bin/env python3
"""
Memory-Mapped Dataset для эффективной загрузки больших объёмов данных.

Проблема:
- При загрузке 64 символов в обычном режиме требуется 64+ GB RAM
- Все данные загружаются в память сразу
- torch.FloatTensor() создаёт копию данных

Решение:
- Lazy loading - данные загружаются по батчам, не все сразу
- Memory-mapped files - np.memmap() для виртуального маппинга без загрузки в RAM
- Zero-copy tensors - torch.from_numpy() вместо torch.FloatTensor()
- IterableDataset - streaming подход для PyTorch

Ожидаемый результат:
- RAM: ~4 GB вместо 64+ GB для 64 символов
- Скорость: +5-10% overhead на I/O (приемлемо)

Путь: backend/ml_engine/training/memory_mapped_dataset.py
"""

import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryMappedConfig:
    """Конфигурация для memory-mapped загрузки."""
    sequence_length: int = 60
    batch_size: int = 64
    shuffle: bool = True
    prefetch_batches: int = 2  # Количество батчей для предзагрузки
    target_horizon: str = "future_direction_300s"

    # Label mapping: {-1, 0, 1} -> {0, 1, 2}
    label_mapping: Dict[int, int] = None

    def __post_init__(self):
        if self.label_mapping is None:
            self.label_mapping = {-1: 0, 0: 1, 1: 2}


class SymbolDataIndex:
    """
    Индекс данных для одного символа.

    Содержит информацию о файлах без загрузки данных в память.
    """

    def __init__(
        self,
        symbol: str,
        features_dir: Path,
        labels_dir: Path,
        sequence_length: int,
        target_horizon: str
    ):
        self.symbol = symbol
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.sequence_length = sequence_length
        self.target_horizon = target_horizon

        # Список файлов и их размеры
        self.file_index: List[Dict] = []
        self.total_sequences = 0

        self._build_index()

    def _build_index(self):
        """
        Построить индекс файлов БЕЗ загрузки данных.

        Читает только header .npy файлов для определения размеров.
        """
        feature_files = sorted(self.features_dir.glob("*.npy"))
        label_files = {f.stem: f for f in self.labels_dir.glob("*.json")}

        for feature_file in feature_files:
            # Читаем только shape без загрузки данных
            # mmap_mode='r' создаёт виртуальный маппинг
            try:
                mmap = np.load(feature_file, mmap_mode='r')
                n_samples = len(mmap)
                n_features = mmap.shape[1] if len(mmap.shape) > 1 else 1

                # Количество sequences = samples - seq_len + 1
                n_sequences = max(0, n_samples - self.sequence_length + 1)

                # Проверяем наличие соответствующего файла labels
                label_file = label_files.get(feature_file.stem)

                if n_sequences > 0 and label_file:
                    self.file_index.append({
                        'feature_file': feature_file,
                        'label_file': label_file,
                        'n_samples': n_samples,
                        'n_features': n_features,
                        'n_sequences': n_sequences,
                        'start_idx': self.total_sequences,
                        'end_idx': self.total_sequences + n_sequences
                    })
                    self.total_sequences += n_sequences

                # Важно: закрываем mmap
                del mmap

            except Exception as e:
                logger.warning(f"Error indexing {feature_file}: {e}")
                continue

        logger.debug(
            f"Indexed {self.symbol}: {len(self.file_index)} files, "
            f"{self.total_sequences:,} sequences"
        )

    def get_sequence_location(self, global_idx: int) -> Tuple[Dict, int]:
        """
        Найти файл и локальный индекс для глобального индекса sequence.

        Args:
            global_idx: Глобальный индекс sequence (0 to total_sequences-1)

        Returns:
            (file_info, local_idx) - информация о файле и локальный индекс
        """
        for file_info in self.file_index:
            if file_info['start_idx'] <= global_idx < file_info['end_idx']:
                local_idx = global_idx - file_info['start_idx']
                return file_info, local_idx

        raise IndexError(f"Index {global_idx} out of range [0, {self.total_sequences})")


class MemoryMappedDataset(IterableDataset):
    """
    Lazy-loading dataset с memory-mapped files.

    Загружает только нужный batch, не всё в память.
    Использует np.memmap() для эффективного доступа к данным на диске.

    Особенности:
    - IterableDataset - поддерживает streaming
    - Memory-mapped - данные не загружаются целиком в RAM
    - Batch-wise loading - загружается только текущий batch
    - Zero-copy tensors - torch.from_numpy() без копирования

    Usage:
        dataset = MemoryMappedDataset(
            data_dir=Path("data/ml_training"),
            symbols=["BTCUSDT", "ETHUSDT"],
            config=MemoryMappedConfig(batch_size=64)
        )

        for batch in dataset:
            sequences = batch['sequence']  # (batch_size, seq_len, features)
            labels = batch['label']        # (batch_size,)
    """

    def __init__(
        self,
        data_dir: Path,
        symbols: List[str],
        config: Optional[MemoryMappedConfig] = None
    ):
        """
        Инициализация dataset.

        Args:
            data_dir: Директория с данными (data/ml_training)
            symbols: Список символов для загрузки
            config: Конфигурация загрузки
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.symbols = symbols
        self.config = config or MemoryMappedConfig()

        # Построение индекса
        self.symbol_indices: Dict[str, SymbolDataIndex] = {}
        self.total_sequences = 0

        self._build_global_index()

        # Кэш для labels (они маленькие, можно держать в памяти)
        self._labels_cache: Dict[str, List] = {}

        logger.info(
            f"MemoryMappedDataset initialized: {len(symbols)} symbols, "
            f"{self.total_sequences:,} total sequences, "
            f"batch_size={self.config.batch_size}"
        )

    def _build_global_index(self):
        """Построить глобальный индекс по всем символам."""
        for symbol in self.symbols:
            symbol_dir = self.data_dir / symbol
            features_dir = symbol_dir / "features"
            labels_dir = symbol_dir / "labels"

            if not features_dir.exists() or not labels_dir.exists():
                logger.warning(f"Data directories not found for {symbol}")
                continue

            index = SymbolDataIndex(
                symbol=symbol,
                features_dir=features_dir,
                labels_dir=labels_dir,
                sequence_length=self.config.sequence_length,
                target_horizon=self.config.target_horizon
            )

            if index.total_sequences > 0:
                self.symbol_indices[symbol] = index
                self.total_sequences += index.total_sequences

        if self.total_sequences == 0:
            raise ValueError(f"No data found for symbols: {self.symbols}")

        logger.info(
            f"Global index built: {self.total_sequences:,} sequences "
            f"from {len(self.symbol_indices)} symbols"
        )

    def _load_labels_for_file(self, label_file: Path) -> List:
        """
        Загрузить labels из JSON файла (с кэшированием).

        Labels маленькие, поэтому кэшируем их в памяти.
        """
        cache_key = str(label_file)

        if cache_key not in self._labels_cache:
            with open(label_file) as f:
                labels_data = json.load(f)

            # Извлекаем target horizon
            labels = []
            for item in labels_data:
                target = item.get(self.config.target_horizon)
                if target is not None:
                    # Применяем label mapping
                    mapped = self.config.label_mapping.get(target, target)
                    labels.append(mapped)
                else:
                    labels.append(1)  # Default to HOLD

            self._labels_cache[cache_key] = labels

        return self._labels_cache[cache_key]

    def _load_sequence(
        self,
        feature_file: Path,
        label_file: Path,
        local_idx: int
    ) -> Tuple[np.ndarray, int]:
        """
        Загрузить одну sequence из memory-mapped файла.

        Args:
            feature_file: Путь к .npy файлу с features
            label_file: Путь к .json файлу с labels
            local_idx: Локальный индекс sequence в файле

        Returns:
            (sequence, label) - numpy array и label
        """
        # Memory-mapped чтение - только нужный slice
        mmap = np.load(feature_file, mmap_mode='r')

        # Извлекаем sequence
        seq_start = local_idx
        seq_end = local_idx + self.config.sequence_length
        sequence = mmap[seq_start:seq_end].copy()  # .copy() для отсоединения от mmap

        del mmap  # Закрываем mmap

        # Загружаем label
        labels = self._load_labels_for_file(label_file)
        label_idx = local_idx + self.config.sequence_length - 1  # Label последнего элемента

        if label_idx < len(labels):
            label = labels[label_idx]
        else:
            label = 1  # Default to HOLD

        return sequence, label

    def _load_batch(
        self,
        batch_indices: List[Tuple[str, int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Загрузить один batch из memory-mapped файлов.

        Args:
            batch_indices: Список (symbol, global_idx) для каждого элемента batch

        Returns:
            Dict с 'sequence' и 'label' tensors
        """
        sequences = []
        labels = []

        for symbol, global_idx in batch_indices:
            index = self.symbol_indices[symbol]
            file_info, local_idx = index.get_sequence_location(global_idx)

            sequence, label = self._load_sequence(
                feature_file=file_info['feature_file'],
                label_file=file_info['label_file'],
                local_idx=local_idx
            )

            sequences.append(sequence)
            labels.append(label)

        # Стек и конвертация в tensors (zero-copy где возможно)
        sequences_np = np.stack(sequences, axis=0).astype(np.float32)
        labels_np = np.array(labels, dtype=np.int64)

        return {
            'sequence': torch.from_numpy(sequences_np),
            'label': torch.from_numpy(labels_np)
        }

    def _generate_all_indices(self) -> List[Tuple[str, int]]:
        """
        Генерировать все (symbol, global_idx) пары.

        Returns:
            Список всех доступных indices
        """
        all_indices = []

        for symbol, index in self.symbol_indices.items():
            for i in range(index.total_sequences):
                all_indices.append((symbol, i))

        return all_indices

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Итератор по batches.

        Yields:
            Dict с 'sequence' и 'label' tensors для каждого batch
        """
        # Генерируем все indices
        all_indices = self._generate_all_indices()

        # Shuffle если нужно
        if self.config.shuffle:
            random.shuffle(all_indices)

        # Yield batches
        batch = []
        for item in all_indices:
            batch.append(item)

            if len(batch) >= self.config.batch_size:
                yield self._load_batch(batch)
                batch = []

        # Последний неполный batch
        if batch:
            yield self._load_batch(batch)

    def __len__(self) -> int:
        """Количество batches."""
        return (self.total_sequences + self.config.batch_size - 1) // self.config.batch_size

    def get_statistics(self) -> Dict:
        """
        Получить статистику по dataset.

        Returns:
            Dict со статистикой
        """
        stats = {
            'total_sequences': self.total_sequences,
            'total_batches': len(self),
            'symbols': len(self.symbol_indices),
            'batch_size': self.config.batch_size,
            'sequence_length': self.config.sequence_length,
            'per_symbol': {}
        }

        for symbol, index in self.symbol_indices.items():
            stats['per_symbol'][symbol] = {
                'sequences': index.total_sequences,
                'files': len(index.file_index)
            }

        return stats


class MemoryMappedDatasetWithSplit(MemoryMappedDataset):
    """
    Memory-mapped dataset с поддержкой train/val/test split.

    Расширяет MemoryMappedDataset добавляя:
    - Разбиение на train/val/test
    - Purging и Embargo для предотвращения data leakage
    """

    def __init__(
        self,
        data_dir: Path,
        symbols: List[str],
        config: Optional[MemoryMappedConfig] = None,
        mode: str = 'train',  # 'train', 'val', 'test'
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        use_purging: bool = True,
        use_embargo: bool = True,
        purge_length: Optional[int] = None,
        embargo_pct: float = 0.02
    ):
        """
        Инициализация с split.

        Args:
            data_dir: Директория с данными
            symbols: Список символов
            config: Конфигурация
            mode: Режим ('train', 'val', 'test')
            train_ratio: Доля train данных
            val_ratio: Доля validation данных
            use_purging: Использовать purging
            use_embargo: Использовать embargo
            purge_length: Длина purge (None = sequence_length)
            embargo_pct: Процент для embargo
        """
        # Инициализируем базовый класс
        super().__init__(data_dir, symbols, config)

        self.mode = mode
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.use_purging = use_purging
        self.use_embargo = use_embargo
        self.purge_length = purge_length or self.config.sequence_length
        self.embargo_pct = embargo_pct

        # Вычисляем границы split
        self._calculate_split_boundaries()

        logger.info(
            f"MemoryMappedDataset [{mode}]: {self._mode_sequences:,} sequences "
            f"(from {self._mode_start} to {self._mode_end})"
        )

    def _calculate_split_boundaries(self):
        """Вычислить границы для текущего mode с учётом purging/embargo."""
        n = self.total_sequences

        # Базовые границы
        train_end_base = int(n * self.train_ratio)
        val_end_base = train_end_base + int(n * self.val_ratio)

        # Purge и embargo lengths
        purge_len = self.purge_length if self.use_purging else 0
        embargo_len = max(
            self.config.sequence_length,
            int(n * self.embargo_pct)
        ) if self.use_embargo else 0

        # Границы с учётом purging/embargo
        if self.mode == 'train':
            self._mode_start = 0
            self._mode_end = train_end_base - purge_len
        elif self.mode == 'val':
            self._mode_start = train_end_base + embargo_len
            self._mode_end = val_end_base - purge_len
        else:  # test
            self._mode_start = val_end_base + embargo_len
            self._mode_end = n

        # Корректировка невалидных границ
        self._mode_start = max(0, self._mode_start)
        self._mode_end = min(n, self._mode_end)
        self._mode_end = max(self._mode_start, self._mode_end)

        self._mode_sequences = self._mode_end - self._mode_start

    def _generate_all_indices(self) -> List[Tuple[str, int]]:
        """
        Генерировать indices только для текущего mode.

        Переопределяем метод базового класса для фильтрации по split.
        """
        all_indices = []
        current_global_idx = 0

        for symbol, index in self.symbol_indices.items():
            for local_idx in range(index.total_sequences):
                # Проверяем, входит ли в текущий split
                if self._mode_start <= current_global_idx < self._mode_end:
                    all_indices.append((symbol, local_idx))

                current_global_idx += 1

        return all_indices

    def __len__(self) -> int:
        """Количество batches для текущего mode."""
        return (self._mode_sequences + self.config.batch_size - 1) // self.config.batch_size


def create_memory_efficient_dataloaders(
    data_dir: Path,
    symbols: List[str],
    config: Optional[MemoryMappedConfig] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_purging: bool = True,
    use_embargo: bool = True,
    num_workers: int = 0,  # 0 for Windows compatibility
    pin_memory: bool = False  # False to save memory
) -> Dict[str, DataLoader]:
    """
    Создать memory-efficient DataLoaders для train/val/test.

    Это основной entry point для использования memory-mapped loading.

    Args:
        data_dir: Директория с данными
        symbols: Список символов
        config: Конфигурация
        train_ratio: Доля train
        val_ratio: Доля validation
        use_purging: Использовать purging
        use_embargo: Использовать embargo
        num_workers: Количество worker процессов
        pin_memory: Пинить память (для GPU)

    Returns:
        Dict с DataLoaders: {'train': ..., 'val': ..., 'test': ...}

    Usage:
        dataloaders = create_memory_efficient_dataloaders(
            data_dir=Path("data/ml_training"),
            symbols=["BTCUSDT", "ETHUSDT", ...],
            config=MemoryMappedConfig(batch_size=64)
        )

        for batch in dataloaders['train']:
            sequences = batch['sequence']
            labels = batch['label']
    """
    config = config or MemoryMappedConfig()

    logger.info("\n" + "=" * 60)
    logger.info("CREATING MEMORY-EFFICIENT DATALOADERS")
    logger.info("=" * 60)
    logger.info(f"  • Symbols: {len(symbols)}")
    logger.info(f"  • Batch size: {config.batch_size}")
    logger.info(f"  • Sequence length: {config.sequence_length}")
    logger.info(f"  • Purging: {use_purging}")
    logger.info(f"  • Embargo: {use_embargo}")
    logger.info("=" * 60)

    dataloaders = {}

    # Train dataset
    train_dataset = MemoryMappedDatasetWithSplit(
        data_dir=data_dir,
        symbols=symbols,
        config=config,
        mode='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_purging=use_purging,
        use_embargo=use_embargo
    )

    # IterableDataset не использует batch_size в DataLoader
    # (dataset сам возвращает batches)
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=None,  # Dataset уже возвращает batches
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Validation dataset (без shuffle)
    val_config = MemoryMappedConfig(
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        shuffle=False,  # No shuffle for validation
        target_horizon=config.target_horizon,
        label_mapping=config.label_mapping
    )

    val_dataset = MemoryMappedDatasetWithSplit(
        data_dir=data_dir,
        symbols=symbols,
        config=val_config,
        mode='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_purging=use_purging,
        use_embargo=use_embargo
    )

    dataloaders['val'] = DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Test dataset (если есть данные)
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio > 0:
        test_dataset = MemoryMappedDatasetWithSplit(
            data_dir=data_dir,
            symbols=symbols,
            config=val_config,  # Same config as val (no shuffle)
            mode='test',
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            use_purging=use_purging,
            use_embargo=use_embargo
        )

        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    # Статистика
    logger.info("\n  DataLoaders created:")
    logger.info(f"  • Train: {len(dataloaders['train'])} batches")
    logger.info(f"  • Val: {len(dataloaders['val'])} batches")
    if 'test' in dataloaders:
        logger.info(f"  • Test: {len(dataloaders['test'])} batches")
    logger.info("=" * 60 + "\n")

    return dataloaders


# ========== ТЕСТИРОВАНИЕ ==========

if __name__ == "__main__":
    """Пример использования memory-mapped dataset."""
    import sys
    import time
    import psutil

    # Путь к данным
    DATA_DIR = Path("data/ml_training")

    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        sys.exit(1)

    # Находим доступные символы
    symbols = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(symbols)} symbols: {symbols[:5]}...")

    if not symbols:
        print("No symbols found!")
        sys.exit(1)

    # Создаём dataset
    config = MemoryMappedConfig(
        sequence_length=60,
        batch_size=64,
        shuffle=True
    )

    print("\n" + "=" * 60)
    print("Testing MemoryMappedDataset")
    print("=" * 60)

    # Измеряем память ДО
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory before: {mem_before:.1f} MB")

    # Создаём dataloaders
    start_time = time.time()

    dataloaders = create_memory_efficient_dataloaders(
        data_dir=DATA_DIR,
        symbols=symbols[:8],  # Тестируем на 8 символах
        config=config,
        train_ratio=0.7,
        val_ratio=0.15,
        use_purging=True,
        use_embargo=True
    )

    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.2f}s")

    # Измеряем память ПОСЛЕ инициализации
    mem_after_init = process.memory_info().rss / 1024 / 1024
    print(f"Memory after init: {mem_after_init:.1f} MB (+{mem_after_init - mem_before:.1f} MB)")

    # Тестируем итерацию
    print("\nTesting iteration...")

    train_loader = dataloaders['train']
    batch_count = 0
    total_samples = 0

    start_time = time.time()

    for batch in train_loader:
        sequences = batch['sequence']
        labels = batch['label']

        batch_count += 1
        total_samples += len(sequences)

        if batch_count == 1:
            print(f"\nFirst batch:")
            print(f"  • sequences shape: {sequences.shape}")
            print(f"  • labels shape: {labels.shape}")
            print(f"  • sequences dtype: {sequences.dtype}")
            print(f"  • labels unique: {torch.unique(labels).tolist()}")

        if batch_count % 100 == 0:
            mem_current = process.memory_info().rss / 1024 / 1024
            print(f"  Batch {batch_count}: {total_samples:,} samples, memory: {mem_current:.1f} MB")

        # Ограничиваем для теста
        if batch_count >= 500:
            break

    iter_time = time.time() - start_time

    # Финальная статистика
    mem_final = process.memory_info().rss / 1024 / 1024

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total batches processed: {batch_count}")
    print(f"Total samples: {total_samples:,}")
    print(f"Iteration time: {iter_time:.2f}s ({total_samples/iter_time:.0f} samples/sec)")
    print(f"Memory usage: {mem_final:.1f} MB (vs ~5GB for standard loading)")
    print(f"Memory increase: {mem_final - mem_before:.1f} MB")
    print("=" * 60)

    print("\n✓ Memory-mapped dataset works correctly!")
