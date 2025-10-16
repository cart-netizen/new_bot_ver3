"""
Data Loader для загрузки и подготовки обучающих данных.

Функциональность:
- Загрузка features и labels из .npy и .json файлов
- Создание временных последовательностей (sequences)
- Train/Val/Test split с walk-forward validation
- Data augmentation для улучшения генерализации
- Batch creation с DataLoader

Путь: backend/ml_engine/training/data_loader.py
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader

from core.logger import get_logger

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
  - Data augmentation
  """

  def __init__(self, config: DataConfig):
    """Инициализация загрузчика."""
    self.config = config
    self.storage_path = Path(config.storage_path)

    if not self.storage_path.exists():
      raise FileNotFoundError(
        f"Директория данных не найдена: {self.storage_path}"
      )

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
        labels: (N,)
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

    # Проверяем соответствие размеров
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]
    timestamps = timestamps[:min_len]

    logger.info(
      f"Загружены данные {symbol}: samples={len(X)}, "
      f"features={X.shape[1]}"
    )

    return X, y, timestamps

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

  def create_dataloaders(
      self,
      train_data: Tuple[np.ndarray, np.ndarray],
      val_data: Tuple[np.ndarray, np.ndarray],
      test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
  ) -> Dict[str, DataLoader]:
    """
    Создать PyTorch DataLoaders.

    Args:
        train_data: (sequences, labels)
        val_data: (sequences, labels)
        test_data: (sequences, labels) опционально

    Returns:
        Dict с DataLoaders
    """
    train_sequences, train_labels = train_data
    val_sequences, val_labels = val_data

    # Создаем datasets
    train_dataset = TradingDataset(train_sequences, train_labels)
    val_dataset = TradingDataset(val_sequences, val_labels)

    # Создаем dataloaders
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
      shuffle=False,
      num_workers=self.config.num_workers,
      pin_memory=True
    )

    dataloaders = {
      'train': train_loader,
      'val': val_loader
    }

    # Test loader если данные предоставлены
    if test_data is not None:
      test_sequences, test_labels = test_data
      test_dataset = TradingDataset(test_sequences, test_labels)
      test_loader = DataLoader(
        test_dataset,
        batch_size=self.config.batch_size,
        shuffle=False,
        num_workers=self.config.num_workers,
        pin_memory=True
      )
      dataloaders['test'] = test_loader

    logger.info(
      f"Созданы DataLoaders: "
      f"train_batches={len(train_loader)}, "
      f"val_batches={len(val_loader)}"
    )

    return dataloaders

  def load_and_prepare(
      self,
      symbols: List[str],
      use_walk_forward: bool = False,
      n_splits: int = 5
  ) -> Dict:
    """
    Полный pipeline загрузки и подготовки данных.

    Args:
        symbols: Список символов для загрузки
        use_walk_forward: Использовать walk-forward split
        n_splits: Количество фолдов для walk-forward

    Returns:
        Dict с подготовленными данными
    """
    all_sequences = []
    all_labels = []

    for symbol in symbols:
      try:
        # Загружаем данные
        features, labels, timestamps = self.load_symbol_data(symbol)

        # Создаем последовательности
        sequences, seq_labels, seq_timestamps = self.create_sequences(
          features, labels, timestamps
        )

        all_sequences.append(sequences)
        all_labels.append(seq_labels)

      except Exception as e:
        logger.error(f"Ошибка загрузки {symbol}: {e}")
        continue

    if not all_sequences:
      raise ValueError("Не удалось загрузить данные ни для одного символа")

    # Объединяем все символы
    X = np.concatenate(all_sequences, axis=0)
    y = np.concatenate(all_labels, axis=0)

    logger.info(
      f"Объединены данные: total_samples={len(X)}, "
      f"symbols={len(symbols)}"
    )

    # Создаем timestamps для всего датасета (dummy)
    timestamps = np.arange(len(X))

    if use_walk_forward:
      # Walk-forward split
      splits = self.walk_forward_split(X, y, timestamps, n_splits)
      return {'walk_forward_splits': splits}
    else:
      # Простой train/val/test split
      n_samples = len(X)
      train_end = int(n_samples * self.config.train_ratio)
      val_end = int(n_samples * (self.config.train_ratio + self.config.val_ratio))

      train_data = (X[:train_end], y[:train_end])
      val_data = (X[train_end:val_end], y[train_end:val_end])
      test_data = (X[val_end:], y[val_end:])

      dataloaders = self.create_dataloaders(
        train_data, val_data, test_data
      )

      return {
        'dataloaders': dataloaders,
        'data': {
          'train': train_data,
          'val': val_data,
          'test': test_data
        }
      }


# Пример использования
if __name__ == "__main__":
  config = DataConfig(
    storage_path="data/ml_training",
    sequence_length=60,
    batch_size=64
  )

  loader = HistoricalDataLoader(config)

  # Загрузка и подготовка данных
  symbols = ["BTCUSDT", "ETHUSDT"]

  result = loader.load_and_prepare(
    symbols=symbols,
    use_walk_forward=False
  )

  # Получаем dataloaders
  dataloaders = result['dataloaders']

  print("DataLoaders готовы:")
  for split, dataloader in dataloaders.items():
    print(f"  {split}: {len(dataloader)} batches")