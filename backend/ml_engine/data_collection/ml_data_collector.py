"""
ML Data Collector - система сбора данных для обучения ML моделей.

Собирает:
- Feature vectors (110 признаков)
- Market state (orderbook snapshot, метрики)
- Labels (future price movement, signals)

Файл: backend/ml_engine/data_collection/ml_data_collector.py
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import numpy as np

from core.logger import get_logger
from ml_engine.features import FeatureVector
from models.orderbook import OrderBookSnapshot, OrderBookMetrics

logger = get_logger(__name__)


class MLDataCollector:
  """
  Сборщик данных для ML обучения.

  Архитектура хранения:
  data/ml_training/
    ├── BTCUSDT/
    │   ├── features/
    │   │   ├── 2025-01-15_batch_0001.npy  # Массивы признаков
    │   │   └── 2025-01-15_batch_0002.npy
    │   ├── labels/
    │   │   ├── 2025-01-15_batch_0001.npy  # Метки (таргеты)
    │   │   └── 2025-01-15_batch_0002.npy
    │   └── metadata/
    │       ├── 2025-01-15_batch_0001.json  # Метаданные
    │       └── 2025-01-15_batch_0002.json
    └── ETHUSDT/
        └── ...
  """

  def __init__(
      self,
      storage_path: str = "data/ml_training",
      max_samples_per_file: int = 10000,
      collection_interval: int = 10  # Каждые N итераций
  ):
    """
    Инициализация сборщика данных.

    Args:
        storage_path: Путь для хранения данных
        max_samples_per_file: Максимум семплов в одном файле
        collection_interval: Интервал сбора (каждые N итераций)
    """
    self.storage_path = Path(storage_path)
    self.max_samples_per_file = max_samples_per_file
    self.collection_interval = collection_interval

    # Буферы для каждого символа
    self.feature_buffers: Dict[str, List[np.ndarray]] = {}
    self.label_buffers: Dict[str, List[Dict[str, Any]]] = {}
    self.metadata_buffers: Dict[str, List[Dict[str, Any]]] = {}

    # Счетчики
    self.sample_counts: Dict[str, int] = {}
    self.batch_numbers: Dict[str, int] = {}
    self.iteration_counter = 0

    # Статистика
    self.total_samples_collected = 0
    self.files_written = 0

    logger.info(
      f"MLDataCollector инициализирован, storage_path={storage_path}, "
      f"interval={collection_interval}"
    )

  async def initialize(self):
    """Инициализация хранилища."""
    try:
      # Создаем директории
      self.storage_path.mkdir(parents=True, exist_ok=True)
      logger.info(f"Создана директория хранилища: {self.storage_path}")

    except Exception as e:
      logger.error(f"Ошибка инициализации хранилища: {e}")
      raise

  def should_collect(self) -> bool:
    """
    Проверка нужно ли собирать данные в этой итерации.

    Returns:
        bool: True если нужно собрать
    """
    self.iteration_counter += 1
    return self.iteration_counter % self.collection_interval == 0

  async def collect_sample(
      self,
      symbol: str,
      feature_vector: FeatureVector,
      orderbook_snapshot: OrderBookSnapshot,
      market_metrics: OrderBookMetrics,
      executed_signal: Optional[Dict[str, Any]] = None
  ):
    """
    Сбор одного семпла данных.

    Args:
        symbol: Торговая пара
        feature_vector: Вектор признаков (110 признаков)
        orderbook_snapshot: Снимок стакана
        market_metrics: Рыночные метрики
        executed_signal: Исполненный сигнал (если есть)
    """
    try:
      # Инициализируем буферы для символа
      if symbol not in self.feature_buffers:
        self.feature_buffers[symbol] = []
        self.label_buffers[symbol] = []
        self.metadata_buffers[symbol] = []
        self.sample_counts[symbol] = 0
        self.batch_numbers[symbol] = 1

      # Извлекаем массив признаков
      features_array = feature_vector.to_array()

      # Создаем метку (label) для supervised learning
      # Здесь нужно будет добавить логику расчета future price movement
      label = self._create_label(
        orderbook_snapshot,
        market_metrics,
        executed_signal
      )

      # Создаем метаданные
      metadata = {
        "timestamp": orderbook_snapshot.timestamp,
        "symbol": symbol,
        "mid_price": orderbook_snapshot.mid_price,
        "spread": orderbook_snapshot.spread,
        "imbalance": market_metrics.imbalance,
        # Сохраняем полную информацию о сигнале
        "signal_type": executed_signal.get("type") if executed_signal else None,
        "signal_confidence": executed_signal.get("confidence") if executed_signal else None,
        "signal_strength": executed_signal.get("strength") if executed_signal else None,
        "feature_count": feature_vector.feature_count
      }

      # Добавляем в буферы
      self.feature_buffers[symbol].append(features_array)
      self.label_buffers[symbol].append(label)
      self.metadata_buffers[symbol].append(metadata)

      self.sample_counts[symbol] += 1
      self.total_samples_collected += 1

      logger.debug(
        f"{symbol} | Собран семпл #{self.sample_counts[symbol]}, "
        f"буфер: {len(self.feature_buffers[symbol])}/{self.max_samples_per_file}"
      )

      # Проверяем нужно ли сохранить batch
      if len(self.feature_buffers[symbol]) >= self.max_samples_per_file:
        await self._save_batch(symbol)

    except Exception as e:
      logger.error(f"{symbol} | Ошибка сбора семпла: {e}")

  def _create_label(
        self,
        orderbook_snapshot: OrderBookSnapshot,
        market_metrics: OrderBookMetrics,
        executed_signal: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
      """
      Создание метки (label) для supervised learning.

      Future targets (direction, movement) будут добавлены ПОЗЖЕ
      через preprocessing скрипт, который обработает собранные данные.

      В режиме real-time мы НЕ МОЖЕМ знать будущую цену,
      поэтому сейчас сохраняем только текущее состояние.

      Args:
          orderbook_snapshot: Снимок стакана
          market_metrics: Метрики
          executed_signal: Исполненный сигнал

      Returns:
          Dict: Метка с текущим состоянием и заглушками для future targets
      """
      label = {
        # ===== FUTURE TARGETS (заполнятся через preprocessing) =====
        # Эти поля будут рассчитаны ПОСЛЕ сбора данных,
        # когда мы будем знать, что произошло с ценой через N секунд
        "future_direction_10s": None,  # 1=up, 0=neutral, -1=down
        "future_direction_30s": None,
        "future_direction_60s": None,
        "future_movement_10s": None,  # % изменения цены
        "future_movement_30s": None,
        "future_movement_60s": None,

        # Current state
        "timestamp": orderbook_snapshot.timestamp,  # ИСПРАВЛЕНО: добавлен timestamp для preprocessing
        "current_mid_price": orderbook_snapshot.mid_price,
        "current_imbalance": market_metrics.imbalance,

        # ===== ИСПРАВЛЕНИЕ: Сохраняем ВСЮ информацию о сигнале =====
        "signal_type": executed_signal.get("type") if executed_signal else None,
        "signal_confidence": executed_signal.get("confidence") if executed_signal else None,
        "signal_strength": executed_signal.get("strength") if executed_signal else None,
      }

      return label

  async def _save_batch(self, symbol: str):
    """
    Сохранение batch данных на диск.

    Args:
        symbol: Торговая пара
    """
    try:
      if not self.feature_buffers[symbol]:
        return

      # Создаем директории для символа
      symbol_dir = self.storage_path / symbol
      features_dir = symbol_dir / "features"
      labels_dir = symbol_dir / "labels"
      metadata_dir = symbol_dir / "metadata"

      for dir_path in [features_dir, labels_dir, metadata_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

      # Формируем имя файла
      date_str = datetime.now().strftime("%Y-%m-%d")
      batch_num = self.batch_numbers[symbol]
      filename_base = f"{date_str}_batch_{batch_num:04d}"

      # Сохраняем features (numpy)
      features_array = np.array(self.feature_buffers[symbol])
      features_file = features_dir / f"{filename_base}.npy"
      np.save(features_file, features_array)

      # Сохраняем labels (numpy)
      # Конвертируем labels в структурированный массив
      labels_file = labels_dir / f"{filename_base}.json"
      with open(labels_file, 'w') as f:
        json.dump(self.label_buffers[symbol], f, indent=2)

      # Сохраняем metadata (json)
      metadata_file = metadata_dir / f"{filename_base}.json"
      with open(metadata_file, 'w') as f:
        json.dump({
          "batch_info": {
            "symbol": symbol,
            "batch_number": batch_num,
            "sample_count": len(self.feature_buffers[symbol]),
            "timestamp": datetime.now().isoformat(),
            "feature_shape": features_array.shape
          },
          "samples": self.metadata_buffers[symbol]
        }, f, indent=2)

      self.files_written += 3  # features, labels, metadata

      logger.info(
        f"{symbol} | Сохранен batch #{batch_num}: "
        f"{len(self.feature_buffers[symbol])} семплов, "
        f"features_shape={features_array.shape}"
      )

      # Очищаем буферы
      self.feature_buffers[symbol].clear()
      self.label_buffers[symbol].clear()
      self.metadata_buffers[symbol].clear()

      # Инкрементируем batch number
      self.batch_numbers[symbol] += 1

    except Exception as e:
      logger.error(f"{symbol} | Ошибка сохранения batch: {e}")

  async def finalize(self):
    """Финализация - сохранение всех оставшихся буферов."""
    logger.info("Финализация MLDataCollector...")

    for symbol in self.feature_buffers.keys():
      if self.feature_buffers[symbol]:
        await self._save_batch(symbol)
        logger.info(f"{symbol} | Финальный batch сохранен")

    logger.info(
      f"MLDataCollector финализирован: "
      f"всего семплов={self.total_samples_collected}, "
      f"файлов={self.files_written}"
    )

  def get_statistics(self) -> Dict[str, Any]:
    """
    Получение статистики сбора данных.

    Returns:
        Dict: Статистика
    """
    symbol_stats = {}
    for symbol in self.sample_counts.keys():
      symbol_stats[symbol] = {
        "total_samples": self.sample_counts[symbol],
        "current_batch": self.batch_numbers[symbol],
        "buffer_size": len(self.feature_buffers.get(symbol, []))
      }

    return {
      "total_samples_collected": self.total_samples_collected,
      "files_written": self.files_written,
      "iteration_counter": self.iteration_counter,
      "collection_interval": self.collection_interval,
      "symbols": symbol_stats
    }