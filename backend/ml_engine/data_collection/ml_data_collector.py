"""
ML Data Collector - система сбора данных для обучения ML моделей.

Собирает:
- Feature vectors (112 признаков)
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
import pandas as pd

from backend.core.logger import get_logger
from backend.ml_engine.features import FeatureVector
from backend.models.orderbook import OrderBookSnapshot, OrderBookMetrics

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
      max_samples_per_file: int = 2000,  # ОПТИМИЗИРОВАНО: 10000 → 2000 (~2 MB/файл)
      collection_interval: int = 10,
      # auto_save_interval_seconds: int = 40000,# Каждые N итераций
      max_buffer_memory_mb: int = 200,  # ОПТИМИЗИРОВАНО: 200 → 80 (запас для адаптивности)
      # НОВОЕ: Feature Store integration
      enable_feature_store: bool = True,  # Записывать в Feature Store (parquet)
      use_legacy_format: bool = True,     # Записывать в legacy формат (.npy/.json)
      feature_store_group: str = "training_features"
  ):
    """
    Инициализация сборщика данных.

    Args:
        storage_path: Путь для хранения данных (legacy формат)
        max_samples_per_file: Максимум семплов в одном файле
        collection_interval: Интервал сбора (каждые N итераций)
        # auto_save_interval_seconds: Интервал автосохранения (секунды)
        max_buffer_memory_mb: Максимум памяти на буфер символа (МБ)
        enable_feature_store: Записывать ли в Feature Store (parquet)
        use_legacy_format: Записывать ли в legacy формат (.npy/.json)
        feature_store_group: Название feature group для Feature Store
    """
    self.storage_path = Path(storage_path)
    self.max_samples_per_file = max_samples_per_file
    self.initial_max_samples_per_file = max_samples_per_file  # НОВОЕ: Сохраняем начальное значение для адаптации
    self.collection_interval = collection_interval
    # self.auto_save_interval = auto_save_interval_seconds
    self.max_buffer_memory_mb = max_buffer_memory_mb

    # Feature Store integration
    self.enable_feature_store = enable_feature_store
    self.use_legacy_format = use_legacy_format
    self.feature_store_group = feature_store_group


    # Буферы для каждого символа
    self.feature_buffers: Dict[str, List[np.ndarray]] = {}
    self.label_buffers: Dict[str, List[Dict[str, Any]]] = {}
    self.metadata_buffers: Dict[str, List[Dict[str, Any]]] = {}

    # Счетчики
    self.sample_counts: Dict[str, int] = {}
    self.batch_numbers: Dict[str, int] = {}
    self.last_save_time: Dict[str, float] = {}  # Время последнего сохранения
    self.iteration_counter = 0
    self.last_cleanup_iteration = 0  # НОВОЕ: Счетчик для периодической очистки

    # Статистика
    self.total_samples_collected = 0
    self.files_written = 0

    # Feature Store (lazy initialization)
    self._feature_store = None

    logger.info(
      f"MLDataCollector инициализирован, storage_path={storage_path}, "
      f"max_samples={max_samples_per_file}, interval={collection_interval}, "
      f"feature_store={'✅' if enable_feature_store else '❌'}, "
      f"legacy={'✅' if use_legacy_format else '❌'}"
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
        feature_vector: Вектор признаков (112 признаков)
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
        self.last_save_time[symbol] = datetime.now().timestamp()

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

      # 🔧 АДАПТИВНАЯ ПОДСТРОЙКА (каждые 100 семплов)
      if self.total_samples_collected % 100 == 0:
        self._adapt_thresholds()

      # 🔥 ПРОАКТИВНАЯ ПРОВЕРКА ПАМЯТИ (КРИТИЧНО!)
      # Проверяем ПОСЛЕ добавления семпла, гарантируя сохранность данных
      buffer_size = len(self.feature_buffers[symbol])
      buffer_memory_mb = self._calculate_buffer_memory(symbol)
      memory_threshold_mb = self.max_buffer_memory_mb * 0.9  # 90% порог

      logger.debug(
        f"{symbol} | Собран семпл #{self.sample_counts[symbol]}, "
        f"буфер: {buffer_size}/{self.max_samples_per_file}, "
        f"память: {buffer_memory_mb:.2f}MB/{self.max_buffer_memory_mb}MB"
      )

      # Проверяем нужно ли сохранить batch (по количеству ИЛИ по памяти)
      should_save = False
      save_reason = ""

      if buffer_size >= self.max_samples_per_file:
        should_save = True
        save_reason = f"превышен лимит семплов ({buffer_size}/{self.max_samples_per_file})"
      elif buffer_memory_mb >= memory_threshold_mb:
        should_save = True
        save_reason = f"превышен лимит памяти ({buffer_memory_mb:.2f}MB/{memory_threshold_mb:.2f}MB)"

      if should_save:
        logger.info(f"{symbol} | 💾 Автосохранение: {save_reason}")
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
        "future_direction_10s": None,  # После preprocessing: 0=DOWN, 1=NEUTRAL, 2=UP
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
    Поддерживает legacy формат (.npy/.json) и Feature Store (parquet).

    Args:
        symbol: Торговая пара
    """
    try:
      if not self.feature_buffers[symbol]:
        return

      # 1. Сохраняем в Feature Store (если включено)
      if self.enable_feature_store:
        await self._save_to_feature_store(symbol)

      # 2. Сохраняем в legacy формат (если включено)
      if self.use_legacy_format:
        await self._save_legacy_batch(symbol)

      # Очищаем буферы (после обоих сохранений)
      buffer_size = len(self.feature_buffers[symbol])
      self.feature_buffers[symbol].clear()
      self.label_buffers[symbol].clear()
      self.metadata_buffers[symbol].clear()

      # Инкрементируем batch number
      self.batch_numbers[symbol] += 1

      logger.info(
        f"✓ {symbol} | Batch сохранен: {buffer_size} семплов "
        f"(FS={'✅' if self.enable_feature_store else '❌'}, "
        f"Legacy={'✅' if self.use_legacy_format else '❌'})"
      )

    except Exception as e:
      logger.error(f"{symbol} | Ошибка сохранения batch: {e}", exc_info=True)

  async def _save_legacy_batch(self, symbol: str):
    """
    Сохранение в legacy формат (.npy/.json).

    Args:
        symbol: Торговая пара
    """
    try:
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

      # Сохраняем labels (json)
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
        f"{symbol} | Legacy batch #{batch_num}: "
        f"{len(self.feature_buffers[symbol])} семплов, "
        f"shape={features_array.shape}"
      )

    except Exception as e:
      logger.error(f"{symbol} | Ошибка сохранения legacy batch: {e}", exc_info=True)

  def _get_feature_store(self):
    """Lazy initialization Feature Store"""
    if self._feature_store is None:
      from backend.ml_engine.feature_store.feature_store import get_feature_store
      self._feature_store = get_feature_store()
    return self._feature_store

  async def _save_to_feature_store(self, symbol: str):
    """
    Сохранение данных в Feature Store (parquet формат).

    Args:
        symbol: Торговая пара
    """
    try:
      if not self.feature_buffers[symbol]:
        return

      logger.info(f"{symbol} | Сохранение в Feature Store...")

      # Получаем буферы
      features_list = self.feature_buffers[symbol]
      labels_list = self.label_buffers[symbol]
      metadata_list = self.metadata_buffers[symbol]

      # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем правильные названия колонок из FeatureStoreSchema
      from backend.ml_engine.feature_store.feature_schema import DEFAULT_SCHEMA
      feature_column_names = DEFAULT_SCHEMA.get_all_feature_columns()

      if len(feature_column_names) != 112:
        logger.error(
          f"Feature schema mismatch: expected 112 columns, got {len(feature_column_names)}"
        )
        return

      # Конвертируем в DataFrame
      rows = []

      for feature_arr, label_dict, meta_dict in zip(features_list, labels_list, metadata_list):
        # Базовая информация
        row = {
          'symbol': symbol,
          'timestamp': meta_dict['timestamp'],
        }

        # Распаковываем 112 признаков с ПРАВИЛЬНЫМИ НАЗВАНИЯМИ
        # feature_arr имеет shape (112,)
        if len(feature_arr) != 112:
          logger.warning(
            f"{symbol} | Feature array length mismatch: expected 112, got {len(feature_arr)}"
          )
          continue

        # Используем правильные названия колонок из схемы
        for i, feature_name in enumerate(feature_column_names):
          row[feature_name] = feature_arr[i]

        # Добавляем метки (labels)
        # Метки могут быть None (будут заполнены позже через preprocessing)
        row['future_direction_10s'] = label_dict.get('future_direction_10s')
        row['future_direction_30s'] = label_dict.get('future_direction_30s')
        row['future_direction_60s'] = label_dict.get('future_direction_60s')
        row['future_movement_10s'] = label_dict.get('future_movement_10s')
        row['future_movement_30s'] = label_dict.get('future_movement_30s')
        row['future_movement_60s'] = label_dict.get('future_movement_60s')

        # Сохраняем current_mid_price для preprocessing
        row['current_mid_price'] = label_dict.get('current_mid_price')

        # Метаданные сигнала (опционально)
        row['signal_type'] = meta_dict.get('signal_type')
        row['signal_confidence'] = meta_dict.get('signal_confidence')
        row['signal_strength'] = meta_dict.get('signal_strength')

        rows.append(row)

      if not rows:
        logger.warning(f"{symbol} | No valid rows to save")
        return

      # Создаем DataFrame
      df = pd.DataFrame(rows)

      # Сортируем по timestamp
      df = df.sort_values('timestamp').reset_index(drop=True)

      # Логируем информацию о колонках для отладки
      logger.info(
        f"{symbol} | DataFrame columns: {len(df.columns)}, "
        f"feature columns: {len([c for c in df.columns if c in feature_column_names])}"
      )

      # Записываем в Feature Store
      feature_store = self._get_feature_store()
      success = feature_store.write_offline_features(
        feature_group=self.feature_store_group,
        features=df,
        timestamp_column='timestamp'
      )

      if success:
        logger.info(
          f"✓ {symbol} | Feature Store: сохранено {len(df)} семплов в parquet с правильными названиями колонок"
        )
      else:
        logger.error(f"{symbol} | Ошибка записи в Feature Store")

      # CRITICAL: Explicitly delete DataFrame to free memory
      # Pandas does not release memory automatically
      # NOTE: gc.collect() removed - too slow (8 sec per save)
      # GC will collect automatically during periodic cleanup
      del df

    except Exception as e:
      logger.error(f"{symbol} | Ошибка сохранения в Feature Store: {e}", exc_info=True)

  async def finalize(self):
    """Финализация - сохранение всех оставшихся буферов."""
    logger.info("Финализация MLDataCollector...")

    # ВАЖНО: При финализации сохраняем ВСЕ буферы, даже маленькие (min_buffer_size=0)
    await self._emergency_save_all_buffers(min_buffer_size=0)

    logger.info(
      f"MLDataCollector финализирован: "
      f"всего семплов={self.total_samples_collected}, "
      f"файлов={self.files_written}"
    )

  def _calculate_buffer_memory(self, symbol: str) -> float:
    """
    Расчет размера буфера в памяти (МБ).

    Args:
        symbol: Торговая пара

    Returns:
        float: Размер буфера в МБ
    """
    if symbol not in self.feature_buffers:
      return 0.0

    try:
      # Размер feature буфера
      features_size = 0
      for arr in self.feature_buffers[symbol]:
        features_size += arr.nbytes

      # Размер label буфера (приблизительно)
      labels_size = len(self.label_buffers[symbol]) * 200  # ~200 bytes per label

      # Размер metadata буфера (приблизительно)
      metadata_size = len(self.metadata_buffers[symbol]) * 300  # ~300 bytes per metadata

      total_bytes = features_size + labels_size + metadata_size
      total_mb = total_bytes / (1024 * 1024)

      return total_mb

    except Exception as e:
      logger.error(f"{symbol} | Ошибка расчета размера буфера: {e}")
      return 0.0

  def _calculate_total_buffer_memory(self) -> float:
    """
    Расчет ОБЩЕГО размера ВСЕХ буферов в памяти (МБ).

    Returns:
        float: Общий размер всех буферов в МБ
    """
    total_mb = 0.0
    for symbol in self.feature_buffers.keys():
      total_mb += self._calculate_buffer_memory(symbol)
    return total_mb

  def _adapt_thresholds(self):
    """
    🔧 АДАПТИВНАЯ ПОДСТРОЙКА ПОРОГОВ.

    Динамически изменяет max_samples_per_file в зависимости от текущего
    использования памяти ВСЕМИ буферами.

    Логика:
    - Высокое давление памяти (>70% от лимита) → снижаем порог до 1000
    - Среднее давление (>50%) → снижаем до 1500
    - Нормальное использование → возвращаем к базовому (2000)

    ЦЕЛЬ: Превентивное сохранение при высокой нагрузке.
    """
    total_memory_mb = self._calculate_total_buffer_memory()
    symbol_count = len(self.feature_buffers)

    if symbol_count == 0:
      return

    # Рассчитываем лимит для всех символов
    total_memory_limit_mb = self.max_buffer_memory_mb * symbol_count
    memory_usage_percent = (total_memory_mb / total_memory_limit_mb) * 100

    old_threshold = self.max_samples_per_file

    # Адаптивная подстройка
    if memory_usage_percent > 70:
      # Критическое использование → агрессивно снижаем порог
      self.max_samples_per_file = int(self.initial_max_samples_per_file * 0.5)  # 1000
    elif memory_usage_percent > 50:
      # Высокое использование → умеренно снижаем
      self.max_samples_per_file = int(self.initial_max_samples_per_file * 0.75)  # 1500
    else:
      # Нормальное использование → базовый порог
      self.max_samples_per_file = self.initial_max_samples_per_file  # 2000

    if old_threshold != self.max_samples_per_file:
      logger.info(
        f"🔧 Адаптивная подстройка: max_samples_per_file {old_threshold} → {self.max_samples_per_file} "
        f"(память: {total_memory_mb:.1f}MB/{total_memory_limit_mb:.1f}MB, {memory_usage_percent:.1f}%)"
      )

  async def _emergency_save_all_buffers(self, min_buffer_size: int = 50):
    """
    🚨 ЭКСТРЕННОЕ СОХРАНЕНИЕ ВСЕХ БУФЕРОВ.

    Вызывается при критическом превышении памяти или периодической очистке.
    В отличие от старого _cleanup_old_buffers(), СОХРАНЯЕТ ВСЕ данные,
    вместо их урезания.

    Args:
        min_buffer_size: Минимальный размер буфера для сохранения (по умолчанию 50)
                        Буферы меньше этого размера не сохраняются (кроме финализации)

    КРИТИЧНО: Ноль потери данных!
    """
    import gc

    logger.info(f"🧹 Проверка буферов ML Data Collector (минимум для сохранения: {min_buffer_size})")

    saved_symbols = []
    skipped_symbols = []
    total_saved_samples = 0

    # Сохраняем только достаточно большие буферы
    for symbol in list(self.feature_buffers.keys()):
      if self.feature_buffers[symbol]:
        buffer_size = len(self.feature_buffers[symbol])
        buffer_memory = self._calculate_buffer_memory(symbol)

        # Проверяем минимальный размер буфера
        if buffer_size >= min_buffer_size:
          # Сохраняем batch
          await self._save_batch(symbol)

          saved_symbols.append(f"{symbol}({buffer_size} семплов, {buffer_memory:.2f}MB)")
          total_saved_samples += buffer_size
        else:
          skipped_symbols.append(f"{symbol}({buffer_size})")

    if saved_symbols:
      logger.warning(
        f"💾 Экстренно сохранено {total_saved_samples} семплов: {', '.join(saved_symbols)}"
      )

    if skipped_symbols:
      logger.info(
        f"⏭️  Пропущены маленькие буферы (< {min_buffer_size}): {', '.join(skipped_symbols)}"
      )

    # Принудительная сборка мусора
    gc.collect()
    logger.info("🧹 Сборка мусора выполнена")

  def get_statistics(self) -> Dict[str, Any]:
    """
    Получение статистики сбора данных.

    Returns:
        Dict: Статистика с информацией о памяти и адаптивных порогах
    """
    symbol_stats = {}
    for symbol in self.sample_counts.keys():
      buffer_memory = self._calculate_buffer_memory(symbol)
      symbol_stats[symbol] = {
        "total_samples": self.sample_counts[symbol],
        "current_batch": self.batch_numbers[symbol],
        "buffer_size": len(self.feature_buffers.get(symbol, [])),
        "buffer_memory_mb": round(buffer_memory, 2),
        "memory_utilization_percent": round(
          (buffer_memory / self.max_buffer_memory_mb) * 100, 1
        ) if self.max_buffer_memory_mb > 0 else 0
      }

    total_memory = self._calculate_total_buffer_memory()

    return {
      "total_samples_collected": self.total_samples_collected,
      "files_written": self.files_written,
      "iteration_counter": self.iteration_counter,
      "collection_interval": self.collection_interval,
      "memory": {
        "total_buffer_memory_mb": round(total_memory, 2),
        "max_buffer_memory_mb_per_symbol": self.max_buffer_memory_mb,
        "current_max_samples_per_file": self.max_samples_per_file,
        "initial_max_samples_per_file": self.initial_max_samples_per_file
      },
      "symbols": symbol_stats
    }