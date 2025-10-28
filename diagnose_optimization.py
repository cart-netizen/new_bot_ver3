#!/usr/bin/env python3
"""
Диагностический скрипт для проверки готовности к оптимизации.

Запуск:
    python diagnose_optimization.py --symbol BTCUSDT
"""
import sys
from pathlib import Path
from datetime import datetime

# Добавляем путь к backend
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))
import time
import logging
import argparse
import numpy as np
import json
from pathlib import Path
from collections import Counter
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict
import multiprocessing as mp

# Настройка логирования
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
  """Проверка окружения."""
  print("\n" + "=" * 60)
  print("ПРОВЕРКА ОКРУЖЕНИЯ")
  print("=" * 60)
  start_time = time.time()
  logger.info("=" * 80)
  logger.info("НАЧАЛО ПРОВЕРКИ ОКРУЖЕНИЯ")
  logger.info("=" * 80)
  try:
    logger.info("Проверка PyTorch...")
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    logger.info(f"✓ PyTorch {torch.__version__} установлен")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
      print(f"  CUDA version: {torch.version.cuda}")
      print(f"  GPU: {torch.cuda.get_device_name(0)}")
      logger.info(f"  CUDA version: {torch.version.cuda}")
      logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
      logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
  except ImportError as e:
    print("❌ PyTorch не установлен!")
    logger.error(f"❌ PyTorch не установлен: {e}")
    return False

  try:
    logger.info("Проверка Scikit-learn...")
    import sklearn
    print(f"✓ Scikit-learn: {sklearn.__version__}")
    logger.info(f"✓ Scikit-learn {sklearn.__version__} установлен")
  except ImportError as e:
    print("❌ Scikit-learn не установлен!")
    logger.error(f"❌ Scikit-learn не установлен: {e}")
    return False

  try:
    logger.info("Проверка imbalanced-learn...")
    from imblearn import over_sampling
    print(f"✓ imbalanced-learn установлен")
    logger.info("✓ imbalanced-learn установлен")
  except ImportError:
    print("⚠️  imbalanced-learn не установлен (опционально)")
    logger.warning("⚠️  imbalanced-learn не установлен (опционально)")

    # Проверка доступных CPU ядер
    cpu_count = mp.cpu_count()
    print(f"✓ CPU cores: {cpu_count}")
    logger.info(f"✓ Доступно CPU ядер: {cpu_count}")

    elapsed = time.time() - start_time
    logger.info(f"✓ Проверка окружения завершена за {elapsed:.2f}s")

  print()
  return True

def _count_samples_in_file(file_path: Path) -> int:
  """Подсчет количества семплов в файле (вспомогательная функция для параллелизации)."""
  try:
    data = np.load(file_path, mmap_mode='r')  # Memory-mapped для быстрой загрузки
    return data.shape[0]
  except Exception as e:
    logger.warning(f"Ошибка при чтении {file_path.name}: {e}")
    return 0

def check_data(symbol: str, data_path: str = "data/ml_training"):
  """Проверка данных."""
  start_time = time.time()
  logger.info("=" * 80)
  logger.info(f"НАЧАЛО ПРОВЕРКИ ДАННЫХ: {symbol}")
  logger.info("=" * 80)
  print("=" * 60)
  print(f"ПРОВЕРКА ДАННЫХ: {symbol}")
  print("=" * 60)

  logger.info(f"Путь к данным: {data_path}")

  symbol_path = Path(data_path) / symbol

  if not symbol_path.exists():
    logger.error(f"❌ Директория не найдена: {symbol_path}")
    print(f"❌ Директория не найдена: {symbol_path}")
    return False

  logger.info(f"✓ Директория найдена: {symbol_path}")
  features_dir = symbol_path / "features"
  labels_dir = symbol_path / "labels"

  if not features_dir.exists():
    print(f"❌ Директория features не найдена!")
    logger.error(f"❌ Директория features не найдена: {features_dir}")
    return False

  if not labels_dir.exists():
    print(f"❌ Директория labels не найдена!")
    logger.error(f"❌ Директория labels не найдена: {labels_dir}")
    return False
  logger.info("✓ Директории features и labels найдены")
  # Подсчет файлов
  logger.info("Подсчет файлов...")
  files_start = time.time()

  feature_files = list(features_dir.glob("*.npy"))
  label_files = list(labels_dir.glob("*.json"))

  files_time = time.time() - files_start
  logger.info(f"✓ Файлы подсчитаны за {files_time:.2f}s")

  print(f"\nФайлы:")
  print(f"  Feature files: {len(feature_files)}")
  print(f"  Label files: {len(label_files)}")

  logger.info(f"Feature files: {len(feature_files)}")
  logger.info(f"Label files: {len(label_files)}")

  if len(feature_files) == 0:
    print("❌ Нет файлов features!")
    logger.error("❌ Нет файлов features!")
    return False

  if len(label_files) == 0:
    print("❌ Нет файлов labels!")
    logger.error("❌ Нет файлов labels!")
    return False

  # Загрузка примеров
  logger.info("Загрузка и анализ примеров данных...")
  print(f"\nЗагрузка примеров...")
  load_start = time.time()
  try:
    # Features
    logger.info(f"Загрузка примера features из {feature_files[0].name}...")
    sample_features = np.load(feature_files[0])
    print(f"  ✓ Features shape: {sample_features.shape}")
    print(f"    Expected: (N, 110)")

    if sample_features.shape[1] != 110:
      logger.warning(f"⚠️  Неожиданное количество признаков: {sample_features.shape[1]} вместо 110")
      print(f"    ⚠️  Неожиданное количество признаков!")

    # Labels
    logger.info(f"Загрузка примера labels из {label_files[0].name}...")
    with open(label_files[0]) as f:
      sample_labels = json.load(f)

    print(f"  ✓ Labels loaded: {len(sample_labels)} samples")
    logger.info(f"✓ Labels загружены: {len(sample_labels)} семплов")

    load_time = time.time() - load_start
    logger.info(f"✓ Примеры загружены за {load_time:.2f}s")

    # Проверка структуры labels
    if sample_labels:
      first_label = sample_labels[0]
      print(f"\n  Label structure:")
      for key in first_label.keys():
        print(f"    - {key}")

      # Проверка обязательных полей
      required = ["timestamp", "current_mid_price"]
      future_fields = ["future_direction_60s", "future_movement_60s"]

      missing_required = [f for f in required if f not in first_label]
      missing_future = [f for f in future_fields if f not in first_label]

      if missing_required:
        print(f"\n  ❌ Отсутствуют обязательные поля: {missing_required}")
        return False

      if missing_future:
        print(f"\n  ⚠️  Отсутствуют future labels: {missing_future}")
        print(f"     Запустите: python preprocessing_add_future_labels.py")
        return False

      # Проверка значений future_direction
      future_dir = first_label.get("future_direction_60s")
      if future_dir not in [-1, 0, 1, None]:
        print(f"\n  ❌ Невалидное значение future_direction_60s: {future_dir}")
        return False

  except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    import traceback
    traceback.print_exc()
    return False

  # Подсчет общего количества семплов
  logger.info("Подсчет общего количества семплов...")
  print(f"\nПодсчет семплов...")
  count_start = time.time()

  # Параллельный подсчет семплов для ускорения
  logger.info(f"Параллельная обработка {len(feature_files)} файлов...")
  total_samples = 0

  # Используем ThreadPoolExecutor для I/O операций
  max_workers = min(mp.cpu_count(), len(feature_files))
  logger.info(f"Использование {max_workers} воркеров для подсчета")

  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Запускаем параллельный подсчет
    futures = {executor.submit(_count_samples_in_file, f): f for f in feature_files}

    # Собираем результаты
    for future in as_completed(futures):
      count = future.result()
      total_samples += count

  count_time = time.time() - count_start
  logger.info(f"✓ Подсчет завершен за {count_time:.2f}s")

  print(f"  Всего семплов: {total_samples:,}")
  logger.info(f"Всего семплов: {total_samples:,}")

  # Рекомендации по объему
  if total_samples < 10_000:
    print(f"  ❌ КРИТИЧНО: Слишком мало данных ({total_samples:,} < 10,000)")
    print(f"     Минимум для тестирования: 10,000")
    print(f"     Рекомендуется: 100,000+")
    return False
  elif total_samples < 50_000:
    print(f"  ⚠️  Маловато данных для оптимизации ({total_samples:,} < 50,000)")
    print(f"     Используйте --quick режим")
    print(f"     Рекомендуется: 100,000+")
  elif total_samples < 100_000:
    print(f"  ✓ Достаточно для quick optimization")
    print(f"     Для full optimization рекомендуется: 100,000+")
  else:
    print(f"  ✓ Отлично! Достаточно данных для full optimization")

  # Анализ распределения классов
  logger.info("Анализ распределения классов...")
  print(f"\nАнализ распределения классов...")
  analysis_start = time.time()


  # Оптимизация: используем векторизацию для более быстрого анализа
  logger.info(f"Загрузка labels из первых {min(10, len(label_files))} файлов для анализа...")

  all_directions = []
  # Загружаем больше файлов для более точного анализа, но не все
  sample_size = min(10, len(label_files))
  for i, label_file in enumerate(label_files[:sample_size]):
    logger.debug(f"Обработка файла {i + 1}/{sample_size}: {label_file.name}")  # Первые 3 файла для примера
    with open(label_file) as f:
      labels = json.load(f)
      for label in labels:
        direction = label.get("future_direction_60s")
        if direction is not None:
          all_directions.append(direction)

  analysis_time = time.time() - analysis_start
  logger.info(f"✓ Анализ выполнен за {analysis_time:.2f}s")

  if all_directions:
    class_dist = Counter(all_directions)
    print(f"  Распределение (первые {len(all_directions)} семплов):")

    for cls in sorted(class_dist.keys()):
      count = class_dist[cls]
      pct = (count / len(all_directions)) * 100
      cls_name = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}.get(cls, str(cls))
      print(f"    {cls_name:8} ({cls:2d}): {count:6,} ({pct:5.1f}%)")

    # Imbalance ratio
    max_count = max(class_dist.values())
    min_count = min(class_dist.values())
    ratio = max_count / min_count

    print(f"\n  Imbalance Ratio: {ratio:.2f}")

    if ratio < 1.5:
      print(f"    ✓ Отличный баланс!")
    elif ratio < 3.0:
      print(f"    ✓ Хороший баланс")
    elif ratio < 5.0:
      print(f"    ⚠️  Средний дисбаланс - используйте Focal Loss")
    else:
      print(f"    ❌ Сильный дисбаланс - используйте Focal Loss + SMOTE")
      logger.info(f"Класс распределение: {dict(class_dist)}")
      logger.info(f"Imbalance ratio: {ratio:.2f}")

    total_time = time.time() - start_time
    logger.info(f"✓ Проверка данных завершена за {total_time:.1f}s")
    logger.info("=" * 80)
  print()
  return True


def test_dataloader(symbol: str):
  """Тест загрузки данных."""
  start_time = time.time()
  logger.info("=" * 80)
  logger.info("НАЧАЛО ТЕСТА DATALOADER")
  logger.info("=" * 80)
  print("=" * 60)
  print("ТЕСТ DATALOADER")
  print("=" * 60)

  try:
    logger.info("Импорт модулей...")
    from ml_engine.training.data_loader import HistoricalDataLoader, DataConfig

    print("Создание DataLoader...")
    logger.info("Создание конфигурации DataLoader...")
    config = DataConfig(
      storage_path="data/ml_training",
      sequence_length=60,
      batch_size=32
    )

    loader = HistoricalDataLoader(config)
    print("✓ DataLoader создан")

    print(f"\nЗагрузка данных для {symbol}...")
    X, y, timestamps = loader.load_symbol_data(symbol)

    print(f"✓ Данные загружены:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  timestamps shape: {timestamps.shape}")
    print(f"  Unique labels: {set(y)}")

    # Проверка на NaN/Inf
    has_nan = np.isnan(X).any()
    has_inf = np.isinf(X).any()

    if has_nan:
      print(f"  ❌ Features содержат NaN!")
      return False

    if has_inf:
      print(f"  ❌ Features содержат Inf!")
      return False

    print(f"  ✓ Нет NaN/Inf значений")

    # Создание sequences
    print(f"\nСоздание sequences...")
    sequences, seq_labels, seq_timestamps = loader.create_sequences(
      X, y, timestamps
    )

    print(f"✓ Sequences созданы:")
    print(f"  Shape: {sequences.shape}")
    print(f"  Expected: (N, 60, 110)")

    print()
    return True

  except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    return False


def test_model_creation():
  """Тест создания модели."""
  print("=" * 60)
  print("ТЕСТ СОЗДАНИЯ МОДЕЛИ")
  print("=" * 60)

  try:
    from ml_engine.models.hybrid_cnn_lstm import create_model

    print("Создание модели...")
    model = create_model()
    print("✓ Модель создана")

    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print()
    return True

  except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    return False


def test_minimal_training(symbol: str):
  """Минимальный тест обучения."""
  print("=" * 60)
  print("МИНИМАЛЬНЫЙ ТЕСТ ОБУЧЕНИЯ (3 эпохи)")
  print("=" * 60)

  try:
    from ml_engine.training.data_loader import HistoricalDataLoader, DataConfig
    from ml_engine.training.model_trainer import ModelTrainer, TrainerConfig
    from ml_engine.models.hybrid_cnn_lstm import create_model

    print("1. Загрузка данных...")
    config = DataConfig(
      storage_path="data/ml_training",
      sequence_length=60,
      batch_size=32
    )

    loader = HistoricalDataLoader(config)
    result = loader.load_and_prepare(symbols=[symbol])

    print(f"✓ Train: {len(result['dataloaders']['train'])} batches")
    print(f"✓ Val: {len(result['dataloaders']['val'])} batches")

    print("\n2. Создание модели...")
    model = create_model()
    print("✓ Модель создана")

    print("\n3. Обучение (3 эпохи на CPU)...")
    trainer_config = TrainerConfig(
      epochs=3,
      learning_rate=0.001,
      device="cpu",  # CPU для стабильности
      early_stopping_patience=10
    )

    trainer = ModelTrainer(model, trainer_config)
    history = trainer.train(
      result['dataloaders']['train'],
      result['dataloaders']['val']
    )

    print(f"\n✓ Обучение завершено: {len(history)} эпох")

    if history:
      last_epoch = history[-1]
      print(f"  Final train_loss: {last_epoch.train_loss:.4f}")
      print(f"  Final val_loss: {last_epoch.val_loss:.4f}")
      print(f"  Final val_accuracy: {last_epoch.val_accuracy:.4f}")

    print()
    return True

  except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    return False


def main():
  """Главная функция."""
  parser = argparse.ArgumentParser(
    description="Диагностика готовности к оптимизации"
  )
  parser.add_argument(
    "--symbol",
    required=True,
    help="Торговая пара (например, BTCUSDT)"
  )
  parser.add_argument(
    "--skip-training-test",
    action="store_true",
    help="Пропустить тест обучения"
  )

  args = parser.parse_args()

  print("\n" + "=" * 60)
  print("ДИАГНОСТИКА СИСТЕМЫ ОПТИМИЗАЦИИ")
  print("=" * 60)
  print(f"Symbol: {args.symbol}")
  print("=" * 60)

  # Чеклист
  results = {
    "Окружение": False,
    "Данные": False,
    "DataLoader": False,
    "Модель": False,
    "Обучение": False
  }

  # 1. Окружение
  results["Окружение"] = check_environment()
  if not results["Окружение"]:
    print("\n❌ Проблемы с окружением! Установите зависимости.")
    sys.exit(1)

  # 2. Данные
  results["Данные"] = check_data(args.symbol)
  if not results["Данные"]:
    print("\n❌ Проблемы с данными! Соберите больше данных или запустите preprocessing.")
    sys.exit(1)

  # 3. DataLoader
  results["DataLoader"] = test_dataloader(args.symbol)
  if not results["DataLoader"]:
    print("\n❌ Проблемы с DataLoader! Проверьте структуру данных.")
    sys.exit(1)

  # 4. Модель
  results["Модель"] = test_model_creation()
  if not results["Модель"]:
    print("\n❌ Проблемы с созданием модели!")
    sys.exit(1)

  # 5. Обучение (опционально)
  if not args.skip_training_test:
    results["Обучение"] = test_minimal_training(args.symbol)
    if not results["Обучение"]:
      print("\n❌ Проблемы с обучением!")
      print("   Возможно недостаточно памяти или данных.")
      sys.exit(1)
  else:
    print("\n⊘ Тест обучения пропущен")

  # Итоговый отчет
  print("\n" + "=" * 60)
  print("ИТОГОВЫЙ ОТЧЕТ")
  print("=" * 60)

  for test_name, passed in results.items():
    status = "✓" if passed else "❌"
    print(f"{status} {test_name}")

  if all(results.values()):
    print("\n🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
    print("\nМожете запускать оптимизацию:")
    print(f"  python hyperparameter_optimizer.py --symbol {args.symbol} --quick")
  else:
    print("\n❌ ЕСТЬ ПРОБЛЕМЫ!")
    print("   Исправьте ошибки перед запуском оптимизации.")

  print("=" * 60 + "\n")


if __name__ == "__main__":
  main()