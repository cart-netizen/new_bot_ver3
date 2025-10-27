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
import argparse
import numpy as np
import json
from pathlib import Path
from collections import Counter
import sys


def check_environment():
  """Проверка окружения."""
  print("\n" + "=" * 60)
  print("ПРОВЕРКА ОКРУЖЕНИЯ")
  print("=" * 60)

  try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
      print(f"  CUDA version: {torch.version.cuda}")
      print(f"  GPU: {torch.cuda.get_device_name(0)}")
  except ImportError:
    print("❌ PyTorch не установлен!")
    return False

  try:
    import sklearn
    print(f"✓ Scikit-learn: {sklearn.__version__}")
  except ImportError:
    print("❌ Scikit-learn не установлен!")
    return False

  try:
    from imblearn import over_sampling
    print(f"✓ imbalanced-learn установлен")
  except ImportError:
    print("⚠️  imbalanced-learn не установлен (опционально)")

  print()
  return True


def check_data(symbol: str, data_path: str = "data/ml_training"):
  """Проверка данных."""
  print("=" * 60)
  print(f"ПРОВЕРКА ДАННЫХ: {symbol}")
  print("=" * 60)

  symbol_path = Path(data_path) / symbol

  if not symbol_path.exists():
    print(f"❌ Директория не найдена: {symbol_path}")
    return False

  features_dir = symbol_path / "features"
  labels_dir = symbol_path / "labels"

  if not features_dir.exists():
    print(f"❌ Директория features не найдена!")
    return False

  if not labels_dir.exists():
    print(f"❌ Директория labels не найдена!")
    return False

  # Подсчет файлов
  feature_files = list(features_dir.glob("*.npy"))
  label_files = list(labels_dir.glob("*.json"))

  print(f"\nФайлы:")
  print(f"  Feature files: {len(feature_files)}")
  print(f"  Label files: {len(label_files)}")

  if len(feature_files) == 0:
    print("❌ Нет файлов features!")
    return False

  if len(label_files) == 0:
    print("❌ Нет файлов labels!")
    return False

  # Загрузка примеров
  print(f"\nЗагрузка примеров...")

  try:
    # Features
    sample_features = np.load(feature_files[0])
    print(f"  ✓ Features shape: {sample_features.shape}")
    print(f"    Expected: (N, 110)")

    if sample_features.shape[1] != 110:
      print(f"    ⚠️  Неожиданное количество признаков!")

    # Labels
    with open(label_files[0]) as f:
      sample_labels = json.load(f)

    print(f"  ✓ Labels loaded: {len(sample_labels)} samples")

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
  print(f"\nПодсчет семплов...")
  total_samples = 0

  for feature_file in feature_files:
    data = np.load(feature_file)
    total_samples += data.shape[0]

  print(f"  Всего семплов: {total_samples:,}")

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
  print(f"\nАнализ распределения классов...")

  all_directions = []
  for label_file in label_files[:3]:  # Первые 3 файла для примера
    with open(label_file) as f:
      labels = json.load(f)
      for label in labels:
        direction = label.get("future_direction_60s")
        if direction is not None:
          all_directions.append(direction)

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

  print()
  return True


def test_dataloader(symbol: str):
  """Тест загрузки данных."""
  print("=" * 60)
  print("ТЕСТ DATALOADER")
  print("=" * 60)

  try:
    from ml_engine.training.data_loader import HistoricalDataLoader, DataConfig

    print("Создание DataLoader...")
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