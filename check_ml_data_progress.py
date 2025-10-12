# backend/scripts/check_ml_data_progress.py
import sys
from pathlib import Path

# Определяем путь к backend директории
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def check_progress():
  """Проверка прогресса сбора данных."""

  target_samples_per_symbol = 5_184_000  # 1 месяц
  data_dir = Path("data/ml_training")

  print("=" * 70)
  print("ПРОГРЕСС СБОРА ML ДАННЫХ")
  print("=" * 70)

  total_samples = 0

  for symbol_dir in data_dir.iterdir():
    if not symbol_dir.is_dir():
      continue

    symbol = symbol_dir.name
    symbol_samples = 0

    for npy_file in symbol_dir.glob("*.npy"):
      data = np.load(npy_file)
      symbol_samples += data.shape[0]

    progress_pct = (symbol_samples / target_samples_per_symbol) * 100
    days_collected = symbol_samples / 172_800

    print(f"\n{symbol}:")
    print(f"  Семплов: {symbol_samples:,} / {target_samples_per_symbol:,}")
    print(f"  Прогресс: {progress_pct:.2f}%")
    print(f"  Дней собрано: {days_collected:.2f} / 30")

    total_samples += symbol_samples

  print("\n" + "=" * 70)
  print(f"ВСЕГО семплов: {total_samples:,}")
  print(f"Общий прогресс: {(total_samples / (target_samples_per_symbol * 3)) * 100:.2f}%")
  print("=" * 70)


if __name__ == "__main__":
  check_progress()