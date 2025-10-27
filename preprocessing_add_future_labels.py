#!/usr/bin/env python3
"""
Preprocessing скрипт для добавления future labels к собранным данным.

Запускается ПОСЛЕ сбора данных, перед обучением модели.
Добавляет метки о будущем движении цены (через 10s, 30s, 60s).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class FutureLabelProcessor:
  """
  Обрабатывает собранные данные и добавляет future labels.
  """

  def __init__(self, data_dir: str = "data/ml_training"):
    self.data_dir = Path(data_dir)

  def process_symbol(self, symbol: str):
    """
    Обработка всех батчей для одного символа.

    Args:
        symbol: Торговая пара (например, BTCUSDT)
    """
    symbol_dir = self.data_dir / symbol
    labels_dir = symbol_dir / "labels"

    print(f"\n{'=' * 70}")
    print(f"Обработка {symbol}")
    print(f"{'=' * 70}")

    # Загружаем все labels файлы
    label_files = sorted(labels_dir.glob("*.json"))

    for label_file in label_files:
      print(f"\nОбработка {label_file.name}...")
      self._process_batch(label_file)

  def _process_batch(self, label_file: Path):
    """
    Обработка одного batch файла.

    Args:
        label_file: Путь к labels.json файлу
    """
    # Загружаем labels
    with open(label_file, 'r') as f:
      labels = json.load(f)

    print(f"  Загружено {len(labels)} семплов")

    # Пытаемся загрузить metadata (для старых данных, где timestamp в metadata)
    metadata_file = label_file.parent.parent / "metadata" / label_file.name
    metadata_samples = []


    if metadata_file.exists():
      try:
        with open(metadata_file, 'r') as f:
          metadata = json.load(f)
          metadata_samples = metadata.get("samples", [])
          print(f"  ✓ Загружен metadata файл ({len(metadata_samples)} записей)")
      except Exception as e:
        print(f"  ⚠️  Ошибка загрузки metadata: {e}")

    # Для каждого семпла рассчитываем future targets
    updated_labels = []
    timestamp_from_metadata_count = 0

    for i, label in enumerate(labels):
      # Текущая цена и timestamp
      current_price = label["current_mid_price"]
      current_timestamp = label.get("timestamp")  # Новые данные

      # Если timestamp нет в label, пытаемся взять из metadata (старые данные)
      if current_timestamp is None and i < len(metadata_samples):
        current_timestamp = metadata_samples[i].get("timestamp")
        if current_timestamp is not None:
          # Сохраняем timestamp в label для будущего использования
          label["timestamp"] = current_timestamp
          timestamp_from_metadata_count += 1

      if current_timestamp is None:
        # Если timestamp не найден нигде, пропускаем
        updated_labels.append(label)
        continue

      # Ищем цены через 10s, 30s, 60s
      future_10s = self._find_future_price(
        labels, metadata_samples, i, current_timestamp, delta_seconds=10
      )
      future_30s = self._find_future_price(
        labels, metadata_samples, i, current_timestamp, delta_seconds=30
      )
      future_60s = self._find_future_price(
        labels, metadata_samples, i, current_timestamp, delta_seconds=60
      )

      # Обновляем label с future targets
      if future_10s:
        label["future_movement_10s"] = self._calculate_movement(
          current_price, future_10s
        )
        label["future_direction_10s"] = self._calculate_direction(
          current_price, future_10s
        )

      if future_30s:
        label["future_movement_30s"] = self._calculate_movement(
          current_price, future_30s
        )
        label["future_direction_30s"] = self._calculate_direction(
          current_price, future_30s
        )

      if future_60s:
        label["future_movement_60s"] = self._calculate_movement(
          current_price, future_60s
        )
        label["future_direction_60s"] = self._calculate_direction(
          current_price, future_60s
        )

      updated_labels.append(label)

    # Сохраняем обновленные labels
    with open(label_file, 'w') as f:
      json.dump(updated_labels, f, indent=2)

    # Статистика
    filled = sum(1 for l in updated_labels if l["future_direction_10s"] is not None)
    print(f"  ✓ Обновлено {filled}/{len(labels)} семплов с future labels")

    if timestamp_from_metadata_count > 0:
      print(f"  ℹ️  Timestamp взят из metadata для {timestamp_from_metadata_count} семплов (старые данные)")

  def _find_future_price(
      self,
      labels: List[Dict],
      metadata_samples: List[Dict],
      current_idx: int,
      current_timestamp: int,
      delta_seconds: int
  ) -> float:
    """
    Находит цену через N секунд после текущего timestamp.

    Args:
        labels: Список всех labels
        metadata_samples: Список metadata (для старых данных)
        current_idx: Индекс текущего семпла
        current_timestamp: Текущий timestamp (ms)
        delta_seconds: Через сколько секунд искать цену

    Returns:
        float: Цена через N секунд или None
    """
    target_timestamp = current_timestamp + (delta_seconds * 1000)  # ms
    tolerance = 2000  # ±2 секунды

    # Ищем ближайший семпл к target_timestamp
    for i in range(current_idx + 1, len(labels)):
      future_label = labels[i]
      future_timestamp = future_label.get("timestamp")  # Новые данные

      # Если timestamp нет в label, берем из metadata (старые данные)
      if future_timestamp is None and i < len(metadata_samples):
        future_timestamp = metadata_samples[i].get("timestamp")

      if future_timestamp is None:
        continue

      # Если timestamp в пределах tolerance
      if abs(future_timestamp - target_timestamp) <= tolerance:
        return future_label["current_mid_price"]

      # Если ушли слишком далеко
      if future_timestamp > target_timestamp + tolerance:
        break

    return None

  def _calculate_movement(self, current_price: float, future_price: float) -> float:
    """
    Рассчитывает процентное изменение цены.

    Args:
        current_price: Текущая цена
        future_price: Будущая цена

    Returns:
        float: Процентное изменение (например, 0.05 = +5%)
    """
    return (future_price - current_price) / current_price

  def _calculate_direction(self, current_price: float, future_price: float) -> int:
    """
    Определяет направление движения цены.

    Args:
        current_price: Текущая цена
        future_price: Будущая цена

    Returns:
        int: 1=up, 0=neutral, -1=down
    """
    movement_pct = self._calculate_movement(current_price, future_price)
    threshold = 0.001  # 0.1% - порог для "neutral"

    if movement_pct > threshold:
      return 1  # UP
    elif movement_pct < -threshold:
      return -1  # DOWN
    else:
      return 0  # NEUTRAL

  def process_all_symbols(self):
    """Обработка всех символов в директории."""
    for symbol_dir in self.data_dir.iterdir():
      if symbol_dir.is_dir():
        self.process_symbol(symbol_dir.name)


def main():
  """Главная функция."""
  processor = FutureLabelProcessor()

  print("\n" + "=" * 70)
  print("PREPROCESSING: Добавление Future Labels")
  print("=" * 70)
  print("\nЭтот скрипт обрабатывает собранные данные и добавляет метки")
  print("о будущем движении цены (через 10s, 30s, 60s).")
  print("\nЗапускается ПОСЛЕ сбора данных, ПЕРЕД обучением модели.")
  print("=" * 70)

  # Обработка всех символов
  processor.process_all_symbols()

  print("\n" + "=" * 70)
  print("✓ Preprocessing завершен!")
  print("=" * 70)
  print("\nТеперь данные готовы для обучения ML модели.")


if __name__ == "__main__":
  main()