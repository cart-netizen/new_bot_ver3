#!/usr/bin/env python3
"""
Инструмент для анализа и визуализации future direction labels.

Функциональность:
- Анализ существующих labels
- Симуляция различных порогов
- Визуализация распределения классов
- Рекомендации по оптимизации

Использование:
    python analyze_future_direction.py --symbol BTCUSDT
    python analyze_future_direction.py --symbol BTCUSDT --threshold 0.0005
    python analyze_future_direction.py --symbol BTCUSDT --visualize
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import argparse


class FutureDirectionAnalyzer:
  """Анализатор future direction labels."""

  def __init__(self, data_dir: str = "data/ml_training"):
    self.data_dir = Path(data_dir)

  def analyze_symbol(
      self,
      symbol: str,
      custom_threshold: float = None
  ) -> Dict:
    """
    Анализ future direction для символа.

    Args:
        symbol: Торговая пара
        custom_threshold: Пользовательский порог (если нужно пересчитать)

    Returns:
        Dict с результатами анализа
    """
    symbol_path = self.data_dir / symbol
    labels_dir = symbol_path / "labels"

    if not labels_dir.exists():
      print(f"❌ Директория {labels_dir} не найдена")
      return {}

    # Загрузка всех labels
    all_labels = []
    label_files = sorted(labels_dir.glob("*.json"))

    for label_file in label_files:
      with open(label_file) as f:
        all_labels.extend(json.load(f))

    if not all_labels:
      print(f"❌ Нет данных для {symbol}")
      return {}

    print(f"\n{'=' * 80}")
    print(f"АНАЛИЗ FUTURE DIRECTION: {symbol}")
    print(f"{'=' * 80}\n")

    results = {
      "symbol": symbol,
      "total_samples": len(all_labels),
      "horizons": {}
    }

    # Анализ для каждого горизонта
    for horizon in ["10s", "30s", "60s"]:
      print(f"\n{'─' * 80}")
      print(f"ГОРИЗОНТ: {horizon}")
      print(f"{'─' * 80}")

      horizon_data = self._analyze_horizon(
        all_labels,
        horizon,
        custom_threshold
      )
      results["horizons"][horizon] = horizon_data

      self._print_horizon_report(horizon_data, horizon)

    # Общие рекомендации
    self._print_recommendations(results)

    return results

  def _analyze_horizon(
      self,
      labels: List[Dict],
      horizon: str,
      custom_threshold: float = None
  ) -> Dict:
    """Анализ одного горизонта."""
    direction_field = f"future_direction_{horizon}"
    movement_field = f"future_movement_{horizon}"

    # Текущие labels (если есть)
    current_directions = []
    current_movements = []

    for label in labels:
      direction = label.get(direction_field)
      movement = label.get(movement_field)

      if direction is not None:
        current_directions.append(direction)
      if movement is not None:
        current_movements.append(movement)

    result = {
      "total_samples": len(labels),
      "samples_with_labels": len(current_directions),
      "samples_without_labels": len(labels) - len(current_directions)
    }

    # Если есть labels, анализируем
    if current_directions:
      result["current"] = self._calculate_distribution(
        current_directions,
        current_movements
      )

    # Если указан custom threshold, пересчитываем
    if custom_threshold is not None and current_movements:
      result["simulated"] = self._simulate_threshold(
        current_movements,
        custom_threshold
      )

    # Анализ движений (если есть)
    if current_movements:
      movements = np.array(current_movements)
      result["movement_stats"] = {
        "mean": float(np.mean(movements)),
        "std": float(np.std(movements)),
        "min": float(np.min(movements)),
        "max": float(np.max(movements)),
        "median": float(np.median(movements)),
        "percentile_95": float(np.percentile(movements, 95)),
        "percentile_5": float(np.percentile(movements, 5))
      }

    return result

  def _calculate_distribution(
      self,
      directions: List[int],
      movements: List[float] = None
  ) -> Dict:
    """Расчет распределения классов."""
    counter = Counter(directions)
    total = len(directions)

    distribution = {
      "UP": counter.get(1, 0),
      "NEUTRAL": counter.get(0, 0),
      "DOWN": counter.get(-1, 0),
      "percentages": {
        "UP": (counter.get(1, 0) / total) * 100,
        "NEUTRAL": (counter.get(0, 0) / total) * 100,
        "DOWN": (counter.get(-1, 0) / total) * 100
      }
    }

    # Imbalance ratio
    counts = [counter.get(c, 0) for c in [-1, 0, 1]]
    max_count = max(counts)
    min_count = min(c for c in counts if c > 0) if any(c > 0 for c in counts) else 1

    distribution["imbalance_ratio"] = max_count / min_count

    # Оценка качества
    if distribution["imbalance_ratio"] < 1.5:
      distribution["balance_quality"] = "Отлично"
    elif distribution["imbalance_ratio"] < 2.5:
      distribution["balance_quality"] = "Хорошо"
    elif distribution["imbalance_ratio"] < 4.0:
      distribution["balance_quality"] = "Приемлемо"
    else:
      distribution["balance_quality"] = "Плохо"

    return distribution

  def _simulate_threshold(
      self,
      movements: List[float],
      threshold: float
  ) -> Dict:
    """Симуляция с другим порогом."""
    simulated_directions = []

    for movement in movements:
      if movement > threshold:
        simulated_directions.append(1)
      elif movement < -threshold:
        simulated_directions.append(-1)
      else:
        simulated_directions.append(0)

    return self._calculate_distribution(simulated_directions, movements)

  def _print_horizon_report(self, data: Dict, horizon: str):
    """Печать отчета по горизонту."""
    total = data["total_samples"]
    with_labels = data["samples_with_labels"]
    without = data["samples_without_labels"]

    print(f"\n📊 Общая статистика:")
    print(f"  • Всего семплов: {total:,}")
    print(f"  • С labels: {with_labels:,} ({(with_labels / total) * 100:.1f}%)")

    if without > 0:
      print(f"  ⚠️  Без labels: {without:,} ({(without / total) * 100:.1f}%)")

    # Текущее распределение
    if "current" in data:
      current = data["current"]
      print(f"\n📈 Текущее распределение (threshold = 0.1%):")
      self._print_distribution(current)

    # Симулированное распределение
    if "simulated" in data:
      simulated = data["simulated"]
      print(f"\n🔄 Симулированное распределение (custom threshold):")
      self._print_distribution(simulated)

    # Статистика движений
    if "movement_stats" in data:
      stats = data["movement_stats"]
      print(f"\n📉 Статистика движений цены:")
      print(f"  • Mean:   {stats['mean'] * 100:+.3f}%")
      print(f"  • Median: {stats['median'] * 100:+.3f}%")
      print(f"  • Std:    {stats['std'] * 100:.3f}%")
      print(f"  • Min:    {stats['min'] * 100:+.3f}%")
      print(f"  • Max:    {stats['max'] * 100:+.3f}%")
      print(f"  • 95%ile: {stats['percentile_95'] * 100:+.3f}%")
      print(f"  • 5%ile:  {stats['percentile_5'] * 100:+.3f}%")

  def _print_distribution(self, dist: Dict):
    """Печать распределения."""
    up = dist["UP"]
    neutral = dist["NEUTRAL"]
    down = dist["DOWN"]

    up_pct = dist["percentages"]["UP"]
    neutral_pct = dist["percentages"]["NEUTRAL"]
    down_pct = dist["percentages"]["DOWN"]

    # Визуальные бары
    bar_length = 40
    up_bar = "█" * int(up_pct / 100 * bar_length)
    neutral_bar = "█" * int(neutral_pct / 100 * bar_length)
    down_bar = "█" * int(down_pct / 100 * bar_length)

    print(f"  • UP (1):      {up:>8,} ({up_pct:5.1f}%)  {up_bar}")
    print(f"  • NEUTRAL (0): {neutral:>8,} ({neutral_pct:5.1f}%)  {neutral_bar}")
    print(f"  • DOWN (-1):   {down:>8,} ({down_pct:5.1f}%)  {down_bar}")
    print(f"\n  • Imbalance Ratio: {dist['imbalance_ratio']:.2f}")

    # Оценка
    quality = dist["balance_quality"]
    emoji = "✅" if quality in ["Отлично", "Хорошо"] else "⚠️"
    print(f"  • Качество баланса: {emoji} {quality}")

  def _print_recommendations(self, results: Dict):
    """Печать рекомендаций."""
    print(f"\n{'=' * 80}")
    print(f"💡 РЕКОМЕНДАЦИИ")
    print(f"{'=' * 80}\n")

    recommendations = []

    # Проверка наличия labels
    for horizon, data in results["horizons"].items():
      without = data["samples_without_labels"]
      total = data["total_samples"]

      if without > 0:
        pct = (without / total) * 100
        recommendations.append(
          f"❌ {horizon}: {pct:.1f}% семплов без labels\n"
          f"   → Запустите: python preprocessing_add_future_labels.py"
        )

    # Проверка баланса классов
    for horizon, data in results["horizons"].items():
      if "current" in data:
        dist = data["current"]
        ratio = dist["imbalance_ratio"]

        if ratio > 3.0:
          recommendations.append(
            f"⚠️  {horizon}: Сильный дисбаланс классов ({ratio:.2f}x)\n"
            f"   → Рекомендации:\n"
            f"     1. Попробуйте другой порог (см. ниже)\n"
            f"     2. Используйте class_weights при обучении\n"
            f"     3. Примените SMOTE/oversampling"
          )

    # Рекомендации по порогу
    for horizon, data in results["horizons"].items():
      if "movement_stats" in data:
        stats = data["movement_stats"]
        std = stats["std"]

        # Рекомендуемый порог = 0.5 * std
        recommended_threshold = std * 0.5

        recommendations.append(
          f"💡 {horizon}: Рекомендуемый порог на основе волатильности\n"
          f"   → Текущий: 0.1% (0.001)\n"
          f"   → Рекомендуемый: {recommended_threshold * 100:.2f}% ({recommended_threshold:.5f})\n"
          f"   → Протестируйте: python analyze_future_direction.py "
          f"--symbol {results['symbol']} --threshold {recommended_threshold:.5f}"
        )

    # Вывод рекомендаций
    if recommendations:
      for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}\n")
    else:
      print("✅ Данные выглядят хорошо!\n")

  def test_multiple_thresholds(
      self,
      symbol: str,
      horizon: str = "60s"
  ):
    """Тестирование различных порогов."""
    symbol_path = self.data_dir / symbol
    labels_dir = symbol_path / "labels"

    if not labels_dir.exists():
      print(f"❌ Директория {labels_dir} не найдена")
      return

    # Загрузка movements
    all_movements = []
    label_files = sorted(labels_dir.glob("*.json"))

    for label_file in label_files:
      with open(label_file) as f:
        labels = json.load(f)

      movement_field = f"future_movement_{horizon}"
      for label in labels:
        movement = label.get(movement_field)
        if movement is not None:
          all_movements.append(movement)

    if not all_movements:
      print(f"❌ Нет данных о движениях для {symbol}")
      return

    print(f"\n{'=' * 80}")
    print(f"ТЕСТИРОВАНИЕ ПОРОГОВ: {symbol} ({horizon})")
    print(f"{'=' * 80}\n")
    print(f"Всего семплов: {len(all_movements):,}\n")

    # Тестируем различные пороги
    thresholds = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003]

    print(f"{'Порог':>10} {'UP':>10} {'NEUTRAL':>10} {'DOWN':>10} {'Ratio':>8} {'Качество':>12}")
    print(f"{'─' * 70}")

    for threshold in thresholds:
      dist = self._simulate_threshold(all_movements, threshold)

      up_pct = dist["percentages"]["UP"]
      neutral_pct = dist["percentages"]["NEUTRAL"]
      down_pct = dist["percentages"]["DOWN"]
      ratio = dist["imbalance_ratio"]
      quality = dist["balance_quality"]

      print(
        f"{threshold * 100:>9.2f}% "
        f"{up_pct:>9.1f}% "
        f"{neutral_pct:>9.1f}% "
        f"{down_pct:>9.1f}% "
        f"{ratio:>7.2f}x "
        f"{quality:>12}"
      )

    print(f"\n💡 Выберите порог с балансом 'Отлично' или 'Хорошо'\n")


def main():
  """Основная функция."""
  parser = argparse.ArgumentParser(
    description="Анализ future direction labels"
  )
  parser.add_argument(
    "--symbol",
    required=True,
    help="Торговая пара (например, BTCUSDT)"
  )
  parser.add_argument(
    "--threshold",
    type=float,
    help="Пользовательский порог для симуляции (например, 0.0005)"
  )
  parser.add_argument(
    "--test-thresholds",
    action="store_true",
    help="Протестировать различные пороги"
  )
  parser.add_argument(
    "--horizon",
    default="60s",
    choices=["10s", "30s", "60s"],
    help="Горизонт для тестирования порогов"
  )
  parser.add_argument(
    "--data-dir",
    default="data/ml_training",
    help="Директория с данными"
  )

  args = parser.parse_args()

  analyzer = FutureDirectionAnalyzer(data_dir=args.data_dir)

  if args.test_thresholds:
    analyzer.test_multiple_thresholds(args.symbol, args.horizon)
  else:
    analyzer.analyze_symbol(args.symbol, args.threshold)


if __name__ == "__main__":
  main()