#!/usr/bin/env python3
"""
Анализ распределения собранных данных.

Проверяет:
- Временное покрытие (gaps, непрерывность)
- Распределение классов (class imbalance)
- Статистика по рыночным условиям
- Рекомендации по улучшению
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import matplotlib.pyplot as plt


class DataDistributionAnalyzer:
  """Анализатор распределения данных."""

  def __init__(self, data_dir: str = "data/ml_training"):
    self.data_dir = Path(data_dir)

  def analyze_symbol(self, symbol: str) -> Dict:
    """Полный анализ данных для символа."""
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
    print(f"АНАЛИЗ РАСПРЕДЕЛЕНИЯ ДАННЫХ: {symbol}")
    print(f"{'=' * 80}\n")

    # Выполняем различные анализы
    results = {
      "symbol": symbol,
      "total_samples": len(all_labels),
      "temporal": self._analyze_temporal(all_labels),
      "class_distribution": self._analyze_class_distribution(all_labels),
      "market_conditions": self._analyze_market_conditions(all_labels),
      "signal_quality": self._analyze_signal_quality(all_labels),
      "data_quality": self._analyze_data_quality(all_labels)
    }

    # Печать отчета
    self._print_report(results)

    # Рекомендации
    self._print_recommendations(results)

    return results

  def _analyze_temporal(self, labels: List[Dict]) -> Dict:
    """Анализ временного распределения."""
    timestamps = [
      label.get("timestamp", 0)
      for label in labels
      if label.get("timestamp")
    ]

    if not timestamps:
      return {"error": "Нет валидных timestamps"}

    timestamps = sorted(timestamps)

    # Конвертация в datetime
    start_time = datetime.fromtimestamp(timestamps[0] / 1000)
    end_time = datetime.fromtimestamp(timestamps[-1] / 1000)
    duration = end_time - start_time

    # Поиск gaps (пропусков > 1 минуты)
    gaps = []
    for i in range(1, len(timestamps)):
      gap_ms = timestamps[i] - timestamps[i - 1]
      if gap_ms > 60_000:  # > 1 минута
        gap_start = datetime.fromtimestamp(timestamps[i - 1] / 1000)
        gap_end = datetime.fromtimestamp(timestamps[i] / 1000)
        gap_duration = gap_end - gap_start
        gaps.append({
          "start": gap_start.isoformat(),
          "end": gap_end.isoformat(),
          "duration_minutes": gap_duration.total_seconds() / 60
        })

    # Средний интервал между семплами
    intervals = np.diff(timestamps) / 1000  # в секундах
    avg_interval = np.mean(intervals)

    return {
      "start_time": start_time.isoformat(),
      "end_time": end_time.isoformat(),
      "duration_days": duration.total_seconds() / 86400,
      "total_samples": len(timestamps),
      "avg_interval_seconds": float(avg_interval),
      "gaps_count": len(gaps),
      "gaps": gaps[:10]  # Показываем первые 10
    }

  def _analyze_class_distribution(self, labels: List[Dict]) -> Dict:
    """Анализ распределения классов."""
    # Проверяем наличие future labels
    has_future = any("future_direction_10s" in label for label in labels)

    if not has_future:
      return {
        "error": "Нет future labels - требуется preprocessing",
        "samples_without_labels": len(labels)
      }

    # Считаем распределение для разных горизонтов
    distributions = {}

    for horizon in ["10s", "30s", "60s"]:
      field = f"future_direction_{horizon}"

      directions = [
        label.get(field)
        for label in labels
        if field in label and label.get(field) is not None
      ]

      if directions:
        counter = Counter(directions)
        total = len(directions)

        distributions[horizon] = {
          "total": total,
          "SELL (-1)": counter.get(-1, 0),
          "HOLD (0)": counter.get(0, 0),
          "BUY (1)": counter.get(1, 0),
          "percentages": {
            "SELL": (counter.get(-1, 0) / total) * 100,
            "HOLD": (counter.get(0, 0) / total) * 100,
            "BUY": (counter.get(1, 0) / total) * 100
          },
          "imbalance_ratio": max(counter.values()) / min(counter.values()) if min(counter.values()) > 0 else float(
            'inf')
        }

    return distributions

  def _analyze_market_conditions(self, labels: List[Dict]) -> Dict:
    """Анализ рыночных условий."""
    mid_prices = [
      label.get("current_mid_price")
      for label in labels
      if label.get("current_mid_price")
    ]

    spreads = [
      label.get("current_spread")
      for label in labels
      if label.get("current_spread")
    ]

    imbalances = [
      label.get("current_imbalance")
      for label in labels
      if label.get("current_imbalance")
    ]

    result = {}

    if mid_prices:
      mid_prices = np.array(mid_prices)
      returns = np.diff(mid_prices) / mid_prices[:-1]

      result["price"] = {
        "min": float(np.min(mid_prices)),
        "max": float(np.max(mid_prices)),
        "mean": float(np.mean(mid_prices)),
        "std": float(np.std(mid_prices)),
        "range_pct": ((np.max(mid_prices) - np.min(mid_prices)) / np.mean(mid_prices)) * 100
      }

      result["returns"] = {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns))
      }

    if spreads:
      spreads = np.array(spreads)
      result["spread"] = {
        "mean": float(np.mean(spreads)),
        "median": float(np.median(spreads)),
        "min": float(np.min(spreads)),
        "max": float(np.max(spreads))
      }

    if imbalances:
      imbalances = np.array(imbalances)
      result["imbalance"] = {
        "mean": float(np.mean(imbalances)),
        "median": float(np.median(imbalances)),
        "std": float(np.std(imbalances))
      }

    return result

  def _analyze_signal_quality(self, labels: List[Dict]) -> Dict:
    """Анализ качества сигналов."""
    signals_with_type = sum(
      1 for label in labels
      if label.get("signal_type") is not None
    )

    signal_types = Counter(
      label.get("signal_type")
      for label in labels
      if label.get("signal_type")
    )

    confidences = [
      label.get("signal_confidence")
      for label in labels
      if label.get("signal_confidence") is not None
    ]

    return {
      "total_samples": len(labels),
      "samples_with_signals": signals_with_type,
      "samples_without_signals": len(labels) - signals_with_type,
      "signal_types": dict(signal_types),
      "avg_confidence": float(np.mean(confidences)) if confidences else None,
      "coverage_pct": (signals_with_type / len(labels)) * 100
    }

  def _analyze_data_quality(self, labels: List[Dict]) -> Dict:
    """Анализ качества данных."""
    missing_timestamp = sum(
      1 for label in labels
      if not label.get("timestamp") or label.get("timestamp") == 0
    )

    missing_price = sum(
      1 for label in labels
      if not label.get("current_mid_price")
    )

    missing_future = sum(
      1 for label in labels
      if "future_direction_10s" not in label
    )

    return {
      "total_samples": len(labels),
      "missing_timestamp": missing_timestamp,
      "missing_price": missing_price,
      "missing_future_labels": missing_future,
      "quality_score": ((len(labels) - missing_timestamp - missing_price) / len(labels)) * 100
    }

  def _print_report(self, results: Dict):
    """Печать отчета анализа."""
    symbol = results["symbol"]
    total = results["total_samples"]

    print(f"📊 ОБЩАЯ СТАТИСТИКА")
    print(f"  • Всего семплов: {total:,}\n")

    # Временной анализ
    temporal = results.get("temporal", {})
    if "error" not in temporal:
      print(f"⏰ ВРЕМЕННОЕ ПОКРЫТИЕ")
      print(f"  • Период: {temporal['start_time']} - {temporal['end_time']}")
      print(f"  • Длительность: {temporal['duration_days']:.2f} дней")
      print(f"  • Средний интервал: {temporal['avg_interval_seconds']:.2f} сек")

      if temporal['gaps_count'] > 0:
        print(f"  ⚠️  Обнаружено пропусков: {temporal['gaps_count']}")
        if temporal['gaps']:
          print(f"  • Первые gaps:")
          for gap in temporal['gaps'][:3]:
            print(f"    - {gap['start']} ({gap['duration_minutes']:.1f} мин)")
      else:
        print(f"  ✅ Пропусков не обнаружено")
      print()

    # Распределение классов
    class_dist = results.get("class_distribution", {})
    if "error" not in class_dist:
      print(f"📈 РАСПРЕДЕЛЕНИЕ КЛАССОВ (Future Direction)")

      for horizon, dist in class_dist.items():
        print(f"\n  Горизонт {horizon}:")
        print(f"    • SELL (-1): {dist['SELL (-1)']:,} ({dist['percentages']['SELL']:.1f}%)")
        print(f"    • HOLD (0):  {dist['HOLD (0)']:,} ({dist['percentages']['HOLD']:.1f}%)")
        print(f"    • BUY (1):   {dist['BUY (1)']:,} ({dist['percentages']['BUY']:.1f}%)")

        if dist['imbalance_ratio'] > 2.0:
          print(f"    ⚠️  Дисбаланс классов: {dist['imbalance_ratio']:.2f}x")
        else:
          print(f"    ✅ Баланс классов приемлемый")
    else:
      print(f"❌ {class_dist['error']}")

    print()

    # Рыночные условия
    market = results.get("market_conditions", {})
    if market:
      print(f"💹 РЫНОЧНЫЕ УСЛОВИЯ")
      if "price" in market:
        p = market["price"]
        print(f"  • Цена: {p['min']:.2f} - {p['max']:.2f} (диапазон: {p['range_pct']:.2f}%)")
      if "spread" in market:
        s = market["spread"]
        print(f"  • Спред: mean={s['mean']:.4f}, median={s['median']:.4f}")
      print()

    # Качество сигналов
    signals = results.get("signal_quality", {})
    if signals:
      print(f"🎯 КАЧЕСТВО СИГНАЛОВ")
      print(f"  • Семплов с сигналами: {signals['samples_with_signals']:,} ({signals['coverage_pct']:.1f}%)")
      if signals['signal_types']:
        print(f"  • Типы сигналов: {signals['signal_types']}")
      if signals['avg_confidence']:
        print(f"  • Средняя уверенность: {signals['avg_confidence']:.3f}")
      print()

    # Качество данных
    quality = results.get("data_quality", {})
    if quality:
      print(f"✅ КАЧЕСТВО ДАННЫХ")
      print(f"  • Score: {quality['quality_score']:.1f}%")
      if quality['missing_timestamp'] > 0:
        print(f"  ⚠️  Отсутствует timestamp: {quality['missing_timestamp']:,}")
      if quality['missing_future_labels'] > 0:
        print(f"  ⚠️  Отсутствуют future labels: {quality['missing_future_labels']:,}")
      print()

  def _print_recommendations(self, results: Dict):
    """Печать рекомендаций."""
    print(f"{'=' * 80}")
    print(f"💡 РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ")
    print(f"{'=' * 80}\n")

    recommendations = []

    # Проверка объема данных
    total = results["total_samples"]
    if total < 100_000:
      recommendations.append(
        f"📊 КРИТИЧНО: Недостаточно данных ({total:,} < 100,000)\n"
        f"   → Продолжите сбор минимум до 1,000,000 семплов (~2 недели)"
      )
    elif total < 1_000_000:
      recommendations.append(
        f"⚠️  Данных маловато для продакшн ({total:,} < 1,000,000)\n"
        f"   → Рекомендуется собрать 5,000,000+ семплов (~1-2 месяца)"
      )

    # Проверка gaps
    temporal = results.get("temporal", {})
    if "error" not in temporal and temporal.get("gaps_count", 0) > 10:
      recommendations.append(
        f"⚠️  Много пропусков в данных ({temporal['gaps_count']})\n"
        f"   → Проверьте стабильность сбора данных\n"
        f"   → Рассмотрите заполнение gaps (forward fill)"
      )

    # Проверка class imbalance
    class_dist = results.get("class_distribution", {})
    if "error" not in class_dist:
      for horizon, dist in class_dist.items():
        if dist['imbalance_ratio'] > 3.0:
          recommendations.append(
            f"⚠️  Сильный дисбаланс классов для {horizon} ({dist['imbalance_ratio']:.1f}x)\n"
            f"   → Используйте class weights при обучении\n"
            f"   → Рассмотрите oversampling/undersampling\n"
            f"   → Попробуйте focal loss"
          )

    # Проверка future labels
    quality = results.get("data_quality", {})
    if quality.get("missing_future_labels", 0) > 0:
      recommendations.append(
        f"❌ КРИТИЧНО: Отсутствуют future labels\n"
        f"   → Запустите: python preprocessing_add_future_labels.py"
      )

    # Проверка сигналов
    signals = results.get("signal_quality", {})
    if signals and signals.get("coverage_pct", 0) < 5:
      recommendations.append(
        f"💡 Почти нет торговых сигналов ({signals['coverage_pct']:.1f}%)\n"
        f"   → Это нормально если модель предсказывает direction\n"
        f"   → Но полезно сохранять реальные сигналы для анализа"
      )

    # Вывод рекомендаций
    if recommendations:
      for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}\n")
    else:
      print("✅ Данные выглядят хорошо! Готовы к обучению.\n")

    print(f"{'=' * 80}\n")


def main():
  """Основная функция."""
  import argparse

  parser = argparse.ArgumentParser(
    description="Анализ распределения данных для ML"
  )
  parser.add_argument(
    "--symbol",
    required=True,
    help="Торговая пара (например, BTCUSDT)"
  )
  parser.add_argument(
    "--data-dir",
    default="data/ml_training",
    help="Директория с данными"
  )

  args = parser.parse_args()

  analyzer = DataDistributionAnalyzer(data_dir=args.data_dir)
  analyzer.analyze_symbol(args.symbol)


if __name__ == "__main__":
  main()