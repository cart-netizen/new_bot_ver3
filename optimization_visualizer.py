#!/usr/bin/env python3
"""
Visualization для результатов оптимизации гиперпараметров.

Создает графики и отчеты для анализа результатов.

Использование:
    python optimization_visualizer.py --results-dir optimization_results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List
import argparse

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class OptimizationVisualizer:
  """Визуализатор результатов оптимизации."""

  def __init__(self, results_dir: str):
    """Инициализация."""
    self.results_dir = Path(results_dir)

    if not self.results_dir.exists():
      raise FileNotFoundError(f"Директория не найдена: {results_dir}")

    # Загружаем сводку
    summary_path = self.results_dir / "optimization_summary.json"
    with open(summary_path) as f:
      self.summary = json.load(f)

    print(f"Загружена оптимизация для {len(self.summary)} символов")

  def create_all_visualizations(self):
    """Создание всех визуализаций."""
    output_dir = self.results_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)

    print("\nСоздание визуализаций...")

    # 1. Сравнение символов
    self.plot_symbol_comparison(output_dir)

    # 2. Для каждого символа
    for symbol in self.summary.keys():
      self.plot_symbol_details(symbol, output_dir)

    print(f"\n✓ Визуализации сохранены в: {output_dir}")

  def plot_symbol_comparison(self, output_dir: Path):
    """Сравнение оптимальных конфигураций между символами."""
    data = []

    for symbol, config in self.summary.items():
      perf = config['expected_performance']
      data.append({
        'Symbol': symbol,
        'F1-Score': perf['f1_weighted'],
        'Accuracy': perf['accuracy'],
        'Imbalance Ratio': perf['imbalance_ratio'],
        'Threshold': config['threshold'],
        'Method': config['balancing_method']
      })

    df = pd.DataFrame(data)

    # График 1: F1-Score и Accuracy
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # F1-Score
    ax1 = axes[0, 0]
    df.plot(x='Symbol', y='F1-Score', kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('F1-Score по символам', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1-Score')
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2 = axes[0, 1]
    df.plot(x='Symbol', y='Accuracy', kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title('Accuracy по символам', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)

    # Imbalance Ratio
    ax3 = axes[1, 0]
    df.plot(x='Symbol', y='Imbalance Ratio', kind='bar', ax=ax3, color='salmon')
    ax3.set_title('Imbalance Ratio по символам', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Imbalance Ratio')
    ax3.axhline(y=2.0, color='r', linestyle='--', label='Порог дисбаланса')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Threshold
    ax4 = axes[1, 1]
    df.plot(x='Symbol', y='Threshold', kind='bar', ax=ax4, color='gold')
    ax4.set_title('Оптимальный Threshold по символам', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Threshold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'symbol_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # График 2: Методы балансировки
    fig, ax = plt.subplots(figsize=(10, 6))
    method_counts = df['Method'].value_counts()
    method_counts.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Частота использования методов балансировки', fontsize=14, fontweight='bold')
    ax.set_xlabel('Количество символов')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'balancing_methods.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  ✓ Сравнение символов")

  def plot_symbol_details(self, symbol: str, output_dir: Path):
    """Детальные графики для одного символа."""
    symbol_dir = self.results_dir / symbol

    if not symbol_dir.exists():
      return

    # Загружаем результаты
    results_path = symbol_dir / "results.csv"
    if not results_path.exists():
      return

    df = pd.read_csv(results_path)

    # График: F1-Score vs Threshold
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # По threshold
    ax1 = axes[0]
    threshold_groups = df.groupby('threshold')['f1_weighted'].mean()
    threshold_groups.plot(kind='line', marker='o', ax=ax1)
    ax1.set_title(f'{symbol}: F1-Score vs Threshold', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('F1-Score')
    ax1.grid(True, alpha=0.3)

    # По методу
    ax2 = axes[1]
    method_groups = df.groupby('method')['f1_weighted'].mean().sort_values()
    method_groups.plot(kind='barh', ax=ax2, color='coral')
    ax2.set_title(f'{symbol}: F1-Score по методам', fontsize=14, fontweight='bold')
    ax2.set_xlabel('F1-Score')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{symbol}_details.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ {symbol}")

  def generate_report(self):
    """Генерация текстового отчета."""
    report_path = self.results_dir / "optimization_report.md"

    with open(report_path, 'w') as f:
      f.write("# Отчет по Оптимизации Гиперпараметров\n\n")
      f.write(f"Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

      f.write("## Сводка по Символам\n\n")

      for symbol, config in self.summary.items():
        perf = config['expected_performance']

        f.write(f"### {symbol}\n\n")
        f.write(f"**Оптимальные параметры:**\n")
        f.write(f"- Threshold: `{config['threshold']:.4f}`\n")
        f.write(f"- Метод: `{config['balancing_method']}`\n")

        if config['balancing_params']:
          f.write(f"- Параметры: {config['balancing_params']}\n")

        f.write(f"\n**Ожидаемая производительность:**\n")
        f.write(f"- F1-Score: `{perf['f1_weighted']:.4f}`\n")
        f.write(f"- Accuracy: `{perf['accuracy']:.4f}`\n")
        f.write(f"- Imbalance Ratio: `{perf['imbalance_ratio']:.2f}`\n\n")

        # Рекомендации
        f.write("**Рекомендации:**\n")

        if perf['imbalance_ratio'] > 3.0:
          f.write("⚠️ Высокий дисбаланс классов - рекомендуется мониторинг\n")

        if perf['f1_weighted'] < 0.5:
          f.write("⚠️ Низкий F1-score - возможно нужно больше данных\n")

        f.write("\n---\n\n")

    print(f"\n✓ Отчет сохранен: {report_path}")


def main():
  """Главная функция."""
  parser = argparse.ArgumentParser(
    description="Визуализация результатов оптимизации"
  )
  parser.add_argument(
    "--results-dir",
    default="optimization_results",
    help="Директория с результатами"
  )

  args = parser.parse_args()

  visualizer = OptimizationVisualizer(args.results_dir)
  visualizer.create_all_visualizations()
  visualizer.generate_report()

  print("\n✓ Готово!")


if __name__ == "__main__":
  main()