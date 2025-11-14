#!/usr/bin/env python3
"""
Анализ данных обучения из Feature Store (Parquet файлы).

Проверяет:
- Объем собранных данных
- Распределение классов (class imbalance)
- Статистика по признакам
- Качество меток
- Готовность к обучению

Использование:
    python analyze_training_data_parquet.py
    python analyze_training_data_parquet.py --feature-group training_features
    python analyze_training_data_parquet.py --show-details
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from backend.ml_engine.feature_store.feature_store import FeatureStore
from backend.core.logger import get_logger

logger = get_logger(__name__)


class ParquetDataAnalyzer:
    """Анализатор данных из Feature Store."""

    def __init__(self, feature_group: str = "training_features"):
        self.feature_group = feature_group
        self.feature_store = FeatureStore()

    def load_data(self) -> pd.DataFrame:
        """Загрузить данные из Feature Store."""
        print("📂 Загрузка данных из Feature Store...")
        print(f"   Feature Group: {self.feature_group}\n")

        try:
            df = self.feature_store.read_features(
                feature_group=self.feature_group,
                start_time=None,  # Все данные
                end_time=None
            )

            if df.empty:
                print("❌ Данные не найдены!")
                return df

            print(f"✅ Загружено {len(df):,} записей")
            print(f"   Период: {df['timestamp'].min()} → {df['timestamp'].max()}\n")

            return df

        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return pd.DataFrame()

    def analyze(self, show_details: bool = False) -> dict:
        """Полный анализ данных."""
        df = self.load_data()

        if df.empty:
            return {}

        print("=" * 80)
        print("АНАЛИЗ ДАННЫХ ДЛЯ ОБУЧЕНИЯ ML МОДЕЛИ")
        print("=" * 80)
        print()

        results = {
            "total_samples": len(df),
            "data_quality": self._analyze_data_quality(df),
            "class_distribution": self._analyze_class_distribution(df),
            "feature_statistics": self._analyze_features(df, show_details),
            "temporal_coverage": self._analyze_temporal(df),
            "readiness": self._assess_readiness(df)
        }

        self._print_summary(results)
        self._print_recommendations(results)

        return results

    def _analyze_data_quality(self, df: pd.DataFrame) -> dict:
        """Анализ качества данных."""
        print("📊 КАЧЕСТВО ДАННЫХ")
        print("-" * 80)

        # Проверяем наличие меток
        has_labels = 'future_direction' in df.columns or 'label' in df.columns
        label_col = 'future_direction' if 'future_direction' in df.columns else 'label'

        if not has_labels:
            print("❌ КРИТИЧНО: Нет колонки с метками!")
            print("   → Запустите: python preprocessing_add_future_labels_parquet.py\n")
            return {"has_labels": False}

        # Подсчет NaN
        total_samples = len(df)
        nan_labels = df[label_col].isna().sum()
        valid_samples = total_samples - nan_labels

        # Проверка признаков
        feature_cols = [col for col in df.columns if col.startswith(('orderbook_', 'candle_', 'indicator_'))]
        nan_features = df[feature_cols].isna().sum().sum()

        print(f"  • Всего записей: {total_samples:,}")
        print(f"  • Валидных меток: {valid_samples:,} ({valid_samples/total_samples*100:.1f}%)")

        if nan_labels > 0:
            print(f"  ⚠️  NaN меток: {nan_labels:,} ({nan_labels/total_samples*100:.1f}%)")

        if nan_features > 0:
            print(f"  ⚠️  NaN в признаках: {nan_features:,}")

        quality_score = (valid_samples / total_samples) * 100
        if quality_score >= 95:
            print(f"  ✅ Качество: {quality_score:.1f}% (отлично)")
        elif quality_score >= 80:
            print(f"  ⚠️  Качество: {quality_score:.1f}% (приемлемо)")
        else:
            print(f"  ❌ Качество: {quality_score:.1f}% (низкое)")

        print()

        return {
            "has_labels": True,
            "label_column": label_col,
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "nan_labels": nan_labels,
            "nan_features": nan_features,
            "quality_score": quality_score
        }

    def _analyze_class_distribution(self, df: pd.DataFrame) -> dict:
        """Анализ распределения классов."""
        print("📈 РАСПРЕДЕЛЕНИЕ КЛАССОВ")
        print("-" * 80)

        label_col = 'future_direction' if 'future_direction' in df.columns else 'label'

        if label_col not in df.columns:
            print("❌ Колонка с метками не найдена\n")
            return {}

        # Убираем NaN
        labels = df[label_col].dropna()

        if len(labels) == 0:
            print("❌ Нет валидных меток\n")
            return {}

        # Подсчет классов
        class_counts = Counter(labels)
        total = len(labels)

        # Маппинг классов
        class_names = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
        if -1 in class_counts:
            class_names = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}

        print(f"  Формат меток: {list(class_counts.keys())}")
        print()

        for cls in sorted(class_counts.keys()):
            count = class_counts[cls]
            pct = (count / total) * 100
            name = class_names.get(cls, f"Class {cls}")
            bar = "█" * int(pct / 2)  # Визуальная полоска
            print(f"  {name:8} ({cls:2}): {count:6,} ({pct:5.1f}%) {bar}")

        print()

        # Оценка дисбаланса
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        if imbalance_ratio <= 2.0:
            print(f"  ✅ Баланс классов: {imbalance_ratio:.2f}x (хорошо)")
        elif imbalance_ratio <= 5.0:
            print(f"  ⚠️  Дисбаланс классов: {imbalance_ratio:.2f}x (приемлемо, используйте class weights)")
        else:
            print(f"  ❌ Сильный дисбаланс: {imbalance_ratio:.2f}x (требуется балансировка)")

        print()

        return {
            "total_labeled": total,
            "class_counts": dict(class_counts),
            "class_percentages": {cls: (count/total)*100 for cls, count in class_counts.items()},
            "imbalance_ratio": imbalance_ratio
        }

    def _analyze_features(self, df: pd.DataFrame, show_details: bool) -> dict:
        """Анализ признаков."""
        print("🔬 ПРИЗНАКИ")
        print("-" * 80)

        # Группы признаков
        orderbook_cols = [col for col in df.columns if col.startswith('orderbook_')]
        candle_cols = [col for col in df.columns if col.startswith('candle_')]
        indicator_cols = [col for col in df.columns if col.startswith('indicator_')]

        print(f"  • OrderBook признаков: {len(orderbook_cols)}")
        print(f"  • Candle признаков: {len(candle_cols)}")
        print(f"  • Indicator признаков: {len(indicator_cols)}")
        print(f"  • ВСЕГО признаков: {len(orderbook_cols) + len(candle_cols) + len(indicator_cols)}")
        print()

        if show_details:
            print("  Статистика по признакам:")
            all_features = orderbook_cols + candle_cols + indicator_cols

            for feat in all_features[:10]:  # Показываем первые 10
                values = df[feat].dropna()
                print(f"    {feat:40}: mean={values.mean():8.4f}, std={values.std():8.4f}")

            if len(all_features) > 10:
                print(f"    ... и ещё {len(all_features) - 10} признаков")
            print()

        return {
            "orderbook_count": len(orderbook_cols),
            "candle_count": len(candle_cols),
            "indicator_count": len(indicator_cols),
            "total_features": len(orderbook_cols) + len(candle_cols) + len(indicator_cols)
        }

    def _analyze_temporal(self, df: pd.DataFrame) -> dict:
        """Анализ временного покрытия."""
        print("⏰ ВРЕМЕННОЕ ПОКРЫТИЕ")
        print("-" * 80)

        if 'timestamp' not in df.columns:
            print("❌ Колонка timestamp не найдена\n")
            return {}

        timestamps = pd.to_datetime(df['timestamp'])
        start_time = timestamps.min()
        end_time = timestamps.max()
        duration = end_time - start_time

        print(f"  • Начало: {start_time}")
        print(f"  • Конец: {end_time}")
        print(f"  • Длительность: {duration.days} дней, {duration.seconds//3600} часов")
        print()

        # Проверка символов
        if 'symbol' in df.columns:
            symbols = df['symbol'].unique()
            print(f"  • Символов: {len(symbols)}")
            if len(symbols) <= 20:
                for sym in sorted(symbols):
                    count = (df['symbol'] == sym).sum()
                    print(f"    - {sym}: {count:,} записей")
            print()

        return {
            "start_time": str(start_time),
            "end_time": str(end_time),
            "duration_days": duration.days,
            "symbols": list(df['symbol'].unique()) if 'symbol' in df.columns else []
        }

    def _assess_readiness(self, df: pd.DataFrame) -> dict:
        """Оценка готовности к обучению."""
        label_col = 'future_direction' if 'future_direction' in df.columns else 'label'

        has_labels = label_col in df.columns
        valid_samples = len(df[label_col].dropna()) if has_labels else 0

        # Минимальные требования
        min_samples = 1000  # Минимум для начала обучения
        recommended_samples = 10000  # Рекомендуемое количество

        readiness_score = 0
        issues = []
        recommendations = []

        # Проверка 1: Наличие меток
        if not has_labels:
            issues.append("Отсутствуют метки (future_direction/label)")
            recommendations.append("Запустите: python preprocessing_add_future_labels_parquet.py")
        else:
            readiness_score += 25

        # Проверка 2: Количество данных
        if valid_samples >= recommended_samples:
            readiness_score += 50
        elif valid_samples >= min_samples:
            readiness_score += 25
            recommendations.append(f"Соберите больше данных (текущих: {valid_samples:,}, рекомендуется: {recommended_samples:,})")
        else:
            issues.append(f"Недостаточно данных ({valid_samples:,} < {min_samples:,})")

        # Проверка 3: Баланс классов
        if has_labels and valid_samples > 0:
            labels = df[label_col].dropna()
            class_counts = Counter(labels)
            if len(class_counts) >= 2:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

                if imbalance_ratio <= 5.0:
                    readiness_score += 25
                else:
                    issues.append(f"Сильный дисбаланс классов ({imbalance_ratio:.1f}x)")
                    recommendations.append("Используйте class_weights или focal_loss при обучении")

        return {
            "ready": readiness_score >= 75,
            "score": readiness_score,
            "issues": issues,
            "recommendations": recommendations
        }

    def _print_summary(self, results: dict):
        """Печать итоговой сводки."""
        print("=" * 80)
        print("ИТОГОВАЯ ОЦЕНКА ГОТОВНОСТИ К ОБУЧЕНИЮ")
        print("=" * 80)
        print()

        readiness = results["readiness"]
        score = readiness["score"]

        # Визуальный индикатор
        bar_length = 50
        filled = int((score / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"  Готовность: [{bar}] {score}%")
        print()

        if readiness["ready"]:
            print("  ✅ ГОТОВО К ОБУЧЕНИЮ!")
            print("     Можете запускать обучение модели.")
        else:
            print("  ⚠️  НЕ ГОТОВО К ОБУЧЕНИЮ")
            print("     Требуются доработки (см. рекомендации ниже)")

        print()

    def _print_recommendations(self, results: dict):
        """Печать рекомендаций."""
        readiness = results["readiness"]

        if readiness["issues"]:
            print("❌ ПРОБЛЕМЫ:")
            for issue in readiness["issues"]:
                print(f"   • {issue}")
            print()

        if readiness["recommendations"]:
            print("💡 РЕКОМЕНДАЦИИ:")
            for rec in readiness["recommendations"]:
                print(f"   • {rec}")
            print()

        if readiness["ready"]:
            print("🚀 СЛЕДУЮЩИЕ ШАГИ:")
            print("   1. Запустите обучение:")
            print("      python backend/ml_engine/training_orchestrator.py")
            print()
            print("   2. Мониторьте прогресс в MLflow:")
            print("      http://localhost:5000")
            print()

        print("=" * 80)


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(
        description="Анализ данных обучения из Feature Store (Parquet)"
    )
    parser.add_argument(
        "--feature-group",
        default="training_features",
        help="Название feature group (default: training_features)"
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Показать детальную статистику по признакам"
    )

    args = parser.parse_args()

    analyzer = ParquetDataAnalyzer(feature_group=args.feature_group)
    analyzer.analyze(show_details=args.show_details)


if __name__ == "__main__":
    main()
