#!/usr/bin/env python3
"""
Analyze Layering ML Training Data

Скрипт для анализа собранных данных Layering ML:
- Показывает статистику сбора
- Анализирует quality данных
- Показывает distribution признаков
- Проверяет готовность к обучению

Usage:
    python analyze_layering_ml_data.py
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

try:
    import pandas as pd
    import numpy as np
    from backend.ml_engine.detection.layering_data_collector import LayeringDataCollector
    from backend.core.logger import get_logger
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nУстановите зависимости:")
    print("  pip install pandas pyarrow scikit-learn")
    sys.exit(1)

logger = get_logger(__name__)


def main():
    print("=" * 80)
    print("🔍 LAYERING ML DATA ANALYSIS")
    print("=" * 80)
    print()

    # Initialize collector
    data_dir = "data/ml_training/layering"
    collector = LayeringDataCollector(
        data_dir=data_dir,
        enabled=True
    )

    # Get statistics
    stats = collector.get_statistics()

    print("📊 COLLECTION STATISTICS")
    print("-" * 80)
    print(f"Status:              {'✅ Enabled' if stats['enabled'] else '❌ Disabled'}")
    print(f"Total collected:     {stats['total_collected']:,}")
    print(f"Total saved:         {stats['total_saved']:,}")
    print(f"Buffer size:         {stats['buffer_size']:,}")
    print(f"Files on disk:       {stats['files_on_disk']:,}")
    print(f"Total on disk:       {stats['total_on_disk']:,}")
    print()

    # Check if we have data
    if stats['total_on_disk'] == 0:
        print("⚠️  NO DATA COLLECTED YET")
        print()
        print("Для начала сбора данных:")
        print("1. Убедитесь что в config/.env установлено:")
        print("   TRADING_MODE=ONLY_TRAINING  (или FULL)")
        print()
        print("2. Запустите бота:")
        print("   python backend/main.py")
        print()
        print("3. Подождите пока detector обнаружит layering паттерны")
        print("   (может занять несколько часов/дней в зависимости от рыночной активности)")
        print()
        return

    # Load all data
    print("📚 Loading data from disk...")
    df = collector.load_all_data()

    if df.empty:
        print("❌ No data loaded")
        return

    print(f"✅ Loaded {len(df):,} samples")
    print()

    # Labeling statistics
    print("🏷️  LABELING STATISTICS")
    print("-" * 80)
    print(f"Total samples:       {len(df):,}")
    print(f"Labeled samples:     {stats['labeled_on_disk']:,} ({stats['labeling_rate']:.1%})")
    print(f"Unlabeled samples:   {len(df) - stats['labeled_on_disk']:,}")
    print()

    if stats['labeled_on_disk'] > 0:
        print(f"True Positives:      {stats['true_positives']:,}")
        print(f"False Positives:     {stats['false_positives']:,}")

        if stats['true_positives'] + stats['false_positives'] > 0:
            tp_rate = stats['true_positives'] / (stats['true_positives'] + stats['false_positives'])
            print(f"True Positive Rate:  {tp_rate:.1%}")
        print()

    # Feature statistics
    print("📈 FEATURE STATISTICS (Top 10)")
    print("-" * 80)

    key_features = [
        'total_volume_btc',
        'total_volume_usdt',
        'placement_duration',
        'cancellation_rate',
        'spoofing_execution_ratio',
        'layer_count',
        'detector_confidence',
        'volatility_24h',
        'liquidity_score',
        'hour_utc'
    ]

    for feature in key_features:
        if feature in df.columns:
            values = df[feature].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                min_val = values.min()
                max_val = values.max()

                print(f"{feature:30s}: "
                      f"mean={mean_val:8.2f}, "
                      f"std={std_val:8.2f}, "
                      f"min={min_val:8.2f}, "
                      f"max={max_val:8.2f}")
    print()

    # Symbol distribution
    print("🎯 SYMBOL DISTRIBUTION")
    print("-" * 80)
    symbol_counts = df['symbol'].value_counts()
    for symbol, count in symbol_counts.head(10).items():
        print(f"{symbol:12s}: {count:5d} samples ({count/len(df)*100:5.1f}%)")
    print()

    # Confidence distribution
    print("🎲 DETECTOR CONFIDENCE DISTRIBUTION")
    print("-" * 80)
    confidence_bins = pd.cut(df['detector_confidence'], bins=[0, 0.5, 0.65, 0.75, 0.85, 1.0])
    conf_dist = confidence_bins.value_counts().sort_index()
    for bin_range, count in conf_dist.items():
        print(f"{str(bin_range):20s}: {count:5d} samples ({count/len(df)*100:5.1f}%)")
    print()

    # Market regime distribution
    if 'market_regime' in df.columns:
        print("🌍 MARKET REGIME DISTRIBUTION")
        print("-" * 80)
        regime_counts = df['market_regime'].value_counts()
        for regime, count in regime_counts.items():
            print(f"{regime:12s}: {count:5d} samples ({count/len(df)*100:5.1f}%)")
        print()

    # Time distribution
    if 'hour_utc' in df.columns:
        print("⏰ TIME DISTRIBUTION (UTC Hours)")
        print("-" * 80)
        hour_counts = df['hour_utc'].value_counts().sort_index()

        # Group by 4-hour blocks for readability
        for start_hour in range(0, 24, 4):
            end_hour = start_hour + 3
            block_count = hour_counts[(hour_counts.index >= start_hour) & (hour_counts.index <= end_hour)].sum()
            print(f"{start_hour:02d}:00 - {end_hour:02d}:59: {block_count:5d} samples ({block_count/len(df)*100:5.1f}%)")
        print()

    # Ready for training?
    print("🎓 TRAINING READINESS")
    print("-" * 80)

    labeled_df = collector.get_labeled_data()

    if len(labeled_df) < 10:
        print("❌ NOT READY - Insufficient labeled data")
        print(f"   Current: {len(labeled_df)} labeled samples")
        print(f"   Minimum: 10 labeled samples")
        print(f"   Recommended: 100+ labeled samples")
        print()
        print("Что делать:")
        print("1. Реализуйте automatic labeling (price action validation)")
        print("2. Или вручную разметьте данные используя:")
        print("   collector.update_label(data_id, label=True/False)")
    elif len(labeled_df) < 100:
        print("⚠️  MARGINAL - Можно обучать, но качество будет низким")
        print(f"   Current: {len(labeled_df)} labeled samples")
        print(f"   Recommended: 100+ labeled samples")
    else:
        print("✅ READY FOR TRAINING!")
        print(f"   Labeled samples: {len(labeled_df)}")

        # Check class balance
        true_count = labeled_df['is_true_layering'].sum()
        false_count = len(labeled_df) - true_count

        print(f"   True Positives:  {true_count} ({true_count/len(labeled_df)*100:.1f}%)")
        print(f"   False Positives: {false_count} ({false_count/len(labeled_df)*100:.1f}%)")

        # Check balance
        min_class = min(true_count, false_count)
        max_class = max(true_count, false_count)
        balance_ratio = min_class / max_class if max_class > 0 else 0

        if balance_ratio < 0.3:
            print(f"   ⚠️  UNBALANCED dataset (ratio: {balance_ratio:.2f})")
            print("      Рекомендуется баланс 40/60 - 60/40")
        else:
            print(f"   ✅ Balanced dataset (ratio: {balance_ratio:.2f})")

        print()
        print("Для обучения модели запустите:")
        print("  python backend/scripts/train_layering_model.py")

    print()
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
