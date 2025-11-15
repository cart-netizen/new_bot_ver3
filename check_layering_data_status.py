#!/usr/bin/env python3
"""
Quick check of Layering ML data status (without pandas dependency)

Usage:
    python check_layering_data_status.py
"""

from pathlib import Path
import json

def main():
    print("=" * 80)
    print("🔍 LAYERING ML DATA STATUS CHECK")
    print("=" * 80)
    print()

    # Check directory
    data_dir = Path("data/ml_training/layering")

    print("📁 DIRECTORY STATUS")
    print("-" * 80)

    if not data_dir.exists():
        print("❌ Directory does not exist yet")
        print(f"   Path: {data_dir.absolute()}")
        print()
        print("Это нормально если бот еще не запускался.")
        print("Директория создастся автоматически при первом запуске.")
        print()
        print("Для начала сбора данных:")
        print("1. Убедитесь что в config/.env установлено:")
        print("   TRADING_MODE=ONLY_TRAINING  (или FULL)")
        print()
        print("2. Запустите бота:")
        print("   python backend/main.py")
        print()
        return

    print(f"✅ Directory exists: {data_dir.absolute()}")
    print()

    # Check parquet files
    parquet_files = list(data_dir.glob("layering_data_*.parquet"))

    print("📦 PARQUET FILES")
    print("-" * 80)
    print(f"Files found: {len(parquet_files)}")
    print()

    if len(parquet_files) == 0:
        print("⚠️  No parquet files yet")
        print()
        print("Причины:")
        print("- Бот только запустился (нужно подождать)")
        print("- Detector еще не обнаружил layering паттерны")
        print("- Buffer еще не накопил 100 samples для сохранения")
        print()
        print("Что делать:")
        print("- Подождите пока detector обнаружит паттерны")
        print("- Проверьте логи: tail -f logs/bot.log | grep -i layering")
        print()
        return

    # Show file info
    total_size = 0
    for i, filepath in enumerate(sorted(parquet_files)[:10], 1):
        size_kb = filepath.stat().st_size / 1024
        total_size += size_kb
        print(f"  {i:2d}. {filepath.name:40s}  {size_kb:8.1f} KB")

    if len(parquet_files) > 10:
        print(f"  ... and {len(parquet_files) - 10} more files")

    print()
    print(f"Total size: {total_size/1024:.2f} MB")
    print()

    # Check statistics file
    stats_file = data_dir / "statistics.json"

    print("📊 STATISTICS")
    print("-" * 80)

    if not stats_file.exists():
        print("⚠️  No statistics file yet")
        print()
    else:
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        print(f"Total collected:     {stats.get('total_collected', 0):,}")
        print(f"Total labeled:       {stats.get('total_labeled', 0):,}")
        print(f"Total saved:         {stats.get('total_saved', 0):,}")
        print(f"Last updated:        {stats.get('last_updated', 'Unknown')}")
        print()

        total_collected = stats.get('total_collected', 0)

        if total_collected == 0:
            print("⚠️  No data collected yet")
        elif total_collected < 100:
            print(f"⏳ Collecting data... ({total_collected}/100 for first batch)")
        elif total_collected < 1000:
            print(f"📊 Good progress! ({total_collected:,} samples collected)")
        else:
            print(f"🎉 Excellent! ({total_collected:,} samples collected)")

    print()

    # Check model
    model_path = Path("data/models/layering_adaptive_v1.pkl")

    print("🧠 ML MODEL STATUS")
    print("-" * 80)

    if not model_path.exists():
        print("❌ Model not trained yet")
        print()
        print("Для обучения модели:")
        print("1. Соберите минимум 100 labeled samples")
        print("2. Запустите: python backend/scripts/train_layering_model.py")
    else:
        size_kb = model_path.stat().st_size / 1024
        print(f"✅ Model exists: {model_path}")
        print(f"   Size: {size_kb:.1f} KB")

    print()
    print("=" * 80)
    print("📚 NEXT STEPS")
    print("=" * 80)
    print()

    if len(parquet_files) == 0:
        print("1. Запустите бота и подождите сбора данных")
        print("   python backend/main.py")
    elif stats_file.exists() and stats.get('total_collected', 0) < 100:
        print("1. Подождите пока соберется больше данных (минимум 100)")
        print(f"   Текущий прогресс: {stats.get('total_collected', 0)}/100")
    else:
        print("1. Проанализируйте собранные данные:")
        print("   pip install pandas pyarrow scikit-learn")
        print("   python analyze_layering_ml_data.py")
        print()
        print("2. Реализуйте labeling (automatic или manual)")
        print()
        print("3. Обучите модель:")
        print("   python backend/scripts/train_layering_model.py")

    print()
    print("📖 Подробная документация:")
    print("   cat LAYERING_ML_GUIDE.md")
    print()
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
