#!/usr/bin/env python3
"""
Инспектор parquet файлов - показывает что внутри.
"""
import pandas as pd
from pathlib import Path
import sys

def inspect_parquet_file(file_path: Path):
    """Показать содержимое parquet файла."""
    print(f"\n{'='*80}")
    print(f"Файл: {file_path}")
    print(f"{'='*80}")

    try:
        df = pd.read_parquet(file_path)

        print(f"\n📊 Общая информация:")
        print(f"  • Строк: {len(df):,}")
        print(f"  • Колонок: {len(df.columns)}")
        print(f"  • Размер в памяти: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        print(f"\n📋 Колонки ({len(df.columns)}):")
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            print(f"  • {col:40} {str(dtype):15} (non-null: {non_null}/{len(df)})")

        print(f"\n🔍 Первые 3 строки:")
        print(df.head(3).to_string())

        print(f"\n📈 Статистика (числовые колонки):")
        print(df.describe().to_string())

        # Check for future_direction or label
        if 'future_direction' in df.columns:
            print(f"\n✅ Колонка future_direction найдена!")
            print(f"  Уникальные значения: {df['future_direction'].unique()}")
            print(f"  Распределение:")
            print(df['future_direction'].value_counts().to_string())
        elif 'label' in df.columns:
            print(f"\n✅ Колонка label найдена!")
            print(f"  Уникальные значения: {df['label'].unique()}")
            print(f"  Распределение:")
            print(df['label'].value_counts().to_string())
        else:
            print(f"\n❌ Колонка future_direction/label НЕ найдена!")

    except Exception as e:
        print(f"\n❌ Ошибка чтения файла: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    # Find parquet files
    feature_store_dir = Path("data/feature_store/offline/training_features")

    if not feature_store_dir.exists():
        print(f"❌ Директория не найдена: {feature_store_dir}")
        return

    parquet_files = list(feature_store_dir.rglob("*.parquet"))

    if not parquet_files:
        print(f"❌ Нет parquet файлов в {feature_store_dir}")
        return

    print(f"\n📁 Найдено {len(parquet_files)} parquet файлов")
    print(f"📂 Директория: {feature_store_dir}\n")

    # Show first file
    print(f"Инспектирую ПЕРВЫЙ файл:")
    inspect_parquet_file(parquet_files[0])

    # Show last file
    if len(parquet_files) > 1:
        print(f"\n\nИнспектирую ПОСЛЕДНИЙ файл:")
        inspect_parquet_file(parquet_files[-1])

    # Show one more from middle
    if len(parquet_files) > 2:
        middle_idx = len(parquet_files) // 2
        print(f"\n\nИнспектирую СРЕДНИЙ файл:")
        inspect_parquet_file(parquet_files[middle_idx])


if __name__ == "__main__":
    main()
