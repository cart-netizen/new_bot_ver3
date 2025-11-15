#!/usr/bin/env python3
"""
Скрипт для анализа дубликатов в parquet файлах
"""
import pandas as pd
from pathlib import Path

def analyze_duplicates():
    """Анализ дубликатов."""
    feature_store_dir = Path("data/feature_store/offline/training_features")

    if not feature_store_dir.exists():
        print(f"❌ Директория не найдена: {feature_store_dir}")
        return

    parquet_files = list(feature_store_dir.rglob("*.parquet"))

    if not parquet_files:
        print(f"❌ Нет parquet файлов в {feature_store_dir}")
        return

    print(f"📁 Найдено {len(parquet_files)} parquet файлов\n")

    # Читаем первый файл
    df = pd.read_parquet(parquet_files[0])

    print(f"📊 Всего строк: {len(df):,}")
    print(f"📊 Колонок: {len(df.columns)}\n")

    # Проверка на полные дубликаты
    duplicated_all = df.duplicated(keep=False)
    print(f"🔍 Полные дубликаты (все колонки):")
    print(f"  • Дублированных строк: {duplicated_all.sum():,}")
    print(f"  • Уникальных строк: {(~duplicated_all).sum():,}\n")

    # Проверка по ключевым колонкам
    key_columns = ['symbol', 'timestamp']
    if all(col in df.columns for col in key_columns):
        duplicated_keys = df.duplicated(subset=key_columns, keep=False)
        print(f"🔍 Дубликаты по ключевым полям (symbol, timestamp):")
        print(f"  • Дублированных строк: {duplicated_keys.sum():,}")
        print(f"  • Уникальных строк: {(~duplicated_keys).sum():,}\n")

        if duplicated_keys.sum() > 0:
            print("📋 Примеры дубликатов:")
            dup_examples = df[duplicated_keys].head(10)
            print(dup_examples[['symbol', 'timestamp', 'mid_price', 'close']].to_string())
            print()

    # Проверка по timestamp только
    if 'timestamp' in df.columns:
        timestamp_counts = df['timestamp'].value_counts()
        duplicated_timestamps = timestamp_counts[timestamp_counts > 1]

        print(f"🔍 Дубликаты по timestamp:")
        print(f"  • Уникальных timestamps: {len(timestamp_counts):,}")
        print(f"  • Дублированных timestamps: {len(duplicated_timestamps):,}\n")

        if len(duplicated_timestamps) > 0:
            print("📋 Топ-10 самых частых timestamps:")
            print(duplicated_timestamps.head(10).to_string())
            print()

if __name__ == "__main__":
    analyze_duplicates()
