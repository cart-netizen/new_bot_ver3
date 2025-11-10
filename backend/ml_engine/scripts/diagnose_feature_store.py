"""
Скрипт для диагностики данных в Feature Store.

Проверяет:
- Какие parquet файлы существуют
- Что в них записано
- Статистика по NaN значениям
- Распределение labels
- Правильность колонок

Usage:
    python -m backend.ml_engine.scripts.diagnose_feature_store --storage-path data/feature_store
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
from backend.core.logger import get_logger
from backend.ml_engine.feature_store.feature_schema import DEFAULT_SCHEMA

logger = get_logger(__name__)


def diagnose_parquet_file(file_path: Path):
    """Диагностика одного parquet файла."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Файл: {file_path.name}")
    logger.info(f"Размер: {file_path.stat().st_size / 1024:.2f} KB")
    logger.info(f"{'='*80}")

    try:
        df = pd.read_parquet(file_path)

        # Базовая информация
        logger.info(f"\n📊 Базовая информация:")
        logger.info(f"  • Строк: {len(df):,}")
        logger.info(f"  • Колонок: {len(df.columns)}")
        logger.info(f"  • Размер в памяти: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        # Список колонок
        logger.info(f"\n📋 Колонки ({len(df.columns)}):")
        logger.info(f"  {list(df.columns[:10])}... (показаны первые 10)")

        # Проверка обязательных колонок
        required_cols = ['symbol', 'timestamp', 'future_direction_60s']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.error(f"❌ Отсутствуют обязательные колонки: {missing_cols}")
        else:
            logger.info(f"✓ Все обязательные колонки присутствуют")

        # Проверка feature колонок
        feature_columns = DEFAULT_SCHEMA.get_all_feature_columns()
        logger.info(f"\n🔍 Feature колонки (ожидается {len(feature_columns)}):")

        present_features = [col for col in feature_columns if col in df.columns]
        missing_features = [col for col in feature_columns if col not in df.columns]

        logger.info(f"  • Присутствуют: {len(present_features)}/{len(feature_columns)}")
        if missing_features:
            logger.warning(f"  • Отсутствуют ({len(missing_features)}): {missing_features[:5]}...")

        # Анализ NaN значений
        logger.info(f"\n🔎 Анализ NaN значений:")

        total_values = df.shape[0] * df.shape[1]
        total_nan = df.isna().sum().sum()
        nan_pct = 100 * total_nan / total_values if total_values > 0 else 0

        logger.info(f"  • Всего NaN: {total_nan:,} из {total_values:,} ({nan_pct:.2f}%)")

        # NaN по колонкам
        nan_per_col = df.isna().sum()
        cols_with_nan = nan_per_col[nan_per_col > 0]

        if len(cols_with_nan) > 0:
            logger.warning(f"  • Колонок с NaN: {len(cols_with_nan)}/{len(df.columns)}")
            logger.warning(f"  • Топ-5 колонок с NaN:")
            for col, count in cols_with_nan.nlargest(5).items():
                pct = 100 * count / len(df)
                logger.warning(f"    - {col}: {count:,} ({pct:.1f}%)")
        else:
            logger.info(f"  ✓ Нет NaN значений в данных")

        # Анализ labels
        if 'future_direction_60s' in df.columns:
            logger.info(f"\n🎯 Анализ labels (future_direction_60s):")

            labels = df['future_direction_60s']
            unique_labels = labels.dropna().unique()

            logger.info(f"  • Уникальные значения: {sorted(unique_labels)}")
            logger.info(f"  • NaN в labels: {labels.isna().sum():,} ({100*labels.isna().sum()/len(labels):.1f}%)")

            # Распределение
            label_dist = Counter(labels.dropna())
            logger.info(f"  • Распределение:")
            for label, count in sorted(label_dist.items()):
                pct = 100 * count / len(labels.dropna())
                logger.info(f"    - {label}: {count:,} ({pct:.1f}%)")

        # Анализ feature values
        logger.info(f"\n📈 Статистика features:")

        feature_cols_present = [col for col in feature_columns if col in df.columns]
        if feature_cols_present:
            feature_df = df[feature_cols_present]

            logger.info(f"  • Min: {feature_df.min().min():.6f}")
            logger.info(f"  • Max: {feature_df.max().max():.6f}")
            logger.info(f"  • Mean: {feature_df.mean().mean():.6f}")
            logger.info(f"  • Std: {feature_df.std().mean():.6f}")

            # Проверка на inf значения
            inf_count = np.isinf(feature_df.values).sum()
            if inf_count > 0:
                logger.warning(f"  ⚠️ Inf значений: {inf_count:,}")

        # Показать несколько строк
        logger.info(f"\n📄 Первые 3 строки:")
        logger.info(f"\n{df.head(3).to_string()}")

        # Проверка timestamp и symbol
        if 'timestamp' in df.columns:
            logger.info(f"\n⏰ Timestamps:")
            logger.info(f"  • Тип: {df['timestamp'].dtype}")
            logger.info(f"  • Min: {df['timestamp'].min()}")
            logger.info(f"  • Max: {df['timestamp'].max()}")
            logger.info(f"  • Уникальных: {df['timestamp'].nunique():,}")

        if 'symbol' in df.columns:
            logger.info(f"\n💱 Symbols:")
            symbols = df['symbol'].value_counts()
            for sym, count in symbols.items():
                logger.info(f"  • {sym}: {count:,} записей")

        return df

    except Exception as e:
        logger.error(f"❌ Ошибка при чтении файла: {e}", exc_info=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="Диагностика Feature Store данных")
    parser.add_argument(
        "--storage-path",
        type=str,
        default="data/feature_store",
        help="Путь к Feature Store"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Максимум файлов для проверки"
    )

    args = parser.parse_args()

    storage_path = Path(args.storage_path)

    logger.info(f"🔍 Диагностика Feature Store")
    logger.info(f"📂 Путь: {storage_path.absolute()}")
    logger.info(f"=" * 80)

    # Проверка существования
    if not storage_path.exists():
        logger.error(f"❌ Путь не существует: {storage_path}")
        return

    # Найти все parquet файлы
    parquet_files = list(storage_path.rglob("*.parquet"))

    if not parquet_files:
        logger.warning(f"⚠️ Не найдено parquet файлов в {storage_path}")
        return

    logger.info(f"✓ Найдено parquet файлов: {len(parquet_files)}")

    # Сортировка по дате модификации (новые первые)
    parquet_files = sorted(parquet_files, key=lambda f: f.stat().st_mtime, reverse=True)

    logger.info(f"\n📋 Список файлов (показаны {min(len(parquet_files), args.limit)} из {len(parquet_files)}):")
    for i, f in enumerate(parquet_files[:args.limit]):
        rel_path = f.relative_to(storage_path)
        logger.info(f"  {i+1}. {rel_path}")

    # Диагностика файлов
    logger.info(f"\n{'='*80}")
    logger.info(f"ДЕТАЛЬНАЯ ДИАГНОСТИКА")
    logger.info(f"{'='*80}")

    for i, file_path in enumerate(parquet_files[:args.limit]):
        df = diagnose_parquet_file(file_path)

        if i < len(parquet_files[:args.limit]) - 1:
            logger.info(f"\n{'='*80}\n")

    logger.info(f"\n✓ Диагностика завершена")


if __name__ == "__main__":
    main()
