#!/usr/bin/env python3
"""
Миграция существующих данных Feature Store: переименование колонок.

Проблема:
- Старые данные сохранены с анонимными названиями: feature_000, feature_001, ...
- Новая схема требует конкретные названия: adl, adx, bid_price_level_0, ...

Решение:
- Читает все parquet файлы из Feature Store
- Переименовывает колонки согласно FeatureStoreSchema
- Сохраняет обновленные файлы

Usage:
    python -m backend.ml_engine.scripts.migrate_feature_store_columns --storage-path data/feature_store --backup
"""

import argparse
import shutil
from pathlib import Path
from typing import List
import pandas as pd

from backend.core.logger import get_logger
from backend.ml_engine.feature_store.feature_schema import DEFAULT_SCHEMA

logger = get_logger(__name__)


def find_parquet_files(storage_path: Path) -> List[Path]:
    """
    Найти все parquet файлы в Feature Store

    Args:
        storage_path: Путь к Feature Store

    Returns:
        Список путей к parquet файлам
    """
    offline_dir = storage_path / "offline"

    if not offline_dir.exists():
        logger.error(f"Offline directory not found: {offline_dir}")
        return []

    parquet_files = list(offline_dir.glob("**/*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files")

    return parquet_files


def migrate_parquet_file(
    file_path: Path,
    feature_column_names: List[str],
    backup: bool = True,
    dry_run: bool = False
) -> bool:
    """
    Мигрировать один parquet файл

    Args:
        file_path: Путь к файлу
        feature_column_names: Правильные названия колонок
        backup: Создать резервную копию
        dry_run: Только проверка, без изменений

    Returns:
        True если успешно
    """
    try:
        logger.info(f"Processing: {file_path}")

        # Читаем файл
        df = pd.read_parquet(file_path)

        # Проверяем, нужна ли миграция
        old_feature_columns = [c for c in df.columns if c.startswith('feature_')]

        if not old_feature_columns:
            logger.info(f"  ✓ Already migrated or no feature columns found")
            return True

        if len(old_feature_columns) != len(feature_column_names):
            logger.error(
                f"  ✗ Column count mismatch: {len(old_feature_columns)} != {len(feature_column_names)}"
            )
            return False

        logger.info(f"  Found {len(old_feature_columns)} feature columns to rename")

        # Создаем маппинг старых названий на новые
        rename_mapping = {}
        for old_col in sorted(old_feature_columns):
            # Извлекаем индекс из 'feature_000' -> 0
            idx_str = old_col.replace('feature_', '')
            try:
                idx = int(idx_str)
                if 0 <= idx < len(feature_column_names):
                    new_col = feature_column_names[idx]
                    rename_mapping[old_col] = new_col
            except ValueError:
                logger.warning(f"  ⚠ Cannot parse index from: {old_col}")
                continue

        if len(rename_mapping) != len(old_feature_columns):
            logger.warning(
                f"  ⚠ Mapping incomplete: {len(rename_mapping)}/{len(old_feature_columns)}"
            )

        # Dry run - только показываем что будет сделано
        if dry_run:
            logger.info(f"  [DRY RUN] Would rename {len(rename_mapping)} columns")
            logger.info(f"  [DRY RUN] Example: {list(rename_mapping.items())[:3]}")
            return True

        # Создаем backup
        if backup:
            backup_path = file_path.with_suffix('.parquet.backup')
            if not backup_path.exists():
                shutil.copy2(file_path, backup_path)
                logger.info(f"  ✓ Backup created: {backup_path.name}")

        # Переименовываем колонки
        df_migrated = df.rename(columns=rename_mapping)

        # Проверяем результат
        migrated_features = [c for c in df_migrated.columns if c in feature_column_names]
        logger.info(f"  After migration: {len(migrated_features)} feature columns")

        # Сохраняем
        df_migrated.to_parquet(file_path, index=False)
        logger.info(f"  ✓ Migrated successfully")

        return True

    except Exception as e:
        logger.error(f"  ✗ Migration failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Feature Store column names from feature_XXX to proper names"
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="data/feature_store",
        help="Path to Feature Store (default: data/feature_store)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup files before migration"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - show what would be done without making changes"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Migrate only specific file (optional)"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("FEATURE STORE MIGRATION: Column Name Fix")
    logger.info("=" * 80)
    logger.info(f"Storage path: {args.storage_path}")
    logger.info(f"Backup: {args.backup}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("")

    # Получаем правильные названия колонок
    feature_column_names = DEFAULT_SCHEMA.get_all_feature_columns()
    logger.info(f"Target schema: {len(feature_column_names)} feature columns")
    logger.info(f"First 5 columns: {feature_column_names[:5]}")
    logger.info(f"Last 5 columns: {feature_column_names[-5:]}")
    logger.info("")

    # Найти файлы для миграции
    storage_path = Path(args.storage_path)

    if args.file:
        # Мигрировать только один файл
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return 1
        parquet_files = [file_path]
    else:
        # Найти все файлы
        parquet_files = find_parquet_files(storage_path)

    if not parquet_files:
        logger.warning("No parquet files found to migrate")
        return 0

    # Мигрируем файлы
    logger.info(f"Starting migration of {len(parquet_files)} files...")
    logger.info("=" * 80)

    success_count = 0
    fail_count = 0

    for file_path in parquet_files:
        success = migrate_parquet_file(
            file_path,
            feature_column_names,
            backup=args.backup,
            dry_run=args.dry_run
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    # Итоговая статистика
    logger.info("")
    logger.info("=" * 80)
    logger.info("MIGRATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total files: {len(parquet_files)}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {fail_count}")

    if args.dry_run:
        logger.info("")
        logger.info("This was a DRY RUN - no changes were made")
        logger.info("Run without --dry-run to perform actual migration")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
