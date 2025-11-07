#!/usr/bin/env python3
"""
Migration Script: Legacy → Feature Store

Переносит данные из legacy формата (.npy/.json) в Feature Store (parquet).

Использование:
    python migrate_legacy_to_parquet.py

Что делает:
1. Сканирует data/ml_training/ для поиска legacy данных
2. Читает .npy файлы (features) и .json файлы (labels, metadata)
3. Конвертирует в DataFrame
4. Записывает в Feature Store (data/feature_store/offline/)

Файл: migrate_legacy_to_parquet.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Добавляем backend в path
sys.path.insert(0, str(Path(__file__).parent))

from backend.core.logger import get_logger
from backend.ml_engine.feature_store.feature_store import get_feature_store

logger = get_logger(__name__)


class LegacyToParquetMigrator:
    """
    Миграция legacy данных в Feature Store.
    """

    def __init__(
        self,
        legacy_dir: str = "data/ml_training",
        feature_store_group: str = "training_features"
    ):
        """
        Args:
            legacy_dir: Директория с legacy данными
            feature_store_group: Feature group для Feature Store
        """
        self.legacy_dir = Path(legacy_dir)
        self.feature_store_group = feature_store_group
        self.feature_store = get_feature_store()

        # Статистика
        self.total_batches_processed = 0
        self.total_samples_migrated = 0
        self.errors = []

    def migrate_all_symbols(self):
        """Миграция всех символов."""
        print("\n" + "=" * 80)
        print("МИГРАЦИЯ: Legacy → Feature Store (Parquet)")
        print("=" * 80)
        print(f"Источник: {self.legacy_dir}")
        print(f"Feature Group: {self.feature_store_group}")
        print("=" * 80 + "\n")

        # Находим все символы
        symbols = [d.name for d in self.legacy_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

        if not symbols:
            print(f"❌ Нет символов в директории {self.legacy_dir}")
            return

        print(f"Найдено символов: {len(symbols)}")
        print(f"Символы: {', '.join(symbols)}\n")

        # Миграция каждого символа
        for symbol in symbols:
            self.migrate_symbol(symbol)

        # Итоговая статистика
        print("\n" + "=" * 80)
        print("ИТОГИ МИГРАЦИИ")
        print("=" * 80)
        print(f"✓ Обработано батчей: {self.total_batches_processed}")
        print(f"✓ Мигрировано семплов: {self.total_samples_migrated:,}")

        if self.errors:
            print(f"\n⚠️  Ошибки ({len(self.errors)}):")
            for error in self.errors[:10]:  # Показываем первые 10
                print(f"  - {error}")

        print("=" * 80 + "\n")

    def migrate_symbol(self, symbol: str):
        """
        Миграция одного символа.

        Args:
            symbol: Торговая пара
        """
        print(f"\n{'=' * 70}")
        print(f"Миграция {symbol}")
        print(f"{'=' * 70}")

        symbol_dir = self.legacy_dir / symbol
        features_dir = symbol_dir / "features"
        labels_dir = symbol_dir / "labels"
        metadata_dir = symbol_dir / "metadata"

        # Проверка наличия директорий
        if not features_dir.exists():
            print(f"  ⚠️  Пропуск: нет директории features")
            return

        # Находим все batch файлы
        feature_files = sorted(features_dir.glob("*.npy"))

        if not feature_files:
            print(f"  ⚠️  Нет .npy файлов")
            return

        print(f"  Найдено батчей: {len(feature_files)}")

        # Обрабатываем каждый batch
        migrated_batches = 0
        migrated_samples = 0

        for feature_file in feature_files:
            try:
                samples_count = self._migrate_batch(
                    symbol,
                    feature_file,
                    labels_dir,
                    metadata_dir
                )

                if samples_count > 0:
                    migrated_batches += 1
                    migrated_samples += samples_count
                    print(f"  ✓ {feature_file.name}: {samples_count} семплов")

            except Exception as e:
                error_msg = f"{symbol}/{feature_file.name}: {e}"
                self.errors.append(error_msg)
                print(f"  ❌ {feature_file.name}: {e}")

        # Обновляем статистику
        self.total_batches_processed += migrated_batches
        self.total_samples_migrated += migrated_samples

        print(f"\n  Итого для {symbol}:")
        print(f"    • Батчей: {migrated_batches}/{len(feature_files)}")
        print(f"    • Семплов: {migrated_samples:,}")

    def _migrate_batch(
        self,
        symbol: str,
        feature_file: Path,
        labels_dir: Path,
        metadata_dir: Path
    ) -> int:
        """
        Миграция одного batch.

        Args:
            symbol: Торговая пара
            feature_file: Путь к .npy файлу с features
            labels_dir: Директория с labels
            metadata_dir: Директория с metadata

        Returns:
            int: Количество мигрированных семплов
        """
        # Имя файла без расширения
        batch_name = feature_file.stem

        # Пути к соответствующим файлам
        labels_file = labels_dir / f"{batch_name}.json"
        metadata_file = metadata_dir / f"{batch_name}.json"

        # Загружаем features
        features = np.load(feature_file)  # Shape: (N, 110)

        # Загружаем labels
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                labels_list = json.load(f)
        else:
            # Создаем пустые labels
            labels_list = [{} for _ in range(len(features))]

        # Загружаем metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata_data = json.load(f)
                metadata_list = metadata_data.get("samples", [])
        else:
            metadata_list = []

        # Проверяем размеры
        n_samples = len(features)
        if len(labels_list) != n_samples:
            logger.warning(
                f"{symbol}/{batch_name}: Несовпадение размеров "
                f"features={n_samples}, labels={len(labels_list)}"
            )
            # Обрезаем до минимального размера
            min_size = min(n_samples, len(labels_list))
            features = features[:min_size]
            labels_list = labels_list[:min_size]
            metadata_list = metadata_list[:min_size] if metadata_list else []

        # Конвертируем в DataFrame
        rows = []

        for i, feature_arr in enumerate(features):
            label_dict = labels_list[i] if i < len(labels_list) else {}
            meta_dict = metadata_list[i] if i < len(metadata_list) else {}

            # Базовая информация
            row = {
                'symbol': symbol,
                'timestamp': meta_dict.get('timestamp', 0),
                'mid_price': meta_dict.get('mid_price', 0.0),
            }

            # Распаковываем 110 признаков
            for j, value in enumerate(feature_arr):
                row[f'feature_{j:03d}'] = float(value)

            # Метки (могут быть None)
            row['future_direction_10s'] = label_dict.get('future_direction_10s')
            row['future_direction_30s'] = label_dict.get('future_direction_30s')
            row['future_direction_60s'] = label_dict.get('future_direction_60s')
            row['future_movement_10s'] = label_dict.get('future_movement_10s')
            row['future_movement_30s'] = label_dict.get('future_movement_30s')
            row['future_movement_60s'] = label_dict.get('future_movement_60s')

            # current_mid_price для preprocessing
            row['current_mid_price'] = label_dict.get('current_mid_price', row['mid_price'])

            # Метаданные сигнала
            row['signal_type'] = meta_dict.get('signal_type')
            row['signal_confidence'] = meta_dict.get('signal_confidence')
            row['signal_strength'] = meta_dict.get('signal_strength')

            rows.append(row)

        # Создаем DataFrame
        df = pd.DataFrame(rows)

        # Сортируем по timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Записываем в Feature Store
        success = self.feature_store.write_offline_features(
            feature_group=self.feature_store_group,
            features=df,
            timestamp_column='timestamp'
        )

        if not success:
            raise Exception("Failed to write to Feature Store")

        return len(df)


def main():
    """Главная функция."""
    migrator = LegacyToParquetMigrator()

    # Проверяем наличие legacy данных
    if not migrator.legacy_dir.exists():
        print(f"❌ Директория не найдена: {migrator.legacy_dir}")
        print("\nУбедитесь что legacy данные находятся в data/ml_training/")
        return

    # Запуск миграции
    migrator.migrate_all_symbols()

    print("\n✅ Миграция завершена!")
    print("\nДанные сохранены в:")
    print("  → data/feature_store/offline/training_features/")
    print("\nТеперь можно использовать parquet данные для обучения модели.")


if __name__ == "__main__":
    main()
