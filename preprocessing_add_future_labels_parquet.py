#!/usr/bin/env python3
"""
Preprocessing скрипт для добавления future labels к данным в Feature Store (parquet).

Запускается ПОСЛЕ сбора данных, ПЕРЕД обучением модели.
Добавляет метки о будущем движении цены (через 10s, 30s, 60s).

Файл: preprocessing_add_future_labels_parquet.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Добавляем backend в path
sys.path.insert(0, str(Path(__file__).parent))

from backend.core.logger import get_logger
from backend.ml_engine.feature_store.feature_store import get_feature_store

logger = get_logger(__name__)


class ParquetFutureLabelProcessor:
    """
    Обрабатывает parquet данные из Feature Store и добавляет future labels.
    """

    def __init__(
        self,
        feature_store_group: str = "training_features",
        start_date: str = None,
        end_date: str = None
    ):
        """
        Args:
            feature_store_group: Feature group для обработки
            start_date: Начальная дата (YYYY-MM-DD) или None для всех данных
            end_date: Конечная дата (YYYY-MM-DD) или None для всех данных
        """
        self.feature_store_group = feature_store_group
        self.start_date = start_date
        self.end_date = end_date
        self.feature_store = get_feature_store()

        # Статистика
        self.total_samples_processed = 0
        self.total_samples_labeled = 0

    def process_all_data(self):
        """Обработка всех данных из Feature Store."""
        print("\n" + "=" * 80)
        print("PREPROCESSING: Добавление Future Labels (Parquet)")
        print("=" * 80)
        print(f"Feature Group: {self.feature_store_group}")
        print(f"Период: {self.start_date or 'начало'} → {self.end_date or 'конец'}")
        print("=" * 80 + "\n")

        # Читаем данные из Feature Store
        print("Загрузка данных из Feature Store...")

        df = self.feature_store.read_offline_features(
            feature_group=self.feature_store_group,
            start_date=self.start_date,
            end_date=self.end_date
        )

        if df.empty:
            print("❌ Нет данных в Feature Store")
            print("\nВозможные причины:")
            print("  1. Данные еще не собраны (запустите бота)")
            print("  2. Неверный feature_group")
            print("  3. Данные за указанный период отсутствуют")
            return

        print(f"✓ Загружено {len(df):,} семплов")
        print(f"  Символы: {df['symbol'].unique().tolist()}")

        # Handle mixed timestamp formats (integers and datetime strings)
        try:
            # Try to convert to datetime (handles both ms timestamps and string dates)
            timestamps = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            # If that failed, try without unit (for string dates)
            if timestamps.isna().all():
                timestamps = pd.to_datetime(df['timestamp'], errors='coerce')

            if not timestamps.isna().all():
                print(f"  Период: {timestamps.min()} → {timestamps.max()}")
            else:
                print(f"  ⚠️  Не удалось определить период (невалидные timestamps)")
        except Exception as e:
            print(f"  ⚠️  Ошибка определения периода: {e}")

        # Normalize timestamps to integers (milliseconds)
        print("\n🔧 Нормализация timestamps...")
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce').astype('int64') // 10**6
        print(f"  ✓ Timestamps нормализованы")

        # Обрабатываем каждый символ отдельно
        symbols = df['symbol'].unique()
        print(f"\nОбработка {len(symbols)} символов...\n")

        all_processed = []

        for symbol in symbols:
            print(f"{'=' * 70}")
            print(f"Обработка {symbol}")
            print(f"{'=' * 70}")

            symbol_df = df[df['symbol'] == symbol].copy()
            processed_df = self._process_symbol_data(symbol, symbol_df)
            all_processed.append(processed_df)

        # Объединяем результаты
        final_df = pd.concat(all_processed, ignore_index=True)

        # Сохраняем обратно в Feature Store
        print(f"\n{'=' * 70}")
        print("Сохранение обновленных данных...")
        print(f"{'=' * 70}")

        # CRITICAL FIX: Delete old parquet files before writing new ones
        # to avoid duplication of data
        self._cleanup_old_parquet_files(final_df)

        success = self.feature_store.write_offline_features(
            feature_group=self.feature_store_group,
            features=final_df,
            timestamp_column='timestamp'
        )

        if success:
            print(f"✓ Сохранено {len(final_df):,} семплов")
        else:
            print("❌ Ошибка сохранения")

        # Итоговая статистика
        print("\n" + "=" * 80)
        print("ИТОГИ PREPROCESSING")
        print("=" * 80)
        print(f"✓ Обработано семплов: {self.total_samples_processed:,}")
        print(f"✓ Помечено future labels: {self.total_samples_labeled:,}")
        print(f"  Процент меток: {100 * self.total_samples_labeled / max(self.total_samples_processed, 1):.1f}%")
        print("=" * 80 + "\n")

    def _cleanup_old_parquet_files(self, df: pd.DataFrame):
        """
        Удаляет старые parquet файлы перед записью новых.
        Это предотвращает дублирование данных.

        Args:
            df: DataFrame с данными для определения партиций
        """
        print("\n🗑️  Очистка старых parquet файлов...")

        # Определяем какие партиции (даты) затронуты
        if 'timestamp' not in df.columns:
            print("  ⚠️  Колонка timestamp не найдена, пропускаем очистку")
            return

        # Получаем уникальные даты
        timestamps = df['timestamp']
        dates = pd.to_datetime(timestamps, unit='ms').dt.strftime('%Y-%m-%d').unique()

        print(f"  Затронуто дат: {len(dates)}")

        # Для каждой даты удаляем старые файлы
        feature_store_dir = Path("data/feature_store/offline") / self.feature_store_group
        deleted_count = 0

        for date_str in dates:
            partition_dir = feature_store_dir / f"date={date_str}"

            if not partition_dir.exists():
                continue

            # Находим все parquet файлы в этой партиции
            parquet_files = list(partition_dir.glob("*.parquet"))

            # Удаляем их
            for file_path in parquet_files:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"  ⚠️  Не удалось удалить {file_path}: {e}")

        print(f"  ✓ Удалено файлов: {deleted_count}")

    def _process_symbol_data(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обработка данных для одного символа.

        Args:
            symbol: Торговая пара
            df: DataFrame с данными символа

        Returns:
            DataFrame с обновленными labels
        """
        # Сортируем по timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"  Семплов: {len(df):,}")

        # Создаем словарь timestamp → price для быстрого поиска
        timestamp_to_price = dict(zip(df['timestamp'], df['current_mid_price']))

        # Для каждого семпла вычисляем future labels
        labeled_count = 0

        for idx, row in df.iterrows():
            current_timestamp = row['timestamp']
            current_price = row['current_mid_price']

            # Пропускаем если уже есть метки
            if pd.notna(row.get('future_direction_60s')):
                labeled_count += 1
                continue

            # Находим будущие цены
            future_10s = self._find_future_price(
                df, timestamp_to_price, idx, current_timestamp, 10
            )
            future_30s = self._find_future_price(
                df, timestamp_to_price, idx, current_timestamp, 30
            )
            future_60s = self._find_future_price(
                df, timestamp_to_price, idx, current_timestamp, 60
            )

            # Обновляем labels
            if future_10s is not None:
                df.at[idx, 'future_movement_10s'] = self._calculate_movement(
                    current_price, future_10s
                )
                df.at[idx, 'future_direction_10s'] = self._calculate_direction(
                    current_price, future_10s
                )
                labeled_count += 1

            if future_30s is not None:
                df.at[idx, 'future_movement_30s'] = self._calculate_movement(
                    current_price, future_30s
                )
                df.at[idx, 'future_direction_30s'] = self._calculate_direction(
                    current_price, future_30s
                )

            if future_60s is not None:
                df.at[idx, 'future_movement_60s'] = self._calculate_movement(
                    current_price, future_60s
                )
                df.at[idx, 'future_direction_60s'] = self._calculate_direction(
                    current_price, future_60s
                )

        self.total_samples_processed += len(df)
        self.total_samples_labeled += labeled_count

        print(f"  ✓ Помечено: {labeled_count}/{len(df)} семплов ({100 * labeled_count / len(df):.1f}%)")

        return df

    def _find_future_price(
        self,
        df: pd.DataFrame,
        timestamp_to_price: Dict[int, float],
        current_idx: int,
        current_timestamp: int,
        delta_seconds: int
    ) -> float:
        """
        Находит цену через N секунд после текущего timestamp.

        Args:
            df: DataFrame с данными
            timestamp_to_price: Словарь timestamp → price
            current_idx: Индекс текущего семпла
            current_timestamp: Текущий timestamp (ms)
            delta_seconds: Через сколько секунд искать цену

        Returns:
            float: Цена через N секунд или None
        """
        target_timestamp = current_timestamp + (delta_seconds * 1000)  # ms
        tolerance = 2000  # ±2 секунды

        # Ищем ближайший семпл к target_timestamp
        # Оптимизация: ищем только в будущем (после current_idx)
        for i in range(current_idx + 1, len(df)):
            future_timestamp = df.iloc[i]['timestamp']

            # Если timestamp в пределах tolerance
            if abs(future_timestamp - target_timestamp) <= tolerance:
                return df.iloc[i]['current_mid_price']

            # Если ушли слишком далеко
            if future_timestamp > target_timestamp + tolerance:
                break

        return None

    def _calculate_movement(self, current_price: float, future_price: float) -> float:
        """
        Рассчитывает процентное изменение цены.

        Args:
            current_price: Текущая цена
            future_price: Будущая цена

        Returns:
            float: Процентное изменение (например, 0.05 = +5%)
        """
        if current_price == 0:
            return 0.0
        return (future_price - current_price) / current_price

    def _calculate_direction(self, current_price: float, future_price: float) -> int:
        """
        Определяет направление движения цены.

        Args:
            current_price: Текущая цена
            future_price: Будущая цена

        Returns:
            int: 2=UP, 1=NEUTRAL, 0=DOWN (для совместимости с ML моделью)
        """
        movement_pct = self._calculate_movement(current_price, future_price)
        threshold = 0.001  # 0.1% - порог для "neutral"

        if movement_pct > threshold:
            return 2  # UP
        elif movement_pct < -threshold:
            return 0  # DOWN
        else:
            return 1  # NEUTRAL


def main():
    """Главная функция."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Добавление future labels к parquet данным"
    )
    parser.add_argument(
        '--feature-group',
        default='training_features',
        help='Feature group для обработки (default: training_features)'
    )
    parser.add_argument(
        '--start-date',
        default=None,
        help='Начальная дата (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        default=None,
        help='Конечная дата (YYYY-MM-DD)'
    )

    args = parser.parse_args()

    # Создаем процессор
    processor = ParquetFutureLabelProcessor(
        feature_store_group=args.feature_group,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # Запуск обработки
    processor.process_all_data()

    print("\n✅ Preprocessing завершен!")
    print("\nДанные в Feature Store обновлены с future labels.")
    print("Теперь данные готовы для обучения ML модели.")


if __name__ == "__main__":
    main()
