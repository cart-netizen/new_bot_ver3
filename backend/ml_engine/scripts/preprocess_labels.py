"""
Preprocessing скрипт для заполнения future labels в Feature Store.

Читает parquet файлы с собранными данными и заполняет:
- future_direction_10s, future_direction_30s, future_direction_60s
- future_movement_10s, future_movement_30s, future_movement_60s

Labels:
- future_direction: -1 (DOWN), 0 (NEUTRAL), 1 (UP)
- future_movement: % изменения цены

Usage:
    python -m backend.ml_engine.scripts.preprocess_labels --storage-path data/feature_store
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple
from backend.core.logger import get_logger

logger = get_logger(__name__)


class LabelPreprocessor:
    """Заполняет future labels для ML обучения."""

    def __init__(
        self,
        threshold_neutral: float = 0.0001,  # 0.01% для NEUTRAL
        time_horizons: List[int] = None
    ):
        """
        Args:
            threshold_neutral: Порог для NEUTRAL класса (в долях, не %)
            time_horizons: Временные горизонты в секундах
        """
        self.threshold_neutral = threshold_neutral
        self.time_horizons = time_horizons or [10, 30, 60]

    def compute_future_labels(
        self,
        df: pd.DataFrame,
        symbol_col: str = 'symbol',
        timestamp_col: str = 'timestamp',
        price_col: str = 'current_mid_price'
    ) -> pd.DataFrame:
        """
        Вычислить future labels для всех строк.

        Args:
            df: DataFrame с данными
            symbol_col: Колонка с символом
            timestamp_col: Колонка с timestamp
            price_col: Колонка с текущей ценой

        Returns:
            DataFrame с заполненными labels
        """
        df = df.copy()

        # Проверка обязательных колонок
        required_cols = [symbol_col, timestamp_col, price_col]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        logger.info(f"Computing future labels for {len(df):,} rows")

        # Сортируем по symbol и timestamp
        df = df.sort_values([symbol_col, timestamp_col]).reset_index(drop=True)

        # Преобразуем timestamp в datetime если нужно
        if df[timestamp_col].dtype != 'datetime64[ns]':
            # Если timestamp в миллисекундах
            if df[timestamp_col].max() > 1e12:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms')
            else:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')

        # Группируем по символу
        for symbol in df[symbol_col].unique():
            logger.info(f"Processing {symbol}...")

            mask = df[symbol_col] == symbol
            symbol_df = df[mask].copy()

            # Для каждого временного горизонта
            for horizon_sec in self.time_horizons:
                direction_col = f'future_direction_{horizon_sec}s'
                movement_col = f'future_movement_{horizon_sec}s'

                logger.info(f"  Computing {horizon_sec}s horizon...")

                # Вычисляем labels для этого горизонта
                directions, movements = self._compute_horizon_labels(
                    timestamps=symbol_df[timestamp_col].values,
                    prices=symbol_df[price_col].values,
                    horizon_sec=horizon_sec
                )

                # Записываем в DataFrame
                df.loc[mask, direction_col] = directions
                df.loc[mask, movement_col] = movements

        # Статистика
        logger.info("\nLabel statistics:")
        for horizon_sec in self.time_horizons:
            direction_col = f'future_direction_{horizon_sec}s'
            if direction_col in df.columns:
                labels = df[direction_col].dropna()
                label_counts = labels.value_counts().sort_index()

                logger.info(f"\n{direction_col}:")
                total = len(labels)
                for label, count in label_counts.items():
                    pct = 100 * count / total
                    label_name = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}.get(label, label)
                    logger.info(f"  {label_name} ({label}): {count:,} ({pct:.1f}%)")

                nan_count = df[direction_col].isna().sum()
                if nan_count > 0:
                    logger.warning(f"  NaN: {nan_count:,} ({100*nan_count/len(df):.1f}%)")

        return df

    def _compute_horizon_labels(
        self,
        timestamps: np.ndarray,
        prices: np.ndarray,
        horizon_sec: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычислить labels для одного временного горизонта.

        Args:
            timestamps: Массив timestamps (datetime64[ns])
            prices: Массив цен
            horizon_sec: Временной горизонт в секундах

        Returns:
            (directions, movements) - массивы labels
        """
        n = len(timestamps)
        directions = np.full(n, np.nan)
        movements = np.full(n, np.nan)

        # Конвертируем timedelta в секунды для сравнения
        target_timedelta = pd.Timedelta(seconds=horizon_sec)

        for i in range(n):
            current_time = timestamps[i]
            current_price = prices[i]

            if pd.isna(current_price) or current_price <= 0:
                continue

            # Ищем цену в будущем (в пределах horizon_sec ± 5 секунд)
            future_time = current_time + target_timedelta
            tolerance = pd.Timedelta(seconds=5)  # ±5 секунд допуск

            # Ищем ближайшую точку во времени
            future_mask = (
                (timestamps > current_time) &
                (timestamps >= future_time - tolerance) &
                (timestamps <= future_time + tolerance)
            )

            future_indices = np.where(future_mask)[0]

            if len(future_indices) == 0:
                # Нет данных в будущем - пытаемся найти хоть что-то после
                future_mask = timestamps > current_time
                future_indices = np.where(future_mask)[0]

                if len(future_indices) == 0:
                    continue  # Совсем нет данных

                # Берем первую доступную точку
                future_idx = future_indices[0]

                # Но только если она не слишком далеко (max 2x horizon)
                time_diff = timestamps[future_idx] - current_time
                if time_diff > target_timedelta * 2:
                    continue

            else:
                # Берем ближайшую к целевому времени
                time_diffs = np.abs(timestamps[future_indices] - future_time)
                closest_idx = future_indices[np.argmin(time_diffs)]
                future_idx = closest_idx

            future_price = prices[future_idx]

            if pd.isna(future_price) or future_price <= 0:
                continue

            # Вычисляем изменение
            price_change = (future_price - current_price) / current_price

            # Movement (в долях, не %)
            movements[i] = price_change

            # Direction
            if abs(price_change) < self.threshold_neutral:
                directions[i] = 0  # NEUTRAL
            elif price_change > 0:
                directions[i] = 1  # UP
            else:
                directions[i] = -1  # DOWN

        return directions, movements

    def process_file(
        self,
        input_path: Path,
        output_path: Path = None,
        overwrite: bool = False
    ) -> bool:
        """
        Обработать один parquet файл.

        Args:
            input_path: Путь к входному файлу
            output_path: Путь к выходному файлу (если None, перезаписывает входной)
            overwrite: Перезаписать если labels уже заполнены

        Returns:
            True если успешно
        """
        try:
            logger.info(f"\nProcessing: {input_path.name}")

            # Читаем
            df = pd.read_parquet(input_path)
            logger.info(f"  Rows: {len(df):,}")

            # Проверяем, нужна ли обработка
            label_cols = [f'future_direction_{h}s' for h in self.time_horizons]
            existing_labels = [col for col in label_cols if col in df.columns]

            if existing_labels and not overwrite:
                # Проверяем, есть ли незаполненные labels
                nan_counts = df[existing_labels].isna().sum()
                total_nan = nan_counts.sum()

                if total_nan == 0:
                    logger.info("  ✓ Labels already filled, skipping")
                    return True
                else:
                    logger.info(f"  Found {total_nan:,} NaN labels, processing...")

            # Вычисляем labels
            df_processed = self.compute_future_labels(df)

            # Сохраняем
            output_path = output_path or input_path
            df_processed.to_parquet(output_path, index=False)

            logger.info(f"  ✓ Saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"  ❌ Error: {e}", exc_info=True)
            return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess Feature Store labels")
    parser.add_argument(
        "--storage-path",
        type=str,
        default="data/feature_store",
        help="Путь к Feature Store"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0001,
        help="Порог для NEUTRAL класса (default: 0.0001 = 0.01%%)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписать уже существующие labels"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Максимум файлов для обработки (для тестирования)"
    )

    args = parser.parse_args()

    storage_path = Path(args.storage_path)

    logger.info(f"{'='*80}")
    logger.info(f"PREPROCESSING FEATURE STORE LABELS")
    logger.info(f"{'='*80}")
    logger.info(f"Storage path: {storage_path.absolute()}")
    logger.info(f"Threshold: {args.threshold} ({args.threshold*100:.2f}%)")
    logger.info(f"Overwrite: {args.overwrite}")
    logger.info(f"{'='*80}\n")

    # Проверка существования
    if not storage_path.exists():
        logger.error(f"❌ Path does not exist: {storage_path}")
        return

    # Найти все parquet файлы
    parquet_files = list(storage_path.rglob("*.parquet"))

    if not parquet_files:
        logger.warning(f"⚠️ No parquet files found in {storage_path}")
        return

    logger.info(f"Found {len(parquet_files)} parquet files")

    if args.limit:
        parquet_files = parquet_files[:args.limit]
        logger.info(f"Processing first {args.limit} files")

    # Создаем preprocessor
    preprocessor = LabelPreprocessor(threshold_neutral=args.threshold)

    # Обрабатываем файлы
    success_count = 0
    fail_count = 0

    for i, file_path in enumerate(parquet_files, 1):
        logger.info(f"\n[{i}/{len(parquet_files)}] {file_path.relative_to(storage_path)}")

        if preprocessor.process_file(file_path, overwrite=args.overwrite):
            success_count += 1
        else:
            fail_count += 1

    # Итоги
    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"✓ Success: {success_count}")
    if fail_count > 0:
        logger.error(f"❌ Failed: {fail_count}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
