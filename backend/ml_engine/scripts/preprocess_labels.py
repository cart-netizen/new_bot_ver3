#!/usr/bin/env python3
"""
Preprocessing скрипт для заполнения future labels в Feature Store.

Поддерживает два метода разметки:
1. Fixed Threshold (legacy) - простой порог для определения направления
2. Triple Barrier Method (рекомендуется) - адаптивные барьеры на основе ATR

Читает parquet файлы с собранными данными и заполняет:
- future_direction_10s, future_direction_30s, future_direction_60s
- future_movement_10s, future_movement_30s, future_movement_60s

Labels:
- 0 (SELL): Движение вниз / нижний барьер достигнут
- 1 (HOLD): Нейтральное движение / timeout
- 2 (BUY): Движение вверх / верхний барьер достигнут

Usage:
    # Fixed threshold (legacy)
    python -m backend.ml_engine.scripts.preprocess_labels --storage-path data/feature_store

    # Triple Barrier Method (recommended)
    python -m backend.ml_engine.scripts.preprocess_labels --storage-path data/feature_store --method triple_barrier

    # Custom Triple Barrier parameters
    python -m backend.ml_engine.scripts.preprocess_labels --method triple_barrier --tp-mult 1.5 --sl-mult 1.0 --max-hold 24
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from backend.core.logger import get_logger
from backend.ml_engine.features.labeling import (
    TripleBarrierLabeler,
    TripleBarrierConfig,
    Direction
)

logger = get_logger(__name__)


class LabelPreprocessor:
    """
    Заполняет future labels для ML обучения.

    Поддерживает два метода:
    1. fixed_threshold - legacy метод с фиксированным порогом
    2. triple_barrier - industry standard с адаптивными барьерами
    """

    def __init__(
        self,
        method: str = "triple_barrier",
        threshold_neutral: float = 0.0001,  # 0.01% для fixed threshold
        time_horizons: Optional[List[int]] = None,
        # Triple Barrier параметры
        tp_multiplier: float = 1.5,
        sl_multiplier: float = 1.0,
        max_holding_period: int = 24,
        atr_period: int = 14
    ):
        """
        Args:
            method: Метод разметки ("fixed_threshold" или "triple_barrier")
            threshold_neutral: Порог для NEUTRAL класса (для fixed_threshold)
            time_horizons: Временные горизонты в секундах [10, 30, 60]
            tp_multiplier: Множитель take profit для triple_barrier
            sl_multiplier: Множитель stop loss для triple_barrier
            max_holding_period: Макс. период удержания для triple_barrier
            atr_period: Период ATR для triple_barrier
        """
        self.method = method.lower()
        self.threshold_neutral = threshold_neutral
        self.time_horizons = time_horizons or [10, 30, 60]

        # Triple Barrier конфигурация
        self.triple_barrier_config = TripleBarrierConfig(
            tp_multiplier=tp_multiplier,
            sl_multiplier=sl_multiplier,
            max_holding_period=max_holding_period,
            atr_period=atr_period
        )

        if self.method == "triple_barrier":
            self.labeler = TripleBarrierLabeler(self.triple_barrier_config)
            logger.info(
                f"Using Triple Barrier Method: "
                f"TP={tp_multiplier}x ATR, SL={sl_multiplier}x ATR, "
                f"max_hold={max_holding_period}"
            )
        else:
            self.labeler = None
            logger.info(
                f"Using Fixed Threshold Method: threshold={threshold_neutral}"
            )

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

        logger.info(f"Computing future labels for {len(df):,} rows using {self.method} method")

        # Сортируем по symbol и timestamp
        df = df.sort_values([symbol_col, timestamp_col]).reset_index(drop=True)

        # Преобразуем timestamp в datetime если нужно
        if df[timestamp_col].dtype != 'datetime64[ns]':
            if df[timestamp_col].max() > 1e12:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms')
            else:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')

        # Группируем по символу
        for symbol in df[symbol_col].unique():
            logger.info(f"Processing {symbol}...")

            mask = df[symbol_col] == symbol
            symbol_df = df[mask].copy()

            if self.method == "triple_barrier":
                df = self._apply_triple_barrier_labels(df, symbol_df, mask, price_col)
            else:
                df = self._apply_fixed_threshold_labels(
                    df, symbol_df, mask, timestamp_col, price_col
                )

        # Статистика
        self._log_label_statistics(df)

        return df

    def _apply_triple_barrier_labels(
        self,
        df: pd.DataFrame,
        symbol_df: pd.DataFrame,
        mask: np.ndarray,
        price_col: str
    ) -> pd.DataFrame:
        """
        Применить Triple Barrier labeling.
        """
        # Проверяем наличие необходимых колонок
        has_high = 'high' in symbol_df.columns
        has_low = 'low' in symbol_df.columns
        has_atr = 'atr_14' in symbol_df.columns

        if not has_high or not has_low:
            logger.warning(
                f"High/Low columns not found. "
                f"Using close prices for barrier checking."
            )

        # Генерируем labels для каждого временного горизонта
        for horizon_sec in self.time_horizons:
            direction_col = f'future_direction_{horizon_sec}s'
            movement_col = f'future_movement_{horizon_sec}s'

            logger.info(f"  Computing {horizon_sec}s horizon with Triple Barrier...")

            # Адаптируем max_holding_period к горизонту
            # Предполагаем 1 бар = 1 секунда для crypto данных
            adapted_config = TripleBarrierConfig(
                tp_multiplier=self.triple_barrier_config.tp_multiplier,
                sl_multiplier=self.triple_barrier_config.sl_multiplier,
                max_holding_period=horizon_sec,  # Адаптируем к горизонту
                atr_period=self.triple_barrier_config.atr_period
            )

            labeler = TripleBarrierLabeler(adapted_config)

            # Подготавливаем данные для labeler
            label_df = pd.DataFrame({
                'close': symbol_df[price_col].values,
                'high': symbol_df['high'].values if has_high else symbol_df[price_col].values,
                'low': symbol_df['low'].values if has_low else symbol_df[price_col].values
            })

            if has_atr:
                label_df['atr'] = symbol_df['atr_14'].values

            # Генерируем метки
            result = labeler.generate_labels(
                label_df,
                price_col='close',
                high_col='high',
                low_col='low',
                atr_col='atr' if has_atr else None
            )

            # Записываем результаты
            df.loc[mask, direction_col] = result.labels
            df.loc[mask, movement_col] = result.returns

        return df

    def _apply_fixed_threshold_labels(
        self,
        df: pd.DataFrame,
        symbol_df: pd.DataFrame,
        mask: np.ndarray,
        timestamp_col: str,
        price_col: str
    ) -> pd.DataFrame:
        """
        Применить Fixed Threshold labeling (legacy).
        """
        for horizon_sec in self.time_horizons:
            direction_col = f'future_direction_{horizon_sec}s'
            movement_col = f'future_movement_{horizon_sec}s'

            logger.info(f"  Computing {horizon_sec}s horizon with Fixed Threshold...")

            directions, movements = self._compute_horizon_labels_fixed(
                timestamps=symbol_df[timestamp_col].values,
                prices=symbol_df[price_col].values,
                horizon_sec=horizon_sec
            )

            df.loc[mask, direction_col] = directions
            df.loc[mask, movement_col] = movements

        return df

    def _compute_horizon_labels_fixed(
        self,
        timestamps: np.ndarray,
        prices: np.ndarray,
        horizon_sec: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычислить labels для одного временного горизонта (fixed threshold).
        """
        n = len(timestamps)
        directions = np.full(n, np.nan)
        movements = np.full(n, np.nan)

        target_timedelta = pd.Timedelta(seconds=horizon_sec)

        for i in range(n):
            current_time = timestamps[i]
            current_price = prices[i]

            if pd.isna(current_price) or current_price <= 0:
                continue

            # Ищем цену в будущем
            future_time = current_time + target_timedelta
            tolerance = pd.Timedelta(seconds=5)

            future_mask = (
                (timestamps > current_time) &
                (timestamps >= future_time - tolerance) &
                (timestamps <= future_time + tolerance)
            )

            future_indices = np.where(future_mask)[0]

            if len(future_indices) == 0:
                future_mask = timestamps > current_time
                future_indices = np.where(future_mask)[0]

                if len(future_indices) == 0:
                    continue

                future_idx = future_indices[0]
                time_diff = timestamps[future_idx] - current_time
                if time_diff > target_timedelta * 2:
                    continue
            else:
                future_times_subset = timestamps[future_indices]
                time_diffs = np.abs(
                    (future_times_subset - future_time).astype('timedelta64[s]').astype(float)
                )
                closest_idx = future_indices[np.argmin(time_diffs)]
                future_idx = closest_idx

            future_price = prices[future_idx]

            if pd.isna(future_price) or future_price <= 0:
                continue

            price_change = (future_price - current_price) / current_price
            movements[i] = price_change

            # Direction с использованием Direction enum для консистентности
            if abs(price_change) < self.threshold_neutral:
                directions[i] = Direction.HOLD  # 1
            elif price_change > 0:
                directions[i] = Direction.BUY   # 2
            else:
                directions[i] = Direction.SELL  # 0

        return directions, movements

    def _log_label_statistics(self, df: pd.DataFrame):
        """Логирование статистики меток."""
        logger.info("\n" + "=" * 60)
        logger.info("LABEL STATISTICS")
        logger.info("=" * 60)

        for horizon_sec in self.time_horizons:
            direction_col = f'future_direction_{horizon_sec}s'
            if direction_col in df.columns:
                labels = df[direction_col].dropna()
                if len(labels) == 0:
                    logger.warning(f"{direction_col}: No valid labels!")
                    continue

                label_counts = labels.value_counts().sort_index()

                logger.info(f"\n{direction_col}:")
                total = len(labels)
                for label, count in label_counts.items():
                    pct = 100 * count / total
                    label_name = {
                        Direction.SELL: "SELL",
                        Direction.HOLD: "HOLD",
                        Direction.BUY: "BUY"
                    }.get(int(label), str(label))
                    logger.info(f"  {label_name} ({int(label)}): {count:,} ({pct:.1f}%)")

                nan_count = df[direction_col].isna().sum()
                if nan_count > 0:
                    logger.warning(f"  NaN: {nan_count:,} ({100*nan_count/len(df):.1f}%)")

        logger.info("=" * 60 + "\n")

    def process_file(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        overwrite: bool = False
    ) -> bool:
        """
        Обработать один parquet файл.
        """
        try:
            logger.info(f"\nProcessing: {input_path.name}")

            df = pd.read_parquet(input_path)
            logger.info(f"  Rows: {len(df):,}")

            # Проверяем, нужна ли обработка
            label_cols = [f'future_direction_{h}s' for h in self.time_horizons]
            existing_labels = [col for col in label_cols if col in df.columns]

            if existing_labels and not overwrite:
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
    parser = argparse.ArgumentParser(
        description="Preprocess Feature Store labels with Fixed Threshold or Triple Barrier Method"
    )

    # Основные параметры
    parser.add_argument(
        "--storage-path",
        type=str,
        default="data/feature_store",
        help="Путь к Feature Store"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["fixed_threshold", "triple_barrier"],
        default="triple_barrier",
        help="Метод разметки (default: triple_barrier)"
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

    # Fixed Threshold параметры
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0001,
        help="Порог для NEUTRAL класса при fixed_threshold (default: 0.0001 = 0.01%%)"
    )

    # Triple Barrier параметры
    parser.add_argument(
        "--tp-mult",
        type=float,
        default=1.5,
        help="Take Profit множитель (x ATR) для triple_barrier (default: 1.5)"
    )
    parser.add_argument(
        "--sl-mult",
        type=float,
        default=1.0,
        help="Stop Loss множитель (x ATR) для triple_barrier (default: 1.0)"
    )
    parser.add_argument(
        "--max-hold",
        type=int,
        default=24,
        help="Максимальное время удержания для triple_barrier (default: 24)"
    )
    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
        help="Период ATR для triple_barrier (default: 14)"
    )

    args = parser.parse_args()

    storage_path = Path(args.storage_path)

    logger.info("=" * 80)
    logger.info("PREPROCESSING FEATURE STORE LABELS")
    logger.info("=" * 80)
    logger.info(f"Storage path: {storage_path.absolute()}")
    logger.info(f"Method: {args.method}")

    if args.method == "triple_barrier":
        logger.info(f"  • TP multiplier: {args.tp_mult}x ATR")
        logger.info(f"  • SL multiplier: {args.sl_mult}x ATR")
        logger.info(f"  • Max holding period: {args.max_hold}")
        logger.info(f"  • ATR period: {args.atr_period}")
    else:
        logger.info(f"  • Threshold: {args.threshold} ({args.threshold*100:.2f}%)")

    logger.info(f"Overwrite: {args.overwrite}")
    logger.info("=" * 80 + "\n")

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
    preprocessor = LabelPreprocessor(
        method=args.method,
        threshold_neutral=args.threshold,
        tp_multiplier=args.tp_mult,
        sl_multiplier=args.sl_mult,
        max_holding_period=args.max_hold,
        atr_period=args.atr_period
    )

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
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✓ Success: {success_count}")
    if fail_count > 0:
        logger.error(f"❌ Failed: {fail_count}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
