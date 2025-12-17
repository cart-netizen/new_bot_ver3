#!/usr/bin/env python3
"""
TLOB Label Synchronization Preprocessor.

Синхронизирует labels из LSTM Feature Store с Raw LOB данными для обучения
TLOB Transformer.

Логика:
1. Загружает labeled LSTM features (с future_direction_60s)
2. Загружает Raw LOB данные
3. Синхронизирует по timestamp (с допуском ±500ms)
4. Создает labeled Raw LOB датасет для обучения

Использование:
    python preprocessing_add_tlob_labels.py --start-date 2025-01-01 --end-date 2025-01-31

Результат:
    data/raw_lob_labeled/{symbol}/date={YYYY-MM-DD}/*.parquet

Файл: preprocessing_add_tlob_labels.py
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm

# Добавляем путь к backend
sys.path.insert(0, str(Path(__file__).parent))

from backend.core.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TLOBLabelConfig:
    """Конфигурация синхронизации labels."""

    # Пути
    lstm_feature_store_path: str = "data/feature_store/offline/training_features"
    raw_lob_path: str = "data/raw_lob"
    output_path: str = "data/raw_lob_labeled"

    # Синхронизация
    timestamp_tolerance_ms: int = 2000  # ±2 секунды для merge_asof

    # Фильтрация
    min_samples_per_file: int = 10

    # Колонки для labels
    label_columns: Tuple[str, ...] = (
        'future_direction_60s',
        'future_direction_30s',
        'future_direction_10s',
        'future_movement_60s',
        'future_movement_30s',
        'future_movement_10s',
    )

    # Compression
    parquet_compression: str = "snappy"


# ============================================================================
# LABEL PROCESSOR
# ============================================================================

class TLOBLabelProcessor:
    """
    Процессор для синхронизации labels с Raw LOB данными.

    Основные шаги:
    1. Загрузка labeled LSTM features
    2. Загрузка Raw LOB parquet файлов
    3. Merge по timestamp с tolerance
    4. Сохранение labeled Raw LOB данных
    """

    def __init__(self, config: Optional[TLOBLabelConfig] = None):
        self.config = config or TLOBLabelConfig()

        self.lstm_store = Path(self.config.lstm_feature_store_path)
        self.raw_lob_path = Path(self.config.raw_lob_path)
        self.output_path = Path(self.config.output_path)

        # Статистика
        self.stats = {
            'total_lstm_samples': 0,
            'total_raw_lob_samples': 0,
            'total_merged': 0,
            'total_dropped': 0,
            'files_processed': 0,
            'files_written': 0,
        }

    def process(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Основной метод обработки.

        Args:
            start_date: Начальная дата (YYYY-MM-DD)
            end_date: Конечная дата (YYYY-MM-DD)
            symbols: Список символов для обработки (None = все)

        Returns:
            Dict со статистикой обработки
        """
        logger.info("=" * 80)
        logger.info("TLOB LABEL SYNCHRONIZATION")
        logger.info("=" * 80)
        logger.info(f"LSTM Features: {self.lstm_store}")
        logger.info(f"Raw LOB: {self.raw_lob_path}")
        logger.info(f"Output: {self.output_path}")
        logger.info(f"Date range: {start_date} → {end_date}")
        logger.info(f"Symbols: {symbols or 'all'}")
        logger.info("=" * 80)

        # Создаем output директорию
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 1. Загрузка LSTM labels
        logger.info("\n[1/3] Загрузка LSTM labels...")
        lstm_df = self._load_lstm_labels(start_date, end_date)

        if lstm_df.empty:
            logger.error("Нет LSTM labels для обработки!")
            return self.stats

        logger.info(f"  Загружено {len(lstm_df):,} LSTM samples")
        self.stats['total_lstm_samples'] = len(lstm_df)

        # 2. Определение символов
        if symbols is None:
            symbols = self._get_available_symbols()

        logger.info(f"\n[2/3] Обработка {len(symbols)} символов...")

        # 3. Обработка каждого символа
        for symbol in tqdm(symbols, desc="Symbols"):
            try:
                self._process_symbol(symbol, lstm_df, start_date, end_date)
            except Exception as e:
                logger.error(f"Ошибка обработки {symbol}: {e}")

        # Финальная статистика
        logger.info("\n" + "=" * 80)
        logger.info("РЕЗУЛЬТАТЫ")
        logger.info("=" * 80)
        logger.info(f"  LSTM samples: {self.stats['total_lstm_samples']:,}")
        logger.info(f"  Raw LOB samples: {self.stats['total_raw_lob_samples']:,}")
        logger.info(f"  Merged: {self.stats['total_merged']:,}")
        logger.info(f"  Dropped: {self.stats['total_dropped']:,}")
        logger.info(f"  Files processed: {self.stats['files_processed']}")
        logger.info(f"  Files written: {self.stats['files_written']}")
        logger.info("=" * 80)

        return self.stats

    def _load_lstm_labels(
        self,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Загрузка labeled LSTM features."""
        if not self.lstm_store.exists():
            logger.warning(f"LSTM store не найден: {self.lstm_store}")
            return pd.DataFrame()

        all_dfs = []

        # Парсим даты
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        # Ищем partitions
        for partition_dir in self.lstm_store.iterdir():
            if not partition_dir.is_dir():
                continue

            # date=YYYY-MM-DD
            if partition_dir.name.startswith("date="):
                date_str = partition_dir.name.split("=")[1]
                try:
                    part_date = datetime.strptime(date_str, "%Y-%m-%d")

                    # Фильтр по дате
                    if start_dt and part_date < start_dt:
                        continue
                    if end_dt and part_date > end_dt:
                        continue

                    # Загружаем parquet файлы
                    for pq_file in partition_dir.glob("*.parquet"):
                        try:
                            df = pd.read_parquet(pq_file)

                            # Выбираем только нужные колонки
                            required_cols = ['symbol', 'timestamp', 'mid_price']
                            available_labels = [
                                c for c in self.config.label_columns
                                if c in df.columns
                            ]

                            if not available_labels:
                                continue

                            cols_to_keep = required_cols + available_labels
                            cols_to_keep = [c for c in cols_to_keep if c in df.columns]

                            df = df[cols_to_keep]
                            all_dfs.append(df)

                        except Exception as e:
                            logger.warning(f"Ошибка чтения {pq_file}: {e}")

                except ValueError:
                    continue

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)

        # Сортируем по timestamp
        combined = combined.sort_values('timestamp').reset_index(drop=True)

        # Удаляем строки без labels
        label_cols = [c for c in self.config.label_columns if c in combined.columns]
        if label_cols:
            combined = combined.dropna(subset=label_cols[:1])  # Хотя бы один label

        return combined

    def _get_available_symbols(self) -> List[str]:
        """Получает список доступных символов из Raw LOB."""
        symbols = []

        for item in self.raw_lob_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                symbols.append(item.name)

        return sorted(symbols)

    def _process_symbol(
        self,
        symbol: str,
        lstm_df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> None:
        """Обработка одного символа."""

        # Фильтруем LSTM по символу
        symbol_lstm = lstm_df[lstm_df['symbol'] == symbol].copy()

        if symbol_lstm.empty:
            logger.debug(f"{symbol}: нет LSTM labels")
            return

        # Загружаем Raw LOB
        raw_lob_df = self._load_raw_lob(symbol, start_date, end_date)

        if raw_lob_df.empty:
            logger.debug(f"{symbol}: нет Raw LOB данных")
            return

        self.stats['total_raw_lob_samples'] += len(raw_lob_df)
        self.stats['files_processed'] += 1

        # Merge по timestamp
        merged_df = self._merge_labels(raw_lob_df, symbol_lstm)

        if merged_df.empty:
            logger.debug(f"{symbol}: нет совпадений по timestamp")
            return

        self.stats['total_merged'] += len(merged_df)
        self.stats['total_dropped'] += len(raw_lob_df) - len(merged_df)

        # Сохраняем
        self._save_labeled_data(symbol, merged_df)

    def _load_raw_lob(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Загрузка Raw LOB данных для символа."""
        symbol_dir = self.raw_lob_path / symbol

        if not symbol_dir.exists():
            return pd.DataFrame()

        all_dfs = []

        # Парсим даты
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        # Ищем все parquet файлы (включая date= партиции)
        for pq_file in symbol_dir.rglob("*.parquet"):
            try:
                # Извлекаем дату из имени файла (SYMBOL_YYYYMMDD_HHMMSS.parquet)
                filename = pq_file.stem
                parts = filename.split('_')

                if len(parts) >= 2:
                    try:
                        file_date = datetime.strptime(parts[1], "%Y%m%d")

                        if start_dt and file_date.date() < start_dt.date():
                            continue
                        if end_dt and file_date.date() > end_dt.date():
                            continue
                    except ValueError:
                        pass

                df = pd.read_parquet(pq_file)
                all_dfs.append(df)

            except Exception as e:
                logger.warning(f"Ошибка чтения {pq_file}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('timestamp').reset_index(drop=True)

        # Дедупликация по timestamp
        combined = combined.drop_duplicates(subset=['timestamp'], keep='first')

        return combined

    def _merge_labels(
        self,
        raw_lob_df: pd.DataFrame,
        lstm_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge labels по timestamp с tolerance."""

        # Сортируем оба DataFrame
        raw_lob_df = raw_lob_df.sort_values('timestamp').reset_index(drop=True)
        lstm_df = lstm_df.sort_values('timestamp').reset_index(drop=True)

        # merge_asof с tolerance
        merged = pd.merge_asof(
            raw_lob_df,
            lstm_df.drop(columns=['symbol'], errors='ignore'),
            on='timestamp',
            tolerance=self.config.timestamp_tolerance_ms,
            direction='nearest'
        )

        # Удаляем строки без labels
        label_cols = [c for c in self.config.label_columns if c in merged.columns]
        if label_cols:
            merged = merged.dropna(subset=label_cols[:1])

        return merged

    def _save_labeled_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Сохранение labeled данных."""

        # Партиционируем по дате
        df['_date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date

        for date_val, date_df in df.groupby('_date'):
            date_str = str(date_val)

            # Создаем директорию
            output_dir = self.output_path / symbol / f"date={date_str}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timestamp}.parquet"
            filepath = output_dir / filename

            # Удаляем временную колонку
            date_df = date_df.drop(columns=['_date'])

            # Сохраняем
            date_df.to_parquet(
                filepath,
                compression=self.config.parquet_compression,
                index=False
            )

            self.stats['files_written'] += 1
            logger.debug(f"Сохранено: {filepath} ({len(date_df)} samples)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TLOB Label Synchronization Preprocessor"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Symbols to process (default: all)"
    )

    parser.add_argument(
        "--lstm-path",
        type=str,
        default="data/feature_store/offline/training_features",
        help="Path to LSTM feature store"
    )

    parser.add_argument(
        "--raw-lob-path",
        type=str,
        default="data/raw_lob",
        help="Path to raw LOB data"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="data/raw_lob_labeled",
        help="Output path for labeled data"
    )

    parser.add_argument(
        "--tolerance-ms",
        type=int,
        default=2000,
        help="Timestamp tolerance for merge (ms)"
    )

    args = parser.parse_args()

    # Конфигурация
    config = TLOBLabelConfig(
        lstm_feature_store_path=args.lstm_path,
        raw_lob_path=args.raw_lob_path,
        output_path=args.output_path,
        timestamp_tolerance_ms=args.tolerance_ms
    )

    # Обработка
    processor = TLOBLabelProcessor(config)
    stats = processor.process(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=args.symbols
    )

    print("\n✅ Обработка завершена!")
    print(f"   Merged: {stats['total_merged']:,} samples")
    print(f"   Files: {stats['files_written']}")


if __name__ == "__main__":
    main()
