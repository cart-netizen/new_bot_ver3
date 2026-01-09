#!/usr/bin/env python3
"""
TLOB Label Preprocessor - Triple Barrier Labeling для Raw LOB данных.

Два режима работы:
1. SYNC MODE: Синхронизирует labels из LSTM Feature Store с Raw LOB данными
2. GENERATE MODE (NEW!): Генерирует labels напрямую из mid_price используя Triple Barrier

Triple Barrier Method для mid_price:
- Использует rolling volatility вместо ATR (т.к. нет OHLC)
- Верхний барьер: mid_price + volatility * multiplier
- Нижний барьер: mid_price - volatility * multiplier
- Таймаут: max_holding_period

Использование:
    # Режим синхронизации (старый)
    python preprocessing_add_tlob_labels.py --mode sync --start-date 2025-01-01

    # Режим генерации (новый, рекомендуется)
    python preprocessing_add_tlob_labels.py --mode generate --start-date 2025-01-01

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


@dataclass
class TripleBarrierConfigLOB:
    """
    Конфигурация Triple Barrier Method для LOB mid_price данных.

    Адаптирована для работы без OHLC - использует rolling volatility.
    """
    # Множители барьеров относительно volatility
    tp_multiplier: float = 1.2  # Take Profit = entry + tp_mult * volatility
    sl_multiplier: float = 1.2  # Stop Loss = entry - sl_mult * volatility

    # Временные параметры (в количестве снимков LOB, ~5 сек каждый)
    max_holding_period: int = 60  # ~5 минут при 5 сек/снимок

    # Volatility параметры
    volatility_window: int = 20  # Окно для расчёта volatility
    min_volatility_pct: float = 0.0002  # Минимум 0.02%

    # Пороги для фильтрации шума
    min_price_change_pct: float = 0.0001  # Игнорировать изменения < 0.01%

    # Горизонты для разных labels
    horizons: Tuple[int, ...] = (60, 30, 10)  # future_direction_60s, 30s, 10s


# ============================================================================
# TRIPLE BARRIER GENERATOR FOR LOB
# ============================================================================

class TripleBarrierLOBGenerator:
    """
    Triple Barrier Label Generator для Raw LOB данных.

    Работает только с mid_price, не требует OHLC.
    Использует rolling volatility вместо ATR.
    """

    def __init__(self, config: Optional[TripleBarrierConfigLOB] = None):
        self.config = config or TripleBarrierConfigLOB()
        logger.info(
            f"TripleBarrierLOBGenerator initialized: "
            f"TP={self.config.tp_multiplier}x vol, "
            f"SL={self.config.sl_multiplier}x vol, "
            f"max_hold={self.config.max_holding_period}"
        )

    def compute_volatility(self, mid_prices: np.ndarray) -> np.ndarray:
        """
        Вычисляет rolling volatility из mid_price.

        Args:
            mid_prices: Массив mid_price

        Returns:
            Массив volatility (абсолютные значения)
        """
        if len(mid_prices) < self.config.volatility_window:
            # Недостаточно данных - используем минимум
            return np.full(len(mid_prices), mid_prices.mean() * self.config.min_volatility_pct)

        # Вычисляем returns
        returns = np.diff(mid_prices) / np.where(mid_prices[:-1] > 0, mid_prices[:-1], 1)
        returns = np.insert(returns, 0, 0)  # Добавляем 0 в начало

        # Rolling std
        volatility = pd.Series(returns).rolling(
            window=self.config.volatility_window,
            min_periods=1
        ).std().values

        # Преобразуем в абсолютные значения
        volatility_abs = volatility * mid_prices

        # Минимальный порог
        min_vol = mid_prices * self.config.min_volatility_pct
        volatility_abs = np.maximum(volatility_abs, min_vol)

        return volatility_abs

    def generate_labels(
        self,
        mid_prices: np.ndarray,
        timestamps: np.ndarray,
        horizon: int = 60
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Генерация Triple Barrier labels из mid_price.

        Args:
            mid_prices: Массив mid_price
            timestamps: Массив timestamps (для валидации)
            horizon: Горизонт прогноза (количество снимков)

        Returns:
            (labels, returns, statistics)
            - labels: 0=SELL, 1=HOLD, 2=BUY
            - returns: Доходность при достижении барьера
            - statistics: Статистика разметки
        """
        n = len(mid_prices)

        if n < horizon + self.config.volatility_window:
            logger.warning(f"Недостаточно данных: {n} < {horizon + self.config.volatility_window}")
            return np.ones(n, dtype=np.int64), np.zeros(n), {'error': 'insufficient_data'}

        # Вычисляем volatility
        volatility = self.compute_volatility(mid_prices)

        # Инициализируем результаты
        labels = np.ones(n, dtype=np.int64)  # Default = HOLD (1)
        returns = np.zeros(n, dtype=np.float64)
        hit_times = np.zeros(n, dtype=np.int32)

        # Статистика
        buy_count = 0
        sell_count = 0
        hold_count = 0

        # Используем адаптивный horizon на основе переданного параметра
        effective_horizon = min(horizon, self.config.max_holding_period)

        for i in range(n - effective_horizon):
            entry_price = mid_prices[i]
            entry_vol = volatility[i]

            if entry_price <= 0 or np.isnan(entry_price):
                hold_count += 1
                continue

            # Вычисляем барьеры
            take_profit = entry_price + (self.config.tp_multiplier * entry_vol)
            stop_loss = entry_price - (self.config.sl_multiplier * entry_vol)

            # Ищем достижение барьеров
            label = 1  # HOLD
            hit_time = effective_horizon
            final_price = mid_prices[i + effective_horizon]

            for j in range(1, effective_horizon + 1):
                idx = i + j
                if idx >= n:
                    break

                current_price = mid_prices[idx]

                # Проверяем верхний барьер (Take Profit → BUY)
                if current_price >= take_profit:
                    label = 2  # BUY
                    hit_time = j
                    final_price = take_profit
                    buy_count += 1
                    break

                # Проверяем нижний барьер (Stop Loss → SELL)
                if current_price <= stop_loss:
                    label = 0  # SELL
                    hit_time = j
                    final_price = stop_loss
                    sell_count += 1
                    break
            else:
                hold_count += 1

            labels[i] = label
            returns[i] = (final_price - entry_price) / entry_price if entry_price > 0 else 0
            hit_times[i] = hit_time

        # Последние samples = HOLD
        labels[-(effective_horizon):] = 1
        hold_count += effective_horizon

        # Статистика
        total_labeled = buy_count + sell_count + hold_count
        statistics = {
            'total_samples': n,
            'labeled_samples': total_labeled,
            'horizon': effective_horizon,
            'distribution': {
                'SELL': sell_count,
                'HOLD': hold_count,
                'BUY': buy_count
            },
            'percentages': {
                'SELL': 100 * sell_count / total_labeled if total_labeled > 0 else 0,
                'HOLD': 100 * hold_count / total_labeled if total_labeled > 0 else 0,
                'BUY': 100 * buy_count / total_labeled if total_labeled > 0 else 0
            },
            'config': {
                'tp_multiplier': self.config.tp_multiplier,
                'sl_multiplier': self.config.sl_multiplier,
                'volatility_window': self.config.volatility_window
            }
        }

        return labels, returns, statistics

    def generate_multi_horizon_labels(
        self,
        mid_prices: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Генерация labels для нескольких горизонтов.

        Returns:
            Dict с labels для каждого горизонта:
            {
                'future_direction_60s': (labels, returns, stats),
                'future_direction_30s': (labels, returns, stats),
                'future_direction_10s': (labels, returns, stats)
            }
        """
        results = {}

        for horizon in self.config.horizons:
            col_name = f'future_direction_{horizon}s'
            labels, returns, stats = self.generate_labels(mid_prices, timestamps, horizon)
            results[col_name] = (labels, returns, stats)

            # Проверяем что stats содержит percentages (может быть error при недостатке данных)
            if 'percentages' in stats:
                logger.info(
                    f"  {col_name}: "
                    f"SELL={stats['percentages']['SELL']:.1f}%, "
                    f"HOLD={stats['percentages']['HOLD']:.1f}%, "
                    f"BUY={stats['percentages']['BUY']:.1f}%"
                )
            elif 'error' in stats:
                logger.warning(f"  {col_name}: {stats['error']}")

        return results


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
# TLOB LABEL GENERATOR (NEW! - generates labels from raw_lob directly)
# ============================================================================

class TLOBLabelGenerator:
    """
    Генератор labels для TLOB напрямую из raw_lob данных.

    НЕ требует Feature Store - работает полностью автономно.
    Использует Triple Barrier метод адаптированный для LOB mid_price.
    """

    def __init__(
        self,
        raw_lob_path: str = "data/raw_lob",
        output_path: str = "data/raw_lob_labeled",
        tb_config: Optional[TripleBarrierConfigLOB] = None
    ):
        self.raw_lob_path = Path(raw_lob_path)
        self.output_path = Path(output_path)
        self.tb_config = tb_config or TripleBarrierConfigLOB()
        self.generator = TripleBarrierLOBGenerator(self.tb_config)

        # Статистика
        self.stats = {
            'total_samples': 0,
            'total_labeled': 0,
            'symbols_processed': 0,
            'files_written': 0,
            'label_distribution': {'SELL': 0, 'HOLD': 0, 'BUY': 0}
        }

    def process(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Основной метод обработки.

        Args:
            start_date: Начальная дата (YYYY-MM-DD)
            end_date: Конечная дата (YYYY-MM-DD)
            symbols: Список символов (None = все)
            days: Количество дней назад (альтернатива start_date/end_date)

        Returns:
            Статистика обработки
        """
        logger.info("=" * 80)
        logger.info("TLOB LABEL GENERATION (Triple Barrier from mid_price)")
        logger.info("=" * 80)
        logger.info(f"Raw LOB: {self.raw_lob_path}")
        logger.info(f"Output: {self.output_path}")
        logger.info(f"TP multiplier: {self.tb_config.tp_multiplier}")
        logger.info(f"SL multiplier: {self.tb_config.sl_multiplier}")
        logger.info(f"Max holding: {self.tb_config.max_holding_period}")
        logger.info("=" * 80)

        # Создаем output директорию
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Определяем символы
        if symbols is None:
            symbols = self._get_available_symbols()

        logger.info(f"\nОбработка {len(symbols)} символов...")

        # Обрабатываем каждый символ
        for symbol in tqdm(symbols, desc="Generating labels"):
            try:
                self._process_symbol(symbol, start_date, end_date, days)
            except Exception as e:
                logger.error(f"Ошибка обработки {symbol}: {e}")
                import traceback
                traceback.print_exc()

        # Финальная статистика
        logger.info("\n" + "=" * 80)
        logger.info("РЕЗУЛЬТАТЫ")
        logger.info("=" * 80)
        logger.info(f"  Символов обработано: {self.stats['symbols_processed']}")
        logger.info(f"  Всего samples: {self.stats['total_samples']:,}")
        logger.info(f"  Labeled samples: {self.stats['total_labeled']:,}")
        logger.info(f"  Файлов записано: {self.stats['files_written']}")
        logger.info(f"  Распределение меток:")
        for label, count in self.stats['label_distribution'].items():
            pct = 100 * count / self.stats['total_labeled'] if self.stats['total_labeled'] > 0 else 0
            logger.info(f"    {label}: {count:,} ({pct:.1f}%)")
        logger.info("=" * 80)

        return self.stats

    def _get_available_symbols(self) -> List[str]:
        """Получает список доступных символов."""
        symbols = []
        if not self.raw_lob_path.exists():
            logger.warning(f"Raw LOB path не существует: {self.raw_lob_path}")
            return symbols

        for item in self.raw_lob_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                symbols.append(item.name)

        return sorted(symbols)

    def _process_symbol(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        days: Optional[int]
    ) -> None:
        """Обработка одного символа."""
        symbol_dir = self.raw_lob_path / symbol

        if not symbol_dir.exists():
            logger.debug(f"{symbol}: директория не найдена")
            return

        # Загружаем все parquet файлы
        all_dfs = []

        # Вычисляем даты
        if days:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=days)
        else:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        for pq_file in symbol_dir.rglob("*.parquet"):
            try:
                # Фильтр по дате из имени файла
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
            logger.debug(f"{symbol}: нет данных")
            return

        # Объединяем и сортируем
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')

        # Проверяем наличие mid_price
        if 'mid_price' not in combined_df.columns:
            logger.warning(f"{symbol}: нет колонки mid_price")
            return

        self.stats['total_samples'] += len(combined_df)

        # Генерируем labels
        mid_prices = combined_df['mid_price'].values
        timestamps = combined_df['timestamp'].values

        # Генерируем multi-horizon labels
        logger.info(f"\n{symbol}: {len(combined_df):,} samples")
        label_results = self.generator.generate_multi_horizon_labels(mid_prices, timestamps)

        # Проверяем что основной горизонт (60s) имеет достаточно данных
        main_horizon_col = 'future_direction_60s'
        if main_horizon_col in label_results:
            _, _, main_stats = label_results[main_horizon_col]
            if 'error' in main_stats:
                logger.warning(f"{symbol}: Пропущен (insufficient data для {main_horizon_col})")
                return

        # Добавляем labels в DataFrame
        for col_name, (labels, returns, stats) in label_results.items():
            combined_df[col_name] = labels
            movement_col = col_name.replace('direction', 'movement')
            combined_df[movement_col] = returns

            # Обновляем статистику (только если есть distribution)
            if '60s' in col_name and 'distribution' in stats:
                self.stats['label_distribution']['SELL'] += stats['distribution']['SELL']
                self.stats['label_distribution']['HOLD'] += stats['distribution']['HOLD']
                self.stats['label_distribution']['BUY'] += stats['distribution']['BUY']
                self.stats['total_labeled'] += stats.get('labeled_samples', 0)

        # Добавляем symbol колонку если нет
        if 'symbol' not in combined_df.columns:
            combined_df['symbol'] = symbol

        # Сохраняем labeled данные
        self._save_labeled_data(symbol, combined_df)
        self.stats['symbols_processed'] += 1

    def _save_labeled_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Сохранение labeled данных с партиционированием по дате."""
        # Партиционируем по дате
        df['_date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date

        for date_val, date_df in df.groupby('_date'):
            date_str = str(date_val)

            # Создаем директорию
            output_dir = self.output_path / symbol / f"date={date_str}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timestamp}_labeled.parquet"
            filepath = output_dir / filename

            # Удаляем временную колонку
            save_df = date_df.drop(columns=['_date'])

            # Сохраняем
            save_df.to_parquet(
                filepath,
                compression="snappy",
                index=False
            )

            self.stats['files_written'] += 1
            logger.debug(f"Сохранено: {filepath} ({len(save_df)} samples)")


# ============================================================================
# API FUNCTION for integration with training
# ============================================================================

def generate_tlob_labels(
    symbols: Optional[List[str]] = None,
    days: int = 30,
    tp_multiplier: float = 1.2,
    sl_multiplier: float = 1.2,
    max_holding_period: int = 60
) -> Dict[str, Any]:
    """
    API функция для генерации TLOB labels.

    Вызывается автоматически при запуске обучения TLOB Transformer.

    Args:
        symbols: Список символов (None = все доступные)
        days: Количество дней данных для обработки
        tp_multiplier: Множитель для Take Profit барьера
        sl_multiplier: Множитель для Stop Loss барьера
        max_holding_period: Максимальный период удержания (в снимках LOB)

    Returns:
        Статистика обработки
    """
    logger.info(f"[API] Generating TLOB labels for {symbols or 'all'} symbols, {days} days")

    config = TripleBarrierConfigLOB(
        tp_multiplier=tp_multiplier,
        sl_multiplier=sl_multiplier,
        max_holding_period=max_holding_period
    )

    generator = TLOBLabelGenerator(
        raw_lob_path="data/raw_lob",
        output_path="data/raw_lob_labeled",
        tb_config=config
    )

    stats = generator.process(
        symbols=symbols,
        days=days
    )

    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TLOB Label Preprocessor - Triple Barrier Labeling"
    )

    # Режим работы
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sync", "generate"],
        default="generate",
        help="Mode: 'sync' = sync from Feature Store, 'generate' = generate from raw_lob (recommended)"
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
        "--days",
        type=int,
        default=None,
        help="Number of days to process (alternative to start/end date)"
    )

    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Symbols to process (default: all)"
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

    # Triple Barrier параметры (для режима generate)
    parser.add_argument(
        "--tp-multiplier",
        type=float,
        default=1.2,
        help="Take Profit multiplier (x volatility)"
    )

    parser.add_argument(
        "--sl-multiplier",
        type=float,
        default=1.2,
        help="Stop Loss multiplier (x volatility)"
    )

    parser.add_argument(
        "--max-holding",
        type=int,
        default=60,
        help="Max holding period (in LOB snapshots)"
    )

    # Sync mode параметры
    parser.add_argument(
        "--lstm-path",
        type=str,
        default="data/feature_store/offline/training_features",
        help="Path to LSTM feature store (for sync mode)"
    )

    parser.add_argument(
        "--tolerance-ms",
        type=int,
        default=2000,
        help="Timestamp tolerance for merge (ms, for sync mode)"
    )

    args = parser.parse_args()

    if args.mode == "generate":
        # === РЕЖИМ ГЕНЕРАЦИИ (рекомендуется) ===
        print("\n" + "=" * 60)
        print("РЕЖИМ: GENERATE (Triple Barrier из raw_lob)")
        print("=" * 60)

        tb_config = TripleBarrierConfigLOB(
            tp_multiplier=args.tp_multiplier,
            sl_multiplier=args.sl_multiplier,
            max_holding_period=args.max_holding
        )

        generator = TLOBLabelGenerator(
            raw_lob_path=args.raw_lob_path,
            output_path=args.output_path,
            tb_config=tb_config
        )

        stats = generator.process(
            start_date=args.start_date,
            end_date=args.end_date,
            symbols=args.symbols,
            days=args.days
        )

        print("\n✅ Генерация завершена!")
        print(f"   Символов: {stats['symbols_processed']}")
        print(f"   Samples: {stats['total_labeled']:,}")
        print(f"   Files: {stats['files_written']}")
        print(f"   Distribution: SELL={stats['label_distribution']['SELL']}, "
              f"HOLD={stats['label_distribution']['HOLD']}, "
              f"BUY={stats['label_distribution']['BUY']}")

    else:
        # === РЕЖИМ СИНХРОНИЗАЦИИ (legacy) ===
        print("\n" + "=" * 60)
        print("РЕЖИМ: SYNC (синхронизация из Feature Store)")
        print("=" * 60)

        config = TLOBLabelConfig(
            lstm_feature_store_path=args.lstm_path,
            raw_lob_path=args.raw_lob_path,
            output_path=args.output_path,
            timestamp_tolerance_ms=args.tolerance_ms
        )

        processor = TLOBLabelProcessor(config)
        stats = processor.process(
            start_date=args.start_date,
            end_date=args.end_date,
            symbols=args.symbols
        )

        print("\n✅ Синхронизация завершена!")
        print(f"   Merged: {stats['total_merged']:,} samples")
        print(f"   Files: {stats['files_written']}")


if __name__ == "__main__":
    main()
