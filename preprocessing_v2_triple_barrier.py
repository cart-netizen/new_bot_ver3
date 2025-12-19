#!/usr/bin/env python3
"""
Preprocessing V2 - Triple Barrier Labeling с улучшенными параметрами.

Изменения по сравнению с V1:
1. Использует Triple Barrier Method (ATR-based) вместо fixed threshold
2. Добавлен 5-минутный горизонт (300s)
3. Более широкие пороги для уменьшения HOLD класса
4. Добавлены lagged фичи для улучшения предсказательной силы
5. Добавлены derived фичи на основе top-корреляций

Запуск:
    python preprocessing_v2_triple_barrier.py --start-date 2025-11-01

Файл: preprocessing_v2_triple_barrier.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# Добавляем backend в path
sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).resolve().parent

from backend.core.logger import get_logger
from backend.ml_engine.feature_store.feature_store import get_feature_store

logger = get_logger(__name__)


# =============================================================================
# КОНФИГУРАЦИЯ - Настраиваемые параметры
# =============================================================================

@dataclass
class LabelingConfig:
    """Конфигурация Triple Barrier для разных горизонтов."""

    # Горизонты предсказания (в секундах)
    horizons: List[int] = None

    # ATR множители для ширины барьеров
    # Чем выше - тем шире барьеры, тем меньше HOLD
    tp_multiplier: float = 2.0   # Take Profit = entry + tp_mult * ATR
    sl_multiplier: float = 2.0   # Stop Loss = entry - sl_mult * ATR

    # Fallback threshold если ATR недоступен (в % от цены)
    fixed_threshold_pct: float = 0.15  # 0.15% для криптовалют

    # Минимальный порог движения для non-HOLD (в %)
    # Если движение < min_movement_pct -> HOLD
    min_movement_pct: float = 0.05  # 0.05%

    def __post_init__(self):
        if self.horizons is None:
            # Горизонты: 1 мин, 3 мин, 5 мин
            self.horizons = [60, 180, 300]


# Конфигурация по умолчанию - оптимизирована для уменьшения HOLD
DEFAULT_CONFIG = LabelingConfig(
    horizons=[60, 180, 300],  # 1, 3, 5 минут
    tp_multiplier=2.0,
    sl_multiplier=2.0,
    fixed_threshold_pct=0.15,
    min_movement_pct=0.05
)


# =============================================================================
# LAGGED FEATURES - Лаговые фичи
# =============================================================================

# Фичи с наивысшей корреляцией (из feature_quality_analyzer)
TOP_CORRELATED_FEATURES = [
    'imbalance_5',
    'depth_imbalance_ratio',
    'imbalance_10',
    'volume_delta_5',
    'gap_size',
    'momentum_10',
    'update_frequency',
    'quote_intensity',
    'rsi_28',
    'roc',
]

# Фичи с наивысшим Fisher Ratio (лучшая сепарабельность классов)
TOP_SEPARATING_FEATURES = [
    'orderbook_volatility',
    'bid_ask_spread_rel',
    'effective_spread',
    'trade_arrival_rate',
    'smart_money_index',
]

# Лаги для создания lagged features (в количестве samples)
# При интервале сохранения ~15s: lag=4 ≈ 1 минута назад
LAGS = [1, 2, 4, 8]  # ~15s, 30s, 1min, 2min назад


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет lagged версии top-корреляционных фич.

    Это помогает модели видеть динамику изменений.

    Args:
        df: DataFrame с фичами

    Returns:
        DataFrame с добавленными lagged фичами
    """
    print("\n📊 Добавление Lagged Features...")

    # Комбинируем списки уникальных фич для лаггинга
    features_to_lag = list(set(TOP_CORRELATED_FEATURES + TOP_SEPARATING_FEATURES))

    # Проверяем какие фичи есть в данных
    available_features = [f for f in features_to_lag if f in df.columns]
    print(f"   Фичи для лаггинга: {len(available_features)}/{len(features_to_lag)}")

    new_columns = {}

    for feature in available_features:
        for lag in LAGS:
            col_name = f"{feature}_lag{lag}"
            new_columns[col_name] = df[feature].shift(lag)

    # Добавляем все новые колонки за раз (эффективнее)
    if new_columns:
        new_df = pd.DataFrame(new_columns, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    n_new_features = len(new_columns)
    print(f"   ✓ Добавлено {n_new_features} lagged features")

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет производные фичи на основе top-корреляций.

    Создает комбинации фич, которые могут иметь лучшую предсказательную силу.

    Args:
        df: DataFrame с фичами

    Returns:
        DataFrame с добавленными derived фичами
    """
    print("\n🔧 Добавление Derived Features...")

    derived = {}

    # 1. Изменения imbalance (momentum of imbalance)
    if 'imbalance_5' in df.columns:
        derived['imbalance_5_change'] = df['imbalance_5'].diff()
        derived['imbalance_5_change_pct'] = df['imbalance_5'].pct_change()

        if 'imbalance_5_lag4' in df.columns:
            # Изменение за 1 минуту
            derived['imbalance_5_momentum'] = df['imbalance_5'] - df['imbalance_5_lag4']

    # 2. Ratio фичи
    if 'imbalance_5' in df.columns and 'imbalance_10' in df.columns:
        # Отношение короткого к длинному imbalance
        derived['imbalance_ratio_5_10'] = df['imbalance_5'] / (df['imbalance_10'] + 1e-10)

    # 3. Volatility-adjusted imbalance
    if 'imbalance_5' in df.columns and 'orderbook_volatility' in df.columns:
        derived['imbalance_vol_adjusted'] = df['imbalance_5'] / (df['orderbook_volatility'] + 1e-10)

    # 4. Spread momentum
    if 'bid_ask_spread_rel' in df.columns:
        derived['spread_change'] = df['bid_ask_spread_rel'].diff()
        derived['spread_momentum'] = df['bid_ask_spread_rel'].diff().rolling(4).mean()

    # 5. Volume-weighted imbalance
    if 'imbalance_5' in df.columns and 'volume' in df.columns:
        derived['imbalance_volume_weighted'] = df['imbalance_5'] * np.log1p(df['volume'])

    # 6. RSI momentum
    if 'rsi_14' in df.columns and 'rsi_28' in df.columns:
        derived['rsi_diff'] = df['rsi_14'] - df['rsi_28']
        derived['rsi_14_momentum'] = df['rsi_14'].diff(4)

    # 7. Composite signals
    if all(f in df.columns for f in ['imbalance_5', 'depth_imbalance_ratio', 'volume_delta_5']):
        # Комбинированный сигнал из top-3 коррелированных фич
        derived['composite_signal'] = (
            df['imbalance_5'].rank(pct=True) * 0.4 +
            df['depth_imbalance_ratio'].rank(pct=True) * 0.4 +
            df['volume_delta_5'].rank(pct=True) * 0.2
        )

    # 8. Orderbook pressure change
    if 'smart_money_index' in df.columns:
        derived['smart_money_momentum'] = df['smart_money_index'].diff()
        derived['smart_money_acceleration'] = df['smart_money_index'].diff().diff()

    # 9. Trade intensity ratio
    if 'trade_arrival_rate' in df.columns and 'quote_intensity' in df.columns:
        derived['trade_quote_ratio'] = df['trade_arrival_rate'] / (df['quote_intensity'] + 1e-10)

    # 10. Volatility regime
    if 'orderbook_volatility' in df.columns:
        vol_ma = df['orderbook_volatility'].rolling(20).mean()
        derived['volatility_regime'] = df['orderbook_volatility'] / (vol_ma + 1e-10)

    # Добавляем все производные фичи
    if derived:
        derived_df = pd.DataFrame(derived, index=df.index)
        df = pd.concat([df, derived_df], axis=1)

    print(f"   ✓ Добавлено {len(derived)} derived features")

    return df


# =============================================================================
# TRIPLE BARRIER LABELING
# =============================================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Вычисляет Average True Range для адаптивных барьеров.

    Args:
        df: DataFrame с OHLC данными
        period: Период ATR

    Returns:
        Series с ATR значениями
    """
    if 'high' not in df.columns or 'low' not in df.columns:
        # Fallback: используем close как proxy для high/low
        if 'close' in df.columns:
            return df['close'] * 0.002  # 0.2% от цены как proxy ATR
        elif 'current_mid_price' in df.columns:
            return df['current_mid_price'] * 0.002
        return None

    high = df['high']
    low = df['low']
    close = df['close'].shift(1) if 'close' in df.columns else df['current_mid_price'].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def apply_triple_barrier_label(
    current_price: float,
    future_price: float,
    atr: float,
    config: LabelingConfig
) -> Tuple[int, float]:
    """
    Применяет Triple Barrier логику для определения label.

    Args:
        current_price: Текущая цена
        future_price: Будущая цена
        atr: Average True Range на момент входа
        config: Конфигурация labeling

    Returns:
        (label, movement) где label: 0=SELL, 1=HOLD, 2=BUY
    """
    if current_price <= 0:
        return 1, 0.0

    movement = (future_price - current_price) / current_price
    movement_pct = abs(movement) * 100

    # Определяем порог на основе ATR или fixed threshold
    if atr is not None and atr > 0 and not np.isnan(atr):
        threshold = (atr / current_price) * config.tp_multiplier
    else:
        threshold = config.fixed_threshold_pct / 100

    # Минимальный порог
    min_threshold = config.min_movement_pct / 100
    threshold = max(threshold, min_threshold)

    # Определяем label
    if movement > threshold:
        return 2, movement  # BUY
    elif movement < -threshold:
        return 0, movement  # SELL
    else:
        return 1, movement  # HOLD


class TripleBarrierPreprocessor:
    """
    Preprocessing с Triple Barrier Method.
    """

    def __init__(
        self,
        config: LabelingConfig = None,
        feature_store_group: str = "training_features",
        start_date: str = None,
        end_date: str = None
    ):
        self.config = config or DEFAULT_CONFIG
        self.feature_store_group = feature_store_group
        self.start_date = start_date
        self.end_date = end_date
        self.feature_store = get_feature_store()

        # Статистика
        self.stats = {
            'total_samples': 0,
            'labeled_samples': 0,
            'label_distribution': {}
        }

    def process_all_data(self):
        """Основной метод обработки."""
        print("\n" + "=" * 80)
        print("PREPROCESSING V2 - TRIPLE BARRIER LABELING")
        print("=" * 80)
        print(f"Feature Group: {self.feature_store_group}")
        print(f"Период: {self.start_date or 'начало'} → {self.end_date or 'конец'}")
        print(f"\nПараметры Triple Barrier:")
        print(f"  • Горизонты: {self.config.horizons} секунд")
        print(f"  • TP множитель: {self.config.tp_multiplier}x ATR")
        print(f"  • SL множитель: {self.config.sl_multiplier}x ATR")
        print(f"  • Fixed threshold: {self.config.fixed_threshold_pct}%")
        print(f"  • Min movement: {self.config.min_movement_pct}%")
        print("=" * 80)

        # Загрузка данных
        print("\n📥 Загрузка данных из Feature Store...")
        df = self.feature_store.read_offline_features(
            feature_group=self.feature_store_group,
            start_date=self.start_date,
            end_date=self.end_date
        )

        if df is None or df.empty:
            print("❌ Нет данных в Feature Store!")
            return

        print(f"   ✓ Загружено {len(df):,} семплов")
        print(f"   ✓ Символы: {df['symbol'].unique().tolist()}")

        # Нормализация timestamps
        df = self._normalize_timestamps(df)

        # Удаляем дубликаты
        initial_count = len(df)
        df = df.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')
        if len(df) < initial_count:
            print(f"   ⚠️ Удалено {initial_count - len(df):,} дубликатов")

        # Обработка по символам
        symbols = df['symbol'].unique()
        all_processed = []

        for symbol in symbols:
            print(f"\n{'─' * 70}")
            print(f"Обработка {symbol}")
            print(f"{'─' * 70}")

            symbol_df = df[df['symbol'] == symbol].copy()
            processed_df = self._process_symbol(symbol, symbol_df)
            all_processed.append(processed_df)

        # Объединяем результаты
        final_df = pd.concat(all_processed, ignore_index=True)

        # Добавляем lagged features
        final_df = add_lagged_features(final_df)

        # Добавляем derived features
        final_df = add_derived_features(final_df)

        # Сохранение
        print(f"\n{'=' * 70}")
        print("💾 Сохранение обновленных данных...")
        print(f"{'=' * 70}")

        self._cleanup_old_files(final_df)

        success = self.feature_store.write_offline_features(
            feature_group=self.feature_store_group,
            features=final_df,
            timestamp_column='timestamp'
        )

        if success:
            print(f"   ✓ Сохранено {len(final_df):,} семплов")
        else:
            print("   ❌ Ошибка сохранения!")

        # Итоговая статистика
        self._print_summary()

    def _normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Нормализует timestamps в миллисекунды."""
        print("\n🔧 Нормализация timestamps...")

        normalized = []
        for ts in df['timestamp']:
            if pd.isna(ts):
                normalized.append(None)
            elif isinstance(ts, (int, np.integer, float, np.floating)):
                normalized.append(int(ts))
            else:
                try:
                    dt = pd.to_datetime(ts)
                    normalized.append(int(dt.timestamp() * 1000))
                except:
                    normalized.append(None)

        df['timestamp'] = normalized
        print("   ✓ Timestamps нормализованы")
        return df

    def _process_symbol(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка данных для одного символа."""
        df = df.sort_values('timestamp').reset_index(drop=True)
        n = len(df)
        print(f"   Семплов: {n:,}")

        # Вычисляем ATR если есть OHLC данные
        atr = compute_atr(df)
        has_atr = atr is not None and not atr.isna().all()
        if has_atr:
            print(f"   ✓ ATR вычислен (mean: {atr.mean():.6f})")
        else:
            print(f"   ⚠️ ATR недоступен, используем fixed threshold")

        # Определяем price column
        price_col = 'current_mid_price' if 'current_mid_price' in df.columns else 'close'

        # Создаем индекс timestamp -> row для быстрого поиска
        timestamp_to_idx = dict(zip(df['timestamp'], range(n)))

        # Обрабатываем каждый горизонт
        for horizon in self.config.horizons:
            label_col = f'future_direction_{horizon}s'
            movement_col = f'future_movement_{horizon}s'

            print(f"\n   Горизонт {horizon}s:")

            labels = np.full(n, 1, dtype=np.int32)  # Default = HOLD
            movements = np.full(n, 0.0, dtype=np.float64)
            labeled_count = 0

            for idx in range(n):
                current_ts = df.iloc[idx]['timestamp']
                current_price = df.iloc[idx][price_col]
                current_atr = atr.iloc[idx] if has_atr else None

                if pd.isna(current_price) or current_price <= 0:
                    continue

                # Ищем future price
                target_ts = current_ts + (horizon * 1000)
                tolerance = 15000  # ±15 секунд

                future_price = None
                for future_idx in range(idx + 1, n):
                    future_ts = df.iloc[future_idx]['timestamp']
                    if abs(future_ts - target_ts) <= tolerance:
                        future_price = df.iloc[future_idx][price_col]
                        break
                    if future_ts > target_ts + tolerance:
                        break

                if future_price is not None and not pd.isna(future_price):
                    label, movement = apply_triple_barrier_label(
                        current_price, future_price, current_atr, self.config
                    )
                    labels[idx] = label
                    movements[idx] = movement
                    labeled_count += 1

            df[label_col] = labels
            df[movement_col] = movements

            # Статистика для этого горизонта
            sell = (labels == 0).sum()
            hold = (labels == 1).sum()
            buy = (labels == 2).sum()

            print(f"      Labeled: {labeled_count:,}/{n:,} ({100*labeled_count/n:.1f}%)")
            print(f"      Distribution: SELL={sell:,} ({100*sell/n:.1f}%) | "
                  f"HOLD={hold:,} ({100*hold/n:.1f}%) | BUY={buy:,} ({100*buy/n:.1f}%)")

            # Обновляем общую статистику
            if horizon == 300:  # Основной горизонт для статистики
                self.stats['total_samples'] += n
                self.stats['labeled_samples'] += labeled_count
                self.stats['label_distribution'] = {
                    'SELL': sell, 'HOLD': hold, 'BUY': buy
                }

        return df

    def _cleanup_old_files(self, df: pd.DataFrame):
        """Удаляет старые parquet файлы перед записью новых."""
        print("\n🗑️ Очистка старых файлов...")

        if 'timestamp' not in df.columns:
            return

        dates = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%Y-%m-%d').unique()

        feature_store_dir = PROJECT_ROOT / "data" / "feature_store" / "offline" / self.feature_store_group
        deleted = 0

        for date_str in dates:
            partition_dir = feature_store_dir / f"date={date_str}"
            if partition_dir.exists():
                for f in partition_dir.glob("*.parquet"):
                    try:
                        f.unlink()
                        deleted += 1
                    except Exception as e:
                        print(f"   ⚠️ Не удалось удалить {f}: {e}")

        print(f"   ✓ Удалено {deleted} старых файлов")

    def _print_summary(self):
        """Выводит итоговую статистику."""
        print("\n" + "=" * 80)
        print("📊 ИТОГИ PREPROCESSING V2")
        print("=" * 80)

        print(f"\n✓ Всего семплов: {self.stats['total_samples']:,}")
        print(f"✓ Размечено: {self.stats['labeled_samples']:,}")

        dist = self.stats.get('label_distribution', {})
        if dist:
            total = sum(dist.values())
            print(f"\n📈 Распределение классов (горизонт 300s):")
            for cls, count in dist.items():
                pct = 100 * count / total if total > 0 else 0
                bar = "█" * int(pct / 2)
                print(f"   {cls:4}: {bar} {count:,} ({pct:.1f}%)")

            # Проверка на mode collapse риск
            hold_pct = 100 * dist.get('HOLD', 0) / total if total > 0 else 0
            if hold_pct > 60:
                print(f"\n⚠️ WARNING: HOLD class = {hold_pct:.1f}%")
                print("   Рассмотрите увеличение tp_multiplier/sl_multiplier")
            elif hold_pct > 40:
                print(f"\n✓ HOLD class = {hold_pct:.1f}% - приемлемо")
            else:
                print(f"\n✓ HOLD class = {hold_pct:.1f}% - отлично!")

        print("\n" + "=" * 80)
        print("✅ Preprocessing завершен!")
        print("=" * 80)


def main():
    """Главная функция."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocessing V2 с Triple Barrier Labeling"
    )
    parser.add_argument(
        '--feature-group',
        default='training_features',
        help='Feature group для обработки'
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
    parser.add_argument(
        '--tp-mult',
        type=float,
        default=2.0,
        help='Take Profit множитель ATR (default: 2.0)'
    )
    parser.add_argument(
        '--sl-mult',
        type=float,
        default=2.0,
        help='Stop Loss множитель ATR (default: 2.0)'
    )
    parser.add_argument(
        '--min-movement',
        type=float,
        default=0.05,
        help='Минимальное движение в %% (default: 0.05)'
    )

    args = parser.parse_args()

    # Создаем конфигурацию
    config = LabelingConfig(
        horizons=[60, 180, 300],  # 1, 3, 5 минут
        tp_multiplier=args.tp_mult,
        sl_multiplier=args.sl_mult,
        min_movement_pct=args.min_movement
    )

    # Создаем процессор
    processor = TripleBarrierPreprocessor(
        config=config,
        feature_store_group=args.feature_group,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # Запуск
    processor.process_all_data()


if __name__ == "__main__":
    main()
