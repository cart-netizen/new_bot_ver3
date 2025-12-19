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
from typing import List, Dict, Optional, Tuple, Any
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

    # === Режим фиксированного процента ===
    # Если True - игнорирует ATR и использует fixed_threshold_pct для всех символов
    use_fixed_pct: bool = False

    # === Масштабирование порога по горизонту (√T scaling) ===
    # Если True - порог увеличивается с горизонтом по правилу √(horizon/base_horizon)
    # Это учитывает, что волатильность растёт пропорционально √T
    use_horizon_scaling: bool = True
    base_horizon: int = 60  # Базовый горизонт для scaling (секунды)

    # === Фильтрация "плоских" символов ===
    # Символы, где HOLD > max_hold_pct во всех горизонтах, исключаются
    max_hold_pct: float = 85.0  # Максимальный % HOLD для включения символа

    def __post_init__(self):
        if self.horizons is None:
            # Горизонты: 1 мин, 3 мин, 5 мин
            self.horizons = [60, 180, 300]

    def get_scaled_threshold(self, horizon: int) -> float:
        """
        Возвращает порог с учётом масштабирования по горизонту.

        По теории случайных блужданий: σ(T) = σ(1) × √T
        Поэтому порог для большего горизонта должен быть больше.
        """
        if not self.use_horizon_scaling:
            return self.fixed_threshold_pct

        # √T scaling: threshold(T) = threshold(base) × √(T/base)
        scale_factor = np.sqrt(horizon / self.base_horizon)
        return self.fixed_threshold_pct * scale_factor


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


def add_lagged_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Добавляет lagged версии top-корреляционных фич.

    Это помогает модели видеть динамику изменений.

    ВАЖНО: Эта функция должна вызываться для каждого символа ОТДЕЛЬНО,
    чтобы shift() не смешивал данные разных символов!

    Args:
        df: DataFrame с фичами
        verbose: Выводить ли информацию о процессе

    Returns:
        DataFrame с добавленными lagged фичами
    """
    if verbose:
        print("\n📊 Добавление Lagged Features...")

    # Комбинируем списки уникальных фич для лаггинга
    features_to_lag = list(set(TOP_CORRELATED_FEATURES + TOP_SEPARATING_FEATURES))

    # Проверяем какие фичи есть в данных
    available_features = [f for f in features_to_lag if f in df.columns]
    if verbose:
        print(f"   Фичи для лаггинга: {len(available_features)}/{len(features_to_lag)}")

    # Удаляем существующие lagged колонки (если запуск на уже обработанных данных)
    existing_lag_cols = [c for c in df.columns if '_lag' in c]
    if existing_lag_cols:
        if verbose:
            print(f"   ⚠️ Удаляем {len(existing_lag_cols)} существующих lagged колонок")
        df = df.drop(columns=existing_lag_cols)

    new_columns = {}

    for feature in available_features:
        for lag in LAGS:
            col_name = f"{feature}_lag{lag}"
            new_columns[col_name] = df[feature].shift(lag).values  # .values для избежания проблем с индексами

    # Добавляем все новые колонки за раз (эффективнее)
    if new_columns:
        for col_name, values in new_columns.items():
            df[col_name] = values

    # Заполняем NaN в начале (первые N строк после shift) значением forward fill,
    # затем backward fill для оставшихся NaN
    lag_cols_added = list(new_columns.keys())
    if lag_cols_added:
        df[lag_cols_added] = df[lag_cols_added].fillna(method='bfill')

    n_new_features = len(new_columns)
    if verbose:
        print(f"   ✓ Добавлено {n_new_features} lagged features")

    return df


def add_derived_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Добавляет производные фичи на основе top-корреляций.

    Создает комбинации фич, которые могут иметь лучшую предсказательную силу.

    ВАЖНО: Эта функция должна вызываться для каждого символа ОТДЕЛЬНО,
    чтобы diff() и rolling() не смешивали данные разных символов!

    Args:
        df: DataFrame с фичами
        verbose: Выводить ли информацию о процессе

    Returns:
        DataFrame с добавленными derived фичами
    """
    if verbose:
        print("\n🔧 Добавление Derived Features...")

    # Список derived фич, которые будем создавать
    derived_feature_names = [
        'imbalance_5_change', 'imbalance_5_change_pct', 'imbalance_5_momentum',
        'imbalance_ratio_5_10', 'imbalance_vol_adjusted', 'spread_change',
        'spread_momentum', 'imbalance_volume_weighted', 'rsi_diff', 'rsi_14_momentum',
        'composite_signal', 'smart_money_momentum', 'smart_money_acceleration',
        'trade_quote_ratio', 'volatility_regime'
    ]

    # Удаляем существующие derived колонки (если запуск на уже обработанных данных)
    existing_derived_cols = [c for c in df.columns if c in derived_feature_names]
    if existing_derived_cols:
        if verbose:
            print(f"   ⚠️ Удаляем {len(existing_derived_cols)} существующих derived колонок")
        df = df.drop(columns=existing_derived_cols)

    added_count = 0

    # 1. Изменения imbalance (momentum of imbalance)
    if 'imbalance_5' in df.columns:
        df['imbalance_5_change'] = df['imbalance_5'].diff().values
        df['imbalance_5_change_pct'] = df['imbalance_5'].pct_change().values
        added_count += 2

        if 'imbalance_5_lag4' in df.columns:
            # Изменение за 1 минуту
            df['imbalance_5_momentum'] = (df['imbalance_5'].values - df['imbalance_5_lag4'].values)
            added_count += 1

    # 2. Ratio фичи
    if 'imbalance_5' in df.columns and 'imbalance_10' in df.columns:
        # Отношение короткого к длинному imbalance
        df['imbalance_ratio_5_10'] = (df['imbalance_5'].values / (df['imbalance_10'].values + 1e-10))
        added_count += 1

    # 3. Volatility-adjusted imbalance
    if 'imbalance_5' in df.columns and 'orderbook_volatility' in df.columns:
        df['imbalance_vol_adjusted'] = (df['imbalance_5'].values / (df['orderbook_volatility'].values + 1e-10))
        added_count += 1

    # 4. Spread momentum
    if 'bid_ask_spread_rel' in df.columns:
        df['spread_change'] = df['bid_ask_spread_rel'].diff().values
        df['spread_momentum'] = df['bid_ask_spread_rel'].diff().rolling(4).mean().values
        added_count += 2

    # 5. Volume-weighted imbalance
    if 'imbalance_5' in df.columns and 'volume' in df.columns:
        df['imbalance_volume_weighted'] = (df['imbalance_5'].values * np.log1p(df['volume'].values))
        added_count += 1

    # 6. RSI momentum
    if 'rsi_14' in df.columns and 'rsi_28' in df.columns:
        df['rsi_diff'] = (df['rsi_14'].values - df['rsi_28'].values)
        df['rsi_14_momentum'] = df['rsi_14'].diff(4).values
        added_count += 2

    # 7. Composite signals
    if all(f in df.columns for f in ['imbalance_5', 'depth_imbalance_ratio', 'volume_delta_5']):
        # Комбинированный сигнал из top-3 коррелированных фич
        df['composite_signal'] = (
            df['imbalance_5'].rank(pct=True).values * 0.4 +
            df['depth_imbalance_ratio'].rank(pct=True).values * 0.4 +
            df['volume_delta_5'].rank(pct=True).values * 0.2
        )
        added_count += 1

    # 8. Orderbook pressure change
    if 'smart_money_index' in df.columns:
        df['smart_money_momentum'] = df['smart_money_index'].diff().values
        df['smart_money_acceleration'] = df['smart_money_index'].diff().diff().values
        added_count += 2

    # 9. Trade intensity ratio
    if 'trade_arrival_rate' in df.columns and 'quote_intensity' in df.columns:
        df['trade_quote_ratio'] = (df['trade_arrival_rate'].values / (df['quote_intensity'].values + 1e-10))
        added_count += 1

    # 10. Volatility regime
    if 'orderbook_volatility' in df.columns:
        vol_ma = df['orderbook_volatility'].rolling(20).mean().values
        df['volatility_regime'] = (df['orderbook_volatility'].values / (vol_ma + 1e-10))
        added_count += 1

    # Заполняем NaN в derived features (bfill для начала серии)
    existing_derived = [c for c in derived_feature_names if c in df.columns]
    if existing_derived:
        df[existing_derived] = df[existing_derived].fillna(method='bfill')
        # Если остались NaN (напр. все значения NaN), заполняем 0
        df[existing_derived] = df[existing_derived].fillna(0)

    if verbose:
        print(f"   ✓ Добавлено {added_count} derived features")

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
    config: LabelingConfig,
    horizon: int = 60
) -> Tuple[int, float]:
    """
    Применяет Triple Barrier логику для определения label.

    Args:
        current_price: Текущая цена
        future_price: Будущая цена
        atr: Average True Range на момент входа
        config: Конфигурация labeling
        horizon: Горизонт предсказания в секундах (для scaling)

    Returns:
        (label, movement) где label: 0=SELL, 1=HOLD, 2=BUY
    """
    if current_price <= 0:
        return 1, 0.0

    movement = (future_price - current_price) / current_price
    movement_pct = abs(movement) * 100

    # Определяем порог
    if config.use_fixed_pct:
        # Режим фиксированного процента с horizon scaling
        threshold_pct = config.get_scaled_threshold(horizon)
        threshold = threshold_pct / 100
    elif atr is not None and atr > 0 and not np.isnan(atr):
        # Режим ATR - адаптивный к волатильности символа
        # threshold = (ATR / price) * multiplier = relative_volatility * multiplier
        threshold = (atr / current_price) * config.tp_multiplier
    else:
        # Fallback с horizon scaling
        threshold_pct = config.get_scaled_threshold(horizon)
        threshold = threshold_pct / 100

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

        if self.config.use_fixed_pct:
            print(f"  • РЕЖИМ: ФИКСИРОВАННЫЙ ПРОЦЕНТ")
            if self.config.use_horizon_scaling:
                print(f"  • Базовый порог: {self.config.fixed_threshold_pct}% (для {self.config.base_horizon}s)")
                print(f"  • √T Scaling: ВКЛЮЧЁН")
                for h in self.config.horizons:
                    scaled = self.config.get_scaled_threshold(h)
                    print(f"    - {h}s → {scaled:.3f}%")
            else:
                print(f"  • Порог: {self.config.fixed_threshold_pct}% для ВСЕХ горизонтов")
        else:
            print(f"  • РЕЖИМ: АДАПТИВНЫЙ ATR")
            print(f"  • TP множитель: {self.config.tp_multiplier}x ATR")
            print(f"  • SL множитель: {self.config.sl_multiplier}x ATR")

        print(f"  • Min movement: {self.config.min_movement_pct}%")
        print(f"  • Фильтр плоских символов: HOLD > {self.config.max_hold_pct}%")
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
        skipped_symbols = []  # Символы с высоким HOLD

        for symbol in symbols:
            print(f"\n{'─' * 70}")
            print(f"Обработка {symbol}")
            print(f"{'─' * 70}")

            symbol_df = df[df['symbol'] == symbol].copy()
            processed_df, hold_pct_by_horizon = self._process_symbol(symbol, symbol_df)

            # Проверяем, является ли символ "плоским" (HOLD > max_hold_pct во ВСЕХ горизонтах)
            is_flat = all(
                hold_pct > self.config.max_hold_pct
                for hold_pct in hold_pct_by_horizon.values()
            )

            if is_flat:
                min_hold = min(hold_pct_by_horizon.values())
                print(f"\n   ⚠️ ПРОПУЩЕН: HOLD > {self.config.max_hold_pct}% во всех горизонтах "
                      f"(min HOLD = {min_hold:.1f}%)")
                skipped_symbols.append((symbol, len(symbol_df), hold_pct_by_horizon))
                self.stats['skipped_symbols'] = self.stats.get('skipped_symbols', [])
                self.stats['skipped_symbols'].append(symbol)
            else:
                # ВАЖНО: Добавляем lagged/derived features ДО объединения символов,
                # чтобы shift() работал только внутри одного символа
                processed_df = add_lagged_features(processed_df, verbose=False)
                processed_df = add_derived_features(processed_df, verbose=False)
                all_processed.append(processed_df)

        # Выводим сводку по пропущенным символам
        if skipped_symbols:
            print(f"\n{'=' * 70}")
            print(f"⚠️ ПРОПУЩЕНО {len(skipped_symbols)} символов с HOLD > {self.config.max_hold_pct}%:")
            total_skipped = 0
            for sym, count, hold_stats in skipped_symbols:
                holds_str = ", ".join([f"{h}s:{p:.0f}%" for h, p in hold_stats.items()])
                print(f"   • {sym}: {count:,} семплов ({holds_str})")
                total_skipped += count
            print(f"   Всего пропущено: {total_skipped:,} семплов")
            print(f"{'=' * 70}")

        # Объединяем результаты
        if not all_processed:
            print("\n❌ Нет данных после фильтрации!")
            return

        final_df = pd.concat(all_processed, ignore_index=True)

        # Lagged и derived features уже добавлены для каждого символа отдельно
        # Это критически важно: shift() должен работать только внутри символа!
        n_lag_cols = len([c for c in final_df.columns if '_lag' in c])
        n_derived_cols = len([c for c in final_df.columns if c in [
            'imbalance_5_change', 'imbalance_5_change_pct', 'imbalance_5_momentum',
            'imbalance_ratio_5_10', 'imbalance_vol_adjusted', 'spread_change',
            'spread_momentum', 'imbalance_volume_weighted', 'rsi_diff', 'rsi_14_momentum',
            'composite_signal', 'smart_money_momentum', 'smart_money_acceleration',
            'trade_quote_ratio', 'volatility_regime'
        ]])
        print(f"\n📊 Добавлено Lagged/Derived Features (per-symbol):")
        print(f"   ✓ {n_lag_cols} lagged features")
        print(f"   ✓ {n_derived_cols} derived features")

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

    def _process_symbol(self, symbol: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, float]]:
        """
        Обработка данных для одного символа.

        Returns:
            (DataFrame с метками, словарь {horizon: hold_pct})
        """
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

        # Статистика HOLD% по горизонтам для фильтрации
        hold_pct_by_horizon = {}

        # Обрабатываем каждый горизонт
        for horizon in self.config.horizons:
            label_col = f'future_direction_{horizon}s'
            movement_col = f'future_movement_{horizon}s'

            # Показываем scaled threshold если включён scaling
            if self.config.use_fixed_pct and self.config.use_horizon_scaling:
                scaled_thresh = self.config.get_scaled_threshold(horizon)
                print(f"\n   Горизонт {horizon}s (порог: {scaled_thresh:.3f}%):")
            else:
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
                        current_price, future_price, current_atr, self.config, horizon
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
            hold_pct = 100 * hold / n if n > 0 else 0

            # Сохраняем HOLD% для фильтрации
            hold_pct_by_horizon[horizon] = hold_pct

            print(f"      Labeled: {labeled_count:,}/{n:,} ({100*labeled_count/n:.1f}%)")
            print(f"      Distribution: SELL={sell:,} ({100*sell/n:.1f}%) | "
                  f"HOLD={hold:,} ({hold_pct:.1f}%) | BUY={buy:,} ({100*buy/n:.1f}%)")

            # Обновляем общую статистику
            if horizon == 300:  # Основной горизонт для статистики
                self.stats['total_samples'] += n
                self.stats['labeled_samples'] += labeled_count
                # Накапливаем распределение по всем символам (а не перезаписываем!)
                self.stats['label_distribution']['SELL'] = self.stats['label_distribution'].get('SELL', 0) + sell
                self.stats['label_distribution']['HOLD'] = self.stats['label_distribution'].get('HOLD', 0) + hold
                self.stats['label_distribution']['BUY'] = self.stats['label_distribution'].get('BUY', 0) + buy

        return df, hold_pct_by_horizon

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

        # Информация о пропущенных символах
        skipped = self.stats.get('skipped_symbols', [])
        if skipped:
            print(f"\n🚫 Пропущено символов (HOLD > {self.config.max_hold_pct}%): {len(skipped)}")

        print(f"\n✓ Всего семплов (после фильтрации): {self.stats['total_samples']:,}")
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
                print("   Рассмотрите уменьшение --fixed-pct или включение --no-scaling")
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
    parser.add_argument(
        '--fixed-pct',
        type=float,
        default=None,
        help='Использовать фиксированный %% порог для ВСЕХ символов (игнорирует ATR). '
             'Например: --fixed-pct 0.1 = 0.1%% порог (базовый для 60s)'
    )
    parser.add_argument(
        '--no-scaling',
        action='store_true',
        help='Отключить √T scaling порогов по горизонтам. '
             'По умолчанию порог масштабируется: 180s=×√3, 300s=×√5'
    )
    parser.add_argument(
        '--max-hold',
        type=float,
        default=85.0,
        help='Максимальный %% HOLD для включения символа (default: 85). '
             'Символы с HOLD > max-hold во ВСЕХ горизонтах будут пропущены'
    )

    args = parser.parse_args()

    # Создаем конфигурацию
    use_fixed = args.fixed_pct is not None
    fixed_threshold = args.fixed_pct if use_fixed else 0.15

    config = LabelingConfig(
        horizons=[60, 180, 300],  # 1, 3, 5 минут
        tp_multiplier=args.tp_mult,
        sl_multiplier=args.sl_mult,
        min_movement_pct=args.min_movement,
        use_fixed_pct=use_fixed,
        fixed_threshold_pct=fixed_threshold,
        use_horizon_scaling=not args.no_scaling,
        max_hold_pct=args.max_hold
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
