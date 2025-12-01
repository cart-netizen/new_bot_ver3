#!/usr/bin/env python3
"""
Triple Barrier Labeling Method - Industry Standard для ML в трейдинге.

Реализует метод разметки данных из книги López de Prado "Advances in Financial Machine Learning".

Triple Barrier Method использует три барьера:
1. Верхний (Take Profit): цена достигла profit target
2. Нижний (Stop Loss): цена достигла stop loss
3. Вертикальный (Timeout): истекло max_holding_period

Преимущества над fixed horizon labeling:
- Адаптивность к волатильности (через ATR)
- Реалистичное моделирование торговых решений
- Лучшее распределение классов

Файл: backend/ml_engine/features/labeling.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import IntEnum
import numpy as np
import pandas as pd

from backend.core.logger import get_logger

logger = get_logger(__name__)


class Direction(IntEnum):
    """Направление движения цены."""
    SELL = 0   # Нижний барьер достигнут первым → short был бы профитен
    HOLD = 1   # Timeout → нет чёткого направления
    BUY = 2    # Верхний барьер достигнут первым → long был бы профитен


class BarrierHitType(IntEnum):
    """Тип достигнутого барьера."""
    TIMEOUT = 0       # Вертикальный барьер (max_holding_period)
    TAKE_PROFIT = 1   # Верхний барьер
    STOP_LOSS = -1    # Нижний барьер


@dataclass
class TripleBarrierConfig:
    """
    Конфигурация Triple Barrier Method.

    Параметры определяют размеры барьеров относительно ATR (волатильности).
    """
    # === Множители барьеров относительно ATR ===
    tp_multiplier: float = 1.5  # Take profit = entry + tp_mult * ATR
    sl_multiplier: float = 1.0  # Stop loss = entry - sl_mult * ATR

    # === Временные параметры ===
    max_holding_period: int = 24  # Максимальное время удержания (в барах)

    # === ATR параметры ===
    atr_period: int = 14  # Период для расчёта ATR
    min_atr_pct: float = 0.001  # Минимальный ATR (0.1% от цены)

    # === Опциональные параметры ===
    use_symmetric_barriers: bool = False  # Симметричные TP/SL (tp_mult = sl_mult)
    dynamic_holding_period: bool = False  # Адаптировать период к волатильности
    vol_scaling_factor: float = 1.0  # Коэффициент масштабирования по волатильности

    # === Пороги для fixed threshold fallback ===
    fixed_threshold: float = 0.0001  # 0.01% для fallback если ATR недоступен

    def __post_init__(self):
        """Валидация параметров."""
        if self.tp_multiplier <= 0:
            raise ValueError(f"tp_multiplier must be > 0, got {self.tp_multiplier}")
        if self.sl_multiplier <= 0:
            raise ValueError(f"sl_multiplier must be > 0, got {self.sl_multiplier}")
        if self.max_holding_period <= 0:
            raise ValueError(f"max_holding_period must be > 0, got {self.max_holding_period}")
        if self.atr_period <= 0:
            raise ValueError(f"atr_period must be > 0, got {self.atr_period}")

        if self.use_symmetric_barriers:
            self.sl_multiplier = self.tp_multiplier


@dataclass
class LabelingResult:
    """Результат разметки данных."""
    labels: np.ndarray  # Метки направления (0=SELL, 1=HOLD, 2=BUY)
    returns: np.ndarray  # Доходности при достижении барьера
    hit_times: np.ndarray  # Время до достижения барьера (в барах)
    hit_types: np.ndarray  # Тип достигнутого барьера
    barriers_tp: np.ndarray  # Уровни take profit
    barriers_sl: np.ndarray  # Уровни stop loss
    statistics: Dict  # Статистика разметки


class TripleBarrierLabeler:
    """
    Triple Barrier Labeling Method (López de Prado, 2018).

    Три барьера:
    1. Верхний (Take Profit): цена достигла profit target
    2. Нижний (Stop Loss): цена достигла stop loss
    3. Вертикальный (Timeout): истекло max_holding_period

    Labels:
    - 2 (BUY): Верхний барьер достигнут первым → long сигнал был бы профитен
    - 0 (SELL): Нижний барьер достигнут первым → short сигнал был бы профитен
    - 1 (HOLD): Timeout → нет чёткого направления

    Использование:
        >>> labeler = TripleBarrierLabeler(config)
        >>> result = labeler.generate_labels(df)
        >>> labels = result.labels  # Массив меток 0, 1, 2
    """

    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        """
        Инициализация labeler.

        Args:
            config: Конфигурация Triple Barrier. Если None - используются defaults.
        """
        self.config = config or TripleBarrierConfig()
        logger.info(
            f"TripleBarrierLabeler initialized: "
            f"TP={self.config.tp_multiplier}x ATR, "
            f"SL={self.config.sl_multiplier}x ATR, "
            f"max_hold={self.config.max_holding_period}"
        )

    def compute_atr(
        self,
        df: pd.DataFrame,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close'
    ) -> pd.Series:
        """
        Вычислить Average True Range.

        ATR измеряет волатильность и используется для адаптивных барьеров.

        Args:
            df: DataFrame с данными
            high_col: Название колонки с high
            low_col: Название колонки с low
            close_col: Название колонки с close

        Returns:
            Series с ATR значениями
        """
        high = df[high_col]
        low = df[low_col]
        close = df[close_col].shift(1)

        # True Range = max(H-L, |H-Cp|, |L-Cp|)
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = EMA или SMA от True Range
        atr = true_range.rolling(window=self.config.atr_period).mean()

        # Минимальный ATR (для избежания слишком узких барьеров)
        min_atr = df[close_col] * self.config.min_atr_pct
        atr = atr.clip(lower=min_atr)

        return atr

    def generate_labels(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        atr_col: Optional[str] = None,
        include_metadata: bool = True
    ) -> LabelingResult:
        """
        Генерация меток с использованием Triple Barrier Method.

        Args:
            df: DataFrame с OHLC данными
            price_col: Колонка с ценой входа (обычно close)
            high_col: Колонка с high (для проверки TP)
            low_col: Колонка с low (для проверки SL)
            atr_col: Колонка с ATR (если None - вычисляется)
            include_metadata: Включить метаданные в результат

        Returns:
            LabelingResult с метками и статистикой
        """
        n = len(df)

        # Инициализируем массивы
        labels = np.full(n, Direction.HOLD, dtype=np.int64)
        returns = np.full(n, np.nan, dtype=np.float64)
        hit_times = np.zeros(n, dtype=np.int32)
        hit_types = np.zeros(n, dtype=np.int32)
        barriers_tp = np.zeros(n, dtype=np.float64)
        barriers_sl = np.zeros(n, dtype=np.float64)

        # Получаем ATR
        if atr_col and atr_col in df.columns:
            atr = df[atr_col].values
            logger.info(f"Using existing ATR column: {atr_col}")
        elif 'atr_14' in df.columns:
            atr = df['atr_14'].values
            logger.info("Using existing atr_14 column")
        elif high_col in df.columns and low_col in df.columns:
            atr = self.compute_atr(df, high_col, low_col, price_col).values
            logger.info("Computed ATR from OHLC data")
        else:
            # Fallback: используем fixed threshold
            logger.warning(
                f"ATR not available, using fixed threshold: {self.config.fixed_threshold}"
            )
            atr = df[price_col].values * self.config.fixed_threshold * 10

        # Получаем ценовые данные
        close = df[price_col].values

        # Используем high/low если доступны, иначе close
        if high_col in df.columns and low_col in df.columns:
            high = df[high_col].values
            low = df[low_col].values
            use_hl = True
        else:
            high = close
            low = close
            use_hl = False
            logger.warning("High/Low not available, using close prices for barrier checking")

        logger.info(f"Generating labels for {n:,} samples...")

        # Основной цикл по каждому sample
        valid_count = 0
        for i in range(n - self.config.max_holding_period):
            entry_price = close[i]
            entry_atr = atr[i]

            # Пропускаем если ATR = NaN или цена невалидна
            if np.isnan(entry_atr) or np.isnan(entry_price) or entry_price <= 0:
                continue

            valid_count += 1

            # Вычисляем барьеры
            take_profit = entry_price + (self.config.tp_multiplier * entry_atr)
            stop_loss = entry_price - (self.config.sl_multiplier * entry_atr)

            barriers_tp[i] = take_profit
            barriers_sl[i] = stop_loss

            # Ищем достижение барьеров
            label = Direction.HOLD
            hit_time = self.config.max_holding_period
            hit_type = BarrierHitType.TIMEOUT
            final_price = close[min(i + self.config.max_holding_period, n - 1)]

            for j in range(1, self.config.max_holding_period + 1):
                idx = i + j
                if idx >= n:
                    break

                # Проверяем верхний барьер (Take Profit)
                check_high = high[idx] if use_hl else close[idx]
                if check_high >= take_profit:
                    label = Direction.BUY  # Long был бы профитен
                    hit_time = j
                    hit_type = BarrierHitType.TAKE_PROFIT
                    final_price = take_profit
                    break

                # Проверяем нижний барьер (Stop Loss)
                check_low = low[idx] if use_hl else close[idx]
                if check_low <= stop_loss:
                    label = Direction.SELL  # Short был бы профитен
                    hit_time = j
                    hit_type = BarrierHitType.STOP_LOSS
                    final_price = stop_loss
                    break

            # Записываем результаты
            labels[i] = label
            returns[i] = (final_price - entry_price) / entry_price
            hit_times[i] = hit_time
            hit_types[i] = hit_type

        # Последние max_holding_period samples = HOLD (недостаточно данных)
        labels[-(self.config.max_holding_period):] = Direction.HOLD

        # Собираем статистику
        valid_labels = labels[labels != Direction.HOLD]  # Только BUY и SELL
        statistics = {
            'total_samples': n,
            'valid_samples': valid_count,
            'label_distribution': {
                'SELL': int((labels == Direction.SELL).sum()),
                'HOLD': int((labels == Direction.HOLD).sum()),
                'BUY': int((labels == Direction.BUY).sum())
            },
            'hit_type_distribution': {
                'TIMEOUT': int((hit_types == BarrierHitType.TIMEOUT).sum()),
                'TAKE_PROFIT': int((hit_types == BarrierHitType.TAKE_PROFIT).sum()),
                'STOP_LOSS': int((hit_types == BarrierHitType.STOP_LOSS).sum())
            },
            'avg_hit_time': float(np.nanmean(hit_times[hit_times > 0])) if (hit_times > 0).any() else 0,
            'avg_return': float(np.nanmean(returns)) if not np.all(np.isnan(returns)) else 0,
            'config': {
                'tp_multiplier': self.config.tp_multiplier,
                'sl_multiplier': self.config.sl_multiplier,
                'max_holding_period': self.config.max_holding_period,
                'atr_period': self.config.atr_period
            }
        }

        # Логируем статистику
        logger.info("\n" + "=" * 60)
        logger.info("TRIPLE BARRIER LABELING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total samples: {statistics['total_samples']:,}")
        logger.info(f"Valid samples: {statistics['valid_samples']:,}")
        logger.info(f"Label distribution:")
        for label_name, count in statistics['label_distribution'].items():
            pct = 100 * count / n
            logger.info(f"  • {label_name}: {count:,} ({pct:.1f}%)")
        logger.info(f"Hit type distribution:")
        for hit_name, count in statistics['hit_type_distribution'].items():
            pct = 100 * count / n if n > 0 else 0
            logger.info(f"  • {hit_name}: {count:,} ({pct:.1f}%)")
        logger.info(f"Average hit time: {statistics['avg_hit_time']:.1f} bars")
        logger.info(f"Average return: {statistics['avg_return']*100:.4f}%")
        logger.info("=" * 60 + "\n")

        return LabelingResult(
            labels=labels,
            returns=returns,
            hit_times=hit_times,
            hit_types=hit_types,
            barriers_tp=barriers_tp,
            barriers_sl=barriers_sl,
            statistics=statistics
        )

    def generate_labels_from_arrays(
        self,
        close: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        atr: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Генерация меток из numpy массивов (для интеграции с legacy кодом).

        Args:
            close: Массив цен закрытия
            high: Массив high (опционально)
            low: Массив low (опционально)
            atr: Массив ATR (опционально, вычисляется если не передан)

        Returns:
            (labels, statistics) - метки и статистика
        """
        # Создаем временный DataFrame
        df = pd.DataFrame({'close': close})

        if high is not None:
            df['high'] = high
        if low is not None:
            df['low'] = low
        if atr is not None:
            df['atr'] = atr

        result = self.generate_labels(
            df,
            price_col='close',
            high_col='high' if high is not None else 'close',
            low_col='low' if low is not None else 'close',
            atr_col='atr' if atr is not None else None
        )

        return result.labels, result.statistics


class FixedThresholdLabeler:
    """
    Legacy fixed threshold labeler для обратной совместимости.

    Использует фиксированный порог для определения направления.
    Рекомендуется использовать TripleBarrierLabeler вместо этого класса.
    """

    def __init__(self, threshold: float = 0.0001):
        """
        Args:
            threshold: Порог для определения направления (в долях)
        """
        self.threshold = threshold
        logger.warning(
            "FixedThresholdLabeler is deprecated. "
            "Use TripleBarrierLabeler for better results."
        )

    def generate_labels(
        self,
        price_changes: np.ndarray
    ) -> np.ndarray:
        """
        Генерация меток на основе фиксированного порога.

        Args:
            price_changes: Массив изменений цены (в долях)

        Returns:
            Массив меток (0=SELL, 1=HOLD, 2=BUY)
        """
        labels = np.ones(len(price_changes), dtype=np.int64)  # Default = HOLD

        labels[price_changes > self.threshold] = Direction.BUY
        labels[price_changes < -self.threshold] = Direction.SELL

        return labels


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_triple_barrier_labeler(
    tp_multiplier: float = 1.5,
    sl_multiplier: float = 1.0,
    max_holding_period: int = 24,
    atr_period: int = 14
) -> TripleBarrierLabeler:
    """
    Factory function для создания TripleBarrierLabeler.

    Args:
        tp_multiplier: Множитель take profit (x ATR)
        sl_multiplier: Множитель stop loss (x ATR)
        max_holding_period: Максимальное время удержания
        atr_period: Период ATR

    Returns:
        Настроенный TripleBarrierLabeler
    """
    config = TripleBarrierConfig(
        tp_multiplier=tp_multiplier,
        sl_multiplier=sl_multiplier,
        max_holding_period=max_holding_period,
        atr_period=atr_period
    )
    return TripleBarrierLabeler(config)


def create_conservative_labeler() -> TripleBarrierLabeler:
    """
    Создать консервативный labeler с узкими барьерами.

    Подходит для:
    - Высоковолатильных рынков
    - Скальпинга
    - Когда нужно больше сигналов
    """
    config = TripleBarrierConfig(
        tp_multiplier=1.0,
        sl_multiplier=1.0,
        max_holding_period=12,
        use_symmetric_barriers=True
    )
    return TripleBarrierLabeler(config)


def create_aggressive_labeler() -> TripleBarrierLabeler:
    """
    Создать агрессивный labeler с широкими барьерами.

    Подходит для:
    - Низковолатильных рынков
    - Свинг-трейдинга
    - Когда нужны более уверенные сигналы
    """
    config = TripleBarrierConfig(
        tp_multiplier=2.0,
        sl_multiplier=1.5,
        max_holding_period=48,
        use_symmetric_barriers=False
    )
    return TripleBarrierLabeler(config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TRIPLE BARRIER LABELING - EXAMPLE")
    print("=" * 80)

    # Создаем тестовые данные
    np.random.seed(42)
    n = 1000

    # Симулируем ценовые данные
    returns = np.random.normal(0, 0.01, n)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))

    df = pd.DataFrame({
        'close': close,
        'high': high,
        'low': low
    })

    # Создаем labeler и генерируем метки
    labeler = create_triple_barrier_labeler(
        tp_multiplier=1.5,
        sl_multiplier=1.0,
        max_holding_period=24
    )

    result = labeler.generate_labels(df)

    print("\n✅ Labeling complete!")
    print(f"  • Labels shape: {result.labels.shape}")
    print(f"  • BUY signals: {(result.labels == Direction.BUY).sum()}")
    print(f"  • SELL signals: {(result.labels == Direction.SELL).sum()}")
    print(f"  • HOLD signals: {(result.labels == Direction.HOLD).sum()}")
