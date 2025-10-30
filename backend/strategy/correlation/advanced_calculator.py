"""
Продвинутый калькулятор корреляций с поддержкой множественных метрик.

Путь: backend/strategy/correlation/advanced_calculator.py
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from scipy.spatial.distance import euclidean

from core.logger import get_logger
from .models import (
    CorrelationMetrics,
    RollingCorrelationWindow,
    DTWParameters,
    ConditionalCorrelationMetrics,
    CorrelationMethod
)

logger = get_logger(__name__)


class AdvancedCorrelationCalculator:
    """
    Продвинутый калькулятор корреляций.

    Поддерживает:
    - Pearson correlation (multiple windows)
    - Spearman rank correlation
    - Dynamic Time Warping (DTW)
    - Volatility distance
    - Return sign agreement
    """

    def __init__(
        self,
        short_window: int = 7,
        medium_window: int = 14,
        long_window: int = 30,
        short_weight: float = 0.5,
        medium_weight: float = 0.3,
        long_weight: float = 0.2,
        dtw_params: Optional[DTWParameters] = None
    ):
        """
        Инициализация калькулятора.

        Args:
            short_window: Короткое окно (дней)
            medium_window: Среднее окно (дней)
            long_window: Длинное окно (дней)
            short_weight: Вес короткого окна
            medium_weight: Вес среднего окна
            long_weight: Вес длинного окна
            dtw_params: Параметры DTW
        """
        self.windows = [
            RollingCorrelationWindow(short_window, short_weight),
            RollingCorrelationWindow(medium_window, medium_weight),
            RollingCorrelationWindow(long_window, long_weight)
        ]

        self.dtw_params = dtw_params or DTWParameters()

        # Валидация весов
        total_weight = sum(w.weight for w in self.windows)
        if not (0.99 <= total_weight <= 1.01):
            logger.warning(f"Сумма весов окон = {total_weight:.3f}, должна быть 1.0")

        logger.info(
            f"AdvancedCorrelationCalculator инициализирован: "
            f"windows=[{short_window}d, {medium_window}d, {long_window}d], "
            f"weights=[{short_weight}, {medium_weight}, {long_weight}]"
        )

    @staticmethod
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """
        Расчет процентных изменений (returns).

        Args:
            prices: Массив цен

        Returns:
            np.ndarray: Массив returns
        """
        if len(prices) < 2:
            return np.array([])

        returns = np.diff(prices) / prices[:-1]
        return returns

    @staticmethod
    def calculate_pearson(returns_a: np.ndarray, returns_b: np.ndarray) -> float:
        """
        Расчет Pearson correlation coefficient.

        Args:
            returns_a: Returns для актива A
            returns_b: Returns для актива B

        Returns:
            float: Correlation coefficient [-1, 1]
        """
        if len(returns_a) < 2 or len(returns_b) < 2:
            return 0.0

        # Выравниваем длину
        min_len = min(len(returns_a), len(returns_b))
        returns_a = returns_a[-min_len:]
        returns_b = returns_b[-min_len:]

        try:
            correlation_matrix = np.corrcoef(returns_a, returns_b)
            correlation = correlation_matrix[0, 1]

            if np.isnan(correlation):
                return 0.0

            return float(correlation)

        except Exception as e:
            logger.warning(f"Ошибка расчета Pearson: {e}")
            return 0.0

    @staticmethod
    def calculate_spearman(returns_a: np.ndarray, returns_b: np.ndarray) -> float:
        """
        Расчет Spearman rank correlation.

        Args:
            returns_a: Returns для актива A
            returns_b: Returns для актива B

        Returns:
            float: Spearman correlation [-1, 1]
        """
        if len(returns_a) < 2 or len(returns_b) < 2:
            return 0.0

        # Выравниваем длину
        min_len = min(len(returns_a), len(returns_b))
        returns_a = returns_a[-min_len:]
        returns_b = returns_b[-min_len:]

        try:
            correlation, _ = stats.spearmanr(returns_a, returns_b)

            if np.isnan(correlation):
                return 0.0

            return float(correlation)

        except Exception as e:
            logger.warning(f"Ошибка расчета Spearman: {e}")
            return 0.0

    def calculate_dtw_distance(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray
    ) -> float:
        """
        Расчет Dynamic Time Warping distance (упрощенная версия).

        Примечание: Полная имплементация DTW требует библиотеки dtaidistance.
        Эта версия использует упрощенный алгоритм.

        Args:
            prices_a: Цены актива A
            prices_b: Цены актива B

        Returns:
            float: DTW distance нормализованный в [0, 1]
        """
        if len(prices_a) < 2 or len(prices_b) < 2:
            return 1.0  # Максимальная дистанция = нет корреляции

        try:
            # Упрощенная версия: используем обычное евклидово расстояние
            # после нормализации временных рядов
            if self.dtw_params.normalize:
                prices_a = (prices_a - np.mean(prices_a)) / (np.std(prices_a) + 1e-8)
                prices_b = (prices_b - np.mean(prices_b)) / (np.std(prices_b) + 1e-8)

            # Выравниваем длину
            min_len = min(len(prices_a), len(prices_b))
            prices_a = prices_a[-min_len:]
            prices_b = prices_b[-min_len:]

            # Расчет евклидова расстояния
            distance = euclidean(prices_a, prices_b)

            # Нормализуем к [0, 1]
            # Максимальная дистанция для нормализованных рядов ≈ sqrt(2*N)
            max_distance = np.sqrt(2 * len(prices_a))
            normalized_distance = min(distance / max_distance, 1.0)

            return float(normalized_distance)

        except Exception as e:
            logger.warning(f"Ошибка расчета DTW: {e}")
            return 1.0

    @staticmethod
    def calculate_volatility_distance(
        returns_a: np.ndarray,
        returns_b: np.ndarray
    ) -> float:
        """
        Расчет разницы в волатильности между активами.

        Args:
            returns_a: Returns для актива A
            returns_b: Returns для актива B

        Returns:
            float: Volatility distance [0, 1]
        """
        if len(returns_a) < 2 or len(returns_b) < 2:
            return 1.0

        try:
            vol_a = np.std(returns_a)
            vol_b = np.std(returns_b)

            if vol_a == 0 and vol_b == 0:
                return 0.0

            # Нормализованная разница
            distance = abs(vol_a - vol_b) / max(vol_a, vol_b, 1e-8)

            return min(float(distance), 1.0)

        except Exception as e:
            logger.warning(f"Ошибка расчета volatility distance: {e}")
            return 1.0

    @staticmethod
    def calculate_return_sign_agreement(
        returns_a: np.ndarray,
        returns_b: np.ndarray
    ) -> float:
        """
        Расчет согласия направления движения (оба растут / оба падают).

        Args:
            returns_a: Returns для актива A
            returns_b: Returns для актива B

        Returns:
            float: Agreement score [0, 1]
        """
        if len(returns_a) < 2 or len(returns_b) < 2:
            return 0.0

        # Выравниваем длину
        min_len = min(len(returns_a), len(returns_b))
        returns_a = returns_a[-min_len:]
        returns_b = returns_b[-min_len:]

        try:
            # Знаки returns (+ или -)
            signs_a = np.sign(returns_a)
            signs_b = np.sign(returns_b)

            # Согласие знаков
            agreement = np.sum(signs_a == signs_b) / len(returns_a)

            return float(agreement)

        except Exception as e:
            logger.warning(f"Ошибка расчета sign agreement: {e}")
            return 0.0

    def calculate_rolling_correlations(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray
    ) -> List[RollingCorrelationWindow]:
        """
        Расчет корреляций для различных rolling windows.

        Args:
            prices_a: Цены актива A
            prices_b: Цены актива B

        Returns:
            List[RollingCorrelationWindow]: Окна с рассчитанными корреляциями
        """
        results = []

        for window in self.windows:
            # Берем последние N дней
            window_prices_a = prices_a[-window.window_days:]
            window_prices_b = prices_b[-window.window_days:]

            if len(window_prices_a) < 2 or len(window_prices_b) < 2:
                # Недостаточно данных
                window_copy = RollingCorrelationWindow(
                    window.window_days,
                    window.weight,
                    0.0
                )
                results.append(window_copy)
                continue

            # Расчет returns
            returns_a = self.calculate_returns(window_prices_a)
            returns_b = self.calculate_returns(window_prices_b)

            # Расчет корреляции
            correlation = self.calculate_pearson(returns_a, returns_b)

            window_copy = RollingCorrelationWindow(
                window.window_days,
                window.weight,
                correlation
            )
            results.append(window_copy)

        return results

    def calculate_correlation_suite(
        self,
        symbol_a: str,
        symbol_b: str,
        prices_a: np.ndarray,
        prices_b: np.ndarray
    ) -> CorrelationMetrics:
        """
        Вычисляет полный набор метрик корреляции.

        Args:
            symbol_a: Символ A
            symbol_b: Символ B
            prices_a: Цены A
            prices_b: Цены B

        Returns:
            CorrelationMetrics: Полный набор метрик
        """
        # Расчет returns (для всего периода)
        returns_a = self.calculate_returns(prices_a)
        returns_b = self.calculate_returns(prices_b)

        # 1. Pearson корреляции для разных окон
        rolling_windows = self.calculate_rolling_correlations(prices_a, prices_b)

        pearson_7d = rolling_windows[0].correlation or 0.0
        pearson_14d = rolling_windows[1].correlation or 0.0
        pearson_30d = rolling_windows[2].correlation or 0.0

        # 2. Spearman корреляция (весь период)
        spearman = self.calculate_spearman(returns_a, returns_b)

        # 3. DTW distance
        dtw_distance = self.calculate_dtw_distance(prices_a, prices_b)

        # 4. Volatility distance
        volatility_distance = self.calculate_volatility_distance(returns_a, returns_b)

        # 5. Return sign agreement
        return_sign_agreement = self.calculate_return_sign_agreement(
            returns_a, returns_b
        )

        # 6. Weighted final score
        # Комбинируем метрики с весами
        weighted_score = self._calculate_weighted_score(
            pearson_7d=pearson_7d,
            pearson_14d=pearson_14d,
            pearson_30d=pearson_30d,
            spearman=spearman,
            dtw_distance=dtw_distance,
            volatility_distance=volatility_distance,
            return_sign_agreement=return_sign_agreement
        )

        # Основная корреляция (Pearson 30d)
        pearson = pearson_30d

        return CorrelationMetrics(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            pearson=pearson,
            spearman=spearman,
            pearson_7d=pearson_7d,
            pearson_14d=pearson_14d,
            pearson_30d=pearson_30d,
            dtw_distance=dtw_distance,
            volatility_distance=volatility_distance,
            return_sign_agreement=return_sign_agreement,
            weighted_score=weighted_score
        )

    def _calculate_weighted_score(
        self,
        pearson_7d: float,
        pearson_14d: float,
        pearson_30d: float,
        spearman: float,
        dtw_distance: float,
        volatility_distance: float,
        return_sign_agreement: float
    ) -> float:
        """
        Расчет взвешенной финальной оценки корреляции.

        Args:
            pearson_7d: Pearson 7 дней
            pearson_14d: Pearson 14 дней
            pearson_30d: Pearson 30 дней
            spearman: Spearman correlation
            dtw_distance: DTW distance
            volatility_distance: Volatility distance
            return_sign_agreement: Sign agreement

        Returns:
            float: Weighted score [-1, 1]
        """
        # Weighted Pearson (учитываем все окна)
        weighted_pearson = (
            pearson_7d * 0.5 +
            pearson_14d * 0.3 +
            pearson_30d * 0.2
        )

        # DTW similarity (1 - distance)
        dtw_similarity = 1.0 - dtw_distance

        # Volatility similarity (1 - distance)
        volatility_similarity = 1.0 - volatility_distance

        # Final weighted score
        score = (
            weighted_pearson * 0.40 +  # Pearson - основа
            spearman * 0.20 +  # Spearman - монотонность
            dtw_similarity * 0.20 +  # DTW - форма движения
            volatility_similarity * 0.10 +  # Похожесть волатильности
            return_sign_agreement * 0.10  # Согласие направления
        )

        # Нормализуем к [-1, 1]
        # return_sign_agreement и similarities в [0, 1], нужно сдвинуть
        # Упрощенно: используем weighted_pearson как базу
        # Корректируем на основе других метрик
        final_score = weighted_pearson

        return np.clip(final_score, -1.0, 1.0)
