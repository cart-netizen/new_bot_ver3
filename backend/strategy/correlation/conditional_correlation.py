"""
Conditional Correlation Analysis - корреляции в разных рыночных условиях.

Анализирует, как меняются корреляции между активами в зависимости от:
- Бычьего/медвежьего тренда
- Высокой/низкой волатильности
- Кризисных ситуаций

Путь: backend/strategy/correlation/conditional_correlation.py
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
from datetime import datetime

from core.logger import get_logger
from .models import ConditionalCorrelationMetrics

logger = get_logger(__name__)


class MarketRegimeClassifier:
    """
    Классификатор рыночных режимов для conditional correlation.
    """

    def __init__(
        self,
        volatility_percentile_high: float = 75.0,
        volatility_percentile_low: float = 25.0,
        crisis_threshold: float = -0.15  # -15% падение за короткий период
    ):
        """
        Инициализация классификатора.

        Args:
            volatility_percentile_high: Процентиль для высокой волатильности
            volatility_percentile_low: Процентиль для низкой волатильности
            crisis_threshold: Порог для определения кризиса (% падения)
        """
        self.volatility_percentile_high = volatility_percentile_high
        self.volatility_percentile_low = volatility_percentile_low
        self.crisis_threshold = crisis_threshold

    def classify_trend_regime(
        self,
        prices: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """
        Классифицирует периоды как бычий/медвежий тренд.

        Args:
            prices: Массив цен
            window: Окно для определения тренда

        Returns:
            np.ndarray: Массив меток (1 = bullish, -1 = bearish, 0 = neutral)
        """
        if len(prices) < window:
            return np.zeros(len(prices))

        # Простая скользящая средняя
        sma = np.convolve(prices, np.ones(window)/window, mode='valid')

        # Расширяем до полной длины
        sma_full = np.concatenate([
            np.full(window - 1, sma[0]),
            sma
        ])

        # Классификация
        regime = np.zeros(len(prices))
        regime[prices > sma_full * 1.02] = 1   # Bullish (выше SMA на 2%)
        regime[prices < sma_full * 0.98] = -1  # Bearish (ниже SMA на 2%)

        return regime

    def classify_volatility_regime(
        self,
        returns: np.ndarray
    ) -> np.ndarray:
        """
        Классифицирует периоды по волатильности.

        Args:
            returns: Массив returns

        Returns:
            np.ndarray: Массив меток (1 = high vol, -1 = low vol, 0 = normal)
        """
        if len(returns) < 20:
            return np.zeros(len(returns))

        # Скользящая волатильность
        window = 20
        rolling_vol = np.array([
            np.std(returns[max(0, i-window):i+1])
            for i in range(len(returns))
        ])

        # Процентили
        high_thresh = np.percentile(rolling_vol, self.volatility_percentile_high)
        low_thresh = np.percentile(rolling_vol, self.volatility_percentile_low)

        # Классификация
        regime = np.zeros(len(returns))
        regime[rolling_vol > high_thresh] = 1   # High volatility
        regime[rolling_vol < low_thresh] = -1   # Low volatility

        return regime

    def detect_crisis_periods(
        self,
        prices: np.ndarray,
        window: int = 5
    ) -> np.ndarray:
        """
        Определяет периоды кризиса (резкое падение).

        Args:
            prices: Массив цен
            window: Окно для определения падения

        Returns:
            np.ndarray: Массив меток (True = crisis, False = normal)
        """
        if len(prices) < window:
            return np.zeros(len(prices), dtype=bool)

        crisis = np.zeros(len(prices), dtype=bool)

        for i in range(window, len(prices)):
            # Падение от локального максимума
            recent_prices = prices[i-window:i+1]
            max_price = np.max(recent_prices)
            current_price = prices[i]

            drawdown = (current_price - max_price) / max_price

            if drawdown < self.crisis_threshold:
                crisis[i] = True

        return crisis


class ConditionalCorrelationAnalyzer:
    """
    Анализатор корреляций в различных рыночных условиях.
    """

    def __init__(self):
        """Инициализация анализатора."""
        self.classifier = MarketRegimeClassifier()

        logger.info("ConditionalCorrelationAnalyzer инициализирован")

    def calculate_conditional_correlations(
        self,
        symbol_a: str,
        symbol_b: str,
        prices_a: np.ndarray,
        prices_b: np.ndarray
    ) -> ConditionalCorrelationMetrics:
        """
        Вычисляет корреляции в различных условиях.

        Args:
            symbol_a: Символ A
            symbol_b: Символ B
            prices_a: Цены A
            prices_b: Цены B

        Returns:
            ConditionalCorrelationMetrics: Набор условных корреляций
        """
        # Рассчитываем returns
        returns_a = np.diff(prices_a) / prices_a[:-1]
        returns_b = np.diff(prices_b) / prices_b[:-1]

        # Используем цены для классификации режимов
        # (берем среднее между двумя активами для определения режима)
        avg_prices = (prices_a + prices_b) / 2
        avg_prices = avg_prices[1:]  # Совмещаем с returns

        # 1. Бычий/медвежий тренд
        trend_regime = self.classifier.classify_trend_regime(avg_prices)

        bullish_corr = self._calculate_regime_correlation(
            returns_a, returns_b, trend_regime == 1
        )
        bearish_corr = self._calculate_regime_correlation(
            returns_a, returns_b, trend_regime == -1
        )

        # 2. Волатильность
        avg_returns = (returns_a + returns_b) / 2
        vol_regime = self.classifier.classify_volatility_regime(avg_returns)

        high_vol_corr = self._calculate_regime_correlation(
            returns_a, returns_b, vol_regime == 1
        )
        low_vol_corr = self._calculate_regime_correlation(
            returns_a, returns_b, vol_regime == -1
        )

        # 3. Кризисные периоды
        crisis_periods = self.classifier.detect_crisis_periods(avg_prices)

        crisis_corr = self._calculate_regime_correlation(
            returns_a, returns_b, crisis_periods
        )

        return ConditionalCorrelationMetrics(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            bullish_correlation=bullish_corr,
            bearish_correlation=bearish_corr,
            high_vol_correlation=high_vol_corr,
            low_vol_correlation=low_vol_corr,
            crisis_correlation=crisis_corr
        )

    def _calculate_regime_correlation(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        regime_mask: np.ndarray
    ) -> Optional[float]:
        """
        Вычисляет корреляцию только для определенного режима.

        Args:
            returns_a: Returns актива A
            returns_b: Returns актива B
            regime_mask: Маска режима (True для включенных периодов)

        Returns:
            Optional[float]: Корреляция или None если недостаточно данных
        """
        if not np.any(regime_mask):
            return None

        # Фильтруем returns по маске
        regime_returns_a = returns_a[regime_mask]
        regime_returns_b = returns_b[regime_mask]

        if len(regime_returns_a) < 10:  # Минимум 10 наблюдений
            return None

        try:
            correlation_matrix = np.corrcoef(regime_returns_a, regime_returns_b)
            correlation = correlation_matrix[0, 1]

            if np.isnan(correlation):
                return None

            return float(correlation)

        except Exception as e:
            logger.warning(f"Ошибка расчета conditional correlation: {e}")
            return None

    def analyze_correlation_stability(
        self,
        symbol_a: str,
        symbol_b: str,
        prices_a: np.ndarray,
        prices_b: np.ndarray
    ) -> Dict[str, float]:
        """
        Анализирует стабильность корреляции в разных режимах.

        Args:
            symbol_a: Символ A
            symbol_b: Символ B
            prices_a: Цены A
            prices_b: Цены B

        Returns:
            Dict[str, float]: Метрики стабильности
        """
        metrics = self.calculate_conditional_correlations(
            symbol_a, symbol_b, prices_a, prices_b
        )

        # Собираем все корреляции
        correlations = []
        if metrics.bullish_correlation is not None:
            correlations.append(metrics.bullish_correlation)
        if metrics.bearish_correlation is not None:
            correlations.append(metrics.bearish_correlation)
        if metrics.high_vol_correlation is not None:
            correlations.append(metrics.high_vol_correlation)
        if metrics.low_vol_correlation is not None:
            correlations.append(metrics.low_vol_correlation)

        if not correlations:
            return {
                "correlation_variance": 0.0,
                "max_correlation_diff": 0.0,
                "stability_score": 0.0
            }

        # Метрики стабильности
        correlation_variance = np.var(correlations)
        max_correlation_diff = np.max(correlations) - np.min(correlations)

        # Stability score (0-1, где 1 = очень стабильная корреляция)
        # Низкая variance и малая разница = высокая стабильность
        stability_score = 1.0 / (1.0 + max_correlation_diff)

        return {
            "correlation_variance": float(correlation_variance),
            "max_correlation_diff": float(max_correlation_diff),
            "stability_score": float(stability_score),
            "crisis_correlation_diff": (
                abs(metrics.crisis_correlation - np.mean(correlations))
                if metrics.crisis_correlation is not None
                else 0.0
            )
        }

    def identify_high_risk_pairs(
        self,
        conditional_metrics: List[ConditionalCorrelationMetrics],
        crisis_threshold: float = 0.85
    ) -> List[Tuple[str, str, float]]:
        """
        Идентифицирует пары с высоким риском в кризис.

        Args:
            conditional_metrics: Список conditional метрик
            crisis_threshold: Порог для высокого риска

        Returns:
            List[Tuple[str, str, float]]: Список (symbol_a, symbol_b, crisis_corr)
        """
        high_risk_pairs = []

        for metrics in conditional_metrics:
            if metrics.crisis_correlation is None:
                continue

            if abs(metrics.crisis_correlation) >= crisis_threshold:
                high_risk_pairs.append((
                    metrics.symbol_a,
                    metrics.symbol_b,
                    metrics.crisis_correlation
                ))

        # Сортируем по убыванию корреляции
        high_risk_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return high_risk_pairs


# Глобальный экземпляр
conditional_correlation_analyzer = ConditionalCorrelationAnalyzer()
