"""
Детектор рыночных режимов корреляций.

Путь: backend/strategy/correlation/regime_detector.py
"""
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime

from backend.core.logger import get_logger
from .models import (
    MarketCorrelationRegime,
    CorrelationRegimeInfo,
    VolatilityCluster
)

logger = get_logger(__name__)


class CorrelationRegimeDetector:
    """
    Детектор текущего режима корреляций на рынке.

    Определяет:
    - Общий уровень корреляций (низкий/средний/высокий/кризис)
    - Рекомендуемые параметры для группировки
    """

    def __init__(
        self,
        low_threshold: float = 0.4,
        moderate_threshold: float = 0.6,
        high_threshold: float = 0.75,
        crisis_threshold: float = 0.85
    ):
        """
        Инициализация детектора.

        Args:
            low_threshold: Порог для низких корреляций
            moderate_threshold: Порог для умеренных корреляций
            high_threshold: Порог для высоких корреляций
            crisis_threshold: Порог для кризисных корреляций
        """
        self.low_threshold = low_threshold
        self.moderate_threshold = moderate_threshold
        self.high_threshold = high_threshold
        self.crisis_threshold = crisis_threshold

        logger.info(
            f"CorrelationRegimeDetector: thresholds="
            f"[{low_threshold}, {moderate_threshold}, "
            f"{high_threshold}, {crisis_threshold}]"
        )

    def detect_regime(
        self,
        correlation_matrix: Dict[Tuple[str, str], float]
    ) -> CorrelationRegimeInfo:
        """
        Определение текущего режима корреляций.

        Args:
            correlation_matrix: Матрица корреляций

        Returns:
            CorrelationRegimeInfo: Информация о режиме
        """
        if not correlation_matrix:
            logger.warning("Пустая correlation matrix")
            return self._default_regime()

        # Извлекаем все корреляции
        correlations = [abs(corr) for corr in correlation_matrix.values()]

        # Средняя корреляция по рынку
        avg_correlation = np.mean(correlations)

        # Считаем количество пар с высокой корреляцией
        high_corr_count = sum(1 for c in correlations if c >= self.high_threshold)

        # Считаем количество пар с низкой корреляцией
        low_corr_count = sum(1 for c in correlations if c < self.moderate_threshold)

        # Определяем режим
        if avg_correlation >= self.crisis_threshold:
            regime = MarketCorrelationRegime.CRISIS_CORRELATION
            recommended_threshold = 0.85
            recommended_max_positions = 1
            confidence = min(
                (avg_correlation - self.crisis_threshold) / (1.0 - self.crisis_threshold),
                1.0
            )

        elif avg_correlation >= self.high_threshold:
            regime = MarketCorrelationRegime.HIGH_CORRELATION
            recommended_threshold = 0.75
            recommended_max_positions = 1
            confidence = (avg_correlation - self.high_threshold) / (
                self.crisis_threshold - self.high_threshold
            )

        elif avg_correlation >= self.moderate_threshold:
            regime = MarketCorrelationRegime.MODERATE_CORRELATION
            recommended_threshold = 0.7
            recommended_max_positions = 2
            confidence = (avg_correlation - self.moderate_threshold) / (
                self.high_threshold - self.moderate_threshold
            )

        else:
            regime = MarketCorrelationRegime.LOW_CORRELATION
            recommended_threshold = 0.6
            recommended_max_positions = 3
            confidence = 1.0 - (avg_correlation / self.moderate_threshold)

        # Общее количество пар
        total_pairs = len(correlations)

        info = CorrelationRegimeInfo(
            regime=regime,
            avg_market_correlation=avg_correlation,
            correlation_threshold=recommended_threshold,
            max_positions_per_group=recommended_max_positions,
            high_correlation_pairs_count=high_corr_count,
            independent_pairs_count=low_corr_count,
            regime_confidence=max(0.0, min(confidence, 1.0))
        )

        logger.info(
            f"Режим корреляций: {regime.value} | "
            f"avg_corr={avg_correlation:.3f} | "
            f"confidence={info.regime_confidence:.2f}"
        )

        return info

    def _default_regime(self) -> CorrelationRegimeInfo:
        """Режим по умолчанию."""
        return CorrelationRegimeInfo(
            regime=MarketCorrelationRegime.MODERATE_CORRELATION,
            avg_market_correlation=0.5,
            correlation_threshold=0.7,
            max_positions_per_group=2,
            high_correlation_pairs_count=0,
            independent_pairs_count=0,
            regime_confidence=0.0
        )


class VolatilityClusterManager:
    """
    Менеджер кластеризации активов по волатильности.
    """

    def __init__(self, n_clusters: int = 3):
        """
        Инициализация.

        Args:
            n_clusters: Количество кластеров волатильности
        """
        self.n_clusters = n_clusters

    def create_volatility_clusters(
        self,
        symbols: List[str],
        returns_cache: Dict[str, np.ndarray]
    ) -> List[VolatilityCluster]:
        """
        Создание кластеров активов по волатильности.

        Args:
            symbols: Список символов
            returns_cache: Кеш returns для каждого символа

        Returns:
            List[VolatilityCluster]: Список кластеров
        """
        if not returns_cache:
            logger.warning("Пустой returns cache")
            return []

        # Рассчитываем волатильность для каждого символа
        volatilities = {}
        for symbol in symbols:
            if symbol not in returns_cache:
                continue

            returns = returns_cache[symbol]
            if len(returns) < 2:
                continue

            vol = np.std(returns)
            volatilities[symbol] = vol

        if not volatilities:
            logger.warning("Не удалось рассчитать волатильности")
            return []

        # Сортируем по волатильности
        sorted_symbols = sorted(volatilities.items(), key=lambda x: x[1])

        # Разбиваем на кластеры (простая квантильная разбивка)
        n = len(sorted_symbols)
        cluster_size = max(1, n // self.n_clusters)

        clusters = []
        for i in range(self.n_clusters):
            start_idx = i * cluster_size
            end_idx = (
                (i + 1) * cluster_size
                if i < self.n_clusters - 1
                else n
            )

            cluster_symbols_with_vol = sorted_symbols[start_idx:end_idx]

            if not cluster_symbols_with_vol:
                continue

            cluster_symbols = [s for s, v in cluster_symbols_with_vol]
            cluster_vols = [v for s, v in cluster_symbols_with_vol]

            vol_min = min(cluster_vols)
            vol_max = max(cluster_vols)
            vol_avg = np.mean(cluster_vols)

            cluster = VolatilityCluster(
                cluster_id=f"vol_cluster_{i}",
                volatility_range=(vol_min, vol_max),
                symbols=cluster_symbols,
                avg_volatility=vol_avg
            )

            clusters.append(cluster)

            logger.info(
                f"Volatility cluster {i}: "
                f"range=[{vol_min:.4f}, {vol_max:.4f}], "
                f"symbols={len(cluster_symbols)}"
            )

        return clusters
