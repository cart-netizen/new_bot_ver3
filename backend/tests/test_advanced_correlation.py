"""
Тесты для продвинутого анализа корреляций.

Путь: backend/tests/test_advanced_correlation.py
"""
import sys
from pathlib import Path
from datetime import datetime

# Добавляем путь к backend
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))
import pytest
import numpy as np
from datetime import datetime

from backend.strategy.correlation.advanced_calculator import AdvancedCorrelationCalculator
from strategy.correlation.grouping_methods import (
    GraphBasedGroupManager,
    HierarchicalGroupManager,
    EnsembleGroupManager
)
from strategy.correlation.regime_detector import (
    CorrelationRegimeDetector,
    VolatilityClusterManager
)
from strategy.correlation.models import (
    DTWParameters,
    MarketCorrelationRegime
)


class TestAdvancedCorrelationCalculator:
    """Тесты для AdvancedCorrelationCalculator."""

    def test_initialization(self):
        """Тест инициализации калькулятора."""
        calc = AdvancedCorrelationCalculator(
            short_window=7,
            medium_window=14,
            long_window=30
        )

        assert len(calc.windows) == 3
        assert calc.windows[0].window_days == 7
        assert calc.windows[1].window_days == 14
        assert calc.windows[2].window_days == 30

    def test_calculate_returns(self):
        """Тест расчета returns."""
        prices = np.array([100, 105, 103, 108, 110])
        returns = AdvancedCorrelationCalculator.calculate_returns(prices)

        assert len(returns) == 4
        assert np.isclose(returns[0], 0.05)  # (105-100)/100 = 0.05

    def test_calculate_pearson(self):
        """Тест расчета Pearson correlation."""
        # Идеально коррелированные
        returns_a = np.array([0.01, 0.02, -0.01, 0.03])
        returns_b = np.array([0.01, 0.02, -0.01, 0.03])

        corr = AdvancedCorrelationCalculator.calculate_pearson(
            returns_a, returns_b
        )

        assert np.isclose(corr, 1.0, atol=0.01)

    def test_calculate_spearman(self):
        """Тест расчета Spearman correlation."""
        returns_a = np.array([0.01, 0.02, 0.03, 0.04])
        returns_b = np.array([0.02, 0.04, 0.06, 0.08])

        corr = AdvancedCorrelationCalculator.calculate_spearman(
            returns_a, returns_b
        )

        # Монотонная зависимость - должна быть высокая корреляция
        assert corr > 0.9

    def test_calculate_volatility_distance(self):
        """Тест расчета volatility distance."""
        # Одинаковая волатильность
        returns_a = np.array([0.01, -0.01, 0.01, -0.01])
        returns_b = np.array([0.01, -0.01, 0.01, -0.01])

        distance = AdvancedCorrelationCalculator.calculate_volatility_distance(
            returns_a, returns_b
        )

        assert distance < 0.1  # Очень малая разница

    def test_calculate_return_sign_agreement(self):
        """Тест расчета sign agreement."""
        # Всегда одинаковое направление
        returns_a = np.array([0.01, 0.02, -0.01, -0.02])
        returns_b = np.array([0.03, 0.01, -0.03, -0.01])

        agreement = AdvancedCorrelationCalculator.calculate_return_sign_agreement(
            returns_a, returns_b
        )

        assert agreement == 1.0  # 100% согласие

    def test_calculate_correlation_suite(self):
        """Тест полного набора метрик корреляции."""
        calc = AdvancedCorrelationCalculator()

        # Создаем коррелирующие цены
        prices_a = np.array([100 + i + np.random.randn() for i in range(30)])
        prices_b = prices_a + np.random.randn(30) * 2  # Сильно коррелированные

        metrics = calc.calculate_correlation_suite(
            "BTCUSDT",
            "ETHUSDT",
            prices_a,
            prices_b
        )

        assert metrics.symbol_a == "BTCUSDT"
        assert metrics.symbol_b == "ETHUSDT"
        assert -1.0 <= metrics.pearson <= 1.0
        assert -1.0 <= metrics.spearman <= 1.0
        assert 0.0 <= metrics.dtw_distance <= 1.0
        assert 0.0 <= metrics.volatility_distance <= 1.0
        assert 0.0 <= metrics.return_sign_agreement <= 1.0


class TestGraphBasedGroupManager:
    """Тесты для GraphBasedGroupManager."""

    def test_initialization(self):
        """Тест инициализации."""
        manager = GraphBasedGroupManager(correlation_threshold=0.7)
        assert manager.correlation_threshold == 0.7

    def test_create_groups_greedy_fallback(self):
        """Тест fallback группировки."""
        manager = GraphBasedGroupManager(correlation_threshold=0.7)

        symbols = ["BTC", "ETH", "SOL", "ADA"]
        correlation_matrix = {
            ("BTC", "ETH"): 0.85,
            ("BTC", "SOL"): 0.78,
            ("ETH", "SOL"): 0.80,
            ("ADA", "BTC"): 0.50  # Низкая корреляция
        }

        groups = manager._create_groups_greedy_fallback(
            symbols, correlation_matrix
        )

        # Должна быть создана хотя бы одна группа
        assert len(groups) >= 1

        # BTC, ETH, SOL должны быть в одной группе
        group_0 = groups[0]
        assert "BTC" in group_0.symbols
        assert "ETH" in group_0.symbols
        assert "SOL" in group_0.symbols


class TestHierarchicalGroupManager:
    """Тесты для HierarchicalGroupManager."""

    def test_initialization(self):
        """Тест инициализации."""
        manager = HierarchicalGroupManager(
            correlation_threshold=0.7,
            linkage='ward'
        )
        assert manager.correlation_threshold == 0.7
        assert manager.linkage == 'ward'

    def test_correlation_to_distance_matrix(self):
        """Тест преобразования корреляций в дистанции."""
        manager = HierarchicalGroupManager()

        symbols = ["BTC", "ETH"]
        correlation_matrix = {
            ("BTC", "ETH"): 0.8
        }

        distance_matrix = manager._correlation_to_distance_matrix(
            symbols, correlation_matrix
        )

        assert distance_matrix.shape == (2, 2)
        assert distance_matrix[0, 0] == 0.0  # Дистанция до себя
        # Distance = (1 - 0.8) / 2 = 0.1
        assert np.isclose(distance_matrix[0, 1], 0.1)


class TestCorrelationRegimeDetector:
    """Тесты для CorrelationRegimeDetector."""

    def test_initialization(self):
        """Тест инициализации."""
        detector = CorrelationRegimeDetector(
            low_threshold=0.4,
            moderate_threshold=0.6,
            high_threshold=0.75,
            crisis_threshold=0.85
        )

        assert detector.low_threshold == 0.4
        assert detector.moderate_threshold == 0.6
        assert detector.high_threshold == 0.75
        assert detector.crisis_threshold == 0.85

    def test_detect_low_correlation_regime(self):
        """Тест определения режима низких корреляций."""
        detector = CorrelationRegimeDetector()

        # Низкие корреляции
        correlation_matrix = {
            ("BTC", "ETH"): 0.3,
            ("BTC", "SOL"): 0.2,
            ("ETH", "SOL"): 0.35
        }

        regime_info = detector.detect_regime(correlation_matrix)

        assert regime_info.regime == MarketCorrelationRegime.LOW_CORRELATION
        assert regime_info.avg_market_correlation < 0.4

    def test_detect_high_correlation_regime(self):
        """Тест определения режима высоких корреляций."""
        detector = CorrelationRegimeDetector()

        # Высокие корреляции
        correlation_matrix = {
            ("BTC", "ETH"): 0.85,
            ("BTC", "SOL"): 0.80,
            ("ETH", "SOL"): 0.82
        }

        regime_info = detector.detect_regime(correlation_matrix)

        assert regime_info.regime == MarketCorrelationRegime.HIGH_CORRELATION
        assert regime_info.avg_market_correlation >= 0.75

    def test_detect_crisis_correlation_regime(self):
        """Тест определения кризисного режима."""
        detector = CorrelationRegimeDetector()

        # Кризисные корреляции (все падает вместе)
        correlation_matrix = {
            ("BTC", "ETH"): 0.95,
            ("BTC", "SOL"): 0.92,
            ("ETH", "SOL"): 0.94
        }

        regime_info = detector.detect_regime(correlation_matrix)

        assert regime_info.regime == MarketCorrelationRegime.CRISIS_CORRELATION
        assert regime_info.max_positions_per_group == 1  # Консервативно


class TestVolatilityClusterManager:
    """Тесты для VolatilityClusterManager."""

    def test_initialization(self):
        """Тест инициализации."""
        manager = VolatilityClusterManager(n_clusters=3)
        assert manager.n_clusters == 3

    def test_create_volatility_clusters(self):
        """Тест создания кластеров по волатильности."""
        manager = VolatilityClusterManager(n_clusters=2)

        symbols = ["BTC", "ETH", "SOL", "ADA"]

        # Создаем returns с разной волатильностью
        returns_cache = {
            "BTC": np.array([0.01, -0.01, 0.01, -0.01]),  # Low vol
            "ETH": np.array([0.02, -0.02, 0.02, -0.02]),  # Medium vol
            "SOL": np.array([0.05, -0.05, 0.05, -0.05]),  # High vol
            "ADA": np.array([0.06, -0.06, 0.06, -0.06])   # High vol
        }

        clusters = manager.create_volatility_clusters(symbols, returns_cache)

        # Должно быть 2 кластера
        assert len(clusters) == 2

        # Проверяем, что волатильности отсортированы
        assert clusters[0].avg_volatility < clusters[1].avg_volatility


@pytest.mark.asyncio
async def test_integration_workflow():
    """Интеграционный тест всего workflow."""
    # 1. Создаем данные
    symbols = ["BTC", "ETH", "SOL"]
    prices_cache = {
        "BTC": np.array([100 + i + np.random.randn() for i in range(30)]),
        "ETH": np.array([50 + i * 0.5 + np.random.randn() for i in range(30)]),
        "SOL": np.array([20 + i * 0.2 + np.random.randn() for i in range(30)])
    }

    # 2. Рассчитываем корреляции
    calc = AdvancedCorrelationCalculator()
    correlation_matrix = {}

    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i + 1:]:
            metrics = calc.calculate_correlation_suite(
                sym1, sym2,
                prices_cache[sym1],
                prices_cache[sym2]
            )
            key = (sym1, sym2) if sym1 < sym2 else (sym2, sym1)
            correlation_matrix[key] = metrics.weighted_score

    # 3. Определяем режим
    detector = CorrelationRegimeDetector()
    regime_info = detector.detect_regime(correlation_matrix)

    assert regime_info is not None
    assert isinstance(regime_info.regime, MarketCorrelationRegime)

    # 4. Создаем группы
    manager = GraphBasedGroupManager(correlation_threshold=0.7)
    groups = manager._create_groups_greedy_fallback(symbols, correlation_matrix)

    assert isinstance(groups, list)

    print(f"✅ Integration test passed: {len(groups)} groups created")
