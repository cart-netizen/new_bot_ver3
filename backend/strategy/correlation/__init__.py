"""
Пакет продвинутого анализа корреляций.

Экспортирует:
- Модели данных
- Продвинутый калькулятор
- Методы группировки
- DTW калькулятор
- Conditional correlation анализатор
- Market cap weighting менеджер
- Детектор режимов корреляций
"""
from .models import (
    CorrelationMethod,
    GroupingMethod,
    MarketCorrelationRegime,
    CorrelationMetrics,
    RollingCorrelationWindow,
    AdvancedCorrelationGroup,
    VolatilityCluster,
    CorrelationRegimeInfo,
    DTWParameters,
    ConditionalCorrelationMetrics
)

from .advanced_calculator import AdvancedCorrelationCalculator

from .grouping_methods import (
    GraphBasedGroupManager,
    HierarchicalGroupManager,
    EnsembleGroupManager,
    ClusterQualityMetrics
)

from .dtw_calculator import DTWCalculator

from .conditional_correlation import (
    MarketRegimeClassifier,
    ConditionalCorrelationAnalyzer
)

from .market_cap_weighting import (
    MarketCapInfo,
    MarketCapWeightingManager,
    market_cap_manager
)

from .regime_detector import (
    CorrelationRegimeDetector,
    VolatilityClusterManager
)

__all__ = [
    # Models
    "CorrelationMethod",
    "GroupingMethod",
    "MarketCorrelationRegime",
    "CorrelationMetrics",
    "RollingCorrelationWindow",
    "AdvancedCorrelationGroup",
    "VolatilityCluster",
    "CorrelationRegimeInfo",
    "DTWParameters",
    "ConditionalCorrelationMetrics",
    # Calculator
    "AdvancedCorrelationCalculator",
    # Grouping
    "GraphBasedGroupManager",
    "HierarchicalGroupManager",
    "EnsembleGroupManager",
    "ClusterQualityMetrics",
    # DTW
    "DTWCalculator",
    # Conditional Correlation
    "MarketRegimeClassifier",
    "ConditionalCorrelationAnalyzer",
    # Market Cap Weighting
    "MarketCapInfo",
    "MarketCapWeightingManager",
    "market_cap_manager",
    # Regime Detection
    "CorrelationRegimeDetector",
    "VolatilityClusterManager",
]
