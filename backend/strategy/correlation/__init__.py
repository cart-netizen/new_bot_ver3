"""
Пакет продвинутого анализа корреляций.

Экспортирует:
- Модели данных
- Продвинутый калькулятор
- Методы группировки
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
    EnsembleGroupManager
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
]
