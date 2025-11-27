"""
Оптимизированные конфигурации для ML модели.

Содержит:
- OptimizedModelConfig: конфигурация модели
- OptimizedTrainerConfig: конфигурация обучения
- OptimizedBalancingConfig: конфигурация балансировки
- ConfigPresets: пресеты для разных сценариев
"""

from .optimized_configs import (
    OptimizedModelConfig,
    OptimizedTrainerConfig,
    OptimizedBalancingConfig,
    OptimizedDataConfig,
    ConfigPresets,
    LRSchedulerType,
    ClassBalancingMethod,
    get_default_configs,
    validate_configs
)

__all__ = [
    'OptimizedModelConfig',
    'OptimizedTrainerConfig', 
    'OptimizedBalancingConfig',
    'OptimizedDataConfig',
    'ConfigPresets',
    'LRSchedulerType',
    'ClassBalancingMethod',
    'get_default_configs',
    'validate_configs'
]
