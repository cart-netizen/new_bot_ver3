"""
ML Model Optimization v2 - Industry Standard Package.

Полный пакет для обучения оптимизированных ML моделей для трейдинга.

Ключевые компоненты:
- configs: Оптимизированные конфигурации
- models: HybridCNNLSTMv2 с Residual/MultiHead Attention
- training: Trainer, Losses, Augmentation, Balancing
- training_orchestrator_v2: Главный оркестратор

Быстрый старт:
    from ml_optimized import TrainingOrchestratorV2, OrchestratorConfig
    
    config = OrchestratorConfig(symbols=["BTCUSDT"])
    orchestrator = TrainingOrchestratorV2(config)
    results = await orchestrator.run_training()

Или через CLI:
    python run_optimized_training.py --preset production
"""

__version__ = "2.0.0"
__author__ = "Trading Bot Team"

# Главный оркестратор
from .training_orchestrator_v2 import (
    TrainingOrchestratorV2,
    OrchestratorConfig
)

# Конфигурации
from .configs import (
    OptimizedModelConfig,
    OptimizedTrainerConfig,
    OptimizedBalancingConfig,
    ConfigPresets,
    get_default_configs
)

# Модели
from .models import (
    HybridCNNLSTMv2,
    ModelConfigV2,
    create_model_v2,
    create_model_v2_from_preset
)

# Training компоненты
from .training import (
    ModelTrainerV2,
    TrainerConfigV2,
    create_trainer_v2,
    FocalLossV2,
    LabelSmoothingCrossEntropy,
    MultiTaskLossV2,
    AugmentationPipeline,
    MixUp,
    ClassBalancingStrategyV2,
    create_balancing_strategy
)

__all__ = [
    # Version
    '__version__',
    
    # Orchestrator
    'TrainingOrchestratorV2',
    'OrchestratorConfig',
    
    # Configs
    'OptimizedModelConfig',
    'OptimizedTrainerConfig',
    'OptimizedBalancingConfig',
    'ConfigPresets',
    'get_default_configs',
    
    # Models
    'HybridCNNLSTMv2',
    'ModelConfigV2',
    'create_model_v2',
    'create_model_v2_from_preset',
    
    # Training
    'ModelTrainerV2',
    'TrainerConfigV2',
    'create_trainer_v2',
    'FocalLossV2',
    'LabelSmoothingCrossEntropy',
    'MultiTaskLossV2',
    'AugmentationPipeline',
    'MixUp',
    'ClassBalancingStrategyV2',
    'create_balancing_strategy'
]
