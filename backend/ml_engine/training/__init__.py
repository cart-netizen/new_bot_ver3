"""
Улучшенные компоненты обучения ML модели.

Содержит:
- ModelTrainerV2: оптимизированный trainer
- Loss functions: Focal Loss, Label Smoothing, Multi-task Loss
- Augmentation: MixUp, Time Masking, Gaussian Noise
- Class Balancing: Adaptive thresholds, Oversampling, SMOTE
"""

# Losses
from .losses import (
    LabelSmoothingCrossEntropy,
    FocalLossV2,
    AsymmetricFocalLoss,
    MultiTaskLossV2,
    DirectionalAccuracyLoss,
    ConfidenceCalibrationLoss,
    LossFactory,
    compute_class_weights
)

# Augmentation
from .augmentation import (
    AugmentationConfig,
    AugmentationPipeline,
    MixUp,
    CutMix,
    TimeMasking,
    FeatureDropout,
    GaussianNoise,
    TimeWarping,
    MagnitudeWarping,
    get_default_augmentation_config,
    get_conservative_augmentation_config,
    get_aggressive_augmentation_config
)

# Class Balancing
from .class_balancing_v2 import (
    ClassBalancingConfigV2,
    ClassBalancingStrategyV2,
    ThresholdOptimizer,
    ClassWeightsCalculator,
    ResamplingStrategy,
    BalancingMethod,
    WeightMethod,
    ThresholdMethod,
    create_balancing_strategy
)

# Trainer
from .model_trainer_v2 import (
    ModelTrainerV2,
    TrainerConfigV2,
    EpochMetrics,
    EarlyStopping,
    create_trainer_v2
)

__all__ = [
    # Losses
    'LabelSmoothingCrossEntropy',
    'FocalLossV2',
    'AsymmetricFocalLoss',
    'MultiTaskLossV2',
    'DirectionalAccuracyLoss',
    'ConfidenceCalibrationLoss',
    'LossFactory',
    'compute_class_weights',
    
    # Augmentation
    'AugmentationConfig',
    'AugmentationPipeline',
    'MixUp',
    'CutMix',
    'TimeMasking',
    'FeatureDropout',
    'GaussianNoise',
    'TimeWarping',
    'MagnitudeWarping',
    'get_default_augmentation_config',
    'get_conservative_augmentation_config',
    'get_aggressive_augmentation_config',
    
    # Class Balancing
    'ClassBalancingConfigV2',
    'ClassBalancingStrategyV2',
    'ThresholdOptimizer',
    'ClassWeightsCalculator',
    'ResamplingStrategy',
    'BalancingMethod',
    'WeightMethod',
    'ThresholdMethod',
    'create_balancing_strategy',
    
    # Trainer
    'ModelTrainerV2',
    'TrainerConfigV2',
    'EpochMetrics',
    'EarlyStopping',
    'create_trainer_v2'
]
