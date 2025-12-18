#!/usr/bin/env python3
"""
Оптимизированные конфигурации для ML модели - Industry Standard.

Данные конфигурации разработаны для достижения максимальной производительности
ML модели на финансовых данных с учётом:
- Малого объёма данных (7-30 дней)
- Высокого class imbalance (Hold доминирует)
- Необходимости генерализации на реальных торгах

КРИТИЧЕСКИЕ ИЗМЕНЕНИЯ относительно базовой конфигурации:
1. Learning Rate: 0.001 → 5e-5 (в 20 раз меньше)
2. Batch Size: 64 → 256 (в 4 раза больше)
3. Weight Decay: ~0 → 0.01 (сильная L2 регуляризация)
4. Scheduler: ReduceLROnPlateau → CosineAnnealingWarmRestarts
5. Dropout: 0.3 → 0.4 (увеличенная регуляризация)
6. Focal Gamma: 2.0 → 2.5 (фокус на hard examples)

Путь: backend/ml_engine/configs/optimized_configs.py
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from pathlib import Path
import torch

# Project root models directory
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # backend/ml_engine/configs -> project_root
_DEFAULT_MODELS_DIR = str(_PROJECT_ROOT / "models")


class LRSchedulerType(str, Enum):
    """Типы Learning Rate Scheduler."""
    REDUCE_ON_PLATEAU = "ReduceLROnPlateau"
    COSINE_ANNEALING = "CosineAnnealing"
    COSINE_WARM_RESTARTS = "CosineAnnealingWarmRestarts"
    ONE_CYCLE = "OneCycleLR"
    EXPONENTIAL = "ExponentialLR"


class ClassBalancingMethod(str, Enum):
    """Методы балансировки классов."""
    NONE = "none"
    CLASS_WEIGHTS = "class_weights"
    FOCAL_LOSS = "focal_loss"
    FOCAL_WITH_WEIGHTS = "focal_with_weights"
    OVERSAMPLING = "oversampling"
    SMOTE = "smote"
    COMBINED = "combined"  # Focal + Oversampling


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class OptimizedModelConfig:
    """
    Оптимизированная конфигурация модели HybridCNNLSTM.
    
    Изменения относительно базовой версии:
    - Уменьшены cnn_channels для малого датасета (меньше параметров)
    - Уменьшен lstm_hidden (256 → 128)
    - Увеличен dropout (0.3 → 0.4)
    - Добавлены residual connections
    - Добавлен Multi-Head Attention
    - Добавлен Layer Normalization
    """
    
    # === Input параметры ===
    input_features: int = 110  # OrderBook(50) + Candle(25) + Indicator(35)
    sequence_length: int = 60  # 60 временных шагов
    
    # === CNN параметры (УМЕНЬШЕНЫ для малого датасета) ===
    # Было: (64, 128, 256) - ~500K параметров
    # Стало: (32, 64, 128) - ~150K параметров
    cnn_channels: Tuple[int, ...] = (32, 64, 128)
    cnn_kernel_sizes: Tuple[int, ...] = (3, 5, 7)
    
    # === LSTM параметры (УМЕНЬШЕНЫ) ===
    # Было: 256 hidden, 2 layers
    # Стало: 128 hidden, 2 layers
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    
    # === Attention параметры (УЛУЧШЕНЫ) ===
    attention_units: int = 64
    attention_heads: int = 4  # НОВОЕ: Multi-Head Attention
    attention_dropout: float = 0.1
    
    # === Output параметры ===
    num_classes: int = 3  # DOWN/SELL=0, NEUTRAL/HOLD=1, UP/BUY=2
    
    # === Regularization (УСИЛЕНА) ===
    # Было: 0.3
    # Стало: 0.4 (для лучшей генерализации на малом датасете)
    dropout: float = 0.4
    
    # === Архитектурные улучшения (НОВОЕ) ===
    use_residual: bool = True  # Residual connections в CNN
    use_layer_norm: bool = True  # LayerNorm вместо BatchNorm в LSTM
    use_multi_head_attention: bool = True  # Multi-Head Attention

    # === Memory Optimization ===
    use_gradient_checkpointing: bool = False  # Enable only if OOM (may conflict with mixed precision)

    # === Инициализация ===
    init_method: str = "kaiming"  # kaiming, xavier, orthogonal
    
    def get_estimated_params(self) -> int:
        """Приблизительная оценка количества параметров."""
        # CNN params
        cnn_params = sum(
            (self.cnn_channels[i-1] if i > 0 else 1) * ch * k
            for i, (ch, k) in enumerate(zip(self.cnn_channels, self.cnn_kernel_sizes))
        )
        
        # LSTM params (bidirectional)
        lstm_input = self.cnn_channels[-1]
        lstm_params = 4 * self.lstm_hidden * (lstm_input + self.lstm_hidden + 1)
        lstm_params *= self.lstm_layers * 2  # bidirectional
        
        # Attention params
        attn_params = self.lstm_hidden * 2 * self.attention_units * 2
        
        # Head params
        head_params = self.lstm_hidden * 2 * 128 + 128 * self.num_classes
        head_params += self.lstm_hidden * 2 * 64 + 64  # confidence
        head_params += self.lstm_hidden * 2 * 64 + 64  # return
        
        return cnn_params + lstm_params + attn_params + head_params
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для логирования."""
        return {
            'input_features': self.input_features,
            'sequence_length': self.sequence_length,
            'cnn_channels': self.cnn_channels,
            'cnn_kernel_sizes': self.cnn_kernel_sizes,
            'lstm_hidden': self.lstm_hidden,
            'lstm_layers': self.lstm_layers,
            'lstm_dropout': self.lstm_dropout,
            'attention_units': self.attention_units,
            'attention_heads': self.attention_heads,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'use_residual': self.use_residual,
            'use_layer_norm': self.use_layer_norm,
            'use_multi_head_attention': self.use_multi_head_attention,
            'estimated_params': self.get_estimated_params()
        }


# ============================================================================
# TRAINER CONFIGURATION
# ============================================================================

@dataclass
class OptimizedTrainerConfig:
    """
    Оптимизированная конфигурация обучения.
    
    КРИТИЧЕСКИЕ ИЗМЕНЕНИЯ:
    1. learning_rate: 0.001 → 5e-5 (в 20 раз меньше!)
    2. batch_size: 64 → 256 (больше для стабильности градиентов)
    3. weight_decay: ~0 → 0.01 (L2 регуляризация)
    4. epochs: 100 → 150 (больше эпох с меньшим LR)
    5. scheduler: ReduceLROnPlateau → CosineAnnealingWarmRestarts
    """
    
    # === Training параметры (ОПТИМИЗИРОВАНЫ) ===
    epochs: int = 150  # Больше эпох с меньшим LR
    
    # КРИТИЧНО: Learning Rate уменьшен в 20 раз!
    # Финансовые данные очень шумные, высокий LR = overfitting
    learning_rate: float = 5e-5  # Было: 0.001
    
    # Альтернативные LR для экспериментов
    min_learning_rate: float = 1e-7
    max_learning_rate: float = 1e-4  # Для OneCycleLR
    
    # Batch Size: без mixed precision можно использовать больший batch
    batch_size: int = 128  # Стабильнее с use_mixed_precision=False
    
    # КРИТИЧНО: Weight Decay добавлен!
    # L2 регуляризация предотвращает overfitting
    weight_decay: float = 0.01  # Было: ~0
    
    # Gradient Clipping (оставляем)
    grad_clip_value: float = 1.0
    grad_clip_norm: Optional[float] = None  # Альтернатива: clip by norm
    
    # === Early Stopping (УВЕЛИЧЕН patience) ===
    early_stopping_patience: int = 20  # Было: 10-15
    early_stopping_delta: float = 1e-4
    early_stopping_metric: str = "val_loss"  # или "val_f1"
    
    # === Learning Rate Scheduler (ИЗМЕНЁН) ===
    lr_scheduler: LRSchedulerType = LRSchedulerType.COSINE_WARM_RESTARTS
    
    # CosineAnnealingWarmRestarts параметры
    scheduler_T_0: int = 10  # Период первого цикла
    scheduler_T_mult: int = 2  # Умножитель периода
    scheduler_eta_min: float = 1e-7  # Минимальный LR
    
    # ReduceLROnPlateau параметры (fallback)
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # === Multi-task Learning Weights ===
    direction_loss_weight: float = 1.0
    confidence_loss_weight: float = 0.5
    return_loss_weight: float = 0.3
    
    # === Label Smoothing (НОВОЕ) ===
    # Сглаживание меток предотвращает overconfidence
    label_smoothing: float = 0.1  # 0 = off, 0.1-0.2 = recommended
    
    # === Data Augmentation (НОВОЕ) ===
    use_augmentation: bool = True
    mixup_alpha: float = 0.2  # MixUp параметр (0 = off)
    gaussian_noise_std: float = 0.01  # Шум к features
    time_mask_ratio: float = 0.1  # Маскирование временных шагов
    feature_dropout_ratio: float = 0.05  # Dropout отдельных features

    # === Memory Optimization ===
    gradient_accumulation_steps: int = 1  # Accumulate gradients (effective_batch = batch_size * steps)
    use_mixed_precision: bool = False  # FP16 training (отключено - вызывает NaN)
    use_gradient_checkpointing: bool = False  # Enable only if OOM

    # === Checkpoint ===
    checkpoint_dir: str = _DEFAULT_MODELS_DIR  # Абсолютный путь к project_root/models
    save_best_only: bool = True
    save_every_n_epochs: int = 10
    
    # === Device ===
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # === Logging ===
    log_interval: int = 10  # Log every N batches
    verbose: bool = True
    use_tqdm: bool = True
    
    # === Reproducibility ===
    seed: int = 42
    deterministic: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для логирования."""
        return {
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'grad_clip_value': self.grad_clip_value,
            'early_stopping_patience': self.early_stopping_patience,
            'lr_scheduler': self.lr_scheduler.value,
            'label_smoothing': self.label_smoothing,
            'use_augmentation': self.use_augmentation,
            'mixup_alpha': self.mixup_alpha,
            'device': self.device
        }


# ============================================================================
# CLASS BALANCING CONFIGURATION
# ============================================================================

@dataclass
class OptimizedBalancingConfig:
    """
    Оптимизированная конфигурация балансировки классов.
    
    КРИТИЧЕСКИЕ ИЗМЕНЕНИЯ:
    1. focal_gamma: 2.0 → 2.5 (больше фокуса на hard examples)
    2. Добавлен oversampling для minority классов
    3. Оптимизированы веса классов
    """

    # === Основной метод ===
    method: ClassBalancingMethod = ClassBalancingMethod.FOCAL_LOSS  # Focal Loss + Oversampling

    # === Class Weights ===
    # DISABLED when using oversampling - they cause double compensation!
    # Oversampling already balances data physically
    use_class_weights: bool = False  # DISABLED: oversampling is primary balancing method
    class_weight_method: str = "balanced"  # balanced, sqrt, log
    
    # Custom веса (если не auto)
    # Типичное распределение: Hold ~70%, Buy ~15%, Sell ~15%
    # Веса: обратно пропорциональны частоте
    custom_class_weights: Optional[Dict[int, float]] = None
    
    # === Focal Loss параметры ===
    use_focal_loss: bool = True
    focal_gamma: float = 1.5  # Reduced from 2.0 - less aggressive with oversampling
    focal_alpha: Optional[List[float]] = None  # Auto-compute if None

    # === Resampling параметры ===
    use_oversampling: bool = True   # ENABLED: Physical data balancing for stability
    oversample_ratio: float = 1.0   # Full balance (100% = equal classes)
    
    use_undersampling: bool = False  # Не рекомендуется при малом датасете
    undersample_ratio: float = 0.8
    
    use_smote: bool = False  # SMOTE для временных рядов спорен
    smote_k_neighbors: int = 5
    
    # === Threshold для Labeling (НОВОЕ) ===
    # Адаптивный threshold вместо фиксированного
    adaptive_threshold: bool = True
    threshold_percentile_sell: float = 0.25  # Bottom 25% = Sell
    threshold_percentile_buy: float = 0.75  # Top 25% = Buy
    
    # Фиксированный threshold (если adaptive=False)
    fixed_threshold_sell: float = -0.001  # -0.1%
    fixed_threshold_buy: float = 0.001  # +0.1%
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для логирования."""
        return {
            'method': self.method.value,
            'use_class_weights': self.use_class_weights,
            'use_focal_loss': self.use_focal_loss,
            'focal_gamma': self.focal_gamma,
            'use_oversampling': self.use_oversampling,
            'oversample_ratio': self.oversample_ratio,
            'adaptive_threshold': self.adaptive_threshold
        }


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

@dataclass
class OptimizedDataConfig:
    """
    Оптимизированная конфигурация данных.
    
    Изменения:
    - batch_size синхронизирован с trainer config
    - Добавлены параметры для Feature Store
    """
    
    # === Storage ===
    storage_path: str = "data/ml_training"
    
    # === Sequence параметры ===
    sequence_length: int = 60
    target_horizon: str = "future_direction_60s"
    
    # === Split параметры ===
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # === DataLoader параметры (СИНХРОНИЗИРОВАНЫ) ===
    batch_size: int = 128  # Синхронизировано с TrainerConfig
    shuffle: bool = True
    num_workers: int = 4  # Оптимально для 12GB GPU
    pin_memory: bool = True
    drop_last: bool = True  # Для стабильности BatchNorm
    
    # === Feature Store ===
    use_feature_store: bool = True
    feature_store_date_range_days: int = 90
    feature_store_group: str = "training_features"
    fallback_to_legacy: bool = True
    
    # === Preprocessing ===
    normalize_features: bool = True
    clip_outliers: bool = True
    outlier_std: float = 5.0  # Clip values > 5 std
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для логирования."""
        return {
            'sequence_length': self.sequence_length,
            'batch_size': self.batch_size,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'use_feature_store': self.use_feature_store,
            'feature_store_date_range_days': self.feature_store_date_range_days
        }


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

class ConfigPresets:
    """Пресеты конфигураций для разных сценариев."""
    
    @staticmethod
    def production_small_data() -> Tuple[OptimizedModelConfig, OptimizedTrainerConfig, OptimizedBalancingConfig]:
        """
        Пресет для production с малым объёмом данных (7-30 дней).
        
        Особенности:
        - Маленькая модель (~150K параметров)
        - Сильная регуляризация
        - Консервативный learning rate
        """
        model_config = OptimizedModelConfig(
            cnn_channels=(32, 64, 128),
            lstm_hidden=128,
            dropout=0.4,
            use_residual=True,
            use_layer_norm=True,
            use_multi_head_attention=True
        )
        
        trainer_config = OptimizedTrainerConfig(
            epochs=150,
            learning_rate=5e-5,
            batch_size=128,
            weight_decay=0.01,
            label_smoothing=0.1,
            use_augmentation=True,
            mixup_alpha=0.2
        )
        
        balancing_config = OptimizedBalancingConfig(
            method=ClassBalancingMethod.FOCAL_LOSS,  # Только Focal Loss
            use_class_weights=False,
            use_focal_loss=True,
            focal_gamma=2.5,
            use_oversampling=False  # Отключено
        )

        return model_config, trainer_config, balancing_config

    @staticmethod
    def production_large_data() -> Tuple[OptimizedModelConfig, OptimizedTrainerConfig, OptimizedBalancingConfig]:
        """
        Пресет для production с большим объёмом данных (60+ дней).
        
        Особенности:
        - Большая модель (~500K параметров)
        - Умеренная регуляризация
        - Более агрессивный learning rate
        """
        model_config = OptimizedModelConfig(
            cnn_channels=(64, 128, 256),
            lstm_hidden=256,
            dropout=0.3,
            use_residual=True,
            use_layer_norm=True,
            use_multi_head_attention=True
        )
        
        trainer_config = OptimizedTrainerConfig(
            epochs=100,
            learning_rate=1e-4,
            batch_size=128,
            weight_decay=0.005,
            label_smoothing=0.05,
            use_augmentation=True,
            mixup_alpha=0.1
        )
        
        balancing_config = OptimizedBalancingConfig(
            method=ClassBalancingMethod.FOCAL_WITH_WEIGHTS,
            focal_gamma=2.0,
            use_oversampling=True,
            oversample_ratio=0.3
        )
        
        return model_config, trainer_config, balancing_config
    
    @staticmethod
    def quick_experiment() -> Tuple[OptimizedModelConfig, OptimizedTrainerConfig, OptimizedBalancingConfig]:
        """
        Пресет для быстрых экспериментов (5-10 минут обучения).
        
        Особенности:
        - Минимальная модель
        - Быстрое обучение
        - Меньше регуляризации
        """
        model_config = OptimizedModelConfig(
            cnn_channels=(32, 64),
            lstm_hidden=64,
            lstm_layers=1,
            dropout=0.3,
            use_residual=False,
            use_layer_norm=False,
            use_multi_head_attention=False
        )
        
        trainer_config = OptimizedTrainerConfig(
            epochs=30,
            learning_rate=1e-4,
            batch_size=128,
            weight_decay=0.001,
            label_smoothing=0.0,
            use_augmentation=False,
            early_stopping_patience=10
        )
        
        balancing_config = OptimizedBalancingConfig(
            method=ClassBalancingMethod.FOCAL_LOSS,
            focal_gamma=2.0,
            use_oversampling=False
        )
        
        return model_config, trainer_config, balancing_config
    
    @staticmethod
    def conservative_trading() -> Tuple[OptimizedModelConfig, OptimizedTrainerConfig, OptimizedBalancingConfig]:
        """
        Пресет для консервативной торговли (минимизация риска).
        
        Особенности:
        - Высокая уверенность в предсказаниях
        - Сильный акцент на precision
        - Меньше false positives
        """
        model_config = OptimizedModelConfig(
            cnn_channels=(32, 64, 128),
            lstm_hidden=128,
            dropout=0.5,  # Увеличенный dropout
            use_residual=True,
            use_layer_norm=True,
            use_multi_head_attention=True
        )
        
        trainer_config = OptimizedTrainerConfig(
            epochs=200,
            learning_rate=3e-5,  # Ещё меньше LR
            batch_size=128,
            weight_decay=0.02,  # Ещё больше регуляризации
            label_smoothing=0.15,
            use_augmentation=True,
            mixup_alpha=0.3
        )
        
        balancing_config = OptimizedBalancingConfig(
            method=ClassBalancingMethod.FOCAL_WITH_WEIGHTS,
            focal_gamma=3.0,  # Сильнее фокус на hard examples
            use_oversampling=True,
            oversample_ratio=0.4,
            # Более консервативные thresholds
            threshold_percentile_sell=0.20,
            threshold_percentile_buy=0.80
        )
        
        return model_config, trainer_config, balancing_config


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_default_configs() -> Tuple[OptimizedModelConfig, OptimizedTrainerConfig, OptimizedBalancingConfig, OptimizedDataConfig]:
    """Получить дефолтные оптимизированные конфигурации."""
    model_config, trainer_config, balancing_config = ConfigPresets.production_small_data()
    data_config = OptimizedDataConfig()
    return model_config, trainer_config, balancing_config, data_config


def validate_configs(
    model_config: OptimizedModelConfig,
    trainer_config: OptimizedTrainerConfig,
    data_config: OptimizedDataConfig
) -> List[str]:
    """
    Валидация согласованности конфигураций.
    
    Returns:
        Список предупреждений (пустой если всё OK)
    """
    warnings = []
    
    # Проверка batch_size
    if trainer_config.batch_size != data_config.batch_size:
        warnings.append(
            f"batch_size рассогласован: trainer={trainer_config.batch_size}, "
            f"data={data_config.batch_size}"
        )
    
    # Проверка sequence_length
    if model_config.sequence_length != data_config.sequence_length:
        warnings.append(
            f"sequence_length рассогласован: model={model_config.sequence_length}, "
            f"data={data_config.sequence_length}"
        )
    
    # Проверка соотношения параметров к данным
    estimated_params = model_config.get_estimated_params()
    if estimated_params > 500000:
        warnings.append(
            f"Модель слишком большая ({estimated_params:,} параметров). "
            f"Рекомендуется < 500K для малых датасетов."
        )
    
    # Проверка learning rate
    if trainer_config.learning_rate > 1e-4:
        warnings.append(
            f"Learning rate слишком высокий ({trainer_config.learning_rate}). "
            f"Рекомендуется 1e-5 - 1e-4 для финансовых данных."
        )
    
    # Проверка weight_decay
    if trainer_config.weight_decay < 0.001:
        warnings.append(
            f"Weight decay слишком маленький ({trainer_config.weight_decay}). "
            f"Рекомендуется 0.001 - 0.01 для предотвращения overfitting."
        )
    
    return warnings


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Получаем дефолтные конфигурации
    model_cfg, trainer_cfg, balancing_cfg, data_cfg = get_default_configs()
    
    print("=" * 80)
    print("ОПТИМИЗИРОВАННЫЕ КОНФИГУРАЦИИ ML МОДЕЛИ")
    print("=" * 80)
    
    print("\n📊 Model Configuration:")
    for key, value in model_cfg.to_dict().items():
        print(f"  • {key}: {value}")
    
    print("\n🎓 Trainer Configuration:")
    for key, value in trainer_cfg.to_dict().items():
        print(f"  • {key}: {value}")
    
    print("\n⚖️ Balancing Configuration:")
    for key, value in balancing_cfg.to_dict().items():
        print(f"  • {key}: {value}")
    
    print("\n📁 Data Configuration:")
    for key, value in data_cfg.to_dict().items():
        print(f"  • {key}: {value}")
    
    # Валидация
    warnings = validate_configs(model_cfg, trainer_cfg, data_cfg)
    
    if warnings:
        print("\n⚠️ Предупреждения:")
        for warning in warnings:
            print(f"  • {warning}")
    else:
        print("\n✅ Все конфигурации согласованы!")
    
    print("\n" + "=" * 80)
