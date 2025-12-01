#!/usr/bin/env python3
"""
Интеграционный модуль для оптимизированной ML системы.

Этот модуль связывает все оптимизированные компоненты с существующей
инфраструктурой бота. Предоставляет единый интерфейс для:
1. Создания оптимизированной модели
2. Настройки обучения
3. Интеграции с существующими компонентами

Путь: backend/ml_engine/integration/optimized_ml_integration.py
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from collections import Counter

from backend.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# OPTIMIZED HYPERPARAMETERS
# ============================================================================

@dataclass
class OptimizedHyperparameters:
    """
    Оптимизированные гиперпараметры - Industry Standard.
    
    КРИТИЧЕСКИЕ ИЗМЕНЕНИЯ относительно базовых значений:
    
    | Параметр       | Было    | Стало   | Причина                    |
    |----------------|---------|---------|----------------------------|
    | learning_rate  | 0.001   | 5e-5    | Финансовые данные шумные   |
    | batch_size     | 64      | 256     | Стабильность градиентов    |
    | weight_decay   | ~0      | 0.01    | L2 регуляризация           |
    | focal_gamma    | 2.0     | 2.5     | Фокус на hard examples     |
    | dropout        | 0.3     | 0.4     | Лучшая генерализация       |
    | epochs         | 100     | 150     | Больше эпох с меньшим LR   |
    """
    
    # === КРИТИЧЕСКИЕ параметры (менять в первую очередь) ===
    learning_rate: float = 5e-5      # НЕ 0.001!
    batch_size: int = 256            # НЕ 64!
    weight_decay: float = 0.01       # L2 регуляризация
    
    # === Training ===
    epochs: int = 150
    early_stopping_patience: int = 20
    grad_clip_value: float = 1.0
    
    # === Scheduler ===
    scheduler_type: str = "cosine_warm_restarts"
    scheduler_T_0: int = 10
    scheduler_T_mult: int = 2
    scheduler_eta_min: float = 1e-7
    
    # === Label Smoothing ===
    label_smoothing: float = 0.1
    
    # === Data Augmentation ===
    use_augmentation: bool = True
    mixup_alpha: float = 0.2
    gaussian_noise_std: float = 0.01
    
    # === Class Balancing ===
    use_focal_loss: bool = True
    focal_gamma: float = 2.5
    use_class_weights: bool = True
    
    # === Model Architecture ===
    cnn_channels: Tuple[int, ...] = (32, 64, 128)
    lstm_hidden: int = 128
    dropout: float = 0.4
    use_residual: bool = True
    use_layer_norm: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        return asdict(self)
    
    def apply_to_trainer_config(self, trainer_config) -> None:
        """
        Применить оптимизированные параметры к существующему TrainerConfig.
        
        Args:
            trainer_config: Существующий TrainerConfig для модификации
        """
        trainer_config.learning_rate = self.learning_rate
        trainer_config.weight_decay = self.weight_decay
        trainer_config.epochs = self.epochs
        trainer_config.early_stopping_patience = self.early_stopping_patience
        trainer_config.grad_clip_value = self.grad_clip_value
        
        # Scheduler
        if hasattr(trainer_config, 'lr_scheduler'):
            trainer_config.lr_scheduler = self.scheduler_type
        
        logger.info(f"✓ Применены оптимизированные гиперпараметры")
    
    def apply_to_data_config(self, data_config) -> None:
        """
        Применить batch_size к DataConfig.
        
        Args:
            data_config: Существующий DataConfig для модификации
        """
        data_config.batch_size = self.batch_size
        logger.info(f"✓ Batch size изменён на {self.batch_size}")
    
    def apply_to_balancing_config(self, balancing_config) -> None:
        """
        Применить параметры балансировки.
        
        Args:
            balancing_config: Существующий ClassBalancingConfig
        """
        balancing_config.use_focal_loss = self.use_focal_loss
        balancing_config.focal_gamma = self.focal_gamma
        balancing_config.use_class_weights = self.use_class_weights
        
        logger.info(f"✓ Применены параметры балансировки классов")


# ============================================================================
# INTEGRATION FUNCTIONS
# ============================================================================

def create_optimized_model_config():
    """
    Создать оптимизированную конфигурацию модели.

    Returns:
        ModelConfigV2 с оптимизированными параметрами
    """
    # UPDATED: Используем v2 версию
    from backend.ml_engine.models.hybrid_cnn_lstm_v2 import ModelConfigV2 as ModelConfig

    return ModelConfig(
        input_features=110,
        sequence_length=60,
        cnn_channels=(32, 64, 128),  # Уменьшено для малого датасета
        lstm_hidden=128,             # Уменьшено
        lstm_layers=2,
        attention_units=64,
        num_classes=3,
        dropout=0.4                  # Увеличено
    )


def create_optimized_trainer_config(checkpoint_dir: str = "checkpoints/models"):
    """
    Создать оптимизированную конфигурацию trainer.

    Args:
        checkpoint_dir: Директория для checkpoints

    Returns:
        TrainerConfigV2 с оптимизированными параметрами
    """
    # UPDATED: Используем v2 версию
    from backend.ml_engine.training.model_trainer_v2 import TrainerConfigV2 as TrainerConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return TrainerConfig(
        epochs=150,
        learning_rate=5e-5,          # КРИТИЧНО!
        weight_decay=0.01,           # КРИТИЧНО!
        grad_clip_value=1.0,
        early_stopping_patience=20,
        label_smoothing=0.1,
        scheduler_type="cosine_warm_restarts",
        scheduler_T_0=10,
        scheduler_T_mult=2,
        checkpoint_dir=checkpoint_dir,
        save_best_only=True,
        device=device
    )


def create_optimized_data_config(storage_path: str = "data/ml_training"):
    """
    Создать оптимизированную конфигурацию данных.
    
    Args:
        storage_path: Путь к данным
    
    Returns:
        DataConfig с оптимизированными параметрами
    """
    from backend.ml_engine.training.data_loader import DataConfig
    
    return DataConfig(
        storage_path=storage_path,
        sequence_length=60,
        target_horizon="future_direction_60s",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=256,              # КРИТИЧНО!
        shuffle=True,
        num_workers=4
    )


def create_optimized_balancing_config():
    """
    Создать оптимизированную конфигурацию балансировки классов.
    
    Returns:
        ClassBalancingConfig с оптимизированными параметрами
    """
    from backend.ml_engine.training.class_balancing import ClassBalancingConfig
    
    return ClassBalancingConfig(
        use_class_weights=True,
        use_focal_loss=True,
        focal_gamma=2.5,             # КРИТИЧНО: увеличено с 2.0
        use_oversampling=True,
        oversample_ratio=0.5,
        use_undersampling=False,
        undersample_ratio=0.8
    )


def setup_optimized_training():
    """
    Полная настройка оптимизированного обучения.
    
    Возвращает все необходимые конфигурации для запуска обучения.
    
    Returns:
        Tuple из (model_config, trainer_config, data_config, balancing_config)
    """
    model_config = create_optimized_model_config()
    trainer_config = create_optimized_trainer_config()
    data_config = create_optimized_data_config()
    balancing_config = create_optimized_balancing_config()
    
    logger.info("\n" + "=" * 80)
    logger.info("ОПТИМИЗИРОВАННАЯ КОНФИГУРАЦИЯ")
    logger.info("=" * 80)
    logger.info("Model:")
    logger.info(f"  • CNN channels: {model_config.cnn_channels}")
    logger.info(f"  • LSTM hidden: {model_config.lstm_hidden}")
    logger.info(f"  • Dropout: {model_config.dropout}")
    logger.info("\nTraining:")
    logger.info(f"  • Learning rate: {trainer_config.learning_rate}")
    logger.info(f"  • Batch size: {data_config.batch_size}")
    logger.info(f"  • Weight decay: {trainer_config.weight_decay}")
    logger.info(f"  • Epochs: {trainer_config.epochs}")
    logger.info("\nClass Balancing:")
    logger.info(f"  • Focal Loss: gamma={balancing_config.focal_gamma}")
    logger.info(f"  • Class weights: {balancing_config.use_class_weights}")
    logger.info(f"  • Oversampling: ratio={balancing_config.oversample_ratio}")
    logger.info("=" * 80 + "\n")
    
    return model_config, trainer_config, data_config, balancing_config


# ============================================================================
# LOSS FUNCTION SETUP
# ============================================================================

def create_optimized_loss(
    train_labels: np.ndarray,
    device: str = "cpu"
) -> nn.Module:
    """
    Создать оптимизированную loss function.
    
    Использует Focal Loss с Label Smoothing и автоматическими весами классов.
    
    Args:
        train_labels: Метки обучающей выборки для расчёта весов
        device: Device для tensors
    
    Returns:
        Настроенная loss function
    """
    from backend.ml_engine.training.class_balancing import ClassBalancingStrategy, ClassBalancingConfig
    
    # Расчёт весов классов
    class_counts = Counter(train_labels)
    num_classes = len(class_counts)
    total = len(train_labels)
    
    # Balanced weights
    weights = np.array([
        total / (num_classes * class_counts.get(i, 1))
        for i in range(num_classes)
    ], dtype=np.float32)
    
    # Normalize
    weights = weights * num_classes / weights.sum()
    
    logger.info(f"Class weights: {weights}")
    
    # Создаём Focal Loss
    from backend.ml_engine.training.class_balancing import FocalLoss
    
    focal_loss = FocalLoss(
        gamma=2.5,
        alpha=torch.tensor(weights, device=device),
        reduction='mean'
    )
    
    return focal_loss


# ============================================================================
# SCHEDULER SETUP
# ============================================================================

def create_optimized_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine_warm_restarts",
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Создать оптимизированный LR scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_type: Тип scheduler
        **kwargs: Дополнительные параметры
    
    Returns:
        Настроенный scheduler
    """
    if scheduler_type == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get("T_0", 10),
            T_mult=kwargs.get("T_mult", 2),
            eta_min=kwargs.get("eta_min", 1e-7)
        )
    
    elif scheduler_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=kwargs.get("patience", 5),
            factor=kwargs.get("factor", 0.5),
            min_lr=kwargs.get("min_lr", 1e-7)
        )
    
    elif scheduler_type == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", 1e-4),
            total_steps=kwargs.get("total_steps", 1000),
            pct_start=0.3,
            anneal_strategy='cos'
        )
    
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 100),
            eta_min=kwargs.get("eta_min", 1e-7)
        )


# ============================================================================
# MIGRATION HELPERS
# ============================================================================

def migrate_existing_config(existing_config: Any) -> Any:
    """
    Мигрировать существующую конфигурацию к оптимизированным значениям.
    
    Сохраняет пользовательские настройки, обновляя только критические параметры.
    
    Args:
        existing_config: Существующая конфигурация (любого типа)
    
    Returns:
        Обновлённая конфигурация
    """
    optimized = OptimizedHyperparameters()
    
    # Обновляем критические параметры
    if hasattr(existing_config, 'learning_rate'):
        if existing_config.learning_rate >= 0.001:
            logger.warning(
                f"⚠️ Learning rate слишком высокий: {existing_config.learning_rate} → {optimized.learning_rate}"
            )
            existing_config.learning_rate = optimized.learning_rate
    
    if hasattr(existing_config, 'batch_size'):
        if existing_config.batch_size < 128:
            logger.warning(
                f"⚠️ Batch size слишком маленький: {existing_config.batch_size} → {optimized.batch_size}"
            )
            existing_config.batch_size = optimized.batch_size
    
    if hasattr(existing_config, 'weight_decay'):
        if existing_config.weight_decay < 0.001:
            logger.warning(
                f"⚠️ Weight decay слишком маленький: {existing_config.weight_decay} → {optimized.weight_decay}"
            )
            existing_config.weight_decay = optimized.weight_decay
    
    return existing_config


def validate_config(config: Any) -> List[str]:
    """
    Валидировать конфигурацию на соответствие оптимальным значениям.
    
    Args:
        config: Конфигурация для валидации
    
    Returns:
        Список предупреждений
    """
    warnings = []
    
    if hasattr(config, 'learning_rate'):
        if config.learning_rate > 1e-4:
            warnings.append(
                f"Learning rate ({config.learning_rate}) слишком высокий. "
                f"Рекомендуется 1e-5 - 1e-4."
            )
    
    if hasattr(config, 'batch_size'):
        if config.batch_size < 128:
            warnings.append(
                f"Batch size ({config.batch_size}) слишком маленький. "
                f"Рекомендуется 128-256."
            )
    
    if hasattr(config, 'weight_decay'):
        if config.weight_decay < 0.001:
            warnings.append(
                f"Weight decay ({config.weight_decay}) слишком маленький. "
                f"Рекомендуется 0.001-0.01."
            )
    
    if hasattr(config, 'focal_gamma'):
        if config.focal_gamma < 2.0:
            warnings.append(
                f"Focal gamma ({config.focal_gamma}) может быть недостаточным. "
                f"Рекомендуется 2.0-3.0."
            )
    
    return warnings


# ============================================================================
# QUICK START
# ============================================================================

def quick_start_training(
    symbols: List[str],
    data_path: str = "data/ml_training",
    output_dir: str = "checkpoints/optimized"
):
    """
    Быстрый запуск оптимизированного обучения.
    
    Использование:
        from backend.ml_engine.integration.optimized_ml_integration import quick_start_training
        quick_start_training(["BTCUSDT", "ETHUSDT"])
    
    Args:
        symbols: Список торговых пар
        data_path: Путь к данным
        output_dir: Директория для сохранения
    """
    # UPDATED: Используем v2 версии
    from backend.ml_engine.models.hybrid_cnn_lstm_v2 import HybridCNNLSTMv2 as HybridCNNLSTM
    from backend.ml_engine.training.model_trainer_v2 import ModelTrainerV2 as ModelTrainer
    from backend.ml_engine.training.data_loader import HistoricalDataLoader

    # 1. Получаем оптимизированные конфигурации
    model_config, trainer_config, data_config, balancing_config = setup_optimized_training()

    # Обновляем пути
    data_config.storage_path = data_path
    trainer_config.checkpoint_dir = output_dir

    # 2. Создаём модель
    model = HybridCNNLSTM(model_config)
    logger.info(f"✓ Модель создана")

    # 3. Загружаем данные
    data_loader = HistoricalDataLoader(data_config, balancing_config)
    result = data_loader.load_and_prepare(symbols, apply_resampling=True)
    dataloaders = result['dataloaders']
    logger.info(f"✓ Данные загружены")

    # 4. Настраиваем trainer с балансировкой (v2 использует create_trainer_v2)
    from backend.ml_engine.training.model_trainer_v2 import create_trainer_v2
    trainer = create_trainer_v2(model, trainer_config)

    # 5. Обучаем
    history = trainer.train(dataloaders['train'], dataloaders['val'])

    logger.info(f"✓ Обучение завершено!")
    logger.info(f"  • Эпох: {len(history)}")
    logger.info(f"  • Best val_loss: {trainer.best_val_loss:.4f}")

    return model, history


# ============================================================================
# EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Демонстрация использования
    print("=" * 80)
    print("OPTIMIZED ML INTEGRATION")
    print("=" * 80)
    
    # Получаем оптимизированные параметры
    params = OptimizedHyperparameters()
    
    print("\n📊 Оптимизированные гиперпараметры:")
    for key, value in params.to_dict().items():
        print(f"  • {key}: {value}")
    
    # Setup полной конфигурации
    print("\n🔧 Настройка конфигурации...")
    model_cfg, trainer_cfg, data_cfg, balance_cfg = setup_optimized_training()
    
    # Валидация
    print("\n✅ Валидация конфигураций...")
    warnings = validate_config(trainer_cfg)
    if warnings:
        print("⚠️ Предупреждения:")
        for w in warnings:
            print(f"  • {w}")
    else:
        print("Все конфигурации оптимальны!")
    
    print("\n" + "=" * 80)
