#!/usr/bin/env python3
"""
ПАТЧИ ДЛЯ СУЩЕСТВУЮЩЕГО КОДА - МИНИМАЛЬНЫЕ ИЗМЕНЕНИЯ

Этот файл содержит конкретные изменения, которые нужно внести
в существующие файлы проекта для достижения industry standard качества.

Применение: Скопируйте соответствующие секции в указанные файлы.
"""

# ============================================================================
# ПАТЧ 1: backend/ml_engine/training/model_trainer.py
# ============================================================================
"""
ИЗМЕНЕНИЯ В TrainerConfig:

Найдите класс TrainerConfig и измените следующие значения по умолчанию:
"""

TRAINER_CONFIG_CHANGES = """
@dataclass
class TrainerConfig:
    '''Конфигурация обучения.'''
    
    # === Training параметры (ИЗМЕНЕНО) ===
    epochs: int = 150                     # Было: 100
    learning_rate: float = 5e-5           # Было: 0.001 ← КРИТИЧНО!
    weight_decay: float = 0.01            # Было: 1e-5 ← КРИТИЧНО!
    grad_clip_value: float = 1.0          # Без изменений
    
    # === Early stopping (ИЗМЕНЕНО) ===
    early_stopping_patience: int = 20     # Было: 10-15
    early_stopping_delta: float = 1e-4    # Без изменений
    
    # === Loss weights (без изменений) ===
    direction_loss_weight: float = 1.0
    confidence_loss_weight: float = 0.5
    return_loss_weight: float = 0.3
    
    # === LR Scheduler (ИЗМЕНЕНО) ===
    lr_scheduler: str = "CosineAnnealingWarmRestarts"  # Было: ReduceLROnPlateau
    lr_patience: int = 5
    lr_factor: float = 0.5
    
    # Новые параметры для CosineAnnealingWarmRestarts
    scheduler_T_0: int = 10               # Период первого цикла
    scheduler_T_mult: int = 2             # Умножитель периода
    scheduler_eta_min: float = 1e-7       # Минимальный LR
    
    # ... остальные параметры без изменений
"""


# ============================================================================
# ПАТЧ 2: backend/ml_engine/training/data_loader.py
# ============================================================================
"""
ИЗМЕНЕНИЯ В DataConfig:

Найдите класс DataConfig и измените batch_size:
"""

DATA_CONFIG_CHANGES = """
@dataclass
class DataConfig:
    '''Конфигурация загрузки данных.'''
    
    storage_path: str = "data/ml_training"
    sequence_length: int = 60
    target_horizon: str = "future_direction_60s"
    
    # Split параметры
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # === DataLoader параметры (ИЗМЕНЕНО) ===
    batch_size: int = 256                 # Было: 64 ← КРИТИЧНО!
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True               # ДОБАВЛЕНО: ускорение GPU
    drop_last: bool = True                # ДОБАВЛЕНО: стабильность BatchNorm
    
    # ... остальные параметры без изменений
"""


# ============================================================================
# ПАТЧ 3: backend/ml_engine/training/class_balancing.py
# ============================================================================
"""
ИЗМЕНЕНИЯ В ClassBalancingConfig:

Найдите ClassBalancingConfig и измените focal_gamma:
"""

CLASS_BALANCING_CHANGES = """
@dataclass
class ClassBalancingConfig:
    '''Конфигурация балансировки классов.'''
    
    use_class_weights: bool = True
    use_focal_loss: bool = True
    focal_gamma: float = 2.5              # Было: 2.0 ← ИЗМЕНЕНО
    
    use_oversampling: bool = True         # ДОБАВЛЕНО
    oversample_ratio: float = 0.5         # ДОБАВЛЕНО
    
    use_undersampling: bool = False
    undersample_ratio: float = 0.8
"""


# ============================================================================
# ПАТЧ 4: Изменение создания scheduler в ModelTrainer
# ============================================================================
"""
В методе __init__ класса ModelTrainer найдите создание scheduler
и замените на CosineAnnealingWarmRestarts:
"""

SCHEDULER_PATCH = """
def _create_scheduler(self):
    '''Создание LR scheduler.'''
    
    if self.config.lr_scheduler == "CosineAnnealingWarmRestarts":
        # НОВЫЙ SCHEDULER (рекомендуется)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=getattr(self.config, 'scheduler_T_0', 10),
            T_mult=getattr(self.config, 'scheduler_T_mult', 2),
            eta_min=getattr(self.config, 'scheduler_eta_min', 1e-7)
        )
    elif self.config.lr_scheduler == "ReduceLROnPlateau":
        # Старый scheduler (fallback)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config.lr_patience,
            factor=self.config.lr_factor,
            min_lr=1e-7
        )
    else:
        # CosineAnnealing (простой)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=1e-7
        )
"""


# ============================================================================
# ПАТЧ 5: Добавление Label Smoothing в loss function
# ============================================================================
"""
В методе _setup_loss_function добавьте Label Smoothing:
"""

LABEL_SMOOTHING_PATCH = """
def _setup_loss_function(self, train_labels=None):
    '''Настройка loss function с Label Smoothing.'''
    
    # Вычисляем class weights если нужно
    class_weights = None
    if self.config.class_balancing and self.config.class_balancing.use_class_weights:
        if train_labels is not None:
            class_weights = self._compute_class_weights(train_labels)
    
    # Label Smoothing (НОВОЕ)
    label_smoothing = getattr(self.config, 'label_smoothing', 0.1)
    
    if self.config.class_balancing and self.config.class_balancing.use_focal_loss:
        # FocalLoss с Label Smoothing
        from backend.ml_engine.training.class_balancing import FocalLoss
        
        direction_criterion = FocalLoss(
            gamma=self.config.class_balancing.focal_gamma,
            alpha=torch.tensor(class_weights, device=self.device) if class_weights is not None else None,
            reduction='mean'
            # Label smoothing применяется отдельно
        )
    else:
        # CrossEntropy с Label Smoothing (встроенная поддержка в PyTorch 1.10+)
        direction_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, device=self.device) if class_weights is not None else None,
            label_smoothing=label_smoothing
        )
    
    # ... остальной код MultiTaskLoss
"""


# ============================================================================
# ПАТЧ 6: Добавление Gaussian Noise augmentation в _train_epoch
# ============================================================================
"""
В методе _train_epoch добавьте augmentation:
"""

AUGMENTATION_PATCH = """
def _train_epoch(self, train_loader, epoch_num):
    '''Обучение одной эпохи с augmentation.'''
    self.model.train()
    
    # Параметры augmentation
    use_augmentation = getattr(self.config, 'use_augmentation', True)
    noise_std = getattr(self.config, 'gaussian_noise_std', 0.01)
    
    for batch in train_loader:
        sequences = batch['sequence'].to(self.device)
        labels = batch['label'].to(self.device)
        
        # === DATA AUGMENTATION (НОВОЕ) ===
        if use_augmentation and self.model.training:
            # Gaussian Noise
            sequences = sequences + torch.randn_like(sequences) * noise_std
        
        # Forward pass
        outputs = self.model(sequences)
        
        # ... остальной код
"""


# ============================================================================
# ПАТЧ 7: Изменения в ModelConfig (hybrid_cnn_lstm.py)
# ============================================================================
"""
ИЗМЕНЕНИЯ В ModelConfig для уменьшения модели:
"""

MODEL_CONFIG_CHANGES = """
@dataclass
class ModelConfig:
    '''Конфигурация модели.'''
    
    input_features: int = 110
    sequence_length: int = 60
    
    # === CNN параметры (ИЗМЕНЕНО для малого датасета) ===
    cnn_channels: Tuple[int, ...] = (32, 64, 128)   # Было: (64, 128, 256)
    cnn_kernel_sizes: Tuple[int, ...] = (3, 5, 7)
    
    # === LSTM параметры (ИЗМЕНЕНО) ===
    lstm_hidden: int = 128                          # Было: 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    
    # === Attention ===
    attention_units: int = 64
    
    # === Output ===
    num_classes: int = 3
    
    # === Regularization (ИЗМЕНЕНО) ===
    dropout: float = 0.4                            # Было: 0.3
"""


# ============================================================================
# ПОЛНЫЙ ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

USAGE_EXAMPLE = """
# ============================================================================
# ПРИМЕР: Запуск обучения с оптимизированными параметрами
# ============================================================================

from backend.ml_engine.models.hybrid_cnn_lstm import HybridCNNLSTM, ModelConfig
from backend.ml_engine.training.model_trainer import ModelTrainer, TrainerConfig
from backend.ml_engine.training.data_loader import HistoricalDataLoader, DataConfig
from backend.ml_engine.training.class_balancing import ClassBalancingConfig

# 1. Оптимизированная конфигурация модели
model_config = ModelConfig(
    cnn_channels=(32, 64, 128),    # Уменьшено
    lstm_hidden=128,               # Уменьшено
    dropout=0.4                    # Увеличено
)

# 2. Оптимизированная конфигурация обучения
trainer_config = TrainerConfig(
    epochs=150,
    learning_rate=5e-5,            # КРИТИЧНО!
    weight_decay=0.01,             # КРИТИЧНО!
    early_stopping_patience=20,
    lr_scheduler="CosineAnnealingWarmRestarts",
    scheduler_T_0=10,
    scheduler_T_mult=2
)

# 3. Оптимизированная конфигурация данных
data_config = DataConfig(
    batch_size=256,                # КРИТИЧНО!
    pin_memory=True,
    drop_last=True
)

# 4. Оптимизированная балансировка классов
balancing_config = ClassBalancingConfig(
    use_focal_loss=True,
    focal_gamma=2.5,               # Увеличено
    use_class_weights=True,
    use_oversampling=True,
    oversample_ratio=0.5
)

# 5. Создание компонентов
model = HybridCNNLSTM(model_config)

data_loader = HistoricalDataLoader(data_config, balancing_config)
result = data_loader.load_and_prepare(["BTCUSDT"], apply_resampling=True)

trainer_config.class_balancing = balancing_config
trainer = ModelTrainer(model, trainer_config)

# 6. Обучение
history = trainer.train(
    result['dataloaders']['train'],
    result['dataloaders']['val']
)

print(f"Best val_loss: {trainer.best_val_loss:.4f}")
"""


# ============================================================================
# QUICK REFERENCE
# ============================================================================

QUICK_REFERENCE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    QUICK REFERENCE: ОПТИМИЗИРОВАННЫЕ ЗНАЧЕНИЯ                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  КРИТИЧЕСКИЕ ПАРАМЕТРЫ (менять в первую очередь):                           ║
║  ┌────────────────────┬─────────────┬─────────────┬─────────────────────────┐║
║  │ Параметр           │ Было        │ Стало       │ Файл                    │║
║  ├────────────────────┼─────────────┼─────────────┼─────────────────────────┤║
║  │ learning_rate      │ 0.001       │ 5e-5        │ model_trainer.py        │║
║  │ batch_size         │ 64          │ 256         │ data_loader.py          │║
║  │ weight_decay       │ ~0          │ 0.01        │ model_trainer.py        │║
║  └────────────────────┴─────────────┴─────────────┴─────────────────────────┘║
║                                                                              ║
║  ВАЖНЫЕ ПАРАМЕТРЫ (менять во вторую очередь):                               ║
║  ┌────────────────────┬─────────────┬─────────────┬─────────────────────────┐║
║  │ focal_gamma        │ 2.0         │ 2.5         │ class_balancing.py      │║
║  │ dropout            │ 0.3         │ 0.4         │ hybrid_cnn_lstm.py      │║
║  │ epochs             │ 100         │ 150         │ model_trainer.py        │║
║  │ early_stop_patience│ 10          │ 20          │ model_trainer.py        │║
║  │ scheduler          │ ReduceOnPlat│ CosineWarmR │ model_trainer.py        │║
║  └────────────────────┴─────────────┴─────────────┴─────────────────────────┘║
║                                                                              ║
║  НОВЫЕ ПАРАМЕТРЫ (добавить):                                                 ║
║  ┌────────────────────┬─────────────┬─────────────────────────────────────────┐║
║  │ label_smoothing    │ 0.1         │ TrainerConfig                          │║
║  │ mixup_alpha        │ 0.2         │ TrainerConfig                          │║
║  │ gaussian_noise_std │ 0.01        │ TrainerConfig                          │║
║  │ scheduler_T_0      │ 10          │ TrainerConfig                          │║
║  │ scheduler_T_mult   │ 2           │ TrainerConfig                          │║
║  │ pin_memory         │ True        │ DataConfig                             │║
║  │ drop_last          │ True        │ DataConfig                             │║
║  └────────────────────┴─────────────┴─────────────────────────────────────────┘║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(QUICK_REFERENCE)
    print("\nПолный пример использования:")
    print(USAGE_EXAMPLE)
