#!/usr/bin/env python3
"""
Улучшенный Model Trainer v2 - Industry Standard.

Ключевые улучшения:
1. Интеграция с оптимизированными конфигурациями
2. Data Augmentation (MixUp, Time Masking, etc.)
3. CosineAnnealingWarmRestarts scheduler
4. Label Smoothing
5. Gradient Accumulation
6. Mixed Precision Training (опционально)
7. Расширенное логирование и метрики
8. Checkpoint management
9. Reproducibility (seed fixing)

Путь: backend/ml_engine/training/model_trainer_v2.py
"""

import os
import random
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any, Union
from collections import deque
from dataclasses import dataclass, asdict, is_dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from backend.core.logger import get_logger

# Project root models directory
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # backend/ml_engine/training -> project_root
_DEFAULT_MODELS_DIR = str(_PROJECT_ROOT / "models")

# Локальные импорты (будут созданы)
# from backend.ml_engine.configs.optimized_configs import (
#     OptimizedTrainerConfig,
#     OptimizedBalancingConfig
# )
# from backend.ml_engine.training.losses import (
#     LossFactory,
#     MultiTaskLossV2,
#     compute_class_weights
# )
# from backend.ml_engine.training.augmentation import (
#     AugmentationPipeline,
#     AugmentationConfig,
#     MixUp
# )

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainerConfigV2:
    """
    Оптимизированная конфигурация обучения v2.
    
    КРИТИЧЕСКИЕ ПАРАМЕТРЫ:
    - learning_rate: 5e-5 (НЕ 0.001!)
    - batch_size: 256
    - weight_decay: 0.01
    """
    
    # === Training ===
    epochs: int = 150
    learning_rate: float = 5e-5  # КРИТИЧНО: В 20 раз меньше стандартного!
    min_learning_rate: float = 1e-7
    batch_size: int = 64  # Уменьшено для oversampled dataset (было 128)
    weight_decay: float = 0.01  # L2 регуляризация

    # === Gradient ===
    grad_clip_value: float = 1.0
    gradient_accumulation_steps: int = 4  # Эффективный batch = 64*4 = 256
    
    # === Early Stopping ===
    early_stopping_patience: int = 20
    early_stopping_delta: float = 1e-4
    early_stopping_metric: str = "val_loss"  # val_loss, val_f1, val_accuracy
    
    # === Scheduler ===
    scheduler_type: str = "cosine_warm_restarts"  # cosine_warm_restarts, reduce_on_plateau
    scheduler_T_0: int = 10  # Период первого цикла
    scheduler_T_mult: int = 2  # Умножитель
    scheduler_eta_min: float = 1e-7
    scheduler_patience: int = 5  # Для ReduceOnPlateau
    scheduler_factor: float = 0.5
    
    # === Label Smoothing ===
    label_smoothing: float = 0.1  # Restored: helps with overconfidence

    # === Data Augmentation ===
    use_augmentation: bool = True  # Restored: helps with generalization
    mixup_alpha: float = 0.2  # Restored
    mixup_prob: float = 0.3  # Reduced from 0.5 for stability
    gaussian_noise_std: float = 0.01
    time_mask_ratio: float = 0.1

    # === Multi-task Loss Weights ===
    direction_loss_weight: float = 1.0
    confidence_loss_weight: float = 0.5
    return_loss_weight: float = 0.3

    # === Class Balancing ===
    # Strategy: Oversampling (data balance) + Focal Loss (hard examples)
    # NO class_weights - they cause double compensation with oversampling!
    use_class_weights: bool = False  # DISABLED: oversampling already balances data
    use_focal_loss: bool = True      # Focuses on hard examples
    focal_gamma: float = 1.5         # Moderate focus with balanced data
    
    # === Mixed Precision ===
    use_mixed_precision: bool = False  # ВРЕМЕННО ОТКЛЮЧЕНО из-за NaN loss (RTX 3060 поддерживает!)

    # === Checkpoint ===
    checkpoint_dir: str = _DEFAULT_MODELS_DIR  # Абсолютный путь к project_root/models
    save_best_only: bool = True
    save_every_n_epochs: int = 10

    # === Device ===
    device: str = "auto"  # auto, cuda, cpu
    
    # === Logging ===
    log_interval: int = 10
    verbose: bool = True
    use_tqdm: bool = True
    
    # === Reproducibility ===
    seed: int = 42
    deterministic: bool = False  # False для скорости, True для воспроизводимости
    
    def get_device(self) -> torch.device:
        """Получить device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """Early Stopping с поддержкой разных метрик."""
    
    def __init__(
        self,
        patience: int = 10,
        delta: float = 1e-4,
        mode: str = "min"
    ):
        """
        Args:
            patience: Количество эпох без улучшения
            delta: Минимальное улучшение
            mode: 'min' для loss, 'max' для accuracy/f1
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Проверить, нужно ли остановить обучение.
        
        Args:
            score: Текущее значение метрики
        
        Returns:
            True если нужно остановить
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.delta
        else:
            improved = score > self.best_score + self.delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        """Сбросить состояние."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


# ============================================================================
# EPOCH METRICS
# ============================================================================

@dataclass
class EpochMetrics:
    """Метрики эпохи."""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float
    learning_rate: float
    epoch_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# MODEL TRAINER V2
# ============================================================================

class ModelTrainerV2:
    """
    Улучшенный Model Trainer v2.
    
    Интегрирует все оптимизации:
    - Оптимизированные гиперпараметры
    - Data Augmentation (MixUp, noise, masking)
    - CosineAnnealingWarmRestarts scheduler
    - Label Smoothing + Focal Loss
    - Gradient clipping и accumulation
    - Mixed Precision Training
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfigV2
    ):
        """
        Args:
            model: Модель для обучения
            config: Конфигурация обучения
        """
        self.model = model
        self.config = config
        self.device = config.get_device()
        
        # Перемещаем модель на device
        self.model.to(self.device)

        # Очищаем GPU память перед началом
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Устанавливаем seed для воспроизводимости
        if config.deterministic:
            self._set_seed(config.seed)
        
        # Loss function (будет настроен в train())
        self.criterion = None
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Early Stopping
        mode = "min" if config.early_stopping_metric == "val_loss" else "max"
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            delta=config.early_stopping_delta,
            mode=mode
        )
        
        # Mixed Precision with safer settings
        if config.use_mixed_precision:
            self.scaler = GradScaler(
                init_scale=2.**10,  # Меньший начальный scale (было 2^16)
                growth_interval=1000  # Реже увеличиваем scale
            )
        else:
            self.scaler = None
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history: List[EpochMetrics] = []
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        
        # MixUp
        if config.use_augmentation and config.mixup_alpha > 0:
            from backend.ml_engine.training.augmentation import MixUp
            self.mixup = MixUp(alpha=config.mixup_alpha)
        else:
            self.mixup = None
        
        logger.info(
            f"✓ ModelTrainerV2 инициализирован:\n"
            f"  • Device: {self.device}\n"
            f"  • Learning rate: {config.learning_rate}\n"
            f"  • Batch size: {config.batch_size}\n"
            f"  • Weight decay: {config.weight_decay}\n"
            f"  • Scheduler: {config.scheduler_type}\n"
            f"  • Label smoothing: {config.label_smoothing}\n"
            f"  • MixUp: {config.mixup_alpha if self.mixup else 'disabled'}\n"
            f"  • Mixed precision: {config.use_mixed_precision}"
        )
    
    def _set_seed(self, seed: int):
        """Установить seed для воспроизводимости."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Создать learning rate scheduler."""
        if self.config.scheduler_type == "cosine_warm_restarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.scheduler_T_0,
                T_mult=self.config.scheduler_T_mult,
                eta_min=self.config.scheduler_eta_min
            )
        elif self.config.scheduler_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
                min_lr=self.config.min_learning_rate
            )
        elif self.config.scheduler_type == "one_cycle":
            # Нужно знать total_steps, устанавливается в train()
            return None
        else:
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_learning_rate
            )
    
    def _setup_loss_function(self, train_labels: np.ndarray):
        """
        Настроить loss function на основе данных.
        
        Args:
            train_labels: Метки обучающих данных для расчёта весов
        """
        from backend.ml_engine.training.losses import (
            LossFactory,
            compute_class_weights
        )
        
        # Вычисляем веса классов
        class_weights = None
        if self.config.use_class_weights:
            class_weights = compute_class_weights(
                train_labels,
                method="balanced"
            )
        
        # Определяем тип loss
        if self.config.use_focal_loss:
            if self.config.label_smoothing > 0:
                loss_type = "focal_smooth"
            else:
                loss_type = "focal"
        else:
            if self.config.label_smoothing > 0:
                loss_type = "ce_smooth"
            else:
                loss_type = "ce"
        
        # Создаём multi-task loss
        self.criterion = LossFactory.create_multi_task_loss(
            direction_loss_type=loss_type,
            num_classes=3,  # BUY, HOLD, SELL
            class_weights=class_weights,
            gamma=self.config.focal_gamma,
            label_smoothing=self.config.label_smoothing,
            direction_weight=self.config.direction_loss_weight,
            confidence_weight=self.config.confidence_loss_weight,
            return_weight=self.config.return_loss_weight,
            device=str(self.device)
        )
        
        logger.info(
            f"✓ Loss function настроен: {loss_type}, "
            f"focal_gamma={self.config.focal_gamma}, "
            f"label_smoothing={self.config.label_smoothing}"
        )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> List[EpochMetrics]:
        """
        Обучение модели.
        
        Args:
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации
            test_loader: DataLoader для тестирования (опционально)
        
        Returns:
            История обучения (list of EpochMetrics)
        """
        # Настраиваем loss на основе данных
        if self.criterion is None:
            logger.info("Анализ обучающих данных для настройки loss...")
            all_labels = []
            for batch in train_loader:
                labels = batch['label'].numpy()
                all_labels.extend(labels)
            self._setup_loss_function(np.array(all_labels))
        
        # Логируем начало обучения через tqdm.write() для немедленного вывода
        tqdm.write("\n" + "=" * 80)
        tqdm.write("НАЧАЛО ОБУЧЕНИЯ")
        tqdm.write("=" * 80)
        tqdm.write(f"Эпох: {self.config.epochs}")
        tqdm.write(f"Train batches: {len(train_loader)}")
        tqdm.write(f"Val batches: {len(val_loader)}")
        tqdm.write(f"Device: {self.device}")
        tqdm.write("=" * 80 + "\n")
        
        # Progress bar для эпох
        # position=0 и dynamic_ncols=False для стабильного вывода в веб-консоли
        epoch_pbar = tqdm(
            range(self.config.epochs),
            desc="Training",
            unit="epoch",
            disable=not self.config.use_tqdm,
            position=0,
            leave=True,
            dynamic_ncols=False,
            ncols=100,
            mininterval=1.0  # Обновлять не чаще раза в секунду
        )
        
        for epoch in epoch_pbar:
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self._train_epoch(
                train_loader,
                epoch_num=epoch + 1
            )
            
            # Validation
            val_loss, val_metrics = self._validate_epoch(
                val_loader,
                epoch_num=epoch + 1
            )
            
            # Scheduler step
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.config.scheduler_type == "reduce_on_plateau":
                # ReduceLROnPlateau принимает метрику (float)
                self.scheduler.step(val_loss)  # type: ignore[arg-type]
            else:
                self.scheduler.step()
            
            # Метрики эпохи
            epoch_time = time.time() - epoch_start
            metrics = EpochMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_acc,
                val_accuracy=val_metrics['accuracy'],
                val_precision=val_metrics['precision'],
                val_recall=val_metrics['recall'],
                val_f1=val_metrics['f1'],
                learning_rate=current_lr,
                epoch_time=epoch_time
            )
            self.history.append(metrics)
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_acc': f'{val_metrics["accuracy"]:.4f}',
                'val_f1': f'{val_metrics["f1"]:.4f}'
            })
            
            # Логирование через tqdm.write() для немедленного вывода
            if self.config.verbose:
                tqdm.write(
                    f"[Epoch {epoch + 1}/{self.config.epochs}] "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"val_acc={val_metrics['accuracy']:.4f}, "
                    f"val_f1={val_metrics['f1']:.4f}, "
                    f"lr={current_lr:.2e}, time={epoch_time:.1f}s"
                )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch + 1, val_loss, val_metrics, "best_loss")
            
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self._save_checkpoint(epoch + 1, val_loss, val_metrics, "best_f1")
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch + 1, val_loss, val_metrics, f"epoch_{epoch + 1}")
            
            # ================================================================
            # MODE COLLAPSE EARLY STOPPING
            # ================================================================
            # Stop training immediately if mode collapse is detected
            mode_collapse_severity = val_metrics.get('mode_collapse_severity', 'none')
            if mode_collapse_severity in ['total', 'severe']:
                tqdm.write(
                    f"\n🛑 [MODE COLLAPSE STOP] Training stopped at epoch {epoch + 1}. "
                    f"Severity: {mode_collapse_severity}. "
                    f"Majority class: {val_metrics.get('majority_class_pct', 0):.1%}"
                )
                tqdm.write(
                    f"   Macro F1: {val_metrics['f1']:.4f} (this is the true metric)"
                )
                # Mark this in history for later analysis
                self.history[-1] = EpochMetrics(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_accuracy=train_acc,
                    val_accuracy=val_metrics['accuracy'],
                    val_precision=val_metrics['precision'],
                    val_recall=val_metrics['recall'],
                    val_f1=val_metrics['f1'],  # This is now macro F1
                    learning_rate=current_lr,
                    epoch_time=epoch_time
                )
                break

            # Regular Early Stopping
            stop_metric = val_loss if self.config.early_stopping_metric == "val_loss" else val_metrics['f1']
            if self.early_stopping(stop_metric):
                tqdm.write(
                    f"\n[Early Stop] Triggered at epoch {epoch + 1}. "
                    f"Best val_loss: {self.best_val_loss:.4f}"
                )
                break
        
        # Финальное логирование через tqdm.write() для немедленного вывода
        tqdm.write("\n" + "=" * 80)
        tqdm.write("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        tqdm.write("=" * 80)
        tqdm.write(f"Всего эпох: {len(self.history)}")
        tqdm.write(f"Best val_loss: {self.best_val_loss:.4f}")
        tqdm.write(f"Best val_f1: {self.best_val_f1:.4f}")
        tqdm.write("=" * 80 + "\n")
        
        # Тестирование (если есть test_loader)
        test_results = None
        if test_loader is not None:
            test_results = self._test_model(test_loader)
            # Store test results for hyperopt logger
            self.test_results = test_results

        return self.history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch_num: int
    ) -> Tuple[float, float]:
        """Обучение одной эпохи."""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # ИСПРАВЛЕНИЕ: Используем один progress bar без вложенных для веб-консоли
        # disable=True для batch-level, используем только logging
        total_batches = len(train_loader)
        log_interval = max(1, total_batches // 10)  # Логируем 10 раз за эпоху

        for batch_idx, batch in enumerate(train_loader):
            # Получаем данные (batch - это Dict[str, Tensor])
            sequences: torch.Tensor = batch['sequence'].to(self.device)
            labels: torch.Tensor = batch['label'].to(self.device)

            # Инициализируем переменные для MixUp
            labels_a: torch.Tensor = labels
            labels_b: torch.Tensor = labels
            lam: float = 1.0

            # Data Augmentation: Gaussian noise
            if self.config.use_augmentation and self.training:
                sequences = sequences + torch.randn_like(sequences) * self.config.gaussian_noise_std

            # MixUp
            use_mixup = (
                self.mixup is not None and
                self.training and
                np.random.random() < self.config.mixup_prob
            )

            if use_mixup:
                sequences, labels_a, labels_b, lam = self.mixup(sequences, labels)
            
            # Forward pass
            if self.config.use_mixed_precision and self.scaler is not None:
                with autocast():
                    outputs = self.model(sequences)
                    if use_mixup:
                        targets_a = {'label': labels_a}
                        targets_b = {'label': labels_b}
                        loss_a, _ = self.criterion(outputs, targets_a)
                        loss_b, _ = self.criterion(outputs, targets_b)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        targets = {'label': labels}
                        loss, _ = self.criterion(outputs, targets)
            else:
                outputs = self.model(sequences)
                if use_mixup:
                    targets_a = {'label': labels_a}
                    targets_b = {'label': labels_b}
                    loss_a, _ = self.criterion(outputs, targets_a)
                    loss_b, _ = self.criterion(outputs, targets_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    targets = {'label': labels}
                    loss, _ = self.criterion(outputs, targets)

            # Save original loss for display
            original_loss = loss.item()

            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected at batch {batch_idx}! Skipping batch.")
                continue

            # Gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                    # Check for NaN gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip_value
                    )

                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logger.warning(f"NaN/Inf gradient detected! Skipping optimizer step.")
                        self.optimizer.zero_grad()
                        continue

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.config.grad_clip_value > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.grad_clip_value
                        )

                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            logger.warning(f"NaN/Inf gradient detected! Skipping optimizer step.")
                            self.optimizer.zero_grad()
                            continue

                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Очистка GPU кеша каждые N шагов для предотвращения фрагментации
                if self.device.type == "cuda" and (batch_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()

            # Metrics
            total_loss += original_loss

            predictions = torch.argmax(
                outputs['direction_logits'], dim=-1
            ).cpu().numpy()

            if use_mixup:
                # Для MixUp используем original labels для metrics
                all_labels.extend(labels_a.cpu().numpy())
            else:
                all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions)

            # Логирование прогресса через tqdm.write() для немедленного вывода в консоль
            if (batch_idx + 1) % log_interval == 0 or batch_idx == total_batches - 1:
                progress_pct = (batch_idx + 1) / total_batches * 100
                tqdm.write(
                    f"  [Train] Epoch {epoch_num}: {batch_idx + 1}/{total_batches} "
                    f"({progress_pct:.0f}%) - loss: {original_loss:.4f}"
                )
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        epoch_num: int
    ) -> Tuple[float, Dict[str, float]]:
        """Валидация одной эпохи."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # ИСПРАВЛЕНИЕ: Убираем вложенный tqdm для предотвращения дублирования строк
        total_batches = len(val_loader)
        log_interval = max(1, total_batches // 5)  # Логируем 5 раз за валидацию
        batch_idx = 0

        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward
                outputs = self.model(sequences)
                
                # Loss
                targets = {'label': labels}
                loss, _ = self.criterion(outputs, targets)
                
                # Metrics
                total_loss += loss.item()
                predictions = torch.argmax(
                    outputs['direction_logits'], dim=-1
                ).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())

                # Логирование прогресса через tqdm.write() для немедленного вывода
                batch_idx += 1
                if batch_idx % log_interval == 0 or batch_idx == total_batches:
                    progress_pct = batch_idx / total_batches * 100
                    tqdm.write(
                        f"  [Val] Epoch {epoch_num}: {batch_idx}/{total_batches} "
                        f"({progress_pct:.0f}%) - loss: {loss.item():.4f}"
                    )
        
        avg_loss = total_loss / len(val_loader)

        # Вычисляем метрики
        accuracy = accuracy_score(all_labels, all_predictions)

        # CRITICAL FIX: Use MACRO average instead of WEIGHTED for F1
        # Weighted F1 rewards mode collapse to majority class!
        # Macro F1 gives equal weight to all classes, penalizing mode collapse
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_predictions,
            average='weighted',
            zero_division=0
        )

        # MACRO F1 - this is what we optimize for
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_predictions,
            average='macro',
            zero_division=0
        )

        # ================================================================
        # MODE COLLAPSE EARLY WARNING
        # ================================================================
        # Check prediction distribution to detect mode collapse
        from collections import Counter
        pred_dist = Counter(all_predictions)
        total_preds = len(all_predictions)

        # Calculate what percentage of predictions are the majority class
        most_common_class, most_common_count = pred_dist.most_common(1)[0]
        majority_pct = most_common_count / total_preds

        # Determine mode collapse severity
        mode_collapse_severity = "none"
        if majority_pct >= 0.99:
            mode_collapse_severity = "total"
        elif majority_pct >= 0.90:
            mode_collapse_severity = "severe"
        elif majority_pct >= 0.75:
            mode_collapse_severity = "mild"

        # Log warnings based on severity
        if mode_collapse_severity == "total":
            tqdm.write(
                f"  🔴 TOTAL MODE COLLAPSE: {majority_pct:.1%} predictions are class {most_common_class}! "
                f"Distribution: {dict(pred_dist)}"
            )
            tqdm.write(
                f"     Weighted F1={f1_weighted:.4f} (misleading!), Macro F1={f1_macro:.4f} (true quality)"
            )
        elif mode_collapse_severity == "severe":
            tqdm.write(
                f"  🟠 SEVERE MODE COLLAPSE: {majority_pct:.1%} predictions are class {most_common_class}! "
                f"Distribution: {dict(pred_dist)}"
            )
        elif mode_collapse_severity == "mild":
            tqdm.write(
                f"  🟡 Prediction imbalance: {majority_pct:.1%} are class {most_common_class}. "
                f"Distribution: {dict(pred_dist)}"
            )

        # Return MACRO F1 as the primary metric (penalizes mode collapse)
        metrics = {
            'accuracy': accuracy,
            'precision': precision_macro,  # Changed to macro
            'recall': recall_macro,        # Changed to macro
            'f1': f1_macro,                # Changed to macro - CRITICAL!
            # Also store weighted for reference
            'f1_weighted': f1_weighted,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            # Mode collapse info
            'mode_collapse_severity': mode_collapse_severity,
            'majority_class_pct': majority_pct
        }

        return avg_loss, metrics
    
    def _test_model(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Тестирование модели с расширенной диагностикой.

        Returns:
            Dict with all test metrics for logging
        """
        from collections import Counter

        tqdm.write("\n" + "=" * 80)
        tqdm.write("ТЕСТИРОВАНИЕ МОДЕЛИ")
        tqdm.write("=" * 80)

        self.model.eval()

        all_predictions = []
        all_labels = []
        all_confidences = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", disable=not self.config.use_tqdm):
                sequences = batch['sequence'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(sequences)

                predictions = torch.argmax(
                    outputs['direction_logits'], dim=-1
                ).cpu().numpy()
                confidences = outputs['confidence'].squeeze(-1).cpu().numpy()

                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences)

        # Mode collapse detection
        pred_dist = Counter(all_predictions)
        label_dist = Counter(all_labels)
        total_preds = len(all_predictions)

        most_common_class, most_common_count = pred_dist.most_common(1)[0]
        majority_pct = most_common_count / total_preds

        # Determine mode collapse severity
        if majority_pct >= 0.99:
            mode_collapse_severity = "total"
        elif majority_pct >= 0.90:
            mode_collapse_severity = "severe"
        elif majority_pct >= 0.75:
            mode_collapse_severity = "mild"
        else:
            mode_collapse_severity = "none"

        # MACRO metrics (true quality measure)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_predictions,
            average='macro',
            zero_division=0
        )

        # WEIGHTED metrics (for reference)
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_predictions,
            average='weighted',
            zero_division=0
        )

        accuracy = accuracy_score(all_labels, all_predictions)

        tqdm.write(f"Test Results (MACRO - true quality):")
        tqdm.write(f"  • Accuracy: {accuracy:.4f}")
        tqdm.write(f"  • Precision (macro): {precision_macro:.4f}")
        tqdm.write(f"  • Recall (macro): {recall_macro:.4f}")
        tqdm.write(f"  • F1 Score (macro): {f1_macro:.4f}")

        if mode_collapse_severity != "none":
            tqdm.write(f"\n⚠️ MODE COLLAPSE DETECTED ({mode_collapse_severity}):")
            tqdm.write(f"  • {majority_pct:.1%} predictions are class {most_common_class}")
            tqdm.write(f"  • F1 (weighted): {f1_weighted:.4f} <- MISLEADING!")
            tqdm.write(f"  • F1 (macro): {f1_macro:.4f} <- TRUE QUALITY")
        else:
            tqdm.write(f"\nWeighted metrics (for reference):")
            tqdm.write(f"  • F1 (weighted): {f1_weighted:.4f}")

        # Class distribution
        class_names = ['SELL', 'HOLD', 'BUY']
        tqdm.write(f"\nClass Distribution:")
        tqdm.write(f"  Labels:      {dict((class_names[k], v) for k, v in sorted(label_dist.items()))}")
        tqdm.write(f"  Predictions: {dict((class_names[k], v) for k, v in sorted(pred_dist.items()))}")

        # Classification report
        report = classification_report(
            all_labels, all_predictions,
            target_names=class_names,
            digits=4
        )
        tqdm.write(f"\nClassification Report:\n{report}")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        tqdm.write(f"\nConfusion Matrix:")
        tqdm.write(f"           SELL  HOLD   BUY")
        for i, row in enumerate(cm):
            tqdm.write(f"  {class_names[i]:>6}  {row[0]:>5}  {row[1]:>5}  {row[2]:>5}")

        # Confidence analysis
        all_confidences = np.array(all_confidences)
        correct_mask = np.array(all_predictions) == np.array(all_labels)

        mean_conf_correct = all_confidences[correct_mask].mean() if correct_mask.any() else 0.0
        mean_conf_incorrect = all_confidences[~correct_mask].mean() if (~correct_mask).any() else 0.0

        tqdm.write(f"\nConfidence Analysis:")
        tqdm.write(f"  • Mean confidence (correct): {mean_conf_correct:.4f}")
        tqdm.write(f"  • Mean confidence (incorrect): {mean_conf_incorrect:.4f}")

        tqdm.write("=" * 80 + "\n")

        # Return results for logging
        return {
            'accuracy': accuracy,
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'mode_collapse_severity': mode_collapse_severity,
            'majority_class_pct': majority_pct,
            'confusion_matrix': cm.tolist(),
            'mean_confidence_correct': mean_conf_correct,
            'mean_confidence_incorrect': mean_conf_incorrect,
            'all_predictions': all_predictions,
            'all_labels': all_labels,
            'all_confidences': all_confidences.tolist()
        }
    
    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        metrics: Dict[str, float],
        name: str
    ):
        """Сохранение checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'metrics': metrics,
            'config': asdict(self.config),  # type: ignore[arg-type]
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'history': [m.to_dict() for m in self.history]
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        """Загрузка checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint
    
    @property
    def training(self) -> bool:
        """Проверка режима обучения."""
        return self.model.training


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_trainer_v2(
    model: nn.Module,
    config: Optional[TrainerConfigV2] = None,
    preset: str = "production_small"
) -> ModelTrainerV2:
    """
    Создать trainer с оптимизированными настройками.
    
    Args:
        model: Модель для обучения
        config: Конфигурация (если None - используется preset)
        preset: Пресет конфигурации
    
    Returns:
        ModelTrainerV2 instance
    """
    if config is None:
        if preset == "production_small":
            config = TrainerConfigV2(
                epochs=150,
                learning_rate=5e-5,
                batch_size=256,
                weight_decay=0.01,
                label_smoothing=0.1,
                use_augmentation=True,
                mixup_alpha=0.2,
                focal_gamma=2.5
            )
        elif preset == "quick_experiment":
            config = TrainerConfigV2(
                epochs=30,
                learning_rate=1e-4,
                batch_size=128,
                weight_decay=0.001,
                label_smoothing=0.0,
                use_augmentation=False,
                early_stopping_patience=10
            )
        else:
            config = TrainerConfigV2()
    
    return ModelTrainerV2(model, config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MODEL TRAINER V2 - EXAMPLE")
    print("=" * 80)
    
    # Создаём тестовую модель
    from backend.ml_engine.models.hybrid_cnn_lstm import create_model
    
    model = create_model()
    
    # Создаём trainer с оптимизированными настройками
    config = TrainerConfigV2(
        epochs=5,  # Для теста
        learning_rate=5e-5,
        batch_size=64,
        use_mixed_precision=False,
        verbose=True
    )
    
    trainer = create_trainer_v2(model, config)
    
    print("\n✅ ModelTrainerV2 создан успешно!")
    print(f"  • Device: {trainer.device}")
    print(f"  • Learning rate: {config.learning_rate}")
    print(f"  • Scheduler: {config.scheduler_type}")
    
    print("\n" + "=" * 80)
