#!/usr/bin/env python3
"""
Multi-Model Trainer - унифицированный тренер для всех моделей.

Поддерживаемые модели:
1. HybridCNNLSTMv2 - Основная модель
2. MPDTransformer - Vision Transformer для временных рядов
3. TLOBTransformer - Transformer для данных стакана

Функции:
- Единый интерфейс обучения для всех моделей
- Поддержка различных типов данных (features, raw LOB)
- MLflow интеграция
- Early stopping и checkpointing

Путь: backend/ml_engine/training/multi_model_trainer.py
"""

import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from backend.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class ModelArchitecture(Enum):
    """Поддерживаемые архитектуры."""
    CNN_LSTM = "cnn_lstm"
    MPD_TRANSFORMER = "mpd_transformer"
    TLOB = "tlob"


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MultiModelTrainerConfig:
    """Конфигурация тренера."""

    # === Model ===
    architecture: ModelArchitecture = ModelArchitecture.CNN_LSTM

    # === Training ===
    learning_rate: float = 5e-5
    batch_size: int = 64
    epochs: int = 150
    gradient_accumulation_steps: int = 4

    # === Optimizer ===
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # === Scheduler ===
    scheduler_T_0: int = 10
    scheduler_T_mult: int = 2

    # === Early Stopping ===
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001

    # === Loss ===
    direction_loss_weight: float = 1.0
    confidence_loss_weight: float = 0.3
    return_loss_weight: float = 0.3
    label_smoothing: float = 0.1

    # === Validation ===
    val_every_n_steps: int = 100

    # === Checkpointing ===
    save_every_n_epochs: int = 10
    checkpoint_dir: str = "checkpoints/models"

    # === Device ===
    device: str = "auto"

    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ============================================================================
# DATASET CLASSES
# ============================================================================

class FeatureDataset(Dataset):
    """Dataset для feature-based моделей (CNN-LSTM, MPDTransformer)."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None
    ):
        """
        Args:
            features: (N, seq_len, num_features)
            labels: (N,) - direction labels (0, 1, 2)
            confidences: (N,) - target confidence
            returns: (N,) - target return
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

        if confidences is not None:
            self.confidences = torch.FloatTensor(confidences)
        else:
            self.confidences = torch.zeros(len(labels))

        if returns is not None:
            self.returns = torch.FloatTensor(returns)
        else:
            self.returns = torch.zeros(len(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx],
            'confidence': self.confidences[idx],
            'expected_return': self.returns[idx]
        }


class LOBDataset(Dataset):
    """Dataset для TLOB модели."""

    def __init__(
        self,
        lob_data: np.ndarray,
        labels: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None
    ):
        """
        Args:
            lob_data: (N, seq_len, num_levels, 4) - raw LOB data
            labels: (N,) - direction labels
            confidences: (N,) - target confidence
            returns: (N,) - target return
        """
        self.lob_data = torch.FloatTensor(lob_data)
        self.labels = torch.LongTensor(labels)

        if confidences is not None:
            self.confidences = torch.FloatTensor(confidences)
        else:
            self.confidences = torch.zeros(len(labels))

        if returns is not None:
            self.returns = torch.FloatTensor(returns)
        else:
            self.returns = torch.zeros(len(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'lob_data': self.lob_data[idx],
            'label': self.labels[idx],
            'confidence': self.confidences[idx],
            'expected_return': self.returns[idx]
        }


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class MultiTaskLoss(nn.Module):
    """Multi-task loss для всех моделей."""

    def __init__(
        self,
        direction_weight: float = 1.0,
        confidence_weight: float = 0.3,
        return_weight: float = 0.3,
        label_smoothing: float = 0.1,
        num_classes: int = 3
    ):
        super().__init__()

        self.direction_weight = direction_weight
        self.confidence_weight = confidence_weight
        self.return_weight = return_weight

        self.direction_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
        self.confidence_loss = nn.MSELoss()
        self.return_loss = nn.MSELoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: Model outputs dict
            targets: Target dict

        Returns:
            total_loss: Combined loss
            loss_dict: Individual losses
        """
        # Direction loss
        dir_loss = self.direction_loss(
            outputs['direction_logits'],
            targets['label']
        )

        # Confidence loss
        conf_loss = self.confidence_loss(
            outputs['confidence'].squeeze(-1),
            targets['confidence']
        )

        # Return loss
        ret_loss = self.return_loss(
            outputs['expected_return'].squeeze(-1),
            targets['expected_return']
        )

        # Total loss
        total_loss = (
            self.direction_weight * dir_loss +
            self.confidence_weight * conf_loss +
            self.return_weight * ret_loss
        )

        loss_dict = {
            'direction_loss': dir_loss.item(),
            'confidence_loss': conf_loss.item(),
            'return_loss': ret_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


# ============================================================================
# MULTI-MODEL TRAINER
# ============================================================================

class MultiModelTrainer:
    """
    Унифицированный тренер для всех моделей.

    Поддерживает:
    - CNN-LSTM v2
    - MPDTransformer
    - TLOB Transformer

    Использование:
    ```python
    trainer = MultiModelTrainer(config)

    # Для CNN-LSTM / MPDTransformer
    trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        architecture=ModelArchitecture.CNN_LSTM
    )

    # Для TLOB
    trainer.train(
        model=tlob_model,
        train_loader=lob_train_loader,
        val_loader=lob_val_loader,
        architecture=ModelArchitecture.TLOB
    )
    ```
    """

    def __init__(self, config: Optional[MultiModelTrainerConfig] = None):
        """
        Args:
            config: Конфигурация тренера
        """
        self.config = config or MultiModelTrainerConfig()
        self.device = self.config.get_device()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # History
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []

        # Loss function
        self.loss_fn = MultiTaskLoss(
            direction_weight=self.config.direction_loss_weight,
            confidence_weight=self.config.confidence_loss_weight,
            return_weight=self.config.return_loss_weight,
            label_smoothing=self.config.label_smoothing
        )

        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"MultiModelTrainer initialized: "
            f"device={self.device}, "
            f"lr={self.config.learning_rate}, "
            f"epochs={self.config.epochs}"
        )

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        architecture: ModelArchitecture = ModelArchitecture.CNN_LSTM,
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Запускает обучение модели.

        Args:
            model: Модель для обучения
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации
            architecture: Тип архитектуры
            experiment_name: Название эксперимента

        Returns:
            Dict с результатами обучения
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 Starting training: {architecture.value}")
        logger.info(f"{'='*80}")

        # Move model to device
        model = model.to(self.device)

        # Setup optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Setup scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.scheduler_T_0,
            T_mult=self.config.scheduler_T_mult
        )

        # Training loop
        start_time = datetime.now()

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self._train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                architecture=architecture
            )

            self.train_history.append(train_metrics)

            # Validation
            if val_loader is not None:
                val_metrics = self._validate(
                    model=model,
                    val_loader=val_loader,
                    architecture=architecture
                )

                self.val_history.append(val_metrics)

                # Early stopping check
                if self._check_early_stopping(val_metrics['total_loss']):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

                # Log progress
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"Train Loss: {train_metrics['total_loss']:.4f} | "
                    f"Val Loss: {val_metrics['total_loss']:.4f} | "
                    f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"Train Loss: {train_metrics['total_loss']:.4f}"
                )

            # Checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    architecture=architecture,
                    epoch=epoch
                )

        # Training completed
        training_time = (datetime.now() - start_time).total_seconds()

        # Save final model
        self._save_checkpoint(
            model=model,
            optimizer=optimizer,
            architecture=architecture,
            epoch=self.current_epoch,
            is_final=True
        )

        results = {
            'architecture': architecture.value,
            'epochs_trained': self.current_epoch + 1,
            'training_time_seconds': training_time,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Training completed: {architecture.value}")
        logger.info(f"   Epochs: {self.current_epoch + 1}")
        logger.info(f"   Best Val Loss: {self.best_val_loss:.4f}")
        logger.info(f"   Time: {training_time/60:.1f} minutes")
        logger.info(f"{'='*80}")

        return results

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        architecture: ModelArchitecture
    ) -> Dict[str, float]:
        """Одна эпоха обучения."""
        model.train()

        total_loss = 0.0
        loss_components = {
            'direction_loss': 0.0,
            'confidence_loss': 0.0,
            'return_loss': 0.0
        }
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Get input based on architecture
            if architecture == ModelArchitecture.TLOB:
                inputs = batch['lob_data'].to(self.device)
            else:
                inputs = batch['features'].to(self.device)

            targets = {
                'label': batch['label'].to(self.device),
                'confidence': batch['confidence'].to(self.device),
                'expected_return': batch['expected_return'].to(self.device)
            }

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss, loss_dict = self.loss_fn(outputs, targets)

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.max_grad_norm
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                self.global_step += 1

            # Accumulate metrics
            total_loss += loss_dict['total_loss']
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            num_batches += 1

        # Average metrics
        metrics = {
            'total_loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in loss_components.items()},
            'learning_rate': scheduler.get_last_lr()[0]
        }

        return metrics

    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        architecture: ModelArchitecture
    ) -> Dict[str, float]:
        """Валидация модели."""
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        loss_components = {
            'direction_loss': 0.0,
            'confidence_loss': 0.0,
            'return_loss': 0.0
        }
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Get input based on architecture
                if architecture == ModelArchitecture.TLOB:
                    inputs = batch['lob_data'].to(self.device)
                else:
                    inputs = batch['features'].to(self.device)

                targets = {
                    'label': batch['label'].to(self.device),
                    'confidence': batch['confidence'].to(self.device),
                    'expected_return': batch['expected_return'].to(self.device)
                }

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss, loss_dict = self.loss_fn(outputs, targets)

                # Accuracy
                predictions = torch.argmax(outputs['direction_logits'], dim=1)
                correct += (predictions == targets['label']).sum().item()
                total += targets['label'].size(0)

                # Accumulate metrics
                total_loss += loss_dict['total_loss']
                for key in loss_components:
                    loss_components[key] += loss_dict[key]
                num_batches += 1

        # Average metrics
        metrics = {
            'total_loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in loss_components.items()},
            'accuracy': correct / total if total > 0 else 0.0
        }

        return metrics

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Проверяет условие early stopping."""
        if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        architecture: ModelArchitecture,
        epoch: int,
        is_final: bool = False
    ):
        """Сохраняет checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if is_final:
            filename = f"{architecture.value}_final_{timestamp}.pt"
        else:
            filename = f"{architecture.value}_epoch_{epoch}_{timestamp}.pt"

        filepath = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'architecture': architecture.value,
            'config': self.config.__dict__,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: str,
        load_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """Загружает checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])

        logger.info(f"Checkpoint loaded: {checkpoint_path}")

        return checkpoint

    async def register_model_mlflow(
        self,
        model: nn.Module,
        architecture: ModelArchitecture,
        metrics: Dict[str, float],
        training_params: Dict[str, Any]
    ) -> Optional[str]:
        """
        Регистрирует модель в MLflow.

        Args:
            model: Обученная модель
            architecture: Тип архитектуры
            metrics: Метрики модели
            training_params: Параметры обучения

        Returns:
            model_version или None при ошибке
        """
        try:
            # Import MLflow tracker
            from backend.ml_engine.mlflow_integration.mlflow_tracker import MLflowTracker
            from backend.core.config import settings

            # Initialize tracker
            tracker = MLflowTracker(
                tracking_uri=settings.MLFLOW_TRACKING_URI,
                experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
                artifact_location=settings.MLFLOW_ARTIFACT_LOCATION
            )

            # Model name mapping
            model_names = {
                ModelArchitecture.CNN_LSTM: "hybrid_cnn_lstm",
                ModelArchitecture.MPD_TRANSFORMER: "mpd_transformer",
                ModelArchitecture.TLOB: "tlob_transformer"
            }
            model_name = model_names.get(architecture, architecture.value)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{model_name}_{timestamp}"

            # Start MLflow run
            run_id = tracker.start_run(
                run_name=run_name,
                tags={
                    "model_type": architecture.value,
                    "framework": "pytorch"
                },
                description=f"Training run for {model_name}"
            )

            # Log params
            tracker.log_params({
                "architecture": architecture.value,
                "learning_rate": training_params.get("learning_rate", self.config.learning_rate),
                "batch_size": training_params.get("batch_size", self.config.batch_size),
                "epochs": training_params.get("epochs", self.config.epochs),
                "weight_decay": self.config.weight_decay,
                "dropout": training_params.get("dropout", 0.1)
            })

            # Log metrics
            tracker.log_metrics(metrics)

            # Save model artifact
            model_path = self.checkpoint_dir / f"{model_name}_{timestamp}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'architecture': architecture.value,
                'config': self.config.__dict__,
                'metrics': metrics
            }, model_path)

            # Log model to MLflow
            model_uri = tracker.log_model(
                model=model,
                model_name=model_name,
                artifacts={"model_checkpoint": str(model_path)}
            )

            # Register model
            model_tags = {
                "framework": "pytorch",
                "model_type": architecture.value,
                "training_mode": "manual",
                "accuracy": f"{metrics.get('accuracy', 0):.4f}",
                "val_loss": f"{metrics.get('val_loss', 0):.4f}",
                "timestamp": timestamp
            }

            model_version = tracker.register_model(
                model_uri=model_uri,
                model_name=model_name,
                tags=model_tags,
                description=f"Trained {model_name} model. Accuracy: {metrics.get('accuracy', 0):.4f}"
            )

            tracker.end_run(status="FINISHED")

            logger.info(f"Model registered in MLflow: {model_name} v{model_version}")

            return model_version

        except ImportError:
            logger.warning("MLflow not available, skipping registration")
            return None
        except Exception as e:
            logger.error(f"Failed to register model in MLflow: {e}")
            return None

    async def register_model_registry(
        self,
        model: nn.Module,
        architecture: ModelArchitecture,
        metrics: Dict[str, float],
        training_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Регистрирует модель во внутреннем реестре.

        Args:
            model: Обученная модель
            architecture: Тип архитектуры
            metrics: Метрики модели
            training_params: Параметры обучения

        Returns:
            ModelInfo или None при ошибке
        """
        try:
            from backend.ml_engine.inference.model_registry import ModelRegistry

            # Initialize registry
            registry = ModelRegistry()

            # Model name mapping
            model_names = {
                ModelArchitecture.CNN_LSTM: "hybrid_cnn_lstm",
                ModelArchitecture.MPD_TRANSFORMER: "mpd_transformer",
                ModelArchitecture.TLOB: "tlob_transformer"
            }
            model_name = model_names.get(architecture, architecture.value)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save model
            model_path = self.checkpoint_dir / f"{model_name}_{timestamp}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'architecture': architecture.value,
                'config': self.config.__dict__,
                'metrics': metrics
            }, model_path)

            # Register
            model_info = await registry.register_model(
                name=model_name,
                version=timestamp,
                model_path=str(model_path),
                model_type=architecture.value,
                description=f"Trained {model_name} model",
                metrics=metrics,
                training_params=training_params
            )

            logger.info(f"Model registered: {model_name} v{timestamp}")

            return model_info

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_trainer(
    architecture: str = "cnn_lstm",
    learning_rate: float = 5e-5,
    epochs: int = 150
) -> MultiModelTrainer:
    """
    Создает тренер для указанной архитектуры.

    Args:
        architecture: Тип архитектуры
        learning_rate: Learning rate
        epochs: Количество эпох

    Returns:
        MultiModelTrainer
    """
    config = MultiModelTrainerConfig(
        architecture=ModelArchitecture(architecture),
        learning_rate=learning_rate,
        epochs=epochs
    )

    return MultiModelTrainer(config)


def create_feature_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Создает DataLoader для feature-based моделей.

    Args:
        features: (N, seq_len, num_features)
        labels: (N,)
        batch_size: Размер батча
        shuffle: Перемешивать ли данные

    Returns:
        DataLoader
    """
    dataset = FeatureDataset(features, labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def create_lob_dataloader(
    lob_data: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Создает DataLoader для TLOB модели.

    Args:
        lob_data: (N, seq_len, num_levels, 4)
        labels: (N,)
        batch_size: Размер батча
        shuffle: Перемешивать ли данные

    Returns:
        DataLoader
    """
    dataset = LOBDataset(lob_data, labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-MODEL TRAINER TEST")
    print("=" * 80)

    # Create trainer
    trainer = create_trainer(
        architecture="cnn_lstm",
        learning_rate=5e-5,
        epochs=5
    )

    # Create dummy data
    features = np.random.randn(1000, 60, 112).astype(np.float32)
    labels = np.random.randint(0, 3, 1000)

    # Create dataloader
    train_loader = create_feature_dataloader(
        features[:800],
        labels[:800],
        batch_size=64
    )

    val_loader = create_feature_dataloader(
        features[800:],
        labels[800:],
        batch_size=64,
        shuffle=False
    )

    print(f"\n📊 Dataset sizes:")
    print(f"   • Train: {len(train_loader.dataset)}")
    print(f"   • Val: {len(val_loader.dataset)}")

    # Create simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(60 * 112, 256)
            self.direction_head = nn.Linear(256, 3)
            self.confidence_head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
            self.return_head = nn.Linear(256, 1)

        def forward(self, x):
            x = x.reshape(x.size(0), -1)
            x = F.relu(self.fc(x))
            return {
                'direction_logits': self.direction_head(x),
                'confidence': self.confidence_head(x),
                'expected_return': self.return_head(x)
            }

    model = SimpleModel()

    print(f"\n🔄 Starting training...")

    results = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        architecture=ModelArchitecture.CNN_LSTM
    )

    print(f"\n📈 Results:")
    print(f"   • Epochs: {results['epochs_trained']}")
    print(f"   • Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"   • Time: {results['training_time_seconds']:.1f}s")

    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print("=" * 80)
