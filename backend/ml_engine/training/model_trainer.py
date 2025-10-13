"""
Model Trainer для обучения ML моделей.

Функциональность:
- Multi-task learning с комбинированным loss
- Early stopping с patience
- Learning rate scheduling
- Gradient clipping для стабильности
- Метрики: accuracy, precision, recall, F1, ROC-AUC
- Checkpoint сохранение лучшей модели
- TensorBoard logging

Путь: backend/ml_engine/training/model_trainer.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Callable, List
from pathlib import Path
import time
from dataclasses import dataclass, field
import numpy as np
from sklearn.metrics import (
  accuracy_score, precision_recall_fscore_support,
  roc_auc_score, confusion_matrix
)

from core.logger import get_logger
from ml_engine.models.hybrid_cnn_lstm import HybridCNNLSTM, ModelConfig

logger = get_logger(__name__)


@dataclass
class TrainerConfig:
  """Конфигурация обучения."""
  # Training параметры
  epochs: int = 100
  learning_rate: float = 0.001
  weight_decay: float = 1e-5
  grad_clip_value: float = 1.0

  # Early stopping
  early_stopping_patience: int = 10
  early_stopping_delta: float = 1e-4

  # Loss weights для multi-task learning
  direction_loss_weight: float = 1.0
  confidence_loss_weight: float = 0.5
  return_loss_weight: float = 0.3

  # Learning rate scheduling
  lr_scheduler: str = "ReduceLROnPlateau"  # или "CosineAnnealing"
  lr_patience: int = 5
  lr_factor: float = 0.5

  # Checkpoint
  checkpoint_dir: str = "checkpoints/models"
  save_best_only: bool = True

  # Device
  device: str = "cuda" if torch.cuda.is_available() else "cpu"

  # Logging
  log_interval: int = 10  # Log every N batches


@dataclass
class TrainingMetrics:
  """Метрики обучения."""
  epoch: int
  train_loss: float
  val_loss: float
  train_accuracy: float
  val_accuracy: float
  val_precision: float
  val_recall: float
  val_f1: float
  val_auc: Optional[float] = None
  learning_rate: float = 0.0
  epoch_time: float = 0.0


class MultiTaskLoss(nn.Module):
  """
  Комбинированный loss для multi-task learning.

  Компоненты:
  - CrossEntropyLoss для direction classification
  - MSELoss для confidence regression
  - MSELoss для expected return regression
  """

  def __init__(
      self,
      direction_weight: float = 1.0,
      confidence_weight: float = 0.5,
      return_weight: float = 0.3
  ):
    super().__init__()

    self.direction_weight = direction_weight
    self.confidence_weight = confidence_weight
    self.return_weight = return_weight

    # Loss functions
    self.direction_criterion = nn.CrossEntropyLoss()
    self.confidence_criterion = nn.MSELoss()
    self.return_criterion = nn.MSELoss()

    logger.info(
      f"Инициализирован MultiTaskLoss: "
      f"direction={direction_weight}, "
      f"confidence={confidence_weight}, "
      f"return={return_weight}"
    )

  def forward(
      self,
      outputs: Dict[str, torch.Tensor],
      targets: Dict[str, torch.Tensor]
  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Вычислить комбинированный loss.

    Args:
        outputs: Dict с предсказаниями модели
        targets: Dict с истинными значениями

    Returns:
        total_loss: Суммарный loss
        loss_components: Dict с компонентами loss
    """
    # Direction classification loss
    direction_loss = self.direction_criterion(
      outputs['direction_logits'],
      targets['label']
    )

    # Confidence regression loss (если есть targets)
    confidence_loss = torch.tensor(0.0, device=direction_loss.device)
    if 'confidence' in targets:
      confidence_loss = self.confidence_criterion(
        outputs['confidence'].squeeze(),
        targets['confidence']
      )

    # Expected return regression loss (если есть targets)
    return_loss = torch.tensor(0.0, device=direction_loss.device)
    if 'return' in targets:
      return_loss = self.return_criterion(
        outputs['expected_return'].squeeze(),
        targets['return']
      )

    # Комбинированный loss
    total_loss = (
        self.direction_weight * direction_loss +
        self.confidence_weight * confidence_loss +
        self.return_weight * return_loss
    )

    loss_components = {
      'direction': direction_loss.item(),
      'confidence': confidence_loss.item(),
      'return': return_loss.item(),
      'total': total_loss.item()
    }

    return total_loss, loss_components


class EarlyStopping:
  """Early stopping для предотвращения переобучения."""

  def __init__(self, patience: int = 10, delta: float = 1e-4):
    """
    Args:
        patience: Количество эпох без улучшения
        delta: Минимальное изменение для считаться улучшением
    """
    self.patience = patience
    self.delta = delta
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.inf

    logger.info(f"Инициализирован EarlyStopping: patience={patience}")

  def __call__(self, val_loss: float) -> bool:
    """
    Проверить условие early stopping.

    Args:
        val_loss: Validation loss текущей эпохи

    Returns:
        True если нужно остановить обучение
    """
    score = -val_loss

    if self.best_score is None:
      self.best_score = score
      self.val_loss_min = val_loss
    elif score < self.best_score + self.delta:
      self.counter += 1
      logger.info(
        f"EarlyStopping counter: {self.counter}/{self.patience}"
      )
      if self.counter >= self.patience:
        self.early_stop = True
    else:
      self.best_score = score
      self.val_loss_min = val_loss
      self.counter = 0

    return self.early_stop


class ModelTrainer:
  """
  Trainer для обучения ML моделей.

  Поддерживает:
  - Multi-task learning
  - Early stopping
  - Learning rate scheduling
  - Gradient clipping
  - Checkpoint management
  """

  def __init__(
      self,
      model: HybridCNNLSTM,
      config: TrainerConfig
  ):
    """
    Инициализация trainer.

    Args:
        model: Модель для обучения
        config: Конфигурация обучения
    """
    self.model = model
    self.config = config
    self.device = torch.device(config.device)

    # Перемещаем модель на device
    self.model.to(self.device)

    # Loss function
    self.criterion = MultiTaskLoss(
      direction_weight=config.direction_loss_weight,
      confidence_weight=config.confidence_loss_weight,
      return_weight=config.return_loss_weight
    )

    # Optimizer
    self.optimizer = optim.AdamW(
      self.model.parameters(),
      lr=config.learning_rate,
      weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    if config.lr_scheduler == "ReduceLROnPlateau":
      self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer,
        mode='min',
        patience=config.lr_patience,
        factor=config.lr_factor
      )
    elif config.lr_scheduler == "CosineAnnealing":
      self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer,
        T_max=config.epochs
      )
    else:
      self.scheduler = None

    # Early stopping
    self.early_stopping = EarlyStopping(
      patience=config.early_stopping_patience,
      delta=config.early_stopping_delta
    )

    # Checkpoint directory
    self.checkpoint_dir = Path(config.checkpoint_dir)
    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    self.history: List[TrainingMetrics] = []
    self.best_val_loss = np.inf

    logger.info(
      f"Инициализирован ModelTrainer: device={self.device}, "
      f"lr={config.learning_rate}, epochs={config.epochs}"
    )

  def train_epoch(
      self,
      train_loader: DataLoader
  ) -> Tuple[float, float]:
    """
    Обучить одну эпоху.

    Args:
        train_loader: DataLoader для обучающих данных

    Returns:
        avg_loss: Средний loss
        accuracy: Точность предсказаний
    """
    self.model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch_idx, batch in enumerate(train_loader):
      # Перемещаем данные на device (type: ignore для DataLoader batch type)
      sequences = batch['sequence'].to(self.device)  # type: ignore[index]
      labels = batch['label'].to(self.device)  # type: ignore[index]

      # Forward pass
      outputs = self.model(sequences)

      # Вычисляем loss
      targets = {'label': labels}
      loss, loss_components = self.criterion(outputs, targets)

      # Backward pass
      self.optimizer.zero_grad()
      loss.backward()

      # Gradient clipping
      if self.config.grad_clip_value > 0:
        torch.nn.utils.clip_grad_norm_(
          self.model.parameters(),
          self.config.grad_clip_value
        )

      self.optimizer.step()

      # Метрики
      total_loss += loss.item()

      # Predictions для accuracy
      predictions = torch.argmax(
        outputs['direction_logits'],
        dim=-1
      ).cpu().numpy()
      all_predictions.extend(predictions)
      all_labels.extend(labels.cpu().numpy())

      # Логирование
      if batch_idx % self.config.log_interval == 0:
        logger.debug(
          f"Batch {batch_idx}/{len(train_loader)}: "
          f"loss={loss.item():.4f}"
        )

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_predictions)

    return avg_loss, accuracy

  def validate_epoch(
      self,
      val_loader: DataLoader
  ) -> Tuple[float, Dict[str, float]]:
    """
    Валидация одной эпохи.

    Args:
        val_loader: DataLoader для валидационных данных

    Returns:
        avg_loss: Средний validation loss
        metrics: Dict с метриками
    """
    self.model.eval()

    total_loss = 0.0
    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
      for batch in val_loader:
        sequences = batch['sequence'].to(self.device)
        labels = batch['label'].to(self.device)

        # Forward pass
        outputs = self.model(sequences)

        # Loss
        targets = {'label': labels}
        loss, _ = self.criterion(outputs, targets)
        total_loss += loss.item()

        # Predictions
        probs = torch.softmax(
          outputs['direction_logits'],
          dim=-1
        ).cpu().numpy()
        predictions = np.argmax(probs, axis=-1)

        all_predictions.extend(predictions)
        all_probabilities.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)

    # Вычисляем метрики
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
      all_labels,
      all_predictions,
      average='macro',
      zero_division=0
    )

    # ROC-AUC для multi-class
    try:
      auc = roc_auc_score(
        all_labels,
        np.array(all_probabilities),
        multi_class='ovr',
        average='macro'
      )
    except Exception as e:
      logger.warning(f"Не удалось вычислить AUC: {e}")
      auc = None

    metrics = {
      'accuracy': accuracy,
      'precision': precision,
      'recall': recall,
      'f1': f1,
      'auc': auc
    }

    return avg_loss, metrics

  def save_checkpoint(
      self,
      epoch: int,
      metrics: TrainingMetrics,
      is_best: bool = False
  ):
    """
    Сохранить checkpoint модели.

    Args:
        epoch: Номер эпохи
        metrics: Метрики эпохи
        is_best: Лучшая модель
    """
    checkpoint = {
      'epoch': epoch,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'metrics': metrics,
      'config': self.config
    }

    if is_best:
      path = self.checkpoint_dir / "best_model.pth"
      torch.save(checkpoint, path)
      logger.info(f"Сохранена лучшая модель: {path}")

    # Также сохраняем последнюю модель
    path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, path)

  def load_checkpoint(self, path: str):
    """Загрузить checkpoint."""
    checkpoint = torch.load(path, map_location=self.device)

    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.info(f"Загружен checkpoint: {path}")
    return checkpoint

  def train(
      self,
      train_loader: DataLoader,
      val_loader: DataLoader
  ) -> List[TrainingMetrics]:
    """
    Полный цикл обучения.

    Args:
        train_loader: DataLoader для обучения
        val_loader: DataLoader для валидации

    Returns:
        История обучения
    """
    logger.info(
      f"Начало обучения: epochs={self.config.epochs}, "
      f"train_batches={len(train_loader)}, "
      f"val_batches={len(val_loader)}"
    )

    for epoch in range(self.config.epochs):
      epoch_start = time.time()

      # Training
      train_loss, train_accuracy = self.train_epoch(train_loader)

      # Validation
      val_loss, val_metrics = self.validate_epoch(val_loader)

      # Learning rate scheduling
      if self.scheduler is not None:
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
          self.scheduler.step(val_loss)
        else:
          self.scheduler.step()

      current_lr = self.optimizer.param_groups[0]['lr']
      epoch_time = time.time() - epoch_start

      # Создаем метрики эпохи
      metrics = TrainingMetrics(
        epoch=epoch + 1,
        train_loss=train_loss,
        val_loss=val_loss,
        train_accuracy=train_accuracy,
        val_accuracy=val_metrics['accuracy'],
        val_precision=val_metrics['precision'],
        val_recall=val_metrics['recall'],
        val_f1=val_metrics['f1'],
        val_auc=val_metrics['auc'],
        learning_rate=current_lr,
        epoch_time=epoch_time
      )

      self.history.append(metrics)

      # Логирование
      logger.info(
        f"Epoch {epoch + 1}/{self.config.epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f} | "
        f"LR: {current_lr:.6f} | "
        f"Time: {epoch_time:.1f}s"
      )

      # Сохранение checkpoint
      is_best = val_loss < self.best_val_loss
      if is_best:
        self.best_val_loss = val_loss

      if self.config.save_best_only:
        if is_best:
          self.save_checkpoint(epoch + 1, metrics, is_best=True)
      else:
        self.save_checkpoint(epoch + 1, metrics, is_best=is_best)

      # Early stopping
      if self.early_stopping(val_loss):
        logger.info(
          f"Early stopping на эпохе {epoch + 1}. "
          f"Лучший val_loss: {self.best_val_loss:.4f}"
        )
        break

    logger.info("Обучение завершено")
    return self.history


# Пример использования
if __name__ == "__main__":
  from ml_engine.models.hybrid_cnn_lstm import create_model
  from ml_engine.training.data_loader import HistoricalDataLoader, DataConfig

  # Создаем модель
  model = create_model()

  # Конфигурация trainer
  trainer_config = TrainerConfig(
    epochs=50,
    learning_rate=0.001,
    early_stopping_patience=10,
    checkpoint_dir="checkpoints/models"
  )

  # Создаем trainer
  trainer = ModelTrainer(model, trainer_config)

  # Загружаем данные
  data_config = DataConfig(
    storage_path="data/ml_training",
    batch_size=64
  )
  loader = HistoricalDataLoader(data_config)

  # Подготавливаем данные
  result = loader.load_and_prepare(["BTCUSDT", "ETHUSDT"])
  dataloaders = result['dataloaders']

  # Обучаем модель
  history = trainer.train(
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val']
  )

  print(f"\nОбучение завершено. Эпох: {len(history)}")
  print(f"Лучший val_loss: {trainer.best_val_loss:.4f}")