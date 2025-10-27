#!/usr/bin/env python3
"""
Class Balancing Module - методы для борьбы с дисбалансом классов.

Функциональность:
1. Class Weights - автоматический расчет весов для loss function
2. Focal Loss - динамическая фокусировка на сложных примерах
3. Oversampling/Undersampling - балансировка датасета
4. SMOTE - синтетическая генерация минорных классов

Путь: backend/ml_engine/training/class_balancing.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
from dataclasses import dataclass
from sklearn.utils.class_weight import compute_class_weight

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClassBalancingConfig:
  """Конфигурация балансировки классов."""
  # Методы балансировки
  use_class_weights: bool = True
  use_focal_loss: bool = False
  use_oversampling: bool = False
  use_undersampling: bool = False
  use_smote: bool = False

  # Параметры Focal Loss
  focal_alpha: Optional[float] = 0.25  # Вес для класса 1
  focal_gamma: float = 2.0  # Focusing parameter

  # Параметры Oversampling
  oversample_strategy: str = "auto"  # "auto", "minority", "all"
  oversample_ratio: float = 1.0  # Целевое соотношение

  # Параметры Undersampling
  undersample_strategy: str = "random"  # "random", "tomek", "enn"
  undersample_ratio: float = 1.0

  # Параметры SMOTE
  smote_k_neighbors: int = 5
  smote_sampling_strategy: str = "auto"

  # Логирование
  verbose: bool = True


class ClassWeightCalculator:
  """Калькулятор весов классов."""

  @staticmethod
  def compute_weights(
      labels: np.ndarray,
      method: str = "balanced",
      normalize: bool = True
  ) -> Dict[int, float]:
    """
    Вычислить веса классов.

    Args:
        labels: Массив меток классов
        method: Метод расчета ("balanced", "inverse_freq", "effective_samples")
        normalize: Нормализовать веса

    Returns:
        Dict с весами для каждого класса
    """
    unique_classes = np.unique(labels)
    n_samples = len(labels)
    n_classes = len(unique_classes)

    if method == "balanced":
      # Sklearn balanced weights: n_samples / (n_classes * n_samples_per_class)
      weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
      )
      class_weights = dict(zip(unique_classes, weights))

    elif method == "inverse_freq":
      # Обратная частота: 1 / freq
      counter = Counter(labels)
      class_weights = {
        cls: n_samples / (n_classes * count)
        for cls, count in counter.items()
      }

    elif method == "effective_samples":
      # Effective number of samples (для сильного дисбаланса)
      beta = 0.9999
      counter = Counter(labels)
      class_weights = {}

      for cls, count in counter.items():
        effective_num = 1.0 - np.power(beta, count)
        class_weights[cls] = (1.0 - beta) / effective_num

    else:
      raise ValueError(f"Неизвестный метод: {method}")

    # Нормализация весов
    if normalize:
      total_weight = sum(class_weights.values())
      class_weights = {
        cls: weight / total_weight * n_classes
        for cls, weight in class_weights.items()
      }

    logger.info(f"Class weights ({method}): {class_weights}")
    return class_weights

  @staticmethod
  def weights_to_tensor(
      class_weights: Dict[int, float],
      device: str = "cpu"
  ) -> torch.Tensor:
    """
    Конвертировать словарь весов в Tensor для PyTorch.

    Args:
        class_weights: Dict {class: weight}
        device: Device для Tensor

    Returns:
        Tensor весов в порядке классов
    """
    sorted_classes = sorted(class_weights.keys())
    weights = [class_weights[cls] for cls in sorted_classes]
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
  """
  Focal Loss для борьбы с дисбалансом классов.

  Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

  Где:
  - p_t: вероятность правильного класса
  - alpha: балансировочный вес
  - gamma: focusing parameter (чем больше, тем сильнее фокусировка на сложных примерах)

  Преимущества:
  - Автоматически фокусируется на сложных примерах
  - Уменьшает вес легко классифицируемых примеров
  - Эффективен при сильном дисбалансе (до 1000:1)

  Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
  """

  def __init__(
      self,
      alpha: Optional[torch.Tensor] = None,
      gamma: float = 2.0,
      reduction: str = 'mean',
      ignore_index: int = -100
  ):
    """
    Инициализация Focal Loss.

    Args:
        alpha: Веса классов [C]. Если None, все классы равны
        gamma: Focusing parameter. Рекомендуется 2.0
        reduction: 'none', 'mean', 'sum'
        ignore_index: Индекс класса для игнорирования
    """
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction
    self.ignore_index = ignore_index

    logger.info(
      f"Инициализирован Focal Loss: "
      f"gamma={gamma}, alpha={alpha is not None}"
    )

  def forward(
      self,
      inputs: torch.Tensor,
      targets: torch.Tensor
  ) -> torch.Tensor:
    """
    Forward pass.

    Args:
        inputs: Logits модели [N, C]
        targets: Целевые классы [N]

    Returns:
        Focal loss
    """
    # Softmax для получения вероятностей
    p = F.softmax(inputs, dim=-1)  # [N, C]

    # Вероятности правильных классов
    ce_loss = F.cross_entropy(
      inputs, targets,
      reduction='none',
      ignore_index=self.ignore_index
    )  # [N]

    # p_t = вероятность правильного класса
    p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]

    # Focal term: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** self.gamma

    # Focal loss
    focal_loss = focal_weight * ce_loss

    # Alpha weighting (если задано)
    if self.alpha is not None:
      if self.alpha.device != inputs.device:
        self.alpha = self.alpha.to(inputs.device)

      # Получаем alpha для каждого семпла
      alpha_t = self.alpha.gather(0, targets)
      focal_loss = alpha_t * focal_loss

    # Reduction
    if self.reduction == 'mean':
      return focal_loss.mean()
    elif self.reduction == 'sum':
      return focal_loss.sum()
    else:
      return focal_loss


class DatasetBalancer:
  """Балансировщик датасета через oversampling/undersampling."""

  @staticmethod
  def oversample(
      X: np.ndarray,
      y: np.ndarray,
      strategy: str = "auto",
      random_state: int = 42
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oversampling минорных классов.

    Args:
        X: Features [N, ...]
        y: Labels [N]
        strategy: "auto", "minority", "all", или dict {class: n_samples}
        random_state: Random seed

    Returns:
        X_resampled, y_resampled
    """
    try:
      from imblearn.over_sampling import RandomOverSampler
    except ImportError:
      logger.error("imbalanced-learn не установлен. pip install imbalanced-learn")
      return X, y

    ros = RandomOverSampler(
      sampling_strategy=strategy,
      random_state=random_state
    )

    # Reshape для imblearn (требуется 2D)
    original_shape = X.shape
    X_2d = X.reshape(len(X), -1)

    X_resampled, y_resampled = ros.fit_resample(X_2d, y)

    # Reshape обратно
    X_resampled = X_resampled.reshape(-1, *original_shape[1:])

    logger.info(
      f"Oversampling: {len(X)} → {len(X_resampled)} семплов"
    )
    logger.info(f"Распределение классов: {Counter(y_resampled)}")

    return X_resampled, y_resampled

  @staticmethod
  def undersample(
      X: np.ndarray,
      y: np.ndarray,
      strategy: str = "auto",
      random_state: int = 42
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Undersampling мажорных классов.

    Args:
        X: Features [N, ...]
        y: Labels [N]
        strategy: "auto", "majority", "all", или dict
        random_state: Random seed

    Returns:
        X_resampled, y_resampled
    """
    try:
      from imblearn.under_sampling import RandomUnderSampler
    except ImportError:
      logger.error("imbalanced-learn не установлен")
      return X, y

    rus = RandomUnderSampler(
      sampling_strategy=strategy,
      random_state=random_state
    )

    original_shape = X.shape
    X_2d = X.reshape(len(X), -1)

    X_resampled, y_resampled = rus.fit_resample(X_2d, y)
    X_resampled = X_resampled.reshape(-1, *original_shape[1:])

    logger.info(
      f"Undersampling: {len(X)} → {len(X_resampled)} семплов"
    )
    logger.info(f"Распределение классов: {Counter(y_resampled)}")

    return X_resampled, y_resampled

  @staticmethod
  def smote(
      X: np.ndarray,
      y: np.ndarray,
      k_neighbors: int = 5,
      strategy: str = "auto",
      random_state: int = 42
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTE - Synthetic Minority Over-sampling Technique.

    Создает синтетические примеры минорных классов путем
    интерполяции между соседними примерами.

    Args:
        X: Features [N, ...]
        y: Labels [N]
        k_neighbors: Количество соседей для SMOTE
        strategy: Стратегия сэмплирования
        random_state: Random seed

    Returns:
        X_resampled, y_resampled
    """
    try:
      from imblearn.over_sampling import SMOTE
    except ImportError:
      logger.error("imbalanced-learn не установлен")
      return X, y

    # SMOTE работает только с 2D данными
    original_shape = X.shape
    X_2d = X.reshape(len(X), -1)

    smote = SMOTE(
      sampling_strategy=strategy,
      k_neighbors=k_neighbors,
      random_state=random_state
    )

    X_resampled, y_resampled = smote.fit_resample(X_2d, y)
    X_resampled = X_resampled.reshape(-1, *original_shape[1:])

    logger.info(
      f"SMOTE: {len(X)} → {len(X_resampled)} семплов"
    )
    logger.info(f"Распределение классов: {Counter(y_resampled)}")

    return X_resampled, y_resampled


class ClassBalancingStrategy:
  """
  Unified стратегия балансировки классов.

  Применяет один или несколько методов балансировки.
  """

  def __init__(self, config: ClassBalancingConfig):
    """Инициализация стратегии."""
    self.config = config

    logger.info("Инициализирована стратегия балансировки классов")
    logger.info(f"  • Class weights: {config.use_class_weights}")
    logger.info(f"  • Focal Loss: {config.use_focal_loss}")
    logger.info(f"  • Oversampling: {config.use_oversampling}")
    logger.info(f"  • Undersampling: {config.use_undersampling}")
    logger.info(f"  • SMOTE: {config.use_smote}")

  def get_loss_function(
      self,
      labels: np.ndarray,
      device: Union[str, torch.device] = "cpu"
  ) -> nn.Module:
    """
    Получить loss function с балансировкой.

    Args:
        labels: Обучающие метки для расчета весов
        device: Device для Tensor

    Returns:
        Loss function (nn.Module)
    """
    if not isinstance(device, str):
      device = str(device)

    if self.config.use_focal_loss:
      # Focal Loss (с опциональными class weights)
      alpha = None

      if self.config.use_class_weights:
        class_weights = ClassWeightCalculator.compute_weights(
          labels, method="balanced"
        )
        alpha = ClassWeightCalculator.weights_to_tensor(
          class_weights, device
        )

      loss_fn = FocalLoss(
        alpha=alpha,
        gamma=self.config.focal_gamma,
        reduction='mean'
      )

    elif self.config.use_class_weights:
      # Standard CrossEntropyLoss с class weights
      class_weights = ClassWeightCalculator.compute_weights(
        labels, method="balanced"
      )
      weight_tensor = ClassWeightCalculator.weights_to_tensor(
        class_weights, device
      )

      loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    else:
      # Без балансировки
      loss_fn = nn.CrossEntropyLoss()

    return loss_fn

  def balance_dataset(
      self,
      X: np.ndarray,
      y: np.ndarray,
      random_state: int = 42
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Применить resampling к датасету.

    Args:
        X: Features
        y: Labels
        random_state: Random seed

    Returns:
        X_balanced, y_balanced
    """
    X_balanced, y_balanced = X, y

    # SMOTE (применяется первым)
    if self.config.use_smote:
      X_balanced, y_balanced = DatasetBalancer.smote(
        X_balanced, y_balanced,
        k_neighbors=self.config.smote_k_neighbors,
        strategy=self.config.smote_sampling_strategy,
        random_state=random_state
      )

    # Oversampling
    if self.config.use_oversampling:
      X_balanced, y_balanced = DatasetBalancer.oversample(
        X_balanced, y_balanced,
        strategy=self.config.oversample_strategy,
        random_state=random_state
      )

    # Undersampling (применяется последним)
    if self.config.use_undersampling:
      X_balanced, y_balanced = DatasetBalancer.undersample(
        X_balanced, y_balanced,
        strategy=self.config.undersample_strategy,
        random_state=random_state
      )

    return X_balanced, y_balanced

  def print_class_distribution(
      self,
      y_before: np.ndarray,
      y_after: np.ndarray
  ):
    """Печать распределения классов до/после балансировки."""
    print("\n" + "=" * 60)
    print("РАСПРЕДЕЛЕНИЕ КЛАССОВ")
    print("=" * 60)

    before = Counter(y_before)
    after = Counter(y_after)

    print("\nДО балансировки:")
    for cls in sorted(before.keys()):
      count = before[cls]
      pct = (count / len(y_before)) * 100
      print(f"  Класс {cls:2d}: {count:8,} ({pct:5.1f}%)")

    print(f"\n  Всего: {len(y_before):,}")
    print(f"  Imbalance Ratio: {max(before.values()) / min(before.values()):.2f}")

    print("\nПОСЛЕ балансировки:")
    for cls in sorted(after.keys()):
      count = after[cls]
      pct = (count / len(y_after)) * 100
      print(f"  Класс {cls:2d}: {count:8,} ({pct:5.1f}%)")

    print(f"\n  Всего: {len(y_after):,}")
    print(f"  Imbalance Ratio: {max(after.values()) / min(after.values()):.2f}")
    print("=" * 60 + "\n")


# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========

if __name__ == "__main__":
  # Пример с искусственным дисбалансом
  np.random.seed(42)

  # Создаем несбалансированный датасет
  n_samples = 10000
  n_features = 110

  # Сильный дисбаланс: 70% класс 0, 20% класс 1, 10% класс -1
  y = np.concatenate([
    np.zeros(7000, dtype=int),
    np.ones(2000, dtype=int),
    np.full(1000, -1, dtype=int)
  ])
  np.random.shuffle(y)

  X = np.random.randn(n_samples, 60, n_features)  # (N, seq_len, features)

  print(f"Исходный датасет: X={X.shape}, y={y.shape}")
  print(f"Распределение классов: {Counter(y)}")

  # ===== ТЕСТ 1: Class Weights =====
  print("\n" + "=" * 80)
  print("ТЕСТ 1: CLASS WEIGHTS")
  print("=" * 80)

  class_weights = ClassWeightCalculator.compute_weights(y, method="balanced")
  weight_tensor = ClassWeightCalculator.weights_to_tensor(class_weights)

  loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
  print(f"Loss function создан с весами: {class_weights}")

  # ===== ТЕСТ 2: Focal Loss =====
  print("\n" + "=" * 80)
  print("ТЕСТ 2: FOCAL LOSS")
  print("=" * 80)

  focal_loss = FocalLoss(
    alpha=weight_tensor,
    gamma=2.0
  )

  # Тестовые предсказания
  logits = torch.randn(32, 3)  # batch=32, classes=3
  targets = torch.randint(0, 3, (32,))

  loss = focal_loss(logits, targets)
  print(f"Focal Loss на тестовом batch: {loss.item():.4f}")

  # ===== ТЕСТ 3: Oversampling =====
  print("\n" + "=" * 80)
  print("ТЕСТ 3: OVERSAMPLING")
  print("=" * 80)

  X_over, y_over = DatasetBalancer.oversample(X, y)

  # ===== ТЕСТ 4: SMOTE =====
  print("\n" + "=" * 80)
  print("ТЕСТ 4: SMOTE")
  print("=" * 80)

  # SMOTE требует 2D, используем упрощенный пример
  X_2d = X.reshape(len(X), -1)
  X_smote, y_smote = DatasetBalancer.smote(X_2d, y)

  # ===== ТЕСТ 5: Unified Strategy =====
  print("\n" + "=" * 80)
  print("ТЕСТ 5: UNIFIED STRATEGY")
  print("=" * 80)

  config = ClassBalancingConfig(
    use_class_weights=True,
    use_focal_loss=True,
    use_oversampling=True,
    focal_gamma=2.0
  )

  strategy = ClassBalancingStrategy(config)

  # Применяем resampling
  X_balanced, y_balanced = strategy.balance_dataset(X, y)

  # Печать результатов
  strategy.print_class_distribution(y, y_balanced)

  # Создаем loss function
  loss_fn = strategy.get_loss_function(y_balanced)
  print(f"Loss function: {loss_fn}")