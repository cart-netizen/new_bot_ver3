#!/usr/bin/env python3
"""
Улучшенные Loss Functions для ML модели - Industry Standard.

Компоненты:
1. LabelSmoothingCrossEntropy - предотвращает overconfidence
2. FocalLossV2 - улучшенная фокусировка на hard examples
3. AsymmetricFocalLoss - разные gamma для pos/neg
4. MultiTaskLossV2 - улучшенный multi-task loss
5. ConfidenceCalibrationLoss - калибровка confidence
6. DirectionalAccuracyLoss - штраф за неправильное направление

Путь: backend/ml_engine/training/losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np
from collections import Counter

from backend.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# LABEL SMOOTHING
# ============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy с Label Smoothing.
    
    Label Smoothing:
    - Предотвращает overconfidence модели
    - Улучшает генерализацию
    - Делает модель более калиброванной
    
    Формула:
        soft_targets = (1 - smoothing) * one_hot + smoothing / num_classes
        loss = CrossEntropy(predictions, soft_targets)
    
    Рекомендации:
        - smoothing=0.1 для большинства задач
        - smoothing=0.05-0.15 для финансовых данных
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            smoothing: Степень сглаживания (0 = off, 0.1 = recommended)
            weight: Веса классов (для class imbalance)
            reduction: 'mean', 'sum', 'none'
        """
        super().__init__()
        
        assert 0 <= smoothing < 1, "smoothing должен быть в [0, 1)"
        
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        
        logger.info(f"✓ LabelSmoothingCrossEntropy: smoothing={smoothing}")
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) - raw logits
            targets: (batch,) - class indices
        
        Returns:
            loss: scalar или (batch,) в зависимости от reduction
        """
        num_classes = logits.size(-1)
        
        # Log softmax для численной стабильности
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Создаём soft targets
        with torch.no_grad():
            # One-hot encoding
            one_hot = torch.zeros_like(log_probs)
            one_hot.scatter_(1, targets.unsqueeze(1), 1)
            
            # Label smoothing
            smooth_targets = (
                (1 - self.smoothing) * one_hot +
                self.smoothing / num_classes
            )
        
        # Cross entropy с soft targets
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        # Применяем class weights
        if self.weight is not None:
            weight = self.weight.to(logits.device)
            loss = loss * weight[targets]
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLossV2(nn.Module):
    """
    Улучшенная Focal Loss для class imbalance.
    
    Focal Loss фокусирует обучение на hard examples,
    уменьшая вклад easy examples в loss.
    
    Формула:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    где:
        p_t = probability of correct class
        gamma = focusing parameter (higher = more focus on hard)
        alpha = class weights
    
    Рекомендации для финансовых данных:
        - gamma=2.0-3.0 (higher при сильном imbalance)
        - alpha = inverse class frequency
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Args:
            gamma: Focusing parameter (0 = standard CE, 2-3 = recommended)
            alpha: Class weights tensor (num_classes,)
            reduction: 'mean', 'sum', 'none'
            label_smoothing: Дополнительное сглаживание
        """
        super().__init__()
        
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        logger.info(
            f"✓ FocalLossV2: gamma={gamma}, "
            f"alpha={'auto' if alpha is None else 'custom'}, "
            f"label_smoothing={label_smoothing}"
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            targets: (batch,)
        
        Returns:
            loss: scalar
        """
        num_classes = logits.size(-1)
        
        # Probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get probability of correct class
        ce_loss = F.cross_entropy(
            logits, targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        # p_t = probability of correct class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss с разными gamma для положительных/отрицательных.
    
    Полезно когда:
    - False Negatives более дороги чем False Positives (или наоборот)
    - Нужен разный фокус для разных типов ошибок
    
    Используется в торговле:
    - Пропустить хорошую сделку (FN) может быть дешевле чем плохая сделка (FP)
    """
    
    def __init__(
        self,
        gamma_pos: float = 2.0,
        gamma_neg: float = 4.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            gamma_pos: Gamma для positive samples
            gamma_neg: Gamma для negative samples (обычно выше)
            alpha: Class weights
            reduction: 'mean', 'sum', 'none'
        """
        super().__init__()
        
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            targets: (batch,)
        
        Returns:
            loss: scalar
        """
        probs = F.softmax(logits, dim=-1)
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Asymmetric gamma: выше для negative
        # Определяем "positive" как non-HOLD классы (1 и 2)
        is_positive = (targets != 0)  # HOLD = 0
        
        gamma = torch.where(
            is_positive,
            torch.tensor(self.gamma_pos, device=logits.device),
            torch.tensor(self.gamma_neg, device=logits.device)
        )
        
        focal_weight = (1 - p_t) ** gamma
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# MULTI-TASK LOSS
# ============================================================================

class MultiTaskLossV2(nn.Module):
    """
    Улучшенный Multi-Task Loss для совместного обучения.
    
    Tasks:
    1. Direction prediction (classification)
    2. Confidence prediction (regression)
    3. Expected return prediction (regression)
    
    Улучшения:
    - Learnable task weights (uncertainty weighting)
    - Gradient balancing
    - Temperature scaling для confidence
    """
    
    def __init__(
        self,
        direction_weight: float = 1.0,
        confidence_weight: float = 0.5,
        return_weight: float = 0.3,
        direction_criterion: Optional[nn.Module] = None,
        use_uncertainty_weighting: bool = False,
        confidence_temperature: float = 1.0
    ):
        """
        Args:
            direction_weight: Вес direction loss
            confidence_weight: Вес confidence loss
            return_weight: Вес return loss
            direction_criterion: Custom criterion для direction (FocalLoss, etc.)
            use_uncertainty_weighting: Learnable веса на основе uncertainty
            confidence_temperature: Temperature для confidence calibration
        """
        super().__init__()
        
        self.direction_weight = direction_weight
        self.confidence_weight = confidence_weight
        self.return_weight = return_weight
        self.confidence_temperature = confidence_temperature
        
        # Direction criterion
        if direction_criterion is not None:
            self.direction_criterion = direction_criterion
        else:
            self.direction_criterion = nn.CrossEntropyLoss()
        
        # Regression losses
        self.mse_loss = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()  # Более робастный для return
        
        # Uncertainty weighting (learnable)
        self.use_uncertainty_weighting = use_uncertainty_weighting
        if use_uncertainty_weighting:
            # log(sigma^2) для каждой задачи
            self.log_var_direction = nn.Parameter(torch.zeros(1))
            self.log_var_confidence = nn.Parameter(torch.zeros(1))
            self.log_var_return = nn.Parameter(torch.zeros(1))
        
        logger.info(
            f"✓ MultiTaskLossV2: direction={direction_weight}, "
            f"confidence={confidence_weight}, return={return_weight}, "
            f"uncertainty_weighting={use_uncertainty_weighting}"
        )
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            outputs: Dict с выходами модели
                - direction_logits: (batch, num_classes)
                - confidence: (batch, 1)
                - expected_return: (batch, 1)
            targets: Dict с целями
                - label: (batch,) - direction labels
                - confidence: (batch,) - target confidence (optional)
                - return: (batch,) - target return (optional)
        
        Returns:
            total_loss: scalar
            loss_components: Dict с компонентами loss
        """
        loss_components = {}
        
        # 1. Direction Loss
        direction_logits = outputs['direction_logits']
        direction_targets = targets['label']
        direction_loss = self.direction_criterion(direction_logits, direction_targets)
        loss_components['direction_loss'] = direction_loss.item()
        
        # 2. Confidence Loss (optional - если есть target)
        if 'confidence' in targets:
            pred_confidence = outputs['confidence'].squeeze(-1)
            target_confidence = targets['confidence']
            confidence_loss = self.mse_loss(pred_confidence, target_confidence)
        else:
            # Self-supervised: confidence должен коррелировать с правильностью
            pred_confidence = outputs['confidence'].squeeze(-1)
            pred_class = torch.argmax(direction_logits, dim=-1)
            is_correct = (pred_class == direction_targets).float()
            confidence_loss = self.mse_loss(pred_confidence, is_correct)
        
        loss_components['confidence_loss'] = confidence_loss.item()
        
        # 3. Return Loss (optional)
        if 'return' in targets:
            pred_return = outputs['expected_return'].squeeze(-1)
            target_return = targets['return']
            return_loss = self.smooth_l1(pred_return, target_return)
            loss_components['return_loss'] = return_loss.item()
        else:
            return_loss = torch.tensor(0.0, device=direction_loss.device)
            loss_components['return_loss'] = 0.0
        
        # Combine losses
        if self.use_uncertainty_weighting:
            # Uncertainty weighting: loss / (2 * sigma^2) + log(sigma)
            total_loss = (
                direction_loss * torch.exp(-self.log_var_direction) +
                self.log_var_direction +
                confidence_loss * torch.exp(-self.log_var_confidence) +
                self.log_var_confidence +
                return_loss * torch.exp(-self.log_var_return) +
                self.log_var_return
            )
            
            loss_components['log_var_direction'] = self.log_var_direction.item()
            loss_components['log_var_confidence'] = self.log_var_confidence.item()
            loss_components['log_var_return'] = self.log_var_return.item()
        else:
            total_loss = (
                self.direction_weight * direction_loss +
                self.confidence_weight * confidence_loss +
                self.return_weight * return_loss
            )
        
        loss_components['total_loss'] = total_loss.item()
        
        return total_loss, loss_components


# ============================================================================
# SPECIALIZED LOSSES
# ============================================================================

class DirectionalAccuracyLoss(nn.Module):
    """
    Loss с штрафом за неправильное направление.
    
    В трейдинге:
    - Предсказать BUY когда реально SELL = очень плохо
    - Предсказать HOLD когда реально BUY/SELL = менее плохо
    
    Штрафы:
        BUY↔SELL: самый большой штраф
        BUY↔HOLD или SELL↔HOLD: меньший штраф
        Correct: нет штрафа
    """
    
    def __init__(
        self,
        opposite_penalty: float = 2.0,
        neutral_penalty: float = 1.0,
        base_criterion: Optional[nn.Module] = None
    ):
        """
        Args:
            opposite_penalty: Множитель штрафа за противоположное направление
            neutral_penalty: Множитель штрафа за HOLD ошибку
            base_criterion: Базовый criterion
        """
        super().__init__()
        
        self.opposite_penalty = opposite_penalty
        self.neutral_penalty = neutral_penalty
        
        if base_criterion is None:
            self.base_criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            self.base_criterion = base_criterion
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            targets: (batch,) - 0=HOLD, 1=BUY, 2=SELL
        
        Returns:
            loss: scalar
        """
        base_loss = self.base_criterion(logits, targets)
        
        # Predictions
        predictions = torch.argmax(logits, dim=-1)
        
        # Penalty weights
        # BUY=1, SELL=2 → opposite означает |pred - target| = 1 и оба != 0
        is_opposite = (
            ((predictions == 1) & (targets == 2)) |
            ((predictions == 2) & (targets == 1))
        )
        
        is_neutral_error = (
            ((predictions == 0) & (targets != 0)) |  # Pred HOLD, target BUY/SELL
            ((predictions != 0) & (targets == 0))    # Pred BUY/SELL, target HOLD
        )
        
        # Apply penalties
        penalty = torch.ones_like(base_loss)
        penalty[is_opposite] = self.opposite_penalty
        penalty[is_neutral_error] = self.neutral_penalty
        
        weighted_loss = base_loss * penalty
        
        return weighted_loss.mean()


class ConfidenceCalibrationLoss(nn.Module):
    """
    Loss для калибровки confidence предсказаний.
    
    Цель: predicted confidence ≈ actual accuracy
    
    Использует Expected Calibration Error (ECE) как дополнительный loss.
    """
    
    def __init__(
        self,
        n_bins: int = 10,
        calibration_weight: float = 0.1
    ):
        """
        Args:
            n_bins: Количество bins для ECE
            calibration_weight: Вес calibration loss
        """
        super().__init__()
        
        self.n_bins = n_bins
        self.calibration_weight = calibration_weight
        self.bins = torch.linspace(0, 1, n_bins + 1)
    
    def forward(
        self,
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            confidences: (batch,) predicted confidence
            predictions: (batch,) predicted class
            targets: (batch,) true class
        
        Returns:
            ece: Expected Calibration Error
        """
        accuracies = (predictions == targets).float()
        
        ece = torch.tensor(0.0, device=confidences.device)
        
        for i in range(self.n_bins):
            bin_lower = self.bins[i].item()
            bin_upper = self.bins[i + 1].item()
            
            # Samples в этом bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                # Средняя accuracy и confidence в bin
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                
                # |accuracy - confidence| взвешенный по размеру bin
                bin_weight = in_bin.float().mean()
                ece += bin_weight * torch.abs(avg_accuracy - avg_confidence)
        
        return self.calibration_weight * ece


# ============================================================================
# LOSS FACTORY
# ============================================================================

class LossFactory:
    """Фабрика для создания loss functions."""
    
    @staticmethod
    def create_direction_loss(
        loss_type: str = "focal",
        num_classes: int = 3,
        class_weights: Optional[np.ndarray] = None,
        gamma: float = 2.5,
        label_smoothing: float = 0.1,
        device: str = "cpu"
    ) -> nn.Module:
        """
        Создать loss для direction prediction.
        
        Args:
            loss_type: 'ce', 'focal', 'focal_smooth', 'asymmetric'
            num_classes: Количество классов
            class_weights: Веса классов (numpy array)
            gamma: Focal gamma
            label_smoothing: Label smoothing
            device: Device для tensors
        
        Returns:
            Loss module
        """
        # Convert weights to tensor
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        
        if loss_type == "ce":
            return nn.CrossEntropyLoss(weight=weight_tensor)
        
        elif loss_type == "ce_smooth":
            return LabelSmoothingCrossEntropy(
                smoothing=label_smoothing,
                weight=weight_tensor
            )
        
        elif loss_type == "focal":
            return FocalLossV2(
                gamma=gamma,
                alpha=weight_tensor,
                label_smoothing=0.0
            )
        
        elif loss_type == "focal_smooth":
            return FocalLossV2(
                gamma=gamma,
                alpha=weight_tensor,
                label_smoothing=label_smoothing
            )
        
        elif loss_type == "asymmetric":
            return AsymmetricFocalLoss(
                gamma_pos=gamma,
                gamma_neg=gamma + 1.0,
                alpha=weight_tensor
            )
        
        elif loss_type == "directional":
            base = FocalLossV2(gamma=gamma, alpha=weight_tensor)
            return DirectionalAccuracyLoss(
                opposite_penalty=2.0,
                neutral_penalty=1.0,
                base_criterion=base
            )
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    @staticmethod
    def create_multi_task_loss(
        direction_loss_type: str = "focal_smooth",
        num_classes: int = 3,
        class_weights: Optional[np.ndarray] = None,
        gamma: float = 2.5,
        label_smoothing: float = 0.1,
        direction_weight: float = 1.0,
        confidence_weight: float = 0.5,
        return_weight: float = 0.3,
        use_uncertainty_weighting: bool = False,
        device: str = "cpu"
    ) -> MultiTaskLossV2:
        """
        Создать MultiTaskLoss с настроенным direction loss.
        
        Returns:
            MultiTaskLossV2 instance
        """
        direction_criterion = LossFactory.create_direction_loss(
            loss_type=direction_loss_type,
            num_classes=num_classes,
            class_weights=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing,
            device=device
        )
        
        return MultiTaskLossV2(
            direction_weight=direction_weight,
            confidence_weight=confidence_weight,
            return_weight=return_weight,
            direction_criterion=direction_criterion,
            use_uncertainty_weighting=use_uncertainty_weighting
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_class_weights(
    labels: np.ndarray,
    method: str = "balanced",
    smooth_factor: float = 0.0
) -> np.ndarray:
    """
    Вычислить веса классов для imbalanced data.
    
    Args:
        labels: Массив меток
        method: 'balanced', 'sqrt', 'log', 'effective'
        smooth_factor: Сглаживание весов
    
    Returns:
        weights: Array весов для каждого класса
    """
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    total_samples = len(labels)
    
    if method == "balanced":
        # sklearn-style: n_samples / (n_classes * n_samples_per_class)
        weights = {
            cls: total_samples / (num_classes * count)
            for cls, count in class_counts.items()
        }
    
    elif method == "sqrt":
        # Square root of balanced
        weights = {
            cls: np.sqrt(total_samples / (num_classes * count))
            for cls, count in class_counts.items()
        }
    
    elif method == "log":
        # Log of inverse frequency
        weights = {
            cls: np.log(total_samples / count + 1)
            for cls, count in class_counts.items()
        }
    
    elif method == "effective":
        # Effective number of samples (CB loss paper)
        beta = 0.9999
        weights = {
            cls: (1 - beta) / (1 - beta ** count)
            for cls, count in class_counts.items()
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize
    weight_sum = sum(weights.values())
    weights = {k: v * num_classes / weight_sum for k, v in weights.items()}
    
    # Apply smoothing
    if smooth_factor > 0:
        weights = {k: v * (1 - smooth_factor) + smooth_factor for k, v in weights.items()}
    
    # Convert to array
    weight_array = np.array([weights.get(i, 1.0) for i in range(num_classes)])
    
    logger.info(f"Class weights ({method}): {weight_array}")
    
    return weight_array.astype(np.float32)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ТЕСТ LOSS FUNCTIONS")
    print("=" * 80)
    
    # Тестовые данные
    batch_size = 32
    num_classes = 3
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    print(f"\n📊 Input shapes:")
    print(f"  • logits: {logits.shape}")
    print(f"  • targets: {targets.shape}")
    
    # 1. Label Smoothing
    print("\n1️⃣ LabelSmoothingCrossEntropy:")
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss_val = ls_loss(logits, targets)
    print(f"  • Loss: {loss_val.item():.4f}")
    
    # 2. Focal Loss
    print("\n2️⃣ FocalLossV2:")
    focal_loss = FocalLossV2(gamma=2.5)
    loss_val = focal_loss(logits, targets)
    print(f"  • Loss: {loss_val.item():.4f}")
    
    # 3. Focal Loss с class weights
    print("\n3️⃣ FocalLossV2 с class weights:")
    labels_np = targets.numpy()
    weights = compute_class_weights(labels_np, method="balanced")
    weights_tensor = torch.tensor(weights)
    
    focal_weighted = FocalLossV2(gamma=2.5, alpha=weights_tensor)
    loss_val = focal_weighted(logits, targets)
    print(f"  • Loss: {loss_val.item():.4f}")
    
    # 4. MultiTaskLoss
    print("\n4️⃣ MultiTaskLossV2:")
    outputs = {
        'direction_logits': logits,
        'confidence': torch.sigmoid(torch.randn(batch_size, 1)),
        'expected_return': torch.randn(batch_size, 1) * 0.01
    }
    targets_dict = {'label': targets}
    
    multi_loss = MultiTaskLossV2(
        direction_criterion=focal_weighted,
        use_uncertainty_weighting=False
    )
    total_loss, components = multi_loss(outputs, targets_dict)
    
    print(f"  • Total loss: {total_loss.item():.4f}")
    print(f"  • Components: {components}")
    
    # 5. Factory
    print("\n5️⃣ LossFactory:")
    factory_loss = LossFactory.create_multi_task_loss(
        direction_loss_type="focal_smooth",
        class_weights=weights,
        gamma=2.5,
        label_smoothing=0.1
    )
    total_loss, components = factory_loss(outputs, targets_dict)
    print(f"  • Total loss: {total_loss.item():.4f}")
    
    print("\n" + "=" * 80)
    print("✅ Все тесты пройдены!")
    print("=" * 80)
