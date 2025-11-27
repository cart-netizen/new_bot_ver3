#!/usr/bin/env python3
"""
Data Augmentation для временных рядов - Industry Standard.

Техники аугментации:
1. MixUp - смешивание samples
2. CutMix - замена части sequence
3. Time Masking - маскирование временных шагов
4. Feature Dropout - dropout отдельных features
5. Gaussian Noise - добавление шума
6. Time Warping - нелинейное искажение времени
7. Magnitude Warping - искажение амплитуды
8. Window Slicing - случайная обрезка окна
9. Permutation - перестановка сегментов

Все техники оптимизированы для финансовых временных рядов.

Путь: backend/ml_engine/training/augmentation.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List, Callable
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline

from backend.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AugmentationConfig:
    """Конфигурация аугментации данных."""
    
    # === MixUp ===
    use_mixup: bool = True
    mixup_alpha: float = 0.2  # Beta distribution parameter
    mixup_prob: float = 0.5  # Вероятность применения
    
    # === CutMix ===
    use_cutmix: bool = False  # Менее эффективен для time series
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.3
    
    # === Time Masking ===
    use_time_masking: bool = True
    time_mask_ratio: float = 0.1  # Доля маскируемых timesteps
    time_mask_value: float = 0.0  # Значение для маскированных
    time_mask_prob: float = 0.3
    
    # === Feature Dropout ===
    use_feature_dropout: bool = True
    feature_dropout_ratio: float = 0.05  # Доля обнуляемых features
    feature_dropout_prob: float = 0.3
    
    # === Gaussian Noise ===
    use_gaussian_noise: bool = True
    noise_std: float = 0.01  # Standard deviation
    noise_prob: float = 0.5
    
    # === Time Warping ===
    use_time_warping: bool = False  # Дорогой по вычислениям
    warp_sigma: float = 0.2
    warp_knots: int = 4
    warp_prob: float = 0.3
    
    # === Magnitude Warping ===
    use_magnitude_warping: bool = False
    magnitude_sigma: float = 0.2
    magnitude_prob: float = 0.3
    
    # === Window Slicing ===
    use_window_slicing: bool = False
    slice_ratio: float = 0.9  # Минимальный размер окна
    slice_prob: float = 0.3
    
    # === Permutation ===
    use_permutation: bool = False  # Нарушает временной порядок
    permutation_segments: int = 4
    permutation_prob: float = 0.2
    
    # === General ===
    apply_on_train_only: bool = True  # Только при обучении


# ============================================================================
# AUGMENTATION TRANSFORMS
# ============================================================================

class MixUp:
    """
    MixUp augmentation для временных рядов.
    
    MixUp создаёт виртуальные training examples через линейную интерполяцию:
        mixed_x = lambda * x_i + (1 - lambda) * x_j
        mixed_y = lambda * y_i + (1 - lambda) * y_j
    
    Преимущества:
    - Улучшает генерализацию
    - Снижает overconfidence
    - Эффективен при малом датасете
    
    Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter.
                   alpha=0: no mixing
                   alpha=0.2: moderate mixing (recommended)
                   alpha=1.0: uniform mixing
        """
        self.alpha = alpha
    
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to batch.
        
        Args:
            x: (batch, seq_len, features) - sequences
            y: (batch,) - labels
        
        Returns:
            mixed_x: (batch, seq_len, features)
            y_a: (batch,) - original labels
            y_b: (batch,) - shuffled labels
            lam: mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    @staticmethod
    def mixup_criterion(
        criterion: Callable,
        pred: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """
        Compute MixUp loss.
        
        Args:
            criterion: Loss function
            pred: Model predictions
            y_a, y_b: Original and shuffled labels
            lam: Mixing coefficient
        
        Returns:
            Mixed loss
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CutMix:
    """
    CutMix augmentation адаптированный для временных рядов.
    
    Вместо пространственной области (как в images), вырезаем
    временной сегмент из одного sample и вставляем в другой.
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to batch.
        
        Args:
            x: (batch, seq_len, features)
            y: (batch,)
        
        Returns:
            mixed_x, y_a, y_b, lam
        """
        batch_size, seq_len, features = x.shape
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random index for mixing
        index = torch.randperm(batch_size, device=x.device)
        
        # Compute cut length
        cut_len = int(seq_len * (1 - lam))
        
        # Random start position
        start = np.random.randint(0, seq_len - cut_len + 1)
        end = start + cut_len
        
        # Apply cut
        mixed_x = x.clone()
        mixed_x[:, start:end, :] = x[index, start:end, :]
        
        # Adjust lambda to actual proportion
        lam = 1 - cut_len / seq_len
        
        return mixed_x, y, y[index], lam


class TimeMasking:
    """
    Time Masking - маскирование случайных временных шагов.
    
    Похоже на SpecAugment для audio, но для временных рядов.
    Помогает модели не полагаться на конкретные временные точки.
    """
    
    def __init__(
        self,
        mask_ratio: float = 0.1,
        mask_value: float = 0.0,
        num_masks: int = 1
    ):
        """
        Args:
            mask_ratio: Доля маскируемых timesteps
            mask_value: Значение для маскированных позиций (0 или mean)
            num_masks: Количество масок
        """
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.num_masks = num_masks
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking.
        
        Args:
            x: (batch, seq_len, features)
        
        Returns:
            Masked tensor
        """
        batch_size, seq_len, features = x.shape
        
        # Total length to mask
        total_mask_len = int(seq_len * self.mask_ratio)
        mask_len_per = max(1, total_mask_len // self.num_masks)
        
        x_masked = x.clone()
        
        for _ in range(self.num_masks):
            for i in range(batch_size):
                start = torch.randint(0, max(1, seq_len - mask_len_per), (1,)).item()
                end = min(start + mask_len_per, seq_len)
                
                if self.mask_value == 0:
                    x_masked[i, start:end, :] = 0
                else:
                    # Use mean of the sample
                    x_masked[i, start:end, :] = x_masked[i].mean()
        
        return x_masked


class FeatureDropout:
    """
    Feature Dropout - обнуление случайных feature каналов.
    
    Помогает модели не полагаться на конкретные features.
    Особенно полезно при multimodal features (orderbook + candles + indicators).
    """
    
    def __init__(self, dropout_ratio: float = 0.05):
        """
        Args:
            dropout_ratio: Доля обнуляемых features
        """
        self.dropout_ratio = dropout_ratio
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feature dropout.
        
        Args:
            x: (batch, seq_len, features)
        
        Returns:
            Tensor с dropout features
        """
        # Create mask: (features,) → broadcast to (batch, seq_len, features)
        mask = torch.rand(x.size(-1), device=x.device) > self.dropout_ratio
        return x * mask.unsqueeze(0).unsqueeze(0)


class GaussianNoise:
    """
    Gaussian Noise - добавление нормального шума.
    
    Простейшая но эффективная аугментация для временных рядов.
    """
    
    def __init__(
        self,
        std: float = 0.01,
        relative: bool = False
    ):
        """
        Args:
            std: Standard deviation шума
            relative: Если True, std относительно амплитуды сигнала
        """
        self.std = std
        self.relative = relative
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise.
        
        Args:
            x: (batch, seq_len, features)
        
        Returns:
            Noisy tensor
        """
        if self.relative:
            # Relative to signal amplitude
            noise_std = self.std * x.abs().mean(dim=(1, 2), keepdim=True)
        else:
            noise_std = self.std
        
        noise = torch.randn_like(x) * noise_std
        return x + noise


class TimeWarping:
    """
    Time Warping - нелинейное искажение временной оси.
    
    Использует smooth random curves для warp временной оси.
    Помогает модели быть invariant к скорости событий.
    """
    
    def __init__(
        self,
        sigma: float = 0.2,
        knots: int = 4
    ):
        """
        Args:
            sigma: Sigma для random warp
            knots: Количество knots для spline
        """
        self.sigma = sigma
        self.knots = knots
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping.
        
        Args:
            x: (batch, seq_len, features)
        
        Returns:
            Time-warped tensor
        """
        batch_size, seq_len, features = x.shape
        device = x.device
        
        # Move to CPU for scipy
        x_np = x.cpu().numpy()
        
        warped = np.zeros_like(x_np)
        
        for i in range(batch_size):
            warped[i] = self._warp_single(x_np[i], seq_len)
        
        return torch.tensor(warped, dtype=x.dtype, device=device)
    
    def _warp_single(self, x: np.ndarray, seq_len: int) -> np.ndarray:
        """Warp single sample."""
        # Original time axis
        orig_steps = np.arange(seq_len)
        
        # Random knots
        knot_xs = np.linspace(0, seq_len - 1, self.knots + 2)
        knot_ys = knot_xs + np.random.normal(0, self.sigma, self.knots + 2) * (seq_len / self.knots)
        
        # Ensure monotonically increasing
        knot_ys = np.sort(knot_ys)
        knot_ys = np.clip(knot_ys, 0, seq_len - 1)
        
        # Spline interpolation
        spline = CubicSpline(knot_xs, knot_ys)
        warped_steps = spline(orig_steps)
        warped_steps = np.clip(warped_steps, 0, seq_len - 1)
        
        # Interpolate features
        warped = np.zeros_like(x)
        for j in range(x.shape[1]):
            warped[:, j] = np.interp(orig_steps, warped_steps, x[:, j])
        
        return warped


class MagnitudeWarping:
    """
    Magnitude Warping - искажение амплитуды.
    
    Умножает сигнал на smooth random curve.
    """
    
    def __init__(
        self,
        sigma: float = 0.2,
        knots: int = 4
    ):
        self.sigma = sigma
        self.knots = knots
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply magnitude warping.
        
        Args:
            x: (batch, seq_len, features)
        
        Returns:
            Magnitude-warped tensor
        """
        batch_size, seq_len, features = x.shape
        device = x.device
        
        # Generate smooth random curves
        orig_steps = np.linspace(0, 1, seq_len)
        knot_xs = np.linspace(0, 1, self.knots + 2)
        
        # Random scales for each batch
        warped = x.clone()
        
        for i in range(batch_size):
            knot_ys = 1.0 + np.random.normal(0, self.sigma, self.knots + 2)
            spline = CubicSpline(knot_xs, knot_ys)
            warp_curve = torch.tensor(
                spline(orig_steps),
                dtype=x.dtype,
                device=device
            ).unsqueeze(-1)  # (seq_len, 1)
            
            warped[i] = x[i] * warp_curve
        
        return warped


class WindowSlicing:
    """
    Window Slicing - случайная обрезка окна.
    
    Выбирает случайное подокно из sequence и масштабирует обратно.
    """
    
    def __init__(self, slice_ratio: float = 0.9):
        """
        Args:
            slice_ratio: Минимальный размер окна (0.9 = min 90% of original)
        """
        self.slice_ratio = slice_ratio
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply window slicing.
        
        Args:
            x: (batch, seq_len, features)
        
        Returns:
            Sliced and resized tensor
        """
        batch_size, seq_len, features = x.shape
        
        # Random slice length
        slice_len = int(seq_len * (self.slice_ratio + (1 - self.slice_ratio) * np.random.random()))
        
        sliced = []
        
        for i in range(batch_size):
            start = np.random.randint(0, seq_len - slice_len + 1)
            slice_x = x[i, start:start + slice_len, :]  # (slice_len, features)
            
            # Resize back to original length using interpolation
            slice_x = slice_x.unsqueeze(0).permute(0, 2, 1)  # (1, features, slice_len)
            resized = F.interpolate(slice_x, size=seq_len, mode='linear', align_corners=True)
            resized = resized.permute(0, 2, 1).squeeze(0)  # (seq_len, features)
            
            sliced.append(resized)
        
        return torch.stack(sliced)


class Permutation:
    """
    Permutation - случайная перестановка сегментов.
    
    ВНИМАНИЕ: Нарушает временной порядок! Использовать с осторожностью.
    """
    
    def __init__(self, num_segments: int = 4):
        self.num_segments = num_segments
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply permutation.
        
        Args:
            x: (batch, seq_len, features)
        
        Returns:
            Permuted tensor
        """
        batch_size, seq_len, features = x.shape
        segment_len = seq_len // self.num_segments
        
        permuted = []
        
        for i in range(batch_size):
            # Split into segments
            segments = [
                x[i, j * segment_len:(j + 1) * segment_len, :]
                for j in range(self.num_segments)
            ]
            
            # Handle remainder
            if seq_len % self.num_segments != 0:
                segments.append(x[i, self.num_segments * segment_len:, :])
            
            # Random permutation
            perm_idx = torch.randperm(len(segments))
            permuted_segments = [segments[idx] for idx in perm_idx]
            
            permuted.append(torch.cat(permuted_segments, dim=0))
        
        return torch.stack(permuted)


# ============================================================================
# AUGMENTATION PIPELINE
# ============================================================================

class AugmentationPipeline:
    """
    Pipeline для применения аугментаций к batch данных.
    
    Применяет различные аугментации с заданными вероятностями.
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Args:
            config: Конфигурация аугментации
        """
        self.config = config
        
        # Initialize transforms
        self.transforms: Dict[str, Tuple[Callable, float]] = {}
        
        if config.use_mixup:
            self.mixup = MixUp(alpha=config.mixup_alpha)
            self.transforms['mixup'] = (self.mixup, config.mixup_prob)
        
        if config.use_cutmix:
            self.cutmix = CutMix(alpha=config.cutmix_alpha)
            self.transforms['cutmix'] = (self.cutmix, config.cutmix_prob)
        
        if config.use_time_masking:
            self.time_mask = TimeMasking(
                mask_ratio=config.time_mask_ratio,
                mask_value=config.time_mask_value
            )
            self.transforms['time_mask'] = (self.time_mask, config.time_mask_prob)
        
        if config.use_feature_dropout:
            self.feature_dropout = FeatureDropout(
                dropout_ratio=config.feature_dropout_ratio
            )
            self.transforms['feature_dropout'] = (self.feature_dropout, config.feature_dropout_prob)
        
        if config.use_gaussian_noise:
            self.gaussian_noise = GaussianNoise(std=config.noise_std)
            self.transforms['gaussian_noise'] = (self.gaussian_noise, config.noise_prob)
        
        if config.use_time_warping:
            self.time_warp = TimeWarping(
                sigma=config.warp_sigma,
                knots=config.warp_knots
            )
            self.transforms['time_warp'] = (self.time_warp, config.warp_prob)
        
        if config.use_magnitude_warping:
            self.magnitude_warp = MagnitudeWarping(
                sigma=config.magnitude_sigma
            )
            self.transforms['magnitude_warp'] = (self.magnitude_warp, config.magnitude_prob)
        
        if config.use_window_slicing:
            self.window_slice = WindowSlicing(slice_ratio=config.slice_ratio)
            self.transforms['window_slice'] = (self.window_slice, config.slice_prob)
        
        if config.use_permutation:
            self.permutation = Permutation(num_segments=config.permutation_segments)
            self.transforms['permutation'] = (self.permutation, config.permutation_prob)
        
        logger.info(
            f"✓ AugmentationPipeline initialized with {len(self.transforms)} transforms: "
            f"{list(self.transforms.keys())}"
        )
    
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[float]]:
        """
        Apply augmentation pipeline.
        
        Args:
            x: (batch, seq_len, features)
            y: (batch,)
            training: Применять ли аугментации (False для inference)
        
        Returns:
            augmented_x: Augmented sequences
            y_a: Primary labels
            y_b: Secondary labels (for mixup) or None
            lam: Mixing coefficient or None
        """
        if not training or not self.transforms:
            return x, y, None, None
        
        y_b = None
        lam = None
        
        # MixUp/CutMix (взаимоисключающие)
        if self.config.use_mixup and np.random.random() < self.config.mixup_prob:
            x, y_a, y_b, lam = self.mixup(x, y)
            y = y_a  # Primary labels
        elif self.config.use_cutmix and np.random.random() < self.config.cutmix_prob:
            x, y_a, y_b, lam = self.cutmix(x, y)
            y = y_a
        
        # Apply other transforms sequentially
        for name, (transform, prob) in self.transforms.items():
            if name in ['mixup', 'cutmix']:
                continue  # Already applied
            
            if np.random.random() < prob:
                x = transform(x)
        
        return x, y, y_b, lam
    
    def apply_simple(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        Apply augmentations without label mixing.
        
        Args:
            x: (batch, seq_len, features)
            training: Apply augmentations
        
        Returns:
            Augmented tensor
        """
        if not training:
            return x
        
        # Apply transforms that don't need labels
        simple_transforms = ['time_mask', 'feature_dropout', 'gaussian_noise',
                           'time_warp', 'magnitude_warp', 'window_slice']
        
        for name in simple_transforms:
            if name in self.transforms:
                transform, prob = self.transforms[name]
                if np.random.random() < prob:
                    x = transform(x)
        
        return x


# ============================================================================
# DEFAULT CONFIGURATIONS
# ============================================================================

def get_default_augmentation_config() -> AugmentationConfig:
    """Получить дефолтную конфигурацию аугментации."""
    return AugmentationConfig(
        use_mixup=True,
        mixup_alpha=0.2,
        mixup_prob=0.5,
        use_time_masking=True,
        time_mask_ratio=0.1,
        time_mask_prob=0.3,
        use_feature_dropout=True,
        feature_dropout_ratio=0.05,
        feature_dropout_prob=0.3,
        use_gaussian_noise=True,
        noise_std=0.01,
        noise_prob=0.5
    )


def get_conservative_augmentation_config() -> AugmentationConfig:
    """Получить консервативную конфигурацию (меньше аугментаций)."""
    return AugmentationConfig(
        use_mixup=True,
        mixup_alpha=0.1,
        mixup_prob=0.3,
        use_time_masking=False,
        use_feature_dropout=False,
        use_gaussian_noise=True,
        noise_std=0.005,
        noise_prob=0.3
    )


def get_aggressive_augmentation_config() -> AugmentationConfig:
    """Получить агрессивную конфигурацию (больше аугментаций)."""
    return AugmentationConfig(
        use_mixup=True,
        mixup_alpha=0.4,
        mixup_prob=0.7,
        use_cutmix=True,
        cutmix_alpha=1.0,
        cutmix_prob=0.3,
        use_time_masking=True,
        time_mask_ratio=0.15,
        time_mask_prob=0.5,
        use_feature_dropout=True,
        feature_dropout_ratio=0.1,
        feature_dropout_prob=0.5,
        use_gaussian_noise=True,
        noise_std=0.02,
        noise_prob=0.7,
        use_magnitude_warping=True,
        magnitude_sigma=0.1,
        magnitude_prob=0.3
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ТЕСТ DATA AUGMENTATION")
    print("=" * 80)
    
    # Тестовые данные
    batch_size = 8
    seq_len = 60
    features = 110
    num_classes = 3
    
    x = torch.randn(batch_size, seq_len, features)
    y = torch.randint(0, num_classes, (batch_size,))
    
    print(f"\n📊 Input shapes:")
    print(f"  • x: {x.shape}")
    print(f"  • y: {y.shape}")
    
    # 1. MixUp
    print("\n1️⃣ MixUp:")
    mixup = MixUp(alpha=0.2)
    mixed_x, y_a, y_b, lam = mixup(x, y)
    print(f"  • mixed_x: {mixed_x.shape}")
    print(f"  • lambda: {lam:.4f}")
    
    # 2. Time Masking
    print("\n2️⃣ Time Masking:")
    time_mask = TimeMasking(mask_ratio=0.1)
    masked_x = time_mask(x)
    print(f"  • masked_x: {masked_x.shape}")
    print(f"  • Zeros count: {(masked_x == 0).sum().item()}")
    
    # 3. Gaussian Noise
    print("\n3️⃣ Gaussian Noise:")
    noise = GaussianNoise(std=0.01)
    noisy_x = noise(x)
    diff = (noisy_x - x).abs().mean().item()
    print(f"  • noisy_x: {noisy_x.shape}")
    print(f"  • Mean diff: {diff:.6f}")
    
    # 4. Feature Dropout
    print("\n4️⃣ Feature Dropout:")
    feat_drop = FeatureDropout(dropout_ratio=0.1)
    dropped_x = feat_drop(x)
    zero_features = (dropped_x.sum(dim=(0, 1)) == 0).sum().item()
    print(f"  • dropped_x: {dropped_x.shape}")
    print(f"  • Zero features: {zero_features}")
    
    # 5. Full Pipeline
    print("\n5️⃣ Full AugmentationPipeline:")
    config = get_default_augmentation_config()
    pipeline = AugmentationPipeline(config)
    
    aug_x, y_a, y_b, lam = pipeline(x, y, training=True)
    print(f"  • aug_x: {aug_x.shape}")
    print(f"  • y_a: {y_a.shape}")
    print(f"  • y_b: {y_b.shape if y_b is not None else 'None'}")
    print(f"  • lam: {lam if lam is not None else 'None'}")
    
    # 6. Simple augmentation (no labels)
    print("\n6️⃣ Simple augmentation:")
    simple_aug_x = pipeline.apply_simple(x, training=True)
    print(f"  • simple_aug_x: {simple_aug_x.shape}")
    
    print("\n" + "=" * 80)
    print("✅ Все тесты пройдены!")
    print("=" * 80)
