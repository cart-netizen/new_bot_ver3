#!/usr/bin/env python3
"""
Улучшенная стратегия балансировки классов v2 - Industry Standard.

Компоненты:
1. Adaptive Threshold Labeling - динамические пороги на основе percentiles
2. Class Weights Calculator - методы balanced, sqrt, log, effective
3. Focal Loss integration - готовые alpha параметры
4. Oversampling/Undersampling - с учётом временных рядов
5. SMOTE для временных рядов - адаптированная версия

Путь: backend/ml_engine/training/class_balancing_v2.py
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from collections import Counter
from enum import Enum
import warnings
from tqdm import tqdm

from backend.core.logger import get_logger

logger = get_logger(__name__)


def _log(msg: str):
    """Логирование через tqdm.write() для немедленного вывода в консоль."""
    tqdm.write(f"[Balancing] {msg}")


# ============================================================================
# ENUMS И CONFIGURATION
# ============================================================================

class BalancingMethod(str, Enum):
    """Методы балансировки."""
    NONE = "none"
    CLASS_WEIGHTS = "class_weights"
    OVERSAMPLING = "oversampling"
    UNDERSAMPLING = "undersampling"
    SMOTE = "smote"
    COMBINED = "combined"  # Oversampling + Class Weights


class WeightMethod(str, Enum):
    """Методы вычисления весов."""
    BALANCED = "balanced"  # sklearn style
    SQRT = "sqrt"  # Square root of balanced
    LOG = "log"  # Log of inverse frequency
    EFFECTIVE = "effective"  # Class-balanced loss paper


class ThresholdMethod(str, Enum):
    """Методы определения порогов для labeling."""
    FIXED = "fixed"
    PERCENTILE = "percentile"
    ADAPTIVE = "adaptive"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


@dataclass
class ClassBalancingConfigV2:
    """
    Конфигурация балансировки классов v2.
    
    Оптимизированные параметры для финансовых данных.
    """
    
    # === Основной метод ===
    method: BalancingMethod = BalancingMethod.COMBINED
    
    # === Class Weights ===
    use_class_weights: bool = True
    weight_method: WeightMethod = WeightMethod.BALANCED
    weight_smoothing: float = 0.0  # Сглаживание весов
    
    # === Focal Loss ===
    use_focal_loss: bool = True
    focal_gamma: float = 2.5
    
    # === Oversampling ===
    use_oversampling: bool = True
    oversample_ratio: float = 0.5  # Minority до 50% от majority
    oversample_with_noise: bool = True  # Добавлять шум при дублировании
    noise_std: float = 0.01
    
    # === Undersampling ===
    use_undersampling: bool = False
    undersample_ratio: float = 0.8
    
    # === SMOTE ===
    use_smote: bool = False
    smote_k_neighbors: int = 5
    
    # === Threshold для Labeling ===
    threshold_method: ThresholdMethod = ThresholdMethod.PERCENTILE
    
    # Фиксированные пороги
    fixed_threshold_sell: float = -0.001  # -0.1%
    fixed_threshold_buy: float = 0.001   # +0.1%
    
    # Percentile пороги
    percentile_sell: float = 0.25  # Bottom 25% = Sell
    percentile_buy: float = 0.75   # Top 25% = Buy
    
    # Adaptive пороги
    min_class_ratio: float = 0.15  # Минимум 15% для каждого класса
    
    # === Целевое распределение ===
    target_distribution: Tuple[float, float, float] = (0.30, 0.40, 0.30)  # Sell, Hold, Buy


# ============================================================================
# THRESHOLD OPTIMIZER
# ============================================================================

class ThresholdOptimizer:
    """
    Оптимизатор порогов для labeling.
    
    Цель: найти пороги, которые дают сбалансированное распределение классов.
    """
    
    def __init__(self, config: ClassBalancingConfigV2):
        self.config = config
    
    def compute_thresholds(
        self,
        returns: np.ndarray,
        method: Optional[ThresholdMethod] = None
    ) -> Tuple[float, float]:
        """
        Вычислить оптимальные пороги.
        
        Args:
            returns: Массив доходностей
            method: Метод (если None - из config)
        
        Returns:
            (sell_threshold, buy_threshold)
        """
        method = method or self.config.threshold_method
        
        if method == ThresholdMethod.FIXED:
            return self.config.fixed_threshold_sell, self.config.fixed_threshold_buy
        
        elif method == ThresholdMethod.PERCENTILE:
            return self._percentile_thresholds(returns)
        
        elif method == ThresholdMethod.ADAPTIVE:
            return self._adaptive_thresholds(returns)
        
        elif method == ThresholdMethod.VOLATILITY_ADJUSTED:
            return self._volatility_adjusted_thresholds(returns)
        
        else:
            return self.config.fixed_threshold_sell, self.config.fixed_threshold_buy
    
    def _percentile_thresholds(
        self,
        returns: np.ndarray
    ) -> Tuple[float, float]:
        """Пороги на основе percentiles."""
        sell_threshold = np.percentile(returns, self.config.percentile_sell * 100)
        buy_threshold = np.percentile(returns, self.config.percentile_buy * 100)
        
        _log(
            f"Percentile thresholds: sell={sell_threshold:.6f} "
            f"({self.config.percentile_sell*100}%), "
            f"buy={buy_threshold:.6f} ({self.config.percentile_buy*100}%)"
        )
        
        return sell_threshold, buy_threshold
    
    def _adaptive_thresholds(
        self,
        returns: np.ndarray
    ) -> Tuple[float, float]:
        """
        Адаптивные пороги для достижения целевого распределения.
        
        Итеративно подбирает пороги чтобы достичь заданного распределения классов.
        """
        target_sell, target_hold, target_buy = self.config.target_distribution
        
        # Начинаем с медианы
        sorted_returns = np.sort(returns)
        n = len(sorted_returns)
        
        # Индексы для целевого распределения
        sell_idx = int(n * target_sell)
        buy_idx = int(n * (target_sell + target_hold))
        
        sell_threshold = sorted_returns[sell_idx]
        buy_threshold = sorted_returns[buy_idx]
        
        # Проверяем распределение
        labels = self._apply_thresholds(returns, sell_threshold, buy_threshold)
        dist = Counter(labels)
        
        _log(
            f"Adaptive thresholds: sell={sell_threshold:.6f}, buy={buy_threshold:.6f}"
        )
        _log(
            f"Resulting distribution: {dict(dist)} "
            f"(target: sell={target_sell:.0%}, hold={target_hold:.0%}, buy={target_buy:.0%})"
        )
        
        return sell_threshold, buy_threshold
    
    def _volatility_adjusted_thresholds(
        self,
        returns: np.ndarray,
        lookback: int = 100
    ) -> Tuple[float, float]:
        """
        Пороги скорректированные на волатильность.
        
        Используем rolling std для адаптации к текущей волатильности.
        """
        # Rolling volatility
        if len(returns) > lookback:
            recent_std = np.std(returns[-lookback:])
        else:
            recent_std = np.std(returns)
        
        # Пороги = k * sigma
        k = 0.5  # Множитель стандартного отклонения
        sell_threshold = -k * recent_std
        buy_threshold = k * recent_std
        
        _log(
            f"Volatility-adjusted thresholds: sell={sell_threshold:.6f}, "
            f"buy={buy_threshold:.6f} (volatility={recent_std:.6f})"
        )
        
        return sell_threshold, buy_threshold
    
    def _apply_thresholds(
        self,
        returns: np.ndarray,
        sell_threshold: float,
        buy_threshold: float
    ) -> np.ndarray:
        """Применить пороги к доходностям."""
        labels = np.ones(len(returns), dtype=np.int64)  # Default: HOLD = 1

        # СТАНДАРТНЫЙ МАППИНГ: SELL=0, HOLD=1, BUY=2
        labels[returns < sell_threshold] = 0  # SELL
        labels[returns > buy_threshold] = 2   # BUY
        labels[(returns >= sell_threshold) & (returns <= buy_threshold)] = 1  # HOLD

        return labels
    
    def relabel_data(
        self,
        returns: np.ndarray,
        method: Optional[ThresholdMethod] = None
    ) -> np.ndarray:
        """
        Перемаркировать данные с оптимальными порогами.
        
        Args:
            returns: Массив доходностей
            method: Метод определения порогов
        
        Returns:
            labels: Массив меток (0=SELL, 1=HOLD, 2=BUY)
        """
        sell_threshold, buy_threshold = self.compute_thresholds(returns, method)
        labels = self._apply_thresholds(returns, sell_threshold, buy_threshold)
        
        dist = Counter(labels)
        _log(f"Relabeling result: {dict(dist)}")
        
        return labels


# ============================================================================
# CLASS WEIGHTS CALCULATOR
# ============================================================================

class ClassWeightsCalculator:
    """Калькулятор весов классов."""
    
    @staticmethod
    def compute(
        labels: np.ndarray,
        method: WeightMethod = WeightMethod.BALANCED,
        smoothing: float = 0.0,
        num_classes: int = 3
    ) -> np.ndarray:
        """
        Вычислить веса классов.
        
        Args:
            labels: Массив меток
            method: Метод вычисления
            smoothing: Сглаживание (0 = off)
            num_classes: Количество классов
        
        Returns:
            weights: Array весов [w0, w1, w2, ...]
        """
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        # Убедимся что все классы представлены
        for i in range(num_classes):
            if i not in class_counts:
                class_counts[i] = 1  # Избегаем деления на ноль
        
        if method == WeightMethod.BALANCED:
            # sklearn style: n_samples / (n_classes * n_samples_per_class)
            weights = {
                cls: total_samples / (num_classes * count)
                for cls, count in class_counts.items()
            }
        
        elif method == WeightMethod.SQRT:
            # Square root of balanced (более мягкие веса)
            weights = {
                cls: np.sqrt(total_samples / (num_classes * count))
                for cls, count in class_counts.items()
            }
        
        elif method == WeightMethod.LOG:
            # Log of inverse frequency
            weights = {
                cls: np.log(total_samples / count + 1)
                for cls, count in class_counts.items()
            }
        
        elif method == WeightMethod.EFFECTIVE:
            # Effective number of samples (Class-Balanced Loss paper)
            beta = 0.9999
            weights = {
                cls: (1 - beta) / (1 - beta ** count)
                for cls, count in class_counts.items()
            }
        
        else:
            # Default: uniform
            weights = {cls: 1.0 for cls in range(num_classes)}
        
        # Normalize so that sum = num_classes
        weight_sum = sum(weights.values())
        weights = {k: v * num_classes / weight_sum for k, v in weights.items()}
        
        # Apply smoothing: weight = weight * (1 - smoothing) + smoothing
        if smoothing > 0:
            weights = {
                k: v * (1 - smoothing) + smoothing
                for k, v in weights.items()
            }
        
        # Convert to array
        weight_array = np.array(
            [weights.get(i, 1.0) for i in range(num_classes)],
            dtype=np.float32
        )
        
        _log(f"Class weights ({method.value}): {weight_array}")
        
        return weight_array
    
    @staticmethod
    def compute_focal_alpha(
        labels: np.ndarray,
        num_classes: int = 3
    ) -> np.ndarray:
        """
        Вычислить alpha для Focal Loss.
        
        Alpha обычно = inverse class frequency, normalized.
        """
        class_counts = Counter(labels)
        total = len(labels)
        
        alpha = np.zeros(num_classes, dtype=np.float32)
        
        for cls in range(num_classes):
            count = class_counts.get(cls, 1)
            alpha[cls] = 1.0 - (count / total)
        
        # Normalize
        alpha = alpha / alpha.sum() * num_classes
        
        _log(f"Focal alpha: {alpha}")
        
        return alpha


# ============================================================================
# RESAMPLING STRATEGIES
# ============================================================================

class ResamplingStrategy:
    """Стратегии ресемплинга для балансировки классов."""
    
    def __init__(self, config: ClassBalancingConfigV2):
        self.config = config
    
    def oversample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_ratio: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oversampling minority классов.
        
        Args:
            X: Features (N, feature_dim) или sequences (N, seq_len, feature_dim)
            y: Labels (N,)
            target_ratio: Целевое соотношение minority/majority
        
        Returns:
            X_resampled, y_resampled
        """
        target_ratio = target_ratio or self.config.oversample_ratio
        
        class_counts = Counter(y)
        majority_count = max(class_counts.values())
        target_count = int(majority_count * target_ratio)
        
        _log(f"Oversampling: majority={majority_count}, target_minority={target_count}")
        
        X_resampled = [X]
        y_resampled = [y]
        
        for cls, count in class_counts.items():
            if count < target_count:
                # Находим индексы этого класса
                indices = np.where(y == cls)[0]
                
                # Сколько нужно добавить
                n_to_add = target_count - count
                
                # Случайно выбираем индексы для дублирования
                sample_indices = np.random.choice(indices, size=n_to_add, replace=True)
                
                # Добавляем с опциональным шумом
                X_new = X[sample_indices].copy()
                
                if self.config.oversample_with_noise:
                    noise = np.random.randn(*X_new.shape) * self.config.noise_std
                    X_new = X_new + noise
                
                X_resampled.append(X_new)
                y_resampled.append(np.full(n_to_add, cls, dtype=y.dtype))
                
                _log(f"  Class {cls}: {count} → {target_count} (+{n_to_add})")
        
        X_final = np.concatenate(X_resampled, axis=0)
        y_final = np.concatenate(y_resampled, axis=0)
        
        # Перемешиваем
        shuffle_idx = np.random.permutation(len(X_final))
        
        return X_final[shuffle_idx], y_final[shuffle_idx]
    
    def undersample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_ratio: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Undersampling majority класса.
        
        Args:
            X: Features
            y: Labels
            target_ratio: Целевое соотношение majority/minority
        
        Returns:
            X_resampled, y_resampled
        """
        target_ratio = target_ratio or self.config.undersample_ratio
        
        class_counts = Counter(y)
        minority_count = min(class_counts.values())
        target_count = int(minority_count / target_ratio)
        
        _log(f"Undersampling: minority={minority_count}, target_majority={target_count}")
        
        indices_to_keep = []
        
        for cls, count in class_counts.items():
            cls_indices = np.where(y == cls)[0]
            
            if count > target_count:
                # Случайно выбираем подмножество
                keep_indices = np.random.choice(
                    cls_indices,
                    size=target_count,
                    replace=False
                )
                indices_to_keep.extend(keep_indices)
                _log(f"  Class {cls}: {count} → {target_count}")
            else:
                indices_to_keep.extend(cls_indices)
        
        indices_to_keep = np.array(indices_to_keep)
        
        # Сортируем для сохранения временного порядка (важно для time series!)
        indices_to_keep = np.sort(indices_to_keep)
        
        return X[indices_to_keep], y[indices_to_keep]
    
    def smote_timeseries(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k_neighbors: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        SMOTE адаптированный для временных рядов.
        
        Вместо интерполяции точек, интерполирует целые sequences.
        
        Args:
            X: Sequences (N, seq_len, feature_dim)
            y: Labels (N,)
            k_neighbors: Количество соседей
        
        Returns:
            X_resampled, y_resampled
        """
        k_neighbors = k_neighbors or self.config.smote_k_neighbors
        
        class_counts = Counter(y)
        majority_count = max(class_counts.values())
        
        X_resampled = [X]
        y_resampled = [y]
        
        for cls, count in class_counts.items():
            if count < majority_count:
                cls_indices = np.where(y == cls)[0]
                X_cls = X[cls_indices]
                
                n_to_generate = majority_count - count
                
                # Генерируем синтетические samples
                synthetic = self._generate_smote_samples(
                    X_cls,
                    n_to_generate,
                    k_neighbors
                )
                
                X_resampled.append(synthetic)
                y_resampled.append(np.full(n_to_generate, cls, dtype=y.dtype))
                
                _log(f"  SMOTE Class {cls}: +{n_to_generate} synthetic samples")
        
        X_final = np.concatenate(X_resampled, axis=0)
        y_final = np.concatenate(y_resampled, axis=0)
        
        shuffle_idx = np.random.permutation(len(X_final))
        
        return X_final[shuffle_idx], y_final[shuffle_idx]
    
    def _generate_smote_samples(
        self,
        X_class: np.ndarray,
        n_samples: int,
        k_neighbors: int
    ) -> np.ndarray:
        """Генерация SMOTE samples для одного класса."""
        n_existing = len(X_class)
        
        if n_existing < k_neighbors:
            k_neighbors = max(1, n_existing - 1)
        
        # Flatten для поиска соседей
        if len(X_class.shape) == 3:
            X_flat = X_class.reshape(n_existing, -1)
        else:
            X_flat = X_class
        
        synthetic = []
        
        for _ in range(n_samples):
            # Выбираем случайный sample
            idx = np.random.randint(0, n_existing)
            sample = X_flat[idx]
            
            # Находим k ближайших соседей
            distances = np.linalg.norm(X_flat - sample, axis=1)
            neighbor_indices = np.argsort(distances)[1:k_neighbors + 1]
            
            # Выбираем случайного соседа
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor = X_flat[neighbor_idx]
            
            # Интерполяция
            alpha = np.random.random()
            new_sample = sample + alpha * (neighbor - sample)
            
            synthetic.append(new_sample)
        
        synthetic = np.array(synthetic)
        
        # Reshape обратно если нужно
        if len(X_class.shape) == 3:
            synthetic = synthetic.reshape(-1, X_class.shape[1], X_class.shape[2])
        
        return synthetic


# ============================================================================
# MAIN CLASS BALANCING STRATEGY
# ============================================================================

class ClassBalancingStrategyV2:
    """
    Комплексная стратегия балансировки классов v2.
    
    Объединяет все методы балансировки в единый интерфейс.
    """
    
    def __init__(self, config: Optional[ClassBalancingConfigV2] = None):
        """
        Args:
            config: Конфигурация балансировки
        """
        self.config = config or ClassBalancingConfigV2()
        self.threshold_optimizer = ThresholdOptimizer(self.config)
        self.resampler = ResamplingStrategy(self.config)
        
        # Cached weights
        self._class_weights: Optional[np.ndarray] = None
        self._focal_alpha: Optional[np.ndarray] = None
        
        _log(
            f"✓ ClassBalancingStrategyV2 initialized: "
            f"method={self.config.method.value}, "
            f"focal={self.config.use_focal_loss}, "
            f"oversample={self.config.use_oversampling}"
        )
    
    def balance_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Применить балансировку к датасету.
        
        Args:
            X: Features или sequences
            y: Labels
            returns: Доходности (для перемаркировки)
        
        Returns:
            X_balanced, y_balanced
        """
        tqdm.write("\n" + "=" * 60)
        tqdm.write("[Balancing] БАЛАНСИРОВКА КЛАССОВ")
        tqdm.write("=" * 60)

        # Логируем исходное распределение
        before_dist = Counter(y)
        _log(f"До балансировки: {dict(before_dist)}")

        # Опционально: перемаркировка с адаптивными порогами
        if returns is not None and self.config.threshold_method != ThresholdMethod.FIXED:
            _log("Применение адаптивных порогов...")
            y = self.threshold_optimizer.relabel_data(returns)
        
        # Применяем метод балансировки
        method = self.config.method
        
        if method == BalancingMethod.NONE:
            pass  # Ничего не делаем
        
        elif method == BalancingMethod.OVERSAMPLING:
            if self.config.use_oversampling:
                X, y = self.resampler.oversample(X, y)
        
        elif method == BalancingMethod.UNDERSAMPLING:
            if self.config.use_undersampling:
                X, y = self.resampler.undersample(X, y)
        
        elif method == BalancingMethod.SMOTE:
            if self.config.use_smote:
                X, y = self.resampler.smote_timeseries(X, y)
        
        elif method == BalancingMethod.COMBINED:
            # Oversampling + будет использовать class weights в loss
            if self.config.use_oversampling:
                X, y = self.resampler.oversample(X, y)
        
        # Логируем результат
        after_dist = Counter(y)
        _log(f"После балансировки: {dict(after_dist)}")
        tqdm.write("=" * 60 + "\n")
        
        return X, y
    
    def get_class_weights(
        self,
        y: np.ndarray,
        num_classes: int = 3
    ) -> np.ndarray:
        """
        Получить веса классов.
        
        Args:
            y: Labels
            num_classes: Количество классов
        
        Returns:
            weights: Array весов
        """
        if not self.config.use_class_weights:
            return np.ones(num_classes, dtype=np.float32)
        
        weights = ClassWeightsCalculator.compute(
            y,
            method=self.config.weight_method,
            smoothing=self.config.weight_smoothing,
            num_classes=num_classes
        )
        
        self._class_weights = weights
        return weights
    
    def get_focal_alpha(
        self,
        y: np.ndarray,
        num_classes: int = 3
    ) -> np.ndarray:
        """
        Получить alpha для Focal Loss.
        
        Args:
            y: Labels
            num_classes: Количество классов
        
        Returns:
            alpha: Array значений alpha
        """
        if not self.config.use_focal_loss:
            return None
        
        alpha = ClassWeightsCalculator.compute_focal_alpha(y, num_classes)
        self._focal_alpha = alpha
        return alpha
    
    def get_loss_params(
        self,
        y: np.ndarray,
        num_classes: int = 3
    ) -> Dict:
        """
        Получить все параметры для loss function.
        
        Args:
            y: Labels
            num_classes: Количество классов
        
        Returns:
            Dict с параметрами для loss
        """
        params = {
            'use_focal_loss': self.config.use_focal_loss,
            'focal_gamma': self.config.focal_gamma,
            'class_weights': None,
            'focal_alpha': None
        }
        
        if self.config.use_class_weights:
            params['class_weights'] = self.get_class_weights(y, num_classes)
        
        if self.config.use_focal_loss:
            params['focal_alpha'] = self.get_focal_alpha(y, num_classes)
        
        return params


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_balancing_strategy(
    preset: str = "production"
) -> ClassBalancingStrategyV2:
    """
    Создать стратегию балансировки из пресета.
    
    Пресеты:
        - production: Оптимальная для production (Focal + Oversampling)
        - conservative: Консервативная (только class weights)
        - aggressive: Агрессивная (SMOTE + высокий gamma)
    
    Returns:
        ClassBalancingStrategyV2 instance
    """
    if preset == "production":
        config = ClassBalancingConfigV2(
            method=BalancingMethod.COMBINED,
            use_class_weights=True,
            use_focal_loss=True,
            focal_gamma=2.5,
            use_oversampling=True,
            oversample_ratio=0.5,
            threshold_method=ThresholdMethod.PERCENTILE,
            percentile_sell=0.25,
            percentile_buy=0.75
        )
    
    elif preset == "conservative":
        config = ClassBalancingConfigV2(
            method=BalancingMethod.CLASS_WEIGHTS,
            use_class_weights=True,
            use_focal_loss=True,
            focal_gamma=2.0,
            use_oversampling=False,
            threshold_method=ThresholdMethod.FIXED
        )
    
    elif preset == "aggressive":
        config = ClassBalancingConfigV2(
            method=BalancingMethod.COMBINED,
            use_class_weights=True,
            use_focal_loss=True,
            focal_gamma=3.0,
            use_oversampling=True,
            oversample_ratio=0.7,
            use_smote=True,
            smote_k_neighbors=5,
            threshold_method=ThresholdMethod.ADAPTIVE
        )
    
    else:
        config = ClassBalancingConfigV2()
    
    return ClassBalancingStrategyV2(config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ТЕСТ CLASS BALANCING V2")
    print("=" * 80)
    
    # Создаём тестовые данные с imbalance
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 110
    
    # Imbalanced labels: 70% HOLD, 15% BUY, 15% SELL
    y = np.random.choice(
        [0, 1, 2],
        size=n_samples,
        p=[0.70, 0.15, 0.15]
    )
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    print(f"\n📊 Original data:")
    print(f"  • X shape: {X.shape}")
    print(f"  • y distribution: {dict(Counter(y))}")
    
    # 1. Class Weights
    print("\n1️⃣ Class Weights Calculator:")
    for method in WeightMethod:
        weights = ClassWeightsCalculator.compute(y, method=method)
        print(f"  • {method.value}: {weights}")
    
    # 2. Threshold Optimizer
    print("\n2️⃣ Threshold Optimizer:")
    returns = np.random.randn(n_samples) * 0.01  # Симуляция доходностей
    
    config = ClassBalancingConfigV2()
    optimizer = ThresholdOptimizer(config)
    
    for method in ThresholdMethod:
        if method != ThresholdMethod.VOLATILITY_ADJUSTED:
            sell_th, buy_th = optimizer.compute_thresholds(returns, method)
            print(f"  • {method.value}: sell={sell_th:.6f}, buy={buy_th:.6f}")
    
    # 3. Full Balancing Strategy
    print("\n3️⃣ Full Balancing Strategy:")
    strategy = create_balancing_strategy("production")
    
    X_balanced, y_balanced = strategy.balance_dataset(X, y)
    
    print(f"\n📊 Balanced data:")
    print(f"  • X shape: {X_balanced.shape}")
    print(f"  • y distribution: {dict(Counter(y_balanced))}")
    
    # 4. Loss Parameters
    print("\n4️⃣ Loss Parameters:")
    loss_params = strategy.get_loss_params(y_balanced)
    print(f"  • use_focal_loss: {loss_params['use_focal_loss']}")
    print(f"  • focal_gamma: {loss_params['focal_gamma']}")
    print(f"  • class_weights: {loss_params['class_weights']}")
    print(f"  • focal_alpha: {loss_params['focal_alpha']}")
    
    print("\n" + "=" * 80)
    print("✅ Все тесты пройдены!")
    print("=" * 80)
