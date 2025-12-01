#!/usr/bin/env python3
"""
Combinatorial Purged Cross-Validation (CPCV) и Probability of Backtest Overfitting (PBO).

Реализация методов из книги López de Prado "Advances in Financial Machine Learning".

CPCV - комбинаторная cross-validation с purging для предотвращения data leakage
в контексте временных рядов с overlapping labels.

PBO - вероятность того, что лучшая in-sample стратегия покажет плохой результат
out-of-sample. Используется для оценки robustness стратегии.

Файл: backend/ml_engine/validation/cpcv.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterator, Callable
from itertools import combinations
import numpy as np
from scipy import stats
import warnings

from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CPCVConfig:
    """
    Конфигурация Combinatorial Purged Cross-Validation.

    Параметры:
        n_splits: Количество групп для разбиения данных
        n_test_splits: Количество групп в test set для каждой комбинации
        purge_length: Количество samples для purging между train/test
        embargo_length: Gap после test set для embargo
        allow_overlap: Разрешить частичное пересечение (не рекомендуется)
    """
    n_splits: int = 6  # Рекомендуется 6-10 для достаточной статистики
    n_test_splits: int = 2  # Количество групп в test
    purge_length: int = 60  # Обычно = sequence_length
    embargo_length: int = 30  # Обычно = label_horizon / 2
    allow_overlap: bool = False

    def __post_init__(self):
        """Валидация параметров."""
        if self.n_splits < 3:
            raise ValueError(f"n_splits must be >= 3, got {self.n_splits}")
        if self.n_test_splits >= self.n_splits:
            raise ValueError(
                f"n_test_splits ({self.n_test_splits}) must be < n_splits ({self.n_splits})"
            )
        if self.n_test_splits < 1:
            raise ValueError(f"n_test_splits must be >= 1, got {self.n_test_splits}")

    @property
    def n_combinations(self) -> int:
        """Количество комбинаций test groups."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)


@dataclass
class CPCVResult:
    """Результат одного fold CPCV."""
    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    test_groups: Tuple[int, ...]
    train_groups: Tuple[int, ...]
    purged_samples: int
    embargoed_samples: int


@dataclass
class PBOResult:
    """Результат расчёта Probability of Backtest Overfitting."""
    pbo: float  # Вероятность overfitting (0-1)
    pbo_adjusted: float  # PBO с коррекцией на кол-во комбинаций
    is_overfit: bool  # pbo > 0.5
    confidence_level: float  # Уровень confidence (1 - pbo)

    # Детали расчёта
    n_combinations: int
    is_sharpe_ratios: List[float]  # In-Sample Sharpe ratios
    oos_sharpe_ratios: List[float]  # Out-of-Sample Sharpe ratios
    rank_correlation: float  # Spearman correlation IS vs OOS

    # Статистика
    best_is_idx: int  # Индекс лучшей IS стратегии
    best_is_sharpe: float  # Sharpe лучшей IS стратегии
    best_is_oos_sharpe: float  # OOS Sharpe лучшей IS стратегии
    best_is_oos_rank: int  # OOS ранг лучшей IS стратегии

    # Интерпретация
    interpretation: str


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation для time series.

    CPCV генерирует все возможные комбинации train/test splits,
    применяя purging и embargo для предотвращения data leakage.

    Преимущества:
    - Все данные используются и в train и в test
    - Более robust оценка модели
    - Возможность расчёта PBO

    Usage:
        >>> cpcv = CombinatorialPurgedCV(config)
        >>> for train_idx, test_idx in cpcv.split(X, y):
        ...     model.fit(X[train_idx], y[train_idx])
        ...     score = model.score(X[test_idx], y[test_idx])
    """

    def __init__(self, config: Optional[CPCVConfig] = None):
        """
        Инициализация CPCV.

        Args:
            config: Конфигурация CPCV
        """
        self.config = config or CPCVConfig()
        logger.info(
            f"CPCV initialized: n_splits={self.config.n_splits}, "
            f"n_test_splits={self.config.n_test_splits}, "
            f"combinations={self.config.n_combinations}"
        )

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Генератор комбинаторных splits с purging.

        Args:
            X: Features array (n_samples, n_features)
            y: Labels array (n_samples,) - не используется, для совместимости
            groups: Group labels - не используется, для совместимости

        Yields:
            (train_indices, test_indices) для каждой комбинации
        """
        n_samples = len(X)
        group_size = n_samples // self.config.n_splits

        # Создаём группы
        group_indices = []
        for g in range(self.config.n_splits):
            start = g * group_size
            end = (g + 1) * group_size if g < self.config.n_splits - 1 else n_samples
            group_indices.append(np.arange(start, end))

        # Генерируем все комбинации test groups
        test_combinations = list(combinations(
            range(self.config.n_splits),
            self.config.n_test_splits
        ))

        logger.info(f"CPCV: Generating {len(test_combinations)} fold combinations")

        for fold_idx, test_groups in enumerate(test_combinations):
            train_groups = tuple(
                g for g in range(self.config.n_splits) if g not in test_groups
            )

            # Собираем test indices
            test_idx = np.concatenate([group_indices[g] for g in test_groups])

            # Собираем train indices с purging
            train_idx = self._get_purged_train_indices(
                group_indices, train_groups, test_groups, n_samples
            )

            yield train_idx, test_idx

    def _get_purged_train_indices(
        self,
        group_indices: List[np.ndarray],
        train_groups: Tuple[int, ...],
        test_groups: Tuple[int, ...],
        n_samples: int
    ) -> np.ndarray:
        """
        Получить train indices с применением purging и embargo.

        Args:
            group_indices: Индексы для каждой группы
            train_groups: Группы для training
            test_groups: Группы для testing
            n_samples: Общее количество samples

        Returns:
            Purged train indices
        """
        # Собираем все train indices
        train_idx_list = []

        for g in train_groups:
            group_idx = group_indices[g].copy()

            # Проверяем соседство с test groups
            for test_g in test_groups:
                # Purging: убираем samples перед test group
                if g == test_g - 1:  # Train group перед test
                    # Убираем последние purge_length samples
                    if len(group_idx) > self.config.purge_length:
                        group_idx = group_idx[:-self.config.purge_length]

                # Embargo: убираем samples после test group
                if g == test_g + 1:  # Train group после test
                    # Убираем первые embargo_length samples
                    if len(group_idx) > self.config.embargo_length:
                        group_idx = group_idx[self.config.embargo_length:]

            if len(group_idx) > 0:
                train_idx_list.append(group_idx)

        if train_idx_list:
            return np.concatenate(train_idx_list)
        else:
            return np.array([], dtype=np.int64)

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Возвращает количество splits (комбинаций)."""
        return self.config.n_combinations

    def get_detailed_splits(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[CPCVResult]:
        """
        Получить детальную информацию о каждом split.

        Returns:
            List[CPCVResult] с информацией о каждом fold
        """
        n_samples = len(X)
        group_size = n_samples // self.config.n_splits

        # Создаём группы
        group_indices = []
        for g in range(self.config.n_splits):
            start = g * group_size
            end = (g + 1) * group_size if g < self.config.n_splits - 1 else n_samples
            group_indices.append(np.arange(start, end))

        # Генерируем все комбинации
        test_combinations = list(combinations(
            range(self.config.n_splits),
            self.config.n_test_splits
        ))

        results = []

        for fold_idx, test_groups in enumerate(test_combinations):
            train_groups = tuple(
                g for g in range(self.config.n_splits) if g not in test_groups
            )

            # Test indices
            test_idx = np.concatenate([group_indices[g] for g in test_groups])

            # Train indices (без purging для сравнения)
            train_idx_raw = np.concatenate([group_indices[g] for g in train_groups])

            # Train indices (с purging)
            train_idx_purged = self._get_purged_train_indices(
                group_indices, train_groups, test_groups, n_samples
            )

            purged_samples = len(train_idx_raw) - len(train_idx_purged)

            results.append(CPCVResult(
                fold_idx=fold_idx,
                train_indices=train_idx_purged,
                test_indices=test_idx,
                test_groups=test_groups,
                train_groups=train_groups,
                purged_samples=purged_samples,
                embargoed_samples=0  # Учтено в purged
            ))

        return results


class ProbabilityOfBacktestOverfitting:
    """
    Расчёт Probability of Backtest Overfitting (PBO).

    PBO измеряет вероятность того, что лучшая in-sample стратегия
    покажет плохой результат out-of-sample.

    Интерпретация PBO:
    - PBO < 0.1: Низкий риск overfitting
    - PBO 0.1-0.3: Умеренный риск
    - PBO 0.3-0.5: Высокий риск
    - PBO > 0.5: Очень высокий риск (модель скорее всего overfit)

    Usage:
        >>> pbo_calc = ProbabilityOfBacktestOverfitting()
        >>> sharpe_is, sharpe_oos = [], []
        >>> for train_idx, test_idx in cpcv.split(X):
        ...     model.fit(X[train_idx], y[train_idx])
        ...     sharpe_is.append(calculate_sharpe(model, X[train_idx]))
        ...     sharpe_oos.append(calculate_sharpe(model, X[test_idx]))
        >>> result = pbo_calc.calculate(sharpe_is, sharpe_oos)
        >>> print(f"PBO: {result.pbo:.2%}")
    """

    def __init__(self, n_simulations: int = 1000):
        """
        Args:
            n_simulations: Количество симуляций для Monte Carlo (если используется)
        """
        self.n_simulations = n_simulations

    def calculate(
        self,
        is_sharpes: List[float],
        oos_sharpes: List[float],
        return_details: bool = True
    ) -> PBOResult:
        """
        Рассчитать PBO на основе IS и OOS Sharpe ratios.

        Алгоритм (López de Prado):
        1. Для каждой пары (IS, OOS) из CPCV комбинаций
        2. Найти стратегию с лучшим IS Sharpe
        3. Посмотреть её ранг в OOS
        4. PBO = доля случаев, когда лучшая IS стратегия в нижней половине OOS

        Args:
            is_sharpes: In-Sample Sharpe ratios для каждой комбинации
            oos_sharpes: Out-of-Sample Sharpe ratios для каждой комбинации
            return_details: Возвращать ли детальную статистику

        Returns:
            PBOResult с PBO и статистикой
        """
        is_sharpes = np.array(is_sharpes)
        oos_sharpes = np.array(oos_sharpes)

        n_combinations = len(is_sharpes)

        if n_combinations < 2:
            logger.warning("Need at least 2 combinations for PBO calculation")
            return PBOResult(
                pbo=0.0,
                pbo_adjusted=0.0,
                is_overfit=False,
                confidence_level=1.0,
                n_combinations=n_combinations,
                is_sharpe_ratios=is_sharpes.tolist(),
                oos_sharpe_ratios=oos_sharpes.tolist(),
                rank_correlation=0.0,
                best_is_idx=0,
                best_is_sharpe=is_sharpes[0] if len(is_sharpes) > 0 else 0.0,
                best_is_oos_sharpe=oos_sharpes[0] if len(oos_sharpes) > 0 else 0.0,
                best_is_oos_rank=1,
                interpretation="Insufficient data for PBO calculation"
            )

        # 1. Найти индекс лучшей IS стратегии
        best_is_idx = int(np.argmax(is_sharpes))
        best_is_sharpe = float(is_sharpes[best_is_idx])

        # 2. OOS Sharpe лучшей IS стратегии
        best_is_oos_sharpe = float(oos_sharpes[best_is_idx])

        # 3. Ранг лучшей IS стратегии в OOS (1 = лучший)
        oos_ranks = stats.rankdata(-oos_sharpes)  # Отрицательный для убывающего порядка
        best_is_oos_rank = int(oos_ranks[best_is_idx])

        # 4. Рассчитываем PBO
        # PBO = вероятность что лучшая IS в нижней половине OOS
        median_rank = n_combinations / 2
        pbo = float(best_is_oos_rank > median_rank)

        # Более точный PBO: доля стратегий которые "проиграли" OOS
        # относительно медианы
        oos_median = np.median(oos_sharpes)
        below_median_count = np.sum(oos_sharpes < oos_median)

        # Если лучшая IS ниже медианы OOS
        if best_is_oos_sharpe < oos_median:
            pbo = 1.0
        else:
            # Позиция относительно медианы
            pbo = float(best_is_oos_rank) / float(n_combinations)

        # Adjusted PBO с учётом кол-ва комбинаций
        # (больше комбинаций = более reliable оценка)
        adjustment_factor = min(1.0, np.sqrt(n_combinations / 10))
        pbo_adjusted = pbo * adjustment_factor

        # Rank correlation между IS и OOS
        try:
            rank_corr, _ = stats.spearmanr(is_sharpes, oos_sharpes)
            if np.isnan(rank_corr):
                rank_corr = 0.0
        except Exception:
            rank_corr = 0.0

        # Интерпретация
        if pbo < 0.1:
            interpretation = "Low overfitting risk. Model shows robust out-of-sample performance."
        elif pbo < 0.3:
            interpretation = "Moderate overfitting risk. Consider additional validation."
        elif pbo < 0.5:
            interpretation = "High overfitting risk. Model may not generalize well."
        else:
            interpretation = "Very high overfitting risk. The best in-sample strategy likely won't perform well out-of-sample."

        result = PBOResult(
            pbo=pbo,
            pbo_adjusted=pbo_adjusted,
            is_overfit=pbo > 0.5,
            confidence_level=1.0 - pbo,
            n_combinations=n_combinations,
            is_sharpe_ratios=is_sharpes.tolist(),
            oos_sharpe_ratios=oos_sharpes.tolist(),
            rank_correlation=rank_corr,
            best_is_idx=best_is_idx,
            best_is_sharpe=best_is_sharpe,
            best_is_oos_sharpe=best_is_oos_sharpe,
            best_is_oos_rank=best_is_oos_rank,
            interpretation=interpretation
        )

        # Логируем результат
        logger.info(
            f"PBO Calculation Complete:\n"
            f"  • PBO: {pbo:.2%} (adjusted: {pbo_adjusted:.2%})\n"
            f"  • Is Overfit: {result.is_overfit}\n"
            f"  • Combinations: {n_combinations}\n"
            f"  • Best IS Sharpe: {best_is_sharpe:.3f}\n"
            f"  • Best IS OOS Sharpe: {best_is_oos_sharpe:.3f}\n"
            f"  • Best IS OOS Rank: {best_is_oos_rank}/{n_combinations}\n"
            f"  • IS-OOS Correlation: {rank_corr:.3f}\n"
            f"  • {interpretation}"
        )

        return result

    def calculate_deflated_sharpe(
        self,
        sharpe_ratio: float,
        n_trials: int,
        variance_of_sharpes: float,
        skewness: float = 0.0,
        kurtosis: float = 3.0
    ) -> float:
        """
        Рассчитать Deflated Sharpe Ratio (DSR).

        DSR корректирует Sharpe Ratio на:
        - Количество проведённых trials (multiple testing)
        - Non-normality returns (skewness, kurtosis)

        Формула Bailey & López de Prado (2014).

        Args:
            sharpe_ratio: Исходный Sharpe Ratio
            n_trials: Количество проведённых trials/стратегий
            variance_of_sharpes: Дисперсия Sharpe ratios всех trials
            skewness: Асимметрия returns (0 для нормального распределения)
            kurtosis: Эксцесс returns (3 для нормального распределения)

        Returns:
            Deflated Sharpe Ratio (вероятность что SR > 0)
        """
        if n_trials <= 1:
            return sharpe_ratio

        # Expected maximum Sharpe under null hypothesis
        # E[max(SR)] ≈ (1 - γ) * Φ^-1(1 - 1/N) + γ * Φ^-1(1 - 1/(N*e))
        # где γ ≈ 0.5772 (Euler-Mascheroni constant)

        gamma = 0.5772156649  # Euler-Mascheroni constant

        try:
            # Approximate expected max Sharpe
            expected_max_sr = (
                (1 - gamma) * stats.norm.ppf(1 - 1/n_trials) +
                gamma * stats.norm.ppf(1 - 1/(n_trials * np.e))
            )

            # Standard deviation of SR estimate
            # Учитываем non-normality
            sr_std = np.sqrt(
                (1 + 0.5 * sharpe_ratio**2 -
                 skewness * sharpe_ratio +
                 ((kurtosis - 3) / 4) * sharpe_ratio**2)
            )

            if sr_std == 0:
                sr_std = 1.0

            # Deflated Sharpe
            z_score = (sharpe_ratio - expected_max_sr) / sr_std
            deflated_sr = float(stats.norm.cdf(z_score))

            return deflated_sr

        except Exception as e:
            logger.warning(f"Error calculating DSR: {e}")
            return sharpe_ratio


def run_cpcv_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    score_fn: Callable,
    config: Optional[CPCVConfig] = None,
    calculate_pbo: bool = True
) -> Tuple[List[float], List[float], Optional[PBOResult]]:
    """
    Запустить полную CPCV валидацию с опциональным расчётом PBO.

    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        model_fn: Функция создания модели: () -> model
        score_fn: Функция оценки: (model, X, y) -> score
        config: Конфигурация CPCV
        calculate_pbo: Рассчитывать ли PBO

    Returns:
        (is_scores, oos_scores, pbo_result)

    Example:
        >>> def create_model():
        ...     return RandomForestClassifier(n_estimators=100)
        >>> def score_model(model, X, y):
        ...     preds = model.predict(X)
        ...     returns = preds * y  # Simplified
        ...     return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        >>> is_scores, oos_scores, pbo = run_cpcv_validation(
        ...     X, y, create_model, score_model
        ... )
    """
    config = config or CPCVConfig()
    cpcv = CombinatorialPurgedCV(config)

    is_scores = []
    oos_scores = []

    logger.info(f"Running CPCV validation with {cpcv.get_n_splits()} combinations...")

    for fold_idx, (train_idx, test_idx) in enumerate(cpcv.split(X, y)):
        try:
            # Create and train model
            model = model_fn()
            model.fit(X[train_idx], y[train_idx])

            # Calculate scores
            is_score = score_fn(model, X[train_idx], y[train_idx])
            oos_score = score_fn(model, X[test_idx], y[test_idx])

            is_scores.append(is_score)
            oos_scores.append(oos_score)

            logger.debug(
                f"Fold {fold_idx + 1}: IS={is_score:.3f}, OOS={oos_score:.3f}"
            )

        except Exception as e:
            logger.error(f"Error in fold {fold_idx}: {e}")
            # Append NaN for failed folds
            is_scores.append(np.nan)
            oos_scores.append(np.nan)

    # Calculate PBO
    pbo_result = None
    if calculate_pbo:
        # Filter out NaN values
        valid_mask = ~np.isnan(is_scores) & ~np.isnan(oos_scores)
        valid_is = [s for s, v in zip(is_scores, valid_mask) if v]
        valid_oos = [s for s, v in zip(oos_scores, valid_mask) if v]

        if len(valid_is) >= 2:
            pbo_calc = ProbabilityOfBacktestOverfitting()
            pbo_result = pbo_calc.calculate(valid_is, valid_oos)
        else:
            logger.warning("Not enough valid folds for PBO calculation")

    return is_scores, oos_scores, pbo_result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CPCV & PBO EXAMPLE")
    print("=" * 80)

    # Создаем тестовые данные
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)  # 3 classes

    # Конфигурация CPCV
    config = CPCVConfig(
        n_splits=6,
        n_test_splits=2,
        purge_length=30,
        embargo_length=15
    )

    # Создаем CPCV
    cpcv = CombinatorialPurgedCV(config)

    print(f"\nCPCV Configuration:")
    print(f"  • N splits: {config.n_splits}")
    print(f"  • N test splits: {config.n_test_splits}")
    print(f"  • Total combinations: {config.n_combinations}")
    print(f"  • Purge length: {config.purge_length}")
    print(f"  • Embargo length: {config.embargo_length}")

    # Генерируем splits
    print(f"\nGenerating splits...")
    splits = cpcv.get_detailed_splits(X, y)

    for result in splits[:3]:  # Показываем первые 3
        print(
            f"  Fold {result.fold_idx}: "
            f"train={len(result.train_indices)}, "
            f"test={len(result.test_indices)}, "
            f"test_groups={result.test_groups}, "
            f"purged={result.purged_samples}"
        )

    # Симулируем Sharpe ratios для PBO
    print(f"\nSimulating Sharpe ratios for PBO calculation...")
    is_sharpes = np.random.normal(1.5, 0.5, config.n_combinations).tolist()
    oos_sharpes = np.random.normal(0.8, 0.6, config.n_combinations).tolist()

    # Рассчитываем PBO
    pbo_calc = ProbabilityOfBacktestOverfitting()
    pbo_result = pbo_calc.calculate(is_sharpes, oos_sharpes)

    print(f"\n✅ PBO Calculation Complete!")
    print(f"  • PBO: {pbo_result.pbo:.2%}")
    print(f"  • Is Overfit: {pbo_result.is_overfit}")
    print(f"  • Confidence: {pbo_result.confidence_level:.2%}")
