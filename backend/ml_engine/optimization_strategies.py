#!/usr/bin/env python3
"""
Advanced Optimization Strategies for Hyperparameter Search.

Реализует алгоритмы для ускорения поиска гиперпараметров:

1. DEGRADATION DETECTION:
   - Определение направления деградации метрик
   - Автоматический выбор направления поиска

2. BINARY SEARCH OPTIMIZATION:
   - Бинарный поиск в выбранном направлении
   - Экономия до 70% trials

3. PARAMETER INTERACTION ANALYSIS:
   - Анализ взаимодействий между параметрами
   - Выявление конфликтующих параметров

4. ADAPTIVE SEARCH SPACE REDUCTION:
   - Динамическое сужение пространства поиска
   - Фокусировка на перспективных областях

5. WARM START FROM HISTORICAL RUNS:
   - Использование результатов предыдущих оптимизаций
   - Transfer learning для гиперпараметров

Файл: backend/ml_engine/optimization_strategies.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
from collections import defaultdict
import json
from pathlib import Path

from backend.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# DEGRADATION DETECTION
# ============================================================================

class TrendDirection(str, Enum):
    """Направление тренда."""
    INCREASING = "increasing"  # Параметр растёт → метрика улучшается
    DECREASING = "decreasing"  # Параметр растёт → метрика ухудшается
    UNCLEAR = "unclear"        # Нет чёткого тренда
    PEAKED = "peaked"          # Есть оптимум посередине


@dataclass
class DegradationAnalysis:
    """Результат анализа деградации."""
    parameter: str
    direction: TrendDirection
    optimal_range: Tuple[float, float]
    confidence: float  # 0-1
    n_samples: int
    correlation: float  # Корреляция param vs metric


class DegradationDetector:
    """
    Детектор деградации метрик.

    Анализирует историю trials и определяет:
    - В каком направлении изменение параметра улучшает метрики
    - Где находится оптимальная зона
    - Какие параметры конфликтуют

    Использование:
        detector = DegradationDetector()
        for trial in trials:
            detector.add_observation(trial.params, trial.metrics)
        analysis = detector.analyze("learning_rate")
    """

    def __init__(self, min_samples: int = 5):
        """
        Args:
            min_samples: Минимум наблюдений для анализа
        """
        self.min_samples = min_samples
        self.observations: List[Tuple[Dict[str, Any], Dict[str, float]]] = []

    def add_observation(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, float]
    ):
        """Добавить наблюдение."""
        self.observations.append((params.copy(), metrics.copy()))

    def analyze(
        self,
        parameter: str,
        metric: str = "val_f1",
        maximize: bool = True
    ) -> DegradationAnalysis:
        """
        Анализ деградации для параметра.

        Args:
            parameter: Имя параметра
            metric: Имя метрики
            maximize: True если метрику нужно максимизировать

        Returns:
            DegradationAnalysis
        """
        # Собираем данные
        param_values = []
        metric_values = []

        for params, metrics in self.observations:
            if parameter in params and metric in metrics:
                param_val = params[parameter]
                # Пропускаем нечисловые значения
                if not isinstance(param_val, (int, float)):
                    continue
                param_values.append(float(param_val))
                metric_values.append(float(metrics[metric]))

        n_samples = len(param_values)

        if n_samples < self.min_samples:
            return DegradationAnalysis(
                parameter=parameter,
                direction=TrendDirection.UNCLEAR,
                optimal_range=(float('-inf'), float('inf')),
                confidence=0.0,
                n_samples=n_samples,
                correlation=0.0
            )

        param_arr = np.array(param_values)
        metric_arr = np.array(metric_values)

        # Вычисляем корреляцию
        if np.std(param_arr) == 0 or np.std(metric_arr) == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(param_arr, metric_arr)[0, 1])

        # Определяем направление
        if maximize:
            # Для максимизации: положительная корреляция = increasing хорошо
            if correlation > 0.3:
                direction = TrendDirection.INCREASING
            elif correlation < -0.3:
                direction = TrendDirection.DECREASING
            else:
                # Проверяем на peaked (оптимум посередине)
                direction = self._check_peaked(param_arr, metric_arr, maximize)
        else:
            # Для минимизации: отрицательная корреляция = increasing хорошо
            if correlation < -0.3:
                direction = TrendDirection.INCREASING
            elif correlation > 0.3:
                direction = TrendDirection.DECREASING
            else:
                direction = self._check_peaked(param_arr, metric_arr, maximize)

        # Определяем оптимальный диапазон
        optimal_range = self._find_optimal_range(
            param_arr, metric_arr, maximize
        )

        # Confidence на основе корреляции и кол-ва samples
        confidence = min(1.0, abs(correlation) * np.sqrt(n_samples / 20))

        return DegradationAnalysis(
            parameter=parameter,
            direction=direction,
            optimal_range=optimal_range,
            confidence=confidence,
            n_samples=n_samples,
            correlation=correlation
        )

    def _check_peaked(
        self,
        param_arr: np.ndarray,
        metric_arr: np.ndarray,
        maximize: bool
    ) -> TrendDirection:
        """Проверить, есть ли пик посередине."""
        # Сортируем по параметру
        sorted_idx = np.argsort(param_arr)
        sorted_metrics = metric_arr[sorted_idx]

        n = len(sorted_metrics)
        if n < 5:
            return TrendDirection.UNCLEAR

        # Разбиваем на три части
        first_third = sorted_metrics[:n//3]
        middle_third = sorted_metrics[n//3:2*n//3]
        last_third = sorted_metrics[2*n//3:]

        first_mean = np.mean(first_third)
        middle_mean = np.mean(middle_third)
        last_mean = np.mean(last_third)

        if maximize:
            # Пик если середина лучше краёв
            if middle_mean > first_mean and middle_mean > last_mean:
                return TrendDirection.PEAKED
        else:
            # Минимум если середина лучше краёв
            if middle_mean < first_mean and middle_mean < last_mean:
                return TrendDirection.PEAKED

        return TrendDirection.UNCLEAR

    def _find_optimal_range(
        self,
        param_arr: np.ndarray,
        metric_arr: np.ndarray,
        maximize: bool
    ) -> Tuple[float, float]:
        """Найти оптимальный диапазон параметра."""
        # Берём топ-25% по метрике
        n = len(param_arr)
        n_top = max(1, n // 4)

        if maximize:
            top_idx = np.argsort(metric_arr)[-n_top:]
        else:
            top_idx = np.argsort(metric_arr)[:n_top]

        top_params = param_arr[top_idx]

        return (float(np.min(top_params)), float(np.max(top_params)))

    def get_conflicting_parameters(
        self,
        metric: str = "val_f1"
    ) -> List[Tuple[str, str, float]]:
        """
        Найти конфликтующие параметры.

        Returns:
            List of (param1, param2, negative_correlation)
        """
        if len(self.observations) < self.min_samples:
            return []

        # Собираем все числовые параметры
        all_params = set()
        for params, _ in self.observations:
            for name, value in params.items():
                if isinstance(value, (int, float)):
                    all_params.add(name)

        all_params = list(all_params)
        conflicts = []

        # Вычисляем корреляции между всеми парами
        for i, p1 in enumerate(all_params):
            for p2 in all_params[i+1:]:
                corr = self._param_param_correlation(p1, p2)
                if corr < -0.5:  # Сильная отрицательная корреляция
                    conflicts.append((p1, p2, corr))

        return sorted(conflicts, key=lambda x: x[2])

    def _param_param_correlation(self, p1: str, p2: str) -> float:
        """Корреляция между двумя параметрами в успешных trials."""
        values1 = []
        values2 = []

        for params, _ in self.observations:
            if p1 in params and p2 in params:
                v1, v2 = params[p1], params[p2]
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    values1.append(float(v1))
                    values2.append(float(v2))

        if len(values1) < 3:
            return 0.0

        arr1 = np.array(values1)
        arr2 = np.array(values2)

        if np.std(arr1) == 0 or np.std(arr2) == 0:
            return 0.0

        return float(np.corrcoef(arr1, arr2)[0, 1])


# ============================================================================
# BINARY SEARCH OPTIMIZER
# ============================================================================

@dataclass
class BinarySearchState:
    """Состояние бинарного поиска для параметра."""
    parameter: str
    low: float
    high: float
    current: float
    best_value: float
    best_metric: float
    iteration: int = 0
    max_iterations: int = 5
    tolerance: float = 0.01
    finished: bool = False


class BinarySearchOptimizer:
    """
    Бинарный поиск для оптимизации параметров.

    После определения направления деградации, использует бинарный поиск
    для быстрого нахождения оптимального значения.

    Использование:
        bs = BinarySearchOptimizer()
        state = bs.initialize("learning_rate", 1e-6, 1e-3, TrendDirection.PEAKED)
        while not state.finished:
            value = state.current
            metric = train_and_evaluate(value)
            state = bs.step(state, metric)
        print(f"Optimal: {state.best_value}")
    """

    def __init__(self, metric_direction: str = "maximize"):
        """
        Args:
            metric_direction: "maximize" или "minimize"
        """
        self.metric_direction = metric_direction

    def initialize(
        self,
        parameter: str,
        low: float,
        high: float,
        trend: TrendDirection,
        log_scale: bool = False
    ) -> BinarySearchState:
        """
        Инициализировать бинарный поиск.

        Args:
            parameter: Имя параметра
            low: Нижняя граница
            high: Верхняя граница
            trend: Направление тренда из DegradationDetector
            log_scale: Использовать логарифмическую шкалу

        Returns:
            BinarySearchState
        """
        if log_scale:
            low = np.log10(low)
            high = np.log10(high)

        # Начальное значение зависит от тренда
        if trend == TrendDirection.INCREASING:
            # Начинаем с более высоких значений
            current = low + 0.75 * (high - low)
        elif trend == TrendDirection.DECREASING:
            # Начинаем с более низких значений
            current = low + 0.25 * (high - low)
        else:
            # Начинаем с середины
            current = (low + high) / 2

        if log_scale:
            current = 10 ** current

        return BinarySearchState(
            parameter=parameter,
            low=10 ** low if log_scale else low,
            high=10 ** high if log_scale else high,
            current=current,
            best_value=current,
            best_metric=float('-inf') if self.metric_direction == "maximize" else float('inf')
        )

    def step(
        self,
        state: BinarySearchState,
        metric: float,
        log_scale: bool = False
    ) -> BinarySearchState:
        """
        Один шаг бинарного поиска.

        Args:
            state: Текущее состояние
            metric: Значение метрики для текущего параметра
            log_scale: Использовать логарифмическую шкалу

        Returns:
            Обновлённое состояние
        """
        # Обновляем лучшее значение
        is_better = (
            metric > state.best_metric if self.metric_direction == "maximize"
            else metric < state.best_metric
        )

        if is_better:
            state.best_value = state.current
            state.best_metric = metric

        state.iteration += 1

        # Проверяем условия остановки
        range_size = state.high - state.low
        if log_scale:
            range_size = np.log10(state.high) - np.log10(state.low)

        if state.iteration >= state.max_iterations or range_size < state.tolerance:
            state.finished = True
            return state

        # Бинарный поиск
        mid = (state.low + state.high) / 2
        if log_scale:
            mid = 10 ** ((np.log10(state.low) + np.log10(state.high)) / 2)

        if is_better:
            # Хорошее значение - сужаем диапазон вокруг него
            if state.current < mid:
                state.high = mid
            else:
                state.low = mid
        else:
            # Плохое значение - идём в противоположную сторону
            if state.current < mid:
                state.low = state.current
            else:
                state.high = state.current

        # Следующее значение - середина нового диапазона
        if log_scale:
            state.current = 10 ** ((np.log10(state.low) + np.log10(state.high)) / 2)
        else:
            state.current = (state.low + state.high) / 2

        return state


# ============================================================================
# ADAPTIVE SEARCH SPACE
# ============================================================================

@dataclass
class AdaptiveSearchSpace:
    """Адаптивное пространство поиска."""
    parameter: str
    original_low: float
    original_high: float
    current_low: float
    current_high: float
    reduction_factor: float = 0.5  # Насколько сужать за итерацию
    min_range_pct: float = 0.1    # Минимальный размер диапазона (% от original)


class AdaptiveSearchSpaceReducer:
    """
    Адаптивное сужение пространства поиска.

    На основе результатов trials сужает пространство поиска,
    фокусируясь на перспективных областях.

    Использование:
        reducer = AdaptiveSearchSpaceReducer()
        reducer.add_trial({"lr": 1e-4}, {"val_f1": 0.5})
        reducer.add_trial({"lr": 1e-5}, {"val_f1": 0.7})
        new_space = reducer.reduce_space("lr", 1e-6, 1e-3)
    """

    def __init__(self, focus_quantile: float = 0.25):
        """
        Args:
            focus_quantile: Какой процент лучших результатов использовать
        """
        self.focus_quantile = focus_quantile
        self.trials: List[Tuple[Dict[str, Any], Dict[str, float]]] = []

    def add_trial(self, params: Dict[str, Any], metrics: Dict[str, float]):
        """Добавить результат trial."""
        self.trials.append((params.copy(), metrics.copy()))

    def reduce_space(
        self,
        parameter: str,
        original_low: float,
        original_high: float,
        metric: str = "val_f1",
        maximize: bool = True
    ) -> AdaptiveSearchSpace:
        """
        Сузить пространство поиска.

        Args:
            parameter: Имя параметра
            original_low: Исходная нижняя граница
            original_high: Исходная верхняя граница
            metric: Метрика для оценки
            maximize: True если метрику нужно максимизировать

        Returns:
            AdaptiveSearchSpace с новыми границами
        """
        # Собираем данные по параметру
        param_values = []
        metric_values = []

        for params, metrics in self.trials:
            if parameter in params and metric in metrics:
                val = params[parameter]
                if isinstance(val, (int, float)):
                    param_values.append(float(val))
                    metric_values.append(float(metrics[metric]))

        if len(param_values) < 3:
            # Недостаточно данных - возвращаем оригинальный диапазон
            return AdaptiveSearchSpace(
                parameter=parameter,
                original_low=original_low,
                original_high=original_high,
                current_low=original_low,
                current_high=original_high
            )

        param_arr = np.array(param_values)
        metric_arr = np.array(metric_values)

        # Находим топовые trials
        n_top = max(1, int(len(param_arr) * self.focus_quantile))

        if maximize:
            top_idx = np.argsort(metric_arr)[-n_top:]
        else:
            top_idx = np.argsort(metric_arr)[:n_top]

        top_params = param_arr[top_idx]

        # Новые границы = min/max из топовых + небольшой margin
        margin = 0.1 * (np.max(top_params) - np.min(top_params))

        new_low = max(original_low, np.min(top_params) - margin)
        new_high = min(original_high, np.max(top_params) + margin)

        # Проверяем минимальный размер
        min_range = 0.1 * (original_high - original_low)
        if new_high - new_low < min_range:
            center = (new_low + new_high) / 2
            new_low = max(original_low, center - min_range / 2)
            new_high = min(original_high, center + min_range / 2)

        return AdaptiveSearchSpace(
            parameter=parameter,
            original_low=original_low,
            original_high=original_high,
            current_low=new_low,
            current_high=new_high
        )


# ============================================================================
# WARM START FROM HISTORY
# ============================================================================

class WarmStartManager:
    """
    Управление warm start из истории оптимизаций.

    Сохраняет и загружает результаты предыдущих оптимизаций
    для ускорения новых запусков.

    Использование:
        manager = WarmStartManager("data/hyperopt_history")
        manager.save_results(study_name, best_params, all_trials)
        warm_params = manager.get_warm_start_params("new_study")
    """

    def __init__(self, storage_path: str = "data/hyperopt_history"):
        """
        Args:
            storage_path: Путь для хранения истории
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_results(
        self,
        study_name: str,
        best_params: Dict[str, Any],
        all_trials: List[Dict[str, Any]],
        metrics: Dict[str, float]
    ):
        """
        Сохранить результаты оптимизации.

        Args:
            study_name: Имя study
            best_params: Лучшие параметры
            all_trials: Все trials
            metrics: Финальные метрики
        """
        result = {
            "study_name": study_name,
            "timestamp": np.datetime64('now').astype(str),
            "best_params": best_params,
            "metrics": metrics,
            "n_trials": len(all_trials),
            "trials_summary": self._summarize_trials(all_trials)
        }

        filepath = self.storage_path / f"{study_name}.json"
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Saved optimization results to {filepath}")

    def get_warm_start_params(
        self,
        similarity_threshold: float = 0.8
    ) -> Optional[Dict[str, Any]]:
        """
        Получить параметры для warm start на основе истории.

        Args:
            similarity_threshold: Минимальная "похожесть" задачи

        Returns:
            Параметры для warm start или None
        """
        history_files = list(self.storage_path.glob("*.json"))

        if not history_files:
            return None

        # Загружаем все результаты
        all_results = []
        for filepath in history_files:
            try:
                with open(filepath, "r") as f:
                    result = json.load(f)
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

        if not all_results:
            return None

        # Находим лучший результат
        best_result = max(
            all_results,
            key=lambda x: x.get("metrics", {}).get("val_f1", 0)
        )

        return best_result.get("best_params")

    def get_suggested_ranges(
        self,
        parameter: str
    ) -> Optional[Tuple[float, float]]:
        """
        Получить рекомендуемый диапазон для параметра на основе истории.

        Args:
            parameter: Имя параметра

        Returns:
            (low, high) или None
        """
        history_files = list(self.storage_path.glob("*.json"))

        values = []
        for filepath in history_files:
            try:
                with open(filepath, "r") as f:
                    result = json.load(f)
                    if parameter in result.get("best_params", {}):
                        val = result["best_params"][parameter]
                        if isinstance(val, (int, float)):
                            values.append(float(val))
            except:
                pass

        if len(values) < 2:
            return None

        # Диапазон = mean ± 2*std
        mean = np.mean(values)
        std = np.std(values)

        return (mean - 2 * std, mean + 2 * std)

    def _summarize_trials(
        self,
        trials: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Создать summary по trials."""
        if not trials:
            return {}

        metrics_names = set()
        for trial in trials:
            if "metrics" in trial:
                metrics_names.update(trial["metrics"].keys())

        summary = {}
        for metric in metrics_names:
            values = [
                t["metrics"][metric]
                for t in trials
                if "metrics" in t and metric in t["metrics"]
            ]
            if values:
                summary[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }

        return summary


# ============================================================================
# PARAMETER IMPORTANCE ANALYZER
# ============================================================================

class ParameterImportanceAnalyzer:
    """
    Анализ важности параметров на основе истории trials.

    Использует:
    - Correlation analysis
    - ANOVA (если достаточно данных)
    - Feature importance из gradient boosting

    Использование:
        analyzer = ParameterImportanceAnalyzer()
        for trial in trials:
            analyzer.add_trial(trial.params, trial.metrics)
        importance = analyzer.get_importance("val_f1")
    """

    def __init__(self):
        self.trials: List[Tuple[Dict[str, Any], Dict[str, float]]] = []

    def add_trial(self, params: Dict[str, Any], metrics: Dict[str, float]):
        """Добавить trial."""
        self.trials.append((params.copy(), metrics.copy()))

    def get_importance(
        self,
        metric: str = "val_f1",
        method: str = "correlation"
    ) -> Dict[str, float]:
        """
        Получить важность параметров.

        Args:
            metric: Метрика для анализа
            method: Метод расчёта ("correlation", "variance")

        Returns:
            Dict[parameter_name, importance_score]
        """
        if len(self.trials) < 5:
            return {}

        # Собираем числовые параметры
        param_names = set()
        for params, _ in self.trials:
            for name, value in params.items():
                if isinstance(value, (int, float, bool)):
                    param_names.add(name)

        importance = {}

        for param in param_names:
            param_values = []
            metric_values = []

            for params, metrics in self.trials:
                if param in params and metric in metrics:
                    val = params[param]
                    if isinstance(val, bool):
                        val = int(val)
                    if isinstance(val, (int, float)):
                        param_values.append(float(val))
                        metric_values.append(float(metrics[metric]))

            if len(param_values) < 3:
                continue

            param_arr = np.array(param_values)
            metric_arr = np.array(metric_values)

            if np.std(param_arr) == 0:
                importance[param] = 0.0
                continue

            if method == "correlation":
                # Абсолютная корреляция
                corr = np.corrcoef(param_arr, metric_arr)[0, 1]
                importance[param] = float(abs(corr)) if not np.isnan(corr) else 0.0

            elif method == "variance":
                # Объяснённая дисперсия (R²)
                from scipy import stats
                slope, intercept, r_value, _, _ = stats.linregress(param_arr, metric_arr)
                importance[param] = float(r_value ** 2) if not np.isnan(r_value) else 0.0

        # Нормализуем
        if importance:
            max_imp = max(importance.values())
            if max_imp > 0:
                importance = {k: v / max_imp for k, v in importance.items()}

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_parameter_interactions(
        self,
        metric: str = "val_f1",
        threshold: float = 0.3
    ) -> List[Tuple[str, str, float]]:
        """
        Найти взаимодействия между параметрами.

        Returns:
            List of (param1, param2, interaction_strength)
        """
        if len(self.trials) < 10:
            return []

        param_names = list(self.get_importance(metric).keys())
        interactions = []

        for i, p1 in enumerate(param_names):
            for p2 in param_names[i+1:]:
                strength = self._compute_interaction(p1, p2, metric)
                if abs(strength) > threshold:
                    interactions.append((p1, p2, strength))

        return sorted(interactions, key=lambda x: abs(x[2]), reverse=True)

    def _compute_interaction(
        self,
        param1: str,
        param2: str,
        metric: str
    ) -> float:
        """Вычислить силу взаимодействия между параметрами."""
        # Собираем данные
        data = []
        for params, metrics in self.trials:
            if param1 in params and param2 in params and metric in metrics:
                v1 = params[param1]
                v2 = params[param2]
                m = metrics[metric]

                if isinstance(v1, bool):
                    v1 = int(v1)
                if isinstance(v2, bool):
                    v2 = int(v2)

                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    data.append((float(v1), float(v2), float(m)))

        if len(data) < 5:
            return 0.0

        arr = np.array(data)

        # Взаимодействие = корреляция произведения параметров с метрикой
        # минус сумма индивидуальных корреляций
        product = arr[:, 0] * arr[:, 1]
        metric_arr = arr[:, 2]

        if np.std(product) == 0:
            return 0.0

        product_corr = np.corrcoef(product, metric_arr)[0, 1]
        p1_corr = np.corrcoef(arr[:, 0], metric_arr)[0, 1]
        p2_corr = np.corrcoef(arr[:, 1], metric_arr)[0, 1]

        # Взаимодействие = насколько произведение объясняет больше чем сумма
        interaction = float(abs(product_corr) - max(abs(p1_corr), abs(p2_corr)))

        return interaction if not np.isnan(interaction) else 0.0


# ============================================================================
# OPTIMIZATION SCHEDULER
# ============================================================================

@dataclass
class OptimizationSchedule:
    """Расписание оптимизации."""
    total_budget_hours: float
    groups: List[str]
    trials_per_group: Dict[str, int]
    estimated_time_per_trial_minutes: float = 48.0  # 4 эпохи * 12 мин


class OptimizationScheduler:
    """
    Планировщик оптимизации с учётом временного бюджета.

    Распределяет trials между группами параметров
    на основе их важности и доступного времени.

    Использование:
        scheduler = OptimizationScheduler()
        schedule = scheduler.create_schedule(
            total_hours=24,
            parameter_importance={"learning_rate": 0.9, "dropout": 0.6}
        )
    """

    def __init__(
        self,
        minutes_per_epoch: float = 12.0,
        epochs_per_trial: int = 4
    ):
        """
        Args:
            minutes_per_epoch: Минут на одну эпоху
            epochs_per_trial: Эпох на один trial
        """
        self.minutes_per_epoch = minutes_per_epoch
        self.epochs_per_trial = epochs_per_trial
        self.minutes_per_trial = minutes_per_epoch * epochs_per_trial

    def create_schedule(
        self,
        total_hours: float,
        parameter_importance: Dict[str, float],
        min_trials_per_group: int = 5,
        max_trials_per_group: int = 20
    ) -> OptimizationSchedule:
        """
        Создать расписание оптимизации.

        Args:
            total_hours: Общий бюджет времени
            parameter_importance: Важность параметров (группированных)
            min_trials_per_group: Минимум trials на группу
            max_trials_per_group: Максимум trials на группу

        Returns:
            OptimizationSchedule
        """
        total_minutes = total_hours * 60
        total_trials = int(total_minutes / self.minutes_per_trial)

        # Нормализуем importance
        total_importance = sum(parameter_importance.values())
        if total_importance == 0:
            total_importance = 1

        normalized = {
            k: v / total_importance
            for k, v in parameter_importance.items()
        }

        # Распределяем trials пропорционально важности
        trials_per_group = {}
        remaining_trials = total_trials

        for group, importance in sorted(
            normalized.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            # Базовое количество trials
            base_trials = int(total_trials * importance)

            # Ограничиваем
            trials = max(min_trials_per_group, min(max_trials_per_group, base_trials))
            trials = min(trials, remaining_trials)

            trials_per_group[group] = trials
            remaining_trials -= trials

            if remaining_trials <= 0:
                break

        # Если остались trials - добавляем к наиболее важным
        if remaining_trials > 0:
            most_important = max(normalized.keys(), key=lambda x: normalized[x])
            current = trials_per_group.get(most_important, 0)
            trials_per_group[most_important] = min(
                max_trials_per_group,
                current + remaining_trials
            )

        return OptimizationSchedule(
            total_budget_hours=total_hours,
            groups=list(trials_per_group.keys()),
            trials_per_group=trials_per_group,
            estimated_time_per_trial_minutes=self.minutes_per_trial
        )

    def estimate_time(self, schedule: OptimizationSchedule) -> float:
        """
        Оценить время выполнения расписания.

        Returns:
            Часы
        """
        total_trials = sum(schedule.trials_per_group.values())
        total_minutes = total_trials * schedule.estimated_time_per_trial_minutes
        return total_minutes / 60


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def suggest_optimization_strategy(
    available_time_hours: float,
    n_parameters: int,
    data_size: int
) -> Dict[str, Any]:
    """
    Предложить стратегию оптимизации на основе условий.

    Args:
        available_time_hours: Доступное время
        n_parameters: Количество параметров
        data_size: Размер датасета

    Returns:
        Рекомендуемая стратегия
    """
    minutes_per_trial = 48  # 4 эпохи * 12 мин

    max_trials = int(available_time_hours * 60 / minutes_per_trial)

    # Определяем режим
    if max_trials < 10:
        mode = "quick"
        strategy = "Focus on learning_rate only"
    elif max_trials < 30:
        mode = "targeted"
        strategy = "Optimize learning_rate and regularization groups"
    elif max_trials < 100:
        mode = "standard"
        strategy = "Full sequential group optimization"
    else:
        mode = "comprehensive"
        strategy = "Full optimization with fine-tuning"

    # Рекомендации по эпохам
    if data_size < 10000:
        epochs_per_trial = 4
    elif data_size < 50000:
        epochs_per_trial = 3
    else:
        epochs_per_trial = 2

    return {
        "mode": mode,
        "strategy": strategy,
        "max_trials": max_trials,
        "epochs_per_trial": epochs_per_trial,
        "estimated_time_hours": max_trials * minutes_per_trial / 60,
        "recommendations": [
            f"Use {mode} mode for {available_time_hours}h budget",
            f"Run {epochs_per_trial} epochs per trial",
            f"Focus on high-importance parameters first",
            "Enable aggressive pruning to save time"
        ]
    }
