"""
A/B Testing Infrastructure для ML моделей

Функции:
- Traffic splitting между моделями
- Метрики сравнения
- Statistical significance testing
- Automatic promotion/rollback
"""

import asyncio
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from pydantic import BaseModel

from backend.core.logger import get_logger

logger = get_logger(__name__)


class ExperimentStatus(str, Enum):
    """Статус A/B эксперимента"""
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelVariant(str, Enum):
    """Вариант модели в эксперименте"""
    CONTROL = "control"  # Текущая production модель
    TREATMENT = "treatment"  # Новая тестируемая модель


@dataclass
class PredictionOutcome:
    """Результат prediction"""
    timestamp: datetime
    symbol: str
    prediction: Any  # Может быть direction, confidence, etc.
    actual_outcome: Optional[float] = None  # Actual return после trade
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class VariantMetrics:
    """Метрики для одного варианта модели"""
    variant: ModelVariant
    model_name: str
    model_version: str

    # Counters
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0

    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Trading metrics (если доступны)
    win_rate: float = 0.0
    avg_return: float = 0.0
    sharpe_ratio: float = 0.0
    total_pnl: float = 0.0

    # Technical metrics
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    error_rate: float = 0.0

    # Raw data для вычислений
    latencies: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    predictions: List[int] = field(default_factory=list)
    actuals: List[int] = field(default_factory=list)


class ABTestConfig(BaseModel):
    """Конфигурация A/B теста"""
    experiment_id: str
    control_model_name: str
    control_model_version: str
    treatment_model_name: str
    treatment_model_version: str

    # Traffic split (control + treatment должно быть = 1.0)
    control_traffic: float = 0.9  # 90% на control
    treatment_traffic: float = 0.1  # 10% на treatment

    # Duration
    duration_hours: int = 24  # Длительность эксперимента
    min_samples_per_variant: int = 100  # Минимум samples для statistical significance

    # Success criteria
    min_accuracy_improvement: float = 0.02  # Минимум +2% accuracy
    max_latency_degradation_ms: float = 2.0  # Максимум +2ms latency
    min_confidence_level: float = 0.95  # 95% confidence для статистических тестов


class ABTestManager:
    """
    A/B Test Manager для сравнения ML моделей

    Traffic splitting:
    - Control (90%): Production модель
    - Treatment (10%): Новая модель

    Автоматическое принятие решения:
    - Если treatment лучше → promote
    - Если treatment хуже → rollback
    """

    def __init__(self):
        self.experiments: Dict[str, ABTestConfig] = {}
        self.metrics: Dict[str, Dict[ModelVariant, VariantMetrics]] = {}
        self.experiment_start_times: Dict[str, datetime] = {}
        self.experiment_status: Dict[str, ExperimentStatus] = {}

        logger.info("A/B Test Manager initialized")

    async def create_experiment(
        self,
        experiment_id: str,
        control_model_name: str,
        control_model_version: str,
        treatment_model_name: str,
        treatment_model_version: str,
        control_traffic: float = 0.9,
        treatment_traffic: float = 0.1,
        duration_hours: int = 24,
        min_samples: int = 100
    ) -> bool:
        """
        Создать новый A/B эксперимент

        Args:
            experiment_id: Уникальный ID эксперимента
            control_model_name: Название control модели
            control_model_version: Версия control модели
            treatment_model_name: Название treatment модели
            treatment_model_version: Versия treatment модели
            control_traffic: % трафика на control (0-1)
            treatment_traffic: % трафика на treatment (0-1)
            duration_hours: Длительность в часах
            min_samples: Минимум samples для каждого варианта

        Returns:
            True если успешно
        """
        if experiment_id in self.experiments:
            logger.error(f"Experiment {experiment_id} already exists")
            return False

        if abs(control_traffic + treatment_traffic - 1.0) > 0.001:
            logger.error(f"Traffic split must sum to 1.0")
            return False

        config = ABTestConfig(
            experiment_id=experiment_id,
            control_model_name=control_model_name,
            control_model_version=control_model_version,
            treatment_model_name=treatment_model_name,
            treatment_model_version=treatment_model_version,
            control_traffic=control_traffic,
            treatment_traffic=treatment_traffic,
            duration_hours=duration_hours,
            min_samples_per_variant=min_samples
        )

        self.experiments[experiment_id] = config
        self.metrics[experiment_id] = {
            ModelVariant.CONTROL: VariantMetrics(
                variant=ModelVariant.CONTROL,
                model_name=control_model_name,
                model_version=control_model_version
            ),
            ModelVariant.TREATMENT: VariantMetrics(
                variant=ModelVariant.TREATMENT,
                model_name=treatment_model_name,
                model_version=treatment_model_version
            )
        }
        self.experiment_start_times[experiment_id] = datetime.now()
        self.experiment_status[experiment_id] = ExperimentStatus.RUNNING

        logger.info(
            f"Created A/B experiment {experiment_id}: "
            f"{control_model_name}:{control_model_version} (control {control_traffic*100:.0f}%) vs "
            f"{treatment_model_name}:{treatment_model_version} (treatment {treatment_traffic*100:.0f}%)"
        )

        return True

    async def route_traffic(self, experiment_id: str) -> ModelVariant:
        """
        Traffic routing для эксперимента

        Args:
            experiment_id: ID эксперимента

        Returns:
            ModelVariant (CONTROL или TREATMENT)
        """
        if experiment_id not in self.experiments:
            logger.warning(f"Experiment {experiment_id} not found, using control")
            return ModelVariant.CONTROL

        config = self.experiments[experiment_id]

        # Random routing based on traffic split
        if random.random() < config.control_traffic:
            return ModelVariant.CONTROL
        else:
            return ModelVariant.TREATMENT

    async def record_prediction(
        self,
        experiment_id: str,
        variant: ModelVariant,
        outcome: PredictionOutcome
    ):
        """
        Записать результат prediction

        Args:
            experiment_id: ID эксперимента
            variant: Какой вариант использовался
            outcome: Результат prediction
        """
        if experiment_id not in self.metrics:
            logger.warning(f"Experiment {experiment_id} not found")
            return

        metrics = self.metrics[experiment_id][variant]

        # Обновить counters
        metrics.total_predictions += 1

        if outcome.error:
            metrics.failed_predictions += 1
        else:
            metrics.successful_predictions += 1

        # Записать latency
        if outcome.latency_ms > 0:
            metrics.latencies.append(outcome.latency_ms)

        # Записать prediction (если есть actual outcome)
        if outcome.actual_outcome is not None:
            metrics.returns.append(outcome.actual_outcome)

            # Предполагаем, что prediction это direction (BUY=1, SELL=-1, HOLD=0)
            # actual_outcome это return (положительный или отрицательный)
            if hasattr(outcome.prediction, 'direction'):
                pred_direction = outcome.prediction.direction
                # Correct prediction если direction совпадает со знаком return
                if pred_direction == "BUY" and outcome.actual_outcome > 0:
                    metrics.predictions.append(1)
                    metrics.actuals.append(1)
                elif pred_direction == "SELL" and outcome.actual_outcome < 0:
                    metrics.predictions.append(1)
                    metrics.actuals.append(1)
                else:
                    metrics.predictions.append(0)
                    metrics.actuals.append(0 if outcome.actual_outcome == 0 else 1)

    async def calculate_metrics(self, experiment_id: str):
        """
        Вычислить метрики для эксперимента

        Args:
            experiment_id: ID эксперимента
        """
        if experiment_id not in self.metrics:
            return

        for variant in [ModelVariant.CONTROL, ModelVariant.TREATMENT]:
            metrics = self.metrics[experiment_id][variant]

            # Error rate
            if metrics.total_predictions > 0:
                metrics.error_rate = metrics.failed_predictions / metrics.total_predictions

            # Latency metrics
            if metrics.latencies:
                metrics.avg_latency_ms = np.mean(metrics.latencies)
                metrics.p95_latency_ms = np.percentile(metrics.latencies, 95)

            # Accuracy metrics
            if metrics.predictions and metrics.actuals:
                predictions_np = np.array(metrics.predictions)
                actuals_np = np.array(metrics.actuals)

                # Accuracy
                metrics.accuracy = np.mean(predictions_np == actuals_np)

                # Precision, Recall, F1 (для binary classification)
                tp = np.sum((predictions_np == 1) & (actuals_np == 1))
                fp = np.sum((predictions_np == 1) & (actuals_np == 0))
                fn = np.sum((predictions_np == 0) & (actuals_np == 1))

                metrics.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                metrics.recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics.f1_score = (
                    2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
                    if (metrics.precision + metrics.recall) > 0 else 0
                )

            # Trading metrics
            if metrics.returns:
                returns_np = np.array(metrics.returns)

                # Win rate
                metrics.win_rate = np.mean(returns_np > 0)

                # Average return
                metrics.avg_return = np.mean(returns_np)

                # Total P&L
                metrics.total_pnl = np.sum(returns_np)

                # Sharpe ratio (annualized, assuming daily returns)
                if len(returns_np) > 1:
                    std_return = np.std(returns_np)
                    if std_return > 0:
                        metrics.sharpe_ratio = (
                            np.mean(returns_np) / std_return * np.sqrt(252)
                        )

    async def analyze_experiment(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        Анализ эксперимента и сравнение вариантов

        Args:
            experiment_id: ID эксперимента

        Returns:
            Словарь с результатами анализа
        """
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return {}

        # Вычислить метрики
        await self.calculate_metrics(experiment_id)

        config = self.experiments[experiment_id]
        control_metrics = self.metrics[experiment_id][ModelVariant.CONTROL]
        treatment_metrics = self.metrics[experiment_id][ModelVariant.TREATMENT]

        # Статистическое сравнение
        statistical_significance = await self._statistical_test(
            control_metrics, treatment_metrics
        )

        # Сравнение метрик
        comparison = {
            "experiment_id": experiment_id,
            "status": self.experiment_status[experiment_id].value,
            "duration_hours": (
                datetime.now() - self.experiment_start_times[experiment_id]
            ).total_seconds() / 3600,

            "control": {
                "model": f"{control_metrics.model_name}:{control_metrics.model_version}",
                "samples": control_metrics.total_predictions,
                "accuracy": control_metrics.accuracy,
                "precision": control_metrics.precision,
                "recall": control_metrics.recall,
                "f1_score": control_metrics.f1_score,
                "win_rate": control_metrics.win_rate,
                "avg_return": control_metrics.avg_return,
                "sharpe_ratio": control_metrics.sharpe_ratio,
                "avg_latency_ms": control_metrics.avg_latency_ms,
                "p95_latency_ms": control_metrics.p95_latency_ms,
                "error_rate": control_metrics.error_rate
            },

            "treatment": {
                "model": f"{treatment_metrics.model_name}:{treatment_metrics.model_version}",
                "samples": treatment_metrics.total_predictions,
                "accuracy": treatment_metrics.accuracy,
                "precision": treatment_metrics.precision,
                "recall": treatment_metrics.recall,
                "f1_score": treatment_metrics.f1_score,
                "win_rate": treatment_metrics.win_rate,
                "avg_return": treatment_metrics.avg_return,
                "sharpe_ratio": treatment_metrics.sharpe_ratio,
                "avg_latency_ms": treatment_metrics.avg_latency_ms,
                "p95_latency_ms": treatment_metrics.p95_latency_ms,
                "error_rate": treatment_metrics.error_rate
            },

            "improvement": {
                "accuracy": treatment_metrics.accuracy - control_metrics.accuracy,
                "sharpe_ratio": treatment_metrics.sharpe_ratio - control_metrics.sharpe_ratio,
                "latency_ms": treatment_metrics.avg_latency_ms - control_metrics.avg_latency_ms,
                "error_rate": treatment_metrics.error_rate - control_metrics.error_rate
            },

            "statistical_significance": statistical_significance,

            "recommendation": await self._make_recommendation(
                config, control_metrics, treatment_metrics, statistical_significance
            )
        }

        return comparison

    async def _statistical_test(
        self,
        control: VariantMetrics,
        treatment: VariantMetrics
    ) -> Dict[str, Any]:
        """
        Statistical significance test (t-test для accuracy)

        Returns:
            Результаты statistical test
        """
        from scipy import stats

        result = {
            "test": "two_sample_t_test",
            "significant": False,
            "p_value": 1.0,
            "confidence_level": 0.0
        }

        # Проверка минимального размера выборки
        if (len(control.predictions) < 30 or len(treatment.predictions) < 30):
            result["note"] = "Insufficient sample size for statistical test (min 30)"
            return result

        try:
            # T-test для сравнения accuracy
            t_stat, p_value = stats.ttest_ind(
                control.predictions,
                treatment.predictions,
                equal_var=False  # Welch's t-test
            )

            result["p_value"] = float(p_value)
            result["t_statistic"] = float(t_stat)
            result["significant"] = p_value < 0.05  # 95% confidence
            result["confidence_level"] = 1 - p_value

        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            result["error"] = str(e)

        return result

    async def _make_recommendation(
        self,
        config: ABTestConfig,
        control: VariantMetrics,
        treatment: VariantMetrics,
        significance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Сделать рекомендацию на основе результатов

        Returns:
            Recommendation с action и reason
        """
        recommendation = {
            "action": "continue",  # continue, promote, rollback
            "reasons": []
        }

        # Проверка минимального количества samples
        min_samples = config.min_samples_per_variant
        if (control.total_predictions < min_samples or
            treatment.total_predictions < min_samples):
            recommendation["reasons"].append(
                f"Insufficient samples (need {min_samples} per variant)"
            )
            return recommendation

        # Проверка latency degradation
        latency_diff = treatment.avg_latency_ms - control.avg_latency_ms
        if latency_diff > config.max_latency_degradation_ms:
            recommendation["action"] = "rollback"
            recommendation["reasons"].append(
                f"Latency degraded by {latency_diff:.2f}ms "
                f"(max allowed: {config.max_latency_degradation_ms}ms)"
            )
            return recommendation

        # Проверка error rate
        if treatment.error_rate > control.error_rate * 1.5:
            recommendation["action"] = "rollback"
            recommendation["reasons"].append(
                f"Error rate increased by {(treatment.error_rate - control.error_rate)*100:.1f}%"
            )
            return recommendation

        # Проверка accuracy improvement
        accuracy_improvement = treatment.accuracy - control.accuracy
        if accuracy_improvement >= config.min_accuracy_improvement:
            if significance.get("significant", False):
                recommendation["action"] = "promote"
                recommendation["reasons"].append(
                    f"Accuracy improved by {accuracy_improvement*100:.1f}% "
                    f"(statistically significant, p={significance['p_value']:.4f})"
                )
            else:
                recommendation["reasons"].append(
                    f"Accuracy improved by {accuracy_improvement*100:.1f}% "
                    f"but not statistically significant (p={significance['p_value']:.4f})"
                )
        elif accuracy_improvement < -0.05:  # Worse by 5%
            recommendation["action"] = "rollback"
            recommendation["reasons"].append(
                f"Accuracy degraded by {abs(accuracy_improvement)*100:.1f}%"
            )
        else:
            recommendation["reasons"].append(
                f"Accuracy change is minimal ({accuracy_improvement*100:.1f}%)"
            )

        # Проверка Sharpe ratio (для trading метрик)
        sharpe_improvement = treatment.sharpe_ratio - control.sharpe_ratio
        if sharpe_improvement > 0.5:
            recommendation["reasons"].append(
                f"Sharpe ratio improved by {sharpe_improvement:.2f}"
            )

        return recommendation

    async def stop_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Остановить эксперимент и получить финальный отчет

        Args:
            experiment_id: ID эксперимента

        Returns:
            Финальный отчет
        """
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return {}

        self.experiment_status[experiment_id] = ExperimentStatus.COMPLETED

        # Финальный анализ
        analysis = await self.analyze_experiment(experiment_id)

        logger.info(
            f"Experiment {experiment_id} stopped. "
            f"Recommendation: {analysis['recommendation']['action']}"
        )

        return analysis

    def get_active_experiments(self) -> List[str]:
        """Получить список активных экспериментов"""
        return [
            exp_id for exp_id, status in self.experiment_status.items()
            if status == ExperimentStatus.RUNNING
        ]


# Singleton instance
_ab_test_manager: Optional[ABTestManager] = None


def get_ab_test_manager() -> ABTestManager:
    """Получить singleton instance A/B Test Manager"""
    global _ab_test_manager
    if _ab_test_manager is None:
        _ab_test_manager = ABTestManager()
    return _ab_test_manager
