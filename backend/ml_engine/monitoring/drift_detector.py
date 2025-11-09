"""
Model Drift Detection для мониторинга деградации модели.

Функциональность:
- Data drift detection (изменение распределения признаков)
- Concept drift detection (изменение зависимости target от features)
- Performance drift monitoring (падение метрик)
- Автоматические алерты при обнаружении drift
- Рекомендации по ретренингу модели

Путь: backend/ml_engine/monitoring/drift_detector.py
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import json
from pathlib import Path
from scipy import stats

from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DriftMetrics:
  """Метрики drift detection."""
  timestamp: int

  # Data drift
  feature_drift_score: float  # Средний KS test score
  drifting_features: List[str]  # Признаки с drift

  # Concept drift
  prediction_accuracy: float
  prediction_drift_score: float

  # Performance drift
  recent_accuracy: float
  baseline_accuracy: float
  accuracy_drop: float

  # Overall assessment
  drift_detected: bool
  severity: str  # "none", "low", "medium", "high", "critical"
  recommendation: str


class DriftDetector:
  """
  Детектор model drift.

  Методы детекции:
  - Kolmogorov-Smirnov test для data drift
  - Population Stability Index (PSI) для prediction drift
  - Rolling window accuracy для performance drift
  """

  def __init__(
      self,
      window_size: int = 10000,
      baseline_window_size: int = 50000,
      drift_threshold: float = 0.1,
      accuracy_drop_threshold: float = 0.05,
      check_interval_hours: int = 24
  ):
    """
    Инициализация детектора.

    Args:
        window_size: Размер окна для текущих данных
        baseline_window_size: Размер окна для baseline
        drift_threshold: Порог для KS test (p-value < threshold = drift)
        accuracy_drop_threshold: Порог падения accuracy для алерта
        check_interval_hours: Интервал проверки drift в часах
    """
    self.window_size = window_size
    self.baseline_window_size = baseline_window_size
    self.drift_threshold = drift_threshold
    self.accuracy_drop_threshold = accuracy_drop_threshold
    self.check_interval_hours = check_interval_hours

    # Baseline данные (для сравнения)
    self.baseline_features: Optional[np.ndarray] = None
    self.baseline_predictions: Optional[np.ndarray] = None
    self.baseline_labels: Optional[np.ndarray] = None
    self.baseline_accuracy: Optional[float] = None

    # Текущие данные (скользящее окно)
    self.current_features = deque(maxlen=window_size)
    self.current_predictions = deque(maxlen=window_size)
    self.current_labels = deque(maxlen=window_size)

    # История drift метрик (FIX: Use deque to limit memory - keep last 1000 checks)
    self.drift_history: deque = deque(maxlen=1000)
    self.last_check_time: Optional[datetime] = None

    # Имена признаков
    self.feature_names: Optional[List[str]] = None

    logger.info(
      f"Инициализирован DriftDetector: "
      f"window={window_size}, baseline={baseline_window_size}, "
      f"threshold={drift_threshold}"
    )

  def set_baseline(
      self,
      features: np.ndarray,
      predictions: np.ndarray,
      labels: np.ndarray,
      feature_names: Optional[List[str]] = None
  ):
    """
    Установить baseline данные.

    Args:
        features: (N, feature_dim) массив признаков
        predictions: (N,) предсказания модели
        labels: (N,) истинные метки
        feature_names: Имена признаков
    """
    # Берем последние baseline_window_size семплов
    if len(features) > self.baseline_window_size:
      features = features[-self.baseline_window_size:]
      predictions = predictions[-self.baseline_window_size:]
      labels = labels[-self.baseline_window_size:]

    self.baseline_features = features
    self.baseline_predictions = predictions
    self.baseline_labels = labels

    # Вычисляем baseline accuracy
    self.baseline_accuracy = np.mean(predictions == labels)

    # Сохраняем имена признаков
    if feature_names is not None:
      self.feature_names = feature_names
    else:
      self.feature_names = [f"feature_{i}" for i in range(features.shape[1])]

    logger.info(
      f"Установлен baseline: samples={len(features)}, "
      f"features={features.shape[1]}, "
      f"accuracy={self.baseline_accuracy:.4f}"
    )

  def add_observation(
      self,
      features: np.ndarray,
      prediction: int,
      label: Optional[int] = None
  ):
    """
    Добавить наблюдение в текущее окно.

    Args:
        features: (feature_dim,) вектор признаков
        prediction: Предсказание модели
        label: Истинная метка (если известна)
    """
    self.current_features.append(features)
    self.current_predictions.append(prediction)

    if label is not None:
      self.current_labels.append(label)

  def detect_data_drift(self) -> Tuple[float, List[str]]:
    """
    Обнаружить data drift (изменение распределения признаков).

    Использует Kolmogorov-Smirnov test для каждого признака.

    Returns:
        avg_ks_score: Средний KS statistic
        drifting_features: Список признаков с обнаруженным drift
    """
    if self.baseline_features is None or len(self.current_features) == 0:
      logger.warning("Нет baseline или текущих данных для drift detection")
      return 0.0, []

    current_features_array = np.array(list(self.current_features))

    ks_scores = []
    drifting_features = []

    # Проверяем каждый признак
    for i in range(self.baseline_features.shape[1]):
      baseline_feature = self.baseline_features[:, i]
      current_feature = current_features_array[:, i]

      # Kolmogorov-Smirnov test
      ks_stat, p_value = stats.ks_2samp(baseline_feature, current_feature)

      ks_scores.append(ks_stat)

      # Drift detected если p-value < threshold
      if p_value < self.drift_threshold:
        feature_name = self.feature_names[i] if self.feature_names else f"feature_{i}"
        drifting_features.append(feature_name)

        logger.warning(
          f"Data drift обнаружен: {feature_name}, "
          f"KS={ks_stat:.4f}, p-value={p_value:.4f}"
        )

    avg_ks_score = float(np.mean(ks_scores))

    return avg_ks_score, drifting_features

  def detect_concept_drift(self) -> float:
    """
    Обнаружить concept drift (изменение зависимости).

    Использует Population Stability Index (PSI).

    Returns:
        psi_score: PSI score (> 0.2 = significant drift)
    """
    if (
        self.baseline_predictions is None
        or len(self.current_predictions) == 0
    ):
      return 0.0

    # Вычисляем распределения предсказаний
    baseline_hist, bins = np.histogram(
      self.baseline_predictions,
      bins=10,
      range=(0, 2),  # 0=HOLD, 1=BUY, 2=SELL
      density=True
    )
    current_hist, _ = np.histogram(
      list(self.current_predictions),
      bins=bins,
      density=True
    )

    # Избегаем деления на ноль
    baseline_hist = np.clip(baseline_hist, 1e-10, None)
    current_hist = np.clip(current_hist, 1e-10, None)

    # Population Stability Index
    psi = np.sum(
      (current_hist - baseline_hist) * np.log(current_hist / baseline_hist)
    )

    return psi

  def detect_performance_drift(self) -> Tuple[float, float]:
    """
    Обнаружить performance drift (падение метрик).

    Returns:
        recent_accuracy: Текущая accuracy
        accuracy_drop: Падение относительно baseline
    """
    if (
        self.baseline_accuracy is None
        or len(self.current_predictions) == 0
        or len(self.current_labels) == 0
    ):
      return 0.0, 0.0

    # Вычисляем текущую accuracy
    recent_predictions = np.array(list(self.current_predictions))
    recent_labels = np.array(list(self.current_labels))

    recent_accuracy = float(np.mean(recent_predictions == recent_labels))
    accuracy_drop = self.baseline_accuracy - recent_accuracy

    return recent_accuracy, accuracy_drop

  def should_check_drift(self) -> bool:
    """Проверить нужно ли запускать drift detection."""
    if self.last_check_time is None:
      return True

    time_since_check = datetime.now() - self.last_check_time
    return time_since_check >= timedelta(hours=self.check_interval_hours)

  def check_drift(self) -> Optional[DriftMetrics]:
    """
    Полная проверка drift.

    Returns:
        DriftMetrics или None если проверка не нужна
    """
    if not self.should_check_drift():
      return None

    if len(self.current_features) < self.window_size // 2:
      logger.info("Недостаточно данных для drift detection")
      return None

    logger.info("Запуск drift detection...")

    # Data drift
    feature_drift_score, drifting_features = self.detect_data_drift()

    # Concept drift
    prediction_drift_score = self.detect_concept_drift()

    # Performance drift
    recent_accuracy, accuracy_drop = self.detect_performance_drift()

    # Determine severity
    drift_detected = False
    severity = "none"
    recommendation = "Модель работает нормально"

    if len(drifting_features) > 0 or prediction_drift_score > 0.2:
      drift_detected = True

      if prediction_drift_score > 0.5 or accuracy_drop > self.accuracy_drop_threshold:
        severity = "critical"
        recommendation = "КРИТИЧНО: Требуется немедленный ретренинг модели"
      elif prediction_drift_score > 0.3 or accuracy_drop > self.accuracy_drop_threshold / 2:
        severity = "high"
        recommendation = "Рекомендуется ретренинг модели в ближайшее время"
      elif prediction_drift_score > 0.2:
        severity = "medium"
        recommendation = "Наблюдается drift, запланируйте ретренинг"
      else:
        severity = "low"
        recommendation = "Небольшой drift обнаружен, мониторьте ситуацию"

    # Создаем метрики
    metrics = DriftMetrics(
      timestamp=int(datetime.now().timestamp() * 1000),
      feature_drift_score=feature_drift_score,
      drifting_features=drifting_features,
      prediction_accuracy=recent_accuracy if recent_accuracy > 0 else self.baseline_accuracy,
      prediction_drift_score=prediction_drift_score,
      recent_accuracy=recent_accuracy,
      baseline_accuracy=self.baseline_accuracy if self.baseline_accuracy else 0.0,
      accuracy_drop=accuracy_drop,
      drift_detected=drift_detected,
      severity=severity,
      recommendation=recommendation
    )

    # Сохраняем в историю
    self.drift_history.append(metrics)
    self.last_check_time = datetime.now()

    # Логирование
    if drift_detected:
      logger.warning(
        f"Drift обнаружен! Severity: {severity}\n"
        f"Feature drift: {feature_drift_score:.4f}, "
        f"drifting features: {len(drifting_features)}\n"
        f"Prediction drift: {prediction_drift_score:.4f}\n"
        f"Accuracy drop: {accuracy_drop:.4f}\n"
        f"Recommendation: {recommendation}"
      )
    else:
      logger.info(
        f"Drift не обнаружен. "
        f"Feature drift: {feature_drift_score:.4f}, "
        f"Prediction drift: {prediction_drift_score:.4f}"
      )

    return metrics

  def get_drift_report(self) -> Dict:
    """
    Получить отчет о состоянии drift.

    Returns:
        Dict с информацией о drift
    """
    if not self.drift_history:
      return {
        'status': 'no_checks',
        'message': 'Drift detection еще не запускался'
      }

    latest_metrics = self.drift_history[-1]

    return {
      'status': 'drift_detected' if latest_metrics.drift_detected else 'ok',
      'severity': latest_metrics.severity,
      'recommendation': latest_metrics.recommendation,
      'latest_check': datetime.fromtimestamp(
        latest_metrics.timestamp / 1000
      ).isoformat(),
      'metrics': {
        'feature_drift_score': latest_metrics.feature_drift_score,
        'drifting_features_count': len(latest_metrics.drifting_features),
        'drifting_features': latest_metrics.drifting_features[:10],  # Top 10
        'prediction_drift_score': latest_metrics.prediction_drift_score,
        'recent_accuracy': latest_metrics.recent_accuracy,
        'baseline_accuracy': latest_metrics.baseline_accuracy,
        'accuracy_drop': latest_metrics.accuracy_drop
      },
      'checks_performed': len(self.drift_history)
    }

  def save_drift_history(self, filepath: str):
    """Сохранить историю drift в файл."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    history_data = [
      {
        'timestamp': m.timestamp,
        'feature_drift_score': m.feature_drift_score,
        'drifting_features': m.drifting_features,
        'prediction_drift_score': m.prediction_drift_score,
        'recent_accuracy': m.recent_accuracy,
        'baseline_accuracy': m.baseline_accuracy,
        'accuracy_drop': m.accuracy_drop,
        'drift_detected': m.drift_detected,
        'severity': m.severity,
        'recommendation': m.recommendation
      }
      for m in self.drift_history
    ]

    with open(filepath, 'w') as f:
      json.dump(history_data, f, indent=2)

    logger.info(f"Drift history сохранена: {filepath}")


# Пример использования
if __name__ == "__main__":
  # Создаем детектор
  detector = DriftDetector(
    window_size=1000,
    baseline_window_size=5000,
    drift_threshold=0.1,
    accuracy_drop_threshold=0.05
  )

  # Генерируем baseline данные
  np.random.seed(42)
  baseline_features = np.random.randn(5000, 110)
  baseline_predictions = np.random.randint(0, 3, 5000)
  baseline_labels = np.random.randint(0, 3, 5000)

  detector.set_baseline(
    baseline_features,
    baseline_predictions,
    baseline_labels
  )

  # Добавляем текущие наблюдения (с небольшим drift)
  for _ in range(1000):
    features = np.random.randn(110) + 0.2  # Небольшой shift
    prediction = np.random.randint(0, 3)
    label = np.random.randint(0, 3)

    detector.add_observation(features, prediction, label)

  # Проверяем drift
  metrics = detector.check_drift()

  if metrics:
    print(f"Drift detected: {metrics.drift_detected}")
    print(f"Severity: {metrics.severity}")
    print(f"Recommendation: {metrics.recommendation}")

  # Получаем отчет
  report = detector.get_drift_report()


# ===================================================================
# Singleton Pattern для DriftDetector
# ===================================================================

_drift_detector_instance: Optional[DriftDetector] = None


def get_drift_detector(
    window_size: int = 10000,
    baseline_window_size: int = 50000,
    drift_threshold: float = 0.1,
    accuracy_drop_threshold: float = 0.05,
    check_interval_hours: int = 24
) -> DriftDetector:
  """
  Получить singleton instance DriftDetector.

  Args:
      window_size: Размер окна для текущих данных
      baseline_window_size: Размер окна для baseline
      drift_threshold: Порог для KS test
      accuracy_drop_threshold: Порог падения accuracy
      check_interval_hours: Интервал проверки drift

  Returns:
      DriftDetector instance
  """
  global _drift_detector_instance

  if _drift_detector_instance is None:
    _drift_detector_instance = DriftDetector(
        window_size=window_size,
        baseline_window_size=baseline_window_size,
        drift_threshold=drift_threshold,
        accuracy_drop_threshold=accuracy_drop_threshold,
        check_interval_hours=check_interval_hours
    )
    logger.info("✓ Created DriftDetector singleton instance")

  return _drift_detector_instance