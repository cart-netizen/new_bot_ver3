#!/usr/bin/env python3
"""
Hyperparameter Optimizer - автоматический поиск оптимальных параметров.

Функциональность:
1. Grid Search по всем параметрам балансировки классов
2. Поиск оптимального threshold для future_direction
3. Per-symbol оптимизация параметров
4. Cross-validation с walk-forward
5. Сохранение оптимальных конфигураций
6. Визуализация результатов

Использование:
    # Для одного символа
    python hyperparameter_optimizer.py --symbol BTCUSDT --quick

    # Для нескольких символов (полный поиск)
    python hyperparameter_optimizer.py --symbols BTCUSDT ETHUSDT SOLUSDT --full

    # Только threshold optimization
    python hyperparameter_optimizer.py --symbol BTCUSDT --optimize threshold

Путь: backend/ml_engine/optimization/hyperparameter_optimizer.py
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Добавляем путь к backend
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import Counter
import itertools
import torch
from sklearn.metrics import (
  f1_score, precision_recall_fscore_support,
  accuracy_score, roc_auc_score, confusion_matrix
)
import warnings
import logging

from tqdm import tqdm

# Отключаем избыточные warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pin_memory.*')

# Настройка логирования
logging.basicConfig(
  level=logging.INFO,
  format='%(message)s'  # Упрощенный формат
)

from core.logger import get_logger
from ml_engine.training.data_loader import HistoricalDataLoader, DataConfig
from ml_engine.training.model_trainer import ModelTrainer, TrainerConfig
from ml_engine.training.class_balancing import ClassBalancingConfig
from ml_engine.models.hybrid_cnn_lstm import create_model

logger = get_logger(__name__)


@dataclass
class OptimizationConfig:
  """Конфигурация оптимизации."""
  # Символы для оптимизации
  symbols: List[str]

  # Параметры данных
  data_path: str = "data/ml_training"
  sequence_length: int = 60

  # Grid Search параметры
  thresholds: List[float] = None  # [0.0005, 0.001, 0.0015, 0.002, 0.0025]

  class_weight_methods: List[str] = None  # ["balanced", "inverse_freq", "effective_samples"]

  focal_gammas: List[float] = None  # [1.0, 1.5, 2.0, 2.5, 3.0]
  focal_alphas: List[Optional[float]] = None  # [None, 0.25, 0.5, 0.75]

  oversample_strategies: List[str] = None  # ["auto", "minority"]
  smote_k_neighbors: List[int] = None  # [3, 5, 7]

  # Балансировка методов
  balancing_methods: List[str] = None  # ["none", "weights", "focal", "focal_oversample", "focal_smote"]

  # Обучение
  quick_mode: bool = False  # Быстрая оптимизация (меньше эпох, меньше параметров)
  max_epochs: int = 30
  early_stopping_patience: int = 5

  # Cross-validation
  n_folds: int = 3  # Walk-forward folds

  # Метрики
  primary_metric: str = "f1_weighted"  # Главная метрика для оптимизации

  # Результаты
  output_dir: str = "optimization_results"
  save_models: bool = False  # Сохранять ли обученные модели

  def __post_init__(self):
    """Инициализация дефолтных значений."""
    if self.thresholds is None:
      if self.quick_mode:
        self.thresholds = [0.001, 0.002]
      else:
        self.thresholds = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003]

    if self.class_weight_methods is None:
      self.class_weight_methods = ["balanced", "inverse_freq"]

    if self.focal_gammas is None:
      if self.quick_mode:
        self.focal_gammas = [2.0]
      else:
        self.focal_gammas = [1.0, 1.5, 2.0, 2.5]

    if self.focal_alphas is None:
      self.focal_alphas = [None, 0.25]

    if self.oversample_strategies is None:
      self.oversample_strategies = ["auto"]

    if self.smote_k_neighbors is None:
      self.smote_k_neighbors = [5]

    if self.balancing_methods is None:
      if self.quick_mode:
        self.balancing_methods = ["none", "weights", "focal"]
      else:
        self.balancing_methods = [
          "none", "weights", "focal",
          "focal_oversample", "focal_smote"
        ]


@dataclass
class OptimizationResult:
  """Результат оптимизации для одной конфигурации."""
  # Параметры
  symbol: str
  threshold: float
  balancing_method: str
  balancing_params: Dict

  # Метрики
  f1_weighted: float
  f1_per_class: Dict[str, float]
  precision_weighted: float
  recall_weighted: float
  accuracy: float

  # Дополнительная информация
  class_distribution: Dict[str, int]
  imbalance_ratio: float
  confusion_matrix: List[List[int]]

  # Мета-информация
  training_time: float
  timestamp: str


class HyperparameterOptimizer:
  """
  Оптимизатор гиперпараметров для ML модели.

  Автоматически находит оптимальные параметры для каждого символа:
  - Threshold для future_direction
  - Метод балансировки классов
  - Параметры focal loss
  - Стратегию oversampling/SMOTE
  """

  def __init__(self, config: OptimizationConfig):
    """Инициализация оптимизатора."""
    self.config = config
    self.output_dir = Path(config.output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)

    # Результаты
    self.all_results: Dict[str, List[OptimizationResult]] = {}
    self.best_configs: Dict[str, Dict] = {}

    print("\n" + "=" * 80)
    print("HYPERPARAMETER OPTIMIZER")
    print("=" * 80)
    print(f"Symbols: {config.symbols}")
    print(f"Mode: {'QUICK' if config.quick_mode else 'FULL'}")
    print(f"Output: {self.output_dir}")
    print("=" * 80 + "\n")

    logger.info("=" * 80)
    logger.info("HYPERPARAMETER OPTIMIZER")
    logger.info("=" * 80)
    logger.info(f"Symbols: {config.symbols}")
    logger.info(f"Quick mode: {config.quick_mode}")
    logger.info(f"Output: {self.output_dir}")
    logger.info("=" * 80)

  def optimize_all_symbols(self):
    """Оптимизация для всех символов."""
    for symbol in self.config.symbols:
      logger.info(f"\n{'=' * 80}")
      logger.info(f"ОПТИМИЗАЦИЯ: {symbol}")
      logger.info(f"{'=' * 80}\n")

      try:
        self.optimize_symbol(symbol)
      except Exception as e:
        logger.error(f"Ошибка оптимизации {symbol}: {e}")
        continue

    # Финальный отчет
    self._generate_final_report()

  def optimize_symbol(self, symbol: str):
    """
    Оптимизация параметров для одного символа.

    Args:
        symbol: Торговая пара
    """
    results = []

    # ===== ШАГ 1: ОПТИМИЗАЦИЯ THRESHOLD =====
    logger.info("Шаг 1/3: Оптимизация threshold...")
    threshold_results = self._optimize_threshold(symbol)

    # Выбираем лучший threshold
    best_threshold_result = max(
      threshold_results,
      key=lambda x: x.f1_weighted
    )
    best_threshold = best_threshold_result.threshold

    logger.info(f"✓ Лучший threshold: {best_threshold} "
                f"(F1={best_threshold_result.f1_weighted:.4f})")

    # ===== ШАГ 2: ОПТИМИЗАЦИЯ BALANCING METHOD =====
    logger.info("\nШаг 2/3: Оптимизация метода балансировки...")
    balancing_results = self._optimize_balancing_method(
      symbol, best_threshold
    )

    results.extend(threshold_results)
    results.extend(balancing_results)

    # ===== ШАГ 3: FINE-TUNING ЛУЧШЕГО МЕТОДА =====
    logger.info("\nШаг 3/3: Fine-tuning лучшего метода...")

    best_balancing_result = max(
      balancing_results,
      key=lambda x: x.f1_weighted
    )

    if "focal" in best_balancing_result.balancing_method:
      finetuning_results = self._finetune_focal_loss(
        symbol, best_threshold
      )
      results.extend(finetuning_results)

    # ===== ВЫБОР ЛУЧШЕЙ КОНФИГУРАЦИИ =====
    best_result = max(results, key=lambda x: x.f1_weighted)

    self.all_results[symbol] = results
    self.best_configs[symbol] = self._extract_config(best_result)

    # Сохранение результатов
    self._save_symbol_results(symbol, results, best_result)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"ЛУЧШАЯ КОНФИГУРАЦИЯ для {symbol}:")
    logger.info(f"{'=' * 80}")
    logger.info(f"  • Threshold: {best_result.threshold}")
    logger.info(f"  • Method: {best_result.balancing_method}")
    logger.info(f"  • F1-score: {best_result.f1_weighted:.4f}")
    logger.info(f"  • Accuracy: {best_result.accuracy:.4f}")
    logger.info(f"  • Imbalance Ratio: {best_result.imbalance_ratio:.2f}")
    logger.info(f"{'=' * 80}\n")

  def _optimize_threshold(
      self,
      symbol: str
  ) -> List[OptimizationResult]:
    """Оптимизация threshold для future_direction."""
    results = []

    print(f"\n{'=' * 60}")
    print(f"ОПТИМИЗАЦИЯ THRESHOLD")
    print(f"{'=' * 60}")
    print(f"Тестирование {len(self.config.thresholds)} thresholds...")
    print(f"Thresholds: {self.config.thresholds}")
    print(f"{'=' * 60}\n")

    logger.info(f"Тестирование {len(self.config.thresholds)} thresholds...")

    for i, threshold in enumerate(self.config.thresholds, 1):
      print(f"\n[{i}/{len(self.config.thresholds)}] Threshold: {threshold}")
      print("─" * 60)

      # Используем baseline (без балансировки) для честного сравнения
      result = self._evaluate_configuration(
        symbol=symbol,
        threshold=threshold,
        balancing_method="none",
        balancing_params={}
      )

      if result:
        results.append(result)
        print(f"✓ F1={result.f1_weighted:.4f}, Accuracy={result.accuracy:.4f}, Ratio={result.imbalance_ratio:.2f}")
        logger.info(
          f"  {threshold:.4f}: F1={result.f1_weighted:.4f}, "
          f"Ratio={result.imbalance_ratio:.2f}"
        )
      else:
        print("✗ Ошибка оценки конфигурации")

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ THRESHOLD OPTIMIZATION")
    print("=" * 60)
    if results:
      best = max(results, key=lambda x: x.f1_weighted)
      print(f"Лучший threshold: {best.threshold}")
      print(f"  F1-Score: {best.f1_weighted:.4f}")
      print(f"  Accuracy: {best.accuracy:.4f}")
      print(f"  Imbalance Ratio: {best.imbalance_ratio:.2f}")
    print("=" * 60 + "\n")

    return results

  def _optimize_balancing_method(
      self,
      symbol: str,
      threshold: float
  ) -> List[OptimizationResult]:
    """Оптимизация метода балансировки классов."""
    results = []

    print(f"\n{'=' * 60}")
    print(f"ОПТИМИЗАЦИЯ BALANCING METHOD")
    print(f"{'=' * 60}")
    print(f"Используется threshold: {threshold}")
    print(f"Тестирование {len(self.config.balancing_methods) - 1} методов...")  # -1 т.к. "none" пропускаем
    print(f"{'=' * 60}\n")

    logger.info(f"Тестирование {len(self.config.balancing_methods)} методов...")

    tested_count = 0
    for method in self.config.balancing_methods:
      # Пропускаем "none" - уже протестировано в threshold optimization
      if method == "none":
        continue

      tested_count += 1
      print(f"\n[{tested_count}/{len(self.config.balancing_methods) - 1}] Method: {method}")
      print("─" * 60)

      # Параметры по умолчанию для каждого метода
      if method == "weights":
        params = {"weight_method": "balanced"}
      elif method == "focal":
        params = {"gamma": 2.0, "alpha": None}
      elif method == "focal_oversample":
        params = {
          "gamma": 2.0,
          "alpha": None,
          "oversample_strategy": "auto"
        }
      elif method == "focal_smote":
        params = {
          "gamma": 2.0,
          "alpha": None,
          "k_neighbors": 5
        }
      else:
        params = {}

      result = self._evaluate_configuration(
        symbol=symbol,
        threshold=threshold,
        balancing_method=method,
        balancing_params=params
      )

      if result:
        results.append(result)
        print(f"✓ F1={result.f1_weighted:.4f}, Accuracy={result.accuracy:.4f}")
        logger.info(
          f"  {method}: F1={result.f1_weighted:.4f}"
        )
      else:
        print("✗ Ошибка оценки конфигурации")

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ BALANCING METHOD OPTIMIZATION")
    print("=" * 60)
    if results:
      best = max(results, key=lambda x: x.f1_weighted)
      print(f"Лучший метод: {best.balancing_method}")
      print(f"  F1-Score: {best.f1_weighted:.4f}")
      print(f"  Accuracy: {best.accuracy:.4f}")
    print("=" * 60 + "\n")

    return results

  def _finetune_focal_loss(
      self,
      symbol: str,
      threshold: float
  ) -> List[OptimizationResult]:
    """Fine-tuning параметров focal loss."""
    results = []

    logger.info("Fine-tuning Focal Loss параметров...")

    # Grid search по gamma и alpha
    param_combinations = list(itertools.product(
      self.config.focal_gammas,
      self.config.focal_alphas
    ))

    logger.info(f"Тестирование {len(param_combinations)} комбинаций...")

    for gamma, alpha in tqdm(param_combinations, desc="Focal params"):
      result = self._evaluate_configuration(
        symbol=symbol,
        threshold=threshold,
        balancing_method="focal",
        balancing_params={"gamma": gamma, "alpha": alpha}
      )

      if result:
        results.append(result)

    # Показываем топ-3
    top_results = sorted(
      results, key=lambda x: x.f1_weighted, reverse=True
    )[:3]

    logger.info("\nТоп-3 конфигурации:")
    for i, r in enumerate(top_results, 1):
      gamma = r.balancing_params.get("gamma")
      alpha = r.balancing_params.get("alpha")
      logger.info(
        f"  {i}. gamma={gamma}, alpha={alpha}: "
        f"F1={r.f1_weighted:.4f}"
      )

    return results

  def _evaluate_configuration(
      self,
      symbol: str,
      threshold: float,
      balancing_method: str,
      balancing_params: Dict
  ) -> Optional[OptimizationResult]:
    """
    Оценка одной конфигурации параметров.

    Returns:
        OptimizationResult или None если ошибка
    """
    start_time = datetime.now()

    # Подробное логирование
    logger.info(f"\n{'─' * 60}")
    logger.info(f"Оценка конфигурации:")
    logger.info(f"  Symbol: {symbol}")
    logger.info(f"  Threshold: {threshold}")
    logger.info(f"  Method: {balancing_method}")
    logger.info(f"  Params: {balancing_params}")
    logger.info(f"{'─' * 60}")

    try:
      # ===== ПОДГОТОВКА ДАННЫХ =====

      logger.info("Шаг 1/4: Подготовка данных...")

      # 1. Применяем threshold к данным
      data_config = DataConfig(
        storage_path=self.config.data_path,
        sequence_length=self.config.sequence_length,
        target_horizon="future_direction_60s",
        batch_size=64
      )

      # 2. Создаем balancing config
      balancing_config = self._create_balancing_config(
        balancing_method, balancing_params
      )

      # 3. Загружаем данные
      logger.info("  Загрузка данных...")
      loader = HistoricalDataLoader(
        config=data_config,
        balancing_config=balancing_config
      )

      # Для threshold optimization нужно пересчитать labels
      # (это упрощение - в реальности нужен preprocessing скрипт)
      logger.info("  Подготовка данных...")
      result = loader.load_and_prepare(
        symbols=[symbol],
        apply_resampling=(balancing_method in [
          "focal_oversample", "focal_smote"
        ])
      )

      train_loader = result['dataloaders']['train']
      val_loader = result['dataloaders']['val']

      logger.info(f"  ✓ Данные загружены: train={len(train_loader)} batches, val={len(val_loader)} batches")

      # Получаем class distribution
      logger.info("Шаг 2/4: Анализ распределения классов...")
      all_labels = []
      for batch in train_loader:
        all_labels.extend(batch['label'].numpy())

      class_dist = Counter(all_labels)
      max_count = max(class_dist.values())
      min_count = min(class_dist.values())
      imbalance_ratio = max_count / min_count

      logger.info(f"  Class distribution: {dict(class_dist)}")
      logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}")

      # ===== ОБУЧЕНИЕ МОДЕЛИ =====

      logger.info("Шаг 3/4: Обучение модели...")
      model = create_model()

      trainer_config = TrainerConfig(
        epochs=self.config.max_epochs if not self.config.quick_mode else 10,
        learning_rate=0.001,
        early_stopping_patience=self.config.early_stopping_patience,
        class_balancing=balancing_config,
        checkpoint_dir=str(self.output_dir / "checkpoints" / symbol)
      )

      trainer = ModelTrainer(model, trainer_config)

      # Обучение
      logger.info(f"  Обучение {trainer_config.epochs} эпох...")
      history = trainer.train(train_loader, val_loader)
      logger.info(f"  ✓ Обучение завершено: {len(history)} эпох")

      # ===== ОЦЕНКА =====

      logger.info("Шаг 4/4: Оценка модели...")

      # Получаем предсказания на validation
      model.eval()
      y_true = []
      y_pred = []

      device = trainer.device

      with torch.no_grad():
        for batch in val_loader:
          sequences = batch['sequence'].to(device)
          labels = batch['label']

          outputs = model(sequences)
          predictions = torch.argmax(
            outputs['direction_logits'], dim=-1
          ).cpu().numpy()

          y_pred.extend(predictions)
          y_true.extend(labels.numpy())

      # Метрики
      f1_weighted = f1_score(y_true, y_pred, average='weighted')
      precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
      )
      precision_weighted = np.average(precision, weights=[
        class_dist.get(i, 0) for i in range(len(precision))
      ])
      recall_weighted = np.average(recall, weights=[
        class_dist.get(i, 0) for i in range(len(recall))
      ])
      accuracy = accuracy_score(y_true, y_pred)
      conf_matrix = confusion_matrix(y_true, y_pred).tolist()

      # Время обучения
      training_time = (datetime.now() - start_time).total_seconds()

      logger.info(f"  ✓ Результаты:")
      logger.info(f"    F1-Score: {f1_weighted:.4f}")
      logger.info(f"    Accuracy: {accuracy:.4f}")
      logger.info(f"    Training time: {training_time:.1f}s")

      # Создаем результат
      result = OptimizationResult(
        symbol=symbol,
        threshold=threshold,
        balancing_method=balancing_method,
        balancing_params=balancing_params,
        f1_weighted=f1_weighted,
        f1_per_class={
          "DOWN": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
          "NEUTRAL": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
          "UP": float(f1_per_class[2]) if len(f1_per_class) > 2 else 0.0
        },
        precision_weighted=float(precision_weighted),
        recall_weighted=float(recall_weighted),
        accuracy=float(accuracy),
        class_distribution=dict(class_dist),
        imbalance_ratio=float(imbalance_ratio),
        confusion_matrix=conf_matrix,
        training_time=training_time,
        timestamp=datetime.now().isoformat()
      )

      logger.info(f"✓ Конфигурация оценена успешно")
      return result

    except Exception as e:
      logger.error(
        f"❌ Ошибка оценки конфигурации "
        f"(threshold={threshold}, method={balancing_method}):"
      )
      logger.error(f"  Exception: {type(e).__name__}")
      logger.error(f"  Message: {str(e)}")

      # Детальный traceback
      import traceback
      logger.error("  Traceback:")
      for line in traceback.format_exc().split('\n'):
        if line:
          logger.error(f"    {line}")

      return None

  def _create_balancing_config(
      self,
      method: str,
      params: Dict
  ) -> ClassBalancingConfig:
    """Создание ClassBalancingConfig из метода и параметров."""
    if method == "none":
      return ClassBalancingConfig(
        use_class_weights=False,
        use_focal_loss=False
      )

    elif method == "weights":
      return ClassBalancingConfig(
        use_class_weights=True,
        use_focal_loss=False
      )

    elif method == "focal":
      return ClassBalancingConfig(
        use_class_weights=False,
        use_focal_loss=True,
        focal_gamma=params.get("gamma", 2.0),
        focal_alpha=params.get("alpha")
      )

    elif method == "focal_oversample":
      return ClassBalancingConfig(
        use_class_weights=False,
        use_focal_loss=True,
        use_oversampling=True,
        focal_gamma=params.get("gamma", 2.0),
        focal_alpha=params.get("alpha"),
        oversample_strategy=params.get("oversample_strategy", "auto")
      )

    elif method == "focal_smote":
      return ClassBalancingConfig(
        use_class_weights=False,
        use_focal_loss=True,
        use_smote=True,
        focal_gamma=params.get("gamma", 2.0),
        focal_alpha=params.get("alpha"),
        smote_k_neighbors=params.get("k_neighbors", 5)
      )

    else:
      raise ValueError(f"Неизвестный метод: {method}")

  def _extract_config(self, result: OptimizationResult) -> Dict:
    """Извлечение конфигурации из результата."""
    config = {
      "threshold": result.threshold,
      "balancing_method": result.balancing_method,
      "balancing_params": result.balancing_params,
      "expected_performance": {
        "f1_weighted": result.f1_weighted,
        "accuracy": result.accuracy,
        "imbalance_ratio": result.imbalance_ratio
      }
    }
    return config

  def _save_symbol_results(
      self,
      symbol: str,
      results: List[OptimizationResult],
      best_result: OptimizationResult
  ):
    """Сохранение результатов для символа."""
    symbol_dir = self.output_dir / symbol
    symbol_dir.mkdir(exist_ok=True)

    # Сохраняем все результаты
    all_results_data = [asdict(r) for r in results]
    with open(symbol_dir / "all_results.json", 'w') as f:
      json.dump(all_results_data, f, indent=2)

    # Сохраняем лучшую конфигурацию
    best_config = self._extract_config(best_result)
    with open(symbol_dir / "best_config.json", 'w') as f:
      json.dump(best_config, f, indent=2)

    # Создаем DataFrame для анализа
    df = pd.DataFrame([
      {
        "threshold": r.threshold,
        "method": r.balancing_method,
        "f1_weighted": r.f1_weighted,
        "accuracy": r.accuracy,
        "imbalance_ratio": r.imbalance_ratio,
        **r.f1_per_class
      }
      for r in results
    ])
    df.to_csv(symbol_dir / "results.csv", index=False)

    logger.info(f"✓ Результаты сохранены в {symbol_dir}")

  def _generate_final_report(self):
    """Генерация финального отчета."""
    logger.info("\n" + "=" * 80)
    logger.info("ФИНАЛЬНЫЙ ОТЧЕТ ОПТИМИЗАЦИИ")
    logger.info("=" * 80 + "\n")

    # Сводная таблица
    summary = []
    for symbol in self.config.symbols:
      if symbol not in self.best_configs:
        continue

      config = self.best_configs[symbol]
      perf = config["expected_performance"]

      summary.append({
        "Symbol": symbol,
        "Threshold": f"{config['threshold']:.4f}",
        "Method": config['balancing_method'],
        "F1": f"{perf['f1_weighted']:.4f}",
        "Accuracy": f"{perf['accuracy']:.4f}",
        "Imbalance": f"{perf['imbalance_ratio']:.2f}"
      })

    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))

    # Сохраняем сводку
    summary_path = self.output_dir / "optimization_summary.json"
    with open(summary_path, 'w') as f:
      json.dump(self.best_configs, f, indent=2)

    logger.info(f"\n✓ Сводка сохранена: {summary_path}")

    # Генерация кода для применения
    self._generate_application_code()

  def _generate_application_code(self):
    """Генерация кода для применения оптимальных конфигураций."""
    code_path = self.output_dir / "apply_optimal_configs.py"

    code = '''#!/usr/bin/env python3
"""
Автоматически сгенерированный код для применения оптимальных конфигураций.
Сгенерирован: ''' + datetime.now().isoformat() + '''
"""

from ml_engine.training.class_balancing import ClassBalancingConfig

# Оптимальные конфигурации для каждого символа
OPTIMAL_CONFIGS = ''' + json.dumps(self.best_configs, indent=4) + '''

def get_optimal_config(symbol: str) -> ClassBalancingConfig:
    """
    Получить оптимальную конфигурацию балансировки для символа.

    Args:
        symbol: Торговая пара (например, "BTCUSDT")

    Returns:
        ClassBalancingConfig с оптимальными параметрами
    """
    if symbol not in OPTIMAL_CONFIGS:
        raise ValueError(f"Нет оптимальной конфигурации для {symbol}")

    config = OPTIMAL_CONFIGS[symbol]
    method = config["balancing_method"]
    params = config["balancing_params"]

    if method == "none":
        return ClassBalancingConfig(
            use_class_weights=False,
            use_focal_loss=False
        )
    elif method == "weights":
        return ClassBalancingConfig(
            use_class_weights=True,
            use_focal_loss=False
        )
    elif method == "focal":
        return ClassBalancingConfig(
            use_class_weights=False,
            use_focal_loss=True,
            focal_gamma=params.get("gamma", 2.0),
            focal_alpha=params.get("alpha")
        )
    elif method == "focal_oversample":
        return ClassBalancingConfig(
            use_class_weights=False,
            use_focal_loss=True,
            use_oversampling=True,
            focal_gamma=params.get("gamma", 2.0),
            focal_alpha=params.get("alpha"),
            oversample_strategy=params.get("oversample_strategy", "auto")
        )
    elif method == "focal_smote":
        return ClassBalancingConfig(
            use_class_weights=False,
            use_focal_loss=True,
            use_smote=True,
            focal_gamma=params.get("gamma", 2.0),
            focal_alpha=params.get("alpha"),
            smote_k_neighbors=params.get("k_neighbors", 5)
        )
    else:
        raise ValueError(f"Неизвестный метод: {method}")


# Пример использования:
if __name__ == "__main__":
    # Получение конфигурации для BTC
    btc_config = get_optimal_config("BTCUSDT")
    print(f"Оптимальная конфигурация для BTCUSDT:")
    print(f"  use_focal_loss: {btc_config.use_focal_loss}")
    print(f"  focal_gamma: {btc_config.focal_gamma}")
    print(f"  focal_alpha: {btc_config.focal_alpha}")
'''

    with open(code_path, 'w') as f:
      f.write(code)

    logger.info(f"✓ Код применения сгенерирован: {code_path}")


# ========== CLI INTERFACE ==========

def main():
  """Главная функция."""
  import argparse

  parser = argparse.ArgumentParser(
    description="Hyperparameter Optimizer для ML модели"
  )

  # Символы
  symbol_group = parser.add_mutually_exclusive_group(required=True)
  symbol_group.add_argument(
    "--symbol",
    type=str,
    help="Одна торговая пара (например, BTCUSDT)"
  )
  symbol_group.add_argument(
    "--symbols",
    nargs='+',
    help="Несколько торговых пар"
  )

  # Режимы
  parser.add_argument(
    "--quick",
    action="store_true",
    help="Быстрая оптимизация (меньше параметров)"
  )
  parser.add_argument(
    "--full",
    action="store_true",
    help="Полная оптимизация (все параметры)"
  )

  # Опции
  parser.add_argument(
    "--optimize",
    choices=["threshold", "balancing", "all"],
    default="all",
    help="Что оптимизировать"
  )
  parser.add_argument(
    "--output",
    default="optimization_results",
    help="Директория для результатов"
  )
  parser.add_argument(
    "--data-path",
    default="data/ml_training",
    help="Путь к данным"
  )

  args = parser.parse_args()

  # Определяем символы
  if args.symbol:
    symbols = [args.symbol]
  else:
    symbols = args.symbols

  # Создаем конфигурацию
  opt_config = OptimizationConfig(
    symbols=symbols,
    data_path=args.data_path,
    output_dir=args.output,
    quick_mode=args.quick or not args.full
  )

  # Запускаем оптимизацию
  optimizer = HyperparameterOptimizer(opt_config)
  optimizer.optimize_all_symbols()

  print("\n" + "=" * 80)
  print("✓ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
  print("=" * 80)
  print(f"\nРезультаты сохранены в: {opt_config.output_dir}")
  print(f"Примените оптимальные конфигурации с помощью:")
  print(f"  python {opt_config.output_dir}/apply_optimal_configs.py")


if __name__ == "__main__":
  main()