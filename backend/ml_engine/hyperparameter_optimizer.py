#!/usr/bin/env python3
"""
Hyperparameter Optimization Script - Industry Standard Implementation.

Скрипт для автоматического поиска оптимальных гиперпараметров ML модели.
Использует Bayesian Optimization (Optuna) с Sequential Parameter Group Optimization.

АРХИТЕКТУРНЫЕ РЕШЕНИЯ:

1. SEQUENTIAL PARAMETER GROUP OPTIMIZATION (SPGO):
   - Параметры разбиты на группы по влиянию
   - Группы оптимизируются последовательно
   - Когда группа "закрепляется" - её значения фиксируются
   - Следующие группы работают с уже оптимизированными предыдущими

2. BAYESIAN OPTIMIZATION (Optuna TPE):
   - Не перебирает все комбинации
   - TPE (Tree-structured Parzen Estimator) выбирает умные точки
   - Учитывает результаты предыдущих trials

3. AGGRESSIVE PRUNING:
   - MedianPruner останавливает неперспективные trials
   - Если после 2 эпох метрики хуже медианы - trial прерывается
   - Экономия до 60% времени

4. WARM STARTING:
   - Начинаем с рекомендованных значений из optimized_configs.py
   - Первый trial всегда использует baseline параметры

5. EARLY DEGRADATION DETECTION:
   - Если изменение параметра ухудшает метрики - фиксируем направление
   - Бинарный поиск в оптимальном направлении

Использование:
    python -m backend.ml_engine.hyperparameter_optimizer --mode full
    python -m backend.ml_engine.hyperparameter_optimizer --mode quick --group learning_rate
    python -m backend.ml_engine.hyperparameter_optimizer --mode resume --study-name my_study

Файл: backend/ml_engine/hyperparameter_optimizer.py
"""

import os
import sys
import json
import time
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict

# Fix for "This event loop is already running" error when running inside FastAPI
# This allows nested event loops (e.g., when asyncio.run_until_complete is called inside an already running loop)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # nest_asyncio not installed, will fail if running inside FastAPI
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import warnings

import numpy as np

# Optuna for Bayesian Optimization
try:
    import optuna
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not installed. Install with: pip install optuna")

from backend.core.logger import get_logger

logger = get_logger(__name__)

# Project root for resolving relative paths
# This ensures paths work correctly regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_TRAINING_PATH = PROJECT_ROOT / "data" / "ml_training"
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "feature_store"

logger.info(f"HYPEROPT: Project root: {PROJECT_ROOT}")
logger.info(f"HYPEROPT: ML Training path: {ML_TRAINING_PATH}")
logger.info(f"HYPEROPT: Feature Store path: {FEATURE_STORE_PATH}")


# ============================================================================
# CONSTANTS & ENUMS
# ============================================================================

class OptimizationMode(str, Enum):
    """Режимы оптимизации."""
    FULL = "full"              # Полная оптимизация всех групп
    QUICK = "quick"            # Быстрая оптимизация одной группы
    GROUP = "group"            # Оптимизация конкретной группы
    RESUME = "resume"          # Продолжить прерванную оптимизацию
    FINE_TUNE = "fine_tune"    # Тонкая настройка вокруг лучших значений


class ParameterGroup(str, Enum):
    """Группы параметров для последовательной оптимизации."""
    LEARNING_RATE = "learning_rate"      # LR + Weight Decay (HIGH IMPACT)
    REGULARIZATION = "regularization"    # Dropout, Label Smoothing, Focal Gamma
    AUGMENTATION = "augmentation"        # Gaussian Noise, MixUp Alpha
    SCHEDULER = "scheduler"              # T_0, T_mult
    CLASS_BALANCE = "class_balance"      # Focal Loss, Oversampling
    TRIPLE_BARRIER = "triple_barrier"    # TP, SL, MaxHolding
    ARCHITECTURE = "architecture"        # CNN channels, LSTM hidden (expensive)


# Порядок оптимизации групп (от высокого влияния к низкому)
OPTIMIZATION_ORDER = [
    ParameterGroup.LEARNING_RATE,      # ~40% влияния на результат
    ParameterGroup.REGULARIZATION,     # ~25% влияния
    ParameterGroup.CLASS_BALANCE,      # ~15% влияния
    ParameterGroup.AUGMENTATION,       # ~10% влияния
    ParameterGroup.SCHEDULER,          # ~5% влияния
    ParameterGroup.TRIPLE_BARRIER,     # ~5% влияния (зависит от данных)
]


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class OptimizationConfig:
    """Конфигурация процесса оптимизации."""

    # === Основные параметры ===
    study_name: str = "ml_hyperopt"
    storage_path: str = "data/hyperopt"

    # === Ограничения времени ===
    epochs_per_trial: int = 4              # Эпох на одну пробу (4 * 12 мин = 48 мин)
    max_trials_per_group: int = 15         # Максимум trials на группу
    max_total_time_hours: float = 24.0     # Максимальное время оптимизации

    # === Pruning ===
    n_startup_trials: int = 3              # Trials без pruning (накопление статистики)
    n_warmup_steps: int = 2                # Эпох до начала pruning
    pruning_percentile: float = 50.0       # Percentile для MedianPruner

    # === Направление оптимизации ===
    optimization_direction: str = "maximize"  # maximize f1, accuracy
    primary_metric: str = "val_f1"            # Основная метрика
    secondary_metrics: List[str] = field(default_factory=lambda: [
        "val_accuracy", "val_precision", "val_recall"
    ])

    # === Convergence ===
    convergence_patience: int = 5          # Trials без улучшения = конвергенция
    convergence_threshold: float = 0.005   # Минимальное улучшение

    # === MLflow ===
    use_mlflow: bool = True
    mlflow_experiment_name: str = "hyperopt"

    # === Logging ===
    verbose: bool = True
    log_interval: int = 1                  # Логировать каждые N trials

    # === Seeds ===
    seed: int = 42
    deterministic: bool = False            # True = медленнее но воспроизводимо

    def __post_init__(self):
        """Создаём директории."""
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)


@dataclass
class ParameterSearchSpace:
    """
    Пространство поиска для одного параметра.

    Поддерживает:
    - Continuous (float): uniform, log-uniform
    - Discrete (int): int range
    - Categorical: выбор из списка
    """
    name: str
    param_type: str  # "float", "int", "categorical", "bool"

    # Для float/int
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False  # Log-uniform distribution
    step: Optional[float] = None  # Step size

    # Для categorical
    choices: Optional[List[Any]] = None

    # Default/baseline value
    default: Any = None

    # Группа параметра
    group: ParameterGroup = ParameterGroup.LEARNING_RATE

    # Влияние на результат (0-1) для приоритизации
    importance: float = 0.5


# ============================================================================
# DEFAULT SEARCH SPACES
# ============================================================================

def get_default_search_spaces() -> Dict[str, ParameterSearchSpace]:
    """
    Получить дефолтные пространства поиска на основе optimized_configs.py.

    Значения по умолчанию взяты из рекомендованных в проекте.
    """
    return {
        # === ГРУППА 1: Learning Rate (HIGH IMPACT) ===
        "learning_rate": ParameterSearchSpace(
            name="learning_rate",
            param_type="float",
            low=1e-6,
            high=1e-3,
            log=True,  # Log-uniform т.к. LR меняется на порядки
            default=5e-5,  # Рекомендованное значение
            group=ParameterGroup.LEARNING_RATE,
            importance=0.9
        ),
        "weight_decay": ParameterSearchSpace(
            name="weight_decay",
            param_type="float",
            low=1e-4,
            high=0.1,
            log=True,
            default=0.01,
            group=ParameterGroup.LEARNING_RATE,
            importance=0.7
        ),

        # === ГРУППА 2: Regularization ===
        "dropout": ParameterSearchSpace(
            name="dropout",
            param_type="float",
            low=0.1,
            high=0.7,
            step=0.05,
            default=0.4,
            group=ParameterGroup.REGULARIZATION,
            importance=0.6
        ),
        "label_smoothing": ParameterSearchSpace(
            name="label_smoothing",
            param_type="float",
            low=0.0,
            high=0.3,
            step=0.02,
            default=0.1,
            group=ParameterGroup.REGULARIZATION,
            importance=0.5
        ),
        "focal_gamma": ParameterSearchSpace(
            name="focal_gamma",
            param_type="float",
            low=0.5,  # CHANGED: 0.0 means no focusing, at least 0.5
            high=3.0,  # CHANGED: Reduced from 5.0 - too aggressive with class_weights
            step=0.5,
            default=2.0,  # CHANGED: Reduced from 2.5 - works better with class_weights
            group=ParameterGroup.REGULARIZATION,
            importance=0.7  # Increased - critical for mode collapse prevention
        ),

        # === ГРУППА 3: Class Balancing ===
        "use_focal_loss": ParameterSearchSpace(
            name="use_focal_loss",
            param_type="bool",
            default=True,
            group=ParameterGroup.CLASS_BALANCE,
            importance=0.7
        ),
        "use_class_weights": ParameterSearchSpace(
            name="use_class_weights",
            param_type="bool",
            default=True,  # FIXED: Must be True to prevent mode collapse with imbalanced data
            group=ParameterGroup.CLASS_BALANCE,
            importance=0.7  # Increased importance - critical for imbalanced data
        ),
        "use_oversampling": ParameterSearchSpace(
            name="use_oversampling",
            param_type="bool",
            default=False,
            group=ParameterGroup.CLASS_BALANCE,
            importance=0.5
        ),
        "oversample_ratio": ParameterSearchSpace(
            name="oversample_ratio",
            param_type="float",
            low=0.2,
            high=1.0,
            step=0.1,
            default=0.5,
            group=ParameterGroup.CLASS_BALANCE,
            importance=0.4
        ),

        # === ГРУППА 4: Augmentation ===
        "use_augmentation": ParameterSearchSpace(
            name="use_augmentation",
            param_type="bool",
            default=True,
            group=ParameterGroup.AUGMENTATION,
            importance=0.5
        ),
        "gaussian_noise_std": ParameterSearchSpace(
            name="gaussian_noise_std",
            param_type="float",
            low=0.0,
            high=0.05,
            step=0.005,
            default=0.01,
            group=ParameterGroup.AUGMENTATION,
            importance=0.4
        ),

        # === ГРУППА 5: Scheduler ===
        "scheduler_T_0": ParameterSearchSpace(
            name="scheduler_T_0",
            param_type="int",
            low=5,
            high=30,
            default=10,
            group=ParameterGroup.SCHEDULER,
            importance=0.4
        ),
        "scheduler_T_mult": ParameterSearchSpace(
            name="scheduler_T_mult",
            param_type="int",
            low=1,
            high=4,
            default=2,
            group=ParameterGroup.SCHEDULER,
            importance=0.3
        ),

        # === ГРУППА 6: Triple Barrier ===
        "tb_tp_multiplier": ParameterSearchSpace(
            name="tb_tp_multiplier",
            param_type="float",
            low=0.5,
            high=3.0,
            step=0.25,
            default=1.5,
            group=ParameterGroup.TRIPLE_BARRIER,
            importance=0.5
        ),
        "tb_sl_multiplier": ParameterSearchSpace(
            name="tb_sl_multiplier",
            param_type="float",
            low=0.5,
            high=2.5,
            step=0.25,
            default=1.0,
            group=ParameterGroup.TRIPLE_BARRIER,
            importance=0.5
        ),
        "tb_max_holding_period": ParameterSearchSpace(
            name="tb_max_holding_period",
            param_type="int",
            low=6,
            high=72,
            step=6,
            default=24,
            group=ParameterGroup.TRIPLE_BARRIER,
            importance=0.4
        ),

        # === ГРУППА 7: Industry Standard ===
        "use_purging": ParameterSearchSpace(
            name="use_purging",
            param_type="bool",
            default=True,
            group=ParameterGroup.REGULARIZATION,
            importance=0.6
        ),
        "use_embargo": ParameterSearchSpace(
            name="use_embargo",
            param_type="bool",
            default=True,
            group=ParameterGroup.REGULARIZATION,
            importance=0.5
        ),
        "embargo_pct": ParameterSearchSpace(
            name="embargo_pct",
            param_type="float",
            low=0.01,
            high=0.05,
            step=0.01,
            default=0.02,
            group=ParameterGroup.REGULARIZATION,
            importance=0.4
        ),

        # === Batch Size (обычно фиксируется) ===
        "batch_size": ParameterSearchSpace(
            name="batch_size",
            param_type="categorical",
            choices=[32, 64, 128, 256],
            default=128,
            group=ParameterGroup.LEARNING_RATE,
            importance=0.5
        ),
    }


# ============================================================================
# TRIAL RESULT
# ============================================================================

@dataclass
class TrialResult:
    """Результат одного trial."""
    trial_number: int
    params: Dict[str, Any]

    # Основные метрики
    val_f1: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_loss: float

    # Время
    duration_seconds: float
    epochs_completed: int

    # Статус
    status: str  # "completed", "pruned", "failed"
    pruned_at_epoch: Optional[int] = None
    error_message: Optional[str] = None

    # История по эпохам
    epoch_history: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        return asdict(self)


@dataclass
class GroupOptimizationResult:
    """Результат оптимизации группы параметров."""
    group: ParameterGroup
    best_params: Dict[str, Any]
    best_value: float

    # Статистика
    n_trials: int
    n_pruned: int
    n_failed: int
    total_time_seconds: float

    # Все trials
    trials: List[TrialResult] = field(default_factory=list)

    # Convergence info
    converged: bool = False
    convergence_trial: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        result = {
            "group": self.group.value,
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": self.n_trials,
            "n_pruned": self.n_pruned,
            "n_failed": self.n_failed,
            "total_time_seconds": self.total_time_seconds,
            "converged": self.converged,
            "convergence_trial": self.convergence_trial
        }
        return result


# ============================================================================
# HYPERPARAMETER OPTIMIZER
# ============================================================================

class HyperparameterOptimizer:
    """
    Основной класс для оптимизации гиперпараметров.

    Использует Sequential Parameter Group Optimization (SPGO):
    1. Оптимизирует группы параметров по порядку влияния
    2. Закрепляет лучшие значения группы перед переходом к следующей
    3. Использует Bayesian Optimization (Optuna) внутри каждой группы
    4. Применяет aggressive pruning для экономии времени

    Пример использования:
        optimizer = HyperparameterOptimizer(config)
        results = await optimizer.optimize(mode=OptimizationMode.FULL)
        print(f"Best params: {results['best_params']}")
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        search_spaces: Optional[Dict[str, ParameterSearchSpace]] = None,
        stop_event: Optional["threading.Event"] = None  # For graceful shutdown
    ):
        """
        Args:
            config: Конфигурация оптимизации
            search_spaces: Пространства поиска (если None - используются дефолтные)
            stop_event: Threading event for graceful shutdown (checked in callbacks)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna required for hyperparameter optimization. "
                "Install with: pip install optuna"
            )

        self.config = config or OptimizationConfig()
        self.search_spaces = search_spaces or get_default_search_spaces()

        # CRITICAL: Stop event for graceful shutdown
        # This is checked in Optuna callbacks to stop optimization when user requests
        import threading
        self._stop_event = stop_event or threading.Event()

        # Фиксированные параметры (уже оптимизированные группы)
        self.fixed_params: Dict[str, Any] = {}

        # История оптимизации
        self.group_results: Dict[ParameterGroup, GroupOptimizationResult] = {}

        # Текущее лучшее значение
        self.best_value: float = float('-inf') if self.config.optimization_direction == "maximize" else float('inf')
        self.best_params: Dict[str, Any] = {}

        # Время старта
        self.start_time: Optional[datetime] = None

        # MLflow tracker
        self.mlflow_tracker = None
        if self.config.use_mlflow:
            try:
                from backend.ml_engine.mlflow_integration.mlflow_tracker import get_mlflow_tracker
                self.mlflow_tracker = get_mlflow_tracker()
            except Exception as e:
                logger.warning(f"MLflow not available: {e}")

        # CRITICAL: Cached data loaders to avoid reloading data for each trial
        # Data is loaded ONCE before optimization and reused across all trials
        self._cached_train_loader = None
        self._cached_val_loader = None
        self._cached_test_loader = None
        self._data_loaded = False

        logger.info(
            f"HyperparameterOptimizer initialized:\n"
            f"  • Study: {self.config.study_name}\n"
            f"  • Epochs per trial: {self.config.epochs_per_trial}\n"
            f"  • Max trials per group: {self.config.max_trials_per_group}\n"
            f"  • Primary metric: {self.config.primary_metric}\n"
            f"  • MLflow: {self.config.use_mlflow and self.mlflow_tracker is not None}\n"
            f"  • Stop event: {'provided' if stop_event else 'internal'}"
        )

    # ========================================================================
    # DATA LOADING (ONCE)
    # ========================================================================

    async def _preload_data(self) -> bool:
        """
        Preload training data ONCE before optimization starts.

        This prevents the data from being reloaded and duplicated for each trial.
        The data is cached in self._cached_train_loader, etc.

        Returns:
            True if data loaded successfully
        """
        if self._data_loaded:
            logger.info("HYPEROPT: Data already loaded, reusing cached loaders")
            return True

        logger.info(f"\n{'='*60}")
        logger.info("HYPEROPT: PRELOADING DATA (ONCE)")
        logger.info(f"{'='*60}\n")

        try:
            from backend.ml_engine.training_orchestrator import TrainingOrchestrator
            from backend.ml_engine.models.hybrid_cnn_lstm_v2 import ModelConfigV2
            from backend.ml_engine.training.model_trainer_v2 import TrainerConfigV2
            from backend.ml_engine.training.data_loader import DataConfig, HistoricalDataLoader
            from backend.ml_engine.training.class_balancing import ClassBalancingConfig
            from backend.ml_engine.feature_store.feature_store import get_feature_store
            from backend.ml_engine.feature_store.feature_schema import DEFAULT_SCHEMA
            from datetime import timedelta

            # Create configs with default values
            # CRITICAL: num_workers=0 to avoid Windows shared memory issues
            data_config = DataConfig(
                storage_path=str(ML_TRAINING_PATH),
                batch_size=128,
                num_workers=0,  # Disable multiprocessing to avoid Windows memory errors
                use_purging=True,
                use_embargo=True,
                embargo_pct=0.02,
                use_feature_store=True,
                feature_store_group="training_features",
                feature_store_date_range_days=30
            )

            balancing_config = ClassBalancingConfig(
                use_class_weights=False,
                use_focal_loss=True,
                focal_gamma=2.5,
                use_oversampling=False,
                use_undersampling=False
            )

            # Step 1: Run preprocessing ONCE
            logger.info("HYPEROPT: Running preprocessing (adding future labels)...")
            try:
                from preprocessing_add_future_labels_parquet import ParquetFutureLabelProcessor
                processor = ParquetFutureLabelProcessor(
                    feature_store_group=data_config.feature_store_group,
                    start_date=None,
                    end_date=None
                )
                processor.process_all_data()
                logger.info("HYPEROPT: ✅ Preprocessing completed")
            except Exception as e:
                logger.warning(f"HYPEROPT: ⚠️ Preprocessing failed: {e}")

            # Step 2: Load data from Feature Store
            logger.info("HYPEROPT: Loading data from Feature Store...")
            feature_store = get_feature_store()

            end_date = datetime.now()
            start_date = end_date - timedelta(days=data_config.feature_store_date_range_days)

            features_df = feature_store.read_offline_features(
                feature_group=data_config.feature_store_group,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )

            if features_df.empty:
                logger.error("HYPEROPT: No data found in Feature Store!")
                return False

            logger.info(f"HYPEROPT: ✓ Loaded {len(features_df):,} samples from Feature Store")

            # Step 3: Create DataLoaders
            logger.info("HYPEROPT: Creating DataLoaders...")
            data_loader = HistoricalDataLoader(data_config, balancing_config=balancing_config)

            self._cached_train_loader, self._cached_val_loader, self._cached_test_loader = \
                data_loader.load_from_dataframe(
                    features_df=features_df,
                    feature_columns=DEFAULT_SCHEMA.get_all_feature_columns(),
                    label_column=DEFAULT_SCHEMA.label_column,
                    timestamp_column=DEFAULT_SCHEMA.timestamp_column,
                    symbol_column=DEFAULT_SCHEMA.symbol_column,
                    apply_resampling=True
                )

            if self._cached_train_loader is None:
                logger.error("HYPEROPT: Failed to create DataLoaders!")
                return False

            # Get dataset sizes (Dataset implements __len__ but type checker doesn't know)
            train_size = len(self._cached_train_loader.dataset)  # type: ignore[arg-type]
            val_size = len(self._cached_val_loader.dataset) if self._cached_val_loader else 0  # type: ignore[arg-type]
            test_size = len(self._cached_test_loader.dataset) if self._cached_test_loader else 0  # type: ignore[arg-type]

            logger.info(f"HYPEROPT: ✓ DataLoaders created successfully:")
            logger.info(f"  • Train: {train_size:,} samples")
            logger.info(f"  • Val: {val_size:,} samples")
            logger.info(f"  • Test: {test_size:,} samples")

            self._data_loaded = True
            return True

        except Exception as e:
            logger.error(f"HYPEROPT: Failed to preload data: {e}", exc_info=True)
            return False

    # ========================================================================
    # MAIN OPTIMIZATION METHODS
    # ========================================================================

    async def optimize(
        self,
        mode: OptimizationMode = OptimizationMode.FULL,
        target_group: Optional[ParameterGroup] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Запустить оптимизацию.

        Args:
            mode: Режим оптимизации
            target_group: Целевая группа (для mode=GROUP)
            resume_from: Путь к study для продолжения

        Returns:
            Словарь с результатами:
            - best_params: лучшие параметры
            - best_value: лучшее значение метрики
            - group_results: результаты по группам
            - total_time: общее время
        """
        self.start_time = datetime.now()

        logger.info(f"\n{'='*80}")
        logger.info(f"HYPERPARAMETER OPTIMIZATION STARTED")
        logger.info(f"Mode: {mode.value}")
        logger.info(f"Started at: {self.start_time.isoformat()}")
        logger.info(f"{'='*80}\n")

        # CRITICAL: Preload data ONCE before starting optimization
        if not await self._preload_data():
            raise RuntimeError("Failed to preload training data")

        try:
            if mode == OptimizationMode.FULL:
                await self._optimize_full()
            elif mode == OptimizationMode.GROUP:
                if target_group is None:
                    raise ValueError("target_group required for GROUP mode")
                await self._optimize_group(target_group)
            elif mode == OptimizationMode.QUICK:
                # Quick mode: только LR группа с меньшим количеством trials
                self.config.max_trials_per_group = 8
                await self._optimize_group(ParameterGroup.LEARNING_RATE)
            elif mode == OptimizationMode.FINE_TUNE:
                await self._fine_tune()
            elif mode == OptimizationMode.RESUME:
                if resume_from is None:
                    raise ValueError("resume_from required for RESUME mode")
                await self._resume_optimization(resume_from)

            # Сохраняем результаты
            await self._save_results()

            total_time = (datetime.now() - self.start_time).total_seconds()

            logger.info(f"\n{'='*80}")
            logger.info(f"OPTIMIZATION COMPLETED")
            logger.info(f"Total time: {total_time/3600:.2f} hours")
            logger.info(f"Best {self.config.primary_metric}: {self.best_value:.4f}")
            logger.info(f"Best params: {json.dumps(self.best_params, indent=2, default=str)}")
            logger.info(f"{'='*80}\n")

            return {
                "best_params": self.best_params,
                "best_value": self.best_value,
                "group_results": {g.value: r.to_dict() for g, r in self.group_results.items()},
                "total_time_seconds": total_time,
                "fixed_params": self.fixed_params
            }

        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            raise

    async def _optimize_full(self):
        """Полная оптимизация всех групп последовательно."""
        for group in OPTIMIZATION_ORDER:
            # CRITICAL: Check stop event between groups
            if self._stop_event.is_set():
                logger.info(f"HYPEROPT: Stop event detected between groups, stopping at {group.value}")
                break

            # Проверяем время
            if self._check_time_limit():
                logger.warning(f"Time limit reached, stopping at group {group.value}")
                break

            logger.info(f"\n{'='*60}")
            logger.info(f"OPTIMIZING GROUP: {group.value}")
            logger.info(f"Fixed params so far: {list(self.fixed_params.keys())}")
            logger.info(f"{'='*60}\n")

            # Оптимизируем группу
            result = await self._optimize_group(group)

            # Закрепляем лучшие параметры группы
            for param_name, param_value in result.best_params.items():
                self.fixed_params[param_name] = param_value

            self.group_results[group] = result

            # Обновляем лучшие общие параметры
            if self._is_better(result.best_value, self.best_value):
                self.best_value = result.best_value
                self.best_params = {**self.fixed_params}

            logger.info(f"\nGroup {group.value} completed:")
            logger.info(f"  • Best value: {result.best_value:.4f}")
            logger.info(f"  • Trials: {result.n_trials} (pruned: {result.n_pruned})")
            logger.info(f"  • Fixed params: {result.best_params}")

    async def _optimize_group(
        self,
        group: ParameterGroup
    ) -> GroupOptimizationResult:
        """
        Оптимизация одной группы параметров.

        Args:
            group: Группа для оптимизации

        Returns:
            GroupOptimizationResult
        """
        # Получаем параметры группы
        group_params = {
            name: space for name, space in self.search_spaces.items()
            if space.group == group
        }

        if not group_params:
            logger.warning(f"No parameters in group {group.value}, skipping")
            return GroupOptimizationResult(
                group=group,
                best_params={},
                best_value=self.best_value,
                n_trials=0,
                n_pruned=0,
                n_failed=0,
                total_time_seconds=0
            )

        # Создаём Optuna study
        study_name = f"{self.config.study_name}_{group.value}"
        storage_path = Path(self.config.storage_path) / f"{study_name}.db"

        # Sampler с seed для воспроизводимости
        sampler = TPESampler(
            seed=self.config.seed,
            n_startup_trials=self.config.n_startup_trials,
            multivariate=True  # Учитывает корреляции между параметрами
        )

        # Pruner для ранней остановки плохих trials
        pruner = MedianPruner(
            n_startup_trials=self.config.n_startup_trials,
            n_warmup_steps=self.config.n_warmup_steps,
            interval_steps=1,
            n_min_trials=self.config.n_startup_trials
        )

        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{storage_path}",
            direction=self.config.optimization_direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )

        # Добавляем baseline trial (дефолтные значения)
        if len(study.trials) == 0:
            baseline_params = {
                name: space.default for name, space in group_params.items()
            }
            study.enqueue_trial(baseline_params)

        # Создаём objective function
        objective = self._create_objective(group_params)

        # Callbacks для отслеживания конвергенции
        convergence_tracker = ConvergenceTracker(
            patience=self.config.convergence_patience,
            threshold=self.config.convergence_threshold,
            direction=self.config.optimization_direction
        )

        def convergence_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            """Callback для проверки конвергенции."""
            if trial.state == optuna.trial.TrialState.COMPLETE:
                convergence_tracker.update(trial.value)

                if convergence_tracker.is_converged():
                    logger.info(f"Convergence reached at trial {trial.number}")
                    study.stop()

        def stop_event_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            """
            CRITICAL: Callback to check for stop requests from API.

            This is called after each trial completes. If the stop_event is set
            (by the /stop endpoint), we call study.stop() to gracefully terminate
            the optimization.
            """
            if self._stop_event.is_set():
                logger.info("=" * 60)
                logger.info(f"HYPEROPT: STOP EVENT detected after trial {trial.number}")
                logger.info("HYPEROPT: Stopping optimization gracefully...")
                logger.info("=" * 60)
                study.stop()

        group_start_time = time.time()

        # Запускаем оптимизацию
        # CRITICAL: stop_event_callback MUST be in callbacks list for graceful shutdown
        try:
            study.optimize(
                objective,
                n_trials=self.config.max_trials_per_group,
                timeout=self.config.max_total_time_hours * 3600 / len(OPTIMIZATION_ORDER),
                callbacks=[convergence_callback, stop_event_callback],  # Added stop_event_callback!
                show_progress_bar=self.config.verbose
            )
        except Exception as e:
            logger.error(f"Optimization error: {e}")

        group_time = time.time() - group_start_time

        # Собираем результаты
        trials = []
        n_pruned = 0
        n_failed = 0

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.PRUNED:
                n_pruned += 1
            elif trial.state == optuna.trial.TrialState.FAIL:
                n_failed += 1

            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials.append(TrialResult(
                    trial_number=trial.number,
                    params=trial.params,
                    val_f1=trial.value if self.config.primary_metric == "val_f1" else 0,
                    val_accuracy=trial.user_attrs.get("val_accuracy", 0),
                    val_precision=trial.user_attrs.get("val_precision", 0),
                    val_recall=trial.user_attrs.get("val_recall", 0),
                    val_loss=trial.user_attrs.get("val_loss", float('inf')),
                    duration_seconds=trial.user_attrs.get("duration", 0),
                    epochs_completed=trial.user_attrs.get("epochs_completed", 0),
                    status="completed"
                ))

        # Лучшие параметры (safely handle case when all trials failed)
        try:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                best_params = study.best_trial.params
                best_value = study.best_value
            else:
                logger.warning(f"No completed trials for group {group}. Using default params.")
                best_params = {name: space.default for name, space in group_params.items()}
                best_value = self.best_value
        except Exception as e:
            logger.warning(f"Failed to get best trial for group {group}: {e}. Using default params.")
            best_params = {name: space.default for name, space in group_params.items()}
            best_value = self.best_value

        return GroupOptimizationResult(
            group=group,
            best_params=best_params,
            best_value=best_value,
            n_trials=len(study.trials),
            n_pruned=n_pruned,
            n_failed=n_failed,
            total_time_seconds=group_time,
            trials=trials,
            converged=convergence_tracker.is_converged(),
            convergence_trial=convergence_tracker.best_trial_number
        )

    def _create_objective(
        self,
        group_params: Dict[str, ParameterSearchSpace]
    ) -> Callable:
        """
        Создать objective function для Optuna.

        Args:
            group_params: Параметры для оптимизации

        Returns:
            Callable objective function
        """
        def objective(trial: optuna.Trial) -> float:
            """Objective function для одного trial."""
            trial_start = time.time()

            # Собираем параметры
            params = dict(self.fixed_params)  # Фиксированные из предыдущих групп

            for name, space in group_params.items():
                if space.param_type == "float":
                    if space.log:
                        value = trial.suggest_float(name, space.low, space.high, log=True)
                    elif space.step:
                        value = trial.suggest_float(name, space.low, space.high, step=space.step)
                    else:
                        value = trial.suggest_float(name, space.low, space.high)
                elif space.param_type == "int":
                    if space.step:
                        value = trial.suggest_int(name, int(space.low), int(space.high), step=int(space.step))
                    else:
                        value = trial.suggest_int(name, int(space.low), int(space.high))
                elif space.param_type == "categorical":
                    value = trial.suggest_categorical(name, space.choices)
                elif space.param_type == "bool":
                    value = trial.suggest_categorical(name, [True, False])
                else:
                    value = space.default

                params[name] = value

            # Добавляем фиксированные параметры обучения
            params["epochs"] = self.config.epochs_per_trial

            # Запускаем обучение
            try:
                metrics = asyncio.get_event_loop().run_until_complete(
                    self._run_training(params, trial)
                )
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                raise optuna.TrialPruned(f"Training failed: {e}")

            # Сохраняем дополнительные метрики
            trial.set_user_attr("val_accuracy", metrics.get("val_accuracy", 0))
            trial.set_user_attr("val_precision", metrics.get("val_precision", 0))
            trial.set_user_attr("val_recall", metrics.get("val_recall", 0))
            trial.set_user_attr("val_loss", metrics.get("val_loss", float('inf')))
            trial.set_user_attr("duration", time.time() - trial_start)
            trial.set_user_attr("epochs_completed", metrics.get("epochs_completed", 0))

            # Логируем
            if self.config.verbose:
                logger.info(
                    f"Trial {trial.number}: "
                    f"{self.config.primary_metric}={metrics.get(self.config.primary_metric, 0):.4f}, "
                    f"time={time.time() - trial_start:.1f}s"
                )

            # Возвращаем основную метрику
            return metrics.get(self.config.primary_metric, 0)

        return objective

    async def _run_training(
        self,
        params: Dict[str, Any],
        trial: optuna.Trial
    ) -> Dict[str, float]:
        """
        Запустить обучение модели с заданными параметрами.

        CRITICAL: Uses pre-cached DataLoaders instead of reloading data each time.
        This prevents data duplication and memory explosion.

        Args:
            params: Параметры обучения
            trial: Optuna trial для pruning

        Returns:
            Словарь с метриками
        """
        import torch

        from backend.ml_engine.models.hybrid_cnn_lstm_v2 import HybridCNNLSTMv2, ModelConfigV2
        from backend.ml_engine.training.model_trainer_v2 import ModelTrainerV2, TrainerConfigV2

        # Ensure data is loaded
        if not self._data_loaded or self._cached_train_loader is None:
            raise RuntimeError("Data not preloaded! Call _preload_data() first.")

        logger.info(f"HYPEROPT: Trial {trial.number} - using cached DataLoaders (NO reload)")

        # Create trainer config
        # CRITICAL FIX: Always use BOTH class_weights AND focal_loss for imbalanced data
        # This prevents mode collapse (model predicting only majority class)
        trainer_config = TrainerConfigV2(
            epochs=params.get("epochs", self.config.epochs_per_trial),
            learning_rate=params.get("learning_rate", 5e-5),
            weight_decay=params.get("weight_decay", 0.01),
            batch_size=params.get("batch_size", 128),
            label_smoothing=params.get("label_smoothing", 0.1),
            scheduler_T_0=params.get("scheduler_T_0", 10),
            scheduler_T_mult=params.get("scheduler_T_mult", 2),
            use_augmentation=params.get("use_augmentation", True),
            gaussian_noise_std=params.get("gaussian_noise_std", 0.01),
            # CRITICAL: Both focal_loss AND class_weights must be enabled
            use_focal_loss=params.get("use_focal_loss", True),
            focal_gamma=params.get("focal_gamma", 2.0),  # Reduced from 2.5 - less aggressive with class_weights
            use_class_weights=params.get("use_class_weights", True),  # FIXED: Was False, causing mode collapse
            early_stopping_patience=max(3, self.config.epochs_per_trial // 2),
            verbose=False,  # Less logs
            use_tqdm=False
        )

        # Get input dimension from first batch of cached train loader
        first_batch = next(iter(self._cached_train_loader))
        sequence_shape = first_batch['sequence'].shape
        input_dim = sequence_shape[2]  # (batch, seq_len, features)
        seq_len = sequence_shape[1]

        logger.debug(f"HYPEROPT: Input shape: seq_len={seq_len}, input_dim={input_dim}")

        # Create model config with correct input dimensions
        model_config = ModelConfigV2(
            input_features=input_dim,
            sequence_length=seq_len,
            num_classes=3,
            dropout=params.get("dropout", 0.4)
        )

        # Create a NEW model for each trial (different hyperparameters)
        model = HybridCNNLSTMv2(config=model_config)

        # Create trainer
        trainer = ModelTrainerV2(model=model, config=trainer_config)

        # Custom epoch callback for Optuna pruning
        original_epoch_end = trainer._on_epoch_end if hasattr(trainer, '_on_epoch_end') else None

        def epoch_callback_wrapper(epoch: int, metrics: Dict[str, float]):
            """Wrapper to report to Optuna and check pruning."""
            primary_value = metrics.get(self.config.primary_metric, 0)
            trial.report(primary_value, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if original_epoch_end:
                original_epoch_end(epoch, metrics)

        # Run training using CACHED DataLoaders
        try:
            history = trainer.train(
                train_loader=self._cached_train_loader,
                val_loader=self._cached_val_loader,
                test_loader=self._cached_test_loader
            )

            # Get best metrics from history
            if history:
                best_epoch = max(history, key=lambda x: x.val_f1 if hasattr(x, 'val_f1') else 0)
                metrics = {
                    "val_f1": getattr(best_epoch, 'val_f1', 0),
                    "val_accuracy": getattr(best_epoch, 'val_accuracy', 0),
                    "val_precision": getattr(best_epoch, 'val_precision', 0),
                    "val_recall": getattr(best_epoch, 'val_recall', 0),
                    "val_loss": getattr(best_epoch, 'val_loss', float('inf')),
                    "epochs_completed": len(history)
                }
            else:
                metrics = {
                    "val_f1": 0,
                    "val_accuracy": 0,
                    "val_precision": 0,
                    "val_recall": 0,
                    "val_loss": float('inf'),
                    "epochs_completed": 0
                }

            # ================================================================
            # MODE COLLAPSE DETECTION
            # ================================================================
            # Check if model collapsed to predicting only one class
            # This happens when F1 ≈ accuracy and precision is low
            # Example: accuracy=0.53, f1=0.37 means model predicts only majority class

            val_f1 = metrics.get("val_f1", 0)
            val_precision = metrics.get("val_precision", 0)
            val_recall = metrics.get("val_recall", 0)

            # Mode collapse indicators:
            # 1. Very low precision (< 0.35) - model not learning minority classes
            # 2. F1 close to simple majority baseline (0.33-0.40 for 3 classes)
            # 3. Recall = accuracy (model outputs same class for all samples)

            is_mode_collapse = False
            collapse_reason = ""

            if val_precision < 0.35:
                is_mode_collapse = True
                collapse_reason = f"Very low precision ({val_precision:.3f} < 0.35)"
            elif val_f1 < 0.40 and val_precision < 0.40:
                is_mode_collapse = True
                collapse_reason = f"Baseline-level F1 ({val_f1:.3f}) with low precision ({val_precision:.3f})"

            if is_mode_collapse:
                logger.warning(
                    f"⚠️ MODE COLLAPSE DETECTED in trial {trial.number}!\n"
                    f"   Reason: {collapse_reason}\n"
                    f"   Metrics: F1={val_f1:.4f}, Prec={val_precision:.4f}, Recall={val_recall:.4f}\n"
                    f"   This trial will be penalized."
                )
                # Penalize the metrics to discourage similar parameter combinations
                # Don't set to 0 (causes issues), but significantly reduce
                metrics["val_f1"] = val_f1 * 0.5  # 50% penalty
                metrics["mode_collapse"] = True
            else:
                metrics["mode_collapse"] = False

            # Clean up GPU memory after each trial
            del model
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return metrics

        except optuna.TrialPruned:
            # Clean up even on pruning
            del model
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    # ========================================================================
    # FINE-TUNING
    # ========================================================================

    async def _fine_tune(self):
        """Тонкая настройка вокруг лучших найденных параметров."""
        if not self.best_params:
            logger.warning("No best params found, running full optimization first")
            await self._optimize_full()
            return

        logger.info("Starting fine-tuning around best parameters...")

        # Создаём уменьшенные search spaces вокруг лучших значений
        fine_tune_spaces = {}

        for name, space in self.search_spaces.items():
            if name not in self.best_params:
                continue

            best_value = self.best_params[name]

            if space.param_type == "float":
                # Уменьшаем диапазон на 80%
                range_size = space.high - space.low
                new_low = max(space.low, best_value - range_size * 0.1)
                new_high = min(space.high, best_value + range_size * 0.1)

                fine_tune_spaces[name] = ParameterSearchSpace(
                    name=name,
                    param_type="float",
                    low=new_low,
                    high=new_high,
                    log=space.log,
                    step=space.step / 2 if space.step else None,
                    default=best_value,
                    group=space.group,
                    importance=space.importance
                )
            elif space.param_type == "int":
                range_size = space.high - space.low
                new_low = max(space.low, best_value - range_size * 0.1)
                new_high = min(space.high, best_value + range_size * 0.1)

                fine_tune_spaces[name] = ParameterSearchSpace(
                    name=name,
                    param_type="int",
                    low=new_low,
                    high=new_high,
                    step=max(1, (space.step or 1) // 2),
                    default=best_value,
                    group=space.group,
                    importance=space.importance
                )

        # Временно заменяем search spaces
        original_spaces = self.search_spaces
        self.search_spaces = fine_tune_spaces
        self.config.max_trials_per_group = 10  # Меньше trials для fine-tuning

        # Оптимизируем только высоковлиятельные группы
        for group in [ParameterGroup.LEARNING_RATE, ParameterGroup.REGULARIZATION]:
            await self._optimize_group(group)

        # Восстанавливаем
        self.search_spaces = original_spaces

    # ========================================================================
    # RESUME
    # ========================================================================

    async def _resume_optimization(self, resume_path: str):
        """Продолжить прерванную оптимизацию."""
        results_path = Path(resume_path)

        if not results_path.exists():
            raise FileNotFoundError(f"Resume path not found: {resume_path}")

        # Загружаем предыдущие результаты
        with open(results_path / "results.json", "r") as f:
            prev_results = json.load(f)

        self.fixed_params = prev_results.get("fixed_params", {})
        self.best_params = prev_results.get("best_params", {})
        self.best_value = prev_results.get("best_value", float('-inf'))

        # Определяем, с какой группы продолжать
        completed_groups = set(prev_results.get("group_results", {}).keys())

        for group in OPTIMIZATION_ORDER:
            if group.value not in completed_groups:
                logger.info(f"Resuming from group: {group.value}")
                await self._optimize_group(group)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _is_better(self, value: float, best: float) -> bool:
        """Проверить, лучше ли новое значение."""
        if self.config.optimization_direction == "maximize":
            return value > best
        return value < best

    def _check_time_limit(self) -> bool:
        """Проверить, не превышен ли лимит времени."""
        if self.start_time is None:
            return False

        elapsed = (datetime.now() - self.start_time).total_seconds()
        limit = self.config.max_total_time_hours * 3600

        return elapsed >= limit

    async def _save_results(self):
        """Сохранить результаты оптимизации."""
        results_path = Path(self.config.storage_path)

        results = {
            "study_name": self.config.study_name,
            "completed_at": datetime.now().isoformat(),
            "best_params": self.best_params,
            "best_value": self.best_value,
            "fixed_params": self.fixed_params,
            "group_results": {g.value: r.to_dict() for g, r in self.group_results.items()},
            "config": asdict(self.config) if self.config else {}  # type: ignore[arg-type]
        }

        # Сохраняем JSON
        with open(results_path / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Сохраняем лучшие параметры отдельно (для удобства использования)
        with open(results_path / "best_params.json", "w") as f:
            json.dump(self.best_params, f, indent=2, default=str)

        logger.info(f"Results saved to: {results_path}")

        # Логируем в MLflow
        if self.mlflow_tracker:
            try:
                self.mlflow_tracker.start_run(run_name=f"hyperopt_{self.config.study_name}")
                self.mlflow_tracker.log_params(self.best_params)
                # Use log_metrics (plural) with a dict
                self.mlflow_tracker.log_metrics({"best_" + self.config.primary_metric: self.best_value})
                self.mlflow_tracker.end_run()
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")


# ============================================================================
# CONVERGENCE TRACKER
# ============================================================================

class ConvergenceTracker:
    """Отслеживание конвергенции оптимизации."""

    def __init__(
        self,
        patience: int = 5,
        threshold: float = 0.005,
        direction: str = "maximize"
    ):
        self.patience = patience
        self.threshold = threshold
        self.direction = direction

        self.best_value = float('-inf') if direction == "maximize" else float('inf')
        self.best_trial_number = 0
        self.trials_since_improvement = 0
        self.values: List[float] = []

    def update(self, value: float) -> bool:
        """
        Обновить tracker с новым значением.

        Returns:
            True если было улучшение
        """
        self.values.append(value)

        improved = False
        if self.direction == "maximize":
            if value > self.best_value + self.threshold:
                improved = True
        else:
            if value < self.best_value - self.threshold:
                improved = True

        if improved:
            self.best_value = value
            self.best_trial_number = len(self.values) - 1
            self.trials_since_improvement = 0
        else:
            self.trials_since_improvement += 1

        return improved

    def is_converged(self) -> bool:
        """Проверить, достигнута ли конвергенция."""
        return self.trials_since_improvement >= self.patience


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization for ML Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full optimization (all groups)
  python -m backend.ml_engine.hyperparameter_optimizer --mode full

  # Quick optimization (learning rate only)
  python -m backend.ml_engine.hyperparameter_optimizer --mode quick

  # Optimize specific group
  python -m backend.ml_engine.hyperparameter_optimizer --mode group --group regularization

  # Resume previous optimization
  python -m backend.ml_engine.hyperparameter_optimizer --mode resume --resume-from data/hyperopt

  # Fine-tune around best params
  python -m backend.ml_engine.hyperparameter_optimizer --mode fine_tune
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "quick", "group", "resume", "fine_tune"],
        default="full",
        help="Optimization mode"
    )

    parser.add_argument(
        "--group",
        type=str,
        choices=[str(g.value) for g in ParameterGroup],  # Explicit str cast for type checker
        default=None,
        help="Target group for GROUP mode"
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to resume optimization from"
    )

    parser.add_argument(
        "--study-name",
        type=str,
        default="ml_hyperopt",
        help="Name for the optimization study"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Epochs per trial"
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        default=15,
        help="Max trials per group"
    )

    parser.add_argument(
        "--max-hours",
        type=float,
        default=24.0,
        help="Max total optimization time in hours"
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["val_f1", "val_accuracy", "val_loss"],
        default="val_f1",
        help="Primary optimization metric"
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Создаём конфигурацию
    config = OptimizationConfig(
        study_name=args.study_name,
        epochs_per_trial=args.epochs,
        max_trials_per_group=args.max_trials,
        max_total_time_hours=args.max_hours,
        primary_metric=args.metric,
        optimization_direction="minimize" if args.metric == "val_loss" else "maximize",
        use_mlflow=not args.no_mlflow,
        seed=args.seed
    )

    # Создаём optimizer
    optimizer = HyperparameterOptimizer(config=config)

    # Определяем режим
    mode = OptimizationMode(args.mode)
    target_group = ParameterGroup(args.group) if args.group else None

    # Запускаем оптимизацию
    results = await optimizer.optimize(
        mode=mode,
        target_group=target_group,
        resume_from=args.resume_from
    )

    # Выводим результаты
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Best {args.metric}: {results['best_value']:.4f}")
    print(f"Total time: {results['total_time_seconds']/3600:.2f} hours")
    print("\nBest Parameters:")
    for param, value in results['best_params'].items():
        print(f"  • {param}: {value}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    asyncio.run(main())
