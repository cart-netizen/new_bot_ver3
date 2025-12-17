"""
Ensemble API - управление мультимодельным ensemble.

Эндпоинты для:
1. Управления моделями (включение/отключение)
2. Настройки весов моделей
3. Выбора стратегии консенсуса
4. Мониторинга производительности
5. Запуска обучения моделей

Путь: backend/api/ensemble_api.py
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio
import concurrent.futures
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from backend.core.logger import get_logger

# Thread pool для обучения моделей (не блокирует FastAPI event loop)
_training_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="model_training")
from backend.ml_engine.ensemble import (
    EnsembleConsensus,
    EnsembleConfig,
    ModelType,
    ConsensusStrategy,
    Direction,
    create_ensemble_consensus
)

logger = get_logger(__name__)

# Router
router = APIRouter(prefix="/api/ensemble", tags=["Ensemble Management"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ModelWeightUpdate(BaseModel):
    """Обновление веса модели."""
    model_type: str = Field(..., description="Тип модели: cnn_lstm, mpd_transformer, tlob")
    weight: float = Field(..., ge=0.0, le=1.0, description="Вес модели (0-1)")


class ModelEnableUpdate(BaseModel):
    """Включение/отключение модели."""
    model_type: str = Field(..., description="Тип модели")
    enabled: bool = Field(..., description="Включена ли модель")


class StrategyUpdate(BaseModel):
    """Обновление стратегии консенсуса."""
    strategy: str = Field(
        ...,
        description="Стратегия: weighted_voting, unanimous, majority, confidence_based, adaptive"
    )


class EnsembleConfigUpdate(BaseModel):
    """Обновление конфигурации ensemble."""
    min_confidence_for_trade: Optional[float] = Field(None, ge=0.0, le=1.0)
    unanimous_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    conflict_resolution: Optional[str] = Field(None)
    enable_adaptive_weights: Optional[bool] = Field(None)


class TrainingRequest(BaseModel):
    """Запрос на обучение модели."""
    model_type: str = Field(..., description="Тип модели для обучения")
    epochs: int = Field(default=150, ge=1, le=1000)
    learning_rate: float = Field(default=5e-5, gt=0)
    symbols: List[str] = Field(default=["BTCUSDT"])
    days: int = Field(default=30, ge=1, le=365)


class PerformanceUpdate(BaseModel):
    """Обновление производительности модели."""
    model_type: str
    actual_direction: int = Field(..., ge=0, le=2)
    predicted_direction: int = Field(..., ge=0, le=2)
    profit_loss: float


class EnsembleStatusResponse(BaseModel):
    """Статус ensemble системы."""
    enabled: bool
    strategy: str
    models: Dict[str, Dict[str, Any]]
    stats: Dict[str, Any]
    config: Dict[str, Any]


class PredictionResponse(BaseModel):
    """Ответ с предсказанием."""
    direction: str
    confidence: float
    meta_confidence: float
    expected_return: float
    should_trade: bool
    consensus_type: str
    agreement_ratio: float
    model_predictions: Dict[str, Any]


# ============================================================================
# GLOBAL ENSEMBLE INSTANCE
# ============================================================================

# Глобальный экземпляр ensemble (инициализируется при запуске)
_ensemble_instance: Optional[EnsembleConsensus] = None

# Training tasks
_training_tasks: Dict[str, Dict[str, Any]] = {}


def get_ensemble() -> EnsembleConsensus:
    """Получает или создает экземпляр ensemble."""
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = create_ensemble_consensus()
    return _ensemble_instance


def set_ensemble(ensemble: EnsembleConsensus):
    """Устанавливает экземпляр ensemble."""
    global _ensemble_instance
    _ensemble_instance = ensemble


# ============================================================================
# STATUS ENDPOINTS
# ============================================================================

@router.get("/status", response_model=EnsembleStatusResponse)
async def get_ensemble_status():
    """
    Получить статус ensemble системы.

    Возвращает:
    - Текущую стратегию
    - Состояние всех моделей
    - Статистику предсказаний
    """
    ensemble = get_ensemble()
    stats = ensemble.get_stats()
    config = ensemble.get_config()

    return EnsembleStatusResponse(
        enabled=True,
        strategy=config['consensus_strategy'],
        models=stats['model_weights'],
        stats={
            'total_predictions': stats['total_predictions'],
            'unanimous_count': stats['unanimous_count'],
            'majority_count': stats['majority_count'],
            'conflict_count': stats['conflict_count'],
            'trades_signaled': stats['trades_signaled'],
            'registered_models': stats['registered_models']
        },
        config={
            'min_confidence_for_trade': config['min_confidence_for_trade'],
            'unanimous_threshold': config['unanimous_threshold'],
            'conflict_resolution': config['conflict_resolution'],
            'enable_adaptive_weights': config['enable_adaptive_weights']
        }
    )


@router.get("/models")
async def get_models():
    """
    Получить список всех моделей и их состояние.
    """
    ensemble = get_ensemble()
    stats = ensemble.get_stats()

    models = []
    for model_name, model_info in stats['model_weights'].items():
        is_registered = model_name in stats['registered_models']

        models.append({
            'name': model_name,
            'display_name': _get_model_display_name(model_name),
            'weight': model_info['weight'],
            'enabled': model_info['enabled'],
            'performance_score': model_info['performance_score'],
            'is_registered': is_registered,
            'description': _get_model_description(model_name)
        })

    return {
        'models': models,
        'total_models': len(models),
        'active_models': sum(1 for m in models if m['enabled'] and m['is_registered'])
    }


# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/models/enable")
async def enable_model(request: ModelEnableUpdate):
    """
    Включить или отключить модель.

    Args:
        model_type: Тип модели (cnn_lstm, mpd_transformer, tlob)
        enabled: True для включения, False для отключения
    """
    ensemble = get_ensemble()

    try:
        model_type = ModelType(request.model_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type: {request.model_type}"
        )

    ensemble.enable_model(model_type, request.enabled)

    return {
        'success': True,
        'model_type': request.model_type,
        'enabled': request.enabled,
        'message': f"Model {request.model_type} {'enabled' if request.enabled else 'disabled'}"
    }


@router.post("/models/weight")
async def update_model_weight(request: ModelWeightUpdate):
    """
    Обновить вес модели.

    Args:
        model_type: Тип модели
        weight: Новый вес (0-1)
    """
    ensemble = get_ensemble()

    try:
        model_type = ModelType(request.model_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type: {request.model_type}"
        )

    ensemble.set_model_weight(model_type, request.weight)

    return {
        'success': True,
        'model_type': request.model_type,
        'weight': request.weight,
        'message': f"Model {request.model_type} weight updated to {request.weight}"
    }


@router.post("/models/weights/batch")
async def update_all_weights(weights: Dict[str, float]):
    """
    Обновить веса всех моделей одновременно.

    Args:
        weights: Dict[model_type, weight]
    """
    ensemble = get_ensemble()

    updated = []
    errors = []

    for model_name, weight in weights.items():
        try:
            model_type = ModelType(model_name)
            ensemble.set_model_weight(model_type, weight)
            updated.append(model_name)
        except ValueError as e:
            errors.append({'model': model_name, 'error': str(e)})

    return {
        'success': len(errors) == 0,
        'updated': updated,
        'errors': errors
    }


# ============================================================================
# STRATEGY ENDPOINTS
# ============================================================================

@router.get("/strategy")
async def get_current_strategy():
    """
    Получить текущую стратегию консенсуса.
    """
    ensemble = get_ensemble()
    config = ensemble.get_config()

    return {
        'current_strategy': config['consensus_strategy'],
        'available_strategies': [s.value for s in ConsensusStrategy],
        'strategy_descriptions': {
            'weighted_voting': 'Взвешенное голосование по весам моделей',
            'unanimous': 'Требуется единогласие всех моделей',
            'majority': 'Решение большинства моделей',
            'confidence_based': 'Выбор модели с максимальной уверенностью',
            'adaptive': 'Адаптивные веса на основе производительности'
        }
    }


@router.post("/strategy")
async def update_strategy(request: StrategyUpdate):
    """
    Изменить стратегию консенсуса.

    Args:
        strategy: Новая стратегия
    """
    ensemble = get_ensemble()

    try:
        strategy = ConsensusStrategy(request.strategy)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy: {request.strategy}"
        )

    ensemble.set_strategy(strategy)

    return {
        'success': True,
        'strategy': request.strategy,
        'message': f"Strategy updated to {request.strategy}"
    }


# ============================================================================
# CONFIG ENDPOINTS
# ============================================================================

@router.get("/config")
async def get_ensemble_config():
    """
    Получить полную конфигурацию ensemble.
    """
    ensemble = get_ensemble()
    return ensemble.get_config()


@router.post("/config")
async def update_ensemble_config(config: EnsembleConfigUpdate):
    """
    Обновить конфигурацию ensemble.
    """
    ensemble = get_ensemble()

    if config.min_confidence_for_trade is not None:
        ensemble.config.min_confidence_for_trade = config.min_confidence_for_trade

    if config.unanimous_threshold is not None:
        ensemble.config.unanimous_threshold = config.unanimous_threshold

    if config.conflict_resolution is not None:
        ensemble.config.conflict_resolution = config.conflict_resolution

    if config.enable_adaptive_weights is not None:
        ensemble.config.enable_adaptive_weights = config.enable_adaptive_weights

    return {
        'success': True,
        'updated_config': ensemble.get_config()
    }


@router.post("/config/save")
async def save_config():
    """
    Сохранить текущую конфигурацию на диск.
    """
    ensemble = get_ensemble()

    try:
        await ensemble.save_config("config/ensemble_config.json")
        return {'success': True, 'path': 'config/ensemble_config.json'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/load")
async def load_config():
    """
    Загрузить конфигурацию с диска.
    """
    ensemble = get_ensemble()

    try:
        await ensemble.load_config("config/ensemble_config.json")
        return {'success': True, 'config': ensemble.get_config()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Config file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PERFORMANCE ENDPOINTS
# ============================================================================

@router.post("/performance/update")
async def update_performance(update: PerformanceUpdate):
    """
    Обновить производительность модели на основе результата сделки.

    Используется для адаптивных весов.
    """
    ensemble = get_ensemble()

    try:
        model_type = ModelType(update.model_type)
        actual = Direction(update.actual_direction)
        predicted = Direction(update.predicted_direction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    ensemble.update_model_performance(
        model_type=model_type,
        actual_direction=actual,
        predicted_direction=predicted,
        profit_loss=update.profit_loss
    )

    return {
        'success': True,
        'model_type': update.model_type,
        'correct': actual == predicted
    }


@router.get("/performance/stats")
async def get_performance_stats():
    """
    Получить статистику производительности всех моделей.
    """
    ensemble = get_ensemble()
    stats = ensemble.get_stats()

    return {
        'model_performance': stats['model_weights'],
        'prediction_stats': {
            'total_predictions': stats['total_predictions'],
            'unanimous_ratio': (
                stats['unanimous_count'] / max(stats['total_predictions'], 1)
            ),
            'majority_ratio': (
                stats['majority_count'] / max(stats['total_predictions'], 1)
            ),
            'conflict_ratio': (
                stats['conflict_count'] / max(stats['total_predictions'], 1)
            ),
            'trade_ratio': (
                stats['trades_signaled'] / max(stats['total_predictions'], 1)
            )
        }
    }


# ============================================================================
# TRAINING ENDPOINTS
# ============================================================================

@router.post("/training/start")
async def start_training(request: TrainingRequest):
    """
    Запустить обучение модели в фоновом режиме.

    Обучение выполняется в отдельном потоке (ThreadPoolExecutor),
    не блокируя основной event loop FastAPI.
    """
    task_id = f"{request.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Проверяем, что модель не обучается
    for tid, task in _training_tasks.items():
        if task['model_type'] == request.model_type and task['status'] == 'running':
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model_type} is already training"
            )

    # Создаем задачу
    _training_tasks[task_id] = {
        'task_id': task_id,
        'model_type': request.model_type,
        'status': 'pending',
        'started_at': None,
        'completed_at': None,
        'progress': 0,
        'epochs': request.epochs,
        'current_epoch': 0,
        'error': None
    }

    # Запускаем в отдельном потоке через ThreadPoolExecutor
    # Это не блокирует FastAPI event loop
    loop = asyncio.get_event_loop()

    def run_training_sync():
        """Wrapper для запуска async функции в отдельном потоке."""
        # Создаем новый event loop для этого потока
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(
                _run_training(
                    task_id=task_id,
                    model_type=request.model_type,
                    epochs=request.epochs,
                    learning_rate=request.learning_rate,
                    symbols=request.symbols,
                    days=request.days
                )
            )
        finally:
            new_loop.close()

    # Запускаем в thread pool (fire-and-forget)
    _training_executor.submit(run_training_sync)

    logger.info(f"Training task {task_id} submitted to ThreadPoolExecutor")

    return {
        'success': True,
        'task_id': task_id,
        'message': f"Training started for {request.model_type} in separate thread"
    }


@router.get("/training/status/{task_id}")
async def get_training_status(task_id: str):
    """
    Получить статус задачи обучения.
    """
    if task_id not in _training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return _training_tasks[task_id]


@router.get("/training/list")
async def list_training_tasks():
    """
    Получить список всех задач обучения.
    """
    return {
        'tasks': list(_training_tasks.values()),
        'total': len(_training_tasks),
        'running': sum(1 for t in _training_tasks.values() if t['status'] == 'running')
    }


@router.post("/training/cancel/{task_id}")
async def cancel_training(task_id: str):
    """
    Отменить задачу обучения.
    """
    if task_id not in _training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _training_tasks[task_id]
    if task['status'] != 'running':
        raise HTTPException(status_code=400, detail="Task is not running")

    task['status'] = 'cancelled'
    task['completed_at'] = datetime.now().isoformat()

    return {'success': True, 'message': f"Task {task_id} cancelled"}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Default data paths
DEFAULT_DATA_PATH = Path("D:/PYTHON/Bot_ver3_stakan_new/data")
FALLBACK_DATA_PATH = Path(__file__).parent.parent.parent / "data"


def _get_data_path() -> Path:
    """Получить путь к данным (Windows или Linux)."""
    if DEFAULT_DATA_PATH.exists():
        return DEFAULT_DATA_PATH
    return FALLBACK_DATA_PATH


def _generate_labels_from_prices(
    mid_prices: np.ndarray,
    horizon: int = 60,
    threshold_pct: float = 0.0005
) -> np.ndarray:
    """
    Генерирует labels на основе будущих изменений цены.

    Args:
        mid_prices: Массив средних цен (mid_price)
        horizon: Горизонт предсказания (количество шагов вперед)
        threshold_pct: Порог изменения цены для определения направления (0.05% = 0.0005)

    Returns:
        labels: Массив меток (0=SELL, 1=HOLD, 2=BUY)
    """
    n_samples = len(mid_prices)
    labels = np.ones(n_samples, dtype=np.int64)  # По умолчанию HOLD (1)

    for i in range(n_samples - horizon):
        current_price = mid_prices[i]
        future_price = mid_prices[i + horizon]

        if current_price > 0:
            price_change = (future_price - current_price) / current_price

            if price_change > threshold_pct:
                labels[i] = 2  # BUY
            elif price_change < -threshold_pct:
                labels[i] = 0  # SELL
            # else: HOLD (1) - уже установлено

    # Последние horizon элементов оставляем как HOLD
    return labels


def _load_lob_data_from_parquet(
    symbol: str,
    days: int,
    num_levels: int = 20
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Загружает данные LOB из Parquet файлов.

    Args:
        symbol: Торговая пара
        days: Количество дней данных
        num_levels: Количество уровней стакана

    Returns:
        Tuple (lob_data, labels, mid_prices) или (None, None, None) если данных нет
    """
    data_path = _get_data_path()
    raw_lob_path = data_path / "raw_lob" / symbol

    if not raw_lob_path.exists():
        logger.warning(f"LOB data path not found: {raw_lob_path}")
        return None, None, None

    # Находим parquet файлы за указанный период
    cutoff_date = datetime.now() - timedelta(days=days)
    parquet_files = sorted(raw_lob_path.glob("*.parquet"))

    if not parquet_files:
        logger.warning(f"No parquet files found in {raw_lob_path}")
        return None, None, None

    logger.info(f"Found {len(parquet_files)} parquet files for {symbol}")

    all_data = []
    for pq_file in parquet_files:
        try:
            # Извлекаем дату из имени файла (SYMBOL_YYYYMMDD_HHMMSS.parquet)
            parts = pq_file.stem.split('_')
            if len(parts) >= 2:
                file_date_str = parts[1]
                try:
                    file_date = datetime.strptime(file_date_str, "%Y%m%d")
                    if file_date < cutoff_date:
                        continue
                except ValueError:
                    pass  # Если не удалось распарсить дату, загружаем файл

            df = pd.read_parquet(pq_file)
            all_data.append(df)
        except Exception as e:
            logger.warning(f"Error loading {pq_file}: {e}")
            continue

    if not all_data:
        logger.warning(f"No valid data loaded for {symbol}")
        return None, None, None

    # Объединяем все данные
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Loaded {len(combined_df)} LOB snapshots for {symbol}")

    # Преобразуем в LOB tensor (N, num_levels, 4)
    n_samples = len(combined_df)
    lob_data = np.zeros((n_samples, num_levels, 4), dtype=np.float32)

    for i in range(num_levels):
        bid_price_col = f'bid_price_{i}'
        bid_vol_col = f'bid_volume_{i}'
        ask_price_col = f'ask_price_{i}'
        ask_vol_col = f'ask_volume_{i}'

        if bid_price_col in combined_df.columns:
            lob_data[:, i, 0] = combined_df[bid_price_col].values
            lob_data[:, i, 1] = combined_df[bid_vol_col].values
            lob_data[:, i, 2] = combined_df[ask_price_col].values
            lob_data[:, i, 3] = combined_df[ask_vol_col].values

    # Извлекаем mid_prices для генерации labels
    mid_prices = combined_df['mid_price'].values

    # Генерируем labels
    labels = _generate_labels_from_prices(mid_prices, horizon=60, threshold_pct=0.0005)

    return lob_data, labels, mid_prices


def _load_feature_data_from_parquet(
    symbol: str,
    days: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Загружает feature данные из Parquet файлов (для CNN-LSTM и MPD).

    Args:
        symbol: Торговая пара
        days: Количество дней данных

    Returns:
        Tuple (features, labels) или (None, None) если данных нет
    """
    data_path = _get_data_path()

    # Проверяем несколько возможных путей
    possible_paths = [
        data_path / "feature_store" / symbol,
        data_path / "ml_training" / symbol,
        data_path / "features" / symbol,
    ]

    features_df = None
    for path in possible_paths:
        if path.exists():
            parquet_files = sorted(path.glob("*.parquet"))
            if parquet_files:
                logger.info(f"Found feature data in {path}")
                try:
                    all_dfs = [pd.read_parquet(f) for f in parquet_files]
                    features_df = pd.concat(all_dfs, ignore_index=True)
                    break
                except Exception as e:
                    logger.warning(f"Error loading from {path}: {e}")
                    continue

    if features_df is None or features_df.empty:
        logger.warning(f"No feature data found for {symbol}")
        return None, None

    # Фильтруем по дате если есть timestamp
    if 'timestamp' in features_df.columns:
        cutoff_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        features_df = features_df[features_df['timestamp'] >= cutoff_ts]

    features_df = features_df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Loaded {len(features_df)} feature samples for {symbol}")

    # Определяем feature columns (исключаем служебные)
    exclude_cols = {'timestamp', 'symbol', 'label', 'future_direction_60s',
                    'future_direction_30s', 'future_direction_15s', 'mid_price'}
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]

    if not feature_cols:
        logger.warning("No feature columns found in data")
        return None, None

    # Извлекаем features
    features = features_df[feature_cols].values.astype(np.float32)

    # Извлекаем или генерируем labels
    if 'future_direction_60s' in features_df.columns:
        labels = features_df['future_direction_60s'].values.astype(np.int64)
        # Маппинг {-1, 0, 1} -> {0, 1, 2}
        label_mapping = {-1: 0, 0: 1, 1: 2}
        labels = np.array([label_mapping.get(l, l) for l in labels], dtype=np.int64)
    elif 'mid_price' in features_df.columns:
        mid_prices = features_df['mid_price'].values
        labels = _generate_labels_from_prices(mid_prices)
    else:
        logger.warning("No labels or mid_price found, generating random labels")
        labels = np.random.randint(0, 3, size=len(features))

    # Обработка NaN
    if np.isnan(features).any():
        features = np.nan_to_num(features, nan=0.0)
    if np.isnan(labels).any():
        valid_mask = ~np.isnan(labels)
        features = features[valid_mask]
        labels = labels[valid_mask].astype(np.int64)

    return features, labels


def _create_sequences(
    data: np.ndarray,
    labels: np.ndarray,
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создает последовательности из данных.

    Args:
        data: Данные shape (N, ...) - может быть (N, features) или (N, levels, 4)
        labels: Метки shape (N,)
        sequence_length: Длина последовательности

    Returns:
        sequences: (N-seq_len+1, seq_len, ...) - последовательности
        seq_labels: (N-seq_len+1,) - метки для каждой последовательности
    """
    n_samples = len(data)
    if n_samples < sequence_length:
        raise ValueError(f"Not enough data: {n_samples} < {sequence_length}")

    num_sequences = n_samples - sequence_length + 1

    # Определяем форму выхода
    if data.ndim == 2:
        # (N, features) -> (num_seq, seq_len, features)
        sequences = np.zeros((num_sequences, sequence_length, data.shape[1]), dtype=np.float32)
    elif data.ndim == 3:
        # (N, levels, 4) -> (num_seq, seq_len, levels, 4)
        sequences = np.zeros((num_sequences, sequence_length, data.shape[1], data.shape[2]), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")

    seq_labels = np.zeros(num_sequences, dtype=np.int64)

    for i in range(num_sequences):
        sequences[i] = data[i:i + sequence_length]
        seq_labels[i] = labels[i + sequence_length - 1]

    return sequences, seq_labels


def _train_val_split(
    sequences: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.8
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Разделяет данные на train и validation.

    Args:
        sequences: Последовательности
        labels: Метки
        train_ratio: Доля обучающей выборки

    Returns:
        (train_sequences, train_labels), (val_sequences, val_labels)
    """
    n_samples = len(sequences)
    split_idx = int(n_samples * train_ratio)

    train_sequences = sequences[:split_idx]
    train_labels = labels[:split_idx]

    val_sequences = sequences[split_idx:]
    val_labels = labels[split_idx:]

    return (train_sequences, train_labels), (val_sequences, val_labels)


async def _run_training(
    task_id: str,
    model_type: str,
    epochs: int,
    learning_rate: float,
    symbols: List[str],
    days: int
):
    """
    Фоновая функция обучения с реальной загрузкой данных.

    Полная реализация обучения моделей:
    1. Загрузка данных из Parquet файлов
    2. Создание labels на основе изменений цены
    3. Создание последовательностей
    4. Train/Val split
    5. Обучение модели
    6. Регистрация в MLflow и реестре
    """
    task = _training_tasks[task_id]
    task['status'] = 'running'
    task['started_at'] = datetime.now().isoformat()

    try:
        # Import here to avoid circular imports
        from backend.ml_engine.training.multi_model_trainer import (
            create_trainer,
            MultiModelTrainer,
            ModelArchitecture,
            FeatureDataset,
            LOBDataset
        )

        logger.info(f"=" * 60)
        logger.info(f"НАЧАЛО ОБУЧЕНИЯ: {model_type}")
        logger.info(f"=" * 60)
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Symbols: {symbols}")
        logger.info(f"  Days: {days}")

        # Get architecture
        if model_type == "cnn_lstm":
            architecture = ModelArchitecture.CNN_LSTM
        elif model_type == "mpd_transformer":
            architecture = ModelArchitecture.MPD_TRANSFORMER
        elif model_type == "tlob":
            architecture = ModelArchitecture.TLOB
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # ==================== ЗАГРУЗКА ДАННЫХ ====================
        task['status'] = 'loading_data'
        task['progress'] = 5

        all_sequences = []
        all_labels = []

        for symbol in symbols:
            logger.info(f"Загрузка данных для {symbol}...")

            if model_type == "tlob":
                # Загрузка LOB данных для TLOB
                lob_data, labels, _ = _load_lob_data_from_parquet(symbol, days)

                if lob_data is not None and len(lob_data) > 60:
                    sequences, seq_labels = _create_sequences(lob_data, labels, sequence_length=60)
                    all_sequences.append(sequences)
                    all_labels.append(seq_labels)
                    logger.info(f"  {symbol}: {len(sequences)} LOB sequences")
                else:
                    logger.warning(f"  {symbol}: недостаточно LOB данных, генерируем синтетические")
                    # Генерируем синтетические данные для демонстрации
                    synthetic_lob = np.random.randn(10000, 20, 4).astype(np.float32)
                    synthetic_labels = np.random.randint(0, 3, size=10000)
                    sequences, seq_labels = _create_sequences(synthetic_lob, synthetic_labels, 60)
                    all_sequences.append(sequences)
                    all_labels.append(seq_labels)
            else:
                # Загрузка feature данных для CNN-LSTM и MPD
                features, labels = _load_feature_data_from_parquet(symbol, days)

                if features is not None and len(features) > 60:
                    sequences, seq_labels = _create_sequences(features, labels, sequence_length=60)
                    all_sequences.append(sequences)
                    all_labels.append(seq_labels)
                    logger.info(f"  {symbol}: {len(sequences)} feature sequences")
                else:
                    logger.warning(f"  {symbol}: недостаточно feature данных, генерируем синтетические")
                    # Генерируем синтетические данные
                    synthetic_features = np.random.randn(10000, 110).astype(np.float32)
                    synthetic_labels = np.random.randint(0, 3, size=10000)
                    sequences, seq_labels = _create_sequences(synthetic_features, synthetic_labels, 60)
                    all_sequences.append(sequences)
                    all_labels.append(seq_labels)

        if not all_sequences:
            raise ValueError("Не удалось загрузить данные ни для одного символа")

        # Объединяем данные
        combined_sequences = np.concatenate(all_sequences, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)

        logger.info(f"Всего данных: {len(combined_sequences)} sequences")
        logger.info(f"  Shape: {combined_sequences.shape}")
        logger.info(f"  Labels distribution: {dict(zip(*np.unique(combined_labels, return_counts=True)))}")

        # ==================== TRAIN/VAL SPLIT ====================
        task['progress'] = 10

        (train_seq, train_labels), (val_seq, val_labels) = _train_val_split(
            combined_sequences, combined_labels, train_ratio=0.8
        )

        logger.info(f"Train: {len(train_seq)} samples")
        logger.info(f"Val: {len(val_seq)} samples")

        # ==================== СОЗДАНИЕ DATALOADERS ====================
        task['progress'] = 15

        batch_size = 64

        if model_type == "tlob":
            # LOB Dataset для TLOB
            train_dataset = LOBDataset(train_seq, train_labels)
            val_dataset = LOBDataset(val_seq, val_labels)
        else:
            # Feature Dataset для CNN-LSTM и MPD
            train_dataset = FeatureDataset(train_seq, train_labels)
            val_dataset = FeatureDataset(val_seq, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")

        # ==================== СОЗДАНИЕ МОДЕЛИ ====================
        task['progress'] = 20

        if model_type == "cnn_lstm":
            from backend.ml_engine.models.hybrid_cnn_lstm_v2 import create_model_v2, ModelConfigV2
            config = ModelConfigV2(input_features=train_seq.shape[2])
            model = create_model_v2(config)
        elif model_type == "mpd_transformer":
            from backend.ml_engine.models.mpd_transformer import create_mpd_transformer, MPDConfig
            config = MPDConfig(input_features=train_seq.shape[2])
            model = create_mpd_transformer(config)
        elif model_type == "tlob":
            from backend.ml_engine.models.tlob_transformer import create_tlob_transformer, TLOBConfig
            config = TLOBConfig(
                num_levels=train_seq.shape[2],
                sequence_length=train_seq.shape[1]
            )
            model = create_tlob_transformer(config)

        # ==================== ОБУЧЕНИЕ ====================
        task['status'] = 'training'

        # Create trainer
        trainer = create_trainer(
            architecture=model_type,
            learning_rate=learning_rate,
            epochs=epochs
        )

        logger.info(f"\n{'=' * 60}")
        logger.info("ЗАПУСК ОБУЧЕНИЯ")
        logger.info(f"{'=' * 60}")

        # Training loop с отслеживанием прогресса
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_accuracy = 0.0
        metrics_history = []

        for epoch in range(epochs):
            if task['status'] == 'cancelled':
                logger.info("Training cancelled by user")
                break

            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                if model_type == "tlob":
                    inputs = batch['lob_data'].to(device)
                else:
                    inputs = batch['features'].to(device)
                labels_batch = batch['label'].to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                # Извлекаем logits
                if isinstance(outputs, dict):
                    logits = outputs.get('direction_logits', outputs.get('logits'))
                else:
                    logits = outputs

                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += labels_batch.size(0)
                train_correct += predicted.eq(labels_batch).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    if model_type == "tlob":
                        inputs = batch['lob_data'].to(device)
                    else:
                        inputs = batch['features'].to(device)
                    labels_batch = batch['label'].to(device)

                    outputs = model(inputs)

                    if isinstance(outputs, dict):
                        logits = outputs.get('direction_logits', outputs.get('logits'))
                    else:
                        logits = outputs

                    loss = criterion(logits, labels_batch)
                    val_loss += loss.item()

                    _, predicted = logits.max(1)
                    val_total += labels_batch.size(0)
                    val_correct += predicted.eq(labels_batch).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            # Update best metrics
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_accuracy:
                best_accuracy = val_acc

            metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })

            # Update task progress
            task['current_epoch'] = epoch + 1
            task['progress'] = 20 + int((epoch + 1) / epochs * 70)  # 20-90%
            task['metrics'] = {
                'train_loss': round(train_loss, 4),
                'train_acc': round(train_acc, 4),
                'val_loss': round(val_loss, 4),
                'val_acc': round(val_acc, 4),
                'best_val_loss': round(best_val_loss, 4),
                'best_accuracy': round(best_accuracy, 4)
            }

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

            # Небольшая задержка для асинхронности
            await asyncio.sleep(0.01)

        # ==================== ЗАВЕРШЕНИЕ ====================
        if task['status'] != 'cancelled':
            task['status'] = 'completed'
            task['completed_at'] = datetime.now().isoformat()
            task['progress'] = 100

            # Final metrics
            metrics = {
                'accuracy': round(best_accuracy, 4),
                'val_loss': round(best_val_loss, 4),
                'train_samples': len(train_seq),
                'val_samples': len(val_seq),
                'epochs_completed': len(metrics_history)
            }
            task['final_metrics'] = metrics

            training_params = {
                'learning_rate': learning_rate,
                'epochs': epochs,
                'symbols': symbols,
                'days': days,
                'batch_size': batch_size
            }

            logger.info(f"\n{'=' * 60}")
            logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО")
            logger.info(f"{'=' * 60}")
            logger.info(f"Best Accuracy: {best_accuracy:.4f}")
            logger.info(f"Best Val Loss: {best_val_loss:.4f}")

            # ==================== РЕГИСТРАЦИЯ МОДЕЛИ ====================
            task['progress'] = 95

            try:
                # Register in MLflow
                await trainer.register_model_mlflow(
                    model=model,
                    architecture=architecture,
                    metrics=metrics,
                    training_params=training_params
                )

                # Register in internal registry
                await trainer.register_model_registry(
                    model=model,
                    architecture=architecture,
                    metrics=metrics,
                    training_params=training_params
                )

                logger.info(f"Model {model_type} registered successfully")
                task['registered'] = True

            except Exception as reg_error:
                logger.warning(f"Model registration failed: {reg_error}")
                task['registered'] = False
                task['registration_error'] = str(reg_error)

            task['progress'] = 100

    except Exception as e:
        import traceback
        task['status'] = 'failed'
        task['error'] = str(e)
        task['traceback'] = traceback.format_exc()
        task['completed_at'] = datetime.now().isoformat()
        logger.error(f"Training failed for {model_type}: {e}")
        logger.error(traceback.format_exc())


def _get_model_display_name(model_type: str) -> str:
    """Получить отображаемое имя модели."""
    names = {
        'cnn_lstm': 'CNN-LSTM v2',
        'mpd_transformer': 'MPD Transformer',
        'tlob': 'TLOB Transformer'
    }
    return names.get(model_type, model_type)


def _get_model_description(model_type: str) -> str:
    """Получить описание модели."""
    descriptions = {
        'cnn_lstm': 'Гибридная CNN-LSTM модель для анализа временных рядов',
        'mpd_transformer': 'Vision Transformer для матричного представления данных',
        'tlob': 'Transformer для анализа структуры стакана ордеров'
    }
    return descriptions.get(model_type, '')


# ============================================================================
# WEBSOCKET FOR REAL-TIME UPDATES
# ============================================================================

# WebSocket endpoint for real-time ensemble updates
# (can be added if needed for frontend)


# ============================================================================
# REGISTER ROUTER
# ============================================================================

def register_ensemble_routes(app):
    """Регистрирует маршруты ensemble в приложении."""
    app.include_router(router)
    logger.info("Ensemble API routes registered")
