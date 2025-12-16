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

from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio

from backend.core.logger import get_logger
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
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Запустить обучение модели в фоновом режиме.
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

    # Запускаем в фоне
    background_tasks.add_task(
        _run_training,
        task_id=task_id,
        model_type=request.model_type,
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        symbols=request.symbols,
        days=request.days
    )

    return {
        'success': True,
        'task_id': task_id,
        'message': f"Training started for {request.model_type}"
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

async def _run_training(
    task_id: str,
    model_type: str,
    epochs: int,
    learning_rate: float,
    symbols: List[str],
    days: int
):
    """Фоновая функция обучения."""
    task = _training_tasks[task_id]
    task['status'] = 'running'
    task['started_at'] = datetime.now().isoformat()

    try:
        # Import here to avoid circular imports
        from backend.ml_engine.training.multi_model_trainer import (
            create_trainer,
            ModelArchitecture
        )
        from backend.ml_engine.models.mpd_transformer import create_mpd_transformer
        from backend.ml_engine.models.tlob_transformer import create_tlob_transformer
        from backend.ml_engine.models.hybrid_cnn_lstm_v2 import create_model_v2

        # Create model based on type
        if model_type == "cnn_lstm":
            model = create_model_v2()
            architecture = ModelArchitecture.CNN_LSTM
        elif model_type == "mpd_transformer":
            model = create_mpd_transformer()
            architecture = ModelArchitecture.MPD_TRANSFORMER
        elif model_type == "tlob":
            model = create_tlob_transformer()
            architecture = ModelArchitecture.TLOB
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create trainer
        trainer = create_trainer(
            architecture=model_type,
            learning_rate=learning_rate,
            epochs=epochs
        )

        # TODO: Load actual data
        # For now, simulating training
        for epoch in range(epochs):
            if task['status'] == 'cancelled':
                break

            task['current_epoch'] = epoch + 1
            task['progress'] = int((epoch + 1) / epochs * 100)

            # Simulate epoch
            await asyncio.sleep(0.1)

        task['status'] = 'completed'
        task['completed_at'] = datetime.now().isoformat()
        task['progress'] = 100

    except Exception as e:
        task['status'] = 'failed'
        task['error'] = str(e)
        task['completed_at'] = datetime.now().isoformat()
        logger.error(f"Training failed for {model_type}: {e}")


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
