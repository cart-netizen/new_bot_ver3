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

from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import asyncio
import concurrent.futures
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from backend.core.logger import get_logger
from backend.api.websocket_manager import (
    WebSocketManager,
    get_websocket_manager,
    set_websocket_manager
)

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


class PredictionRequest(BaseModel):
    """Запрос на предсказание ensemble."""
    features: Optional[List[List[float]]] = Field(
        None,
        description="Feature данные (seq_len, num_features) для CNN-LSTM и MPD"
    )
    raw_lob: Optional[List[List[List[float]]]] = Field(
        None,
        description="LOB данные (seq_len, num_levels, 4) для TLOB"
    )
    broadcast: bool = Field(
        default=True,
        description="Отправлять ли предсказание через WebSocket"
    )


# ============================================================================
# GLOBAL ENSEMBLE INSTANCE
# ============================================================================

# Глобальный экземпляр ensemble (инициализируется при запуске)
_ensemble_instance: Optional[EnsembleConsensus] = None

# Training tasks
_training_tasks: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# WEBSOCKET CONNECTION MANAGER (SHARED)
# ============================================================================

# Используем shared WebSocket manager из websocket_manager.py
# Это позволяет разным API модулям (ensemble, hyperopt) использовать один менеджер

def get_ws_manager() -> WebSocketManager:
    """Получить экземпляр менеджера WebSocket."""
    return get_websocket_manager()


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


# Флаг для отслеживания загрузки моделей
_models_loaded: bool = False


async def _load_production_models():
    """
    Загружает production модели из Model Registry в Ensemble.

    Вызывается при первом запросе статуса ensemble.
    Это позволяет показывать корректный статус моделей на странице /strategies.
    """
    global _models_loaded

    if _models_loaded:
        return

    try:
        from dataclasses import fields
        from backend.ml_engine.inference.model_registry import get_model_registry, ModelStage

        registry = get_model_registry()
        ensemble = get_ensemble()

        # Маппинг имен моделей в registry на ModelType
        model_mapping = {
            'hybrid_cnn_lstm': ModelType.CNN_LSTM,
            'mpd_transformer': ModelType.MPD_TRANSFORMER,
            'tlob_transformer': ModelType.TLOB,
        }

        def filter_config_fields(config_class, config_dict: dict) -> dict:
            """Фильтрует config dict, оставляя только поля, существующие в dataclass."""
            known_fields = {f.name for f in fields(config_class)}
            return {k: v for k, v in config_dict.items() if k in known_fields}

        loaded_count = 0

        for registry_name, model_type in model_mapping.items():
            try:
                # Пытаемся получить production модель
                model_info = await registry.get_production_model(registry_name)

                if model_info and model_info.model_exists():
                    # Загружаем модель
                    model_path = model_info.model_path

                    # Определяем класс модели по типу
                    if model_type == ModelType.CNN_LSTM:
                        from backend.ml_engine.models.hybrid_cnn_lstm_v2 import HybridCNNLSTMv2, ModelConfigV2
                        # Загружаем checkpoint
                        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                        # Получаем конфиг из checkpoint или используем default
                        if 'config' in checkpoint:
                            # Фильтруем только известные поля (исключаем 'architecture' и др.)
                            filtered_config = filter_config_fields(ModelConfigV2, checkpoint['config'])
                            config = ModelConfigV2(**filtered_config)
                        else:
                            config = ModelConfigV2()
                        model = HybridCNNLSTMv2(config)
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            model.load_state_dict(checkpoint)

                    elif model_type == ModelType.MPD_TRANSFORMER:
                        from backend.ml_engine.models.mpd_transformer import MPDTransformer, MPDTransformerConfig
                        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                        if 'config' in checkpoint:
                            # Фильтруем только известные поля (исключаем 'architecture' и др.)
                            filtered_config = filter_config_fields(MPDTransformerConfig, checkpoint['config'])
                            logger.debug(f"MPD config from checkpoint: {filtered_config}")
                            config = MPDTransformerConfig(**filtered_config)
                        else:
                            config = MPDTransformerConfig()
                        model = MPDTransformer(config)

                        state_dict = checkpoint.get('model_state_dict', checkpoint)

                        # Проверяем и исправляем размер pos_embed если нужно
                        pos_key = 'pos_embed.pos_embed'
                        if pos_key in state_dict:
                            saved_pos = state_dict[pos_key]
                            model_pos = model.state_dict()[pos_key]
                            if saved_pos.shape != model_pos.shape:
                                logger.warning(
                                    f"MPD pos_embed shape mismatch: checkpoint {saved_pos.shape} vs model {model_pos.shape}. "
                                    f"Resizing via interpolation."
                                )
                                # Resize через интерполяцию (1, seq_len, embed_dim)
                                saved_pos = saved_pos.permute(0, 2, 1)  # (1, embed, seq)
                                resized = torch.nn.functional.interpolate(
                                    saved_pos, size=model_pos.shape[1], mode='linear', align_corners=False
                                )
                                state_dict[pos_key] = resized.permute(0, 2, 1)  # (1, seq, embed)

                        model.load_state_dict(state_dict)

                    elif model_type == ModelType.TLOB:
                        from backend.ml_engine.models.tlob_transformer import TLOBTransformer, TLOBConfig
                        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                        if 'config' in checkpoint:
                            # Фильтруем только известные поля (исключаем 'architecture' и др.)
                            filtered_config = filter_config_fields(TLOBConfig, checkpoint['config'])
                            logger.debug(f"TLOB config from checkpoint: {filtered_config}")
                            config = TLOBConfig(**filtered_config)
                        else:
                            config = TLOBConfig()
                        model = TLOBTransformer(config)

                        state_dict = checkpoint.get('model_state_dict', checkpoint)

                        # Проверяем и исправляем размер pos_embed если нужно
                        pos_key = 'pos_embed'
                        if pos_key in state_dict:
                            saved_pos = state_dict[pos_key]
                            model_pos = model.state_dict()[pos_key]
                            if saved_pos.shape != model_pos.shape:
                                logger.warning(
                                    f"TLOB pos_embed shape mismatch: checkpoint {saved_pos.shape} vs model {model_pos.shape}. "
                                    f"Resizing via interpolation."
                                )
                                # Resize через интерполяцию (1, seq_len, embed_dim)
                                saved_pos = saved_pos.permute(0, 2, 1)  # (1, embed, seq)
                                resized = torch.nn.functional.interpolate(
                                    saved_pos, size=model_pos.shape[1], mode='linear', align_corners=False
                                )
                                state_dict[pos_key] = resized.permute(0, 2, 1)  # (1, seq, embed)

                        model.load_state_dict(state_dict)

                    model.eval()  # Переключаем в режим inference

                    # Регистрируем в ensemble
                    ensemble.register_model(model_type, model)
                    loaded_count += 1

                    logger.info(
                        f"Loaded production model: {registry_name} v{model_info.metadata.version} "
                        f"-> {model_type.value}"
                    )
                else:
                    logger.debug(f"No production model found for {registry_name}")

            except Exception as e:
                logger.warning(f"Failed to load model {registry_name}: {e}")
                continue

        _models_loaded = True
        logger.info(f"Ensemble models loaded: {loaded_count}/3 production models")

    except Exception as e:
        logger.error(f"Failed to load production models: {e}")
        _models_loaded = True  # Помечаем как выполненное чтобы не пытаться снова


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
    # Автозагрузка production моделей из Model Registry при первом запросе
    await _load_production_models()

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
    # Автозагрузка production моделей из Model Registry при первом запросе
    await _load_production_models()

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
# PREDICTION ENDPOINT
# ============================================================================

@router.post("/predict", response_model=PredictionResponse)
async def get_prediction(request: PredictionRequest):
    """
    Получить предсказание ensemble.

    Отправляет данные всем активным моделям и возвращает
    консенсусное решение на основе выбранной стратегии.

    Args:
        features: Feature данные для CNN-LSTM и MPD моделей
        raw_lob: LOB данные для TLOB модели
        broadcast: Отправлять ли результат через WebSocket

    Returns:
        PredictionResponse с направлением, уверенностью и деталями
    """
    ensemble = get_ensemble()

    # Конвертируем входные данные в тензоры
    features_tensor = None
    raw_lob_tensor = None

    if request.features is not None:
        features_tensor = torch.tensor(request.features, dtype=torch.float32).unsqueeze(0)

    if request.raw_lob is not None:
        raw_lob_tensor = torch.tensor(request.raw_lob, dtype=torch.float32).unsqueeze(0)

    if features_tensor is None and raw_lob_tensor is None:
        raise HTTPException(
            status_code=400,
            detail="Either 'features' or 'raw_lob' must be provided"
        )

    try:
        # Получаем предсказание от ensemble
        prediction = await ensemble.predict(
            features=features_tensor,
            raw_lob=raw_lob_tensor
        )

        # Формируем ответ
        response_data = {
            'direction': prediction.direction.name,
            'confidence': prediction.confidence,
            'meta_confidence': prediction.meta_confidence,
            'expected_return': prediction.expected_return,
            'should_trade': prediction.should_trade,
            'consensus_type': prediction.consensus_type,
            'agreement_ratio': prediction.agreement_ratio,
            'model_predictions': {
                model_type.value: {
                    'direction': pred.direction.name,
                    'confidence': pred.confidence,
                    'probabilities': pred.probabilities.tolist() if hasattr(pred.probabilities, 'tolist') else pred.probabilities
                }
                for model_type, pred in prediction.model_predictions.items()
            }
        }

        # WebSocket broadcast
        if request.broadcast:
            try:
                ws_manager = get_ws_manager()
                await ws_manager.broadcast_prediction(
                    direction=response_data['direction'],
                    confidence=response_data['confidence'],
                    model_predictions=response_data['model_predictions'],
                    should_trade=response_data['should_trade']
                )
            except Exception as e:
                logger.debug(f"WebSocket broadcast error: {e}")

        return PredictionResponse(**response_data)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

    # Получаем старое значение для broadcast
    stats = ensemble.get_stats()
    old_enabled = stats['model_weights'].get(request.model_type, {}).get('enabled', False)

    ensemble.enable_model(model_type, request.enabled)

    # WebSocket broadcast
    try:
        ws_manager = get_ws_manager()
        await ws_manager.broadcast_status_change(
            model_type=request.model_type,
            change_type="enabled",
            old_value=old_enabled,
            new_value=request.enabled
        )
    except Exception as e:
        logger.debug(f"WebSocket broadcast error: {e}")

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

    # Получаем старое значение для broadcast
    stats = ensemble.get_stats()
    old_weight = stats['model_weights'].get(request.model_type, {}).get('weight', 0.0)

    ensemble.set_model_weight(model_type, request.weight)

    # WebSocket broadcast
    try:
        ws_manager = get_ws_manager()
        await ws_manager.broadcast_status_change(
            model_type=request.model_type,
            change_type="weight",
            old_value=old_weight,
            new_value=request.weight
        )
    except Exception as e:
        logger.debug(f"WebSocket broadcast error: {e}")

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

    # Получаем старые веса для broadcast
    stats = ensemble.get_stats()
    old_weights = {
        name: info.get('weight', 0.0)
        for name, info in stats['model_weights'].items()
    }

    updated = []
    errors = []

    for model_name, weight in weights.items():
        try:
            model_type = ModelType(model_name)
            ensemble.set_model_weight(model_type, weight)
            updated.append(model_name)
        except ValueError as e:
            errors.append({'model': model_name, 'error': str(e)})

    # WebSocket broadcast для каждой обновленной модели
    try:
        ws_manager = get_ws_manager()
        for model_name in updated:
            await ws_manager.broadcast_status_change(
                model_type=model_name,
                change_type="weight",
                old_value=old_weights.get(model_name, 0.0),
                new_value=weights[model_name]
            )
    except Exception as e:
        logger.debug(f"WebSocket broadcast error: {e}")

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

    # Получаем старую стратегию для broadcast
    old_config = ensemble.get_config()
    old_strategy = old_config.get('consensus_strategy', 'unknown')

    try:
        strategy = ConsensusStrategy(request.strategy)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy: {request.strategy}"
        )

    ensemble.set_strategy(strategy)

    # WebSocket broadcast
    try:
        ws_manager = get_ws_manager()
        await ws_manager.broadcast(
            {
                "type": "strategy_changed",
                "old_strategy": old_strategy,
                "new_strategy": request.strategy
            },
            event_type="status"
        )
    except Exception as e:
        logger.debug(f"WebSocket broadcast error: {e}")

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

    # Сохраняем старую конфигурацию для broadcast
    old_config = ensemble.get_config()
    changed_fields = {}

    if config.min_confidence_for_trade is not None:
        changed_fields['min_confidence_for_trade'] = {
            'old': old_config.get('min_confidence_for_trade'),
            'new': config.min_confidence_for_trade
        }
        ensemble.config.min_confidence_for_trade = config.min_confidence_for_trade

    if config.unanimous_threshold is not None:
        changed_fields['unanimous_threshold'] = {
            'old': old_config.get('unanimous_threshold'),
            'new': config.unanimous_threshold
        }
        ensemble.config.unanimous_threshold = config.unanimous_threshold

    if config.conflict_resolution is not None:
        changed_fields['conflict_resolution'] = {
            'old': old_config.get('conflict_resolution'),
            'new': config.conflict_resolution
        }
        ensemble.config.conflict_resolution = config.conflict_resolution

    if config.enable_adaptive_weights is not None:
        changed_fields['enable_adaptive_weights'] = {
            'old': old_config.get('enable_adaptive_weights'),
            'new': config.enable_adaptive_weights
        }
        ensemble.config.enable_adaptive_weights = config.enable_adaptive_weights

    # WebSocket broadcast
    if changed_fields:
        try:
            ws_manager = get_ws_manager()
            await ws_manager.broadcast(
                {
                    "type": "config_updated",
                    "changed_fields": changed_fields,
                    "full_config": ensemble.get_config()
                },
                event_type="status"
            )
        except Exception as e:
            logger.debug(f"WebSocket broadcast error: {e}")

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

    # Получаем старый performance score
    stats = ensemble.get_stats()
    old_score = stats['model_weights'].get(update.model_type, {}).get('performance_score', 1.0)

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

    # Получаем новый score после обновления
    new_stats = ensemble.get_stats()
    new_score = new_stats['model_weights'].get(update.model_type, {}).get('performance_score', 1.0)

    is_correct = actual == predicted

    # WebSocket broadcast
    try:
        ws_manager = get_ws_manager()
        await ws_manager.broadcast(
            {
                "type": "performance_updated",
                "model_type": update.model_type,
                "was_correct": is_correct,
                "profit_loss": update.profit_loss,
                "old_score": old_score,
                "new_score": new_score,
                "actual_direction": update.actual_direction,
                "predicted_direction": update.predicted_direction
            },
            event_type="status"
        )
    except Exception as e:
        logger.debug(f"WebSocket broadcast error: {e}")

    return {
        'success': True,
        'model_type': update.model_type,
        'correct': is_correct
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

    # ===== ВАЛИДАЦИЯ СИМВОЛОВ =====
    # Проверяем наличие данных для запрошенных символов
    data_path = _get_data_path()
    symbols_with_data = []
    symbols_without_data = []

    # Функция для получения доступных символов
    def get_available_symbols_for_model(model_type: str) -> set:
        available = set()
        if model_type == "tlob":
            raw_lob_path = data_path / "raw_lob"
            if raw_lob_path.exists():
                for symbol_dir in raw_lob_path.iterdir():
                    if symbol_dir.is_dir() and list(symbol_dir.rglob("*.parquet")):
                        available.add(symbol_dir.name)
        else:
            feature_path = data_path / "feature_store" / "offline" / "training_features"
            if feature_path.exists():
                for pq_file in feature_path.rglob("*.parquet"):
                    try:
                        df = pd.read_parquet(pq_file, columns=['symbol'])
                        available.update(df['symbol'].unique())
                    except Exception:
                        pass
        return available

    # Если символы не указаны - автоматически используем все доступные
    symbols_to_check = request.symbols
    if not symbols_to_check:
        available_symbols = get_available_symbols_for_model(request.model_type)
        if available_symbols:
            symbols_to_check = sorted(list(available_symbols))
            logger.info(f"No symbols specified, auto-selecting all available: {symbols_to_check}")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"No symbols specified and no data available for {request.model_type}"
            )

    for symbol in symbols_to_check:
        if request.model_type == "tlob":
            # Проверяем raw_lob или raw_lob_labeled
            raw_path = data_path / "raw_lob" / symbol
            labeled_path = data_path / "raw_lob_labeled" / symbol
            has_data = (
                (raw_path.exists() and list(raw_path.rglob("*.parquet"))) or
                (labeled_path.exists() and list(labeled_path.rglob("*.parquet")))
            )
        else:
            # Проверяем feature_store для CNN-LSTM и MPD
            feature_path = data_path / "feature_store" / "offline" / "training_features"
            has_data = False
            if feature_path.exists():
                for pq_file in feature_path.rglob("*.parquet"):
                    try:
                        df = pd.read_parquet(pq_file, columns=['symbol'])
                        if symbol in df['symbol'].values:
                            has_data = True
                            break
                    except Exception:
                        pass

        if has_data:
            symbols_with_data.append(symbol)
        else:
            symbols_without_data.append(symbol)

    # Если нет данных ни для одного символа - ошибка
    if not symbols_with_data:
        available_symbols = get_available_symbols_for_model(request.model_type)
        raise HTTPException(
            status_code=400,
            detail=f"No data found for symbols: {symbols_to_check}. "
                   f"Available symbols: {sorted(list(available_symbols)) if available_symbols else 'None'}"
        )

    # Предупреждаем о символах без данных
    if symbols_without_data:
        logger.warning(f"No data for symbols: {symbols_without_data}, training only with: {symbols_with_data}")

    # Используем только символы с данными
    validated_symbols = symbols_with_data

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
    # Сохраняем ссылку на main event loop для WebSocket broadcasts
    main_loop = asyncio.get_event_loop()

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
                    symbols=validated_symbols,  # Используем только символы с данными
                    days=request.days,
                    main_loop=main_loop  # Передаём main event loop
                )
            )
        finally:
            new_loop.close()

    # Запускаем в thread pool (fire-and-forget)
    _training_executor.submit(run_training_sync)

    logger.info(f"Training task {task_id} submitted to ThreadPoolExecutor")
    logger.info(f"  Training symbols: {validated_symbols}")

    return {
        'success': True,
        'task_id': task_id,
        'message': f"Training started for {request.model_type} in separate thread",
        'symbols_used': validated_symbols,
        'symbols_skipped': symbols_without_data if symbols_without_data else None
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


@router.get("/training/available-symbols")
async def get_available_symbols():
    """
    Получить список доступных символов из собранных данных.

    Проверяет наличие данных в:
    - Feature Store (для CNN-LSTM, MPD)
    - Raw LOB (для TLOB)

    Returns:
        Dict с доступными символами и preset группами
    """
    data_path = _get_data_path()

    # Символы из Feature Store (LSTM/MPD)
    feature_store_symbols = set()
    feature_store_path = data_path / "feature_store" / "offline" / "training_features"

    if feature_store_path.exists():
        for partition in feature_store_path.iterdir():
            if partition.is_dir() and partition.name.startswith("date="):
                for pq_file in partition.glob("*.parquet"):
                    try:
                        df = pd.read_parquet(pq_file, columns=['symbol'])
                        feature_store_symbols.update(df['symbol'].unique())
                    except Exception:
                        pass

    # Символы из Raw LOB (TLOB)
    raw_lob_symbols = set()
    raw_lob_path = data_path / "raw_lob"

    if raw_lob_path.exists():
        for symbol_dir in raw_lob_path.iterdir():
            if symbol_dir.is_dir() and not symbol_dir.name.startswith('.'):
                # Проверяем есть ли данные
                parquet_files = list(symbol_dir.glob("*.parquet")) + list(symbol_dir.rglob("date=*/*.parquet"))
                if parquet_files:
                    raw_lob_symbols.add(symbol_dir.name)

    # Preset группы
    preset_groups = {
        'major': ['BTCUSDT', 'ETHUSDT'],
        'l1_chains': ['SOLUSDT', 'AVAXUSDT', 'NEARUSDT', 'APTUSDT', 'SUIUSDT', 'SEIUSDT'],
        'l2_defi': ['ARBUSDT', 'OPUSDT', 'MATICUSDT', 'LINKUSDT', 'AAVEUSDT', 'UNIUSDT'],
        'meme': ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT', 'WIFUSDT'],
        'ai_gaming': ['FETUSDT', 'AGIXUSDT', 'RENDERUSDT', 'AXSUSDT', 'SANDUSDT', 'MANAUSDT'],
    }

    # Фильтруем preset группы - только символы с данными
    all_available = feature_store_symbols | raw_lob_symbols
    filtered_groups = {}
    for group_name, group_symbols in preset_groups.items():
        available_in_group = [s for s in group_symbols if s in all_available]
        if available_in_group:
            filtered_groups[group_name] = available_in_group

    # Add "all" preset with ALL available symbols (both raw_lob and feature_store)
    if all_available:
        filtered_groups['all'] = sorted(list(all_available))

    # Add "raw_lob_all" preset specifically for TLOB training
    if raw_lob_symbols:
        filtered_groups['raw_lob_all'] = sorted(list(raw_lob_symbols))

    return {
        'feature_store_symbols': sorted(list(feature_store_symbols)),
        'raw_lob_symbols': sorted(list(raw_lob_symbols)),
        'all_symbols': sorted(list(all_available)),
        'preset_groups': filtered_groups,
        'total_feature_store': len(feature_store_symbols),
        'total_raw_lob': len(raw_lob_symbols),
        'total_unique': len(all_available)
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Default data paths - используем абсолютный путь из config
from backend.config import get_project_data_path
_PROJECT_DATA_PATH = Path(get_project_data_path())


def _get_data_path() -> Path:
    """Получить путь к данным (абсолютный путь к project_root/data)."""
    return _PROJECT_DATA_PATH


def _generate_labels_from_prices(
    mid_prices: np.ndarray,
    horizon: int = 60,
    threshold_pct: float = 0.0005,
    use_dynamic_threshold: bool = True,
    volatility_window: int = 20,
    volatility_multiplier: float = 1.0  # Уменьшено с 1.5 для лучшего баланса классов
) -> np.ndarray:
    """
    Генерирует labels на основе будущих изменений цены.

    Поддерживает два режима:
    1. Фиксированный порог (threshold_pct) - для обратной совместимости
    2. Динамический порог на основе волатильности (ATR-подобный) - рекомендуется

    Args:
        mid_prices: Массив средних цен (mid_price)
        horizon: Горизонт предсказания (количество шагов вперед)
        threshold_pct: Фиксированный порог (используется если use_dynamic_threshold=False)
        use_dynamic_threshold: Использовать динамический порог на основе волатильности
        volatility_window: Окно для расчёта волатильности (по умолчанию 20)
        volatility_multiplier: Множитель волатильности для порога (1.0 = 1 sigma)

    Returns:
        labels: Массив меток (0=SELL, 1=HOLD, 2=BUY)
    """
    import pandas as pd

    n_samples = len(mid_prices)
    labels = np.ones(n_samples, dtype=np.int64)  # По умолчанию HOLD (1)

    # ===== РАСЧЁТ ДИНАМИЧЕСКИХ ПОРОГОВ =====
    if use_dynamic_threshold and n_samples > volatility_window:
        # Вычисляем returns
        returns = np.diff(mid_prices) / np.where(mid_prices[:-1] > 0, mid_prices[:-1], 1)

        # Rolling standard deviation (аналог ATR для returns)
        returns_series = pd.Series(returns)
        rolling_volatility = returns_series.rolling(
            window=volatility_window,
            min_periods=max(5, volatility_window // 4)
        ).std().fillna(returns_series.std()).values

        # Добавляем первый элемент чтобы размеры совпадали
        rolling_volatility = np.insert(rolling_volatility, 0, rolling_volatility[0] if len(rolling_volatility) > 0 else threshold_pct)

        # Минимальный порог чтобы избежать слишком чувствительных labels
        min_threshold = 0.0002  # 0.02%
        dynamic_thresholds = np.maximum(rolling_volatility * volatility_multiplier, min_threshold)

        logger.debug(f"[Labels] Dynamic thresholds: mean={np.mean(dynamic_thresholds):.6f}, "
                     f"min={np.min(dynamic_thresholds):.6f}, max={np.max(dynamic_thresholds):.6f}")
    else:
        # Фиксированный порог для всех точек
        dynamic_thresholds = np.full(n_samples, threshold_pct)

    # ===== ГЕНЕРАЦИЯ LABELS =====
    for i in range(n_samples - horizon):
        current_price = mid_prices[i]
        future_price = mid_prices[i + horizon]

        if current_price > 0:
            price_change = (future_price - current_price) / current_price
            threshold = dynamic_thresholds[i]

            if price_change > threshold:
                labels[i] = 2  # BUY
            elif price_change < -threshold:
                labels[i] = 0  # SELL
            # else: HOLD (1) - уже установлено

    # Последние horizon элементов оставляем как HOLD

    # Логируем распределение результирующих labels
    label_counts = np.bincount(labels, minlength=3)
    total = len(labels)
    logger.info(f"[Fallback Labels] Generated with {'dynamic' if use_dynamic_threshold else 'fixed'} threshold:")
    logger.info(f"  SELL: {label_counts[0]:,} ({100*label_counts[0]/total:.1f}%)")
    logger.info(f"  HOLD: {label_counts[1]:,} ({100*label_counts[1]/total:.1f}%)")
    logger.info(f"  BUY:  {label_counts[2]:,} ({100*label_counts[2]/total:.1f}%)")

    return labels


def _load_lob_data_from_parquet(
    symbol: str,
    days: int,
    num_levels: int = 20
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Загружает данные LOB из Parquet файлов.

    Порядок загрузки:
    1. Сначала пробуем загрузить labeled данные из raw_lob_labeled/
    2. Если нет - запускаем preprocessing для синхронизации labels
    3. Если preprocessing не сработал - генерируем labels из цен (fallback)

    Args:
        symbol: Торговая пара
        days: Количество дней данных
        num_levels: Количество уровней стакана

    Returns:
        Tuple (lob_data, labels, mid_prices) или (None, None, None) если данных нет
    """
    data_path = _get_data_path()
    cutoff_date = datetime.now() - timedelta(days=days)

    # ===== 1. ПОПЫТКА ЗАГРУЗКИ LABELED ДАННЫХ =====
    labeled_path = data_path / "raw_lob_labeled" / symbol
    raw_lob_path = data_path / "raw_lob" / symbol

    combined_df = None

    if labeled_path.exists():
        logger.info(f"Checking labeled LOB data: {labeled_path}")
        combined_df = _load_lob_parquet_files(labeled_path, cutoff_date, days)

        if combined_df is not None and 'future_direction_60s' in combined_df.columns:
            logger.info(f"✓ Loaded {len(combined_df)} LABELED LOB snapshots for {symbol}")
        else:
            combined_df = None  # Нет labels - пробуем другой путь

    # ===== 2. АВТОМАТИЧЕСКИЙ PREPROCESSING =====
    if combined_df is None and raw_lob_path.exists():
        logger.info(f"No labeled data found, attempting auto-preprocessing for {symbol}...")

        try:
            # Запускаем синхронизацию labels
            _run_tlob_label_preprocessing(symbol, days)

            # Пробуем загрузить снова
            if labeled_path.exists():
                combined_df = _load_lob_parquet_files(labeled_path, cutoff_date, days)

                if combined_df is not None and 'future_direction_60s' in combined_df.columns:
                    logger.info(f"✓ Auto-preprocessed {len(combined_df)} LOB snapshots for {symbol}")

        except Exception as e:
            logger.warning(f"Auto-preprocessing failed for {symbol}: {e}")

    # ===== 3. FALLBACK: ЗАГРУЗКА RAW + ГЕНЕРАЦИЯ LABELS =====
    if combined_df is None and raw_lob_path.exists():
        logger.warning(f"Loading raw LOB with generated labels (fallback) for {symbol}")
        combined_df = _load_lob_parquet_files(raw_lob_path, cutoff_date, days)

    if combined_df is None or combined_df.empty:
        logger.warning(f"No LOB data available for {symbol}")
        return None, None, None

    logger.info(f"Total LOB snapshots for {symbol}: {len(combined_df)}")

    # ===== ИЗВЛЕЧЕНИЕ MID PRICES ДЛЯ НОРМАЛИЗАЦИИ =====
    mid_prices = combined_df['mid_price'].values

    # ===== ПРЕОБРАЗОВАНИЕ В TENSOR С НОРМАЛИЗАЦИЕЙ =====
    n_samples = len(combined_df)
    lob_data = np.zeros((n_samples, num_levels, 4), dtype=np.float32)

    for i in range(num_levels):
        bid_price_col = f'bid_price_{i}'
        bid_vol_col = f'bid_volume_{i}'
        ask_price_col = f'ask_price_{i}'
        ask_vol_col = f'ask_volume_{i}'

        if bid_price_col in combined_df.columns:
            # Сырые данные
            bid_prices = combined_df[bid_price_col].values
            bid_volumes = combined_df[bid_vol_col].values
            ask_prices = combined_df[ask_price_col].values
            ask_volumes = combined_df[ask_vol_col].values

            # ===== НОРМАЛИЗАЦИЯ ДЛЯ MULTI-SYMBOL ОБУЧЕНИЯ =====
            # Цены: относительно mid_price в basis points (универсально для всех символов)
            # Это позволяет обучать на BTC ($100k) и SOL ($200) одновременно
            safe_mid = np.where(mid_prices > 0, mid_prices, 1.0)
            lob_data[:, i, 0] = (bid_prices - mid_prices) / safe_mid * 10000  # basis points
            lob_data[:, i, 2] = (ask_prices - mid_prices) / safe_mid * 10000  # basis points

            # Объёмы: log-нормализация (сжимает большие объёмы, универсально)
            lob_data[:, i, 1] = np.log1p(bid_volumes)
            lob_data[:, i, 3] = np.log1p(ask_volumes)

    logger.info(f"  LOB data normalized: prices in basis points, volumes log-transformed")

    # Используем synchronized labels если есть
    if 'future_direction_60s' in combined_df.columns:
        labels = combined_df['future_direction_60s'].values.astype(np.int64)
        # Конвертируем -1,0,1 → 0,1,2 если нужно
        if labels.min() < 0:
            labels = labels + 1
        logger.info(f"  Using synchronized labels: {np.bincount(labels)}")
    else:
        # Fallback: генерируем labels из цен
        labels = _generate_labels_from_prices(mid_prices, horizon=60, threshold_pct=0.0005)
        logger.info(f"  Using generated labels: {np.bincount(labels)}")

    return lob_data, labels, mid_prices


def _load_lob_parquet_files(
    lob_path: Path,
    cutoff_date: datetime,
    days: int
) -> Optional[pd.DataFrame]:
    """
    Загружает parquet файлы из указанной директории.

    Поддерживает структуры:
    - {symbol}/*.parquet
    - {symbol}/date=YYYY-MM-DD/*.parquet
    """
    all_data = []

    # Ищем файлы в корне и партициях
    parquet_files = list(lob_path.glob("*.parquet")) + list(lob_path.rglob("date=*/*.parquet"))

    if not parquet_files:
        return None

    for pq_file in parquet_files:
        try:
            # Проверяем дату файла
            parts = pq_file.stem.split('_')
            if len(parts) >= 2:
                try:
                    file_date = datetime.strptime(parts[1], "%Y%m%d")
                    if file_date < cutoff_date:
                        continue
                except ValueError:
                    pass

            df = pd.read_parquet(pq_file)
            all_data.append(df)

        except Exception as e:
            logger.warning(f"Error loading {pq_file}: {e}")
            continue

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    combined = combined.drop_duplicates(subset=['timestamp'], keep='first')

    return combined


def _run_tlob_label_preprocessing(symbol: str, days: int) -> bool:
    """
    Запускает синхронизацию labels для TLOB.

    Использует TLOBLabelProcessor для merge raw LOB с LSTM labels.
    """
    try:
        # Импортируем процессор
        import sys
        from pathlib import Path as PathLib

        # Добавляем путь к проекту
        project_root = PathLib(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Импортируем процессор
        from preprocessing_add_tlob_labels import TLOBLabelProcessor, TLOBLabelConfig

        # Вычисляем даты
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Конфигурация
        config = TLOBLabelConfig(
            lstm_feature_store_path=str(_get_data_path() / "feature_store" / "offline" / "training_features"),
            raw_lob_path=str(_get_data_path() / "raw_lob"),
            output_path=str(_get_data_path() / "raw_lob_labeled"),
            timestamp_tolerance_ms=2000
        )

        # Обработка
        processor = TLOBLabelProcessor(config)
        stats = processor.process(
            start_date=start_date,
            end_date=end_date,
            symbols=[symbol]
        )

        return stats.get('total_merged', 0) > 0

    except ImportError as e:
        logger.warning(f"TLOBLabelProcessor not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Label preprocessing failed: {e}")
        return False


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
    features_df = None

    # ===== 1. Пробуем загрузить из Feature Store (offline/training_features) =====
    feature_store_path = data_path / "feature_store" / "offline" / "training_features"
    if feature_store_path.exists():
        logger.info(f"Checking Feature Store: {feature_store_path}")

        # Вычисляем диапазон дат
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Собираем все партиции
        all_partition_files = []
        filtered_files = []

        for partition_dir in feature_store_path.iterdir():
            if not partition_dir.is_dir():
                continue

            # Извлекаем дату из имени партиции (date=YYYY-MM-DD)
            if partition_dir.name.startswith("date="):
                partition_date = partition_dir.name.split("=")[1]
                partition_parquets = list(partition_dir.glob("*.parquet"))
                all_partition_files.extend(partition_parquets)

                try:
                    part_dt = datetime.strptime(partition_date, "%Y-%m-%d")
                    # Фильтруем по диапазону дат
                    if start_date.date() <= part_dt.date() <= end_date.date():
                        filtered_files.extend(partition_parquets)
                except ValueError:
                    continue

        # Используем отфильтрованные файлы, или все если фильтр пустой
        parquet_files = filtered_files if filtered_files else all_partition_files

        if not filtered_files and all_partition_files:
            logger.info(f"No files in date range ({start_date.date()} to {end_date.date()}), "
                       f"loading all {len(all_partition_files)} available files")

        if parquet_files:
            logger.info(f"Found {len(parquet_files)} parquet files in Feature Store")
            try:
                all_dfs = [pd.read_parquet(f) for f in parquet_files]
                features_df = pd.concat(all_dfs, ignore_index=True)

                # Фильтруем по symbol если есть колонка
                if 'symbol' in features_df.columns:
                    features_df = features_df[features_df['symbol'] == symbol]
                    logger.info(f"Filtered for {symbol}: {len(features_df)} samples")

                if features_df.empty:
                    logger.warning(f"No data for {symbol} in Feature Store")
                    features_df = None
            except Exception as e:
                logger.warning(f"Error loading from Feature Store: {e}")
                features_df = None

    # ===== 2. Fallback: старые пути =====
    if features_df is None or features_df.empty:
        possible_paths = [
            data_path / "feature_store" / symbol,
            data_path / "ml_training" / symbol,
            data_path / "features" / symbol,
        ]

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

    # Определяем feature columns (исключаем служебные и нечисловые)
    exclude_cols = {'timestamp', 'symbol', 'label', 'future_direction_60s',
                    'future_direction_30s', 'future_direction_15s', 'mid_price',
                    'direction', 'signal', 'side', 'action'}  # Строковые колонки

    # Только числовые колонки (исключаем object, string, category)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    if not feature_cols:
        logger.warning("No numeric feature columns found in data")
        return None, None

    logger.info(f"Using {len(feature_cols)} numeric feature columns")

    # Извлекаем features
    features = features_df[feature_cols].values.astype(np.float32)

    # Извлекаем или генерируем labels
    if 'future_direction_60s' in features_df.columns:
        raw_labels = features_df['future_direction_60s'].values

        # КРИТИЧНО: Обрабатываем NaN ДО преобразования в int
        # NaN в float при astype(int64) становится некорректным числом!
        valid_mask = ~pd.isna(raw_labels)
        if not valid_mask.all():
            nan_count = (~valid_mask).sum()
            logger.warning(f"Found {nan_count} NaN labels, filtering them out")
            features = features[valid_mask]
            raw_labels = raw_labels[valid_mask]

        # Теперь безопасно преобразуем в int
        labels = raw_labels.astype(np.int64)

        # Маппинг {-1, 0, 1} -> {0, 1, 2}
        label_mapping = {-1: 0, 0: 1, 1: 2}
        labels = np.array([label_mapping.get(int(l), 1) for l in labels], dtype=np.int64)

        # Финальная проверка - все метки должны быть в [0, 2]
        invalid_labels = (labels < 0) | (labels > 2)
        if invalid_labels.any():
            logger.warning(f"Found {invalid_labels.sum()} invalid labels, setting to HOLD(1)")
            labels[invalid_labels] = 1

    elif 'mid_price' in features_df.columns:
        mid_prices = features_df['mid_price'].values
        labels = _generate_labels_from_prices(mid_prices)
    else:
        logger.warning("No labels or mid_price found, generating random labels")
        labels = np.random.randint(0, 3, size=len(features))

    # Обработка NaN в features
    if np.isnan(features).any():
        features = np.nan_to_num(features, nan=0.0)

    logger.info(f"Labels distribution: 0(SELL)={np.sum(labels==0)}, 1(HOLD)={np.sum(labels==1)}, 2(BUY)={np.sum(labels==2)}")

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
    days: int,
    main_loop: asyncio.AbstractEventLoop = None
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

    Args:
        main_loop: Main FastAPI event loop для WebSocket broadcasts
    """
    task = _training_tasks[task_id]
    task['status'] = 'running'
    task['started_at'] = datetime.now().isoformat()

    def broadcast_to_main_loop(coro):
        """Отправляет coroutine в main event loop для WebSocket broadcast."""
        try:
            if main_loop and main_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coro, main_loop)
                # Добавляем callback для логирования ошибок
                def on_done(f):
                    exc = f.exception()
                    if exc:
                        logger.warning(f"[WS] Broadcast coroutine failed: {exc}")
                future.add_done_callback(on_done)
            else:
                logger.warning(f"[WS] Main loop not available: main_loop={main_loop is not None}, running={main_loop.is_running() if main_loop else 'N/A'}")
        except Exception as e:
            logger.warning(f"[WS] Failed to schedule broadcast: {e}")

    try:
        # Import here to avoid circular imports
        from backend.ml_engine.training.multi_model_trainer import (
            create_trainer,
            MultiModelTrainer,
            ModelArchitecture,
            FeatureDataset,
            LOBDataset
        )

        logger.info("=" * 60)
        logger.info(f"НАЧАЛО ОБУЧЕНИЯ: {model_type}")
        logger.info("=" * 60)
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Symbols: {symbols}")
        logger.info(f"  Days: {days}")
        print(f"\n{'='*60}\nНАЧАЛО ОБУЧЕНИЯ: {model_type}\n{'='*60}", flush=True)  # Debug print

        # ========== WEBSOCKET: TRAINING STARTED (через main event loop) ==========
        try:
            ws_manager = get_ws_manager()
            broadcast_to_main_loop(
                ws_manager.broadcast_training_progress(
                    task_id=task_id,
                    model_type=model_type,
                    epoch=0,
                    total_epochs=epochs,
                    metrics={"status": "started"},
                    status="started"
                )
            )
        except Exception as ws_error:
            logger.debug(f"WebSocket broadcast error: {ws_error}")

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
                    # Пропускаем символ - не генерируем синтетические данные (они вызывают nan loss)
                    logger.warning(f"  {symbol}: недостаточно LOB данных ({len(lob_data) if lob_data is not None else 0} snapshots), пропускаем")
            else:
                # Загрузка feature данных для CNN-LSTM и MPD
                features, labels = _load_feature_data_from_parquet(symbol, days)

                if features is not None and len(features) > 60:
                    sequences, seq_labels = _create_sequences(features, labels, sequence_length=60)
                    all_sequences.append(sequences)
                    all_labels.append(seq_labels)
                    logger.info(f"  {symbol}: {len(sequences)} feature sequences")
                else:
                    # Пропускаем символ - не генерируем синтетические данные
                    logger.warning(f"  {symbol}: недостаточно feature данных ({len(features) if features is not None else 0} samples), пропускаем")

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
            from backend.ml_engine.models.mpd_transformer import create_mpd_transformer, MPDTransformerConfig
            config = MPDTransformerConfig(input_features=train_seq.shape[2])
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

        # ===== CLASS WEIGHTS для борьбы с дисбалансом классов =====
        # Подсчёт распределения классов
        from collections import Counter
        class_counts = Counter(train_labels.tolist() if hasattr(train_labels, 'tolist') else train_labels)
        total_samples = sum(class_counts.values())
        num_classes = 3  # SELL=0, HOLD=1, BUY=2

        # Вычисляем веса: больший вес для меньшинства
        # Формула: weight[c] = total / (num_classes * count[c])
        class_weights = []
        for c in range(num_classes):
            count = class_counts.get(c, 1)  # Избегаем деления на 0
            weight = total_samples / (num_classes * count)
            class_weights.append(weight)

        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

        # Логируем распределение и веса
        logger.info(f"\n{'=' * 50}")
        logger.info("CLASS DISTRIBUTION & WEIGHTS")
        logger.info(f"{'=' * 50}")
        for c in range(num_classes):
            label_name = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}[c]
            count = class_counts.get(c, 0)
            pct = 100.0 * count / total_samples if total_samples > 0 else 0
            logger.info(f"  {label_name}: {count:,} ({pct:.1f}%) → weight={class_weights[c]:.3f}")
        logger.info(f"{'=' * 50}\n")

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

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

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

            # ========== WEBSOCKET BROADCAST (через main event loop) ==========
            # Отправляем прогресс обучения через WebSocket
            try:
                ws_manager = get_ws_manager()
                # Логируем для диагностики
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(f"[WS] Broadcasting epoch {epoch + 1}/{epochs} to WebSocket clients")
                broadcast_to_main_loop(
                    ws_manager.broadcast_training_progress(
                        task_id=task_id,
                        model_type=model_type,
                        epoch=epoch + 1,
                        total_epochs=epochs,
                        metrics=task['metrics'],
                        status="training"
                    )
                )
            except Exception as ws_error:
                # Логируем ошибки для диагностики (было debug, теперь info)
                logger.info(f"[WS] WebSocket broadcast error: {ws_error}")

            # Небольшая задержка для асинхронности
            await asyncio.sleep(0.01)

        # ==================== ЗАВЕРШЕНИЕ ====================
        if task['status'] != 'cancelled':
            task['status'] = 'completed'
            task['completed_at'] = datetime.now().isoformat()
            task['progress'] = 100

            # Calculate precision, recall, f1 on validation set
            model.eval()
            all_preds = []
            all_labels = []

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

                    _, predicted = logits.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels_batch.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            # Calculate precision, recall, f1 (weighted average for multi-class)
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

            # Final metrics
            metrics = {
                'accuracy': round(best_accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'val_loss': round(best_val_loss, 4),
                'train_samples': len(train_seq),
                'val_samples': len(val_seq),
                'epochs_completed': len(metrics_history)
            }
            task['final_metrics'] = metrics

            logger.info(f"Final metrics: Accuracy={best_accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

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

            # ========== WEBSOCKET: TRAINING COMPLETED (через main event loop) ==========
            try:
                ws_manager = get_ws_manager()
                broadcast_to_main_loop(
                    ws_manager.broadcast_training_progress(
                        task_id=task_id,
                        model_type=model_type,
                        epoch=epochs,
                        total_epochs=epochs,
                        metrics=metrics,
                        status="completed"
                    )
                )
            except Exception as ws_error:
                logger.debug(f"WebSocket broadcast error: {ws_error}")

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

        # ========== WEBSOCKET: TRAINING FAILED (через main event loop) ==========
        try:
            ws_manager = get_ws_manager()
            broadcast_to_main_loop(
                ws_manager.broadcast_training_progress(
                    task_id=task_id,
                    model_type=model_type,
                    epoch=task.get('current_epoch', 0),
                    total_epochs=epochs,
                    metrics={"error": str(e)},
                    status="failed"
                )
            )
        except Exception:
            pass  # Игнорируем ошибки WebSocket при неудачном обучении


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

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint для real-time обновлений ensemble.

    Подключение: ws://host:port/api/ensemble/ws

    После подключения клиент может отправить JSON с подпиской:
    {"action": "subscribe", "events": ["training", "predictions", "status"]}

    Получаемые события:
    - training: Прогресс обучения моделей
    - predictions: Новые предсказания ensemble
    - status: Изменения статуса/весов моделей

    Формат сообщений (server -> client):
    {
        "event_type": "training|predictions|status",
        "timestamp": "2024-01-01T00:00:00",
        "type": "training_progress|prediction|status_change",
        ... event-specific data
    }
    """
    ws_manager = get_ws_manager()
    logger.info("[Ensemble WS] New connection attempt")
    await ws_manager.connect(websocket)
    logger.info("[Ensemble WS] Connection accepted, waiting for messages...")

    try:
        while True:
            # Получаем сообщения от клиента
            data = await websocket.receive_text()
            logger.debug(f"[Ensemble WS] Received: {data[:100]}...")

            try:
                message = json.loads(data)
                action = message.get("action")
                logger.info(f"[Ensemble WS] Action: {action}")

                if action == "subscribe":
                    # Подписка на события
                    events = message.get("events", ["all"])
                    for event_type in events:
                        if event_type in ws_manager.subscriptions:
                            ws_manager.subscriptions[event_type].add(websocket)

                    await ws_manager.send_personal_message(
                        {
                            "type": "subscription_confirmed",
                            "events": events
                        },
                        websocket
                    )

                elif action == "unsubscribe":
                    # Отписка от событий
                    events = message.get("events", [])
                    for event_type in events:
                        if event_type in ws_manager.subscriptions:
                            ws_manager.subscriptions[event_type].discard(websocket)

                    await ws_manager.send_personal_message(
                        {
                            "type": "unsubscription_confirmed",
                            "events": events
                        },
                        websocket
                    )

                elif action == "ping":
                    # Heartbeat
                    await ws_manager.send_personal_message(
                        {"type": "pong", "timestamp": datetime.now().isoformat()},
                        websocket
                    )

                elif action == "get_status":
                    # Запрос текущего статуса
                    ensemble = get_ensemble()
                    stats = ensemble.get_stats()
                    training_status = {
                        tid: {
                            "model_type": t["model_type"],
                            "status": t["status"],
                            "progress": t["progress"],
                            "current_epoch": t.get("current_epoch", 0),
                            "epochs": t.get("epochs", 0)
                        }
                        for tid, t in _training_tasks.items()
                        if t["status"] == "running"
                    }

                    await ws_manager.send_personal_message(
                        {
                            "type": "current_status",
                            "models": stats["model_weights"],
                            "active_training": training_status,
                            "total_predictions": stats["total_predictions"]
                        },
                        websocket
                    )

                else:
                    # Неизвестное действие
                    await ws_manager.send_personal_message(
                        {
                            "type": "error",
                            "message": f"Unknown action: {action}"
                        },
                        websocket
                    )

            except json.JSONDecodeError:
                await ws_manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "Invalid JSON format"
                    },
                    websocket
                )

    except WebSocketDisconnect:
        logger.info("[Ensemble WS] Client disconnected normally")
        ws_manager.disconnect(websocket)
    except asyncio.CancelledError:
        # Graceful shutdown - не логируем как ошибку
        logger.info("[Ensemble WS] Connection cancelled (shutdown)")
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"[Ensemble WS] Error in WebSocket handler: {e}", exc_info=True)
        ws_manager.disconnect(websocket)


@router.get("/ws/stats")
async def get_websocket_stats():
    """
    Получить статистику WebSocket подключений.
    """
    ws_manager = get_ws_manager()
    return {
        "total_connections": len(ws_manager.active_connections),
        "subscriptions": {
            event_type: len(subscribers)
            for event_type, subscribers in ws_manager.subscriptions.items()
        }
    }


# ============================================================================
# REGISTER ROUTER
# ============================================================================

def register_ensemble_routes(app):
    """Регистрирует маршруты ensemble в приложении."""
    app.include_router(router)
    logger.info("Ensemble API routes registered")
