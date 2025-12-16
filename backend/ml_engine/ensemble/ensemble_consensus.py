#!/usr/bin/env python3
"""
Ensemble Consensus Module - объединение предсказаний нескольких моделей.

Поддерживаемые модели:
1. CNN-LSTM v2 (HybridCNNLSTMv2) - Основная модель
2. MPDTransformer - Vision Transformer для временных рядов
3. TLOB - Transformer для данных стакана

Стратегии консенсуса:
1. Weighted Voting - взвешенное голосование
2. Unanimous - единогласное решение
3. Majority - большинство голосов
4. Confidence-based - на основе уверенности

Путь: backend/ml_engine/ensemble/ensemble_consensus.py
"""

import asyncio
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import json
from pathlib import Path

from backend.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class ModelType(Enum):
    """Типы поддерживаемых моделей."""
    CNN_LSTM = "cnn_lstm"
    MPD_TRANSFORMER = "mpd_transformer"
    TLOB = "tlob"


class ConsensusStrategy(Enum):
    """Стратегии консенсуса."""
    WEIGHTED_VOTING = "weighted_voting"
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    CONFIDENCE_BASED = "confidence_based"
    ADAPTIVE = "adaptive"


class Direction(Enum):
    """Направление сигнала."""
    SELL = 0
    HOLD = 1
    BUY = 2


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelWeight:
    """Веса для отдельной модели."""
    model_type: ModelType
    weight: float = 1.0
    enabled: bool = True
    min_confidence: float = 0.3
    performance_score: float = 0.5  # Историческая производительность

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type.value,
            'weight': self.weight,
            'enabled': self.enabled,
            'min_confidence': self.min_confidence,
            'performance_score': self.performance_score
        }


@dataclass
class EnsembleConfig:
    """Конфигурация Ensemble системы."""

    # === Strategy ===
    consensus_strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_VOTING

    # === Model Weights ===
    model_weights: Dict[ModelType, ModelWeight] = field(default_factory=lambda: {
        ModelType.CNN_LSTM: ModelWeight(
            model_type=ModelType.CNN_LSTM,
            weight=0.4,
            enabled=True
        ),
        ModelType.MPD_TRANSFORMER: ModelWeight(
            model_type=ModelType.MPD_TRANSFORMER,
            weight=0.35,
            enabled=True
        ),
        ModelType.TLOB: ModelWeight(
            model_type=ModelType.TLOB,
            weight=0.25,
            enabled=True
        )
    })

    # === Thresholds ===
    min_confidence_for_trade: float = 0.5
    unanimous_threshold: float = 0.7  # Порог для unanimous
    conflict_resolution: str = "hold"  # hold, highest_confidence, weighted

    # === Adaptive ===
    enable_adaptive_weights: bool = True
    weight_update_interval_hours: int = 24
    performance_window_days: int = 7

    # === Output ===
    output_meta_confidence: bool = True  # Meta-confidence из ensemble

    def to_dict(self) -> Dict[str, Any]:
        return {
            'consensus_strategy': self.consensus_strategy.value,
            'model_weights': {
                k.value: v.to_dict() for k, v in self.model_weights.items()
            },
            'min_confidence_for_trade': self.min_confidence_for_trade,
            'unanimous_threshold': self.unanimous_threshold,
            'conflict_resolution': self.conflict_resolution,
            'enable_adaptive_weights': self.enable_adaptive_weights
        }


# ============================================================================
# PREDICTION DATA STRUCTURES
# ============================================================================

@dataclass
class ModelPrediction:
    """Предсказание одной модели."""
    model_type: ModelType
    direction: Direction
    direction_probs: np.ndarray  # (3,) - [SELL, HOLD, BUY]
    confidence: float
    expected_return: float
    timestamp: int
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type.value,
            'direction': self.direction.value,
            'direction_probs': self.direction_probs.tolist(),
            'confidence': self.confidence,
            'expected_return': self.expected_return,
            'timestamp': self.timestamp,
            'latency_ms': self.latency_ms,
            'metadata': self.metadata
        }


@dataclass
class EnsemblePrediction:
    """Результат консенсуса ensemble."""
    # === Финальное решение ===
    direction: Direction
    confidence: float
    meta_confidence: float  # Уверенность в консенсусе
    expected_return: float
    should_trade: bool

    # === Детали консенсуса ===
    consensus_type: str  # unanimous, majority, weighted, conflict
    agreement_ratio: float  # Процент согласия моделей
    direction_probs: np.ndarray  # Взвешенные вероятности

    # === Предсказания моделей ===
    model_predictions: Dict[ModelType, ModelPrediction]
    enabled_models: List[ModelType]

    # === Метаданные ===
    timestamp: int
    total_latency_ms: float
    strategy_used: ConsensusStrategy

    def to_dict(self) -> Dict[str, Any]:
        return {
            'direction': self.direction.value,
            'direction_name': self.direction.name,
            'confidence': self.confidence,
            'meta_confidence': self.meta_confidence,
            'expected_return': self.expected_return,
            'should_trade': self.should_trade,
            'consensus_type': self.consensus_type,
            'agreement_ratio': self.agreement_ratio,
            'direction_probs': self.direction_probs.tolist(),
            'model_predictions': {
                k.value: v.to_dict() for k, v in self.model_predictions.items()
            },
            'enabled_models': [m.value for m in self.enabled_models],
            'timestamp': self.timestamp,
            'total_latency_ms': self.total_latency_ms,
            'strategy_used': self.strategy_used.value
        }


# ============================================================================
# ENSEMBLE CONSENSUS
# ============================================================================

class EnsembleConsensus:
    """
    Ensemble Consensus - объединение предсказаний нескольких моделей.

    Архитектура:
    ```
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  CNN-LSTM    │  │ MPDTransformer│  │    TLOB      │
    │  (features)  │  │  (features)  │  │  (raw LOB)   │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Prediction  │  │  Prediction  │  │  Prediction  │
    │  + Confidence│  │  + Confidence│  │  + Confidence│
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ CONSENSUS MODULE │
                    │                  │
                    │ • Weighted Vote  │
                    │ • Unanimous      │
                    │ • Majority       │
                    │ • Adaptive       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  FINAL SIGNAL    │
                    │ + Meta-Confidence│
                    └──────────────────┘
    ```

    Использование:
    ```python
    consensus = EnsembleConsensus(config)

    # Добавляем модели
    consensus.register_model(ModelType.CNN_LSTM, cnn_lstm_model)
    consensus.register_model(ModelType.MPD_TRANSFORMER, mpd_model)
    consensus.register_model(ModelType.TLOB, tlob_model)

    # Получаем предсказание
    result = await consensus.predict(features, raw_lob)

    if result.should_trade:
        print(f"Signal: {result.direction.name}, Confidence: {result.confidence}")
    ```
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        """
        Args:
            config: Конфигурация ensemble
        """
        self.config = config or EnsembleConfig()

        # Зарегистрированные модели
        self._models: Dict[ModelType, Any] = {}

        # История предсказаний для адаптивных весов
        self._prediction_history: List[Dict] = []
        self._max_history = 1000

        # Статистика
        self._stats = {
            'total_predictions': 0,
            'unanimous_count': 0,
            'majority_count': 0,
            'conflict_count': 0,
            'trades_signaled': 0
        }

        # Последнее обновление весов
        self._last_weight_update: Optional[datetime] = None

        logger.info(
            f"EnsembleConsensus initialized: "
            f"strategy={self.config.consensus_strategy.value}, "
            f"models={list(self.config.model_weights.keys())}"
        )

    def register_model(
        self,
        model_type: ModelType,
        model: Any,
        weight: Optional[float] = None
    ):
        """
        Регистрирует модель в ensemble.

        Args:
            model_type: Тип модели
            model: Инстанс модели
            weight: Вес модели (опционально)
        """
        self._models[model_type] = model

        if weight is not None and model_type in self.config.model_weights:
            self.config.model_weights[model_type].weight = weight

        logger.info(
            f"Registered model: {model_type.value}, "
            f"weight={self.config.model_weights.get(model_type, ModelWeight(model_type)).weight}"
        )

    def unregister_model(self, model_type: ModelType):
        """Удаляет модель из ensemble."""
        if model_type in self._models:
            del self._models[model_type]
            logger.info(f"Unregistered model: {model_type.value}")

    def enable_model(self, model_type: ModelType, enabled: bool = True):
        """Включает/выключает модель."""
        if model_type in self.config.model_weights:
            self.config.model_weights[model_type].enabled = enabled
            logger.info(f"Model {model_type.value} enabled={enabled}")

    def set_model_weight(self, model_type: ModelType, weight: float):
        """Устанавливает вес модели."""
        if model_type in self.config.model_weights:
            self.config.model_weights[model_type].weight = max(0.0, min(1.0, weight))
            self._normalize_weights()
            logger.info(f"Model {model_type.value} weight set to {weight}")

    def set_strategy(self, strategy: ConsensusStrategy):
        """Устанавливает стратегию консенсуса."""
        self.config.consensus_strategy = strategy
        logger.info(f"Consensus strategy set to {strategy.value}")

    async def predict(
        self,
        features: Optional[torch.Tensor] = None,
        raw_lob: Optional[torch.Tensor] = None,
        model_predictions: Optional[Dict[ModelType, ModelPrediction]] = None
    ) -> EnsemblePrediction:
        """
        Получает консенсусное предсказание.

        Args:
            features: (batch, seq, features) для CNN-LSTM и MPDTransformer
            raw_lob: (batch, seq, levels, 4) для TLOB
            model_predictions: Готовые предсказания (если уже вычислены)

        Returns:
            EnsemblePrediction
        """
        start_time = datetime.now()

        # Получаем предсказания от всех моделей
        if model_predictions is None:
            model_predictions = await self._get_model_predictions(features, raw_lob)

        # Фильтруем только включенные модели
        enabled_predictions = {
            k: v for k, v in model_predictions.items()
            if self._is_model_enabled(k) and v is not None
        }

        if not enabled_predictions:
            # Нет предсказаний - возвращаем HOLD
            return self._create_hold_prediction(model_predictions, start_time)

        # Применяем стратегию консенсуса
        result = self._apply_consensus_strategy(enabled_predictions)

        # Обновляем статистику
        self._update_stats(result)

        # Вычисляем latency
        result.total_latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        return result

    async def _get_model_predictions(
        self,
        features: Optional[torch.Tensor],
        raw_lob: Optional[torch.Tensor]
    ) -> Dict[ModelType, ModelPrediction]:
        """Получает предсказания от всех моделей."""
        predictions = {}
        timestamp = int(datetime.now().timestamp() * 1000)

        # CNN-LSTM
        if ModelType.CNN_LSTM in self._models and features is not None:
            try:
                pred = await self._predict_cnn_lstm(features, timestamp)
                if pred:
                    predictions[ModelType.CNN_LSTM] = pred
            except Exception as e:
                logger.error(f"CNN-LSTM prediction error: {e}")

        # MPDTransformer
        if ModelType.MPD_TRANSFORMER in self._models and features is not None:
            try:
                pred = await self._predict_mpd_transformer(features, timestamp)
                if pred:
                    predictions[ModelType.MPD_TRANSFORMER] = pred
            except Exception as e:
                logger.error(f"MPDTransformer prediction error: {e}")

        # TLOB
        if ModelType.TLOB in self._models and raw_lob is not None:
            try:
                pred = await self._predict_tlob(raw_lob, timestamp)
                if pred:
                    predictions[ModelType.TLOB] = pred
            except Exception as e:
                logger.error(f"TLOB prediction error: {e}")

        return predictions

    async def _predict_cnn_lstm(
        self,
        features: torch.Tensor,
        timestamp: int
    ) -> Optional[ModelPrediction]:
        """Предсказание CNN-LSTM."""
        start = datetime.now()
        model = self._models[ModelType.CNN_LSTM]

        with torch.no_grad():
            outputs = model.predict(features)

        latency = (datetime.now() - start).total_seconds() * 1000

        return ModelPrediction(
            model_type=ModelType.CNN_LSTM,
            direction=Direction(outputs['direction'][0].item()),
            direction_probs=outputs['direction_probs'][0].cpu().numpy(),
            confidence=outputs['confidence'][0].item(),
            expected_return=outputs['expected_return'][0].item(),
            timestamp=timestamp,
            latency_ms=latency
        )

    async def _predict_mpd_transformer(
        self,
        features: torch.Tensor,
        timestamp: int
    ) -> Optional[ModelPrediction]:
        """Предсказание MPDTransformer."""
        start = datetime.now()
        model = self._models[ModelType.MPD_TRANSFORMER]

        with torch.no_grad():
            outputs = model.predict(features)

        latency = (datetime.now() - start).total_seconds() * 1000

        return ModelPrediction(
            model_type=ModelType.MPD_TRANSFORMER,
            direction=Direction(outputs['direction'][0].item()),
            direction_probs=outputs['direction_probs'][0].cpu().numpy(),
            confidence=outputs['confidence'][0].item(),
            expected_return=outputs['expected_return'][0].item(),
            timestamp=timestamp,
            latency_ms=latency
        )

    async def _predict_tlob(
        self,
        raw_lob: torch.Tensor,
        timestamp: int
    ) -> Optional[ModelPrediction]:
        """Предсказание TLOB."""
        start = datetime.now()
        model = self._models[ModelType.TLOB]

        with torch.no_grad():
            outputs = model.predict(raw_lob)

        latency = (datetime.now() - start).total_seconds() * 1000

        return ModelPrediction(
            model_type=ModelType.TLOB,
            direction=Direction(outputs['direction'][0].item()),
            direction_probs=outputs['direction_probs'][0].cpu().numpy(),
            confidence=outputs['confidence'][0].item(),
            expected_return=outputs['expected_return'][0].item(),
            timestamp=timestamp,
            latency_ms=latency
        )

    def _apply_consensus_strategy(
        self,
        predictions: Dict[ModelType, ModelPrediction]
    ) -> EnsemblePrediction:
        """Применяет стратегию консенсуса."""
        strategy = self.config.consensus_strategy

        if strategy == ConsensusStrategy.WEIGHTED_VOTING:
            return self._weighted_voting_consensus(predictions)
        elif strategy == ConsensusStrategy.UNANIMOUS:
            return self._unanimous_consensus(predictions)
        elif strategy == ConsensusStrategy.MAJORITY:
            return self._majority_consensus(predictions)
        elif strategy == ConsensusStrategy.CONFIDENCE_BASED:
            return self._confidence_based_consensus(predictions)
        elif strategy == ConsensusStrategy.ADAPTIVE:
            return self._adaptive_consensus(predictions)
        else:
            return self._weighted_voting_consensus(predictions)

    def _weighted_voting_consensus(
        self,
        predictions: Dict[ModelType, ModelPrediction]
    ) -> EnsemblePrediction:
        """Взвешенное голосование."""
        # Собираем веса и вероятности
        weighted_probs = np.zeros(3)
        total_weight = 0.0
        weighted_confidence = 0.0
        weighted_return = 0.0

        for model_type, pred in predictions.items():
            weight = self.config.model_weights[model_type].weight
            weighted_probs += pred.direction_probs * weight
            weighted_confidence += pred.confidence * weight
            weighted_return += pred.expected_return * weight
            total_weight += weight

        if total_weight > 0:
            weighted_probs /= total_weight
            weighted_confidence /= total_weight
            weighted_return /= total_weight

        # Финальное направление
        final_direction = Direction(np.argmax(weighted_probs))

        # Agreement ratio
        directions = [p.direction for p in predictions.values()]
        agreement = sum(1 for d in directions if d == final_direction) / len(directions)

        # Meta-confidence
        meta_confidence = weighted_confidence * agreement

        # Consensus type
        if agreement == 1.0:
            consensus_type = "unanimous"
        elif agreement >= 0.5:
            consensus_type = "majority"
        else:
            consensus_type = "weighted"

        # Should trade
        should_trade = (
            meta_confidence >= self.config.min_confidence_for_trade and
            final_direction != Direction.HOLD
        )

        return EnsemblePrediction(
            direction=final_direction,
            confidence=weighted_confidence,
            meta_confidence=meta_confidence,
            expected_return=weighted_return,
            should_trade=should_trade,
            consensus_type=consensus_type,
            agreement_ratio=agreement,
            direction_probs=weighted_probs,
            model_predictions=predictions,
            enabled_models=list(predictions.keys()),
            timestamp=int(datetime.now().timestamp() * 1000),
            total_latency_ms=0.0,
            strategy_used=ConsensusStrategy.WEIGHTED_VOTING
        )

    def _unanimous_consensus(
        self,
        predictions: Dict[ModelType, ModelPrediction]
    ) -> EnsemblePrediction:
        """Единогласный консенсус."""
        directions = [p.direction for p in predictions.values()]
        confidences = [p.confidence for p in predictions.values()]

        # Проверяем единогласие
        unique_directions = set(directions)

        if len(unique_directions) == 1 and min(confidences) >= self.config.unanimous_threshold:
            # Единогласие с высокой уверенностью
            final_direction = directions[0]
            consensus_type = "unanimous"
            agreement = 1.0
        else:
            # Нет единогласия - HOLD
            final_direction = Direction.HOLD
            consensus_type = "conflict"
            agreement = 1.0 / len(unique_directions) if unique_directions else 0.0

        # Средние метрики
        avg_confidence = np.mean(confidences)
        avg_return = np.mean([p.expected_return for p in predictions.values()])

        # Взвешенные вероятности
        weighted_probs = np.mean(
            [p.direction_probs for p in predictions.values()],
            axis=0
        )

        should_trade = (
            consensus_type == "unanimous" and
            final_direction != Direction.HOLD and
            avg_confidence >= self.config.min_confidence_for_trade
        )

        return EnsemblePrediction(
            direction=final_direction,
            confidence=avg_confidence,
            meta_confidence=avg_confidence * agreement,
            expected_return=avg_return,
            should_trade=should_trade,
            consensus_type=consensus_type,
            agreement_ratio=agreement,
            direction_probs=weighted_probs,
            model_predictions=predictions,
            enabled_models=list(predictions.keys()),
            timestamp=int(datetime.now().timestamp() * 1000),
            total_latency_ms=0.0,
            strategy_used=ConsensusStrategy.UNANIMOUS
        )

    def _majority_consensus(
        self,
        predictions: Dict[ModelType, ModelPrediction]
    ) -> EnsemblePrediction:
        """Консенсус большинства."""
        directions = [p.direction for p in predictions.values()]

        # Подсчет голосов
        vote_counts = {Direction.SELL: 0, Direction.HOLD: 0, Direction.BUY: 0}
        for d in directions:
            vote_counts[d] += 1

        # Находим большинство
        max_votes = max(vote_counts.values())
        winners = [d for d, v in vote_counts.items() if v == max_votes]

        if len(winners) == 1:
            final_direction = winners[0]
            consensus_type = "majority"
        else:
            # Tie - используем confidence
            if self.config.conflict_resolution == "highest_confidence":
                best_pred = max(predictions.values(), key=lambda p: p.confidence)
                final_direction = best_pred.direction
            else:
                final_direction = Direction.HOLD
            consensus_type = "tie_resolved"

        agreement = vote_counts[final_direction] / len(directions)

        # Метрики
        avg_confidence = np.mean([p.confidence for p in predictions.values()])
        avg_return = np.mean([p.expected_return for p in predictions.values()])
        weighted_probs = np.mean(
            [p.direction_probs for p in predictions.values()],
            axis=0
        )

        should_trade = (
            agreement >= 0.5 and
            final_direction != Direction.HOLD and
            avg_confidence >= self.config.min_confidence_for_trade
        )

        return EnsemblePrediction(
            direction=final_direction,
            confidence=avg_confidence,
            meta_confidence=avg_confidence * agreement,
            expected_return=avg_return,
            should_trade=should_trade,
            consensus_type=consensus_type,
            agreement_ratio=agreement,
            direction_probs=weighted_probs,
            model_predictions=predictions,
            enabled_models=list(predictions.keys()),
            timestamp=int(datetime.now().timestamp() * 1000),
            total_latency_ms=0.0,
            strategy_used=ConsensusStrategy.MAJORITY
        )

    def _confidence_based_consensus(
        self,
        predictions: Dict[ModelType, ModelPrediction]
    ) -> EnsemblePrediction:
        """Консенсус на основе уверенности."""
        # Выбираем модель с максимальной уверенностью
        best_pred = max(predictions.values(), key=lambda p: p.confidence)
        final_direction = best_pred.direction

        # Проверяем согласие других моделей
        directions = [p.direction for p in predictions.values()]
        agreement = sum(1 for d in directions if d == final_direction) / len(directions)

        # Взвешенные метрики по confidence
        total_conf = sum(p.confidence for p in predictions.values())
        if total_conf > 0:
            weighted_return = sum(
                p.expected_return * p.confidence for p in predictions.values()
            ) / total_conf
            weighted_probs = sum(
                p.direction_probs * p.confidence for p in predictions.values()
            ) / total_conf
        else:
            weighted_return = best_pred.expected_return
            weighted_probs = best_pred.direction_probs

        meta_confidence = best_pred.confidence * agreement

        should_trade = (
            meta_confidence >= self.config.min_confidence_for_trade and
            final_direction != Direction.HOLD
        )

        return EnsemblePrediction(
            direction=final_direction,
            confidence=best_pred.confidence,
            meta_confidence=meta_confidence,
            expected_return=weighted_return,
            should_trade=should_trade,
            consensus_type="confidence_leader",
            agreement_ratio=agreement,
            direction_probs=weighted_probs,
            model_predictions=predictions,
            enabled_models=list(predictions.keys()),
            timestamp=int(datetime.now().timestamp() * 1000),
            total_latency_ms=0.0,
            strategy_used=ConsensusStrategy.CONFIDENCE_BASED
        )

    def _adaptive_consensus(
        self,
        predictions: Dict[ModelType, ModelPrediction]
    ) -> EnsemblePrediction:
        """Адаптивный консенсус на основе исторической производительности."""
        # Используем performance_score для весов
        weighted_probs = np.zeros(3)
        total_weight = 0.0

        for model_type, pred in predictions.items():
            mw = self.config.model_weights[model_type]
            # Adaptive weight = base_weight * performance_score
            adaptive_weight = mw.weight * mw.performance_score
            weighted_probs += pred.direction_probs * adaptive_weight
            total_weight += adaptive_weight

        if total_weight > 0:
            weighted_probs /= total_weight

        final_direction = Direction(np.argmax(weighted_probs))

        # Остальные метрики как в weighted voting
        directions = [p.direction for p in predictions.values()]
        agreement = sum(1 for d in directions if d == final_direction) / len(directions)

        avg_confidence = np.mean([p.confidence for p in predictions.values()])
        avg_return = np.mean([p.expected_return for p in predictions.values()])

        meta_confidence = avg_confidence * agreement

        should_trade = (
            meta_confidence >= self.config.min_confidence_for_trade and
            final_direction != Direction.HOLD
        )

        return EnsemblePrediction(
            direction=final_direction,
            confidence=avg_confidence,
            meta_confidence=meta_confidence,
            expected_return=avg_return,
            should_trade=should_trade,
            consensus_type="adaptive",
            agreement_ratio=agreement,
            direction_probs=weighted_probs,
            model_predictions=predictions,
            enabled_models=list(predictions.keys()),
            timestamp=int(datetime.now().timestamp() * 1000),
            total_latency_ms=0.0,
            strategy_used=ConsensusStrategy.ADAPTIVE
        )

    def _create_hold_prediction(
        self,
        predictions: Dict[ModelType, ModelPrediction],
        start_time: datetime
    ) -> EnsemblePrediction:
        """Создает HOLD предсказание когда нет данных."""
        return EnsemblePrediction(
            direction=Direction.HOLD,
            confidence=0.0,
            meta_confidence=0.0,
            expected_return=0.0,
            should_trade=False,
            consensus_type="no_models",
            agreement_ratio=0.0,
            direction_probs=np.array([0.0, 1.0, 0.0]),
            model_predictions=predictions,
            enabled_models=[],
            timestamp=int(datetime.now().timestamp() * 1000),
            total_latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
            strategy_used=self.config.consensus_strategy
        )

    def _is_model_enabled(self, model_type: ModelType) -> bool:
        """Проверяет, включена ли модель."""
        if model_type not in self.config.model_weights:
            return False
        return self.config.model_weights[model_type].enabled

    def _normalize_weights(self):
        """Нормализует веса моделей."""
        total = sum(
            mw.weight for mw in self.config.model_weights.values()
            if mw.enabled
        )

        if total > 0:
            for mw in self.config.model_weights.values():
                if mw.enabled:
                    mw.weight = mw.weight / total

    def _update_stats(self, result: EnsemblePrediction):
        """Обновляет статистику."""
        self._stats['total_predictions'] += 1

        if result.consensus_type == "unanimous":
            self._stats['unanimous_count'] += 1
        elif result.consensus_type in ["majority", "weighted"]:
            self._stats['majority_count'] += 1
        elif result.consensus_type == "conflict":
            self._stats['conflict_count'] += 1

        if result.should_trade:
            self._stats['trades_signaled'] += 1

    def update_model_performance(
        self,
        model_type: ModelType,
        actual_direction: Direction,
        predicted_direction: Direction,
        profit_loss: float
    ):
        """
        Обновляет производительность модели для адаптивных весов.

        Args:
            model_type: Тип модели
            actual_direction: Реальное направление
            predicted_direction: Предсказанное направление
            profit_loss: Прибыль/убыток
        """
        if model_type not in self.config.model_weights:
            return

        mw = self.config.model_weights[model_type]

        # Простая обновление score
        correct = actual_direction == predicted_direction
        alpha = 0.1  # Learning rate

        if correct:
            mw.performance_score = mw.performance_score * (1 - alpha) + 1.0 * alpha
        else:
            mw.performance_score = mw.performance_score * (1 - alpha) + 0.0 * alpha

        # Сохраняем историю
        self._prediction_history.append({
            'model_type': model_type.value,
            'correct': correct,
            'profit_loss': profit_loss,
            'timestamp': datetime.now().isoformat()
        })

        # Ограничиваем историю
        if len(self._prediction_history) > self._max_history:
            self._prediction_history = self._prediction_history[-self._max_history:]

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику."""
        return {
            **self._stats,
            'model_weights': {
                k.value: {
                    'weight': v.weight,
                    'enabled': v.enabled,
                    'performance_score': v.performance_score
                }
                for k, v in self.config.model_weights.items()
            },
            'strategy': self.config.consensus_strategy.value,
            'registered_models': [m.value for m in self._models.keys()]
        }

    def get_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию."""
        return self.config.to_dict()

    async def save_config(self, path: str):
        """Сохраняет конфигурацию."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Ensemble config saved to {path}")

    async def load_config(self, path: str):
        """Загружает конфигурацию."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.config.consensus_strategy = ConsensusStrategy(data['consensus_strategy'])
        self.config.min_confidence_for_trade = data['min_confidence_for_trade']
        self.config.unanimous_threshold = data['unanimous_threshold']

        for model_name, weight_data in data['model_weights'].items():
            model_type = ModelType(model_name)
            if model_type in self.config.model_weights:
                self.config.model_weights[model_type].weight = weight_data['weight']
                self.config.model_weights[model_type].enabled = weight_data['enabled']

        logger.info(f"Ensemble config loaded from {path}")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_ensemble_consensus(
    strategy: str = "weighted_voting",
    cnn_lstm_weight: float = 0.4,
    mpd_weight: float = 0.35,
    tlob_weight: float = 0.25
) -> EnsembleConsensus:
    """
    Создает EnsembleConsensus с заданными параметрами.

    Args:
        strategy: Стратегия консенсуса
        cnn_lstm_weight: Вес CNN-LSTM
        mpd_weight: Вес MPDTransformer
        tlob_weight: Вес TLOB

    Returns:
        EnsembleConsensus
    """
    config = EnsembleConfig(
        consensus_strategy=ConsensusStrategy(strategy),
        model_weights={
            ModelType.CNN_LSTM: ModelWeight(
                model_type=ModelType.CNN_LSTM,
                weight=cnn_lstm_weight,
                enabled=True
            ),
            ModelType.MPD_TRANSFORMER: ModelWeight(
                model_type=ModelType.MPD_TRANSFORMER,
                weight=mpd_weight,
                enabled=True
            ),
            ModelType.TLOB: ModelWeight(
                model_type=ModelType.TLOB,
                weight=tlob_weight,
                enabled=True
            )
        }
    )

    return EnsembleConsensus(config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        print("=" * 80)
        print("ENSEMBLE CONSENSUS TEST")
        print("=" * 80)

        # Create ensemble
        ensemble = create_ensemble_consensus(
            strategy="weighted_voting",
            cnn_lstm_weight=0.4,
            mpd_weight=0.35,
            tlob_weight=0.25
        )

        # Симулируем предсказания моделей
        mock_predictions = {
            ModelType.CNN_LSTM: ModelPrediction(
                model_type=ModelType.CNN_LSTM,
                direction=Direction.BUY,
                direction_probs=np.array([0.1, 0.2, 0.7]),
                confidence=0.75,
                expected_return=0.02,
                timestamp=1700000000000
            ),
            ModelType.MPD_TRANSFORMER: ModelPrediction(
                model_type=ModelType.MPD_TRANSFORMER,
                direction=Direction.BUY,
                direction_probs=np.array([0.15, 0.25, 0.6]),
                confidence=0.68,
                expected_return=0.015,
                timestamp=1700000000000
            ),
            ModelType.TLOB: ModelPrediction(
                model_type=ModelType.TLOB,
                direction=Direction.HOLD,
                direction_probs=np.array([0.2, 0.5, 0.3]),
                confidence=0.55,
                expected_return=0.005,
                timestamp=1700000000000
            )
        }

        # Получаем консенсус
        result = await ensemble.predict(model_predictions=mock_predictions)

        print(f"\n📊 Ensemble Prediction:")
        print(f"   • Direction: {result.direction.name}")
        print(f"   • Confidence: {result.confidence:.3f}")
        print(f"   • Meta-Confidence: {result.meta_confidence:.3f}")
        print(f"   • Agreement: {result.agreement_ratio:.2%}")
        print(f"   • Consensus Type: {result.consensus_type}")
        print(f"   • Should Trade: {result.should_trade}")

        print(f"\n📈 Model Predictions:")
        for model, pred in result.model_predictions.items():
            print(f"   • {model.value}: {pred.direction.name} ({pred.confidence:.2f})")

        print(f"\n📉 Stats:")
        stats = ensemble.get_stats()
        for key, value in stats.items():
            if not isinstance(value, dict):
                print(f"   • {key}: {value}")

        print("\n" + "=" * 80)
        print("✅ Test completed!")
        print("=" * 80)

    asyncio.run(main())
