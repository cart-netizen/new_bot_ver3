"""
Ensemble Module - объединение нескольких ML моделей.

Компоненты:
- EnsembleConsensus: Консенсус предсказаний
- ModelType: Типы моделей
- ConsensusStrategy: Стратегии консенсуса

Путь: backend/ml_engine/ensemble/__init__.py
"""

from backend.ml_engine.ensemble.ensemble_consensus import (
    EnsembleConsensus,
    EnsembleConfig,
    EnsemblePrediction,
    ModelPrediction,
    ModelWeight,
    ModelType,
    ConsensusStrategy,
    Direction,
    create_ensemble_consensus
)

__all__ = [
    'EnsembleConsensus',
    'EnsembleConfig',
    'EnsemblePrediction',
    'ModelPrediction',
    'ModelWeight',
    'ModelType',
    'ConsensusStrategy',
    'Direction',
    'create_ensemble_consensus'
]
