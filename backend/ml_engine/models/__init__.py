"""
Улучшенные ML модели для трейдинга.

Содержит:
- HybridCNNLSTMv2: улучшенная гибридная модель
- ModelConfigV2: конфигурация модели
- Фабричные функции для создания моделей
"""

from .hybrid_cnn_lstm_v2 import (
    HybridCNNLSTMv2,
    ModelConfigV2,
    ResidualConvBlock,
    MultiHeadTemporalAttention,
    LSTMWithLayerNorm,
    create_model_v2,
    create_model_v2_from_preset,
    load_from_v1_checkpoint
)

__all__ = [
    'HybridCNNLSTMv2',
    'ModelConfigV2',
    'ResidualConvBlock',
    'MultiHeadTemporalAttention',
    'LSTMWithLayerNorm',
    'create_model_v2',
    'create_model_v2_from_preset',
    'load_from_v1_checkpoint'
]
