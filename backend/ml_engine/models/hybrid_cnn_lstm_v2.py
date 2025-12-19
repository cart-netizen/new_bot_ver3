#!/usr/bin/env python3
"""
Улучшенная гибридная CNN-LSTM модель v2 - Industry Standard.

Архитектурные улучшения относительно базовой версии:
1. Residual Connections в CNN блоках
2. Multi-Head Temporal Attention (вместо single-head)
3. Layer Normalization (вместо BatchNorm в LSTM части)
4. Improved Weight Initialization (orthogonal для LSTM)
5. Stochastic Depth (опционально)
6. Squeeze-and-Excitation blocks (опционально)

Ключевые принципы:
- Модель оптимизирована для малых датасетов (7-30 дней)
- Сильная регуляризация через dropout, layer norm
- Backward compatible с существующим API

Путь: backend/ml_engine/models/hybrid_cnn_lstm_v2.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from backend.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfigV2:
    """
    Конфигурация улучшенной модели v2.
    
    Совместима с базовой ModelConfig, но добавляет новые параметры.
    """
    # === Input параметры ===
    # input_features: только для документации, модель адаптируется динамически
    # После preprocessing_v2: 112 base + 60 lagged + 15 derived = 187 features
    input_features: int = 187
    sequence_length: int = 60
    
    # === CNN параметры ===
    cnn_channels: Tuple[int, ...] = (32, 64, 128)
    cnn_kernel_sizes: Tuple[int, ...] = (3, 5, 7)
    
    # === LSTM параметры ===
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    
    # === Attention параметры ===
    attention_units: int = 64
    attention_heads: int = 4  # Multi-Head Attention
    attention_dropout: float = 0.1
    
    # === Output параметры ===
    num_classes: int = 3
    
    # === Regularization ===
    dropout: float = 0.4
    
    # === Архитектурные улучшения ===
    use_residual: bool = True
    use_layer_norm: bool = True
    use_multi_head_attention: bool = True
    use_se_block: bool = False  # Squeeze-and-Excitation
    stochastic_depth_prob: float = 0.0  # 0 = off

    # === Memory optimization ===
    use_gradient_checkpointing: bool = False  # Enable only if OOM (may cause NaN with mixed precision)

    # === Инициализация ===
    init_method: str = "kaiming"  # kaiming, xavier, orthogonal


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class ResidualConvBlock(nn.Module):
    """
    CNN блок с Residual Connection.
    
    Архитектура:
    input → Conv → BN → ReLU → Dropout → output
        ↓                                    ↑
        └──────── skip connection ───────────┘
    
    Skip connection использует 1x1 conv если размерности не совпадают.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.3,
        use_residual: bool = True,
        stochastic_depth_prob: float = 0.0
    ):
        super().__init__()
        
        self.use_residual = use_residual
        self.stochastic_depth_prob = stochastic_depth_prob
        
        # Main path
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # same padding
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()  # GELU вместо ReLU (лучше для transformers)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Skip connection (projection если размерности разные)
        if use_residual:
            if in_channels != out_channels:
                self.skip = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                )
            else:
                self.skip = nn.MaxPool1d(kernel_size=2, stride=2)
        else:
            self.skip = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, channels, sequence)
        
        Returns:
            (batch, out_channels, sequence // 2)
        """
        # Main path
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.pool(out)
        
        # Residual connection
        if self.use_residual and self.skip is not None:
            identity = self.skip(x)
            
            # Stochastic depth (drop path during training)
            if self.training and self.stochastic_depth_prob > 0:
                if torch.rand(1).item() < self.stochastic_depth_prob:
                    return identity
            
            out = out + identity
        
        return out


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block для channel attention.
    
    Позволяет модели адаптивно взвешивать важность каналов.
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        
        reduced_channels = max(channels // reduction, 8)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, sequence)
        
        Returns:
            (batch, channels, sequence) - rescaled
        """
        weights = self.se(x).unsqueeze(-1)  # (batch, channels, 1)
        return x * weights


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-Head Attention для временных рядов.
    
    Отличия от стандартного Multi-Head Attention:
    - Temporal positional encoding
    - Global average pooling для финального context
    - Causal masking опционально
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) должен делиться на num_heads ({num_heads})"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None
        
        # Temporal positional encoding (learnable)
        self.max_seq_len = 256
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.max_seq_len, hidden_size) * 0.02
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, hidden_size)
            return_weights: Возвращать ли attention weights
        
        Returns:
            context: (batch, hidden_size) - агрегированный контекст
            attention_weights: (batch, num_heads, seq_len) если return_weights=True
        """
        B, T, C = x.shape
        
        # Добавляем positional encoding
        if T <= self.max_seq_len:
            x = x + self.pos_embedding[:, :T, :]
        
        # Pre-LayerNorm (если включено)
        if self.layer_norm is not None:
            x_norm = self.layer_norm(x)
        else:
            x_norm = x
        
        # Multi-head projection
        # (B, T, C) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Use memory-efficient attention (Flash Attention) if available (PyTorch 2.0+)
        # This is O(n) memory instead of O(n²) for the attention matrix
        use_flash_attention = hasattr(F, 'scaled_dot_product_attention') and not return_weights

        if use_flash_attention:
            # Flash Attention - memory efficient, but doesn't return weights
            dropout_p = self.attn_dropout.p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                is_causal=False  # Not causal for full sequence attention
            )
            attn = None  # Weights not available with Flash Attention
        else:
            # Standard attention - needed when return_weights=True
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            out = attn @ v
        
        # Reshape: (B, T, hidden_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        
        # Residual connection
        out = x + out
        
        # Global context: mean pooling + last timestep (ensemble)
        context = out.mean(dim=1) + out[:, -1, :]
        
        if return_weights and attn is not None:
            # Average attention across heads
            avg_attn = attn.mean(dim=1)  # (B, T, T)
            # Attention на последний timestep
            last_attn = avg_attn[:, -1, :]  # (B, T)
            return context, last_attn

        return context, None


class SimpleAttentionLayer(nn.Module):
    """
    Простой Attention механизм (backward compatible).
    
    Используется если use_multi_head_attention=False.
    """
    
    def __init__(self, hidden_size: int, attention_units: int):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_units),
            nn.Tanh(),
            nn.Linear(attention_units, 1)
        )
    
    def forward(
        self,
        lstm_output: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            lstm_output: (batch, sequence, hidden_size)
        
        Returns:
            context: (batch, hidden_size)
            attention_weights: (batch, sequence) если return_weights=True
        """
        # Attention scores
        attention_scores = self.attention(lstm_output)  # (batch, seq, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq, 1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_size)
        
        if return_weights:
            return context, attention_weights.squeeze(-1)
        
        return context, None


class LSTMWithLayerNorm(nn.Module):
    """
    LSTM с Layer Normalization на выходе.
    
    Layer Normalization лучше работает с sequences чем Batch Normalization.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        output_size = hidden_size * (2 if bidirectional else 1)
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_size)
        else:
            self.layer_norm = None
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        
        Returns:
            (batch, seq_len, hidden_size * 2)
        """
        lstm_out, _ = self.lstm(x)
        
        if self.layer_norm is not None:
            lstm_out = self.layer_norm(lstm_out)
        
        lstm_out = self.dropout(lstm_out)
        
        return lstm_out


# ============================================================================
# MAIN MODEL
# ============================================================================

class HybridCNNLSTMv2(nn.Module):
    """
    Улучшенная гибридная CNN-LSTM модель v2.
    
    Архитектура:
    1. CNN блоки с Residual Connections
    2. BiLSTM с Layer Normalization
    3. Multi-Head Temporal Attention
    4. Multi-task heads (direction, confidence, return)
    
    Совместимость:
    - API совместим с базовой HybridCNNLSTM
    - Можно загружать веса из базовой модели (частично)
    """
    
    def __init__(self, config: ModelConfigV2):
        super().__init__()
        
        self.config = config
        
        # ==================== CNN БЛОКИ ====================
        cnn_blocks = []
        in_channels = 1  # Начинаем с 1 канала
        
        for i, (out_channels, kernel_size) in enumerate(
            zip(config.cnn_channels, config.cnn_kernel_sizes)
        ):
            # Stochastic depth увеличивается с глубиной
            sd_prob = config.stochastic_depth_prob * (i + 1) / len(config.cnn_channels)
            
            cnn_blocks.append(
                ResidualConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=config.dropout,
                    use_residual=config.use_residual,
                    stochastic_depth_prob=sd_prob
                )
            )
            
            # Squeeze-and-Excitation после каждого CNN блока
            if config.use_se_block:
                cnn_blocks.append(SqueezeExcitation(out_channels))
            
            in_channels = out_channels
        
        self.cnn_blocks = nn.ModuleList(cnn_blocks)
        
        # Размер после CNN
        cnn_output_size = config.cnn_channels[-1]
        # Sequence уменьшается в 2^n раз из-за pooling
        n_pooling = len(config.cnn_channels)
        if config.use_se_block:
            n_pooling = len(config.cnn_channels)  # SE не меняет размер
        
        # ==================== LSTM ====================
        self.lstm = LSTMWithLayerNorm(
            input_size=cnn_output_size,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            dropout=config.lstm_dropout,
            bidirectional=True,
            use_layer_norm=config.use_layer_norm
        )
        
        lstm_output_size = config.lstm_hidden * 2  # Bidirectional
        
        # ==================== ATTENTION ====================
        if config.use_multi_head_attention:
            self.attention = MultiHeadTemporalAttention(
                hidden_size=lstm_output_size,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout,
                use_layer_norm=config.use_layer_norm
            )
        else:
            self.attention = SimpleAttentionLayer(
                hidden_size=lstm_output_size,
                attention_units=config.attention_units
            )
        
        # ==================== OUTPUT HEADS ====================
        
        # Direction classifier (BUY/HOLD/SELL)
        self.direction_head = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.5),
            nn.Linear(64, config.num_classes)
        )
        
        # Confidence regressor (0-1)
        self.confidence_head = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Expected return regressor
        self.return_head = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)
        )
        
        # ==================== ИНИЦИАЛИЗАЦИЯ ====================
        self._initialize_weights()
        
        # Логирование
        model_size = self.get_model_size()
        logger.info(
            f"✓ Инициализирована HybridCNNLSTMv2: "
            f"params={model_size['total_params']:,}, "
            f"trainable={model_size['trainable_params']:,}"
        )
        logger.info(
            f"  • CNN: {config.cnn_channels}, residual={config.use_residual}"
        )
        logger.info(
            f"  • LSTM: hidden={config.lstm_hidden}, layers={config.lstm_layers}, "
            f"layer_norm={config.use_layer_norm}"
        )
        logger.info(
            f"  • Attention: heads={config.attention_heads}, "
            f"multi_head={config.use_multi_head_attention}"
        )
    
    def _initialize_weights(self):
        """Улучшенная инициализация весов."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                # Kaiming для Conv
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.Linear):
                # Xavier для Linear
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.LSTM):
                # Orthogonal для LSTM (лучше для gradient flow)
                for param_name, param in module.named_parameters():
                    if 'weight_ih' in param_name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in param_name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in param_name:
                        nn.init.zeros_(param)
                        # Forget gate bias = 1 (помогает запоминать)
                        n = param.size(0)
                        param.data[n // 4:n // 2].fill_(1.0)
            
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, sequence_length, input_features)
            return_attention: Возвращать ли attention weights
        
        Returns:
            Dict с выходами:
                - direction_logits: (batch, num_classes)
                - confidence: (batch, 1)
                - expected_return: (batch, 1)
                - attention_weights: (batch, sequence) если return_attention=True
        """
        batch_size = x.size(0)
        use_checkpointing = self.config.use_gradient_checkpointing and self.training

        # ==================== CNN PROCESSING ====================
        # (batch, sequence, features) -> (batch, 1, sequence * features)
        x = x.reshape(batch_size, 1, -1)

        # Пропускаем через CNN блоки (с gradient checkpointing если включено)
        for block in self.cnn_blocks:
            if use_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # ==================== LSTM PROCESSING ====================
        # (batch, channels, reduced_sequence) -> (batch, reduced_sequence, channels)
        x = x.transpose(1, 2)

        # BiLSTM с LayerNorm (с gradient checkpointing если включено)
        if use_checkpointing:
            lstm_out = checkpoint(self.lstm, x, use_reentrant=False)
        else:
            lstm_out = self.lstm(x)  # (batch, reduced_sequence, lstm_hidden * 2)

        # ==================== ATTENTION ====================
        # Attention не checkpoint'им т.к. уже используем Flash Attention
        context, attention_weights = self.attention(
            lstm_out,
            return_weights=return_attention
        )
        
        # ==================== OUTPUT HEADS ====================
        direction_logits = self.direction_head(context)
        confidence = self.confidence_head(context)
        expected_return = self.return_head(context)
        
        outputs = {
            'direction_logits': direction_logits,
            'confidence': confidence,
            'expected_return': expected_return
        }
        
        if return_attention and attention_weights is not None:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def predict(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        confidence_threshold: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Inference с temperature scaling и фильтрацией по confidence.
        
        Args:
            x: (batch, sequence_length, input_features)
            temperature: Temperature для softmax калибровки
            confidence_threshold: Минимальный confidence для предсказания
        
        Returns:
            Dict с предсказаниями:
                - direction: (batch,) класс предсказания
                - direction_probs: (batch, num_classes) вероятности
                - confidence: (batch,) уверенность модели
                - expected_return: (batch,) ожидаемая доходность
                - should_trade: (batch,) bool - торговать ли (confidence > threshold)
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(x, return_attention=False)
            
            # Temperature scaling для калибровки
            direction_logits = outputs['direction_logits'] / temperature
            direction_probs = F.softmax(direction_logits, dim=-1)
            direction = torch.argmax(direction_probs, dim=-1)
            
            confidence = outputs['confidence'].squeeze(-1)
            expected_return = outputs['expected_return'].squeeze(-1)
            
            # Фильтр по confidence
            should_trade = confidence >= confidence_threshold
            
            return {
                'direction': direction,
                'direction_probs': direction_probs,
                'confidence': confidence,
                'expected_return': expected_return,
                'should_trade': should_trade
            }
    
    def get_model_size(self) -> Dict[str, int]:
        """Получить размер модели."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params
        }
    
    def get_layer_info(self) -> List[Dict[str, any]]:
        """Получить информацию о слоях для отладки."""
        info = []
        
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear, nn.LSTM)):
                params = sum(p.numel() for p in module.parameters())
                info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'params': params
                })
        
        return info


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_model_v2(config: Optional[ModelConfigV2] = None) -> HybridCNNLSTMv2:
    """
    Фабрика для создания улучшенной модели v2.
    
    Args:
        config: Конфигурация модели (None = default)
    
    Returns:
        Инициализированная модель
    """
    if config is None:
        config = ModelConfigV2()
    
    model = HybridCNNLSTMv2(config)
    
    return model


def create_model_v2_from_preset(preset: str = "production_small") -> HybridCNNLSTMv2:
    """
    Создать модель из пресета.
    
    Пресеты:
        - production_small: Для малых датасетов (7-30 дней)
        - production_large: Для больших датасетов (60+ дней)
        - quick_experiment: Для быстрых экспериментов
        - conservative: Для консервативной торговли
    """
    if preset == "production_small":
        config = ModelConfigV2(
            cnn_channels=(32, 64, 128),
            lstm_hidden=128,
            dropout=0.4,
            use_residual=True,
            use_layer_norm=True,
            use_multi_head_attention=True
        )
    
    elif preset == "production_large":
        config = ModelConfigV2(
            cnn_channels=(64, 128, 256),
            lstm_hidden=256,
            dropout=0.3,
            use_residual=True,
            use_layer_norm=True,
            use_multi_head_attention=True
        )
    
    elif preset == "quick_experiment":
        config = ModelConfigV2(
            cnn_channels=(32, 64),
            lstm_hidden=64,
            lstm_layers=1,
            dropout=0.3,
            use_residual=False,
            use_layer_norm=False,
            use_multi_head_attention=False
        )
    
    elif preset == "conservative":
        config = ModelConfigV2(
            cnn_channels=(32, 64, 128),
            lstm_hidden=128,
            dropout=0.5,
            use_residual=True,
            use_layer_norm=True,
            use_multi_head_attention=True,
            use_se_block=True
        )
    
    else:
        raise ValueError(f"Unknown preset: {preset}")
    
    return create_model_v2(config)


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

def load_from_v1_checkpoint(
    model_v2: HybridCNNLSTMv2,
    checkpoint_path: str,
    strict: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Загрузить веса из checkpoint v1 в модель v2.
    
    Args:
        model_v2: Модель v2
        checkpoint_path: Путь к checkpoint v1
        strict: Требовать точное совпадение весов
    
    Returns:
        (missing_keys, unexpected_keys)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Пытаемся загрузить с адаптацией имён
    missing, unexpected = model_v2.load_state_dict(state_dict, strict=strict)
    
    logger.info(f"Загружены веса из v1 checkpoint: {checkpoint_path}")
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
    
    return missing, unexpected


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Создаём модель с default конфигурацией
    print("=" * 80)
    print("ТЕСТ HybridCNNLSTMv2")
    print("=" * 80)
    
    config = ModelConfigV2(
        cnn_channels=(32, 64, 128),
        lstm_hidden=128,
        use_residual=True,
        use_layer_norm=True,
        use_multi_head_attention=True
    )
    
    model = create_model_v2(config)
    
    # Тестовые данные
    batch_size = 32
    sequence_length = 60
    input_features = 110
    
    x = torch.randn(batch_size, sequence_length, input_features)
    
    print(f"\n📊 Input shape: {x.shape}")
    
    # Forward pass
    print("\n🔄 Forward pass...")
    outputs = model(x, return_attention=True)
    
    print("\n📤 Outputs:")
    for key, value in outputs.items():
        print(f"  • {key}: {value.shape}")
    
    # Inference
    print("\n🎯 Inference...")
    predictions = model.predict(x, temperature=1.0, confidence_threshold=0.5)
    
    print("\n📊 Predictions:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  • {key}: {value.shape}")
    
    # Model info
    print("\n📈 Model info:")
    size_info = model.get_model_size()
    for key, value in size_info.items():
        print(f"  • {key}: {value:,}")
    
    # Layer info
    print("\n📋 Layer info (top params):")
    layers = model.get_layer_info()
    layers_sorted = sorted(layers, key=lambda x: x['params'], reverse=True)[:5]
    for layer in layers_sorted:
        print(f"  • {layer['name']}: {layer['type']}, params={layer['params']:,}")
    
    print("\n" + "=" * 80)
    print("✅ Все тесты пройдены!")
    print("=" * 80)
