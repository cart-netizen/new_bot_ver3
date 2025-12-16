#!/usr/bin/env python3
"""
MPDTransformer - Matrix Profile Decomposition Transformer for Financial Time Series.

Архитектура основана на статье "MPDTransformer: Stock Price Prediction via
Multi-Period Decomposition and Self-Attention" с адаптацией для криптовалют.

Ключевые особенности:
1. 2D Matrix Encoding: Преобразование временного ряда в матрицу (time × features)
2. Patch Embedding: Разбиение матрицы на патчи (как в Vision Transformer)
3. Multi-Head Self-Attention: Захват кросс-факторных зависимостей
4. Multi-task Learning: Direction + Confidence + Expected Return

Путь: backend/ml_engine/models/mpd_transformer.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from backend.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MPDTransformerConfig:
    """
    Конфигурация MPDTransformer модели.

    Параметры оптимизированы для финансовых временных рядов.
    """

    # === Input параметры ===
    input_features: int = 112  # 50 LOB + 25 Candle + 37 Indicator
    sequence_length: int = 60  # Временное окно

    # === Patch Embedding ===
    patch_size_time: int = 5  # Размер патча по времени
    patch_size_feature: int = 16  # Размер патча по признакам
    embed_dim: int = 256  # Размерность эмбеддинга

    # === Transformer ===
    num_layers: int = 6  # Количество Transformer блоков
    num_heads: int = 8  # Количество голов внимания
    mlp_ratio: float = 4.0  # Множитель для FFN
    dropout: float = 0.1  # Dropout rate
    attention_dropout: float = 0.1  # Dropout в attention

    # === Classification Head ===
    num_classes: int = 3  # SELL, HOLD, BUY

    # === Regularization ===
    drop_path_rate: float = 0.1  # Stochastic Depth
    layer_norm_eps: float = 1e-6

    # === Positional Encoding ===
    use_learnable_pos_embed: bool = True
    use_cls_token: bool = True  # Использовать CLS токен как в ViT

    # === Output ===
    pool_type: str = "cls"  # cls, mean, max

    def __post_init__(self):
        """Вычисляем производные параметры."""
        # Количество патчей
        self.num_patches_time = self.sequence_length // self.patch_size_time
        self.num_patches_feature = self.input_features // self.patch_size_feature

        # Остаток признаков (если не делится нацело)
        self.feature_remainder = self.input_features % self.patch_size_feature

        if self.feature_remainder > 0:
            self.num_patches_feature += 1

        self.total_num_patches = self.num_patches_time * self.num_patches_feature

        # Размер патча (с padding)
        self.patch_dim = self.patch_size_time * self.patch_size_feature


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Patch Embedding для 2D матрицы временного ряда.

    Преобразует входную матрицу (batch, time, features) в последовательность
    патчей с эмбеддингами: (batch, num_patches, embed_dim)
    """

    def __init__(self, config: MPDTransformerConfig):
        super().__init__()

        self.config = config
        self.patch_size_time = config.patch_size_time
        self.patch_size_feature = config.patch_size_feature

        # Линейная проекция патчей
        patch_dim = config.patch_size_time * config.patch_size_feature
        self.projection = nn.Linear(patch_dim, config.embed_dim)

        # Layer Normalization
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, sequence_length, input_features)

        Returns:
            (batch, num_patches, embed_dim)
        """
        B, T, F = x.shape

        # Padding по features если нужно
        if F % self.patch_size_feature != 0:
            pad_size = self.patch_size_feature - (F % self.patch_size_feature)
            x = F.pad(x, (0, pad_size), mode='constant', value=0)
            F = F + pad_size

        # Padding по времени если нужно
        if T % self.patch_size_time != 0:
            pad_size = self.patch_size_time - (T % self.patch_size_time)
            x = F.pad(x, (0, 0, 0, pad_size), mode='constant', value=0)
            T = T + pad_size

        # Reshape в патчи
        # (B, T, F) -> (B, T/pt, pt, F/pf, pf) -> (B, T/pt, F/pf, pt, pf)
        num_patches_t = T // self.patch_size_time
        num_patches_f = F // self.patch_size_feature

        x = x.reshape(
            B,
            num_patches_t, self.patch_size_time,
            num_patches_f, self.patch_size_feature
        )
        x = x.permute(0, 1, 3, 2, 4)  # (B, num_t, num_f, pt, pf)

        # Flatten патчи
        x = x.reshape(B, num_patches_t * num_patches_f, -1)  # (B, num_patches, patch_dim)

        # Проекция
        x = self.projection(x)
        x = self.norm(x)

        return x


class PositionalEncoding(nn.Module):
    """
    Позиционное кодирование для Transformer.

    Поддерживает learnable и sinusoidal encoding.
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 1000,
        learnable: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        if learnable:
            self.pos_embed = nn.Parameter(
                torch.randn(1, max_len, embed_dim) * 0.02
            )
        else:
            # Sinusoidal encoding
            pos_embed = self._get_sinusoidal_encoding(max_len, embed_dim)
            self.register_buffer('pos_embed', pos_embed)

    def _get_sinusoidal_encoding(
        self,
        max_len: int,
        embed_dim: int
    ) -> torch.Tensor:
        """Создает sinusoidal positional encoding."""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            (batch, seq_len, embed_dim)
        """
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        return self.dropout(x)


class DropPath(nn.Module):
    """
    Drop Path (Stochastic Depth) для регуляризации.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention с поддержкой Flash Attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            return_attention: Возвращать ли attention weights

        Returns:
            output: (batch, seq_len, embed_dim)
            attention: (batch, num_heads, seq_len, seq_len) если return_attention=True
        """
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Flash Attention if available
        use_flash = hasattr(F, 'scaled_dot_product_attention') and not return_attention

        if use_flash:
            # Use Flash Attention
            dropout_p = self.attn_dropout.p if self.training else 0.0
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                is_causal=False
            )
            attn = None
        else:
            # Standard attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_dropout(attn)
            x = attn @ v

        # Reshape and project
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x, attn


class MLP(nn.Module):
    """
    MLP (Feed-Forward Network) с GELU activation.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 4)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Block с Pre-LayerNorm.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )

        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            output: (batch, seq_len, embed_dim)
            attention: optional attention weights
        """
        # Self-Attention with residual
        attn_out, attn_weights = self.attn(self.norm1(x), return_attention)
        x = x + self.drop_path(attn_out)

        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn_weights


# ============================================================================
# MAIN MODEL
# ============================================================================

class MPDTransformer(nn.Module):
    """
    MPDTransformer - Matrix Profile Decomposition Transformer.

    Архитектура:
    1. Patch Embedding: Входная матрица -> последовательность патчей
    2. Positional Encoding: Добавление позиционной информации
    3. Transformer Encoder: Self-Attention + FFN блоки
    4. Multi-task Heads: Direction, Confidence, Expected Return

    Преимущества:
    - Захватывает кросс-факторные зависимости через 2D patch embedding
    - Эффективен для данных с большим количеством признаков
    - Масштабируется лучше RNN для длинных последовательностей
    """

    def __init__(self, config: MPDTransformerConfig):
        super().__init__()

        self.config = config

        # ==================== PATCH EMBEDDING ====================
        self.patch_embed = PatchEmbedding(config)

        # ==================== CLS TOKEN ====================
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ==================== POSITIONAL ENCODING ====================
        num_positions = config.total_num_patches + (1 if config.use_cls_token else 0)
        self.pos_embed = PositionalEncoding(
            embed_dim=config.embed_dim,
            max_len=num_positions + 100,  # Extra buffer
            learnable=config.use_learnable_pos_embed,
            dropout=config.dropout
        )

        # ==================== TRANSFORMER ENCODER ====================
        # Stochastic depth decay
        dpr = [
            config.drop_path_rate * i / (config.num_layers - 1)
            for i in range(config.num_layers)
        ]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                drop_path=dpr[i],
                layer_norm_eps=config.layer_norm_eps
            )
            for i in range(config.num_layers)
        ])

        # Final LayerNorm
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

        # ==================== OUTPUT HEADS ====================

        # Direction classifier (SELL=0, HOLD=1, BUY=2)
        self.direction_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 2, config.num_classes)
        )

        # Confidence regressor (0-1)
        self.confidence_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 4, 1),
            nn.Sigmoid()
        )

        # Expected return regressor
        self.return_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 4, 1)
        )

        # ==================== INITIALIZATION ====================
        self._init_weights()

        # Logging
        model_size = self.get_model_size()
        logger.info(
            f"✓ MPDTransformer initialized: "
            f"params={model_size['total_params']:,}, "
            f"embed_dim={config.embed_dim}, "
            f"layers={config.num_layers}, "
            f"heads={config.num_heads}"
        )

    def _init_weights(self):
        """Инициализация весов."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pooling strategy для получения финального представления.

        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            (batch, embed_dim)
        """
        if self.config.pool_type == "cls" and self.config.use_cls_token:
            return x[:, 0]  # CLS token
        elif self.config.pool_type == "mean":
            return x.mean(dim=1)
        elif self.config.pool_type == "max":
            return x.max(dim=1)[0]
        else:
            # Default: mean + last
            return x.mean(dim=1) + x[:, -1]

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
                - attention_weights: list если return_attention=True
        """
        B = x.size(0)

        # ==================== PATCH EMBEDDING ====================
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # ==================== CLS TOKEN ====================
        if self.config.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # ==================== POSITIONAL ENCODING ====================
        x = self.pos_embed(x)

        # ==================== TRANSFORMER BLOCKS ====================
        attention_weights = []

        for block in self.blocks:
            x, attn = block(x, return_attention=return_attention)
            if return_attention and attn is not None:
                attention_weights.append(attn)

        # ==================== FINAL NORM + POOLING ====================
        x = self.norm(x)
        pooled = self._pool(x)  # (B, embed_dim)

        # ==================== OUTPUT HEADS ====================
        direction_logits = self.direction_head(pooled)
        confidence = self.confidence_head(pooled)
        expected_return = self.return_head(pooled)

        outputs = {
            'direction_logits': direction_logits,
            'confidence': confidence,
            'expected_return': expected_return
        }

        if return_attention:
            outputs['attention_weights'] = attention_weights

        return outputs

    def predict(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        confidence_threshold: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Inference с temperature scaling.

        Args:
            x: (batch, sequence_length, input_features)
            temperature: Temperature для softmax калибровки
            confidence_threshold: Минимальный confidence

        Returns:
            Dict с предсказаниями
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(x, return_attention=False)

            # Temperature scaling
            direction_logits = outputs['direction_logits'] / temperature
            direction_probs = F.softmax(direction_logits, dim=-1)
            direction = torch.argmax(direction_probs, dim=-1)

            confidence = outputs['confidence'].squeeze(-1)
            expected_return = outputs['expected_return'].squeeze(-1)

            # Filter by confidence
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


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_mpd_transformer(
    config: Optional[MPDTransformerConfig] = None
) -> MPDTransformer:
    """
    Создает MPDTransformer модель.

    Args:
        config: Конфигурация (None = default)

    Returns:
        MPDTransformer модель
    """
    if config is None:
        config = MPDTransformerConfig()

    return MPDTransformer(config)


def create_mpd_transformer_from_preset(preset: str = "base") -> MPDTransformer:
    """
    Создает модель из пресета.

    Пресеты:
        - tiny: Для быстрых экспериментов
        - base: Стандартная конфигурация
        - large: Для больших датасетов
    """
    if preset == "tiny":
        config = MPDTransformerConfig(
            embed_dim=128,
            num_layers=3,
            num_heads=4,
            dropout=0.2
        )

    elif preset == "base":
        config = MPDTransformerConfig(
            embed_dim=256,
            num_layers=6,
            num_heads=8,
            dropout=0.1
        )

    elif preset == "large":
        config = MPDTransformerConfig(
            embed_dim=512,
            num_layers=12,
            num_heads=16,
            dropout=0.1,
            mlp_ratio=4.0
        )

    else:
        raise ValueError(f"Unknown preset: {preset}")

    return create_mpd_transformer(config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MPDTransformer TEST")
    print("=" * 80)

    # Create config
    config = MPDTransformerConfig(
        input_features=112,
        sequence_length=60,
        embed_dim=256,
        num_layers=6,
        num_heads=8
    )

    print(f"\n📋 Config:")
    print(f"   • Input: ({config.sequence_length}, {config.input_features})")
    print(f"   • Patches: {config.num_patches_time} × {config.num_patches_feature}")
    print(f"   • Total patches: {config.total_num_patches}")
    print(f"   • Embed dim: {config.embed_dim}")
    print(f"   • Layers: {config.num_layers}")
    print(f"   • Heads: {config.num_heads}")

    # Create model
    model = create_mpd_transformer(config)

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, config.sequence_length, config.input_features)

    print(f"\n📊 Input shape: {x.shape}")

    print("\n🔄 Forward pass...")
    outputs = model(x, return_attention=False)

    print("\n📤 Outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   • {key}: {value.shape}")

    # Inference
    print("\n🎯 Inference...")
    predictions = model.predict(x, temperature=1.0, confidence_threshold=0.5)

    print("\n📊 Predictions:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"   • {key}: {value.shape}")

    # Model info
    print("\n📈 Model info:")
    size_info = model.get_model_size()
    for key, value in size_info.items():
        print(f"   • {key}: {value:,}")

    print("\n" + "=" * 80)
    print("✅ MPDTransformer test passed!")
    print("=" * 80)
