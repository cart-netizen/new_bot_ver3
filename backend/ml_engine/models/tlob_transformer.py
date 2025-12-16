#!/usr/bin/env python3
"""
TLOB - Transformer for Limit Order Book.

Специализированная архитектура для анализа данных стакана ордеров (LOB).
Основана на исследовании "Transformers for Limit Order Books" (2023).

Ключевые особенности:
1. Spatial CNN: Извлечение локальных паттернов из структуры стакана
2. Temporal Transformer: Захват долгосрочных временных зависимостей
3. Multi-scale Analysis: Анализ на разных временных масштабах
4. HFT-optimized: Оптимизирован для высокочастотных предсказаний

Входные данные:
- Raw LOB tensor: (batch, sequence, levels, 4)
- 4 канала: bid_price, bid_volume, ask_price, ask_volume

Путь: backend/ml_engine/models/tlob_transformer.py
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
class TLOBConfig:
    """
    Конфигурация TLOB модели.

    Параметры оптимизированы для данных стакана криптовалют.
    """

    # === Input параметры ===
    num_levels: int = 20  # Количество уровней стакана
    num_features: int = 4  # bid_price, bid_vol, ask_price, ask_vol
    sequence_length: int = 60  # Временное окно

    # === Spatial CNN ===
    cnn_channels: Tuple[int, ...] = (32, 64, 128)
    cnn_kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    cnn_dropout: float = 0.2

    # === Temporal Transformer ===
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # === Multi-scale ===
    use_multi_scale: bool = True
    scales: Tuple[int, ...] = (1, 5, 15)  # 1 = каждый шаг, 5 = каждые 5, 15 = каждые 15

    # === Output ===
    num_classes: int = 3  # SELL, HOLD, BUY
    prediction_horizons: Tuple[int, ...] = (10, 30, 60)  # Горизонты предсказания в шагах

    # === Regularization ===
    drop_path_rate: float = 0.1
    layer_norm_eps: float = 1e-6

    # === Auxiliary Features ===
    use_auxiliary_features: bool = True  # Дополнительные признаки (spread, imbalance)


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class SpatialCNNEncoder(nn.Module):
    """
    Spatial CNN для извлечения паттернов из структуры стакана.

    Обрабатывает пространственную структуру LOB:
    - Вертикально: уровни цен (bid/ask levels)
    - Каналы: price + volume для bid и ask
    """

    def __init__(self, config: TLOBConfig):
        super().__init__()

        self.config = config

        # Начальный размер: (num_levels, num_features) = (20, 4)
        in_channels = config.num_features

        layers = []
        for i, (out_channels, kernel_size) in enumerate(
            zip(config.cnn_channels, config.cnn_kernel_sizes)
        ):
            layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(config.cnn_dropout)
            ])
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)

        # Global pooling для получения фиксированного представления
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Размер выхода
        self.output_dim = config.cnn_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_levels, num_features)

        Returns:
            (batch, output_dim) - пространственное представление
        """
        # Transpose для Conv1d: (batch, features, levels)
        x = x.transpose(1, 2)

        # CNN encoding
        x = self.encoder(x)

        # Global pooling
        x = self.global_pool(x).squeeze(-1)

        return x


class TemporalTransformerBlock(nn.Module):
    """
    Transformer блок для временных зависимостей.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        drop_path: float = 0.0,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.drop_path = nn.Identity()
        if drop_path > 0:
            self.drop_path = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            attn_mask: Optional attention mask

        Returns:
            output: (batch, seq_len, embed_dim)
            attention_weights: (batch, seq_len, seq_len)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            need_weights=True
        )
        x = x + self.drop_path(attn_out)

        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn_weights


class DropPath(nn.Module):
    """Stochastic Depth."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        return x.div(keep_prob) * random_tensor


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale encoding для захвата паттернов на разных временных масштабах.
    """

    def __init__(self, config: TLOBConfig, input_dim: int):
        super().__init__()

        self.config = config
        self.scales = config.scales

        # Encoder для каждого масштаба
        self.scale_encoders = nn.ModuleDict()

        for scale in self.scales:
            self.scale_encoders[f"scale_{scale}"] = nn.Sequential(
                nn.Linear(input_dim, config.embed_dim),
                nn.LayerNorm(config.embed_dim),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.embed_dim * len(self.scales), config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, seq_len, embed_dim)
        """
        B, T, D = x.shape
        scale_outputs = []

        for scale in self.scales:
            # Subsample по времени
            if scale > 1:
                indices = torch.arange(0, T, scale, device=x.device)
                x_scaled = x[:, indices, :]

                # Upsample обратно
                x_scaled = F.interpolate(
                    x_scaled.transpose(1, 2),
                    size=T,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                x_scaled = x

            # Encode
            encoded = self.scale_encoders[f"scale_{scale}"](x_scaled)
            scale_outputs.append(encoded)

        # Concatenate и fusion
        x = torch.cat(scale_outputs, dim=-1)
        x = self.fusion(x)

        return x


class AuxiliaryFeatureEncoder(nn.Module):
    """
    Encoder для дополнительных признаков LOB (spread, imbalance, etc).
    """

    def __init__(self, config: TLOBConfig, output_dim: int):
        super().__init__()

        # Вычисляем признаки из LOB данных
        # spread, mid_price_change, imbalance_5, imbalance_10, volume_ratio
        num_aux_features = 8

        self.encoder = nn.Sequential(
            nn.Linear(num_aux_features, 32),
            nn.GELU(),
            nn.Linear(32, output_dim)
        )

    def compute_auxiliary_features(self, lob: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет дополнительные признаки из LOB.

        Args:
            lob: (batch, seq_len, num_levels, 4) - [bid_price, bid_vol, ask_price, ask_vol]

        Returns:
            (batch, seq_len, num_aux_features)
        """
        B, T, L, _ = lob.shape

        # Извлекаем каналы
        bid_prices = lob[:, :, :, 0]  # (B, T, L)
        bid_volumes = lob[:, :, :, 1]
        ask_prices = lob[:, :, :, 2]
        ask_volumes = lob[:, :, :, 3]

        # Best bid/ask
        best_bid = bid_prices[:, :, 0]  # (B, T)
        best_ask = ask_prices[:, :, 0]

        # Spread
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        relative_spread = spread / (mid_price + 1e-8)

        # Mid price change
        mid_price_change = torch.zeros_like(mid_price)
        mid_price_change[:, 1:] = (mid_price[:, 1:] - mid_price[:, :-1]) / (mid_price[:, :-1] + 1e-8)

        # Imbalance at different levels
        bid_vol_5 = bid_volumes[:, :, :5].sum(dim=-1)
        ask_vol_5 = ask_volumes[:, :, :5].sum(dim=-1)
        imbalance_5 = (bid_vol_5 - ask_vol_5) / (bid_vol_5 + ask_vol_5 + 1e-8)

        bid_vol_10 = bid_volumes[:, :, :10].sum(dim=-1)
        ask_vol_10 = ask_volumes[:, :, :10].sum(dim=-1)
        imbalance_10 = (bid_vol_10 - ask_vol_10) / (bid_vol_10 + ask_vol_10 + 1e-8)

        # Volume ratio
        total_bid_vol = bid_volumes.sum(dim=-1)
        total_ask_vol = ask_volumes.sum(dim=-1)
        volume_ratio = total_bid_vol / (total_ask_vol + 1e-8)

        # VWAP distance
        weighted_bid = (bid_prices * bid_volumes).sum(dim=-1) / (total_bid_vol + 1e-8)
        weighted_ask = (ask_prices * ask_volumes).sum(dim=-1) / (total_ask_vol + 1e-8)
        vwap_spread = (weighted_ask - weighted_bid) / (mid_price + 1e-8)

        # Stack features
        aux_features = torch.stack([
            relative_spread,
            mid_price_change,
            imbalance_5,
            imbalance_10,
            volume_ratio.clamp(-5, 5),  # Clip outliers
            vwap_spread,
            torch.log1p(total_bid_vol),
            torch.log1p(total_ask_vol)
        ], dim=-1)

        return aux_features

    def forward(self, lob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lob: (batch, seq_len, num_levels, 4)

        Returns:
            (batch, seq_len, output_dim)
        """
        aux_features = self.compute_auxiliary_features(lob)
        return self.encoder(aux_features)


# ============================================================================
# MAIN MODEL
# ============================================================================

class TLOBTransformer(nn.Module):
    """
    TLOB - Transformer for Limit Order Book.

    Архитектура:
    1. Spatial CNN: Извлечение паттернов из структуры стакана
    2. Multi-scale Encoding: Анализ на разных временных масштабах
    3. Temporal Transformer: Захват долгосрочных зависимостей
    4. Multi-horizon Prediction: Предсказания на разных горизонтах

    Input:
        Raw LOB: (batch, sequence, levels, 4)
        4 channels: [bid_price, bid_volume, ask_price, ask_volume]

    Output:
        - direction_logits: (batch, num_classes)
        - confidence: (batch, 1)
        - expected_return: (batch, 1)
        - horizon_predictions: Dict[horizon, predictions] (optional)
    """

    def __init__(self, config: TLOBConfig):
        super().__init__()

        self.config = config

        # ==================== SPATIAL CNN ====================
        self.spatial_encoder = SpatialCNNEncoder(config)
        spatial_dim = self.spatial_encoder.output_dim

        # ==================== AUXILIARY FEATURES ====================
        if config.use_auxiliary_features:
            self.aux_encoder = AuxiliaryFeatureEncoder(config, spatial_dim)
            combined_dim = spatial_dim * 2
        else:
            self.aux_encoder = None
            combined_dim = spatial_dim

        # ==================== MULTI-SCALE ENCODING ====================
        if config.use_multi_scale:
            self.multi_scale = MultiScaleEncoder(config, combined_dim)
            embed_dim = config.embed_dim
        else:
            self.multi_scale = None
            embed_dim = combined_dim
            self.projection = nn.Linear(combined_dim, config.embed_dim)

        # ==================== POSITIONAL ENCODING ====================
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.sequence_length, config.embed_dim) * 0.02
        )

        # ==================== TEMPORAL TRANSFORMER ====================
        dpr = [
            config.drop_path_rate * i / (config.num_layers - 1)
            for i in range(config.num_layers)
        ]

        self.transformer_blocks = nn.ModuleList([
            TemporalTransformerBlock(
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

        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

        # ==================== OUTPUT HEADS ====================

        # Main classification head
        self.direction_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 2, config.num_classes)
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 4, 1),
            nn.Sigmoid()
        )

        # Return prediction head
        self.return_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 4, 1)
        )

        # Multi-horizon heads (optional)
        self.horizon_heads = nn.ModuleDict()
        for horizon in config.prediction_horizons:
            self.horizon_heads[f"horizon_{horizon}"] = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim // 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.embed_dim // 4, config.num_classes)
            )

        # ==================== INITIALIZATION ====================
        self._init_weights()

        # Logging
        model_size = self.get_model_size()
        logger.info(
            f"✓ TLOBTransformer initialized: "
            f"params={model_size['total_params']:,}, "
            f"levels={config.num_levels}, "
            f"embed_dim={config.embed_dim}, "
            f"layers={config.num_layers}"
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
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_horizons: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, sequence_length, num_levels, num_features)
               или (batch, sequence_length, num_levels * num_features) - будет reshape
            return_attention: Возвращать ли attention weights
            return_horizons: Возвращать ли предсказания для разных горизонтов

        Returns:
            Dict с выходами
        """
        B = x.size(0)
        T = x.size(1)

        # Reshape если нужно
        if x.dim() == 3:
            # (B, T, L*F) -> (B, T, L, F)
            x = x.reshape(B, T, self.config.num_levels, self.config.num_features)

        # ==================== SPATIAL ENCODING ====================
        # Process each timestep through spatial CNN
        spatial_features = []

        for t in range(T):
            lob_t = x[:, t, :, :]  # (B, L, F)
            spatial_t = self.spatial_encoder(lob_t)  # (B, spatial_dim)
            spatial_features.append(spatial_t)

        spatial_out = torch.stack(spatial_features, dim=1)  # (B, T, spatial_dim)

        # ==================== AUXILIARY FEATURES ====================
        if self.aux_encoder is not None:
            aux_out = self.aux_encoder(x)  # (B, T, spatial_dim)
            combined = torch.cat([spatial_out, aux_out], dim=-1)
        else:
            combined = spatial_out

        # ==================== MULTI-SCALE ENCODING ====================
        if self.multi_scale is not None:
            temporal_in = self.multi_scale(combined)
        else:
            temporal_in = self.projection(combined)

        # ==================== POSITIONAL ENCODING ====================
        temporal_in = temporal_in + self.pos_embed[:, :T, :]

        # ==================== TEMPORAL TRANSFORMER ====================
        attention_weights = []

        for block in self.transformer_blocks:
            temporal_in, attn = block(temporal_in)
            if return_attention:
                attention_weights.append(attn)

        temporal_out = self.norm(temporal_in)

        # ==================== POOLING ====================
        # Use last timestep + mean
        pooled = temporal_out[:, -1, :] + temporal_out.mean(dim=1)

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

        if return_horizons:
            outputs['horizon_predictions'] = {}
            for horizon in self.config.prediction_horizons:
                outputs['horizon_predictions'][horizon] = \
                    self.horizon_heads[f"horizon_{horizon}"](pooled)

        return outputs

    def predict(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        confidence_threshold: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Inference.

        Args:
            x: Input LOB data
            temperature: Temperature scaling
            confidence_threshold: Minimum confidence

        Returns:
            Dict с предсказаниями
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(x, return_attention=False, return_horizons=False)

            direction_logits = outputs['direction_logits'] / temperature
            direction_probs = F.softmax(direction_logits, dim=-1)
            direction = torch.argmax(direction_probs, dim=-1)

            confidence = outputs['confidence'].squeeze(-1)
            expected_return = outputs['expected_return'].squeeze(-1)

            should_trade = confidence >= confidence_threshold

            return {
                'direction': direction,
                'direction_probs': direction_probs,
                'confidence': confidence,
                'expected_return': expected_return,
                'should_trade': should_trade
            }

    def get_model_size(self) -> Dict[str, int]:
        """Размер модели."""
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

def create_tlob_transformer(
    config: Optional[TLOBConfig] = None
) -> TLOBTransformer:
    """Создает TLOB модель."""
    if config is None:
        config = TLOBConfig()

    return TLOBTransformer(config)


def create_tlob_from_preset(preset: str = "base") -> TLOBTransformer:
    """
    Создает модель из пресета.

    Пресеты:
        - tiny: Для быстрых экспериментов
        - base: Стандартная конфигурация
        - hft: Оптимизирован для HFT (меньше latency)
    """
    if preset == "tiny":
        config = TLOBConfig(
            num_levels=10,
            cnn_channels=(16, 32, 64),
            embed_dim=64,
            num_layers=2,
            num_heads=2
        )

    elif preset == "base":
        config = TLOBConfig(
            num_levels=20,
            cnn_channels=(32, 64, 128),
            embed_dim=128,
            num_layers=4,
            num_heads=4
        )

    elif preset == "hft":
        config = TLOBConfig(
            num_levels=10,
            sequence_length=30,
            cnn_channels=(32, 64),
            embed_dim=64,
            num_layers=2,
            num_heads=2,
            use_multi_scale=False,
            dropout=0.05
        )

    else:
        raise ValueError(f"Unknown preset: {preset}")

    return create_tlob_transformer(config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TLOB TRANSFORMER TEST")
    print("=" * 80)

    # Create config
    config = TLOBConfig(
        num_levels=20,
        sequence_length=60,
        embed_dim=128,
        num_layers=4,
        num_heads=4
    )

    print(f"\n📋 Config:")
    print(f"   • Levels: {config.num_levels}")
    print(f"   • Sequence: {config.sequence_length}")
    print(f"   • Embed dim: {config.embed_dim}")
    print(f"   • Layers: {config.num_layers}")
    print(f"   • Multi-scale: {config.use_multi_scale}")

    # Create model
    model = create_tlob_transformer(config)

    # Test input: (batch, sequence, levels, 4)
    batch_size = 32
    x = torch.randn(
        batch_size,
        config.sequence_length,
        config.num_levels,
        config.num_features
    )

    print(f"\n📊 Input shape: {x.shape}")

    print("\n🔄 Forward pass...")
    outputs = model(x, return_attention=False, return_horizons=True)

    print("\n📤 Outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   • {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"   • {key}:")
            for k, v in value.items():
                print(f"      - {k}: {v.shape}")

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
    print("✅ TLOB Transformer test passed!")
    print("=" * 80)
