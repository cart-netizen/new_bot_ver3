# Multi-Model Ensemble System

## Overview

This guide documents the Multi-Model Ensemble system for the trading bot. The system implements three advanced ML architectures that work together through an Ensemble Consensus mechanism to improve prediction accuracy.

**Path:** `backend/ml_engine/`

## Architecture Components

```
ml_engine/
├── models/
│   ├── cnn_lstm_v2.py          # Existing CNN-LSTM model
│   ├── mpd_transformer.py      # NEW: MPDTransformer (Vision Transformer)
│   └── tlob_transformer.py     # NEW: TLOB Transformer (Order Book)
├── ensemble/
│   ├── __init__.py
│   └── ensemble_consensus.py   # NEW: Ensemble consensus module
├── training/
│   └── multi_model_trainer.py  # NEW: Unified training pipeline
├── data_collection/
│   └── raw_lob_collector.py    # NEW: Raw LOB data collector for TLOB
└── api/
    └── ensemble_api.py         # NEW: FastAPI endpoints for management
```

---

## 1. MPDTransformer (Matrix Profile Distribution Transformer)

### Description

MPDTransformer is a Vision Transformer (ViT) adapted for financial time series. It converts sequential data into 2D matrix representation and processes it using patch embedding and transformer blocks.

**File:** `backend/ml_engine/models/mpd_transformer.py`

### Architecture

```
Input: (batch_size, sequence_length=60, features=112)
    ↓
┌─────────────────────────────────────┐
│      2D Matrix Conversion           │
│   (batch, 60, 112) → (batch, 60, 112)
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│        Patch Embedding              │
│   patch_size = (10, 14)             │
│   → 48 patches per sample           │
│   → Linear projection to embed_dim  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Positional Encoding             │
│   Learnable position embeddings     │
│   + CLS token prepended             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Transformer Blocks (×6)         │
│   - Multi-Head Attention (8 heads)  │
│   - Feed-Forward Network            │
│   - Layer Normalization             │
│   - Dropout (0.1)                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│        Output Heads                 │
│   - Direction: 3 classes (softmax)  │
│   - Confidence: 1 value (sigmoid)   │
│   - Expected Return: 1 value        │
└─────────────────────────────────────┘
```

### Configuration

```python
from backend.ml_engine.models.mpd_transformer import MPDConfig, MPDTransformer

config = MPDConfig(
    input_features=112,         # Number of input features
    sequence_length=60,         # Sequence length
    patch_size=(10, 14),        # Patch dimensions
    embed_dim=256,              # Embedding dimension
    num_layers=6,               # Number of transformer layers
    num_heads=8,                # Attention heads
    mlp_ratio=4.0,              # MLP hidden dimension ratio
    dropout=0.1,                # Dropout rate
    num_classes=3,              # Direction classes: DOWN, NEUTRAL, UP
    use_cls_token=True          # Use CLS token for classification
)

model = MPDTransformer(config)
```

### Usage

```python
import torch

# Input: (batch_size, sequence_length, features)
x = torch.randn(32, 60, 112)

# Forward pass
outputs = model(x)

# Outputs
direction_logits = outputs['direction_logits']  # Shape: (32, 3)
confidence = outputs['confidence']               # Shape: (32, 1)
expected_return = outputs['expected_return']     # Shape: (32, 1)

# Get predictions
direction_probs = torch.softmax(direction_logits, dim=-1)
predicted_direction = torch.argmax(direction_probs, dim=-1)  # 0=DOWN, 1=NEUTRAL, 2=UP
```

### Key Features

- **Patch-based processing**: Captures local patterns in feature space
- **Multi-task learning**: Predicts direction, confidence, and expected return simultaneously
- **Scalable**: Efficient attention mechanism for longer sequences
- **GPU optimized**: Uses mixed precision training

---

## 2. TLOB Transformer (Transformer for Limit Order Book)

### Description

TLOB is a specialized transformer designed for raw limit order book data. It combines Spatial CNN for cross-level patterns with Temporal Transformer for sequential dependencies.

**File:** `backend/ml_engine/models/tlob_transformer.py`

### Architecture

```
Input: (batch_size, sequence_length=100, levels=20, features=4)
       features = [bid_price, bid_volume, ask_price, ask_volume]
    ↓
┌─────────────────────────────────────┐
│      Spatial CNN Encoder            │
│   - Conv1d across levels            │
│   - Captures bid-ask interactions   │
│   - Multi-scale patterns            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Multi-Scale Encoder             │
│   - Short-term: Conv1d kernel=3     │
│   - Medium-term: Conv1d kernel=5    │
│   - Long-term: Conv1d kernel=7      │
│   - Concatenate + project           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Auxiliary Feature Encoder         │
│   - Optional: Add 112 extra features│
│   - MLP projection                  │
│   - Feature fusion                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Temporal Transformer (×4)         │
│   - Multi-Head Attention (4 heads)  │
│   - Sequence-level patterns         │
│   - Positional encoding             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Multi-Horizon Heads             │
│   - Short horizon (10 steps)        │
│   - Medium horizon (30 steps)       │
│   - Long horizon (50 steps)         │
│   Each: direction + confidence      │
└─────────────────────────────────────┘
```

### Configuration

```python
from backend.ml_engine.models.tlob_transformer import TLOBConfig, TLOBTransformer

config = TLOBConfig(
    num_levels=20,                  # Order book depth
    lob_features=4,                 # bid_price, bid_vol, ask_price, ask_vol
    sequence_length=100,            # LOB snapshots per sample
    spatial_channels=[32, 64, 128], # CNN channel progression
    temporal_hidden=128,            # Transformer hidden size
    num_temporal_layers=4,          # Transformer layers
    num_heads=4,                    # Attention heads
    dropout=0.1,                    # Dropout rate
    num_classes=3,                  # Direction classes
    horizons=[10, 30, 50],          # Prediction horizons
    aux_features=112                # Optional auxiliary features
)

model = TLOBTransformer(config)
```

### Usage

```python
import torch

# LOB Input: (batch_size, sequence_length, levels, 4)
lob_data = torch.randn(32, 100, 20, 4)

# Optional auxiliary features: (batch_size, aux_features)
aux_features = torch.randn(32, 112)

# Forward pass
outputs = model(lob_data, aux_features)

# Outputs for each horizon
for horizon in [10, 30, 50]:
    direction = outputs[f'direction_{horizon}']      # Shape: (32, 3)
    confidence = outputs[f'confidence_{horizon}']    # Shape: (32, 1)

# Primary prediction (medium horizon)
direction_logits = outputs['direction_30']
predicted_direction = torch.argmax(torch.softmax(direction_logits, dim=-1), dim=-1)
```

### Raw LOB Data Collection

TLOB requires raw order book data. Use the `RawLOBCollector`:

**File:** `backend/ml_engine/data_collection/raw_lob_collector.py`

```python
from backend.ml_engine.data_collection.raw_lob_collector import (
    RawLOBCollector, RawLOBConfig
)

# Configuration
config = RawLOBConfig(
    symbol="BTCUSDT",
    num_levels=20,           # Order book depth
    collection_interval=0.1, # 100ms intervals
    sequence_length=100,     # Snapshots per sequence
    buffer_size=10000,       # Memory buffer
    save_interval=1000       # Save to disk every N snapshots
)

# Initialize collector
collector = RawLOBCollector(config)

# Start collection (background task)
await collector.start()

# Get sequence for training
sequence = collector.get_latest_sequence()
if sequence:
    tensor = sequence.to_normalized_tensor()  # Shape: (100, 20, 4)

# Stop collection
await collector.stop()

# Export data for training
collector.export_to_parquet("data/raw_lob/")
```

---

## 3. Ensemble Consensus Module

### Description

The Ensemble Consensus module combines predictions from multiple models using various consensus strategies.

**File:** `backend/ml_engine/ensemble/ensemble_consensus.py`

### Supported Models

```python
from backend.ml_engine.ensemble import ModelType

# Available model types
ModelType.CNN_LSTM       # Existing CNN-LSTM v2
ModelType.MPD_TRANSFORMER  # MPDTransformer
ModelType.TLOB           # TLOB Transformer
```

### Consensus Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `WEIGHTED_VOTING` | Weighted average of predictions | Default, balanced approach |
| `UNANIMOUS` | All models must agree | High confidence trades |
| `MAJORITY` | Simple majority vote | Robust to outliers |
| `CONFIDENCE_BASED` | Weight by model confidence | Dynamic weighting |
| `ADAPTIVE` | Adjusts weights based on recent performance | Changing market conditions |

### Configuration

```python
from backend.ml_engine.ensemble import (
    EnsembleConsensus,
    EnsembleConfig,
    ModelWeight,
    ModelType,
    ConsensusStrategy
)

# Define model weights
weights = [
    ModelWeight(
        model_type=ModelType.CNN_LSTM,
        weight=0.4,
        enabled=True,
        min_confidence=0.6
    ),
    ModelWeight(
        model_type=ModelType.MPD_TRANSFORMER,
        weight=0.35,
        enabled=True,
        min_confidence=0.55
    ),
    ModelWeight(
        model_type=ModelType.TLOB,
        weight=0.25,
        enabled=True,
        min_confidence=0.5
    )
]

# Configuration
config = EnsembleConfig(
    strategy=ConsensusStrategy.WEIGHTED_VOTING,
    model_weights=weights,
    min_agreement=0.6,          # Minimum agreement for consensus
    confidence_threshold=0.5,   # Minimum confidence to act
    neutral_threshold=0.1       # Threshold for neutral prediction
)

# Create ensemble
ensemble = EnsembleConsensus(config)
```

### Usage

```python
from backend.ml_engine.ensemble import ModelPrediction, Direction

# Collect predictions from each model
predictions = [
    ModelPrediction(
        model_type=ModelType.CNN_LSTM,
        direction=Direction.UP,
        confidence=0.75,
        probabilities=[0.1, 0.15, 0.75],  # [DOWN, NEUTRAL, UP]
        expected_return=0.002,
        timestamp=time.time()
    ),
    ModelPrediction(
        model_type=ModelType.MPD_TRANSFORMER,
        direction=Direction.UP,
        confidence=0.68,
        probabilities=[0.12, 0.2, 0.68],
        expected_return=0.0018,
        timestamp=time.time()
    ),
    ModelPrediction(
        model_type=ModelType.TLOB,
        direction=Direction.NEUTRAL,
        confidence=0.55,
        probabilities=[0.2, 0.55, 0.25],
        expected_return=0.0001,
        timestamp=time.time()
    )
]

# Get ensemble prediction
result = ensemble.predict(predictions)

# Result contains
print(f"Direction: {result.direction}")           # Direction.UP
print(f"Confidence: {result.confidence:.2f}")     # 0.67
print(f"Meta-confidence: {result.meta_confidence:.2f}")  # 0.85
print(f"Agreement: {result.agreement_score:.2f}") # 0.67 (2/3 agree)
print(f"Should trade: {result.should_trade}")     # True/False

# Individual model contributions
for model, contrib in result.model_contributions.items():
    print(f"{model.value}: {contrib:.2%}")
```

### Dynamic Weight Adjustment

```python
# Update weights based on model performance
ensemble.update_model_performance(
    model_type=ModelType.CNN_LSTM,
    accuracy=0.72,
    profit_factor=1.8
)

# Disable poorly performing model
ensemble.disable_model(ModelType.TLOB)

# Re-enable model
ensemble.enable_model(ModelType.TLOB)

# Change strategy
ensemble.set_strategy(ConsensusStrategy.ADAPTIVE)

# Get current status
status = ensemble.get_status()
print(status)
```

---

## 4. Multi-Model Training Pipeline

### Description

Unified training pipeline that supports all three model architectures with consistent interface.

**File:** `backend/ml_engine/training/multi_model_trainer.py`

### Supported Architectures

```python
from backend.ml_engine.training.multi_model_trainer import ModelArchitecture

ModelArchitecture.CNN_LSTM
ModelArchitecture.MPD_TRANSFORMER
ModelArchitecture.TLOB
```

### Training Configuration

```python
from backend.ml_engine.training.multi_model_trainer import (
    MultiModelTrainer,
    TrainingConfig
)

config = TrainingConfig(
    # Model selection
    architecture=ModelArchitecture.MPD_TRANSFORMER,

    # Data parameters
    sequence_length=60,
    batch_size=32,

    # Training parameters
    learning_rate=1e-4,
    weight_decay=1e-5,
    epochs=100,
    early_stopping_patience=10,

    # Loss weights (multi-task learning)
    direction_weight=1.0,
    confidence_weight=0.5,
    return_weight=0.3,

    # Validation
    validation_split=0.2,
    walk_forward_splits=5,

    # Optimization
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    use_amp=True,  # Mixed precision

    # Checkpointing
    save_best_only=True,
    checkpoint_dir="checkpoints/"
)
```

### Training Example

```python
# Initialize trainer
trainer = MultiModelTrainer(config)

# Load data
trainer.load_data(
    features_path="data/features/",
    lob_path="data/raw_lob/",  # Required for TLOB
    symbol="BTCUSDT"
)

# Train with walk-forward validation
results = trainer.train_walk_forward()

# Results contain
print(f"Best validation accuracy: {results['best_val_accuracy']:.2%}")
print(f"Best validation loss: {results['best_val_loss']:.4f}")
print(f"Training time: {results['training_time_seconds']:.0f}s")

# Model is saved automatically
print(f"Model saved to: {results['model_path']}")
```

### Training All Models

```python
# Train all models sequentially
architectures = [
    ModelArchitecture.CNN_LSTM,
    ModelArchitecture.MPD_TRANSFORMER,
    ModelArchitecture.TLOB
]

results = {}
for arch in architectures:
    config.architecture = arch
    trainer = MultiModelTrainer(config)
    trainer.load_data(...)
    results[arch] = trainer.train_walk_forward()

# Compare results
for arch, result in results.items():
    print(f"{arch.value}: accuracy={result['best_val_accuracy']:.2%}")
```

---

## 5. Ensemble API

### Description

FastAPI endpoints for managing the ensemble system from the frontend.

**File:** `backend/api/ensemble_api.py`

### Endpoints

#### Get Ensemble Status
```http
GET /api/ensemble/status
```
Response:
```json
{
    "strategy": "weighted_voting",
    "total_models": 3,
    "enabled_models": 3,
    "models": {
        "cnn_lstm": {"enabled": true, "weight": 0.4, "accuracy": 0.72},
        "mpd_transformer": {"enabled": true, "weight": 0.35, "accuracy": 0.68},
        "tlob": {"enabled": true, "weight": 0.25, "accuracy": 0.65}
    },
    "last_prediction": {...}
}
```

#### List Models
```http
GET /api/ensemble/models
```

#### Enable/Disable Model
```http
POST /api/ensemble/models/enable
Content-Type: application/json

{
    "model_type": "mpd_transformer",
    "enabled": false
}
```

#### Update Model Weight
```http
POST /api/ensemble/models/weight
Content-Type: application/json

{
    "model_type": "cnn_lstm",
    "weight": 0.5
}
```

#### Change Strategy
```http
POST /api/ensemble/strategy
Content-Type: application/json

{
    "strategy": "adaptive"
}
```

#### Start Training
```http
POST /api/ensemble/training/start
Content-Type: application/json

{
    "architecture": "mpd_transformer",
    "symbol": "BTCUSDT",
    "epochs": 100,
    "batch_size": 32
}
```

#### Get Training Status
```http
GET /api/ensemble/training/status
```
Response:
```json
{
    "is_training": true,
    "current_epoch": 45,
    "total_epochs": 100,
    "current_loss": 0.234,
    "best_loss": 0.198,
    "architecture": "mpd_transformer",
    "progress_percent": 45.0
}
```

#### Update Model Performance
```http
POST /api/ensemble/performance/update
Content-Type: application/json

{
    "model_type": "cnn_lstm",
    "accuracy": 0.73,
    "profit_factor": 1.95
}
```

### Integration with Main Router

```python
# In your main FastAPI app
from fastapi import FastAPI
from backend.api.ensemble_api import router as ensemble_router

app = FastAPI()
app.include_router(ensemble_router, prefix="/api/ensemble", tags=["Ensemble"])
```

---

## 6. Integration Guide

### Step 1: Setup Data Collection

```python
# Start raw LOB collection for TLOB
from backend.ml_engine.data_collection.raw_lob_collector import RawLOBCollector

collector = RawLOBCollector(config)
await collector.start()
```

### Step 2: Train Models

```python
# Train each model architecture
from backend.ml_engine.training.multi_model_trainer import MultiModelTrainer

# 1. Train CNN-LSTM (existing)
# 2. Train MPDTransformer
# 3. Train TLOB

for arch in architectures:
    trainer = MultiModelTrainer(config)
    trainer.train_walk_forward()
```

### Step 3: Initialize Ensemble

```python
from backend.ml_engine.ensemble import create_ensemble_consensus

ensemble = create_ensemble_consensus(
    strategy=ConsensusStrategy.WEIGHTED_VOTING,
    model_weights={
        ModelType.CNN_LSTM: 0.4,
        ModelType.MPD_TRANSFORMER: 0.35,
        ModelType.TLOB: 0.25
    }
)
```

### Step 4: Use in Trading

```python
async def get_trading_signal(features, lob_data):
    # Get predictions from each model
    predictions = []

    # CNN-LSTM prediction
    cnn_pred = cnn_lstm_model.predict(features)
    predictions.append(ModelPrediction(
        model_type=ModelType.CNN_LSTM,
        direction=cnn_pred['direction'],
        confidence=cnn_pred['confidence'],
        probabilities=cnn_pred['probabilities'],
        expected_return=cnn_pred['expected_return'],
        timestamp=time.time()
    ))

    # MPDTransformer prediction
    mpd_pred = mpd_model(features)
    predictions.append(ModelPrediction(
        model_type=ModelType.MPD_TRANSFORMER,
        direction=mpd_pred['direction'],
        confidence=mpd_pred['confidence'],
        ...
    ))

    # TLOB prediction
    tlob_pred = tlob_model(lob_data, features)
    predictions.append(ModelPrediction(
        model_type=ModelType.TLOB,
        direction=tlob_pred['direction'],
        confidence=tlob_pred['confidence'],
        ...
    ))

    # Get ensemble consensus
    result = ensemble.predict(predictions)

    return {
        'direction': result.direction,
        'confidence': result.confidence,
        'should_trade': result.should_trade,
        'meta_confidence': result.meta_confidence
    }
```

---

## 7. Feature Count Verification

The system uses **112 features** total:

| Category | Count | Description |
|----------|-------|-------------|
| LOB Features | 50 | Order book derived features |
| Candle Features | 25 | OHLCV price features |
| Indicator Features | 37 | Technical indicators |
| **Total** | **112** | |

**Note:** Some documentation may reference 114 features, but the actual implementation uses 112.

---

## 8. Computational Requirements

### Model Sizes (Approximate)

| Model | Parameters | VRAM (Training) | VRAM (Inference) |
|-------|------------|-----------------|------------------|
| CNN-LSTM v2 | ~2M | 2-4 GB | 0.5 GB |
| MPDTransformer | ~8M | 4-6 GB | 1 GB |
| TLOB | ~5M | 6-8 GB | 1.5 GB |

### Training Time Estimates

| Model | GPU (RTX 3080) | GPU (RTX 4090) |
|-------|----------------|----------------|
| CNN-LSTM | 2-4 hours | 1-2 hours |
| MPDTransformer | 4-6 hours | 2-3 hours |
| TLOB | 6-10 hours | 3-5 hours |

### Recommendations

1. **GPU Memory**: Minimum 8GB VRAM for training all models
2. **Mixed Precision**: Enable `use_amp=True` to reduce memory usage
3. **Gradient Accumulation**: Use `gradient_accumulation_steps=4` for larger effective batch size
4. **Data Storage**: Raw LOB data requires ~10GB per day of collection

---

## 9. Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size
config.batch_size = 16

# Enable gradient accumulation
config.gradient_accumulation_steps = 8

# Enable mixed precision
config.use_amp = True
```

**2. TLOB Data Not Available**
```python
# Ensure raw LOB collector is running
collector = RawLOBCollector(config)
await collector.start()

# Wait for sufficient data (minimum 1000 snapshots)
while collector.get_buffer_size() < 1000:
    await asyncio.sleep(1)
```

**3. Model Disagreement**
```python
# Use confidence-based strategy for handling disagreement
ensemble.set_strategy(ConsensusStrategy.CONFIDENCE_BASED)

# Or require unanimous agreement for trading
ensemble.set_strategy(ConsensusStrategy.UNANIMOUS)
```

**4. Poor Ensemble Performance**
```python
# Disable underperforming models
ensemble.disable_model(ModelType.TLOB)

# Increase weight of best model
ensemble.update_weight(ModelType.CNN_LSTM, 0.6)
```

---

## 10. Future Improvements

1. **A/B Testing Framework**: Compare ensemble vs individual models
2. **Online Learning**: Update model weights based on real trading results
3. **Additional Models**: Support for Stockformer, DGT architectures
4. **Distributed Training**: Multi-GPU support for faster training
5. **Model Compression**: Quantization for faster inference

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-12 | Initial release with MPD, TLOB, Ensemble |

---

**Authors:** Trading Bot ML Team
**Last Updated:** December 2024
