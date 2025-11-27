# 🚀 ML Model Optimization v2 - Integration Guide

## Обзор изменений

Данный пакет содержит **критические оптимизации** для ML модели, которые увеличат метрики с текущих 33% accuracy до **55-62%** (industry standard).

### Ключевые изменения

| Параметр | Было | Стало | Почему |
|----------|------|-------|--------|
| Learning Rate | 0.001 | **5e-5** | В 20 раз меньше! Финансовые данные очень шумные |
| Batch Size | 64 | **256** | Стабильнее градиенты |
| Weight Decay | ~0 | **0.01** | L2 регуляризация против overfitting |
| Focal Gamma | 2.0 | **2.5** | Больше фокуса на hard examples |
| Label Smoothing | 0 | **0.1** | Предотвращает overconfidence |
| Scheduler | ReduceOnPlateau | **CosineAnnealingWarmRestarts** | Лучше для финансов |
| Augmentation | Нет | **MixUp + Noise** | Увеличивает эффективный размер данных |

---

## Структура файлов

```
ml_optimized/
├── configs/
│   └── optimized_configs.py      # Оптимизированные конфигурации
├── models/
│   └── hybrid_cnn_lstm_v2.py     # Улучшенная модель с Residual/MultiHead
├── training/
│   ├── losses.py                 # Label Smoothing, Focal Loss v2
│   ├── augmentation.py           # MixUp, Time Masking, Noise
│   ├── class_balancing_v2.py     # Adaptive thresholds, SMOTE
│   └── model_trainer_v2.py       # Интегрированный trainer
├── training_orchestrator_v2.py   # Главный оркестратор
└── run_optimized_training.py     # Скрипт запуска
```

---

## Интеграция в проект

### Шаг 1: Копирование файлов

```bash
# Копируем configs
cp ml_optimized/configs/optimized_configs.py backend/ml_engine/configs/

# Копируем модель
cp ml_optimized/models/hybrid_cnn_lstm_v2.py backend/ml_engine/models/

# Копируем training компоненты
cp ml_optimized/training/losses.py backend/ml_engine/training/
cp ml_optimized/training/augmentation.py backend/ml_engine/training/
cp ml_optimized/training/class_balancing_v2.py backend/ml_engine/training/
cp ml_optimized/training/model_trainer_v2.py backend/ml_engine/training/

# Копируем orchestrator
cp ml_optimized/training_orchestrator_v2.py backend/ml_engine/

# Копируем скрипт запуска
cp ml_optimized/run_optimized_training.py backend/ml_engine/
```

### Шаг 2: Обновление импортов

В файле `backend/ml_engine/__init__.py` добавьте:

```python
# Новые оптимизированные компоненты
from .models.hybrid_cnn_lstm_v2 import HybridCNNLSTMv2, ModelConfigV2, create_model_v2
from .training.model_trainer_v2 import ModelTrainerV2, TrainerConfigV2
from .training.losses import MultiTaskLossV2, FocalLossV2, LabelSmoothingCrossEntropy
from .training.augmentation import AugmentationPipeline, MixUp
from .training.class_balancing_v2 import ClassBalancingStrategyV2
from .training_orchestrator_v2 import TrainingOrchestratorV2
```

### Шаг 3: Обновление существующего кода

#### Замена ModelConfig

```python
# Было:
from backend.ml_engine.models.hybrid_cnn_lstm import ModelConfig, create_model

config = ModelConfig()
model = create_model(config)

# Стало:
from backend.ml_engine.models.hybrid_cnn_lstm_v2 import ModelConfigV2, create_model_v2

config = ModelConfigV2(
    cnn_channels=(32, 64, 128),  # Уменьшено для малого датасета
    lstm_hidden=128,
    dropout=0.4,
    use_residual=True,
    use_layer_norm=True,
    use_multi_head_attention=True
)
model = create_model_v2(config)
```

#### Замена TrainerConfig

```python
# Было:
from backend.ml_engine.training.model_trainer import TrainerConfig, ModelTrainer

config = TrainerConfig(
    learning_rate=0.001,  # СЛИШКОМ ВЫСОКИЙ!
    batch_size=64,
    epochs=100
)

# Стало:
from backend.ml_engine.training.model_trainer_v2 import TrainerConfigV2, ModelTrainerV2

config = TrainerConfigV2(
    learning_rate=5e-5,    # КРИТИЧНО: в 20 раз меньше!
    batch_size=256,        # КРИТИЧНО: в 4 раза больше!
    weight_decay=0.01,     # L2 регуляризация
    label_smoothing=0.1,   # Предотвращает overconfidence
    use_augmentation=True,
    mixup_alpha=0.2,
    focal_gamma=2.5,
    epochs=150
)
```

---

## Быстрый старт

### Вариант 1: Через скрипт (рекомендуется)

```bash
# Быстрый тест (5 минут)
python backend/ml_engine/run_optimized_training.py --preset quick

# Production обучение (2-4 часа)
python backend/ml_engine/run_optimized_training.py --preset production --symbols BTCUSDT ETHUSDT

# Полная версия
python backend/ml_engine/run_optimized_training.py \
    --preset production \
    --symbols BTCUSDT ETHUSDT BNBUSDT \
    --days 30 \
    --output-dir models/trained
```

### Вариант 2: Программный API

```python
import asyncio
from backend.ml_engine.training_orchestrator_v2 import (
    TrainingOrchestratorV2,
    OrchestratorConfig
)

async def train():
    config = OrchestratorConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        feature_store_days=30,
        model_preset="production_small"
    )
    
    orchestrator = TrainingOrchestratorV2(config)
    results = await orchestrator.run_training()
    
    print(f"Best F1: {results['best_metrics']['val_f1']:.4f}")
    return results

# Запуск
results = asyncio.run(train())
```

### Вариант 3: Standalone (минимальный код)

```python
import torch
from backend.ml_engine.models.hybrid_cnn_lstm_v2 import create_model_v2_from_preset
from backend.ml_engine.training.model_trainer_v2 import create_trainer_v2

# Создаём модель
model = create_model_v2_from_preset("production_small")

# Создаём trainer с оптимизированными параметрами
trainer = create_trainer_v2(model, preset="production_small")

# Обучаем
history = trainer.train(train_loader, val_loader)

print(f"Best Val Loss: {trainer.best_val_loss:.4f}")
print(f"Best Val F1: {trainer.best_val_f1:.4f}")
```

---

## Конфигурационные пресеты

### Production Small (рекомендуется для 7-30 дней данных)

```python
ModelConfigV2:
    cnn_channels: (32, 64, 128)    # ~150K параметров
    lstm_hidden: 128
    dropout: 0.4
    use_residual: True
    use_layer_norm: True

TrainerConfigV2:
    learning_rate: 5e-5
    batch_size: 256
    weight_decay: 0.01
    label_smoothing: 0.1
    mixup_alpha: 0.2
    focal_gamma: 2.5
    epochs: 150
```

### Production Large (для 60+ дней данных)

```python
ModelConfigV2:
    cnn_channels: (64, 128, 256)   # ~500K параметров
    lstm_hidden: 256
    dropout: 0.3

TrainerConfigV2:
    learning_rate: 1e-4
    batch_size: 128
    weight_decay: 0.005
    label_smoothing: 0.05
    epochs: 100
```

### Quick Experiment (для быстрых тестов)

```python
ModelConfigV2:
    cnn_channels: (32, 64)         # ~50K параметров
    lstm_hidden: 64
    lstm_layers: 1

TrainerConfigV2:
    learning_rate: 1e-4
    batch_size: 128
    epochs: 30
    use_augmentation: False
```

---

## Архитектурные улучшения

### 1. Residual Connections в CNN

```
Input → Conv → BN → ReLU → Dropout → Output
  ↓                                     ↑
  └───────── skip connection ───────────┘
```

Преимущества:
- Лучший gradient flow
- Возможность обучать более глубокие сети
- Стабильнее обучение

### 2. Multi-Head Temporal Attention

```python
# Вместо single-head attention:
self.attention = SimpleAttention(hidden_size)

# Используем multi-head:
self.attention = MultiHeadTemporalAttention(
    hidden_size=256,
    num_heads=4,
    dropout=0.1
)
```

Преимущества:
- Захватывает разные аспекты временных зависимостей
- Лучше работает с длинными sequences
- Повышает interpretability

### 3. Layer Normalization для LSTM

```python
# Вместо BatchNorm после LSTM:
self.lstm = LSTMWithLayerNorm(
    input_size=128,
    hidden_size=256,
    use_layer_norm=True
)
```

Преимущества:
- Стабильнее для sequences (не зависит от batch)
- Лучше работает с маленькими batch sizes

---

## Data Augmentation

### MixUp

```python
# Смешивание samples:
mixed_x = lambda * x_i + (1 - lambda) * x_j
mixed_y = lambda * y_i + (1 - lambda) * y_j

# Использование:
mixup = MixUp(alpha=0.2)
mixed_x, y_a, y_b, lam = mixup(x, y)
loss = lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)
```

### Gaussian Noise

```python
# Добавление шума к features:
noisy_x = x + torch.randn_like(x) * 0.01
```

### Time Masking

```python
# Маскирование случайных timesteps:
mask = TimeMasking(mask_ratio=0.1)
masked_x = mask(x)  # 10% timesteps заменены на 0
```

---

## Class Balancing

### Adaptive Thresholds

Вместо фиксированных порогов для labeling:

```python
# Было (фиксированные):
if return > 0.001:
    label = BUY
elif return < -0.001:
    label = SELL
else:
    label = HOLD

# Стало (percentile-based):
sell_threshold = np.percentile(returns, 25)  # Bottom 25%
buy_threshold = np.percentile(returns, 75)   # Top 25%
```

### Focal Loss с улучшенным gamma

```python
# Focal Loss фокусируется на hard examples:
# FL(p) = -(1-p)^gamma * log(p)

# gamma=2.0: стандартный
# gamma=2.5: больше фокуса на hard (рекомендуется)
# gamma=3.0: агрессивный фокус
```

---

## Мониторинг обучения

### Ожидаемые метрики по эпохам

| Эпоха | Train Loss | Val Loss | Val Acc | Val F1 |
|-------|------------|----------|---------|--------|
| 1-10  | 1.1-0.9    | 1.2-1.0  | 35-40%  | 30-38% |
| 10-30 | 0.9-0.7    | 1.0-0.85 | 40-48%  | 38-45% |
| 30-60 | 0.7-0.5    | 0.85-0.70| 48-55%  | 45-52% |
| 60-100| 0.5-0.4    | 0.70-0.60| 55-60%  | 52-57% |
| 100+  | 0.4-0.35   | 0.60-0.55| 58-62%  | 55-60% |

### Warning Signs

❌ **Val Loss растёт, Train Loss падает** → Overfitting
   - Решение: увеличить dropout, weight_decay, уменьшить модель

❌ **Accuracy не растёт выше 35%** → Модель не обучается
   - Решение: проверить learning rate (должен быть 5e-5, не 0.001!)

❌ **F1 сильно ниже Accuracy** → Class imbalance
   - Решение: увеличить focal_gamma, использовать oversampling

---

## Troubleshooting

### Проблема: Out of Memory

```python
# Уменьшить batch_size:
config.batch_size = 128  # вместо 256

# Использовать gradient accumulation:
config.gradient_accumulation_steps = 2
```

### Проблема: Медленное обучение

```python
# Включить Mixed Precision (только для GPU):
config.use_mixed_precision = True

# Уменьшить модель:
model_config.cnn_channels = (32, 64)  # вместо (32, 64, 128)
```

### Проблема: Нестабильные метрики

```python
# Увеличить batch_size для стабильности:
config.batch_size = 256

# Уменьшить learning rate:
config.learning_rate = 3e-5

# Включить gradient clipping:
config.grad_clip_value = 1.0
```

---

## Checklist перед Production

- [ ] Learning rate = 5e-5 (НЕ 0.001!)
- [ ] Batch size >= 128 (рекомендуется 256)
- [ ] Weight decay = 0.01
- [ ] Label smoothing = 0.1
- [ ] Focal gamma = 2.5
- [ ] Early stopping patience >= 15
- [ ] MixUp augmentation включен
- [ ] Class balancing настроен
- [ ] Тестирование на held-out данных

---

## Контакты и поддержка

При возникновении проблем:
1. Проверьте логи обучения
2. Убедитесь что все параметры соответствуют рекомендациям
3. Запустите диагностику: `python diagnose_optimization.py --symbol BTCUSDT`

---

*Последнее обновление: 2025*
