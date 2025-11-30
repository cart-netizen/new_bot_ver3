# 🚀 ML v2 Migration Complete - Документация

## ✅ Миграция завершена (обновлено 2025-11-27)

Проект полностью переведен на использование **оптимизированных v2 версий** ML компонентов.

### 🎯 Текущий статус интеграции

**Используемая версия:** `training_orchestrator.py` с v2 компонентами внутри
- ✅ Все v2 модели и trainer импортируются через алиасы
- ✅ Полная обратная совместимость с существующим API
- ✅ `TrainingOrchestratorV2` доступен для прямого импорта
- ✅ Совместимость EpochMetrics (v2) и dict (v1) форматов

---

## 📋 Что изменено

### 1. Обновленные файлы (используют v2)

**Backend ML Engine:**
- ✅ `backend/ml_engine/training_orchestrator.py`
- ✅ `backend/ml_engine/auto_retraining/retraining_pipeline.py`
- ✅ `backend/ml_engine/inference/model_server.py`

**API:**
- ✅ `backend/api/ml_management_api.py`

**Scripts:**
- ✅ `train_model.py`

### 2. Используемые v2 компоненты

**Модель:**
```python
from backend.ml_engine.models.hybrid_cnn_lstm_v2 import (
    HybridCNNLSTMv2 as HybridCNNLSTM,
    ModelConfigV2 as ModelConfig
)
```

**Trainer:**
```python
from backend.ml_engine.training.model_trainer_v2 import (
    ModelTrainerV2 as ModelTrainer,
    TrainerConfigV2 as TrainerConfig
)
```

### 3. Алиасы для совместимости

Все v2 классы импортируются с алиасами, чтобы сохранить совместимость с существующим кодом:
- `HybridCNNLSTMv2` → `HybridCNNLSTM`
- `ModelConfigV2` → `ModelConfig`
- `ModelTrainerV2` → `ModelTrainer`
- `TrainerConfigV2` → `TrainerConfig`

---

## 🎯 Оптимизации в v2

### ModelConfigV2 (hybrid_cnn_lstm_v2.py)

**Архитектурные улучшения:**
- ✅ **Residual Connections** в CNN блоках
- ✅ **Multi-Head Temporal Attention** (4 heads вместо 1)
- ✅ **Layer Normalization** для LSTM
- ✅ **GELU activation** вместо ReLU
- ✅ **Squeeze-and-Excitation** блоки

**Оптимизированные параметры для малого датасета:**
```python
cnn_channels: (32, 64, 128)     # Было: (64, 128, 256)
lstm_hidden: 128                # Было: 256
dropout: 0.4                    # Было: 0.3
```

### TrainerConfigV2 (model_trainer_v2.py)

**Критические изменения:**
```python
learning_rate: 5e-5             # Было: 0.001 ← КРИТИЧНО!
weight_decay: 0.01              # Было: 1e-5 ← КРИТИЧНО!
batch_size: 256                 # Было: 64 ← КРИТИЧНО!
epochs: 150                     # Было: 100
```

**Новые возможности:**
- ✅ **CosineAnnealingWarmRestarts** scheduler
- ✅ **Label Smoothing** (0.1)
- ✅ **Gaussian Noise Augmentation** (std=0.01)
- ✅ **MixUp Data Augmentation** (alpha=0.2)

### Loss Functions (training/losses.py)

- ✅ **FocalLossV2** с gamma=2.5 (фокус на hard examples)
- ✅ **LabelSmoothingCrossEntropy** (предотвращает overconfidence)
- ✅ **AsymmetricFocalLoss** (разные веса для FP и FN)
- ✅ **DirectionalAccuracyLoss** (штраф за противоположное направление)

### Data Augmentation (training/augmentation.py)

- ✅ **MixUp** - смешивание samples
- ✅ **Time Masking** - маскирование временных шагов
- ✅ **Gaussian Noise** - добавление шума
- ✅ **Feature Dropout** - dropout отдельных features
- ✅ **Time Warping** - временное растяжение/сжатие

### Class Balancing (class_balancing_v2.py)

- ✅ **Adaptive Threshold Labeling** (percentile-based)
- ✅ **Improved Oversampling** (ratio=0.5)
- ✅ **Focal Loss enabled** по умолчанию

---

## 📊 Ожидаемые улучшения метрик

| Метрика | Было (v1) | Ожидается (v2) | Industry Standard |
|---------|-----------|----------------|-------------------|
| **Accuracy** | 33.89% | **55-62%** | 55-65% |
| **F1 Score** | 32.50% | **52-58%** | 50-60% |
| **Val Loss** | 1.219 | **<0.7** | <0.6 |
| **Train-Val Gap** | 0.295 | **<0.08** | <0.1 |

---

## 🔄 Обратная совместимость

### Старые модели (v1)

Старые чекпоинты автоматически конвертируются при загрузке:

```python
from backend.ml_engine.models.hybrid_cnn_lstm_v2 import load_from_v1_checkpoint

# Загрузка старого чекпоинта
model_v2 = load_from_v1_checkpoint(
    checkpoint_path="checkpoints/old_model.pt",
    config_v2=ModelConfigV2()
)
```

### Миграция данных

Все старые датасеты совместимы с v2:
- ✅ Feature Store
- ✅ `.npy` файлы
- ✅ Label mapping (автоматический)

---

## 🚀 Как использовать v2

### Вариант 1: Через Training Orchestrator (рекомендуется, текущий способ)

```python
from backend.ml_engine.training_orchestrator import TrainingOrchestrator

# Автоматически использует v2 компоненты внутри
orchestrator = TrainingOrchestrator()
result = await orchestrator.train_model()
```

**Статус:** ✅ Используется везде в проекте. Внутри использует v2 модели и trainer через алиасы.

### Вариант 1b: Через TrainingOrchestratorV2 (продвинутый)

```python
from backend.ml_engine.training_orchestrator import TrainingOrchestratorV2, OrchestratorConfig

# Конфигурация с пресетами
config = OrchestratorConfig(
    model_preset="production_small",
    trainer_preset="production_small",
    feature_store_days=30,
    symbols=["BTCUSDT", "ETHUSDT"]
)

# Создание orchestrator
orchestrator = TrainingOrchestratorV2(config)
result = await orchestrator.run_training()
```

**Преимущества v2 orchestrator:**
- ✅ Единая конфигурация через `OrchestratorConfig`
- ✅ Поддержка пресетов (production_small, production_large, quick_experiment, conservative)
- ✅ Автоматический выбор оптимальных параметров
- ✅ Встроенная поддержка Feature Store и MLflow

**Статус:** ✅ Доступен для импорта, можно использовать для продвинутых сценариев.

### Вариант 2: Явное создание компонентов

```python
from backend.ml_engine.models.hybrid_cnn_lstm_v2 import create_model_v2_from_preset
from backend.ml_engine.training.model_trainer_v2 import create_trainer_v2

# Создание модели с пресетом
model = create_model_v2_from_preset("production_small")

# Создание trainer
trainer = create_trainer_v2(model, preset="production_small")

# Обучение
history = trainer.train(train_loader, val_loader)
```

### Вариант 3: Через CLI

```bash
# Используется оптимизированная v2 версия
python train_model.py --epochs 150 --batch-size 256

# Или через оптимизированный скрипт
python backend/ml_optimized/scripts/run_optimized_training.py --preset production_small
```

---

## 🔧 Пресеты конфигураций

### production_small (7-30 дней данных)

```python
learning_rate: 5e-5
batch_size: 256
weight_decay: 0.01
epochs: 150
dropout: 0.4
focal_gamma: 2.5
mixup_alpha: 0.2
label_smoothing: 0.1
```

### production_large (60+ дней данных)

```python
learning_rate: 1e-4
batch_size: 128
weight_decay: 0.005
epochs: 100
dropout: 0.3
focal_gamma: 2.0
```

### quick_experiment (быстрые тесты)

```python
learning_rate: 1e-4
batch_size: 128
epochs: 30
use_augmentation: False
```

### conservative (консервативная торговля)

```python
learning_rate: 3e-5
batch_size: 256
weight_decay: 0.02
dropout: 0.5
focal_gamma: 3.0
label_smoothing: 0.15
```

---

## 📁 Структура v2 файлов

```
backend/ml_engine/
├── models/
│   ├── hybrid_cnn_lstm.py          # v1 (обновлен, но устарел)
│   └── hybrid_cnn_lstm_v2.py       # ✅ v2 (ИСПОЛЬЗУЕТСЯ)
│
├── training/
│   ├── model_trainer.py            # v1 (обновлен, но устарел)
│   ├── model_trainer_v2.py         # ✅ v2 (ИСПОЛЬЗУЕТСЯ)
│   ├── class_balancing.py          # v1 (обновлен, но устарел)
│   ├── class_balancing_v2.py       # ✅ v2 (ИСПОЛЬЗУЕТСЯ)
│   ├── losses.py                   # ✅ v2 (НОВОЕ)
│   └── augmentation.py             # ✅ v2 (НОВОЕ)
│
├── inference/
│   ├── model_server.py             # Обновлен для v2
│   └── model_server_v2.py          # Альтернативная версия
│
└── training_orchestrator_v2.py     # ✅ Расширенный orchestrator
```

---

## ⚠️ Важные замечания

### 1. GPU Memory

v2 модель с batch_size=256 требует ~4GB GPU памяти. На CPU обучение будет медленным.

**Решение для ограниченной памяти:**
```python
config = TrainerConfigV2(
    batch_size=128,  # Уменьшить
    gradient_accumulation_steps=2  # Компенсировать
)
```

### 2. Compatibility с Production

Все endpoints и API сохраняют обратную совместимость благодаря алиасам.

### 3. Monitoring

v2 trainer автоматически логирует все новые метрики в MLflow:
- Learning rate по эпохам
- Augmentation parameters
- Loss components (direction, confidence, return)

---

## 🐛 Troubleshooting

### Проблема: ImportError при импорте v2

**Решение:**
```bash
# Убедитесь, что все файлы v2 существуют
ls backend/ml_engine/models/hybrid_cnn_lstm_v2.py
ls backend/ml_engine/training/model_trainer_v2.py
```

### Проблема: Out of Memory

**Решение:**
```python
# Уменьшить batch_size
config.batch_size = 128

# Или использовать gradient accumulation
config.gradient_accumulation_steps = 2
```

### Проблема: Метрики не улучшаются

**Решение:**
1. Проверьте learning rate (должен быть 5e-5, не 0.001!)
2. Проверьте batch_size (должен быть ≥128)
3. Проверьте данные на NaN/Inf

---

## 📞 Поддержка

При возникновении проблем:
1. Проверьте логи обучения
2. Убедитесь, что используются v2 компоненты
3. Проверьте параметры согласно рекомендациям

---

**Статус:** ✅ Миграция завершена
**Дата:** 2025-01-27
**Версия:** v2.0
