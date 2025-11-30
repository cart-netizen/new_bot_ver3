# 🚀 Оптимизация ML модели для Production Trading

## Обзор изменений

Данный пакет содержит **industry-standard оптимизации** для ML модели криптотрейдинга.
Все изменения направлены на достижение **реальных торговых результатов** с минимальным риском.

---

## 📊 Целевые метрики

| Метрика | Было | После оптимизации | Industry Standard |
|---------|------|-------------------|-------------------|
| Accuracy | 33.89% | 55-62% | 55-65% |
| F1 Score | 32.50% | 52-58% | 50-60% |
| Val Loss | 1.219 | <0.7 | <0.6 |
| Train-Val Gap | 0.295 | <0.08 | <0.1 |

---

## 🔴 КРИТИЧЕСКИЕ ИЗМЕНЕНИЯ

### 1. Learning Rate: `0.001` → `5e-5`

**Почему это критично:**
- Финансовые временные ряды очень шумные
- Высокий LR = модель "перескакивает" оптимумы
- При LR=0.001 модель не может стабильно обучаться

```python
# ❌ БЫЛО (плохо)
learning_rate = 0.001

# ✅ СТАЛО (правильно)
learning_rate = 5e-5  # В 20 раз меньше!
```

### 2. Batch Size: `64` → `256`

**Почему это критично:**
- Маленький batch = нестабильные градиенты
- Особенно важно при шумных финансовых данных
- Больший batch = более надёжные обновления весов

```python
# ❌ БЫЛО (плохо)
batch_size = 64

# ✅ СТАЛО (правильно)
batch_size = 256  # В 4 раза больше
```

### 3. Weight Decay: `~0` → `0.01`

**Почему это критично:**
- Отсутствие L2 регуляризации = overfitting
- Особенно при малом датасете (~15K samples)
- Weight decay помогает генерализации

```python
# ❌ БЫЛО (плохо)
weight_decay = 1e-5  # Почти ноль

# ✅ СТАЛО (правильно)
weight_decay = 0.01  # Значимая регуляризация
```

---

## 📁 Структура файлов

```
ml_optimized/
├── configs/
│   └── optimized_configs.py      # Все конфигурации
├── models/
│   └── hybrid_cnn_lstm_v2.py     # Улучшенная модель
├── training/
│   ├── losses.py                 # Loss functions
│   ├── augmentation.py           # Data augmentation
│   ├── model_trainer_v2.py       # Улучшенный trainer
│   └── class_balancing_v2.py     # Class balancing
├── integration/
│   └── optimized_ml_integration.py  # Интеграция
├── scripts/
│   └── run_optimized_training.py    # Скрипт запуска
└── OPTIMIZATION_GUIDE.md         # Этот файл
```

---

## 🛠 Инструкция по применению

### Вариант 1: Минимальные изменения (Quick Fix)

Изменить **только 3 параметра** в существующем коде:

```python
# backend/ml_engine/training/model_trainer.py

@dataclass
class TrainerConfig:
    # ИЗМЕНИТЬ ЭТИ 3 ПАРАМЕТРА:
    learning_rate: float = 5e-5      # Было: 0.001
    weight_decay: float = 0.01       # Было: ~0
    # batch_size изменить в DataConfig!

# backend/ml_engine/training/data_loader.py

@dataclass
class DataConfig:
    batch_size: int = 256            # Было: 64
```

### Вариант 2: Полная интеграция

1. **Скопировать файлы** из `ml_optimized/` в соответствующие директории:

```bash
# Копирование файлов
cp ml_optimized/training/losses.py backend/ml_engine/training/
cp ml_optimized/training/augmentation.py backend/ml_engine/training/
cp ml_optimized/training/class_balancing_v2.py backend/ml_engine/training/
cp ml_optimized/integration/optimized_ml_integration.py backend/ml_engine/integration/
```

2. **Использовать оптимизированные конфигурации:**

```python
from backend.ml_engine.integration.optimized_ml_integration import (
    setup_optimized_training,
    quick_start_training
)

# Полная настройка
model_cfg, trainer_cfg, data_cfg, balance_cfg = setup_optimized_training()

# Или быстрый старт
model, history = quick_start_training(["BTCUSDT", "ETHUSDT"])
```

### Вариант 3: Запуск через скрипт

```bash
python backend/ml_engine/scripts/run_optimized_training.py \
    --symbols BTCUSDT ETHUSDT \
    --days 30 \
    --preset production_small \
    --output-dir checkpoints/optimized
```

---

## 🎯 Пресеты конфигураций

### `production_small` (Рекомендуется для 7-30 дней)

```python
{
    "learning_rate": 5e-5,
    "batch_size": 256,
    "weight_decay": 0.01,
    "epochs": 150,
    "dropout": 0.4,
    "focal_gamma": 2.5,
    "mixup_alpha": 0.2,
    "label_smoothing": 0.1
}
```

### `production_large` (Для 60+ дней)

```python
{
    "learning_rate": 1e-4,
    "batch_size": 128,
    "weight_decay": 0.005,
    "epochs": 100,
    "dropout": 0.3,
    "focal_gamma": 2.0,
    "mixup_alpha": 0.1,
    "label_smoothing": 0.05
}
```

### `quick_experiment` (Быстрые тесты)

```python
{
    "learning_rate": 1e-4,
    "batch_size": 128,
    "epochs": 30,
    "use_augmentation": False,
    "early_stopping_patience": 10
}
```

### `conservative` (Консервативная торговля)

```python
{
    "learning_rate": 3e-5,
    "batch_size": 256,
    "weight_decay": 0.02,
    "dropout": 0.5,
    "focal_gamma": 3.0,
    "label_smoothing": 0.15
}
```

---

## 📋 Чеклист перед запуском Production

- [ ] Learning rate ≤ 1e-4 (рекомендуется 5e-5)
- [ ] Batch size ≥ 128 (рекомендуется 256)
- [ ] Weight decay ≥ 0.001 (рекомендуется 0.01)
- [ ] Focal Loss включён (gamma ≥ 2.0)
- [ ] Class weights включены
- [ ] Early stopping patience ≥ 15
- [ ] Validation accuracy > 50%
- [ ] Train-Val loss gap < 0.15

---

## 🔬 Компоненты оптимизации

### 1. Loss Functions (`training/losses.py`)

- **LabelSmoothingCrossEntropy**: Предотвращает overconfidence
- **FocalLossV2**: Фокус на hard examples (gamma=2.5)
- **MultiTaskLossV2**: Direction + Confidence + Return
- **DirectionalAccuracyLoss**: Штраф за противоположное направление

### 2. Data Augmentation (`training/augmentation.py`)

- **MixUp**: Смешивание samples (alpha=0.2)
- **Time Masking**: Маскирование временных шагов
- **Gaussian Noise**: Добавление шума (std=0.01)
- **Feature Dropout**: Dropout отдельных features

### 3. Class Balancing (`training/class_balancing_v2.py`)

- **Adaptive Threshold**: Percentile-based labeling
- **Class Weights**: Balanced / Sqrt / Effective
- **Oversampling**: Random oversampling для minority

### 4. Model Architecture (`models/hybrid_cnn_lstm_v2.py`)

- **Residual Connections**: Улучшенный gradient flow
- **Multi-Head Attention**: 4 heads вместо 1
- **Layer Normalization**: Лучше для sequences
- **GELU Activation**: Лучше чем ReLU для transformers

---

## ⚠️ Известные ограничения

1. **Объём данных**: При < 7 днях данных даже оптимизированная модель
   может показывать нестабильные результаты. Рекомендуется минимум 14 дней.

2. **Class Imbalance**: При сильном дисбалансе (HOLD > 80%) используйте
   агрессивный oversampling или измените thresholds для labeling.

3. **GPU Memory**: При batch_size=256 и sequence_length=60 требуется
   ~4GB GPU памяти. На CPU обучение будет медленным.

---

## 📈 Ожидаемый прогресс обучения

```
Epoch 1-10:   val_loss ↓ быстро, val_acc ~35-40%
Epoch 10-30:  val_loss ↓ умеренно, val_acc ~40-48%
Epoch 30-70:  val_loss ↓ медленно, val_acc ~48-55%
Epoch 70-100: val_loss стабилизируется, val_acc ~55-60%
Epoch 100+:   Fine-tuning, val_acc 58-62%
```

Если val_loss не падает после 20 эпох:
1. Проверьте learning rate (должен быть ~5e-5)
2. Проверьте batch size (должен быть ≥128)
3. Проверьте данные на наличие NaN/Inf

---

## 🆘 Troubleshooting

### Проблема: Val loss не падает

```python
# Решение 1: Уменьшить learning rate
learning_rate = 1e-5  # Ещё меньше

# Решение 2: Увеличить weight decay
weight_decay = 0.02  # Больше регуляризации

# Решение 3: Проверить данные
assert not np.isnan(X_train).any()
assert not np.isinf(X_train).any()
```

### Проблема: Overfitting (train_loss << val_loss)

```python
# Решение: Увеличить регуляризацию
dropout = 0.5  # Было 0.4
weight_decay = 0.02  # Было 0.01
mixup_alpha = 0.3  # Было 0.2
label_smoothing = 0.15  # Было 0.1
```

### Проблема: Accuracy ~33% (random)

```python
# Решение 1: Проверить class imbalance
from collections import Counter
print(Counter(y_train))  # Должно быть ~равномерно

# Решение 2: Использовать adaptive threshold
use_adaptive_threshold = True
percentile_sell = 0.30  # Было 0.25
percentile_buy = 0.70   # Было 0.75
```

---

## 📝 Версионирование

- **v1.0** (текущая): Базовые оптимизации
- **v1.1** (планируется): Curriculum Learning
- **v1.2** (планируется): Ensemble моделей
- **v2.0** (планируется): Transformer architecture

---

## 👤 Контакты

При возникновении проблем обращайтесь в Issues или создавайте PR.

---

*Документация актуальна для версии оптимизаций от 2024-01-XX*
