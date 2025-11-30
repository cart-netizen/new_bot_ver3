# 🎯 Class Balancing Guide

## Проблема дисбаланса классов

В торговых сигналах часто бывает дисбаланс:
- **HOLD (0)**: 70-80% данных (большинство времени рынок стабилен)
- **BUY (1)**: 10-15% данных
- **SELL (2)**: 10-15% данных

**Проблема**: Модель учится всегда предсказывать HOLD → высокая accuracy, но не полезно!

---

## ✅ Реализованное решение

### Двухуровневая балансировка:

#### 1. **Data-level Balancing** (в DataLoader)
```python
ClassBalancingConfig(
    use_oversampling=True,       # Дублируем minority классы (BUY, SELL)
    oversample_strategy="auto",  # Автоматический выбор стратегии
    oversample_ratio=1.0,        # Целевое соотношение классов
)
```

**Как работает:**
- Случайное копирование семплов из minority классов
- Приводит к равному распределению: HOLD≈33%, BUY≈33%, SELL≈33%
- Применяется **ДО** создания sequences

**Пример:**
```
ДО:
  • HOLD: 7000 samples (70%)
  • BUY:  1500 samples (15%)
  • SELL: 1500 samples (15%)

ПОСЛЕ oversampling:
  • HOLD: 7000 samples (33%)
  • BUY:  7000 samples (33%) ← дублированы
  • SELL: 7000 samples (33%) ← дублированы
```

#### 2. **Loss-level Balancing** (в Trainer)
```python
TrainerConfigV2(
    use_class_weights=True,  # Взвешенный loss
    use_focal_loss=True,     # Фокус на сложных примерах
    focal_gamma=2.5          # Сила фокусировки
)
```

**Как работает:**
- **Class Weights**: Больший вес для minority классов в loss function
- **Focal Loss**: Уменьшает вес легких примеров, увеличивает сложных
- Модель больше "штрафуется" за ошибки на minority классах

---

## 📊 Сравнение методов

| Метод | Где применяется | Преимущества | Недостатки |
|-------|----------------|--------------|------------|
| **Oversampling** | DataLoader | Просто, работает всегда | Может переобучиться на дубликатах |
| **Class Weights** | Loss function | Не дублирует данные | Требует подбора весов |
| **Focal Loss** | Loss function | Фокус на сложных случаях | Сложнее в настройке |

**Лучший результат**: Комбинация всех методов ✅

---

## 🔧 Текущая конфигурация

В `training_orchestrator.py`:

```python
balancing_config = ClassBalancingConfig(
    use_class_weights=True,   # ✅ Для loss function
    use_oversampling=True,    # ✅ Для data loader
    use_focal_loss=False,     # ❌ Focal loss уже в TrainerConfig
    oversample_strategy="auto",
    oversample_ratio=1.0,
    verbose=True
)
```

В `model_trainer_v2.py` (TrainerConfigV2):

```python
use_class_weights=True  # Веса классов для loss
use_focal_loss=True     # Focal loss (gamma=2.5)
focal_gamma=2.5         # Сила фокусировки
```

---

## 📈 Ожидаемые результаты

### До балансировки:
```
Accuracy: 75%
Precision: HOLD=0.80, BUY=0.30, SELL=0.30
Recall:    HOLD=0.95, BUY=0.10, SELL=0.10
F1:        HOLD=0.87, BUY=0.15, SELL=0.15

Проблема: Модель просто предсказывает HOLD!
```

### После балансировки:
```
Accuracy: 68% (немного ниже, но полезнее!)
Precision: HOLD=0.70, BUY=0.65, SELL=0.65
Recall:    HOLD=0.75, BUY=0.60, SELL=0.60
F1:        HOLD=0.72, BUY=0.62, SELL=0.62

Результат: Модель научилась находить BUY и SELL сигналы!
```

---

## 🛠️ Мониторинг балансировки

### В логах обучения:

```
✓ Class Balancing включен в DataLoader
Распределение классов ДО resampling:
  • Class 0: 7,000 (70.0%)
  • Class 1: 1,500 (15.0%)
  • Class 2: 1,500 (15.0%)

ПРИМЕНЕНИЕ CLASS BALANCING
Распределение классов ПОСЛЕ resampling:
  • Class 0: 7,000 (33.3%)
  • Class 1: 7,000 (33.3%)
  • Class 2: 7,000 (33.3%)
Новый размер данных: 21,000 samples
```

---

## 🎮 Альтернативные стратегии

### Aggressive (для сильного дисбаланса):
```python
ClassBalancingConfig(
    use_oversampling=True,
    use_undersampling=True,  # + undersampling majority
    oversample_ratio=1.0,
    undersample_ratio=0.5    # Уменьшить HOLD
)
```

### Conservative (если переобучение):
```python
ClassBalancingConfig(
    use_oversampling=True,
    oversample_ratio=0.5,    # Меньше дубликатов
    use_class_weights=True,
    use_focal_loss=False     # Без focal loss
)
```

### SMOTE (синтетические данные):
```python
ClassBalancingConfig(
    use_smote=True,           # Вместо копирования
    smote_k_neighbors=5,
    smote_sampling_strategy="auto"
)
```

---

## 🐛 Troubleshooting

### Модель всё ещё предсказывает только HOLD:

**Причина 1**: Балансировка не применяется
```bash
# Проверить в логах:
grep "Class Balancing" logs/training.log

# Должно быть:
✓ Class Balancing включен в DataLoader
```

**Решение**: Убедитесь, что `apply_resampling=True` в load_from_dataframe()

---

**Причина 2**: Слишком слабая балансировка
```python
# Попробовать aggressive:
oversample_ratio=1.0  # Полная балансировка
use_undersampling=True  # + уменьшить majority
```

---

**Причина 3**: Focal loss конфликтует
```python
# Временно отключить focal loss:
use_focal_loss=False
focal_gamma=0.0
```

---

### Переобучение на minority классах:

**Симптомы**: Train F1=0.90, Val F1=0.45

**Решение**: Уменьшить дубликаты
```python
oversample_ratio=0.5  # Было 1.0
use_oversampling=True
use_focal_loss=True  # Компенсировать через loss
```

---

### Out of Memory после балансировки:

**Причина**: Oversampling увеличивает размер датасета

**Решение**:
```python
# Вариант 1: Undersampling вместо oversampling
use_oversampling=False
use_undersampling=True
undersample_ratio=0.5  # Уменьшить HOLD

# Вариант 2: Меньший batch size
batch_size=64  # Было 128
```

---

## 📝 Checklist

- [ ] `balancing_config` передан в TrainingOrchestrator
- [ ] `apply_resampling=True` в load_from_dataframe()
- [ ] В логах видно "Class Balancing включен"
- [ ] Распределение классов изменилось после resampling
- [ ] `use_class_weights=True` в TrainerConfig
- [ ] `use_focal_loss=True` в TrainerConfig
- [ ] F1 score для BUY и SELL > 0.5

---

## 📚 Дополнительные ресурсы

- [Imbalanced-learn docs](https://imbalanced-learn.org/)
- [Focal Loss paper](https://arxiv.org/abs/1708.02002)
- [SMOTE algorithm](https://arxiv.org/abs/1106.1813)

---

Создано: 2025-11-27
Статус: ✅ Активно в training_orchestrator.py
Методы: Oversampling + Class Weights + Focal Loss
