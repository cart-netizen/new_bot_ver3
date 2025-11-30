# 🔍 NaN Loss Troubleshooting Guide

## Проблема
```
Train Epoch 1:   0%|          | 1/859 [00:07<1:46:25,  7.44s/batch, loss=nan]
```

Loss становится NaN на первом (или ранних) батчах обучения.

---

## ✅ Применённые исправления (v1.0)

### 1. **Отключен Mixed Precision** (ВРЕМЕННО)
```python
# model_trainer_v2.py:123
use_mixed_precision: bool = False  # Было True
```

**Причина**: Mixed precision (FP16) + gradient accumulation может вызывать переполнение

**Когда включать обратно**:
- После успешного обучения без mixed precision
- Когда будет найдена и исправлена root cause
- Постепенно, с мониторингом loss

### 2. **Добавлена проверка на NaN/Inf**
```python
# В _train_epoch():
if torch.isnan(loss) or torch.isinf(loss):
    logger.warning(f"NaN/Inf loss detected at batch {batch_idx}! Skipping batch.")
    continue
```

**Эффект**: Пропускает "плохие" батчи вместо краша обучения

### 3. **Проверка градиентов**
```python
grad_norm = torch.nn.utils.clip_grad_norm_(...)
if torch.isnan(grad_norm) or torch.isinf(grad_norm):
    logger.warning(f"NaN/Inf gradient detected! Skipping optimizer step.")
    self.optimizer.zero_grad()
    continue
```

**Эффект**: Предотвращает update весов с NaN градиентами

### 4. **Улучшен GradScaler** (когда mixed precision включен)
```python
GradScaler(
    init_scale=2.**10,      # Меньший начальный scale (было 2^16)
    growth_interval=1000    # Реже увеличиваем scale
)
```

**Эффект**: Более консервативное масштабирование градиентов

---

## 🔍 Возможные причины NaN loss

### 1. Mixed Precision Overflow ⚡
**Симптомы**: NaN на первых батчах с mixed precision

**Решение**:
```python
use_mixed_precision=False  # ✅ Уже применено
```

**Детали**: FP16 имеет меньший динамический диапазон чем FP32. Большие значения → overflow → NaN

---

### 2. Невалидные данные 📊
**Симптомы**: NaN в features или labels

**Проверка**:
```python
# В DataLoader добавить:
print(f"Features NaN: {torch.isnan(sequences).any()}")
print(f"Features Inf: {torch.isinf(sequences).any()}")
print(f"Features min/max: {sequences.min()}, {sequences.max()}")
```

**Решение**:
```python
# В data preprocessing:
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
```

---

### 3. Слишком большой Learning Rate 📈
**Симптомы**: Loss растёт или сразу NaN

**Текущее значение**: `5e-5` (очень консервативное)

**Если проблема**: Попробовать `1e-5`

---

### 4. Focal Loss + Class Imbalance 🎯
**Симптомы**: NaN после балансировки классов

**Проверка**:
```python
# Временно отключить focal loss:
use_focal_loss=False
focal_gamma=0.0
```

**Если помогло**: Уменьшить gamma
```python
focal_gamma=1.0  # Было 2.5
```

---

### 5. Gradient Explosion 💥
**Симптомы**: Градиенты становятся очень большими

**Текущая защита**:
```python
grad_clip_value=1.0  # Уже включено
```

**Если не помогает**:
```python
grad_clip_value=0.5  # Более агрессивный clipping
```

---

### 6. Batch Normalization Issues
**Симптомы**: NaN при маленьком batch size

**Текущий batch**: 128 (должно быть ОК)

**Если проблема**: Увеличить до 256 или использовать Layer Norm

---

### 7. LSTM Hidden State Overflow
**Симптомы**: NaN в LSTM forward pass

**Решение**: Уже есть в модели - orthogonal initialization

---

## 🛠️ Пошаговая диагностика

### Шаг 1: Запустить обучение с отключенным mixed precision
```bash
# Уже применено в коде
```

**Ожидаемый результат**: Loss должен быть числом (не NaN)

**Если всё ещё NaN** → переходим к шагу 2

---

### Шаг 2: Проверить данные
```python
# В training_orchestrator.py после загрузки данных:
for batch in train_loader:
    sequences = batch['sequence']
    labels = batch['label']

    print(f"Sequences shape: {sequences.shape}")
    print(f"Sequences NaN: {torch.isnan(sequences).sum()}")
    print(f"Sequences Inf: {torch.isinf(sequences).sum()}")
    print(f"Sequences range: [{sequences.min():.4f}, {sequences.max():.4f}]")
    print(f"Labels: {labels.unique()}")
    break
```

**Если есть NaN/Inf** → проблема в preprocessing

---

### Шаг 3: Упростить loss function
```python
# Временно:
use_focal_loss=False
use_class_weights=False
label_smoothing=0.0
```

**Если помогло** → проблема в loss function настройках

---

### Шаг 4: Уменьшить learning rate
```python
learning_rate=1e-5  # Было 5e-5
```

---

### Шаг 5: Отключить augmentation
```python
use_augmentation=False
mixup_alpha=0.0
```

**Если помогло** → проблема в MixUp или augmentation

---

### Шаг 6: Проверить model forward pass
```python
# Тест модели:
model = create_model()
x = torch.randn(2, 60, 110)  # batch=2, seq=60, features=110
output = model(x)

print(f"Output NaN: {torch.isnan(output['direction_logits']).any()}")
print(f"Output range: {output['direction_logits'].min()}, {output['direction_logits'].max()}")
```

---

## 📋 Checklist

После каждого изменения проверить:

- [ ] Loss - число (не NaN)
- [ ] Gradients finite (в логах нет "NaN/Inf gradient detected")
- [ ] Accuracy растёт (хотя бы немного)
- [ ] GPU memory stable (~7-8 GB)

---

## 🎯 Ожидаемые результаты

### С отключенным mixed precision:
```
Train Epoch 1:   1%|  | 10/859 [00:15<21:12, 1.50s/batch, loss=0.9234]  ✅
Train Epoch 1:   5%|▌ | 50/859 [01:15<20:45, 1.54s/batch, loss=0.8761]  ✅
```

### Если NaN всё ещё появляется:
```
⚠️ NaN/Inf loss detected at batch 42! Skipping batch.
Train Epoch 1:   5%|▌ | 50/859 [01:15<20:45, 1.54s/batch, loss=0.8761]  ⚠️
```
→ Смотреть логи, какие батчи пропускаются
→ Проверить эти батчи отдельно

---

## 🔄 План re-enable mixed precision

Когда обучение стабильно работает без mixed precision:

### 1. Проверить стабильность
- Обучение >10 эпох без NaN
- Val loss плавно снижается
- Нет warnings "NaN detected"

### 2. Re-enable с консервативными настройками
```python
use_mixed_precision=True
# GradScaler уже настроен консервативно:
# init_scale=2.**10, growth_interval=1000
```

### 3. Мониторить первые 100 батчей
```
- Если NaN → вернуть False
- Если OK → продолжить обучение
```

### 4. Если работает
```
# Можно попробовать более агрессивный scaling:
init_scale=2.**12  # Было 2.**10
```

---

## 📚 Дополнительные ресурсы

- [PyTorch AMP Troubleshooting](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [Debugging NaN in Neural Networks](https://github.com/pytorch/pytorch/issues/12633)
- [Focal Loss Numerical Stability](https://arxiv.org/abs/1708.02002)

---

Создано: 2025-11-27
Статус: ✅ Mixed precision ОТКЛЮЧЕН
NaN detection: ✅ АКТИВНА
Next: Тестировать обучение без mixed precision
