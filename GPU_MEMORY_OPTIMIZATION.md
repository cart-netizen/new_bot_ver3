# 🎮 GPU Memory Optimization для RTX 3060 12GB

## ✅ Применённые исправления (v1.0)

### Проблема
```
CUDA out of memory. Tried to allocate 1.35 GiB.
GPU 0 has a total capacity of 12.00 GiB of which 4.62 GiB is free.
```

### Решение

#### 1. **Уменьшен batch_size**
```python
# model_trainer_v2.py:82
batch_size: int = 128  # Было 256
```
**Экономия:** ~3.5 GB GPU памяти

#### 2. **Gradient Accumulation** (сохраняем эффективный batch)
```python
# model_trainer_v2.py:87
gradient_accumulation_steps: int = 2  # Эффективный batch = 128*2 = 256
```
**Эффект:** Тот же размер batch, но в 2 прохода (меньше памяти за раз)

#### 3. **CUDA Memory Allocator**
```python
# training_orchestrator.py:26
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```
**Эффект:** Предотвращает фрагментацию памяти, динамически расширяет сегменты

#### 4. **Периодическая очистка GPU кеша**
```python
# model_trainer_v2.py:270
torch.cuda.empty_cache()  # При инициализации

# model_trainer_v2.py:653
if (batch_idx + 1) % 50 == 0:
    torch.cuda.empty_cache()  # Каждые 50 батчей
```
**Эффект:** Освобождает фрагментированную память

#### 5. **Оптимизация num_workers**
```python
# data_loader.py:52
num_workers: int = 4  # Было 8
```
**Причина:** Слишком много workers → больше CPU памяти → давление на GPU

---

## 📊 Использование памяти

### До оптимизации:
- Batch size: 256
- Memory allocated: 5.04 GiB
- Memory reserved: 1.28 GiB
- **Attempted allocation: 1.35 GiB → OOM** ❌

### После оптимизации:
- Batch size: 128 (× 2 accumulation)
- Expected allocated: ~3.0-3.5 GiB
- Memory reserved: ~0.8 GiB
- **Free memory: ~7-8 GiB** ✅

---

## 🔍 Мониторинг памяти

### Во время обучения:
```bash
# Windows PowerShell
nvidia-smi -l 1

# Ищите:
# - Memory-Usage: должно быть ~50-70% (6-8 GB из 12 GB)
# - GPU-Util: должно быть 95-100%
```

### Python код (в trainer):
```python
import torch

# Текущее использование
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Максимальное использование
max_allocated = torch.cuda.max_memory_allocated() / 1e9
print(f"Max allocated: {max_allocated:.2f} GB")
```

---

## 🛠️ Дальнейшие оптимизации (если нужно)

### Вариант A: Ещё меньше batch_size
```python
batch_size: int = 64  # Было 128
gradient_accumulation_steps: int = 4  # Эффективный = 256
```
**Когда:** Если всё ещё OOM

### Вариант B: Отключить Mixed Precision (временно)
```python
use_mixed_precision: bool = False  # Было True
```
**Когда:** Для отладки (mixed precision иногда требует больше памяти)

### Вариант C: Gradient Checkpointing
```python
# В модели HybridCNNLSTM добавить:
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Вместо обычного forward:
    x = checkpoint(self.cnn_blocks, x)
    x = checkpoint(self.lstm, x)
    ...
```
**Эффект:** Экономия ~30-50% памяти за счёт перевычисления в backward

### Вариант D: Уменьшить размер модели
```python
# В ModelConfig:
cnn_channels: Tuple[int, ...] = (32, 64, 128)  # Было (64, 128, 256)
lstm_hidden: int = 128  # Было 256
```
**Эффект:** Меньше параметров = меньше памяти

---

## 🎯 Рекомендуемая конфигурация для RTX 3060 12GB

### Production (текущая):
```python
TrainerConfigV2(
    batch_size=128,
    gradient_accumulation_steps=2,
    use_mixed_precision=True,
    num_workers=4,
)
```
**Memory usage:** ~60-70% (7-8 GB)
**Speed:** ~2-3 batch/s

### Conservative (если OOM):
```python
TrainerConfigV2(
    batch_size=64,
    gradient_accumulation_steps=4,
    use_mixed_precision=False,
    num_workers=2,
)
```
**Memory usage:** ~40-50% (5-6 GB)
**Speed:** ~1.5-2 batch/s

### Aggressive (если есть запас):
```python
TrainerConfigV2(
    batch_size=192,
    gradient_accumulation_steps=1,
    use_mixed_precision=True,
    num_workers=6,
)
```
**Memory usage:** ~80-90% (10-11 GB)
**Speed:** ~3-4 batch/s
⚠️ **Риск OOM!**

---

## 🐛 Troubleshooting

### OOM в середине эпохи:
**Причина:** Фрагментация памяти
**Решение:**
```python
# Увеличить частоту очистки кеша
if (batch_idx + 1) % 25 == 0:  # Было 50
    torch.cuda.empty_cache()
```

### OOM на валидации:
**Причина:** Validation batch тоже большой
**Решение:**
```python
# В _validate_epoch использовать меньший batch
val_batch_size = self.config.batch_size // 2
```

### OOM после нескольких эпох:
**Причина:** Утечка памяти (references)
**Решение:**
```python
# После каждой эпохи:
torch.cuda.empty_cache()
gc.collect()
```

### Mixed Precision не помогает:
**Причина:** Mixed precision сохраняет 2 копии весов (FP32 + FP16)
**Решение:** Отключить на время отладки

---

## 📝 Checklist перед запуском обучения

- [ ] `batch_size <= 128` для RTX 3060 12GB
- [ ] `gradient_accumulation_steps >= 2` для эффективного большого batch
- [ ] `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` установлен
- [ ] `num_workers <= 4` для экономии CPU/GPU памяти
- [ ] Закрыты другие GPU процессы (браузер, игры, etc.)
- [ ] Запущен `nvidia-smi -l 1` для мониторинга
- [ ] Проверено наличие `torch.cuda.empty_cache()` в коде

---

## 📚 Полезные ссылки

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Mixed Precision Training](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)

---

Создано: 2025-11-27
Видеокарта: RTX 3060 12GB
Статус: ✅ Исправлено
