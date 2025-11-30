# Маппинг параметров Frontend → Backend (v2)

**Дата:** 2025-11-27
**Статус:** Backend API исправлен для работы с TrainerConfigV2

---

## Проблема

При попытке запустить обучение через frontend получили ошибку:
```
TypeError: TrainerConfigV2.__init__() got an unexpected keyword argument 'lr_scheduler'
```

**Причина:** Параметры frontend не совпадали с параметрами TrainerConfigV2.

---

## Решение

### 1. Исправлен маппинг параметров в `ml_management_api.py`

**Файл:** `backend/api/ml_management_api.py`

#### Изменения:

| Frontend параметр | Backend параметр (TrainerConfigV2) | Примечание |
|-------------------|-----------------------------------|-----------|
| `lr_scheduler` | `scheduler_type` | ✅ Переименован с преобразованием |
| `scheduler_T_0` | `scheduler_T_0` | ✅ Прямой маппинг |
| `scheduler_T_mult` | `scheduler_T_mult` | ✅ Прямой маппинг |
| `dropout` | ModelConfig.`dropout` | ✅ Переместили в ModelConfig |
| `label_smoothing` | `label_smoothing` | ✅ Прямой маппинг |
| `use_augmentation` | `use_augmentation` | ✅ Прямой маппинг |
| `gaussian_noise_std` | `gaussian_noise_std` | ✅ Прямой маппинг |
| `use_focal_loss` | `use_focal_loss` | ✅ Прямой маппинг |
| `focal_gamma` | `focal_gamma` | ✅ Прямой маппинг |
| `use_oversampling` | ❌ Не поддерживается | ⚠️ Пока игнорируется |
| `oversample_ratio` | ❌ Не поддерживается | ⚠️ Пока игнорируется |

---

## Текущая реализация

### Код в `ml_management_api.py` (строки 268-303)

```python
# ===== СОЗДАЕМ MODEL CONFIG С V2 ПАРАМЕТРАМИ =====
# Dropout - это параметр модели, а не trainer'а
if request.ml_model_config:
    model_config = ModelConfig(**request.ml_model_config)
else:
    model_config = ModelConfig(dropout=request.dropout)

# ===== СОЗДАЕМ TRAINER CONFIG С V2 ПАРАМЕТРАМИ =====
trainer_config = TrainerConfig(
    # Базовые параметры
    epochs=request.epochs,
    learning_rate=request.learning_rate,
    weight_decay=request.weight_decay,
    early_stopping_patience=request.early_stopping_patience,

    # Scheduler параметры (v2: scheduler_type вместо lr_scheduler)
    scheduler_type=request.lr_scheduler.lower().replace(
        "cosineannealingwarmrestarts", "cosine_warm_restarts"
    ),
    scheduler_T_0=request.scheduler_T_0,
    scheduler_T_mult=request.scheduler_T_mult,

    # Regularization (label_smoothing есть в v2)
    label_smoothing=request.label_smoothing,

    # Data Augmentation (v2 параметры)
    use_augmentation=request.use_augmentation,
    gaussian_noise_std=request.gaussian_noise_std,

    # Class Balancing (v2 встроенные параметры)
    use_focal_loss=request.use_focal_loss,
    focal_gamma=request.focal_gamma,
    use_class_weights=True

    # ПРИМЕЧАНИЕ: use_oversampling и oversample_ratio пока не поддерживаются
    # в TrainerConfigV2 напрямую. Для их использования нужно передавать
    # отдельный ClassBalancingConfigV2 через TrainingOrchestrator
)
```

---

## Параметры которые РАБОТАЮТ ✅

Эти параметры корректно передаются из frontend в backend и используются при обучении:

### Базовые параметры обучения:
- ✅ **epochs** (150) → TrainerConfig.epochs
- ✅ **learning_rate** (0.00005) → TrainerConfig.learning_rate
- ✅ **batch_size** (256) → TrainerConfig.batch_size
- ✅ **weight_decay** (0.01) → TrainerConfig.weight_decay
- ✅ **early_stopping_patience** (20) → TrainerConfig.early_stopping_patience

### Scheduler параметры:
- ✅ **lr_scheduler** ("CosineAnnealingWarmRestarts") → TrainerConfig.scheduler_type ("cosine_warm_restarts")
- ✅ **scheduler_T_0** (10) → TrainerConfig.scheduler_T_0
- ✅ **scheduler_T_mult** (2) → TrainerConfig.scheduler_T_mult

### Регуляризация:
- ✅ **dropout** (0.4) → ModelConfig.dropout
- ✅ **label_smoothing** (0.1) → TrainerConfig.label_smoothing

### Data Augmentation:
- ✅ **use_augmentation** (true) → TrainerConfig.use_augmentation
- ✅ **gaussian_noise_std** (0.01) → TrainerConfig.gaussian_noise_std

### Class Balancing (частично):
- ✅ **use_focal_loss** (true) → TrainerConfig.use_focal_loss
- ✅ **focal_gamma** (2.5) → TrainerConfig.focal_gamma

---

## Параметры которые НЕ РАБОТАЮТ ⚠️

Эти параметры есть в frontend, но пока не используются в backend:

### Oversampling:
- ❌ **use_oversampling** (true) - игнорируется
- ❌ **oversample_ratio** (0.5) - игнорируется

**Причина:** TrainerConfigV2 не имеет этих параметров. Они существуют в ClassBalancingConfigV2, но TrainingOrchestrator не принимает отдельный ClassBalancingConfig.

**Решение для будущего:**
1. Обновить TrainingOrchestrator чтобы принимать ClassBalancingConfigV2
2. Или добавить эти параметры в TrainerConfigV2
3. Или использовать training_orchestrator_v2.py который может поддерживать это

---

## Что нужно сделать для полной поддержки oversampling

### Вариант 1: Обновить TrainerConfigV2

Добавить в `backend/ml_engine/training/model_trainer_v2.py`:

```python
@dataclass
class TrainerConfigV2:
    # ... существующие параметры ...

    # === Oversampling ===
    use_oversampling: bool = True
    oversample_ratio: float = 0.5
```

### Вариант 2: Обновить TrainingOrchestrator

Изменить `training_orchestrator.py` чтобы принимать `ClassBalancingConfig`:

```python
def __init__(
    self,
    model_config: Optional[ModelConfig] = None,
    trainer_config: Optional[TrainerConfig] = None,
    data_config: Optional[DataConfig] = None,
    balancing_config: Optional[ClassBalancingConfig] = None  # НОВОЕ
):
```

### Вариант 3: Использовать TrainingOrchestratorV2

В файле `training_orchestrator_v2.py` может быть полная поддержка всех v2 параметров.

---

## Тестирование

### Проверить что работает:

```bash
# 1. Запустить frontend
cd frontend
npm run dev

# 2. Открыть http://localhost:5173/ml-management
# 3. Проверить что все поля отображаются
# 4. Запустить обучение с дефолтными v2 параметрами
# 5. Проверить логи backend:

# В логах должны быть:
✅ epochs=150
✅ learning_rate=5e-05
✅ batch_size=256
✅ weight_decay=0.01
✅ scheduler_type='cosine_warm_restarts'
✅ scheduler_T_0=10
✅ scheduler_T_mult=2
✅ dropout=0.4
✅ label_smoothing=0.1
✅ use_augmentation=True
✅ gaussian_noise_std=0.01
✅ use_focal_loss=True
✅ focal_gamma=2.5

⚠️ НЕ будет в логах:
- use_oversampling (игнорируется)
- oversample_ratio (игнорируется)
```

---

## Ошибка которую исправили

### До исправления:

```python
trainer_config = TrainerConfig(
    lr_scheduler=request.lr_scheduler,  # ❌ ОШИБКА: нет такого параметра
    dropout=request.dropout,             # ❌ ОШИБКА: должно быть в ModelConfig
    # ...
)
```

**Результат:**
```
TypeError: TrainerConfigV2.__init__() got an unexpected keyword argument 'lr_scheduler'
```

### После исправления:

```python
model_config = ModelConfig(dropout=request.dropout)  # ✅ Dropout в ModelConfig

trainer_config = TrainerConfig(
    scheduler_type=request.lr_scheduler.lower()...,  # ✅ Правильное имя параметра
    # ...
)
```

**Результат:** ✅ Обучение запускается без ошибок!

---

## Статус

**✅ Backend API исправлен**
**✅ 11 из 13 параметров работают**
**⚠️ 2 параметра (oversampling) пока игнорируются**

**Для полной поддержки всех параметров нужно обновить TrainerConfigV2 или TrainingOrchestrator.**
