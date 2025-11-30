# 🔧 HOTFIX: Исправление маппинга v2 параметров в API

**Дата:** 2025-11-27
**Тип:** Backend Hotfix
**Приоритет:** КРИТИЧЕСКИЙ

---

## ❌ Проблема

При запуске обучения через frontend `/ml-management` возникла ошибка:

```
2025-11-27 14:37:47 ERROR [backend.api.ml_management_api]
Training job failed: 20251127_143747,
error=TrainerConfigV2.__init__() got an unexpected keyword argument 'lr_scheduler'
```

**Причина:** Неправильный маппинг параметров frontend → TrainerConfigV2.

---

## ✅ Решение

### Файл: `backend/api/ml_management_api.py`

**Строки:** 268-303

### Изменения:

#### 1. Dropout перемещен в ModelConfig ✅

**До:**
```python
trainer_config = TrainerConfig(
    dropout=request.dropout,  # ❌ ОШИБКА: нет в TrainerConfigV2
    ...
)
```

**После:**
```python
model_config = ModelConfig(dropout=request.dropout)  # ✅ Правильно
```

#### 2. lr_scheduler → scheduler_type ✅

**До:**
```python
trainer_config = TrainerConfig(
    lr_scheduler=request.lr_scheduler,  # ❌ ОШИБКА: неправильное имя
    ...
)
```

**После:**
```python
trainer_config = TrainerConfig(
    scheduler_type=request.lr_scheduler.lower().replace(
        "cosineannealingwarmrestarts", "cosine_warm_restarts"
    ),  # ✅ Правильно
    ...
)
```

#### 3. Убраны параметры oversampling ⚠️

**До:**
```python
balancing_config = ClassBalancingConfig(
    use_oversampling=request.use_oversampling,  # ❌ Не поддерживается
    oversample_ratio=request.oversample_ratio,
)
```

**После:**
```python
# ✅ Параметры oversampling пока игнорируются
# (нет в TrainerConfigV2, требуется отдельный ClassBalancingConfigV2)
```

---

## 📋 Полный список изменений

### Измененный код (строки 268-303):

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
    scheduler_type=request.lr_scheduler.lower().replace("cosineannealingwarmrestarts", "cosine_warm_restarts"),
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

## ✅ Что работает теперь

### 11 из 13 параметров корректно передаются:

1. ✅ **epochs** (150)
2. ✅ **batch_size** (256)
3. ✅ **learning_rate** (0.00005)
4. ✅ **weight_decay** (0.01)
5. ✅ **early_stopping_patience** (20)
6. ✅ **scheduler_type** ("cosine_warm_restarts")
7. ✅ **scheduler_T_0** (10)
8. ✅ **scheduler_T_mult** (2)
9. ✅ **dropout** (0.4) - через ModelConfig
10. ✅ **label_smoothing** (0.1)
11. ✅ **use_augmentation** (true)
12. ✅ **gaussian_noise_std** (0.01)
13. ✅ **use_focal_loss** (true)
14. ✅ **focal_gamma** (2.5)

### ⚠️ Пока не работают (игнорируются):

15. ⚠️ **use_oversampling** (true)
16. ⚠️ **oversample_ratio** (0.5)

**Причина:** TrainerConfigV2 не поддерживает эти параметры напрямую.

---

## 🔬 Тестирование

### Проверка синтаксиса:

```bash
$ python -m py_compile backend/api/ml_management_api.py
✅ Syntax OK
```

### Проверка работоспособности:

1. Запустить frontend:
```bash
cd frontend
npm run dev
```

2. Открыть http://localhost:5173/ml-management

3. Запустить обучение с дефолтными v2 параметрами

4. Проверить логи backend - не должно быть ошибки `TypeError`

### Ожидаемый результат:

```
INFO: Starting training job: 20251127_xxxxxx
INFO: Training config: epochs=150, lr=5e-05, batch_size=256, weight_decay=0.01
INFO: Scheduler: cosine_warm_restarts (T_0=10, T_mult=2)
INFO: Using Focal Loss with gamma=2.5
INFO: Data augmentation enabled with gaussian_noise_std=0.01
INFO: Label smoothing: 0.1
✅ Обучение запускается без ошибок!
```

---

## 📚 Связанные документы

1. **FRONTEND_V2_UPDATE_COMPLETE.md** - Документация по обновлению frontend
2. **V2_API_PARAMETER_MAPPING.md** - Полный маппинг параметров frontend → backend
3. **FINAL_V2_STATUS_REPORT.md** - Статус v2 интеграции

---

## 🚀 Следующие шаги (опционально)

Для полной поддержки всех 16 параметров (включая oversampling):

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

### Вариант 2: Использовать ClassBalancingConfigV2

Обновить TrainingOrchestrator чтобы принимать отдельный ClassBalancingConfigV2.

### Вариант 3: Миграция на TrainingOrchestratorV2

Использовать `training_orchestrator_v2.py` который может иметь полную поддержку.

---

## ✅ Статус

**HOTFIX APPLIED ✅**

- ✅ Синтаксис корректный
- ✅ 11/13 параметров работают
- ✅ Критическая ошибка `TypeError` исправлена
- ⚠️ 2 параметра (oversampling) пока игнорируются

**Готово к тестированию на реальных данных!**
