# 🔧 HOTFIX: v2 Compatibility Fix

## Проблема

```
AttributeError: 'EpochMetrics' object has no attribute 'get'
```

**Причина:** `training_orchestrator.py` ожидал, что `trainer.train()` вернет список словарей, но v2 trainer возвращает список объектов `EpochMetrics`.

## ✅ Решение

### Исправленный файл: `backend/ml_engine/training_orchestrator.py`

**Строки 196-222** обновлены для поддержки обоих форматов:

```python
# Handle both v2 (EpochMetrics) and v1 (dict) formats
if hasattr(final_epoch, 'to_dict'):
    # v2 format: EpochMetrics object
    final_epoch_dict = final_epoch.to_dict()
    history_dicts = [m.to_dict() if hasattr(m, 'to_dict') else m for m in training_history]
else:
    # v1 format: dict
    final_epoch_dict = final_epoch
    history_dicts = training_history
```

**Ключевые изменения:**
1. ✅ Проверка типа объекта с помощью `hasattr(final_epoch, 'to_dict')`
2. ✅ Автоматическая конвертация `EpochMetrics` в dict через `to_dict()`
3. ✅ Поддержка обоих форматов имен полей (`train_accuracy` и `train_acc`)
4. ✅ Обратная совместимость с v1 trainer

## 📊 Структура EpochMetrics (v2)

```python
@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float       # v2 имя
    val_accuracy: float         # v2 имя
    val_precision: float
    val_recall: float
    val_f1: float
    learning_rate: float
    epoch_time: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

**vs v1 формат (dict):**
```python
{
    "epoch": 1,
    "train_loss": 0.5,
    "val_loss": 0.6,
    "train_acc": 0.7,           # v1 имя
    "val_acc": 0.65,            # v1 имя
    ...
}
```

## 🎯 Compatibility Layer

Код теперь поддерживает:
- ✅ v2 trainer → `EpochMetrics` объекты
- ✅ v1 trainer → словари
- ✅ Разные имена полей (`accuracy` vs `acc`)
- ✅ Автоматическое определение формата

## 📝 Где применяется

**Файл:** `backend/ml_engine/training_orchestrator.py`
**Метод:** `train_model()` → обработка результатов обучения

**Затрагивает:**
- MLflow logging
- Final metrics extraction
- Training history processing

## ⚠️ Альтернатива: Использовать training_orchestrator_v2.py

Для полной v2 интеграции можно переключиться на `training_orchestrator_v2.py`:

```python
# Вместо:
from backend.ml_engine.training_orchestrator import TrainingOrchestrator

# Использовать:
from backend.ml_engine.training_orchestrator_v2 import TrainingOrchestratorV2 as TrainingOrchestrator
```

**Преимущества v2 orchestrator:**
- Полная интеграция с v2 компонентами
- Поддержка пресетов (production_small, production_large и т.д.)
- Расширенная конфигурация через `OrchestratorConfig`
- Нет compatibility issues

## 🚀 Рекомендации

### Краткосрочно (сейчас)
- ✅ Используйте исправленный `training_orchestrator.py`
- ✅ Все работает с v2 trainer
- ✅ Полная обратная совместимость

### Долгосрочно (рекомендуется)
- 🔄 Постепенный переход на `training_orchestrator_v2.py`
- 🔄 Обновление импортов в API и скриптах
- 🔄 Использование пресетов для упрощения конфигурации

## ✅ Статус

**Исправлено:** 2025-01-27
**Версия:** v2.0.1
**Тестировано:** ✅ Обучение проходит без ошибок
