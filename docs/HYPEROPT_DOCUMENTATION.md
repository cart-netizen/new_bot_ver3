# Hyperparameter Optimization Module - Документация

## Обзор

Модуль оптимизации гиперпараметров использует **Optuna** для автоматического поиска оптимальных параметров обучения LSTM модели.

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYPEROPT ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│   Frontend UI    │────▶│  FastAPI REST    │────▶│  HyperparameterOptimizer │
│   (React)        │     │  hyperopt_api.py │     │  hyperparameter_         │
│                  │◀────│                  │◀────│  optimizer.py            │
└──────────────────┘     └──────────────────┘     └──────────────────────────┘
                                │                            │
                                │                            │
                         ┌──────▼──────┐              ┌──────▼──────┐
                         │  state.json │              │  Optuna     │
                         │  (progress) │              │  SQLite DBs │
                         └─────────────┘              └─────────────┘
```

---

## Диаграмма полного цикла работы

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ПОЛНЫЙ ЦИКЛ HYPEROPT                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │   START     │
                              └──────┬──────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │  POST /api/hyperopt/start      │
                    │  • stop_event.clear()          │
                    │  • Create job_id               │
                    │  • Save state.json             │
                    │  • Start background task       │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────┐
            │           _run_optimization()                  │
            │  • Import HyperparameterOptimizer              │
            │  • Create OptimizationConfig                   │
            │  • Pass stop_event to optimizer                │
            │  • Run in ThreadPoolExecutor                   │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────┐
            │         optimizer.optimize()                   │
            │  • Preload data ONCE                           │
            │  • Loop through parameter groups               │
            └───────────────────────┬───────────────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
                   ▼                                 ▼
    ┌──────────────────────────┐     ┌──────────────────────────┐
    │  FOR each group:         │     │  User clicks STOP        │
    │  • learning_rate         │     │  POST /api/hyperopt/stop │
    │  • regularization        │     │  • stop_event.set()      │
    │  • augmentation          │     │  • Save state.json       │
    │  • scheduler             │     │  (status="paused")       │
    │  • class_balance         │     └───────────┬──────────────┘
    │  • triple_barrier        │                 │
    └────────────┬─────────────┘                 │
                 │                               │
                 ▼                               │
    ┌──────────────────────────┐                 │
    │  _optimize_group()       │                 │
    │  • Create Optuna Study   │                 │
    │  • load_if_exists=True   │◀────────────────┘
    │  • Run study.optimize()  │    (callback checks stop_event)
    └────────────┬─────────────┘
                 │
                 ▼
    ┌──────────────────────────┐
    │  FOR each trial:         │
    │  • Sample parameters     │
    │  • Train model (N epochs)│
    │  • Evaluate metrics      │
    │  • Check mode collapse   │
    │  • Report to Optuna      │
    └────────────┬─────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌───────────────┐  ┌───────────────┐
│ Trial Done    │  │ Stop Event?   │
│ More trials?  │  │ Converged?    │
└───────┬───────┘  └───────┬───────┘
        │                  │
        │ YES              │ YES
        ▼                  ▼
┌───────────────┐  ┌───────────────┐
│ Next Trial    │  │ study.stop()  │
└───────────────┘  └───────┬───────┘
                           │
                           ▼
            ┌──────────────────────────┐
            │  Save results:           │
            │  • results.json          │
            │  • best_params.json      │
            │  • group_*.db (Optuna)   │
            └────────────┬─────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
┌───────────────────┐         ┌───────────────────┐
│  All groups done  │         │  Stopped/Paused   │
│  status=completed │         │  status=paused    │
└─────────┬─────────┘         └─────────┬─────────┘
          │                             │
          ▼                             ▼
┌───────────────────┐         ┌───────────────────┐
│ GET /results      │         │ POST /resume      │
│ Returns best      │         │ • stop_event.clear│
│ params & metrics  │         │ • Load state.json │
└───────────────────┘         │ • RESUME mode     │
                              │ • Skip completed  │
                              │   groups          │
                              └─────────┬─────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │ Continue from     │
                              │ where stopped     │
                              └───────────────────┘
```

---

## API Endpoints

### POST /api/hyperopt/start
Запуск новой оптимизации.

**Request:**
```json
{
  "mode": "full",           // full, quick, group, fine_tune
  "target_group": null,     // для mode=group
  "epochs_per_trial": 4,
  "max_trials_per_group": 15,
  "max_total_hours": 24.0,
  "primary_metric": "val_f1",
  "study_name": "ml_hyperopt",
  "use_mlflow": true,
  "seed": 42
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "20231217_215832",
  "status": "starting",
  "time_estimate": {
    "estimated_hours": 8.5,
    "total_trials": 90
  }
}
```

### POST /api/hyperopt/stop
Остановка текущей оптимизации (graceful).

**Response:**
```json
{
  "success": true,
  "status": "paused",
  "message": "Остановка после текущего trial",
  "can_resume": true
}
```

### POST /api/hyperopt/resume
Продолжение остановленной оптимизации.

**Response:**
```json
{
  "success": true,
  "status": "resumed",
  "job_id": "20231217_215832",
  "progress": {
    "current_group": "regularization",
    "current_trial": 8,
    "groups_completed": ["learning_rate"]
  }
}
```

### GET /api/hyperopt/status
Текущий статус оптимизации.

### GET /api/hyperopt/results
Результаты оптимизации.

---

## Группы параметров

Оптимизация проходит **последовательно** по группам:

| # | Группа | Параметры | Описание |
|---|--------|-----------|----------|
| 1 | learning_rate | learning_rate, batch_size | Скорость обучения |
| 2 | regularization | weight_decay, dropout, label_smoothing, focal_gamma | Регуляризация |
| 3 | class_balance | use_focal_loss, use_class_weights, use_oversampling | Балансировка классов |
| 4 | augmentation | use_augmentation, gaussian_noise_std, mixup_alpha | Аугментация данных |
| 5 | scheduler | scheduler_T_0, scheduler_T_mult | LR scheduler |
| 6 | triple_barrier | threshold, take_profit, stop_loss | Triple Barrier метки |

---

## Сохраняемые файлы

```
data/hyperopt/
├── state.json              # Статус, job_id, config, progress
├── results.json            # best_params, group_results, fixed_params
├── best_params.json        # Только лучшие параметры
├── ml_hyperopt_learning_rate.db      # Optuna SQLite
├── ml_hyperopt_regularization.db
├── ml_hyperopt_class_balance.db
├── ml_hyperopt_augmentation.db
├── ml_hyperopt_scheduler.db
└── ml_hyperopt_triple_barrier.db
```

---

## Защита от Mode Collapse

### Проблема
При несбалансированных данных (53% HOLD vs 23% SELL/BUY) модель может "схлопнуться" к предсказанию только majority класса.

### Решение
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODE COLLAPSE PREVENTION                              │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌───────────────────────┐
                    │   Training Config     │
                    ├───────────────────────┤
                    │ use_class_weights=True│ ← Веса классов в loss
                    │ use_focal_loss=True   │ ← Фокус на hard examples
                    │ focal_gamma=2.0       │ ← Умеренный gamma
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   During Training     │
                    ├───────────────────────┤
                    │ Check prediction      │
                    │ distribution per epoch│
                    │                       │
                    │ If >90% same class:   │
                    │   ⚠️ MODE COLLAPSE    │
                    │   WARNING             │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   After Trial         │
                    ├───────────────────────┤
                    │ If precision < 0.35:  │
                    │   Mode collapse!      │
                    │   Penalize metrics    │
                    │   (F1 * 0.5)          │
                    └───────────────────────┘
```

---

## Механизм остановки

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GRACEFUL STOP MECHANISM                               │
└─────────────────────────────────────────────────────────────────────────┘

  User clicks STOP          threading.Event          Optuna Callback
        │                         │                        │
        │   stop_event.set()      │                        │
        ├────────────────────────▶│                        │
        │                         │                        │
        │                         │    (after trial)       │
        │                         │◀───────────────────────┤
        │                         │    is_set()?           │
        │                         │                        │
        │                         │    YES                 │
        │                         ├───────────────────────▶│
        │                         │                        │
        │                         │                study.stop()
        │                         │                        │
        │                         │    Graceful exit       │
        │◀─────────────────────────────────────────────────┤
        │                         │                        │
   Status = paused                │                        │
   Results saved                  │                        │
```

**Важно:** Остановка происходит ПОСЛЕ завершения текущего trial (может занять несколько минут).

---

## Механизм Resume

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RESUME MECHANISM                                      │
└─────────────────────────────────────────────────────────────────────────┘

  POST /resume
       │
       ▼
  ┌─────────────────────┐
  │ 1. stop_event.clear()│ ← Очистить флаг остановки
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │ 2. Load state.json  │ ← Загрузить сохраненное состояние
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │ 3. is_resume=True   │ ← Передать флаг в optimizer
  │    mode=RESUME      │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │ 4. Load results.json│ ← Загрузить предыдущие результаты
  │    • fixed_params   │
  │    • best_params    │
  │    • best_value     │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │ 5. Check completed  │ ← Определить завершённые группы
  │    groups           │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │ 6. Skip completed   │ ← Пропустить уже оптимизированные
  │    Continue from    │
  │    interrupted group│
  └─────────────────────┘

  Optuna: load_if_exists=True
  ├── Загружает существующие trials из SQLite
  └── Продолжает с последнего trial
```

---

## Конфигурация

### OptimizationConfig

```python
@dataclass
class OptimizationConfig:
    study_name: str = "ml_hyperopt"
    storage_path: str = "data/hyperopt"
    epochs_per_trial: int = 4
    max_trials_per_group: int = 15
    max_total_time_hours: float = 72.0
    n_startup_trials: int = 3          # Random trials before TPE
    n_warmup_steps: int = 2            # Pruner warmup
    pruning_percentile: float = 50.0   # Prune bottom 50%
    optimization_direction: str = "maximize"
    primary_metric: str = "val_f1"
    convergence_patience: int = 5
    convergence_threshold: float = 0.005
    use_mlflow: bool = True
    verbose: bool = True
    seed: int = 42
```

### TrainerConfigV2 defaults (исправлены)

```python
use_class_weights: bool = True    # CRITICAL: Prevents mode collapse
use_focal_loss: bool = True       # CRITICAL: Focuses on hard examples
focal_gamma: float = 2.0          # Reduced when using class_weights
label_smoothing: float = 0.1
use_augmentation: bool = True
```

---

## Пример использования

### CLI
```bash
# Полная оптимизация
python -m backend.ml_engine.hyperparameter_optimizer --mode full

# Быстрая (только learning_rate)
python -m backend.ml_engine.hyperparameter_optimizer --mode quick

# Одна группа
python -m backend.ml_engine.hyperparameter_optimizer --mode group --target regularization
```

### API
```python
import requests

# Запуск
response = requests.post("http://localhost:8000/api/hyperopt/start", json={
    "mode": "full",
    "epochs_per_trial": 4,
    "max_trials_per_group": 15
})

# Статус
status = requests.get("http://localhost:8000/api/hyperopt/status").json()

# Остановка
requests.post("http://localhost:8000/api/hyperopt/stop")

# Возобновление
requests.post("http://localhost:8000/api/hyperopt/resume")

# Результаты
results = requests.get("http://localhost:8000/api/hyperopt/results").json()
print(results["best_params"])
```

---

## Troubleshooting

### Mode Collapse (модель предсказывает только HOLD)
- **Причина:** `use_class_weights=False` или `use_focal_loss=False`
- **Решение:** Оба параметра должны быть `True`

### Кнопка "Стоп" не работает
- **Причина:** Ранее использовался `asyncio.Task.cancel()` без `threading.Event`
- **Решение:** Исправлено - используется `stop_event` проверяемый в Optuna callback

### Resume не восстанавливает прогресс
- **Причина:** Ранее не использовался RESUME mode
- **Решение:** Исправлено - `/resume` передаёт `is_resume=True` и загружает `results.json`

---

## Версия

**Последнее обновление:** 2024-12-18
**Коммиты:**
- `bef9a0b` - fix: Prevent mode collapse (class_weights + focal_loss)
- `40338e3` - fix: Add graceful stop mechanism (threading.Event)
- `c6c78a0` - fix: Implement proper resume functionality
