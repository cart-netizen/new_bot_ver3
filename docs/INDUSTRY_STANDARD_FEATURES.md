# Industry Standard ML Features

Документация по новым функциям, реализованным согласно industry standard практикам для ML в трейдинге.

## Содержание

1. [Purging & Embargo](#1-purging--embargo)
2. [Triple Barrier Labeling](#2-triple-barrier-labeling)
3. [Drift-Triggered Retraining](#3-drift-triggered-retraining)
4. [Rolling Normalization](#4-rolling-normalization)
5. [CPCV & PBO](#5-cpcv--pbo)

---

## 1. Purging & Embargo

**Файл**: `backend/ml_engine/training/data_loader.py`

### Проблема
При разбиении временных рядов на train/val/test возникает data leakage из-за:
- Overlapping sequences: labels в конце train могут пересекаться с началом val
- Автокорреляция: соседние samples не независимы

### Решение
- **Purging**: Удаляет `purge_length` samples из конца train (обычно = sequence_length)
- **Embargo**: Добавляет gap `embargo_length` samples между sets (обычно = label_horizon или 2% от dataset)

### Схема
```
[TRAIN] --purge-- [EMBARGO GAP] -- [VAL] --purge-- [EMBARGO GAP] -- [TEST]
```

### Использование
```python
from backend.ml_engine.training.data_loader import DataConfig, HistoricalDataLoader

config = DataConfig(
    sequence_length=60,
    label_horizon=60,

    # Purging & Embargo
    use_purging=True,      # Включить purging
    use_embargo=True,      # Включить embargo
    purge_length=None,     # None = sequence_length (60)
    embargo_length=None,   # None = max(label_horizon, 2% от dataset)
    embargo_pct=0.02       # 2% от dataset для embargo
)

loader = HistoricalDataLoader(config)
train_data, val_data, test_data = loader.train_val_test_split(
    sequences, labels, timestamps
)
```

### Параметры конфигурации

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `use_purging` | bool | True | Включить purging |
| `use_embargo` | bool | True | Включить embargo |
| `purge_length` | int | None | Длина purging (None = sequence_length) |
| `embargo_length` | int | None | Длина embargo (None = max(label_horizon, embargo_pct * n)) |
| `embargo_pct` | float | 0.02 | Процент от dataset для embargo |

### Ожидаемый эффект
- Устранение data leakage
- Более реалистичная оценка модели
- +2-5% к "честной" accuracy

---

## 2. Triple Barrier Labeling

**Файл**: `backend/ml_engine/features/labeling.py`

### Проблема
Fixed horizon labeling (`future_direction_60s`) не учитывает волатильность:
- В периоды высокой волатильности: много ложных сигналов
- В периоды низкой волатильности: пропуск возможностей

### Решение
Triple Barrier Method (López de Prado, 2018) использует три барьера:
1. **Take Profit**: цена достигла profit target (ATR * tp_multiplier)
2. **Stop Loss**: цена достигла stop loss (ATR * sl_multiplier)
3. **Timeout**: истекло max_holding_period

### Labels
- **2 (BUY)**: Верхний барьер достигнут первым → long был бы профитен
- **0 (SELL)**: Нижний барьер достигнут первым → short был бы профитен
- **1 (HOLD)**: Timeout → нет чёткого направления

### Использование
```python
from backend.ml_engine.features.labeling import (
    TripleBarrierLabeler,
    TripleBarrierConfig
)

config = TripleBarrierConfig(
    tp_multiplier=1.5,     # Take profit = entry + 1.5 * ATR
    sl_multiplier=1.0,     # Stop loss = entry - 1.0 * ATR
    max_holding_period=24, # Максимум 24 бара
    atr_period=14          # Период ATR
)

labeler = TripleBarrierLabeler(config)
result = labeler.generate_labels(df, price_col='close')

# Результаты
labels = result.labels          # np.array с метками 0, 1, 2
returns = result.returns        # Доходности при достижении барьера
hit_times = result.hit_times    # Время до достижения барьера
statistics = result.statistics  # Статистика разметки
```

### Перепроцессинг существующих данных
```bash
# Triple Barrier (рекомендуется)
python -m backend.ml_engine.scripts.preprocess_labels \
    --storage-path data/feature_store \
    --method triple_barrier \
    --tp-mult 1.5 \
    --sl-mult 1.0 \
    --max-hold 24 \
    --overwrite

# Legacy Fixed Threshold
python -m backend.ml_engine.scripts.preprocess_labels \
    --storage-path data/feature_store \
    --method fixed_threshold \
    --threshold 0.0001 \
    --overwrite
```

### Ожидаемый эффект
- +3-7% к качеству labels
- Адаптация к волатильности
- Более реалистичное моделирование торговых решений

---

## 3. Drift-Triggered Retraining

**Файл**: `backend/ml_engine/auto_retraining/retraining_pipeline.py`

### Проблема
Модель деградирует со временем из-за:
- Data drift: изменение распределения features
- Concept drift: изменение зависимости target от features
- Performance drift: падение accuracy

### Решение
Интеграция DriftDetector с RetrainingPipeline для автоматического retraining.

### Использование
```python
from backend.ml_engine.auto_retraining.retraining_pipeline import (
    RetrainingPipeline,
    RetrainingConfig
)

config = RetrainingConfig(
    # Drift detection
    enable_drift_trigger=True,
    drift_threshold=0.15,
    drift_check_interval_minutes=60,

    # Performance monitoring
    enable_performance_trigger=True,
    performance_threshold=0.75,

    # Auto-promotion
    auto_promote_to_production=True,
    min_accuracy_for_promotion=0.80
)

pipeline = RetrainingPipeline(config)

# Запуск
await pipeline.start()

# Добавление samples для drift monitoring (в production)
await pipeline.add_prediction_sample(
    features=feature_vector,
    prediction=model_prediction,
    label=actual_label  # если известен
)
```

### Triggers для Retraining

| Trigger | Условие | Описание |
|---------|---------|----------|
| SCHEDULED | По расписанию | Ежедневно в 03:00 |
| DRIFT_DETECTED | severity in ['high', 'critical'] | Data/Concept drift |
| PERFORMANCE_DROP | accuracy < 0.75 | Падение accuracy |
| MANUAL | Ручной запуск | По запросу |

### Ожидаемый эффект
- Автоматическое реагирование на drift
- Стабильность production performance
- Меньше ручного вмешательства

---

## 4. Rolling Normalization

**Файл**: `backend/ml_engine/features/feature_scaler_manager.py`

### Проблема
Статические scalers (fit на training data) не адаптируются к non-stationary market conditions.

### Решение
RollingScaler использует скользящее окно для расчёта статистик нормализации:
- **robust**: median и IQR (устойчив к выбросам)
- **standard**: mean и std

### Использование
```python
from backend.ml_engine.features.feature_scaler_manager import (
    FeatureScalerManager,
    ScalerConfig
)

config = ScalerConfig(
    # Включить Rolling Normalization
    use_rolling_normalization=True,
    rolling_window_size=500,      # Размер окна
    rolling_method="robust",       # robust или standard
    rolling_update_interval=10     # Обновлять каждые 10 samples
)

manager = FeatureScalerManager(symbol="BTCUSDT", config=config)

# Scaling с rolling normalization
scaled_vector = await manager.scale_features(feature_vector)
```

### Методы Rolling Normalization

| Метод | Центрирование | Масштабирование | Когда использовать |
|-------|---------------|-----------------|-------------------|
| `robust` | median | IQR | Волатильные рынки, выбросы |
| `standard` | mean | std | Стабильные рынки |

### Ожидаемый эффект
- Адаптация к изменяющимся market conditions
- Более стабильная нормализация
- Лучшая generalization

---

## 5. CPCV & PBO

**Файл**: `backend/ml_engine/validation/cpcv.py`

### Combinatorial Purged Cross-Validation (CPCV)

CPCV генерирует все возможные комбинации train/test splits с purging:
- Более robust оценка модели
- Все данные используются и в train и в test
- Возможность расчёта PBO

### Probability of Backtest Overfitting (PBO)

PBO измеряет вероятность что лучшая in-sample стратегия покажет плохой результат out-of-sample.

### Интерпретация PBO

| PBO | Риск | Рекомендация |
|-----|------|--------------|
| < 0.1 | Низкий | Модель robust |
| 0.1-0.3 | Умеренный | Дополнительная валидация |
| 0.3-0.5 | Высокий | Пересмотр модели |
| > 0.5 | Очень высокий | Вероятно overfit |

### Использование
```python
from backend.ml_engine.validation.cpcv import (
    CombinatorialPurgedCV,
    CPCVConfig,
    ProbabilityOfBacktestOverfitting,
    run_cpcv_validation
)

# Конфигурация CPCV
config = CPCVConfig(
    n_splits=6,           # 6 групп
    n_test_splits=2,      # 2 группы в test
    purge_length=60,      # Purging
    embargo_length=30     # Embargo
)

# Создание CPCV
cpcv = CombinatorialPurgedCV(config)

# Генерация splits
for train_idx, test_idx in cpcv.split(X, y):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])

# Расчёт PBO
pbo_calc = ProbabilityOfBacktestOverfitting()
result = pbo_calc.calculate(is_sharpes, oos_sharpes)

print(f"PBO: {result.pbo:.2%}")
print(f"Is Overfit: {result.is_overfit}")
print(f"Interpretation: {result.interpretation}")
```

### Полный пример с run_cpcv_validation
```python
def create_model():
    return YourMLModel()

def score_model(model, X, y):
    # Рассчитать Sharpe ratio
    predictions = model.predict(X)
    returns = calculate_returns(predictions, y)
    return np.mean(returns) / np.std(returns) * np.sqrt(252)

is_scores, oos_scores, pbo_result = run_cpcv_validation(
    X, y,
    model_fn=create_model,
    score_fn=score_model,
    config=CPCVConfig(n_splits=6, n_test_splits=2),
    calculate_pbo=True
)

print(f"PBO: {pbo_result.pbo:.2%}")
```

---

## Резюме изменений

| Фича | Файл | Статус |
|------|------|--------|
| Purging & Embargo | `data_loader.py` | ✅ Реализовано |
| Triple Barrier | `labeling.py` (новый) | ✅ Реализовано |
| Drift-Triggered Retraining | `retraining_pipeline.py` | ✅ Реализовано |
| Rolling Normalization | `feature_scaler_manager.py` | ✅ Реализовано |
| CPCV & PBO | `cpcv.py` (новый) | ✅ Реализовано |

---

## Ожидаемый эффект

| Улучшение | Ожидаемый эффект | Влияние на данные |
|-----------|------------------|-------------------|
| Purging & Embargo | +2-5% реальная accuracy | Не требует |
| Triple Barrier | +3-7% качество labels | Перепроцессинг labels |
| Drift Detection | Стабильность в production | Не требует |
| Rolling Normalization | Адаптация к рынку | Не требует |
| CPCV & PBO | Robust оценка модели | Не требует |

**Общий ожидаемый прирост**: +7-15% при правильном применении.

---

## Миграция

### Шаг 1: Перепроцессинг labels (опционально)
```bash
python -m backend.ml_engine.scripts.preprocess_labels \
    --method triple_barrier \
    --overwrite
```

### Шаг 2: Обновление конфигурации training
```python
# В DataConfig
config = DataConfig(
    use_purging=True,
    use_embargo=True
)
```

### Шаг 3: Включение Rolling Normalization (опционально)
```python
# В ScalerConfig
config = ScalerConfig(
    use_rolling_normalization=True,
    rolling_window_size=500
)
```

### Шаг 4: Запуск retraining с CPCV validation (рекомендуется)
```python
# Использовать CPCV для валидации перед promotion в production
```
