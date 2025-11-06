# ML Infrastructure - Complete Guide

Полное руководство по ML инфраструктуре проекта с MLflow, Feature Store и Auto-Retraining.

## 📋 Содержание

1. [Обзор](#обзор)
2. [Архитектура](#архитектура)
3. [Быстрый старт](#быстрый-старт)
4. [Компоненты](#компоненты)
5. [API Reference](#api-reference)
6. [Frontend UI](#frontend-ui)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Обзор

Реализованная ML инфраструктура включает:

### ✅ Что реализовано

1. **MLflow Integration**
   - Experiment Tracking (все метрики, параметры, графики)
   - Model Registry (версионирование моделей)
   - Artifact Storage (модели, конфиги, графики)
   - Run Comparison (сравнение экспериментов)

2. **Feature Store**
   - Online Store (real-time serving с кешированием)
   - Offline Store (training data с партиционированием)
   - Feature Metadata (версионирование фич)
   - Feature Consistency (train/serve parity)

3. **Auto-Retraining Pipeline**
   - Scheduled Retraining (по расписанию)
   - Drift-Triggered Retraining (при обнаружении drift)
   - Performance-Triggered (при падении метрик)
   - Walk-Forward Validation
   - Auto-Promotion к production

4. **Training Orchestrator**
   - One-command training
   - Automatic MLflow tracking
   - Automatic model registration
   - ONNX export
   - Auto-promotion логика

5. **REST API для Frontend**
   - `/api/ml-management/train` - Start training
   - `/api/ml-management/models` - List models
   - `/api/ml-management/models/{name}/{version}/promote` - Promote model
   - `/api/ml-management/retraining/start` - Start auto-retraining
   - И многое другое...

### 🎁 Преимущества

- **Полная автоматизация**: От сбора данных до production deployment
- **Experiment Tracking**: Все эксперименты записываются в MLflow
- **Reproducibility**: Полная воспроизводимость результатов
- **Model Versioning**: Версионирование моделей с Git-like workflow
- **Feature Consistency**: Одинаковые фичи в training и serving
- **Auto-Retraining**: Модели переобучаются автоматически
- **Simple UI**: Простой интерфейс - выбрать модель → нажать "Обучить"

---

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                         FRONTEND                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ML Management UI (React)                            │   │
│  │  - Train Model Button                                │   │
│  │  - Model List & Selection                            │   │
│  │  - Training Progress                                 │   │
│  │  - Model Promotion                                   │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │ REST API
┌────────────────────┴────────────────────────────────────────┐
│                      BACKEND API                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ML Management API (FastAPI)                         │   │
│  │  /api/ml-management/*                                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────┬──────────────┬──────────────┬────────────────────┘
          │              │              │
┌─────────┴──────┐ ┌────┴──────┐ ┌────┴────────────┐
│   Training     │ │  Feature  │ │  Auto-Retraining│
│  Orchestrator  │ │   Store   │ │    Pipeline     │
└─────────┬──────┘ └────┬──────┘ └────┬────────────┘
          │              │              │
     ┌────┴──────────────┴──────────────┴────┐
     │         MLflow Integration             │
     │  - Tracking Server                     │
     │  - Model Registry                      │
     │  - Artifact Store                      │
     └────────────────────────────────────────┘
```

### Поток данных

```
1. Feature Engineering → Feature Store (Offline)
2. Training → MLflow Tracking + Model Registry
3. Evaluation → Auto-Promotion Logic
4. Deployment → Model Server (Production)
5. Serving → Feature Store (Online) → Predictions
6. Monitoring → Auto-Retraining Trigger
```

---

## 🚀 Быстрый старт

### Метод 1: Командная строка (самый простой)

```bash
# 1. Обучить модель с дефолтными параметрами
python train_model.py

# 2. Или с кастомными параметрами
python train_model.py --epochs 100 --lr 0.0001

# 3. Или двойной клик на Windows
train_model.bat
```

### Метод 2: Python API

```python
import asyncio
from backend.ml_engine.training_orchestrator import TrainingOrchestrator

async def train():
    orchestrator = TrainingOrchestrator()

    result = await orchestrator.quick_train(
        epochs=50,
        batch_size=64,
        learning_rate=0.001
    )

    print(f"Training completed: {result['success']}")
    print(f"Accuracy: {result['test_metrics']['accuracy']:.4f}")

asyncio.run(train())
```

### Метод 3: Frontend UI (после реализации)

1. Открыть веб-интерфейс
2. Перейти в "ML Management"
3. Выбрать параметры обучения
4. Нажать "Start Training"
5. Наблюдать progress в реальном времени
6. Автоматический promotion в production

### Метод 4: REST API

```bash
# Start training
curl -X POST http://localhost:8000/api/ml-management/train \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001
  }'

# Check status
curl http://localhost:8000/api/ml-management/training/status

# List models
curl http://localhost:8000/api/ml-management/models

# Promote model
curl -X POST http://localhost:8000/api/ml-management/models/hybrid_cnn_lstm/20250106_120000/promote
```

---

## 📦 Компоненты

### 1. MLflow Integration

**Расположение**: `backend/ml_engine/mlflow_integration/`

**Функции**:
- Automatic experiment tracking
- Parameter & metric logging
- Model artifact storage
- Model registry с staging workflow

**Использование**:

```python
from backend.ml_engine.mlflow_integration import get_mlflow_tracker

tracker = get_mlflow_tracker()

# Start run
tracker.start_run(run_name="my_experiment")

# Log params
tracker.log_params({"learning_rate": 0.001, "epochs": 50})

# Log metrics
tracker.log_metrics({"accuracy": 0.85, "loss": 0.15}, step=10)

# Log model
model_uri = tracker.log_model(model, "my_model")

# Register model
version = tracker.register_model(model_uri, "my_model")

# Promote to production
tracker.transition_model_stage("my_model", version, "Production")

# End run
tracker.end_run()
```

**MLflow UI**:
```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns --port 5000

# Open browser
http://localhost:5000
```

### 2. Feature Store

**Расположение**: `backend/ml_engine/feature_store/`

**Архитектура**:
- **Offline Store**: Parquet files (для training)
- **Online Store**: In-memory cache + disk (для serving)
- **Metadata Store**: JSON files (feature definitions)

**Использование**:

```python
from backend.ml_engine.feature_store import get_feature_store, FeatureMetadata

store = get_feature_store()

# Register feature
metadata = FeatureMetadata(
    name="rsi",
    version="1.0",
    description="RSI indicator",
    feature_type="technical",
    data_type="float",
    source="indicator_calculator",
    dependencies=[]
)
store.register_feature(metadata)

# Write offline features (for training)
features_df = pd.DataFrame(...)  # Your features
store.write_offline_features("orderbook_features", features_df)

# Read offline features
features = store.read_offline_features(
    feature_group="orderbook_features",
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Write online features (for serving)
feature_vector = np.array([...])
store.write_online_features("BTCUSDT", feature_vector)

# Read online features
features = store.read_online_features("BTCUSDT")
```

### 3. Auto-Retraining Pipeline

**Расположение**: `backend/ml_engine/auto_retraining/`

**Triggers**:
1. **Scheduled**: По расписанию (например, каждый день в 3:00)
2. **Drift-Detected**: При обнаружении data drift
3. **Performance-Drop**: При падении accuracy ниже порога
4. **Manual**: Ручной запуск

**Использование**:

```python
from backend.ml_engine.auto_retraining import get_retraining_pipeline, RetrainingConfig

# Create config
config = RetrainingConfig(
    enable_scheduled=True,
    retraining_interval_hours=24,
    enable_drift_trigger=True,
    drift_threshold=0.15,
    enable_performance_trigger=True,
    performance_threshold=0.75,
    auto_promote_to_production=True
)

# Get pipeline
pipeline = get_retraining_pipeline(config)

# Start pipeline (runs in background)
await pipeline.start()

# Manual trigger
result = await pipeline.trigger_retraining(
    trigger=RetrainingTrigger.MANUAL
)

# Stop pipeline
await pipeline.stop()
```

### 4. Training Orchestrator

**Расположение**: `backend/ml_engine/training_orchestrator.py`

**Workflow**:
1. Load data from Feature Store
2. Initialize model & trainer
3. Train with MLflow tracking
4. Evaluate on test set
5. Save model & register
6. Export to ONNX
7. Auto-promote to production

**Использование**:

```python
from backend.ml_engine.training_orchestrator import get_training_orchestrator
from backend.ml_engine.models.hybrid_cnn_lstm import ModelConfig
from backend.ml_engine.training.model_trainer import TrainerConfig

# Create orchestrator
orchestrator = get_training_orchestrator(
    model_config=ModelConfig(),
    trainer_config=TrainerConfig(epochs=50)
)

# Train model
result = await orchestrator.train_model(
    model_name="hybrid_cnn_lstm",
    export_onnx=True,
    auto_promote=True,
    min_accuracy_for_promotion=0.80
)

# Or quick train
result = await orchestrator.quick_train(
    epochs=50,
    batch_size=64,
    learning_rate=0.001
)
```

---

## 🔌 API Reference

### Training Endpoints

#### POST `/api/ml-management/train`

Start model training.

**Request Body**:
```json
{
  "model_name": "hybrid_cnn_lstm",
  "epochs": 50,
  "batch_size": 64,
  "learning_rate": 0.001,
  "export_onnx": true,
  "auto_promote": true,
  "min_accuracy": 0.80
}
```

**Response**:
```json
{
  "job_id": "20250106_120000",
  "status": "started",
  "message": "Training started in background",
  "started_at": "2025-01-06T12:00:00"
}
```

#### GET `/api/ml-management/training/status`

Get training status.

**Response**:
```json
{
  "is_training": true,
  "current_job": {
    "job_id": "20250106_120000",
    "status": "running",
    "started_at": "2025-01-06T12:00:00",
    "progress": {
      "current_epoch": 25,
      "total_epochs": 50,
      "best_val_accuracy": 0.82
    }
  }
}
```

### Model Management Endpoints

#### GET `/api/ml-management/models`

List all models.

**Query Params**:
- `stage` (optional): Filter by stage ("production", "staging", "archived")

**Response**:
```json
{
  "models": [
    {
      "name": "hybrid_cnn_lstm",
      "version": "20250106_120000",
      "stage": "production",
      "created_at": "2025-01-06T12:00:00",
      "metrics": {
        "accuracy": 0.85,
        "precision": 0.83
      }
    }
  ],
  "total": 1
}
```

#### POST `/api/ml-management/models/{name}/{version}/promote`

Promote model to stage.

**Query Params**:
- `stage`: Target stage ("production", "staging", "archived")

**Response**:
```json
{
  "success": true,
  "model_name": "hybrid_cnn_lstm",
  "version": "20250106_120000",
  "new_stage": "production"
}
```

### Auto-Retraining Endpoints

#### POST `/api/ml-management/retraining/start`

Start auto-retraining pipeline.

**Request Body** (optional):
```json
{
  "enable_scheduled": true,
  "retraining_interval_hours": 24,
  "enable_drift_trigger": true,
  "drift_threshold": 0.15
}
```

#### POST `/api/ml-management/retraining/stop`

Stop auto-retraining pipeline.

#### GET `/api/ml-management/retraining/status`

Get pipeline status.

**Response**:
```json
{
  "is_running": true,
  "config": {...},
  "last_training_time": "2025-01-06T03:00:00",
  "last_drift_check_time": "2025-01-06T12:00:00"
}
```

#### POST `/api/ml-management/retraining/trigger`

Manually trigger retraining.

**Query Params**:
- `trigger`: "manual", "drift", "performance", "scheduled"

---

## 🖥️ Frontend UI

### Главная страница ML Management

```
┌─────────────────────────────────────────────────────────┐
│  ML Model Management                                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📊 Training Status                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ● Training in progress...                       │   │
│  │ Epoch: 25/50  |  Accuracy: 0.82                 │   │
│  │ [████████████░░░░░░░░░] 50%                     │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  🎯 Quick Train                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Epochs:         [50     ▼]                      │   │
│  │ Batch Size:     [64     ▼]                      │   │
│  │ Learning Rate:  [0.001  ▼]                      │   │
│  │                                                  │   │
│  │ [ ] Export to ONNX                              │   │
│  │ [x] Auto-promote to production                  │   │
│  │                                                  │   │
│  │           [🚀 Start Training]                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  📦 Models                                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Name             Version      Stage    Accuracy │   │
│  │─────────────────────────────────────────────────│   │
│  │ hybrid_cnn_lstm  20250106... Production  0.85   │   │
│  │   [Promote] [Download] [Delete]                 │   │
│  │                                                  │   │
│  │ hybrid_cnn_lstm  20250105... Staging     0.82   │   │
│  │   [Promote] [Download] [Delete]                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  🔄 Auto-Retraining                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Status: ● Running                               │   │
│  │ Last Training: 2025-01-06 03:00:00              │   │
│  │ Next Training: 2025-01-07 03:00:00              │   │
│  │                                                  │   │
│  │ [⏸ Stop] [▶ Start] [🔧 Configure]              │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Troubleshooting

### Проблема: MLflow tracking не работает

**Решение**:
```bash
# Проверить, что MLflow tracking URI настроен
export MLFLOW_TRACKING_URI=file:./mlruns

# Или в коде
mlflow.set_tracking_uri("file:./mlruns")
```

### Проблема: Feature Store возвращает пустые данные

**Решение**:
```python
# Проверить, что данные записаны
store = get_feature_store()
features = store.read_offline_features("orderbook_features")
print(f"Found {len(features)} rows")

# Проверить партиции
import os
print(os.listdir("data/feature_store/offline/orderbook_features"))
```

### Проблема: Auto-retraining не запускается

**Решение**:
```python
# Проверить статус
pipeline = get_retraining_pipeline()
print(f"Is running: {pipeline.is_running}")

# Запустить вручную
await pipeline.start()

# Проверить логи
tail -f logs/retraining/*.log
```

---

## 📚 Дополнительные ресурсы

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Feature Store Best Practices](https://www.featurestore.org/)
- [Model Versioning Guide](./docs/model_versioning.md)
- [Auto-Retraining Strategies](./docs/auto_retraining.md)

---

## 🎓 Примеры использования

См. полные примеры в:
- `examples/train_model_example.py`
- `examples/feature_store_example.py`
- `examples/auto_retraining_example.py`
- `examples/api_usage_example.py`
