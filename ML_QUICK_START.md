# ML Infrastructure - Quick Start

Максимально быстрый старт для обучения моделей.

## 🚀 Быстрый старт (5 минут)

### 1. Установить зависимости

```bash
pip install mlflow pandas pyarrow scikit-learn
```

### 2. Обучить модель

**Вариант A: Один клик (Windows)**
```bash
train_model.bat
```

**Вариант B: Командная строка**
```bash
python train_model.py
```

**Вариант C: С параметрами**
```bash
python train_model.py --epochs 100 --lr 0.0001
```

### 3. Проверить результаты

```bash
# ВАЖНО: MLflow использует PostgreSQL как backend store
# Убедитесь, что PostgreSQL запущен и настроен в .env:
# MLFLOW_TRACKING_URI=postgresql://trading_bot:robocop@localhost:5432/trading_bot

# Открыть MLflow UI
mlflow ui --backend-store-uri postgresql://trading_bot:robocop@localhost:5432/trading_bot

# Или если MLFLOW_TRACKING_URI установлен в environment:
mlflow ui

# Открыть в браузере: http://localhost:5000
```

### 4. Запустить ML Model Server

```bash
python run_ml_server.py
```

### 5. Проверить модели

```bash
curl http://localhost:8001/api/ml/models
```

---

## 📝 Что произойдет при обучении

1. ✅ Загрузка данных из Feature Store (или legacy loader)
2. ✅ Инициализация HybridCNNLSTM модели
3. ✅ Обучение с MLflow tracking (все метрики логируются)
4. ✅ Evaluation на test set
5. ✅ Сохранение модели
6. ✅ Регистрация в Model Registry
7. ✅ Экспорт в ONNX
8. ✅ **Автоматический promotion в production** (если accuracy >= 0.80)

---

## 🎯 Результат

После успешного обучения:

```
TRAINING COMPLETED
═══════════════════════════════════════════════════
✅ Success!
Version: 20250106_120000
Model Path: checkpoints/models/20250106_120000/hybrid_cnn_lstm.pt

Test Metrics:
  Accuracy:  0.8542
  Precision: 0.8345
  Recall:    0.8123
  F1 Score:  0.8232

ONNX Model: checkpoints/models/20250106_120000/model.onnx

🚀 Model promoted to PRODUCTION!
═══════════════════════════════════════════════════
```

---

## 🔧 Настройка параметров

### Через командную строку

```bash
python train_model.py \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.0001 \
  --no-onnx        # Пропустить ONNX export
  --no-promote     # Пропустить auto-promotion
```

### Через Python API

```python
import asyncio
from backend.ml_engine.training_orchestrator import TrainingOrchestrator
from backend.ml_engine.models.hybrid_cnn_lstm import ModelConfig
from backend.ml_engine.training.model_trainer import TrainerConfig

async def main():
    # Custom configs
    model_config = ModelConfig(
        lstm_hidden=512,
        lstm_layers=3,
        dropout=0.4
    )

    trainer_config = TrainerConfig(
        epochs=100,
        learning_rate=0.0001,
        early_stopping_patience=15
    )

    # Train
    orchestrator = TrainingOrchestrator(
        model_config=model_config,
        trainer_config=trainer_config
    )

    result = await orchestrator.train_model(
        export_onnx=True,
        auto_promote=True
    )

    print(f"Success: {result['success']}")
    print(f"Accuracy: {result['test_metrics']['accuracy']:.4f}")

asyncio.run(main())
```

### Через REST API

```bash
curl -X POST http://localhost:8000/api/ml-management/train \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001,
    "export_onnx": true,
    "auto_promote": true
  }'
```

---

## 🔄 Auto-Retraining

Включить автоматическое переобучение:

```python
from backend.ml_engine.auto_retraining import get_retraining_pipeline, RetrainingConfig

async def setup_auto_retraining():
    config = RetrainingConfig(
        enable_scheduled=True,
        retraining_interval_hours=24,
        retraining_time="03:00",  # Каждый день в 3:00 утра
        enable_drift_trigger=True,
        auto_promote_to_production=True
    )

    pipeline = get_retraining_pipeline(config)
    await pipeline.start()

    print("Auto-retraining pipeline started!")

# Запустить
asyncio.run(setup_auto_retraining())
```

Или через API:

```bash
curl -X POST http://localhost:8000/api/ml-management/retraining/start
```

---

## 📦 Управление моделями

### Список моделей

```bash
curl http://localhost:8000/api/ml-management/models
```

### Promote модель в production

```bash
curl -X POST http://localhost:8000/api/ml-management/models/hybrid_cnn_lstm/20250106_120000/promote?stage=production
```

### Скачать модель

```bash
# Model Registry путь
models/registry/hybrid_cnn_lstm/production/model.pt

# Или из MLflow
mlflow artifacts download --run-id <run_id>
```

---

## 🐛 Troubleshooting

### Ошибка: "No training data"

Убедитесь, что есть данные для обучения:

```bash
ls -la data/ml_training/
```

Если данных нет, запустите сбор данных:

```python
from backend.ml_engine.data_collection import MLDataCollector

collector = MLDataCollector()
await collector.start()
```

### Ошибка: "MLflow tracking URI not set"

Установите tracking URI в `.env`:

```bash
# .env файл
MLFLOW_TRACKING_URI=postgresql://trading_bot:robocop@localhost:5432/trading_bot
MLFLOW_ARTIFACT_LOCATION=./mlruns/artifacts
MLFLOW_EXPERIMENT_NAME=trading_bot_ml
```

Или в environment:
```bash
export MLFLOW_TRACKING_URI=postgresql://trading_bot:robocop@localhost:5432/trading_bot
```

**ВАЖНО**: Убедитесь, что PostgreSQL запущен и доступен.

### Ошибка: "Model not loading"

Проверьте Model Registry:

```bash
ls -la models/registry/
```

---

## 📚 Следующие шаги

1. **Experiment Tracking**: Откройте MLflow UI и исследуйте эксперименты
2. **Model Comparison**: Сравните разные версии моделей
3. **Production Deployment**: Продвиньте лучшую модель в production
4. **Auto-Retraining**: Включите автоматическое переобучение
5. **Monitoring**: Настройте мониторинг drift и performance

---

## 🎓 Полная документация

См. [`ML_INFRASTRUCTURE_GUIDE.md`](./ML_INFRASTRUCTURE_GUIDE.md) для полной документации.
