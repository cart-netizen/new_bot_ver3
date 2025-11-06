# ML Model Serving Infrastructure

Complete production-ready ML infrastructure для trading bot.

## 📋 Оглавление

1. [Обзор](#обзор)
2. [Компоненты](#компоненты)
3. [Quick Start](#quick-start)
4. [Model Registry](#model-registry)
5. [Model Server](#model-server)
6. [A/B Testing](#ab-testing)
7. [ONNX Optimization](#onnx-optimization)
8. [Integration](#integration)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Обзор

ML Model Serving Infrastructure предоставляет полный lifecycle management для ML моделей:

- **Model Registry**: Версионирование и управление моделями
- **Model Server**: FastAPI сервер для inference (port 8001)
- **A/B Testing**: Testing новых моделей в production
- **ONNX Optimization**: Экспорт и оптимизация для latency < 3ms

### Архитектура

```
┌──────────────────────────────────────────────────────────┐
│                    Trading Bot (Main)                     │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │          Model Client (HTTP Client)                  │ │
│  └────────────────────┬─────────────────────────────────┘ │
└───────────────────────┼───────────────────────────────────┘
                        │ HTTP (port 8001)
                        ▼
┌──────────────────────────────────────────────────────────┐
│               Model Server (FastAPI)                      │
├──────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌───────────┐  ┌───────────────────┐   │
│  │   Model    │  │   A/B     │  │  ONNX Sessions    │   │
│  │  Registry  │  │  Testing  │  │  (Optimized)      │   │
│  └────────────┘  └───────────┘  └───────────────────┘   │
├──────────────────────────────────────────────────────────┤
│            Loaded Models (PyTorch/ONNX)                   │
└──────────────────────────────────────────────────────────┘
                        ▼
                  File System
            models/
            ├── hybrid_cnn_lstm/
            │   ├── v1.0.0/
            │   │   ├── model.pt
            │   │   ├── model.onnx
            │   │   └── metadata.json
            │   ├── v1.1.0/
            │   └── production -> v1.0.0
```

---

## 🧩 Компоненты

### 1. Model Registry
**Файл**: `backend/ml_engine/inference/model_registry.py`

Управление lifecycle моделей:
- Регистрация новых версий
- Staging → Production promotion
- Метрики и метаданные
- Symlink management для stages

### 2. Model Server
**Файл**: `backend/ml_engine/inference/model_server_v2.py`

FastAPI сервер для inference:
- Single и batch predictions
- Hot reload моделей
- A/B testing support
- Health monitoring

### 3. A/B Testing Manager
**Файл**: `backend/ml_engine/inference/ab_testing.py`

Testing новых моделей:
- Traffic splitting (90/10)
- Statistical significance testing
- Automatic promotion/rollback
- Comprehensive metrics

### 4. ONNX Optimizer
**Файл**: `backend/ml_engine/optimization/onnx_optimizer.py`

Оптимизация моделей:
- PyTorch → ONNX export
- INT8 quantization
- Benchmarking
- Latency optimization

### 5. Model Client
**Файл**: `backend/ml_engine/inference/model_client.py`

HTTP клиент для интеграции:
- Async predictions
- Batch support
- Health checks
- A/B test management

---

## 🚀 Quick Start

### 1. Установка зависимостей

```bash
pip install onnx onnxruntime fastapi uvicorn scipy
```

### 2. Запуск Model Server

```bash
# В отдельном терминале
cd backend
python -m backend.ml_engine.inference.model_server_v2

# Или через uvicorn
uvicorn backend.ml_engine.inference.model_server_v2:app --host 0.0.0.0 --port 8001
```

Server запустится на `http://localhost:8001`

### 3. Регистрация первой модели

```python
import asyncio
from pathlib import Path
from backend.ml_engine.inference.model_registry import get_model_registry

async def register_model():
    registry = get_model_registry()

    # Предполагаем, что у вас есть trained модель
    model_info = await registry.register_model(
        name="hybrid_cnn_lstm",
        version="1.0.0",
        model_path=Path("path/to/model.pt"),
        model_type="HybridCNNLSTM",
        description="Initial production model",
        metrics={
            "accuracy": 0.85,
            "sharpe_ratio": 2.5,
            "latency_ms": 5.0
        },
        training_params={
            "input_size": 110,
            "lstm_hidden_size": 256,
            "lstm_layers": 2,
            "dropout": 0.3
        }
    )

    # Promote to production
    await registry.promote_to_production("hybrid_cnn_lstm", "1.0.0")
    print(f"Model registered: {model_info.metadata.name} v{model_info.metadata.version}")

asyncio.run(register_model())
```

### 4. Использование в боте

```python
from backend.ml_engine.inference.model_client import get_model_client
import numpy as np

# Инициализация клиента
client = get_model_client("http://localhost:8001")
await client.initialize()

# Prediction
features = np.random.randn(60, 110)  # 60 timesteps, 110 features
prediction = await client.predict(
    symbol="BTCUSDT",
    features=features
)

print(f"Prediction: {prediction['prediction']}")
print(f"Latency: {prediction['latency_ms']:.2f}ms")
```

---

## 📦 Model Registry

### Структура хранения

```
models/
├── hybrid_cnn_lstm/
│   ├── v1.0.0/
│   │   ├── model.pt           # PyTorch веса
│   │   ├── model.onnx         # ONNX (optional)
│   │   ├── metadata.json      # Метаданные
│   │   └── metrics.json       # Метрики (deprecated, используйте metadata)
│   ├── v1.1.0/
│   ├── v2.0.0/
│   ├── production -> v1.0.0   # Symlink
│   └── staging -> v1.1.0      # Symlink
```

### API

#### Регистрация модели

```python
from backend.ml_engine.inference.model_registry import get_model_registry

registry = get_model_registry()

model_info = await registry.register_model(
    name="hybrid_cnn_lstm",
    version="1.1.0",
    model_path=Path("models/trained_model.pt"),
    model_type="HybridCNNLSTM",
    description="Improved model with better features",
    metrics={
        "accuracy": 0.87,
        "precision": 0.85,
        "recall": 0.89,
        "sharpe_ratio": 2.8
    },
    training_params={
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.001
    },
    tags=["improved", "production-candidate"]
)
```

#### Получение модели

```python
# По версии
model_info = await registry.get_model("hybrid_cnn_lstm", "1.1.0")

# Production версия
model_info = await registry.get_production_model("hybrid_cnn_lstm")

# Staging версия
model_info = await registry.get_staging_model("hybrid_cnn_lstm")

# Latest версия (если production не установлен)
model_info = await registry.get_model("hybrid_cnn_lstm")
```

#### Управление stages

```python
# Установить staging
await registry.set_model_stage("hybrid_cnn_lstm", "1.1.0", ModelStage.STAGING)

# Promote to production
await registry.promote_to_production("hybrid_cnn_lstm", "1.1.0")

# Retire (archive)
await registry.retire_model("hybrid_cnn_lstm", "1.0.0")
```

#### Список моделей

```python
# Все версии конкретной модели
models = await registry.list_models("hybrid_cnn_lstm")

# Все модели
all_models = await registry.list_models()

for model in models:
    print(f"{model.metadata.name} v{model.metadata.version} - {model.metadata.stage}")
```

#### Сравнение моделей

```python
comparison = await registry.compare_models(
    name="hybrid_cnn_lstm",
    version1="1.0.0",
    version2="1.1.0"
)

print(f"Accuracy improvement: {comparison['metrics_comparison']['accuracy']['diff_pct']:.2f}%")
print(f"Size difference: {comparison['size_comparison']['diff_mb']:.2f} MB")
```

---

## 🖥️ Model Server

### Endpoints

#### POST /api/ml/predict
Single prediction

**Request**:
```json
{
  "symbol": "BTCUSDT",
  "features": [0.1, 0.2, ..., 0.5],  // Flattened feature vector
  "model_name": "hybrid_cnn_lstm",    // Optional
  "model_version": "1.0.0"            // Optional
}
```

**Response**:
```json
{
  "symbol": "BTCUSDT",
  "prediction": {
    "direction": 0,      // 0=HOLD, 1=BUY, 2=SELL
    "confidence": 0.85,
    "expected_return": 0.025
  },
  "model_name": "hybrid_cnn_lstm",
  "model_version": "1.0.0",
  "variant": null,
  "latency_ms": 3.5,
  "timestamp": "2025-11-06T12:00:00"
}
```

#### POST /api/ml/predict/batch
Batch predictions

**Request**:
```json
{
  "requests": [
    {"symbol": "BTCUSDT", "features": [...]},
    {"symbol": "ETHUSDT", "features": [...]}
  ],
  "max_batch_size": 32
}
```

#### GET /api/ml/models
Список загруженных моделей

**Response**:
```json
[
  {
    "name": "hybrid_cnn_lstm",
    "version": "1.0.0",
    "stage": "Production",
    "model_type": "HybridCNNLSTM",
    "metrics": {"accuracy": 0.85, "sharpe": 2.5},
    "size_mb": 12.5,
    "loaded": true
  }
]
```

#### POST /api/ml/models/reload
Hot reload модели

**Request**:
```json
{
  "model_name": "hybrid_cnn_lstm",
  "version": "1.1.0"  // Optional, default = production
}
```

#### GET /api/ml/health
Health check

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-06T12:00:00",
  "loaded_models": ["hybrid_cnn_lstm:1.0.0"],
  "active_experiments": ["v1_vs_v2"],
  "uptime_seconds": 3600.5
}
```

### curl Examples

```bash
# Health check
curl http://localhost:8001/api/ml/health

# Prediction
curl -X POST http://localhost:8001/api/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "features": [0.1, 0.2, 0.3, ..., 0.5]
  }'

# List models
curl http://localhost:8001/api/ml/models

# Reload model
curl -X POST http://localhost:8001/api/ml/models/reload \
  -H "Content-Type: application/json" \
  -d '{"model_name": "hybrid_cnn_lstm", "version": "1.1.0"}'
```

---

## 🧪 A/B Testing

### Workflow

1. **Create Experiment**: Определить control и treatment модели
2. **Traffic Splitting**: Автоматически route 90% → control, 10% → treatment
3. **Collect Metrics**: Record predictions и outcomes
4. **Analyze**: Statistical significance testing
5. **Decision**: Promote treatment или rollback

### API

#### Создание A/B теста

```python
from backend.ml_engine.inference.model_client import get_model_client

client = get_model_client()
await client.initialize()

# Create experiment
success = await client.create_ab_test(
    experiment_id="v1_vs_v2",
    control_model="hybrid_cnn_lstm",
    control_version="1.0.0",
    treatment_model="hybrid_cnn_lstm",
    treatment_version="1.1.0",
    traffic_split=0.9  # 90% control, 10% treatment
)

print(f"Experiment created: {success}")
```

#### Predictions через эксперимент

```python
# Predictions автоматически routing
prediction = await client.predict(
    symbol="BTCUSDT",
    features=features,
    # experiment_id будет использоваться автоматически если active
)

# Вариант указан в response
print(f"Variant: {prediction.get('variant')}")  # control или treatment
```

#### Анализ эксперимента

```python
# Получить текущий анализ
analysis = await client.get_ab_test_analysis("v1_vs_v2")

print(f"Control accuracy: {analysis['control']['accuracy']:.2%}")
print(f"Treatment accuracy: {analysis['treatment']['accuracy']:.2%}")
print(f"Improvement: {analysis['improvement']['accuracy']:.2%}")
print(f"Recommendation: {analysis['recommendation']['action']}")
print(f"Reasons: {analysis['recommendation']['reasons']}")
```

#### Остановка эксперимента

```python
# Stop и получить final report
report = await client.stop_ab_test("v1_vs_v2")

print(f"Final recommendation: {report['recommendation']['action']}")
# Actions: "promote", "rollback", "continue"

if report['recommendation']['action'] == "promote":
    # Promote treatment to production
    registry = get_model_registry()
    await registry.promote_to_production("hybrid_cnn_lstm", "1.1.0")

    # Reload на server
    await client.reload_model("hybrid_cnn_lstm", "1.1.0")
```

### Метрики сравнения

A/B тест собирает и сравнивает:

**Performance Metrics**:
- Accuracy, Precision, Recall, F1
- Win rate, Average return, Sharpe ratio, Total P&L

**Technical Metrics**:
- Average latency, P95 latency
- Error rate
- Throughput

**Statistical Tests**:
- Two-sample t-test для accuracy
- Confidence level (default 95%)
- P-value для significance

### Критерии принятия решения

Автоматические recommendations based on:

1. **Promote treatment если**:
   - Accuracy improvement >= 2% (configurable)
   - Statistical significance (p < 0.05)
   - Latency degradation < 2ms
   - Error rate не увеличился

2. **Rollback если**:
   - Latency degradation > 2ms
   - Error rate увеличился > 50%
   - Accuracy degraded > 5%

3. **Continue если**:
   - Недостаточно samples
   - Improvement marginal
   - Not statistically significant

---

## ⚡ ONNX Optimization

### Export to ONNX

```python
from backend.ml_engine.optimization.onnx_optimizer import get_onnx_optimizer
from backend.ml_engine.models.hybrid_cnn_lstm import HybridCNNLSTM
from pathlib import Path

optimizer = get_onnx_optimizer()

# Load PyTorch model
model = HybridCNNLSTM(input_size=110, ...)
model_path = Path("models/hybrid_cnn_lstm/v1.0.0/model.pt")
onnx_path = Path("models/hybrid_cnn_lstm/v1.0.0/model.onnx")

# Export
success = await optimizer.export_to_onnx(
    model=model,
    model_path=model_path,
    output_path=onnx_path,
    input_shape=(1, 60, 110),  # batch, timesteps, features
    opset_version=14
)

print(f"Export success: {success}")
```

### Quantization (FP32 → INT8)

```python
# Quantize для -75% memory, -40% latency
quantized_path = Path("models/hybrid_cnn_lstm/v1.0.0/model_quantized.onnx")

success = await optimizer.quantize_model(
    onnx_path=onnx_path,
    output_path=quantized_path,
    quantization_type="dynamic"
)

print(f"Quantization success: {success}")
```

### Benchmarking

```python
# Benchmark original ONNX
metrics = await optimizer.benchmark(
    onnx_path=onnx_path,
    input_shape=(1, 60, 110),
    num_iterations=1000,
    warmup_iterations=100
)

print(f"Average latency: {metrics['latency_ms']:.2f}ms")
print(f"P95 latency: {metrics['p95_ms']:.2f}ms")
print(f"Throughput: {metrics['throughput']:.0f} predictions/sec")

# Benchmark quantized
quant_metrics = await optimizer.benchmark(
    onnx_path=quantized_path,
    input_shape=(1, 60, 110),
    num_iterations=1000
)

speedup = metrics['latency_ms'] / quant_metrics['latency_ms']
print(f"Quantized speedup: {speedup:.2f}x")
```

### Full Optimization Pipeline

```python
# Complete: Export + Quantize + Benchmark
results = await optimizer.export_and_optimize(
    model=model,
    model_path=model_path,
    output_dir=Path("models/hybrid_cnn_lstm/v1.0.0"),
    input_shape=(1, 60, 110),
    quantize=True,
    benchmark_iterations=1000
)

print(f"Export: {results['export_success']}")
print(f"Quantize: {results['quantize_success']}")
print(f"ONNX FP32: {results['benchmarks']['onnx_fp32']['latency_ms']:.2f}ms")
print(f"ONNX INT8: {results['benchmarks']['onnx_int8']['latency_ms']:.2f}ms")
print(f"Speedup: {results['comparison']['speedup']:.2f}x")
```

---

## 🔗 Integration

### Интеграция с основным ботом

#### main.py updates

```python
from backend.ml_engine.inference.model_client import get_model_client

class TradingBot:
    def __init__(self):
        # ...existing code...

        # Initialize Model Client
        self.model_client = get_model_client(
            server_url=settings.ML_SERVER_URL
        )

    async def start(self):
        # ...existing code...

        # Initialize ML client
        await self.model_client.initialize()

        # Health check
        healthy = await self.model_client.health_check()
        if not healthy:
            logger.warning("ML Model Server is not healthy")

    async def stop(self):
        # ...existing code...

        # Cleanup ML client
        await self.model_client.cleanup()
```

#### Использование в analysis loop

```python
async def _analysis_loop(self):
    while self.running:
        for symbol in self.symbols:
            # ...extract features...

            # ML prediction
            ml_prediction = await self.model_client.predict(
                symbol=symbol,
                features=features_array
            )

            if ml_prediction:
                # Use prediction
                direction = ml_prediction['prediction']['direction']
                confidence = ml_prediction['prediction']['confidence']

                # Integrate with strategy consensus
                # ...
```

---

## 📚 API Reference

### Model Registry

```python
class ModelRegistry:
    async def register_model(name, version, model_path, model_type, ...) -> ModelInfo
    async def get_model(name, version=None, stage=None) -> ModelInfo
    async def list_models(name=None) -> List[ModelInfo]
    async def set_model_stage(name, version, stage) -> bool
    async def promote_to_production(name, version) -> bool
    async def retire_model(name, version) -> bool
    async def delete_model(name, version) -> bool
    async def update_metrics(name, version, metrics) -> bool
    async def compare_models(name, version1, version2) -> Dict
```

### Model Client

```python
class ModelClient:
    async def initialize()
    async def cleanup()
    async def predict(symbol, features, model_name=None, model_version=None) -> Dict
    async def batch_predict(requests, max_batch_size=32) -> List[Dict]
    async def health_check() -> bool
    async def list_models() -> List[Dict]
    async def reload_model(model_name, version=None) -> bool
    async def create_ab_test(experiment_id, control_model, ...) -> bool
    async def get_ab_test_analysis(experiment_id) -> Dict
    async def stop_ab_test(experiment_id) -> Dict
```

### ONNX Optimizer

```python
class ONNXOptimizer:
    async def export_to_onnx(model, model_path, output_path, input_shape, ...) -> bool
    async def quantize_model(onnx_path, output_path, quantization_type="dynamic") -> bool
    async def optimize_graph(onnx_path, output_path) -> bool
    async def benchmark(onnx_path, input_shape, num_iterations=1000, ...) -> Dict
    async def compare_pytorch_onnx(pytorch_model, onnx_path, ...) -> Dict
    async def export_and_optimize(model, model_path, output_dir, ...) -> Dict
```

---

## 🐛 Troubleshooting

### Model Server не запускается

**Problem**: `ModuleNotFoundError: No module named 'onnx'`

**Solution**:
```bash
pip install onnx onnxruntime
```

---

### ONNX export fails

**Problem**: `RuntimeError: ONNX export failed: Unsupported operator`

**Solution**:
- Проверьте opset_version (попробуйте 11, 13, 14)
- Убедитесь, что все операторы PyTorch поддерживаются в ONNX
- Используйте `verbose=True` для детального лога

---

### Predictions slow

**Problem**: Latency > 10ms

**Solutions**:
1. Use ONNX version: `use_onnx=True` при loading
2. Quantize model: INT8 quantization
3. Reduce model size: Prune или distill
4. Check batch size: Batch predictions для throughput

---

### A/B test not routing

**Problem**: Все predictions идут на control

**Solution**:
- Проверьте, что experiment создан
- Убедитесь, что experiment status = RUNNING
- Проверьте traffic split (должно быть control + treatment = 1.0)

---

### Model Registry symlinks not working (Windows)

**Problem**: Symlinks не работают на Windows

**Solution**:
- Enable Developer Mode в Windows
- Или используйте version напрямую вместо stage
- Или используйте Linux/Mac для development

---

## 🎓 Best Practices

### 1. Model Versioning

- Используйте semantic versioning: `major.minor.patch`
- `major`: Breaking changes (new architecture)
- `minor`: Improvements (better features, hyperparams)
- `patch`: Bug fixes (training improvements)

### 2. A/B Testing

- Всегда начинайте с 90/10 split
- Минимум 100 samples per variant
- Run минимум 24 часа
- Monitor latency AND accuracy

### 3. Production Deployment

- Всегда test в staging перед production
- Use A/B test для validation
- Keep previous version в production до успешного теста
- Monitor drift после deployment

### 4. Performance

- Export to ONNX для production
- Quantize если latency критична
- Benchmark перед deployment
- Monitor latency в production

---

## 📝 Changelog

### v2.0.0 (2025-11-06)
- ✅ Initial release
- ✅ Model Registry
- ✅ Model Server v2
- ✅ A/B Testing Infrastructure
- ✅ ONNX Optimization
- ✅ Complete integration tests

### Planned v2.1.0
- [ ] MLflow integration
- [ ] Auto-retraining pipeline
- [ ] Feature store
- [ ] Multi-GPU support

---

## 📞 Support

Для вопросов и issues:
- GitHub Issues: https://github.com/cart-netizen/new_bot_ver3/issues
- Documentation: См. этот файл
- Code Examples: `backend/tests/test_ml_serving.py`

---

**Status**: ✅ Production Ready
**Version**: 2.0.0
**Last Updated**: 2025-11-06
