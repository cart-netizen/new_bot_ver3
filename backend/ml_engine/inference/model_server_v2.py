"""
ML Model Server v2 - FastAPI сервер для ML inference

Отдельный процесс для обслуживания ML моделей:
- Port: 8001
- Endpoints: /predict, /reload, /health
- A/B Testing support
- ONNX optimization
- Model Registry integration
"""

import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.ml_engine.inference.model_registry import (
    get_model_registry,
    ModelRegistry,
    ModelStage
)
from backend.ml_engine.inference.ab_testing import (
    get_ab_test_manager,
    ABTestManager,
    ModelVariant,
    PredictionOutcome
)
from backend.ml_engine.models.hybrid_cnn_lstm import HybridCNNLSTM
from backend.core.logger import get_logger

logger = get_logger(__name__)


# === Request/Response Models ===

class PredictRequest(BaseModel):
    """Request для single prediction"""
    symbol: str
    features: List[float] = Field(..., min_length=1)
    # Опционально: можно указать конкретную модель
    model_name: Optional[str] = None
    model_version: Optional[str] = None


class PredictResponse(BaseModel):
    """Response для prediction"""
    symbol: str
    prediction: Dict[str, Any]  # direction, confidence, expected_return
    model_name: str
    model_version: str
    variant: Optional[str] = None  # control или treatment (если A/B test)
    latency_ms: float
    timestamp: datetime


class BatchPredictRequest(BaseModel):
    """Request для batch predictions"""
    requests: List[PredictRequest]
    max_batch_size: int = 32


class BatchPredictResponse(BaseModel):
    """Response для batch predictions"""
    predictions: List[PredictResponse]
    total_latency_ms: float


class ModelInfo(BaseModel):
    """Информация о модели"""
    name: str
    version: str
    stage: str
    model_type: str
    metrics: Dict[str, float]
    size_mb: float
    loaded: bool


class HealthResponse(BaseModel):
    """Health check response"""
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    loaded_models: List[str]
    active_experiments: List[str]
    uptime_seconds: float


class ReloadRequest(BaseModel):
    """Request для reload модели"""
    model_name: str
    version: Optional[str] = None  # Если None, загружается production


class ABTestRequest(BaseModel):
    """Request для создания A/B теста"""
    experiment_id: str
    control_model_name: str
    control_model_version: str
    treatment_model_name: str
    treatment_model_version: str
    control_traffic: float = 0.9
    treatment_traffic: float = 0.1
    duration_hours: int = 24


# === Model Server ===

class ModelServer:
    """
    ML Model Server

    Функции:
    - Загрузка моделей из Model Registry
    - Inference (single + batch)
    - A/B testing
    - Hot reload
    - ONNX support
    """

    def __init__(self):
        self.registry: ModelRegistry = get_model_registry()
        self.ab_manager: ABTestManager = get_ab_test_manager()

        # Loaded models cache
        self.loaded_models: Dict[str, torch.nn.Module] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}

        # ONNX sessions
        self.onnx_sessions: Dict[str, Any] = {}

        # Stats
        self.start_time = datetime.now()
        self.total_predictions = 0
        self.total_latency_ms = 0.0

        logger.info("Model Server initialized")

    async def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        use_onnx: bool = False
    ) -> bool:
        """
        Загрузить модель в память

        Args:
            model_name: Название модели
            version: Версия (если None, загружается production)
            use_onnx: Использовать ONNX версию

        Returns:
            True если успешно
        """
        try:
            # Получить модель из registry
            model_info = await self.registry.get_model(model_name, version)
            if not model_info:
                logger.error(f"Model {model_name} not found in registry")
                return False

            model_key = f"{model_name}:{model_info.metadata.version}"

            # ONNX version
            if use_onnx and model_info.onnx_exists():
                try:
                    import onnxruntime as ort
                    session = ort.InferenceSession(
                        str(model_info.onnx_path),
                        providers=['CPUExecutionProvider']
                    )
                    self.onnx_sessions[model_key] = session
                    logger.info(f"Loaded ONNX model: {model_key}")
                except Exception as e:
                    logger.error(f"Failed to load ONNX model: {e}")
                    use_onnx = False

            # PyTorch version (fallback или основная)
            if not use_onnx:
                # Создать архитектуру модели (зависит от model_type)
                model_type = model_info.metadata.model_type
                if model_type == "HybridCNNLSTM":
                    # Импортировать ModelConfig
                    from backend.ml_engine.models.hybrid_cnn_lstm import ModelConfig

                    # Загрузить параметры из metadata
                    training_params = model_info.metadata.training_params

                    # Создать config
                    config = ModelConfig(
                        input_features=training_params.get("input_features", 110),
                        sequence_length=training_params.get("sequence_length", 60),
                        cnn_channels=tuple(training_params.get("cnn_channels", [64, 128, 256])),
                        cnn_kernel_sizes=tuple(training_params.get("cnn_kernel_sizes", [3, 5, 7])),
                        lstm_hidden=training_params.get("lstm_hidden", 256),
                        lstm_layers=training_params.get("lstm_layers", 2),
                        lstm_dropout=training_params.get("lstm_dropout", 0.2),
                        attention_units=training_params.get("attention_units", 128),
                        num_classes=training_params.get("num_classes", 3),
                        dropout=training_params.get("dropout", 0.3)
                    )

                    # Создать модель с config
                    model = HybridCNNLSTM(config)

                    # Загрузить веса
                    checkpoint = torch.load(model_info.model_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)

                    model.eval()
                    self.loaded_models[model_key] = model
                    logger.info(f"Loaded PyTorch model: {model_key}")
                else:
                    logger.error(f"Unknown model type: {model_type}")
                    return False

            # Сохранить metadata
            self.model_metadata[model_key] = {
                "name": model_name,
                "version": model_info.metadata.version,
                "stage": model_info.metadata.stage.value,
                "model_type": model_info.metadata.model_type,
                "metrics": model_info.metadata.metrics,
                "size_mb": model_info.metadata.model_size_mb,
                "use_onnx": use_onnx,
                "loaded_at": datetime.now()
            }

            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    async def unload_model(self, model_name: str, version: str) -> bool:
        """Выгрузить модель из памяти"""
        model_key = f"{model_name}:{version}"

        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            logger.info(f"Unloaded PyTorch model: {model_key}")

        if model_key in self.onnx_sessions:
            del self.onnx_sessions[model_key]
            logger.info(f"Unloaded ONNX model: {model_key}")

        if model_key in self.model_metadata:
            del self.model_metadata[model_key]

        return True

    async def predict(
        self,
        symbol: str,
        features: np.ndarray,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Single prediction

        Args:
            symbol: Trading symbol
            features: Feature vector (shape: [timesteps, features])
            model_name: Конкретная модель (опционально)
            model_version: Версия модели (опционально)
            experiment_id: ID A/B эксперимента (опционально)

        Returns:
            Prediction result
        """
        start_time = time.perf_counter()

        # Initialize variant outside try block to avoid "referenced before assignment" error
        variant = None

        try:
            # Определить модель для использования
            if experiment_id:
                # A/B testing
                variant = await self.ab_manager.route_traffic(experiment_id)
                experiment = self.ab_manager.experiments.get(experiment_id)
                if experiment:
                    if variant == ModelVariant.CONTROL:
                        model_name = experiment.control_model_name
                        model_version = experiment.control_model_version
                    else:
                        model_name = experiment.treatment_model_name
                        model_version = experiment.treatment_model_version

            # Если модель не указана, используем production
            if not model_name:
                model_name = "hybrid_cnn_lstm"  # Default

            # Загрузить модель если не загружена
            model_key = f"{model_name}:{model_version}" if model_version else None
            if not model_key or model_key not in self.loaded_models:
                # Загружаем production версию
                success = await self.load_model(model_name, model_version)
                if not success:
                    raise ValueError(f"Failed to load model {model_name}")

                # Обновить model_key после загрузки
                loaded_keys = [k for k in self.loaded_models.keys() if k.startswith(model_name)]
                if not loaded_keys:
                    raise ValueError(f"No loaded model found for {model_name}")
                model_key = loaded_keys[0]

            # Inference
            use_onnx = model_key in self.onnx_sessions
            metadata = self.model_metadata[model_key]

            if use_onnx:
                # ONNX inference
                session = self.onnx_sessions[model_key]
                input_name = session.get_inputs()[0].name

                # Reshape для ONNX (добавить batch dimension)
                if features.ndim == 2:
                    features = np.expand_dims(features, axis=0)

                # Run inference
                outputs = session.run(None, {input_name: features.astype(np.float32)})
                prediction_tensor = outputs[0]

            else:
                # PyTorch inference
                model = self.loaded_models[model_key]

                # Convert to tensor
                if isinstance(features, np.ndarray):
                    features_tensor = torch.from_numpy(features).float()
                else:
                    features_tensor = features

                # Reshape если нужно (добавить batch dimension)
                if features_tensor.ndim == 2:
                    features_tensor = features_tensor.unsqueeze(0)

                # Inference
                with torch.no_grad():
                    output = model(features_tensor)

                # Получить outputs (зависит от архитектуры модели)
                if isinstance(output, tuple):
                    # Multi-task output: (direction, confidence, return)
                    direction_logits, confidence, expected_return = output
                    prediction_tensor = {
                        "direction": torch.argmax(direction_logits, dim=1).item(),
                        "confidence": confidence.item(),
                        "expected_return": expected_return.item()
                    }
                else:
                    # Single output
                    prediction_tensor = output.cpu().numpy()

            # Latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Формат результата
            if isinstance(prediction_tensor, dict):
                prediction = prediction_tensor
            else:
                # Default format - handle numpy array output
                if hasattr(prediction_tensor, 'tolist'):
                    output_value = prediction_tensor.tolist()
                elif hasattr(prediction_tensor, 'item'):
                    # Single scalar numpy value
                    output_value = float(prediction_tensor.item())
                else:
                    # Fallback for other array-like types
                    output_value = float(prediction_tensor.flatten()[0])

                prediction = {
                    "output": output_value,
                    "confidence": 0.5
                }

            # Stats
            self.total_predictions += 1
            self.total_latency_ms += latency_ms

            # Record для A/B testing (если применимо)
            if experiment_id and variant:
                outcome = PredictionOutcome(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    prediction=prediction,
                    latency_ms=latency_ms
                )
                await self.ab_manager.record_prediction(
                    experiment_id, variant, outcome
                )

            return {
                "prediction": prediction,
                "model_name": metadata["name"],
                "model_version": metadata["version"],
                "variant": variant.value if variant else None,
                "latency_ms": latency_ms,
                "use_onnx": use_onnx
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Prediction failed for {symbol}: {e}")

            # Record error для A/B testing
            if experiment_id and variant:
                outcome = PredictionOutcome(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    prediction=None,
                    latency_ms=latency_ms,
                    error=str(e)
                )
                await self.ab_manager.record_prediction(
                    experiment_id, variant, outcome
                )

            raise

    async def batch_predict(
        self,
        requests: List[Dict[str, Any]],
        max_batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Batch predictions (для оптимизации throughput)

        Args:
            requests: Список request dictionaries
            max_batch_size: Максимальный размер batch

        Returns:
            Список predictions
        """
        # TODO: Реализовать настоящий batching с padding
        # Пока делаем последовательно
        results = []
        for req in requests:
            try:
                pred = await self.predict(
                    symbol=req["symbol"],
                    features=np.array(req["features"]),
                    model_name=req.get("model_name"),
                    model_version=req.get("model_version"),
                    experiment_id=req.get("experiment_id")
                )
                results.append(pred)
            except Exception as e:
                logger.error(f"Batch prediction failed for {req['symbol']}: {e}")
                results.append({"error": str(e)})

        return results

    def get_loaded_models(self) -> List[ModelInfo]:
        """Получить список загруженных моделей"""
        models = []
        for key, metadata in self.model_metadata.items():
            models.append(ModelInfo(
                name=metadata["name"],
                version=metadata["version"],
                stage=metadata["stage"],
                model_type=metadata["model_type"],
                metrics=metadata["metrics"],
                size_mb=metadata["size_mb"],
                loaded=True
            ))
        return models

    async def health_check(self) -> HealthResponse:
        """Health check"""
        loaded_models = list(self.model_metadata.keys())
        active_experiments = self.ab_manager.get_active_experiments()

        # Determine health status
        if loaded_models:
            status = "healthy"
        elif active_experiments:
            status = "degraded"  # Есть experiments но нет loaded models
        else:
            status = "unhealthy"

        uptime = (datetime.now() - self.start_time).total_seconds()

        return HealthResponse(
            status=status,
            timestamp=datetime.now(),
            loaded_models=loaded_models,
            active_experiments=active_experiments,
            uptime_seconds=uptime
        )


# === FastAPI App ===

app = FastAPI(
    title="ML Model Server",
    description="ML Inference Server with A/B Testing",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Server instance
server = ModelServer()


@app.on_event("startup")
async def startup():
    """Startup: загрузить production модели"""
    logger.info("ML Model Server starting up...")

    # Загрузить default production модель
    try:
        await server.load_model("hybrid_cnn_lstm", version=None)
        logger.info("Production model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load production model: {e}")


@app.on_event("shutdown")
async def shutdown():
    """Shutdown: cleanup"""
    logger.info("ML Model Server shutting down...")


@app.post("/api/ml/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Single prediction"""
    try:
        features_array = np.array(request.features).reshape(60, -1)  # Предполагаем 60 timesteps

        result = await server.predict(
            symbol=request.symbol,
            features=features_array,
            model_name=request.model_name,
            model_version=request.model_version
        )

        return PredictResponse(
            symbol=request.symbol,
            prediction=result["prediction"],
            model_name=result["model_name"],
            model_version=result["model_version"],
            variant=result.get("variant"),
            latency_ms=result["latency_ms"],
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/predict/batch", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest):
    """Batch predictions"""
    start_time = time.perf_counter()

    try:
        # Convert requests to dict format
        requests_data = []
        for req in request.requests:
            requests_data.append({
                "symbol": req.symbol,
                "features": req.features,
                "model_name": req.model_name,
                "model_version": req.model_version
            })

        # Batch predict
        results = await server.batch_predict(requests_data, request.max_batch_size)

        # Convert to response format
        predictions = []
        for req, res in zip(request.requests, results):
            if "error" not in res:
                predictions.append(PredictResponse(
                    symbol=req.symbol,
                    prediction=res["prediction"],
                    model_name=res["model_name"],
                    model_version=res["model_version"],
                    variant=res.get("variant"),
                    latency_ms=res["latency_ms"],
                    timestamp=datetime.now()
                ))

        total_latency = (time.perf_counter() - start_time) * 1000

        return BatchPredictResponse(
            predictions=predictions,
            total_latency_ms=total_latency
        )

    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/models", response_model=List[ModelInfo])
async def list_models():
    """Список загруженных моделей"""
    return server.get_loaded_models()


@app.post("/api/ml/models/reload")
async def reload_model(request: ReloadRequest):
    """Reload модели"""
    try:
        success = await server.load_model(request.model_name, request.version)
        if success:
            return {"status": "success", "message": f"Model {request.model_name} reloaded"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        logger.error(f"Reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/ab-test/create")
async def create_ab_test(request: ABTestRequest):
    """Создать A/B test"""
    try:
        success = await server.ab_manager.create_experiment(
            experiment_id=request.experiment_id,
            control_model_name=request.control_model_name,
            control_model_version=request.control_model_version,
            treatment_model_name=request.treatment_model_name,
            treatment_model_version=request.treatment_model_version,
            control_traffic=request.control_traffic,
            treatment_traffic=request.treatment_traffic,
            duration_hours=request.duration_hours
        )

        if success:
            # Загрузить обе модели
            await server.load_model(request.control_model_name, request.control_model_version)
            await server.load_model(request.treatment_model_name, request.treatment_model_version)

            return {"status": "success", "experiment_id": request.experiment_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to create experiment")

    except Exception as e:
        logger.error(f"A/B test creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/ab-test/{experiment_id}/analyze")
async def analyze_experiment(experiment_id: str):
    """Анализ A/B теста"""
    try:
        analysis = await server.ab_manager.analyze_experiment(experiment_id)
        return analysis
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/ab-test/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """Остановить A/B test"""
    try:
        report = await server.ab_manager.stop_experiment(experiment_id)
        return report
    except Exception as e:
        logger.error(f"Stop experiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/health", response_model=HealthResponse)
async def health():
    """Health check"""
    return await server.health_check()


# ==================== COMPATIBILITY ALIASES ====================
# Добавляем endpoints без префикса /api/ml/ для совместимости с MLSignalValidator

@app.get("/health")
async def health_alias():
    """Health check (alias for compatibility)"""
    return await server.health_check()


@app.post("/predict")
async def predict_alias(request: PredictRequest):
    """Single prediction (alias for compatibility)"""
    return await predict(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
