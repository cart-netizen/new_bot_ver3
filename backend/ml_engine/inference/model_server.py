"""
Model Serving Server для real-time ML inference.

Функциональность:
- REST API для inference запросов
- Model versioning и hot reload
- A/B testing между моделями
- Batch prediction для производительности
- Health checks и monitoring
- Graceful shutdown

Путь: backend/ml_engine/inference/model_server.py
"""
from contextlib import asynccontextmanager

import torch
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
from datetime import datetime
from collections import defaultdict
import numpy as np

from backend.core.logger import get_logger
# UPDATED: Используем оптимизированные v2 версии
from backend.ml_engine.models.hybrid_cnn_lstm_v2 import (
    HybridCNNLSTMv2 as HybridCNNLSTM,
    ModelConfigV2 as ModelConfig,
    create_model_v2 as create_model
)

logger = get_logger(__name__)


# ==================== API MODELS ====================

class PredictionRequest(BaseModel):
  """Запрос на предсказание."""
  symbol: str = Field(..., description="Торговая пара")
  sequence: List[List[float]] = Field(
    ...,
    description="Временная последовательность (seq_len, features)"
  )
  model_version: Optional[str] = Field(
    "latest",
    description="Версия модели"
  )
  return_probabilities: bool = Field(
    False,
    description="Вернуть вероятности для каждого класса"
  )


class BatchPredictionRequest(BaseModel):
  """Batch запрос на предсказания."""
  requests: List[PredictionRequest]


class PredictionResponse(BaseModel):
  """Ответ с предсказанием."""
  symbol: str
  direction: str  # "BUY", "HOLD", "SELL"
  direction_class: int  # 1, 0, 2
  confidence: float
  expected_return: float
  probabilities: Optional[Dict[str, float]] = None
  inference_time_ms: float
  model_version: str
  timestamp: int


class BatchPredictionResponse(BaseModel):
  """Batch ответ с предсказаниями."""
  predictions: List[PredictionResponse]
  total_inference_time_ms: float


class ModelInfo(BaseModel):
  """Информация о модели."""
  version: str
  loaded_at: str
  parameters: int
  device: str
  inference_count: int
  avg_inference_time_ms: float


class ServerStatus(BaseModel):
  """Статус сервера."""
  status: str
  uptime_seconds: float
  models_loaded: int
  total_predictions: int
  avg_latency_ms: float


# ==================== MODEL REGISTRY ====================

class ModelRegistry:
  """
  Реестр моделей с версионированием и hot reload.

  Поддерживает:
  - Загрузка нескольких версий модели
  - Hot reload без downtime
  - A/B testing между версиями
  - Статистика использования
  """

  def __init__(self, checkpoint_dir: str = "models"):
    """
    Инициализация реестра.

    Args:
        checkpoint_dir: Директория с checkpoint файлами
    """
    self.checkpoint_dir = Path(checkpoint_dir)
    self.models: Dict[str, HybridCNNLSTM] = {}
    self.model_info: Dict[str, Dict] = {}
    self.model_stats: Dict[str, Dict] = defaultdict(
      lambda: {
        'inference_count': 0,
        'total_inference_time': 0.0,
        'errors': 0
      }
    )

    self.device = torch.device(
      "cuda" if torch.cuda.is_available() else "cpu"
    )

    self.default_version = "latest"

    logger.info(
      f"Инициализирован ModelRegistry: device={self.device}, "
      f"checkpoint_dir={self.checkpoint_dir}"
    )

  def load_model(
      self,
      version: str,
      checkpoint_path: Optional[str] = None
  ) -> bool:
    """
    Загрузить модель.

    Args:
        version: Версия модели
        checkpoint_path: Путь к checkpoint (None = auto-detect)

    Returns:
        True если успешно загружено
    """
    try:
      if checkpoint_path is None:
        # Auto-detect checkpoint
        if version == "latest":
          checkpoint_path = self.checkpoint_dir / "best_model.pth"
        else:
          checkpoint_path = self.checkpoint_dir / f"{version}.pth"

      checkpoint_path = Path(checkpoint_path)

      if not checkpoint_path.exists():
        logger.error(f"Checkpoint не найден: {checkpoint_path}")
        return False

      # Загружаем checkpoint
      checkpoint = torch.load(
        checkpoint_path,
        map_location=self.device
      )

      # Создаем модель с конфигурацией из checkpoint
      model_config = checkpoint.get('config')
      if model_config:
        # Извлекаем ModelConfig из TrainerConfig
        model = create_model(ModelConfig())
      else:
        model = create_model()

      # Загружаем веса
      model.load_state_dict(checkpoint['model_state_dict'])
      model.to(self.device)
      model.eval()  # Inference mode

      # Compile для ускорения (PyTorch 2.0+)
      if hasattr(torch, 'compile'):
        try:
          model = torch.compile(model, mode='reduce-overhead')
          logger.info(f"Модель {version} скомпилирована")
        except Exception as e:
          logger.warning(f"Не удалось скомпилировать модель: {e}")

      # Сохраняем модель
      self.models[version] = model

      # Сохраняем информацию
      model_size = model.get_model_size()
      self.model_info[version] = {
        'loaded_at': datetime.now().isoformat(),
        'checkpoint_path': str(checkpoint_path),
        'epoch': checkpoint.get('epoch', 'unknown'),
        'parameters': model_size['total_params'],
        'device': str(self.device)
      }

      logger.info(
        f"Загружена модель {version}: "
        f"epoch={checkpoint.get('epoch')}, "
        f"params={model_size['total_params']:,}"
      )

      return True

    except Exception as e:
      logger.error(f"Ошибка загрузки модели {version}: {e}")
      return False

  def get_model(self, version: str = "latest") -> Optional[HybridCNNLSTM]:
    """
    Получить модель по версии.

    Args:
        version: Версия модели

    Returns:
        Модель или None
    """
    if version not in self.models:
      logger.warning(f"Модель {version} не загружена")
      return None

    return self.models[version]

  def unload_model(self, version: str) -> bool:
    """Выгрузить модель из памяти."""
    if version in self.models:
      del self.models[version]
      del self.model_info[version]
      logger.info(f"Модель {version} выгружена")
      return True
    return False

  def list_models(self) -> List[str]:
    """Список загруженных моделей."""
    return list(self.models.keys())

  def get_model_info(self, version: str) -> Optional[Dict]:
    """Получить информацию о модели."""
    return self.model_info.get(version)

  def update_stats(
      self,
      version: str,
      inference_time: float,
      error: bool = False
  ):
    """Обновить статистику модели."""
    stats = self.model_stats[version]
    stats['inference_count'] += 1
    stats['total_inference_time'] += inference_time
    if error:
      stats['errors'] += 1

  def get_stats(self, version: str) -> Dict:
    """Получить статистику модели."""
    stats = self.model_stats[version]

    count = stats['inference_count']
    avg_time = (
      stats['total_inference_time'] / count
      if count > 0
      else 0.0
    )

    return {
      'inference_count': count,
      'avg_inference_time_ms': avg_time * 1000,
      'errors': stats['errors']
    }


# ==================== MODEL SERVER ====================

class MLModelServer:
  """
  Сервер для ML inference.

  Функциональность:
  - Синхронные и batch предсказания
  - Автоматическое управление моделями
  - Мониторинг производительности
  """

  def __init__(self, checkpoint_dir: str = "models"):
    """Инициализация сервера."""
    self.registry = ModelRegistry(checkpoint_dir)
    self.start_time = time.time()
    self.total_predictions = 0
    self.total_inference_time = 0.0

    # Маппинг классов на направления (СТАНДАРТ: 0=SELL, 1=HOLD, 2=BUY)
    self.direction_map = {
      0: "SELL",
      1: "HOLD",
      2: "BUY"
    }

    logger.info("Инициализирован MLModelServer")

  async def initialize(self):
    """Инициализация: загрузка моделей."""
    # Загружаем latest модель
    success = self.registry.load_model("latest")

    if not success:
      logger.warning(
        "Не удалось загрузить latest модель. "
        "Создаем новую модель для тестирования."
      )
      # Создаем и сохраняем тестовую модель
      model = create_model()
      model.to(self.registry.device)
      self.registry.models["latest"] = model
      self.registry.model_info["latest"] = {
        'loaded_at': datetime.now().isoformat(),
        'parameters': model.get_model_size()['total_params'],
        'device': str(self.registry.device)
      }

    logger.info("MLModelServer инициализирован")

  async def predict(
      self,
      request: PredictionRequest
  ) -> PredictionResponse:
    """
    Выполнить предсказание.

    Args:
        request: Запрос на предсказание

    Returns:
        Ответ с предсказанием
    """
    start_time = time.time()

    try:
      # Получаем модель
      model = self.registry.get_model(request.model_version)

      if model is None:
        raise HTTPException(
          status_code=404,
          detail=f"Модель {request.model_version} не найдена"
        )

      # Подготовка данных
      sequence = np.array(request.sequence, dtype=np.float32)

      # Проверка размерности
      if len(sequence.shape) != 2:
        raise HTTPException(
          status_code=400,
          detail="sequence должен иметь размерность (seq_len, features)"
        )

      # Добавляем batch dimension
      sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
      sequence_tensor = sequence_tensor.to(self.registry.device)

      # Inference
      with torch.no_grad():
        predictions = model.predict(sequence_tensor)

      # Извлекаем результаты
      direction_class = predictions['direction'].item()
      direction = self.direction_map[direction_class]
      confidence = predictions['confidence'].item()
      expected_return = predictions['expected_return'].item()

      # Probabilities если запрошены (СТАНДАРТ: probs[0]=SELL, probs[1]=HOLD, probs[2]=BUY)
      probabilities = None
      if request.return_probabilities:
        probs = predictions['direction_probs'][0].cpu().numpy()
        probabilities = {
          "SELL": float(probs[0]),
          "HOLD": float(probs[1]),
          "BUY": float(probs[2])
        }

      inference_time = (time.time() - start_time) * 1000  # ms

      # Обновляем статистику
      self.registry.update_stats(
        request.model_version,
        time.time() - start_time
      )
      self.total_predictions += 1
      self.total_inference_time += time.time() - start_time

      return PredictionResponse(
        symbol=request.symbol,
        direction=direction,
        direction_class=direction_class,
        confidence=confidence,
        expected_return=expected_return,
        probabilities=probabilities,
        inference_time_ms=inference_time,
        model_version=request.model_version,
        timestamp=int(time.time() * 1000)
      )

    except Exception as e:
      self.registry.update_stats(
        request.model_version,
        time.time() - start_time,
        error=True
      )
      logger.error(f"Ошибка предсказания: {e}")
      raise HTTPException(status_code=500, detail=str(e))

  async def predict_batch(
      self,
      request: BatchPredictionRequest
  ) -> BatchPredictionResponse:
    """
    Batch предсказания.

    Args:
        request: Batch запрос

    Returns:
        Batch ответ
    """
    start_time = time.time()

    # Обрабатываем все запросы параллельно
    tasks = [self.predict(req) for req in request.requests]
    predictions = await asyncio.gather(*tasks, return_exceptions=True)

    # Фильтруем ошибки
    valid_predictions = [
      p for p in predictions
      if not isinstance(p, Exception)
    ]

    total_time = (time.time() - start_time) * 1000

    return BatchPredictionResponse(
      predictions=valid_predictions,
      total_inference_time_ms=total_time
    )

  def get_status(self) -> ServerStatus:
    """Получить статус сервера."""
    uptime = time.time() - self.start_time
    avg_latency = (
      (self.total_inference_time / self.total_predictions * 1000)
      if self.total_predictions > 0
      else 0.0
    )

    return ServerStatus(
      status="running",
      uptime_seconds=uptime,
      models_loaded=len(self.registry.list_models()),
      total_predictions=self.total_predictions,
      avg_latency_ms=avg_latency
    )


# ==================== FASTAPI APPLICATION ====================

# Создаем глобальный экземпляр сервера
model_server = MLModelServer()





# @app.on_event("startup")
# async def startup_event():
#   """Инициализация при запуске."""
#   await model_server.initialize()
#   logger.info("ML Model Server запущен")
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler."""
    # Startup
    await model_server.initialize()
    logger.info("ML Model Server запущен")
    yield
    # Shutdown (если нужна очистка)
    logger.info("ML Model Server остановлен")


# FastAPI app
app = FastAPI(
  title="ML Model Serving API",
  description="Real-time ML inference для торгового бота",
  version="1.0.0",
  lifespan=lifespan
)

@app.get("/")
async def root():
  """Root endpoint."""
  return {"message": "ML Model Serving API", "status": "running"}


@app.get("/health")
async def health_check():
  """Health check endpoint."""
  return {"status": "healthy", "timestamp": int(time.time() * 1000)}


@app.get("/status", response_model=ServerStatus)
async def get_status():
  """Получить статус сервера."""
  return model_server.get_status()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
  """Получить предсказание."""
  return await model_server.predict(request)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
  """Batch предсказания."""
  return await model_server.predict_batch(request)


@app.get("/models")
async def list_models():
  """Список загруженных моделей."""
  models = model_server.registry.list_models()

  models_info = []
  for version in models:
    info = model_server.registry.get_model_info(version)
    stats = model_server.registry.get_stats(version)

    models_info.append({
      'version': version,
      'info': info,
      'stats': stats
    })

  return {'models': models_info}


@app.get("/models/{version}", response_model=ModelInfo)
async def get_model_info(version: str):
  """Получить информацию о модели."""
  info = model_server.registry.get_model_info(version)

  if info is None:
    raise HTTPException(
      status_code=404,
      detail=f"Модель {version} не найдена"
    )

  stats = model_server.registry.get_stats(version)

  return ModelInfo(
    version=version,
    loaded_at=info['loaded_at'],
    parameters=info['parameters'],
    device=info['device'],
    inference_count=stats['inference_count'],
    avg_inference_time_ms=stats['avg_inference_time_ms']
  )


@app.post("/models/{version}/reload")
async def reload_model(version: str, background_tasks: BackgroundTasks):
  """Hot reload модели."""

  def reload_task():
    success = model_server.registry.load_model(version)
    if success:
      logger.info(f"Модель {version} перезагружена")
    else:
      logger.error(f"Ошибка перезагрузки модели {version}")

  background_tasks.add_task(reload_task)

  return {"message": f"Перезагрузка модели {version} запущена"}


# Пример запуска
if __name__ == "__main__":
  import uvicorn

  uvicorn.run(
    app,
    host="0.0.0.0",
    port=8001,
    log_level="info"
  )