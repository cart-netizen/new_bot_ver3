"""
Model Registry - версионирование и управление ML моделями

Функции:
- Регистрация новых моделей с версионированием
- Управление жизненным циклом моделей (staging → production)
- Метаданные и метрики моделей
- Хранение моделей в файловой системе
"""

import json
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import torch
import asyncio
from backend.core.logger import get_logger

logger = get_logger(__name__)


class ModelStage(str, Enum):
    """Стадии жизненного цикла модели"""
    NONE = "None"  # Только зарегистрирована
    STAGING = "Staging"  # Тестируется
    PRODUCTION = "Production"  # В production
    ARCHIVED = "Archived"  # Устарела


class ModelMetadata(BaseModel):
    """Метаданные модели"""
    name: str
    version: str
    stage: ModelStage = ModelStage.NONE
    created_at: datetime
    updated_at: datetime
    description: Optional[str] = None

    # Информация о модели
    model_type: str  # "HybridCNNLSTM", "Stockformer", etc.
    framework: str = "pytorch"  # "pytorch", "onnx", "tensorflow"

    # Метрики производительности
    metrics: Dict[str, float] = Field(default_factory=dict)
    # Примеры: accuracy, precision, recall, sharpe_ratio, latency_ms

    # Информация о обучении
    training_params: Dict[str, Any] = Field(default_factory=dict)
    dataset_info: Dict[str, Any] = Field(default_factory=dict)

    # Технические детали
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    model_size_mb: Optional[float] = None

    # Тэги для поиска
    tags: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True


class ModelInfo(BaseModel):
    """Информация о зарегистрированной модели"""
    metadata: ModelMetadata
    model_path: Path
    onnx_path: Optional[Path] = None

    def model_exists(self) -> bool:
        """Проверка существования модели"""
        return self.model_path.exists()

    def onnx_exists(self) -> bool:
        """Проверка существования ONNX версии"""
        return self.onnx_path is not None and self.onnx_path.exists()


class ModelRegistry:
    """
    Model Registry для управления ML моделями

    Структура хранения:
    models/
    ├── hybrid_cnn_lstm/
    │   ├── v1.0.0/
    │   │   ├── model.pt
    │   │   ├── model.onnx (optional)
    │   │   ├── metadata.json
    │   │   └── metrics.json
    │   ├── v1.1.0/
    │   ├── production -> v1.0.0  (symlink)
    │   └── staging -> v1.1.0     (symlink)
    """

    def __init__(self, registry_dir: str = "models"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model Registry initialized at {self.registry_dir}")

    async def register_model(
        self,
        name: str,
        version: str,
        model_path: Path,
        model_type: str,
        description: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        training_params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        onnx_path: Optional[Path] = None
    ) -> ModelInfo:
        """
        Регистрация новой модели

        Args:
            name: Название модели (например, "hybrid_cnn_lstm")
            version: Версия (например, "1.0.0")
            model_path: Путь к файлу модели
            model_type: Тип модели
            description: Описание
            metrics: Метрики производительности
            training_params: Параметры обучения
            tags: Тэги для поиска
            onnx_path: Путь к ONNX версии (опционально)

        Returns:
            ModelInfo с информацией о зарегистрированной модели
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Создать директорию для модели
        model_dir = self.registry_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Скопировать файлы модели
        target_model_path = model_dir / "model.pt"
        shutil.copy2(model_path, target_model_path)
        logger.info(f"Copied model to {target_model_path}")

        target_onnx_path = None
        if onnx_path and onnx_path.exists():
            target_onnx_path = model_dir / "model.onnx"
            shutil.copy2(onnx_path, target_onnx_path)
            logger.info(f"Copied ONNX model to {target_onnx_path}")

        # Вычислить размер модели
        model_size_mb = target_model_path.stat().st_size / (1024 * 1024)

        # Создать метаданные
        metadata = ModelMetadata(
            name=name,
            version=version,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            description=description,
            model_type=model_type,
            metrics=metrics or {},
            training_params=training_params or {},
            tags=tags or [],
            model_size_mb=round(model_size_mb, 2)
        )

        # Сохранить метаданные
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.model_dump(), f, indent=2, default=str)

        logger.info(
            f"Registered model {name} v{version} "
            f"({model_size_mb:.2f} MB, {metadata.model_type})"
        )

        return ModelInfo(
            metadata=metadata,
            model_path=target_model_path,
            onnx_path=target_onnx_path
        )

    async def get_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Optional[ModelInfo]:
        """
        Получить модель по имени и версии или стадии

        Args:
            name: Название модели
            version: Версия (если None, берется production или latest)
            stage: Стадия (альтернатива version)

        Returns:
            ModelInfo или None если не найдена
        """
        model_base_dir = self.registry_dir / name
        if not model_base_dir.exists():
            logger.warning(f"Model {name} not found in registry")
            return None

        # Если указана стадия, ищем symlink
        if stage:
            stage_link = model_base_dir / stage.value.lower()
            if stage_link.is_symlink():
                version = stage_link.readlink().name
                logger.debug(f"Stage {stage} points to version {version}")
            else:
                logger.warning(f"No {stage} model for {name}")
                return None

        # Если версия не указана, берем production или latest
        if not version:
            # Пытаемся взять production
            production_link = model_base_dir / "production"
            if production_link.is_symlink():
                version = production_link.readlink().name
                logger.debug(f"Using production version {version}")
            else:
                # Берем последнюю версию
                versions = [
                    d.name for d in model_base_dir.iterdir()
                    if d.is_dir() and not d.name.startswith('.')
                ]
                if not versions:
                    logger.warning(f"No versions found for {name}")
                    return None
                version = sorted(versions)[-1]
                logger.debug(f"Using latest version {version}")

        model_dir = model_base_dir / version
        if not model_dir.exists():
            logger.warning(f"Model {name} v{version} not found")
            return None

        # Загрузить метаданные
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            logger.warning(f"Metadata not found for {name} v{version}")
            return None

        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
            metadata = ModelMetadata(**metadata_dict)

        model_path = model_dir / "model.pt"
        onnx_path = model_dir / "model.onnx"

        return ModelInfo(
            metadata=metadata,
            model_path=model_path,
            onnx_path=onnx_path if onnx_path.exists() else None
        )

    async def list_models(self, name: Optional[str] = None) -> List[ModelInfo]:
        """
        Список всех моделей или версий конкретной модели

        Args:
            name: Название модели (если None, возвращает все модели)

        Returns:
            Список ModelInfo
        """
        models = []

        if name:
            # Список версий конкретной модели
            model_dir = self.registry_dir / name
            if not model_dir.exists():
                return []

            for version_dir in model_dir.iterdir():
                if version_dir.is_dir() and not version_dir.name.startswith('.'):
                    model_info = await self.get_model(name, version_dir.name)
                    if model_info:
                        models.append(model_info)
        else:
            # Список всех моделей
            for model_dir in self.registry_dir.iterdir():
                if model_dir.is_dir():
                    for version_dir in model_dir.iterdir():
                        if version_dir.is_dir() and not version_dir.name.startswith('.'):
                            model_info = await self.get_model(
                                model_dir.name,
                                version_dir.name
                            )
                            if model_info:
                                models.append(model_info)

        return models

    async def set_model_stage(
        self,
        name: str,
        version: str,
        stage: ModelStage
    ) -> bool:
        """
        Установить стадию для модели (создать/обновить symlink)

        Args:
            name: Название модели
            version: Версия
            stage: Новая стадия

        Returns:
            True если успешно
        """
        model_info = await self.get_model(name, version)
        if not model_info:
            logger.error(f"Model {name} v{version} not found")
            return False

        model_base_dir = self.registry_dir / name
        stage_link = model_base_dir / stage.value.lower()

        # Удалить старый symlink если есть
        if stage_link.exists() or stage_link.is_symlink():
            stage_link.unlink()

        # Создать новый symlink
        version_dir = model_base_dir / version
        stage_link.symlink_to(version_dir.name)

        # Обновить метаданные
        model_info.metadata.stage = stage
        model_info.metadata.updated_at = datetime.now()

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_info.metadata.model_dump(), f, indent=2, default=str)

        logger.info(f"Set {name} v{version} to {stage}")
        return True

    async def promote_to_production(self, name: str, version: str) -> bool:
        """
        Продвинуть модель в production (staging → production)

        Args:
            name: Название модели
            version: Версия

        Returns:
            True если успешно
        """
        return await self.set_model_stage(name, version, ModelStage.PRODUCTION)

    async def retire_model(self, name: str, version: str) -> bool:
        """
        Отправить модель в архив (production → archived)

        Args:
            name: Название модели
            version: Версия

        Returns:
            True если успешно
        """
        return await self.set_model_stage(name, version, ModelStage.ARCHIVED)

    async def delete_model(self, name: str, version: str) -> bool:
        """
        Удалить модель из registry

        Args:
            name: Название модели
            version: Версия

        Returns:
            True если успешно
        """
        model_dir = self.registry_dir / name / version
        if not model_dir.exists():
            logger.warning(f"Model {name} v{version} not found")
            return False

        # Проверить, не в production ли модель
        production_link = self.registry_dir / name / "production"
        if production_link.is_symlink():
            production_version = production_link.readlink().name
            if production_version == version:
                logger.error(
                    f"Cannot delete {name} v{version}: it's in production"
                )
                return False

        # Удалить директорию модели
        shutil.rmtree(model_dir)
        logger.info(f"Deleted model {name} v{version}")
        return True

    async def get_production_model(self, name: str) -> Optional[ModelInfo]:
        """Получить production версию модели"""
        return await self.get_model(name, stage=ModelStage.PRODUCTION)

    async def get_staging_model(self, name: str) -> Optional[ModelInfo]:
        """Получить staging версию модели"""
        return await self.get_model(name, stage=ModelStage.STAGING)

    async def compare_models(
        self,
        name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Сравнить две версии модели

        Returns:
            Словарь с результатами сравнения
        """
        model1 = await self.get_model(name, version1)
        model2 = await self.get_model(name, version2)

        if not model1 or not model2:
            raise ValueError(f"One or both models not found")

        comparison = {
            "model_name": name,
            "version1": version1,
            "version2": version2,
            "metrics_comparison": {},
            "size_comparison": {
                "v1_mb": model1.metadata.model_size_mb,
                "v2_mb": model2.metadata.model_size_mb,
                "diff_mb": model2.metadata.model_size_mb - model1.metadata.model_size_mb
            }
        }

        # Сравнить метрики
        all_metrics = set(model1.metadata.metrics.keys()) | set(model2.metadata.metrics.keys())
        for metric in all_metrics:
            val1 = model1.metadata.metrics.get(metric)
            val2 = model2.metadata.metrics.get(metric)

            if val1 is not None and val2 is not None:
                diff = val2 - val1
                diff_pct = (diff / val1 * 100) if val1 != 0 else 0
                comparison["metrics_comparison"][metric] = {
                    "v1": val1,
                    "v2": val2,
                    "diff": diff,
                    "diff_pct": round(diff_pct, 2)
                }

        return comparison

    async def get_model_metadata(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """Получить только метаданные модели"""
        model_info = await self.get_model(name, version)
        return model_info.metadata if model_info else None

    async def update_metrics(
        self,
        name: str,
        version: str,
        metrics: Dict[str, float]
    ) -> bool:
        """
        Обновить метрики модели

        Args:
            name: Название модели
            version: Версия
            metrics: Новые метрики

        Returns:
            True если успешно
        """
        model_info = await self.get_model(name, version)
        if not model_info:
            return False

        # Обновить метрики
        model_info.metadata.metrics.update(metrics)
        model_info.metadata.updated_at = datetime.now()

        # Сохранить
        model_dir = self.registry_dir / name / version
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_info.metadata.model_dump(), f, indent=2, default=str)

        logger.info(f"Updated metrics for {name} v{version}")
        return True


# Singleton instance
_registry_instance: Optional[ModelRegistry] = None


def get_model_registry(registry_dir: str = "models") -> ModelRegistry:
    """Получить singleton instance Model Registry"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry(registry_dir)
    return _registry_instance
