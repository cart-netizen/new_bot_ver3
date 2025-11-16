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
        logger.debug(f"Model Registry initialized at {self.registry_dir}")

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
        logger.info(f"[GET_MODEL] Request: name={name}, version={version}, stage={stage}")

        model_base_dir = self.registry_dir / name
        if not model_base_dir.exists():
            logger.warning(f"[GET_MODEL] Model directory {model_base_dir} does not exist")
            return None

        logger.info(f"[GET_MODEL] Model directory exists: {model_base_dir}")

        # Если указана стадия, ищем stage marker или symlink (backward compatibility)
        if stage:
            # Нормализуем stage - может быть строкой или enum
            if isinstance(stage, ModelStage):
                stage_str = stage.value.lower()
            elif isinstance(stage, str):
                stage_str = stage.lower()
            else:
                stage_str = str(stage).lower()

            stage_marker = model_base_dir / f".{stage_str}"
            stage_link = model_base_dir / stage_str

            # Try new marker format first
            if stage_marker.exists():
                with open(stage_marker, 'r') as f:
                    version = f.read().strip()
                logger.debug(f"Stage {stage} points to version {version}")
            # Fallback to old symlink format
            elif stage_link.is_symlink():
                version = stage_link.readlink().name
                logger.debug(f"Stage {stage} points to version {version} (symlink)")
            else:
                logger.warning(f"No {stage} model for {name}")
                return None

        # Если версия не указана, берем production или latest
        if not version:
            logger.info(f"[GET_MODEL] No version specified, looking for production or latest")

            # Пытаемся взять production (try marker first, then symlink)
            production_marker = model_base_dir / ".production"
            production_link = model_base_dir / "production"

            logger.info(f"[GET_MODEL] Checking production_marker: {production_marker}, exists={production_marker.exists()}")

            if production_marker.exists():
                with open(production_marker, 'r') as f:
                    version = f.read().strip()
                logger.info(f"[GET_MODEL] Found production marker pointing to version: {version}")
            elif production_link.is_symlink():
                version = production_link.readlink().name
                logger.info(f"[GET_MODEL] Found production symlink pointing to version: {version}")
            else:
                logger.info(f"[GET_MODEL] No production marker/symlink, searching for latest version")
                # Берем последнюю версию
                all_items = list(model_base_dir.iterdir())
                logger.info(f"[GET_MODEL] All items in {model_base_dir}: {[item.name for item in all_items]}")

                versions = [
                    d.name for d in all_items
                    if d.is_dir() and not d.name.startswith('.')
                ]
                logger.info(f"[GET_MODEL] Found versions: {versions}")

                if not versions:
                    logger.warning(f"[GET_MODEL] No versions found for {name} in {model_base_dir}")
                    return None
                version = sorted(versions)[-1]
                logger.info(f"[GET_MODEL] Using latest version: {version}")

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
        Установить стадию для модели (создать/обновить stage marker)

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
        version_dir = model_base_dir / version

        # Нормализуем stage - может быть строкой или enum
        if isinstance(stage, ModelStage):
            stage_str = stage.value.lower()
            stage_enum = stage
        elif isinstance(stage, str):
            stage_str = stage.lower()
            # Пытаемся преобразовать строку в enum
            try:
                stage_enum = ModelStage(stage.capitalize())
            except ValueError:
                stage_enum = None
        else:
            stage_str = str(stage).lower()
            stage_enum = None

        stage_marker = model_base_dir / f".{stage_str}"

        # If promoting to PRODUCTION, demote all other versions from PRODUCTION
        if stage_enum == ModelStage.PRODUCTION or stage_str == "production":
            all_versions = await self.list_models(name)
            for other_model in all_versions:
                if other_model.metadata.version != version and other_model.metadata.stage == ModelStage.PRODUCTION:
                    # Demote to STAGING
                    other_version_dir = model_base_dir / other_model.metadata.version
                    other_metadata_path = other_version_dir / "metadata.json"
                    other_model.metadata.stage = ModelStage.STAGING
                    other_model.metadata.updated_at = datetime.now()
                    with open(other_metadata_path, 'w') as f:
                        json.dump(other_model.metadata.model_dump(), f, indent=2, default=str)
                    logger.info(f"Demoted {name} v{other_model.metadata.version} from PRODUCTION to STAGING")

        # Удалить старый marker/symlink если есть
        if stage_marker.exists():
            stage_marker.unlink()

        # Also try to remove old symlink for backward compatibility
        old_symlink = model_base_dir / stage_str
        if old_symlink.exists() or old_symlink.is_symlink():
            try:
                old_symlink.unlink()
            except:
                pass

        # Создать новый stage marker (текстовый файл с версией)
        # Это работает на всех ОС без прав администратора
        with open(stage_marker, 'w') as f:
            f.write(version)

        # Обновить метаданные (убедимся что это enum)
        if stage_enum:
            model_info.metadata.stage = stage_enum
        elif isinstance(stage, ModelStage):
            model_info.metadata.stage = stage
        else:
            # Пытаемся преобразовать строку в enum
            try:
                model_info.metadata.stage = ModelStage(stage_str.capitalize())
            except ValueError:
                # Если не получилось, используем как есть (может быть уже enum)
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
