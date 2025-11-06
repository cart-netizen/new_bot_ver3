"""
ML Model Client - клиент для взаимодействия с Model Server

Используется основным ботом для получения ML predictions
"""

import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
import numpy as np

from backend.core.logger import get_logger

logger = get_logger(__name__)


class ModelClient:
    """
    Client для взаимодействия с ML Model Server

    Использование:
        client = ModelClient("http://localhost:8001")
        prediction = await client.predict("BTCUSDT", features)
    """

    def __init__(self, server_url: str = "http://localhost:8001"):
        self.server_url = server_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialized = False

        logger.info(f"Model Client initialized for {self.server_url}")

    async def initialize(self):
        """Инициализация HTTP сессии"""
        if not self._initialized:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            self._initialized = True
            logger.info("Model Client HTTP session created")

    async def cleanup(self):
        """Cleanup HTTP сессии"""
        if self.session and not self.session.closed:
            await self.session.close()
            self._initialized = False
            logger.info("Model Client HTTP session closed")

    async def predict(
        self,
        symbol: str,
        features: np.ndarray,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Получить prediction от Model Server

        Args:
            symbol: Trading symbol
            features: Feature vector/matrix
            model_name: Specific model (optional)
            model_version: Model version (optional)

        Returns:
            Prediction dict или None при ошибке
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Подготовить features
            if isinstance(features, np.ndarray):
                features_list = features.flatten().tolist()
            elif isinstance(features, list):
                features_list = features
            else:
                logger.error(f"Invalid features type: {type(features)}")
                return None

            # Request payload
            payload = {
                "symbol": symbol,
                "features": features_list
            }

            if model_name:
                payload["model_name"] = model_name
            if model_version:
                payload["model_version"] = model_version

            # POST request
            url = f"{self.server_url}/api/ml/predict"
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Prediction request failed: {response.status} - {error_text}"
                    )
                    return None

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

    async def batch_predict(
        self,
        requests: List[Dict[str, Any]],
        max_batch_size: int = 32
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Batch predictions

        Args:
            requests: List of request dicts (symbol, features, ...)
            max_batch_size: Max batch size

        Returns:
            List of prediction dicts
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Подготовить payload
            payload = {
                "requests": requests,
                "max_batch_size": max_batch_size
            }

            url = f"{self.server_url}/api/ml/predict/batch"
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("predictions", [])
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Batch prediction failed: {response.status} - {error_text}"
                    )
                    return []

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return []

    async def health_check(self) -> bool:
        """
        Health check Model Server

        Returns:
            True если server healthy
        """
        if not self._initialized:
            await self.initialize()

        try:
            url = f"{self.server_url}/api/ml/health"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    status = data.get("status")
                    logger.debug(f"Model Server health: {status}")
                    return status in ["healthy", "degraded"]
                else:
                    logger.warning(f"Health check failed: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Получить список загруженных моделей

        Returns:
            List of model info dicts
        """
        if not self._initialized:
            await self.initialize()

        try:
            url = f"{self.server_url}/api/ml/models"
            async with self.session.get(url) as response:
                if response.status == 200:
                    models = await response.json()
                    return models
                else:
                    logger.error(f"List models failed: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"List models error: {e}")
            return []

    async def reload_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Reload модели на сервере

        Args:
            model_name: Model name
            version: Version (None = production)

        Returns:
            True если успешно
        """
        if not self._initialized:
            await self.initialize()

        try:
            payload = {"model_name": model_name}
            if version:
                payload["version"] = version

            url = f"{self.server_url}/api/ml/models/reload"
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"Model {model_name} reloaded successfully")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Reload failed: {error_text}")
                    return False

        except Exception as e:
            logger.error(f"Reload error: {e}")
            return False

    async def create_ab_test(
        self,
        experiment_id: str,
        control_model: str,
        control_version: str,
        treatment_model: str,
        treatment_version: str,
        traffic_split: float = 0.9
    ) -> bool:
        """
        Создать A/B test

        Args:
            experiment_id: Experiment ID
            control_model: Control model name
            control_version: Control version
            treatment_model: Treatment model name
            treatment_version: Treatment version
            traffic_split: Control traffic % (treatment = 1 - split)

        Returns:
            True если успешно
        """
        if not self._initialized:
            await self.initialize()

        try:
            payload = {
                "experiment_id": experiment_id,
                "control_model_name": control_model,
                "control_model_version": control_version,
                "treatment_model_name": treatment_model,
                "treatment_model_version": treatment_version,
                "control_traffic": traffic_split,
                "treatment_traffic": 1.0 - traffic_split
            }

            url = f"{self.server_url}/api/ml/ab-test/create"
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"A/B test {experiment_id} created")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Create A/B test failed: {error_text}")
                    return False

        except Exception as e:
            logger.error(f"Create A/B test error: {e}")
            return False

    async def get_ab_test_analysis(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить анализ A/B теста

        Args:
            experiment_id: Experiment ID

        Returns:
            Analysis dict или None
        """
        if not self._initialized:
            await self.initialize()

        try:
            url = f"{self.server_url}/api/ml/ab-test/{experiment_id}/analyze"
            async with self.session.get(url) as response:
                if response.status == 200:
                    analysis = await response.json()
                    return analysis
                else:
                    logger.error(f"Get analysis failed: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Get analysis error: {e}")
            return None

    async def stop_ab_test(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Остановить A/B test

        Args:
            experiment_id: Experiment ID

        Returns:
            Final report dict или None
        """
        if not self._initialized:
            await self.initialize()

        try:
            url = f"{self.server_url}/api/ml/ab-test/{experiment_id}/stop"
            async with self.session.post(url) as response:
                if response.status == 200:
                    report = await response.json()
                    logger.info(f"A/B test {experiment_id} stopped")
                    return report
                else:
                    logger.error(f"Stop test failed: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Stop test error: {e}")
            return None


# Singleton instance
_client_instance: Optional[ModelClient] = None


def get_model_client(server_url: str = "http://localhost:8001") -> ModelClient:
    """Получить singleton instance Model Client"""
    global _client_instance
    if _client_instance is None:
        _client_instance = ModelClient(server_url)
    return _client_instance
