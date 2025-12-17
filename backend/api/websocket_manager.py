"""
Shared WebSocket Manager для real-time обновлений.

Используется несколькими API модулями:
- ensemble_api.py (обучение моделей, предсказания)
- hyperopt_api.py (оптимизация гиперпараметров)

Путь: backend/api/websocket_manager.py
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from fastapi import WebSocket
import json

from backend.core.logger import get_logger

logger = get_logger(__name__)


class WebSocketManager:
    """
    Unified WebSocket manager для real-time обновлений.

    Поддерживает события:
    - training: Обновления обучения моделей
    - predictions: Предсказания ensemble
    - status: Изменения статуса моделей
    - hyperopt: Прогресс оптимизации гиперпараметров
    - all: Все события
    """

    def __init__(self):
        # Все активные подключения
        self.active_connections: List[WebSocket] = []
        # Подключения по типу подписки
        self.subscriptions: Dict[str, Set[WebSocket]] = {
            "training": set(),      # Обновления обучения
            "predictions": set(),   # Предсказания ensemble
            "status": set(),        # Изменения статуса моделей
            "hyperopt": set(),      # Оптимизация гиперпараметров
            "all": set()            # Все события
        }

    async def connect(self, websocket: WebSocket, subscription: str = "all"):
        """Подключить нового клиента."""
        await websocket.accept()
        self.active_connections.append(websocket)
        if subscription in self.subscriptions:
            self.subscriptions[subscription].add(websocket)
        self.subscriptions["all"].add(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Отключить клиента."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        for subscription_set in self.subscriptions.values():
            subscription_set.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Отправить сообщение конкретному клиенту."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any], event_type: str = "all"):
        """
        Отправить сообщение всем подписанным клиентам.

        Args:
            message: Сообщение для отправки
            event_type: Тип события (training, predictions, status, hyperopt, all)
        """
        # Определяем получателей
        recipients: Set[WebSocket] = set()
        if event_type in self.subscriptions:
            recipients = self.subscriptions[event_type].copy()
        recipients.update(self.subscriptions["all"])

        if not recipients:
            return  # Нет подписчиков

        # Добавляем метаданные
        message_with_meta = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **message
        }

        # Отправляем всем получателям
        disconnected = []
        for websocket in recipients:
            try:
                await websocket.send_json(message_with_meta)
            except Exception as e:
                logger.warning(f"Failed to broadcast to client: {e}")
                disconnected.append(websocket)

        # Удаляем отключенных клиентов
        for ws in disconnected:
            self.disconnect(ws)

    # ============================================================================
    # TRAINING BROADCASTS
    # ============================================================================

    async def broadcast_training_progress(
        self,
        task_id: str,
        model_type: str,
        epoch: int,
        total_epochs: int,
        metrics: Dict[str, float],
        status: str = "training"
    ):
        """
        Broadcast прогресса обучения модели.

        Args:
            task_id: ID задачи обучения
            model_type: Тип модели
            epoch: Текущая эпоха
            total_epochs: Всего эпох
            metrics: Метрики обучения
            status: Статус (started, training, completed, failed)
        """
        await self.broadcast(
            {
                "type": "training_progress",
                "task_id": task_id,
                "model_type": model_type,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "progress_pct": round((epoch / total_epochs) * 100, 1) if total_epochs > 0 else 0,
                "metrics": metrics,
                "status": status
            },
            event_type="training"
        )

    # ============================================================================
    # PREDICTION BROADCASTS
    # ============================================================================

    async def broadcast_prediction(
        self,
        direction: str,
        confidence: float,
        model_predictions: Dict[str, Any],
        should_trade: bool
    ):
        """
        Broadcast нового предсказания ensemble.

        Args:
            direction: Направление (BUY, SELL, HOLD)
            confidence: Уверенность consensus
            model_predictions: Предсказания отдельных моделей
            should_trade: Рекомендация к сделке
        """
        await self.broadcast(
            {
                "type": "prediction",
                "direction": direction,
                "confidence": confidence,
                "model_predictions": model_predictions,
                "should_trade": should_trade
            },
            event_type="predictions"
        )

    # ============================================================================
    # STATUS BROADCASTS
    # ============================================================================

    async def broadcast_status_change(
        self,
        model_type: str,
        change_type: str,
        old_value: Any,
        new_value: Any
    ):
        """
        Broadcast изменения статуса модели.

        Args:
            model_type: Тип модели
            change_type: Тип изменения (enabled, weight, performance)
            old_value: Старое значение
            new_value: Новое значение
        """
        await self.broadcast(
            {
                "type": "status_change",
                "model_type": model_type,
                "change_type": change_type,
                "old_value": old_value,
                "new_value": new_value
            },
            event_type="status"
        )

    # ============================================================================
    # HYPEROPT BROADCASTS
    # ============================================================================

    async def broadcast_hyperopt_started(
        self,
        job_id: str,
        mode: str,
        total_trials_estimate: int,
        time_estimate: Dict[str, Any]
    ):
        """
        Broadcast начала оптимизации гиперпараметров.

        Args:
            job_id: ID задачи оптимизации
            mode: Режим оптимизации (full, quick, group, fine_tune)
            total_trials_estimate: Оценка количества trials
            time_estimate: Оценка времени
        """
        await self.broadcast(
            {
                "type": "hyperopt_started",
                "job_id": job_id,
                "mode": mode,
                "total_trials_estimate": total_trials_estimate,
                "time_estimate": time_estimate
            },
            event_type="hyperopt"
        )

    async def broadcast_hyperopt_progress(
        self,
        job_id: str,
        current_trial: int,
        total_trials: int,
        current_group: str,
        best_value: float,
        current_value: float,
        elapsed_time: str
    ):
        """
        Broadcast прогресса оптимизации.

        Args:
            job_id: ID задачи
            current_trial: Текущий trial
            total_trials: Всего trials
            current_group: Текущая группа параметров
            best_value: Лучшее значение метрики
            current_value: Текущее значение метрики
            elapsed_time: Прошедшее время
        """
        await self.broadcast(
            {
                "type": "hyperopt_progress",
                "job_id": job_id,
                "current_trial": current_trial,
                "total_trials": total_trials,
                "progress_pct": round((current_trial / total_trials) * 100, 1) if total_trials > 0 else 0,
                "current_group": current_group,
                "best_value": best_value,
                "current_value": current_value,
                "elapsed_time": elapsed_time
            },
            event_type="hyperopt"
        )

    async def broadcast_hyperopt_completed(
        self,
        job_id: str,
        best_params: Dict[str, Any],
        best_value: float,
        total_trials: int,
        elapsed_time: str
    ):
        """
        Broadcast завершения оптимизации.

        Args:
            job_id: ID задачи
            best_params: Лучшие найденные параметры
            best_value: Лучшее значение метрики
            total_trials: Общее количество trials
            elapsed_time: Общее время
        """
        await self.broadcast(
            {
                "type": "hyperopt_completed",
                "job_id": job_id,
                "best_params": best_params,
                "best_value": best_value,
                "total_trials": total_trials,
                "elapsed_time": elapsed_time
            },
            event_type="hyperopt"
        )

    async def broadcast_hyperopt_failed(
        self,
        job_id: str,
        error: str,
        error_type: str
    ):
        """
        Broadcast ошибки оптимизации.

        Args:
            job_id: ID задачи
            error: Сообщение об ошибке
            error_type: Тип ошибки
        """
        await self.broadcast(
            {
                "type": "hyperopt_failed",
                "job_id": job_id,
                "error": error,
                "error_type": error_type
            },
            event_type="hyperopt"
        )


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Глобальный экземпляр менеджера
_ws_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Получить глобальный экземпляр WebSocket manager."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager


def set_websocket_manager(manager: WebSocketManager):
    """Установить глобальный экземпляр WebSocket manager."""
    global _ws_manager
    _ws_manager = manager
