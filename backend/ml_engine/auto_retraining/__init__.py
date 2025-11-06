"""
Auto-Retraining Pipeline Module

Автоматическое переобучение моделей:
- Scheduled retraining (cron-like)
- Drift-triggered retraining
- Walk-forward validation
- Auto model promotion
"""

from backend.ml_engine.auto_retraining.retraining_pipeline import (
    RetrainingPipeline,
    get_retraining_pipeline,
    RetrainingConfig,
    RetrainingTrigger
)

__all__ = [
    "RetrainingPipeline",
    "get_retraining_pipeline",
    "RetrainingConfig",
    "RetrainingTrigger"
]
