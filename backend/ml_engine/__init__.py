"""
ML Engine модуль для машинного обучения в торговом боте.
"""

__version__ = "0.1.0"

from backend.core.logger import get_logger

logger = get_logger(__name__)
logger.info(f"ML Engine модуль загружен (версия {__version__})")