"""
ML Signal Validator с улучшенной обработкой недоступности сервера.

Изменения:
1. Добавлен health check при инициализации
2. Периодическая проверка доступности ML сервера
3. Улучшенное логирование состояния fallback
4. Graceful degradation при недоступности ML
"""

import asyncio
import aiohttp
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import deque

from core.logger import get_logger
from models.signal import TradingSignal, SignalType, SignalStrength
from ml_engine.features import FeatureVector
from utils.helpers import safe_enum_value

logger = get_logger(__name__)


@dataclass
class ValidationConfig:
  """Конфигурация валидатора."""
  # Model Server
  model_server_url: str = "http://localhost:8001"
  model_version: str = "latest"
  request_timeout: float = 5.0  # seconds

  # Health Check
  health_check_enabled: bool = True
  health_check_interval: int = 30  # seconds
  health_check_timeout: float = 2.0  # seconds

  # Validation thresholds
  min_ml_confidence: float = 0.6
  confidence_boost_factor: float = 1.2
  confidence_penalty_factor: float = 0.7

  # Hybrid decision
  ml_weight: float = 0.6
  strategy_weight: float = 0.4

  # Fallback
  use_fallback_on_error: bool = True
  fallback_to_strategy: bool = True

  # Caching
  cache_predictions: bool = True
  cache_ttl_seconds: int = 30


@dataclass
class ValidationResult:
  """Результат валидации сигнала."""
  original_signal: TradingSignal
  ml_direction: Optional[str]
  ml_confidence: Optional[float]
  ml_expected_return: Optional[float]

  validated: bool
  final_signal_type: SignalType
  final_confidence: float

  agreement: bool
  reason: str

  inference_time_ms: float
  used_fallback: bool


class MLSignalValidator:
  """
  Валидатор торговых сигналов через ML модель с улучшенной отказоустойчивостью.
  """

  def __init__(self, config: ValidationConfig):
    """
    Инициализация валидатора.

    Args:
        config: Конфигурация валидатора
    """
    self.config = config

    # HTTP session для запросов
    self.session: Optional[aiohttp.ClientSession] = None

    # Кэш предсказаний
    self.prediction_cache: Dict[str, Tuple[Dict, float]] = {}

    # Статус ML сервера
    self.ml_server_available: bool = False
    self.last_health_check: Optional[float] = None
    self.health_check_task: Optional[asyncio.Task] = None

    # Статистика
    self.total_validations = 0
    self.ml_success_count = 0
    self.ml_error_count = 0
    self.fallback_count = 0
    self.agreement_count = 0
    self.disagreement_count = 0
    self.health_check_failures = 0

    # Маппинг направлений
    self.direction_to_signal_type = {
      "BUY": SignalType.BUY,
      "SELL": SignalType.SELL,
      "HOLD": None
    }

    logger.info(
      f"🤖 Инициализирован MLSignalValidator: "
      f"server={config.model_server_url}, "
      f"ml_weight={config.ml_weight}, "
      f"health_check_enabled={config.health_check_enabled}"
    )

  async def initialize(self):
    """Инициализация HTTP сессии и проверка ML сервера."""
    if self.session is None:
      timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
      self.session = aiohttp.ClientSession(timeout=timeout)
      logger.info("✓ HTTP session создана")

    # Первичная проверка доступности ML сервера
    await self._check_ml_server_health()

    # Запуск периодической проверки здоровья
    if self.config.health_check_enabled:
      self.health_check_task = asyncio.create_task(
        self._health_check_loop()
      )
      logger.info(
        f"✓ Запущен health check loop с интервалом "
        f"{self.config.health_check_interval}s"
      )

  async def close(self):
    """Закрыть HTTP сессию и остановить health check."""
    # Остановка health check loop
    if self.health_check_task:
      self.health_check_task.cancel()
      try:
        await self.health_check_task
      except asyncio.CancelledError:
        pass
      logger.info("✓ Health check loop остановлен")

    # Закрытие сессии
    if self.session:
      await self.session.close()
      self.session = None
      logger.info("✓ HTTP session закрыта")

  async def _check_ml_server_health(self) -> bool:
    """
    Проверка доступности ML сервера.

    Returns:
        bool: True если сервер доступен
    """
    try:
      health_url = f"{self.config.model_server_url}/health"
      timeout = aiohttp.ClientTimeout(total=self.config.health_check_timeout)

      async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(health_url) as response:
          if response.status == 200:
            self.ml_server_available = True
            self.last_health_check = datetime.now().timestamp()
            self.health_check_failures = 0

            if not self.ml_server_available:
              logger.info(
                f"✅ ML Server восстановлен и доступен: "
                f"{self.config.model_server_url}"
              )

            return True

    except Exception as e:
      self.ml_server_available = False
      self.health_check_failures += 1

      logger.warning(
        f"⚠️ ML Server недоступен (попытка {self.health_check_failures}): "
        f"{self.config.model_server_url} - {str(e)[:100]}"
      )

      return False

  async def _health_check_loop(self):
    """Периодическая проверка здоровья ML сервера."""
    while True:
      try:
        await asyncio.sleep(self.config.health_check_interval)
        await self._check_ml_server_health()

      except asyncio.CancelledError:
        logger.info("Health check loop cancelled")
        break
      except Exception as e:
        logger.error(f"Ошибка в health check loop: {e}")

  def _get_cache_key(self, symbol: str, feature_vector: FeatureVector) -> str:
    """Генерация ключа для кэша."""
    features_hash = hash(tuple(feature_vector.to_array()))
    return f"{symbol}_{features_hash}_{feature_vector.timestamp}"

  def _get_cached_prediction(self, cache_key: str) -> Optional[Dict]:
    """Получить предсказание из кэша."""
    if not self.config.cache_predictions:
      return None

    if cache_key in self.prediction_cache:
      prediction, cached_at = self.prediction_cache[cache_key]

      # Проверяем TTL
      age = datetime.now().timestamp() - cached_at
      if age < self.config.cache_ttl_seconds:
        logger.debug(f"✓ Cache hit: {cache_key}, age={age:.1f}s")
        return prediction
      else:
        # Удаляем устаревший кэш
        del self.prediction_cache[cache_key]

    return None

  def _cache_prediction(self, cache_key: str, prediction: Dict):
    """Сохранить предсказание в кэш."""
    if self.config.cache_predictions:
      self.prediction_cache[cache_key] = (
        prediction,
        datetime.now().timestamp()
      )

  async def _request_ml_prediction(
      self,
      symbol: str,
      feature_vector: FeatureVector
  ) -> Optional[Dict]:
    """
    Запросить предсказание от ML сервера.

    Returns:
        Dict с предсказанием или None при ошибке
    """
    # Проверка доступности сервера
    if not self.ml_server_available:
      logger.debug(
        f"⚠️ Пропуск запроса к ML Server - сервер недоступен"
      )
      return None

    try:
      if not self.session:
        raise RuntimeError("HTTP session not initialized")

      # URL для предсказания
      predict_url = f"{self.config.model_server_url}/predict"

      # Подготовка данных
      request_data = {
        "symbol": symbol,
        "features": feature_vector.to_dict(),
        "model_version": self.config.model_version
      }

      # Отправка запроса
      async with self.session.post(predict_url, json=request_data) as response:
        if response.status == 200:
          result = await response.json()
          self.ml_success_count += 1
          return result
        else:
          error_text = await response.text()
          logger.error(
            f"ML Server ответил с ошибкой {response.status}: "
            f"{error_text[:200]}"
          )
          self.ml_error_count += 1
          return None

    except asyncio.TimeoutError:
      logger.warning(
        f"⚠️ Таймаут запроса к ML Server для {symbol}"
      )
      self.ml_error_count += 1
      # Помечаем сервер как недоступный
      self.ml_server_available = False
      return None

    except Exception as e:
      logger.error(
        f"❌ Ошибка запроса к ML Server: {str(e)[:200]}"
      )
      self.ml_error_count += 1
      # Помечаем сервер как недоступный
      self.ml_server_available = False
      return None

  async def validate(
      self,
      signal: TradingSignal,
      feature_vector: FeatureVector
  ) -> ValidationResult:
    """
    Валидация торгового сигнала через ML модель.

    Args:
        signal: Торговый сигнал от стратегии
        feature_vector: Вектор признаков для ML

    Returns:
        ValidationResult с финальным решением
    """
    self.total_validations += 1
    start_time = datetime.now()

    # Проверяем кэш
    cache_key = self._get_cache_key(signal.symbol, feature_vector)
    cached_prediction = self._get_cached_prediction(cache_key)

    if cached_prediction:
      ml_prediction = cached_prediction
    else:
      # Запрос к ML серверу
      ml_prediction = await self._request_ml_prediction(
        signal.symbol,
        feature_vector
      )

      # Кэшируем результат
      if ml_prediction:
        self._cache_prediction(cache_key, ml_prediction)

    inference_time = (datetime.now() - start_time).total_seconds() * 1000

    # Fallback при недоступности ML
    if ml_prediction is None:
      self.fallback_count += 1

      if self.config.fallback_to_strategy:
        logger.info(
          f"✓ Сигнал ВАЛИДИРОВАН {signal.symbol}: {safe_enum_value(signal.signal_type)}, "
          f"confidence={signal.confidence:.4f}, "
          f"agreement=False, "
          f"reason=ML unavailable, using strategy signal (fallback)"
        )

        return ValidationResult(
          original_signal=signal,
          ml_direction=None,
          ml_confidence=None,
          ml_expected_return=None,
          validated=True,
          final_signal_type=signal.signal_type,
          final_confidence=signal.confidence * self.config.strategy_weight,
          agreement=False,
          reason="ML unavailable, using strategy signal (fallback)",
          inference_time_ms=inference_time,
          used_fallback=True
        )
      else:
        logger.warning(
          f"✗ Сигнал ОТКЛОНЕН {signal.symbol}: ML недоступен, "
          f"fallback отключен"
        )

        return ValidationResult(
          original_signal=signal,
          ml_direction=None,
          ml_confidence=None,
          ml_expected_return=None,
          validated=False,
          final_signal_type=signal.signal_type,
          final_confidence=0.0,
          agreement=False,
          reason="ML unavailable, fallback disabled",
          inference_time_ms=inference_time,
          used_fallback=False
        )

    # Извлекаем данные из предсказания
    ml_direction = ml_prediction.get("direction")
    ml_confidence = ml_prediction.get("confidence", 0.0)
    ml_expected_return = ml_prediction.get("expected_return", 0.0)

    # Проверка минимального confidence
    if ml_confidence < self.config.min_ml_confidence:
      logger.info(
        f"✗ ML confidence слишком низкий для {signal.symbol}: "
        f"{ml_confidence:.4f} < {self.config.min_ml_confidence}"
      )

      return ValidationResult(
        original_signal=signal,
        ml_direction=ml_direction,
        ml_confidence=ml_confidence,
        ml_expected_return=ml_expected_return,
        validated=False,
        final_signal_type=signal.signal_type,
        final_confidence=0.0,
        agreement=False,
        reason=f"ML confidence too low: {ml_confidence:.4f}",
        inference_time_ms=inference_time,
        used_fallback=False
      )

    # Конвертируем ML direction в SignalType
    ml_signal_type = self.direction_to_signal_type.get(ml_direction)

    # Проверяем согласие между ML и стратегией
    agreement = (ml_signal_type == signal.signal_type)

    if agreement:
      self.agreement_count += 1

      # ML и стратегия согласны - усиливаем уверенность
      final_confidence = min(
        signal.confidence * self.config.confidence_boost_factor,
        1.0
      )

      logger.info(
        f"✓ Сигнал ВАЛИДИРОВАН {signal.symbol}: {signal.signal_type.value}, "
        f"confidence={final_confidence:.4f}, "
        f"agreement=True, "
        f"reason=ML and strategy agree"
      )

      return ValidationResult(
        original_signal=signal,
        ml_direction=ml_direction,
        ml_confidence=ml_confidence,
        ml_expected_return=ml_expected_return,
        validated=True,
        final_signal_type=signal.signal_type,
        final_confidence=final_confidence,
        agreement=True,
        reason="ML and strategy agree, signal boosted",
        inference_time_ms=inference_time,
        used_fallback=False
      )
    else:
      self.disagreement_count += 1

      # ML и стратегия не согласны - взвешенное решение

      # Если ML говорит HOLD - отклоняем сигнал
      if ml_direction == "HOLD":
        logger.info(
          f"✗ Сигнал ОТКЛОНЕН {signal.symbol}: ML предлагает HOLD"
        )

        return ValidationResult(
          original_signal=signal,
          ml_direction=ml_direction,
          ml_confidence=ml_confidence,
          ml_expected_return=ml_expected_return,
          validated=False,
          final_signal_type=signal.signal_type,
          final_confidence=0.0,
          agreement=False,
          reason="ML suggests HOLD, rejecting signal",
          inference_time_ms=inference_time,
          used_fallback=False
        )

      # ML предлагает противоположное направление - штрафуем confidence
      final_confidence = signal.confidence * self.config.confidence_penalty_factor

      logger.warning(
        f"⚠️ Сигнал ОСЛАБЛЕН {signal.symbol}: "
        f"ML={ml_direction}, Strategy={signal.signal_type.value}, "
        f"confidence={final_confidence:.4f}"
      )

      return ValidationResult(
        original_signal=signal,
        ml_direction=ml_direction,
        ml_confidence=ml_confidence,
        ml_expected_return=ml_expected_return,
        validated=True,
        final_signal_type=signal.signal_type,
        final_confidence=final_confidence,
        agreement=False,
        reason="ML disagrees, confidence penalized",
        inference_time_ms=inference_time,
        used_fallback=False
      )

  def get_statistics(self) -> Dict:
    """Получить статистику валидатора."""
    return {
      "total_validations": self.total_validations,
      "ml_success_count": self.ml_success_count,
      "ml_error_count": self.ml_error_count,
      "fallback_count": self.fallback_count,
      "agreement_count": self.agreement_count,
      "disagreement_count": self.disagreement_count,
      "ml_server_available": self.ml_server_available,
      "health_check_failures": self.health_check_failures,
      "last_health_check": self.last_health_check,
      "success_rate": (
        self.ml_success_count / self.total_validations * 100
        if self.total_validations > 0 else 0.0
      ),
      "agreement_rate": (
        self.agreement_count / self.total_validations * 100
        if self.total_validations > 0 else 0.0
      ),
      "fallback_rate": (
        self.fallback_count / self.total_validations * 100
        if self.total_validations > 0 else 0.0
      )
    }