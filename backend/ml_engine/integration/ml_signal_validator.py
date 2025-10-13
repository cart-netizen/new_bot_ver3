"""
ML Signal Validator для валидации торговых сигналов через ML модель.

Функциональность:
- Валидация сигналов от стратегий через ML
- Асинхронные запросы к Model Server
- Кэширование предсказаний
- Fallback стратегия при недоступности ML
- Гибридное принятие решений (ML + Rules)

Путь: backend/ml_engine/integration/ml_signal_validator.py
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

logger = get_logger(__name__)


@dataclass
class ValidationConfig:
  """Конфигурация валидатора."""
  # Model Server
  model_server_url: str = "http://localhost:8001"
  model_version: str = "latest"
  request_timeout: float = 5.0  # seconds

  # Validation thresholds
  min_ml_confidence: float = 0.6  # Минимальная уверенность ML
  confidence_boost_factor: float = 1.2  # Усиление при согласии
  confidence_penalty_factor: float = 0.7  # Штраф при несогласии

  # Hybrid decision
  ml_weight: float = 0.6  # Вес ML в решении (0-1)
  strategy_weight: float = 0.4  # Вес стратегии в решении

  # Fallback
  use_fallback_on_error: bool = True
  fallback_to_strategy: bool = True  # Использовать сигнал стратегии при ошибке ML

  # Caching
  cache_predictions: bool = True
  cache_ttl_seconds: int = 30


@dataclass
class ValidationResult:
  """Результат валидации сигнала."""
  original_signal: TradingSignal
  ml_direction: Optional[str]  # "BUY", "HOLD", "SELL"
  ml_confidence: Optional[float]
  ml_expected_return: Optional[float]

  validated: bool  # Сигнал прошел валидацию
  final_signal_type: SignalType
  final_confidence: float

  agreement: bool  # ML и стратегия согласны
  reason: str  # Причина решения

  inference_time_ms: float
  used_fallback: bool


class MLSignalValidator:
  """
  Валидатор торговых сигналов через ML модель.

  Логика валидации:
  1. Получить сигнал от стратегии
  2. Запросить предсказание ML модели
  3. Сравнить сигнал и предсказание
  4. Принять решение на основе гибридной логики
  5. Вернуть валидированный сигнал
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

    # Статистика
    self.total_validations = 0
    self.ml_success_count = 0
    self.ml_error_count = 0
    self.fallback_count = 0
    self.agreement_count = 0
    self.disagreement_count = 0

    # Маппинг направлений
    self.direction_to_signal_type = {
      "BUY": SignalType.BUY,
      "SELL": SignalType.SELL,
      "HOLD": None  # HOLD = нет сигнала
    }

    logger.info(
      f"Инициализирован MLSignalValidator: "
      f"server={config.model_server_url}, "
      f"ml_weight={config.ml_weight}"
    )

  async def initialize(self):
    """Инициализация HTTP сессии."""
    if self.session is None:
      timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
      self.session = aiohttp.ClientSession(timeout=timeout)
      logger.info("HTTP session создана")

  async def close(self):
    """Закрыть HTTP сессию."""
    if self.session:
      await self.session.close()
      self.session = None
      logger.info("HTTP session закрыта")

  def _get_cache_key(self, symbol: str, feature_vector: FeatureVector) -> str:
    """Генерация ключа для кэша."""
    # Используем hash признаков и timestamp
    features_hash = hash(tuple(feature_vector.to_array()))
    return f"{symbol}_{features_hash}_{feature_vector.timestamp}"

  def _get_cached_prediction(
      self,
      cache_key: str
  ) -> Optional[Dict]:
    """Получить предсказание из кэша."""
    if not self.config.cache_predictions:
      return None

    if cache_key in self.prediction_cache:
      prediction, cached_at = self.prediction_cache[cache_key]

      # Проверяем TTL
      age = datetime.now().timestamp() - cached_at
      if age < self.config.cache_ttl_seconds:
        logger.debug(f"Cache hit: {cache_key}, age={age:.1f}s")
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

    Args:
        symbol: Торговая пара
        feature_vector: Вектор признаков

    Returns:
        Dict с предсказанием или None при ошибке
    """
    if self.session is None:
      await self.initialize()

    try:
      # Проверяем кэш
      cache_key = self._get_cache_key(symbol, feature_vector)
      cached = self._get_cached_prediction(cache_key)
      if cached:
        return cached

      # Создаем последовательность из feature_vector
      # Для упрощения используем один вектор, но в production
      # нужно собирать последовательность из истории
      features_array = feature_vector.to_array()

      # Создаем dummy последовательность (повторяем текущий вектор)
      # TODO: В production использовать реальную историю
      sequence = [features_array.tolist()] * 60  # 60 timesteps

      # Формируем запрос
      request_data = {
        "symbol": symbol,
        "sequence": sequence,
        "model_version": self.config.model_version,
        "return_probabilities": True
      }

      # Отправляем запрос
      url = f"{self.config.model_server_url}/predict"

      async with self.session.post(url, json=request_data) as response:
        if response.status == 200:
          prediction = await response.json()

          # Кэшируем результат
          self._cache_prediction(cache_key, prediction)

          self.ml_success_count += 1

          logger.debug(
            f"ML prediction для {symbol}: "
            f"direction={prediction['direction']}, "
            f"confidence={prediction['confidence']:.4f}"
          )

          return prediction
        else:
          error_text = await response.text()
          logger.error(
            f"ML Server ошибка: status={response.status}, "
            f"error={error_text}"
          )
          self.ml_error_count += 1
          return None

    except asyncio.TimeoutError:
      logger.error(f"ML Server timeout для {symbol}")
      self.ml_error_count += 1
      return None

    except Exception as e:
      logger.error(f"Ошибка запроса к ML Server: {e}")
      self.ml_error_count += 1
      return None

  def _make_hybrid_decision(
      self,
      signal: TradingSignal,
      ml_prediction: Optional[Dict]
  ) -> ValidationResult:
    """
    Принять решение на основе гибридной логики.

    Args:
        signal: Исходный сигнал от стратегии
        ml_prediction: Предсказание ML или None

    Returns:
        Результат валидации
    """
    self.total_validations += 1

    # Если ML недоступна и включен fallback
    if ml_prediction is None:
      if self.config.use_fallback_on_error and self.config.fallback_to_strategy:
        self.fallback_count += 1

        return ValidationResult(
          original_signal=signal,
          ml_direction=None,
          ml_confidence=None,
          ml_expected_return=None,
          validated=True,
          final_signal_type=signal.signal_type,
          final_confidence=signal.confidence,
          agreement=False,
          reason="ML unavailable, using strategy signal (fallback)",
          inference_time_ms=0.0,
          used_fallback=True
        )
      else:
        # Отклоняем сигнал
        return ValidationResult(
          original_signal=signal,
          ml_direction=None,
          ml_confidence=None,
          ml_expected_return=None,
          validated=False,
          final_signal_type=signal.signal_type,
          final_confidence=0.0,
          agreement=False,
          reason="ML unavailable, signal rejected",
          inference_time_ms=0.0,
          used_fallback=False
        )

    # Извлекаем данные ML
    ml_direction = ml_prediction['direction']
    ml_confidence = ml_prediction['confidence']
    ml_expected_return = ml_prediction['expected_return']
    inference_time = ml_prediction['inference_time_ms']

    # Проверяем минимальную уверенность ML
    if ml_confidence < self.config.min_ml_confidence:
      return ValidationResult(
        original_signal=signal,
        ml_direction=ml_direction,
        ml_confidence=ml_confidence,
        ml_expected_return=ml_expected_return,
        validated=False,
        final_signal_type=signal.signal_type,
        final_confidence=0.0,
        agreement=False,
        reason=f"ML confidence too low: {ml_confidence:.4f} < {self.config.min_ml_confidence}",
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
        return ValidationResult(
          original_signal=signal,
          ml_direction=ml_direction,
          ml_confidence=ml_confidence,
          ml_expected_return=ml_expected_return,
          validated=False,
          final_signal_type=signal.signal_type,
          final_confidence=0.0,
          agreement=False,
          reason="ML suggests HOLD, signal rejected",
          inference_time_ms=inference_time,
          used_fallback=False
        )

      # Взвешенное решение: приоритет у ML
      if self.config.ml_weight >= self.config.strategy_weight:
        # Следуем ML
        final_signal_type = ml_signal_type
        final_confidence = ml_confidence * self.config.ml_weight
        reason = f"ML disagrees (ML: {ml_direction}, Strategy: {signal.signal_type.value}), following ML"
      else:
        # Следуем стратегии, но со штрафом
        final_signal_type = signal.signal_type
        final_confidence = signal.confidence * self.config.confidence_penalty_factor
        reason = f"ML disagrees (ML: {ml_direction}, Strategy: {signal.signal_type.value}), following strategy with penalty"

      # Проверяем финальную уверенность
      validated = final_confidence >= self.config.min_ml_confidence

      return ValidationResult(
        original_signal=signal,
        ml_direction=ml_direction,
        ml_confidence=ml_confidence,
        ml_expected_return=ml_expected_return,
        validated=validated,
        final_signal_type=final_signal_type,
        final_confidence=final_confidence if validated else 0.0,
        agreement=False,
        reason=reason,
        inference_time_ms=inference_time,
        used_fallback=False
      )

  async def validate_signal(
      self,
      signal: TradingSignal,
      feature_vector: FeatureVector
  ) -> ValidationResult:
    """
    Валидировать торговый сигнал через ML.

    Args:
        signal: Сигнал от стратегии
        feature_vector: Вектор признаков

    Returns:
        Результат валидации
    """
    logger.debug(
      f"Валидация сигнала {signal.symbol}: "
      f"type={signal.signal_type.value}, "
      f"confidence={signal.confidence:.4f}"
    )

    # Запрашиваем предсказание ML
    ml_prediction = await self._request_ml_prediction(
      signal.symbol,
      feature_vector
    )

    # Принимаем решение
    result = self._make_hybrid_decision(signal, ml_prediction)

    # Логирование результата
    if result.validated:
      logger.info(
        f"✓ Сигнал ВАЛИДИРОВАН {signal.symbol}: "
        f"{result.final_signal_type.value}, "
        f"confidence={result.final_confidence:.4f}, "
        f"agreement={result.agreement}, "
        f"reason={result.reason}"
      )
    else:
      logger.info(
        f"✗ Сигнал ОТКЛОНЕН {signal.symbol}: "
        f"reason={result.reason}"
      )

    return result

  def get_statistics(self) -> Dict:
    """Получить статистику валидатора."""
    success_rate = (
      self.ml_success_count / (self.ml_success_count + self.ml_error_count)
      if (self.ml_success_count + self.ml_error_count) > 0
      else 0.0
    )

    agreement_rate = (
      self.agreement_count / self.total_validations
      if self.total_validations > 0
      else 0.0
    )

    fallback_rate = (
      self.fallback_count / self.total_validations
      if self.total_validations > 0
      else 0.0
    )

    return {
      'total_validations': self.total_validations,
      'ml_success_count': self.ml_success_count,
      'ml_error_count': self.ml_error_count,
      'ml_success_rate': success_rate,
      'agreement_count': self.agreement_count,
      'disagreement_count': self.disagreement_count,
      'agreement_rate': agreement_rate,
      'fallback_count': self.fallback_count,
      'fallback_rate': fallback_rate,
      'cache_size': len(self.prediction_cache)
    }


# Пример использования
if __name__ == "__main__":
  from models.signal import TradingSignal, SignalType, SignalSource, SignalStrength
  from ml_engine.features import FeatureVector


  async def main():
    # Создаем валидатор
    config = ValidationConfig(
      model_server_url="http://localhost:8001",
      min_ml_confidence=0.6,
      ml_weight=0.6
    )

    validator = MLSignalValidator(config)
    await validator.initialize()

    # Создаем тестовый сигнал
    signal = TradingSignal(
      symbol="BTCUSDT",
      signal_type=SignalType.BUY,
      confidence=0.8,
      signal_source=SignalSource.STRATEGY,
      signal_strength=SignalStrength.STRONG,
      price=50000.0,
      reason="Test signal"
    )

    # Создаем тестовый feature vector
    feature_vector = FeatureVector(
      symbol="BTCUSDT",
      timestamp=int(datetime.now().timestamp() * 1000),
      features={}
    )

    # Заполняем dummy признаками
    import numpy as np
    dummy_features = np.random.randn(110)
    for i, value in enumerate(dummy_features):
      feature_vector.features[f"feature_{i}"] = value

    # Валидируем сигнал
    result = await validator.validate_signal(signal, feature_vector)

    print(f"Validation result:")
    print(f"  Validated: {result.validated}")
    print(f"  Final signal: {result.final_signal_type}")
    print(f"  Final confidence: {result.final_confidence:.4f}")
    print(f"  Agreement: {result.agreement}")
    print(f"  Reason: {result.reason}")

    # Статистика
    stats = validator.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total validations: {stats['total_validations']}")
    print(f"  ML success rate: {stats['ml_success_rate']:.2%}")
    print(f"  Agreement rate: {stats['agreement_rate']:.2%}")

    await validator.close()


  asyncio.run(main())