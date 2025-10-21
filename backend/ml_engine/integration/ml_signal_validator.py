"""
ML Signal Validator с расширенным ValidationResult.

Изменения:
1. ValidationResult расширен полями: predicted_mae, manipulation_risk, market_regime, feature_quality
2. Улучшена обработка feature_vector (поддержка Dict и FeatureVector)
3. Добавлены методы расчета дополнительных метрик
"""

import asyncio
import aiohttp
from typing import Dict, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import deque

from core.logger import get_logger
from models.signal import TradingSignal, SignalType, SignalStrength
from ml_engine.features import FeatureVector
from strategy.risk_models import MarketRegime
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

  # Advanced metrics
  enable_mae_prediction: bool = True
  enable_manipulation_detection: bool = True
  enable_regime_detection: bool = True
  enable_feature_quality_check: bool = True


@dataclass
class ValidationResult:
  """
  Расширенный результат валидации сигнала.

  Добавлены поля:
  - predicted_mae: Предсказанная средняя абсолютная ошибка
  - manipulation_risk: Риск манипуляции (0.0-1.0)
  - market_regime: Определенный режим рынка
  - feature_quality: Качество признаков (0.0-1.0)
  """
  # Основные поля
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

  # ========================================
  # НОВЫЕ РАСШИРЕННЫЕ ПОЛЯ
  # ========================================
  predicted_mae: Optional[float] = None  # Mean Absolute Error предсказания
  manipulation_risk: float = 0.0  # Риск манипуляции [0.0, 1.0]
  market_regime: Optional[MarketRegime] = None  # Режим рынка
  feature_quality: float = 1.0  # Качество признаков [0.0, 1.0]

  # Дополнительные метаданные
  metadata: Dict = field(default_factory=dict)


class MLSignalValidator:
  """
  Валидатор торговых сигналов с расширенной ML аналитикой.

  Новые возможности:
  - Предсказание MAE для оценки точности
  - Детекция манипуляций стакана
  - Определение режима рынка
  - Оценка качества признаков
  """

  def __init__(self, config: ValidationConfig):
    """
    Инициализация валидатора.

    Args:
        config: Конфигурация валидатора
    """
    self.config = config

    # HTTP session для запросов к ML серверу
    self.session: Optional[aiohttp.ClientSession] = None

    # Статистика
    self.total_validations = 0
    self.ml_success_count = 0
    self.ml_error_count = 0
    self.fallback_count = 0
    self.agreement_count = 0
    self.disagreement_count = 0

    # Health check
    self.ml_server_available = False
    self.health_check_failures = 0
    self.last_health_check: Optional[datetime] = None
    self._health_check_task: Optional[asyncio.Task] = None

    # Prediction cache
    self._prediction_cache: deque = deque(maxlen=100)
    self._cache_dict: Dict[str, Dict] = {}

    # Историческая статистика для расчета метрик
    self._mae_history: deque = deque(maxlen=1000)  # История ошибок для MAE
    self._manipulation_scores: deque = deque(maxlen=100)  # История детекции

    logger.info(
      f"MLSignalValidator инициализирован: "
      f"server={config.model_server_url}, "
      f"min_confidence={config.min_ml_confidence:.2f}, "
      f"advanced_metrics={config.enable_mae_prediction}"
    )

  async def initialize(self):
    """Инициализация HTTP сессии и health check."""
    self.session = aiohttp.ClientSession()

    # Проверяем доступность ML сервера
    if self.config.health_check_enabled:
      await self._perform_health_check()

      # Запускаем периодический health check
      self._health_check_task = asyncio.create_task(
        self._periodic_health_check()
      )

    logger.info("MLSignalValidator initialized successfully")

  async def cleanup(self):
    """Очистка ресурсов."""
    if self._health_check_task:
      self._health_check_task.cancel()
      try:
        await self._health_check_task
      except asyncio.CancelledError:
        pass

    if self.session:
      await self.session.close()

    logger.info("MLSignalValidator cleaned up")

  async def validate(
      self,
      signal: TradingSignal,
      feature_vector: Union[FeatureVector, Dict]
  ) -> ValidationResult:
    """
    Валидация торгового сигнала с расширенной ML аналитикой.

    Args:
        signal: Торговый сигнал от стратегии
        feature_vector: Вектор признаков (FeatureVector или Dict)

    Returns:
        ValidationResult с полной аналитикой
    """
    self.total_validations += 1
    start_time = datetime.now()

    # ========================================
    # КОНВЕРТАЦИЯ FEATURE_VECTOR
    # ========================================
    # Поддерживаем оба формата: Dict и FeatureVector
    if isinstance(feature_vector, dict):
      logger.debug(
        f"{signal.symbol} | Feature vector передан как Dict, "
        f"будем использовать напрямую"
      )
      feature_dict = feature_vector
    elif isinstance(feature_vector, FeatureVector):
      logger.debug(
        f"{signal.symbol} | Feature vector передан как FeatureVector, "
        f"конвертируем в Dict"
      )
      feature_dict = feature_vector.to_dict()
    else:
      logger.error(
        f"{signal.symbol} | Неподдерживаемый тип feature_vector: "
        f"{type(feature_vector)}"
      )
      return self._create_fallback_result(
        signal,
        "Invalid feature_vector type",
        start_time
      )

    # ========================================
    # РАСЧЕТ ДОПОЛНИТЕЛЬНЫХ МЕТРИК
    # ========================================
    # Рассчитываем метрики ДО запроса к ML серверу
    feature_quality = self._calculate_feature_quality(feature_dict)
    manipulation_risk = self._detect_manipulation_risk(feature_dict)
    market_regime = self._detect_market_regime(feature_dict)

    logger.debug(
      f"{signal.symbol} | Дополнительные метрики: "
      f"quality={feature_quality:.3f}, "
      f"manipulation={manipulation_risk:.3f}, "
      f"regime={market_regime.value if market_regime else 'None'}"
    )

    # ========================================
    # ПОЛУЧЕНИЕ ML ПРЕДСКАЗАНИЯ
    # ========================================
    # Проверяем кэш
    cache_key = self._get_cache_key(signal.symbol, feature_dict)
    cached_prediction = self._get_cached_prediction(cache_key)

    if cached_prediction:
      ml_prediction = cached_prediction
      logger.debug(f"{signal.symbol} | ML prediction из кэша")
    else:
      # Запрос к ML серверу
      ml_prediction = await self._request_ml_prediction(
        signal.symbol,
        feature_dict
      )

      # Кэшируем результат
      if ml_prediction:
        self._cache_prediction(cache_key, ml_prediction)
        logger.debug(f"{signal.symbol} | ML prediction закэширован")

    inference_time = (datetime.now() - start_time).total_seconds() * 1000

    # ========================================
    # FALLBACK ПРИ НЕДОСТУПНОСТИ ML
    # ========================================
    if ml_prediction is None:
      self.fallback_count += 1

      if self.config.fallback_to_strategy:
        logger.info(
          f"✓ Сигнал ВАЛИДИРОВАН {signal.symbol}: "
          f"{safe_enum_value(signal.signal_type)}, "
          f"confidence={signal.confidence:.4f}, "
          f"reason=ML unavailable (fallback)"
        )

        return ValidationResult(
          original_signal=signal,
          ml_direction=None,
          ml_confidence=signal.confidence,  # ✅ ИСПРАВЛЕНО: Передаем оригинальный confidence
          ml_expected_return=None,
          validated=True,
          final_signal_type=signal.signal_type,
          final_confidence=signal.confidence,  # ✅ ИСПРАВЛЕНО: Без штрафа
          agreement=False,
          reason="ML unavailable, using strategy signal (fallback)",
          inference_time_ms=inference_time,
          used_fallback=True,
          # Используем рассчитанные метрики даже при fallback
          predicted_mae=self._estimate_mae_from_history(),
          manipulation_risk=manipulation_risk,
          market_regime=market_regime,
          feature_quality=feature_quality
        )
      else:
        logger.warning(
          f"✗ Сигнал ОТКЛОНЕН {signal.symbol}: ML недоступен"
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
          used_fallback=True,
          predicted_mae=None,
          manipulation_risk=manipulation_risk,
          market_regime=market_regime,
          feature_quality=feature_quality
        )

    # ========================================
    # ОБРАБОТКА ML ПРЕДСКАЗАНИЯ
    # ========================================
    self.ml_success_count += 1

    ml_direction = ml_prediction.get("direction", "HOLD")
    ml_confidence = ml_prediction.get("confidence", 0.0)
    ml_expected_return = ml_prediction.get("expected_return", 0.0)

    # Извлекаем расширенные метрики из ML ответа
    predicted_mae = ml_prediction.get("predicted_mae")

    # Если ML не предоставил метрики, используем рассчитанные локально
    if predicted_mae is None and self.config.enable_mae_prediction:
      predicted_mae = self._estimate_mae_from_history()

    # Проверяем соответствие направлений
    ml_signal_type = self._convert_direction_to_signal_type(ml_direction)
    agreement = (ml_signal_type == signal.signal_type)

    # ========================================
    # ФОРМИРОВАНИЕ ФИНАЛЬНОГО РЕЗУЛЬТАТА
    # ========================================
    if agreement:
      self.agreement_count += 1

      # ML и стратегия согласны - усиливаем сигнал
      final_confidence = min(
        signal.confidence * self.config.confidence_boost_factor,
        1.0
      )

      logger.info(
        f"✓ Сигнал ВАЛИДИРОВАН {signal.symbol}: "
        f"{safe_enum_value(signal.signal_type)}, "
        f"confidence={final_confidence:.4f}, "
        f"agreement=True, mae={predicted_mae:.4f if predicted_mae else 'N/A'}"
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
        used_fallback=False,
        predicted_mae=predicted_mae,
        manipulation_risk=manipulation_risk,
        market_regime=market_regime,
        feature_quality=feature_quality
      )
    else:
      self.disagreement_count += 1

      # ML и стратегия не согласны
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
          used_fallback=False,
          predicted_mae=predicted_mae,
          manipulation_risk=manipulation_risk,
          market_regime=market_regime,
          feature_quality=feature_quality
        )

      # ML предлагает противоположное направление - штрафуем
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
        used_fallback=False,
        predicted_mae=predicted_mae,
        manipulation_risk=manipulation_risk,
        market_regime=market_regime,
        feature_quality=feature_quality
      )

  # ========================================
  # МЕТОДЫ РАСЧЕТА ДОПОЛНИТЕЛЬНЫХ МЕТРИК
  # ========================================

  def _calculate_feature_quality(self, feature_dict: Dict) -> float:
    """
    Оценка качества признаков.

    Проверяет:
    - Наличие всех необходимых признаков
    - Отсутствие NaN/Inf значений
    - Разумные диапазоны значений

    Returns:
        float: Качество признаков [0.0, 1.0]
    """
    if not self.config.enable_feature_quality_check:
      return 1.0

    try:
      quality_score = 1.0

      # 1. Проверка наличия ключевых групп признаков
      required_groups = ['orderbook', 'candle', 'indicator']
      missing_groups = [g for g in required_groups if g not in feature_dict]

      if missing_groups:
        quality_score *= (len(required_groups) - len(missing_groups)) / len(required_groups)
        logger.warning(
          f"Missing feature groups: {missing_groups}, quality={quality_score:.2f}"
        )

      # 2. Проверка на NaN/Inf в значениях
      def check_values(data: Dict) -> tuple[int, int]:
        """Рекурсивная проверка значений."""
        total_fields = 0
        valid_fields = 0

        for key, value in data.items():
          if isinstance(value, dict):
            sub_total, sub_valid = check_values(value)
            total_fields += sub_total
            valid_fields += sub_valid
          elif isinstance(value, (int, float)):
            total_fields += 1
            if not (np.isnan(value) or np.isinf(value)):
              valid_fields += 1

        return total_fields, valid_fields

      total_fields, valid_fields = check_values(feature_dict)

      if total_fields > 0:
        validity_ratio = valid_fields / total_fields
        quality_score *= validity_ratio

        if validity_ratio < 0.95:
          logger.warning(
            f"Feature validity low: {validity_ratio:.2%}, "
            f"valid={valid_fields}/{total_fields}"
          )

      # 3. Проверка временной актуальности
      if 'timestamp' in feature_dict:
        age_seconds = (datetime.now().timestamp() * 1000 - feature_dict['timestamp']) / 1000

        # Штрафуем за старые данные (> 10 секунд)
        if age_seconds > 10:
          age_penalty = max(0.5, 1.0 - (age_seconds - 10) / 100)
          quality_score *= age_penalty

          if age_penalty < 0.9:
            logger.warning(
              f"Feature data age: {age_seconds:.1f}s, "
              f"penalty={age_penalty:.2f}"
            )

      return max(0.0, min(1.0, quality_score))

    except Exception as e:
      logger.error(f"Error calculating feature quality: {e}", exc_info=True)
      return 0.5  # Средний уровень при ошибке

  def _detect_manipulation_risk(self, feature_dict: Dict) -> float:
    """
    Детекция риска манипуляции стаканом.

    Анализирует:
    - Аномальные кластеры в стакане
    - Резкие изменения объемов
    - Паттерны spoofing/layering

    Returns:
        float: Риск манипуляции [0.0, 1.0]
    """
    if not self.config.enable_manipulation_detection:
      return 0.0

    try:
      manipulation_score = 0.0

      # Извлекаем orderbook признаки
      ob_features = feature_dict.get('orderbook', {})

      # 1. Проверка аномальных кластеров
      cluster_concentration = ob_features.get('cluster_concentration_avg', 0.0)
      if cluster_concentration > 0.7:  # Высокая концентрация
        manipulation_score += 0.3
        logger.debug(f"High cluster concentration: {cluster_concentration:.3f}")

      # 2. Проверка резких изменений объемов
      volume_change = abs(ob_features.get('volume_change_percent', 0.0))
      if volume_change > 50:  # >50% изменение
        manipulation_score += 0.2
        logger.debug(f"Sharp volume change: {volume_change:.1f}%")

      # 3. Проверка spread аномалий
      spread = ob_features.get('spread_bps', 0.0)
      if spread > 100:  # Широкий spread - возможен spoofing
        manipulation_score += 0.2
        logger.debug(f"Wide spread detected: {spread:.1f} bps")

      # 4. Проверка дисбаланса ликвидности
      liquidity_imbalance = abs(ob_features.get('liquidity_imbalance', 0.0))
      if liquidity_imbalance > 0.8:  # Сильный перекос
        manipulation_score += 0.3
        logger.debug(f"High liquidity imbalance: {liquidity_imbalance:.3f}")

      # Нормализуем к [0, 1]
      manipulation_score = min(1.0, manipulation_score)

      # Обновляем историю
      self._manipulation_scores.append(manipulation_score)

      if manipulation_score > 0.5:
        logger.warning(
          f"High manipulation risk detected: {manipulation_score:.3f}"
        )

      return manipulation_score

    except Exception as e:
      logger.error(f"Error detecting manipulation risk: {e}", exc_info=True)
      return 0.0

  def _detect_market_regime(self, feature_dict: Dict) -> Optional[MarketRegime]:
    """
    Определение режима рынка на основе признаков.

    Анализирует:
    - Тренд (ADX, Moving Averages)
    - Волатильность (ATR, Bollinger Bands)
    - Объемы и активность

    Returns:
        MarketRegime или None
    """
    if not self.config.enable_regime_detection:
      return None

    try:
      indicator_features = feature_dict.get('indicator', {})

      # 1. Определяем силу тренда через ADX
      adx = indicator_features.get('trend_adx', 0.0)

      # 2. Определяем волатильность через ATR
      atr = indicator_features.get('volatility_atr', 0.0)
      candle_features = feature_dict.get('candle', {})
      atr_ratio = atr / candle_features.get('close', 1.0) if candle_features.get('close') else 0.0

      # 3. Определяем направление тренда
      macd = indicator_features.get('trend_macd', 0.0)
      rsi = indicator_features.get('momentum_rsi', 50.0)

      # ========================================
      # КЛАССИФИКАЦИЯ РЕЖИМА
      # ========================================

      # HIGH VOLATILITY: ATR ratio > 2% и ADX > 30
      if atr_ratio > 0.02 and adx > 30:
        logger.debug(
          f"Market regime: HIGH_VOLATILITY "
          f"(atr_ratio={atr_ratio:.3f}, adx={adx:.1f})"
        )
        return MarketRegime.HIGH_VOLATILITY

      # STRONG TREND: ADX > 40
      if adx > 40:
        logger.debug(
          f"Market regime: STRONG_TREND (adx={adx:.1f})"
        )
        return MarketRegime.STRONG_TREND

      # MILD TREND: ADX 25-40
      if 25 <= adx <= 40:
        logger.debug(
          f"Market regime: MILD_TREND (adx={adx:.1f})"
        )
        return MarketRegime.MILD_TREND

      # RANGING: ADX < 25 и низкая волатильность
      if adx < 25 and atr_ratio < 0.015:
        logger.debug(
          f"Market regime: RANGING "
          f"(adx={adx:.1f}, atr_ratio={atr_ratio:.3f})"
        )
        return MarketRegime.RANGING

      # DISTRIBUTION/ACCUMULATION: по объемам и MACD
      volume_features = indicator_features.get('volume_obv', 0.0)

      if macd < 0 and rsi < 30:  # Продажи
        logger.debug(
          f"Market regime: DISTRIBUTION (macd={macd:.3f}, rsi={rsi:.1f})"
        )
        return MarketRegime.DISTRIBUTION
      elif macd > 0 and rsi > 70:  # Покупки
        logger.debug(
          f"Market regime: ACCUMULATION (macd={macd:.3f}, rsi={rsi:.1f})"
        )
        return MarketRegime.ACCUMULATION

      # По умолчанию - RANGING
      return MarketRegime.RANGING

    except Exception as e:
      logger.error(f"Error detecting market regime: {e}", exc_info=True)
      return None

  def _estimate_mae_from_history(self) -> Optional[float]:
    """
    Оценка MAE на основе исторических ошибок.

    Returns:
        float: Предсказанная MAE или None
    """
    if not self.config.enable_mae_prediction:
      return None

    try:
      if len(self._mae_history) < 10:
        # Недостаточно данных - используем дефолтное значение
        return 0.015  # 1.5% default MAE

      # Вычисляем скользящее среднее MAE
      recent_mae = list(self._mae_history)[-50:]  # Последние 50 значений
      estimated_mae = np.mean(recent_mae)

      logger.debug(
        f"Estimated MAE from history: {estimated_mae:.4f} "
        f"(samples={len(recent_mae)})"
      )

      return float(estimated_mae)

    except Exception as e:
      logger.error(f"Error estimating MAE: {e}", exc_info=True)
      return 0.015

  def update_mae_history(self, actual_error: float):
    """
    Обновление истории MAE после получения фактической ошибки.

    Args:
        actual_error: Фактическая абсолютная ошибка предсказания
    """
    self._mae_history.append(actual_error)
    logger.debug(
      f"MAE history updated: new_error={actual_error:.4f}, "
      f"history_size={len(self._mae_history)}"
    )

  # ========================================
  # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
  # ========================================

  async def _request_ml_prediction(
      self,
      symbol: str,
      feature_dict: Dict
  ) -> Optional[Dict]:
    """Запрос ML предсказания от сервера."""
    if not self.ml_server_available:
      logger.debug(f"{symbol} | ML server unavailable, skipping request")
      return None

    try:
      url = f"{self.config.model_server_url}/predict"

      payload = {
        "symbol": symbol,
        "features": feature_dict,
        "model_version": self.config.model_version
      }

      async with self.session.post(
          url,
          json=payload,
          timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
      ) as response:
        if response.status == 200:
          result = await response.json()
          logger.debug(f"{symbol} | ML prediction received successfully")
          return result
        else:
          logger.warning(
            f"{symbol} | ML server returned status {response.status}"
          )
          self.ml_error_count += 1
          return None

    except asyncio.TimeoutError:
      logger.warning(f"{symbol} | ML prediction timeout")
      self.ml_error_count += 1
      return None
    except Exception as e:
      logger.error(f"{symbol} | ML prediction error: {e}", exc_info=True)
      self.ml_error_count += 1
      return None

  async def _perform_health_check(self) -> bool:
    """Проверка доступности ML сервера."""
    try:
      url = f"{self.config.model_server_url}/health"

      async with self.session.get(
          url,
          timeout=aiohttp.ClientTimeout(total=self.config.health_check_timeout)
      ) as response:
        if response.status == 200:
          self.ml_server_available = True
          self.health_check_failures = 0
          self.last_health_check = datetime.now()
          logger.info("ML server health check: OK")
          return True
        else:
          self.ml_server_available = False
          self.health_check_failures += 1
          logger.warning(f"ML server health check failed: {response.status}")
          return False

    except Exception as e:
      self.ml_server_available = False
      self.health_check_failures += 1
      logger.warning(f"ML server health check error: {e}")
      return False

  async def _periodic_health_check(self):
    """Периодическая проверка доступности ML сервера."""
    while True:
      try:
        await asyncio.sleep(self.config.health_check_interval)
        await self._perform_health_check()
      except asyncio.CancelledError:
        break
      except Exception as e:
        logger.error(f"Error in periodic health check: {e}", exc_info=True)

  def _get_cache_key(self, symbol: str, feature_dict: Dict) -> str:
    """Генерация ключа кэша."""
    # Используем timestamp из признаков
    timestamp = feature_dict.get('timestamp', int(datetime.now().timestamp() * 1000))
    return f"{symbol}_{timestamp}"

  def _get_cached_prediction(self, cache_key: str) -> Optional[Dict]:
    """Получение предсказания из кэша."""
    if not self.config.cache_predictions:
      return None

    cached = self._cache_dict.get(cache_key)

    if cached:
      # Проверяем TTL
      age = (datetime.now() - cached['cached_at']).total_seconds()

      if age < self.config.cache_ttl_seconds:
        return cached['prediction']
      else:
        # Удаляем устаревший кэш
        del self._cache_dict[cache_key]

    return None

  def _cache_prediction(self, cache_key: str, prediction: Dict):
    """Кэширование предсказания."""
    if not self.config.cache_predictions:
      return

    self._cache_dict[cache_key] = {
      'prediction': prediction,
      'cached_at': datetime.now()
    }

    # Ограничиваем размер кэша
    if len(self._cache_dict) > 1000:
      # Удаляем старейшие записи
      sorted_keys = sorted(
        self._cache_dict.keys(),
        key=lambda k: self._cache_dict[k]['cached_at']
      )

      for old_key in sorted_keys[:100]:
        del self._cache_dict[old_key]

  def _convert_direction_to_signal_type(self, direction: str) -> SignalType:
    """Конвертация ML направления в SignalType."""
    direction_upper = direction.upper()

    if direction_upper in ("BUY", "LONG"):
      return SignalType.BUY
    elif direction_upper in ("SELL", "SHORT"):
      return SignalType.SELL
    else:
      return SignalType.HOLD

  def _create_fallback_result(
      self,
      signal: TradingSignal,
      reason: str,
      start_time: datetime
  ) -> ValidationResult:
    """Создание fallback результата при ошибках."""
    inference_time = (datetime.now() - start_time).total_seconds() * 1000

    return ValidationResult(
      original_signal=signal,
      ml_direction=None,
      ml_confidence=None,
      ml_expected_return=None,
      validated=False,
      final_signal_type=signal.signal_type,
      final_confidence=0.0,
      agreement=False,
      reason=reason,
      inference_time_ms=inference_time,
      used_fallback=True,
      predicted_mae=None,
      manipulation_risk=0.0,
      market_regime=None,
      feature_quality=0.0
    )

  def get_statistics(self) -> Dict:
    """Получить расширенную статистику валидатора."""
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
      ),
      "mae_samples": len(self._mae_history),
      "avg_mae": (
        np.mean(list(self._mae_history))
        if len(self._mae_history) > 0 else None
      ),
      "manipulation_samples": len(self._manipulation_scores),
      "avg_manipulation_risk": (
        np.mean(list(self._manipulation_scores))
        if len(self._manipulation_scores) > 0 else 0.0
      )
    }