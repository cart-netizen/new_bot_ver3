"""
Reversal Detector - Обнаружение разворотов тренда.

МЕТОДЫ ДЕТЕКЦИИ:
1. Price Action Patterns (doji, engulfing, hammer)
2. Momentum Divergence (RSI, MACD)
3. Volume Anomaly (exhaustion volume)
4. Support/Resistance collision
5. Higher Timeframe confluence

STRENGTH LEVELS:
- WEAK: 1-2 indicators
- MODERATE: 3-4 indicators
- STRONG: 5-6 indicators
- CRITICAL: 7+ indicators + extreme readings
"""
from enum import Enum
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import numpy as np

from backend.core.logger import get_logger
from backend.config import settings
from backend.ml_engine.features.candle_feature_extractor import Candle
from backend.models.signal import SignalType
from backend.strategy.risk_models import ReversalSignal, ReversalStrength, ReversalAction

logger = get_logger(__name__)




class ReversalDetector:
  """
  Детектор разворотов тренда с multi-indicator подтверждением.
  """

  def __init__(self):
    """Инициализация."""
    self.enabled = settings.REVERSAL_DETECTOR_ENABLED
    self.min_indicators = settings.REVERSAL_MIN_INDICATORS_CONFIRM
    self.cooldown_seconds = settings.REVERSAL_COOLDOWN_SECONDS
    self.auto_action = settings.REVERSAL_AUTO_ACTION

    # История обнаруженных разворотов (для cooldown)
    self.reversal_history: Dict[str, datetime] = {}

    logger.info(
      f"ReversalDetector initialized: "
      f"enabled={self.enabled}, "
      f"min_indicators={self.min_indicators}, "
      f"auto_action={self.auto_action}"
    )

  def detect_reversal(
      self,
      symbol: str,
      candles: List[Candle],
      current_trend: SignalType,
      indicators: Dict,
      orderbook_metrics: Optional[Dict] = None
  ) -> Optional[ReversalSignal]:
    """
    Обнаружение сигнала разворота.

    Args:
        symbol: Торговая пара
        candles: История свечей (min 50)
        current_trend: Текущий тренд позиции
        indicators: Текущие индикаторы (RSI, MACD, etc)
        orderbook_metrics: Метрики стакана (optional)

    Returns:
        ReversalSignal если обнаружен разворот, иначе None
    """
    if not self.enabled:
      return None

    if len(candles) < 50:
      logger.debug(f"{symbol} | Insufficient candles for reversal detection")
      return None

    # Проверка cooldown
    if not self._check_cooldown(symbol):
      return None

    # Анализируем индикаторы разворота
    reversal_indicators = []

    # 1. PRICE ACTION PATTERNS
    price_action = self._detect_price_action_reversal(candles, current_trend)
    if price_action:
      reversal_indicators.append(price_action)

    # 2. MOMENTUM DIVERGENCE
    momentum_div = self._detect_momentum_divergence(candles, indicators, current_trend)
    if momentum_div:
      reversal_indicators.append(momentum_div)

    # 3. VOLUME EXHAUSTION
    volume_signal = self._detect_volume_exhaustion(candles, current_trend)
    if volume_signal:
      reversal_indicators.append(volume_signal)

    # 4. RSI EXTREME + REVERSAL
    rsi_signal = self._detect_rsi_reversal(indicators, current_trend)
    if rsi_signal:
      reversal_indicators.append(rsi_signal)

    # 5. MACD CROSS
    macd_signal = self._detect_macd_cross(indicators, current_trend)
    if macd_signal:
      reversal_indicators.append(macd_signal)

    # 6. ORDERBOOK PRESSURE SHIFT
    if orderbook_metrics:
      ob_signal = self._detect_orderbook_shift(orderbook_metrics, current_trend)
      if ob_signal:
        reversal_indicators.append(ob_signal)

    # 7. SUPPORT/RESISTANCE COLLISION
    sr_signal = self._detect_sr_collision(candles, current_trend)
    if sr_signal:
      reversal_indicators.append(sr_signal)

    # Проверяем минимальное количество подтверждений
    if len(reversal_indicators) < self.min_indicators:
      logger.debug(
        f"{symbol} | Reversal indicators insufficient: "
        f"{len(reversal_indicators)}/{self.min_indicators}"
      )
      return None

    # Определяем силу сигнала
    strength = self._calculate_reversal_strength(len(reversal_indicators))

    # Определяем рекомендуемое действие
    suggested_action = self._determine_action(strength, current_trend)

    # Создаем сигнал разворота
    reversal_signal = ReversalSignal(
      symbol=symbol,
      detected_at=datetime.now(),
      strength=strength,
      indicators_confirming=reversal_indicators,
      confidence=len(reversal_indicators) / 7.0,  # Максимум 7 индикаторов
      suggested_action=suggested_action,
      reason=self._build_reason(reversal_indicators, current_trend)
    )

    # Записываем в историю
    self.reversal_history[symbol] = datetime.now()

    logger.warning(
      f"{symbol} | 🔄 REVERSAL DETECTED | "
      f"Strength: {strength.value}, "
      f"Indicators: {len(reversal_indicators)}/{self.min_indicators}, "
      f"Action: {suggested_action}"
    )

    return reversal_signal

  def _detect_price_action_reversal(
        self,
        candles: List[Candle],
        current_trend: SignalType
    ) -> Optional[str]:
      """Обнаружение разворотных паттернов Price Action."""
      if len(candles) < 3:
        return None

      last_candle = candles[-1]
      prev_candle = candles[-2]

      body = abs(last_candle.close - last_candle.open)
      total_range = last_candle.high - last_candle.low

      # ✅ ИНИЦИАЛИЗИРУЕМ body_ratio СРАЗУ
      body_ratio = 0.0
      if total_range > 0:
        body_ratio = body / total_range

      # DOJI (маленькое тело)
      if body_ratio < 0.1 and total_range > 0:  # ✅ Теперь body_ratio всегда определена
        return "doji_at_extreme"

      # ENGULFING
      if current_trend == SignalType.BUY:
        # Bearish engulfing
        if (last_candle.open > last_candle.close and
            prev_candle.close > prev_candle.open and
            last_candle.open >= prev_candle.close and
            last_candle.close <= prev_candle.open):
          return "bearish_engulfing"
      else:  # SHORT position
        # Bullish engulfing
        if (last_candle.close > last_candle.open and
            prev_candle.open > prev_candle.close and
            last_candle.open <= prev_candle.close and
            last_candle.close >= prev_candle.open):
          return "bullish_engulfing"

      # HAMMER / SHOOTING STAR
      if total_range > 0:
        upper_shadow = last_candle.high - max(last_candle.open, last_candle.close)
        lower_shadow = min(last_candle.open, last_candle.close) - last_candle.low

        upper_ratio = upper_shadow / total_range
        lower_ratio = lower_shadow / total_range

        if current_trend == SignalType.BUY:
          # Shooting star (long upper shadow)
          if upper_ratio > 0.6 and body_ratio < 0.3:  # ✅ body_ratio теперь определена
            return "shooting_star"
        else:  # SHORT
          # Hammer (long lower shadow)
          if lower_ratio > 0.6 and body_ratio < 0.3:  # ✅ body_ratio теперь определена
            return "hammer"

      return None

  def _detect_momentum_divergence(
      self,
      candles: List[Candle],
      indicators: Dict,
      current_trend: SignalType
  ) -> Optional[str]:
    """
    Обнаружение дивергенции momentum индикаторов.

    Дивергенция = Цена делает новый high/low, но индикатор не подтверждает.
    """
    if len(candles) < 20:
      return None

    # Получаем цены и RSI
    closes = np.array([c.close for c in candles[-20:]])
    rsi = indicators.get('rsi')

    if rsi is None or not isinstance(rsi, (list, np.ndarray)) or len(rsi) < 20:
      return None

    rsi = np.array(rsi[-20:])

    # Ищем дивергенцию на последних 10 свечах
    recent_closes = closes[-10:]
    recent_rsi = rsi[-10:]

    if current_trend == SignalType.BUY:
      # Bearish divergence: цена растет, RSI падает
      price_trend = recent_closes[-1] > recent_closes[0]
      rsi_trend = recent_rsi[-1] < recent_rsi[0]

      # Проверяем: новый high по цене, но не по RSI
      price_made_new_high = recent_closes[-1] >= np.max(recent_closes[:-1])
      rsi_made_new_high = recent_rsi[-1] >= np.max(recent_rsi[:-1])

      if price_trend and not rsi_trend and price_made_new_high and not rsi_made_new_high:
        return "bearish_divergence"

    else:  # SHORT
      # Bullish divergence: цена падает, RSI растет
      price_trend = recent_closes[-1] < recent_closes[0]
      rsi_trend = recent_rsi[-1] > recent_rsi[0]

      # Проверяем: новый low по цене, но не по RSI
      price_made_new_low = recent_closes[-1] <= np.min(recent_closes[:-1])
      rsi_made_new_low = recent_rsi[-1] <= np.min(recent_rsi[:-1])

      if price_trend and not rsi_trend and price_made_new_low and not rsi_made_new_low:
        return "bullish_divergence"

    return None

  def _detect_volume_exhaustion(
      self,
      candles: List[Candle],
      current_trend: SignalType
  ) -> Optional[str]:
    """
    Обнаружение exhaustion volume (истощение объема на экстремумах).

    Признаки:
    - Резкий скачок объема на пике/дне
    - Последующее снижение объема
    - Цена не может пробить уровень
    """
    if len(candles) < 10:
      return None

    volumes = np.array([c.volume for c in candles[-10:]])
    closes = np.array([c.close for c in candles[-10:]])

    # Средний объем за период
    avg_volume = np.mean(volumes[:-2])  # Исключаем последние 2

    # Последний объем
    last_volume = volumes[-1]
    prev_volume = volumes[-2]

    # Проверка spike в объеме
    if prev_volume > avg_volume * 2.0:  # Spike 2x
      # Проверка снижения после spike
      if last_volume < prev_volume * 0.7:  # Снижение на 30%

        if current_trend == SignalType.BUY:
          # Цена должна быть около максимумов
          is_near_high = closes[-2] >= np.max(closes[:-2]) * 0.98
          if is_near_high:
            return "volume_exhaustion_uptrend"

        else:  # SHORT
          # Цена около минимумов
          is_near_low = closes[-2] <= np.min(closes[:-2]) * 1.02
          if is_near_low:
            return "volume_exhaustion_downtrend"

    return None

  def _detect_rsi_reversal(
      self,
      indicators: Dict,
      current_trend: SignalType
  ) -> Optional[str]:
    """
    Обнаружение разворота по RSI.

    Условия:
    - RSI в экстремальной зоне (>75 или <25)
    - RSI начинает разворачиваться
    """
    rsi = indicators.get('rsi')

    if rsi is None or not isinstance(rsi, (list, np.ndarray)) or len(rsi) < 3:
      return None

    rsi = np.array(rsi[-3:])

    current_rsi = rsi[-1]
    prev_rsi = rsi[-2]

    if current_trend == SignalType.BUY:
      # Overbought + reversal
      if prev_rsi > 75 and current_rsi < prev_rsi:
        return "rsi_overbought_reversal"

    else:  # SHORT
      # Oversold + reversal
      if prev_rsi < 25 and current_rsi > prev_rsi:
        return "rsi_oversold_reversal"

    return None

  def _detect_macd_cross(
      self,
      indicators: Dict,
      current_trend: SignalType
  ) -> Optional[str]:
    """
    Обнаружение пересечения MACD.
    """
    macd = indicators.get('macd')
    macd_signal = indicators.get('macd_signal')

    if macd is None or macd_signal is None:
      return None

    if not isinstance(macd, (list, np.ndarray)) or len(macd) < 2:
      return None

    if not isinstance(macd_signal, (list, np.ndarray)) or len(macd_signal) < 2:
      return None

    macd = np.array(macd[-2:])
    macd_signal = np.array(macd_signal[-2:])

    prev_macd = macd[-2]
    curr_macd = macd[-1]
    prev_signal = macd_signal[-2]
    curr_signal = macd_signal[-1]

    if current_trend == SignalType.BUY:
      # Bearish cross: MACD crosses below signal
      if prev_macd > prev_signal and curr_macd < curr_signal:
        return "macd_bearish_cross"

    else:  # SHORT
      # Bullish cross: MACD crosses above signal
      if prev_macd < prev_signal and curr_macd > curr_signal:
        return "macd_bullish_cross"

    return None

  def _detect_orderbook_shift(
      self,
      orderbook_metrics: Dict,
      current_trend: SignalType
  ) -> Optional[str]:
    """
    Обнаружение изменения pressure в стакане.
    """
    imbalance = orderbook_metrics.get('imbalance', 0.0)

    # Значительный сдвиг в давлении
    if current_trend == SignalType.BUY:
      # Сильное давление продавцов
      if imbalance < -0.4:  # Sellers dominate
        return "orderbook_sell_pressure"

    else:  # SHORT
      # Сильное давление покупателей
      if imbalance > 0.4:  # Buyers dominate
        return "orderbook_buy_pressure"

    return None

  def _detect_sr_collision(
      self,
      candles: List[Candle],
      current_trend: SignalType
  ) -> Optional[str]:
    """
    Обнаружение столкновения с S/R уровнем.

    Простая реализация: проверка максимумов/минимумов за последние 50 свечей.
    """
    if len(candles) < 50:
      return None

    recent_candles = candles[-50:]
    current_price = candles[-1].close

    highs = [c.high for c in recent_candles]
    lows = [c.low for c in recent_candles]

    if current_trend == SignalType.BUY:
      # Приближение к сопротивлению
      resistance = np.max(highs)
      distance_to_resistance = abs(current_price - resistance) / current_price

      if distance_to_resistance < 0.005:  # В пределах 0.5%
        return "near_resistance"

    else:  # SHORT
      # Приближение к поддержке
      support = np.min(lows)
      distance_to_support = abs(current_price - support) / current_price

      if distance_to_support < 0.005:  # В пределах 0.5%
        return "near_support"

    return None

  def _calculate_reversal_strength(self, num_indicators: int) -> ReversalStrength:
    """Определение силы сигнала разворота."""
    if num_indicators >= 7:
      return ReversalStrength.CRITICAL
    elif num_indicators >= 5:
      return ReversalStrength.STRONG
    elif num_indicators >= 3:
      return ReversalStrength.MODERATE
    else:
      return ReversalStrength.WEAK

  def _determine_action(
        self,
        strength: ReversalStrength,
        current_trend: SignalType
    ) -> ReversalAction:  # ✅ Возвращаем ReversalAction вместо str
      """Определение рекомендуемого действия."""
      if strength == ReversalStrength.CRITICAL:
        return ReversalAction.CLOSE_POSITION
      elif strength == ReversalStrength.STRONG:
        return ReversalAction.REDUCE_SIZE
      elif strength == ReversalStrength.MODERATE:
        return ReversalAction.TIGHTEN_SL
      else:
        return ReversalAction.NO_ACTION

  def _build_reason(
      self,
      indicators: List[str],
      current_trend: SignalType
  ) -> str:
    """Формирование причины разворота."""
    trend_name = "uptrend" if current_trend == SignalType.BUY else "downtrend"
    indicators_str = ", ".join(indicators)

    return (
      f"Reversal detected in {trend_name}: "
      f"{len(indicators)} indicators confirm ({indicators_str})"
    )

  def _check_cooldown(self, symbol: str) -> bool:
    """Проверка cooldown периода."""
    if symbol not in self.reversal_history:
      return True

    last_reversal = self.reversal_history[symbol]
    time_since = (datetime.now() - last_reversal).total_seconds()

    if time_since < self.cooldown_seconds:
      logger.debug(
        f"{symbol} | Reversal detection in cooldown: "
        f"{time_since:.0f}s / {self.cooldown_seconds}s"
      )
      return False

    return True


# Глобальный экземпляр
reversal_detector = ReversalDetector()