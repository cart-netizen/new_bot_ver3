"""
Candle Feature Extractor для извлечения 25 признаков из свечных данных (OHLCV).

ИСПРАВЛЕНИЕ: Добавлен @property typical_price в класс Candle.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

from core.logger import get_logger
from core.periodic_logger import periodic_logger

logger = get_logger(__name__)


@dataclass
class Candle:
  """Модель свечи (OHLCV)"""
  timestamp: int
  open: float
  high: float
  low: float
  close: float
  volume: float

  @property
  def typical_price(self) -> float:
    """Typical price для VWAP: (H+L+C)/3"""
    return (self.high + self.low + self.close) / 3

  @property
  def range(self) -> float:
    """Диапазон свечи (high - low)"""
    return self.high - self.low

  @property
  def body_size(self) -> float:
    """Размер тела свечи"""
    return abs(self.close - self.open)

  @property
  def is_bullish(self) -> bool:
    """Бычья свеча (закрытие выше открытия)"""
    return self.close > self.open

  @property
  def is_bearish(self) -> bool:
    """Медвежья свеча (закрытие ниже открытия)"""
    return self.close < self.open


@dataclass
class CandleFeatures:
  """Контейнер для признаков из свечных данных"""

  symbol: str
  timestamp: int

  # Базовые OHLCV (6)
  open: float
  high: float
  low: float
  close: float
  volume: float
  typical_price: float  # (H+L+C)/3

  # Производные метрики (7)
  returns: float  # (close - prev_close) / prev_close
  log_returns: float  # ln(close / prev_close)
  high_low_range: float  # (high - low) / close
  close_open_diff: float  # (close - open) / open
  upper_shadow: float  # Верхняя тень
  lower_shadow: float  # Нижняя тень
  body_size: float  # Размер тела / диапазон

  # Волатильность (3)
  realized_volatility: float  # Std returns
  parkinson_volatility: float  # High-Low estimator
  garman_klass_volatility: float  # OHLC estimator

  # Volume features (5)
  volume_ma_ratio: float  # volume / MA(volume)
  volume_change_rate: float  # Изменение объема
  price_volume_trend: float  # PVT
  volume_weighted_price: float  # VWAP
  money_flow: float  # Денежный поток

  # Pattern indicators (4)
  doji_strength: float  # Сила паттерна дожи
  hammer_strength: float  # Сила паттерна молот
  engulfing_strength: float  # Сила поглощения
  gap_size: float  # Размер гэпа

  def to_array(self) -> np.ndarray:
    """Преобразование в numpy array"""
    return np.array([
      # Базовые (6)
      self.open,
      self.high,
      self.low,
      self.close,
      self.volume,
      self.typical_price,
      # Производные (7)
      self.returns,
      self.log_returns,
      self.high_low_range,
      self.close_open_diff,
      self.upper_shadow,
      self.lower_shadow,
      self.body_size,
      # Волатильность (3)
      self.realized_volatility,
      self.parkinson_volatility,
      self.garman_klass_volatility,
      # Volume (5)
      self.volume_ma_ratio,
      self.volume_change_rate,
      self.price_volume_trend,
      self.volume_weighted_price,
      self.money_flow,
      # Patterns (4)
      self.doji_strength,
      self.hammer_strength,
      self.engulfing_strength,
      self.gap_size
    ], dtype=np.float32)

  def to_dict(self) -> Dict[str, float]:
    """Преобразование в словарь"""
    return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class CandleFeatureExtractor:
  """
  Извлекает 25 признаков из свечных данных (OHLCV).
  """

  def __init__(self, symbol: str, lookback_period: int = 20):
    """
    Args:
        symbol: Торговая пара
        lookback_period: Период для скользящих средних и волатильности
    """
    self.symbol = symbol
    self.lookback_period = lookback_period

    # История свечей
    self.candle_history: List[Candle] = []
    self.max_history_size = 200

    # Кумулятивный PVT
    self.cumulative_pvt = 0.0

    logger.info(
      f"CandleFeatureExtractor инициализирован для {symbol}, "
      f"lookback={lookback_period}"
    )

  def extract(
      self,
      candle: Candle,
      prev_candle: Optional[Candle] = None
  ) -> CandleFeatures:
    """
    Извлекает все признаки из свечи.

    Args:
        candle: Текущая свеча
        prev_candle: Предыдущая свеча (для расчета returns)

    Returns:
        CandleFeatures: Извлеченные признаки
    """
    logger.debug(f"{self.symbol} | Извлечение признаков из свечи")

    try:
      # Добавляем в историю
      self.candle_history.append(candle)
      if len(self.candle_history) > self.max_history_size:
        self.candle_history.pop(0)

      # Используем предыдущую свечу из истории если не передана
      if prev_candle is None and len(self.candle_history) >= 2:
        prev_candle = self.candle_history[-2]

      # 1. Базовые OHLCV
      basic_features = self._extract_basic_features(candle)
      logger.debug(f"{self.symbol} | Базовые признаки извлечены")

      # 2. Производные метрики
      derivative_features = self._extract_derivative_features(
        candle,
        prev_candle
      )
      logger.debug(f"{self.symbol} | Производные признаки извлечены")

      # 3. Волатильность
      volatility_features = self._extract_volatility_features(candle)
      logger.debug(f"{self.symbol} | Признаки волатильности извлечены")

      # 4. Volume features
      volume_features = self._extract_volume_features(candle, prev_candle)
      logger.debug(f"{self.symbol} | Volume признаки извлечены")

      # 5. Pattern indicators
      pattern_features = self._extract_pattern_features(candle, prev_candle)
      logger.debug(f"{self.symbol} | Pattern признаки извлечены")

      # Объединяем
      features = CandleFeatures(
        symbol=self.symbol,
        timestamp=candle.timestamp,
        **basic_features,
        **derivative_features,
        **volatility_features,
        **volume_features,
        **pattern_features
      )

      key = f"orderbook_features_{self.symbol}"
      should_log, count = periodic_logger.should_log(key, every_n=500, first_n=1)

      if should_log:
        logger.info(
          f"{self.symbol} | Извлечено 25 признаков из стакана "
          f"(#{count}), close={features.close:.2f}, returns={features.returns:.4f}"
        )

      # logger.info(
      #   f"{self.symbol} | Извлечено 25 признаков из свечи, "
      #   f"close={features.close:.2f}, returns={features.returns:.4f}"
      # )

      return features

    except Exception as e:
      logger.error(f"{self.symbol} | Ошибка извлечения признаков: {e}")
      raise

  def _extract_basic_features(self, candle: Candle) -> Dict[str, float]:
    """Извлечение базовых OHLCV признаков (6)"""

    # Typical price (средняя цена) - используем property
    typical_price = candle.typical_price

    return {
      "open": candle.open,
      "high": candle.high,
      "low": candle.low,
      "close": candle.close,
      "volume": candle.volume,
      "typical_price": typical_price
    }

  def _extract_derivative_features(
      self,
      candle: Candle,
      prev_candle: Optional[Candle]
  ) -> Dict[str, float]:
    """Извлечение производных метрик (7)"""

    # Returns
    if prev_candle and prev_candle.close > 0:
      returns = (candle.close - prev_candle.close) / prev_candle.close
      log_returns = np.log(candle.close / prev_candle.close)
    else:
      returns = 0.0
      log_returns = 0.0

    # High-Low range
    range_val = candle.high - candle.low
    high_low_range = range_val / candle.close if candle.close > 0 else 0.0

    # Close-Open difference
    if candle.open > 0:
      close_open_diff = (candle.close - candle.open) / candle.open
    else:
      close_open_diff = 0.0

    # Shadows (тени)
    max_oc = max(candle.open, candle.close)
    min_oc = min(candle.open, candle.close)

    if range_val > 0:
      upper_shadow = (candle.high - max_oc) / range_val
      lower_shadow = (min_oc - candle.low) / range_val
      body_size = abs(candle.close - candle.open) / range_val
    else:
      upper_shadow = 0.0
      lower_shadow = 0.0
      body_size = 0.0

    return {
      "returns": returns,
      "log_returns": log_returns,
      "high_low_range": high_low_range,
      "close_open_diff": close_open_diff,
      "upper_shadow": upper_shadow,
      "lower_shadow": lower_shadow,
      "body_size": body_size
    }

  def _extract_volatility_features(self, candle: Candle) -> Dict[str, float]:
    """Извлечение признаков волатильности (3)"""

    # Realized volatility (std returns)
    if len(self.candle_history) >= 2:
      closes = [c.close for c in self.candle_history[-self.lookback_period:]]
      if len(closes) >= 2:
        returns_arr = np.diff(closes) / closes[:-1]
        realized_volatility = float(np.std(returns_arr))
      else:
        realized_volatility = 0.0
    else:
      realized_volatility = 0.0

    # Parkinson volatility (High-Low estimator)
    if len(self.candle_history) >= 2:
      hl_ratios = [
        np.log(c.high / c.low) ** 2
        for c in self.candle_history[-self.lookback_period:]
        if c.low > 0
      ]
      if hl_ratios:
        parkinson_volatility = np.sqrt(
          np.mean(hl_ratios) / (4 * np.log(2))
        )
      else:
        parkinson_volatility = 0.0
    else:
      parkinson_volatility = 0.0

    # Garman-Klass volatility (OHLC estimator)
    if len(self.candle_history) >= 2:
      gk_components = []
      for c in self.candle_history[-self.lookback_period:]:
        if c.low > 0 and c.open > 0 and c.close > 0:
          hl = (np.log(c.high / c.low)) ** 2
          co = (np.log(c.close / c.open)) ** 2
          gk_components.append(0.5 * hl - (2 * np.log(2) - 1) * co)

      if gk_components:
        garman_klass_volatility = np.sqrt(np.mean(gk_components))
      else:
        garman_klass_volatility = 0.0
    else:
      garman_klass_volatility = 0.0

    return {
      "realized_volatility": realized_volatility,
      "parkinson_volatility": parkinson_volatility,
      "garman_klass_volatility": garman_klass_volatility
    }

  def _extract_volume_features(
      self,
      candle: Candle,
      prev_candle: Optional[Candle]
  ) -> Dict[str, float]:
    """Извлечение volume признаков (5)"""

    # Volume MA ratio
    if len(self.candle_history) >= self.lookback_period:
      volumes = [c.volume for c in self.candle_history[-self.lookback_period:]]
      volume_ma = np.mean(volumes)
      volume_ma_ratio = candle.volume / volume_ma if volume_ma > 0 else 1.0
    else:
      volume_ma_ratio = 1.0

    # Volume change rate
    if prev_candle and prev_candle.volume > 0:
      volume_change_rate = (
          (candle.volume - prev_candle.volume) / prev_candle.volume
      )
    else:
      volume_change_rate = 0.0

    # Price Volume Trend (кумулятивный)
    if prev_candle and prev_candle.close > 0:
      price_change = (candle.close - prev_candle.close) / prev_candle.close
      self.cumulative_pvt += price_change * candle.volume
    price_volume_trend = self.cumulative_pvt

    # Volume Weighted Price (VWAP approximation)
    # ИСПРАВЛЕНО: используем property typical_price напрямую
    if len(self.candle_history) >= 2:
      recent = self.candle_history[-self.lookback_period:]
      total_volume = sum(c.volume for c in recent)

      if total_volume > 0:
        # Теперь typical_price работает корректно как property
        volume_weighted_price = sum(
          c.typical_price * c.volume for c in recent
        ) / total_volume
      else:
        volume_weighted_price = candle.close
    else:
      volume_weighted_price = candle.close

    # Money Flow (денежный поток)
    typical_price = candle.typical_price
    money_flow = typical_price * candle.volume

    return {
      "volume_ma_ratio": volume_ma_ratio,
      "volume_change_rate": volume_change_rate,
      "price_volume_trend": price_volume_trend,
      "volume_weighted_price": volume_weighted_price,
      "money_flow": money_flow
    }

  def _extract_pattern_features(
      self,
      candle: Candle,
      prev_candle: Optional[Candle]
  ) -> Dict[str, float]:
    """Извлечение pattern признаков (4)"""

    # Doji strength (маленькое тело)
    range_val = candle.high - candle.low
    body = abs(candle.close - candle.open)

    if range_val > 0:
      doji_strength = 1 - (body / range_val)
    else:
      doji_strength = 0.0

    # Hammer strength (длинная нижняя тень, маленькое тело сверху)
    if range_val > 0:
      lower_wick = min(candle.open, candle.close) - candle.low
      hammer_strength = (
        (lower_wick / range_val) * (1 - body / range_val)
        if lower_wick > 0 else 0.0
      )
    else:
      hammer_strength = 0.0

    # Engulfing strength (поглощение предыдущей свечи)
    if prev_candle:
      prev_body = abs(prev_candle.close - prev_candle.open)
      current_body = abs(candle.close - candle.open)

      # Бычье поглощение: текущая бычья, предыдущая медвежья
      if candle.is_bullish and prev_candle.is_bearish:
        if (candle.close > prev_candle.open and
            candle.open < prev_candle.close):
          engulfing_strength = current_body / max(prev_body, 0.001)
        else:
          engulfing_strength = 0.0
      # Медвежье поглощение
      elif candle.is_bearish and prev_candle.is_bullish:
        if (candle.close < prev_candle.open and
            candle.open > prev_candle.close):
          engulfing_strength = -(current_body / max(prev_body, 0.001))
        else:
          engulfing_strength = 0.0
      else:
        engulfing_strength = 0.0
    else:
      engulfing_strength = 0.0

    # Gap size (гэп между свечами)
    if prev_candle:
      if candle.open > prev_candle.close:
        # Восходящий гэп
        gap_size = (candle.open - prev_candle.close) / prev_candle.close
      elif candle.open < prev_candle.close:
        # Нисходящий гэп
        gap_size = (candle.open - prev_candle.close) / prev_candle.close
      else:
        gap_size = 0.0
    else:
      gap_size = 0.0

    return {
      "doji_strength": doji_strength,
      "hammer_strength": hammer_strength,
      "engulfing_strength": engulfing_strength,
      "gap_size": gap_size
    }