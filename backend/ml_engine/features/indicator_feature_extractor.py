"""
Indicator Feature Extractor для извлечения 35+ признаков из технических индикаторов.

Professional Industry-Standard Implementation:
- MACD Signal: EMA-based (not SMA)
- RSI: Wilder's smoothing
- ADX: Wilder's smoothing
- Stochastic: %D = SMA(%K, 3)
- Aroon: Up/Down momentum indicator

Использует numpy для вычислений (numpy-based, no TA-Lib dependency).
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from backend.core.logger import get_logger
from backend.core.periodic_logger import periodic_logger
from backend.ml_engine.features.candle_feature_extractor import Candle

logger = get_logger(__name__)


# ===== State Classes for Stateful Indicators =====

@dataclass
class MACDState:
  """State для MACD индикатора с историей для Signal EMA"""
  macd_history: deque = field(default_factory=lambda: deque(maxlen=50))
  signal_ema: Optional[float] = None


@dataclass
class StochasticState:
  """State для Stochastic с историей %K для расчета %D"""
  k_history: deque = field(default_factory=lambda: deque(maxlen=3))  # For %D = SMA(%K, 3)


@dataclass
class ADXState:
  """State для ADX с Wilder's smoothing"""
  smooth_tr: Optional[float] = None
  smooth_plus_dm: Optional[float] = None
  smooth_minus_dm: Optional[float] = None
  dx_history: deque = field(default_factory=lambda: deque(maxlen=14))
  adx_value: Optional[float] = None


@dataclass
class RSIState:
  """State для RSI с Wilder's smoothing"""
  avg_gain: Optional[float] = None
  avg_loss: Optional[float] = None
  period: int = 14


@dataclass
class IndicatorFeatures:
  """Контейнер для признаков технических индикаторов"""

  symbol: str
  timestamp: int

  # Trend indicators (12)
  sma_10: float
  sma_20: float
  sma_50: float
  ema_10: float
  ema_20: float
  ema_50: float
  macd: float
  macd_signal: float
  macd_histogram: float
  adx: float
  plus_di: float
  minus_di: float

  # Momentum indicators (9)
  rsi_14: float
  rsi_28: float
  stochastic_k: float
  stochastic_d: float
  williams_r: float
  cci: float
  momentum_10: float
  roc: float  # Rate of Change
  mfi: float  # Money Flow Index

  # Volatility indicators (8)
  bollinger_upper: float
  bollinger_middle: float
  bollinger_lower: float
  bollinger_width: float
  bollinger_pct: float  # Положение цены в полосах
  atr_14: float
  keltner_upper: float
  keltner_lower: float

  # Volume indicators (6)
  obv: float  # On-Balance Volume
  vwap: float
  ad_line: float  # Accumulation/Distribution
  cmf: float  # Chaikin Money Flow
  vpt: float  # Volume Price Trend
  nvi: float  # Negative Volume Index

  # Aroon indicators (2) - NEW
  aroon_up: float  # Aroon Up (0-100)
  aroon_down: float  # Aroon Down (0-100)

  def to_array(self) -> np.ndarray:
    """Преобразование в numpy array"""
    return np.array([
      # Trend (12)
      self.sma_10, self.sma_20, self.sma_50,
      self.ema_10, self.ema_20, self.ema_50,
      self.macd, self.macd_signal, self.macd_histogram,
      self.adx, self.plus_di, self.minus_di,
      # Momentum (9)
      self.rsi_14, self.rsi_28,
      self.stochastic_k, self.stochastic_d,
      self.williams_r, self.cci, self.momentum_10,
      self.roc, self.mfi,
      # Volatility (8)
      self.bollinger_upper, self.bollinger_middle, self.bollinger_lower,
      self.bollinger_width, self.bollinger_pct,
      self.atr_14, self.keltner_upper, self.keltner_lower,
      # Volume (6)
      self.obv, self.vwap, self.ad_line,
      self.cmf, self.vpt, self.nvi,
      # Aroon (2)
      self.aroon_up, self.aroon_down
    ], dtype=np.float32)

  def to_dict(self) -> Dict[str, float]:
    """Преобразование в словарь"""
    return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class IndicatorFeatureExtractor:
  """
  Извлекает 37+ признаков из технических индикаторов.

  Professional Industry-Standard Implementation (numpy-based):
  - MACD: EMA-based signal (not SMA)
  - RSI: Wilder's smoothing
  - ADX: Wilder's smoothing
  - Stochastic: %D = SMA(%K, 3)
  - Aroon: Up/Down momentum
  """

  def __init__(self, symbol: str):
    """
    Args:
        symbol: Торговая пара
    """
    self.symbol = symbol

    # История для индикаторов
    self.candle_history: List[Candle] = []
    self.max_history_size = 100  # MEMORY FIX: 200 → 100 свечей

    # Кумулятивные индикаторы
    self.obv_cumulative = 0.0
    self.ad_cumulative = 0.0
    self.vpt_cumulative = 0.0
    self.nvi = 1000.0  # Начальное значение NVI

    # ===== Stateful Indicators (Professional) =====
    self.macd_state = MACDState()
    self.stochastic_state = StochasticState()
    self.adx_state = ADXState()
    self.rsi_14_state = RSIState(period=14)
    self.rsi_28_state = RSIState(period=28)

    logger.info(f"✅ Professional IndicatorFeatureExtractor инициализирован для {symbol}")

  def extract(self, candles: List[Candle]) -> IndicatorFeatures:
    """
    Извлекает все индикаторы из списка свечей.

    Args:
        candles: Список свечей (минимум 50 для надежных расчетов)

    Returns:
        IndicatorFeatures: Извлеченные признаки
    """
    logger.debug(f"{self.symbol} | Извлечение индикаторов")

    try:
      # Обновляем историю
      self.candle_history.extend(candles)
      if len(self.candle_history) > self.max_history_size:
        self.candle_history = self.candle_history[-self.max_history_size:]

      # Нужно минимум 50 свечей для надежных индикаторов
      if len(self.candle_history) < 50:
        logger.warning(
          f"{self.symbol} | Недостаточно данных для индикаторов "
          f"({len(self.candle_history)} < 50)"
        )
        return self._create_default_features(candles[-1] if candles else None)

      current_candle = self.candle_history[-1]

      # 1. Trend indicators
      trend_features = self._extract_trend_indicators()
      logger.debug(f"{self.symbol} | Trend индикаторы извлечены")

      # 2. Momentum indicators
      momentum_features = self._extract_momentum_indicators()
      logger.debug(f"{self.symbol} | Momentum индикаторы извлечены")

      # 3. Volatility indicators
      volatility_features = self._extract_volatility_indicators()
      logger.debug(f"{self.symbol} | Volatility индикаторы извлечены")

      # 4. Volume indicators
      volume_features = self._extract_volume_indicators()
      logger.debug(f"{self.symbol} | Volume индикаторы извлечены")

      # 5. Aroon indicators (NEW)
      aroon_features = self._extract_aroon_indicators()
      logger.debug(f"{self.symbol} | Aroon индикаторы извлечены")

      # Объединяем
      features = IndicatorFeatures(
        symbol=self.symbol,
        timestamp=current_candle.timestamp,
        **trend_features,
        **momentum_features,
        **volatility_features,
        **volume_features,
        **aroon_features
      )

      key = f"indicator_features_{self.symbol}"
      should_log, count = periodic_logger.should_log(key, every_n=500, first_n=1)

      if should_log:
        logger.info(
          f"{self.symbol} | Извлечено 37 индикаторов "
          f"(#{count}), RSI={features.rsi_14:.2f}, MACD={features.macd:.4f}, "
          f"Aroon_Up={features.aroon_up:.1f}"
        )

      # logger.info(
      #   f"{self.symbol} | Извлечено 35 индикаторов, "
      #   f"RSI={features.rsi_14:.2f}, MACD={features.macd:.4f}"
      # )

      return features

    except Exception as e:
      logger.error(f"{self.symbol} | Ошибка извлечения индикаторов: {e}")
      raise

  def _extract_trend_indicators(self) -> Dict[str, float]:
    """Извлечение trend индикаторов (12)"""

    closes = np.array([c.close for c in self.candle_history])
    highs = np.array([c.high for c in self.candle_history])
    lows = np.array([c.low for c in self.candle_history])

    # SMA (Simple Moving Average)
    sma_10 = float(np.mean(closes[-10:])) if len(closes) >= 10 else closes[-1]
    sma_20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else closes[-1]
    sma_50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else closes[-1]

    # EMA (Exponential Moving Average)
    ema_10 = self._calculate_ema(closes, 10)
    ema_20 = self._calculate_ema(closes, 20)
    ema_50 = self._calculate_ema(closes, 50)

    # MACD (Moving Average Convergence Divergence)
    ema_12 = self._calculate_ema(closes, 12)
    ema_26 = self._calculate_ema(closes, 26)
    macd = ema_12 - ema_26

    # ===== MACD Signal (9-period EMA of MACD) - PROFESSIONAL =====
    # Store MACD in history
    self.macd_state.macd_history.append(macd)

    # Calculate Signal line (EMA of MACD, not SMA!)
    if len(self.macd_state.macd_history) >= 9:
      if self.macd_state.signal_ema is None:
        # Initialize with SMA for first time
        self.macd_state.signal_ema = float(
          np.mean(list(self.macd_state.macd_history)[-9:])
        )
      else:
        # Update with EMA formula
        multiplier = 2 / (9 + 1)
        self.macd_state.signal_ema = (
          macd * multiplier + self.macd_state.signal_ema * (1 - multiplier)
        )
      macd_signal = self.macd_state.signal_ema
    else:
      macd_signal = macd  # Not enough history yet

    macd_histogram = macd - macd_signal

    # ===== ADX (Average Directional Index) - PROFESSIONAL with Wilder's smoothing =====
    adx, plus_di, minus_di = self._calculate_adx_professional(highs, lows, closes)

    return {
      "sma_10": sma_10,
      "sma_20": sma_20,
      "sma_50": sma_50,
      "ema_10": ema_10,
      "ema_20": ema_20,
      "ema_50": ema_50,
      "macd": macd,
      "macd_signal": macd_signal,
      "macd_histogram": macd_histogram,
      "adx": adx,
      "plus_di": plus_di,
      "minus_di": minus_di
    }

  def _extract_momentum_indicators(self) -> Dict[str, float]:
    """Извлечение momentum индикаторов (9)"""

    closes = np.array([c.close for c in self.candle_history])
    highs = np.array([c.high for c in self.candle_history])
    lows = np.array([c.low for c in self.candle_history])
    volumes = np.array([c.volume for c in self.candle_history])

    # ===== RSI (Relative Strength Index) - PROFESSIONAL with Wilder's smoothing =====
    rsi_14 = self._calculate_rsi_professional(closes, self.rsi_14_state)
    rsi_28 = self._calculate_rsi_professional(closes, self.rsi_28_state)

    # ===== Stochastic Oscillator - PROFESSIONAL with %D = SMA(%K, 3) =====
    stochastic_k, stochastic_d = self._calculate_stochastic_professional(highs, lows, closes)

    # Williams %R
    williams_r = self._calculate_williams_r(highs, lows, closes)

    # CCI (Commodity Channel Index)
    cci = self._calculate_cci(highs, lows, closes)

    # Momentum
    momentum_10 = (
      (closes[-1] - closes[-10]) / closes[-10]
      if len(closes) >= 10 else 0.0
    )

    # ROC (Rate of Change)
    roc = (
      ((closes[-1] - closes[-12]) / closes[-12]) * 100
      if len(closes) >= 12 else 0.0
    )

    # MFI (Money Flow Index)
    mfi = self._calculate_mfi(highs, lows, closes, volumes)

    return {
      "rsi_14": rsi_14,
      "rsi_28": rsi_28,
      "stochastic_k": stochastic_k,
      "stochastic_d": stochastic_d,
      "williams_r": williams_r,
      "cci": cci,
      "momentum_10": momentum_10,
      "roc": roc,
      "mfi": mfi
    }

  def _extract_volatility_indicators(self) -> Dict[str, float]:
    """Извлечение volatility индикаторов (8)"""

    closes = np.array([c.close for c in self.candle_history])
    highs = np.array([c.high for c in self.candle_history])
    lows = np.array([c.low for c in self.candle_history])

    # Bollinger Bands
    period = 20
    if len(closes) >= period:
      sma = np.mean(closes[-period:])
      std = np.std(closes[-period:])

      bollinger_middle = sma
      bollinger_upper = sma + (2 * std)
      bollinger_lower = sma - (2 * std)
      bollinger_width = (bollinger_upper - bollinger_lower) / bollinger_middle

      # Position в полосах (0 = нижняя граница, 1 = верхняя граница)
      if bollinger_upper > bollinger_lower:
        bollinger_pct = (
            (closes[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower)
        )
      else:
        bollinger_pct = 0.5
    else:
      bollinger_middle = closes[-1]
      bollinger_upper = closes[-1]
      bollinger_lower = closes[-1]
      bollinger_width = 0.0
      bollinger_pct = 0.5

    # ATR (Average True Range)
    atr_14 = self._calculate_atr(highs, lows, closes, 14)

    # Keltner Channels
    ema_20 = self._calculate_ema(closes, 20)
    keltner_upper = ema_20 + (2 * atr_14)
    keltner_lower = ema_20 - (2 * atr_14)

    return {
      "bollinger_upper": bollinger_upper,
      "bollinger_middle": bollinger_middle,
      "bollinger_lower": bollinger_lower,
      "bollinger_width": bollinger_width,
      "bollinger_pct": bollinger_pct,
      "atr_14": atr_14,
      "keltner_upper": keltner_upper,
      "keltner_lower": keltner_lower
    }

  def _extract_volume_indicators(self) -> Dict[str, float]:
    """Извлечение volume индикаторов (6)"""

    closes = np.array([c.close for c in self.candle_history])
    highs = np.array([c.high for c in self.candle_history])
    lows = np.array([c.low for c in self.candle_history])
    volumes = np.array([c.volume for c in self.candle_history])

    # OBV (On-Balance Volume) - кумулятивный
    if len(self.candle_history) >= 2:
      prev_close = self.candle_history[-2].close
      current_close = self.candle_history[-1].close
      current_volume = self.candle_history[-1].volume

      if current_close > prev_close:
        self.obv_cumulative += current_volume
      elif current_close < prev_close:
        self.obv_cumulative -= current_volume

    obv = self.obv_cumulative

    # VWAP (Volume Weighted Average Price)
    if len(volumes) >= 20:
      typical_prices = (highs[-20:] + lows[-20:] + closes[-20:]) / 3
      vwap = float(np.sum(typical_prices * volumes[-20:]) / np.sum(volumes[-20:]))
    else:
      vwap = closes[-1]

    # A/D Line (Accumulation/Distribution)
    if len(self.candle_history) >= 2:
      high = highs[-1]
      low = lows[-1]
      close = closes[-1]
      volume = volumes[-1]

      if high != low:
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_volume = mf_multiplier * volume
        self.ad_cumulative += mf_volume

    ad_line = self.ad_cumulative

    # CMF (Chaikin Money Flow)
    cmf = self._calculate_cmf(highs, lows, closes, volumes)

    # VPT (Volume Price Trend) - кумулятивный
    if len(self.candle_history) >= 2:
      prev_close = self.candle_history[-2].close
      current_close = self.candle_history[-1].close
      current_volume = self.candle_history[-1].volume

      if prev_close > 0:
        price_change_pct = (current_close - prev_close) / prev_close
        self.vpt_cumulative += current_volume * price_change_pct

    vpt = self.vpt_cumulative

    # NVI (Negative Volume Index)
    if len(self.candle_history) >= 2:
      prev_volume = self.candle_history[-2].volume
      current_volume = self.candle_history[-1].volume
      prev_close = self.candle_history[-2].close
      current_close = self.candle_history[-1].close

      if current_volume < prev_volume and prev_close > 0:
        price_change = (current_close - prev_close) / prev_close
        self.nvi *= (1 + price_change)

    nvi = self.nvi

    return {
      "obv": obv,
      "vwap": vwap,
      "ad_line": ad_line,
      "cmf": cmf,
      "vpt": vpt,
      "nvi": nvi
    }

  # ===== Helper Methods =====

  @staticmethod
  def _calculate_ema(data: np.ndarray, period: int) -> float:
    """Вычисляет EMA (Exponential Moving Average)"""
    if len(data) < period:
      return float(data[-1])

    multiplier = 2 / (period + 1)
    ema = np.mean(data[:period])

    for price in data[period:]:
      ema = (price * multiplier) + (ema * (1 - multiplier))

    return float(ema)

  @staticmethod
  def _calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Вычисляет RSI (Relative Strength Index)"""
    if len(closes) < period + 1:
      return 50.0

    deltas = np.diff(closes[-period - 1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
      return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return float(rsi)

  @staticmethod
  def _calculate_williams_r(
      highs: np.ndarray,
      lows: np.ndarray,
      closes: np.ndarray,
      period: int = 14
  ) -> float:
    """Вычисляет Williams %R"""
    if len(closes) < period:
      return -50.0

    highest_high = np.max(highs[-period:])
    lowest_low = np.min(lows[-period:])

    if highest_high == lowest_low:
      return -50.0

    williams_r = (
        (highest_high - closes[-1]) / (highest_high - lowest_low) * -100
    )

    return float(williams_r)

  @staticmethod
  def _calculate_cci(
      highs: np.ndarray,
      lows: np.ndarray,
      closes: np.ndarray,
      period: int = 20
  ) -> float:
    """Вычисляет CCI (Commodity Channel Index)"""
    if len(closes) < period:
      return 0.0

    typical_prices = (highs[-period:] + lows[-period:] + closes[-period:]) / 3
    sma_tp = np.mean(typical_prices)
    mean_deviation = np.mean(np.abs(typical_prices - sma_tp))

    if mean_deviation == 0:
      return 0.0

    cci = (typical_prices[-1] - sma_tp) / (0.015 * mean_deviation)

    return float(cci)

  @staticmethod
  def _calculate_mfi(
      highs: np.ndarray,
      lows: np.ndarray,
      closes: np.ndarray,
      volumes: np.ndarray,
      period: int = 14
  ) -> float:
    """Вычисляет MFI (Money Flow Index)"""
    if len(closes) < period + 1:
      return 50.0

    typical_prices = (highs + lows + closes) / 3
    money_flows = typical_prices * volumes

    positive_flow = 0.0
    negative_flow = 0.0

    for i in range(-period, 0):
      if typical_prices[i] > typical_prices[i - 1]:
        positive_flow += money_flows[i]
      elif typical_prices[i] < typical_prices[i - 1]:
        negative_flow += money_flows[i]

    if negative_flow == 0:
      return 100.0

    money_ratio = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + money_ratio))

    return float(mfi)

  @staticmethod
  def _calculate_atr(
      highs: np.ndarray,
      lows: np.ndarray,
      closes: np.ndarray,
      period: int = 14
  ) -> float:
    """Вычисляет ATR (Average True Range)"""
    if len(closes) < period + 1:
      return 0.0

    true_ranges = []
    for i in range(1, len(closes)):
      high_low = float(highs[i] - lows[i])
      high_close = float(abs(highs[i] - closes[i - 1]))
      low_close = float(abs(lows[i] - closes[i - 1]))
      true_ranges.append(max(high_low, high_close, low_close))

    atr = np.mean(true_ranges[-period:])

    return float(atr)

  @staticmethod
  def _calculate_cmf(
      highs: np.ndarray,
      lows: np.ndarray,
      closes: np.ndarray,
      volumes: np.ndarray,
      period: int = 20
  ) -> float:
    """Вычисляет CMF (Chaikin Money Flow)"""
    if len(closes) < period:
      return 0.0

    mf_volumes = []
    for i in range(-period, 0):
      high = highs[i]
      low = lows[i]
      close = closes[i]
      volume = volumes[i]

      if high != low:
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_volumes.append(mf_multiplier * volume)
      else:
        mf_volumes.append(0)

    cmf = sum(mf_volumes) / sum(volumes[-period:]) if sum(volumes[-period:]) > 0 else 0.0

    return float(cmf)

  # ===== PROFESSIONAL INDICATOR METHODS (Industry-Standard) =====

  def _calculate_rsi_professional(
      self,
      closes: np.ndarray,
      rsi_state: RSIState
  ) -> float:
    """
    Professional RSI with Wilder's smoothing (industry standard).

    Wilder's smoothing:
    - First avg_gain/loss: SMA(period)
    - Subsequent: (prev_avg * (period - 1) + current_value) / period

    Args:
        closes: Close prices
        rsi_state: State object with avg_gain/avg_loss

    Returns:
        RSI value (0-100)
    """
    period = rsi_state.period

    if len(closes) < period + 1:
      return 50.0

    # Calculate price changes
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    current_gain = gains[-1]
    current_loss = losses[-1]

    # Initialize or update avg_gain/avg_loss with Wilder's smoothing
    if rsi_state.avg_gain is None or rsi_state.avg_loss is None:
      # First calculation: use SMA
      if len(gains) >= period:
        rsi_state.avg_gain = float(np.mean(gains[-period:]))
        rsi_state.avg_loss = float(np.mean(losses[-period:]))
      else:
        return 50.0
    else:
      # Wilder's smoothing: (prev_avg * (n-1) + current) / n
      rsi_state.avg_gain = (
        (rsi_state.avg_gain * (period - 1) + current_gain) / period
      )
      rsi_state.avg_loss = (
        (rsi_state.avg_loss * (period - 1) + current_loss) / period
      )

    # Calculate RSI
    if rsi_state.avg_loss == 0:
      return 100.0

    rs = rsi_state.avg_gain / rsi_state.avg_loss
    rsi = 100 - (100 / (1 + rs))

    return float(rsi)

  def _calculate_stochastic_professional(
      self,
      highs: np.ndarray,
      lows: np.ndarray,
      closes: np.ndarray,
      period: int = 14
  ) -> tuple:
    """
    Professional Stochastic Oscillator with %D = SMA(%K, 3).

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Lookback period (default 14)

    Returns:
        Tuple[float, float]: (%K, %D)
    """
    if len(closes) < period:
      return 50.0, 50.0

    # Calculate %K
    highest_high = np.max(highs[-period:])
    lowest_low = np.min(lows[-period:])

    if highest_high == lowest_low:
      stochastic_k = 50.0
    else:
      stochastic_k = (
        (closes[-1] - lowest_low) / (highest_high - lowest_low) * 100
      )

    # Store %K in history
    self.stochastic_state.k_history.append(stochastic_k)

    # Calculate %D = 3-period SMA of %K (PROFESSIONAL)
    if len(self.stochastic_state.k_history) >= 3:
      stochastic_d = float(np.mean(list(self.stochastic_state.k_history)))
    else:
      stochastic_d = stochastic_k  # Not enough history yet

    return float(stochastic_k), float(stochastic_d)

  def _calculate_adx_professional(
      self,
      highs: np.ndarray,
      lows: np.ndarray,
      closes: np.ndarray,
      period: int = 14
  ) -> tuple:
    """
    Professional ADX with Wilder's smoothing (industry standard).

    Wilder's smoothing for TR, +DM, -DM:
    - First: SMA(period)
    - Subsequent: (prev_smooth * (period - 1) + current) / period

    ADX = Wilder's smoothing of DX

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: ADX period (default 14)

    Returns:
        Tuple[float, float, float]: (ADX, +DI, -DI)
    """
    if len(closes) < period + 1:
      return 25.0, 25.0, 25.0

    # Calculate current TR, +DM, -DM
    high_diff = highs[-1] - highs[-2]
    low_diff = lows[-2] - lows[-1]

    # +DM
    if high_diff > low_diff and high_diff > 0:
      plus_dm_current = high_diff
    else:
      plus_dm_current = 0.0

    # -DM
    if low_diff > high_diff and low_diff > 0:
      minus_dm_current = low_diff
    else:
      minus_dm_current = 0.0

    # TR
    high_low = float(highs[-1] - lows[-1])
    high_close = float(abs(highs[-1] - closes[-2]))
    low_close = float(abs(lows[-1] - closes[-2]))
    tr_current = max(high_low, high_close, low_close)

    # Apply Wilder's smoothing to TR, +DM, -DM
    if self.adx_state.smooth_tr is None:
      # Initialize with SMA
      if len(closes) < period + 1:
        return 25.0, 25.0, 25.0

      # Calculate initial smoothed values
      tr_values = []
      plus_dm_values = []
      minus_dm_values = []

      for i in range(1, len(closes)):
        h_diff = highs[i] - highs[i-1]
        l_diff = lows[i-1] - lows[i]

        # +DM
        if h_diff > l_diff and h_diff > 0:
          plus_dm_values.append(h_diff)
        else:
          plus_dm_values.append(0)

        # -DM
        if l_diff > h_diff and l_diff > 0:
          minus_dm_values.append(l_diff)
        else:
          minus_dm_values.append(0)

        # TR
        hl = float(highs[i] - lows[i])
        hc = float(abs(highs[i] - closes[i-1]))
        lc = float(abs(lows[i] - closes[i-1]))
        tr_values.append(max(hl, hc, lc))

      if len(tr_values) >= period:
        self.adx_state.smooth_tr = float(np.mean(tr_values[-period:]))
        self.adx_state.smooth_plus_dm = float(np.mean(plus_dm_values[-period:]))
        self.adx_state.smooth_minus_dm = float(np.mean(minus_dm_values[-period:]))
      else:
        return 25.0, 25.0, 25.0
    else:
      # Wilder's smoothing: (prev * (n-1) + current) / n
      self.adx_state.smooth_tr = (
        (self.adx_state.smooth_tr * (period - 1) + tr_current) / period
      )
      self.adx_state.smooth_plus_dm = (
        (self.adx_state.smooth_plus_dm * (period - 1) + plus_dm_current) / period
      )
      self.adx_state.smooth_minus_dm = (
        (self.adx_state.smooth_minus_dm * (period - 1) + minus_dm_current) / period
      )

    # Calculate +DI, -DI
    if self.adx_state.smooth_tr > 0:
      plus_di = (self.adx_state.smooth_plus_dm / self.adx_state.smooth_tr) * 100
      minus_di = (self.adx_state.smooth_minus_dm / self.adx_state.smooth_tr) * 100
    else:
      plus_di = 25.0
      minus_di = 25.0

    # Calculate DX
    if (plus_di + minus_di) > 0:
      dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    else:
      dx = 25.0

    # Store DX in history
    self.adx_state.dx_history.append(dx)

    # Calculate ADX (Wilder's smoothing of DX)
    if len(self.adx_state.dx_history) >= period:
      if self.adx_state.adx_value is None:
        # Initialize with SMA
        self.adx_state.adx_value = float(np.mean(list(self.adx_state.dx_history)[-period:]))
      else:
        # Wilder's smoothing
        self.adx_state.adx_value = (
          (self.adx_state.adx_value * (period - 1) + dx) / period
        )
      adx = self.adx_state.adx_value
    else:
      adx = dx  # Not enough history yet

    return float(adx), float(plus_di), float(minus_di)

  def _extract_aroon_indicators(self) -> Dict[str, float]:
    """
    Извлечение Aroon Up/Down индикаторов (NEW).

    Aroon измеряет время с момента последнего максимума/минимума.
    - Aroon Up = ((period - periods_since_high) / period) * 100
    - Aroon Down = ((period - periods_since_low) / period) * 100

    Values:
    - 100 = недавний максимум/минимум (сильный тренд)
    - 0 = давно не было максимума/минимума

    Returns:
        Dict with aroon_up, aroon_down
    """
    highs = np.array([c.high for c in self.candle_history])
    lows = np.array([c.low for c in self.candle_history])

    period = 25  # Standard Aroon period

    if len(highs) < period:
      return {
        "aroon_up": 50.0,
        "aroon_down": 50.0
      }

    # Find periods since highest high and lowest low
    recent_highs = highs[-period:]
    recent_lows = lows[-period:]

    # Periods since highest high (0 = most recent)
    periods_since_high = period - 1 - np.argmax(recent_highs)

    # Periods since lowest low (0 = most recent)
    periods_since_low = period - 1 - np.argmin(recent_lows)

    # Calculate Aroon Up/Down
    aroon_up = ((period - periods_since_high) / period) * 100
    aroon_down = ((period - periods_since_low) / period) * 100

    return {
      "aroon_up": float(aroon_up),
      "aroon_down": float(aroon_down)
    }

  def _create_default_features(
      self,
      candle: Optional[Candle]
  ) -> IndicatorFeatures:
    """Создает признаки со значениями по умолчанию"""
    current_price = candle.close if candle else 0.0

    return IndicatorFeatures(
      symbol=self.symbol,
      timestamp=candle.timestamp if candle else 0,
      # Trend (12)
      sma_10=current_price, sma_20=current_price, sma_50=current_price,
      ema_10=current_price, ema_20=current_price, ema_50=current_price,
      macd=0.0, macd_signal=0.0, macd_histogram=0.0,
      adx=25.0, plus_di=25.0, minus_di=25.0,
      # Momentum (9)
      rsi_14=50.0, rsi_28=50.0,
      stochastic_k=50.0, stochastic_d=50.0,
      williams_r=-50.0, cci=0.0, momentum_10=0.0,
      roc=0.0, mfi=50.0,
      # Volatility (8)
      bollinger_upper=current_price, bollinger_middle=current_price,
      bollinger_lower=current_price, bollinger_width=0.0, bollinger_pct=0.5,
      atr_14=0.0, keltner_upper=current_price, keltner_lower=current_price,
      # Volume (6)
      obv=0.0, vwap=current_price, ad_line=0.0,
      cmf=0.0, vpt=0.0, nvi=1000.0,
      # Aroon (2) - NEW
      aroon_up=50.0, aroon_down=50.0
    )