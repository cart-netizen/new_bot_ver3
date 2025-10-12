"""
Indicator Feature Extractor для извлечения 35+ признаков из технических индикаторов.

Использует TA-Lib для вычисления индикаторов.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

from core.logger import get_logger
from ml_engine.features.candle_feature_extractor import Candle

logger = get_logger(__name__)


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
      self.cmf, self.vpt, self.nvi
    ], dtype=np.float32)

  def to_dict(self) -> Dict[str, float]:
    """Преобразование в словарь"""
    return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class IndicatorFeatureExtractor:
  """
  Извлекает 35+ признаков из технических индикаторов.

  Использует numpy для вычислений (без зависимости от TA-Lib для упрощения).
  Можно заменить на TA-Lib в production для более точных расчетов.
  """

  def __init__(self, symbol: str):
    """
    Args:
        symbol: Торговая пара
    """
    self.symbol = symbol

    # История для индикаторов
    self.candle_history: List[Candle] = []
    self.max_history_size = 200

    # Кумулятивные индикаторы
    self.obv_cumulative = 0.0
    self.ad_cumulative = 0.0
    self.vpt_cumulative = 0.0
    self.nvi = 1000.0  # Начальное значение NVI

    logger.info(f"IndicatorFeatureExtractor инициализирован для {symbol}")

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

      # Объединяем
      features = IndicatorFeatures(
        symbol=self.symbol,
        timestamp=current_candle.timestamp,
        **trend_features,
        **momentum_features,
        **volatility_features,
        **volume_features
      )

      logger.info(
        f"{self.symbol} | Извлечено 35 индикаторов, "
        f"RSI={features.rsi_14:.2f}, MACD={features.macd:.4f}"
      )

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

    # MACD Signal (9-period EMA of MACD)
    # Упрощенная версия - используем SMA вместо EMA для signal
    if len(self.candle_history) >= 35:
      macd_values = []
      for i in range(26, len(closes)):
        ema12_i = self._calculate_ema(closes[:i + 1], 12)
        ema26_i = self._calculate_ema(closes[:i + 1], 26)
        macd_values.append(ema12_i - ema26_i)

      if len(macd_values) >= 9:
        macd_signal = float(np.mean(macd_values[-9:]))
      else:
        macd_signal = macd
    else:
      macd_signal = macd

    macd_histogram = macd - macd_signal

    # ADX (Average Directional Index) - упрощенная версия
    adx, plus_di, minus_di = self._calculate_adx(highs, lows, closes)

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

    # RSI (Relative Strength Index)
    rsi_14 = self._calculate_rsi(closes, 14)
    rsi_28 = self._calculate_rsi(closes, 28)

    # Stochastic Oscillator
    stochastic_k, stochastic_d = self._calculate_stochastic(highs, lows, closes)

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
  def _calculate_stochastic(
      highs: np.ndarray,
      lows: np.ndarray,
      closes: np.ndarray,
      period: int = 14
  ) -> tuple:
    """Вычисляет Stochastic Oscillator (%K, %D)"""
    if len(closes) < period:
      return 50.0, 50.0

    highest_high = np.max(highs[-period:])
    lowest_low = np.min(lows[-period:])

    if highest_high == lowest_low:
      stochastic_k = 50.0
    else:
      stochastic_k = (
          (closes[-1] - lowest_low) / (highest_high - lowest_low) * 100
      )

    # %D = 3-period SMA of %K (упрощение)
    stochastic_d = stochastic_k  # В полной версии нужна история %K

    return float(stochastic_k), float(stochastic_d)

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
      high_low = highs[i] - lows[i]
      high_close = abs(highs[i] - closes[i - 1])
      low_close = abs(lows[i] - closes[i - 1])
      true_ranges.append(max(high_low, high_close, low_close))

    atr = np.mean(true_ranges[-period:])

    return float(atr)

  @staticmethod
  def _calculate_adx(
      highs: np.ndarray,
      lows: np.ndarray,
      closes: np.ndarray,
      period: int = 14
  ) -> tuple:
    """Вычисляет ADX, +DI, -DI (упрощенная версия)"""
    if len(closes) < period + 1:
      return 25.0, 25.0, 25.0

    # True Range
    tr_values = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(closes)):
      high_diff = highs[i] - highs[i - 1]
      low_diff = lows[i - 1] - lows[i]

      # +DM
      if high_diff > low_diff and high_diff > 0:
        plus_dm.append(high_diff)
      else:
        plus_dm.append(0)

      # -DM
      if low_diff > high_diff and low_diff > 0:
        minus_dm.append(low_diff)
      else:
        minus_dm.append(0)

      # TR
      high_low = highs[i] - lows[i]
      high_close = abs(highs[i] - closes[i - 1])
      low_close = abs(lows[i] - closes[i - 1])
      tr_values.append(max(high_low, high_close, low_close))

    # Сглаживание
    smooth_tr = np.mean(tr_values[-period:])
    smooth_plus_dm = np.mean(plus_dm[-period:])
    smooth_minus_dm = np.mean(minus_dm[-period:])

    # DI
    if smooth_tr > 0:
      plus_di = (smooth_plus_dm / smooth_tr) * 100
      minus_di = (smooth_minus_dm / smooth_tr) * 100
    else:
      plus_di = 25.0
      minus_di = 25.0

    # DX и ADX
    if (plus_di + minus_di) > 0:
      dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
      adx = dx  # Упрощение: в полной версии ADX = EMA(DX)
    else:
      adx = 25.0

    return float(adx), float(plus_di), float(minus_di)

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
      cmf=0.0, vpt=0.0, nvi=1000.0
    )