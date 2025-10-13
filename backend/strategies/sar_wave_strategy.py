"""
SAR Wave Strategy - торговля на основе Parabolic SAR и волнового анализа.

Методология:
- Parabolic SAR для определения направления тренда
- Wave detection для входов на откатах
- ADX для фильтрации силы тренда
- Volume profile для подтверждения
- Fibonacci retracement levels

Путь: backend/strategies/sar_wave_strategy.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime

from core.logger import get_logger
from models.signal import TradingSignal, SignalType, SignalStrength, SignalSource
from strategy.candle_manager import Candle

logger = get_logger(__name__)


@dataclass
class SARWaveConfig:
  """Конфигурация SAR Wave стратегии."""
  # Parabolic SAR параметры
  sar_acceleration: float = 0.02  # Начальное ускорение
  sar_max_acceleration: float = 0.2  # Максимальное ускорение

  # ADX параметры (фильтр силы тренда)
  adx_period: int = 14
  adx_threshold: float = 25.0  # ADX > 25 = сильный тренд

  # Wave detection
  swing_detection_period: int = 5  # Период для определения swing points
  min_wave_amplitude_pct: float = 0.5  # Минимальная амплитуда волны 0.5%

  # Fibonacci levels для входов
  fib_entry_levels: List[float] = None  # [0.382, 0.5, 0.618]
  fib_tolerance_pct: float = 0.2  # Допуск ±0.2% от уровня

  # Risk management
  stop_loss_pct: float = 1.5
  take_profit_pct: float = 4.5  # 3:1 risk/reward

  def __post_init__(self):
    if self.fib_entry_levels is None:
      self.fib_entry_levels = [0.382, 0.5, 0.618]


@dataclass
class SwingPoint:
  """Точка разворота (swing high/low)."""
  index: int
  price: float
  timestamp: int
  swing_type: str  # "high" или "low"


@dataclass
class Wave:
  """Волна (от swing к swing)."""
  start: SwingPoint
  end: SwingPoint
  amplitude: float  # В процентах
  direction: str  # "up" или "down"

  def get_fibonacci_levels(self) -> Dict[float, float]:
    """Вычислить Fibonacci retracement уровни."""
    diff = self.end.price - self.start.price

    levels = {}
    for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]:
      level_price = self.end.price - (diff * ratio)
      levels[ratio] = level_price

    return levels


class ParabolicSAR:
  """
  Parabolic SAR (Stop and Reverse) индикатор.

  SAR показывает потенциальные точки разворота тренда.
  """

  def __init__(
      self,
      acceleration: float = 0.02,
      max_acceleration: float = 0.2
  ):
    """
    Args:
        acceleration: Начальный фактор ускорения
        max_acceleration: Максимальный фактор ускорения
    """
    self.acceleration = acceleration
    self.max_acceleration = max_acceleration

  def calculate(
      self,
      highs: np.ndarray,
      lows: np.ndarray,
      closes: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычислить Parabolic SAR.

    Returns:
        (sar_values, trend_direction)
        trend_direction: 1 = uptrend, -1 = downtrend
    """
    n = len(closes)
    sar = np.zeros(n)
    trend = np.zeros(n)
    ep = np.zeros(n)  # Extreme Point
    af = np.zeros(n)  # Acceleration Factor

    # Инициализация
    sar[0] = lows[0]
    trend[0] = 1  # Начинаем с uptrend
    ep[0] = highs[0]
    af[0] = self.acceleration

    for i in range(1, n):
      # Предыдущие значения
      prev_sar = sar[i - 1]
      prev_trend = trend[i - 1]
      prev_ep = ep[i - 1]
      prev_af = af[i - 1]

      # Вычисляем новый SAR
      sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)

      # Uptrend
      if prev_trend == 1:
        # SAR не должен быть выше low предыдущих 2 свечей
        sar[i] = float(np.minimum(sar[i], lows[i - 1]))
        if i > 1:
          sar[i] = float(np.minimum(sar[i], lows[i - 2]))

        # Проверяем разворот
        if lows[i] <= sar[i]:
          # Разворот на downtrend
          trend[i] = -1
          sar[i] = prev_ep  # SAR становится предыдущим EP
          ep[i] = lows[i]
          af[i] = self.acceleration
        else:
          # Продолжаем uptrend
          trend[i] = 1

          # Обновляем EP если новый high
          if highs[i] > prev_ep:
            ep[i] = highs[i]
            af[i] = float(np.minimum(prev_af + self.acceleration, self.max_acceleration))
          else:
            ep[i] = prev_ep
            af[i] = prev_af

      # Downtrend
      else:
        # SAR не должен быть ниже high предыдущих 2 свечей
        sar[i] = float(np.maximum(sar[i], highs[i - 1]))
        if i > 1:
          sar[i] = float(np.maximum(sar[i], highs[i - 2]))

        # Проверяем разворот
        if highs[i] >= sar[i]:
          # Разворот на uptrend
          trend[i] = 1
          sar[i] = prev_ep  # SAR становится предыдущим EP
          ep[i] = highs[i]
          af[i] = self.acceleration
        else:
          # Продолжаем downtrend
          trend[i] = -1

          # Обновляем EP если новый low
          if lows[i] < prev_ep:
            ep[i] = lows[i]
            af[i] = float(np.minimum(prev_af + self.acceleration, self.max_acceleration))
          else:
            ep[i] = prev_ep
            af[i] = prev_af

    return sar, trend


class ADXIndicator:
  """Average Directional Index - измерение силы тренда."""

  @staticmethod
  def calculate(
      highs: np.ndarray,
      lows: np.ndarray,
      closes: np.ndarray,
      period: int = 14
  ) -> np.ndarray:
    """
    Вычислить ADX.

    ADX показывает силу тренда (не направление).
    ADX > 25 = сильный тренд
    ADX < 20 = слабый тренд / флэт
    """
    n = len(closes)

    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
      high_low = highs[i] - lows[i]
      high_close = abs(highs[i] - closes[i - 1])
      low_close = abs(lows[i] - closes[i - 1])
      tr[i] = float(np.maximum(np.maximum(high_low, high_close), low_close))

    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
      high_diff = highs[i] - highs[i - 1]
      low_diff = lows[i - 1] - lows[i]

      if high_diff > low_diff and high_diff > 0:
        plus_dm[i] = high_diff

      if low_diff > high_diff and low_diff > 0:
        minus_dm[i] = low_diff

    # Smoothed TR, +DM, -DM
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)

    # Первое значение - простое среднее
    atr[period] = np.mean(tr[1:period + 1])
    plus_di[period] = np.mean(plus_dm[1:period + 1])
    minus_di[period] = np.mean(minus_dm[1:period + 1])

    # Экспоненциальное сглаживание
    for i in range(period + 1, n):
      atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
      plus_di[i] = (plus_di[i - 1] * (period - 1) + plus_dm[i]) / period
      minus_di[i] = (minus_di[i - 1] * (period - 1) + minus_dm[i]) / period

    # Directional Indicators
    plus_di_pct = np.zeros(n)
    minus_di_pct = np.zeros(n)

    for i in range(period, n):
      if atr[i] != 0:
        plus_di_pct[i] = 100 * plus_di[i] / atr[i]
        minus_di_pct[i] = 100 * minus_di[i] / atr[i]

    # DX (Directional Index)
    dx = np.zeros(n)
    for i in range(period, n):
      di_sum = plus_di_pct[i] + minus_di_pct[i]
      if di_sum != 0:
        dx[i] = 100 * abs(plus_di_pct[i] - minus_di_pct[i]) / di_sum

    # ADX (сглаженный DX)
    adx = np.zeros(n)
    adx[period * 2] = np.mean(dx[period:period * 2 + 1])

    for i in range(period * 2 + 1, n):
      adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


class SARWaveStrategy:
  """
  SAR Wave Trading Strategy.

  Логика:
  1. Используем Parabolic SAR для определения тренда
  2. Фильтруем через ADX (только сильные тренды)
  3. Определяем swing points и волны
  4. Входим на откатах к Fibonacci уровням
  5. Stop loss под/над SAR
  """

  def __init__(self, config: SARWaveConfig):
    """Инициализация стратегии."""
    self.config = config

    self.sar_indicator = ParabolicSAR(
      acceleration=config.sar_acceleration,
      max_acceleration=config.sar_max_acceleration
    )

    # Состояние
    self.swing_points: Dict[str, List[SwingPoint]] = {}
    self.waves: Dict[str, List[Wave]] = {}
    self.active_signals: Dict[str, TradingSignal] = {}

    # Статистика
    self.signals_generated = 0

    logger.info(
      f"Инициализирована SARWaveStrategy: "
      f"sar_accel={config.sar_acceleration}, "
      f"adx_threshold={config.adx_threshold}"
    )

  def _find_swing_points(
      self,
      candles: List[Candle],
      period: int
  ) -> List[SwingPoint]:
    """Найти swing high и swing low точки."""
    swing_points = []
    n = len(candles)

    for i in range(period, n - period):
      # Swing High
      is_swing_high = True
      for j in range(i - period, i + period + 1):
        if j != i and candles[j].high >= candles[i].high:
          is_swing_high = False
          break

      if is_swing_high:
        swing_points.append(SwingPoint(
          index=i,
          price=candles[i].high,
          timestamp=candles[i].timestamp,
          swing_type="high"
        ))

      # Swing Low
      is_swing_low = True
      for j in range(i - period, i + period + 1):
        if j != i and candles[j].low <= candles[i].low:
          is_swing_low = False
          break

      if is_swing_low:
        swing_points.append(SwingPoint(
          index=i,
          price=candles[i].low,
          timestamp=candles[i].timestamp,
          swing_type="low"
        ))

    # Сортируем по индексу
    swing_points.sort(key=lambda x: x.index)

    return swing_points

  def _create_waves(
      self,
      swing_points: List[SwingPoint]
  ) -> List[Wave]:
    """Создать волны из swing points."""
    waves = []

    for i in range(len(swing_points) - 1):
      start = swing_points[i]
      end = swing_points[i + 1]

      # Волна должна чередоваться high-low-high или low-high-low
      if start.swing_type == end.swing_type:
        continue

      # Вычисляем амплитуду
      amplitude = abs(end.price - start.price) / start.price * 100

      # Фильтруем слишком маленькие волны
      if amplitude < self.config.min_wave_amplitude_pct:
        continue

      # Определяем направление
      if start.swing_type == "low" and end.swing_type == "high":
        direction = "up"
      else:
        direction = "down"

      wave = Wave(
        start=start,
        end=end,
        amplitude=amplitude,
        direction=direction
      )

      waves.append(wave)

    return waves

  def analyze(
      self,
      symbol: str,
      candles: List[Candle],
      current_price: float
  ) -> Optional[TradingSignal]:
    """Анализ и генерация сигнала."""
    min_candles = max(
      self.config.adx_period * 3,
      self.config.swing_detection_period * 3
    )

    if len(candles) < min_candles:
      return None

    # Извлекаем данные
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    closes = np.array([c.close for c in candles])

    # Вычисляем Parabolic SAR
    sar_values, sar_trend = self.sar_indicator.calculate(highs, lows, closes)

    current_sar = sar_values[-1]
    current_trend = sar_trend[-1]  # 1 = up, -1 = down

    # Вычисляем ADX
    adx = ADXIndicator.calculate(highs, lows, closes, self.config.adx_period)
    current_adx = adx[-1]

    # Фильтр: только сильные тренды
    if current_adx < self.config.adx_threshold:
      return None

    # Определяем swing points
    swing_points = self._find_swing_points(
      candles,
      self.config.swing_detection_period
    )

    self.swing_points[symbol] = swing_points

    # Создаем волны
    waves = self._create_waves(swing_points)
    self.waves[symbol] = waves

    if not waves:
      return None

    # Берем последнюю завершенную волну
    last_wave = waves[-1]

    # Проверяем условия входа
    signal_type = None
    reason_parts = []
    entry_level = None

    # LONG условия: uptrend по SAR + откат в волне down
    if current_trend == 1 and last_wave.direction == "down":
      # Вычисляем Fibonacci levels для волны
      fib_levels = last_wave.get_fibonacci_levels()

      # Проверяем близость к Fibonacci уровням
      for fib_ratio in self.config.fib_entry_levels:
        if fib_ratio not in fib_levels:
          continue

        fib_price = fib_levels[fib_ratio]
        distance_pct = abs(current_price - fib_price) / fib_price * 100

        if distance_pct <= self.config.fib_tolerance_pct:
          signal_type = SignalType.BUY
          entry_level = fib_ratio
          reason_parts.append(f"Uptrend (SAR), ADX={current_adx:.1f}")
          reason_parts.append(f"Pullback to Fib {fib_ratio:.3f} level (${fib_price:.2f})")
          reason_parts.append(f"Wave amplitude: {last_wave.amplitude:.2f}%")
          break

    # SHORT условия: downtrend по SAR + откат в волне up
    elif current_trend == -1 and last_wave.direction == "up":
      fib_levels = last_wave.get_fibonacci_levels()

      for fib_ratio in self.config.fib_entry_levels:
        if fib_ratio not in fib_levels:
          continue

        fib_price = fib_levels[fib_ratio]
        distance_pct = abs(current_price - fib_price) / fib_price * 100

        if distance_pct <= self.config.fib_tolerance_pct:
          signal_type = SignalType.SELL
          entry_level = fib_ratio
          reason_parts.append(f"Downtrend (SAR), ADX={current_adx:.1f}")
          reason_parts.append(f"Pullback to Fib {fib_ratio:.3f} level (${fib_price:.2f})")
          reason_parts.append(f"Wave amplitude: {last_wave.amplitude:.2f}%")
          break

    if signal_type is None:
      return None

    # Вычисляем confidence на основе ADX и wave amplitude
    adx_component = float(np.minimum((current_adx - self.config.adx_threshold) / 25.0, 1.0))
    wave_component = min(last_wave.amplitude / 5.0, 1.0)  # Нормализуем до 5%

    confidence = (adx_component * 0.6 + wave_component * 0.4)

    # Сила сигнала
    if confidence > 0.7:
      signal_strength = SignalStrength.STRONG
    elif confidence > 0.5:
      signal_strength = SignalStrength.MEDIUM
    else:
      signal_strength = SignalStrength.WEAK

    # Создаем сигнал
    signal = TradingSignal(
      symbol=symbol,
      signal_type=signal_type,
      source=SignalSource.STRATEGY,
      strength=signal_strength,
      price=current_price,
      confidence=confidence,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason=" | ".join(reason_parts),
      metadata={
        'strategy': 'sar_wave',
        'sar_value': current_sar,
        'sar_trend': 'uptrend' if current_trend == 1 else 'downtrend',
        'adx': current_adx,
        'wave_amplitude': last_wave.amplitude,
        'fib_entry_level': entry_level,
        'stop_loss_pct': self.config.stop_loss_pct,
        'take_profit_pct': self.config.take_profit_pct
      }
    )

    self.active_signals[symbol] = signal
    self.signals_generated += 1

    logger.info(
      f"🌊 SAR WAVE SIGNAL [{symbol}]: {signal_type.value}, "
      f"confidence={confidence:.2f}, "
      f"ADX={current_adx:.1f}, Fib={entry_level}"
    )

    return signal

  def get_stop_loss_price(
      self,
      symbol: str,
      entry_price: float,
      position_side: str
  ) -> float:
    """Вычислить цену stop loss на основе SAR."""
    # Базовый stop loss
    if position_side == "long":
      stop_loss = entry_price * (1 - self.config.stop_loss_pct / 100)
    else:
      stop_loss = entry_price * (1 + self.config.stop_loss_pct / 100)

    return stop_loss

  def get_statistics(self) -> Dict:
    """Получить статистику стратегии."""
    total_waves = sum(len(waves) for waves in self.waves.values())

    return {
      'strategy': 'sar_wave',
      'signals_generated': self.signals_generated,
      'active_signals': len(self.active_signals),
      'symbols_tracked': len(self.waves),
      'total_waves_detected': total_waves
    }


# Пример использования
if __name__ == "__main__":
  import random

  config = SARWaveConfig(
    sar_acceleration=0.02,
    adx_threshold=25.0,
    swing_detection_period=5
  )

  strategy = SARWaveStrategy(config)

  # Генерируем тестовые свечи
  base_price = 50000.0
  candles = []

  # Создаем волновое движение
  for i in range(100):
    wave_phase = i / 10.0
    wave = np.sin(wave_phase) * 500
    trend = i * 5
    noise = random.uniform(-50, 50)

    price = base_price + trend + wave + noise

    candle = Candle(
      timestamp=int(datetime.now().timestamp() * 1000) + i * 60000,
      open=price - 5,
      high=price + abs(random.uniform(10, 30)),
      low=price - abs(random.uniform(10, 30)),
      close=price,
      volume=1000 + random.uniform(-200, 200)
    )
    candles.append(candle)

  # Анализируем
  signal = strategy.analyze("BTCUSDT", candles, candles[-1].close)

  if signal:
    print(f"Signal: {signal.signal_type.value}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Reason: {signal.reason}")
    print(f"Metadata: {signal.metadata}")
  else:
    print("No signal")

  stats = strategy.get_statistics()
  print(f"\nStatistics: {stats}")