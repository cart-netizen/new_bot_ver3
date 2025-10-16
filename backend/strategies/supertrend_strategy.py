"""
SuperTrend Strategy - торговля на основе SuperTrend индикатора.

Методология:
- SuperTrend (ATR-based) для определения тренда
- Multi-timeframe analysis для подтверждения
- Volume and momentum confirmation
- Adaptive stops на основе волатильности

Путь: backend/strategies/supertrend_strategy.py
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
class SuperTrendConfig:
  """Конфигурация SuperTrend стратегии."""
  # SuperTrend параметры
  atr_period: int = 10  # Период для ATR
  atr_multiplier: float = 3.0  # Множитель для bands

  # Confirmation filters
  use_momentum_filter: bool = True
  momentum_period: int = 14
  momentum_threshold: float = 0.0  # Momentum > 0 для long

  use_volume_filter: bool = True
  volume_ma_period: int = 20
  volume_threshold: float = 1.1  # 1.1x средний объем

  # Risk management
  stop_loss_atr_multiplier: float = 2.0  # Stop loss = ATR * 2
  take_profit_atr_multiplier: float = 6.0  # Take profit = ATR * 6
  use_trailing_stop: bool = True
  trailing_stop_atr_multiplier: float = 1.5

  # Signal strength thresholds
  strong_signal_bars: int = 3  # Тренд > N баров = сильный сигнал


class SuperTrendIndicator:
  """
  SuperTrend индикатор.

  SuperTrend = базовая линия ± (ATR × множитель)
  Пересечение цены и SuperTrend = смена тренда
  """

  @staticmethod
  def calculate_atr(
      highs: np.ndarray,
      lows: np.ndarray,
      closes: np.ndarray,
      period: int
  ) -> np.ndarray:
    """Вычислить Average True Range."""
    n = len(closes)
    tr = np.zeros(n)

    for i in range(1, n):
      high_low = highs[i] - lows[i]
      high_close = abs(highs[i] - closes[i - 1])
      low_close = abs(lows[i] - closes[i - 1])
      tr[i] = float(np.maximum(np.maximum(high_low, high_close), low_close))

    # ATR = EMA of TR
    atr = np.zeros(n)
    atr[period] = np.mean(tr[1:period + 1])

    alpha = 2.0 / (period + 1)
    for i in range(period + 1, n):
      atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    return atr

  @staticmethod
  def calculate(
      highs: np.ndarray,
      lows: np.ndarray,
      closes: np.ndarray,
      period: int = 10,
      multiplier: float = 3.0
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычислить SuperTrend.

    Returns:
        (supertrend_values, trend_direction)
        trend_direction: 1 = uptrend, -1 = downtrend
    """
    n = len(closes)

    # Вычисляем ATR
    atr = SuperTrendIndicator.calculate_atr(highs, lows, closes, period)

    # Базовая линия (HL/2)
    hl_avg = (highs + lows) / 2

    # Upper и Lower bands
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)

    # SuperTrend и направление тренда
    supertrend = np.zeros(n)
    trend = np.zeros(n)

    # Инициализация
    supertrend[0] = upper_band[0]
    trend[0] = 1

    for i in range(1, n):
      # Uptrend
      if trend[i - 1] == 1:
        # Обновляем lower band (не может уменьшаться)
        if lower_band[i] > supertrend[i - 1]:
          supertrend[i] = lower_band[i]
        else:
          supertrend[i] = supertrend[i - 1]

        # Проверяем разворот
        if closes[i] <= supertrend[i]:
          trend[i] = -1
          supertrend[i] = upper_band[i]
        else:
          trend[i] = 1

      # Downtrend
      else:
        # Обновляем upper band (не может увеличиваться)
        if upper_band[i] < supertrend[i - 1]:
          supertrend[i] = upper_band[i]
        else:
          supertrend[i] = supertrend[i - 1]

        # Проверяем разворот
        if closes[i] >= supertrend[i]:
          trend[i] = 1
          supertrend[i] = lower_band[i]
        else:
          trend[i] = -1

    return supertrend, trend


class SuperTrendStrategy:
  """
  SuperTrend Trading Strategy.

  Логика:
  1. SuperTrend определяет направление тренда
  2. Входим при смене тренда (crossover)
  3. Фильтруем через momentum и volume
  4. Stop loss на основе ATR
  5. Trailing stop для защиты прибыли
  """

  def __init__(self, config: SuperTrendConfig):
    """Инициализация стратегии."""
    self.config = config

    # Состояние
    self.active_signals: Dict[str, TradingSignal] = {}
    self.trend_history: Dict[str, List[int]] = {}  # История тренда
    self.entry_atr: Dict[str, float] = {}  # ATR при входе

    # Статистика
    self.signals_generated = 0
    self.trend_changes = 0

    logger.info(
      f"Инициализирована SuperTrendStrategy: "
      f"atr_period={config.atr_period}, "
      f"multiplier={config.atr_multiplier}"
    )

  def _calculate_momentum(
      self,
      closes: np.ndarray,
      period: int
  ) -> float:
    """Вычислить momentum."""
    if len(closes) < period + 1:
      return 0.0

    return float(closes[-1] - closes[-period-1])

  def analyze(
      self,
      symbol: str,
      candles: List[Candle],
      current_price: float
  ) -> Optional[TradingSignal]:
    """Анализ и генерация сигнала."""
    min_candles = max(
      self.config.atr_period * 2,
      self.config.momentum_period if self.config.use_momentum_filter else 0,
      self.config.volume_ma_period if self.config.use_volume_filter else 0
    ) + 10

    if len(candles) < min_candles:
      return None

    # Извлекаем данные
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    closes = np.array([c.close for c in candles])
    volumes = np.array([c.volume for c in candles])

    # Вычисляем SuperTrend
    supertrend, trend = SuperTrendIndicator.calculate(
      highs,
      lows,
      closes,
      self.config.atr_period,
      self.config.atr_multiplier
    )

    current_supertrend = supertrend[-1]
    current_trend = trend[-1]  # 1 = up, -1 = down
    previous_trend = trend[-2] if len(trend) > 1 else current_trend

    # Сохраняем историю тренда
    if symbol not in self.trend_history:
      self.trend_history[symbol] = []

    self.trend_history[symbol].append(int(current_trend))

    # Ограничиваем историю
    if len(self.trend_history[symbol]) > 100:
      self.trend_history[symbol] = self.trend_history[symbol][-100:]

    # Проверяем смену тренда (crossover)
    trend_changed = (current_trend != previous_trend)

    if not trend_changed:
      return None

    self.trend_changes += 1

    # Определяем направление сигнала
    if current_trend == 1:
      signal_type = SignalType.BUY
    else:
      signal_type = SignalType.SELL

    reason_parts = []
    reason_parts.append(
      f"SuperTrend crossover: {'Uptrend' if current_trend == 1 else 'Downtrend'}"
    )
    reason_parts.append(f"SuperTrend level: ${current_supertrend:.2f}")

    # Фильтр: Momentum
    if self.config.use_momentum_filter:
      momentum = self._calculate_momentum(closes, self.config.momentum_period)

      # Для long: momentum должен быть положительным
      if signal_type == SignalType.BUY and momentum <= self.config.momentum_threshold:
        logger.debug(
          f"Signal filtered by momentum: {momentum:.2f} <= {self.config.momentum_threshold}"
        )
        return None

      # Для short: momentum должен быть отрицательным
      if signal_type == SignalType.SELL and momentum >= -self.config.momentum_threshold:
        logger.debug(
          f"Signal filtered by momentum: {momentum:.2f} >= {-self.config.momentum_threshold}"
        )
        return None

      reason_parts.append(f"Momentum: {momentum:.2f}")

    # Фильтр: Volume
    if self.config.use_volume_filter:
      volume_ma = np.mean(volumes[-self.config.volume_ma_period:])
      current_volume = volumes[-1]
      volume_ratio = current_volume / volume_ma if volume_ma > 0 else 0

      if volume_ratio < self.config.volume_threshold:
        logger.debug(
          f"Signal filtered by volume: {volume_ratio:.2f} < {self.config.volume_threshold}"
        )
        return None

      reason_parts.append(f"Volume: {volume_ratio:.2f}x average")

    # Вычисляем силу сигнала
    # На основе продолжительности предыдущего тренда
    prev_trend_bars = 0
    for i in range(len(self.trend_history[symbol]) - 1, -1, -1):
      if self.trend_history[symbol][i] == previous_trend:
        prev_trend_bars += 1
      else:
        break

    confidence = min(prev_trend_bars / self.config.strong_signal_bars, 1.0)

    if confidence > 0.7:
      signal_strength = SignalStrength.STRONG
    elif confidence > 0.5:
      signal_strength = SignalStrength.MEDIUM
    else:
      signal_strength = SignalStrength.WEAK

    # Вычисляем ATR для risk management
    atr = SuperTrendIndicator.calculate_atr(
      highs,
      lows,
      closes,
      self.config.atr_period
    )
    current_atr = float(atr[-1])

    # Сохраняем ATR для использования в trailing stop
    self.entry_atr[symbol] = current_atr

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
        'strategy': 'supertrend',
        'supertrend_value': current_supertrend,
        'trend': 'uptrend' if current_trend == 1 else 'downtrend',
        'prev_trend_bars': prev_trend_bars,
        'atr': current_atr,
        'stop_loss_distance': current_atr * self.config.stop_loss_atr_multiplier,
        'take_profit_distance': current_atr * self.config.take_profit_atr_multiplier
      }
    )

    self.active_signals[symbol] = signal
    self.signals_generated += 1

    logger.info(
      f"📈 SUPERTREND SIGNAL [{symbol}]: {signal_type.value}, "
      f"confidence={confidence:.2f}, "
      f"SuperTrend=${current_supertrend:.2f}, "
      f"ATR=${current_atr:.2f}"
    )

    return signal

  def get_stop_loss_price(
      self,
      symbol: str,
      entry_price: float,
      position_side: str
  ) -> Optional[float]:
    """Вычислить цену stop loss на основе ATR."""
    if symbol not in self.entry_atr:
      return None

    atr = self.entry_atr[symbol]
    stop_distance = atr * self.config.stop_loss_atr_multiplier

    if position_side == "long":
      return entry_price - stop_distance
    else:
      return entry_price + stop_distance

  def get_take_profit_price(
      self,
      symbol: str,
      entry_price: float,
      position_side: str
  ) -> Optional[float]:
    """Вычислить цену take profit на основе ATR."""
    if symbol not in self.entry_atr:
      return None

    atr = self.entry_atr[symbol]
    tp_distance = atr * self.config.take_profit_atr_multiplier

    if position_side == "long":
      return entry_price + tp_distance
    else:
      return entry_price - tp_distance

  def should_trail_stop(
      self,
      symbol: str,
      entry_price: float,
      current_price: float,
      position_side: str
  ) -> Optional[float]:
    """Вычислить новую цену trailing stop."""
    if not self.config.use_trailing_stop:
      return None

    if symbol not in self.entry_atr:
      return None

    atr = self.entry_atr[symbol]
    trail_distance = atr * self.config.trailing_stop_atr_multiplier

    if position_side == "long":
      # Для long: trailing stop под текущей ценой
      return current_price - trail_distance
    else:
      # Для short: trailing stop над текущей ценой
      return current_price + trail_distance

  def get_statistics(self) -> Dict:
    """Получить статистику стратегии."""
    return {
      'strategy': 'supertrend',
      'signals_generated': self.signals_generated,
      'trend_changes': self.trend_changes,
      'active_signals': len(self.active_signals),
      'symbols_tracked': len(self.trend_history)
    }


# Пример использования
if __name__ == "__main__":
  import random

  config = SuperTrendConfig(
    atr_period=10,
    atr_multiplier=3.0,
    use_momentum_filter=True,
    use_volume_filter=True
  )

  strategy = SuperTrendStrategy(config)

  # Генерируем тестовые свечи с трендом
  base_price = 50000.0
  candles = []

  for i in range(150):
    # Создаем тренд с разворотом
    if i < 75:
      trend = i * 10  # Uptrend
    else:
      trend = (150 - i) * 10  # Downtrend

    noise = random.uniform(-100, 100)
    price = base_price + trend + noise

    volatility = random.uniform(50, 150)

    candle = Candle(
      timestamp=int(datetime.now().timestamp() * 1000) + i * 60000,
      open=price - 5,
      high=price + volatility,
      low=price - volatility,
      close=price,
      volume=1000 + random.uniform(-200, 200)
    )
    candles.append(candle)

  # Анализируем несколько точек
  for i in [50, 80, 120]:
    signal = strategy.analyze("BTCUSDT", candles[:i], candles[i - 1].close)

    if signal:
      print(f"\n[Bar {i}] Signal: {signal.signal_type.value}")
      print(f"  Confidence: {signal.confidence:.2f}")
      print(f"  Reason: {signal.reason}")

      # Stop loss и take profit
      stop_loss = strategy.get_stop_loss_price(
        "BTCUSDT",
        signal.price,
        "long" if signal.signal_type == SignalType.BUY else "short"
      )
      take_profit = strategy.get_take_profit_price(
        "BTCUSDT",
        signal.price,
        "long" if signal.signal_type == SignalType.BUY else "short"
      )

      print(f"  Entry: ${signal.price:.2f}")
      print(f"  Stop Loss: ${stop_loss:.2f}")
      print(f"  Take Profit: ${take_profit:.2f}")

  stats = strategy.get_statistics()
  print(f"\nStatistics: {stats}")