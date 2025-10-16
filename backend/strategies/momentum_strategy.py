"""
Momentum Strategy - торговля на основе силы тренда.

Методология:
- Rate of Change (ROC) для измерения momentum
- RSI для фильтрации перекупленности/перепроданности
- Volume confirmation
- Adaptive position sizing на основе силы momentum
- Dynamic trailing stop

Путь: backend/strategies/momentum_strategy.py
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
class MomentumConfig:
  """Конфигурация Momentum стратегии."""
  # ROC параметры
  roc_period: int = 14  # Период для Rate of Change
  roc_threshold_long: float = 2.0  # % для long сигнала
  roc_threshold_short: float = -2.0  # % для short сигнала

  # RSI фильтр
  rsi_period: int = 14
  rsi_overbought: float = 70.0
  rsi_oversold: float = 30.0

  # Volume confirmation
  volume_ma_period: int = 20
  volume_threshold: float = 1.2  # 1.2x средний объем

  # Momentum strength
  momentum_ma_short: int = 5
  momentum_ma_long: int = 20

  # Position sizing
  base_position_size: float = 1.0
  max_position_multiplier: float = 2.0  # Макс увеличение позиции

  # Risk management
  stop_loss_pct: float = 2.0  # 2% stop loss
  take_profit_pct: float = 6.0  # 3:1 risk/reward
  trailing_stop_activation_pct: float = 3.0  # Активация trailing после 3%
  trailing_stop_distance_pct: float = 1.5  # Дистанция trailing 1.5%


class MomentumIndicators:
  """Вычисление индикаторов для Momentum стратегии."""

  @staticmethod
  def calculate_roc(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Rate of Change - скорость изменения цены.

    ROC = ((Price - Price[period]) / Price[period]) * 100
    """
    if len(prices) < period + 1:
      return np.array([])

    roc = np.zeros(len(prices))
    for i in range(period, len(prices)):
      roc[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100

    return roc

  @staticmethod
  def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index.

    RSI = 100 - (100 / (1 + RS))
    где RS = Average Gain / Average Loss
    """
    if len(prices) < period + 1:
      return np.array([])

    deltas = np.diff(prices)

    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Первое среднее - простое
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rsi = np.zeros(len(prices))

    for i in range(period, len(deltas)):
      avg_gain = (avg_gain * (period - 1) + gains[i]) / period
      avg_loss = (avg_loss * (period - 1) + losses[i]) / period

      if avg_loss == 0:
        rsi[i + 1] = 100
      else:
        rs = avg_gain / avg_loss
        rsi[i + 1] = 100 - (100 / (1 + rs))

    return rsi

  @staticmethod
  def calculate_momentum_strength(
      roc: float,
      rsi: float,
      volume_ratio: float,
      config: MomentumConfig
  ) -> float:
    """
    Вычислить силу momentum (0-1).

    Компоненты:
    - ROC magnitude
    - RSI положение
    - Volume confirmation
    """
    strength = 0.0

    # ROC component (40% веса)
    roc_abs = abs(roc)
    if roc_abs >= abs(config.roc_threshold_long):
      roc_component = min(roc_abs / 5.0, 1.0)  # Нормализуем до 5%
      strength += roc_component * 0.4

    # RSI component (30% веса)
    # Для long: RSI в зоне 50-70 = сильно
    # Для short: RSI в зоне 30-50 = сильно
    if roc > 0:  # Long momentum
      if 50 <= rsi <= 70:
        rsi_component = (rsi - 50) / 20
        strength += rsi_component * 0.3
    else:  # Short momentum
      if 30 <= rsi <= 50:
        rsi_component = (50 - rsi) / 20
        strength += rsi_component * 0.3

    # Volume component (30% веса)
    if volume_ratio >= config.volume_threshold:
      volume_component = min((volume_ratio - 1.0) / 1.0, 1.0)
      strength += volume_component * 0.3

    return min(strength, 1.0)


class MomentumStrategy:
  """
  Momentum Trading Strategy.

  Логика:
  1. Вычисляем ROC для определения силы тренда
  2. Фильтруем через RSI (избегаем экстремальных зон)
  3. Подтверждаем через volume
  4. Генерируем сигналы с adaptive position sizing
  5. Управляем позицией через trailing stop
  """

  def __init__(self, config: MomentumConfig):
    """
    Инициализация стратегии.

    Args:
        config: Конфигурация стратегии
    """
    self.config = config
    self.indicators = MomentumIndicators()

    # Состояние стратегии
    self.active_signals: Dict[str, TradingSignal] = {}
    self.momentum_history: Dict[str, List[float]] = {}

    # Статистика
    self.signals_generated = 0
    self.strong_signals = 0

    logger.info(
      f"Инициализирована MomentumStrategy: "
      f"roc_period={config.roc_period}, "
      f"rsi_period={config.rsi_period}"
    )

  def analyze(
      self,
      symbol: str,
      candles: List[Candle],
      current_price: float
  ) -> Optional[TradingSignal]:
    """
    Анализ и генерация сигнала.

    Args:
        symbol: Торговая пара
        candles: История свечей
        current_price: Текущая цена

    Returns:
        TradingSignal или None
    """
    if len(candles) < max(
        self.config.roc_period,
        self.config.rsi_period,
        self.config.volume_ma_period
    ) + 1:
      logger.debug(f"Недостаточно свечей для {symbol}")
      return None

    # Извлекаем данные
    closes = np.array([c.close for c in candles])
    volumes = np.array([c.volume for c in candles])

    # Вычисляем индикаторы
    roc = self.indicators.calculate_roc(closes, self.config.roc_period)
    rsi = self.indicators.calculate_rsi(closes, self.config.rsi_period)

    if len(roc) == 0 or len(rsi) == 0:
      return None

    # Текущие значения
    current_roc = float(roc[-1])
    current_rsi = float(rsi[-1])

    # Volume analysis
    volume_ma = np.mean(volumes[-self.config.volume_ma_period:])
    current_volume = volumes[-1]
    volume_ratio = current_volume / volume_ma if volume_ma > 0 else 0

    # Проверяем условия для сигналов
    signal_type = None
    reason_parts = []

    # LONG условия
    if (
        current_roc >= self.config.roc_threshold_long
        and current_rsi < self.config.rsi_overbought
        and volume_ratio >= self.config.volume_threshold
    ):
      signal_type = SignalType.BUY
      reason_parts.append(f"Strong upward momentum: ROC={current_roc:.2f}%")
      reason_parts.append(f"RSI={current_rsi:.1f} (not overbought)")
      reason_parts.append(f"Volume {volume_ratio:.2f}x average")

    # SHORT условия
    elif (
        current_roc <= self.config.roc_threshold_short
        and current_rsi > self.config.rsi_oversold
        and volume_ratio >= self.config.volume_threshold
    ):
      signal_type = SignalType.SELL
      reason_parts.append(f"Strong downward momentum: ROC={current_roc:.2f}%")
      reason_parts.append(f"RSI={current_rsi:.1f} (not oversold)")
      reason_parts.append(f"Volume {volume_ratio:.2f}x average")

    if signal_type is None:
      return None

    # Вычисляем силу momentum
    momentum_strength = self.indicators.calculate_momentum_strength(
      current_roc,
      current_rsi,
      volume_ratio,
      self.config
    )

    # Сохраняем историю momentum
    if symbol not in self.momentum_history:
      self.momentum_history[symbol] = []
    self.momentum_history[symbol].append(momentum_strength)

    # Ограничиваем историю
    if len(self.momentum_history[symbol]) > 100:
      self.momentum_history[symbol] = self.momentum_history[symbol][-100:]

    # Определяем силу сигнала
    if momentum_strength > 0.7:
      signal_strength = SignalStrength.STRONG
      self.strong_signals += 1
    elif momentum_strength > 0.5:
      signal_strength = SignalStrength.MEDIUM
    else:
      signal_strength = SignalStrength.WEAK

    # Adaptive position sizing
    position_multiplier = 1.0 + (
        momentum_strength * (self.config.max_position_multiplier - 1.0)
    )

    # Создаем сигнал
    signal = TradingSignal(
      symbol=symbol,
      signal_type=signal_type,
      source=SignalSource.STRATEGY,
      strength=signal_strength,
      price=current_price,
      confidence=momentum_strength,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason=" | ".join(reason_parts),
      metadata={
        'strategy': 'momentum',
        'roc': current_roc,
        'rsi': current_rsi,
        'volume_ratio': volume_ratio,
        'momentum_strength': momentum_strength,
        'position_multiplier': position_multiplier,
        'stop_loss_pct': self.config.stop_loss_pct,
        'take_profit_pct': self.config.take_profit_pct,
        'trailing_stop_activation': self.config.trailing_stop_activation_pct,
        'trailing_stop_distance': self.config.trailing_stop_distance_pct
      }
    )

    # Сохраняем активный сигнал
    self.active_signals[symbol] = signal
    self.signals_generated += 1

    logger.info(
      f"🎯 MOMENTUM SIGNAL [{symbol}]: {signal_type.value}, "
      f"strength={momentum_strength:.2f}, "
      f"ROC={current_roc:.2f}%, RSI={current_rsi:.1f}"
    )

    return signal

  def should_exit(
      self,
      symbol: str,
      candles: List[Candle],
      entry_price: float,
      position_side: str  # "long" или "short"
  ) -> Tuple[bool, Optional[str]]:
    """
    Проверить условия выхода из позиции.

    Returns:
        (should_exit, reason)
    """
    if len(candles) < 2:
      return False, None

    current_price = candles[-1].close

    # Вычисляем P&L
    if position_side == "long":
      pnl_pct = ((current_price - entry_price) / entry_price) * 100
    else:  # short
      pnl_pct = ((entry_price - current_price) / entry_price) * 100

    # Stop Loss
    if pnl_pct <= -self.config.stop_loss_pct:
      return True, f"Stop Loss hit: {pnl_pct:.2f}%"

    # Take Profit
    if pnl_pct >= self.config.take_profit_pct:
      return True, f"Take Profit hit: {pnl_pct:.2f}%"

    # Trailing Stop (если активирован)
    if pnl_pct >= self.config.trailing_stop_activation_pct:
      # Вычисляем максимальный P&L за последние свечи
      recent_closes = np.array([c.close for c in candles[-10:]])

      if position_side == "long":
        max_price = np.max(recent_closes)
        max_pnl = ((max_price - entry_price) / entry_price) * 100
        drawdown = max_pnl - pnl_pct

        if drawdown >= self.config.trailing_stop_distance_pct:
          return True, f"Trailing Stop: drawdown={drawdown:.2f}%"
      else:  # short
        min_price = np.min(recent_closes)
        max_pnl = ((entry_price - min_price) / entry_price) * 100
        drawdown = max_pnl - pnl_pct

        if drawdown >= self.config.trailing_stop_distance_pct:
          return True, f"Trailing Stop: drawdown={drawdown:.2f}%"

    # Momentum reversal
    if len(candles) >= self.config.roc_period + 1:
      closes = np.array([c.close for c in candles])
      roc = self.indicators.calculate_roc(closes, self.config.roc_period)

      if len(roc) > 0:
        current_roc = roc[-1]

        # Для long позиции - выходим если momentum ослаб
        if position_side == "long" and current_roc < 0:
          return True, f"Momentum reversal: ROC={current_roc:.2f}%"

        # Для short позиции - выходим если momentum развернулся вверх
        if position_side == "short" and current_roc > 0:
          return True, f"Momentum reversal: ROC={current_roc:.2f}%"

    return False, None

  def get_statistics(self) -> Dict:
    """Получить статистику стратегии."""
    strong_signal_rate = (
      self.strong_signals / self.signals_generated
      if self.signals_generated > 0
      else 0.0
    )

    # Средняя сила momentum
    all_momentum = []
    for history in self.momentum_history.values():
      all_momentum.extend(history)

    avg_momentum = np.mean(all_momentum) if all_momentum else 0.0

    return {
      'strategy': 'momentum',
      'signals_generated': self.signals_generated,
      'strong_signals': self.strong_signals,
      'strong_signal_rate': strong_signal_rate,
      'avg_momentum_strength': avg_momentum,
      'active_signals': len(self.active_signals),
      'symbols_tracked': len(self.momentum_history)
    }


# Пример использования
if __name__ == "__main__":
  from strategy.candle_manager import Candle
  import random

  config = MomentumConfig(
    roc_period=14,
    roc_threshold_long=2.0,
    rsi_period=14,
    volume_ma_period=20
  )

  strategy = MomentumStrategy(config)

  # Генерируем тестовые свечи с трендом
  base_price = 50000.0
  candles = []

  for i in range(100):
    # Симулируем восходящий тренд
    trend = i * 10
    noise = random.uniform(-100, 100)
    price = base_price + trend + noise

    candle = Candle(
      timestamp=int(datetime.now().timestamp() * 1000) + i * 60000,
      open=price - 5,
      high=price + 10,
      low=price - 10,
      close=price,
      volume=1000 + random.uniform(-200, 200)
    )
    candles.append(candle)

  # Анализируем
  signal = strategy.analyze("BTCUSDT", candles, candles[-1].close)

  if signal:
    print(f"Signal generated: {signal.signal_type.value}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Reason: {signal.reason}")
    print(f"Metadata: {signal.metadata}")
  else:
    print("No signal generated")

  # Статистика
  stats = strategy.get_statistics()
  print(f"\nStatistics:")
  for key, value in stats.items():
    print(f"  {key}: {value}")