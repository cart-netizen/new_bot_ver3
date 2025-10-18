"""
Volume Profile Strategy - торговля на основе профиля объема.

Методология:
- Volume Profile для определения ключевых уровней
- Point of Control (POC) - уровень максимального объема
- Value Area (VA) - зона 70% объема
- High Volume Nodes (HVN) и Low Volume Nodes (LVN)
- Торговля на пробоях LVN и отскоках от HVN

Путь: backend/strategies/volume_profile_strategy.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
from collections import defaultdict

from core.logger import get_logger
from models.signal import TradingSignal, SignalType, SignalStrength, SignalSource
from strategy.candle_manager import Candle

logger = get_logger(__name__)


@dataclass
class VolumeProfileConfig:
  """Конфигурация Volume Profile стратегии."""
  # Profile параметры
  lookback_periods: int = 100  # Период для построения профиля
  price_bins: int = 50  # Количество ценовых уровней
  value_area_percent: float = 0.70  # 70% объема для Value Area

  # HVN/LVN detection
  hvn_threshold_percentile: float = 80  # Top 20% = HVN
  lvn_threshold_percentile: float = 20  # Bottom 20% = LVN

  # Trading logic
  poc_tolerance_pct: float = 0.2  # Допуск от POC для входов
  lvn_breakout_confirmation_bars: int = 2  # Баров для подтверждения пробоя

  # Volume confirmation
  volume_surge_threshold: float = 1.5  # 1.5x средний объем

  # Risk management
  stop_loss_pct: float = 1.5
  take_profit_pct: float = 4.5
  use_value_area_targets: bool = True  # TP на границах VA


@dataclass
class VolumeNode:
  """Узел объема на ценовом уровне."""
  price: float
  volume: float
  node_type: str  # "HVN", "LVN", "POC", "VA_HIGH", "VA_LOW", "normal"


@dataclass
class VolumeProfile:
  """Volume Profile - распределение объема по ценам."""
  price_levels: np.ndarray  # Ценовые уровни
  volumes: np.ndarray  # Объемы на уровнях

  poc_price: float  # Point of Control
  poc_volume: float

  value_area_high: float  # Верхняя граница Value Area
  value_area_low: float  # Нижняя граница Value Area

  hvn_nodes: List[VolumeNode]  # High Volume Nodes
  lvn_nodes: List[VolumeNode]  # Low Volume Nodes

  timestamp: int  # Когда построен профиль


class VolumeProfileAnalyzer:
  """Анализатор Volume Profile."""

  @staticmethod
  def build_profile(
      candles: List[Candle],
      price_bins: int,
      value_area_percent: float
  ) -> VolumeProfile:
    """
    Построить Volume Profile из свечей.

    Args:
        candles: История свечей
        price_bins: Количество ценовых уровней
        value_area_percent: % объема для Value Area

    Returns:
        VolumeProfile
    """
    if not candles:
      raise ValueError("Нет свечей для построения профиля")

    # Определяем диапазон цен
    all_prices = []
    for candle in candles:
      all_prices.extend([candle.open, candle.high, candle.low, candle.close])

    min_price = min(all_prices)
    max_price = max(all_prices)

    # Создаем ценовые бины
    price_levels = np.linspace(min_price, max_price, price_bins)
    price_step = (max_price - min_price) / price_bins

    # Распределяем объем по бинам
    volume_distribution = np.zeros(price_bins)

    for candle in candles:
      # Для каждой свечи распределяем объем между ценами
      # Упрощенно: равномерно между OHLC
      prices = [candle.open, candle.high, candle.low, candle.close]
      volume_per_price = candle.volume / 4

      for price in prices:
        # Находим соответствующий бин
        bin_idx = int((price - min_price) / price_step)
        bin_idx = min(bin_idx, price_bins - 1)
        volume_distribution[bin_idx] += volume_per_price

    # Point of Control (максимальный объем)
    poc_idx = np.argmax(volume_distribution)
    poc_price = float(price_levels[poc_idx])
    poc_volume = float(volume_distribution[poc_idx])

    # Value Area (70% объема)
    total_volume = np.sum(volume_distribution)
    target_volume = total_volume * value_area_percent

    # Начинаем с POC и расширяем в обе стороны
    va_indices = {poc_idx}
    va_volume = poc_volume

    lower_idx = poc_idx - 1
    upper_idx = poc_idx + 1

    while va_volume < target_volume:
      lower_vol = volume_distribution[lower_idx] if lower_idx >= 0 else 0
      upper_vol = volume_distribution[upper_idx] if upper_idx < price_bins else 0

      if lower_vol == 0 and upper_vol == 0:
        break

      # Добавляем сторону с большим объемом
      if lower_vol >= upper_vol and lower_idx >= 0:
        va_indices.add(lower_idx)
        va_volume += lower_vol
        lower_idx -= 1
      elif upper_idx < price_bins:
        va_indices.add(upper_idx)
        va_volume += upper_vol
        upper_idx += 1

    va_indices_sorted = sorted(va_indices)
    value_area_low = float(price_levels[va_indices_sorted[0]])
    value_area_high = float(price_levels[va_indices_sorted[-1]])

    return VolumeProfile(
      price_levels=price_levels,
      volumes=volume_distribution,
      poc_price=poc_price,
      poc_volume=poc_volume,
      value_area_high=value_area_high,
      value_area_low=value_area_low,
      hvn_nodes=[],  # Будет заполнено в detect_nodes
      lvn_nodes=[],
      timestamp=candles[-1].timestamp
    )

  @staticmethod
  def detect_nodes(
      profile: VolumeProfile,
      hvn_percentile: float,
      lvn_percentile: float
  ) -> Tuple[List[VolumeNode], List[VolumeNode]]:
    """
    Обнаружить High Volume Nodes и Low Volume Nodes.

    Returns:
        (hvn_nodes, lvn_nodes)
    """
    volumes = profile.volumes
    prices = profile.price_levels

    # Определяем пороги
    hvn_threshold = np.percentile(volumes, hvn_percentile)
    lvn_threshold = np.percentile(volumes, lvn_percentile)

    hvn_nodes = []
    lvn_nodes = []

    for i, (price, volume) in enumerate(zip(prices, volumes)):
      if volume >= hvn_threshold:
        node = VolumeNode(
          price=price,
          volume=volume,
          node_type="HVN"
        )
        hvn_nodes.append(node)

      elif volume <= lvn_threshold and volume > 0:
        node = VolumeNode(
          price=price,
          volume=volume,
          node_type="LVN"
        )
        lvn_nodes.append(node)

    return hvn_nodes, lvn_nodes


class VolumeProfileStrategy:
  """
  Volume Profile Trading Strategy.

  Логика:
  1. Строим Volume Profile из истории
  2. Определяем POC, VA, HVN, LVN
  3. Торгуем:
     - Пробои LVN (низкого сопротивления)
     - Отскоки от HVN (поддержка/сопротивление)
     - Возврат к POC из экстремумов
  """

  def __init__(self, config: VolumeProfileConfig):
    """Инициализация стратегии."""
    self.config = config

    # Кэш профилей
    self.profiles: Dict[str, VolumeProfile] = {}

    # Состояние
    self.active_signals: Dict[str, TradingSignal] = {}
    self.last_profile_update: Dict[str, int] = {}

    # Статистика
    self.signals_generated = 0
    self.profiles_built = 0

    logger.info(
      f"Инициализирована VolumeProfileStrategy: "
      f"lookback={config.lookback_periods}, "
      f"bins={config.price_bins}"
    )

  def _should_rebuild_profile(
      self,
      symbol: str,
      current_timestamp: int
  ) -> bool:
    """Проверить нужно ли перестроить профиль."""
    if symbol not in self.profiles:
      return True

    if symbol not in self.last_profile_update:
      return True

    # Обновляем профиль каждые 30 минут
    time_since_update = current_timestamp - self.last_profile_update[symbol]
    return time_since_update > (30 * 60 * 1000)  # 30 минут в ms

  def _build_profile(
      self,
      symbol: str,
      candles: List[Candle]
  ) -> VolumeProfile:
    """Построить Volume Profile."""
    # Берем последние N свечей
    profile_candles = candles[-self.config.lookback_periods:]

    # Строим профиль
    profile = VolumeProfileAnalyzer.build_profile(
      profile_candles,
      self.config.price_bins,
      self.config.value_area_percent
    )

    # Обнаруживаем HVN/LVN
    hvn_nodes, lvn_nodes = VolumeProfileAnalyzer.detect_nodes(
      profile,
      self.config.hvn_threshold_percentile,
      self.config.lvn_threshold_percentile
    )

    profile.hvn_nodes = hvn_nodes
    profile.lvn_nodes = lvn_nodes

    # Сохраняем
    self.profiles[symbol] = profile
    self.last_profile_update[symbol] = profile.timestamp
    self.profiles_built += 1

    logger.debug(
      f"Построен Volume Profile [{symbol}]: "
      f"POC=${profile.poc_price:.2f}, "
      f"VA=[${profile.value_area_low:.2f}, ${profile.value_area_high:.2f}], "
      f"HVN={len(hvn_nodes)}, LVN={len(lvn_nodes)}"
    )

    return profile

  def analyze(
      self,
      symbol: str,
      candles: List[Candle],
      current_price: float
  ) -> Optional[TradingSignal]:
    """Анализ и генерация сигнала."""
    if len(candles) < self.config.lookback_periods:
      return None

    # Обновляем профиль если нужно
    if self._should_rebuild_profile(symbol, candles[-1].timestamp):
      profile = self._build_profile(symbol, candles)
    else:
      profile = self.profiles.get(symbol)
      if not profile:
        return None

    # Анализируем позицию цены относительно профиля
    signal_type = None
    reason_parts = []
    confidence = 0.0

    # 1. Торговля около POC
    poc_distance_pct = abs(current_price - profile.poc_price) / profile.poc_price * 100

    if poc_distance_pct <= self.config.poc_tolerance_pct:
      # Цена около POC - ждем пробоя или отскока
      # Определяем direction по положению в VA
      if current_price > profile.poc_price:
        # Выше POC - потенциал для роста
        if current_price < profile.value_area_high:
          signal_type = SignalType.BUY
          reason_parts.append(f"Price near POC (${profile.poc_price:.2f}), upside potential to VA high")
          confidence += 0.3
      else:
        # Ниже POC - потенциал для падения
        if current_price > profile.value_area_low:
          signal_type = SignalType.SELL
          reason_parts.append(f"Price near POC (${profile.poc_price:.2f}), downside potential to VA low")
          confidence += 0.3

    # 2. Пробой LVN (низкого сопротивления)
    nearest_lvn = self._find_nearest_node(current_price, profile.lvn_nodes)

    if nearest_lvn:
      lvn_distance_pct = abs(current_price - nearest_lvn.price) / current_price * 100

      if lvn_distance_pct < 0.3:  # Очень близко к LVN
        # Проверяем направление пробоя
        recent_candles = candles[-self.config.lvn_breakout_confirmation_bars:]

        if all(c.close > nearest_lvn.price for c in recent_candles):
          # Пробой вверх через LVN
          if signal_type is None or signal_type == SignalType.BUY:
            signal_type = SignalType.BUY
            reason_parts.append(f"Breakout through LVN at ${nearest_lvn.price:.2f}")
            confidence += 0.4

        elif all(c.close < nearest_lvn.price for c in recent_candles):
          # Пробой вниз через LVN
          if signal_type is None or signal_type == SignalType.SELL:
            signal_type = SignalType.SELL
            reason_parts.append(f"Breakdown through LVN at ${nearest_lvn.price:.2f}")
            confidence += 0.4

    # 3. Отскок от HVN
    nearest_hvn = self._find_nearest_node(current_price, profile.hvn_nodes)

    if nearest_hvn:
      hvn_distance_pct = abs(current_price - nearest_hvn.price) / current_price * 100

      if hvn_distance_pct < 0.3:  # Близко к HVN
        # HVN действует как поддержка/сопротивление
        if current_price > nearest_hvn.price:
          # HVN как поддержка
          if signal_type is None or signal_type == SignalType.BUY:
            signal_type = SignalType.BUY
            reason_parts.append(f"Bounce from HVN support at ${nearest_hvn.price:.2f}")
            confidence += 0.3
        else:
          # HVN как сопротивление
          if signal_type is None or signal_type == SignalType.SELL:
            signal_type = SignalType.SELL
            reason_parts.append(f"Rejection at HVN resistance ${nearest_hvn.price:.2f}")
            confidence += 0.3

    # Нет сигнала
    if signal_type is None or confidence < 0.3:
      return None

    # Volume confirmation
    volumes = np.array([c.volume for c in candles[-20:]])
    avg_volume = np.mean(volumes)
    current_volume = candles[-1].volume
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

    if volume_ratio >= self.config.volume_surge_threshold:
      confidence += 0.2
      reason_parts.append(f"Volume surge: {volume_ratio:.2f}x")

    # Ограничиваем confidence
    confidence = min(confidence, 1.0)

    # Определяем силу сигнала
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
        'strategy': 'volume_profile',
        'poc_price': profile.poc_price,
        'value_area_high': profile.value_area_high,
        'value_area_low': profile.value_area_low,
        'hvn_count': len(profile.hvn_nodes),
        'lvn_count': len(profile.lvn_nodes),
        'volume_ratio': volume_ratio,
        'stop_loss_pct': self.config.stop_loss_pct,
        'take_profit_pct': self.config.take_profit_pct
      }
    )

    self.active_signals[symbol] = signal
    self.signals_generated += 1

    # logger.info(
    #   f"📊 VOLUME PROFILE SIGNAL [{symbol}]: {signal_type.value}, "
    #   f"confidence={confidence:.2f}, "
    #   f"POC=${profile.poc_price:.2f}"
    # )

    return signal

  def _find_nearest_node(
      self,
      price: float,
      nodes: List[VolumeNode]
  ) -> Optional[VolumeNode]:
    """Найти ближайший node к цене."""
    if not nodes:
      return None

    nearest = min(nodes, key=lambda n: abs(n.price - price))
    return nearest

  def get_value_area_targets(
      self,
      symbol: str,
      position_side: str
  ) -> Optional[Tuple[float, float]]:
    """
    Получить цели на основе Value Area.

    Returns:
        (target1, target2) или None
    """
    if symbol not in self.profiles:
      return None

    profile = self.profiles[symbol]

    if position_side == "long":
      # Для long: target1 = POC, target2 = VA high
      return (profile.poc_price, profile.value_area_high)
    else:
      # Для short: target1 = POC, target2 = VA low
      return (profile.poc_price, profile.value_area_low)

  def get_statistics(self) -> Dict:
    """Получить статистику стратегии."""
    return {
      'strategy': 'volume_profile',
      'signals_generated': self.signals_generated,
      'profiles_built': self.profiles_built,
      'active_signals': len(self.active_signals),
      'symbols_tracked': len(self.profiles)
    }


# Пример использования
if __name__ == "__main__":
  import random

  config = VolumeProfileConfig(
    lookback_periods=100,
    price_bins=50,
    value_area_percent=0.70
  )

  strategy = VolumeProfileStrategy(config)

  # Генерируем тестовые свечи с концентрацией объема
  base_price = 50000.0
  candles = []

  for i in range(150):
    # Создаем зону высокого объема около 50500
    if 49800 <= base_price + (i * 20) <= 50200:
      volume_multiplier = 3.0
    else:
      volume_multiplier = 1.0

    price = base_price + (i * 20) + random.uniform(-200, 200)

    candle = Candle(
      timestamp=int(datetime.now().timestamp() * 1000) + i * 60000,
      open=price - 5,
      high=price + abs(random.uniform(20, 50)),
      low=price - abs(random.uniform(20, 50)),
      close=price,
      volume=(1000 + random.uniform(-200, 200)) * volume_multiplier
    )
    candles.append(candle)

  # Анализируем
  signal = strategy.analyze("BTCUSDT", candles, candles[-1].close)

  if signal:
    print(f"Signal: {signal.signal_type.value}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Reason: {signal.reason}")

    # Value Area targets
    targets = strategy.get_value_area_targets(
      "BTCUSDT",
      "long" if signal.signal_type == SignalType.BUY else "short"
    )
    if targets:
      print(f"Targets: ${targets[0]:.2f}, ${targets[1]:.2f}")
  else:
    print("No signal")

  # Профиль
  if "BTCUSDT" in strategy.profiles:
    profile = strategy.profiles["BTCUSDT"]
    print(f"\nVolume Profile:")
    print(f"  POC: ${profile.poc_price:.2f}")
    print(f"  VA: [${profile.value_area_low:.2f}, ${profile.value_area_high:.2f}]")
    print(f"  HVN nodes: {len(profile.hvn_nodes)}")
    print(f"  LVN nodes: {len(profile.lvn_nodes)}")

  stats = strategy.get_statistics()
  print(f"\nStatistics: {stats}")