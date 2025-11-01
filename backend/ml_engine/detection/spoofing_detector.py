"""
Spoofing Detector для обнаружения манипуляций ордерами.

Spoofing - размещение крупных ордеров без намерения исполнения для манипуляции ценой.

Признаки spoofing:
- Крупные ордера на одной стороне стакана
- Быстрая отмена ордеров при приближении цены
- Короткое время жизни (TTL) крупных ордеров
- Ордера далеко от текущей цены
- Паттерн: размещение -> движение цены -> отмена

Путь: backend/ml_engine/detection/spoofing_detector.py
"""

from typing import Dict, List, Optional, Tuple, Deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np

from backend.core.logger import get_logger
from backend.models.orderbook import OrderBookSnapshot, OrderBookLevel

logger = get_logger(__name__)


@dataclass
class SpoofingConfig:
  """Конфигурация детектора."""
  # Пороги для определения крупного ордера
  large_order_threshold_usdt: float = 50000.0  # $50k+
  large_order_relative_threshold: float = 0.1  # 10% от total depth

  # Пороги для TTL
  suspicious_ttl_seconds: float = 10.0  # < 10 секунд подозрительно

  # Расстояние от mid price
  min_distance_from_mid: float = 0.001  # 0.1% от mid price
  max_distance_from_mid: float = 0.01  # 1% от mid price

  # История для анализа паттернов
  history_window_seconds: int = 300  # 5 минут

  # Пороги для детекции
  cancel_rate_threshold: float = 0.7  # 70%+ отмен = подозрительно
  min_events_for_detection: int = 3  # Минимум событий для паттерна


@dataclass
class OrderEvent:
  """Событие с ордером."""
  timestamp: int  # ms
  side: str  # "bid" или "ask"
  price: float
  volume: float
  event_type: str  # "placed", "cancelled", "filled", "modified"
  distance_from_mid: float  # % от mid price


@dataclass
class SpoofingPattern:
  """Обнаруженный паттерн spoofing."""
  symbol: str
  timestamp: int
  side: str  # "bid" или "ask"
  confidence: float  # 0-1

  # Детали паттерна
  large_orders_count: int
  total_volume: float
  avg_ttl_seconds: float
  cancel_rate: float
  avg_distance_from_mid: float

  # События
  events: List[OrderEvent]

  # Причина
  reason: str


class LevelHistory:
  """История изменений уровня в стакане."""

  def __init__(self, price: float, side: str):
    """
    Инициализация истории уровня.

    Args:
        price: Цена уровня
        side: Сторона ("bid" или "ask")
    """
    self.price = price
    self.side = side
    self.first_seen: Optional[int] = None  # timestamp ms
    self.last_seen: Optional[int] = None
    self.max_volume: float = 0.0
    self.volume_history: Deque[Tuple[int, float]] = deque(maxlen=100)
    self.events: List[OrderEvent] = []

  def update(self, timestamp: int, volume: float, mid_price: float):
    """Обновить историю уровня."""
    if self.first_seen is None:
      self.first_seen = timestamp

    self.last_seen = timestamp
    self.max_volume = max(self.max_volume, volume)
    self.volume_history.append((timestamp, volume))

    # Определяем тип события
    if len(self.volume_history) == 1:
      event_type = "placed"
    elif volume > self.volume_history[-2][1]:
      event_type = "modified"  # Увеличение объема
    elif volume < self.volume_history[-2][1]:
      event_type = "modified"  # Уменьшение объема
    else:
      return  # Без изменений

    distance = abs(self.price - mid_price) / mid_price

    event = OrderEvent(
      timestamp=timestamp,
      side=self.side,
      price=self.price,
      volume=volume,
      event_type=event_type,
      distance_from_mid=distance
    )
    self.events.append(event)

  def mark_cancelled(self, timestamp: int, mid_price: float):
    """Отметить отмену уровня."""
    distance = abs(self.price - mid_price) / mid_price

    event = OrderEvent(
      timestamp=timestamp,
      side=self.side,
      price=self.price,
      volume=0.0,
      event_type="cancelled",
      distance_from_mid=distance
    )
    self.events.append(event)

  def get_ttl(self) -> float:
    """Получить время жизни уровня в секундах."""
    if self.first_seen and self.last_seen:
      return (self.last_seen - self.first_seen) / 1000.0
    return 0.0

  def is_suspicious(self, config: SpoofingConfig) -> bool:
    """Проверить подозрительность уровня."""
    ttl = self.get_ttl()

    # Крупный объем
    is_large = self.max_volume * self.price >= config.large_order_threshold_usdt

    # Короткий TTL
    short_lived = 0 < ttl < config.suspicious_ttl_seconds

    # Быстро отменен
    was_cancelled = (
        len(self.events) > 0
        and self.events[-1].event_type == "cancelled"
    )

    return is_large and short_lived and was_cancelled


class SpoofingDetector:
  """
  Детектор spoofing манипуляций.

  Алгоритм:
  1. Отслеживаем историю каждого уровня в стакане
  2. Вычисляем TTL (Time To Live) для уровней
  3. Анализируем паттерны размещения/отмены крупных ордеров
  4. Детектируем spoofing на основе признаков
  """

  def __init__(self, config: SpoofingConfig):
    """
    Инициализация детектора.

    Args:
        config: Конфигурация детектора
    """
    self.config = config

    # История уровней для каждого символа
    # symbol -> side -> price -> LevelHistory
    self.level_history: Dict[str, Dict[str, Dict[float, LevelHistory]]] = defaultdict(
      lambda: {"bid": {}, "ask": {}}
    )

    # Обнаруженные паттерны
    self.detected_patterns: Dict[str, List[SpoofingPattern]] = defaultdict(list)

    # Статистика
    self.total_checks = 0
    self.patterns_detected = 0

    logger.info(
      f"Инициализирован SpoofingDetector: "
      f"large_order_threshold={config.large_order_threshold_usdt}, "
      f"suspicious_ttl={config.suspicious_ttl_seconds}s"
    )

  def update(self, snapshot: OrderBookSnapshot):
    """
    Обновить детектор новым snapshot стакана.

    Args:
        snapshot: Snapshot стакана
    """
    symbol = snapshot.symbol
    timestamp = snapshot.timestamp
    mid_price = snapshot.mid_price

    if mid_price is None or mid_price == 0:
      return

    # Текущие уровни из snapshot (bids/asks - это list[tuple[float, float]])
    current_bids = {price: volume for price, volume in snapshot.bids}
    current_asks = {price: volume for price, volume in snapshot.asks}

    # Обновляем bid уровни
    self._update_side(
      symbol,
      "bid",
      current_bids,
      timestamp,
      mid_price
    )

    # Обновляем ask уровни
    self._update_side(
      symbol,
      "ask",
      current_asks,
      timestamp,
      mid_price
    )

    # Периодически проверяем на паттерны
    self.total_checks += 1
    if self.total_checks % 100 == 0:  # Каждые 100 обновлений
      self._detect_patterns(symbol, timestamp, mid_price)
      self._cleanup_old_history(symbol, timestamp)

  def _update_side(
      self,
      symbol: str,
      side: str,
      current_levels: Dict[float, float],
      timestamp: int,
      mid_price: float
  ):
    """Обновить одну сторону стакана."""
    history_side = self.level_history[symbol][side]

    # Обновляем существующие уровни
    for price, volume in current_levels.items():
      if price not in history_side:
        # Новый уровень
        history_side[price] = LevelHistory(price, side)

      history_side[price].update(timestamp, volume, mid_price)

    # Проверяем удаленные уровни (отмененные)
    removed_prices = set(history_side.keys()) - set(current_levels.keys())
    for price in removed_prices:
      history_side[price].mark_cancelled(timestamp, mid_price)

  def _detect_patterns(
      self,
      symbol: str,
      timestamp: int,
      mid_price: float
  ):
    """Обнаружить spoofing паттерны."""
    # Проверяем каждую сторону
    for side in ["bid", "ask"]:
      pattern = self._analyze_side_for_spoofing(
        symbol,
        side,
        timestamp,
        mid_price
      )

      if pattern:
        self.detected_patterns[symbol].append(pattern)
        self.patterns_detected += 1

        logger.warning(
          f"🚨 SPOOFING ОБНАРУЖЕН [{symbol}]: "
          f"side={side}, confidence={pattern.confidence:.2f}, "
          f"reason={pattern.reason}"
        )

  def _analyze_side_for_spoofing(
      self,
      symbol: str,
      side: str,
      timestamp: int,
      mid_price: float
  ) -> Optional[SpoofingPattern]:
    """Анализ одной стороны на spoofing."""
    history_side = self.level_history[symbol][side]

    if not history_side:
      return None

    # Фильтруем подозрительные уровни
    suspicious_levels = [
      level for level in history_side.values()
      if level.is_suspicious(self.config)
    ]

    if len(suspicious_levels) < self.config.min_events_for_detection:
      return None

    # Анализируем паттерн
    total_volume = sum(level.max_volume for level in suspicious_levels)
    total_volume_usdt = sum(
      level.max_volume * level.price
      for level in suspicious_levels
    )

    ttls = [level.get_ttl() for level in suspicious_levels]
    avg_ttl = np.mean(ttls) if ttls else 0.0

    # Считаем cancel rate
    all_events = []
    for level in suspicious_levels:
      all_events.extend(level.events)

    cancelled_count = sum(
      1 for e in all_events
      if e.event_type == "cancelled"
    )
    cancel_rate = cancelled_count / len(all_events) if all_events else 0.0

    # Проверяем пороги
    if cancel_rate < self.config.cancel_rate_threshold:
      return None

    # Средняя дистанция от mid price
    distances = [
      level.events[-1].distance_from_mid
      for level in suspicious_levels
      if level.events
    ]
    avg_distance = np.mean(distances) if distances else 0.0

    # Вычисляем уверенность
    confidence = self._calculate_confidence(
      cancel_rate=cancel_rate,
      avg_ttl=avg_ttl,
      total_volume_usdt=total_volume_usdt,
      avg_distance=avg_distance
    )

    # Формируем причину
    reason = (
      f"{len(suspicious_levels)} крупных ордеров "
      f"(${total_volume_usdt:,.0f}) с высокой отменой ({cancel_rate:.1%}) "
      f"и коротким TTL ({avg_ttl:.1f}s)"
    )

    return SpoofingPattern(
      symbol=symbol,
      timestamp=timestamp,
      side=side,
      confidence=confidence,
      large_orders_count=len(suspicious_levels),
      total_volume=total_volume,
      avg_ttl_seconds=avg_ttl,
      cancel_rate=cancel_rate,
      avg_distance_from_mid=avg_distance,
      events=all_events[-20:],  # Последние 20 событий
      reason=reason
    )

  def _calculate_confidence(
      self,
      cancel_rate: float,
      avg_ttl: float,
      total_volume_usdt: float,
      avg_distance: float
  ) -> float:
    """Вычислить уверенность в spoofing паттерне."""
    confidence = 0.0

    # Высокая отмена
    if cancel_rate > 0.9:
      confidence += 0.4
    elif cancel_rate > 0.8:
      confidence += 0.3
    elif cancel_rate > 0.7:
      confidence += 0.2

    # Короткий TTL
    if avg_ttl < 5.0:
      confidence += 0.3
    elif avg_ttl < 10.0:
      confidence += 0.2

    # Крупный объем
    if total_volume_usdt > 100000:
      confidence += 0.2
    elif total_volume_usdt > 50000:
      confidence += 0.1

    # Подозрительная дистанция
    if (
        self.config.min_distance_from_mid
        < avg_distance
        < self.config.max_distance_from_mid
    ):
      confidence += 0.1

    return min(confidence, 1.0)

  def _cleanup_old_history(self, symbol: str, timestamp: int):
    """Очистка старой истории."""
    cutoff_time = timestamp - (self.config.history_window_seconds * 1000)

    for side in ["bid", "ask"]:
      history_side = self.level_history[symbol][side]

      # Удаляем старые уровни
      old_prices = [
        price for price, level in history_side.items()
        if level.last_seen and level.last_seen < cutoff_time
      ]

      for price in old_prices:
        del history_side[price]

  def get_recent_patterns(
      self,
      symbol: str,
      time_window_seconds: int = 60
  ) -> List[SpoofingPattern]:
    """
    Получить недавние паттерны spoofing.

    Args:
        symbol: Торговая пара
        time_window_seconds: Временное окно

    Returns:
        Список паттернов
    """
    current_time = int(datetime.now().timestamp() * 1000)
    cutoff_time = current_time - (time_window_seconds * 1000)

    patterns = self.detected_patterns.get(symbol, [])

    return [
      p for p in patterns
      if p.timestamp >= cutoff_time
    ]

  def is_spoofing_active(
      self,
      symbol: str,
      side: Optional[str] = None,
      time_window_seconds: int = 60
  ) -> bool:
    """
    Проверить активность spoofing.

    Args:
        symbol: Торговая пара
        side: Сторона ("bid"/"ask") или None для обеих
        time_window_seconds: Временное окно

    Returns:
        True если обнаружен активный spoofing
    """
    patterns = self.get_recent_patterns(symbol, time_window_seconds)

    if side:
      patterns = [p for p in patterns if p.side == side]

    return len(patterns) > 0

  def get_statistics(self) -> Dict:
    """Получить статистику детектора."""
    total_patterns = sum(
      len(patterns)
      for patterns in self.detected_patterns.values()
    )

    return {
      'total_checks': self.total_checks,
      'patterns_detected': self.patterns_detected,
      'total_patterns': total_patterns,
      'symbols_monitored': len(self.level_history),
      'detection_rate': (
        self.patterns_detected / self.total_checks
        if self.total_checks > 0
        else 0.0
      )
    }


# Пример использования
if __name__ == "__main__":
  from backend.models.orderbook import OrderBookSnapshot, OrderBookLevel

  # Создаем детектор
  config = SpoofingConfig(
    large_order_threshold_usdt=50000.0,
    suspicious_ttl_seconds=10.0,
    cancel_rate_threshold=0.7
  )

  detector = SpoofingDetector(config)

  # Симулируем spoofing паттерн
  base_time = int(datetime.now().timestamp() * 1000)

  # Snapshot 1: Большой bid ордер появляется
  snapshot1 = OrderBookSnapshot(
    symbol="BTCUSDT",
    bids=[
      OrderBookLevel(50000.0, 10.0),  # $500k - крупный ордер
      OrderBookLevel(49999.0, 1.0),
    ],
    asks=[
      OrderBookLevel(50001.0, 1.0),
    ],
    timestamp=base_time
  )

  detector.update(snapshot1)

  # Snapshot 2: 5 секунд спустя, ордер все еще там
  snapshot2 = OrderBookSnapshot(
    symbol="BTCUSDT",
    bids=[
      OrderBookLevel(50000.0, 10.0),
      OrderBookLevel(49999.0, 1.0),
    ],
    asks=[
      OrderBookLevel(50001.0, 1.0),
    ],
    timestamp=base_time + 5000
  )

  detector.update(snapshot2)

  # Snapshot 3: 8 секунд спустя, крупный ордер отменен (spoofing!)
  snapshot3 = OrderBookSnapshot(
    symbol="BTCUSDT",
    bids=[
      OrderBookLevel(49999.0, 1.0),  # Крупный bid исчез
    ],
    asks=[
      OrderBookLevel(50001.0, 1.0),
    ],
    timestamp=base_time + 8000
  )

  detector.update(snapshot3)

  # Проверяем обнаружение
  patterns = detector.get_recent_patterns("BTCUSDT", time_window_seconds=60)

  print(f"Обнаружено паттернов: {len(patterns)}")
  for pattern in patterns:
    print(f"  Side: {pattern.side}")
    print(f"  Confidence: {pattern.confidence:.2f}")
    print(f"  Reason: {pattern.reason}")

  # Статистика
  stats = detector.get_statistics()
  print(f"\nСтатистика:")
  print(f"  Total checks: {stats['total_checks']}")
  print(f"  Patterns detected: {stats['patterns_detected']}")