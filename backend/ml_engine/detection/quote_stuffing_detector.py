"""
Professional Industry-Standard Quote Stuffing Detector.

Quote Stuffing (HFT Manipulation):
Массовая отправка и отмена ордеров за миллисекунды для создания
информационного шума, перегрузки системы, и сокрытия реальных намерений.

Industry-Standard Approach:
1. Order Update Rate Analysis: Измерение частоты обновлений orderbook
2. Micro Order Detection: Выявление аномально малых ордеров
3. High Cancellation Rate: 95%+ cancellations в короткий период
4. Price Range Concentration: Узкий диапазон цен (< 5 bps)
5. Temporal Pattern Analysis: Burst activity с последующим затишьем

Key Patterns:
- 20+ orderbook updates per second (нормально: 1-5/sec)
- 95%+ cancellation rate (нормально: 30-50%)
- Micro orders < 0.01 BTC (institutional: 0.1+ BTC)
- Price concentration < 5 basis points
- Burst-idle cycles (30 sec burst → 60 sec idle)

Path: backend/ml_engine/detection/quote_stuffing_detector.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
import time

from backend.core.logger import get_logger
from backend.models.orderbook import OrderBookSnapshot

logger = get_logger(__name__)


@dataclass
class QuoteStuffingConfig:
  """Configuration for quote stuffing detection."""
  # Update rate thresholds
  min_updates_per_second: float = 20.0      # 20+ updates/sec suspicious
  high_updates_per_second: float = 50.0     # 50+ very suspicious
  extreme_updates_per_second: float = 100.0 # 100+ extreme

  # Order characteristics
  min_cancellation_rate: float = 0.90       # 90%+ cancellations
  high_cancellation_rate: float = 0.95      # 95%+ very suspicious
  max_avg_order_size_btc: float = 0.01      # Micro orders

  # Price concentration
  max_price_range_bps: float = 5.0          # < 5 bps concentrated

  # Detection windows
  detection_window_seconds: float = 5.0     # Analyze last 5 seconds
  burst_detection_window: float = 30.0      # Burst period
  idle_detection_window: float = 60.0       # Idle period after burst

  # Minimum thresholds
  min_orders_for_detection: int = 20        # Need 20+ orders
  min_confidence: float = 0.65              # Minimum confidence


@dataclass
class QuoteStuffingPattern:
  """Detected quote stuffing pattern."""
  symbol: str
  timestamp: int
  confidence: float

  # Update metrics
  updates_per_second: float
  total_updates: int

  # Order metrics
  order_count: int
  avg_order_size: float
  total_volume: float

  # Cancellation metrics
  cancellation_rate: float
  cancellation_count: int

  # Price metrics
  price_range_bps: float
  min_price: float
  max_price: float

  # Pattern type
  pattern_type: str  # "burst", "sustained", "cyclic"

  # Scoring components
  update_rate_score: float
  cancellation_score: float
  order_size_score: float
  concentration_score: float

  reason: str


class OrderBookUpdateTracker:
  """Tracks orderbook update frequency and patterns."""

  def __init__(self, symbol: str):
    """
    Initialize update tracker.

    Args:
        symbol: Trading pair
    """
    self.symbol = symbol

    # Update timestamps (milliseconds)
    # MEMORY FIX: 1000 → 200 timestamps (sufficient for 10-40 sec analysis)
    self.update_timestamps: deque = deque(maxlen=200)

    # Update content tracking
    # MEMORY FIX: 100 → 20 full snapshots (80% reduction, saves ~500KB per symbol)
    self.update_snapshots: deque = deque(maxlen=20)

    # Statistics
    self.total_updates = 0
    self.last_burst_time: Optional[int] = None

  def add_update(self, snapshot: OrderBookSnapshot):
    """Record orderbook update."""
    self.update_timestamps.append(snapshot.timestamp)
    self.update_snapshots.append(snapshot)
    self.total_updates += 1

  def calculate_update_rate(
      self,
      window_seconds: float,
      current_time: int
  ) -> float:
    """
    Calculate updates per second over window.

    Args:
        window_seconds: Time window in seconds
        current_time: Current timestamp (ms)

    Returns:
        Updates per second
    """
    cutoff_time = current_time - int(window_seconds * 1000)

    recent_updates = [
      ts for ts in self.update_timestamps
      if ts >= cutoff_time
    ]

    if not recent_updates or window_seconds == 0:
      return 0.0

    return len(recent_updates) / window_seconds

  def get_recent_snapshots(
      self,
      window_seconds: float,
      current_time: int
  ) -> List[OrderBookSnapshot]:
    """Get recent snapshots within window."""
    cutoff_time = current_time - int(window_seconds * 1000)

    return [
      snap for snap in self.update_snapshots
      if snap.timestamp >= cutoff_time
    ]

  def detect_burst_pattern(
      self,
      current_time: int,
      burst_window: float,
      idle_window: float
  ) -> bool:
    """
    Detect burst-idle cycle pattern.

    Returns:
        True if burst detected followed by relative idle
    """
    # Calculate rate in burst window
    burst_rate = self.calculate_update_rate(burst_window, current_time)

    # Calculate rate in previous idle window
    idle_start = current_time - int((burst_window + idle_window) * 1000)
    idle_end = current_time - int(burst_window * 1000)

    idle_updates = [
      ts for ts in self.update_timestamps
      if idle_start <= ts < idle_end
    ]

    idle_rate = len(idle_updates) / idle_window if idle_window > 0 else 0.0

    # Burst: high rate now, low rate before
    return burst_rate > 20.0 and idle_rate < 5.0


class QuoteStuffingDetector:
  """
  Professional Industry-Standard Quote Stuffing Detector.

  Algorithm:
  1. Track orderbook update frequency (updates per second)
  2. Analyze order characteristics (size, cancellations, price range)
  3. Detect burst patterns (high activity → low activity cycles)
  4. Multi-factor confidence scoring
  5. Real-time alerts for HFT manipulation
  """

  def __init__(self, config: QuoteStuffingConfig):
    """
    Initialize quote stuffing detector.

    Args:
        config: Configuration parameters
    """
    self.config = config

    # Update trackers per symbol
    self.update_trackers: Dict[str, OrderBookUpdateTracker] = {}

    # Detected patterns
    self.detected_patterns: Dict[str, List[QuoteStuffingPattern]] = defaultdict(list)

    # Statistics
    self.total_checks = 0
    self.patterns_detected = 0

    # Last detection time (for cooldown)
    self.last_detection_time: Dict[str, int] = {}
    self.detection_cooldown_ms = 5000  # 5 seconds

    logger.info(
      f"✅ QuoteStuffingDetector initialized: "
      f"min_updates={config.min_updates_per_second}/sec, "
      f"min_cancel_rate={config.min_cancellation_rate:.0%}"
    )

  def update(self, snapshot: OrderBookSnapshot):
    """
    Update detector with new orderbook snapshot.

    Args:
        snapshot: OrderBook snapshot
    """
    symbol = snapshot.symbol
    timestamp = snapshot.timestamp

    # Initialize tracker if needed
    if symbol not in self.update_trackers:
      self.update_trackers[symbol] = OrderBookUpdateTracker(symbol)

    # Record update
    tracker = self.update_trackers[symbol]
    tracker.add_update(snapshot)

    # Check for quote stuffing periodically
    self.total_checks += 1

    # Calculate current update rate
    update_rate = tracker.calculate_update_rate(
      self.config.detection_window_seconds,
      timestamp
    )

    # Trigger detection if high update rate
    if update_rate >= self.config.min_updates_per_second:
      self._detect_stuffing(symbol, timestamp)

  def _detect_stuffing(self, symbol: str, timestamp: int):
    """
    Detect quote stuffing pattern.

    Args:
        symbol: Trading pair
        timestamp: Current timestamp
    """
    # Cooldown check
    if symbol in self.last_detection_time:
      time_since_last = timestamp - self.last_detection_time[symbol]
      if time_since_last < self.detection_cooldown_ms:
        return

    tracker = self.update_trackers[symbol]

    # Get recent snapshots
    recent_snapshots = tracker.get_recent_snapshots(
      self.config.detection_window_seconds,
      timestamp
    )

    if len(recent_snapshots) < 2:
      return

    # Analyze pattern
    pattern = self._analyze_stuffing_pattern(
      symbol,
      timestamp,
      tracker,
      recent_snapshots
    )

    if pattern and pattern.confidence >= self.config.min_confidence:
      self.detected_patterns[symbol].append(pattern)
      self.patterns_detected += 1
      self.last_detection_time[symbol] = timestamp

      logger.warning(
        f"🚨 QUOTE STUFFING DETECTED [{symbol}]: "
        f"updates={pattern.updates_per_second:.1f}/sec, "
        f"cancel_rate={pattern.cancellation_rate:.1%}, "
        f"confidence={pattern.confidence:.2%}, "
        f"type={pattern.pattern_type}"
      )

  def _analyze_stuffing_pattern(
      self,
      symbol: str,
      timestamp: int,
      tracker: OrderBookUpdateTracker,
      recent_snapshots: List[OrderBookSnapshot]
  ) -> Optional[QuoteStuffingPattern]:
    """
    Analyze if recent activity is quote stuffing.

    Args:
        symbol: Trading pair
        timestamp: Current timestamp
        tracker: Update tracker
        recent_snapshots: Recent orderbook snapshots

    Returns:
        QuoteStuffingPattern if detected, None otherwise
    """
    window_sec = self.config.detection_window_seconds

    # Calculate update rate
    updates_per_second = tracker.calculate_update_rate(window_sec, timestamp)
    total_updates = len(recent_snapshots)

    # Analyze order changes across snapshots
    order_metrics = self._analyze_order_metrics(recent_snapshots)

    if not order_metrics:
      return None

    order_count = order_metrics['order_count']
    avg_order_size = order_metrics['avg_order_size']
    total_volume = order_metrics['total_volume']
    cancellation_rate = order_metrics['cancellation_rate']
    cancellation_count = order_metrics['cancellation_count']
    price_range_bps = order_metrics['price_range_bps']
    min_price = order_metrics['min_price']
    max_price = order_metrics['max_price']

    # Minimum order threshold
    if order_count < self.config.min_orders_for_detection:
      return None

    # Detect pattern type
    is_burst = tracker.detect_burst_pattern(
      timestamp,
      self.config.burst_detection_window,
      self.config.idle_detection_window
    )

    if is_burst:
      pattern_type = "burst"
    elif updates_per_second > self.config.high_updates_per_second:
      pattern_type = "sustained"
    else:
      pattern_type = "elevated"

    # Multi-factor confidence scoring
    scores = self._calculate_stuffing_confidence(
      updates_per_second=updates_per_second,
      cancellation_rate=cancellation_rate,
      avg_order_size=avg_order_size,
      price_range_bps=price_range_bps,
      pattern_type=pattern_type
    )

    confidence = scores['total']

    if confidence < self.config.min_confidence:
      return None

    # Build reason string
    reason = (
      f"{total_updates} updates at {updates_per_second:.1f}/sec, "
      f"{order_count} orders (avg {avg_order_size:.4f} BTC), "
      f"cancel_rate={cancellation_rate:.1%}, "
      f"price_range={price_range_bps:.1f}bps, "
      f"pattern={pattern_type}"
    )

    return QuoteStuffingPattern(
      symbol=symbol,
      timestamp=timestamp,
      confidence=confidence,
      updates_per_second=updates_per_second,
      total_updates=total_updates,
      order_count=order_count,
      avg_order_size=avg_order_size,
      total_volume=total_volume,
      cancellation_rate=cancellation_rate,
      cancellation_count=cancellation_count,
      price_range_bps=price_range_bps,
      min_price=min_price,
      max_price=max_price,
      pattern_type=pattern_type,
      update_rate_score=scores['update_rate'],
      cancellation_score=scores['cancellation'],
      order_size_score=scores['order_size'],
      concentration_score=scores['concentration'],
      reason=reason
    )

  def _analyze_order_metrics(
      self,
      snapshots: List[OrderBookSnapshot]
  ) -> Optional[Dict]:
    """
    Analyze order characteristics across snapshots.

    Returns:
        Dict with order metrics or None
    """
    if len(snapshots) < 2:
      return None

    # Track orders across snapshots
    all_orders = {}  # price -> [volumes across time]

    for snapshot in snapshots:
      current_orders = {}

      # Combine bids and asks
      for price, qty in snapshot.bids + snapshot.asks:
        current_orders[price] = qty

      # Track in all_orders
      for price, qty in current_orders.items():
        if price not in all_orders:
          all_orders[price] = []
        all_orders[price].append(qty)

    if not all_orders:
      return None

    # Calculate metrics
    order_count = len(all_orders)

    # Average order size (mean across all observations)
    all_sizes = []
    for volumes in all_orders.values():
      all_sizes.extend(volumes)

    avg_order_size = float(np.mean(all_sizes)) if all_sizes else 0.0
    total_volume = float(sum(all_sizes))

    # Cancellation rate (orders that appeared then disappeared)
    cancelled_orders = 0
    for price, volumes in all_orders.items():
      # If order appeared (volume > 0) then disappeared (not in last snapshot)
      if len(volumes) > 1 and volumes[0] > 0:
        # Check if it's still in the last snapshot
        last_snapshot_prices = set(
          [p for p, _ in snapshots[-1].bids] + [p for p, _ in snapshots[-1].asks]
        )
        if price not in last_snapshot_prices:
          cancelled_orders += 1

    cancellation_rate = cancelled_orders / order_count if order_count > 0 else 0.0

    # Price range (basis points)
    prices = list(all_orders.keys())
    if len(prices) > 1:
      min_price = min(prices)
      max_price = max(prices)
      avg_price = np.mean(prices)
      price_range_bps = ((max_price - min_price) / avg_price) * 10000
    else:
      min_price = prices[0] if prices else 0.0
      max_price = min_price
      price_range_bps = 0.0

    return {
      'order_count': order_count,
      'avg_order_size': avg_order_size,
      'total_volume': total_volume,
      'cancellation_rate': cancellation_rate,
      'cancellation_count': cancelled_orders,
      'price_range_bps': price_range_bps,
      'min_price': min_price,
      'max_price': max_price
    }

  def _calculate_stuffing_confidence(
      self,
      updates_per_second: float,
      cancellation_rate: float,
      avg_order_size: float,
      price_range_bps: float,
      pattern_type: str
  ) -> Dict[str, float]:
    """
    Multi-factor confidence scoring for quote stuffing.

    Returns:
        Dict with component scores and total
    """
    scores = {
      'update_rate': 0.0,
      'cancellation': 0.0,
      'order_size': 0.0,
      'concentration': 0.0,
      'total': 0.0
    }

    # 1. Update Rate Score (weight: 0.30)
    if updates_per_second >= self.config.extreme_updates_per_second:
      scores['update_rate'] = 0.30
    elif updates_per_second >= self.config.high_updates_per_second:
      scores['update_rate'] = 0.25
    elif updates_per_second >= self.config.min_updates_per_second:
      scores['update_rate'] = 0.15

    # 2. Cancellation Score (weight: 0.30)
    if cancellation_rate >= self.config.high_cancellation_rate:
      scores['cancellation'] = 0.30
    elif cancellation_rate >= self.config.min_cancellation_rate:
      scores['cancellation'] = 0.20
    elif cancellation_rate >= 0.70:
      scores['cancellation'] = 0.10

    # 3. Order Size Score (weight: 0.20)
    if avg_order_size <= self.config.max_avg_order_size_btc:
      scores['order_size'] = 0.20
    elif avg_order_size <= 0.05:
      scores['order_size'] = 0.10

    # 4. Concentration Score (weight: 0.20)
    if price_range_bps <= self.config.max_price_range_bps:
      scores['concentration'] = 0.20
    elif price_range_bps <= 10.0:
      scores['concentration'] = 0.10

    # Bonus for burst pattern
    if pattern_type == "burst":
      scores['update_rate'] += 0.05

    # Total confidence
    scores['total'] = min(
      scores['update_rate'] +
      scores['cancellation'] +
      scores['order_size'] +
      scores['concentration'],
      1.0
    )

    return scores

  def get_recent_patterns(
      self,
      symbol: str,
      time_window_seconds: int = 60
  ) -> List[QuoteStuffingPattern]:
    """Get recent quote stuffing patterns."""
    current_time = int(time.time() * 1000)
    cutoff_time = current_time - (time_window_seconds * 1000)

    patterns = self.detected_patterns.get(symbol, [])

    return [
      p for p in patterns
      if p.timestamp >= cutoff_time
    ]

  def is_stuffing_active(
      self,
      symbol: str,
      time_window_seconds: int = 30
  ) -> bool:
    """Check if quote stuffing is currently active."""
    patterns = self.get_recent_patterns(symbol, time_window_seconds)
    return len(patterns) > 0

  def get_statistics(self) -> Dict:
    """Get detector statistics."""
    total_patterns = sum(
      len(patterns)
      for patterns in self.detected_patterns.values()
    )

    return {
      'total_checks': self.total_checks,
      'patterns_detected': self.patterns_detected,
      'total_patterns': total_patterns,
      'symbols_monitored': len(self.update_trackers),
      'detection_rate': (
        self.patterns_detected / self.total_checks
        if self.total_checks > 0
        else 0.0
      )
    }


# Example usage
if __name__ == "__main__":
  config = QuoteStuffingConfig()
  detector = QuoteStuffingDetector(config)

  print("QuoteStuffingDetector initialized")
  print(f"Config: {config}")
