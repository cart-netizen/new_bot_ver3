"""
Professional Industry-Standard Layering Detector.

Layering (Market Manipulation) Detection:
Размещение множественных ордеров на разных уровнях цены для создания
ложного впечатления о спросе/предложении с целью манипулирования ценой.

Industry-Standard Approach:
1. Two-Sided Analysis: Spoofing side (large fake orders) + Execution side (real trades)
2. Temporal Correlation: Placement timing → Trade execution → Quick cancellation
3. Price Impact Analysis: Expected vs actual market impact
4. Volume Ratio Analysis: Spoofing volume vs execution volume
5. Professional Confidence Scoring: Multi-factor weighted algorithm

Key Patterns:
- Multiple orders at similar price levels (layering)
- Orders placed sequentially in short time window
- Similar order sizes (systematic placement)
- One-sided orderbook manipulation
- Correlated trade activity on opposite side
- Quick cancellation after price movement or trade execution
- High volume ratio (spoofing >> execution)

Path: backend/ml_engine/detection/layering_detector.py
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
import time
import asyncio

from backend.core.logger import get_logger
from backend.models.orderbook import OrderBookSnapshot, OrderBookLevel

# Import new components (optional)
try:
  from backend.ml_engine.detection.pattern_database import HistoricalPatternDatabase
  from backend.ml_engine.detection.layering_data_collector import LayeringDataCollector
  from backend.ml_engine.detection.adaptive_layering_model import AdaptiveLayeringModel
  ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
  ADVANCED_FEATURES_AVAILABLE = False
  HistoricalPatternDatabase = None
  LayeringDataCollector = None
  AdaptiveLayeringModel = None

if TYPE_CHECKING:
  from backend.strategy.trade_manager import TradeManager

logger = get_logger(__name__)


@dataclass
class LayeringConfig:
  """Professional configuration for layering detection."""
  # Layer identification thresholds
  min_orders_in_layer: int = 3  # Minimum orders to form a layer
  max_price_spread_pct: float = 0.005  # Max price spread within layer (0.5%)

  # Temporal thresholds
  placement_window_seconds: float = 30.0  # Window for order placement
  cancellation_window_seconds: float = 60.0  # Window for cancellation detection
  execution_correlation_window: float = 60.0  # Window for trade correlation

  # Volume thresholds
  volume_similarity_threshold: float = 0.3  # Max 30% volume variance (CV)
  min_layer_volume_btc: float = 0.5  # Minimum 0.5 BTC per layer

  # Spoofing vs Execution thresholds
  min_spoofing_execution_ratio: float = 5.0  # Spoofing volume / Execution volume
  high_spoofing_execution_ratio: float = 10.0  # Very suspicious ratio

  # Price impact thresholds
  min_expected_impact_bps: float = 5.0  # Minimum expected impact (basis points)
  low_actual_impact_multiplier: float = 0.3  # Actual impact < 30% of expected

  # History and detection
  history_window_seconds: int = 300  # 5 minutes history
  min_confidence: float = 0.65  # Minimum confidence for alert

  # Event-driven detection
  check_on_cancellation: bool = True  # Trigger check on order cancellations
  check_on_trade_burst: bool = True  # Trigger check on trade activity spikes
  min_cancellation_rate: float = 0.5  # 50%+ orders cancelled = suspicious


@dataclass
class OrderLayer:
  """Layer of orders at similar price levels."""
  side: str  # "bid" or "ask"
  prices: List[float]
  volumes: List[float]
  timestamps: List[int]

  # Layer metrics
  price_spread: float
  total_volume: float
  avg_volume: float
  volume_std: float
  placement_duration: float  # seconds

  @property
  def order_count(self) -> int:
    return len(self.prices)

  @property
  def mid_price(self) -> float:
    """Average price within the layer."""
    return float(np.mean(self.prices))

  @property
  def min_price(self) -> float:
    return float(min(self.prices))

  @property
  def max_price(self) -> float:
    return float(max(self.prices))

  @property
  def volume_cv(self) -> float:
    """Coefficient of Variation for volumes."""
    if self.avg_volume > 0:
      return self.volume_std / self.avg_volume
    return 0.0


@dataclass
class ExecutionMetrics:
  """Metrics from actual trade execution on opposite side."""
  side: str  # "bid" or "ask" - side where trades occurred
  total_volume: float  # Total volume executed
  trade_count: int  # Number of trades
  avg_trade_size: float  # Average trade size
  aggressive_ratio: float  # Ratio of aggressive trades
  time_since_placement: float  # Time since layer placement (seconds)
  correlation_score: float  # Temporal correlation with placement


@dataclass
class PriceImpactMetrics:
  """Price impact analysis metrics."""
  initial_price: float  # Price before layering
  current_price: float  # Current price
  price_change_bps: float  # Actual price change in basis points
  expected_impact_bps: float  # Expected impact based on volume
  impact_ratio: float  # Actual / Expected (< 1.0 = suspicious)
  direction_matches: bool  # Price moved in expected direction


@dataclass
class LayeringPattern:
  """Detected layering manipulation pattern."""
  symbol: str
  timestamp: int

  # Pattern identification
  spoofing_side: str  # Side with fake orders (bid/ask)
  execution_side: Optional[str]  # Side with real trades (opposite)
  confidence: float

  # Spoofing details
  layers: List[OrderLayer]
  total_orders: int
  total_spoofing_volume: float
  placement_duration: float
  cancellation_detected: bool
  cancellation_rate: float

  # Execution details (if available)
  execution_metrics: Optional[ExecutionMetrics]
  spoofing_execution_ratio: Optional[float]

  # Price impact
  price_impact: Optional[PriceImpactMetrics]

  # Pattern scoring components
  volume_score: float
  timing_score: float
  cancellation_score: float
  execution_correlation_score: float
  price_impact_score: float

  reason: str


@dataclass
class PendingValidationPattern:
  """Pattern pending price validation for ML labeling."""
  data_id: str  # ML data collection ID
  pattern: LayeringPattern
  symbol: str
  entry_price: float
  entry_timestamp: int
  expected_direction: str  # "up" or "down"
  validation_window_seconds: int  # How long to wait

  # Will be populated after validation
  validation_completed: bool = False
  price_moved_as_expected: Optional[bool] = None
  actual_price_change_bps: Optional[float] = None


class OrderTracker:
  """Professional order tracking with comprehensive history."""

  def __init__(self, symbol: str, side: str):
    """
    Initialize order tracker.

    Args:
        symbol: Trading pair
        side: Side ("bid" or "ask")
    """
    self.symbol = symbol
    self.side = side

    # Order history: price -> [(timestamp, volume), ...]
    self.order_history: Dict[float, List[Tuple[int, float]]] = defaultdict(list)

    # Current active orders
    self.active_orders: Dict[float, float] = {}  # price -> volume

    # Cancellation tracking
    self.recent_cancellations: deque = deque(maxlen=100)
    self.placement_times: Dict[float, int] = {}  # price -> first_placement_time

  def update(self, levels: List[OrderBookLevel], timestamp: int):
    """Update tracker with new orderbook levels."""
    current_prices = {level.price: level.quantity for level in levels}

    # New orders
    new_prices = set(current_prices.keys()) - set(self.active_orders.keys())
    for price in new_prices:
      volume = current_prices[price]
      self.order_history[price].append((timestamp, volume))
      if price not in self.placement_times:
        self.placement_times[price] = timestamp

    # Updated orders (volume change)
    for price in set(current_prices.keys()) & set(self.active_orders.keys()):
      if current_prices[price] != self.active_orders[price]:
        self.order_history[price].append((timestamp, current_prices[price]))

    # Cancelled orders
    cancelled_prices = set(self.active_orders.keys()) - set(current_prices.keys())
    for price in cancelled_prices:
      self.order_history[price].append((timestamp, 0.0))  # 0 = cancelled

      # Track cancellation with lifetime
      if price in self.placement_times:
        lifetime = (timestamp - self.placement_times[price]) / 1000.0
        self.recent_cancellations.append({
          'price': price,
          'timestamp': timestamp,
          'lifetime': lifetime,
          'side': self.side
        })
        del self.placement_times[price]

    # Update active orders
    self.active_orders = current_prices

  def find_recent_placements(
      self,
      window_seconds: float,
      current_time: int
  ) -> List[Tuple[float, int, float]]:
    """
    Find recently placed orders.

    Returns:
        List[(price, timestamp, volume)]
    """
    cutoff_time = current_time - int(window_seconds * 1000)

    placements = []
    for price, history in self.order_history.items():
      for timestamp, volume in history:
        if timestamp >= cutoff_time and volume > 0:
          # This is placement or increase
          placements.append((price, timestamp, volume))

    return placements

  def find_recent_cancellations(
      self,
      window_seconds: float,
      current_time: int
  ) -> List[Tuple[float, int]]:
    """
    Find recently cancelled orders.

    Returns:
        List[(price, timestamp)]
    """
    cutoff_time = current_time - int(window_seconds * 1000)

    cancellations = []
    for price, history in self.order_history.items():
      for timestamp, volume in history:
        if timestamp >= cutoff_time and volume == 0:
          cancellations.append((price, timestamp))

    return cancellations

  def get_cancellation_rate(
      self,
      window_seconds: float,
      current_time: int
  ) -> float:
    """
    Calculate cancellation rate (cancelled orders / total placed orders).

    Returns:
        float: Cancellation rate (0.0 to 1.0)
    """
    placements = self.find_recent_placements(window_seconds, current_time)
    cancellations = self.find_recent_cancellations(window_seconds, current_time)

    if len(placements) == 0:
      return 0.0

    # Count unique prices that were cancelled
    cancelled_prices = set(price for price, _ in cancellations)
    placed_prices = set(price for price, _, _ in placements)

    if len(placed_prices) == 0:
      return 0.0

    return len(cancelled_prices) / len(placed_prices)

  def cleanup_old_history(self, cutoff_time: int):
    """Remove old history data."""
    for price in list(self.order_history.keys()):
      # Filter old entries
      self.order_history[price] = [
        (ts, vol) for ts, vol in self.order_history[price]
        if ts >= cutoff_time
      ]

      # Remove empty entries
      if not self.order_history[price]:
        del self.order_history[price]

    # Cleanup old placement times
    for price in list(self.placement_times.keys()):
      if price not in self.active_orders:
        # Order was cancelled or removed from history
        if price not in self.order_history:
          del self.placement_times[price]


class LayeringDetector:
  """
  Professional Industry-Standard Layering Detector.

  Algorithm:
  1. Track order placement patterns on both sides (bid/ask)
  2. Group similar orders into layers (clustering by price/time/volume)
  3. Analyze actual trade execution on opposite side (via TradeManager)
  4. Calculate temporal correlation (placement → trades → cancellation)
  5. Compute price impact (expected vs actual)
  6. Multi-factor confidence scoring
  7. Event-driven detection for real-time alerts
  """

  def __init__(
      self,
      config: LayeringConfig,
      trade_managers: Optional[Dict[str, 'TradeManager']] = None,
      pattern_database: Optional['HistoricalPatternDatabase'] = None,
      data_collector: Optional['LayeringDataCollector'] = None,
      adaptive_model: Optional['AdaptiveLayeringModel'] = None,
      enable_ml_features: bool = True
  ):
    """
    Initialize professional layering detector.

    Args:
        config: Configuration parameters
        trade_managers: Dict of TradeManagers for each symbol (for execution analysis)
        pattern_database: Historical pattern database for learning
        data_collector: Data collector for ML training
        adaptive_model: ML model for adaptive thresholds
        enable_ml_features: Enable advanced ML features
    """
    self.config = config
    self.trade_managers = trade_managers or {}

    # Advanced ML components (optional)
    self.pattern_database = pattern_database
    self.data_collector = data_collector
    self.adaptive_model = adaptive_model
    self.enable_ml_features = enable_ml_features and ADVANCED_FEATURES_AVAILABLE

    # Trackers for each symbol: symbol -> side -> OrderTracker
    self.trackers: Dict[str, Dict[str, OrderTracker]] = {}

    # Price history for impact analysis: symbol -> deque[(timestamp, mid_price)]
    self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    # Detected patterns
    self.detected_patterns: Dict[str, List[LayeringPattern]] = defaultdict(list)

    # Statistics
    self.total_checks = 0
    self.patterns_detected = 0
    self.event_driven_checks = 0

    # Last detection time per symbol (for throttling)
    self.last_detection_time: Dict[str, int] = {}
    self.detection_cooldown_ms = 5000  # 5 seconds cooldown

    # ML auto-labeling: patterns pending price validation
    self.pending_validations: List[PendingValidationPattern] = []
    self.validation_tasks: List[asyncio.Task] = []  # Track running validation tasks

    logger.info(
      f"✅ Professional LayeringDetector initialized: "
      f"min_orders={config.min_orders_in_layer}, "
      f"price_spread={config.max_price_spread_pct:.2%}, "
      f"trade_integration={'✅' if trade_managers else '❌'}, "
      f"ml_features={'✅' if self.enable_ml_features else '❌'}"
    )

    if self.enable_ml_features:
      logger.info(
        f"   ├─ Pattern Database: {'✅' if pattern_database else '❌'}"
      )
      logger.info(
        f"   ├─ Data Collector: {'✅' if data_collector else '❌'}"
      )
      logger.info(
        f"   └─ Adaptive Model: {'✅' if adaptive_model else '❌'}"
      )

  def update(self, snapshot: OrderBookSnapshot):
    """
    Update detector with new orderbook snapshot.
    Real-time event-driven detection.

    Args:
        snapshot: OrderBook snapshot
    """
    symbol = snapshot.symbol
    timestamp = snapshot.timestamp

    # Initialize trackers if needed
    if symbol not in self.trackers:
      self.trackers[symbol] = {
        "bid": OrderTracker(symbol, "bid"),
        "ask": OrderTracker(symbol, "ask")
      }

    # Store price history for impact analysis
    if snapshot.mid_price:
      self.price_history[symbol].append((timestamp, snapshot.mid_price))

    # Convert tuples to OrderBookLevel
    bid_levels = [
      OrderBookLevel(price=price, quantity=qty)
      for price, qty in snapshot.bids
    ]
    ask_levels = [
      OrderBookLevel(price=price, quantity=qty)
      for price, qty in snapshot.asks
    ]

    # Update trackers and detect cancellations
    bid_tracker = self.trackers[symbol]["bid"]
    ask_tracker = self.trackers[symbol]["ask"]

    prev_bid_count = len(bid_tracker.active_orders)
    prev_ask_count = len(ask_tracker.active_orders)

    bid_tracker.update(bid_levels, timestamp)
    ask_tracker.update(ask_levels, timestamp)

    # Event-driven detection triggers
    self.total_checks += 1

    # Trigger 1: Significant cancellations detected
    if self.config.check_on_cancellation:
      bid_cancellations = prev_bid_count - len(bid_tracker.active_orders)
      ask_cancellations = prev_ask_count - len(ask_tracker.active_orders)

      if bid_cancellations >= 3 or ask_cancellations >= 3:
        # Significant cancellations - check for layering
        self._detect_patterns_event_driven(
          symbol, timestamp, snapshot.mid_price,
          trigger="cancellations"
        )

    # Trigger 2: Trade burst detection (if TradeManager available)
    if self.config.check_on_trade_burst and symbol in self.trade_managers:
      trade_manager = self.trade_managers[symbol]
      arrival_rate = trade_manager.calculate_arrival_rate(window_seconds=10)

      # High arrival rate = potential execution after layering
      if arrival_rate > 5.0:  # More than 5 trades per second
        self._detect_patterns_event_driven(
          symbol, timestamp, snapshot.mid_price,
          trigger="trade_burst"
        )

    # Trigger 3: Periodic check (every 50 updates as fallback)
    if self.total_checks % 50 == 0:
      self._detect_patterns(symbol, timestamp, snapshot.mid_price)
      self._cleanup_old_data(symbol, timestamp)

  def _detect_patterns_event_driven(
      self,
      symbol: str,
      timestamp: int,
      mid_price: Optional[float],
      trigger: str
  ):
    """
    Event-driven pattern detection with cooldown.

    Args:
        symbol: Trading pair
        timestamp: Current timestamp
        mid_price: Current mid price
        trigger: Event that triggered detection
    """
    # Cooldown check (prevent spam)
    if symbol in self.last_detection_time:
      time_since_last = timestamp - self.last_detection_time[symbol]
      if time_since_last < self.detection_cooldown_ms:
        return

    self.event_driven_checks += 1
    self.last_detection_time[symbol] = timestamp

    logger.debug(
      f"🔍 Event-driven layering check: {symbol}, trigger={trigger}"
    )

    self._detect_patterns(symbol, timestamp, mid_price)

  def _detect_patterns(
      self,
      symbol: str,
      timestamp: int,
      mid_price: Optional[float]
  ):
    """Detect layering patterns with two-sided analysis."""
    if mid_price is None or mid_price <= 0:
      return

    # Analyze both sides for potential spoofing
    for spoofing_side in ["bid", "ask"]:
      execution_side = "ask" if spoofing_side == "bid" else "bid"

      pattern = self._analyze_two_sided_layering(
        symbol,
        spoofing_side,
        execution_side,
        timestamp,
        mid_price
      )

      if pattern:
        self.detected_patterns[symbol].append(pattern)
        self.patterns_detected += 1

        logger.warning(
          f"🚨 LAYERING DETECTED [{symbol}]: "
          f"spoofing_side={pattern.spoofing_side}, "
          f"layers={len(pattern.layers)}, "
          f"confidence={pattern.confidence:.2%}, "
          f"ratio={pattern.spoofing_execution_ratio:.1f}x, "
          f"reason={pattern.reason}"
        )

  def _analyze_two_sided_layering(
      self,
      symbol: str,
      spoofing_side: str,
      execution_side: str,
      timestamp: int,
      mid_price: float
  ) -> Optional[LayeringPattern]:
    """
    Professional two-sided layering analysis.

    Analyzes:
    1. Spoofing side: Large order layers
    2. Execution side: Actual trade activity
    3. Temporal correlation between placement and execution
    4. Price impact analysis
    5. Multi-factor confidence scoring

    Args:
        symbol: Trading pair
        spoofing_side: Side with potential fake orders
        execution_side: Side with potential real execution
        timestamp: Current timestamp
        mid_price: Current mid price (USDT price for calculation)

    Returns:
        LayeringPattern if detected, None otherwise
    """
    spoofing_tracker = self.trackers[symbol][spoofing_side]

    # ===== STEP 1: Analyze spoofing side for layers =====
    placements = spoofing_tracker.find_recent_placements(
      self.config.placement_window_seconds,
      timestamp
    )

    if len(placements) < self.config.min_orders_in_layer:
      return None

    # Group orders into layers
    layers = self._group_into_layers(placements, spoofing_side, mid_price)

    if not layers:
      return None

    # Filter valid layers (fixed: use mid_price for USDT conversion)
    valid_layers = [
      layer for layer in layers
      if self._is_valid_layer(layer, mid_price)
    ]

    if not valid_layers:
      return None

    # Calculate spoofing metrics
    total_orders = sum(layer.order_count for layer in valid_layers)
    total_spoofing_volume = sum(layer.total_volume for layer in valid_layers)

    # Placement duration
    all_timestamps = []
    for layer in valid_layers:
      all_timestamps.extend(layer.timestamps)

    if len(all_timestamps) > 1:
      placement_duration = (max(all_timestamps) - min(all_timestamps)) / 1000.0
    else:
      placement_duration = 0.0

    # ===== STEP 2: Check cancellations =====
    cancellations = spoofing_tracker.find_recent_cancellations(
      self.config.cancellation_window_seconds,
      timestamp
    )

    cancellation_detected = len(cancellations) >= self.config.min_orders_in_layer
    cancellation_rate = spoofing_tracker.get_cancellation_rate(
      self.config.cancellation_window_seconds,
      timestamp
    )

    # ===== STEP 3: Analyze execution side (if TradeManager available) =====
    execution_metrics = None
    spoofing_execution_ratio = None

    if symbol in self.trade_managers:
      execution_metrics = self._analyze_execution_side(
        symbol,
        execution_side,
        timestamp,
        placement_duration
      )

      if execution_metrics and execution_metrics.total_volume > 0:
        spoofing_execution_ratio = (
          total_spoofing_volume / execution_metrics.total_volume
        )

    # ===== STEP 4: Price impact analysis =====
    price_impact = self._analyze_price_impact(
      symbol,
      timestamp,
      spoofing_side,
      total_spoofing_volume,
      mid_price
    )

    # ===== STEP 5: Multi-factor confidence scoring =====
    confidence_components = self._calculate_professional_confidence(
      layers=valid_layers,
      total_spoofing_volume=total_spoofing_volume,
      placement_duration=placement_duration,
      cancellation_detected=cancellation_detected,
      cancellation_rate=cancellation_rate,
      execution_metrics=execution_metrics,
      spoofing_execution_ratio=spoofing_execution_ratio,
      price_impact=price_impact,
      mid_price=mid_price
    )

    confidence = confidence_components['total']

    if confidence < self.config.min_confidence:
      return None

    # ===== STEP 6: Build reason string =====
    reason = self._build_reason_string(
      valid_layers,
      total_orders,
      total_spoofing_volume,
      placement_duration,
      cancellation_detected,
      cancellation_rate,
      execution_metrics,
      spoofing_execution_ratio,
      price_impact,
      mid_price
    )

    pattern = LayeringPattern(
      symbol=symbol,
      timestamp=timestamp,
      spoofing_side=spoofing_side,
      execution_side=execution_side if execution_metrics else None,
      confidence=confidence,
      layers=valid_layers,
      total_orders=total_orders,
      total_spoofing_volume=total_spoofing_volume,
      placement_duration=placement_duration,
      cancellation_detected=cancellation_detected,
      cancellation_rate=cancellation_rate,
      execution_metrics=execution_metrics,
      spoofing_execution_ratio=spoofing_execution_ratio,
      price_impact=price_impact,
      volume_score=confidence_components['volume'],
      timing_score=confidence_components['timing'],
      cancellation_score=confidence_components['cancellation'],
      execution_correlation_score=confidence_components['execution_correlation'],
      price_impact_score=confidence_components['price_impact'],
      reason=reason
    )

    # ===== STEP 7: ML Integration (if enabled) =====
    if self.enable_ml_features:
      pattern = self._integrate_ml_features(
        pattern,
        symbol,
        mid_price,
        confidence_components
      )

    return pattern

  def _group_into_layers(
      self,
      placements: List[Tuple[float, int, float]],
      side: str,
      mid_price: float
  ) -> List[OrderLayer]:
    """
    Group orders into layers using price clustering.

    Args:
        placements: [(price, timestamp, volume), ...]
        side: "bid" or "ask"
        mid_price: Current mid price for spread calculation

    Returns:
        List of OrderLayer objects
    """
    if not placements:
      return []

    # Sort by price
    placements_sorted = sorted(placements, key=lambda x: x[0])

    layers = []
    current_layer_prices = [placements_sorted[0][0]]
    current_layer_volumes = [placements_sorted[0][2]]
    current_layer_timestamps = [placements_sorted[0][1]]

    for i in range(1, len(placements_sorted)):
      price, timestamp, volume = placements_sorted[i]

      # Calculate spread relative to first order in layer
      base_price = current_layer_prices[0]
      spread = abs(price - base_price) / mid_price

      if spread <= self.config.max_price_spread_pct:
        # Add to current layer
        current_layer_prices.append(price)
        current_layer_volumes.append(volume)
        current_layer_timestamps.append(timestamp)
      else:
        # Create new layer from current data
        if len(current_layer_prices) >= self.config.min_orders_in_layer:
          layer = self._create_layer(
            side,
            current_layer_prices,
            current_layer_volumes,
            current_layer_timestamps
          )
          if layer:
            layers.append(layer)

        # Start new layer
        current_layer_prices = [price]
        current_layer_volumes = [volume]
        current_layer_timestamps = [timestamp]

    # Add last layer
    if len(current_layer_prices) >= self.config.min_orders_in_layer:
      layer = self._create_layer(
        side,
        current_layer_prices,
        current_layer_volumes,
        current_layer_timestamps
      )
      if layer:
        layers.append(layer)

    return layers

  def _create_layer(
      self,
      side: str,
      prices: List[float],
      volumes: List[float],
      timestamps: List[int]
  ) -> Optional[OrderLayer]:
    """
    Create layer from orders.
    FIXED: Correctly pass side parameter.

    Args:
        side: "bid" or "ask" - correctly determined side
        prices: List of prices
        volumes: List of volumes
        timestamps: List of timestamps

    Returns:
        OrderLayer object or None
    """
    if not prices:
      return None

    # Calculate metrics
    price_spread = float((max(prices) - min(prices)) / np.mean(prices))
    total_volume = float(sum(volumes))
    avg_volume = float(np.mean(volumes))
    volume_std = float(np.std(volumes)) if len(volumes) > 1 else 0.0

    placement_duration = (
      (max(timestamps) - min(timestamps)) / 1000.0
      if len(timestamps) > 1
      else 0.0
    )

    return OrderLayer(
      side=side,  # FIXED: Use actual side parameter
      prices=prices,
      volumes=volumes,
      timestamps=timestamps,
      price_spread=price_spread,
      total_volume=total_volume,
      avg_volume=avg_volume,
      volume_std=volume_std,
      placement_duration=placement_duration
    )

  def _is_valid_layer(self, layer: OrderLayer, mid_price: float) -> bool:
    """
    Validate layer based on professional criteria.
    FIXED: Correct mathematical formula for volume check.

    Args:
        layer: OrderLayer to validate
        mid_price: Current mid price (USDT price for conversion)

    Returns:
        bool: True if layer is valid
    """
    # FIXED: Correct formula - convert BTC volume to USDT
    layer_volume_usdt = layer.total_volume * mid_price
    if layer_volume_usdt < self.config.min_layer_volume_btc * mid_price:
      return False

    # Volume similarity check (Coefficient of Variation)
    if layer.order_count > 1 and layer.avg_volume > 0:
      cv = layer.volume_std / layer.avg_volume
      if cv > self.config.volume_similarity_threshold:
        return False

    return True

  def _analyze_execution_side(
      self,
      symbol: str,
      execution_side: str,
      timestamp: int,
      placement_duration: float
  ) -> Optional[ExecutionMetrics]:
    """
    Analyze actual trade execution on opposite side.
    Uses TradeManager to get real market trades.

    Args:
        symbol: Trading pair
        execution_side: "bid" or "ask" - side where we expect execution
        timestamp: Current timestamp
        placement_duration: Duration of order placement

    Returns:
        ExecutionMetrics if significant execution detected
    """
    if symbol not in self.trade_managers:
      return None

    trade_manager = self.trade_managers[symbol]

    # Get trades in correlation window
    window_seconds = self.config.execution_correlation_window
    current_time_sec = timestamp / 1000.0
    cutoff_time_ms = timestamp - int(window_seconds * 1000)

    # Filter trades on execution side
    relevant_trades = []
    for trade in trade_manager.recent_trades:
      if trade.timestamp < cutoff_time_ms:
        continue

      # execution_side "bid" means we expect aggressive SELLS (hitting bids)
      # execution_side "ask" means we expect aggressive BUYS (lifting asks)
      if execution_side == "bid" and trade.is_sell:
        relevant_trades.append(trade)
      elif execution_side == "ask" and trade.is_buy:
        relevant_trades.append(trade)

    if not relevant_trades:
      return None

    # Calculate execution metrics
    total_volume = sum(t.quantity for t in relevant_trades)
    trade_count = len(relevant_trades)
    avg_trade_size = total_volume / trade_count if trade_count > 0 else 0.0

    # All market trades are aggressive (takers)
    aggressive_ratio = 1.0

    # Temporal correlation: trades happened during/after placement
    if relevant_trades:
      first_trade_time = min(t.timestamp for t in relevant_trades) / 1000.0
      time_since_placement = current_time_sec - first_trade_time

      # Correlation score: closer in time = higher score
      if time_since_placement < 10.0:
        correlation_score = 1.0
      elif time_since_placement < 30.0:
        correlation_score = 0.7
      elif time_since_placement < 60.0:
        correlation_score = 0.4
      else:
        correlation_score = 0.2
    else:
      time_since_placement = 0.0
      correlation_score = 0.0

    return ExecutionMetrics(
      side=execution_side,
      total_volume=total_volume,
      trade_count=trade_count,
      avg_trade_size=avg_trade_size,
      aggressive_ratio=aggressive_ratio,
      time_since_placement=time_since_placement,
      correlation_score=correlation_score
    )

  def _analyze_price_impact(
      self,
      symbol: str,
      timestamp: int,
      spoofing_side: str,
      spoofing_volume: float,
      current_price: float
  ) -> Optional[PriceImpactMetrics]:
    """
    Analyze price impact: expected vs actual.
    Low actual impact compared to expected = suspicious (fake orders).

    Args:
        symbol: Trading pair
        timestamp: Current timestamp
        spoofing_side: Side with potential layering
        spoofing_volume: Total volume in layers
        current_price: Current mid price

    Returns:
        PriceImpactMetrics if analysis possible
    """
    if symbol not in self.price_history:
      return None

    price_hist = self.price_history[symbol]
    if len(price_hist) < 10:
      return None

    # Find price 30 seconds ago
    cutoff_time = timestamp - 30000  # 30 seconds

    initial_price = None
    for ts, price in price_hist:
      if ts >= cutoff_time:
        initial_price = price
        break

    if not initial_price or initial_price <= 0:
      # Use oldest price in history
      initial_price = price_hist[0][1]

    # Calculate actual price change
    price_change = current_price - initial_price
    price_change_bps = (price_change / initial_price) * 10000

    # Calculate expected impact based on volume
    # Rule of thumb: 1 BTC volume ~ 1-2 bps impact for liquid pairs
    expected_impact_bps = spoofing_volume * 1.5  # Conservative estimate

    # Direction check
    if spoofing_side == "bid":
      # Large bid orders should push price UP
      direction_matches = price_change > 0
    else:
      # Large ask orders should push price DOWN
      direction_matches = price_change < 0

    # Impact ratio
    if expected_impact_bps > 0:
      impact_ratio = abs(price_change_bps) / expected_impact_bps
    else:
      impact_ratio = 0.0

    return PriceImpactMetrics(
      initial_price=initial_price,
      current_price=current_price,
      price_change_bps=price_change_bps,
      expected_impact_bps=expected_impact_bps,
      impact_ratio=impact_ratio,
      direction_matches=direction_matches
    )

  def _calculate_professional_confidence(
      self,
      layers: List[OrderLayer],
      total_spoofing_volume: float,
      placement_duration: float,
      cancellation_detected: bool,
      cancellation_rate: float,
      execution_metrics: Optional[ExecutionMetrics],
      spoofing_execution_ratio: Optional[float],
      price_impact: Optional[PriceImpactMetrics],
      mid_price: float
  ) -> Dict[str, float]:
    """
    Professional multi-factor confidence scoring.
    Industry-standard weighted approach.

    Returns:
        Dict with component scores and total confidence
    """
    scores = {
      'volume': 0.0,
      'timing': 0.0,
      'cancellation': 0.0,
      'execution_correlation': 0.0,
      'price_impact': 0.0,
      'total': 0.0
    }

    # ===== 1. Volume Score (weight: 0.20) =====
    # FIXED: Use real mid_price instead of hardcoded 50000
    total_usdt = total_spoofing_volume * mid_price

    if total_usdt > 200000:  # $200k+
      scores['volume'] = 0.20
    elif total_usdt > 100000:  # $100k+
      scores['volume'] = 0.15
    elif total_usdt > 50000:  # $50k+
      scores['volume'] = 0.10
    else:
      scores['volume'] = 0.05

    # Layer count bonus
    if len(layers) >= 5:
      scores['volume'] += 0.05
    elif len(layers) >= 3:
      scores['volume'] += 0.03

    # ===== 2. Timing Score (weight: 0.20) =====
    # Fast placement = more suspicious
    if placement_duration < 5.0:  # Very fast
      scores['timing'] = 0.20
    elif placement_duration < 15.0:
      scores['timing'] = 0.15
    elif placement_duration < 30.0:
      scores['timing'] = 0.10
    else:
      scores['timing'] = 0.05

    # ===== 3. Cancellation Score (weight: 0.25) =====
    if cancellation_detected:
      scores['cancellation'] = 0.15

      # High cancellation rate = very suspicious
      if cancellation_rate >= 0.7:  # 70%+
        scores['cancellation'] += 0.10
      elif cancellation_rate >= 0.5:  # 50%+
        scores['cancellation'] += 0.07
      elif cancellation_rate >= 0.3:  # 30%+
        scores['cancellation'] += 0.03

    # ===== 4. Execution Correlation Score (weight: 0.20) =====
    if execution_metrics and spoofing_execution_ratio:
      # High ratio = suspicious
      if spoofing_execution_ratio >= self.config.high_spoofing_execution_ratio:
        scores['execution_correlation'] = 0.20
      elif spoofing_execution_ratio >= self.config.min_spoofing_execution_ratio:
        scores['execution_correlation'] = 0.15
      else:
        scores['execution_correlation'] = 0.05

      # Temporal correlation bonus
      scores['execution_correlation'] += execution_metrics.correlation_score * 0.05

    # ===== 5. Price Impact Score (weight: 0.15) =====
    if price_impact:
      # Low actual impact vs expected = suspicious
      if price_impact.impact_ratio < self.config.low_actual_impact_multiplier:
        scores['price_impact'] = 0.15
      elif price_impact.impact_ratio < 0.5:
        scores['price_impact'] = 0.10
      elif price_impact.impact_ratio < 0.7:
        scores['price_impact'] = 0.05

      # Wrong direction = very suspicious
      if not price_impact.direction_matches:
        scores['price_impact'] += 0.05

    # ===== Total Confidence =====
    scores['total'] = min(
      scores['volume'] +
      scores['timing'] +
      scores['cancellation'] +
      scores['execution_correlation'] +
      scores['price_impact'],
      1.0
    )

    return scores

  def _build_reason_string(
      self,
      layers: List[OrderLayer],
      total_orders: int,
      total_spoofing_volume: float,
      placement_duration: float,
      cancellation_detected: bool,
      cancellation_rate: float,
      execution_metrics: Optional[ExecutionMetrics],
      spoofing_execution_ratio: Optional[float],
      price_impact: Optional[PriceImpactMetrics],
      mid_price: float
  ) -> str:
    """Build comprehensive reason string for detection."""
    # FIXED: Use real mid_price for USDT calculation
    total_usdt = total_spoofing_volume * mid_price

    reason_parts = [
      f"{len(layers)} layers with {total_orders} orders "
      f"(${total_usdt:,.0f}) placed in {placement_duration:.1f}s"
    ]

    if cancellation_detected:
      reason_parts.append(
        f"cancellation_rate={cancellation_rate:.1%}"
      )

    if execution_metrics and spoofing_execution_ratio:
      reason_parts.append(
        f"spoofing/execution_ratio={spoofing_execution_ratio:.1f}x "
        f"({execution_metrics.trade_count} trades)"
      )

    if price_impact:
      reason_parts.append(
        f"price_impact={price_impact.price_change_bps:.1f}bps "
        f"(expected={price_impact.expected_impact_bps:.1f}bps, "
        f"ratio={price_impact.impact_ratio:.2f})"
      )

    return ", ".join(reason_parts)

  def _integrate_ml_features(
      self,
      pattern: LayeringPattern,
      symbol: str,
      mid_price: float,
      confidence_components: Dict[str, float]
  ) -> LayeringPattern:
    """
    Integrate ML features: Historical matching, data collection, adaptive model.

    Args:
        pattern: Detected layering pattern
        symbol: Trading symbol
        mid_price: Current mid price
        confidence_components: Confidence score components

    Returns:
        Enhanced pattern with ML adjustments
    """
    # ===== 1. Historical Pattern Matching =====
    if self.pattern_database:
      pattern_features = self._extract_pattern_features(pattern, mid_price)

      # Find similar patterns (using sync wrapper)
      match_result = self.pattern_database.find_similar_pattern_sync(
        pattern_features,
        similarity_threshold=0.80
      )

      if match_result:
        historical, similarity = match_result

        if historical:
          # Boost confidence for known patterns
          confidence_boost = 0.15 if historical.blacklist else 0.10
          pattern.confidence = min(pattern.confidence + confidence_boost, 1.0)

          # Update reason
          pattern.reason += (
            f" | KNOWN PATTERN (id={historical.pattern_id[:8]}, "
            f"seen={historical.occurrence_count}x, "
            f"risk={historical.risk_level})"
          )

          logger.info(
            f"🔍 Historical match: {historical.pattern_id[:12]}, "
            f"similarity={similarity:.2f}, "
            f"occurrences={historical.occurrence_count}, "
            f"blacklist={historical.blacklist}"
          )

      # Save pattern to database (using sync wrapper)
      try:
        price_impact_bps = (
          pattern.price_impact.price_change_bps
          if pattern.price_impact else 0.0
        )

        self.pattern_database.save_pattern_sync(
          pattern_features,
          symbol,
          pattern.confidence,
          success_rate=0.0,
          price_impact_bps=price_impact_bps
        )
      except Exception as e:
        logger.error(f"Error saving pattern to database: {e}")

    # ===== 2. ML Data Collection =====
    if self.data_collector:
      try:
        # Prepare data for collection
        pattern_data = {
          'timestamp': pattern.timestamp,
          'symbol': symbol,
          'total_spoofing_volume': pattern.total_spoofing_volume,
          'total_volume_usdt': pattern.total_spoofing_volume * mid_price,
          'placement_duration': pattern.placement_duration,
          'cancellation_rate': pattern.cancellation_rate,
          'spoofing_execution_ratio': pattern.spoofing_execution_ratio,
          'layer_count': len(pattern.layers),
          'total_orders': pattern.total_orders,
          'avg_order_size': (
            pattern.total_spoofing_volume / pattern.total_orders
            if pattern.total_orders > 0 else 0.0
          ),
          'price_spread_bps': (
            np.mean([layer.price_spread for layer in pattern.layers]) * 10000
            if pattern.layers else 0.0
          ),
          'confidence': pattern.confidence,
          'price_change_bps': (
            pattern.price_impact.price_change_bps
            if pattern.price_impact else None
          ),
          'expected_impact_bps': (
            pattern.price_impact.expected_impact_bps
            if pattern.price_impact else None
          ),
          'impact_ratio': (
            pattern.price_impact.impact_ratio
            if pattern.price_impact else None
          ),
          'execution_volume': (
            pattern.execution_metrics.total_volume
            if pattern.execution_metrics else None
          ),
          'execution_trade_count': (
            pattern.execution_metrics.trade_count
            if pattern.execution_metrics else None
          ),
          'aggressive_ratio': (
            pattern.execution_metrics.aggressive_ratio
            if pattern.execution_metrics else None
          )
        }

        # Get market context (simplified - can be enhanced)
        market_context = {
          'market_regime': 'unknown',  # Can integrate with MarketRegimeDetector
          'volatility_24h': 0.0,       # Can get from market stats
          'volume_24h': 0.0,
          'liquidity_score': 0.0,
          'spread_bps': 0.0
        }

        # Collect data (unlabeled - will be labeled later)
        data_id = self.data_collector.collect(
          pattern_data,
          market_context,
          confidence_components,
          label=None,  # Unlabeled initially
          label_source="automatic",
          label_confidence=0.0
        )

        logger.debug(f"📊 Training data collected: {data_id}")

        # ===== Schedule automatic price validation for labeling =====
        if data_id:
          self._schedule_price_validation(
            data_id=data_id,
            pattern=pattern,
            symbol=symbol,
            mid_price=mid_price
          )

      except Exception as e:
        logger.error(f"Error collecting training data: {e}")

    # ===== 3. Adaptive ML Model Prediction =====
    if self.adaptive_model and self.adaptive_model.enabled:
      try:
        # Prepare features for ML prediction
        features = self._extract_ml_features(pattern, mid_price)

        # Get ML prediction
        ml_is_true, ml_confidence = self.adaptive_model.predict(features)

        # Adjust confidence based on ML prediction
        if ml_is_true:
          # ML confirms - boost confidence slightly
          pattern.confidence = (pattern.confidence + ml_confidence) / 2
        else:
          # ML disagrees - reduce confidence
          pattern.confidence = pattern.confidence * 0.7

        pattern.confidence = min(max(pattern.confidence, 0.0), 1.0)

        logger.debug(
          f"🤖 ML prediction: is_true={ml_is_true}, "
          f"ml_conf={ml_confidence:.2f}, "
          f"adjusted_conf={pattern.confidence:.2f}"
        )

      except Exception as e:
        logger.error(f"Error in ML prediction: {e}")

    return pattern

  def _extract_pattern_features(
      self,
      pattern: LayeringPattern,
      mid_price: float
  ) -> Dict:
    """Extract features for historical pattern matching."""
    avg_spread = (
      np.mean([layer.price_spread for layer in pattern.layers])
      if pattern.layers else 0.0
    )

    return {
      'avg_layer_count': len(pattern.layers),
      'cancellation_rate': pattern.cancellation_rate,
      'total_volume': pattern.total_spoofing_volume,
      'placement_duration': pattern.placement_duration,
      'avg_spread_pct': avg_spread,
      'total_orders': pattern.total_orders,
      'spoofing_execution_ratio': pattern.spoofing_execution_ratio,
      'avg_lifetime_seconds': 0.0  # Can be calculated from tracker
    }

  def _extract_ml_features(
      self,
      pattern: LayeringPattern,
      mid_price: float
  ) -> Dict:
    """Extract features for ML model prediction."""
    avg_order_size = (
      pattern.total_spoofing_volume / pattern.total_orders
      if pattern.total_orders > 0 else 0.0
    )

    avg_spread = (
      np.mean([layer.price_spread for layer in pattern.layers])
      if pattern.layers else 0.0
    )

    return {
      # Pattern features
      'total_volume_btc': pattern.total_spoofing_volume,
      'total_volume_usdt': pattern.total_spoofing_volume * mid_price,
      'placement_duration': pattern.placement_duration,
      'cancellation_rate': pattern.cancellation_rate,
      'spoofing_execution_ratio': pattern.spoofing_execution_ratio,
      'layer_count': len(pattern.layers),
      'total_orders': pattern.total_orders,
      'avg_order_size': avg_order_size,
      'price_spread_bps': avg_spread * 10000,

      # Market context (simplified)
      'volatility_24h': 0.0,
      'volume_24h': 0.0,
      'liquidity_score': 0.0,
      'spread_bps': 0.0,
      'hour_utc': datetime.utcfromtimestamp(pattern.timestamp / 1000).hour,
      'day_of_week': datetime.utcfromtimestamp(pattern.timestamp / 1000).weekday(),

      # Price impact
      'price_change_bps': (
        pattern.price_impact.price_change_bps
        if pattern.price_impact else 0.0
      ),
      'expected_impact_bps': (
        pattern.price_impact.expected_impact_bps
        if pattern.price_impact else 0.0
      ),
      'impact_ratio': (
        pattern.price_impact.impact_ratio
        if pattern.price_impact else 0.0
      ),

      # Component scores
      'volume_score': pattern.volume_score,
      'timing_score': pattern.timing_score,
      'cancellation_score': pattern.cancellation_score,
      'execution_correlation_score': pattern.execution_correlation_score,
      'price_impact_score': pattern.price_impact_score,

      # Detector confidence (for reference)
      'detector_confidence': pattern.confidence
    }

  def _cleanup_old_data(self, symbol: str, timestamp: int):
    """Cleanup old tracking data."""
    cutoff_time = timestamp - (self.config.history_window_seconds * 1000)

    # Cleanup order trackers
    for side in ["bid", "ask"]:
      tracker = self.trackers[symbol][side]
      if tracker:
        tracker.cleanup_old_history(cutoff_time)

    # Cleanup price history
    if symbol in self.price_history:
      cutoff_time_ms = cutoff_time
      self.price_history[symbol] = deque(
        [(ts, p) for ts, p in self.price_history[symbol] if ts >= cutoff_time_ms],
        maxlen=1000
      )

  def get_recent_patterns(
      self,
      symbol: str,
      time_window_seconds: int = 60
  ) -> List[LayeringPattern]:
    """Get recent layering patterns."""
    current_time = int(time.time() * 1000)
    cutoff_time = current_time - (time_window_seconds * 1000)

    patterns = self.detected_patterns.get(symbol, [])

    return [
      p for p in patterns
      if p.timestamp >= cutoff_time
    ]

  def is_layering_active(
      self,
      symbol: str,
      side: Optional[str] = None,
      time_window_seconds: int = 60
  ) -> bool:
    """Check if layering is currently active."""
    patterns = self.get_recent_patterns(symbol, time_window_seconds)

    if side:
      patterns = [p for p in patterns if p.spoofing_side == side]

    return len(patterns) > 0

  def get_statistics(self) -> Dict:
    """Get detector statistics."""
    total_patterns = sum(
      len(patterns)
      for patterns in self.detected_patterns.values()
    )

    stats = {
      'total_checks': self.total_checks,
      'event_driven_checks': self.event_driven_checks,
      'patterns_detected': self.patterns_detected,
      'total_patterns': total_patterns,
      'symbols_monitored': len(self.trackers),
      'detection_rate': (
        self.patterns_detected / self.total_checks
        if self.total_checks > 0
        else 0.0
      ),
      'trade_integration_enabled': len(self.trade_managers) > 0,
      'ml_features_enabled': self.enable_ml_features
    }

    # Add ML component statistics
    if self.enable_ml_features:
      if self.pattern_database:
        stats['pattern_database'] = self.pattern_database.get_statistics_sync()

      if self.data_collector:
        stats['data_collector'] = self.data_collector.get_statistics()

      if self.adaptive_model:
        stats['adaptive_model'] = self.adaptive_model.get_info()

    return stats

  def _schedule_price_validation(
      self,
      data_id: str,
      pattern: LayeringPattern,
      symbol: str,
      mid_price: float
  ):
    """
    Schedule automatic price validation for ML auto-labeling.

    Args:
        data_id: ML data collection ID
        pattern: Detected layering pattern
        symbol: Trading symbol
        mid_price: Current mid price
    """
    if not self.data_collector:
      return

    # Determine expected price movement direction based on layering side
    # Layering on bid (fake demand) → expect price to go DOWN after cancellation
    # Layering on ask (fake supply) → expect price to go UP after cancellation
    expected_direction = "down" if pattern.spoofing_side == "bid" else "up"

    # Create pending validation record
    pending = PendingValidationPattern(
      data_id=data_id,
      pattern=pattern,
      symbol=symbol,
      entry_price=mid_price,
      entry_timestamp=pattern.timestamp,
      expected_direction=expected_direction,
      validation_window_seconds=30  # Validate after 30 seconds
    )

    self.pending_validations.append(pending)

    # Schedule async validation task
    try:
      task = asyncio.create_task(
        self._validate_pattern_price_action(pending)
      )
      self.validation_tasks.append(task)

      logger.debug(
        f"⏰ Scheduled price validation: {data_id}, "
        f"expected_direction={expected_direction}, "
        f"window=30s"
      )

    except RuntimeError:
      # No event loop running (sync context)
      logger.warning(
        f"⚠️  Cannot schedule async validation (no event loop): {data_id}"
      )

  async def _validate_pattern_price_action(
      self,
      pending: PendingValidationPattern
  ):
    """
    Validate pattern via price action and update ML label.

    Waits validation_window_seconds then checks if price moved as expected.

    Args:
        pending: Pending validation pattern
    """
    try:
      # Wait for validation window
      await asyncio.sleep(pending.validation_window_seconds)

      # Get current price
      symbol = pending.symbol
      if symbol not in self.price_history or not self.price_history[symbol]:
        logger.warning(
          f"⚠️  No price history for validation: {pending.data_id}"
        )
        return

      # Get most recent price
      latest_timestamp, latest_price = self.price_history[symbol][-1]

      # Calculate price change
      price_change_bps = (
        (latest_price - pending.entry_price) / pending.entry_price * 10000
      )

      # Determine if price moved as expected
      # Threshold: at least 3 bps movement in expected direction
      threshold_bps = 3.0

      if pending.expected_direction == "down":
        price_moved_as_expected = price_change_bps < -threshold_bps
      else:  # "up"
        price_moved_as_expected = price_change_bps > threshold_bps

      # Update ML data label
      label_confidence = min(abs(price_change_bps) / 10.0, 1.0)  # Scale confidence

      self.data_collector.update_label(
        data_id=pending.data_id,
        label=price_moved_as_expected,
        label_source="price_action_30s",
        label_confidence=label_confidence,
        notes=f"Price change: {price_change_bps:.1f} bps, "
             f"expected: {pending.expected_direction}"
      )

      # Update pending record
      pending.validation_completed = True
      pending.price_moved_as_expected = price_moved_as_expected
      pending.actual_price_change_bps = price_change_bps

      logger.info(
        f"✅ Price validation completed: {pending.data_id}, "
        f"label={price_moved_as_expected}, "
        f"price_change={price_change_bps:.1f}bps, "
        f"confidence={label_confidence:.2f}"
      )

    except Exception as e:
      logger.error(f"Error in price validation: {e}")

    finally:
      # Remove from pending list
      if pending in self.pending_validations:
        self.pending_validations.remove(pending)


# Example usage and testing
if __name__ == "__main__":
  from backend.models.orderbook import OrderBookSnapshot, OrderBookLevel

  config = LayeringConfig(
    min_orders_in_layer=3,
    max_price_spread_pct=0.005,
    min_layer_volume_btc=0.5
  )

  detector = LayeringDetector(config)

  # Simulate layering: multiple bid orders close together
  base_time = int(time.time() * 1000)

  # Snapshot 1: Start of layering
  snapshot1 = OrderBookSnapshot(
    symbol="BTCUSDT",
    bids=[
      (50000.0, 2.0),
      (49995.0, 2.1),
      (49990.0, 1.9),
    ],
    asks=[(50100.0, 1.0)],
    timestamp=base_time
  )

  detector.update(snapshot1)

  # Snapshot 2: More orders added
  snapshot2 = OrderBookSnapshot(
    symbol="BTCUSDT",
    bids=[
      (50000.0, 2.0),
      (49995.0, 2.1),
      (49990.0, 1.9),
      (49985.0, 2.0),  # New
      (49980.0, 1.8),  # New
    ],
    asks=[(50100.0, 1.0)],
    timestamp=base_time + 5000
  )

  detector.update(snapshot2)

  # Check for patterns
  patterns = detector.get_recent_patterns("BTCUSDT")
  print(f"Detected {len(patterns)} layering patterns")

  for pattern in patterns:
    print(f"\n  Side: {pattern.spoofing_side}")
    print(f"  Layers: {len(pattern.layers)}")
    print(f"  Confidence: {pattern.confidence:.2%}")
    print(f"  Reason: {pattern.reason}")

  # Statistics
  stats = detector.get_statistics()
  print(f"\nStatistics: {stats}")
