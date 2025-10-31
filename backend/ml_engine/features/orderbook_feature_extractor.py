"""
OrderBook Feature Extractor для извлечения 50+ признаков из стакана.

Интегрируется с существующими:
- backend/models/orderbook.py (OrderBookSnapshot, OrderBookMetrics)
- backend/strategy/analyzer.py (OrderBookAnalyzer)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from numba import jit

from core.logger import get_logger
from core.periodic_logger import periodic_logger
from models.orderbook import OrderBookSnapshot, OrderBookMetrics
from strategy.analyzer import OrderBookAnalyzer

logger = get_logger(__name__)


@dataclass
class OrderBookFeatures:
  """Контейнер для признаков стакана"""

  symbol: str
  timestamp: int

  # Базовые микроструктурные (15)
  bid_ask_spread_abs: float
  bid_ask_spread_rel: float
  mid_price: float
  micro_price: float
  vwap_bid_5: float
  vwap_ask_5: float
  vwap_bid_10: float
  vwap_ask_10: float
  depth_bid_5: float
  depth_ask_5: float
  depth_bid_10: float
  depth_ask_10: float
  total_bid_volume: float
  total_ask_volume: float
  book_depth_ratio: float

  # Дисбаланс и давление (10)
  imbalance_5: float
  imbalance_10: float
  imbalance_total: float
  price_pressure: float
  volume_delta_5: float
  order_flow_imbalance: float
  bid_intensity: float
  ask_intensity: float
  buy_sell_ratio: float
  smart_money_index: float

  # Кластеры и уровни (10)
  largest_bid_cluster_price: float
  largest_bid_cluster_volume: float
  largest_ask_cluster_price: float
  largest_ask_cluster_volume: float
  num_bid_clusters: int
  num_ask_clusters: int
  support_level_1: float
  resistance_level_1: float
  distance_to_support: float
  distance_to_resistance: float

  # Ликвидность (8)
  liquidity_bid_5: float
  liquidity_ask_5: float
  liquidity_asymmetry: float
  effective_spread: float
  kyle_lambda: float
  amihud_illiquidity: float
  roll_spread: float
  depth_imbalance_ratio: float

  # Временные (7)
  level_ttl_avg: float
  level_ttl_std: float
  orderbook_volatility: float
  update_frequency: float
  quote_intensity: float
  trade_arrival_rate: float
  spread_volatility: float

  def to_array(self) -> np.ndarray:
    """Преобразование в numpy array для ML моделей"""
    return np.array([
      # Базовые (15)
      self.bid_ask_spread_abs,
      self.bid_ask_spread_rel,
      self.mid_price,
      self.micro_price,
      self.vwap_bid_5,
      self.vwap_ask_5,
      self.vwap_bid_10,
      self.vwap_ask_10,
      self.depth_bid_5,
      self.depth_ask_5,
      self.depth_bid_10,
      self.depth_ask_10,
      self.total_bid_volume,
      self.total_ask_volume,
      self.book_depth_ratio,
      # Дисбаланс (10)
      self.imbalance_5,
      self.imbalance_10,
      self.imbalance_total,
      self.price_pressure,
      self.volume_delta_5,
      self.order_flow_imbalance,
      self.bid_intensity,
      self.ask_intensity,
      self.buy_sell_ratio,
      self.smart_money_index,
      # Кластеры (10)
      self.largest_bid_cluster_price,
      self.largest_bid_cluster_volume,
      self.largest_ask_cluster_price,
      self.largest_ask_cluster_volume,
      float(self.num_bid_clusters),
      float(self.num_ask_clusters),
      self.support_level_1,
      self.resistance_level_1,
      self.distance_to_support,
      self.distance_to_resistance,
      # Ликвидность (8)
      self.liquidity_bid_5,
      self.liquidity_ask_5,
      self.liquidity_asymmetry,
      self.effective_spread,
      self.kyle_lambda,
      self.amihud_illiquidity,
      self.roll_spread,
      self.depth_imbalance_ratio,
      # Временные (7)
      self.level_ttl_avg,
      self.level_ttl_std,
      self.orderbook_volatility,
      self.update_frequency,
      self.quote_intensity,
      self.trade_arrival_rate,
      self.spread_volatility
    ], dtype=np.float32)

  def to_dict(self) -> Dict[str, float]:
    """Преобразование в словарь"""
    return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class OrderBookFeatureExtractor:
  """
  Извлекает 50+ признаков из стакана ордеров.

  Использует существующий OrderBookAnalyzer для базовых метрик
  и добавляет продвинутые признаки.
  """

  def __init__(self, symbol: str):
    """
    Args:
        symbol: Торговая пара
    """
    self.symbol = symbol
    self.analyzer = OrderBookAnalyzer(symbol)

    # История для временных признаков
    self.snapshot_history: List[OrderBookSnapshot] = []
    self.max_history_size = 100  # Последние 100 снимков

    # Level TTL tracking для spoofing detection
    # Отслеживаем время жизни каждого ценового уровня
    self.level_tracker: Dict[str, Dict[float, Dict]] = {
      "bid": {},  # price -> {first_seen, last_seen, max_volume}
      "ask": {}
    }
    self.level_ttl_history: List[float] = []  # История TTL (секунды)
    self.max_ttl_history = 200  # Последние 200 TTL значений

    logger.info(f"OrderBookFeatureExtractor инициализирован для {symbol}")

  def extract(
      self,
      snapshot: OrderBookSnapshot,
      prev_snapshot: Optional[OrderBookSnapshot] = None
  ) -> OrderBookFeatures:
    """
    Извлекает все признаки из снимка стакана.

    Args:
        snapshot: Текущий снимок стакана
        prev_snapshot: Предыдущий снимок для временных признаков

    Returns:
        OrderBookFeatures: Извлеченные признаки
    """
    logger.debug(f"{self.symbol} | Извлечение признаков из стакана")

    try:
      # Добавляем в историю
      self.snapshot_history.append(snapshot)
      if len(self.snapshot_history) > self.max_history_size:
        self.snapshot_history.pop(0)

      # Обновляем отслеживание уровней для TTL
      self._update_level_tracking(snapshot)

      # 1. Базовые микроструктурные признаки
      basic_features = self._extract_basic_features(snapshot)
      logger.debug(f"{self.symbol} | Базовые признаки извлечены")

      # 2. Признаки дисбаланса и давления
      imbalance_features = self._extract_imbalance_features(snapshot)
      logger.debug(f"{self.symbol} | Признаки дисбаланса извлечены")

      # 3. Кластерные признаки
      cluster_features = self._extract_cluster_features(snapshot)
      logger.debug(f"{self.symbol} | Кластерные признаки извлечены")

      # 4. Признаки ликвидности
      liquidity_features = self._extract_liquidity_features(snapshot)
      logger.debug(f"{self.symbol} | Признаки ликвидности извлечены")

      # 5. Временные признаки
      temporal_features = self._extract_temporal_features(
        snapshot,
        prev_snapshot
      )
      logger.debug(f"{self.symbol} | Временные признаки извлечены")

      # Объединяем все признаки
      features = OrderBookFeatures(
        symbol=self.symbol,
        timestamp=snapshot.timestamp,
        **basic_features,
        **imbalance_features,
        **cluster_features,
        **liquidity_features,
        **temporal_features
      )

      key = f"orderbook_features_{self.symbol}"
      should_log, count = periodic_logger.should_log(key, every_n=500, first_n=1)

      if should_log:
        logger.info(
          f"{self.symbol} | Извлечено 50 признаков из стакана "
          f"(#{count}), mid_price={features.mid_price:.2f}, imbalance={features.imbalance_5:.3f}"
        )

      # logger.info(
      #   f"{self.symbol} | Извлечено 50 признаков из стакана, "
      #   f"mid_price={features.mid_price:.2f}, "
      #   f"imbalance={features.imbalance_5:.3f}"
      # )

      return features

    except Exception as e:
      logger.error(f"{self.symbol} | Ошибка извлечения признаков: {e}")
      raise

  def _extract_basic_features(
      self,
      snapshot: OrderBookSnapshot
  ) -> Dict[str, float]:
    """Извлечение базовых микроструктурных признаков (15)"""

    best_bid = snapshot.best_bid or 0.0
    best_ask = snapshot.best_ask or 0.0
    mid_price = snapshot.mid_price or 0.0

    # Spread
    spread_abs = snapshot.spread or 0.0
    spread_rel = (spread_abs / mid_price * 100) if mid_price > 0 else 0.0

    # Micro price (weighted mid)
    bid_vol = sum(vol for _, vol in snapshot.bids[:1])
    ask_vol = sum(vol for _, vol in snapshot.asks[:1])
    total_vol = bid_vol + ask_vol

    if total_vol > 0:
      micro_price = (best_bid * ask_vol + best_ask * bid_vol) / total_vol
    else:
      micro_price = mid_price

    # VWAP и Depth для разных уровней
    vwap_bid_5, depth_bid_5 = self._calculate_vwap_depth(snapshot.bids, 5)
    vwap_ask_5, depth_ask_5 = self._calculate_vwap_depth(snapshot.asks, 5)
    vwap_bid_10, depth_bid_10 = self._calculate_vwap_depth(snapshot.bids, 10)
    vwap_ask_10, depth_ask_10 = self._calculate_vwap_depth(snapshot.asks, 10)

    # Общие объемы
    total_bid_volume = sum(vol for _, vol in snapshot.bids)
    total_ask_volume = sum(vol for _, vol in snapshot.asks)

    book_depth_ratio = (
      total_bid_volume / total_ask_volume
      if total_ask_volume > 0 else 1.0
    )

    return {
      "bid_ask_spread_abs": spread_abs,
      "bid_ask_spread_rel": spread_rel,
      "mid_price": mid_price,
      "micro_price": micro_price,
      "vwap_bid_5": vwap_bid_5,
      "vwap_ask_5": vwap_ask_5,
      "vwap_bid_10": vwap_bid_10,
      "vwap_ask_10": vwap_ask_10,
      "depth_bid_5": depth_bid_5,
      "depth_ask_5": depth_ask_5,
      "depth_bid_10": depth_bid_10,
      "depth_ask_10": depth_ask_10,
      "total_bid_volume": total_bid_volume,
      "total_ask_volume": total_ask_volume,
      "book_depth_ratio": book_depth_ratio
    }

  def _extract_imbalance_features(
      self,
      snapshot: OrderBookSnapshot
  ) -> Dict[str, float]:
    """Извлечение признаков дисбаланса и давления (10)"""

    # Imbalance для разных глубин
    imbalance_5 = self._calculate_imbalance(snapshot.bids, snapshot.asks, 5)
    imbalance_10 = self._calculate_imbalance(snapshot.bids, snapshot.asks, 10)

    total_bid = sum(vol for _, vol in snapshot.bids)
    total_ask = sum(vol for _, vol in snapshot.asks)
    imbalance_total = (
      (total_bid - total_ask) / (total_bid + total_ask)
      if (total_bid + total_ask) > 0 else 0.0
    )

    # Price pressure
    best_bid = snapshot.best_bid or 0.0
    best_ask = snapshot.best_ask or 0.0
    bid_vol_5 = sum(vol for _, vol in snapshot.bids[:5])
    ask_vol_5 = sum(vol for _, vol in snapshot.asks[:5])

    if (bid_vol_5 + ask_vol_5) > 0:
      price_pressure = (
          (bid_vol_5 * best_ask - ask_vol_5 * best_bid) /
          (bid_vol_5 + ask_vol_5)
      )
    else:
      price_pressure = 0.0

    # Volume delta - изменение объема на топ-5 уровнях
    volume_delta_5 = self._calculate_volume_delta(snapshot, depth=5)

    # Order flow imbalance - взвешенное изменение объемов
    # Учитывает близость к mid price (ближе = более агрессивно)
    order_flow_imbalance = self._calculate_order_flow_imbalance(snapshot, depth=10)

    # Intensities (количество ордеров / средний размер)
    bid_intensity = self._calculate_intensity(snapshot.bids[:10])
    ask_intensity = self._calculate_intensity(snapshot.asks[:10])

    # Buy/Sell ratio (из imbalance)
    buy_sell_ratio = (1 + imbalance_total) / max(1 - imbalance_total, 0.01)

    # Smart money index (крупные ордера vs мелкие)
    smart_money_index = self._calculate_smart_money_index(snapshot)

    return {
      "imbalance_5": imbalance_5,
      "imbalance_10": imbalance_10,
      "imbalance_total": imbalance_total,
      "price_pressure": price_pressure,
      "volume_delta_5": volume_delta_5,
      "order_flow_imbalance": order_flow_imbalance,
      "bid_intensity": bid_intensity,
      "ask_intensity": ask_intensity,
      "buy_sell_ratio": buy_sell_ratio,
      "smart_money_index": smart_money_index
    }

  def _extract_cluster_features(
      self,
      snapshot: OrderBookSnapshot
  ) -> Dict[str, float]:
    """Извлечение кластерных признаков (10)"""

    # Находим кластеры
    bid_clusters = self._find_clusters(snapshot.bids)
    ask_clusters = self._find_clusters(snapshot.asks)

    # Крупнейшие кластеры
    if bid_clusters:
      largest_bid = max(bid_clusters, key=lambda c: c[1])
      largest_bid_price, largest_bid_volume = largest_bid
    else:
      largest_bid_price, largest_bid_volume = 0.0, 0.0

    if ask_clusters:
      largest_ask = max(ask_clusters, key=lambda c: c[1])
      largest_ask_price, largest_ask_volume = largest_ask
    else:
      largest_ask_price, largest_ask_volume = 0.0, 0.0

    # Уровни поддержки/сопротивления
    support_level_1 = largest_bid_price if largest_bid_price > 0 else (snapshot.best_bid or 0.0)
    resistance_level_1 = largest_ask_price if largest_ask_price > 0 else (snapshot.best_ask or 0.0)

    mid_price = snapshot.mid_price or 0.0

    # Расстояния до уровней
    distance_to_support = (
      (mid_price - support_level_1) / mid_price * 100
      if mid_price > 0 else 0.0
    )
    distance_to_resistance = (
      (resistance_level_1 - mid_price) / mid_price * 100
      if mid_price > 0 else 0.0
    )

    return {
      "largest_bid_cluster_price": largest_bid_price,
      "largest_bid_cluster_volume": largest_bid_volume,
      "largest_ask_cluster_price": largest_ask_price,
      "largest_ask_cluster_volume": largest_ask_volume,
      "num_bid_clusters": len(bid_clusters),
      "num_ask_clusters": len(ask_clusters),
      "support_level_1": support_level_1,
      "resistance_level_1": resistance_level_1,
      "distance_to_support": distance_to_support,
      "distance_to_resistance": distance_to_resistance
    }

  def _extract_liquidity_features(
      self,
      snapshot: OrderBookSnapshot
  ) -> Dict[str, float]:
    """Извлечение признаков ликвидности (8)"""

    # Liquidity (объем необходимый для движения на N уровней)
    liquidity_bid_5 = sum(vol for _, vol in snapshot.bids[:5])
    liquidity_ask_5 = sum(vol for _, vol in snapshot.asks[:5])

    liquidity_asymmetry = (
      liquidity_bid_5 / liquidity_ask_5
      if liquidity_ask_5 > 0 else 1.0
    )

    # Effective spread (взвешенный по объему)
    spread = snapshot.spread or 0.0
    mid_price = snapshot.mid_price or 1.0
    effective_spread = spread / mid_price if mid_price > 0 else 0.0

    # Kyle's Lambda - полная формула (price impact coefficient)
    # λ = ΔPrice / ΔVolume
    kyle_lambda = self._calculate_kyle_lambda(snapshot)

    # Amihud Illiquidity - полная формула
    # ILLIQ = mean(|Return| / Volume)
    amihud_illiquidity = self._calculate_amihud_illiquidity(snapshot)

    # Roll's Spread Estimator - полная формула
    # Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
    roll_spread = self._calculate_roll_spread(snapshot)

    # Depth imbalance ratio
    depth_imbalance_ratio = (
      (liquidity_bid_5 - liquidity_ask_5) / (liquidity_bid_5 + liquidity_ask_5)
      if (liquidity_bid_5 + liquidity_ask_5) > 0 else 0.0
    )

    return {
      "liquidity_bid_5": liquidity_bid_5,
      "liquidity_ask_5": liquidity_ask_5,
      "liquidity_asymmetry": liquidity_asymmetry,
      "effective_spread": effective_spread,
      "kyle_lambda": kyle_lambda,
      "amihud_illiquidity": amihud_illiquidity,
      "roll_spread": roll_spread,
      "depth_imbalance_ratio": depth_imbalance_ratio
    }

  def _extract_temporal_features(
      self,
      snapshot: OrderBookSnapshot,
      prev_snapshot: Optional[OrderBookSnapshot]
  ) -> Dict[str, float]:
    """Извлечение временных признаков (7)"""

    # Для временных признаков нужна история

    # Level TTL (среднее и std) из истории отслеживания уровней
    level_ttl_avg, level_ttl_std = self._calculate_level_ttl_stats()

    # Orderbook volatility
    orderbook_volatility = self._calculate_orderbook_volatility()

    # Update frequency
    update_frequency = self._calculate_update_frequency()

    # Quote intensity (updates per second)
    quote_intensity = update_frequency

    # Trade arrival rate - оценка на основе активности стакана
    # TODO: Заменить на реальные market trades когда будет WebSocket stream
    trade_arrival_rate = self._estimate_trade_arrival_rate(snapshot)

    # Spread volatility
    spread_volatility = self._calculate_spread_volatility()

    return {
      "level_ttl_avg": level_ttl_avg,
      "level_ttl_std": level_ttl_std,
      "orderbook_volatility": orderbook_volatility,
      "update_frequency": update_frequency,
      "quote_intensity": quote_intensity,
      "trade_arrival_rate": trade_arrival_rate,
      "spread_volatility": spread_volatility
    }

  # ===== Helper Methods =====

  @staticmethod
  def _calculate_vwap_depth(
      levels: List[Tuple[float, float]],
      n: int
  ) -> Tuple[float, float]:
    """Вычисляет VWAP и глубину для N уровней"""
    if not levels:
      return 0.0, 0.0

    levels_n = levels[:n]
    total_volume = sum(vol for _, vol in levels_n)

    if total_volume == 0:
      return 0.0, 0.0

    vwap = sum(price * vol for price, vol in levels_n) / total_volume

    return vwap, total_volume

  @staticmethod
  def _calculate_imbalance(
      bids: List[Tuple[float, float]],
      asks: List[Tuple[float, float]],
      n: int
  ) -> float:
    """Вычисляет imbalance для N уровней"""
    bid_volume = sum(vol for _, vol in bids[:n])
    ask_volume = sum(vol for _, vol in asks[:n])

    total = bid_volume + ask_volume
    if total == 0:
      return 0.5  # Нейтральный баланс

    return (bid_volume - ask_volume) / total

  @staticmethod
  def _calculate_intensity(levels: List[Tuple[float, float]]) -> float:
    """Вычисляет intensity (количество ордеров / средний размер)"""
    if not levels:
      return 0.0

    total_volume = sum(vol for _, vol in levels)
    num_orders = len(levels)

    if num_orders == 0:
      return 0.0

    avg_order_size = total_volume / num_orders

    # Intensity = num_orders / avg_size (нормализовано)
    return num_orders / max(avg_order_size, 0.001)

  def _calculate_smart_money_index(
      self,
      snapshot: OrderBookSnapshot
  ) -> float:
    """
    Индекс институциональной активности.
    Крупные ордера (>90th percentile) vs мелкие.
    """
    all_volumes = [vol for _, vol in snapshot.bids + snapshot.asks]

    if not all_volumes:
      return 0.5

    threshold = np.percentile(all_volumes, 90)

    large_volume = sum(vol for vol in all_volumes if vol >= threshold)
    small_volume = sum(vol for vol in all_volumes if vol < threshold)

    total = large_volume + small_volume
    if total == 0:
      return 0.5

    return large_volume / total

  def _find_clusters(
      self,
      levels: List[Tuple[float, float]],
      price_proximity_pct: float = 0.1
  ) -> List[Tuple[float, float]]:
    """
    Находит кластеры близко расположенных уровней.

    Returns:
        List[(price, total_volume)]
    """
    if not levels:
      return []

    clusters = []
    current_cluster_price = levels[0][0]
    current_cluster_volume = 0.0

    for price, volume in levels:
      # Проверяем близость к текущему кластеру
      price_diff_pct = abs(price - current_cluster_price) / current_cluster_price * 100

      if price_diff_pct <= price_proximity_pct:
        # Добавляем к текущему кластеру
        current_cluster_volume += volume
      else:
        # Сохраняем предыдущий кластер
        if current_cluster_volume > 0:
          clusters.append((current_cluster_price, current_cluster_volume))

        # Начинаем новый кластер
        current_cluster_price = price
        current_cluster_volume = volume

    # Добавляем последний кластер
    if current_cluster_volume > 0:
      clusters.append((current_cluster_price, current_cluster_volume))

    return clusters

  def _calculate_orderbook_volatility(self) -> float:
    """Вычисляет волатильность изменений стакана"""
    if len(self.snapshot_history) < 2:
      return 0.0

    # Volatility = std изменений mid_price
    mid_prices = [
      s.mid_price for s in self.snapshot_history[-20:]
      if s.mid_price is not None
    ]

    if len(mid_prices) < 2:
      return 0.0

    returns = np.diff(mid_prices) / mid_prices[:-1]

    return float(np.std(returns))

  def _calculate_update_frequency(self) -> float:
    """Вычисляет частоту обновлений стакана (updates/sec)"""
    if len(self.snapshot_history) < 2:
      return 0.0

    # Берем последние 10 снимков
    recent = self.snapshot_history[-10:]

    if len(recent) < 2:
      return 0.0

    time_diff_ms = recent[-1].timestamp - recent[0].timestamp

    if time_diff_ms == 0:
      return 0.0

    time_diff_sec = time_diff_ms / 1000.0
    num_updates = len(recent) - 1

    return num_updates / time_diff_sec

  def _calculate_spread_volatility(self) -> float:
    """Вычисляет волатильность спреда"""
    if len(self.snapshot_history) < 2:
      return 0.0

    spreads = [
      s.spread for s in self.snapshot_history[-20:]
      if s.spread is not None
    ]

    if len(spreads) < 2:
      return 0.0

    return float(np.std(spreads))

  def _calculate_volume_delta(
      self,
      snapshot: OrderBookSnapshot,
      depth: int = 5
  ) -> float:
    """
    Вычисляет изменение объема на топ-N уровнях.

    Volume Delta показывает агрессивность покупателей/продавцов:
    - Положительное значение = агрессивная покупка (bid объем растет)
    - Отрицательное значение = агрессивная продажа (ask объем растет)

    Args:
        snapshot: Текущий снимок стакана
        depth: Глубина анализа (топ-N уровней)

    Returns:
        float: Изменение net volume (bid - ask) между снимками
    """
    # Нужен предыдущий снимок для сравнения
    if len(self.snapshot_history) < 2:
      return 0.0

    prev_snapshot = self.snapshot_history[-2]

    # Текущие объемы на топ-N уровнях
    current_bid_vol = sum(vol for _, vol in snapshot.bids[:depth])
    current_ask_vol = sum(vol for _, vol in snapshot.asks[:depth])
    current_net_volume = current_bid_vol - current_ask_vol

    # Предыдущие объемы на топ-N уровнях
    prev_bid_vol = sum(vol for _, vol in prev_snapshot.bids[:depth])
    prev_ask_vol = sum(vol for _, vol in prev_snapshot.asks[:depth])
    prev_net_volume = prev_bid_vol - prev_ask_vol

    # Дельта (изменение net volume)
    volume_delta = current_net_volume - prev_net_volume

    return volume_delta

  def _update_level_tracking(self, snapshot: OrderBookSnapshot):
    """
    Обновляет отслеживание времени жизни уровней в стакане.

    Для каждого уровня:
    - Если новый -> записать first_seen
    - Если существует -> обновить last_seen
    - Если исчез -> вычислить TTL и сохранить в историю

    Это критично для spoofing detection!

    Args:
        snapshot: Текущий снимок стакана
    """
    timestamp = snapshot.timestamp
    current_bids = {price: vol for price, vol in snapshot.bids}
    current_asks = {price: vol for price, vol in snapshot.asks}

    # Обновляем bid уровни
    self._update_side_levels("bid", current_bids, timestamp)

    # Обновляем ask уровни
    self._update_side_levels("ask", current_asks, timestamp)

  def _update_side_levels(
      self,
      side: str,
      current_levels: Dict[float, float],
      timestamp: int
  ):
    """
    Обновляет отслеживание уровней для одной стороны стакана.

    Args:
        side: "bid" или "ask"
        current_levels: Dict[price, volume]
        timestamp: Временная метка (ms)
    """
    tracker_side = self.level_tracker[side]

    # Обновляем существующие и добавляем новые уровни
    for price, volume in current_levels.items():
      if price not in tracker_side:
        # Новый уровень появился
        tracker_side[price] = {
          "first_seen": timestamp,
          "last_seen": timestamp,
          "max_volume": volume
        }
      else:
        # Уровень все еще существует - обновляем
        tracker_side[price]["last_seen"] = timestamp
        tracker_side[price]["max_volume"] = max(
          tracker_side[price]["max_volume"],
          volume
        )

    # Проверяем исчезнувшие уровни
    disappeared_prices = set(tracker_side.keys()) - set(current_levels.keys())

    for price in disappeared_prices:
      level_info = tracker_side[price]

      # Вычисляем TTL (секунды)
      ttl_ms = level_info["last_seen"] - level_info["first_seen"]
      ttl_sec = ttl_ms / 1000.0

      # Сохраняем в историю (только если TTL > 0)
      if ttl_sec > 0:
        self.level_ttl_history.append(ttl_sec)

        # Ограничиваем размер истории
        if len(self.level_ttl_history) > self.max_ttl_history:
          self.level_ttl_history.pop(0)

      # Удаляем исчезнувший уровень
      del tracker_side[price]

  def _calculate_level_ttl_stats(self) -> Tuple[float, float]:
    """
    Вычисляет статистику TTL уровней.

    Returns:
        Tuple[avg_ttl, std_ttl] в секундах
    """
    if not self.level_ttl_history:
      return 0.0, 0.0

    # Берем последние 100 значений для статистики
    recent_ttls = self.level_ttl_history[-100:]

    avg_ttl = float(np.mean(recent_ttls))
    std_ttl = float(np.std(recent_ttls))

    return avg_ttl, std_ttl

  def _estimate_trade_arrival_rate(self, snapshot: OrderBookSnapshot) -> float:
    """
    Оценивает частоту сделок на основе активности стакана.

    ВРЕМЕННАЯ РЕАЛИЗАЦИЯ до получения доступа к market trades stream.

    Логика:
    - Частые обновления стакана + большие изменения объема = высокая торговая активность
    - Используем update_frequency и volume_delta как proxy

    Формула:
    trade_arrival_rate ≈ update_freq * (1 + |volume_delta| / total_volume)

    TODO: Заменить на реальный подсчет из market trades WebSocket stream

    Args:
        snapshot: Текущий снимок стакана

    Returns:
        float: Оценка trades per second
    """
    # Базовая частота обновлений стакана
    update_freq = self._calculate_update_frequency()

    if update_freq == 0:
      return 0.0

    # Изменение объема (используем уже вычисленный volume_delta)
    volume_delta = abs(self._calculate_volume_delta(snapshot, depth=5))

    # Общий объем в стакане
    total_bid_vol = sum(vol for _, vol in snapshot.bids[:10])
    total_ask_vol = sum(vol for _, vol in snapshot.asks[:10])
    total_volume = total_bid_vol + total_ask_vol

    if total_volume == 0:
      return update_freq

    # Нормализованная активность (0-1)
    volume_activity = min(volume_delta / total_volume, 1.0)

    # Оценка trade arrival rate
    # Логика: больше изменений в объеме = больше сделок
    estimated_trade_rate = update_freq * (1.0 + volume_activity)

    return estimated_trade_rate

  def _calculate_kyle_lambda(self, snapshot: OrderBookSnapshot) -> float:
    """
    Вычисляет Kyle's Lambda - коэффициент влияния объема на цену.

    Полная формула:
    λ = ΔMid_Price / ΔVolume

    Где:
    - ΔMid_Price = изменение средней цены между snapshot'ами
    - ΔVolume = изменение объема на топ-5 уровнях

    Показывает: насколько изменится цена при торговле единицей объема.
    Высокий λ = низкая ликвидность (цена сильно реагирует на объем)
    Низкий λ = высокая ликвидность (цена стабильна при больших объемах)

    Args:
        snapshot: Текущий снимок стакана

    Returns:
        float: Kyle's Lambda (price impact per unit volume)
    """
    # Нужна история для расчета изменений
    if len(self.snapshot_history) < 10:
      # Fallback к упрощенной формуле
      spread = snapshot.spread or 0.0
      total_vol = sum(vol for _, vol in snapshot.bids[:5]) + sum(vol for _, vol in snapshot.asks[:5])
      return spread / total_vol if total_vol > 0 else 0.0

    # Берем последние 10 snapshots для расчета
    recent_snapshots = self.snapshot_history[-10:]

    price_changes = []
    volume_changes = []

    for i in range(1, len(recent_snapshots)):
      prev_snap = recent_snapshots[i - 1]
      curr_snap = recent_snapshots[i]

      # Изменение mid price
      prev_mid = prev_snap.mid_price or 0.0
      curr_mid = curr_snap.mid_price or 0.0

      if prev_mid == 0:
        continue

      delta_price = abs(curr_mid - prev_mid)

      # Изменение объема на топ-5 уровнях
      prev_vol = sum(vol for _, vol in prev_snap.bids[:5]) + sum(vol for _, vol in prev_snap.asks[:5])
      curr_vol = sum(vol for _, vol in curr_snap.bids[:5]) + sum(vol for _, vol in curr_snap.asks[:5])

      delta_volume = abs(curr_vol - prev_vol)

      # Сохраняем только значимые изменения
      if delta_volume > 0:
        price_changes.append(delta_price)
        volume_changes.append(delta_volume)

    if not price_changes or not volume_changes:
      # Нет данных - используем fallback
      spread = snapshot.spread or 0.0
      total_vol = sum(vol for _, vol in snapshot.bids[:5]) + sum(vol for _, vol in snapshot.asks[:5])
      return spread / total_vol if total_vol > 0 else 0.0

    # Вычисляем Kyle's Lambda как среднее отношение ΔPrice/ΔVolume
    lambdas = [dp / dv for dp, dv in zip(price_changes, volume_changes) if dv > 0]

    if not lambdas:
      return 0.0

    # Используем median для устойчивости к outliers
    kyle_lambda = float(np.median(lambdas))

    return kyle_lambda

  def _calculate_amihud_illiquidity(self, snapshot: OrderBookSnapshot) -> float:
    """
    Вычисляет Amihud Illiquidity Measure - меру неликвидности.

    Полная формула:
    ILLIQ = mean(|Return_t| / Volume_t)

    Где:
    - Return_t = (Price_t - Price_{t-1}) / Price_{t-1}
    - Volume_t = торговый объем в период t

    Показывает: насколько сильно цена реагирует на объем торговли.
    Высокий ILLIQ = низкая ликвидность (большие движения цены на малых объемах)
    Низкий ILLIQ = высокая ликвидность (малые движения даже на больших объемах)

    Используется для:
    - Оценки транзакционных издержек
    - Определения риска ликвидности
    - Сравнения ликвидности разных активов

    Args:
        snapshot: Текущий снимок стакана

    Returns:
        float: Amihud Illiquidity Measure
    """
    # Нужна история для расчета returns
    if len(self.snapshot_history) < 10:
      # Fallback к упрощенной формуле
      spread = snapshot.spread or 0.0
      total_vol = sum(vol for _, vol in snapshot.bids[:5]) + sum(vol for _, vol in snapshot.asks[:5])
      return spread / max(total_vol, 1.0)

    # Берем последние 20 snapshots для расчета
    recent_snapshots = self.snapshot_history[-20:]

    illiquidity_values = []

    for i in range(1, len(recent_snapshots)):
      prev_snap = recent_snapshots[i - 1]
      curr_snap = recent_snapshots[i]

      # Вычисляем return
      prev_mid = prev_snap.mid_price or 0.0
      curr_mid = curr_snap.mid_price or 0.0

      if prev_mid == 0:
        continue

      # Return = (P_t - P_{t-1}) / P_{t-1}
      return_pct = abs((curr_mid - prev_mid) / prev_mid)

      # Объем торговли (используем изменение объема в стакане как proxy)
      prev_vol = sum(vol for _, vol in prev_snap.bids[:5]) + sum(vol for _, vol in prev_snap.asks[:5])
      curr_vol = sum(vol for _, vol in curr_snap.bids[:5]) + sum(vol for _, vol in curr_snap.asks[:5])

      # Используем среднее значение как proxy для traded volume
      avg_volume = (prev_vol + curr_vol) / 2.0

      if avg_volume > 0:
        # Amihud measure для этого периода
        illiq_t = return_pct / avg_volume
        illiquidity_values.append(illiq_t)

    if not illiquidity_values:
      # Fallback
      spread = snapshot.spread or 0.0
      total_vol = sum(vol for _, vol in snapshot.bids[:5]) + sum(vol for _, vol in snapshot.asks[:5])
      return spread / max(total_vol, 1.0)

    # Среднее значение Amihud illiquidity
    amihud = float(np.mean(illiquidity_values))

    return amihud

  def _calculate_roll_spread(self, snapshot: OrderBookSnapshot) -> float:
    """
    Вычисляет Roll's Spread Estimator - оценку эффективного спреда.

    Полная формула Roll (1984):
    Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))

    Где:
    - ΔP_t = P_t - P_{t-1} (изменение цены)
    - Cov() = ковариация последовательных изменений цен

    Логика:
    Bid-ask bounce создает отрицательную автокорреляцию в ценах.
    Когда сделка происходит по ask, следующая может быть по bid (и наоборот).
    Это создает паттерн: +spread, -spread, +spread...
    Roll использовал эту отрицательную ковариацию для оценки спреда.

    Показывает:
    - Неявные транзакционные издержки
    - Эффективный спред (не quoted spread)
    - Качество price discovery

    Args:
        snapshot: Текущий снимок стакана

    Returns:
        float: Roll's Spread Estimate
    """
    # Нужна история для расчета ковариации
    if len(self.snapshot_history) < 20:
      # Fallback к effective spread
      spread = snapshot.spread or 0.0
      mid_price = snapshot.mid_price or 1.0
      return 2.0 * (spread / mid_price) if mid_price > 0 else 0.0

    # Берем последние 50 snapshots для расчета
    recent_snapshots = self.snapshot_history[-50:]

    # Извлекаем mid prices
    mid_prices = []
    for snap in recent_snapshots:
      mid = snap.mid_price
      if mid and mid > 0:
        mid_prices.append(mid)

    if len(mid_prices) < 20:
      # Недостаточно данных - fallback
      spread = snapshot.spread or 0.0
      mid_price = snapshot.mid_price or 1.0
      return 2.0 * (spread / mid_price) if mid_price > 0 else 0.0

    # Вычисляем price changes (ΔP_t)
    price_changes = np.diff(mid_prices)

    if len(price_changes) < 2:
      # Fallback
      spread = snapshot.spread or 0.0
      mid_price = snapshot.mid_price or 1.0
      return 2.0 * (spread / mid_price) if mid_price > 0 else 0.0

    # Вычисляем ковариацию между ΔP_t и ΔP_{t-1}
    # Для этого сдвигаем массив на 1
    delta_t = price_changes[1:]  # ΔP_t
    delta_t_minus_1 = price_changes[:-1]  # ΔP_{t-1}

    # Ковариация
    covariance = float(np.cov(delta_t, delta_t_minus_1)[0, 1])

    # Roll spread = 2 * sqrt(-cov)
    # Ковариация должна быть отрицательной из-за bid-ask bounce
    if covariance < 0:
      roll_spread = 2.0 * np.sqrt(-covariance)
    else:
      # Если ковариация положительная (нет bid-ask bounce),
      # используем fallback к quoted spread
      spread = snapshot.spread or 0.0
      mid_price = snapshot.mid_price or 1.0
      roll_spread = 2.0 * (spread / mid_price) if mid_price > 0 else 0.0

    return float(roll_spread)

  def _calculate_order_flow_imbalance(
      self,
      snapshot: OrderBookSnapshot,
      depth: int = 10
  ) -> float:
    """
    Вычисляет Order Flow Imbalance - взвешенное изменение объемов.

    OFI учитывает:
    - Изменение объемов на каждом уровне
    - Взвешивание по близости к mid price (ближе = важнее)
    - Направление потока капитала

    Формула:
    OFI = Σ(ΔBid_i * weight_i) - Σ(ΔAsk_i * weight_i)
    где weight_i = 1 / (1 + distance_from_mid_i)

    Положительный OFI = агрессивное покупательное давление
    Отрицательный OFI = агрессивное продавательное давление

    Args:
        snapshot: Текущий снимок стакана
        depth: Глубина анализа

    Returns:
        float: Order Flow Imbalance
    """
    # Нужен предыдущий снимок
    if len(self.snapshot_history) < 2:
      return 0.0

    prev_snapshot = self.snapshot_history[-2]
    mid_price = snapshot.mid_price or 0.0

    if mid_price == 0:
      return 0.0

    # Создаем словари для быстрого доступа
    prev_bids = {price: vol for price, vol in prev_snapshot.bids[:depth]}
    prev_asks = {price: vol for price, vol in prev_snapshot.asks[:depth]}

    weighted_bid_flow = 0.0
    weighted_ask_flow = 0.0

    # Анализируем изменения на bid стороне
    for price, volume in snapshot.bids[:depth]:
      # Вычисляем изменение объема
      prev_volume = prev_bids.get(price, 0.0)
      delta_volume = volume - prev_volume

      # Взвешиваем по близости к mid price
      distance_pct = abs(price - mid_price) / mid_price
      weight = 1.0 / (1.0 + distance_pct)

      weighted_bid_flow += delta_volume * weight

    # Анализируем изменения на ask стороне
    for price, volume in snapshot.asks[:depth]:
      # Вычисляем изменение объема
      prev_volume = prev_asks.get(price, 0.0)
      delta_volume = volume - prev_volume

      # Взвешиваем по близости к mid price
      distance_pct = abs(price - mid_price) / mid_price
      weight = 1.0 / (1.0 + distance_pct)

      weighted_ask_flow += delta_volume * weight

    # Order Flow Imbalance = bid flow - ask flow
    ofi = weighted_bid_flow - weighted_ask_flow

    return ofi