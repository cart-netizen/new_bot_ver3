"""
Layering Detector для обнаружения layering манипуляций.

Layering - размещение множественных ордеров на разных уровнях цены
для создания ложного впечатления о спросе/предложении.

Признаки layering:
- Множественные ордера на похожих уровнях
- Ордера размещаются последовательно (в короткий промежуток времени)
- Похожие размеры ордеров
- Ордера с одной стороны стакана
- Быстрая отмена всех ордеров после движения цены

Путь: backend/ml_engine/detection/layering_detector.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, deque
import numpy as np

from core.logger import get_logger
from models.orderbook import OrderBookSnapshot, OrderBookLevel

logger = get_logger(__name__)


@dataclass
class LayeringConfig:
  """Конфигурация детектора."""
  # Пороги для определения layer
  min_orders_in_layer: int = 3  # Минимум ордеров в слое
  max_price_spread_pct: float = 0.005  # Макс разброс цен в слое (0.5%)

  # Временные пороги
  placement_window_seconds: float = 30.0  # Окно для размещения
  cancellation_window_seconds: float = 60.0  # Окно для отмены

  # Пороги объема
  volume_similarity_threshold: float = 0.3  # 30% разница макс
  min_layer_volume_usdt: float = 30000.0  # $30k+ на слой

  # История
  history_window_seconds: int = 300  # 5 минут

  # Детекция
  min_confidence: float = 0.6


@dataclass
class OrderLayer:
  """Слой ордеров."""
  side: str  # "bid" или "ask"
  prices: List[float]
  volumes: List[float]
  timestamps: List[int]

  # Метрики слоя
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
    return float(np.mean(self.prices))


@dataclass
class LayeringPattern:
  """Обнаруженный паттерн layering."""
  symbol: str
  timestamp: int
  side: str
  confidence: float

  # Детали паттерна
  layers: List[OrderLayer]
  total_orders: int
  total_volume: float
  placement_duration: float
  cancellation_detected: bool

  reason: str


class OrderTracker:
  """Отслеживание истории ордеров для детекции layering."""

  def __init__(self, symbol: str, side: str):
    """
    Инициализация tracker.

    Args:
        symbol: Торговая пара
        side: Сторона ("bid" или "ask")
    """
    self.symbol = symbol
    self.side = side

    # История ордеров: price -> (timestamp, volume)
    self.order_history: Dict[float, List[Tuple[int, float]]] = defaultdict(list)

    # Текущие активные ордера
    self.active_orders: Dict[float, float] = {}  # price -> volume

  def update(self, levels: List[OrderBookLevel], timestamp: int):
    """Обновить tracker с новыми уровнями."""
    current_prices = {level.price: level.quantity for level in levels}

    # Новые ордера
    new_prices = set(current_prices.keys()) - set(self.active_orders.keys())
    for price in new_prices:
      volume = current_prices[price]
      self.order_history[price].append((timestamp, volume))

    # Обновленные ордера
    for price in set(current_prices.keys()) & set(self.active_orders.keys()):
      if current_prices[price] != self.active_orders[price]:
        self.order_history[price].append((timestamp, current_prices[price]))

    # Отмененные ордера
    cancelled_prices = set(self.active_orders.keys()) - set(current_prices.keys())
    for price in cancelled_prices:
      self.order_history[price].append((timestamp, 0.0))  # 0 = cancelled

    # Обновляем активные
    self.active_orders = current_prices

  def find_recent_placements(
      self,
      window_seconds: float,
      current_time: int
  ) -> List[Tuple[float, int, float]]:
    """
    Найти недавно размещенные ордера.

    Returns:
        List[(price, timestamp, volume)]
    """
    cutoff_time = current_time - int(window_seconds * 1000)

    placements = []
    for price, history in self.order_history.items():
      for timestamp, volume in history:
        if timestamp >= cutoff_time and volume > 0:
          # Это размещение или увеличение
          placements.append((price, timestamp, volume))

    return placements

  def find_recent_cancellations(
      self,
      window_seconds: float,
      current_time: int
  ) -> List[Tuple[float, int]]:
    """
    Найти недавно отмененные ордера.

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

  def cleanup_old_history(self, cutoff_time: int):
    """Удалить старую историю."""
    for price in list(self.order_history.keys()):
      # Фильтруем старые записи
      self.order_history[price] = [
        (ts, vol) for ts, vol in self.order_history[price]
        if ts >= cutoff_time
      ]

      # Удаляем пустые
      if not self.order_history[price]:
        del self.order_history[price]


class LayeringDetector:
  """
  Детектор layering манипуляций.

  Алгоритм:
  1. Отслеживаем размещение ордеров на каждой стороне
  2. Группируем близкие ордера в "слои" (layers)
  3. Анализируем характеристики слоев
  4. Детектируем layering паттерны
  5. Отслеживаем отмену слоев
  """

  def __init__(self, config: LayeringConfig):
    """
    Инициализация детектора.

    Args:
        config: Конфигурация
    """
    self.config = config

    # Trackers для каждого символа
    # symbol -> side -> OrderTracker
    self.trackers: Dict[str, Dict[str, OrderTracker]] = {}


    # Обнаруженные паттерны
    self.detected_patterns: Dict[str, List[LayeringPattern]] = defaultdict(list)

    # Статистика
    self.total_checks = 0
    self.patterns_detected = 0

    logger.info(
      f"Инициализирован LayeringDetector: "
      f"min_orders={config.min_orders_in_layer}, "
      f"price_spread={config.max_price_spread_pct:.2%}"
    )

  def update(self, snapshot: OrderBookSnapshot):
    """
    Обновить детектор новым snapshot.

    Args:
        snapshot: Snapshot стакана
    """
    symbol = snapshot.symbol
    timestamp = snapshot.timestamp

    # Инициализируем trackers если нужно
    if symbol not in self.trackers:
      self.trackers[symbol] = {
        "bid": OrderTracker(symbol, "bid"),
        "ask": OrderTracker(symbol, "ask")
      }

    # Обновляем trackers
    # Конвертируем tuples в OrderBookLevel
    from models.orderbook import OrderBookLevel

    bid_levels = [
      OrderBookLevel(price=price, quantity=qty)
      for price, qty in snapshot.bids
    ]
    ask_levels = [
      OrderBookLevel(price=price, quantity=qty)
      for price, qty in snapshot.asks
    ]

    # Обновляем trackers
    self.trackers[symbol]["bid"].update(bid_levels, timestamp)
    self.trackers[symbol]["ask"].update(ask_levels, timestamp)

    # Периодически детектируем паттерны
    self.total_checks += 1
    if self.total_checks % 50 == 0:  # Каждые 50 обновлений
      self._detect_patterns(symbol, timestamp, snapshot.mid_price)
      self._cleanup_old_data(symbol, timestamp)

  def _detect_patterns(
      self,
      symbol: str,
      timestamp: int,
      mid_price: Optional[float]
  ):
    """Обнаружить layering паттерны."""
    if mid_price is None:
      return

    # Проверяем каждую сторону
    for side in ["bid", "ask"]:
      pattern = self._analyze_side_for_layering(
        symbol,
        side,
        timestamp,
        mid_price
      )

      if pattern:
        self.detected_patterns[symbol].append(pattern)
        self.patterns_detected += 1

        logger.warning(
          f"🚨 LAYERING ОБНАРУЖЕН [{symbol}]: "
          f"side={side}, layers={len(pattern.layers)}, "
          f"confidence={pattern.confidence:.2f}, "
          f"reason={pattern.reason}"
        )

  def _analyze_side_for_layering(
      self,
      symbol: str,
      side: str,
      timestamp: int,
      mid_price: float
  ) -> Optional[LayeringPattern]:
    """Анализ одной стороны на layering."""
    tracker = self.trackers[symbol][side]

    # Находим недавно размещенные ордера
    placements = tracker.find_recent_placements(
      self.config.placement_window_seconds,
      timestamp
    )

    if len(placements) < self.config.min_orders_in_layer:
      return None

    # Группируем ордера в слои
    layers = self._group_into_layers(placements, mid_price)

    if not layers:
      return None

    # Фильтруем слои по критериям
    valid_layers = [
      layer for layer in layers
      if self._is_valid_layer(layer)
    ]

    if not valid_layers:
      return None

    # Проверяем отмены
    cancellations = tracker.find_recent_cancellations(
      self.config.cancellation_window_seconds,
      timestamp
    )

    cancellation_detected = len(cancellations) >= self.config.min_orders_in_layer

    # Вычисляем метрики
    total_orders = sum(layer.order_count for layer in valid_layers)
    total_volume = sum(layer.total_volume for layer in valid_layers)

    # Placement duration
    all_timestamps = []
    for layer in valid_layers:
      all_timestamps.extend(layer.timestamps)

    if len(all_timestamps) > 1:
      placement_duration = (
                               max(all_timestamps) - min(all_timestamps)
                           ) / 1000.0
    else:
      placement_duration = 0.0

    # Вычисляем confidence
    confidence = self._calculate_confidence(
      layers=valid_layers,
      total_volume=total_volume,
      placement_duration=placement_duration,
      cancellation_detected=cancellation_detected
    )

    if confidence < self.config.min_confidence:
      return None

    # Формируем reason
    reason = (
      f"{len(valid_layers)} слоев с {total_orders} ордерами "
      f"(${total_volume * mid_price:,.0f}) "
      f"размещены за {placement_duration:.1f}s"
    )

    if cancellation_detected:
      reason += f", {len(cancellations)} отмен обнаружено"

    return LayeringPattern(
      symbol=symbol,
      timestamp=timestamp,
      side=side,
      confidence=confidence,
      layers=valid_layers,
      total_orders=total_orders,
      total_volume=total_volume,
      placement_duration=placement_duration,
      cancellation_detected=cancellation_detected,
      reason=reason
    )

  def _group_into_layers(
      self,
      placements: List[Tuple[float, int, float]],
      mid_price: float
  ) -> List[OrderLayer]:
    """
    Группировать ордера в слои.

    Args:
        placements: [(price, timestamp, volume)]
        mid_price: Средняя цена

    Returns:
        Список слоев
    """
    if not placements:
      return []

    # Сортируем по цене
    placements_sorted = sorted(placements, key=lambda x: x[0])

    layers = []
    current_layer_prices = [placements_sorted[0][0]]
    current_layer_volumes = [placements_sorted[0][2]]
    current_layer_timestamps = [placements_sorted[0][1]]

    for i in range(1, len(placements_sorted)):
      price, timestamp, volume = placements_sorted[i]

      # Вычисляем spread относительно первого ордера в слое
      base_price = current_layer_prices[0]
      spread = abs(price - base_price) / mid_price

      if spread <= self.config.max_price_spread_pct:
        # Добавляем в текущий слой
        current_layer_prices.append(price)
        current_layer_volumes.append(volume)
        current_layer_timestamps.append(timestamp)
      else:
        # Создаем новый слой
        if len(current_layer_prices) >= self.config.min_orders_in_layer:
          layer = self._create_layer(
            current_layer_prices,
            current_layer_volumes,
            current_layer_timestamps
          )
          if layer:
            layers.append(layer)

        # Начинаем новый слой
        current_layer_prices = [price]
        current_layer_volumes = [volume]
        current_layer_timestamps = [timestamp]

    # Последний слой
    if len(current_layer_prices) >= self.config.min_orders_in_layer:
      layer = self._create_layer(
        current_layer_prices,
        current_layer_volumes,
        current_layer_timestamps
      )
      if layer:
        layers.append(layer)

    return layers

  def _create_layer(
      self,
      prices: List[float],
      volumes: List[float],
      timestamps: List[int]
  ) -> Optional[OrderLayer]:
    """Создать слой из ордеров."""
    if not prices:
      return None

    # Вычисляем метрики
    price_spread = float((max(prices) - min(prices)) / np.mean(prices))
    total_volume = float(sum(volumes))
    avg_volume = float(np.mean(volumes))
    volume_std = float(np.std(volumes)) if len(volumes) > 1 else 0.0

    placement_duration = (
      (max(timestamps) - min(timestamps)) / 1000.0
      if len(timestamps) > 1
      else 0.0
    )

    # Определяем сторону (bid или ask) по цене относительно других
    # Упрощенно - всегда используем сторону от tracker
    side = "bid"  # Будет переопределено позже

    return OrderLayer(
      side=side,
      prices=prices,
      volumes=volumes,
      timestamps=timestamps,
      price_spread=price_spread,
      total_volume=total_volume,
      avg_volume=avg_volume,
      volume_std=volume_std,
      placement_duration=placement_duration
    )

  def _is_valid_layer(self, layer: OrderLayer) -> bool:
    """Проверить валидность слоя."""
    # Минимальный объем
    if layer.total_volume < self.config.min_layer_volume_usdt / layer.mid_price:
      return False

    # Схожесть объемов
    if layer.order_count > 1:
      cv = layer.volume_std / layer.avg_volume  # Coefficient of variation
      if cv > self.config.volume_similarity_threshold:
        return False

    return True

  def _calculate_confidence(
      self,
      layers: List[OrderLayer],
      total_volume: float,
      placement_duration: float,
      cancellation_detected: bool
  ) -> float:
    """Вычислить уверенность в layering паттерне."""
    confidence = 0.0

    # Количество слоев
    if len(layers) >= 5:
      confidence += 0.3
    elif len(layers) >= 3:
      confidence += 0.2

    # Быстрое размещение
    if placement_duration < 10.0:
      confidence += 0.3
    elif placement_duration < 30.0:
      confidence += 0.2

    # Крупный объем
    # (total_volume уже в базовой валюте, умножаем на примерную цену)
    # Упрощение: предполагаем среднюю цену ~50000
    total_usdt = total_volume * 50000
    if total_usdt > 100000:
      confidence += 0.2
    elif total_usdt > 50000:
      confidence += 0.1

    # Отмены обнаружены
    if cancellation_detected:
      confidence += 0.2

    return min(confidence, 1.0)

  def _cleanup_old_data(self, symbol: str, timestamp: int):
    """Очистка старых данных."""
    cutoff_time = timestamp - (self.config.history_window_seconds * 1000)

    for side in ["bid", "ask"]:
      tracker = self.trackers[symbol][side]
      if tracker:
        tracker.cleanup_old_history(cutoff_time)

  def get_recent_patterns(
      self,
      symbol: str,
      time_window_seconds: int = 60
  ) -> List[LayeringPattern]:
    """Получить недавние паттерны layering."""
    current_time = int(datetime.now().timestamp() * 1000)
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
    """Проверить активность layering."""
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
      'symbols_monitored': len(self.trackers),
      'detection_rate': (
        self.patterns_detected / self.total_checks
        if self.total_checks > 0
        else 0.0
      )
    }


# Пример использования
if __name__ == "__main__":
  from models.orderbook import OrderBookSnapshot, OrderBookLevel

  config = LayeringConfig(
    min_orders_in_layer=3,
    max_price_spread_pct=0.005,
    min_layer_volume_usdt=30000.0
  )

  detector = LayeringDetector(config)

  # Симулируем layering: множественные bid ордера близко друг к другу
  base_time = int(datetime.now().timestamp() * 1000)

  # Snapshot 1: Начало layering
  snapshot1 = OrderBookSnapshot(
    symbol="BTCUSDT",
    bids=[
      OrderBookLevel(50000.0, 2.0),
      OrderBookLevel(49995.0, 2.1),
      OrderBookLevel(49990.0, 1.9),
    ],
    asks=[OrderBookLevel(50100.0, 1.0)],
    timestamp=base_time
  )

  detector.update(snapshot1)

  # Snapshot 2: Добавление еще ордеров
  snapshot2 = OrderBookSnapshot(
    symbol="BTCUSDT",
    bids=[
      OrderBookLevel(50000.0, 2.0),
      OrderBookLevel(49995.0, 2.1),
      OrderBookLevel(49990.0, 1.9),
      OrderBookLevel(49985.0, 2.0),  # Новый
    ],
    asks=[OrderBookLevel(50100.0, 1.0)],
    timestamp=base_time + 5000
  )

  detector.update(snapshot2)

  # Проверка
  patterns = detector.get_recent_patterns("BTCUSDT")
  print(f"Обнаружено layering паттернов: {len(patterns)}")

  for pattern in patterns:
    print(f"  Side: {pattern.side}")
    print(f"  Layers: {len(pattern.layers)}")
    print(f"  Confidence: {pattern.confidence:.2f}")
    print(f"  Reason: {pattern.reason}")