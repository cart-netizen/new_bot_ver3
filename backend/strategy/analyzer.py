"""
Модуль анализа рыночных данных.
Расчет метрик стакана для генерации торговых сигналов.
"""

from typing import Dict, Optional

from core.logger import get_logger
from core.exceptions import AnalysisError
from models.orderbook import OrderBookSnapshot, OrderBookMetrics
from strategy.orderbook_manager import OrderBookManager
from backend.config import settings
from utils.helpers import safe_divide, get_timestamp_ms

logger = get_logger(__name__)


class OrderBookAnalyzer:
  """Анализатор стакана ордеров для расчета метрик."""

  def __init__(self, symbol: str):
    """
    Инициализация анализатора.

    Args:
        symbol: Торговая пара
    """
    self.symbol = symbol
    logger.debug(f"Инициализирован анализатор для {symbol}")

  def analyze(self, orderbook_manager: OrderBookManager) -> OrderBookMetrics:
    """
    Полный анализ стакана с расчетом всех метрик.

    Args:
        orderbook_manager: Менеджер стакана

    Returns:
        OrderBookMetrics: Рассчитанные метрики
    """
    try:
      snapshot = orderbook_manager.get_snapshot()

      if not snapshot:
        logger.warning(f"{self.symbol} | Нет данных стакана для анализа")
        return self._create_empty_metrics()

      metrics = OrderBookMetrics(
        symbol=self.symbol,
        timestamp=snapshot.timestamp or get_timestamp_ms()
      )

      # Базовые метрики
      metrics.best_bid = snapshot.best_bid
      metrics.best_ask = snapshot.best_ask
      metrics.spread = snapshot.spread
      metrics.mid_price = snapshot.mid_price

      # Объемные метрики
      self._calculate_volume_metrics(snapshot, metrics)

      # Дисбаланс
      self._calculate_imbalance(snapshot, metrics)

      # VWAP метрики
      self._calculate_vwap_metrics(snapshot, metrics)

      # Кластеры
      self._calculate_clusters(snapshot, metrics)

      logger.debug(
        f"{self.symbol} | Метрики рассчитаны: "
        f"imbalance={metrics.imbalance:.4f}, "
        f"spread={metrics.spread:.8f}"
      )

      return metrics

    except Exception as e:
      logger.error(f"{self.symbol} | Ошибка анализа стакана: {e}")
      raise AnalysisError(f"Failed to analyze orderbook: {str(e)}")

  def _create_empty_metrics(self) -> OrderBookMetrics:
    """
    Создание пустых метрик.

    Returns:
        OrderBookMetrics: Пустые метрики
    """
    return OrderBookMetrics(
      symbol=self.symbol,
      timestamp=get_timestamp_ms()
    )

  def _calculate_volume_metrics(
      self,
      snapshot: OrderBookSnapshot,
      metrics: OrderBookMetrics
  ):
    """
    Расчет объемных метрик.

    Args:
        snapshot: Снимок стакана
        metrics: Объект метрик для заполнения
    """
    # Общие объемы
    metrics.total_bid_volume = sum(qty for _, qty in snapshot.bids)
    metrics.total_ask_volume = sum(qty for _, qty in snapshot.asks)

    # Объемы на разной глубине
    metrics.bid_volume_depth_5 = sum(
      qty for _, qty in snapshot.bids[:5]
    )
    metrics.ask_volume_depth_5 = sum(
      qty for _, qty in snapshot.asks[:5]
    )

    metrics.bid_volume_depth_10 = sum(
      qty for _, qty in snapshot.bids[:10]
    )
    metrics.ask_volume_depth_10 = sum(
      qty for _, qty in snapshot.asks[:10]
    )

  def _calculate_imbalance(
      self,
      snapshot: OrderBookSnapshot,
      metrics: OrderBookMetrics
  ):
    """
    Расчет дисбаланса спроса и предложения.

    Args:
        snapshot: Снимок стакана
        metrics: Объект метрик для заполнения
    """
    # Общий дисбаланс
    total_volume = metrics.total_bid_volume + metrics.total_ask_volume
    metrics.imbalance = safe_divide(
      metrics.total_bid_volume,
      total_volume,
      default=0.5
    )

    # Дисбаланс на глубине 5
    volume_5 = metrics.bid_volume_depth_5 + metrics.ask_volume_depth_5
    metrics.imbalance_depth_5 = safe_divide(
      metrics.bid_volume_depth_5,
      volume_5,
      default=0.5
    )

    # Дисбаланс на глубине 10
    volume_10 = metrics.bid_volume_depth_10 + metrics.ask_volume_depth_10
    metrics.imbalance_depth_10 = safe_divide(
      metrics.bid_volume_depth_10,
      volume_10,
      default=0.5
    )

  def _calculate_vwap_metrics(
      self,
      snapshot: OrderBookSnapshot,
      metrics: OrderBookMetrics
  ):
    """
    Расчет VWAP (Volume Weighted Average Price) метрик.

    Args:
        snapshot: Снимок стакана
        metrics: Объект метрик для заполнения
    """
    # VWAP для bid стороны
    if snapshot.bids and metrics.total_bid_volume > 0:
      bid_weighted_sum = sum(
        price * qty for price, qty in snapshot.bids[:20]
      )
      metrics.vwap_bid = bid_weighted_sum / sum(
        qty for _, qty in snapshot.bids[:20]
      )

    # VWAP для ask стороны
    if snapshot.asks and metrics.total_ask_volume > 0:
      ask_weighted_sum = sum(
        price * qty for price, qty in snapshot.asks[:20]
      )
      metrics.vwap_ask = ask_weighted_sum / sum(
        qty for _, qty in snapshot.asks[:20]
      )

    # Volume Weighted Mid Price
    if metrics.vwap_bid and metrics.vwap_ask:
      total_volume = metrics.total_bid_volume + metrics.total_ask_volume
      if total_volume > 0:
        metrics.vwmp = (
            (metrics.vwap_bid * metrics.total_bid_volume +
             metrics.vwap_ask * metrics.total_ask_volume) /
            total_volume
        )

  def _calculate_clusters(
      self,
      snapshot: OrderBookSnapshot,
      metrics: OrderBookMetrics
  ):
    """
    Определение кластеров объема.

    Args:
        snapshot: Снимок стакана
        metrics: Объект метрик для заполнения
    """
    min_cluster_volume = settings.MIN_CLUSTER_VOLUME

    # Поиск кластера на bid стороне
    if snapshot.bids:
      max_bid_volume = 0
      max_bid_price = None

      for price, qty in snapshot.bids[:20]:
        if qty > max_bid_volume:
          max_bid_volume = qty
          max_bid_price = price

      if max_bid_volume >= min_cluster_volume:
        metrics.largest_bid_cluster_price = max_bid_price
        metrics.largest_bid_cluster_volume = max_bid_volume

    # Поиск кластера на ask стороне
    if snapshot.asks:
      max_ask_volume = 0
      max_ask_price = None

      for price, qty in snapshot.asks[:20]:
        if qty > max_ask_volume:
          max_ask_volume = qty
          max_ask_price = price

      if max_ask_volume >= min_cluster_volume:
        metrics.largest_ask_cluster_price = max_ask_price
        metrics.largest_ask_cluster_volume = max_ask_volume


class MarketAnalyzer:
  """Анализатор рынка для множества пар."""

  def __init__(self):
    """Инициализация анализатора рынка."""
    self.analyzers: Dict[str, OrderBookAnalyzer] = {}
    self.latest_metrics: Dict[str, OrderBookMetrics] = {}
    logger.info("Инициализирован анализатор рынка")

  def add_symbol(self, symbol: str):
    """
    Добавление символа для анализа.

    Args:
        symbol: Торговая пара
    """
    if symbol not in self.analyzers:
      self.analyzers[symbol] = OrderBookAnalyzer(symbol)
      logger.info(f"Добавлен анализатор для {symbol}")

  def analyze_symbol(
      self,
      symbol: str,
      orderbook_manager: OrderBookManager
  ) -> OrderBookMetrics:
    """
    Анализ конкретного символа.

    Args:
        symbol: Торговая пара
        orderbook_manager: Менеджер стакана

    Returns:
        OrderBookMetrics: Рассчитанные метрики
    """
    if symbol not in self.analyzers:
      self.add_symbol(symbol)

    metrics = self.analyzers[symbol].analyze(orderbook_manager)
    self.latest_metrics[symbol] = metrics

    return metrics

  def get_latest_metrics(self, symbol: str) -> Optional[OrderBookMetrics]:
    """
    Получение последних метрик для символа.

    Args:
        symbol: Торговая пара

    Returns:
        OrderBookMetrics: Последние метрики или None
    """
    return self.latest_metrics.get(symbol)

  def get_all_metrics(self) -> Dict[str, OrderBookMetrics]:
    """
    Получение метрик для всех символов.

    Returns:
        Dict[str, OrderBookMetrics]: Словарь метрик
    """
    return self.latest_metrics.copy()

  def get_metrics_summary(self) -> Dict:
    """
    Получение сводки по метрикам.

    Returns:
        Dict: Сводная информация
    """
    summary = {
      "total_symbols": len(self.analyzers),
      "symbols_with_data": len(self.latest_metrics),
      "avg_imbalance": 0.0,
      "symbols": []
    }

    if self.latest_metrics:
      imbalances = [m.imbalance for m in self.latest_metrics.values()]
      summary["avg_imbalance"] = sum(imbalances) / len(imbalances)

      for symbol, metrics in self.latest_metrics.items():
        summary["symbols"].append({
          "symbol": symbol,
          "imbalance": metrics.imbalance,
          "spread": metrics.spread,
          "mid_price": metrics.mid_price,
        })

    return summary