"""
Менеджер стакана ордеров.
Управляет локальным состоянием стакана для каждой торговой пары.
"""

from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

from backend.core.logger import get_logger
from backend.core.exceptions import OrderBookError, OrderBookSyncError
from backend.models.orderbook import OrderBookSnapshot, OrderBookDelta
from backend.utils.helpers import get_timestamp_ms

logger = get_logger(__name__)


class OrderBookManager:
  """Менеджер для управления состоянием стакана ордеров."""

  def __init__(self, symbol: str, max_depth: int = 50):
    """
    Инициализация менеджера стакана.

    Args:
        symbol: Торговая пара
        max_depth: Максимальная глубина стакана (уровней) для экономии памяти
    """
    self.symbol = symbol
    self.max_depth = max_depth  # НОВОЕ: Ограничение глубины для экономии памяти

    # Хранилище уровней стакана
    self.bids: OrderedDict[float, float] = OrderedDict()  # {price: quantity}
    self.asks: OrderedDict[float, float] = OrderedDict()  # {price: quantity}

    # Метаданные
    self.last_update_id: Optional[int] = None
    self.last_sequence_id: Optional[int] = None
    self.last_update_timestamp: Optional[int] = None
    self.snapshot_received: bool = False

    # Счетчики
    self.snapshot_count: int = 0
    self.delta_count: int = 0

    logger.info(f"Инициализирован OrderBook менеджер для {symbol}, max_depth={max_depth}")

  def apply_snapshot(self, data: Dict) -> OrderBookSnapshot:
    """
    Применение snapshot (полного снимка) стакана.

    Args:
        data: Данные snapshot от WebSocket

    Returns:
        OrderBookSnapshot: Объект снимка стакана
    """
    try:
      # Очищаем текущее состояние
      self.bids.clear()
      self.asks.clear()

      # Применяем новые данные
      bids_list = []
      asks_list = []

      # ИСПРАВЛЕНО: Берем только top N уровней из snapshot
      bids_data = data.get("b", [])[:self.max_depth]
      asks_data = data.get("a", [])[:self.max_depth]

      for bid_data in bids_data:
        price = float(bid_data[0])
        quantity = float(bid_data[1])
        self.bids[price] = quantity
        bids_list.append((price, quantity))

      for ask_data in asks_data:
        price = float(ask_data[0])
        quantity = float(ask_data[1])
        self.asks[price] = quantity
        asks_list.append((price, quantity))

      # Сортируем
      self.bids = OrderedDict(sorted(self.bids.items(), reverse=True))
      self.asks = OrderedDict(sorted(self.asks.items()))

      # Обновляем метаданные
      self.last_update_id = data.get("u")
      self.last_sequence_id = data.get("seq")

      # CRITICAL: Ensure timestamp is valid (not None, not 0, not empty string)
      ts = data.get("ts")
      if ts and ts != 0:
        self.last_update_timestamp = ts
      else:
        current_time = get_timestamp_ms()
        self.last_update_timestamp = current_time
        logger.warning(
          f"{self.symbol} | [OrderBook.apply_snapshot] Invalid timestamp received: ts={ts} (type={type(ts)}), "
          f"using current time: {current_time}"
        )

      self.snapshot_received = True  # ВАЖНО: Флаг что snapshot получен
      self.snapshot_count += 1

      logger.info(
        f"{self.symbol} | Snapshot применен: "
        f"bids={len(self.bids)}, asks={len(self.asks)}, "
        f"seq={self.last_sequence_id}"
      )

      # Создаем объект snapshot
      snapshot = OrderBookSnapshot(
        symbol=self.symbol,
        bids=bids_list,
        asks=asks_list,
        timestamp=self.last_update_timestamp,
        update_id=self.last_update_id,
        sequence_id=self.last_sequence_id
      )

      return snapshot

    except Exception as e:
      logger.error(f"{self.symbol} | Ошибка применения snapshot: {e}")
      raise OrderBookError(f"Failed to apply snapshot: {str(e)}")

  def clear_old_data(self):
    """
    Очистка старых данных для освобождения памяти.
    Вызывается периодически для предотвращения утечек памяти.
    """
    # Ограничиваем количество уровней до max_depth
    if len(self.bids) > self.max_depth:
      # Оставляем только top N лучших bid (самые высокие цены)
      sorted_bids = sorted(self.bids.items(), reverse=True)[:self.max_depth]
      self.bids = OrderedDict(sorted_bids)
      logger.debug(f"{self.symbol} | Ограничена глубина bids до {self.max_depth}")

    if len(self.asks) > self.max_depth:
      # Оставляем только top N лучших ask (самые низкие цены)
      sorted_asks = sorted(self.asks.items())[:self.max_depth]
      self.asks = OrderedDict(sorted_asks)
      logger.debug(f"{self.symbol} | Ограничена глубина asks до {self.max_depth}")

  def apply_delta(self, data: Dict) -> Optional[OrderBookDelta]:
    """
    Применение delta (инкрементального обновления) стакана.

    Args:
        data: Данные delta от WebSocket

    Returns:
        OrderBookDelta: Объект дельта-обновления или None если snapshot не получен
    """
    if not self.snapshot_received:
      # ИСПРАВЛЕНИЕ: Не выбрасываем исключение, просто возвращаем None
      logger.debug(f"{self.symbol} | Delta пропущена: snapshot не получен")
      return None

    try:
      bids_update = []
      asks_update = []

      # Обновляем bids
      for bid_data in data.get("b", []):
        price = float(bid_data[0])
        quantity = float(bid_data[1])

        if quantity == 0:
          # Удаляем уровень
          self.bids.pop(price, None)
        else:
          # Добавляем или обновляем уровень
          self.bids[price] = quantity

        bids_update.append((price, quantity))

      # Обновляем asks
      for ask_data in data.get("a", []):
        price = float(ask_data[0])
        quantity = float(ask_data[1])

        if quantity == 0:
          # Удаляем уровень
          self.asks.pop(price, None)
        else:
          # Добавляем или обновляем уровень
          self.asks[price] = quantity

        asks_update.append((price, quantity))

      # Пересортировываем для сохранения порядка
      self.bids = OrderedDict(sorted(self.bids.items(), reverse=True))
      self.asks = OrderedDict(sorted(self.asks.items()))

      # Обновляем метаданные
      self.last_update_id = data.get("u")
      self.last_sequence_id = data.get("seq")

      # CRITICAL: Ensure timestamp is valid (not None, not 0)
      ts = data.get("ts")
      if ts and ts != 0:
        self.last_update_timestamp = ts
      else:
        current_time = get_timestamp_ms()
        self.last_update_timestamp = current_time
        logger.warning(
          f"{self.symbol} | [OrderBook.apply_delta] Invalid timestamp received: ts={ts} (type={type(ts)}), "
          f"using current time: {current_time}"
        )

      self.delta_count += 1

      logger.debug(
        f"{self.symbol} | Delta применена: "
        f"bids={len(bids_update)}, asks={len(asks_update)}, "
        f"seq={self.last_sequence_id}"
      )

      # Создаем объект delta
      delta = OrderBookDelta(
        symbol=self.symbol,
        bids_update=bids_update,
        asks_update=asks_update,
        timestamp=self.last_update_timestamp,
        update_id=self.last_update_id,
        sequence_id=self.last_sequence_id
      )

      return delta

    except Exception as e:
      logger.error(f"{self.symbol} | Ошибка применения delta: {e}")
      raise OrderBookError(f"Failed to apply delta: {str(e)}")

  def get_snapshot(self) -> Optional[OrderBookSnapshot]:
    """
    Получение текущего состояния стакана.

    Returns:
        OrderBookSnapshot: Снимок текущего состояния или None
    """
    if not self.snapshot_received:
      return None

    # CRITICAL: Ensure timestamp is never None or 0
    # If last_update_timestamp is None/0, use current time
    timestamp = self.last_update_timestamp
    if timestamp is None or timestamp == 0:
      timestamp = get_timestamp_ms()
      logger.warning(
        f"{self.symbol} | [OrderBook.get_snapshot] Invalid timestamp detected: "
        f"last_update_timestamp={self.last_update_timestamp} (type={type(self.last_update_timestamp)}), "
        f"using current time: {timestamp}"
      )
    else:
      # DEBUG: Логируем валидный timestamp (каждый 100-й раз, чтобы не спамить)
      if self.snapshot_count % 100 == 0:
        logger.debug(
          f"{self.symbol} | [OrderBook.get_snapshot] Using valid timestamp: {timestamp}"
        )

    return OrderBookSnapshot(
      symbol=self.symbol,
      bids=list(self.bids.items()),
      asks=list(self.asks.items()),
      timestamp=timestamp,
      update_id=self.last_update_id,
      sequence_id=self.last_sequence_id
    )

  def get_best_bid(self) -> Optional[Tuple[float, float]]:
    """
    Получение лучшего bid.

    Returns:
        Tuple[float, float]: (price, quantity) или None
    """
    if self.bids:
      price = next(iter(self.bids))
      return (price, self.bids[price])
    return None

  def get_best_ask(self) -> Optional[Tuple[float, float]]:
    """
    Получение лучшего ask.

    Returns:
        Tuple[float, float]: (price, quantity) или None
    """
    if self.asks:
      price = next(iter(self.asks))
      return (price, self.asks[price])
    return None

  def get_spread(self) -> Optional[float]:
    """
    Получение спреда.

    Returns:
        float: Спред или None
    """
    best_bid = self.get_best_bid()
    best_ask = self.get_best_ask()

    if best_bid and best_ask:
      return best_ask[0] - best_bid[0]
    return None

  def get_mid_price(self) -> Optional[float]:
    """
    Получение средней цены.

    Returns:
        float: Средняя цена или None
    """
    best_bid = self.get_best_bid()
    best_ask = self.get_best_ask()

    if best_bid and best_ask:
      return (best_bid[0] + best_ask[0]) / 2
    return None

  def get_depth_volume(self, side: str, levels: int = 10) -> float:
    """
    Получение суммарного объема на заданной глубине.

    Args:
        side: Сторона ("bid" или "ask")
        levels: Количество уровней

    Returns:
        float: Суммарный объем
    """
    if side == "bid":
      items = list(self.bids.items())[:levels]
    else:
      items = list(self.asks.items())[:levels]

    return sum(quantity for _, quantity in items)

  def get_stats(self) -> Dict:
    """
    Получение статистики стакана.

    Returns:
        Dict: Статистика
    """
    best_bid = self.get_best_bid()
    best_ask = self.get_best_ask()

    return {
      "symbol": self.symbol,
      "snapshot_received": self.snapshot_received,
      "snapshot_count": self.snapshot_count,
      "delta_count": self.delta_count,
      "levels": {
        "bids": len(self.bids),
        "asks": len(self.asks),
      },
      "best_prices": {
        "bid": best_bid[0] if best_bid else None,
        "ask": best_ask[0] if best_ask else None,
      },
      "spread": self.get_spread(),
      "mid_price": self.get_mid_price(),
      "last_update_timestamp": self.last_update_timestamp,
      "last_update_id": self.last_update_id,
    }