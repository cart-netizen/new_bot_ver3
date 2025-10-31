"""
TradeManager для отслеживания и анализа market trades в реальном времени.

Этот модуль управляет историей публичных сделок с биржи (не собственных сделок бота),
полученных через WebSocket stream publicTrade.

Используется для:
- Расчета реального trade arrival rate (trades per second)
- Определения buy/sell pressure (агрессивность участников)
- Order flow toxicity analysis (информированные vs случайные трейдеры)
- Block trades detection (институциональная активность)
- Реального VWAP расчета из исполненных сделок
- Улучшения ML признаков для более точных предсказаний

Путь: backend/strategy/trade_manager.py
"""

from typing import List, Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import time
import numpy as np

from core.logger import get_logger
from models.market_data import MarketTrade

logger = get_logger(__name__)


@dataclass
class TradeStatistics:
  """Статистика market trades за временное окно."""

  symbol: str
  window_seconds: int

  # Основные метрики
  total_trades: int
  trades_per_second: float

  # Buy/Sell анализ
  buy_trades: int
  sell_trades: int
  buy_volume: float
  sell_volume: float
  buy_sell_ratio: float

  # Агрессивность
  aggressive_buy_volume: float
  aggressive_sell_volume: float
  aggressive_ratio: float

  # Размеры сделок
  avg_trade_size: float
  median_trade_size: float
  max_trade_size: float

  # Институционалы
  block_trades_count: int
  block_trades_volume: float

  # VWAP
  vwap: float

  # Токсичность потока
  order_flow_toxicity: float


class TradeManager:
  """
  Менеджер для хранения и анализа market trades в реальном времени.

  Профессиональная реализация с:
  - Эффективным хранением (deque с ограничением памяти)
  - Быстрыми вычислениями (кэширование, incremental updates)
  - Множественными временными окнами (10s, 30s, 60s, 5m)
  - Детальной статистикой для ML
  - Обработкой edge cases
  """

  def __init__(
      self,
      symbol: str,
      max_history: int = 5000,  # Последние 5000 сделок (~5-10 минут истории)
      enable_statistics: bool = True
  ):
    """
    Инициализация TradeManager.

    Args:
        symbol: Торговая пара (BTCUSDT)
        max_history: Максимум сделок в истории (для ограничения памяти)
        enable_statistics: Включить расширенную статистику
    """
    self.symbol = symbol
    self.max_history = max_history
    self.enable_statistics = enable_statistics

    # История trades (deque для эффективного добавления/удаления)
    self.recent_trades: deque[MarketTrade] = deque(maxlen=max_history)

    # Счетчики для быстрых вычислений
    self._total_trades_received = 0
    self._last_cleanup_time = time.time()

    # Кэш статистики (обновляется каждые N trades)
    self._stats_cache: Dict[int, Optional[TradeStatistics]] = {
      10: None,   # 10 секунд
      30: None,   # 30 секунд
      60: None,   # 1 минута
      300: None   # 5 минут
    }
    self._last_stats_update = 0
    self._stats_update_interval = 10  # Обновлять статистику каждые 10 trades

    logger.info(
      f"TradeManager инициализирован для {symbol}, "
      f"max_history={max_history}, stats={'ON' if enable_statistics else 'OFF'}"
    )

  def add_trade(self, trade: MarketTrade):
    """
    Добавление новой market trade в историю.

    Args:
        trade: Объект MarketTrade из WebSocket
    """
    # Валидация
    if trade.symbol != self.symbol:
      logger.warning(
        f"Trade symbol mismatch: expected {self.symbol}, got {trade.symbol}"
      )
      return

    # Добавляем в историю (deque автоматически удаляет старые если > maxlen)
    self.recent_trades.append(trade)
    self._total_trades_received += 1

    # Инвалидируем кэш статистики
    if self._total_trades_received % self._stats_update_interval == 0:
      self._invalidate_stats_cache()

    # Периодическая очистка старых данных (каждые 60 секунд)
    current_time = time.time()
    if current_time - self._last_cleanup_time > 60:
      self._cleanup_old_trades()
      self._last_cleanup_time = current_time

    # Логируем первые trades для диагностики
    if self._total_trades_received <= 5:
      logger.info(
        f"{self.symbol} | Trade #{self._total_trades_received}: "
        f"side={trade.side}, price={trade.price:.2f}, "
        f"volume={trade.quantity:.4f}, block={trade.is_block_trade}"
      )

  def calculate_arrival_rate(self, window_seconds: int = 60) -> float:
    """
    Вычислить реальную частоту сделок (trades per second).

    Args:
        window_seconds: Временное окно для расчета (секунды)

    Returns:
        float: Trades per second
    """
    if not self.recent_trades:
      return 0.0

    current_time = time.time() * 1000  # ms
    cutoff_time = current_time - (window_seconds * 1000)

    # Подсчет trades за окно
    recent_count = sum(
      1 for t in self.recent_trades
      if t.timestamp >= cutoff_time
    )

    if recent_count == 0:
      return 0.0

    return recent_count / window_seconds

  def calculate_buy_sell_pressure(
      self,
      window_seconds: int = 60
  ) -> Tuple[float, float, float]:
    """
    Вычислить реальное давление покупателей/продавцов.

    Args:
        window_seconds: Временное окно

    Returns:
        Tuple[buy_volume, sell_volume, buy_sell_ratio]
    """
    if not self.recent_trades:
      return 0.0, 0.0, 1.0

    current_time = time.time() * 1000
    cutoff_time = current_time - (window_seconds * 1000)

    buy_volume = 0.0
    sell_volume = 0.0

    for trade in self.recent_trades:
      if trade.timestamp < cutoff_time:
        continue

      if trade.is_buy:
        buy_volume += trade.quantity
      else:
        sell_volume += trade.quantity

    # Buy/Sell Ratio
    if sell_volume > 0:
      ratio = buy_volume / sell_volume
    else:
      ratio = 10.0 if buy_volume > 0 else 1.0

    return buy_volume, sell_volume, ratio

  def calculate_order_flow_toxicity(
      self,
      window_seconds: int = 60,
      price_change_window: int = 10
  ) -> float:
    """
    Вычислить Order Flow Toxicity - меру информированности трейдеров.

    Токсичность показывает насколько агрессивные сделки предсказывают
    движение цены. Высокая токсичность = умные деньги (informed traders).

    Формула:
    toxicity = corr(volume_imbalance, price_movement)

    Args:
        window_seconds: Окно для анализа
        price_change_window: Окно для измерения движения цены (секунды)

    Returns:
        float: Toxicity score (-1 до +1)
    """
    if len(self.recent_trades) < 20:
      return 0.0

    current_time = time.time() * 1000
    cutoff_time = current_time - (window_seconds * 1000)

    # Фильтруем trades в окне
    recent = [t for t in self.recent_trades if t.timestamp >= cutoff_time]

    if len(recent) < 10:
      return 0.0

    try:
      # Разбиваем на мини-окна по 10 секунд
      mini_window_ms = price_change_window * 1000

      imbalances = []
      price_movements = []

      # Группируем trades по мини-окнам
      window_start = recent[0].timestamp

      for i in range(len(recent)):
        if recent[i].timestamp - window_start > mini_window_ms:
          # Вычисляем imbalance для этого окна
          window_trades = [
            t for t in recent
            if window_start <= t.timestamp < window_start + mini_window_ms
          ]

          if len(window_trades) >= 2:
            buy_vol = sum(t.quantity for t in window_trades if t.is_buy)
            sell_vol = sum(t.quantity for t in window_trades if t.is_sell)

            total_vol = buy_vol + sell_vol
            if total_vol > 0:
              imbalance = (buy_vol - sell_vol) / total_vol
              imbalances.append(imbalance)

              # Движение цены после окна
              start_price = window_trades[0].price
              end_price = window_trades[-1].price
              price_move = (end_price - start_price) / start_price
              price_movements.append(price_move)

          window_start = recent[i].timestamp

      # Вычисляем корреляцию
      if len(imbalances) >= 3:
        correlation = float(np.corrcoef(imbalances, price_movements)[0, 1])
        return correlation if not np.isnan(correlation) else 0.0

      return 0.0

    except Exception as e:
      logger.error(f"{self.symbol} | Ошибка расчета toxicity: {e}")
      return 0.0

  def calculate_vwap(self, window_seconds: int = 60) -> float:
    """
    Вычислить реальный VWAP из исполненных сделок.

    Args:
        window_seconds: Временное окно

    Returns:
        float: Volume-Weighted Average Price
    """
    if not self.recent_trades:
      return 0.0

    current_time = time.time() * 1000
    cutoff_time = current_time - (window_seconds * 1000)

    total_value = 0.0
    total_volume = 0.0

    for trade in self.recent_trades:
      if trade.timestamp < cutoff_time:
        continue

      total_value += trade.price * trade.quantity
      total_volume += trade.quantity

    if total_volume == 0:
      return 0.0

    return total_value / total_volume

  def get_statistics(self, window_seconds: int = 60) -> TradeStatistics:
    """
    Получить полную статистику trades за временное окно.

    Args:
        window_seconds: Временное окно (секунды)

    Returns:
        TradeStatistics: Детальная статистика
    """
    # Проверяем кэш
    if window_seconds in self._stats_cache and self._stats_cache[window_seconds]:
      cache_age = self._total_trades_received - self._last_stats_update
      if cache_age < self._stats_update_interval:
        return self._stats_cache[window_seconds]

    # Вычисляем статистику
    if not self.recent_trades:
      return self._create_empty_statistics(window_seconds)

    current_time = time.time() * 1000
    cutoff_time = current_time - (window_seconds * 1000)

    # Фильтруем trades
    window_trades = [t for t in self.recent_trades if t.timestamp >= cutoff_time]

    if not window_trades:
      return self._create_empty_statistics(window_seconds)

    # Основные метрики
    total_trades = len(window_trades)
    arrival_rate = total_trades / window_seconds

    # Buy/Sell анализ
    buy_trades = sum(1 for t in window_trades if t.is_buy)
    sell_trades = sum(1 for t in window_trades if t.is_sell)

    buy_volume = sum(t.quantity for t in window_trades if t.is_buy)
    sell_volume = sum(t.quantity for t in window_trades if t.is_sell)

    buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else 10.0

    # Агрессивность (все trades в publicTrade - это агрессивные taker orders)
    aggressive_buy = buy_volume
    aggressive_sell = sell_volume
    aggressive_ratio = aggressive_buy / aggressive_sell if aggressive_sell > 0 else 10.0

    # Размеры сделок
    sizes = [t.quantity for t in window_trades]
    avg_size = float(np.mean(sizes))
    median_size = float(np.median(sizes))
    max_size = float(np.max(sizes))

    # Block trades
    block_trades = [t for t in window_trades if t.is_block_trade]
    block_count = len(block_trades)
    block_volume = sum(t.quantity for t in block_trades)

    # VWAP
    total_value = sum(t.price * t.quantity for t in window_trades)
    total_volume = sum(t.quantity for t in window_trades)
    vwap = total_value / total_volume if total_volume > 0 else 0.0

    # Toxicity
    toxicity = self.calculate_order_flow_toxicity(window_seconds)

    stats = TradeStatistics(
      symbol=self.symbol,
      window_seconds=window_seconds,
      total_trades=total_trades,
      trades_per_second=arrival_rate,
      buy_trades=buy_trades,
      sell_trades=sell_trades,
      buy_volume=buy_volume,
      sell_volume=sell_volume,
      buy_sell_ratio=buy_sell_ratio,
      aggressive_buy_volume=aggressive_buy,
      aggressive_sell_volume=aggressive_sell,
      aggressive_ratio=aggressive_ratio,
      avg_trade_size=avg_size,
      median_trade_size=median_size,
      max_trade_size=max_size,
      block_trades_count=block_count,
      block_trades_volume=block_volume,
      vwap=vwap,
      order_flow_toxicity=toxicity
    )

    # Кэшируем
    if window_seconds in self._stats_cache:
      self._stats_cache[window_seconds] = stats
      self._last_stats_update = self._total_trades_received

    return stats

  def get_recent_trades(self, count: int = 100) -> List[MarketTrade]:
    """
    Получить последние N trades.

    Args:
        count: Количество trades

    Returns:
        List[MarketTrade]: Последние trades (от новых к старым)
    """
    return list(self.recent_trades)[-count:]

  def _cleanup_old_trades(self):
    """
    Очистка устаревших trades для экономии памяти.

    Удаляет trades старше 10 минут.
    """
    if not self.recent_trades:
      return

    current_time = time.time() * 1000
    cutoff_time = current_time - (600 * 1000)  # 10 минут

    # Подсчитываем сколько нужно удалить
    old_count = sum(1 for t in self.recent_trades if t.timestamp < cutoff_time)

    if old_count > 0:
      # Удаляем старые trades
      self.recent_trades = deque(
        (t for t in self.recent_trades if t.timestamp >= cutoff_time),
        maxlen=self.max_history
      )

      logger.debug(f"{self.symbol} | Очищено {old_count} старых trades")

  def _invalidate_stats_cache(self):
    """Инвалидация кэша статистики."""
    for window in self._stats_cache:
      self._stats_cache[window] = None

  def _create_empty_statistics(self, window_seconds: int) -> TradeStatistics:
    """Создать пустую статистику."""
    return TradeStatistics(
      symbol=self.symbol,
      window_seconds=window_seconds,
      total_trades=0,
      trades_per_second=0.0,
      buy_trades=0,
      sell_trades=0,
      buy_volume=0.0,
      sell_volume=0.0,
      buy_sell_ratio=1.0,
      aggressive_buy_volume=0.0,
      aggressive_sell_volume=0.0,
      aggressive_ratio=1.0,
      avg_trade_size=0.0,
      median_trade_size=0.0,
      max_trade_size=0.0,
      block_trades_count=0,
      block_trades_volume=0.0,
      vwap=0.0,
      order_flow_toxicity=0.0
    )

  def get_info(self) -> Dict:
    """
    Получить информацию о состоянии TradeManager.

    Returns:
        Dict: Информация и статистика
    """
    return {
      "symbol": self.symbol,
      "total_trades_received": self._total_trades_received,
      "current_history_size": len(self.recent_trades),
      "max_history": self.max_history,
      "memory_usage_mb": len(self.recent_trades) * 0.0001,  # Приблизительно
      "statistics_enabled": self.enable_statistics,
      "oldest_trade_age_seconds": (
        (time.time() * 1000 - self.recent_trades[0].timestamp) / 1000
        if self.recent_trades else 0
      )
    }
