# backend/core/screener_processor.py
"""
Screener Data Processor - ядро обработки данных скринера.

Функционал:
- Хранение состояния торговых пар
- Построение свечей из тиков
- Расчет динамики по таймфреймам (1m, 3m, 5m, 15m)
- Расчет технических индикаторов
- Memory-optimized подход

Оптимизации:
- Использование deque для ограничения истории
- Ленивые вычисления индикаторов
- Эффективная фильтрация пар
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
from datetime import datetime

from core.logger import get_logger

logger = get_logger(__name__)


class ScreenerPairData:
  """Данные одной торговой пары для скринера."""

  def __init__(self, symbol: str, initial_data: Optional[Dict[str, Any]] = None):
    """
    Инициализация данных пары.

    Args:
        symbol: Символ торговой пары
        initial_data: Начальные данные (цена, объем, изменения)
    """
    self.symbol = symbol
    self.last_price = initial_data.get('last_price', 0.0) if initial_data else 0.0

    # История свечей (ограничена для экономии памяти)
    self.candles_1m = deque(maxlen=300)  # 5 часов
    self.candles_5m = deque(maxlen=300)  # 25 часов

    # Метаданные
    self.last_update_ts = 0
    self.is_active = True

    # Метрики динамики
    self.metrics = {
      'change_1m': None,
      'change_3m': None,
      'change_5m': None,
      'change_15m': None,
      'change_1h': None,
      'change_4h': initial_data.get('change_4h') if initial_data else None,
      'change_24h': initial_data.get('change_24h') if initial_data else None,
      'volume_24h': initial_data.get('volume_24h', 0.0) if initial_data else 0.0,
      'high_24h': initial_data.get('high_24h', 0.0) if initial_data else 0.0,
      'low_24h': initial_data.get('low_24h', 0.0) if initial_data else 0.0,
      'prev_price_24h': initial_data.get('prev_price_24h', 0.0) if initial_data else 0.0,
    }

    logger.debug(f"Инициализирована пара {symbol}: price={self.last_price}")

  def update_price(self, price: float, timestamp: float):
    """
    Обновление цены тиком.

    Args:
        price: Новая цена
        timestamp: Временная метка
    """
    self.last_price = price
    self.last_update_ts = timestamp
    self.is_active = True

    # Обновляем свечу
    self._update_candle(price, timestamp)

  def _update_candle(self, price: float, timestamp: float):
    """
    Обновление 1-минутной свечи.

    Args:
        price: Цена
        timestamp: Временная метка
    """
    # Округляем timestamp до минуты
    minute_ts = int(timestamp // 60) * 60

    # Если нет свечей или новая минута
    if not self.candles_1m or self.candles_1m[-1]['timestamp'] < minute_ts:
      # Создаем новую свечу
      self.candles_1m.append({
        'timestamp': minute_ts,
        'open': price,
        'high': price,
        'low': price,
        'close': price,
        'volume': 0.0,
      })
    else:
      # Обновляем текущую свечу
      candle = self.candles_1m[-1]
      candle['high'] = max(candle['high'], price)
      candle['low'] = min(candle['low'], price)
      candle['close'] = price

  def calculate_timeframe_changes(self):
    """Расчет изменений цены по различным таймфреймам."""
    if not self.candles_1m or len(self.candles_1m) < 2:
      return

    current_price = self.last_price

    # Функция расчета изменения
    def calc_change(minutes_ago: int) -> Optional[float]:
      """Расчет изменения за N минут назад."""
      if len(self.candles_1m) <= minutes_ago:
        return None

      # Берем свечу N минут назад
      past_candle = self.candles_1m[-(minutes_ago + 1)]
      past_price = past_candle['close']

      if past_price <= 0:
        return None

      return ((current_price - past_price) / past_price) * 100

    # Расчет изменений
    self.metrics['change_1m'] = calc_change(1)
    self.metrics['change_3m'] = calc_change(3)
    self.metrics['change_5m'] = calc_change(5)
    self.metrics['change_15m'] = calc_change(15)
    self.metrics['change_1h'] = calc_change(60)

  def update_24h_data(self, data: Dict[str, Any]):
    """
    Обновление 24-часовых данных от биржи.

    Args:
        data: Данные тикера от Bybit
    """
    self.metrics['change_24h'] = float(data.get('price24hPcnt', 0)) * 100
    self.metrics['volume_24h'] = float(data.get('turnover24h', 0))
    self.metrics['high_24h'] = float(data.get('highPrice24h', 0))
    self.metrics['low_24h'] = float(data.get('lowPrice24h', 0))
    self.metrics['prev_price_24h'] = float(data.get('prevPrice24h', 0))

  def to_dict(self) -> Dict[str, Any]:
    """
    Сериализация в словарь для отправки на фронтенд.

    Returns:
        Словарь с данными пары
    """
    return {
      'symbol': self.symbol,
      'lastPrice': self.last_price,
      'price24hPcnt': self.metrics['change_24h'] or 0.0,
      'volume24h': self.metrics['volume_24h'],
      'highPrice24h': self.metrics['high_24h'],
      'lowPrice24h': self.metrics['low_24h'],
      'prevPrice24h': self.metrics['prev_price_24h'],
      'change_1m': self.metrics['change_1m'],
      'change_3m': self.metrics['change_3m'],
      'change_5m': self.metrics['change_5m'],
      'change_15m': self.metrics['change_15m'],
      'change_1h': self.metrics['change_1h'],
      'lastUpdate': int(self.last_update_ts * 1000),  # в миллисекундах
    }


class ScreenerProcessor:
  """
  Процессор данных скринера.

  Управляет состоянием всех торговых пар, обрабатывает обновления
  и предоставляет данные для broadcast через WebSocket.
  """

  def __init__(self, min_volume: float = 4_000_000):
    """
    Инициализация процессора.

    Args:
        min_volume: Минимальный объем за 24ч для фильтрации (USDT)
    """
    self.pairs: Dict[str, ScreenerPairData] = {}
    self.min_volume = min_volume
    self.last_cleanup_ts = time.time()

    # Конфигурация
    self.MAX_PAIRS = 200  # Максимум пар в памяти
    self.CLEANUP_INTERVAL = 60  # Очистка каждую минуту
    self.INACTIVE_TTL = 300  # 5 минут без обновлений = неактивна

    logger.info(f"Инициализирован ScreenerProcessor с min_volume={min_volume:,.0f} USDT")

  def update_from_ticker(self, ticker_data: Dict[str, Any]):
    """
    Обновление данных из тикера от WebSocket Bybit.

    Args:
        ticker_data: Данные тикера
    """
    try:
      symbol = ticker_data.get('symbol')
      if not symbol or not symbol.endswith('USDT'):
        return

      # Проверка объема
      volume_24h = float(ticker_data.get('turnover24h', 0))
      if volume_24h < self.min_volume:
        # Удаляем из памяти если был
        if symbol in self.pairs:
          del self.pairs[symbol]
        return

      # Создаем или получаем пару
      if symbol not in self.pairs:
        # Проверяем лимит
        if len(self.pairs) >= self.MAX_PAIRS:
          logger.warning(f"Достигнут лимит пар ({self.MAX_PAIRS}), пропускаем {symbol}")
          return

        self.pairs[symbol] = ScreenerPairData(symbol)
        logger.debug(f"Добавлена новая пара: {symbol}")

      pair = self.pairs[symbol]

      # Обновляем цену
      last_price = float(ticker_data.get('lastPrice', 0))
      if last_price > 0:
        pair.update_price(last_price, time.time())

      # Обновляем 24h данные
      pair.update_24h_data(ticker_data)

    except (ValueError, TypeError, KeyError) as e:
      logger.warning(f"Ошибка обработки тикера: {e}")

  def calculate_all_metrics(self):
    """Расчет метрик для всех пар."""
    for pair in self.pairs.values():
      pair.calculate_timeframe_changes()

  def cleanup_inactive_pairs(self):
    """Очистка неактивных пар из памяти."""
    current_ts = time.time()

    # Проверяем интервал очистки
    if current_ts - self.last_cleanup_ts < self.CLEANUP_INTERVAL:
      return

    # Ищем неактивные пары
    inactive_symbols = []
    for symbol, pair in self.pairs.items():
      if current_ts - pair.last_update_ts > self.INACTIVE_TTL:
        inactive_symbols.append(symbol)

    # Удаляем неактивные
    for symbol in inactive_symbols:
      del self.pairs[symbol]
      logger.debug(f"Удалена неактивная пара: {symbol}")

    if inactive_symbols:
      logger.info(f"Очищено {len(inactive_symbols)} неактивных пар, "
                  f"осталось {len(self.pairs)} активных")

    self.last_cleanup_ts = current_ts

  def get_screener_data(self) -> Dict[str, Any]:
    """
    Получение данных для отправки на фронтенд.

    Returns:
        Словарь с данными всех пар
    """
    # Очищаем неактивные пары
    self.cleanup_inactive_pairs()

    # Рассчитываем метрики
    self.calculate_all_metrics()

    # Формируем данные
    pairs_data = [pair.to_dict() for pair in self.pairs.values()]

    # Сортируем по объему (по убыванию)
    pairs_data.sort(key=lambda x: x['volume24h'], reverse=True)

    return {
      'type': 'screener_data',
      'pairs': pairs_data,
      'total': len(pairs_data),
      'timestamp': int(time.time() * 1000),
      'min_volume': self.min_volume,
    }

  def get_statistics(self) -> Dict[str, Any]:
    """
    Получение статистики процессора.

    Returns:
        Словарь со статистикой
    """
    return {
      'total_pairs': len(self.pairs),
      'active_pairs': sum(1 for p in self.pairs.values() if p.is_active),
      'min_volume': self.min_volume,
      'max_pairs_limit': self.MAX_PAIRS,
    }