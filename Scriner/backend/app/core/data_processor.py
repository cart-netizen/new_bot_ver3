# backend/app/core/data_processor.py

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import List, Dict, Any
from collections import deque
import time
import copy

from .logger import log


class ScreenerDataProcessor:
  """
  Хранит состояние торговых пар, строит свечи, обрабатывает тики в реальном времени
  и вычисляет все необходимые метрики и индикаторы.
  """

  def __init__(self, symbols: List[str], initial_data: Dict[str, Any],
               historical_5m_data: Dict[str, List[Dict]] = None):
    self.symbols = symbols
    self.pairs_data: Dict[str, Dict[str, Any]] = {}

    for symbol in symbols:
      pair_data = self._initialize_pair_data(symbol, initial_data.get(symbol, {}))

      # Загружаем исторические 5м свечи если они есть
      if historical_5m_data and symbol in historical_5m_data:
        for candle in historical_5m_data[symbol]:
          pair_data["candles_5m"].append(candle)

        # ДОБАВИТЬ ОТЛАДКУ:
        volumes = [c.get('volume', 0) for c in historical_5m_data[symbol][:5]]
        log.debug(f"Загружено {len(historical_5m_data[symbol])} 5м свечей для {symbol}, первые 5 объемов: {volumes}")

      self.pairs_data[symbol] = pair_data

    log.info(f"Процессор данных инициализирован для {len(symbols)} пар.")

  def _initialize_pair_data(self, symbol: str, initial: Dict[str, Any]) -> Dict[str, Any]:
    return {
      "symbol": symbol,
      "last_price": initial.get('last_price', 0.0),
      "candles_1m": deque(maxlen=300),
      "candles_5m": deque(maxlen=300),  # Добавить для 5м данных
      "last_update_ts": 0,
      "metrics": {
        "change_1m": None, "change_2m": None, "change_3m": None, "change_5m": None,
        "change_15m": None, "change_1h": None,
        "change_4h": initial.get('change_4h'),
        "change_24h": initial.get('change_24h'),
        "change_1h_for_24h_display": None,
        "volume_change_pct": initial.get('volume_change_pct', 0),  # Новое поле
        "rsi_14": None,
        "atr_14": None,
        "is_above_ema200": None,
        "rsi_trend": 0,
        "psar_trend": 0,
        "aroon_trend": 0,
        "reversal_signal": 0,
      }
    }

  def update_with_tick(self, tick_data: Dict[str, Any]):
    """ Обновляет состояние пары на основе нового тика от WebSocket. """
    try:
      symbol = tick_data.get("symbol")
      if not symbol or symbol not in self.pairs_data:
        return

      pair = self.pairs_data[symbol]

      if 'last_price' in tick_data:
        price = float(tick_data["last_price"])
        pair["last_price"] = price
        pair["last_update_ts"] = time.time()
        self._update_candle(pair, price, pair["last_update_ts"])

      # Обновляем 24h% если он пришел в сообщении
      if 'price24hPcnt' in tick_data:
        pair["metrics"]["change_24h"] = float(tick_data['price24hPcnt']) * 100


    except (ValueError, TypeError, KeyError) as e:
      log.warning(f"Некорректный формат тика: {tick_data}. Ошибка: {e}")

  def _update_candle(self, pair: Dict, price: float, ts: float):
    """Обновляет текущие минутную и 5-минутную свечи."""
    current_minute_ts = int(ts // 60) * 60
    current_5min_ts = int(ts // 300) * 300

    # Обновление 1-минутных свечей
    if not pair["candles_1m"] or pair["candles_1m"][-1]["timestamp"] != current_minute_ts:
      new_candle = {
        "timestamp": current_minute_ts,
        "open": price, "high": price, "low": price, "close": price
      }
      pair["candles_1m"].append(new_candle)
    else:
      last_candle = pair["candles_1m"][-1]
      last_candle["high"] = max(last_candle["high"], price)
      last_candle["low"] = min(last_candle["low"], price)
      last_candle["close"] = price

    # Обновление 5-минутных свечей (ИСПРАВЛЕНО)
    if not pair["candles_5m"] or pair["candles_5m"][-1]["timestamp"] != current_5min_ts:
      new_candle_5m = {
        "timestamp": current_5min_ts,
        "open": price, "high": price, "low": price, "close": price,
        "volume": 0.0  # ДОБАВЛЕНО: начальный объем
      }
      pair["candles_5m"].append(new_candle_5m)
    else:
      last_candle_5m = pair["candles_5m"][-1]
      last_candle_5m["high"] = max(last_candle_5m["high"], price)
      last_candle_5m["low"] = min(last_candle_5m["low"], price)
      last_candle_5m["close"] = price
      # Объем не обновляется через WebSocket, используем исторические данные

  def calculate_all_metrics(self):
    """ Главный метод для вычислений. """
    for symbol, pair in self.pairs_data.items():
      try:
        self._calculate_metrics_for_pair(pair)
      except Exception as e:
        log.error(f"Ошибка при расчете метрик для {symbol}: {e}", exc_info=False)

  def _calculate_metrics_for_pair(self, pair: Dict[str, Any]):
    """Вычисляет метрики для одной конкретной пары."""
    candles_1m = list(pair["candles_1m"])
    candles_5m = list(pair["candles_5m"])

    current_price = pair["last_price"]

    # 1. Расчет процентных изменений на 1м данных (если есть)
    if len(candles_1m) >= 2:
      periods = {"1m": 1, "2m": 2, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
      for key, p_len in periods.items():
        if len(candles_1m) > p_len:
          open_price = candles_1m[-p_len - 1]['open']
          change = ((current_price - open_price) / open_price) * 100
          pair["metrics"][f"change_{key}"] = round(change, 4)

      # Расчет изменения за последний час для отображения в колонке 24h
      if len(candles_1m) > 60:
        price_1h_ago = candles_1m[-61]['open']
        change_1h = ((current_price - price_1h_ago) / price_1h_ago) * 100
        pair["metrics"]["change_1h_for_24h_display"] = round(change_1h, 2)

    # 2. Расчет индикаторов на 5-минутном ТФ
    if len(candles_5m) < 50:
      return

    try:
      df_5m = pd.DataFrame(candles_5m)
      df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'], unit='s')
      df_5m.set_index('timestamp', inplace=True)

      # --- ДОБАВЛЯЕМ ЛОГИКУ НОВОГО ИНДИКАТОРА ---

      # Вычисляем все необходимые индикаторы
      ema9 = ta.ema(df_5m['close'], length=9)
      ema50 = ta.ema(df_5m['close'], length=50)
      rsi14 = ta.rsi(df_5m['close'], length=14)
      volume_sma20 = ta.sma(df_5m['volume'], length=20)

      # Обнуляем сигнал по умолчанию
      pair["metrics"]["reversal_signal"] = 0

      # Проверяем, что все данные рассчитались
      if ema9 is not None and ema50 is not None and rsi14 is not None and volume_sma20 is not None:
        # Получаем последние и предпоследние значения для анализа пересечений
        last = df_5m.iloc[-1]
        prev = df_5m.iloc[-2]

        last_close = last['close']
        prev_close = prev['close']

        last_ema9 = ema9.iloc[-1]
        prev_ema9 = ema9.iloc[-2]

        last_ema50 = ema50.iloc[-1]

        last_rsi = rsi14.iloc[-1]
        prev_rsi = rsi14.iloc[-2]

        last_volume = last['volume']
        last_volume_sma = volume_sma20.iloc[-1]

        # Условие 1: Находимся в нисходящем тренде
        is_downtrend = last_close < last_ema50

        # Условие 2: RSI вышел из зоны перепроданности
        rsi_crossed_up_30 = prev_rsi < 30 and last_rsi >= 30

        # Условие 3: Цена пересекла быструю EMA вверх
        price_crossed_ema9 = prev_close < prev_ema9 and last_close > last_ema9

        # Условие 4 (опционально): Объем выше среднего
        is_volume_spike = last_volume > (last_volume_sma * 1.5)

        # Собираем все условия вместе
        if is_downtrend and rsi_crossed_up_30 and price_crossed_ema9 and is_volume_spike:
          pair["metrics"]["reversal_signal"] = 1  # 1 - сигнал на покупку (разворот вверх)

      # RSI на 5м данных
      rsi = ta.rsi(df_5m['close'], length=14)
      if rsi is not None and not rsi.empty and not pd.isna(rsi.iloc[-1]):
        # pair["metrics"]["rsi_14"] = round(rsi.iloc[-1], 2)
        pair["metrics"]["rsi_14"] = float(rsi.iloc[-1])

      # ATR на 5м данных
      atr = ta.atr(df_5m['high'], df_5m['low'], df_5m['close'], length=14)
      if atr is not None and not atr.empty and not pd.isna(atr.iloc[-1]):
        pair["metrics"]["atr_14"] = round(atr.iloc[-1], 6)

      # PSAR на 5м данных
      psar = ta.psar(df_5m['high'], df_5m['low'])
      if psar is not None and not psar.empty:
        psar_col = 'PSARl_0.02_0.2' if 'PSARl_0.02_0.2' in psar.columns else 'PSARs_0.02_0.2'
        if psar_col in psar.columns and not pd.isna(psar.iloc[-1][psar_col]):
          last_psar = psar.iloc[-1][psar_col]
          pair["metrics"]["psar_trend"] = 1 if current_price > last_psar else -1

      # Aroon на 5м данных
      aroon = ta.aroon(df_5m['high'], df_5m['low'], length=14)
      if aroon is not None and not aroon.empty:
        aroon_up = aroon.iloc[-1]['AROONU_14']
        aroon_down = aroon.iloc[-1]['AROOND_14']
        if not pd.isna(aroon_up) and not pd.isna(aroon_down):
          pair["metrics"]["aroon_trend"] = 1 if aroon_up > aroon_down else -1

      # EMA на 5м данных (используем 40-период вместо 200 для 5м)
      ema40 = ta.ema(df_5m['close'], length=40)
      if ema40 is not None and not ema40.empty and not pd.isna(ema40.iloc[-1]):
        ema_value = float(ema40.iloc[-1])  # ДОБАВЛЕНО: float()
        pair["metrics"]["is_above_ema200"] = bool(current_price > ema_value)  # ДОБАВЛЕНО: bool()

      # RSI Trend на 5м данных
      self._calculate_rsitrend(pair, df_5m)

      # 3. Расчет динамики объема (ИСПРАВЛЕННАЯ ВЕРСИЯ)
      try:
        if len(candles_5m) >= 20:
          # Получаем объемы последних 20 свечей
          volumes = []
          for candle in candles_5m[-20:]:
            vol = candle.get('volume', 0)
            if vol is not None and vol > 0:
              volumes.append(float(vol))

          if len(volumes) >= 10:
            # Текущий объем (последняя свеча)
            current_volume = volumes[-1]
            # Средний объем за предыдущие свечи
            avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[0]

            if avg_volume > 0:
              volume_change = ((current_volume - avg_volume) / avg_volume) * 100
              pair["metrics"]["volume_change_pct"] = round(float(volume_change), 2)

            else:
              pair["metrics"]["volume_change_pct"] = 0.0
          else:
            pair["metrics"]["volume_change_pct"] = 0.0
        else:
          pair["metrics"]["volume_change_pct"] = 0.0

      except Exception as e:
        log.warning(f"Ошибка при расчете объема для {pair['symbol']}: {e}")
        pair["metrics"]["volume_change_pct"] = 0.0

    except Exception as e:
      log.warning(f"Ошибка при расчете индикаторов для {pair['symbol']}: {e}")

  def get_screener_data(self) -> Dict[str, Any]:
    pairs_for_frontend = []
    for pair_data in self.pairs_data.values():
      try:
        data_copy = copy.deepcopy(pair_data)
        if 'candles_1m' in data_copy:
          del data_copy['candles_1m']
        if 'candles_5m' in data_copy:
          del data_copy['candles_5m']

        # ИСПРАВЛЕНО: Конвертируем все значения в JSON-совместимые типы
        metrics = data_copy.get('metrics', {})
        for key, value in metrics.items():
          if isinstance(value, bool):
            metrics[key] = value  # bool остается bool - это нормально для JSON
          elif isinstance(value, (int, float)):
            metrics[key] = float(value) if value is not None else None
          elif value is None:
            metrics[key] = None
          else:
            metrics[key] = str(value)  # Конвертируем остальное в строку

        # Конвертируем другие поля
        data_copy['last_price'] = float(data_copy.get('last_price', 0.0))
        data_copy['last_update_ts'] = float(data_copy.get('last_update_ts', 0.0))

        pairs_for_frontend.append(data_copy)

      except Exception as e:
        log.error(f"Ошибка при подготовке данных для {pair_data.get('symbol', 'Unknown')}: {e}")
        continue

    result = {"timestamp": time.time(), "pairs": pairs_for_frontend}
    log.debug(f"Подготовлено {len(pairs_for_frontend)} пар для отправки.")
    return result

  def _calculate_rsitrend(self, pair: Dict, df: pd.DataFrame):
    """Рассчитывает RSI Trend индикатор на основе HMA."""
    if len(df) < 40:
      pair["metrics"]["rsi_trend"] = 0
      return

    try:
      # Используем HMA (Hull Moving Average) для определения тренда
      bbmc = ta.hma(df['close'], length=30)
      if bbmc is None or len(bbmc) < 3:
        pair["metrics"]["rsi_trend"] = 0
        return

      # Получаем предыдущие значения HMA
      prev_bbmc = ta.hma(df['close'].iloc[:-1], length=30)
      if prev_bbmc is None or len(prev_bbmc) < 3:
        pair["metrics"]["rsi_trend"] = 0
        return

      # Проверяем пересечения для определения сигналов
      current_hma = bbmc.iloc[-1]
      prev_hma = bbmc.iloc[-3] if len(bbmc) >= 3 else bbmc.iloc[-1]
      prev_current_hma = prev_bbmc.iloc[-1] if len(prev_bbmc) >= 1 else current_hma
      prev_prev_hma = prev_bbmc.iloc[-3] if len(prev_bbmc) >= 3 else prev_current_hma

      # Проверяем на NaN значения
      if pd.isna(current_hma) or pd.isna(prev_hma) or pd.isna(prev_current_hma) or pd.isna(prev_prev_hma):
        pair["metrics"]["rsi_trend"] = 0
        return

      # Логика определения сигналов:
      # Buy signal: текущий HMA выше чем 3 периода назад И предыдущий HMA был ниже
      if (current_hma > prev_hma) and (prev_current_hma < prev_prev_hma):
        pair["metrics"]["rsi_trend"] = 1  # Buy
      # Sell signal: текущий HMA ниже чем 3 периода назад И предыдущий HMA был выше
      elif (current_hma < prev_hma) and (prev_current_hma > prev_prev_hma):
        pair["metrics"]["rsi_trend"] = -1  # Sell
      else:
        pair["metrics"]["rsi_trend"] = 0  # Neutral

    except Exception as e:
      log.warning(f"Ошибка при расчете RSI Trend для {pair['symbol']}: {e}")
      pair["metrics"]["rsi_trend"] = 0
