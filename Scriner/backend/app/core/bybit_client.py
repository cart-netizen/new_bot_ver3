# backend/app/core/bybit_client.py

import httpx
from typing import List, Optional, Dict, Any

from ..config import settings
from .logger import log


class BybitRESTClient:
  """ Клиент для взаимодействия с Bybit REST API v5. """

  def __init__(self):
    self.api_url = settings.bybit_api_url
    self.min_volume = settings.min_daily_volume_usd
    self.top_n = settings.top_n_pairs
    log.info("Клиент Bybit REST v5 инициализирован.")

  async def get_5m_klines_for_symbols(self, symbols: List[str]) -> Dict[str, List[Dict]]:
    """Загружает 5-минутные свечи для всех символов."""
    result = {}

    for symbol in symbols:
      try:
        endpoint = "/v5/market/kline"
        url = f"{self.api_url}{endpoint}"
        # Загружаем 150 5-минутных свечей (12 часов)
        params = {"category": "linear", "symbol": symbol, "interval": "5", "limit": "150"}

        async with httpx.AsyncClient() as client:
          response = await client.get(url, params=params, timeout=10.0)
          response.raise_for_status()

        data = response.json()
        if data.get("retCode") == 0:
          klines = data.get("result", {}).get("list", [])
          # Преобразуем в нужный формат и сортируем по времени
          formatted_klines = []
          for k in reversed(klines):  # API возвращает в обратном порядке
            formatted_klines.append({
              "timestamp": int(k[0]) // 1000,  # в секундах
              "open": float(k[1]),
              "high": float(k[2]),
              "low": float(k[3]),
              "close": float(k[4]),
              "volume": float(k[5]) if len(k) > 5 else 0.0,  # ИСПРАВЛЕНО: проверяем наличие volume
            })
          result[symbol] = formatted_klines

      except Exception as e:
        log.error(f"Ошибка при загрузке 5м свечей для {symbol}: {e}")
        result[symbol] = []

    log.info(f"Загружены 5м свечи для {len(result)} символов.")
    return result

  async def get_top_volume_pairs(self) -> Optional[Dict[str, Any]]:
    """
    Получает топ-N пар и их начальные данные (цена, изменения 4h/24h).
    """
    endpoint = "/v5/market/tickers"
    url = f"{self.api_url}{endpoint}"
    params = {"category": "linear"}

    log.debug(f"Отправка запроса на эндпоинт v5: {url} с параметрами: {params}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()

        data = response.json()
        if data.get("retCode") != 0:
            log.error(f"Ошибка от API Bybit v5: {data.get('retMsg', 'Нет сообщения')}")
            return None

        all_tickers: List[Dict[str, Any]] = data.get("result", {}).get("list", [])
        if not all_tickers:
            log.warning("API Bybit v5 вернул пустой список тикеров.")
            return {}

        log.info(f"Получено {len(all_tickers)} тикеров с Bybit v5.")

        liquid_tickers = [
            ticker for ticker in all_tickers
            if ticker.get("turnover24h") and float(ticker["turnover24h"]) > self.min_volume
        ]
        log.info(f"Найдено {len(liquid_tickers)} пар с объемом > ${self.min_volume / 1_000_000:.1f}M.")

        if not liquid_tickers:
            return {"symbols": [], "initial_data": {}}

        sorted_tickers = sorted(
            liquid_tickers, key=lambda x: float(x["turnover24h"]), reverse=True
        )

        initial_data = {}
        top_symbols = []
        for ticker in sorted_tickers[:self.top_n]:
            symbol = ticker["symbol"]
            top_symbols.append(symbol)
            initial_data[symbol] = {
                'last_price': float(ticker.get('lastPrice', 0.0)),
                'change_24h': float(ticker.get('price24hPcnt', '0').replace('%', '')) * 100,
                'change_4h': float(ticker.get('price4hPcnt', '0').replace('%', '')) * 100,
                'volume_change_pct': 0.0,  # Заглушка для начала
            }

        log.success(f"Отобрано топ-{len(top_symbols)} пар для мониторинга.")
        return {"symbols": top_symbols, "initial_data": initial_data}

    except httpx.HTTPStatusError as e:
        log.error(f"Ошибка HTTP при запросе к Bybit v5: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        log.error(f"Непредвиденная ошибка при получении тикеров v5: {e}", exc_info=True)
        return None

  async def get_historical_klines(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
    """ Загружает исторические минутные свечи для графика. """
    endpoint = "/v5/market/kline"
    url = f"{self.api_url}{endpoint}"
    # Загружаем 240 свечей (4 часа)
    params = {"category": "linear", "symbol": symbol, "interval": "1", "limit": "240"}

    try:
      async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=10.0)
        response.raise_for_status()

      data = response.json()
      if data.get("retCode") != 0:
        log.error(f"Ошибка от API kline: {data.get('retMsg')}")
        return None

      klines = data.get("result", {}).get("list", [])
      # Форматируем для lightweight-charts
      formatted_klines = [
        {
          "time": int(k[0]) // 1000,  # в секундах
          "open": float(k[1]),
          "high": float(k[2]),
          "low": float(k[3]),
          "close": float(k[4]),
        }
        for k in klines
      ]
      # Сортируем по времени, так как API возвращает в обратном порядке
      return sorted(formatted_klines, key=lambda x: x['time'])

    except Exception as e:
      log.error(f"Ошибка при получении klines для {symbol}: {e}")
      return None

  async def get_volume_data(self, symbol: str) -> Optional[Dict[str, Any]]:
      """Загружает данные объема для расчета динамики."""
      endpoint = "/v5/market/kline"
      url = f"{self.api_url}{endpoint}"
      # Загружаем 288 5-минутных свечей (24 часа)
      params = {"category": "linear", "symbol": symbol, "interval": "5", "limit": "288"}

      try:
        async with httpx.AsyncClient() as client:
          response = await client.get(url, params=params, timeout=10.0)
          response.raise_for_status()

        data = response.json()
        if data.get("retCode") != 0:
          return None

        klines = data.get("result", {}).get("list", [])
        if not klines:
          return None

        # Возвращаем последнюю свечу и среднее за 24ч
        volumes = [float(k[5]) for k in klines]  # volume is index 5
        current_volume = volumes[0] if volumes else 0  # последняя свеча первая в списке
        avg_24h_volume = sum(volumes) / len(volumes) if volumes else 0

        return {
          "current_volume": current_volume,
          "avg_24h_volume": avg_24h_volume,
          "volume_change_pct": ((current_volume - avg_24h_volume) / avg_24h_volume * 100) if avg_24h_volume > 0 else 0
        }

      except Exception as e:
        log.error(f"Ошибка при получении данных объема для {symbol}: {e}")
        return None

  async def get_enhanced_tickers(self) -> Optional[Dict[str, Any]]:
    """Получает расширенные данные тикеров с 4h данными."""
    endpoint = "/v5/market/tickers"
    url = f"{self.api_url}{endpoint}"
    params = {"category": "linear"}

    try:
      async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=10.0)
        response.raise_for_status()

      data = response.json()
      if data.get("retCode") != 0:
        return None

      all_tickers = data.get("result", {}).get("list", [])
      liquid_tickers = [
        ticker for ticker in all_tickers
        if ticker.get("turnover24h") and float(ticker["turnover24h"]) > self.min_volume
      ]

      sorted_tickers = sorted(liquid_tickers, key=lambda x: float(x["turnover24h"]), reverse=True)

      initial_data = {}
      top_symbols = []

      for ticker in sorted_tickers[:self.top_n]:
        symbol = ticker["symbol"]
        top_symbols.append(symbol)

        # Получаем данные объема
        volume_data = await self.get_volume_data(symbol)

        initial_data[symbol] = {
          'last_price': float(ticker.get('lastPrice', 0.0)),
          'change_24h': float(ticker.get('price24hPcnt', '0').replace('%', '')) * 100,
          'change_4h': float(ticker.get('price4hPcnt', '0').replace('%', '')) * 100,
          'volume_change_pct': volume_data.get('volume_change_pct', 0) if volume_data else 0
        }

      return {"symbols": top_symbols, "initial_data": initial_data}

    except Exception as e:
      log.error(f"Ошибка при получении расширенных тикеров: {e}")
      return None
