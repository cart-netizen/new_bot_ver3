#!/usr/bin/env python3
"""
Скрипт для скачивания исторических данных GMTUSDT за 24 часа.
Сохраняет данные в CSV для анализа пропущенных сигналов.
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
import json
import os

SYMBOL = "GMTUSDT"
BASE_URL = "https://api.bybit.com"

async def fetch_klines(session, symbol: str, interval: str, limit: int = 200):
    """Скачать свечи с Bybit."""
    url = f"{BASE_URL}/v5/market/kline"

    # Интервалы Bybit: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
    interval_map = {
        "1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"
    }

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval_map.get(interval, interval),
        "limit": limit
    }

    async with session.get(url, params=params) as resp:
        data = await resp.json()
        if data.get("retCode") == 0:
            return data.get("result", {}).get("list", [])
        return []

async def fetch_orderbook(session, symbol: str):
    """Скачать текущий orderbook."""
    url = f"{BASE_URL}/v5/market/orderbook"
    params = {"category": "linear", "symbol": symbol, "limit": 50}

    async with session.get(url, params=params) as resp:
        data = await resp.json()
        if data.get("retCode") == 0:
            return data.get("result", {})
        return {}

async def fetch_recent_trades(session, symbol: str, limit: int = 1000):
    """Скачать последние сделки."""
    url = f"{BASE_URL}/v5/market/recent-trade"
    params = {"category": "linear", "symbol": symbol, "limit": limit}

    async with session.get(url, params=params) as resp:
        data = await resp.json()
        if data.get("retCode") == 0:
            return data.get("result", {}).get("list", [])
        return []

async def main():
    print(f"=== Скачивание данных {SYMBOL} ===")
    print(f"Время: {datetime.now()}")

    async with aiohttp.ClientSession() as session:
        # 1. Скачать свечи по разным таймфреймам
        timeframes = ["1m", "5m", "15m", "1h", "4h"]
        all_candles = {}

        for tf in timeframes:
            # Для 1m берём 1000 свечей (~16 часов), для остальных адаптивно
            limit = {"1m": 1000, "5m": 288, "15m": 96, "1h": 48, "4h": 12}.get(tf, 200)
            candles = await fetch_klines(session, SYMBOL, tf, limit)
            all_candles[tf] = candles
            print(f"  {tf}: {len(candles)} свечей")

        # 2. Скачать orderbook
        orderbook = await fetch_orderbook(session, SYMBOL)
        print(f"  Orderbook: {len(orderbook.get('a', []))} asks, {len(orderbook.get('b', []))} bids")

        # 3. Скачать последние сделки
        trades = await fetch_recent_trades(session, SYMBOL, 1000)
        print(f"  Trades: {len(trades)} сделок")

    # Конвертируем в DataFrame и сохраняем
    output_dir = r"/analyse"

    # Создаём папку для результатов если нет
    os.makedirs(output_dir, exist_ok=True)

    for tf, candles in all_candles.items():
        if not candles:
            continue

        # Bybit формат: [timestamp, open, high, low, close, volume, turnover]
        df = pd.DataFrame(candles, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])

        # Конвертация типов
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)

        # Сортировка по времени (старые → новые)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Добавляем расчётные колонки
        df["price_change_pct"] = ((df["close"] - df["open"]) / df["open"] * 100).round(2)
        df["volume_ma20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = (df["volume"] / df["volume_ma20"]).round(2)
        df["high_low_range_pct"] = ((df["high"] - df["low"]) / df["low"] * 100).round(2)

        # ROC (Rate of Change)
        df["roc_5"] = ((df["close"] - df["close"].shift(5)) / df["close"].shift(5) * 100).round(2)
        df["roc_10"] = ((df["close"] - df["close"].shift(10)) / df["close"].shift(10) * 100).round(2)

        # Volume acceleration
        df["volume_change"] = df["volume"].pct_change().round(2)

        # Cumulative price change from start
        first_close = df["close"].iloc[0]
        df["cumulative_change_pct"] = ((df["close"] - first_close) / first_close * 100).round(2)

        filename = f"{output_dir}/gmtusdt_{tf}_analysis.csv"
        df.to_csv(filename, index=False)
        print(f"  Сохранено: {filename}")

    # Сохраняем сводку
    summary = {
        "symbol": SYMBOL,
        "downloaded_at": datetime.now().isoformat(),
        "timeframes": list(all_candles.keys()),
        "candle_counts": {tf: len(c) for tf, c in all_candles.items()},
        "trades_count": len(trades),
        "orderbook_depth": {
            "asks": len(orderbook.get("a", [])),
            "bids": len(orderbook.get("b", []))
        }
    }

    with open(f"{output_dir}/gmtusdt_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Готово! ===")
    print(f"Файлы сохранены в {output_dir}/gmtusdt_*.csv")

    # Выводим краткую статистику по часовым свечам
    if "1h" in all_candles and all_candles["1h"]:
        df_1h = pd.DataFrame(all_candles["1h"], columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df_1h[col] = df_1h[col].astype(float)
        df_1h = df_1h.sort_values("timestamp")

        print(f"\n=== Статистика 1H (последние 48 часов) ===")
        print(f"Мин цена: ${df_1h['low'].min():.4f}")
        print(f"Макс цена: ${df_1h['high'].max():.4f}")
        print(f"Изменение: {((df_1h['close'].iloc[-1] - df_1h['open'].iloc[0]) / df_1h['open'].iloc[0] * 100):.2f}%")
        print(f"Средний объём: {df_1h['volume'].mean():,.0f}")
        print(f"Макс объём: {df_1h['volume'].max():,.0f}")
        print(f"Макс volume ratio: {(df_1h['volume'].max() / df_1h['volume'].mean()):.2f}x")

if __name__ == "__main__":
    asyncio.run(main())
