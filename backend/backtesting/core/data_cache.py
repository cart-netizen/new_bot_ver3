"""
Data Cache - кэширование сгенерированных OrderBook и Market Trades данных.

Проблема: Генерация orderbook snapshots и market trades занимает много времени.
Решение: Кэшировать сгенерированные данные на диске для повторного использования.

Использование:
    cache = DataCache()

    # Попытка загрузить из кэша
    orderbooks = cache.load_orderbooks(symbol, start_date, end_date, interval)
    if not orderbooks:
        # Генерируем если нет в кэше
        orderbooks = handler.generate_orderbook_sequence(candles, symbol)
        cache.save_orderbooks(orderbooks, symbol, start_date, end_date, interval)
"""

import hashlib
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from backend.models.orderbook import OrderBookSnapshot
from backend.models.market_data import MarketTrade

logger = logging.getLogger(__name__)


class DataCache:
    """
    Кэш для сгенерированных данных бэктестинга.

    Хранит:
    - OrderBook snapshots
    - Market Trades
    - Метаданные (дата генерации, параметры)
    """

    def __init__(self, cache_dir: str = ".backtesting_cache"):
        """
        Args:
            cache_dir: Директория для хранения кэша
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Поддиректории
        self.orderbook_dir = self.cache_dir / "orderbooks"
        self.trades_dir = self.cache_dir / "market_trades"

        self.orderbook_dir.mkdir(exist_ok=True)
        self.trades_dir.mkdir(exist_ok=True)

        logger.info(f"DataCache инициализирован: {self.cache_dir}")

    def _generate_cache_key(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        config_hash: Optional[str] = None
    ) -> str:
        """
        Генерирует уникальный ключ кэша.

        Args:
            symbol: Торговая пара
            start_date: Начальная дата
            end_date: Конечная дата
            interval: Интервал свечей
            config_hash: Hash конфигурации (для orderbook/trades параметров)

        Returns:
            str: Хэш-ключ для кэша
        """
        key_parts = [
            symbol,
            start_date.strftime("%Y%m%d"),
            end_date.strftime("%Y%m%d"),
            interval
        ]

        if config_hash:
            key_parts.append(config_hash)

        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def _generate_config_hash(self, config_params: Dict[str, Any]) -> str:
        """Генерирует hash из параметров конфигурации."""
        config_str = json.dumps(config_params, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def save_orderbooks(
        self,
        orderbooks: List[OrderBookSnapshot],
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        config_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Сохранить orderbook snapshots в кэш.

        Args:
            orderbooks: Список OrderBookSnapshot
            symbol: Торговая пара
            start_date: Начальная дата
            end_date: Конечная дата
            interval: Интервал свечей
            config_params: Параметры конфигурации (num_levels, spread_bps, etc.)

        Returns:
            bool: True если успешно сохранено
        """
        try:
            config_hash = self._generate_config_hash(config_params) if config_params else None
            cache_key = self._generate_cache_key(symbol, start_date, end_date, interval, config_hash)

            cache_file = self.orderbook_dir / f"{cache_key}.pkl"
            metadata_file = self.orderbook_dir / f"{cache_key}.meta.json"

            # Сохраняем данные
            with open(cache_file, 'wb') as f:
                pickle.dump(orderbooks, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Сохраняем метаданные
            metadata = {
                "symbol": symbol,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "interval": interval,
                "count": len(orderbooks),
                "created_at": datetime.now().isoformat(),
                "config_params": config_params or {}
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"✅ Orderbooks cached: {symbol} {start_date.date()} -> {end_date.date()}, "
                f"count={len(orderbooks)}, key={cache_key}"
            )

            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения orderbooks в кэш: {e}")
            return False

    def load_orderbooks(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        config_params: Optional[Dict[str, Any]] = None
    ) -> Optional[List[OrderBookSnapshot]]:
        """
        Загрузить orderbook snapshots из кэша.

        Args:
            symbol: Торговая пара
            start_date: Начальная дата
            end_date: Конечная дата
            interval: Интервал свечей
            config_params: Параметры конфигурации

        Returns:
            List[OrderBookSnapshot] или None если не найдено
        """
        try:
            config_hash = self._generate_config_hash(config_params) if config_params else None
            cache_key = self._generate_cache_key(symbol, start_date, end_date, interval, config_hash)

            cache_file = self.orderbook_dir / f"{cache_key}.pkl"

            if not cache_file.exists():
                logger.debug(f"Orderbooks cache miss: key={cache_key}")
                return None

            # Загружаем данные
            with open(cache_file, 'rb') as f:
                orderbooks = pickle.load(f)

            logger.info(
                f"✅ Orderbooks loaded from cache: {symbol} {start_date.date()} -> {end_date.date()}, "
                f"count={len(orderbooks)}"
            )

            return orderbooks

        except Exception as e:
            logger.error(f"Ошибка загрузки orderbooks из кэша: {e}")
            return None

    def save_market_trades(
        self,
        trades: List[MarketTrade],
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        config_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Сохранить market trades в кэш.

        Args:
            trades: Список MarketTrade
            symbol: Торговая пара
            start_date: Начальная дата
            end_date: Конечная дата
            interval: Интервал свечей
            config_params: Параметры конфигурации (trades_per_volume_unit, etc.)

        Returns:
            bool: True если успешно сохранено
        """
        try:
            config_hash = self._generate_config_hash(config_params) if config_params else None
            cache_key = self._generate_cache_key(symbol, start_date, end_date, interval, config_hash)

            cache_file = self.trades_dir / f"{cache_key}.pkl"
            metadata_file = self.trades_dir / f"{cache_key}.meta.json"

            # Сохраняем данные
            with open(cache_file, 'wb') as f:
                pickle.dump(trades, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Сохраняем метаданные
            metadata = {
                "symbol": symbol,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "interval": interval,
                "count": len(trades),
                "created_at": datetime.now().isoformat(),
                "config_params": config_params or {}
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"✅ Market trades cached: {symbol} {start_date.date()} -> {end_date.date()}, "
                f"count={len(trades)}, key={cache_key}"
            )

            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения market trades в кэш: {e}")
            return False

    def load_market_trades(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        config_params: Optional[Dict[str, Any]] = None
    ) -> Optional[List[MarketTrade]]:
        """
        Загрузить market trades из кэша.

        Args:
            symbol: Торговая пара
            start_date: Начальная дата
            end_date: Конечная дата
            interval: Интервал свечей
            config_params: Параметры конфигурации

        Returns:
            List[MarketTrade] или None если не найдено
        """
        try:
            config_hash = self._generate_config_hash(config_params) if config_params else None
            cache_key = self._generate_cache_key(symbol, start_date, end_date, interval, config_hash)

            cache_file = self.trades_dir / f"{cache_key}.pkl"

            if not cache_file.exists():
                logger.debug(f"Market trades cache miss: key={cache_key}")
                return None

            # Загружаем данные
            with open(cache_file, 'rb') as f:
                trades = pickle.load(f)

            logger.info(
                f"✅ Market trades loaded from cache: {symbol} {start_date.date()} -> {end_date.date()}, "
                f"count={len(trades)}"
            )

            return trades

        except Exception as e:
            logger.error(f"Ошибка загрузки market trades из кэша: {e}")
            return None

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Очистить кэш.

        Args:
            older_than_days: Удалить файлы старше N дней (None = удалить все)
        """
        import time

        now = time.time()
        cutoff_time = now - (older_than_days * 24 * 3600) if older_than_days else 0

        deleted_count = 0

        for cache_file in list(self.orderbook_dir.glob("*.pkl")) + list(self.trades_dir.glob("*.pkl")):
            if cache_file.stat().st_mtime < cutoff_time:
                # Удаляем файл и метаданные
                cache_file.unlink()
                meta_file = cache_file.with_suffix(".meta.json")
                if meta_file.exists():
                    meta_file.unlink()
                deleted_count += 1

        logger.info(f"✅ Cache cleared: {deleted_count} files deleted")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получить статистику кэша."""
        orderbook_files = list(self.orderbook_dir.glob("*.pkl"))
        trades_files = list(self.trades_dir.glob("*.pkl"))

        total_size = sum(f.stat().st_size for f in orderbook_files + trades_files)

        return {
            "orderbook_count": len(orderbook_files),
            "market_trades_count": len(trades_files),
            "total_files": len(orderbook_files) + len(trades_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }
