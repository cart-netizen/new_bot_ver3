# backend/core/dynamic_symbols.py
"""
Управление динамическим списком торговых пар.
Логика отбора: объем > 4M → топ-200 по объему → топ-40 растущих + топ-20 падающих.

Поддерживает два режима:
1. С скринером: получает данные от ScreenerManager
2. Standalone: сам получает тикеры от Bybit REST API (без скринера)
"""

import aiohttp
import asyncio
from typing import List, Dict, Optional
from backend.core.logger import get_logger
from backend.config import settings

logger = get_logger(__name__)


class DynamicSymbolsManager:
    """Управление динамическим списком торговых пар для мониторинга."""

    def __init__(
        self,
        min_volume: float = 4_000_000,
        max_volume_pairs: int = 200,
        top_gainers: int = 40,
        top_losers: int = 20
    ):
        """
        Инициализация менеджера.

        Args:
            min_volume: Минимальный объем за 24ч (USDT)
            max_volume_pairs: Максимум пар после фильтра по объему
            top_gainers: Количество растущих пар
            top_losers: Количество падающих пар
        """
        self.min_volume = min_volume
        self.max_volume_pairs = max_volume_pairs
        self.top_gainers = top_gainers
        self.top_losers = top_losers
        self.current_symbols: List[str] = []

        logger.info(
            f"DynamicSymbolsManager инициализирован: "
            f"min_volume={min_volume:,.0f}, "
            f"max_volume={max_volume_pairs}, "
            f"gainers={top_gainers}, losers={top_losers}"
        )

    def select_symbols(self, pairs: List[Dict]) -> List[str]:
        """
        Отбор торговых пар по критериям.

        Логика:
        1. Фильтр: объем > min_volume
        2. Сортировка по объему → топ-200
        3. Сортировка по изменению цены за 24ч
        4. Топ-40 растущих + топ-20 падающих = 60 пар

        Args:
            pairs: Список пар от screener

        Returns:
            List[str]: Список символов для мониторинга
        """
        if not pairs:
            logger.warning("Пустой список пар от screener")
            return []

        # Шаг 1: Фильтр по объему
        filtered = [
            p for p in pairs
            if p.get('volume_24h', 0) >= self.min_volume
        ]
        logger.info(
            f"Шаг 1: Фильтр по объему > {self.min_volume:,.0f} → "
            f"{len(filtered)} пар"
        )

        if not filtered:
            logger.warning("Нет пар с достаточным объемом")
            return []

        # Шаг 2: Топ по объему
        by_volume = sorted(
            filtered,
            key=lambda x: x.get('volume_24h', 0),
            reverse=True
        )[:self.max_volume_pairs]

        logger.info(
            f"Шаг 2: Топ-{self.max_volume_pairs} по объему → "
            f"{len(by_volume)} пар"
        )

        # Шаг 3: Сортировка по изменению цены
        by_change = sorted(
            by_volume,
            key=lambda x: x.get('price_change_24h_percent', 0),
            reverse=True
        )

        # Шаг 4: Топ-40 растущих
        gainers = by_change[:self.top_gainers]

        # Шаг 5: Топ-20 падающих (с конца)
        losers = by_change[-self.top_losers:]

        # Объединяем
        selected_pairs = gainers + losers
        symbols = [p['symbol'] for p in selected_pairs]

        # Убираем дубликаты (на случай если пар < 60)
        symbols = list(dict.fromkeys(symbols))

        logger.info(
            f"✓ Отобрано {len(symbols)} пар: "
            f"{len(gainers)} растущих + {len(losers)} падающих"
        )

        # Логируем топ-5 каждой категории
        if gainers:
            top5_gainers = [
                f"{p['symbol']} (+{p.get('price_change_24h_percent', 0):.1f}%)"
                for p in gainers[:5]
            ]
            logger.info(f"Топ-5 растущих: {', '.join(top5_gainers)}")

        if losers:
            top5_losers = [
                f"{p['symbol']} ({p.get('price_change_24h_percent', 0):.1f}%)"
                for p in losers[:5]
            ]
            logger.info(f"Топ-5 падающих: {', '.join(top5_losers)}")

        self.current_symbols = symbols
        return symbols

    def get_changes(self, new_symbols: List[str]) -> Dict[str, List[str]]:
        """
        Определить изменения в списке пар.

        Args:
            new_symbols: Новый список символов

        Returns:
            Dict с added и removed символами
        """
        current_set = set(self.current_symbols)
        new_set = set(new_symbols)

        added = list(new_set - current_set)
        removed = list(current_set - new_set)

        return {
            'added': added,
            'removed': removed
        }

    # ========== STANDALONE MODE - работа без скринера ==========

    async def fetch_tickers_standalone(self) -> List[Dict]:
        """
        Получение тикеров напрямую от Bybit REST API.

        Standalone режим: позволяет динамический выбор пар без запуска полного скринера.
        Делает один HTTP запрос для получения всех USDT фьючерсов.

        Returns:
            List[Dict]: Список пар с данными для select_symbols()
                - symbol: str
                - volume_24h: float (turnover24h)
                - price_change_24h_percent: float
                - last_price: float
        """
        api_url = (
            "https://api-testnet.bybit.com"
            if settings.BYBIT_MODE == "testnet"
            else "https://api.bybit.com"
        )

        pairs = []

        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{api_url}/v5/market/tickers"
                params = {"category": "linear"}  # Все USDT perpetual фьючерсы

                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"[Standalone] API вернул статус {response.status}")
                        return []

                    data = await response.json()

                    # MEMORY FIX: Явно освобождаем response
                    await response.release()

                    if data.get("retCode") != 0:
                        logger.error(f"[Standalone] API ошибка: {data.get('retMsg')}")
                        return []

                    tickers = data.get("result", {}).get("list", [])

                    if not tickers:
                        logger.warning("[Standalone] API вернул пустой список тикеров")
                        return []

                    # Обрабатываем тикеры
                    for ticker in tickers:
                        symbol = ticker.get("symbol", "")

                        # Только USDT пары
                        if not symbol.endswith("USDT"):
                            continue

                        try:
                            volume_24h = float(ticker.get("turnover24h", 0))
                            price_change_pct = float(ticker.get("price24hPcnt", 0)) * 100
                            last_price = float(ticker.get("lastPrice", 0))

                            pairs.append({
                                "symbol": symbol,
                                "volume_24h": volume_24h,
                                "price_change_24h_percent": price_change_pct,
                                "last_price": last_price
                            })
                        except (ValueError, TypeError) as e:
                            logger.debug(f"[Standalone] Ошибка парсинга {symbol}: {e}")
                            continue

            logger.info(f"[Standalone] Получено {len(pairs)} USDT пар от Bybit API")
            return pairs

        except asyncio.TimeoutError:
            logger.error("[Standalone] Таймаут при запросе к Bybit API")
            return []
        except aiohttp.ClientError as e:
            logger.error(f"[Standalone] Ошибка HTTP: {e}")
            return []
        except Exception as e:
            logger.error(f"[Standalone] Ошибка получения тикеров: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return []

    async def select_symbols_standalone(self) -> List[str]:
        """
        Полный цикл выбора пар в standalone режиме.

        Комбинирует fetch_tickers_standalone() + select_symbols().
        Используется когда скринер отключен.

        Returns:
            List[str]: Отобранные символы
        """
        logger.info("[Standalone] Запуск автономного выбора пар...")

        # Получаем тикеры
        pairs = await self.fetch_tickers_standalone()

        if not pairs:
            logger.warning("[Standalone] Не удалось получить тикеры, возвращаем текущий список")
            return self.current_symbols

        # Отбираем по критериям
        symbols = self.select_symbols(pairs)

        logger.info(f"[Standalone] ✓ Отобрано {len(symbols)} пар")
        return symbols