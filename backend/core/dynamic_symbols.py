# backend/core/dynamic_symbols.py
"""
Управление динамическим списком торговых пар.
Логика отбора: объем > 4M → топ-200 по объему → топ-40 растущих + топ-20 падающих.
"""

from typing import List, Dict
from backend.core.logger import get_logger

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