"""
Market Cap Weighting для корреляционного анализа.

Учитывает доминирование крупных активов (BTC, ETH) в корреляционной структуре рынка.

Путь: backend/strategy/correlation/market_cap_weighting.py
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MarketCapInfo:
    """
    Информация о market cap актива.

    Attributes:
        symbol: Символ актива
        market_cap_usd: Market cap в USD
        dominance_percent: Процент доминирования в рынке
        tier: Tier актива (1 = major, 2 = mid, 3 = small)
    """
    symbol: str
    market_cap_usd: float
    dominance_percent: float
    tier: int


class MarketCapWeightingManager:
    """
    Менеджер взвешивания по market cap.

    Позволяет:
    - Придавать больший вес корреляциям с крупными активами
    - Использовать разные пороги для разных tiers
    - Учитывать системный риск от доминирующих активов
    """

    # Примерные market caps (можно обновлять динамически)
    DEFAULT_MARKET_CAPS = {
        "BTCUSDT": 1_000_000_000_000,  # $1T
        "ETHUSDT": 400_000_000_000,     # $400B
        "BNBUSDT": 80_000_000_000,      # $80B
        "SOLUSDT": 70_000_000_000,      # $70B
        "XRPUSDT": 60_000_000_000,      # $60B
        "ADAUSDT": 30_000_000_000,      # $30B
        "AVAXUSDT": 15_000_000_000,     # $15B
        "DOGEUSDT": 25_000_000_000,     # $25B
        "LINKUSDT": 10_000_000_000,     # $10B
        "MATICUSDT": 8_000_000_000,     # $8B
    }

    def __init__(
        self,
        market_caps: Optional[Dict[str, float]] = None,
        major_tier_threshold: float = 100_000_000_000,  # $100B
        mid_tier_threshold: float = 10_000_000_000       # $10B
    ):
        """
        Инициализация менеджера.

        Args:
            market_caps: Словарь {symbol: market_cap_usd}
            major_tier_threshold: Порог для major tier (>= этой суммы)
            mid_tier_threshold: Порог для mid tier
        """
        self.market_caps = market_caps or self.DEFAULT_MARKET_CAPS.copy()
        self.major_tier_threshold = major_tier_threshold
        self.mid_tier_threshold = mid_tier_threshold

        self._calculate_dominance()

        logger.info(
            f"MarketCapWeightingManager: {len(self.market_caps)} активов, "
            f"major_threshold=${major_tier_threshold/1e9:.1f}B"
        )

    def _calculate_dominance(self):
        """Рассчитывает процент доминирования для каждого актива."""
        total_market_cap = sum(self.market_caps.values())

        self.dominance = {
            symbol: (cap / total_market_cap * 100)
            for symbol, cap in self.market_caps.items()
        }

    def get_market_cap_info(self, symbol: str) -> Optional[MarketCapInfo]:
        """
        Получает информацию о market cap для символа.

        Args:
            symbol: Символ актива

        Returns:
            Optional[MarketCapInfo]: Информация или None если нет данных
        """
        if symbol not in self.market_caps:
            return None

        market_cap = self.market_caps[symbol]
        dominance = self.dominance.get(symbol, 0.0)

        # Определяем tier
        if market_cap >= self.major_tier_threshold:
            tier = 1  # Major (BTC, ETH, BNB)
        elif market_cap >= self.mid_tier_threshold:
            tier = 2  # Mid cap (SOL, ADA, AVAX)
        else:
            tier = 3  # Small cap

        return MarketCapInfo(
            symbol=symbol,
            market_cap_usd=market_cap,
            dominance_percent=dominance,
            tier=tier
        )

    def calculate_weighted_correlation(
        self,
        symbol_a: str,
        symbol_b: str,
        correlation: float,
        weighting_strategy: str = "dominance"
    ) -> float:
        """
        Рассчитывает взвешенную корреляцию с учетом market cap.

        Args:
            symbol_a: Первый символ
            symbol_b: Второй символ
            correlation: Базовая корреляция
            weighting_strategy: Стратегия взвешивания
                - "dominance": Вес по доминированию
                - "systemic_risk": Учет системного риска
                - "equal": Без взвешивания

        Returns:
            float: Взвешенная корреляция
        """
        if weighting_strategy == "equal":
            return correlation

        info_a = self.get_market_cap_info(symbol_a)
        info_b = self.get_market_cap_info(symbol_b)

        # Если нет данных - возвращаем как есть
        if not info_a or not info_b:
            return correlation

        if weighting_strategy == "dominance":
            # Увеличиваем вес корреляции с доминирующими активами
            max_dominance = max(info_a.dominance_percent, info_b.dominance_percent)

            # Вес от 1.0 (small cap) до 1.5 (major dominant)
            weight = 1.0 + (max_dominance / 100.0) * 0.5

            # Взвешиваем корреляцию (усиливаем сигнал)
            weighted = correlation * weight

            # Нормализуем обратно в [-1, 1]
            weighted = np.clip(weighted, -1.0, 1.0)

            return weighted

        elif weighting_strategy == "systemic_risk":
            # Если оба актива major - повышаем значимость корреляции
            if info_a.tier == 1 and info_b.tier == 1:
                # Major-major корреляция важнее
                weight = 1.3
            elif info_a.tier <= 2 and info_b.tier <= 2:
                # Mid-mid корреляция
                weight = 1.1
            else:
                # Small cap корреляция менее важна
                weight = 0.9

            weighted = correlation * weight
            weighted = np.clip(weighted, -1.0, 1.0)

            return weighted

        return correlation

    def get_adaptive_threshold(
        self,
        symbol_a: str,
        symbol_b: str,
        base_threshold: float = 0.7
    ) -> float:
        """
        Получает адаптивный порог корреляции в зависимости от market cap.

        Args:
            symbol_a: Первый символ
            symbol_b: Второй символ
            base_threshold: Базовый порог

        Returns:
            float: Адаптированный порог
        """
        info_a = self.get_market_cap_info(symbol_a)
        info_b = self.get_market_cap_info(symbol_b)

        if not info_a or not info_b:
            return base_threshold

        # Для major-major пар используем более строгий порог
        if info_a.tier == 1 and info_b.tier == 1:
            # BTC-ETH и подобные - более строгий порог
            return base_threshold - 0.05

        # Для small cap пар - более мягкий порог
        elif info_a.tier == 3 or info_b.tier == 3:
            return base_threshold + 0.05

        return base_threshold

    def identify_systemic_risk_groups(
        self,
        correlation_matrix: Dict[Tuple[str, str], float],
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        Идентифицирует группы с системным риском.

        Args:
            correlation_matrix: Матрица корреляций
            threshold: Порог корреляции

        Returns:
            List[Dict]: Список групп с системным риском
        """
        systemic_groups = []

        # Ищем группы major активов с высокой корреляцией
        major_symbols = [
            symbol for symbol, info in self.market_caps.items()
            if self.get_market_cap_info(symbol).tier == 1
        ]

        for i, sym1 in enumerate(major_symbols):
            for sym2 in major_symbols[i+1:]:
                key = (sym1, sym2) if sym1 < sym2 else (sym2, sym1)

                if key not in correlation_matrix:
                    continue

                correlation = correlation_matrix[key]

                if abs(correlation) >= threshold:
                    info_1 = self.get_market_cap_info(sym1)
                    info_2 = self.get_market_cap_info(sym2)

                    # Системный риск = доминирование обоих активов * корреляция
                    systemic_risk_score = (
                        (info_1.dominance_percent + info_2.dominance_percent) / 100.0
                        * abs(correlation)
                    )

                    systemic_groups.append({
                        "symbols": [sym1, sym2],
                        "correlation": correlation,
                        "combined_dominance": (
                            info_1.dominance_percent + info_2.dominance_percent
                        ),
                        "systemic_risk_score": systemic_risk_score,
                        "tier": "major-major"
                    })

        # Сортируем по системному риску
        systemic_groups.sort(key=lambda x: x["systemic_risk_score"], reverse=True)

        return systemic_groups

    def calculate_portfolio_market_cap_exposure(
        self,
        open_positions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Рассчитывает экспозицию портфеля по tiers.

        Args:
            open_positions: {symbol: position_size_usdt}

        Returns:
            Dict[str, float]: Экспозиция по tiers
        """
        tier_exposure = {
            "tier_1_major": 0.0,
            "tier_2_mid": 0.0,
            "tier_3_small": 0.0,
            "unknown": 0.0
        }

        total_exposure = sum(open_positions.values())

        for symbol, size in open_positions.items():
            info = self.get_market_cap_info(symbol)

            if not info:
                tier_exposure["unknown"] += size
                continue

            if info.tier == 1:
                tier_exposure["tier_1_major"] += size
            elif info.tier == 2:
                tier_exposure["tier_2_mid"] += size
            else:
                tier_exposure["tier_3_small"] += size

        # Нормализуем к процентам
        if total_exposure > 0:
            for tier in tier_exposure:
                tier_exposure[tier] = (tier_exposure[tier] / total_exposure) * 100

        return tier_exposure

    def update_market_caps(
        self,
        new_market_caps: Dict[str, float]
    ):
        """
        Обновляет данные о market caps.

        Args:
            new_market_caps: Новые данные {symbol: market_cap_usd}
        """
        self.market_caps.update(new_market_caps)
        self._calculate_dominance()

        logger.info(f"Market caps обновлены: {len(new_market_caps)} активов")


# Глобальный экземпляр
market_cap_manager = MarketCapWeightingManager()
