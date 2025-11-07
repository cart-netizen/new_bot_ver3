"""
OrderBook Data Handler для бэктестинга.

Генерирует реалистичные orderbook snapshots из исторических свечей.
Использует статистические методы для симуляции bid/ask уровней.
"""

from typing import List, Tuple, Optional
from datetime import datetime
import numpy as np
from dataclasses import dataclass

from backend.core.logger import get_logger
from backend.models.orderbook import OrderBookSnapshot
from backend.backtesting.models import Candle

logger = get_logger(__name__)


@dataclass
class OrderBookSimulationConfig:
    """Конфигурация симуляции orderbook."""

    num_levels: int = 20  # Количество уровней в каждую сторону
    base_spread_bps: float = 2.0  # Базовый spread в basis points (0.02%)
    volatility_spread_multiplier: float = 2.0  # Множитель spread при высокой волатильности
    depth_decay_factor: float = 0.85  # Скорость убывания объема по уровням
    min_level_size: float = 0.01  # Минимальный размер уровня в BTC
    max_level_size: float = 5.0  # Максимальный размер уровня в BTC
    price_tick_size: float = 0.5  # Шаг цены между уровнями ($0.5)


class OrderBookDataHandler:
    """
    Обработчик для симуляции исторических orderbook данных.

    Методология:
    1. Spread оценивается на основе волатильности свечи
    2. Глубина стакана моделируется экспоненциальным убыванием
    3. Дисбаланс bid/ask определяется направлением движения цены
    4. Добавляется случайный шум для реалистичности
    """

    def __init__(self, config: Optional[OrderBookSimulationConfig] = None):
        """
        Инициализация обработчика.

        Args:
            config: Конфигурация симуляции
        """
        self.config = config or OrderBookSimulationConfig()

        # Кэш для хранения волатильности
        self.volatility_cache: List[float] = []
        self.volatility_window = 20  # Окно для расчета волатильности

        logger.info(
            f"Инициализирован OrderBook симулятор: "
            f"levels={self.config.num_levels}, "
            f"base_spread={self.config.base_spread_bps} bps"
        )

    def generate_orderbook(
        self,
        candle: Candle,
        symbol: str,
        prev_candle: Optional[Candle] = None
    ) -> OrderBookSnapshot:
        """
        Сгенерировать orderbook snapshot из свечи.

        Args:
            candle: Текущая свеча
            symbol: Символ торговой пары
            prev_candle: Предыдущая свеча (для расчета направления)

        Returns:
            OrderBookSnapshot: Симулированный snapshot стакана
        """
        # Расчет метрик свечи
        candle_range = candle.high - candle.low
        candle_volatility = candle_range / candle.close if candle.close > 0 else 0.001

        # Обновляем кэш волатильности
        self.volatility_cache.append(candle_volatility)
        if len(self.volatility_cache) > self.volatility_window:
            self.volatility_cache.pop(0)

        avg_volatility = np.mean(self.volatility_cache) if self.volatility_cache else candle_volatility

        # Расчет spread на основе волатильности
        base_spread = candle.close * (self.config.base_spread_bps / 10000)
        volatility_multiplier = 1 + (avg_volatility * self.config.volatility_spread_multiplier)
        spread = base_spread * volatility_multiplier

        # Определение mid price (используем close цену свечи)
        mid_price = candle.close

        # Определение дисбаланса на основе направления свечи
        price_change = candle.close - candle.open
        price_change_pct = price_change / candle.open if candle.open > 0 else 0

        # Дисбаланс: положительный = больше buyers, отрицательный = больше sellers
        imbalance = np.tanh(price_change_pct * 100)  # Нормализуем в [-1, 1]

        # Генерация bid/ask уровней
        bids = self._generate_side_levels(
            mid_price=mid_price,
            spread=spread,
            side='bid',
            imbalance=imbalance,
            volume=candle.volume
        )

        asks = self._generate_side_levels(
            mid_price=mid_price,
            spread=spread,
            side='ask',
            imbalance=imbalance,
            volume=candle.volume
        )

        return OrderBookSnapshot(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=candle.timestamp,
            update_id=None,
            sequence_id=None
        )

    def _generate_side_levels(
        self,
        mid_price: float,
        spread: float,
        side: str,
        imbalance: float,
        volume: float
    ) -> List[Tuple[float, float]]:
        """
        Сгенерировать уровни для одной стороны стакана.

        Args:
            mid_price: Средняя цена
            spread: Спред
            side: 'bid' или 'ask'
            imbalance: Дисбаланс [-1, 1]
            volume: Объем свечи

        Returns:
            List[Tuple[float, float]]: Список (price, quantity) уровней
        """
        levels = []

        # Определяем стартовую цену
        if side == 'bid':
            start_price = mid_price - (spread / 2)
            price_direction = -1
            # Больше объема на bid при положительном имбалансе (bullish)
            volume_multiplier = 1.0 + (imbalance * 0.3)
        else:  # ask
            start_price = mid_price + (spread / 2)
            price_direction = 1
            # Больше объема на ask при отрицательном имбалансе (bearish)
            volume_multiplier = 1.0 - (imbalance * 0.3)

        # Оцениваем общий объем для стороны
        # Используем объем свечи как базу
        total_side_volume = volume * 0.5 * volume_multiplier  # 50% объема свечи на сторону

        # Генерируем уровни
        for i in range(self.config.num_levels):
            # Цена уровня
            price_offset = i * self.config.price_tick_size * price_direction
            level_price = start_price + price_offset

            # Количество на уровне (экспоненциальное убывание)
            decay = self.config.depth_decay_factor ** i
            base_quantity = total_side_volume / self.config.num_levels
            level_quantity = base_quantity * decay

            # Добавляем случайный шум ±20%
            noise = np.random.uniform(0.8, 1.2)
            level_quantity *= noise

            # Ограничиваем размер уровня
            level_quantity = np.clip(
                level_quantity,
                self.config.min_level_size,
                self.config.max_level_size
            )

            levels.append((level_price, level_quantity))

        return levels

    def generate_orderbook_sequence(
        self,
        candles: List[Candle],
        symbol: str
    ) -> List[OrderBookSnapshot]:
        """
        Сгенерировать последовательность orderbook snapshots из свечей.

        Args:
            candles: Список свечей
            symbol: Символ торговой пары

        Returns:
            List[OrderBookSnapshot]: Список snapshots
        """
        orderbooks = []
        prev_candle = None

        for candle in candles:
            orderbook = self.generate_orderbook(candle, symbol, prev_candle)
            orderbooks.append(orderbook)
            prev_candle = candle

        logger.info(
            f"Сгенерировано {len(orderbooks)} orderbook snapshots для {symbol}"
        )

        return orderbooks

    def calculate_vwap(
        self,
        levels: List[Tuple[float, float]],
        depth: int = 5
    ) -> float:
        """
        Рассчитать VWAP для N уровней.

        Args:
            levels: Список (price, quantity) уровней
            depth: Количество уровней для расчета

        Returns:
            float: VWAP цена
        """
        if not levels:
            return 0.0

        top_levels = levels[:depth]
        total_value = sum(price * qty for price, qty in top_levels)
        total_volume = sum(qty for _, qty in top_levels)

        return total_value / total_volume if total_volume > 0 else 0.0

    def calculate_depth(
        self,
        levels: List[Tuple[float, float]],
        depth: int = 5
    ) -> float:
        """
        Рассчитать общий объем на N уровнях.

        Args:
            levels: Список (price, quantity) уровней
            depth: Количество уровней

        Returns:
            float: Общий объем
        """
        return sum(qty for _, qty in levels[:depth])
