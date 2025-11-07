"""
Trade Data Handler для бэктестинга.

Генерирует реалистичные market trades из исторических свечей.
Использует статистические методы для симуляции buy/sell агрессивности.
"""

from typing import List, Optional
from datetime import datetime
import numpy as np
from dataclasses import dataclass

from backend.core.logger import get_logger
from backend.models.market_data import MarketTrade
from backend.backtesting.models import Candle

logger = get_logger(__name__)


@dataclass
class TradeSimulationConfig:
    """Конфигурация симуляции market trades."""

    # Количество trades на единицу объема
    trades_per_volume_unit: float = 100.0  # ~100 trades per 1 BTC
    min_trade_size: float = 0.001  # Минимальный размер трейда (0.001 BTC)
    max_trade_size: float = 2.0  # Максимальный размер трейда (2 BTC)
    block_trade_threshold: float = 5.0  # Порог для определения block trade (5 BTC)

    # Распределение trades внутри свечи
    time_distribution: str = "exponential"  # "uniform", "exponential", "poisson"

    # Шум и реалистичность
    price_noise_bps: float = 1.0  # Шум цены в basis points
    volume_noise_factor: float = 0.2  # Случайный разброс объема ±20%


class TradeDataHandler:
    """
    Обработчик для симуляции исторических market trades данных.

    Методология:
    1. Определяем buy/sell давление из направления свечи (close vs open)
    2. Распределяем объем свечи по времени
    3. Генерируем отдельные trades с реалистичными размерами
    4. Добавляем случайный шум для естественности
    5. Определяем block trades (крупные институциональные сделки)
    """

    def __init__(self, config: Optional[TradeSimulationConfig] = None):
        """
        Инициализация обработчика.

        Args:
            config: Конфигурация симуляции
        """
        self.config = config or TradeSimulationConfig()

        logger.info(
            f"Инициализирован Trade симулятор: "
            f"trades_per_volume={self.config.trades_per_volume_unit}, "
            f"block_threshold={self.config.block_trade_threshold} BTC"
        )

    def generate_trades_from_candle(
        self,
        candle: Candle,
        symbol: str
    ) -> List[MarketTrade]:
        """
        Сгенерировать market trades из одной свечи.

        Args:
            candle: Свеча OHLCV
            symbol: Символ торговой пары

        Returns:
            List[MarketTrade]: Список сгенерированных trades
        """
        if candle.volume <= 0:
            return []

        # Определяем buy/sell давление
        price_change = candle.close - candle.open
        price_range = candle.high - candle.low

        if price_range > 0:
            # Bullish pressure: насколько цена закрылась ближе к high
            bull_pressure = (candle.close - candle.low) / price_range
        else:
            bull_pressure = 0.5  # Нейтральный

        # Распределяем объем на buy/sell
        buy_volume = candle.volume * bull_pressure
        sell_volume = candle.volume * (1 - bull_pressure)

        # Рассчитываем количество trades
        num_trades = max(
            int(candle.volume * self.config.trades_per_volume_unit),
            10  # Минимум 10 trades на свечу
        )

        # Генерируем временные метки внутри свечи
        trade_timestamps = self._generate_timestamps(
            start_time=candle.timestamp,
            candle_interval_ms=60000,  # 1 минута
            num_trades=num_trades
        )

        # Генерируем цены внутри OHLC диапазона
        trade_prices = self._generate_prices(
            candle=candle,
            num_trades=num_trades
        )

        # Генерируем размеры trades
        buy_sizes = self._generate_trade_sizes(buy_volume, num_trades // 2)
        sell_sizes = self._generate_trade_sizes(sell_volume, num_trades - num_trades // 2)

        # Создаем trades
        trades = []
        buy_idx = 0
        sell_idx = 0

        for i in range(num_trades):
            # Определяем сторону (buy или sell) с учетом давления
            is_buy = np.random.random() < bull_pressure

            if is_buy and buy_idx < len(buy_sizes):
                side = "Buy"
                size = buy_sizes[buy_idx]
                buy_idx += 1
            elif sell_idx < len(sell_sizes):
                side = "Sell"
                size = sell_sizes[sell_idx]
                sell_idx += 1
            else:
                # Если один пул закончился, берем из другого
                if buy_idx < len(buy_sizes):
                    side = "Buy"
                    size = buy_sizes[buy_idx]
                    buy_idx += 1
                else:
                    side = "Sell"
                    size = sell_sizes[sell_idx]
                    sell_idx += 1

            # Определяем block trade
            is_block = size >= self.config.block_trade_threshold

            trade = MarketTrade(
                trade_id=f"{symbol}_{candle.timestamp}_{i}",
                symbol=symbol,
                side=side,
                price=trade_prices[i],
                quantity=size,
                timestamp=trade_timestamps[i],
                is_block_trade=is_block
            )

            trades.append(trade)

        return trades

    def _generate_timestamps(
        self,
        start_time: int,
        candle_interval_ms: int,
        num_trades: int
    ) -> List[int]:
        """
        Сгенерировать временные метки для trades внутри свечи.

        Args:
            start_time: Начало свечи (timestamp ms)
            candle_interval_ms: Длительность свечи (ms)
            num_trades: Количество trades

        Returns:
            List[int]: Список timestamps
        """
        if self.config.time_distribution == "uniform":
            # Равномерное распределение
            offsets = np.linspace(0, candle_interval_ms, num_trades, dtype=int)
        elif self.config.time_distribution == "exponential":
            # Экспоненциальное (больше в начале)
            offsets = np.random.exponential(
                scale=candle_interval_ms / 3,
                size=num_trades
            )
            offsets = np.clip(offsets, 0, candle_interval_ms)
            offsets = np.sort(offsets).astype(int)
        else:  # poisson
            # Пуассоновское (случайное)
            offsets = np.sort(
                np.random.uniform(0, candle_interval_ms, num_trades)
            ).astype(int)

        timestamps = [start_time + int(offset) for offset in offsets]
        return timestamps

    def _generate_prices(
        self,
        candle: Candle,
        num_trades: int
    ) -> List[float]:
        """
        Сгенерировать цены для trades внутри OHLC диапазона.

        Args:
            candle: Свеча
            num_trades: Количество trades

        Returns:
            List[float]: Список цен
        """
        # Создаем ценовую траекторию O -> H/L -> C
        prices = []

        # 1. От Open к High или Low (первая половина)
        if candle.close > candle.open:  # Bullish
            # O -> L -> H -> C
            quarter = num_trades // 4
            prices.extend(np.linspace(candle.open, candle.low, quarter))
            prices.extend(np.linspace(candle.low, candle.high, quarter))
            prices.extend(np.linspace(candle.high, candle.close, num_trades - 2 * quarter))
        else:  # Bearish
            # O -> H -> L -> C
            quarter = num_trades // 4
            prices.extend(np.linspace(candle.open, candle.high, quarter))
            prices.extend(np.linspace(candle.high, candle.low, quarter))
            prices.extend(np.linspace(candle.low, candle.close, num_trades - 2 * quarter))

        # Добавляем шум
        noise_scale = candle.close * (self.config.price_noise_bps / 10000)
        noise = np.random.normal(0, noise_scale, len(prices))
        prices = [p + n for p, n in zip(prices, noise)]

        # Ограничиваем диапазоном [low, high]
        prices = [
            np.clip(p, candle.low, candle.high)
            for p in prices
        ]

        return prices

    def _generate_trade_sizes(
        self,
        total_volume: float,
        num_trades: int
    ) -> List[float]:
        """
        Сгенерировать размеры trades с реалистичным распределением.

        Args:
            total_volume: Общий объем для распределения
            num_trades: Количество trades

        Returns:
            List[float]: Список размеров trades
        """
        if num_trades == 0:
            return []

        # Используем степенное распределение (power law)
        # Большинство trades маленькие, редкие - большие
        alpha = 2.5  # Параметр power law
        sizes = np.random.pareto(alpha, num_trades) + 1

        # Нормализуем к total_volume
        sizes = sizes / sizes.sum() * total_volume

        # Добавляем случайный шум
        noise = np.random.uniform(
            1 - self.config.volume_noise_factor,
            1 + self.config.volume_noise_factor,
            num_trades
        )
        sizes = sizes * noise

        # Ограничиваем размеры
        sizes = np.clip(
            sizes,
            self.config.min_trade_size,
            self.config.max_trade_size
        )

        # Перенормализуем после clipping
        sizes = sizes / sizes.sum() * total_volume

        return sizes.tolist()

    def generate_trades_from_candles(
        self,
        candles: List[Candle],
        symbol: str
    ) -> List[MarketTrade]:
        """
        Сгенерировать market trades из последовательности свечей.

        Args:
            candles: Список свечей
            symbol: Символ торговой пары

        Returns:
            List[MarketTrade]: Список всех trades
        """
        all_trades = []

        for candle in candles:
            trades = self.generate_trades_from_candle(candle, symbol)
            all_trades.extend(trades)

        logger.info(
            f"Сгенерировано {len(all_trades)} market trades "
            f"из {len(candles)} свечей для {symbol}"
        )

        # Статистика
        if all_trades:
            buy_trades = [t for t in all_trades if t.is_buy]
            sell_trades = [t for t in all_trades if t.is_sell]
            block_trades = [t for t in all_trades if t.is_block_trade]

            logger.info(
                f"  Buy: {len(buy_trades)} ({len(buy_trades)/len(all_trades)*100:.1f}%), "
                f"Sell: {len(sell_trades)} ({len(sell_trades)/len(all_trades)*100:.1f}%), "
                f"Block: {len(block_trades)}"
            )

        return all_trades

    def calculate_buy_sell_pressure(
        self,
        trades: List[MarketTrade],
        window_ms: int = 60000
    ) -> float:
        """
        Рассчитать buy/sell давление из trades.

        Args:
            trades: Список trades
            window_ms: Временное окно (мс)

        Returns:
            float: Давление [-1, 1], положительное = bullish
        """
        if not trades:
            return 0.0

        # Фильтруем trades в окне
        latest_time = max(t.timestamp for t in trades)
        window_trades = [
            t for t in trades
            if t.timestamp >= latest_time - window_ms
        ]

        if not window_trades:
            return 0.0

        # Считаем объемы
        buy_volume = sum(t.quantity for t in window_trades if t.is_buy)
        sell_volume = sum(t.quantity for t in window_trades if t.is_sell)
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return 0.0

        # Нормализуем в [-1, 1]
        pressure = (buy_volume - sell_volume) / total_volume
        return pressure
