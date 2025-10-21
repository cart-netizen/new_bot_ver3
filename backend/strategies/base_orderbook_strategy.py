"""
Базовый класс для OrderBook-aware стратегий.

Функциональность:
- Абстрактный интерфейс для стратегий, работающих со стаканом
- Утилиты для анализа микроструктуры рынка
- Детекция манипуляций
- Управление состоянием стратегии

Путь: backend/strategies/base_orderbook_strategy.py
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import numpy as np

from core.logger import get_logger
from models.signal import TradingSignal, SignalType, SignalStrength, SignalSource
from models.orderbook import OrderBookSnapshot, OrderBookMetrics
from strategy.candle_manager import Candle

logger = get_logger(__name__)


@dataclass
class OrderBookAnalysisResult:
    """Результат анализа стакана."""
    has_manipulation: bool
    manipulation_type: Optional[str]  # "spoofing", "layering", "wash_trading"
    manipulation_confidence: float
    liquidity_quality: float  # 0-1, качество ликвидности
    market_pressure: str  # "bullish", "bearish", "neutral"
    pressure_strength: float  # 0-1


class BaseOrderBookStrategy(ABC):
    """
    Базовый класс для стратегий, работающих со стаканом ордеров.
    
    Предоставляет общие утилиты для:
    - Детекции манипуляций
    - Анализа ликвидности
    - Работы с объемными кластерами
    - Статистики стратегии
    """

    def __init__(self, strategy_name: str):
        """
        Инициализация базовой стратегии.

        Args:
            strategy_name: Имя стратегии
        """
        self.strategy_name = strategy_name
        
        # История snapshot'ов для временного анализа
        self.snapshot_history: Dict[str, deque] = {}
        self.max_snapshot_history = 100
        
        # История метрик
        self.metrics_history: Dict[str, deque] = {}
        
        # Активные сигналы
        self.active_signals: Dict[str, TradingSignal] = {}
        
        # Статистика
        self.signals_generated = 0
        self.manipulation_blocks = 0
        self.liquidity_blocks = 0
        
        logger.info(
            f"Инициализирована {strategy_name} (OrderBook-aware стратегия)"
        )

    @abstractmethod
    def analyze(
        self,
        symbol: str,
        candles: List[Candle],
        current_price: float,
        orderbook: OrderBookSnapshot,
        metrics: OrderBookMetrics
    ) -> Optional[TradingSignal]:
        """
        Анализ и генерация сигнала.

        Args:
            symbol: Торговая пара
            candles: История свечей
            current_price: Текущая цена
            orderbook: Снимок стакана
            metrics: Метрики стакана

        Returns:
            TradingSignal или None
        """
        pass

    def update_snapshot_history(
        self, 
        symbol: str, 
        snapshot: OrderBookSnapshot
    ):
        """Обновить историю snapshot'ов."""
        if symbol not in self.snapshot_history:
            self.snapshot_history[symbol] = deque(maxlen=self.max_snapshot_history)
        
        self.snapshot_history[symbol].append(snapshot)

    def update_metrics_history(
        self, 
        symbol: str, 
        metrics: OrderBookMetrics
    ):
        """Обновить историю метрик."""
        if symbol not in self.metrics_history:
            self.metrics_history[symbol] = deque(maxlen=self.max_snapshot_history)
        
        self.metrics_history[symbol].append(metrics)

    def analyze_orderbook_quality(
        self,
        symbol: str,
        orderbook: OrderBookSnapshot,
        metrics: OrderBookMetrics
    ) -> OrderBookAnalysisResult:
        """
        Комплексный анализ качества стакана.

        Returns:
            OrderBookAnalysisResult
        """
        # Обновляем историю
        self.update_snapshot_history(symbol, orderbook)
        self.update_metrics_history(symbol, metrics)
        
        # Детекция манипуляций
        has_manipulation, manip_type, manip_confidence = self._detect_manipulation(
            symbol, orderbook, metrics
        )
        
        if has_manipulation:
            self.manipulation_blocks += 1
        
        # Оценка качества ликвидности
        liquidity_quality = self._assess_liquidity_quality(orderbook, metrics)
        
        # Определение давления рынка
        pressure, pressure_strength = self._determine_market_pressure(
            symbol, orderbook, metrics
        )
        
        return OrderBookAnalysisResult(
            has_manipulation=has_manipulation,
            manipulation_type=manip_type,
            manipulation_confidence=manip_confidence,
            liquidity_quality=liquidity_quality,
            market_pressure=pressure,
            pressure_strength=pressure_strength
        )

    def _detect_manipulation(
        self,
        symbol: str,
        orderbook: OrderBookSnapshot,
        metrics: OrderBookMetrics
    ) -> Tuple[bool, Optional[str], float]:
        """
        Упрощенная детекция манипуляций на уровне стратегии.

        Returns:
            (has_manipulation, type, confidence)
        """
        # Признаки spoofing: крупные ордера с коротким TTL
        if len(self.snapshot_history.get(symbol, [])) < 5:
            return False, None, 0.0
        
        recent_snapshots = list(self.snapshot_history[symbol])[-5:]
        
        # Ищем крупные ордера, которые быстро исчезли
        large_order_threshold = metrics.total_bid_volume * 0.1  # 10% от общего объема
        
        spoofing_score = 0.0
        
        for i in range(len(recent_snapshots) - 1):
            prev_snap = recent_snapshots[i]
            curr_snap = recent_snapshots[i + 1]
            
            # Проверяем bid сторону
            prev_bids = {price: vol for price, vol in prev_snap.bids[:5]}
            curr_bids = {price: vol for price, vol in curr_snap.bids[:5]}
            
            for price, vol in prev_bids.items():
                if vol > large_order_threshold and price not in curr_bids:
                    spoofing_score += 0.2
            
            # Проверяем ask сторону
            prev_asks = {price: vol for price, vol in prev_snap.asks[:5]}
            curr_asks = {price: vol for price, vol in curr_snap.asks[:5]}
            
            for price, vol in prev_asks.items():
                if vol > large_order_threshold and price not in curr_asks:
                    spoofing_score += 0.2
        
        if spoofing_score > 0.5:
            return True, "spoofing", min(spoofing_score, 1.0)
        
        return False, None, 0.0

    def _assess_liquidity_quality(
        self,
        orderbook: OrderBookSnapshot,
        metrics: OrderBookMetrics
    ) -> float:
        """
        Оценка качества ликвидности (0-1).

        Высокое качество:
        - Узкий спред
        - Глубокий стакан
        - Равномерное распределение объемов
        """
        quality_score = 0.0
        
        # Компонент 1: Спред (30%)
        if metrics.mid_price and metrics.spread:
            spread_pct = (metrics.spread / metrics.mid_price) * 100
            # Спред < 0.01% = отлично, > 0.1% = плохо
            spread_score = max(0.0, 1.0 - (spread_pct / 0.1))
            quality_score += spread_score * 0.3
        
        # Компонент 2: Глубина (40%)
        total_volume = metrics.total_bid_volume + metrics.total_ask_volume
        # Нормализуем логарифмически
        depth_score = min(np.log1p(total_volume) / 15.0, 1.0)
        quality_score += depth_score * 0.4
        
        # Компонент 3: Баланс (30%)
        # Имбалланс близкий к 0.5 = хороший баланс
        balance_score = 1.0 - abs(metrics.imbalance - 0.5) * 2.0
        quality_score += balance_score * 0.3
        
        return min(quality_score, 1.0)

    def _determine_market_pressure(
        self,
        symbol: str,
        orderbook: OrderBookSnapshot,
        metrics: OrderBookMetrics
    ) -> Tuple[str, float]:
        """
        Определение давления рынка.

        Returns:
            (pressure_type, strength)
        """
        # Анализируем imbalance и volume delta
        imbalance = metrics.imbalance
        
        # Имбалланс > 0.6 = бычье давление, < 0.4 = медвежье
        if imbalance > 0.6:
            pressure = "bullish"
            strength = (imbalance - 0.5) * 2.0  # Нормализуем 0.6-1.0 в 0.2-1.0
        elif imbalance < 0.4:
            pressure = "bearish"
            strength = (0.5 - imbalance) * 2.0  # Нормализуем 0.0-0.4 в 1.0-0.2
        else:
            pressure = "neutral"
            strength = 0.0
        
        # Усиливаем через историю (тренд imbalance)
        if len(self.metrics_history.get(symbol, [])) >= 5:
            recent_imbalances = [
                m.imbalance for m in list(self.metrics_history[symbol])[-5:]
            ]
            imbalance_trend = recent_imbalances[-1] - recent_imbalances[0]
            
            # Если тренд согласуется с давлением, усиливаем
            if (pressure == "bullish" and imbalance_trend > 0) or \
               (pressure == "bearish" and imbalance_trend < 0):
                strength = min(strength * 1.2, 1.0)
        
        return pressure, min(strength, 1.0)

    def find_volume_clusters(
        self,
        orderbook: OrderBookSnapshot,
        side: str = "both"
    ) -> List[Tuple[float, float]]:
        """
        Найти кластеры объема в стакане.

        Args:
            orderbook: Снимок стакана
            side: "bid", "ask" или "both"

        Returns:
            List of (price, volume)
        """
        clusters = []
        
        # Анализируем bid сторону
        if side in ["bid", "both"]:
            bids = orderbook.bids[:20]
            if bids:
                volumes = [vol for _, vol in bids]
                avg_volume = np.mean(volumes)
                threshold = avg_volume * 2.0  # Кластер = 2x средний объем
                
                for price, vol in bids:
                    if vol > threshold:
                        clusters.append((price, vol))
        
        # Анализируем ask сторону
        if side in ["ask", "both"]:
            asks = orderbook.asks[:20]
            if asks:
                volumes = [vol for _, vol in asks]
                avg_volume = np.mean(volumes)
                threshold = avg_volume * 2.0
                
                for price, vol in asks:
                    if vol > threshold:
                        clusters.append((price, vol))
        
        # Сортируем по объему (убывание)
        clusters.sort(key=lambda x: x[1], reverse=True)
        
        return clusters

    def calculate_volume_delta(
        self,
        symbol: str,
        lookback: int = 5
    ) -> float:
        """
        Вычислить volume delta за последние N snapshot'ов.

        Positive delta = больше bid объема добавилось
        Negative delta = больше ask объема добавилось
        """
        if symbol not in self.metrics_history:
            return 0.0
        
        history = list(self.metrics_history[symbol])
        if len(history) < lookback:
            return 0.0
        
        recent = history[-lookback:]
        
        bid_delta = recent[-1].total_bid_volume - recent[0].total_bid_volume
        ask_delta = recent[-1].total_ask_volume - recent[0].total_ask_volume
        
        return bid_delta - ask_delta

    def check_large_walls(
        self,
        orderbook: OrderBookSnapshot,
        threshold_usdt: float = 100000.0
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Проверить наличие крупных стен (walls) в стакане.

        Args:
            orderbook: Снимок стакана
            threshold_usdt: Порог в USDT для определения "стены"

        Returns:
            {"bid_walls": [...], "ask_walls": [...]}
        """
        bid_walls = []
        ask_walls = []
        
        # Bid walls
        for price, vol in orderbook.bids[:20]:
            volume_usdt = price * vol
            if volume_usdt >= threshold_usdt:
                bid_walls.append((price, vol))
        
        # Ask walls
        for price, vol in orderbook.asks[:20]:
            volume_usdt = price * vol
            if volume_usdt >= threshold_usdt:
                ask_walls.append((price, vol))
        
        return {
            "bid_walls": bid_walls,
            "ask_walls": ask_walls
        }

    def get_statistics(self) -> Dict:
        """Получить статистику стратегии."""
        return {
            'strategy': self.strategy_name,
            'signals_generated': self.signals_generated,
            'manipulation_blocks': self.manipulation_blocks,
            'liquidity_blocks': self.liquidity_blocks,
            'active_signals': len(self.active_signals),
            'symbols_tracked': len(self.snapshot_history)
        }
