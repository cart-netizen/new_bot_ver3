"""
Volume Flow Strategy - отслеживание потоков крупных игроков через динамику объемов.

Методология:
- Детекция крупных блоковых заявок (whale orders)
- Анализ агрессивных market orders через проскальзывание уровней
- Трекинг кластеров объема и их "поглощения"
- Order Flow Imbalance (OFI) анализ
- Управление позицией на основе объемных зон

Путь: backend/strategies/volume_flow_strategy.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np

from core.logger import get_logger
from models.signal import TradingSignal, SignalType, SignalStrength, SignalSource
from models.orderbook import OrderBookSnapshot, OrderBookMetrics
from strategy.candle_manager import Candle
from strategies.base_orderbook_strategy import BaseOrderBookStrategy

logger = get_logger(__name__)


@dataclass
class VolumeCluster:
    """Кластер объема на определенном ценовом уровне."""
    price: float
    volume: float
    side: str  # "bid" или "ask"
    first_seen: int  # timestamp ms
    last_seen: int
    peak_volume: float  # Максимальный объем за время жизни
    is_absorbed: bool = False  # Был ли кластер "поглощен"
    absorption_timestamp: Optional[int] = None


@dataclass
class WhaleOrder:
    """Крупная заявка (whale order)."""
    price: float
    volume: float
    volume_usdt: float
    side: str
    timestamp: int
    is_aggressive: bool = False  # Агрессивная market order


@dataclass
class VolumeFlowConfig:
    """Конфигурация Volume Flow стратегии."""
    # Whale orders
    whale_threshold_percentile: float = 95.0  # Перцентиль для определения whale
    min_whale_volume_usdt: float = 50000.0  # Минимальный объем для whale
    
    # Volume clusters
    cluster_volume_multiplier: float = 3.0  # X раз больше среднего объема
    cluster_merge_distance_pct: float = 0.001  # 0.1% для слияния кластеров
    max_cluster_age_minutes: int = 30  # Макс возраст кластера
    
    # Order Flow Imbalance (OFI)
    ofi_window: int = 10  # Окно для расчета OFI
    ofi_buy_threshold: float = 0.3  # Порог для bullish OFI
    ofi_sell_threshold: float = -0.3  # Порог для bearish OFI
    
    # Absorption detection
    absorption_threshold_pct: float = 0.5  # 50% снижение объема = поглощение
    absorption_confirmation_snapshots: int = 3
    
    # Signal generation
    min_whale_orders_for_signal: int = 2  # Минимум whale orders
    min_cluster_strength: float = 0.6
    
    # Risk management
    stop_loss_cluster_distance: float = 1.5  # X расстояний до ближайшего кластера


class VolumeFlowStrategy(BaseOrderBookStrategy):
    """
    Стратегия на основе анализа потоков объемов.
    
    Принцип:
    Крупные игроки оставляют следы в виде:
    1. Блоковых заявок
    2. Агрессивных market orders
    3. Поглощения уровней
    
    Следуем за "умными деньгами".
    """

    def __init__(self, config: VolumeFlowConfig):
        """
        Инициализация стратегии.

        Args:
            config: Конфигурация
        """
        super().__init__("volume_flow")
        self.config = config
        
        # Трекинг кластеров
        self.volume_clusters: Dict[str, List[VolumeCluster]] = {}
        
        # История whale orders
        self.whale_orders: Dict[str, deque] = {}
        self.max_whale_history = 50
        
        # Order Flow Imbalance history
        self.ofi_history: Dict[str, deque] = {}
        
        # Статистика
        self.whale_orders_detected = 0
        self.clusters_absorbed = 0
        self.ofi_signals = 0
        
        logger.info(
            f"Инициализирована VolumeFlowStrategy: "
            f"whale_threshold={config.whale_threshold_percentile}%, "
            f"min_whale_volume={config.min_whale_volume_usdt} USDT"
        )

    def analyze(
        self,
        symbol: str,
        candles: List[Candle],
        current_price: float,
        orderbook: OrderBookSnapshot,
        metrics: OrderBookMetrics
    ) -> Optional[TradingSignal]:
        """
        Анализ потоков объемов и генерация сигнала.
        """
        # Шаг 1: Базовый анализ качества
        analysis = self.analyze_orderbook_quality(symbol, orderbook, metrics)
        
        if analysis.has_manipulation:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | "
                f"Манипуляция: {analysis.manipulation_type} - БЛОКИРУЕМ"
            )
            self.manipulation_blocks += 1
            return None
        
        # Шаг 2: Детекция whale orders
        whale_orders = self._detect_whale_orders(
            symbol, orderbook, metrics, current_price
        )

        if whale_orders:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | "
                f"Обнаружено {len(whale_orders)} whale orders"
            )

        # НОВОЕ: Подтверждение через реальные block trades из TradeManager
        real_block_trades_count = 0
        if self.trade_manager:
            try:
                stats = self.trade_manager.get_statistics(window_seconds=60)
                real_block_trades_count = stats.block_trade_count
                if real_block_trades_count > 0:
                    logger.debug(
                        f"[{self.strategy_name}] {symbol} | "
                        f"РЕАЛЬНЫЕ block trades: {real_block_trades_count} за 60 сек"
                    )
            except Exception as e:
                logger.debug(f"TradeManager stats error: {e}")

        # Шаг 3: Обновление и анализ volume clusters
        self._update_volume_clusters(symbol, orderbook, metrics, current_price)
        
        # Детекция поглощения кластеров
        absorbed_clusters = self._detect_absorbed_clusters(symbol)
        
        if absorbed_clusters:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | "
                f"Поглощено кластеров: {len(absorbed_clusters)}"
            )
        
        # Шаг 4: Order Flow Imbalance (OFI) расчет
        ofi = self._calculate_ofi(symbol, orderbook, metrics)
        
        # Шаг 5: Определение направления на основе всех факторов
        signal_analysis = self._analyze_flow_direction(
            symbol=symbol,
            whale_orders=whale_orders,
            absorbed_clusters=absorbed_clusters,
            ofi=ofi,
            market_pressure=analysis.market_pressure,
            pressure_strength=analysis.pressure_strength
        )
        
        if not signal_analysis['has_signal']:
            return None
        
        signal_type = signal_analysis['signal_type']
        flow_strength = signal_analysis['strength']
        
        # Шаг 6: Вычисление confidence (с новыми фичами из TradeManager)
        confidence = self._calculate_confidence(
            flow_strength=flow_strength,
            whale_count=len(whale_orders),
            absorption_count=len(absorbed_clusters),
            ofi=ofi,
            liquidity_quality=analysis.liquidity_quality,
            signal_type=signal_type,  # НОВОЕ: для проверки buy/sell pressure
            block_trades_count=real_block_trades_count  # НОВОЕ: реальные block trades
        )
        
        if confidence < 0.6:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | "
                f"Низкая confidence: {confidence:.2f} - ПРОПУСКАЕМ"
            )
            return None
        
        # Шаг 7: Определение силы сигнала
        if confidence >= 0.85:
            signal_strength = SignalStrength.STRONG
        elif confidence >= 0.70:
            signal_strength = SignalStrength.MEDIUM
        else:
            signal_strength = SignalStrength.WEAK
        
        # Шаг 8: Формирование reason (с новыми фичами)
        reason_parts = [
            f"Volume flow {signal_type.value}: strength={flow_strength:.2f}",
            f"Whale orders: {len(whale_orders)}"
        ]

        if absorbed_clusters:
            reason_parts.append(f"Absorbed clusters: {len(absorbed_clusters)}")

        reason_parts.append(f"OFI: {ofi:.2f}")

        # НОВОЕ: Добавляем информацию из TradeManager
        if real_block_trades_count > 0:
            reason_parts.append(f"Real block trades: {real_block_trades_count}")

        if self.trade_manager:
            try:
                _, _, pressure_ratio = self.trade_manager.calculate_buy_sell_pressure(60)
                reason_parts.append(f"Buy pressure: {pressure_ratio:.1%}")
            except Exception:
                pass
        
        # Шаг 9: Определение stop-loss на основе кластеров
        stop_loss_info = self._calculate_cluster_based_stop(
            symbol, signal_type, current_price
        )
        
        # Шаг 10: Создание сигнала (с новыми фичами из TradeManager)
        metadata = {
            'strategy': self.strategy_name,
            'flow_strength': flow_strength,
            'whale_orders': len(whale_orders),
            'absorbed_clusters': len(absorbed_clusters),
            'ofi': ofi,
            'clusters_nearby': stop_loss_info['clusters_nearby'],
            'stop_loss_distance': stop_loss_info['distance'],
            'liquidity_quality': analysis.liquidity_quality,
            # НОВОЕ: Real market trades статистики
            'real_block_trades': real_block_trades_count
        }

        # Добавляем дополнительные метрики из TradeManager
        if self.trade_manager:
            try:
                stats = self.trade_manager.get_statistics(window_seconds=60)
                metadata.update({
                    'buy_pressure_ratio': stats.buy_sell_ratio / (1 + stats.buy_sell_ratio),  # Normalize to 0-1
                    'order_flow_toxicity': stats.order_flow_toxicity,
                    'trade_arrival_rate': stats.arrival_rate,
                    'real_vwap': stats.vwap
                })
            except Exception:
                pass

        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            source=SignalSource.STRATEGY,
            strength=signal_strength,
            price=current_price,
            confidence=confidence,
            timestamp=int(datetime.now().timestamp() * 1000),
            reason=" | ".join(reason_parts),
            metadata=metadata
        )
        
        self.active_signals[symbol] = signal
        self.signals_generated += 1
        
        logger.info(
            f"🌊 VOLUME FLOW SIGNAL [{symbol}]: {signal_type.value}, "
            f"confidence={confidence:.2f}, "
            f"whale_orders={len(whale_orders)}, "
            f"ofi={ofi:.2f}"
        )
        
        return signal

    def _detect_whale_orders(
        self,
        symbol: str,
        orderbook: OrderBookSnapshot,
        metrics: OrderBookMetrics,
        current_price: float
    ) -> List[WhaleOrder]:
        """
        Детекция крупных заявок (whale orders).
        """
        whale_orders = []
        
        # Инициализируем историю whale orders
        if symbol not in self.whale_orders:
            self.whale_orders[symbol] = deque(maxlen=self.max_whale_history)
        
        # Вычисляем статистику объемов
        all_volumes = [vol for _, vol in orderbook.bids[:20]] + \
                     [vol for _, vol in orderbook.asks[:20]]
        
        if not all_volumes:
            return []
        
        # Перцентиль для определения whale
        whale_volume_threshold = np.percentile(
            all_volumes, 
            self.config.whale_threshold_percentile
        )
        
        timestamp = int(datetime.now().timestamp() * 1000)
        
        # Проверяем bid сторону
        for price, vol in orderbook.bids[:10]:
            volume_usdt = price * vol
            
            if (vol >= whale_volume_threshold and 
                volume_usdt >= self.config.min_whale_volume_usdt):
                
                whale = WhaleOrder(
                    price=price,
                    volume=vol,
                    volume_usdt=volume_usdt,
                    side="bid",
                    timestamp=timestamp,
                    is_aggressive=self._is_aggressive_order(price, current_price, "bid")
                )
                
                whale_orders.append(whale)
                self.whale_orders[symbol].append(whale)
                self.whale_orders_detected += 1
        
        # Проверяем ask сторону
        for price, vol in orderbook.asks[:10]:
            volume_usdt = price * vol
            
            if (vol >= whale_volume_threshold and 
                volume_usdt >= self.config.min_whale_volume_usdt):
                
                whale = WhaleOrder(
                    price=price,
                    volume=vol,
                    volume_usdt=volume_usdt,
                    side="ask",
                    timestamp=timestamp,
                    is_aggressive=self._is_aggressive_order(price, current_price, "ask")
                )
                
                whale_orders.append(whale)
                self.whale_orders[symbol].append(whale)
                self.whale_orders_detected += 1
        
        return whale_orders

    def _is_aggressive_order(
        self, 
        order_price: float, 
        current_price: float, 
        side: str
    ) -> bool:
        """
        Определить является ли ордер агрессивным (проскальзывает через спред).
        """
        # Агрессивный bid: цена выше текущей
        if side == "bid" and order_price >= current_price * 1.0001:
            return True
        
        # Агрессивный ask: цена ниже текущей
        if side == "ask" and order_price <= current_price * 0.9999:
            return True
        
        return False

    def _update_volume_clusters(
        self,
        symbol: str,
        orderbook: OrderBookSnapshot,
        metrics: OrderBookMetrics,
        current_price: float
    ):
        """
        Обновить кластеры объема.
        """
        if symbol not in self.volume_clusters:
            self.volume_clusters[symbol] = []
        
        timestamp = int(datetime.now().timestamp() * 1000)
        
        # Находим текущие кластеры в стакане
        current_clusters = self.find_volume_clusters(orderbook, side="both")
        
        # Обновляем существующие или создаем новые
        for cluster_price, cluster_volume in current_clusters:
            # Определяем side
            side = "bid" if cluster_price < current_price else "ask"
            
            # Ищем существующий кластер на похожей цене
            existing_cluster = None
            for cluster in self.volume_clusters[symbol]:
                price_diff_pct = abs(cluster.price - cluster_price) / cluster_price
                
                if price_diff_pct <= self.config.cluster_merge_distance_pct:
                    existing_cluster = cluster
                    break
            
            if existing_cluster:
                # Обновляем существующий
                existing_cluster.volume = cluster_volume
                existing_cluster.last_seen = timestamp
                existing_cluster.peak_volume = max(
                    existing_cluster.peak_volume, 
                    cluster_volume
                )
            else:
                # Создаем новый
                new_cluster = VolumeCluster(
                    price=cluster_price,
                    volume=cluster_volume,
                    side=side,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    peak_volume=cluster_volume
                )
                self.volume_clusters[symbol].append(new_cluster)
        
        # Удаляем старые кластеры
        max_age_ms = self.config.max_cluster_age_minutes * 60 * 1000
        self.volume_clusters[symbol] = [
            c for c in self.volume_clusters[symbol]
            if (timestamp - c.last_seen) < max_age_ms
        ]

    def _detect_absorbed_clusters(self, symbol: str) -> List[VolumeCluster]:
        """
        Детекция поглощенных кластеров.
        
        Кластер считается поглощенным если:
        - Его объем упал больше чем на X% от пика
        - Это произошло за несколько последних snapshot'ов
        """
        if symbol not in self.volume_clusters:
            return []
        
        absorbed = []
        
        for cluster in self.volume_clusters[symbol]:
            if cluster.is_absorbed:
                continue
            
            # Проверяем снижение объема от пика
            volume_drop_pct = (
                (cluster.peak_volume - cluster.volume) / cluster.peak_volume
            )
            
            if volume_drop_pct >= self.config.absorption_threshold_pct:
                cluster.is_absorbed = True
                cluster.absorption_timestamp = int(datetime.now().timestamp() * 1000)
                absorbed.append(cluster)
                self.clusters_absorbed += 1
        
        return absorbed

    def _calculate_ofi(
        self,
        symbol: str,
        orderbook: OrderBookSnapshot,
        metrics: OrderBookMetrics
    ) -> float:
        """
        Расчет Order Flow Imbalance (OFI).
        
        OFI = (bid_volume_change - ask_volume_change) / total_volume_change
        
        Positive OFI = больше buying pressure
        Negative OFI = больше selling pressure
        """
        if symbol not in self.ofi_history:
            self.ofi_history[symbol] = deque(maxlen=self.config.ofi_window)
        
        # Сохраняем текущее состояние
        self.ofi_history[symbol].append({
            'timestamp': orderbook.timestamp,
            'bid_volume': metrics.total_bid_volume,
            'ask_volume': metrics.total_ask_volume
        })
        
        if len(self.ofi_history[symbol]) < 2:
            return 0.0
        
        # Берем первое и последнее состояние из окна
        first = self.ofi_history[symbol][0]
        last = self.ofi_history[symbol][-1]
        
        bid_change = last['bid_volume'] - first['bid_volume']
        ask_change = last['ask_volume'] - first['ask_volume']
        total_change = abs(bid_change) + abs(ask_change)
        
        if total_change == 0:
            return 0.0
        
        ofi = (bid_change - ask_change) / total_change
        
        return ofi

    def _analyze_flow_direction(
        self,
        symbol: str,
        whale_orders: List[WhaleOrder],
        absorbed_clusters: List[VolumeCluster],
        ofi: float,
        market_pressure: str,
        pressure_strength: float
    ) -> Dict:
        """
        Определить направление потока на основе всех факторов.
        """
        # Подсчитываем bullish/bearish факторы
        bullish_score = 0.0
        bearish_score = 0.0
        
        # Фактор 1: Whale orders (30%)
        bid_whales = [w for w in whale_orders if w.side == "bid"]
        ask_whales = [w for w in whale_orders if w.side == "ask"]
        
        # Агрессивные whale orders имеют больший вес
        bid_whale_strength = sum(
            2.0 if w.is_aggressive else 1.0 for w in bid_whales
        )
        ask_whale_strength = sum(
            2.0 if w.is_aggressive else 1.0 for w in ask_whales
        )
        
        if bid_whale_strength > ask_whale_strength:
            bullish_score += 0.3
        elif ask_whale_strength > bid_whale_strength:
            bearish_score += 0.3
        
        # Фактор 2: Поглощенные кластеры (25%)
        # Поглощение ask кластеров = bullish (сопротивление преодолено)
        # Поглощение bid кластеров = bearish (поддержка пробита)
        absorbed_asks = [c for c in absorbed_clusters if c.side == "ask"]
        absorbed_bids = [c for c in absorbed_clusters if c.side == "bid"]
        
        if len(absorbed_asks) > len(absorbed_bids):
            bullish_score += 0.25
        elif len(absorbed_bids) > len(absorbed_asks):
            bearish_score += 0.25
        
        # Фактор 3: OFI (30%)
        if ofi >= self.config.ofi_buy_threshold:
            bullish_score += 0.3 * min(abs(ofi), 1.0)
        elif ofi <= self.config.ofi_sell_threshold:
            bearish_score += 0.3 * min(abs(ofi), 1.0)
        
        # Фактор 4: Market pressure (15%)
        if market_pressure == "bullish":
            bullish_score += 0.15 * pressure_strength
        elif market_pressure == "bearish":
            bearish_score += 0.15 * pressure_strength
        
        # Определяем направление
        if bullish_score > bearish_score and bullish_score >= 0.5:
            return {
                'has_signal': True,
                'signal_type': SignalType.BUY,
                'strength': bullish_score
            }
        elif bearish_score > bullish_score and bearish_score >= 0.5:
            return {
                'has_signal': True,
                'signal_type': SignalType.SELL,
                'strength': bearish_score
            }
        
        return {'has_signal': False}

    def _calculate_confidence(
        self,
        flow_strength: float,
        whale_count: int,
        absorption_count: int,
        ofi: float,
        liquidity_quality: float,
        signal_type: Optional[SignalType] = None,
        block_trades_count: int = 0
    ) -> float:
        """
        Вычислить итоговую confidence с учетом реальных market trades.
        """
        # Базовая confidence от силы потока
        base_confidence = flow_strength * 0.5

        # Бонус за количество whale orders (до +0.2)
        whale_bonus = min(whale_count / 5.0, 1.0) * 0.2

        # Бонус за поглощения (до +0.15)
        absorption_bonus = min(absorption_count / 3.0, 1.0) * 0.15

        # Качество ликвидности (0.15)
        liquidity_component = liquidity_quality * 0.15

        confidence = (
            base_confidence +
            whale_bonus +
            absorption_bonus +
            liquidity_component
        )

        # ==================== НОВЫЕ ФИЧИ ИЗ TRADEMANAGER ====================

        # 1. Bonus за реальные block trades (до +0.15)
        if block_trades_count > 0:
            block_trades_bonus = min(block_trades_count / 5.0, 1.0) * 0.15
            confidence += block_trades_bonus

        # 2. Buy/Sell Pressure Confirmation (до +0.15 или -0.2 при конфликте)
        if self.trade_manager and signal_type:
            try:
                buy_vol, sell_vol, pressure_ratio = self.trade_manager.calculate_buy_sell_pressure(window_seconds=60)

                if signal_type == SignalType.BUY:
                    # Long сигнал: ожидаем давление покупателей > 60%
                    if pressure_ratio > 0.7:
                        confidence += 0.15  # Сильное подтверждение
                    elif pressure_ratio > 0.6:
                        confidence += 0.08  # Умеренное подтверждение
                    elif pressure_ratio < 0.4:
                        confidence -= 0.2  # Конфликт с реальными трейдами!

                elif signal_type == SignalType.SELL:
                    # Short сигнал: ожидаем давление продавцов > 60%
                    if pressure_ratio < 0.3:
                        confidence += 0.15  # Сильное подтверждение
                    elif pressure_ratio < 0.4:
                        confidence += 0.08  # Умеренное подтверждение
                    elif pressure_ratio > 0.6:
                        confidence -= 0.2  # Конфликт с реальными трейдами!

            except Exception:
                pass  # Нет данных - ок

        # 3. Order Flow Toxicity Bonus (до +0.1)
        if self.trade_manager:
            try:
                toxicity = self.trade_manager.calculate_order_flow_toxicity(window_seconds=60)
                # Высокая токсичность = информированные трейдеры активны
                if abs(toxicity) > 0.5:
                    toxicity_bonus = min(abs(toxicity), 1.0) * 0.1
                    confidence += toxicity_bonus
            except Exception:
                pass  # Нет данных - ок

        # ======================================================================

        return max(min(confidence, 1.0), 0.0)  # Clamp [0, 1]

    def _calculate_cluster_based_stop(
        self,
        symbol: str,
        signal_type: SignalType,
        current_price: float
    ) -> Dict:
        """
        Рассчитать stop-loss на основе ближайших кластеров.
        """
        if symbol not in self.volume_clusters:
            return {'clusters_nearby': 0, 'distance': None}
        
        # Ищем ближайший кластер в противоположном направлении
        relevant_clusters = []
        
        if signal_type == SignalType.BUY:
            # Для long: ищем bid кластеры ниже текущей цены
            relevant_clusters = [
                c for c in self.volume_clusters[symbol]
                if c.side == "bid" and c.price < current_price and not c.is_absorbed
            ]
        else:
            # Для short: ищем ask кластеры выше текущей цены
            relevant_clusters = [
                c for c in self.volume_clusters[symbol]
                if c.side == "ask" and c.price > current_price and not c.is_absorbed
            ]
        
        if not relevant_clusters:
            return {'clusters_nearby': 0, 'distance': None}
        
        # Находим ближайший кластер
        nearest = min(relevant_clusters, key=lambda c: abs(c.price - current_price))
        
        # Расстояние в % от текущей цены
        distance_pct = abs(nearest.price - current_price) / current_price * 100
        
        # Stop-loss: дальше кластера на X расстояний
        stop_distance = distance_pct * self.config.stop_loss_cluster_distance
        
        return {
            'clusters_nearby': len(relevant_clusters),
            'distance': stop_distance
        }

    def get_statistics(self) -> Dict:
        """Получить расширенную статистику."""
        base_stats = super().get_statistics()
        base_stats.update({
            'whale_orders_detected': self.whale_orders_detected,
            'clusters_absorbed': self.clusters_absorbed,
            'ofi_signals': self.ofi_signals
        })
        return base_stats