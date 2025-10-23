"""
Imbalance Strategy - торговля на дисбалансе спроса/предложения в стакане.

Методология:
- Анализ дисбаланса bid/ask объемов на разных глубинах
- Подтверждение через volume delta
- Фильтрация через детекцию манипуляций
- Проверка отсутствия крупных стен против направления сигнала
- Адаптивное управление уверенностью на основе качества стакана

Путь: backend/strategies/imbalance_strategy.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from core.logger import get_logger
from models.signal import TradingSignal, SignalType, SignalStrength, SignalSource
from models.orderbook import OrderBookSnapshot, OrderBookMetrics
from strategy.candle_manager import Candle
from strategies.base_orderbook_strategy import (
    BaseOrderBookStrategy, 
    OrderBookAnalysisResult
)

logger = get_logger(__name__)


@dataclass
class ImbalanceConfig:
    """Конфигурация Imbalance стратегии."""
    # Пороги дисбаланса
    imbalance_buy_threshold: float = 0.75  # > 75% bid объема = BUY
    imbalance_sell_threshold: float = 0.25  # < 25% bid объема = SELL
    
    # Volume delta
    min_volume_delta_usdt: float = 50000.0  # Минимальный volume delta для подтверждения
    volume_delta_lookback: int = 5  # Количество snapshot'ов для расчета
    
    # Крупные стены
    large_wall_threshold_usdt: float = 100000.0  # Порог для определения "стены"
    max_opposing_wall_ratio: float = 0.5  # Макс соотношение противоположной стены
    
    # Качество ликвидности
    min_liquidity_quality: float = 0.5  # Минимальное качество ликвидности
    
    # Подтверждение через глубину
    use_depth_confirmation: bool = True
    depth_levels: List[int] = None  # [5, 10] - проверять на разных глубинах
    
    # Risk management
    base_confidence: float = 0.7
    manipulation_penalty: float = 0.3  # Снижение confidence при манипуляциях
    
    def __post_init__(self):
        if self.depth_levels is None:
            self.depth_levels = [5, 10]


class ImbalanceStrategy(BaseOrderBookStrategy):
    """
    Стратегия на основе дисбаланса в стакане ордеров.
    
    Принцип:
    Сильное доминирование bid/ask объемов указывает на предстоящее движение цены.
    Но требуется фильтрация через качество ликвидности и отсутствие манипуляций.
    """

    def __init__(self, config: ImbalanceConfig):
        """
        Инициализация стратегии.

        Args:
            config: Конфигурация стратегии
        """
        super().__init__("imbalance")
        self.config = config
        
        # Дополнительная статистика
        self.imbalance_signals = 0
        self.wall_blocks = 0
        
        logger.info(
            f"Инициализирована ImbalanceStrategy: "
            f"buy_threshold={config.imbalance_buy_threshold}, "
            f"sell_threshold={config.imbalance_sell_threshold}, "
            f"min_volume_delta={config.min_volume_delta_usdt} USDT"
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
        Анализ дисбаланса и генерация сигнала.

        Args:
            symbol: Торговая пара
            candles: История свечей (не используется напрямую)
            current_price: Текущая цена
            orderbook: Снимок стакана
            metrics: Метрики стакана

        Returns:
            TradingSignal или None
        """
        # Шаг 1: Комплексный анализ качества стакана
        analysis = self.analyze_orderbook_quality(symbol, orderbook, metrics)
        
        # Фильтр: Блокируем при манипуляциях
        if analysis.has_manipulation:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | "
                f"Манипуляция обнаружена: {analysis.manipulation_type}, "
                f"confidence={analysis.manipulation_confidence:.2f} - БЛОКИРУЕМ"
            )
            self.manipulation_blocks += 1
            return None
        
        # Фильтр: Минимальное качество ликвидности
        if analysis.liquidity_quality < self.config.min_liquidity_quality:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | "
                f"Низкое качество ликвидности: {analysis.liquidity_quality:.2f} - ПРОПУСКАЕМ"
            )
            self.liquidity_blocks += 1
            return None
        
        # Шаг 2: Анализ дисбаланса на разных глубинах
        imbalance_signals = self._analyze_multi_depth_imbalance(symbol, metrics)
        
        if not imbalance_signals['has_signal']:
            return None
        
        signal_type = imbalance_signals['signal_type']
        imbalance_strength = imbalance_signals['strength']
        
        # Шаг 3: Подтверждение через volume delta
        volume_delta_usdt = self._calculate_volume_delta_usdt(symbol, current_price)
        
        volume_delta_confirms = self._check_volume_delta_confirmation(
            signal_type, 
            volume_delta_usdt
        )
        
        if not volume_delta_confirms:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | "
                f"Volume delta не подтверждает: {volume_delta_usdt:.2f} USDT - ПРОПУСКАЕМ"
            )
            return None
        
        # Шаг 4: Проверка крупных стен
        walls = self.check_large_walls(orderbook, self.config.large_wall_threshold_usdt)
        
        wall_check = self._check_opposing_walls(signal_type, walls, current_price)
        
        if not wall_check['passed']:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | "
                f"Крупная противоположная стена: {wall_check['reason']} - БЛОКИРУЕМ"
            )
            self.wall_blocks += 1
            return None
        
        # Шаг 5: Вычисление итоговой confidence
        final_confidence = self._calculate_signal_confidence(
            imbalance_strength=imbalance_strength,
            volume_delta_usdt=volume_delta_usdt,
            liquidity_quality=analysis.liquidity_quality,
            market_pressure=analysis.market_pressure,
            pressure_strength=analysis.pressure_strength,
            signal_type=signal_type
        )
        
        # Проверка минимальной confidence
        if final_confidence < 0.6:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | "
                f"Низкая итоговая confidence: {final_confidence:.2f} - ПРОПУСКАЕМ"
            )
            return None
        
        # Шаг 6: Определение силы сигнала
        if final_confidence >= 0.85:
            signal_strength = SignalStrength.STRONG
        elif final_confidence >= 0.70:
            signal_strength = SignalStrength.MEDIUM
        else:
            signal_strength = SignalStrength.WEAK
        
        # Шаг 7: Формирование reason
        reason_parts = [
            f"Imbalance {signal_type.value}: {imbalance_strength:.2f}",
            f"Volume delta: {volume_delta_usdt:+,.0f} USDT",
            f"Liquidity quality: {analysis.liquidity_quality:.2f}",
            f"Market pressure: {analysis.market_pressure} ({analysis.pressure_strength:.2f})"
        ]
        
        if walls['bid_walls'] or walls['ask_walls']:
            reason_parts.append(
                f"Walls: bid={len(walls['bid_walls'])}, ask={len(walls['ask_walls'])}"
            )
        
        # Шаг 8: Создание сигнала
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            source=SignalSource.STRATEGY,
            strength=signal_strength,
            price=current_price,
            confidence=final_confidence,
            timestamp=int(datetime.now().timestamp() * 1000),
            reason=" | ".join(reason_parts),
            metadata={
                'strategy': self.strategy_name,
                'imbalance': metrics.imbalance,
                'imbalance_depth_5': getattr(metrics, 'imbalance_depth_5', None),
                'imbalance_depth_10': getattr(metrics, 'imbalance_depth_10', None),
                'volume_delta_usdt': volume_delta_usdt,
                'liquidity_quality': analysis.liquidity_quality,
                'market_pressure': analysis.market_pressure,
                'has_bid_walls': len(walls['bid_walls']) > 0,
                'has_ask_walls': len(walls['ask_walls']) > 0,
                'total_bid_volume': metrics.total_bid_volume,
                'total_ask_volume': metrics.total_ask_volume
            }
        )
        
        # Сохраняем активный сигнал
        self.active_signals[symbol] = signal
        self.signals_generated += 1
        self.imbalance_signals += 1
        
        logger.info(
            f"🎯 IMBALANCE SIGNAL [{symbol}]: {signal_type.value}, "
            f"confidence={final_confidence:.2f}, "
            f"imbalance={metrics.imbalance:.2f}, "
            f"volume_delta={volume_delta_usdt:+,.0f} USDT"
        )
        
        return signal

    def _analyze_multi_depth_imbalance(
        self, 
        symbol: str, 
        metrics: OrderBookMetrics
    ) -> Dict:
        """
        Анализ дисбаланса на нескольких уровнях глубины.

        Returns:
            {
                'has_signal': bool,
                'signal_type': SignalType,
                'strength': float,
                'levels': {...}
            }
        """
        # Основной дисбаланс (весь стакан)
        main_imbalance = metrics.imbalance
        
        # Дисбаланс на глубине 5
        imbalance_5 = getattr(metrics, 'imbalance_depth_5', main_imbalance)
        
        # Дисбаланс на глубине 10
        imbalance_10 = getattr(metrics, 'imbalance_depth_10', main_imbalance)
        
        # Взвешенный композитный дисбаланс (больший вес ближним уровням)
        composite_imbalance = (
            imbalance_5 * 0.5 +
            imbalance_10 * 0.3 +
            main_imbalance * 0.2
        )
        
        # Проверка условий BUY
        if composite_imbalance >= self.config.imbalance_buy_threshold:
            # Подтверждение: хотя бы 2 из 3 уровней поддерживают BUY
            supports = sum([
                main_imbalance > 0.65,
                imbalance_5 > 0.70,
                imbalance_10 > 0.70
            ])
            
            if supports >= 2:
                # Сила основана на том, насколько превышен порог
                strength = min((composite_imbalance - 0.5) * 2.0, 1.0)
                
                return {
                    'has_signal': True,
                    'signal_type': SignalType.BUY,
                    'strength': strength,
                    'levels': {
                        'main': main_imbalance,
                        'depth_5': imbalance_5,
                        'depth_10': imbalance_10,
                        'composite': composite_imbalance
                    }
                }
        
        # Проверка условий SELL
        elif composite_imbalance <= self.config.imbalance_sell_threshold:
            # Подтверждение: хотя бы 2 из 3 уровней поддерживают SELL
            supports = sum([
                main_imbalance < 0.35,
                imbalance_5 < 0.30,
                imbalance_10 < 0.30
            ])
            
            if supports >= 2:
                # Сила основана на том, насколько ниже порог
                strength = min((0.5 - composite_imbalance) * 2.0, 1.0)
                
                return {
                    'has_signal': True,
                    'signal_type': SignalType.SELL,
                    'strength': strength,
                    'levels': {
                        'main': main_imbalance,
                        'depth_5': imbalance_5,
                        'depth_10': imbalance_10,
                        'composite': composite_imbalance
                    }
                }
        
        return {'has_signal': False}

    def _calculate_volume_delta_usdt(
        self, 
        symbol: str, 
        current_price: float
    ) -> float:
        """
        Рассчитать volume delta в USDT за последние N snapshot'ов.

        Returns:
            Volume delta в USDT (положительный = bid доминирует)
        """
        volume_delta = self.calculate_volume_delta(
            symbol, 
            lookback=self.config.volume_delta_lookback
        )
        
        # Конвертируем в USDT (приблизительно)
        volume_delta_usdt = volume_delta * current_price
        
        return volume_delta_usdt

    def _check_volume_delta_confirmation(
        self, 
        signal_type: SignalType, 
        volume_delta_usdt: float
    ) -> bool:
        """
        Проверить подтверждение через volume delta.

        Args:
            signal_type: Тип сигнала (BUY/SELL)
            volume_delta_usdt: Volume delta в USDT

        Returns:
            True если подтверждается
        """
        min_delta = self.config.min_volume_delta_usdt
        
        if signal_type == SignalType.BUY:
            # Для BUY нужен положительный volume delta
            return volume_delta_usdt >= min_delta
        
        elif signal_type == SignalType.SELL:
            # Для SELL нужен отрицательный volume delta
            return volume_delta_usdt <= -min_delta
        
        return False

    def _check_opposing_walls(
        self,
        signal_type: SignalType,
        walls: Dict[str, List[Tuple[float, float]]],
        current_price: float
    ) -> Dict:
        """
        Проверить наличие крупных противоположных стен.

        Args:
            signal_type: Тип сигнала
            walls: Словарь с bid/ask стенами
            current_price: Текущая цена

        Returns:
            {'passed': bool, 'reason': str}
        """
        bid_walls = walls['bid_walls']
        ask_walls = walls['ask_walls']
        
        # Вычисляем общий объем стен в USDT
        total_bid_wall_usdt = sum(price * vol for price, vol in bid_walls)
        total_ask_wall_usdt = sum(price * vol for price, vol in ask_walls)
        
        if signal_type == SignalType.BUY:
            # Для BUY: проверяем нет ли крупных ask стен (сопротивление)
            if ask_walls:
                # Вычисляем соотношение ask walls к bid walls
                if total_bid_wall_usdt > 0:
                    wall_ratio = total_ask_wall_usdt / total_bid_wall_usdt
                    
                    if wall_ratio > self.config.max_opposing_wall_ratio:
                        return {
                            'passed': False,
                            'reason': f"Ask wall слишком крупная (ratio={wall_ratio:.2f})"
                        }
        
        elif signal_type == SignalType.SELL:
            # Для SELL: проверяем нет ли крупных bid стен (поддержка)
            if bid_walls:
                # Вычисляем соотношение bid walls к ask walls
                if total_ask_wall_usdt > 0:
                    wall_ratio = total_bid_wall_usdt / total_ask_wall_usdt
                    
                    if wall_ratio > self.config.max_opposing_wall_ratio:
                        return {
                            'passed': False,
                            'reason': f"Bid wall слишком крупная (ratio={wall_ratio:.2f})"
                        }
        
        return {'passed': True, 'reason': 'OK'}

    def _calculate_signal_confidence(
        self,
        imbalance_strength: float,
        volume_delta_usdt: float,
        liquidity_quality: float,
        market_pressure: str,
        pressure_strength: float,
        signal_type: SignalType
    ) -> float:
        """
        Вычислить итоговую confidence для сигнала.

        Композитная метрика из:
        - Сила дисбаланса (40%)
        - Величина volume delta (30%)
        - Качество ликвидности (20%)
        - Согласованность с market pressure (10%)
        """
        # Компонент 1: Имбаланс (40%)
        imbalance_component = imbalance_strength * 0.4
        
        # Компонент 2: Volume delta (30%)
        # Нормализуем volume delta (100k USDT = 1.0)
        volume_delta_normalized = min(
            abs(volume_delta_usdt) / 100000.0, 
            1.0
        )
        volume_delta_component = volume_delta_normalized * 0.3
        
        # Компонент 3: Качество ликвидности (20%)
        liquidity_component = liquidity_quality * 0.2
        
        # Компонент 4: Согласованность с давлением (10%)
        pressure_component = 0.0
        
        if market_pressure == "bullish" and signal_type == SignalType.BUY:
            pressure_component = pressure_strength * 0.1
        elif market_pressure == "bearish" and signal_type == SignalType.SELL:
            pressure_component = pressure_strength * 0.1
        elif market_pressure == "neutral":
            # Нейтральное давление - небольшой бонус за стабильность
            pressure_component = 0.05
        # Если противоположное давление - не добавляем ничего
        
        # Итоговая confidence
        confidence = (
            imbalance_component +
            volume_delta_component +
            liquidity_component +
            pressure_component
        )
        
        # Применяем базовую confidence как множитель
        confidence = confidence * (self.config.base_confidence / 0.7)
        
        return min(confidence, 1.0)

    def get_statistics(self) -> Dict:
        """Получить расширенную статистику."""
        base_stats = super().get_statistics()
        base_stats.update({
            'imbalance_signals': self.imbalance_signals,
            'wall_blocks': self.wall_blocks
        })
        return base_stats
