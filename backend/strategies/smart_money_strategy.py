"""
Smart Money Strategy (Hybrid) - следование за институциональными игроками.

Методология:
- Multi-Signal подход:
  1. Определение тренда через свечной анализ (SuperTrend, ADX, MA)
  2. Поиск точек входа через микроструктуру стакана
  3. Подтверждение через Volume Profile и ML предсказания
- Комбинирует технический анализ свечей с анализом стакана
- Фильтрация через детекцию манипуляций
- Адаптивное управление рисками

Путь: backend/strategies/smart_money_strategy.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np

from backend.core.logger import get_logger
from backend.models.signal import TradingSignal, SignalType, SignalStrength, SignalSource
from backend.models.orderbook import OrderBookSnapshot, OrderBookMetrics
from backend.strategy.candle_manager import Candle
from backend.strategies.base_orderbook_strategy import BaseOrderBookStrategy

logger = get_logger(__name__)


class TrendDirection(Enum):
    """Направление тренда."""
    STRONG_UP = "strong_uptrend"
    WEAK_UP = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWN = "weak_downtrend"
    STRONG_DOWN = "strong_downtrend"


class MarketPhase(Enum):
    """Фаза рынка."""
    ACCUMULATION = "accumulation"  # Умные деньги накапливают
    MARKUP = "markup"  # Рост
    DISTRIBUTION = "distribution"  # Умные деньги распределяют
    MARKDOWN = "markdown"  # Падение


@dataclass
class SmartMoneyConfig:
    """Конфигурация Smart Money стратегии."""
    # Trend detection (свечи)
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    adx_period: int = 14
    adx_strong_threshold: float = 25.0
    adx_weak_threshold: float = 15.0
    ema_fast: int = 9
    ema_slow: int = 21
    
    # Volatility (ATR)
    atr_period: int = 14
    atr_multiplier_stop: float = 2.0
    atr_multiplier_target: float = 3.0
    
    # OrderBook entry signals
    imbalance_entry_threshold: float = 0.70  # Сильный имбаланс для входа
    min_wall_size_usdt: float = 80000.0  # Минимальный размер стены
    whale_detection_percentile: float = 90.0
    
    # Volume Profile
    use_volume_profile: bool = True
    poc_confirmation_distance_pct: float = 0.5
    
    # ML Integration
    use_ml_validation: bool = True
    min_ml_confidence: float = 0.7
    
    # Multi-signal consensus
    min_signals_required: int = 2  # Минимум сигналов из 3 этапов
    
    # Risk management
    max_risk_per_trade_pct: float = 1.0
    trailing_stop_activation_pct: float = 2.0


@dataclass
class TrendAnalysis:
    """Результат анализа тренда."""
    direction: TrendDirection
    strength: float  # 0-1
    supertrend_signal: str  # "bullish", "bearish", "neutral"
    adx_value: float
    ema_alignment: bool  # Fast EMA > Slow EMA для uptrend
    atr_value: float


@dataclass
class EntryPoint:
    """Точка входа от анализа стакана."""
    has_entry: bool
    signal_type: Optional[SignalType]
    confidence: float
    reasons: List[str]
    supporting_factors: Dict


class SmartMoneyStrategy(BaseOrderBookStrategy):
    """
    Hybrid стратегия следования за институциональными игроками.
    
    Философия:
    Умные деньги (институциональные игроки) оставляют следы:
    1. В трендах на свечах (они формируют тренды)
    2. В стакане (крупные заявки, дисбаланс)
    3. В объемах (накопление/распределение)
    
    Мы комбинируем все три источника для высокоточных входов.
    """

    def __init__(self, config: SmartMoneyConfig):
        """
        Инициализация стратегии.

        Args:
            config: Конфигурация
        """
        super().__init__("smart_money")
        self.config = config
        
        # Кэш для индикаторов
        self.indicator_cache: Dict[str, Dict] = {}
        
        # Статистика
        self.trend_entries = 0
        self.pullback_entries = 0
        self.breakout_entries = 0
        
        logger.info(
            f"Инициализирована SmartMoneyStrategy (Hybrid): "
            f"adx_threshold={config.adx_strong_threshold}, "
            f"ml_enabled={config.use_ml_validation}"
        )

    def analyze(
        self,
        symbol: str,
        candles: List[Candle],
        current_price: float,
        orderbook: OrderBookSnapshot,
        metrics: OrderBookMetrics,
        volume_profile: Optional[Dict] = None,
        ml_prediction: Optional[Dict] = None
    ) -> Optional[TradingSignal]:
        """
        Трехэтапный анализ для генерации сигнала.

        Args:
            symbol: Торговая пара
            candles: История свечей
            current_price: Текущая цена
            orderbook: Снимок стакана
            metrics: Метрики стакана
            volume_profile: Volume profile (опционально)
            ml_prediction: ML предсказание (опционально)

        Returns:
            TradingSignal или None
        """
        if len(candles) < 100:
            return None
        
        # ==================== ЭТАП 1: ОПРЕДЕЛЕНИЕ ТРЕНДА (СВЕЧИ) ====================
        trend_analysis = self._analyze_trend(symbol, candles, current_price)
        
        logger.debug(
            f"[{self.strategy_name}] {symbol} | "
            f"Trend: {trend_analysis.direction.value}, "
            f"strength={trend_analysis.strength:.2f}, "
            f"ADX={trend_analysis.adx_value:.1f}"
        )
        
        # Фильтр: Только сильные тренды или ranging для mean reversion
        if trend_analysis.direction == TrendDirection.RANGING:
            # В ranging не торгуем по Smart Money (требуется тренд)
            logger.debug(
                f"[{self.strategy_name}] {symbol} | Ranging market - пропускаем"
            )
            return None
        
        # ==================== ЭТАП 2: ТОЧКА ВХОДА (СТАКАН) ====================
        orderbook_analysis = self.analyze_orderbook_quality(
            symbol, orderbook, metrics
        )
        
        # Фильтр: Манипуляции
        if orderbook_analysis.has_manipulation:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | "
                f"Манипуляция: {orderbook_analysis.manipulation_type} - БЛОКИРУЕМ"
            )
            self.manipulation_blocks += 1
            return None
        
        entry_point = self._find_entry_point(
            symbol=symbol,
            trend_analysis=trend_analysis,
            orderbook=orderbook,
            metrics=metrics,
            orderbook_analysis=orderbook_analysis,
            current_price=current_price
        )
        
        if not entry_point.has_entry:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | Точка входа не найдена"
            )
            return None
        
        # ==================== ЭТАП 3: ПОДТВЕРЖДЕНИЕ (VOLUME PROFILE + ML) ====================
        confirmation = self._get_confirmation(
            symbol=symbol,
            signal_type=entry_point.signal_type,
            current_price=current_price,
            volume_profile=volume_profile,
            ml_prediction=ml_prediction
        )
        
        # Проверка минимального количества подтверждений
        signals_count = sum([
            True,  # Trend всегда есть
            entry_point.has_entry,  # Entry point найден
            confirmation['volume_profile_confirms'],
            confirmation['ml_confirms']
        ])
        
        if signals_count < self.config.min_signals_required:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | "
                f"Недостаточно подтверждений: {signals_count}/{self.config.min_signals_required}"
            )
            return None
        
        # ==================== РАСЧЕТ ИТОГОВОЙ CONFIDENCE ====================
        final_confidence = self._calculate_final_confidence(
            trend_analysis=trend_analysis,
            entry_point=entry_point,
            confirmation=confirmation,
            orderbook_quality=orderbook_analysis.liquidity_quality
        )
        
        if final_confidence < 0.65:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | "
                f"Низкая итоговая confidence: {final_confidence:.2f}"
            )
            return None
        
        # ==================== ФОРМИРОВАНИЕ СИГНАЛА ====================
        signal_type = entry_point.signal_type
        
        # Определение силы сигнала
        if final_confidence >= 0.85:
            signal_strength = SignalStrength.STRONG
        elif final_confidence >= 0.75:
            signal_strength = SignalStrength.MEDIUM
        else:
            signal_strength = SignalStrength.WEAK
        
        # Формирование reason
        reason_parts = [
            f"Smart Money {signal_type.value}",
            f"Trend: {trend_analysis.direction.value} (ADX={trend_analysis.adx_value:.1f})",
            f"Entry: {', '.join(entry_point.reasons[:2])}"  # Первые 2 причины
        ]
        
        if confirmation['volume_profile_confirms']:
            reason_parts.append("POC confirms")
        
        if confirmation['ml_confirms']:
            reason_parts.append(f"ML confidence={confirmation['ml_confidence']:.2f}")
        
        # ==================== УПРАВЛЕНИЕ РИСКАМИ ====================
        risk_params = self._calculate_risk_parameters(
            signal_type=signal_type,
            current_price=current_price,
            atr_value=trend_analysis.atr_value,
            orderbook=orderbook
        )
        
        # ==================== СОЗДАНИЕ СИГНАЛА ====================
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
                'trend_direction': trend_analysis.direction.value,
                'trend_strength': trend_analysis.strength,
                'adx': trend_analysis.adx_value,
                'atr': trend_analysis.atr_value,
                'entry_confidence': entry_point.confidence,
                'entry_factors': entry_point.supporting_factors,
                'ml_confidence': confirmation.get('ml_confidence'),
                'stop_loss_pct': risk_params['stop_loss_pct'],
                'take_profit_pct': risk_params['take_profit_pct'],
                'trailing_stop_enabled': True
            }
        )
        
        self.active_signals[symbol] = signal
        self.signals_generated += 1
        
        # Классифицируем тип входа
        if 'pullback' in entry_point.reasons[0].lower():
            self.pullback_entries += 1
        elif 'breakout' in entry_point.reasons[0].lower():
            self.breakout_entries += 1
        else:
            self.trend_entries += 1
        
        logger.info(
            f"💎 SMART MONEY SIGNAL [{symbol}]: {signal_type.value}, "
            f"confidence={final_confidence:.2f}, "
            f"trend={trend_analysis.direction.value}, "
            f"ADX={trend_analysis.adx_value:.1f}"
        )
        
        return signal

    def _analyze_trend(
        self, 
        symbol: str, 
        candles: List[Candle], 
        current_price: float
    ) -> TrendAnalysis:
        """
        Этап 1: Комплексный анализ тренда.
        
        Используем:
        - SuperTrend для направления
        - ADX для силы тренда
        - EMA alignment для подтверждения
        - ATR для волатильности
        """
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        
        # 1. SuperTrend
        supertrend_signal, supertrend_line = self._calculate_supertrend(
            highs, lows, closes,
            period=self.config.supertrend_period,
            multiplier=self.config.supertrend_multiplier
        )
        
        # 2. ADX (Average Directional Index)
        adx_value = self._calculate_adx(highs, lows, closes, self.config.adx_period)
        
        # 3. EMA alignment
        ema_fast = self._calculate_ema(closes, self.config.ema_fast)
        ema_slow = self._calculate_ema(closes, self.config.ema_slow)
        ema_alignment = ema_fast[-1] > ema_slow[-1] if len(ema_fast) > 0 else False
        
        # 4. ATR (Average True Range)
        atr_value = self._calculate_atr(highs, lows, closes, self.config.atr_period)
        
        # Определение направления тренда
        if supertrend_signal == "bullish":
            if adx_value > self.config.adx_strong_threshold and ema_alignment:
                direction = TrendDirection.STRONG_UP
                strength = min(adx_value / 50.0, 1.0)  # Нормализуем
            elif adx_value > self.config.adx_weak_threshold:
                direction = TrendDirection.WEAK_UP
                strength = adx_value / 50.0
            else:
                direction = TrendDirection.RANGING
                strength = 0.3
        elif supertrend_signal == "bearish":
            if adx_value > self.config.adx_strong_threshold and not ema_alignment:
                direction = TrendDirection.STRONG_DOWN
                strength = min(adx_value / 50.0, 1.0)
            elif adx_value > self.config.adx_weak_threshold:
                direction = TrendDirection.WEAK_DOWN
                strength = adx_value / 50.0
            else:
                direction = TrendDirection.RANGING
                strength = 0.3
        else:
            direction = TrendDirection.RANGING
            strength = 0.3
        
        return TrendAnalysis(
            direction=direction,
            strength=strength,
            supertrend_signal=supertrend_signal,
            adx_value=adx_value,
            ema_alignment=ema_alignment,
            atr_value=atr_value
        )

    def _find_entry_point(
        self,
        symbol: str,
        trend_analysis: TrendAnalysis,
        orderbook: OrderBookSnapshot,
        metrics: OrderBookMetrics,
        orderbook_analysis,
        current_price: float
    ) -> EntryPoint:
        """
        Этап 2: Поиск оптимальной точки входа через стакан.
        
        Стратегии входа:
        1. Pullback к зоне ликвидности в направлении тренда
        2. Появление крупной стены в направлении тренда
        3. Агрессивные market orders в направлении тренда
        """
        reasons = []
        supporting_factors = {}
        entry_confidence = 0.0
        
        # Определяем желаемое направление на основе тренда
        if trend_analysis.direction in [TrendDirection.STRONG_UP, TrendDirection.WEAK_UP]:
            desired_direction = SignalType.BUY
        elif trend_analysis.direction in [TrendDirection.STRONG_DOWN, TrendDirection.WEAK_DOWN]:
            desired_direction = SignalType.SELL
        else:
            return EntryPoint(
                has_entry=False,
                signal_type=None,
                confidence=0.0,
                reasons=[],
                supporting_factors={}
            )
        
        # ========== Фактор 1: Imbalance в направлении тренда (40%) ==========
        imbalance_score = 0.0
        
        if desired_direction == SignalType.BUY:
            if metrics.imbalance >= self.config.imbalance_entry_threshold:
                imbalance_score = 0.4
                reasons.append("Strong bid imbalance")
                supporting_factors['imbalance'] = metrics.imbalance
        else:
            if metrics.imbalance <= (1.0 - self.config.imbalance_entry_threshold):
                imbalance_score = 0.4
                reasons.append("Strong ask imbalance")
                supporting_factors['imbalance'] = metrics.imbalance
        
        entry_confidence += imbalance_score
        
        # ========== Фактор 2: Крупные стены в направлении тренда (30%) ==========
        walls = self.check_large_walls(orderbook, self.config.min_wall_size_usdt)
        wall_score = 0.0
        
        if desired_direction == SignalType.BUY and walls['bid_walls']:
            # Крупные bid стены = институционалы готовятся покупать
            largest_bid_wall = max(walls['bid_walls'], key=lambda x: x[1])
            wall_score = 0.3
            reasons.append(f"Large bid wall at ${largest_bid_wall[0]:.2f}")
            supporting_factors['bid_wall_size'] = largest_bid_wall[0] * largest_bid_wall[1]
        elif desired_direction == SignalType.SELL and walls['ask_walls']:
            # Крупные ask стены = институционалы готовятся продавать
            largest_ask_wall = max(walls['ask_walls'], key=lambda x: x[1])
            wall_score = 0.3
            reasons.append(f"Large ask wall at ${largest_ask_wall[0]:.2f}")
            supporting_factors['ask_wall_size'] = largest_ask_wall[0] * largest_ask_wall[1]
        
        entry_confidence += wall_score
        
        # ========== Фактор 3: Market pressure согласуется (30%) ==========
        pressure_score = 0.0
        
        if desired_direction == SignalType.BUY and orderbook_analysis.market_pressure == "bullish":
            pressure_score = 0.3 * orderbook_analysis.pressure_strength
            reasons.append("Bullish market pressure")
            supporting_factors['market_pressure'] = orderbook_analysis.pressure_strength
        elif desired_direction == SignalType.SELL and orderbook_analysis.market_pressure == "bearish":
            pressure_score = 0.3 * orderbook_analysis.pressure_strength
            reasons.append("Bearish market pressure")
            supporting_factors['market_pressure'] = orderbook_analysis.pressure_strength
        
        entry_confidence += pressure_score

        # ========== НОВОЕ: Фактор 4: Institutional Activity Detection (до +0.3) ==========
        institutional_score = 0.0

        if self.trade_manager:
            try:
                stats = self.trade_manager.get_statistics(window_seconds=300)  # 5 минут

                # Признаки "умных денег" (institutional players):

                # 1. Block Trades (крупные институциональные сделки)
                if stats.block_trade_count > 3:
                    institutional_score += 0.15
                    reasons.append(f"Institutional block trades: {stats.block_trade_count}")
                    supporting_factors['block_trades'] = stats.block_trade_count

                # 2. High Order Flow Toxicity (информированные трейдеры)
                if abs(stats.order_flow_toxicity) > 0.6:
                    institutional_score += 0.10
                    reasons.append(f"High order flow toxicity: {stats.order_flow_toxicity:.2f}")
                    supporting_factors['toxicity'] = stats.order_flow_toxicity

                # 3. Sustained Directional Pressure (устойчивое давление в одну сторону)
                if desired_direction == SignalType.BUY and stats.buy_sell_ratio > 2.0:
                    institutional_score += 0.05
                    reasons.append(f"Strong buy pressure: {stats.buy_sell_ratio:.1f}")
                    supporting_factors['buy_sell_ratio'] = stats.buy_sell_ratio
                elif desired_direction == SignalType.SELL and stats.buy_sell_ratio < 0.5:
                    institutional_score += 0.05
                    reasons.append(f"Strong sell pressure: {stats.buy_sell_ratio:.1f}")
                    supporting_factors['buy_sell_ratio'] = stats.buy_sell_ratio

            except Exception:
                pass  # Нет данных - продолжаем без institutional score

        entry_confidence += institutional_score

        # Проверка минимальной entry confidence
        has_entry = entry_confidence >= 0.5 and len(reasons) >= 2

        return EntryPoint(
            has_entry=has_entry,
            signal_type=desired_direction if has_entry else None,
            confidence=entry_confidence,
            reasons=reasons,
            supporting_factors=supporting_factors
        )

    def _get_confirmation(
        self,
        symbol: str,
        signal_type: Optional[SignalType],
        current_price: float,
        volume_profile: Optional[Dict],
        ml_prediction: Optional[Dict]
    ) -> Dict:
        """
        Этап 3: Получение подтверждений от Volume Profile и ML.
        """
        confirmation = {
            'volume_profile_confirms': False,
            'ml_confirms': False,
            'ml_confidence': 0.0
        }
        
        # ========== Volume Profile подтверждение ==========
        if self.config.use_volume_profile and volume_profile:
            poc_price = volume_profile.get('poc_price')
            
            if poc_price:
                distance_pct = abs(current_price - poc_price) / current_price * 100
                
                # POC поддерживает направление если:
                # - BUY: POC ниже текущей цены (цена выше равновесия)
                # - SELL: POC выше текущей цены (цена ниже равновесия)
                
                if signal_type == SignalType.BUY and poc_price < current_price:
                    if distance_pct <= self.config.poc_confirmation_distance_pct:
                        confirmation['volume_profile_confirms'] = True
                elif signal_type == SignalType.SELL and poc_price > current_price:
                    if distance_pct <= self.config.poc_confirmation_distance_pct:
                        confirmation['volume_profile_confirms'] = True
        
        # ========== ML подтверждение ==========
        if self.config.use_ml_validation and ml_prediction:
            ml_confidence = ml_prediction.get('confidence', 0.0)
            ml_direction = ml_prediction.get('prediction')  # 'bullish' или 'bearish'
            
            if ml_confidence >= self.config.min_ml_confidence:
                # Проверяем согласуется ли ML с нашим сигналом
                if signal_type == SignalType.BUY and ml_direction == 'bullish':
                    confirmation['ml_confirms'] = True
                    confirmation['ml_confidence'] = ml_confidence
                elif signal_type == SignalType.SELL and ml_direction == 'bearish':
                    confirmation['ml_confirms'] = True
                    confirmation['ml_confidence'] = ml_confidence
        
        return confirmation

    def _calculate_final_confidence(
        self,
        trend_analysis: TrendAnalysis,
        entry_point: EntryPoint,
        confirmation: Dict,
        orderbook_quality: float
    ) -> float:
        """
        Вычислить итоговую confidence из всех компонентов.
        """
        # Компонент 1: Сила тренда (30%)
        trend_component = trend_analysis.strength * 0.3
        
        # Компонент 2: Entry point confidence (40%)
        entry_component = entry_point.confidence * 0.4
        
        # Компонент 3: Подтверждения (20%)
        confirmation_score = 0.0
        if confirmation['volume_profile_confirms']:
            confirmation_score += 0.5
        if confirmation['ml_confirms']:
            confirmation_score += 0.5
        confirmation_component = confirmation_score * 0.2
        
        # Компонент 4: Качество ликвидности (10%)
        liquidity_component = orderbook_quality * 0.1
        
        final_confidence = (
            trend_component +
            entry_component +
            confirmation_component +
            liquidity_component
        )
        
        return min(final_confidence, 1.0)

    def _calculate_risk_parameters(
        self,
        signal_type: SignalType,
        current_price: float,
        atr_value: float,
        orderbook: OrderBookSnapshot
    ) -> Dict:
        """
        Рассчитать параметры риск-менеджмента.
        
        Stop-loss: на основе ATR
        Take-profit: risk/reward 2:1 или 3:1
        """
        # Stop-loss на основе ATR
        stop_distance = atr_value * self.config.atr_multiplier_stop
        stop_loss_pct = (stop_distance / current_price) * 100
        
        # Take-profit
        take_profit_distance = atr_value * self.config.atr_multiplier_target
        take_profit_pct = (take_profit_distance / current_price) * 100
        
        return {
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'atr_value': atr_value,
            'trailing_activation_pct': self.config.trailing_stop_activation_pct
        }

    # ==================== ИНДИКАТОРЫ ====================

    def _calculate_supertrend(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray,
        period: int,
        multiplier: float
    ) -> Tuple[str, np.ndarray]:
        """
        SuperTrend индикатор.
        
        Returns:
            (signal, supertrend_line)
        """
        # ATR для SuperTrend
        atr = self._calculate_atr(highs, lows, closes, period)
        
        hl_avg = (highs + lows) / 2
        
        # Базовые верхний и нижний диапазоны
        basic_ub = hl_avg + multiplier * atr
        basic_lb = hl_avg - multiplier * atr
        
        # Final bands с учетом предыдущих значений
        final_ub = np.zeros_like(basic_ub)
        final_lb = np.zeros_like(basic_lb)
        supertrend = np.zeros_like(closes)
        
        for i in range(len(closes)):
            if i == 0:
                final_ub[i] = basic_ub[i]
                final_lb[i] = basic_lb[i]
            else:
                final_ub[i] = basic_ub[i] if basic_ub[i] < final_ub[i-1] or closes[i-1] > final_ub[i-1] else final_ub[i-1]
                final_lb[i] = basic_lb[i] if basic_lb[i] > final_lb[i-1] or closes[i-1] < final_lb[i-1] else final_lb[i-1]
            
            # Определение SuperTrend
            if i == 0:
                supertrend[i] = final_ub[i]
            else:
                if supertrend[i-1] == final_ub[i-1] and closes[i] <= final_ub[i]:
                    supertrend[i] = final_ub[i]
                elif supertrend[i-1] == final_ub[i-1] and closes[i] > final_ub[i]:
                    supertrend[i] = final_lb[i]
                elif supertrend[i-1] == final_lb[i-1] and closes[i] >= final_lb[i]:
                    supertrend[i] = final_lb[i]
                elif supertrend[i-1] == final_lb[i-1] and closes[i] < final_lb[i]:
                    supertrend[i] = final_ub[i]
        
        # Текущий сигнал
        if closes[-1] > supertrend[-1]:
            signal = "bullish"
        elif closes[-1] < supertrend[-1]:
            signal = "bearish"
        else:
            signal = "neutral"
        
        return signal, supertrend

    def _calculate_adx(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int
    ) -> float:
        """Average Directional Index."""
        if len(highs) < period + 1:
            return 0.0

        # True Range
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                abs(highs[1:] - closes[:-1]),
                abs(lows[1:] - closes[:-1])
            )
        )

        # Directional Movement
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed values
        atr_smooth = np.zeros(len(tr))
        plus_di_smooth = np.zeros(len(plus_dm))
        minus_di_smooth = np.zeros(len(minus_dm))

        atr_smooth[period-1] = np.mean(tr[:period])
        plus_di_smooth[period-1] = np.mean(plus_dm[:period])
        minus_di_smooth[period-1] = np.mean(minus_dm[:period])

        for i in range(period, len(tr)):
            atr_smooth[i] = (atr_smooth[i-1] * (period - 1) + tr[i]) / period
            plus_di_smooth[i] = (plus_di_smooth[i-1] * (period - 1) + plus_dm[i]) / period
            minus_di_smooth[i] = (minus_di_smooth[i-1] * (period - 1) + minus_dm[i]) / period

        # DI (защита от деления на ноль)
        # Заменяем нулевые значения atr_smooth на маленькое число
        atr_smooth_safe = np.where(atr_smooth > 0, atr_smooth, 1e-10)
        plus_di = 100 * plus_di_smooth / atr_smooth_safe
        minus_di = 100 * minus_di_smooth / atr_smooth_safe

        # DX (защита от деления на ноль)
        # np.where всё равно вычисляет оба условия, поэтому заменяем нули ДО деления

        dx_denom = plus_di + minus_di
        dx_denom_safe = np.where(dx_denom > 0, dx_denom, 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / dx_denom_safe

        # ADX (сглаженный DX)
        adx = np.mean(dx[-period:])

        return float(adx)

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        if len(prices) < period:
            return np.array([])
        
        ema = np.zeros(len(prices))
        ema[period-1] = np.mean(prices[:period])
        
        multiplier = 2.0 / (period + 1)
        
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema

    def _calculate_atr(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray, 
        period: int
    ) -> float:
        """Average True Range."""
        if len(highs) < period + 1:
            return 0.0
        
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                abs(highs[1:] - closes[:-1]),
                abs(lows[1:] - closes[:-1])
            )
        )
        
        atr = np.mean(tr[-period:])
        return float(atr)

    def get_statistics(self) -> Dict:
        """Получить расширенную статистику."""
        base_stats = super().get_statistics()
        base_stats.update({
            'trend_entries': self.trend_entries,
            'pullback_entries': self.pullback_entries,
            'breakout_entries': self.breakout_entries
        })
        return base_stats
