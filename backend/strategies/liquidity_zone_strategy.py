"""
Liquidity Zone Strategy - торговля от зон высокой ликвидности.

Методология:
- Использование S/R уровней из SRLevelDetector
- Анализ High Volume Nodes (HVN) и Low Volume Nodes (LVN)
- Point of Control (POC) из volume profile
- Mean Reversion от HVN
- Breakout через LVN с объемным подтверждением
- Rejection паттерны на уровнях

Путь: backend/strategies/liquidity_zone_strategy.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from core.logger import get_logger
from models.signal import TradingSignal, SignalType, SignalStrength, SignalSource
from models.orderbook import OrderBookSnapshot, OrderBookMetrics
from strategy.candle_manager import Candle
from strategies.base_orderbook_strategy import BaseOrderBookStrategy
# Импортируем SRLevel и SRLevelDetector, если нужны типы
# from ml_engine.detection.sr_level_detector import SRLevel, SRLevelDetector

logger = get_logger(__name__)


@dataclass
class LiquidityZoneConfig:
    """Конфигурация Liquidity Zone стратегии."""
    # Зоны ликвидности
    hvn_distance_threshold_pct: float = 0.5  # Макс 0.5% до HVN для mean reversion
    lvn_breakout_volume_multiplier: float = 1.5  # Volume для подтверждения breakout
    
    # S/R уровни
    use_sr_levels: bool = True
    min_sr_level_strength: float = 0.5  # Минимальная сила S/R уровня
    sr_touch_tolerance_pct: float = 0.1  # 0.1% для определения "касания"
    
    # Point of Control (POC)
    use_poc: bool = True
    poc_distance_threshold_pct: float = 1.0  # Дистанция до POC для сигнала
    
    # Rejection detection
    rejection_candles: int = 3  # Количество свечей для детекции rejection
    rejection_body_ratio: float = 0.3  # Макс размер тела для rejection candle
    
    # Mean reversion
    mean_reversion_enabled: bool = True
    reversion_confidence_base: float = 0.65
    
    # Breakout
    breakout_enabled: bool = True
    breakout_confirmation_candles: int = 2
    breakout_confidence_base: float = 0.70
    
    # Risk management
    stop_loss_beyond_level_pct: float = 0.3  # Stop за уровнем на X%


@dataclass
class LiquidityZone:
    """Зона ликвидности."""
    price: float
    zone_type: str  # "HVN" (High Volume Node) или "LVN" (Low Volume Node)
    strength: float  # 0-1
    volume: float
    source: str  # "orderbook", "sr_level", "volume_profile"
    
    # POC маркер
    is_poc: bool = False
    
    # S/R level связь
    is_support: bool = False
    is_resistance: bool = False
    
    # История взаимодействий
    touch_count: int = 0
    last_touch_timestamp: Optional[int] = None
    rejection_count: int = 0


class LiquidityZoneStrategy(BaseOrderBookStrategy):
    """
    Стратегия торговли от зон ликвидности.
    
    Принципы:
    1. Mean Reversion: отскок от HVN (high volume nodes)
    2. Breakout: пробой LVN (low volume nodes) с объемом
    3. Rejection: множественные отбои = разворот
    
    Зоны ликвидности формируются из:
    - Volume Profile (POC, HVN, LVN)
    - S/R уровни от SRLevelDetector
    - Кластеры из стакана
    """

    def __init__(self, config: LiquidityZoneConfig):
        """
        Инициализация стратегии.

        Args:
            config: Конфигурация
        """
        super().__init__("liquidity_zone")
        self.config = config
        
        # Зоны ликвидности для каждого символа
        self.liquidity_zones: Dict[str, List[LiquidityZone]] = {}
        
        # Статистика
        self.mean_reversion_signals = 0
        self.breakout_signals = 0
        self.rejection_signals = 0
        
        logger.info(
            f"Инициализирована LiquidityZoneStrategy: "
            f"mean_reversion={config.mean_reversion_enabled}, "
            f"breakout={config.breakout_enabled}"
        )

    def analyze(
        self,
        symbol: str,
        candles: List[Candle],
        current_price: float,
        orderbook: OrderBookSnapshot,
        metrics: OrderBookMetrics,
        sr_levels: Optional[List] = None,  # List[SRLevel]
        volume_profile: Optional[Dict] = None
    ) -> Optional[TradingSignal]:
        """
        Анализ зон ликвидности и генерация сигнала.

        Args:
            symbol: Торговая пара
            candles: История свечей
            current_price: Текущая цена
            orderbook: Снимок стакана
            metrics: Метрики стакана
            sr_levels: S/R уровни от SRLevelDetector (опционально)
            volume_profile: Volume profile данные (опционально)

        Returns:
            TradingSignal или None
        """
        # Проверка достаточности данных
        if len(candles) < 50:
            return None
        
        # Шаг 1: Базовый анализ качества
        analysis = self.analyze_orderbook_quality(symbol, orderbook, metrics)
        
        if analysis.has_manipulation:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | "
                f"Манипуляция: {analysis.manipulation_type} - БЛОКИРУЕМ"
            )
            self.manipulation_blocks += 1
            return None
        
        # Шаг 2: Обновление зон ликвидности
        self._update_liquidity_zones(
            symbol, 
            orderbook, 
            metrics, 
            sr_levels, 
            volume_profile,
            current_price
        )
        
        zones = self.liquidity_zones.get(symbol, [])
        
        if not zones:
            logger.debug(
                f"[{self.strategy_name}] {symbol} | Зоны ликвидности не найдены"
            )
            return None
        
        # Шаг 3: Поиск ближайших зон
        nearest_zones = self._find_nearest_zones(zones, current_price)
        
        if not nearest_zones:
            return None
        
        # Шаг 4: Детекция rejection паттернов
        rejection_detected = self._detect_rejection_pattern(
            candles, 
            nearest_zones, 
            current_price
        )
        
        # Шаг 5: Анализ типа сигнала
        signal_analysis = None
        
        # Приоритет 1: Rejection (самый сильный сигнал)
        if rejection_detected['has_rejection']:
            signal_analysis = self._analyze_rejection_signal(
                rejection_detected, 
                nearest_zones, 
                current_price
            )
            if signal_analysis and signal_analysis['has_signal']:
                self.rejection_signals += 1
        
        # Приоритет 2: Mean Reversion от HVN
        if not signal_analysis or not signal_analysis.get('has_signal'):
            if self.config.mean_reversion_enabled:
                signal_analysis = self._analyze_mean_reversion(
                    nearest_zones, 
                    current_price, 
                    analysis
                )
                if signal_analysis and signal_analysis['has_signal']:
                    self.mean_reversion_signals += 1
        
        # Приоритет 3: Breakout через LVN
        if not signal_analysis or not signal_analysis.get('has_signal'):
            if self.config.breakout_enabled:
                signal_analysis = self._analyze_breakout(
                    candles, 
                    nearest_zones, 
                    current_price, 
                    metrics
                )
                if signal_analysis and signal_analysis['has_signal']:
                    self.breakout_signals += 1
        
        if not signal_analysis or not signal_analysis.get('has_signal'):
            return None
        
        # Шаг 6: Формирование сигнала
        signal_type = signal_analysis['signal_type']
        confidence = signal_analysis['confidence']
        pattern_type = signal_analysis['pattern_type']
        
        # Проверка минимальной confidence
        if confidence < 0.6:
            return None
        
        # Определение силы
        if confidence >= 0.85:
            signal_strength = SignalStrength.STRONG
        elif confidence >= 0.70:
            signal_strength = SignalStrength.MEDIUM
        else:
            signal_strength = SignalStrength.WEAK
        
        # Reason
        reason_parts = [
            f"{pattern_type} {signal_type.value}: confidence={confidence:.2f}"
        ]
        
        involved_zones = signal_analysis.get('involved_zones', [])
        if involved_zones:
            zone_types = [z.zone_type for z in involved_zones]
            reason_parts.append(f"Zones: {', '.join(zone_types)}")
        
        # Stop-loss based on nearest level
        stop_info = self._calculate_level_based_stop(
            signal_type, 
            current_price, 
            nearest_zones
        )
        
        # Создание сигнала
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            source=SignalSource.STRATEGY,
            strength=signal_strength,
            price=current_price,
            confidence=confidence,
            timestamp=int(datetime.now().timestamp() * 1000),
            reason=" | ".join(reason_parts),
            metadata={
                'strategy': self.strategy_name,
                'pattern_type': pattern_type,
                'zones_count': len(zones),
                'nearest_hvn': self._find_nearest_hvn(zones, current_price),
                'nearest_lvn': self._find_nearest_lvn(zones, current_price),
                'stop_loss_level': stop_info['level_price'],
                'stop_loss_distance_pct': stop_info['distance_pct']
            }
        )
        
        self.active_signals[symbol] = signal
        self.signals_generated += 1
        
        logger.info(
            f"🎯 LIQUIDITY ZONE SIGNAL [{symbol}]: {signal_type.value}, "
            f"pattern={pattern_type}, "
            f"confidence={confidence:.2f}"
        )
        
        return signal

    def _update_liquidity_zones(
        self,
        symbol: str,
        orderbook: OrderBookSnapshot,
        metrics: OrderBookMetrics,
        sr_levels: Optional[List],
        volume_profile: Optional[Dict],
        current_price: float
    ):
        """
        Обновить зоны ликвидности из всех источников.
        """
        zones = []
        
        # Источник 1: Volume clusters из стакана (HVN)
        clusters = self.find_volume_clusters(orderbook, side="both")
        
        for price, volume in clusters:
            zone = LiquidityZone(
                price=price,
                zone_type="HVN",  # Кластеры = high volume
                strength=min(volume / (metrics.total_bid_volume + metrics.total_ask_volume), 1.0),
                volume=volume,
                source="orderbook"
            )
            zones.append(zone)
        
        # Источник 2: S/R уровни от SRLevelDetector
        if sr_levels and self.config.use_sr_levels:
            for level in sr_levels:
                # Проверяем минимальную силу
                if level.strength < self.config.min_sr_level_strength:
                    continue
                
                zone = LiquidityZone(
                    price=level.price,
                    zone_type="HVN",  # S/R уровни обычно HVN
                    strength=level.strength,
                    volume=level.avg_volume,
                    source="sr_level",
                    is_support=(level.level_type == "support"),
                    is_resistance=(level.level_type == "resistance"),
                    touch_count=level.touch_count
                )
                zones.append(zone)
        
        # Источник 3: Volume Profile (POC, HVN, LVN)
        if volume_profile and self.config.use_poc:
            # POC (Point of Control)
            if 'poc_price' in volume_profile:
                poc_zone = LiquidityZone(
                    price=volume_profile['poc_price'],
                    zone_type="HVN",
                    strength=1.0,  # POC = максимальная сила
                    volume=volume_profile.get('poc_volume', 0.0),
                    source="volume_profile",
                    is_poc=True
                )
                zones.append(poc_zone)
            
            # HVN nodes
            if 'hvn_nodes' in volume_profile:
                for node in volume_profile['hvn_nodes']:
                    zone = LiquidityZone(
                        price=node['price'],
                        zone_type="HVN",
                        strength=node.get('strength', 0.7),
                        volume=node.get('volume', 0.0),
                        source="volume_profile"
                    )
                    zones.append(zone)
            
            # LVN nodes
            if 'lvn_nodes' in volume_profile:
                for node in volume_profile['lvn_nodes']:
                    zone = LiquidityZone(
                        price=node['price'],
                        zone_type="LVN",
                        strength=node.get('strength', 0.5),
                        volume=node.get('volume', 0.0),
                        source="volume_profile"
                    )
                    zones.append(zone)
        
        # Удаляем дубликаты (близкие по цене зоны)
        zones = self._merge_close_zones(zones)
        
        # Сортируем по силе
        zones.sort(key=lambda z: z.strength, reverse=True)
        
        self.liquidity_zones[symbol] = zones
        
        logger.debug(
            f"[{self.strategy_name}] {symbol} | "
            f"Обновлено зон: {len(zones)} "
            f"(HVN={len([z for z in zones if z.zone_type == 'HVN'])}, "
            f"LVN={len([z for z in zones if z.zone_type == 'LVN'])})"
        )

    def _merge_close_zones(self, zones: List[LiquidityZone]) -> List[LiquidityZone]:
        """Объединить близкие зоны."""
        if not zones:
            return []
        
        merged = []
        zones_sorted = sorted(zones, key=lambda z: z.price)
        
        current_zone = zones_sorted[0]
        
        for next_zone in zones_sorted[1:]:
            distance_pct = abs(next_zone.price - current_zone.price) / current_zone.price
            
            # Если зоны близко - объединяем (берем сильнейшую)
            if distance_pct < 0.001:  # 0.1%
                if next_zone.strength > current_zone.strength:
                    current_zone = next_zone
            else:
                merged.append(current_zone)
                current_zone = next_zone
        
        merged.append(current_zone)
        return merged

    def _find_nearest_zones(
        self, 
        zones: List[LiquidityZone], 
        current_price: float,
        max_distance_pct: float = 2.0
    ) -> List[LiquidityZone]:
        """
        Найти ближайшие зоны к текущей цене.
        
        Returns:
            Список ближайших зон (в пределах max_distance_pct)
        """
        nearby = []
        
        for zone in zones:
            distance_pct = abs(zone.price - current_price) / current_price * 100
            
            if distance_pct <= max_distance_pct:
                nearby.append((zone, distance_pct))
        
        # Сортируем по дистанции
        nearby.sort(key=lambda x: x[1])
        
        return [zone for zone, _ in nearby]

    def _detect_rejection_pattern(
        self,
        candles: List[Candle],
        zones: List[LiquidityZone],
        current_price: float
    ) -> Dict:
        """
        Детекция rejection паттерна (отбой от уровня).
        
        Rejection candle: длинная тень (wick) и маленькое тело.
        """
        if len(candles) < self.config.rejection_candles:
            return {'has_rejection': False}
        
        recent_candles = candles[-self.config.rejection_candles:]
        
        for zone in zones:
            # Проверяем коснулась ли цена зоны
            touches = 0
            rejection_candles_found = []
            
            for candle in recent_candles:
                # Проверка касания зоны
                touched = False
                if zone.is_support or zone.price < current_price:
                    # Support zone - проверяем low
                    if abs(candle.low - zone.price) / zone.price < self.config.sr_touch_tolerance_pct / 100:
                        touched = True
                else:
                    # Resistance zone - проверяем high
                    if abs(candle.high - zone.price) / zone.price < self.config.sr_touch_tolerance_pct / 100:
                        touched = True
                
                if touched:
                    touches += 1
                    
                    # Проверяем является ли это rejection candle
                    body_size = abs(candle.close - candle.open)
                    total_range = candle.high - candle.low
                    
                    if total_range > 0:
                        body_ratio = body_size / total_range
                        
                        # Rejection = маленькое тело
                        if body_ratio < self.config.rejection_body_ratio:
                            rejection_candles_found.append(candle)
            
            # Если было несколько касаний с rejection - это сигнал
            if touches >= 2 and len(rejection_candles_found) >= 1:
                # Определяем направление отскока
                if zone.price < current_price:
                    # Отскок от support = BUY
                    signal_type = SignalType.BUY
                else:
                    # Отскок от resistance = SELL
                    signal_type = SignalType.SELL
                
                return {
                    'has_rejection': True,
                    'zone': zone,
                    'touches': touches,
                    'rejection_candles': len(rejection_candles_found),
                    'signal_type': signal_type
                }
        
        return {'has_rejection': False}

    def _analyze_rejection_signal(
        self,
        rejection_data: Dict,
        zones: List[LiquidityZone],
        current_price: float
    ) -> Dict:
        """Анализ rejection сигнала."""
        zone = rejection_data['zone']
        
        # Confidence базируется на:
        # - Количестве касаний
        # - Силе зоны
        # - Количестве rejection candles
        
        touch_score = min(rejection_data['touches'] / 3.0, 1.0) * 0.4
        strength_score = zone.strength * 0.4
        rejection_score = min(rejection_data['rejection_candles'] / 2.0, 1.0) * 0.2
        
        confidence = touch_score + strength_score + rejection_score
        
        return {
            'has_signal': True,
            'signal_type': rejection_data['signal_type'],
            'confidence': min(confidence, 1.0),
            'pattern_type': 'Rejection',
            'involved_zones': [zone]
        }

    def _analyze_mean_reversion(
        self,
        zones: List[LiquidityZone],
        current_price: float,
        analysis
    ) -> Dict:
        """
        Анализ Mean Reversion сигнала.
        
        Цена близка к HVN -> ожидаем отскок обратно к равновесию.
        """
        # Ищем ближайший HVN
        hvn_zones = [z for z in zones if z.zone_type == "HVN"]
        
        if not hvn_zones:
            return {'has_signal': False}
        
        # Находим ближайший HVN
        nearest_hvn = min(hvn_zones, key=lambda z: abs(z.price - current_price))
        distance_pct = abs(nearest_hvn.price - current_price) / current_price * 100
        
        # Проверяем близость
        if distance_pct > self.config.hvn_distance_threshold_pct:
            return {'has_signal': False}
        
        # Определяем направление
        if nearest_hvn.price < current_price:
            # HVN ниже - ожидаем отскок вниз
            signal_type = SignalType.SELL
        else:
            # HVN выше - ожидаем отскок вверх
            signal_type = SignalType.BUY
        
        # Confidence
        # Чем ближе к HVN и чем сильнее зона - тем выше confidence
        distance_score = (1.0 - distance_pct / self.config.hvn_distance_threshold_pct) * 0.4
        strength_score = nearest_hvn.strength * 0.4
        
        # Бонус если это POC
        poc_bonus = 0.15 if nearest_hvn.is_poc else 0.0
        
        confidence = self.config.reversion_confidence_base + distance_score + strength_score + poc_bonus
        
        return {
            'has_signal': True,
            'signal_type': signal_type,
            'confidence': min(confidence, 1.0),
            'pattern_type': 'Mean Reversion',
            'involved_zones': [nearest_hvn]
        }

    def _analyze_breakout(
        self,
        candles: List[Candle],
        zones: List[LiquidityZone],
        current_price: float,
        metrics: OrderBookMetrics
    ) -> Dict:
        """
        Анализ Breakout сигнала через LVN.
        
        LVN = low volume node = слабое сопротивление -> легче пробить.
        """
        # Ищем LVN зоны
        lvn_zones = [z for z in zones if z.zone_type == "LVN"]
        
        if not lvn_zones:
            return {'has_signal': False}
        
        # Проверяем был ли недавний пробой LVN
        recent_candles = candles[-self.config.breakout_confirmation_candles:]
        
        for lvn in lvn_zones:
            # Проверяем пробила ли цена LVN за последние свечи
            breakout_detected = False
            breakout_direction = None
            
            for candle in recent_candles:
                # Breakout вверх
                if candle.close > lvn.price and candle.open <= lvn.price:
                    breakout_detected = True
                    breakout_direction = "up"
                    break
                
                # Breakout вниз
                if candle.close < lvn.price and candle.open >= lvn.price:
                    breakout_detected = True
                    breakout_direction = "down"
                    break
            
            if breakout_detected:
                # Проверяем объемное подтверждение
                recent_volumes = [c.volume for c in recent_candles]
                avg_volume = np.mean([c.volume for c in candles[-20:]])
                
                volume_spike = max(recent_volumes) > avg_volume * self.config.lvn_breakout_volume_multiplier
                
                if volume_spike:
                    # Breakout confirmed
                    signal_type = SignalType.BUY if breakout_direction == "up" else SignalType.SELL
                    
                    # Confidence
                    volume_ratio = max(recent_volumes) / avg_volume
                    volume_score = min(volume_ratio / 2.0, 1.0) * 0.4
                    
                    # LVN strength (обратная - слабый LVN = легче пробить)
                    lvn_score = (1.0 - lvn.strength) * 0.3
                    
                    confidence = self.config.breakout_confidence_base + volume_score + lvn_score
                    
                    return {
                        'has_signal': True,
                        'signal_type': signal_type,
                        'confidence': min(confidence, 1.0),
                        'pattern_type': 'Breakout',
                        'involved_zones': [lvn]
                    }
        
        return {'has_signal': False}

    def _calculate_level_based_stop(
        self,
        signal_type: SignalType,
        current_price: float,
        zones: List[LiquidityZone]
    ) -> Dict:
        """Рассчитать stop-loss на основе ближайшего уровня."""
        relevant_zones = []
        
        if signal_type == SignalType.BUY:
            # Для long: stop ниже ближайшего support
            relevant_zones = [
                z for z in zones
                if z.price < current_price and (z.is_support or z.zone_type == "HVN")
            ]
        else:
            # Для short: stop выше ближайшего resistance
            relevant_zones = [
                z for z in zones
                if z.price > current_price and (z.is_resistance or z.zone_type == "HVN")
            ]
        
        if not relevant_zones:
            return {'level_price': None, 'distance_pct': None}
        
        nearest = min(relevant_zones, key=lambda z: abs(z.price - current_price))
        distance_pct = abs(nearest.price - current_price) / current_price * 100
        
        # Stop дальше уровня на X%
        stop_distance = distance_pct + self.config.stop_loss_beyond_level_pct
        
        if signal_type == SignalType.BUY:
            stop_price = current_price * (1.0 - stop_distance / 100)
        else:
            stop_price = current_price * (1.0 + stop_distance / 100)
        
        return {
            'level_price': stop_price,
            'distance_pct': stop_distance
        }

    def _find_nearest_hvn(
        self, 
        zones: List[LiquidityZone], 
        current_price: float
    ) -> Optional[float]:
        """Найти цену ближайшего HVN."""
        hvns = [z for z in zones if z.zone_type == "HVN"]
        if not hvns:
            return None
        nearest = min(hvns, key=lambda z: abs(z.price - current_price))
        return nearest.price

    def _find_nearest_lvn(
        self, 
        zones: List[LiquidityZone], 
        current_price: float
    ) -> Optional[float]:
        """Найти цену ближайшего LVN."""
        lvns = [z for z in zones if z.zone_type == "LVN"]
        if not lvns:
            return None
        nearest = min(lvns, key=lambda z: abs(z.price - current_price))
        return nearest.price

    def get_statistics(self) -> Dict:
        """Получить расширенную статистику."""
        base_stats = super().get_statistics()
        base_stats.update({
            'mean_reversion_signals': self.mean_reversion_signals,
            'breakout_signals': self.breakout_signals,
            'rejection_signals': self.rejection_signals
        })
        return base_stats
