"""
Market Regime Detector - идентификация текущей фазы рынка.

Функциональность:
- Детекция трендовых режимов (Strong/Weak Up/Down, Ranging)
- Определение волатильности (High/Normal/Low)
- Оценка ликвидности (High/Normal/Low)
- Детекция структурных изменений (Chow Test)
- Mapping стратегий на оптимальные режимы
- Рекомендации по весам на основе режима

Путь: backend/strategies/adaptive/market_regime_detector.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
from scipy import stats

from core.logger import get_logger
from strategy.candle_manager import Candle
from models.orderbook import OrderBookMetrics

logger = get_logger(__name__)


class TrendRegime(Enum):
    """Режимы тренда."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class VolatilityRegime(Enum):
    """Режимы волатильности."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class LiquidityRegime(Enum):
    """Режимы ликвидности."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class MarketRegime:
    """Текущий режим рынка."""
    symbol: str
    timestamp: int
    
    # Режимы
    trend: TrendRegime
    trend_strength: float  # 0-1
    volatility: VolatilityRegime
    volatility_value: float
    liquidity: LiquidityRegime
    liquidity_score: float
    
    # Технические индикаторы
    adx_value: float
    atr_value: float
    volume_ratio: float  # Текущий объем / средний
    
    # Структурные изменения
    structural_break_detected: bool
    
    # Рекомендации
    recommended_strategy_weights: Dict[str, float]
    confidence_multiplier: float  # Множитель для min_consensus_confidence
    recommended_filters: Dict[str, any]


@dataclass
class RegimeDetectorConfig:
    """Конфигурация Regime Detector."""
    # Trend detection
    adx_strong_threshold: float = 25.0
    adx_weak_threshold: float = 15.0
    adx_period: int = 14
    
    # Moving averages
    sma_short: int = 20
    sma_long: int = 50
    
    # Volatility
    atr_period: int = 14
    volatility_high_percentile: float = 80.0  # 80th percentile = high
    volatility_low_percentile: float = 20.0   # 20th percentile = low
    volatility_lookback: int = 100
    
    # Liquidity
    volume_ma_period: int = 20
    liquidity_high_threshold: float = 1.2  # 120% of average
    liquidity_low_threshold: float = 0.8   # 80% of average
    
    # Structural breaks
    enable_structural_break_detection: bool = True
    chow_test_window: int = 50
    chow_test_significance: float = 0.05
    
    # Update frequency
    update_frequency_seconds: int = 300  # 5 минут


# Матрица: Режим -> Оптимальные веса стратегий
REGIME_STRATEGY_MAPPING = {
    # Strong Uptrend + High Volatility
    ('strong_uptrend', 'high'): {
        'momentum': 0.30,
        'supertrend': 0.25,
        'sar_wave': 0.20,
        'volume_profile': 0.10,
        'imbalance': 0.05,
        'volume_flow': 0.05,
        'liquidity_zone': 0.05,
        'smart_money': 0.00  # В сильной волатильности hybrid не эффективен
    },
    
    # Strong Uptrend + Normal/Low Volatility
    ('strong_uptrend', 'normal'): {
        'momentum': 0.25,
        'supertrend': 0.20,
        'smart_money': 0.20,  # Hybrid хорош в стабильном тренде
        'volume_profile': 0.15,
        'sar_wave': 0.10,
        'imbalance': 0.05,
        'volume_flow': 0.05,
        'liquidity_zone': 0.00
    },
    
    # Ranging + Low Volatility
    ('ranging', 'low'): {
        'liquidity_zone': 0.35,  # Mean reversion от HVN
        'imbalance': 0.25,       # Дисбаланс эффективен во флэте
        'volume_flow': 0.15,
        'volume_profile': 0.10,
        'momentum': 0.05,
        'sar_wave': 0.05,
        'supertrend': 0.05,
        'smart_money': 0.00
    },
    
    # Ranging + High Volatility (хаотичный рынок)
    ('ranging', 'high'): {
        # Очень консервативный подход
        'liquidity_zone': 0.40,
        'volume_flow': 0.30,
        'imbalance': 0.20,
        'momentum': 0.05,
        'sar_wave': 0.05,
        'supertrend': 0.00,
        'volume_profile': 0.00,
        'smart_money': 0.00
    },
    
    # Strong Downtrend + High Volatility
    ('strong_downtrend', 'high'): {
        'momentum': 0.30,     # Momentum эффективен в падении
        'supertrend': 0.25,
        'volume_flow': 0.15,  # Whale orders показывают выходы
        'sar_wave': 0.15,
        'volume_profile': 0.10,
        'imbalance': 0.05,
        'liquidity_zone': 0.00,
        'smart_money': 0.00
    },
    
    # High Volatility + Low Liquidity (опасный режим)
    ('ranging', 'high_low_liquidity'): {
        # Минимальная торговля
        'volume_flow': 0.50,  # Только whale tracking
        'liquidity_zone': 0.30,
        'imbalance': 0.20,
        'momentum': 0.00,
        'sar_wave': 0.00,
        'supertrend': 0.00,
        'volume_profile': 0.00,
        'smart_money': 0.00
    }
}


class MarketRegimeDetector:
    """
    Детектор режимов рынка.
    
    Определяет текущую фазу рынка (тренд, волатильность, ликвидность)
    и рекомендует оптимальные веса стратегий.
    """

    def __init__(self, config: RegimeDetectorConfig):
        """
        Инициализация детектора.

        Args:
            config: Конфигурация
        """
        self.config = config
        
        # Текущие режимы для каждого символа
        self.current_regimes: Dict[str, MarketRegime] = {}
        
        # История режимов (для анализа переходов)
        self.regime_history: Dict[str, List[MarketRegime]] = {}
        self.max_history_size = 100
        
        # Кэш индикаторов
        self.indicator_cache: Dict[str, Dict] = {}
        self.cache_timestamp: Dict[str, int] = {}
        
        # Статистика
        self.total_detections = 0
        self.regime_changes = 0
        
        logger.info(
            f"Инициализирован MarketRegimeDetector: "
            f"adx_strong={config.adx_strong_threshold}, "
            f"update_freq={config.update_frequency_seconds}s"
        )

    def detect_regime(
        self,
        symbol: str,
        candles: List[Candle],
        orderbook_metrics: Optional[OrderBookMetrics] = None
    ) -> MarketRegime:
        """
        Определить текущий режим рынка.

        Args:
            symbol: Торговая пара
            candles: История свечей (минимум 100)
            orderbook_metrics: Метрики стакана (опционально)

        Returns:
            MarketRegime
        """
        if len(candles) < 100:
            logger.warning(f"{symbol} | Недостаточно свечей для детекции режима")
            return self._get_default_regime(symbol)
        
        # Проверяем нужно ли обновление (кэш)
        if not self._should_update(symbol):
            if symbol in self.current_regimes:
                return self.current_regimes[symbol]
        
        # ========== ДЕТЕКЦИЯ ТРЕНДА ==========
        trend_regime, trend_strength, adx_value = self._detect_trend_regime(candles)
        
        # ========== ДЕТЕКЦИЯ ВОЛАТИЛЬНОСТИ ==========
        volatility_regime, volatility_value, atr_value = self._detect_volatility_regime(candles)
        
        # ========== ДЕТЕКЦИЯ ЛИКВИДНОСТИ ==========
        liquidity_regime, liquidity_score, volume_ratio = self._detect_liquidity_regime(
            candles, orderbook_metrics
        )
        
        # ========== СТРУКТУРНЫЕ ИЗМЕНЕНИЯ ==========
        structural_break = False
        if self.config.enable_structural_break_detection:
            structural_break = self._detect_structural_break(candles)
        
        # ========== РЕКОМЕНДАЦИИ ПО ВЕСАМ ==========
        recommended_weights = self._get_recommended_weights(
            trend_regime, volatility_regime, liquidity_regime
        )
        
        # ========== ФИЛЬТРЫ И ПАРАМЕТРЫ ==========
        confidence_multiplier, filters = self._get_regime_specific_parameters(
            trend_regime, volatility_regime, liquidity_regime
        )
        
        # ========== СОЗДАНИЕ РЕЖИМА ==========
        regime = MarketRegime(
            symbol=symbol,
            timestamp=int(datetime.now().timestamp() * 1000),
            trend=trend_regime,
            trend_strength=trend_strength,
            volatility=volatility_regime,
            volatility_value=volatility_value,
            liquidity=liquidity_regime,
            liquidity_score=liquidity_score,
            adx_value=adx_value,
            atr_value=atr_value,
            volume_ratio=volume_ratio,
            structural_break_detected=structural_break,
            recommended_strategy_weights=recommended_weights,
            confidence_multiplier=confidence_multiplier,
            recommended_filters=filters
        )
        
        # Проверка изменения режима
        if symbol in self.current_regimes:
            prev_regime = self.current_regimes[symbol]
            
            if (prev_regime.trend != regime.trend or 
                prev_regime.volatility != regime.volatility):
                self.regime_changes += 1
                logger.info(
                    f"🔄 РЕЖИМ ИЗМЕНЕН [{symbol}]: "
                    f"{prev_regime.trend.value}/{prev_regime.volatility.value} → "
                    f"{regime.trend.value}/{regime.volatility.value}"
                )
        
        # Сохраняем текущий режим
        self.current_regimes[symbol] = regime
        
        # Добавляем в историю
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        
        self.regime_history[symbol].append(regime)
        
        if len(self.regime_history[symbol]) > self.max_history_size:
            self.regime_history[symbol].pop(0)
        
        # Обновляем кэш timestamp
        self.cache_timestamp[symbol] = int(datetime.now().timestamp())
        self.total_detections += 1
        
        logger.debug(
            f"[Regime] {symbol}: trend={regime.trend.value} ({trend_strength:.2f}), "
            f"volatility={regime.volatility.value}, "
            f"liquidity={regime.liquidity.value}, "
            f"ADX={adx_value:.1f}"
        )
        
        return regime

    def get_current_regime(self, symbol: str) -> Optional[MarketRegime]:
        """Получить текущий режим для символа."""
        return self.current_regimes.get(symbol)

    def _detect_trend_regime(
        self, 
        candles: List[Candle]
    ) -> Tuple[TrendRegime, float, float]:
        """
        Детекция режима тренда.
        
        Использует:
        - ADX для силы тренда
        - SMA crossover для направления
        - Linear regression slope
        """
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        
        # ADX расчет
        adx_value = self._calculate_adx(highs, lows, closes)
        
        # SMA для направления
        sma_short = np.mean(closes[-self.config.sma_short:])
        sma_long = np.mean(closes[-self.config.sma_long:])
        
        # Linear regression для тренда
        x = np.arange(len(closes[-50:]))
        y = closes[-50:]
        slope, _ = np.polyfit(x, y, 1)
        slope_normalized = slope / np.mean(y) * 100  # в процентах
        
        # Определение направления
        is_uptrend = sma_short > sma_long and slope_normalized > 0
        is_downtrend = sma_short < sma_long and slope_normalized < 0
        
        # Определение режима
        if adx_value > self.config.adx_strong_threshold:
            # Сильный тренд
            if is_uptrend:
                regime = TrendRegime.STRONG_UPTREND
                strength = min(adx_value / 50.0, 1.0)
            elif is_downtrend:
                regime = TrendRegime.STRONG_DOWNTREND
                strength = min(adx_value / 50.0, 1.0)
            else:
                # ADX высокий но направление неясное
                regime = TrendRegime.RANGING
                strength = 0.5
        
        elif adx_value > self.config.adx_weak_threshold:
            # Слабый тренд
            if is_uptrend:
                regime = TrendRegime.WEAK_UPTREND
                strength = adx_value / 50.0
            elif is_downtrend:
                regime = TrendRegime.WEAK_DOWNTREND
                strength = adx_value / 50.0
            else:
                regime = TrendRegime.RANGING
                strength = 0.3
        
        else:
            # Ranging
            regime = TrendRegime.RANGING
            strength = 0.2
        
        return regime, strength, adx_value

    def _detect_volatility_regime(
        self, 
        candles: List[Candle]
    ) -> Tuple[VolatilityRegime, float, float]:
        """
        Детекция режима волатильности.
        
        Использует ATR в перцентилях.
        """
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])
        
        # ATR
        atr_values = []
        lookback = min(len(candles), self.config.volatility_lookback)
        
        for i in range(self.config.atr_period, lookback):
            atr = self._calculate_atr_single(
                highs[i-self.config.atr_period:i+1],
                lows[i-self.config.atr_period:i+1],
                closes[i-self.config.atr_period:i+1]
            )
            atr_values.append(atr)
        
        if not atr_values:
            return VolatilityRegime.NORMAL, 0.0, 0.0
        
        current_atr = atr_values[-1]
        
        # Перцентили
        high_percentile = np.percentile(atr_values, self.config.volatility_high_percentile)
        low_percentile = np.percentile(atr_values, self.config.volatility_low_percentile)
        
        # Нормализуем ATR (как процент от цены)
        current_price = closes[-1]
        volatility_value = (current_atr / current_price) * 100
        
        # Определение режима
        if current_atr > high_percentile:
            regime = VolatilityRegime.HIGH
        elif current_atr < low_percentile:
            regime = VolatilityRegime.LOW
        else:
            regime = VolatilityRegime.NORMAL
        
        return regime, volatility_value, current_atr

    def _detect_liquidity_regime(
        self,
        candles: List[Candle],
        orderbook_metrics: Optional[OrderBookMetrics]
    ) -> Tuple[LiquidityRegime, float, float]:
        """
        Детекция режима ликвидности.
        
        Использует объемы и опционально метрики стакана.
        """
        volumes = np.array([c.volume for c in candles])
        
        # Volume MA
        volume_ma = np.mean(volumes[-self.config.volume_ma_period:])
        current_volume = volumes[-1]
        
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
        
        # Liquidity score
        if orderbook_metrics:
            # Комбинируем volume ratio со spread
            spread_score = 1.0
            
            if orderbook_metrics.spread and orderbook_metrics.mid_price:
                spread_pct = (orderbook_metrics.spread / orderbook_metrics.mid_price) * 100
                # Узкий спред = высокая ликвидность
                spread_score = max(0.0, 1.0 - (spread_pct / 0.1))  # 0.1% spread = 0 score
            
            # Depth score
            total_volume = orderbook_metrics.total_bid_volume + orderbook_metrics.total_ask_volume
            depth_score = min(np.log1p(total_volume) / 15.0, 1.0)
            
            # Композитный liquidity score
            liquidity_score = (volume_ratio * 0.4 + spread_score * 0.3 + depth_score * 0.3)
        else:
            # Только volume ratio
            liquidity_score = volume_ratio
        
        # Определение режима
        if volume_ratio >= self.config.liquidity_high_threshold:
            regime = LiquidityRegime.HIGH
        elif volume_ratio <= self.config.liquidity_low_threshold:
            regime = LiquidityRegime.LOW
        else:
            regime = LiquidityRegime.NORMAL
        
        return regime, liquidity_score, volume_ratio

    def _detect_structural_break(self, candles: List[Candle]) -> bool:
        """
        Детекция структурных изменений через Chow Test.
        
        Упрощенная версия: сравниваем volatility до и после точки разрыва.
        """
        if len(candles) < self.config.chow_test_window * 2:
            return False
        
        returns = np.diff(np.log([c.close for c in candles[-self.config.chow_test_window * 2:]]))
        
        # Разделяем на две части
        mid_point = len(returns) // 2
        first_half = returns[:mid_point]
        second_half = returns[mid_point:]
        
        # F-test для сравнения variance
        f_stat = np.var(second_half) / np.var(first_half) if np.var(first_half) > 0 else 1.0
        
        # Degrees of freedom
        dfn = len(second_half) - 1
        dfd = len(first_half) - 1
        
        # P-value
        p_value = 1 - stats.f.cdf(f_stat, dfn, dfd)
        
        # Структурный разрыв если p < significance
        structural_break = p_value < self.config.chow_test_significance
        
        if structural_break:
            logger.warning(
                f"Структурный разрыв обнаружен: p_value={p_value:.4f}, "
                f"f_stat={f_stat:.2f}"
            )
        
        return structural_break

    def _get_recommended_weights(
        self,
        trend: TrendRegime,
        volatility: VolatilityRegime,
        liquidity: LiquidityRegime
    ) -> Dict[str, float]:
        """
        Получить рекомендуемые веса стратегий для режима.
        """
        # Упрощенный ключ для mapping
        trend_simple = 'strong_uptrend' if trend in [TrendRegime.STRONG_UPTREND, TrendRegime.WEAK_UPTREND] \
                  else 'strong_downtrend' if trend in [TrendRegime.STRONG_DOWNTREND, TrendRegime.WEAK_DOWNTREND] \
                  else 'ranging'
        
        volatility_simple = volatility.value
        
        # Специальный случай: High Volatility + Low Liquidity
        if volatility == VolatilityRegime.HIGH and liquidity == LiquidityRegime.LOW:
            key = ('ranging', 'high_low_liquidity')
        else:
            key = (trend_simple, volatility_simple)
        
        # Получаем из mapping или default
        if key in REGIME_STRATEGY_MAPPING:
            return REGIME_STRATEGY_MAPPING[key].copy()
        
        # Default веса (равномерное распределение)
        return {
            'momentum': 0.20,
            'sar_wave': 0.15,
            'supertrend': 0.20,
            'volume_profile': 0.15,
            'imbalance': 0.10,
            'volume_flow': 0.10,
            'liquidity_zone': 0.10,
            'smart_money': 0.00
        }

    def _get_regime_specific_parameters(
        self,
        trend: TrendRegime,
        volatility: VolatilityRegime,
        liquidity: LiquidityRegime
    ) -> Tuple[float, Dict]:
        """
        Получить специфичные для режима параметры.
        
        Returns:
            (confidence_multiplier, filters)
        """
        # Confidence multiplier
        # В опасных режимах повышаем требования
        if volatility == VolatilityRegime.HIGH and liquidity == LiquidityRegime.LOW:
            confidence_multiplier = 0.7  # Снижаем confidence (повышаем порог)
        elif volatility == VolatilityRegime.HIGH:
            confidence_multiplier = 0.85
        elif trend == TrendRegime.RANGING:
            confidence_multiplier = 0.90
        else:
            confidence_multiplier = 1.0
        
        # Фильтры
        filters = {}
        
        # В ranging требуем больше стратегий для consensus
        if trend == TrendRegime.RANGING:
            filters['min_strategies_required'] = 3
        else:
            filters['min_strategies_required'] = 2
        
        # В high volatility повышаем min_consensus_confidence
        if volatility == VolatilityRegime.HIGH:
            filters['min_consensus_confidence'] = 0.70
        else:
            filters['min_consensus_confidence'] = 0.60
        
        return confidence_multiplier, filters

    def _calculate_adx(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Average Directional Index."""
        period = self.config.adx_period
        
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
        
        # Smoothed
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
        
        # DI
        plus_di = 100 * plus_di_smooth / (atr_smooth + 1e-10)
        minus_di = 100 * minus_di_smooth / (atr_smooth + 1e-10)
        
        # DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        # ADX
        adx = np.mean(dx[-period:])
        
        return float(adx)

    def _calculate_atr_single(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray
    ) -> float:
        """Single ATR value."""
        if len(highs) < 2:
            return 0.0
        
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                abs(highs[1:] - closes[:-1]),
                abs(lows[1:] - closes[:-1])
            )
        )
        
        return float(np.mean(tr))

    def _should_update(self, symbol: str) -> bool:
        """Проверить нужно ли обновление режима."""
        if symbol not in self.cache_timestamp:
            return True
        
        last_update = self.cache_timestamp[symbol]
        current_time = int(datetime.now().timestamp())
        
        return (current_time - last_update) >= self.config.update_frequency_seconds

    def _get_default_regime(self, symbol: str) -> MarketRegime:
        """Получить default режим при недостатке данных."""
        return MarketRegime(
            symbol=symbol,
            timestamp=int(datetime.now().timestamp() * 1000),
            trend=TrendRegime.RANGING,
            trend_strength=0.3,
            volatility=VolatilityRegime.NORMAL,
            volatility_value=0.5,
            liquidity=LiquidityRegime.NORMAL,
            liquidity_score=1.0,
            adx_value=0.0,
            atr_value=0.0,
            volume_ratio=1.0,
            structural_break_detected=False,
            recommended_strategy_weights={
                'momentum': 0.20,
                'sar_wave': 0.15,
                'supertrend': 0.20,
                'volume_profile': 0.15,
                'imbalance': 0.10,
                'volume_flow': 0.10,
                'liquidity_zone': 0.10,
                'smart_money': 0.00
            },
            confidence_multiplier=1.0,
            recommended_filters={
                'min_strategies_required': 2,
                'min_consensus_confidence': 0.60
            }
        )

    def get_statistics(self) -> Dict:
        """Получить статистику детектора."""
        return {
            'total_detections': self.total_detections,
            'regime_changes': self.regime_changes,
            'symbols_tracked': len(self.current_regimes),
            'cache_size': len(self.indicator_cache)
        }
