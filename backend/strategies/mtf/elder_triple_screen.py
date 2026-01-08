"""
Elder's Triple Screen - Industry Standard MTF Consensus System.

Система трёх экранов Elder'a - классический алгоритм MTF анализа,
адаптированный для криптовалют с использованием современных индикаторов.

Оригинальная система (1985):
- Screen 1 (Tide): Недельный график - определяет основной тренд
- Screen 2 (Wave): Дневной график - ищет откаты против тренда
- Screen 3 (Ripple): 4H график - точный вход

Адаптация для крипто (24/7 рынок):
- Screen 1 (Tide): H1 - определяет основной тренд
- Screen 2 (Wave): M15 - ищет откаты/подтверждения
- Screen 3 (Ripple): M5 - точка входа

Правила входа:
1. Screen 1 определяет НАПРАВЛЕНИЕ (только BUY или только SELL)
2. Screen 2 даёт ПОДТВЕРЖДЕНИЕ (откат в тренде = opportunity)
3. Screen 3 даёт ТАЙМИНГ (конкретная точка входа)

КРИТИЧНО: Сигнал генерируется ТОЛЬКО когда все три экрана согласованы!

Путь: backend/strategies/mtf/elder_triple_screen.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from backend.core.logger import get_logger
from backend.models.signal import SignalType, SignalStrength, TradingSignal, SignalSource

from .timeframe_coordinator import Timeframe
from .timeframe_analyzer import TimeframeAnalysisResult, MarketRegimeType

logger = get_logger(__name__)


class ScreenDirection(Enum):
    """Направление на экране."""
    BULLISH = "bullish"       # Бычий тренд/сигнал
    BEARISH = "bearish"       # Медвежий тренд/сигнал
    NEUTRAL = "neutral"       # Нейтрально/неопределённо


class WavePhase(Enum):
    """Фаза волны на Screen 2."""
    PULLBACK_IN_UPTREND = "pullback_in_uptrend"     # Откат в бычьем тренде (opportunity BUY)
    PULLBACK_IN_DOWNTREND = "pullback_in_downtrend" # Откат в медвежьем тренде (opportunity SELL)
    WITH_TREND = "with_trend"                        # Движение по тренду (слишком поздно)
    AGAINST_TREND = "against_trend"                  # Сильное движение против (опасно)
    NEUTRAL = "neutral"                              # Нет чёткой фазы


@dataclass
class ScreenAnalysis:
    """Результат анализа одного экрана."""
    screen_number: int          # 1, 2 или 3
    timeframe: Timeframe
    direction: ScreenDirection
    confidence: float           # 0-1
    indicators_used: List[str]  # Какие индикаторы использованы
    reason: str
    wave_phase: Optional[WavePhase] = None  # Только для Screen 2


@dataclass
class TripleScreenResult:
    """Результат анализа Triple Screen."""
    # Результаты экранов
    screen1: ScreenAnalysis  # Tide (H1)
    screen2: ScreenAnalysis  # Wave (M15)
    screen3: ScreenAnalysis  # Ripple (M5)

    # Итоговое решение
    all_screens_aligned: bool
    final_direction: Optional[ScreenDirection]
    final_signal_type: Optional[SignalType]
    combined_confidence: float

    # Метаданные
    entry_quality: str         # "excellent", "good", "moderate", "poor"
    timing_score: float        # 0-1, насколько хороший момент входа
    risk_level: str            # "low", "medium", "high"

    # Причины и warnings
    reasons: List[str]
    warnings: List[str]

    # Timestamp
    timestamp: int


@dataclass
class ElderTripleScreenConfig:
    """Конфигурация Elder's Triple Screen."""
    # Таймфреймы для экранов (адаптированы для крипто)
    tide_timeframe: Timeframe = Timeframe.H1     # Screen 1
    wave_timeframe: Timeframe = Timeframe.M15    # Screen 2
    ripple_timeframe: Timeframe = Timeframe.M5   # Screen 3

    # Порог уверенности для каждого экрана
    min_tide_confidence: float = 0.65    # Screen 1 должен быть уверенным
    min_wave_confidence: float = 0.55    # Screen 2 может быть менее уверенным
    min_ripple_confidence: float = 0.50  # Screen 3 - точка входа

    # Требовать ли строгое выравнивание
    require_all_aligned: bool = True

    # Разрешить ли вход при нейтральном Screen 3
    allow_neutral_ripple: bool = False

    # Веса экранов для combined confidence
    tide_weight: float = 0.5     # Screen 1 - самый важный
    wave_weight: float = 0.3     # Screen 2 - подтверждение
    ripple_weight: float = 0.2   # Screen 3 - тайминг


class ElderTripleScreen:
    """
    Elder's Triple Screen Trading System.

    Реализация классического MTF алгоритма с адаптациями для криптовалют.

    Ключевые принципы:
    1. "Trade in the direction of the tide" - торгуем только по тренду H1
    2. "Use the waves against the tide to enter" - входим на откатах M15
    3. "Use ripples to time your entry" - M5 для точного входа
    """

    def __init__(self, config: Optional[ElderTripleScreenConfig] = None):
        """Инициализация Triple Screen системы."""
        self.config = config or ElderTripleScreenConfig()

        # Статистика
        self.total_analyses = 0
        self.signals_generated = 0
        self.all_aligned_count = 0
        self.partial_aligned_count = 0
        self.no_signal_count = 0

        logger.info(
            f"Инициализирован ElderTripleScreen: "
            f"tide={self.config.tide_timeframe.value}, "
            f"wave={self.config.wave_timeframe.value}, "
            f"ripple={self.config.ripple_timeframe.value}"
        )

    def analyze(
        self,
        tf_results: Dict[Timeframe, TimeframeAnalysisResult],
        current_price: float
    ) -> TripleScreenResult:
        """
        Выполнить Triple Screen анализ.

        Args:
            tf_results: Результаты анализа по таймфреймам
            current_price: Текущая цена

        Returns:
            TripleScreenResult с итоговым решением
        """
        self.total_analyses += 1

        # Screen 1: Tide (H1) - определяем направление тренда
        screen1 = self._analyze_screen1_tide(tf_results)

        # Screen 2: Wave (M15) - ищем подходящую фазу волны
        screen2 = self._analyze_screen2_wave(tf_results, screen1.direction)

        # Screen 3: Ripple (M5) - определяем точку входа
        screen3 = self._analyze_screen3_ripple(tf_results, screen1.direction, screen2)

        # Проверяем выравнивание экранов
        all_aligned, final_direction = self._check_alignment(screen1, screen2, screen3)

        # Определяем качество входа
        entry_quality, timing_score = self._evaluate_entry_quality(
            screen1, screen2, screen3, all_aligned
        )

        # Вычисляем combined confidence
        combined_confidence = self._calculate_combined_confidence(
            screen1, screen2, screen3, all_aligned
        )

        # Определяем уровень риска
        risk_level = self._evaluate_risk(screen1, screen2, screen3, combined_confidence)

        # Формируем reasons и warnings
        reasons, warnings = self._compile_reasons_and_warnings(
            screen1, screen2, screen3, all_aligned
        )

        # Определяем финальный тип сигнала
        final_signal_type = None
        if all_aligned and final_direction:
            if final_direction == ScreenDirection.BULLISH:
                final_signal_type = SignalType.BUY
            elif final_direction == ScreenDirection.BEARISH:
                final_signal_type = SignalType.SELL

        # Обновляем статистику
        if all_aligned:
            self.all_aligned_count += 1
            if final_signal_type:
                self.signals_generated += 1
        elif final_direction:
            self.partial_aligned_count += 1
        else:
            self.no_signal_count += 1

        result = TripleScreenResult(
            screen1=screen1,
            screen2=screen2,
            screen3=screen3,
            all_screens_aligned=all_aligned,
            final_direction=final_direction,
            final_signal_type=final_signal_type,
            combined_confidence=combined_confidence,
            entry_quality=entry_quality,
            timing_score=timing_score,
            risk_level=risk_level,
            reasons=reasons,
            warnings=warnings,
            timestamp=int(datetime.now().timestamp() * 1000)
        )

        self._log_result(result)

        return result

    def _analyze_screen1_tide(
        self,
        tf_results: Dict[Timeframe, TimeframeAnalysisResult]
    ) -> ScreenAnalysis:
        """
        Screen 1: Tide (H1) - определение основного тренда.

        Использует:
        - Market regime (trending vs ranging)
        - Trend direction индикаторы
        - Momentum
        """
        result = tf_results.get(self.config.tide_timeframe)

        if not result:
            return ScreenAnalysis(
                screen_number=1,
                timeframe=self.config.tide_timeframe,
                direction=ScreenDirection.NEUTRAL,
                confidence=0.0,
                indicators_used=[],
                reason="No data for tide timeframe"
            )

        # Анализируем market regime
        regime = result.regime.market_regime
        trend_strength = result.regime.trend_strength

        # Определяем направление на основе режима и технического анализа
        direction = ScreenDirection.NEUTRAL
        confidence = 0.0
        indicators_used = []
        reasons = []

        # 1. Market regime direction
        if regime == MarketRegimeType.TRENDING_UP:
            direction = ScreenDirection.BULLISH
            confidence += 0.3
            indicators_used.append("market_regime")
            reasons.append("H1 uptrend")
        elif regime == MarketRegimeType.TRENDING_DOWN:
            direction = ScreenDirection.BEARISH
            confidence += 0.3
            indicators_used.append("market_regime")
            reasons.append("H1 downtrend")

        # 2. Trend strength усиливает уверенность
        if trend_strength > 0.6:
            confidence += 0.2
            indicators_used.append("trend_strength")
            reasons.append(f"Strong trend ({trend_strength:.2f})")
        elif trend_strength > 0.4:
            confidence += 0.1

        # 3. Технический анализ (если есть сигнал от стратегий)
        if result.timeframe_signal:
            signal_type = result.timeframe_signal.signal_type
            signal_conf = result.timeframe_signal.confidence

            if signal_type == SignalType.BUY:
                if direction == ScreenDirection.BULLISH:
                    confidence += 0.3 * signal_conf
                    reasons.append(f"Strategy BUY confirms ({signal_conf:.2f})")
                elif direction == ScreenDirection.NEUTRAL:
                    direction = ScreenDirection.BULLISH
                    confidence += 0.25 * signal_conf
                    reasons.append(f"Strategy suggests BUY")
                # Если direction BEARISH, но сигнал BUY - конфликт
                else:
                    confidence *= 0.7  # Уменьшаем уверенность
                    reasons.append("Conflicting BUY signal")

            elif signal_type == SignalType.SELL:
                if direction == ScreenDirection.BEARISH:
                    confidence += 0.3 * signal_conf
                    reasons.append(f"Strategy SELL confirms ({signal_conf:.2f})")
                elif direction == ScreenDirection.NEUTRAL:
                    direction = ScreenDirection.BEARISH
                    confidence += 0.25 * signal_conf
                    reasons.append(f"Strategy suggests SELL")
                else:
                    confidence *= 0.7
                    reasons.append("Conflicting SELL signal")

            indicators_used.append("strategy_consensus")

        # 4. Volatility check (высокая волатильность = осторожность)
        if result.regime.volatility_state == "high":
            confidence *= 0.9
            reasons.append("High volatility caution")

        # Нормализуем confidence
        confidence = min(confidence, 1.0)

        return ScreenAnalysis(
            screen_number=1,
            timeframe=self.config.tide_timeframe,
            direction=direction,
            confidence=confidence,
            indicators_used=indicators_used,
            reason=" | ".join(reasons) if reasons else "Neutral market"
        )

    def _analyze_screen2_wave(
        self,
        tf_results: Dict[Timeframe, TimeframeAnalysisResult],
        tide_direction: ScreenDirection
    ) -> ScreenAnalysis:
        """
        Screen 2: Wave (M15) - поиск откатов и подтверждений.

        Ключевой принцип: Входим когда волна идёт против течения (откат).

        В бычьем тренде (tide BULLISH):
        - Ищем pullback down = opportunity to buy
        - Wave идёт с трендом = слишком поздно покупать

        В медвежьем тренде (tide BEARISH):
        - Ищем pullback up = opportunity to sell
        - Wave идёт с трендом = слишком поздно продавать
        """
        result = tf_results.get(self.config.wave_timeframe)

        if not result:
            return ScreenAnalysis(
                screen_number=2,
                timeframe=self.config.wave_timeframe,
                direction=ScreenDirection.NEUTRAL,
                confidence=0.0,
                indicators_used=[],
                reason="No data for wave timeframe",
                wave_phase=WavePhase.NEUTRAL
            )

        regime = result.regime.market_regime
        indicators_used = []
        reasons = []
        confidence = 0.0
        wave_phase = WavePhase.NEUTRAL
        direction = ScreenDirection.NEUTRAL

        # Определяем фазу волны относительно tide
        if tide_direction == ScreenDirection.BULLISH:
            # В бычьем тренде ищем откат вниз (opportunity)
            if regime in [MarketRegimeType.TRENDING_DOWN, MarketRegimeType.VOLATILE_DOWN]:
                wave_phase = WavePhase.PULLBACK_IN_UPTREND
                direction = ScreenDirection.BULLISH  # Готовимся покупать!
                confidence = 0.7
                reasons.append("Pullback in uptrend - buy opportunity")
            elif regime == MarketRegimeType.RANGING:
                wave_phase = WavePhase.PULLBACK_IN_UPTREND
                direction = ScreenDirection.BULLISH
                confidence = 0.5
                reasons.append("Consolidation in uptrend - potential entry")
            elif regime == MarketRegimeType.TRENDING_UP:
                wave_phase = WavePhase.WITH_TREND
                direction = ScreenDirection.NEUTRAL  # Слишком поздно
                confidence = 0.3
                reasons.append("Wave with trend - may be late")
            else:
                wave_phase = WavePhase.AGAINST_TREND
                direction = ScreenDirection.NEUTRAL
                confidence = 0.2
                reasons.append("Strong counter-move - caution")

        elif tide_direction == ScreenDirection.BEARISH:
            # В медвежьем тренде ищем откат вверх (opportunity)
            if regime in [MarketRegimeType.TRENDING_UP, MarketRegimeType.VOLATILE_UP]:
                wave_phase = WavePhase.PULLBACK_IN_DOWNTREND
                direction = ScreenDirection.BEARISH  # Готовимся продавать!
                confidence = 0.7
                reasons.append("Pullback in downtrend - sell opportunity")
            elif regime == MarketRegimeType.RANGING:
                wave_phase = WavePhase.PULLBACK_IN_DOWNTREND
                direction = ScreenDirection.BEARISH
                confidence = 0.5
                reasons.append("Consolidation in downtrend - potential entry")
            elif regime == MarketRegimeType.TRENDING_DOWN:
                wave_phase = WavePhase.WITH_TREND
                direction = ScreenDirection.NEUTRAL  # Слишком поздно
                confidence = 0.3
                reasons.append("Wave with trend - may be late")
            else:
                wave_phase = WavePhase.AGAINST_TREND
                direction = ScreenDirection.NEUTRAL
                confidence = 0.2
                reasons.append("Strong counter-move - caution")
        else:
            # Tide нейтральный - не торгуем
            wave_phase = WavePhase.NEUTRAL
            direction = ScreenDirection.NEUTRAL
            confidence = 0.0
            reasons.append("No tide direction - no trade")

        indicators_used.append("wave_phase_analysis")

        # Учитываем сигнал стратегии если есть
        if result.timeframe_signal:
            signal_type = result.timeframe_signal.signal_type
            signal_conf = result.timeframe_signal.confidence

            # Сигнал должен соответствовать ожидаемому направлению
            if direction == ScreenDirection.BULLISH and signal_type == SignalType.BUY:
                confidence += 0.15 * signal_conf
                reasons.append(f"Strategy confirms BUY")
                indicators_used.append("strategy_signal")
            elif direction == ScreenDirection.BEARISH and signal_type == SignalType.SELL:
                confidence += 0.15 * signal_conf
                reasons.append(f"Strategy confirms SELL")
                indicators_used.append("strategy_signal")
            elif signal_type != SignalType.HOLD:
                # Противоречащий сигнал
                confidence *= 0.8
                reasons.append(f"Conflicting strategy signal")

        confidence = min(confidence, 1.0)

        return ScreenAnalysis(
            screen_number=2,
            timeframe=self.config.wave_timeframe,
            direction=direction,
            confidence=confidence,
            indicators_used=indicators_used,
            reason=" | ".join(reasons),
            wave_phase=wave_phase
        )

    def _analyze_screen3_ripple(
        self,
        tf_results: Dict[Timeframe, TimeframeAnalysisResult],
        tide_direction: ScreenDirection,
        screen2: ScreenAnalysis
    ) -> ScreenAnalysis:
        """
        Screen 3: Ripple (M5) - точная точка входа.

        Если Screen 1 и 2 согласованы, Screen 3 даёт точку входа:
        - Для BUY: ищем локальный минимум / breakout вверх
        - Для SELL: ищем локальный максимум / breakdown вниз
        """
        result = tf_results.get(self.config.ripple_timeframe)

        if not result:
            return ScreenAnalysis(
                screen_number=3,
                timeframe=self.config.ripple_timeframe,
                direction=ScreenDirection.NEUTRAL,
                confidence=0.0,
                indicators_used=[],
                reason="No data for ripple timeframe"
            )

        indicators_used = []
        reasons = []
        confidence = 0.0
        direction = ScreenDirection.NEUTRAL

        # Если Screen 2 не даёт направления, Screen 3 не может дать вход
        if screen2.direction == ScreenDirection.NEUTRAL:
            return ScreenAnalysis(
                screen_number=3,
                timeframe=self.config.ripple_timeframe,
                direction=ScreenDirection.NEUTRAL,
                confidence=0.0,
                indicators_used=["screen2_check"],
                reason="No wave direction - no entry"
            )

        expected_direction = screen2.direction

        # Проверяем сигнал стратегии на M5
        if result.timeframe_signal:
            signal_type = result.timeframe_signal.signal_type
            signal_conf = result.timeframe_signal.confidence

            if expected_direction == ScreenDirection.BULLISH:
                if signal_type == SignalType.BUY:
                    direction = ScreenDirection.BULLISH
                    confidence = signal_conf * 0.8
                    reasons.append(f"M5 BUY entry signal ({signal_conf:.2f})")
                    indicators_used.append("entry_signal")
                elif signal_type == SignalType.SELL:
                    # Противоположный сигнал - не входим
                    direction = ScreenDirection.NEUTRAL
                    confidence = 0.0
                    reasons.append("M5 shows SELL - wait")
                else:
                    direction = ScreenDirection.NEUTRAL
                    confidence = 0.3
                    reasons.append("M5 neutral - no clear entry")

            elif expected_direction == ScreenDirection.BEARISH:
                if signal_type == SignalType.SELL:
                    direction = ScreenDirection.BEARISH
                    confidence = signal_conf * 0.8
                    reasons.append(f"M5 SELL entry signal ({signal_conf:.2f})")
                    indicators_used.append("entry_signal")
                elif signal_type == SignalType.BUY:
                    direction = ScreenDirection.NEUTRAL
                    confidence = 0.0
                    reasons.append("M5 shows BUY - wait")
                else:
                    direction = ScreenDirection.NEUTRAL
                    confidence = 0.3
                    reasons.append("M5 neutral - no clear entry")
        else:
            # Нет сигнала от стратегии - проверяем regime
            regime = result.regime.market_regime

            if expected_direction == ScreenDirection.BULLISH:
                if regime in [MarketRegimeType.TRENDING_UP, MarketRegimeType.VOLATILE_UP]:
                    direction = ScreenDirection.BULLISH
                    confidence = 0.5
                    reasons.append("M5 turning bullish - entry possible")
                else:
                    direction = ScreenDirection.NEUTRAL
                    confidence = 0.2
                    reasons.append("M5 not yet bullish - wait")

            elif expected_direction == ScreenDirection.BEARISH:
                if regime in [MarketRegimeType.TRENDING_DOWN, MarketRegimeType.VOLATILE_DOWN]:
                    direction = ScreenDirection.BEARISH
                    confidence = 0.5
                    reasons.append("M5 turning bearish - entry possible")
                else:
                    direction = ScreenDirection.NEUTRAL
                    confidence = 0.2
                    reasons.append("M5 not yet bearish - wait")

            indicators_used.append("regime_check")

        confidence = min(confidence, 1.0)

        return ScreenAnalysis(
            screen_number=3,
            timeframe=self.config.ripple_timeframe,
            direction=direction,
            confidence=confidence,
            indicators_used=indicators_used,
            reason=" | ".join(reasons) if reasons else "Entry analysis"
        )

    def _check_alignment(
        self,
        screen1: ScreenAnalysis,
        screen2: ScreenAnalysis,
        screen3: ScreenAnalysis
    ) -> Tuple[bool, Optional[ScreenDirection]]:
        """
        Проверить выравнивание всех трёх экранов.

        Returns:
            (all_aligned, final_direction)
        """
        # Screen 1 должен иметь чёткое направление
        if screen1.direction == ScreenDirection.NEUTRAL:
            return False, None

        # Screen 1 confidence должен быть достаточным
        if screen1.confidence < self.config.min_tide_confidence:
            return False, None

        # Screen 2 должен подтверждать направление
        if screen2.direction != screen1.direction:
            return False, screen1.direction  # Частичное выравнивание

        if screen2.confidence < self.config.min_wave_confidence:
            return False, screen1.direction

        # Screen 3 проверка
        if screen3.direction == screen1.direction:
            if screen3.confidence >= self.config.min_ripple_confidence:
                return True, screen1.direction
            else:
                return False, screen1.direction  # Недостаточная уверенность входа
        elif screen3.direction == ScreenDirection.NEUTRAL:
            if self.config.allow_neutral_ripple:
                return True, screen1.direction  # Разрешён нейтральный Screen 3
            else:
                return False, screen1.direction
        else:
            # Screen 3 противоположен - нет выравнивания
            return False, screen1.direction

    def _evaluate_entry_quality(
        self,
        screen1: ScreenAnalysis,
        screen2: ScreenAnalysis,
        screen3: ScreenAnalysis,
        all_aligned: bool
    ) -> Tuple[str, float]:
        """
        Оценить качество точки входа.

        Returns:
            (quality_label, timing_score)
        """
        if not all_aligned:
            return "poor", 0.2

        # Базовый timing score
        timing_score = (
            screen1.confidence * 0.4 +
            screen2.confidence * 0.3 +
            screen3.confidence * 0.3
        )

        # Бонус за идеальную фазу волны
        if screen2.wave_phase in [WavePhase.PULLBACK_IN_UPTREND, WavePhase.PULLBACK_IN_DOWNTREND]:
            timing_score += 0.1

        # Penalty за слабый Screen 3
        if screen3.confidence < 0.5:
            timing_score *= 0.9

        timing_score = min(timing_score, 1.0)

        # Определяем качество
        if timing_score >= 0.8:
            quality = "excellent"
        elif timing_score >= 0.65:
            quality = "good"
        elif timing_score >= 0.5:
            quality = "moderate"
        else:
            quality = "poor"

        return quality, timing_score

    def _calculate_combined_confidence(
        self,
        screen1: ScreenAnalysis,
        screen2: ScreenAnalysis,
        screen3: ScreenAnalysis,
        all_aligned: bool
    ) -> float:
        """Вычислить combined confidence."""
        if not all_aligned:
            return 0.0

        combined = (
            screen1.confidence * self.config.tide_weight +
            screen2.confidence * self.config.wave_weight +
            screen3.confidence * self.config.ripple_weight
        )

        # Бонус за полное выравнивание
        combined *= 1.1

        return min(combined, 1.0)

    def _evaluate_risk(
        self,
        screen1: ScreenAnalysis,
        screen2: ScreenAnalysis,
        screen3: ScreenAnalysis,
        combined_confidence: float
    ) -> str:
        """Оценить уровень риска."""
        risk_score = 0.0

        # Низкая уверенность = высокий риск
        if combined_confidence < 0.5:
            risk_score += 0.4
        elif combined_confidence < 0.7:
            risk_score += 0.2

        # Wave with trend = более рискованно (возможно поздний вход)
        if screen2.wave_phase == WavePhase.WITH_TREND:
            risk_score += 0.3
        elif screen2.wave_phase == WavePhase.AGAINST_TREND:
            risk_score += 0.4

        # Слабый Screen 1 = неопределённый тренд
        if screen1.confidence < 0.6:
            risk_score += 0.2

        if risk_score >= 0.5:
            return "high"
        elif risk_score >= 0.3:
            return "medium"
        else:
            return "low"

    def _compile_reasons_and_warnings(
        self,
        screen1: ScreenAnalysis,
        screen2: ScreenAnalysis,
        screen3: ScreenAnalysis,
        all_aligned: bool
    ) -> Tuple[List[str], List[str]]:
        """Собрать reasons и warnings."""
        reasons = []
        warnings = []

        # Добавляем reasons от экранов
        reasons.append(f"S1({screen1.timeframe.value}): {screen1.reason}")
        reasons.append(f"S2({screen2.timeframe.value}): {screen2.reason}")
        reasons.append(f"S3({screen3.timeframe.value}): {screen3.reason}")

        # Warnings
        if not all_aligned:
            warnings.append("Screens not fully aligned - no trade recommended")

        if screen1.confidence < 0.5:
            warnings.append("Weak tide direction - uncertain trend")

        if screen2.wave_phase == WavePhase.WITH_TREND:
            warnings.append("Entry may be late - wave moving with trend")
        elif screen2.wave_phase == WavePhase.AGAINST_TREND:
            warnings.append("Strong counter-move - high risk")

        if screen3.confidence < 0.4:
            warnings.append("Weak entry signal - consider waiting")

        return reasons, warnings

    def _log_result(self, result: TripleScreenResult):
        """Логировать результат."""
        if result.all_screens_aligned and result.final_signal_type:
            logger.info(
                f"🎯 TRIPLE SCREEN: {result.final_signal_type.value}, "
                f"confidence={result.combined_confidence:.2f}, "
                f"quality={result.entry_quality}, "
                f"timing={result.timing_score:.2f}, "
                f"risk={result.risk_level}"
            )
        else:
            logger.debug(
                f"Triple Screen: no signal, "
                f"S1={result.screen1.direction.value}({result.screen1.confidence:.2f}), "
                f"S2={result.screen2.direction.value}({result.screen2.confidence:.2f}), "
                f"S3={result.screen3.direction.value}({result.screen3.confidence:.2f})"
            )

    def create_trading_signal(
        self,
        result: TripleScreenResult,
        symbol: str,
        current_price: float
    ) -> Optional[TradingSignal]:
        """
        Создать TradingSignal из результата Triple Screen.

        Args:
            result: Результат Triple Screen анализа
            symbol: Торговая пара
            current_price: Текущая цена

        Returns:
            TradingSignal или None
        """
        if not result.all_screens_aligned or not result.final_signal_type:
            return None

        # Определяем силу сигнала
        if result.combined_confidence >= 0.8:
            strength = SignalStrength.STRONG
        elif result.combined_confidence >= 0.65:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.WEAK

        signal = TradingSignal(
            symbol=symbol,
            signal_type=result.final_signal_type,
            source=SignalSource.MTF_CONSENSUS,
            strength=strength,
            price=current_price,
            confidence=result.combined_confidence,
            timestamp=result.timestamp,
            reason=f"Triple Screen [{result.entry_quality}]: " + " | ".join(result.reasons[:2]),
            metadata={
                'system': 'elder_triple_screen',
                'screens': {
                    's1_direction': result.screen1.direction.value,
                    's1_confidence': result.screen1.confidence,
                    's2_direction': result.screen2.direction.value,
                    's2_confidence': result.screen2.confidence,
                    's2_wave_phase': result.screen2.wave_phase.value if result.screen2.wave_phase else None,
                    's3_direction': result.screen3.direction.value,
                    's3_confidence': result.screen3.confidence,
                },
                'entry_quality': result.entry_quality,
                'timing_score': result.timing_score,
                'risk_level': result.risk_level,
                'warnings': result.warnings
            }
        )

        return signal

    def get_statistics(self) -> Dict:
        """Получить статистику."""
        return {
            'total_analyses': self.total_analyses,
            'signals_generated': self.signals_generated,
            'all_aligned_count': self.all_aligned_count,
            'partial_aligned_count': self.partial_aligned_count,
            'no_signal_count': self.no_signal_count,
            'alignment_rate': (
                self.all_aligned_count / self.total_analyses
                if self.total_analyses > 0 else 0.0
            ),
            'signal_rate': (
                self.signals_generated / self.total_analyses
                if self.total_analyses > 0 else 0.0
            ),
            'config': {
                'tide_tf': self.config.tide_timeframe.value,
                'wave_tf': self.config.wave_timeframe.value,
                'ripple_tf': self.config.ripple_timeframe.value,
                'require_all_aligned': self.config.require_all_aligned
            }
        }
