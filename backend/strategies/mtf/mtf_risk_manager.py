"""
Professional Multi-Timeframe Risk Management Module.

Этот модуль заменяет упрощенную логику расчета risk/reward параметров
в timeframe_signal_synthesizer.py профессиональными industry-standard алгоритмами.

ЗАМЕНЯЕТ УПРОЩЕННУЮ ЛОГИКУ:
1. Take-Profit с фиксированным R:R 2:1 → ML/ATR/Regime-based TP/SL
2. Простые position size multipliers → Kelly Criterion / Volatility-adjusted sizing
3. Счетчик risk factors → Weighted risk scoring
4. Отсутствующий reliability score → Historical performance tracking

ИНТЕГРАЦИИ:
- UnifiedSLTPCalculator: Профессиональный расчет Stop-Loss и Take-Profit
- AdaptiveRiskCalculator: Kelly Criterion и адаптивный position sizing
- SignalReliabilityTracker: Отслеживание исторической производительности сигналов

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────┐
│              MTFRiskManager (Orchestrator)                  │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ UnifiedSLTPCalc  │  │ AdaptiveRiskCalc │                │
│  │ - ML-based TP/SL │  │ - Kelly Criterion│                │
│  │ - ATR fallback   │  │ - Volatility adj │                │
│  │ - Regime adjust  │  │ - Win rate scale │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │      SignalReliabilityTracker                 │          │
│  │  - Historical win rate by synthesis mode      │          │
│  │  - Performance by timeframe combination       │          │
│  │  - Quality score correlation with outcomes    │          │
│  └──────────────────────────────────────────────┘          │
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │      WeightedRiskAssessment                   │          │
│  │  - Multi-factor risk scoring                  │          │
│  │  - Factor correlation analysis                │          │
│  │  - Dynamic threshold adjustment               │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘

Path: backend/strategies/mtf/mtf_risk_manager.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from backend.core.logger import get_logger
from backend.models.signal import TradingSignal, SignalType
from backend.strategy.sltp_calculator import UnifiedSLTPCalculator, sltp_calculator
from backend.strategy.adaptive_risk_calculator import AdaptiveRiskCalculator, adaptive_risk_calculator
from backend.strategy.risk_models import SLTPCalculation, RiskPerTradeParams, MarketRegime
from backend.strategies.mtf.timeframe_coordinator import Timeframe

logger = get_logger(__name__)


@dataclass
class SignalPerformanceRecord:
    """Запись о производительности сигнала."""

    # Идентификация
    synthesis_mode: str
    timeframes_used: List[Timeframe]
    signal_quality: float
    alignment_score: float

    # Рыночные условия
    market_regime: Optional[MarketRegime]
    volatility_regime: Optional[str]

    # Результат
    was_profitable: bool
    pnl_percent: float
    max_adverse_excursion: float  # Максимальная просадка
    max_favorable_excursion: float  # Максимальная прибыль

    # Метаданные
    entry_time: datetime
    exit_time: Optional[datetime]
    hold_duration_minutes: Optional[int]

    # TP/SL execution
    hit_take_profit: bool
    hit_stop_loss: bool
    actual_rr_achieved: float


@dataclass
class ReliabilityMetrics:
    """Метрики надежности для определенного типа сигналов."""

    total_signals: int
    win_rate: float
    avg_pnl_percent: float
    avg_rr_achieved: float

    # Консистентность
    win_streak: int
    loss_streak: int
    max_win_streak: int
    max_loss_streak: int

    # Quality correlation
    quality_score_correlation: float  # Корреляция quality score с outcomes

    # Confidence
    reliability_score: float  # 0.0-1.0, композитная метрика


@dataclass
class WeightedRiskFactors:
    """Weighted risk factors для professional risk assessment."""

    # Market factors (вес 40%)
    high_volatility: float = 0.0  # 0.0-1.0
    regime_uncertainty: float = 0.0
    low_liquidity: float = 0.0

    # Signal factors (вес 35%)
    low_alignment: float = 0.0
    divergence_present: float = 0.0
    low_quality: float = 0.0

    # Historical factors (вес 25%)
    poor_reliability: float = 0.0
    unfavorable_regime: float = 0.0
    recent_losses: float = 0.0

    # Weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        'market': 0.40,
        'signal': 0.35,
        'historical': 0.25
    })

    def calculate_composite_risk(self) -> float:
        """
        Рассчитать композитный risk score.

        Returns:
            float: Risk score 0.0-1.0 (0=low risk, 1=extreme risk)
        """
        market_score = (
            self.high_volatility * 0.4 +
            self.regime_uncertainty * 0.35 +
            self.low_liquidity * 0.25
        ) * self.weights['market']

        signal_score = (
            self.low_alignment * 0.35 +
            self.divergence_present * 0.40 +
            self.low_quality * 0.25
        ) * self.weights['signal']

        historical_score = (
            self.poor_reliability * 0.40 +
            self.unfavorable_regime * 0.30 +
            self.recent_losses * 0.30
        ) * self.weights['historical']

        return market_score + signal_score + historical_score


class SignalReliabilityTracker:
    """
    Отслеживание исторической производительности MTF сигналов.

    Отслеживает:
    - Win rate по synthesis mode
    - Performance по комбинациям timeframes
    - Корреляцию quality score с outcomes
    - Режимы рынка, в которых сигналы работают лучше/хуже
    """

    def __init__(self, max_history: int = 1000):
        """
        Инициализация tracker.

        Args:
            max_history: Максимальное количество записей в истории
        """
        self.max_history = max_history

        # История всех сигналов
        self.performance_history: deque = deque(maxlen=max_history)

        # Aggregated metrics по synthesis mode
        self.metrics_by_mode: Dict[str, ReliabilityMetrics] = {}

        # Aggregated metrics по timeframe combinations
        self.metrics_by_tf_combo: Dict[str, ReliabilityMetrics] = {}

        # Aggregated metrics по market regime
        self.metrics_by_regime: Dict[MarketRegime, ReliabilityMetrics] = {}

        # Recent performance window (последние 50 сигналов)
        self.recent_signals: deque = deque(maxlen=50)

        logger.info(f"SignalReliabilityTracker initialized: max_history={max_history}")

    def record_signal_outcome(self, record: SignalPerformanceRecord):
        """
        Записать результат сигнала.

        Args:
            record: Запись о производительности
        """
        self.performance_history.append(record)
        self.recent_signals.append(record)

        # Пересчитываем aggregated metrics
        self._recalculate_metrics()

        logger.debug(
            f"Signal outcome recorded: mode={record.synthesis_mode}, "
            f"profitable={record.was_profitable}, pnl={record.pnl_percent:.2%}"
        )

    def get_reliability_score(
        self,
        synthesis_mode: str,
        timeframes: List[Timeframe],
        market_regime: Optional[MarketRegime],
        signal_quality: float
    ) -> float:
        """
        Получить reliability score для определенного типа сигнала.

        Args:
            synthesis_mode: Режим synthesis
            timeframes: Используемые timeframes
            market_regime: Режим рынка
            signal_quality: Качество текущего сигнала

        Returns:
            float: Reliability score 0.0-1.0
        """
        # Базовая reliability от synthesis mode
        mode_reliability = self._get_mode_reliability(synthesis_mode)

        # Reliability от timeframe combination
        tf_reliability = self._get_tf_combo_reliability(timeframes)

        # Reliability от market regime
        regime_reliability = self._get_regime_reliability(market_regime)

        # Recent performance trend
        recent_trend = self._get_recent_trend()

        # Quality correlation factor
        quality_factor = self._get_quality_correlation_factor(
            synthesis_mode, signal_quality
        )

        # Weighted composite
        reliability_score = (
            mode_reliability * 0.30 +
            tf_reliability * 0.20 +
            regime_reliability * 0.20 +
            recent_trend * 0.15 +
            quality_factor * 0.15
        )

        return np.clip(reliability_score, 0.0, 1.0)

    def _recalculate_metrics(self):
        """Пересчитать aggregated metrics."""
        if not self.performance_history:
            return

        # По synthesis mode
        mode_records = defaultdict(list)
        for record in self.performance_history:
            mode_records[record.synthesis_mode].append(record)

        for mode, records in mode_records.items():
            self.metrics_by_mode[mode] = self._calculate_metrics(records)

        # По timeframe combinations
        tf_records = defaultdict(list)
        for record in self.performance_history:
            tf_key = self._tf_combo_key(record.timeframes_used)
            tf_records[tf_key].append(record)

        for tf_key, records in tf_records.items():
            self.metrics_by_tf_combo[tf_key] = self._calculate_metrics(records)

        # По market regime
        regime_records = defaultdict(list)
        for record in self.performance_history:
            if record.market_regime:
                regime_records[record.market_regime].append(record)

        for regime, records in regime_records.items():
            self.metrics_by_regime[regime] = self._calculate_metrics(records)

    def _calculate_metrics(self, records: List[SignalPerformanceRecord]) -> ReliabilityMetrics:
        """Рассчитать metrics для списка записей."""
        if not records:
            return ReliabilityMetrics(
                total_signals=0,
                win_rate=0.5,
                avg_pnl_percent=0.0,
                avg_rr_achieved=0.0,
                win_streak=0,
                loss_streak=0,
                max_win_streak=0,
                max_loss_streak=0,
                quality_score_correlation=0.0,
                reliability_score=0.5
            )

        total = len(records)
        wins = sum(1 for r in records if r.was_profitable)
        win_rate = wins / total

        avg_pnl = np.mean([r.pnl_percent for r in records])
        avg_rr = np.mean([r.actual_rr_achieved for r in records])

        # Streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for record in records:
            if record.was_profitable:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)

        # Quality correlation
        if len(records) >= 10:
            qualities = [r.signal_quality for r in records]
            outcomes = [1.0 if r.was_profitable else 0.0 for r in records]
            correlation = np.corrcoef(qualities, outcomes)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # Composite reliability score
        reliability = (
            win_rate * 0.40 +  # Win rate is most important
            np.clip(avg_rr / 2.0, 0, 1) * 0.25 +  # Normalized R:R
            np.clip(correlation + 0.5, 0, 1) * 0.20 +  # Quality correlation
            (1.0 - min(max_loss_streak / 10.0, 1.0)) * 0.15  # Streak penalty
        )

        return ReliabilityMetrics(
            total_signals=total,
            win_rate=win_rate,
            avg_pnl_percent=avg_pnl,
            avg_rr_achieved=avg_rr,
            win_streak=current_win_streak,
            loss_streak=current_loss_streak,
            max_win_streak=max_win_streak,
            max_loss_streak=max_loss_streak,
            quality_score_correlation=correlation,
            reliability_score=reliability
        )

    def _get_mode_reliability(self, mode: str) -> float:
        """Получить reliability для synthesis mode."""
        if mode in self.metrics_by_mode:
            return self.metrics_by_mode[mode].reliability_score
        return 0.5  # Neutral для неизвестных

    def _get_tf_combo_reliability(self, timeframes: List[Timeframe]) -> float:
        """Получить reliability для timeframe combination."""
        tf_key = self._tf_combo_key(timeframes)
        if tf_key in self.metrics_by_tf_combo:
            return self.metrics_by_tf_combo[tf_key].reliability_score
        return 0.5

    def _get_regime_reliability(self, regime: Optional[MarketRegime]) -> float:
        """Получить reliability для market regime."""
        if regime and regime in self.metrics_by_regime:
            return self.metrics_by_regime[regime].reliability_score
        return 0.5

    def _get_recent_trend(self) -> float:
        """
        Получить recent performance trend.

        Returns:
            float: 0.0-1.0 (плохой trend -> хороший trend)
        """
        if len(self.recent_signals) < 5:
            return 0.5

        # Последние 10 сигналов vs предыдущие 40
        recent_10 = list(self.recent_signals)[-10:]
        previous_40 = list(self.recent_signals)[:-10] if len(self.recent_signals) > 10 else []

        if not previous_40:
            return 0.5

        recent_wr = sum(1 for r in recent_10 if r.was_profitable) / len(recent_10)
        previous_wr = sum(1 for r in previous_40 if r.was_profitable) / len(previous_40)

        # Trend score: 0.5 = no change, 0.0 = worsening, 1.0 = improving
        if previous_wr == 0:
            return 0.5

        trend_ratio = recent_wr / previous_wr
        trend_score = np.clip(0.5 + (trend_ratio - 1.0) * 0.5, 0.0, 1.0)

        return trend_score

    def _get_quality_correlation_factor(self, mode: str, current_quality: float) -> float:
        """
        Получить factor на основе корреляции quality с outcomes.

        Args:
            mode: Synthesis mode
            current_quality: Текущее качество сигнала

        Returns:
            float: 0.0-1.0
        """
        if mode not in self.metrics_by_mode:
            return 0.5

        metrics = self.metrics_by_mode[mode]

        # Если корреляция положительная и сильная
        if metrics.quality_score_correlation > 0.3:
            # High quality signal в режиме с хорошей корреляцией = boost
            return np.clip(current_quality * (1.0 + metrics.quality_score_correlation), 0.0, 1.0)

        # Если корреляция отрицательная (качество не помогает)
        elif metrics.quality_score_correlation < -0.3:
            # Не верим quality score в этом режиме
            return 0.5

        # Слабая корреляция
        return current_quality

    def _tf_combo_key(self, timeframes: List[Timeframe]) -> str:
        """Создать ключ для timeframe combination."""
        return "_".join(sorted(tf.value for tf in timeframes))

    def get_statistics(self) -> Dict:
        """Получить статистику tracker."""
        return {
            'total_records': len(self.performance_history),
            'metrics_by_mode': {
                mode: {
                    'total': m.total_signals,
                    'win_rate': m.win_rate,
                    'avg_rr': m.avg_rr_achieved,
                    'reliability': m.reliability_score
                }
                for mode, m in self.metrics_by_mode.items()
            },
            'recent_performance': {
                'total': len(self.recent_signals),
                'win_rate': (
                    sum(1 for r in self.recent_signals if r.was_profitable) / len(self.recent_signals)
                    if self.recent_signals else 0.0
                )
            }
        }


class MTFRiskManager:
    """
    Professional Multi-Timeframe Risk Manager.

    Интегрирует:
    - UnifiedSLTPCalculator для TP/SL
    - AdaptiveRiskCalculator для position sizing
    - SignalReliabilityTracker для reliability scoring
    - WeightedRiskFactors для risk assessment
    """

    def __init__(
        self,
        sltp_calc: Optional[UnifiedSLTPCalculator] = None,
        risk_calc: Optional[AdaptiveRiskCalculator] = None,
        reliability_tracker: Optional[SignalReliabilityTracker] = None
    ):
        """
        Инициализация MTF Risk Manager.

        Args:
            sltp_calc: UnifiedSLTPCalculator instance (или использует глобальный)
            risk_calc: AdaptiveRiskCalculator instance (или использует глобальный)
            reliability_tracker: SignalReliabilityTracker instance (или создает новый)
        """
        self.sltp_calc = sltp_calc or sltp_calculator
        self.risk_calc = risk_calc or adaptive_risk_calculator
        self.reliability_tracker = reliability_tracker or SignalReliabilityTracker()

        logger.info("MTFRiskManager initialized with professional calculators")

    def calculate_risk_parameters(
        self,
        signal: TradingSignal,
        current_price: float,
        synthesis_mode: str,
        timeframes_analyzed: List[Timeframe],
        signal_quality: float,
        alignment_score: float,
        divergence_severity: float,
        market_regime: Optional[MarketRegime],
        volatility_regime: Optional[str],
        atr: Optional[float] = None,
        ml_result: Optional[dict] = None,
        balance: Optional[float] = None
    ) -> Dict:
        """
        Рассчитать все risk management параметры.

        Args:
            signal: TradingSignal
            current_price: Текущая цена
            synthesis_mode: Режим synthesis (top_down/consensus/confluence)
            timeframes_analyzed: Список анализируемых timeframes
            signal_quality: Качество сигнала 0.0-1.0
            alignment_score: Alignment score 0.0-1.0
            divergence_severity: Severity дивергенции 0.0-1.0
            market_regime: Режим рынка
            volatility_regime: Режим волатильности
            atr: Average True Range
            ml_result: ML predictions (optional)
            balance: Доступный баланс (optional)

        Returns:
            dict: {
                'stop_loss_price': float,
                'take_profit_price': float,
                'risk_reward_ratio': float,
                'position_size_multiplier': float,
                'reliability_score': float,
                'risk_level': str,
                'calculation_method': str,
                'confidence': float,
                'warnings': List[str]
            }
        """
        warnings = []

        # 1. Calculate Reliability Score
        reliability_score = self.reliability_tracker.get_reliability_score(
            synthesis_mode=synthesis_mode,
            timeframes=timeframes_analyzed,
            market_regime=market_regime,
            signal_quality=signal_quality
        )

        # 2. Calculate Stop-Loss and Take-Profit
        sltp_result = self.sltp_calc.calculate(
            signal=signal,
            entry_price=current_price,
            ml_result=ml_result,
            atr=atr,
            market_regime=market_regime
        )

        # 3. Calculate Position Size Multiplier
        if balance:
            # Используем AdaptiveRiskCalculator
            risk_params = self.risk_calc.calculate(
                signal=signal,
                balance=balance,
                stop_loss_price=sltp_result.stop_loss,
                current_volatility=atr / current_price if atr else None,
                ml_confidence=sltp_result.confidence
            )

            # Position size multiplier = final_risk / base_risk
            position_multiplier = risk_params.final_risk_percent / risk_params.base_risk_percent
        else:
            # Fallback: базовый multiplier на основе reliability и quality
            position_multiplier = self._calculate_fallback_multiplier(
                reliability_score, signal_quality, sltp_result.confidence
            )

        # 4. Weighted Risk Assessment
        risk_factors = self._assess_risk_factors(
            volatility_regime=volatility_regime,
            market_regime=market_regime,
            alignment_score=alignment_score,
            divergence_severity=divergence_severity,
            signal_quality=signal_quality,
            reliability_score=reliability_score
        )

        composite_risk = risk_factors.calculate_composite_risk()
        risk_level = self._classify_risk_level(composite_risk)

        # 5. Generate Warnings
        if composite_risk > 0.7:
            warnings.append(f"High composite risk score: {composite_risk:.2f}")

        if reliability_score < 0.4:
            warnings.append(f"Low reliability for this signal type: {reliability_score:.2f}")

        if sltp_result.calculation_method == "fixed":
            warnings.append("Using fixed SL/TP fallback (ML/ATR unavailable)")

        logger.info(
            f"MTF Risk Parameters calculated: "
            f"SL={sltp_result.stop_loss:.2f}, TP={sltp_result.take_profit:.2f}, "
            f"R:R={sltp_result.risk_reward_ratio:.2f}, "
            f"pos_mult={position_multiplier:.2f}, "
            f"reliability={reliability_score:.2f}, "
            f"risk_level={risk_level}"
        )

        return {
            'stop_loss_price': sltp_result.stop_loss,
            'take_profit_price': sltp_result.take_profit,
            'risk_reward_ratio': sltp_result.risk_reward_ratio,
            'trailing_start_profit': sltp_result.trailing_start_profit,
            'position_size_multiplier': position_multiplier,
            'reliability_score': reliability_score,
            'risk_level': risk_level,
            'composite_risk_score': composite_risk,
            'calculation_method': sltp_result.calculation_method,
            'confidence': sltp_result.confidence,
            'warnings': warnings,
            'sltp_reasoning': sltp_result.reasoning
        }

    def _calculate_fallback_multiplier(
        self,
        reliability: float,
        quality: float,
        confidence: float
    ) -> float:
        """
        Fallback position size multiplier когда balance не доступен.

        Args:
            reliability: Reliability score
            quality: Signal quality
            confidence: Confidence score

        Returns:
            float: Position multiplier [0.3, 1.5]
        """
        # Weighted average
        base_multiplier = (
            reliability * 0.40 +
            quality * 0.35 +
            confidence * 0.25
        )

        # Scale to [0.3, 1.5]
        multiplier = 0.3 + (base_multiplier * 1.2)

        return np.clip(multiplier, 0.3, 1.5)

    def _assess_risk_factors(
        self,
        volatility_regime: Optional[str],
        market_regime: Optional[MarketRegime],
        alignment_score: float,
        divergence_severity: float,
        signal_quality: float,
        reliability_score: float
    ) -> WeightedRiskFactors:
        """
        Assess weighted risk factors.

        Returns:
            WeightedRiskFactors
        """
        factors = WeightedRiskFactors()

        # Market factors
        if volatility_regime == "high":
            factors.high_volatility = 1.0
        elif volatility_regime == "medium":
            factors.high_volatility = 0.5

        if market_regime in [MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY]:
            factors.regime_uncertainty = 0.7

        # Signal factors
        if alignment_score < 0.65:
            factors.low_alignment = 1.0 - alignment_score

        factors.divergence_present = divergence_severity

        if signal_quality < 0.65:
            factors.low_quality = 1.0 - signal_quality

        # Historical factors
        if reliability_score < 0.5:
            factors.poor_reliability = 1.0 - reliability_score

        if market_regime in [MarketRegime.DISTRIBUTION, MarketRegime.ACCUMULATION]:
            # Специфичные режимы с повышенным риском
            factors.unfavorable_regime = 0.6

        # Recent losses (из tracker)
        if len(self.reliability_tracker.recent_signals) >= 5:
            recent_5 = list(self.reliability_tracker.recent_signals)[-5:]
            losses = sum(1 for r in recent_5 if not r.was_profitable)
            factors.recent_losses = losses / 5.0

        return factors

    def _classify_risk_level(self, composite_risk: float) -> str:
        """
        Classify risk level based on composite score.

        Args:
            composite_risk: Composite risk score 0.0-1.0

        Returns:
            str: "LOW" / "NORMAL" / "HIGH" / "EXTREME"
        """
        if composite_risk < 0.25:
            return "LOW"
        elif composite_risk < 0.50:
            return "NORMAL"
        elif composite_risk < 0.75:
            return "HIGH"
        else:
            return "EXTREME"

    def record_signal_outcome(
        self,
        synthesis_mode: str,
        timeframes_used: List[Timeframe],
        signal_quality: float,
        alignment_score: float,
        market_regime: Optional[MarketRegime],
        volatility_regime: Optional[str],
        was_profitable: bool,
        pnl_percent: float,
        max_adverse_excursion: float,
        max_favorable_excursion: float,
        entry_time: datetime,
        exit_time: Optional[datetime] = None,
        hit_take_profit: bool = False,
        hit_stop_loss: bool = False,
        actual_rr_achieved: float = 0.0
    ):
        """
        Записать результат сигнала для обучения reliability tracker.

        Args:
            synthesis_mode: Режим synthesis
            timeframes_used: Использованные timeframes
            signal_quality: Качество сигнала
            alignment_score: Alignment score
            market_regime: Режим рынка
            volatility_regime: Режим волатильности
            was_profitable: Прибыльная ли сделка
            pnl_percent: P&L в процентах
            max_adverse_excursion: Максимальная просадка
            max_favorable_excursion: Максимальная прибыль
            entry_time: Время входа
            exit_time: Время выхода
            hit_take_profit: Достиг ли TP
            hit_stop_loss: Достиг ли SL
            actual_rr_achieved: Фактический R:R
        """
        hold_duration = None
        if exit_time:
            hold_duration = int((exit_time - entry_time).total_seconds() / 60)

        record = SignalPerformanceRecord(
            synthesis_mode=synthesis_mode,
            timeframes_used=timeframes_used,
            signal_quality=signal_quality,
            alignment_score=alignment_score,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            was_profitable=was_profitable,
            pnl_percent=pnl_percent,
            max_adverse_excursion=max_adverse_excursion,
            max_favorable_excursion=max_favorable_excursion,
            entry_time=entry_time,
            exit_time=exit_time,
            hold_duration_minutes=hold_duration,
            hit_take_profit=hit_take_profit,
            hit_stop_loss=hit_stop_loss,
            actual_rr_achieved=actual_rr_achieved
        )

        self.reliability_tracker.record_signal_outcome(record)

        # Также записываем в AdaptiveRiskCalculator для Kelly/adaptive расчетов
        self.risk_calc.record_trade(
            is_win=was_profitable,
            pnl=pnl_percent  # В процентах, не в USDT, но для статистики подходит
        )

    def get_statistics(self) -> Dict:
        """Получить статистику MTF Risk Manager."""
        return {
            'reliability_tracker': self.reliability_tracker.get_statistics(),
            'adaptive_risk_calculator': self.risk_calc.get_statistics()
        }


# Глобальный экземпляр
mtf_risk_manager = MTFRiskManager()
