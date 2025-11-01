"""
Timeframe Signal Synthesizer - синтез финального multi-timeframe сигнала.

Функциональность:
- Комбинирование сигналов со всех таймфреймов
- Три режима synthesis: Top-Down, Consensus, Confluence-Required
- Модуляция confidence на основе alignment
- Professional Risk Management интеграция (MTFRiskManager)
- Quality scoring финального сигнала

Synthesis Modes:
1. Top-Down: HTF определяет направление, LTF - точку входа
2. Consensus: Взвешенный консенсус всех TF
3. Confluence-Required: Только при согласии всех TF

УЛУЧШЕНИЯ (заменяет упрощенную логику):
- ML/ATR/Regime-based TP/SL вместо фиксированного R:R 2:1
- Kelly Criterion position sizing вместо простых multipliers
- Weighted risk assessment вместо счетчика факторов
- Historical reliability tracking вместо статичного 0.0

Путь: backend/strategies/mtf/timeframe_signal_synthesizer.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

from core.logger import get_logger
from models.signal import TradingSignal, SignalType, SignalSource, SignalStrength
from strategies.mtf import ConfluenceZone
from strategies.mtf.timeframe_coordinator import Timeframe
from strategies.mtf.timeframe_analyzer import TimeframeAnalysisResult
from strategies.mtf.timeframe_aligner import (
  TimeframeAlignment,
  AlignmentType,
  DivergenceType
)
from strategies.mtf.mtf_risk_manager import MTFRiskManager, mtf_risk_manager
from strategy.risk_models import MarketRegime

logger = get_logger(__name__)


class SynthesisMode(Enum):
  """Режимы синтеза сигналов."""
  TOP_DOWN = "top_down"  # HTF → LTF каскад
  CONSENSUS = "consensus"  # Взвешенный консенсус
  CONFLUENCE = "confluence"  # Требуется полное согласие


@dataclass
class MultiTimeframeSignal:
  """Финальный multi-timeframe сигнал."""
  # Основной сигнал
  signal: TradingSignal

  # MTF контекст
  synthesis_mode: SynthesisMode
  timeframes_analyzed: int
  timeframes_agreeing: int

  # Alignment информация
  alignment_score: float
  alignment_type: AlignmentType

  # Confluence detection
  has_confluence: bool
  confluence_zones_count: int
  confluence_zones: List['ConfluenceZone']

  # Divergence detection
  divergence_detected: bool
  divergence_type: Optional['DivergenceType']
  divergence_severity: float
  divergence_details: str


  # Quality metrics
  signal_quality: float  # 0.0-1.0, композитная метрика качества
  reliability_score: float  # Надежность на основе истории

  # Risk management
  recommended_position_size_multiplier: float
  recommended_stop_loss_price: Optional[float] = None
  recommended_take_profit_price: Optional[float] = None
  stop_loss_timeframe: Optional[Timeframe] = None

  # Детали по таймфреймам
  timeframe_signals: Dict[Timeframe, Optional[TradingSignal]] = field(default_factory=dict)
  higher_timeframe_context: Dict = field(default_factory=dict)

  # Warnings и caveats
  warnings: List[str] = field(default_factory=list)
  risk_level: str = "NORMAL"  # LOW, NORMAL, HIGH, EXTREME

  # Metadata
  timestamp: int = 0
  synthesis_duration_ms: float = 0.0


@dataclass
class SynthesizerConfig:
  """Конфигурация Signal Synthesizer."""
  # Режим синтеза
  mode: SynthesisMode = SynthesisMode.TOP_DOWN

  # Веса таймфреймов (для consensus mode)
  timeframe_weights: Dict[Timeframe, float] = field(default_factory=lambda: {
    Timeframe.M1: 0.10,
    Timeframe.M5: 0.20,
    Timeframe.M15: 0.30,
    Timeframe.H1: 0.40
  })

  # Таймфреймы
  primary_timeframe: Timeframe = Timeframe.H1  # Для тренда
  execution_timeframe: Timeframe = Timeframe.M1  # Для входа
  stop_loss_timeframe: Timeframe = Timeframe.M15  # Для стоп-лосса

  # Thresholds
  min_signal_quality: float = 0.6  # Минимальное качество сигнала
  min_timeframes_required: int = 2  # Минимум TF для сигнала

  # Top-Down mode settings
  require_htf_confirmation: bool = True  # Требовать подтверждение от HTF
  allow_ltf_contrary_signal: bool = False  # Разрешить LTF против HTF

  # Consensus mode settings
  consensus_threshold: float = 0.7  # Минимальный вес согласия

  # Confluence mode settings
  require_all_timeframes: bool = True  # Все TF должны согласиться
  allow_neutral_timeframes: bool = True  # Разрешить HOLD как согласие

  # Risk Management
  enable_dynamic_position_sizing: bool = True
  base_position_size: float = 1.0
  max_position_multiplier: float = 1.5
  min_position_multiplier: float = 0.3

  # Stop-loss placement
  use_higher_tf_for_stops: bool = True  # Использовать swing levels с HTF
  atr_multiplier_for_stops: float = 2.0  # ATR × multiplier для stop

  # Quality scoring weights
  quality_weights: Dict[str, float] = field(default_factory=lambda: {
    'alignment_score': 0.30,
    'higher_tf_confirmation': 0.25,
    'confluence_presence': 0.20,
    'divergence_absence': 0.15,
    'volume_confirmation': 0.10
  })


class TimeframeSignalSynthesizer:
  """
  Синтезатор multi-timeframe сигналов.

  Комбинирует анализ всех таймфреймов в единый
  высококачественный торговый сигнал.
  """

  def __init__(
      self,
      config: SynthesizerConfig,
      risk_manager: Optional[MTFRiskManager] = None
  ):
    """
    Инициализация синтезатора.

    Args:
        config: Конфигурация synthesis
        risk_manager: MTF Risk Manager (или использует глобальный)
    """
    self.config = config
    self.risk_manager = risk_manager or mtf_risk_manager

    # Статистика
    self.total_syntheses = 0
    self.signals_generated = 0
    self.signals_by_mode = {mode: 0 for mode in SynthesisMode}
    self.high_quality_signals = 0

    logger.info(
      f"Инициализирован TimeframeSignalSynthesizer: "
      f"mode={config.mode.value}, "
      f"primary_tf={config.primary_timeframe.value}, "
      f"execution_tf={config.execution_timeframe.value}, "
      f"professional_risk_management=True"
    )

  def synthesize_signal(
      self,
      tf_results: Dict[Timeframe, TimeframeAnalysisResult],
      alignment: TimeframeAlignment,
      symbol: str,
      current_price: float
  ) -> Optional[MultiTimeframeSignal]:
    """
    Синтезировать финальный multi-timeframe сигнал.

    Args:
        tf_results: Результаты анализа каждого TF
        alignment: Результат alignment check
        symbol: Торговая пара
        current_price: Текущая цена

    Returns:
        MultiTimeframeSignal или None если нет консенсуса
    """
    import time
    start_time = time.time()

    self.total_syntheses += 1

    if not tf_results:
      logger.warning(f"Нет данных таймфреймов для синтеза сигнала {symbol}")
      return None

    # Выбор метода синтеза
    if self.config.mode == SynthesisMode.TOP_DOWN:
      signal = self._synthesize_top_down(
        tf_results, alignment, symbol, current_price
      )
    elif self.config.mode == SynthesisMode.CONSENSUS:
      signal = self._synthesize_consensus(
        tf_results, alignment, symbol, current_price
      )
    elif self.config.mode == SynthesisMode.CONFLUENCE:
      signal = self._synthesize_confluence(
        tf_results, alignment, symbol, current_price
      )
    else:
      logger.error(f"Unknown synthesis mode: {self.config.mode}")
      return None

    if signal:
      # Расчет quality metrics
      signal = self._calculate_signal_quality(signal, tf_results, alignment)

      # Risk management parameters
      signal = self._calculate_risk_parameters(
        signal, tf_results, alignment, current_price
      )

      # Валидация качества
      if signal.signal_quality < self.config.min_signal_quality:
        logger.debug(
          f"Сигнал отклонен: качество {signal.signal_quality:.2f} < "
          f"порог {self.config.min_signal_quality}"
        )
        return None

      # Статистика
      self.signals_generated += 1
      self.signals_by_mode[self.config.mode] += 1

      if signal.signal_quality >= 0.8:
        self.high_quality_signals += 1

      # Метаданные
      signal.timestamp = int(datetime.now().timestamp() * 1000)
      signal.synthesis_duration_ms = (time.time() - start_time) * 1000

      logger.info(
        f"✅ MTF SIGNAL [{symbol}]: {signal.signal.signal_type.value}, "
        f"quality={signal.signal_quality:.2f}, "
        f"confidence={signal.signal.confidence:.2f}, "
        f"alignment={signal.alignment_score:.2f}, "
        f"position_mult={signal.recommended_position_size_multiplier:.2f}"
      )

    return signal

  def _synthesize_top_down(
      self,
      tf_results: Dict[Timeframe, TimeframeAnalysisResult],
      alignment: TimeframeAlignment,
      symbol: str,
      current_price: float
  ) -> Optional[MultiTimeframeSignal]:
    """
    Top-Down синтез: HTF определяет направление, LTF - точку входа.

    Логика:
    1. Проверить HTF тренд (H1)
    2. Если тренд есть, ждать подтверждение от M15
    3. Искать точку входа на M5/M1
    4. Все должны согласиться с направлением HTF
    """
    # Шаг 1: Получить HTF тренд
    htf = self.config.primary_timeframe

    if htf not in tf_results:
      logger.warning(f"HTF {htf.value} отсутствует в результатах")
      return None

    htf_result = tf_results[htf]
    htf_trend = htf_result.regime.trend_direction
    htf_signal = htf_result.timeframe_signal

    if htf_trend == 0:
      logger.debug("HTF в ranging режиме, пропускаем сигнал")
      return None

    # Требуется подтверждение от HTF?
    if self.config.require_htf_confirmation:
      if not htf_signal or htf_signal.signal_type == SignalType.HOLD:
        logger.debug("HTF не дает сигнал, пропускаем")
        return None

      # Проверяем что сигнал согласуется с трендом
      if (htf_trend > 0 and htf_signal.signal_type != SignalType.BUY) or \
          (htf_trend < 0 and htf_signal.signal_type != SignalType.SELL):
        logger.debug("HTF сигнал не согласуется с трендом")
        return None

    # Шаг 2: Проверить intermediate TF (M15)
    intermediate_tfs = [Timeframe.M15, Timeframe.M5]
    intermediate_confirmation = False

    for itf in intermediate_tfs:
      if itf in tf_results:
        itf_signal = tf_results[itf].timeframe_signal

        if itf_signal and itf_signal.signal_type != SignalType.HOLD:
          # Проверяем согласие с HTF
          if (htf_trend > 0 and itf_signal.signal_type == SignalType.BUY) or \
              (htf_trend < 0 and itf_signal.signal_type == SignalType.SELL):
            intermediate_confirmation = True
            break

    if not intermediate_confirmation:
      logger.debug("Нет подтверждения от intermediate TF")
      return None

    # Шаг 3: Получить точку входа с execution TF
    etf = self.config.execution_timeframe

    if etf not in tf_results:
      logger.warning(f"Execution TF {etf.value} отсутствует")
      # Fallback - используем HTF сигнал
      final_signal_type = SignalType.BUY if htf_trend > 0 else SignalType.SELL
      execution_signal = htf_signal
    else:
      etf_signal = tf_results[etf].timeframe_signal

      # Если execution TF дает противоположный сигнал
      if etf_signal and not self.config.allow_ltf_contrary_signal:
        if (htf_trend > 0 and etf_signal.signal_type == SignalType.SELL) or \
            (htf_trend < 0 and etf_signal.signal_type == SignalType.BUY):
          logger.debug("Execution TF дает противоположный сигнал, пропускаем")
          return None

      final_signal_type = SignalType.BUY if htf_trend > 0 else SignalType.SELL
      execution_signal = etf_signal if etf_signal else htf_signal

    # Подсчет согласия
    agreeing_tfs = []
    for tf, result in tf_results.items():
      if result.timeframe_signal:
        if result.timeframe_signal.signal_type == final_signal_type:
          agreeing_tfs.append(tf)

    # Создание базового сигнала
    base_confidence = 0.70  # Базовая для top-down

    # Boost за согласие TF
    agreement_boost = len(agreeing_tfs) / len(tf_results) * 0.20

    # Boost за HTF trend strength
    htf_boost = htf_result.regime.trend_strength * 0.10

    final_confidence = min(base_confidence + agreement_boost + htf_boost, 0.95)

    # Создаем TradingSignal
    signal = TradingSignal(
      symbol=symbol,
      signal_type=final_signal_type,
      source=SignalSource.STRATEGY,
      strength=self._confidence_to_strength(final_confidence),
      price=current_price,
      confidence=final_confidence,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason=(
        f"[MTF Top-Down] HTF={htf.value} trend={htf_trend}, "
        f"agreement={len(agreeing_tfs)}/{len(tf_results)} TF"
      ),
      metadata={
        'synthesis_mode': 'top_down',
        'htf': htf.value,
        'htf_trend': htf_trend,
        'execution_tf': etf.value
      }
    )

    # Собираем TF signals
    timeframe_signals = {
      tf: result.timeframe_signal
      for tf, result in tf_results.items()
    }

    mtf_signal = MultiTimeframeSignal(
      signal=signal,
      synthesis_mode=SynthesisMode.TOP_DOWN,
      timeframes_analyzed=len(tf_results),
      timeframes_agreeing=len(agreeing_tfs),
      alignment_score=alignment.alignment_score,
      alignment_type=alignment.alignment_type,
      has_confluence=alignment.has_strong_confluence,
      confluence_zones_count=len(alignment.confluence_zones),
      confluence_zones=alignment.confluence_zones,

      divergence_detected=(alignment.divergence_type != DivergenceType.NO_DIVERGENCE),
      divergence_type=alignment.divergence_type,
      divergence_severity=alignment.divergence_severity,
      divergence_details=alignment.divergence_details,
      signal_quality=0.0,  # Будет рассчитано позже
      reliability_score=0.0,
      recommended_position_size_multiplier=1.0,
      timeframe_signals=timeframe_signals,
      higher_timeframe_context={
        'trend': htf_trend,
        'trend_strength': htf_result.regime.trend_strength,
        'volatility': htf_result.regime.volatility_regime.value
      }
    )

    return mtf_signal

  def _synthesize_consensus(
      self,
      tf_results: Dict[Timeframe, TimeframeAnalysisResult],
      alignment: TimeframeAlignment,
      symbol: str,
      current_price: float
  ) -> Optional[MultiTimeframeSignal]:
    """
    Consensus синтез: взвешенный консенсус всех TF.

    Логика:
    - Каждый TF голосует своим весом
    - Требуется минимальный weighted agreement
    - Confidence = weighted average
    """
    # Собираем голоса
    buy_weight = 0.0
    sell_weight = 0.0
    hold_weight = 0.0

    tf_votes = {}

    for tf, result in tf_results.items():
      weight = self.config.timeframe_weights.get(tf, 0.1)

      if result.timeframe_signal:
        signal_type = result.timeframe_signal.signal_type

        if signal_type == SignalType.BUY:
          buy_weight += weight
          tf_votes[tf] = ('BUY', weight)
        elif signal_type == SignalType.SELL:
          sell_weight += weight
          tf_votes[tf] = ('SELL', weight)
        else:
          hold_weight += weight
          tf_votes[tf] = ('HOLD', weight)
      else:
        hold_weight += weight
        tf_votes[tf] = ('HOLD', weight)

    total_weight = buy_weight + sell_weight + hold_weight

    if total_weight == 0:
      logger.debug("Нет голосов от TF")
      return None

    # Нормализация
    buy_ratio = buy_weight / total_weight
    sell_ratio = sell_weight / total_weight

    # Определяем победителя
    if buy_ratio > sell_ratio and buy_ratio >= self.config.consensus_threshold:
      final_signal_type = SignalType.BUY
      consensus_ratio = buy_ratio
    elif sell_ratio > buy_ratio and sell_ratio >= self.config.consensus_threshold:
      final_signal_type = SignalType.SELL
      consensus_ratio = sell_ratio
    else:
      logger.debug(
        f"Нет консенсуса: BUY={buy_ratio:.2f}, SELL={sell_ratio:.2f}, "
        f"threshold={self.config.consensus_threshold}"
      )
      return None

    # Подсчет согласных TF
    agreeing_tfs = [
      tf for tf, (vote, _) in tf_votes.items()
      if vote == final_signal_type.value
    ]

    # Расчет weighted confidence
    weighted_confidences = []

    for tf in agreeing_tfs:
      result = tf_results[tf]
      if result.timeframe_signal:
        weight = self.config.timeframe_weights.get(tf, 0.1)
        weighted_confidences.append(
          result.timeframe_signal.confidence * weight
        )

    if not weighted_confidences:
      final_confidence = consensus_ratio
    else:
      # Weighted average confidence
      final_confidence = sum(weighted_confidences) / sum(
        self.config.timeframe_weights.get(tf, 0.1) for tf in agreeing_tfs
      )

    # Модуляция на основе consensus ratio
    final_confidence *= consensus_ratio
    final_confidence = min(final_confidence, 0.95)

    # Создаем TradingSignal
    signal = TradingSignal(
      symbol=symbol,
      signal_type=final_signal_type,
      source=SignalSource.CONSENSUS,
      strength=self._confidence_to_strength(final_confidence),
      price=current_price,
      confidence=final_confidence,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason=(
        f"[MTF Consensus] {len(agreeing_tfs)}/{len(tf_results)} TF agree, "
        f"weighted_ratio={consensus_ratio:.2f}"
      ),
      metadata={
        'synthesis_mode': 'consensus',
        'consensus_ratio': consensus_ratio,
        'buy_weight': buy_weight,
        'sell_weight': sell_weight
      }
    )

    timeframe_signals = {
      tf: result.timeframe_signal
      for tf, result in tf_results.items()
    }

    mtf_signal = MultiTimeframeSignal(
      signal=signal,
      synthesis_mode=SynthesisMode.CONSENSUS,
      timeframes_analyzed=len(tf_results),
      timeframes_agreeing=len(agreeing_tfs),
      alignment_score=alignment.alignment_score,
      alignment_type=alignment.alignment_type,
      has_confluence=alignment.has_strong_confluence,
      confluence_zones_count=len(alignment.confluence_zones),
      confluence_zones=alignment.confluence_zones,

      divergence_detected=(alignment.divergence_type != DivergenceType.NO_DIVERGENCE),
      divergence_type=alignment.divergence_type,
      divergence_severity=alignment.divergence_severity,
      divergence_details=alignment.divergence_details,
      signal_quality=0.0,
      reliability_score=0.0,
      recommended_position_size_multiplier=1.0,
      timeframe_signals=timeframe_signals
    )

    return mtf_signal

  def _synthesize_confluence(
      self,
      tf_results: Dict[Timeframe, TimeframeAnalysisResult],
      alignment: TimeframeAlignment,
      symbol: str,
      current_price: float
  ) -> Optional[MultiTimeframeSignal]:
    """
    Confluence синтез: требуется полное согласие всех TF.

    Логика:
    - Все TF должны дать одинаковый сигнал
    - Или: разрешаем HOLD как согласие
    - Самый строгий режим
    """
    if len(tf_results) < self.config.min_timeframes_required:
      logger.debug(
        f"Недостаточно TF для confluence: "
        f"{len(tf_results)} < {self.config.min_timeframes_required}"
      )
      return None

    # Собираем сигналы
    signals = []

    for tf, result in tf_results.items():
      if result.timeframe_signal:
        signals.append(result.timeframe_signal.signal_type)
      else:
        if self.config.allow_neutral_timeframes:
          signals.append(SignalType.HOLD)
        else:
          logger.debug(f"TF {tf.value} не имеет сигнала")
          return None

    # Проверяем единогласие
    if self.config.require_all_timeframes:
      # Все должны быть одинаковые (кроме HOLD если разрешено)
      non_hold_signals = [s for s in signals if s != SignalType.HOLD]

      if not non_hold_signals:
        logger.debug("Все TF в HOLD")
        return None

      first_signal = non_hold_signals[0]

      if not all(s == first_signal or s == SignalType.HOLD for s in signals):
        logger.debug("Нет полного confluence: TF дают разные сигналы")
        return None

      final_signal_type = first_signal
      agreeing_tfs = list(tf_results.keys())

    else:
      # Достаточно большинства
      buy_count = signals.count(SignalType.BUY)
      sell_count = signals.count(SignalType.SELL)

      required_count = len(signals) if self.config.require_all_timeframes else (len(signals) // 2) + 1

      if buy_count >= required_count:
        final_signal_type = SignalType.BUY
        agreeing_tfs = [
          tf for tf, result in tf_results.items()
          if result.timeframe_signal and result.timeframe_signal.signal_type == SignalType.BUY
        ]
      elif sell_count >= required_count:
        final_signal_type = SignalType.SELL
        agreeing_tfs = [
          tf for tf, result in tf_results.items()
          if result.timeframe_signal and result.timeframe_signal.signal_type == SignalType.SELL
        ]
      else:
        logger.debug("Нет confluence: недостаточно согласия")
        return None

    # Высокая confidence для confluence mode (все согласны)
    base_confidence = 0.85

    # Boost за единогласие
    unanimity_boost = len(agreeing_tfs) / len(tf_results) * 0.10

    final_confidence = min(base_confidence + unanimity_boost, 0.95)

    # Создаем TradingSignal
    signal = TradingSignal(
      symbol=symbol,
      signal_type=final_signal_type,
      source=SignalSource.CONSENSUS,
      strength=SignalStrength.STRONG,  # Confluence всегда STRONG
      price=current_price,
      confidence=final_confidence,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason=(
        f"[MTF Confluence] ALL {len(agreeing_tfs)} TF agree on {final_signal_type.value}"
      ),
      metadata={
        'synthesis_mode': 'confluence',
        'unanimity': len(agreeing_tfs) == len(tf_results)
      }
    )

    timeframe_signals = {
      tf: result.timeframe_signal
      for tf, result in tf_results.items()
    }

    mtf_signal = MultiTimeframeSignal(
      signal=signal,
      synthesis_mode=SynthesisMode.CONFLUENCE,
      timeframes_analyzed=len(tf_results),
      timeframes_agreeing=len(agreeing_tfs),
      alignment_score=alignment.alignment_score,
      alignment_type=alignment.alignment_type,
      has_confluence=alignment.has_strong_confluence,
      confluence_zones_count=len(alignment.confluence_zones),
      confluence_zones=alignment.confluence_zones,

      divergence_detected=(alignment.divergence_type != DivergenceType.NO_DIVERGENCE),
      divergence_type=alignment.divergence_type,
      divergence_severity=alignment.divergence_severity,
      divergence_details=alignment.divergence_details,
      signal_quality=0.0,
      reliability_score=0.0,
      recommended_position_size_multiplier=1.0,
      timeframe_signals=timeframe_signals
    )

    return mtf_signal

  def _calculate_signal_quality(
      self,
      mtf_signal: MultiTimeframeSignal,
      tf_results: Dict[Timeframe, TimeframeAnalysisResult],
      alignment: TimeframeAlignment
  ) -> MultiTimeframeSignal:
    """
    Рассчитать композитную метрику качества сигнала.

    Quality score складывается из:
    - Alignment score
    - HTF confirmation
    - Confluence presence
    - Divergence absence
    - Volume confirmation
    """
    scores = {}
    weights = self.config.quality_weights

    # 1. Alignment score
    scores['alignment_score'] = alignment.alignment_score

    # 2. Higher TF confirmation
    htf = self.config.primary_timeframe
    if htf in tf_results:
      htf_result = tf_results[htf]

      if htf_result.timeframe_signal:
        # Проверяем согласие HTF с финальным сигналом
        if htf_result.timeframe_signal.signal_type == mtf_signal.signal.signal_type:
          scores['higher_tf_confirmation'] = 1.0
        else:
          scores['higher_tf_confirmation'] = 0.0
      else:
        scores['higher_tf_confirmation'] = 0.5  # Нейтрально
    else:
      scores['higher_tf_confirmation'] = 0.0

    # 3. Confluence presence
    if alignment.has_strong_confluence:
      scores['confluence_presence'] = 1.0
    elif alignment.confluence_zones:
      scores['confluence_presence'] = 0.5
    else:
      scores['confluence_presence'] = 0.0

    # 4. Divergence absence (обратная метрика)
    if alignment.divergence_type == DivergenceType.NO_DIVERGENCE:
      scores['divergence_absence'] = 1.0
    else:
      scores['divergence_absence'] = max(0.0, 1.0 - alignment.divergence_severity)

    # 5. Volume confirmation
    # Проверяем volume ratio на согласных TF
    volume_confirmations = 0
    total_checked = 0

    for tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15]:
      if tf in tf_results:
        result = tf_results[tf]
        if result.indicators.volume_ratio:
          total_checked += 1
          # Volume выше среднего = подтверждение
          if result.indicators.volume_ratio > 1.0:
            volume_confirmations += 1

    if total_checked > 0:
      scores['volume_confirmation'] = volume_confirmations / total_checked
    else:
      scores['volume_confirmation'] = 0.5  # Нейтрально

    # Weighted sum
    quality_score = sum(
      scores.get(key, 0.0) * weight
      for key, weight in weights.items()
    )

    mtf_signal.signal_quality = quality_score

    logger.debug(
      f"Signal quality: {quality_score:.2f}, "
      f"components={scores}"
    )

    return mtf_signal

  def _calculate_risk_parameters(
      self,
      mtf_signal: MultiTimeframeSignal,
      tf_results: Dict[Timeframe, TimeframeAnalysisResult],
      alignment: TimeframeAlignment,
      current_price: float
  ) -> MultiTimeframeSignal:
    """
    Рассчитать risk management параметры используя professional MTFRiskManager.

    УЛУЧШЕНИЯ относительно упрощенной версии:
    - ML/ATR/Regime-based TP/SL вместо фиксированного R:R 2:1
    - Kelly Criterion position sizing вместо простых multipliers
    - Weighted risk assessment вместо счетчика факторов
    - Historical reliability tracking вместо статичного 0.0

    Args:
        mtf_signal: Multi-timeframe сигнал
        tf_results: Результаты анализа по timeframes
        alignment: Timeframe alignment
        current_price: Текущая цена

    Returns:
        mtf_signal с заполненными risk parameters
    """
    # Собираем данные для MTFRiskManager

    # 1. Определяем market regime (используем HTF)
    market_regime = None
    volatility_regime = None
    atr = None

    htf = self.config.primary_timeframe
    if htf in tf_results:
      htf_result = tf_results[htf]

      # Маппинг из TrendRegime/VolatilityRegime в MarketRegime
      market_regime = self._map_to_market_regime(
        htf_result.regime.trend_direction,
        htf_result.regime.volatility_regime.value
      )

      volatility_regime = htf_result.regime.volatility_regime.value
      atr = htf_result.indicators.atr

    # 2. Собираем список timeframes для reliability tracking
    timeframes_list = list(tf_results.keys())

    # 3. Вызываем MTFRiskManager для professional расчета
    risk_params = self.risk_manager.calculate_risk_parameters(
      signal=mtf_signal.signal,
      current_price=current_price,
      synthesis_mode=mtf_signal.synthesis_mode.value,
      timeframes_analyzed=timeframes_list,
      signal_quality=mtf_signal.signal_quality,
      alignment_score=alignment.alignment_score,
      divergence_severity=alignment.divergence_severity,
      market_regime=market_regime,
      volatility_regime=volatility_regime,
      atr=atr,
      ml_result=None,  # TODO: Интегрировать ML predictions если доступны
      balance=None  # TODO: Передавать balance если доступен для Kelly sizing
    )

    # 4. Заполняем mtf_signal из professional расчетов
    mtf_signal.recommended_stop_loss_price = risk_params['stop_loss_price']
    mtf_signal.recommended_take_profit_price = risk_params['take_profit_price']
    mtf_signal.recommended_position_size_multiplier = risk_params['position_size_multiplier']
    mtf_signal.reliability_score = risk_params['reliability_score']
    mtf_signal.risk_level = risk_params['risk_level']

    # Добавляем warnings из MTFRiskManager
    mtf_signal.warnings.extend(risk_params['warnings'])

    # Для stop_loss_timeframe используем stop_loss_timeframe из конфига
    if self.config.use_higher_tf_for_stops:
      mtf_signal.stop_loss_timeframe = self.config.stop_loss_timeframe

    logger.debug(
      f"Professional risk parameters calculated: "
      f"SL={mtf_signal.recommended_stop_loss_price:.2f}, "
      f"TP={mtf_signal.recommended_take_profit_price:.2f}, "
      f"R:R={risk_params['risk_reward_ratio']:.2f}, "
      f"position_mult={mtf_signal.recommended_position_size_multiplier:.2f}, "
      f"reliability={mtf_signal.reliability_score:.2f}, "
      f"risk_level={mtf_signal.risk_level}, "
      f"method={risk_params['calculation_method']}"
    )

    return mtf_signal

  def _map_to_market_regime(
      self,
      trend_direction: int,
      volatility_regime: str
  ) -> MarketRegime:
    """
    Маппинг из TrendRegime/VolatilityRegime в MarketRegime.

    Args:
        trend_direction: -1 (down), 0 (ranging), 1 (up)
        volatility_regime: 'low', 'medium', 'high'

    Returns:
        MarketRegime
    """
    # High volatility override
    if volatility_regime == 'high':
      return MarketRegime.HIGH_VOLATILITY

    # Ranging market
    if trend_direction == 0:
      return MarketRegime.RANGING

    # Trending market
    if abs(trend_direction) >= 0.7:
      return MarketRegime.STRONG_TREND
    elif abs(trend_direction) >= 0.3:
      return MarketRegime.MILD_TREND

    return MarketRegime.RANGING

  def _confidence_to_strength(self, confidence: float) -> SignalStrength:
    """Конвертировать confidence в strength."""
    if confidence >= 0.80:
      return SignalStrength.STRONG
    elif confidence >= 0.60:
      return SignalStrength.MEDIUM
    else:
      return SignalStrength.WEAK

  def get_statistics(self) -> Dict:
    """Получить статистику синтезатора."""
    return {
      'total_syntheses': self.total_syntheses,
      'signals_generated': self.signals_generated,
      'signal_generation_rate': (
        self.signals_generated / self.total_syntheses
        if self.total_syntheses > 0 else 0.0
      ),
      'signals_by_mode': {
        mode.value: count
        for mode, count in self.signals_by_mode.items()
      },
      'high_quality_signals': self.high_quality_signals,
      'high_quality_rate': (
        self.high_quality_signals / self.signals_generated
        if self.signals_generated > 0 else 0.0
      )
    }