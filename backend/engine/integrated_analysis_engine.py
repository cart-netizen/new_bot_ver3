"""
Integrated Analysis Engine - объединение всех продвинутых компонентов.

Интегрирует:
- Фаза 1: OrderBook-Aware Strategies
- Фаза 2: Adaptive Consensus Management
- Фаза 3: Multi-Timeframe Analysis

Предоставляет единый интерфейс для полного анализа рынка
с использованием всех доступных инструментов.

Путь: backend/engine/integrated_analysis_engine.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio

from backend.core.logger import get_logger
from backend.models.orderbook import OrderBookSnapshot, OrderBookMetrics
from backend.models.signal import TradingSignal, SignalType, SignalSource

from backend.strategy.candle_manager import Candle

# Фаза 1: OrderBook-Aware Strategies
from backend.strategies.strategy_manager import (
  ExtendedStrategyManager,
  ExtendedStrategyManagerConfig,
  ConsensusSignal
)

# Фаза 2: Adaptive Consensus
from backend.strategies.adaptive import AdaptiveConsensusConfig, AdaptiveConsensusManager

# Фаза 3: Multi-Timeframe Analysis
from backend.strategies.mtf import (
  MultiTimeframeManager,
  MTFManagerConfig,
  MultiTimeframeSignal,
  SynthesisMode
)

logger = get_logger(__name__)


class AnalysisMode(Enum):
  """Режимы работы integrated engine."""
  SINGLE_TF_ONLY = "single_tf_only"  # Только single-TF анализ
  MTF_ONLY = "mtf_only"  # Только MTF анализ
  HYBRID = "hybrid"  # Комбинация single-TF + MTF
  ADAPTIVE = "adaptive"  # Автоматический выбор на основе условий


@dataclass
class IntegratedSignal:
  """
  Финальный интегрированный сигнал от всей системы.

  Объединяет результаты от:
  - Strategy Manager (с адаптивными весами)
  - Multi-Timeframe Analysis
  - Consensus building
  """
  # Финальный сигнал для исполнения
  final_signal: TradingSignal

  # Источники сигнала
  source_analysis_mode: AnalysisMode
  used_single_tf: bool
  used_mtf: bool

  # Single-TF компоненты
  single_tf_consensus: Optional[ConsensusSignal] = None
  adaptive_weights: Optional[Dict[str, float]] = None
  market_regime: Optional[str] = None

  # MTF компоненты
  mtf_signal: Optional[MultiTimeframeSignal] = None
  mtf_alignment_score: Optional[float] = None
  mtf_quality: Optional[float] = None

  # Объединенные параметры
  combined_confidence: float = 0.0
  combined_quality_score: float = 0.0

  # Risk management
  recommended_position_multiplier: float = 1.0
  recommended_stop_loss: Optional[float] = None
  recommended_take_profit: Optional[float] = None
  risk_level: str = "NORMAL"

  # Метаданные
  analysis_timestamp: int = 0
  analysis_duration_ms: float = 0.0
  warnings: List[str] = field(default_factory=list)
  debug_info: Dict = field(default_factory=dict)


@dataclass
class IntegratedAnalysisConfig:
  """Конфигурация Integrated Analysis Engine."""
  # Режим работы
  analysis_mode: AnalysisMode = AnalysisMode.HYBRID

  # Включение компонентов
  enable_adaptive_consensus: bool = True
  enable_mtf_analysis: bool = True

  # Конфигурации компонентов
  strategy_manager_config: ExtendedStrategyManagerConfig = field(
    default_factory=ExtendedStrategyManagerConfig
  )
  adaptive_consensus_config: AdaptiveConsensusConfig = field(
    default_factory=AdaptiveConsensusConfig
  )
  mtf_config: MTFManagerConfig = field(
    default_factory=MTFManagerConfig
  )

  # Hybrid mode settings
  hybrid_mtf_priority: float = 0.6  # 60% вес MTF, 40% single-TF
  hybrid_min_agreement: bool = True  # Требовать согласия MTF и single-TF
  hybrid_conflict_resolution: str = "mtf"  # "mtf", "single_tf", "highest_quality"

  # Adaptive mode settings
  adaptive_use_mtf_when: List[str] = field(default_factory=lambda: [
    "trending_market",
    "high_alignment",
    "strong_confluence"
  ])
  adaptive_use_single_tf_when: List[str] = field(default_factory=lambda: [
    "ranging_market",
    "high_volatility",
    "low_alignment"
  ])

  # Quality thresholds
  min_combined_quality: float = 0.65
  min_confidence: float = 0.60

  # Logging
  verbose_logging: bool = True
  log_analysis_details: bool = True


class IntegratedAnalysisEngine:
  """
  Главный движок интегрированного анализа.

  Объединяет все продвинутые компоненты системы:
  - OrderBook-Aware Strategies
  - Adaptive Consensus
  - Multi-Timeframe Analysis

  Предоставляет единый интерфейс для получения
  высококачественных торговых сигналов.
  """

  def __init__(self, config: IntegratedAnalysisConfig):
    """
    Инициализация integrated engine.

    Args:
        config: Конфигурация engine
    """
    self.config = config

    # === Фаза 1: Strategy Manager ===
    self.strategy_manager = ExtendedStrategyManager(
      config.strategy_manager_config
    )

    # === Фаза 2: Adaptive Consensus ===
    if config.enable_adaptive_consensus:
      self.adaptive_consensus = AdaptiveConsensusManager(
        strategy_manager=self.strategy_manager,
        config=config.adaptive_consensus_config
      )
    else:
      self.adaptive_consensus = None

    # === Фаза 3: Multi-Timeframe Manager ===
    if config.enable_mtf_analysis:
      self.mtf_manager = MultiTimeframeManager(
        strategy_manager=self.strategy_manager,
        config=config.mtf_config
      )
    else:
      self.mtf_manager = None

    # Статистика
    self.total_analyses = 0
    self.signals_generated = 0
    self.signals_by_mode = {mode: 0 for mode in AnalysisMode}
    self.high_quality_signals = 0
    self.conflicts_detected = 0
    self.conflicts_resolved = 0

    logger.info(
      f"Инициализирован IntegratedAnalysisEngine: "
      f"mode={config.analysis_mode.value}, "
      f"adaptive_consensus={'✅' if config.enable_adaptive_consensus else '❌'}, "
      f"mtf={'✅' if config.enable_mtf_analysis else '❌'}"
    )

  async def initialize_symbol(self, symbol: str) -> bool:
    """
    Инициализировать символ для анализа.

    Args:
        symbol: Торговая пара

    Returns:
        True если успешно
    """
    success = True

    # Инициализация MTF (если включен)
    if self.mtf_manager:
      mtf_success = await self.mtf_manager.initialize_symbol(symbol)
      if not mtf_success:
        logger.warning(f"MTF инициализация для {symbol} не удалась")
        success = False

    # Инициализация Adaptive Consensus (если включен)
    if self.adaptive_consensus:
      # Adaptive consensus не требует явной инициализации символа
      pass

    if success:
      logger.info(f"✅ {symbol} инициализирован в IntegratedAnalysisEngine")
    else:
      logger.error(f"❌ Ошибка инициализации {symbol}")

    return success

  async def analyze(
      self,
      symbol: str,
      candles: List[Candle],
      current_price: float,
      orderbook: Optional[OrderBookSnapshot] = None,
      metrics: Optional[OrderBookMetrics] = None
  ) -> Optional[IntegratedSignal]:
    """
    Полный интегрированный анализ символа.

    Выполняет анализ согласно выбранному режиму:
    - SINGLE_TF_ONLY: только single-TF + adaptive weights
    - MTF_ONLY: только MTF анализ
    - HYBRID: комбинация single-TF и MTF
    - ADAPTIVE: автоматический выбор режима

    Args:
        symbol: Торговая пара
        candles: Свечи (primary timeframe)
        current_price: Текущая цена
        orderbook: Снимок стакана
        metrics: Метрики стакана

    Returns:
        IntegratedSignal или None
    """
    import time
    start_time = time.time()

    self.total_analyses += 1

    try:
      # Определяем режим анализа
      effective_mode = self._determine_analysis_mode(
        symbol, candles, orderbook, metrics
      )

      if self.config.verbose_logging:
        logger.info(
          f"[{symbol}] Анализ в режиме: {effective_mode.value}"
        )

      # Выполняем анализ по режиму
      if effective_mode == AnalysisMode.SINGLE_TF_ONLY:
        integrated_signal = await self._analyze_single_tf_mode(
          symbol, candles, current_price, orderbook, metrics
        )

      elif effective_mode == AnalysisMode.MTF_ONLY:
        integrated_signal = await self._analyze_mtf_mode(
          symbol, orderbook, metrics
        )

      elif effective_mode == AnalysisMode.HYBRID:
        integrated_signal = await self._analyze_hybrid_mode(
          symbol, candles, current_price, orderbook, metrics
        )

      else:
        logger.error(f"Unknown analysis mode: {effective_mode}")
        return None

      # Post-processing
      if integrated_signal:
        # Метаданные
        integrated_signal.analysis_timestamp = int(
          datetime.now().timestamp() * 1000
        )
        integrated_signal.analysis_duration_ms = (
            (time.time() - start_time) * 1000
        )

        # Quality check
        if integrated_signal.combined_quality_score < self.config.min_combined_quality:
          logger.debug(
            f"Signal rejected: quality {integrated_signal.combined_quality_score:.2f} "
            f"< threshold {self.config.min_combined_quality}"
          )
          return None

        if integrated_signal.combined_confidence < self.config.min_confidence:
          logger.debug(
            f"Signal rejected: confidence {integrated_signal.combined_confidence:.2f} "
            f"< threshold {self.config.min_confidence}"
          )
          return None

        # Статистика
        self.signals_generated += 1
        self.signals_by_mode[effective_mode] += 1

        if integrated_signal.combined_quality_score >= 0.85:
          self.high_quality_signals += 1

        # Логирование
        self._log_integrated_signal(symbol, integrated_signal)

      return integrated_signal

    except Exception as e:
      logger.error(f"Ошибка интегрированного анализа {symbol}: {e}", exc_info=True)
      return None

  def _determine_analysis_mode(
      self,
      symbol: str,
      candles: List[Candle],
      orderbook: Optional[OrderBookSnapshot],
      metrics: Optional[OrderBookMetrics]
  ) -> AnalysisMode:
    """
    Определить эффективный режим анализа.

    В ADAPTIVE режиме выбирает между single-TF и MTF
    на основе текущих рыночных условий используя профессиональный MarketRegimeDetector.

    Args:
        symbol: Торговая пара
        candles: Свечи
        orderbook: Стакан
        metrics: Метрики

    Returns:
        Эффективный режим анализа
    """
    if self.config.analysis_mode != AnalysisMode.ADAPTIVE:
      return self.config.analysis_mode

    # ============================================================================
    # PROFESSIONAL MARKET REGIME DETECTION
    # Replaces simplified SMA-based logic (old lines 373-383)
    # ============================================================================

    # Use professional MarketRegimeDetector if available via AdaptiveConsensusManager
    if self.adaptive_consensus and self.adaptive_consensus.regime_detector:
      regime = self.adaptive_consensus.regime_detector.detect_regime(
        symbol=symbol,
        candles=candles,
        orderbook_metrics=metrics
      )

      # Decision logic based on professional regime analysis
      # MTF is optimal for:
      # - Strong trends (STRONG_UPTREND, STRONG_DOWNTREND)
      # - Normal/Low volatility (stable market)
      # - High liquidity (quality data across timeframes)

      # Single-TF is optimal for:
      # - Ranging markets (RANGING)
      # - High volatility (CHOPPY_VOLATILE)
      # - Low liquidity (less reliable multi-timeframe data)

      from backend.strategies.adaptive.market_regime_detector import (
        TrendRegime, VolatilityRegime, LiquidityRegime
      )

      # Strong trend + stable volatility → MTF
      if regime.trend in [TrendRegime.STRONG_UPTREND, TrendRegime.STRONG_DOWNTREND]:
        if regime.volatility != VolatilityRegime.HIGH:
          logger.debug(
            f"{symbol}: Strong {regime.trend.value} + {regime.volatility.value} volatility "
            f"(ADX={regime.adx_value:.1f}) → MTF"
          )
          return AnalysisMode.MTF_ONLY if self.mtf_manager else AnalysisMode.SINGLE_TF_ONLY

      # High volatility → Single-TF (more responsive)
      if regime.volatility == VolatilityRegime.HIGH:
        logger.debug(
          f"{symbol}: High volatility (ATR={regime.atr_value:.2f}) → Single-TF"
        )
        return AnalysisMode.SINGLE_TF_ONLY

      # Ranging market → Single-TF (mean reversion strategies work better)
      if regime.trend == TrendRegime.RANGING:
        logger.debug(
          f"{symbol}: Ranging market (ADX={regime.adx_value:.1f}) → Single-TF"
        )
        return AnalysisMode.SINGLE_TF_ONLY

      # Low liquidity + High volatility → Single-TF only
      if regime.liquidity == LiquidityRegime.LOW and regime.volatility == VolatilityRegime.HIGH:
        logger.debug(
          f"{symbol}: Low liquidity + High volatility → Single-TF (dangerous regime)"
        )
        return AnalysisMode.SINGLE_TF_ONLY

      # Weak trend → HYBRID (use both for confirmation)
      if regime.trend in [TrendRegime.WEAK_UPTREND, TrendRegime.WEAK_DOWNTREND]:
        logger.debug(
          f"{symbol}: Weak {regime.trend.value} → HYBRID (require confirmation)"
        )
        return AnalysisMode.HYBRID if self.mtf_manager else AnalysisMode.SINGLE_TF_ONLY

      # Default: MTF if available
      logger.debug(
        f"{symbol}: Standard regime ({regime.trend.value}/{regime.volatility.value}) → "
        f"{'MTF' if self.mtf_manager else 'Single-TF'}"
      )
      return AnalysisMode.MTF_ONLY if self.mtf_manager else AnalysisMode.SINGLE_TF_ONLY

    # ============================================================================
    # END PROFESSIONAL MARKET REGIME DETECTION
    # ============================================================================

    # Fallback if MarketRegimeDetector not available
    logger.warning(
      f"{symbol}: MarketRegimeDetector not available, using fallback mode selection"
    )

    # Simple fallback: MTF if available, otherwise Single-TF
    if self.mtf_manager:
      return AnalysisMode.MTF_ONLY
    else:
      return AnalysisMode.SINGLE_TF_ONLY

  async def _analyze_single_tf_mode(
      self,
      symbol: str,
      candles: List[Candle],
      current_price: float,
      orderbook: Optional[OrderBookSnapshot],
      metrics: Optional[OrderBookMetrics]
  ) -> Optional[IntegratedSignal]:
    """
    Анализ в single-TF режиме с adaptive consensus.

    Returns:
        IntegratedSignal или None
    """
    # Получаем adaptive веса (если включено)
    adaptive_weights = None
    market_regime = None

    if self.adaptive_consensus:
      # Получаем список имен стратегий из strategy_manager
      strategy_names = list(self.strategy_manager.candle_strategies.keys()) + \
                       list(self.strategy_manager.orderbook_strategies.keys()) + \
                       list(self.strategy_manager.hybrid_strategies.keys())

      adaptive_weights = self.adaptive_consensus.weight_optimizer.get_optimal_weights(
        symbol=symbol,
        strategy_names=strategy_names
      )

    # Запускаем стратегии с adaptive consensus
    consensus = self.strategy_manager.analyze_with_consensus(
      symbol=symbol,
      candles=candles,
      current_price=current_price,
      orderbook=orderbook,
      metrics=metrics
    )

    if not consensus or not consensus.final_signal:
      return None

    # Создаем IntegratedSignal
    integrated_signal = IntegratedSignal(
      final_signal=consensus.final_signal,
      source_analysis_mode=AnalysisMode.SINGLE_TF_ONLY,
      used_single_tf=True,
      used_mtf=False,
      single_tf_consensus=consensus,
      adaptive_weights=adaptive_weights,
      market_regime=market_regime,
      combined_confidence=consensus.final_signal.confidence,
      combined_quality_score=consensus.consensus_confidence,
      recommended_position_multiplier=1.0,  # Default для single-TF
      risk_level="NORMAL"
    )

    return integrated_signal

  async def _analyze_mtf_mode(
      self,
      symbol: str,
      orderbook: Optional[OrderBookSnapshot],
      metrics: Optional[OrderBookMetrics]
  ) -> Optional[IntegratedSignal]:
    """
    Анализ в MTF-only режиме.

    Returns:
        IntegratedSignal или None
    """
    if not self.mtf_manager:
      logger.warning("MTF manager не инициализирован")
      return None

    # MTF анализ
    mtf_signal = await self.mtf_manager.analyze_symbol(
      symbol=symbol,
      orderbook=orderbook,
      metrics=metrics
    )

    if not mtf_signal:
      return None

    # ============================================================
    # CRITICAL FIX: Передача MTF параметров через metadata
    # ============================================================
    # Проблема: ExecutionManager получает только final_signal (TradingSignal)
    # и НЕ ВИДИТ recommended параметры из IntegratedSignal.
    # Решение: Добавляем MTF параметры в signal.metadata для передачи в ExecutionManager.
    #
    # Это избегает дублирования расчета SL/TP:
    # - MTFRiskManager рассчитывает SL/TP один раз (здесь)
    # - ExecutionManager использует эти значения (не пересчитывает)
    # ============================================================

    final_signal = mtf_signal.signal

    # Инициализируем metadata если его нет
    if final_signal.metadata is None:
      final_signal.metadata = {}

    # Добавляем MTF risk parameters
    final_signal.metadata['mtf_recommended_stop_loss'] = mtf_signal.recommended_stop_loss_price
    final_signal.metadata['mtf_recommended_take_profit'] = mtf_signal.recommended_take_profit_price
    final_signal.metadata['mtf_position_multiplier'] = mtf_signal.recommended_position_size_multiplier
    final_signal.metadata['mtf_reliability_score'] = mtf_signal.reliability_score
    final_signal.metadata['mtf_risk_level'] = mtf_signal.risk_level
    final_signal.metadata['mtf_signal_quality'] = mtf_signal.signal_quality
    final_signal.metadata['mtf_alignment_score'] = mtf_signal.alignment_score

    # Флаг для ExecutionManager
    final_signal.metadata['has_mtf_risk_params'] = True

    # MTF synthesis metadata
    final_signal.metadata['synthesis_mode'] = mtf_signal.synthesis_mode.value
    final_signal.metadata['timeframes_analyzed'] = mtf_signal.timeframes_analyzed
    final_signal.metadata['timeframes_agreeing'] = mtf_signal.timeframes_agreeing

    logger.info(
      f"{mtf_signal.signal.symbol} | ✅ MTF risk parameters добавлены в metadata: "
      f"SL=${mtf_signal.recommended_stop_loss_price:.2f}, "
      f"TP=${mtf_signal.recommended_take_profit_price:.2f}, "
      f"reliability={mtf_signal.reliability_score:.3f}, "
      f"risk_level={mtf_signal.risk_level}"
    )

    # Создаем IntegratedSignal
    integrated_signal = IntegratedSignal(
      final_signal=final_signal,
      source_analysis_mode=AnalysisMode.MTF_ONLY,
      used_single_tf=False,
      used_mtf=True,
      mtf_signal=mtf_signal,
      mtf_alignment_score=mtf_signal.alignment_score,
      mtf_quality=mtf_signal.signal_quality,
      combined_confidence=mtf_signal.signal.confidence,
      combined_quality_score=mtf_signal.signal_quality,
      recommended_position_multiplier=mtf_signal.recommended_position_size_multiplier,
      recommended_stop_loss=mtf_signal.recommended_stop_loss_price,
      recommended_take_profit=mtf_signal.recommended_take_profit_price,
      risk_level=mtf_signal.risk_level,
      warnings=mtf_signal.warnings
    )

    return integrated_signal

  async def _analyze_hybrid_mode(
      self,
      symbol: str,
      candles: List[Candle],
      current_price: float,
      orderbook: Optional[OrderBookSnapshot],
      metrics: Optional[OrderBookMetrics]
  ) -> Optional[IntegratedSignal]:
    """
    Анализ в hybrid режиме - комбинация single-TF и MTF.

    Логика:
    1. Запустить оба анализа параллельно
    2. Сравнить результаты
    3. Объединить с учетом приоритета и согласия

    Returns:
        IntegratedSignal или None
    """
    # Запускаем оба анализа параллельно
    single_tf_task = self._analyze_single_tf_mode(
      symbol, candles, current_price, orderbook, metrics
    )

    mtf_task = self._analyze_mtf_mode(
      symbol, orderbook, metrics
    )

    single_tf_signal, mtf_signal = await asyncio.gather(
      single_tf_task,
      mtf_task,
      return_exceptions=True
    )

    # Обработка ошибок
    if isinstance(single_tf_signal, Exception):
      logger.error(f"Single-TF analysis error: {single_tf_signal}")
      single_tf_signal = None

    if isinstance(mtf_signal, Exception):
      logger.error(f"MTF analysis error: {mtf_signal}")
      mtf_signal = None

    # Если оба None - нет сигнала
    if not single_tf_signal and not mtf_signal:
      return None

    # Если только один доступен - используем его
    if single_tf_signal and not mtf_signal:
      logger.debug("Используем single-TF (MTF недоступен)")
      return single_tf_signal

    if mtf_signal and not single_tf_signal:
      logger.debug("Используем MTF (single-TF недоступен)")
      return mtf_signal

    # Оба сигнала доступны - объединяем
    return self._combine_signals(
      symbol,
      single_tf_signal,
      mtf_signal
    )

  def _combine_signals(
      self,
      symbol: str,
      single_tf_signal: IntegratedSignal,
      mtf_signal: IntegratedSignal
  ) -> Optional[IntegratedSignal]:
    """
    Объединить сигналы от single-TF и MTF.

    Args:
        symbol: Торговая пара
        single_tf_signal: Single-TF сигнал
        mtf_signal: MTF сигнал

    Returns:
        Объединенный IntegratedSignal
    """
    # Проверка согласия
    signals_agree = (
        single_tf_signal.final_signal.signal_type ==
        mtf_signal.final_signal.signal_type
    )

    if not signals_agree:
      self.conflicts_detected += 1

      if self.config.hybrid_min_agreement:
        logger.warning(
          f"{symbol}: Конфликт сигналов - "
          f"single-TF={single_tf_signal.final_signal.signal_type.value}, "
          f"MTF={mtf_signal.final_signal.signal_type.value}"
        )

        # Разрешение конфликта
        return self._resolve_conflict(
          symbol,
          single_tf_signal,
          mtf_signal
        )

    # Сигналы согласны - объединяем
    mtf_weight = self.config.hybrid_mtf_priority
    single_tf_weight = 1.0 - mtf_weight

    # Weighted confidence
    combined_confidence = (
        mtf_signal.combined_confidence * mtf_weight +
        single_tf_signal.combined_confidence * single_tf_weight
    )

    # Weighted quality
    combined_quality = (
        mtf_signal.combined_quality_score * mtf_weight +
        single_tf_signal.combined_quality_score * single_tf_weight
    )

    # Position multiplier - берем от MTF (более консервативный)
    position_multiplier = mtf_signal.recommended_position_multiplier

    # Risk level - берем худший из двух
    risk_levels_priority = {"LOW": 1, "NORMAL": 2, "HIGH": 3, "EXTREME": 4}
    risk_level = max(
      [single_tf_signal.risk_level, mtf_signal.risk_level],
      key=lambda r: risk_levels_priority.get(r, 2)
    )

    # Создаем объединенный сигнал
    # Используем MTF сигнал как базу (приоритет)
    final_signal = mtf_signal.final_signal
    final_signal.confidence = combined_confidence

    integrated = IntegratedSignal(
      final_signal=final_signal,
      source_analysis_mode=AnalysisMode.HYBRID,
      used_single_tf=True,
      used_mtf=True,
      single_tf_consensus=single_tf_signal.single_tf_consensus,
      adaptive_weights=single_tf_signal.adaptive_weights,
      mtf_signal=mtf_signal.mtf_signal,
      mtf_alignment_score=mtf_signal.mtf_alignment_score,
      mtf_quality=mtf_signal.mtf_quality,
      combined_confidence=combined_confidence,
      combined_quality_score=combined_quality,
      recommended_position_multiplier=position_multiplier,
      recommended_stop_loss=mtf_signal.recommended_stop_loss,
      recommended_take_profit=mtf_signal.recommended_take_profit,
      risk_level=risk_level,
      warnings=(
          single_tf_signal.warnings +
          mtf_signal.warnings +
          ["Hybrid mode: combined single-TF + MTF"]
      )
    )

    logger.info(
      f"{symbol}: Объединены сигналы - "
      f"confidence={combined_confidence:.2%}, "
      f"quality={combined_quality:.2%}"
    )

    return integrated

  def _resolve_conflict(
      self,
      symbol: str,
      single_tf_signal: IntegratedSignal,
      mtf_signal: IntegratedSignal
  ) -> Optional[IntegratedSignal]:
    """
    Разрешить конфликт между single-TF и MTF сигналами.

    Args:
        symbol: Торговая пара
        single_tf_signal: Single-TF сигнал
        mtf_signal: MTF сигнал

    Returns:
        Выбранный сигнал или None
    """
    self.conflicts_resolved += 1

    resolution = self.config.hybrid_conflict_resolution

    if resolution == "mtf":
      logger.info(f"{symbol}: Конфликт разрешен → MTF (по приоритету)")
      return mtf_signal

    elif resolution == "single_tf":
      logger.info(f"{symbol}: Конфликт разрешен → Single-TF (по приоритету)")
      return single_tf_signal

    elif resolution == "highest_quality":
      # Выбираем по quality score
      if mtf_signal.combined_quality_score > single_tf_signal.combined_quality_score:
        logger.info(
          f"{symbol}: Конфликт разрешен → MTF "
          f"(quality={mtf_signal.combined_quality_score:.2f})"
        )
        return mtf_signal
      else:
        logger.info(
          f"{symbol}: Конфликт разрешен → Single-TF "
          f"(quality={single_tf_signal.combined_quality_score:.2f})"
        )
        return single_tf_signal

    else:
      logger.error(f"Unknown conflict resolution: {resolution}")
      return None

  def _log_integrated_signal(
      self,
      symbol: str,
      signal: IntegratedSignal
  ):
    """Логирование интегрированного сигнала."""
    logger.info(
      f"🎯 INTEGRATED SIGNAL [{symbol}]: "
      f"{signal.final_signal.signal_type.value}, "
      f"mode={signal.source_analysis_mode.value}, "
      f"confidence={signal.combined_confidence:.2%}, "
      f"quality={signal.combined_quality_score:.2%}, "
      f"position_mult={signal.recommended_position_multiplier:.2f}x, "
      f"risk={signal.risk_level}"
    )

    if self.config.log_analysis_details:
      # Детали single-TF
      if signal.used_single_tf and signal.single_tf_consensus:
        logger.debug(
          f"  Single-TF: {signal.single_tf_consensus.agreement_count}/"
          f"{signal.single_tf_consensus.agreement_count + signal.single_tf_consensus.disagreement_count} "
          f"strategies agree"
        )

      # Детали MTF
      if signal.used_mtf and signal.mtf_signal:
        logger.debug(
          f"  MTF: {signal.mtf_signal.timeframes_agreeing}/"
          f"{signal.mtf_signal.timeframes_analyzed} TF agree, "
          f"alignment={signal.mtf_alignment_score:.2%}"
        )

      # Warnings
      if signal.warnings:
        logger.warning(f"  Warnings: {', '.join(signal.warnings)}")

  def get_statistics(self) -> Dict:
    """Получить статистику integrated engine."""
    stats = {
      'engine': {
        'total_analyses': self.total_analyses,
        'signals_generated': self.signals_generated,
        'signal_rate': (
          self.signals_generated / self.total_analyses
          if self.total_analyses > 0 else 0.0
        ),
        'signals_by_mode': {
          mode.value: count
          for mode, count in self.signals_by_mode.items()
        },
        'high_quality_signals': self.high_quality_signals,
        'high_quality_rate': (
          self.high_quality_signals / self.signals_generated
          if self.signals_generated > 0 else 0.0
        ),
        'conflicts_detected': self.conflicts_detected,
        'conflicts_resolved': self.conflicts_resolved
      }
    }

    # Добавляем статистику компонентов
    if self.adaptive_consensus:
      stats['adaptive_consensus'] = self.adaptive_consensus.get_statistics()

    if self.mtf_manager:
      stats['mtf'] = self.mtf_manager.get_statistics()

    stats['strategy_manager'] = self.strategy_manager.get_statistics()

    return stats

  def get_health_status(self) -> Dict:
    """Проверить health status всей системы."""
    status = {
      'healthy': True,
      'components': {},
      'issues': []
    }

    # Strategy Manager
    status['components']['strategy_manager'] = True

    # Adaptive Consensus
    if self.adaptive_consensus:
      ac_health = self.adaptive_consensus.get_health_status()
      status['components']['adaptive_consensus'] = ac_health.get('healthy', True)
      if not ac_health.get('healthy'):
        status['healthy'] = False
        status['issues'].extend(ac_health.get('issues', []))

    # MTF Manager
    if self.mtf_manager:
      mtf_health = self.mtf_manager.get_health_status()
      status['components']['mtf'] = mtf_health.get('healthy', True)
      if not mtf_health.get('healthy'):
        status['healthy'] = False
        status['issues'].extend(mtf_health.get('issues', []))

    return status