"""
Multi-Timeframe Manager - главный оркестратор MTF анализа.

Функциональность:
- Объединение всех MTF компонентов
- End-to-end MTF analysis pipeline
- Координация обновлений данных
- Генерация финальных MTF сигналов
- Мониторинг и статистика

Pipeline:
    1. TimeframeCoordinator: обновление свечей всех TF
    2. TimeframeAnalyzer: анализ каждого TF независимо
    3. TimeframeAligner: проверка согласованности
    4. TimeframeSignalSynthesizer: генерация финального сигнала

Путь: backend/strategies/mtf/multi_timeframe_manager.py
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from core.logger import get_logger
from models.orderbook import OrderBookSnapshot, OrderBookMetrics
from strategies.strategy_manager import ExtendedStrategyManager

from .timeframe_coordinator import (
  TimeframeCoordinator,
  Timeframe,
  MultiTimeframeConfig
)
from .timeframe_analyzer import (
  TimeframeAnalyzer,
  TimeframeAnalysisResult
)
from .timeframe_aligner import (
  TimeframeAligner,
  TimeframeAlignment,
  AlignmentConfig
)
from .timeframe_signal_synthesizer import (
  TimeframeSignalSynthesizer,
  MultiTimeframeSignal,
  SynthesizerConfig,
  SynthesisMode
)

logger = get_logger(__name__)


@dataclass
class MTFManagerConfig:
  """Конфигурация Multi-Timeframe Manager."""
  # Включить MTF анализ
  enabled: bool = True

  # Конфигурации компонентов
  coordinator_config: MultiTimeframeConfig = field(default_factory=MultiTimeframeConfig)
  aligner_config: AlignmentConfig = field(default_factory=AlignmentConfig)
  synthesizer_config: SynthesizerConfig = field(default_factory=SynthesizerConfig)

  # Обновление данных
  auto_update_enabled: bool = True
  update_on_each_analysis: bool = False  # Обновлять ли при каждом вызове

  # Fallback behavior
  fallback_to_single_tf: bool = True  # Использовать single TF если MTF недоступен
  fallback_timeframe: Timeframe = Timeframe.M1

  # Logging
  verbose_logging: bool = False


class MultiTimeframeManager:
  """
  Главный менеджер Multi-Timeframe анализа.

  Координирует работу всех MTF компонентов и предоставляет
  единый интерфейс для получения MTF сигналов.
  """

  def __init__(
      self,
      strategy_manager: ExtendedStrategyManager,
      config: MTFManagerConfig
  ):
    """
    Инициализация MTF Manager.

    Args:
        strategy_manager: Менеджер стратегий для запуска на каждом TF
        config: Конфигурация MTF Manager
    """
    self.config = config
    self.strategy_manager = strategy_manager

    # Компоненты MTF
    self.coordinator = TimeframeCoordinator(config.coordinator_config)

    self.analyzer = TimeframeAnalyzer(
      strategy_manager=strategy_manager
    )

    self.aligner = TimeframeAligner(config.aligner_config)

    self.synthesizer = TimeframeSignalSynthesizer(config.synthesizer_config)

    # Кэш последних результатов
    self._last_tf_results: Dict[str, Dict[Timeframe, TimeframeAnalysisResult]] = {}
    self._last_alignment: Dict[str, TimeframeAlignment] = {}
    self._last_mtf_signal: Dict[str, MultiTimeframeSignal] = {}

    # Статистика
    self.total_analyses = 0
    self.mtf_signals_generated = 0
    self.fallback_used = 0

    # Инициализация завершена
    self._initialized_symbols = set()

    logger.info(
      f"Инициализирован MultiTimeframeManager: "
      f"enabled={config.enabled}, "
      f"timeframes={[tf.value for tf in config.coordinator_config.active_timeframes]}, "
      f"synthesis_mode={config.synthesizer_config.mode.value}"
    )

  async def initialize_symbol(self, symbol: str) -> bool:
    """
    Инициализировать символ для MTF анализа.

    Args:
        symbol: Торговая пара

    Returns:
        True если успешно
    """
    if not self.config.enabled:
      logger.warning("MTF анализ отключен в конфигурации")
      return False

    try:
      # Инициализация координатора (загрузка исторических данных)
      success = await self.coordinator.initialize_symbol(symbol)

      if success:
        self._initialized_symbols.add(symbol)
        logger.info(f"✅ Символ {symbol} инициализирован для MTF анализа")
      else:
        logger.error(f"❌ Ошибка инициализации {symbol} для MTF")

      return success

    except Exception as e:
      logger.error(f"Ошибка инициализации MTF для {symbol}: {e}", exc_info=True)
      return False

  async def analyze_symbol(
      self,
      symbol: str,
      orderbook: Optional[OrderBookSnapshot] = None,
      metrics: Optional[OrderBookMetrics] = None
  ) -> Optional[MultiTimeframeSignal]:
    """
    Полный MTF анализ символа.

    Pipeline:
    1. Обновить свечи всех TF (опционально)
    2. Анализировать каждый TF независимо
    3. Проверить alignment между TF
    4. Синтезировать финальный сигнал

    Args:
        symbol: Торговая пара
        orderbook: Снимок стакана (для гибридных стратегий)
        metrics: Метрики стакана

    Returns:
        MultiTimeframeSignal или None
    """
    if not self.config.enabled:
      logger.debug("MTF анализ отключен")
      return None

    # Проверка инициализации
    if symbol not in self._initialized_symbols:
      logger.warning(f"{symbol} не инициализирован для MTF, инициализация...")
      success = await self.initialize_symbol(symbol)
      if not success:
        return self._fallback_analysis(symbol, orderbook, metrics)

    self.total_analyses += 1

    try:
      # Шаг 1: Обновление данных (опционально)
      if self.config.update_on_each_analysis:
        await self.coordinator.update_all_timeframes(symbol)

      # Шаг 2: Получить текущие свечи всех TF
      all_candles = self.coordinator.get_all_timeframes_candles(symbol)

      if not all_candles:
        logger.warning(f"Нет данных свечей для {symbol}")
        return self._fallback_analysis(symbol, orderbook, metrics)

      # Получить текущую цену (из самого свежего TF)
      current_price = None
      for tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1]:
        if tf in all_candles and all_candles[tf]:
          current_price = all_candles[tf][-1].close
          break

      if not current_price:
        logger.error(f"Не удалось получить текущую цену для {symbol}")
        return self._fallback_analysis(symbol, orderbook, metrics)

      # Шаг 3: Анализировать каждый TF
      tf_results: Dict[Timeframe, TimeframeAnalysisResult] = {}

      for timeframe, candles in all_candles.items():
        if not candles:
          logger.warning(f"Нет свечей для {symbol} {timeframe.value}")
          continue

        result = await self.analyzer.analyze_timeframe(
          symbol=symbol,
          timeframe=timeframe,
          candles=candles,
          current_price=current_price,
          orderbook=orderbook,
          metrics=metrics
        )

        tf_results[timeframe] = result

        if self.config.verbose_logging:
          logger.debug(
            f"[{timeframe.value}] {symbol}: "
            f"regime={result.regime.market_regime.value}, "
            f"signal={result.timeframe_signal.signal_type.value if result.timeframe_signal else 'NONE'}"
          )

      if not tf_results:
        logger.error(f"Не удалось проанализировать ни один TF для {symbol}")
        return self._fallback_analysis(symbol, orderbook, metrics)

      # Кэшируем результаты
      self._last_tf_results[symbol] = tf_results

      # Шаг 4: Проверка alignment
      alignment = self.aligner.check_alignment(tf_results, current_price)
      self._last_alignment[symbol] = alignment

      if self.config.verbose_logging:
        logger.debug(
          f"Alignment check: type={alignment.alignment_type.value}, "
          f"score={alignment.alignment_score:.2f}, "
          f"recommended={alignment.recommended_action.value}"
        )

      # Шаг 5: Синтез финального сигнала
      mtf_signal = self.synthesizer.synthesize_signal(
        tf_results=tf_results,
        alignment=alignment,
        symbol=symbol,
        current_price=current_price
      )

      if mtf_signal:
        self.mtf_signals_generated += 1
        self._last_mtf_signal[symbol] = mtf_signal

        logger.info(
          f"🎯 MTF SIGNAL [{symbol}]: {mtf_signal.signal.signal_type.value}, "
          f"confidence={mtf_signal.signal.confidence:.2f}, "
          f"quality={mtf_signal.signal_quality:.2f}, "
          f"mode={mtf_signal.synthesis_mode.value}, "
          f"alignment={mtf_signal.alignment_score:.2f}"
        )
      else:
        logger.debug(f"Нет MTF сигнала для {symbol}")

      return mtf_signal

    except Exception as e:
      logger.error(f"Ошибка MTF анализа {symbol}: {e}", exc_info=True)
      return self._fallback_analysis(symbol, orderbook, metrics)

  async def update_timeframes(self, symbol: str) -> bool:
    """
    Обновить данные всех таймфреймов для символа.

    Args:
        symbol: Торговая пара

    Returns:
        True если успешно
    """
    try:
      results = await self.coordinator.update_all_timeframes(symbol)

      success_count = sum(1 for success in results.values() if success)
      total_count = len(results)

      logger.debug(
        f"Обновлено {success_count}/{total_count} таймфреймов для {symbol}"
      )

      return success_count > 0

    except Exception as e:
      logger.error(f"Ошибка обновления таймфреймов {symbol}: {e}")
      return False

  def get_last_tf_results(self, symbol: str) -> Optional[Dict[Timeframe, TimeframeAnalysisResult]]:
    """Получить последние результаты анализа таймфреймов."""
    return self._last_tf_results.get(symbol)

  def get_last_alignment(self, symbol: str) -> Optional[TimeframeAlignment]:
    """Получить последний alignment check."""
    return self._last_alignment.get(symbol)

  def get_last_mtf_signal(self, symbol: str) -> Optional[MultiTimeframeSignal]:
    """Получить последний MTF сигнал."""
    return self._last_mtf_signal.get(symbol)

  def validate_data_consistency(self, symbol: str) -> Dict:
    """
    Валидировать консистентность данных между таймфреймами.

    Args:
        symbol: Торговая пара

    Returns:
        Dict с результатами валидации
    """
    return self.coordinator.validate_data_consistency(symbol)

  def _fallback_analysis(
      self,
      symbol: str,
      orderbook: Optional[OrderBookSnapshot],
      metrics: Optional[OrderBookMetrics]
  ) -> Optional[MultiTimeframeSignal]:
    """
    Fallback к single-timeframe анализу при проблемах с MTF.

    Args:
        symbol: Торговая пара
        orderbook: Снимок стакана
        metrics: Метрики

    Returns:
        MultiTimeframeSignal (эмулированный) или None
    """
    if not self.config.fallback_to_single_tf:
      logger.debug("Fallback к single TF отключен")
      return None

    self.fallback_used += 1

    logger.warning(
      f"Использован fallback к single TF для {symbol}, "
      f"timeframe={self.config.fallback_timeframe.value}"
    )

    try:
      # Получаем свечи fallback TF
      candles = self.coordinator.get_candles(
        symbol,
        self.config.fallback_timeframe
      )

      if not candles:
        logger.error(f"Нет свечей для fallback TF {self.config.fallback_timeframe.value}")
        return None

      current_price = candles[-1].close

      # Запускаем стратегии на single TF
      consensus = self.strategy_manager.analyze_with_consensus(
        symbol=symbol,
        candles=candles,
        current_price=current_price,
        orderbook=orderbook,
        metrics=metrics
      )

      if not consensus or not consensus.final_signal:
        return None

      # Эмулируем MTF сигнал
      from .timeframe_aligner import AlignmentType, DivergenceType

      mtf_signal = MultiTimeframeSignal(
        signal=consensus.final_signal,
        synthesis_mode=SynthesisMode.TOP_DOWN,  # Fallback mode
        timeframes_analyzed=1,
        timeframes_agreeing=1,
        alignment_score=consensus.consensus_confidence,
        alignment_type=AlignmentType.NEUTRAL,
        signal_quality=consensus.consensus_confidence,
        reliability_score=0.5,
        recommended_position_size_multiplier=0.8,  # Пониженный для fallback
        warnings=[f"Fallback to single TF: {self.config.fallback_timeframe.value}"],
        risk_level="HIGH",
        timestamp=int(datetime.now().timestamp() * 1000)
      )

      return mtf_signal

    except Exception as e:
      logger.error(f"Ошибка fallback анализа: {e}", exc_info=True)
      return None

  def get_statistics(self) -> Dict:
    """Получить статистику MTF Manager."""
    coordinator_stats = self.coordinator.get_statistics()
    analyzer_stats = self.analyzer.get_statistics()
    aligner_stats = self.aligner.get_statistics()
    synthesizer_stats = self.synthesizer.get_statistics()

    return {
      'manager': {
        'enabled': self.config.enabled,
        'initialized_symbols': len(self._initialized_symbols),
        'total_analyses': self.total_analyses,
        'mtf_signals_generated': self.mtf_signals_generated,
        'signal_generation_rate': (
          self.mtf_signals_generated / self.total_analyses
          if self.total_analyses > 0 else 0.0
        ),
        'fallback_used': self.fallback_used,
        'fallback_rate': (
          self.fallback_used / self.total_analyses
          if self.total_analyses > 0 else 0.0
        )
      },
      'coordinator': coordinator_stats,
      'analyzer': analyzer_stats,
      'aligner': aligner_stats,
      'synthesizer': synthesizer_stats
    }

  def get_health_status(self) -> Dict:
    """Получить health status MTF системы."""
    # Проверяем компоненты
    components_healthy = {
      'coordinator': self.coordinator is not None,
      'analyzer': self.analyzer is not None,
      'aligner': self.aligner is not None,
      'synthesizer': self.synthesizer is not None
    }

    all_healthy = all(components_healthy.values())

    # Проверяем инициализированные символы
    initialized_count = len(self._initialized_symbols)

    # Проверяем валидацию данных для инициализированных символов
    data_issues = []
    for symbol in list(self._initialized_symbols)[:3]:  # Проверяем первые 3
      validation = self.coordinator.validate_data_consistency(symbol)
      if not validation.get('valid', True):
        data_issues.extend(validation.get('issues', []))

    status = {
      'healthy': all_healthy and len(data_issues) == 0,
      'components': components_healthy,
      'initialized_symbols': initialized_count,
      'data_issues': data_issues[:5],  # Первые 5 проблем
      'enabled': self.config.enabled
    }

    return status

  def reset_symbol_cache(self, symbol: str):
    """Очистить кэш для символа."""
    if symbol in self._last_tf_results:
      del self._last_tf_results[symbol]

    if symbol in self._last_alignment:
      del self._last_alignment[symbol]

    if symbol in self._last_mtf_signal:
      del self._last_mtf_signal[symbol]

    logger.debug(f"Кэш для {symbol} очищен")

  def reset_all_cache(self):
    """Очистить весь кэш."""
    self._last_tf_results.clear()
    self._last_alignment.clear()
    self._last_mtf_signal.clear()

    logger.info("Весь кэш MTF очищен")