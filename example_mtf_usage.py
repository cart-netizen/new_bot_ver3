"""
Пример использования Multi-Timeframe Analysis System.

Демонстрирует:
1. Инициализацию MTF Manager
2. Настройку различных режимов synthesis
3. Анализ символов с MTF подходом
4. Работу с результатами и risk management
5. Мониторинг и статистику

Путь: examples/example_mtf_usage.py
"""

import asyncio
from datetime import datetime

from core.logger import get_logger
from strategies.strategy_manager import ExtendedStrategyManager, ExtendedStrategyManagerConfig
from strategies.mtf import (
  MultiTimeframeManager,
  MTFManagerConfig,
  MultiTimeframeConfig,
  AlignmentConfig,
  SynthesizerConfig,
  SynthesisMode,
  Timeframe
)
from models.signal import SignalType

logger = get_logger(__name__)


async def example_basic_mtf_analysis():
  """
  Пример 1: Базовый MTF анализ с Top-Down режимом.
  """
  logger.info("=" * 60)
  logger.info("ПРИМЕР 1: Базовый MTF анализ (Top-Down)")
  logger.info("=" * 60)

  # Шаг 1: Создаем StrategyManager
  strategy_config = ExtendedStrategyManagerConfig(
    enable_orderbook_strategies=True,
    enable_hybrid_strategies=True
  )
  strategy_manager = ExtendedStrategyManager(strategy_config)

  # Шаг 2: Настраиваем MTF конфигурацию
  mtf_config = MTFManagerConfig(
    enabled=True,
    coordinator_config=MultiTimeframeConfig(
      active_timeframes=[
        Timeframe.M1,
        Timeframe.M5,
        Timeframe.M15,
        Timeframe.H1
      ],
      primary_timeframe=Timeframe.H1,
      execution_timeframe=Timeframe.M1
    ),
    synthesizer_config=SynthesizerConfig(
      mode=SynthesisMode.TOP_DOWN,
      primary_timeframe=Timeframe.H1,
      execution_timeframe=Timeframe.M1,
      require_htf_confirmation=True
    ),
    verbose_logging=True
  )

  # Шаг 3: Создаем MTF Manager
  mtf_manager = MultiTimeframeManager(strategy_manager, mtf_config)

  # Шаг 4: Инициализируем символ
  symbol = "BTCUSDT"
  logger.info(f"Инициализация {symbol}...")

  success = await mtf_manager.initialize_symbol(symbol)

  if not success:
    logger.error(f"Не удалось инициализировать {symbol}")
    return

  logger.info(f"✅ {symbol} инициализирован")

  # Шаг 5: Выполняем MTF анализ
  logger.info(f"Запуск MTF анализа для {symbol}...")

  mtf_signal = await mtf_manager.analyze_symbol(symbol)

  if mtf_signal:
    logger.info("=" * 60)
    logger.info("✅ MTF СИГНАЛ ПОЛУЧЕН")
    logger.info("=" * 60)
    logger.info(f"Направление: {mtf_signal.signal.signal_type.value}")
    logger.info(f"Цена: {mtf_signal.signal.price:.2f}")
    logger.info(f"Confidence: {mtf_signal.signal.confidence:.2%}")
    logger.info(f"Качество: {mtf_signal.signal_quality:.2%}")
    logger.info(f"Режим синтеза: {mtf_signal.synthesis_mode.value}")
    logger.info(f"Alignment score: {mtf_signal.alignment_score:.2%}")
    logger.info(f"Alignment type: {mtf_signal.alignment_type.value}")
    logger.info(f"Таймфреймов проанализировано: {mtf_signal.timeframes_analyzed}")
    logger.info(f"Таймфреймов согласны: {mtf_signal.timeframes_agreeing}")

    # Risk Management параметры
    logger.info("=" * 60)
    logger.info("RISK MANAGEMENT")
    logger.info("=" * 60)
    logger.info(f"Position size multiplier: {mtf_signal.recommended_position_size_multiplier:.2f}x")
    logger.info(f"Stop-loss: {mtf_signal.recommended_stop_loss_price}")
    logger.info(f"Take-profit: {mtf_signal.recommended_take_profit_price}")
    logger.info(f"Risk level: {mtf_signal.risk_level}")

    if mtf_signal.warnings:
      logger.warning(f"Warnings: {', '.join(mtf_signal.warnings)}")
  else:
    logger.info("❌ MTF сигнал не сгенерирован (нет консенсуса)")

  # Шаг 6: Статистика
  stats = mtf_manager.get_statistics()
  logger.info("=" * 60)
  logger.info("СТАТИСТИКА")
  logger.info("=" * 60)
  logger.info(f"Всего анализов: {stats['manager']['total_analyses']}")
  logger.info(f"Сигналов сгенерировано: {stats['manager']['mtf_signals_generated']}")
  logger.info(f"Signal rate: {stats['manager']['signal_generation_rate']:.2%}")


async def example_consensus_mode():
  """
  Пример 2: MTF анализ с Consensus режимом.
  """
  logger.info("=" * 60)
  logger.info("ПРИМЕР 2: MTF анализ (Consensus)")
  logger.info("=" * 60)

  strategy_manager = ExtendedStrategyManager(ExtendedStrategyManagerConfig())

  # Consensus mode - взвешенный консенсус всех TF
  mtf_config = MTFManagerConfig(
    enabled=True,
    synthesizer_config=SynthesizerConfig(
      mode=SynthesisMode.CONSENSUS,
      consensus_threshold=0.70,  # Требуется 70% weighted agreement
      timeframe_weights={
        Timeframe.M1: 0.10,
        Timeframe.M5: 0.20,
        Timeframe.M15: 0.30,
        Timeframe.H1: 0.40
      }
    )
  )

  mtf_manager = MultiTimeframeManager(strategy_manager, mtf_config)

  symbol = "ETHUSDT"
  await mtf_manager.initialize_symbol(symbol)

  mtf_signal = await mtf_manager.analyze_symbol(symbol)

  if mtf_signal:
    logger.info(f"Consensus signal: {mtf_signal.signal.signal_type.value}")
    logger.info(f"Weighted confidence: {mtf_signal.signal.confidence:.2%}")

    # Детали по каждому TF
    logger.info("\nСигналы по таймфреймам:")
    for tf, signal in mtf_signal.timeframe_signals.items():
      if signal:
        logger.info(f"  {tf.value}: {signal.signal_type.value} (conf={signal.confidence:.2f})")
  else:
    logger.info("Нет consensus между таймфреймами")


async def example_confluence_mode():
  """
  Пример 3: MTF анализ с Confluence режимом (строгий).
  """
  logger.info("=" * 60)
  logger.info("ПРИМЕР 3: MTF анализ (Confluence - строгий режим)")
  logger.info("=" * 60)

  strategy_manager = ExtendedStrategyManager(ExtendedStrategyManagerConfig())

  # Confluence mode - требуется согласие ВСЕХ TF
  mtf_config = MTFManagerConfig(
    enabled=True,
    synthesizer_config=SynthesizerConfig(
      mode=SynthesisMode.CONFLUENCE,
      require_all_timeframes=True,
      allow_neutral_timeframes=False  # Даже HOLD не разрешен
    )
  )

  mtf_manager = MultiTimeframeManager(strategy_manager, mtf_config)

  symbol = "SOLUSDT"
  await mtf_manager.initialize_symbol(symbol)

  mtf_signal = await mtf_manager.analyze_symbol(symbol)

  if mtf_signal:
    logger.info("🎯 PERFECT CONFLUENCE - все TF согласны!")
    logger.info(f"Signal: {mtf_signal.signal.signal_type.value}")
    logger.info(f"Quality: {mtf_signal.signal_quality:.2%}")
    logger.info("Это высоконадежный сигнал с максимальной уверенностью")
  else:
    logger.info("Нет полного confluence - требуется согласие всех TF")


async def example_alignment_analysis():
  """
  Пример 4: Детальный анализ alignment между таймфреймами.
  """
  logger.info("=" * 60)
  logger.info("ПРИМЕР 4: Детальный анализ alignment")
  logger.info("=" * 60)

  strategy_manager = ExtendedStrategyManager(ExtendedStrategyManagerConfig())

  mtf_config = MTFManagerConfig(
    enabled=True,
    aligner_config=AlignmentConfig(
      min_alignment_score=0.60,
      allow_trend_counter_signals=False,
      confluence_price_tolerance_percent=0.5
    )
  )

  mtf_manager = MultiTimeframeManager(strategy_manager, mtf_config)

  symbol = "BTCUSDT"
  await mtf_manager.initialize_symbol(symbol)
  await mtf_manager.analyze_symbol(symbol)

  # Получаем детали alignment
  alignment = mtf_manager.get_last_alignment(symbol)

  if alignment:
    logger.info("=" * 60)
    logger.info("ALIGNMENT DETAILS")
    logger.info("=" * 60)
    logger.info(f"Alignment type: {alignment.alignment_type.value}")
    logger.info(f"Alignment score: {alignment.alignment_score:.2%}")
    logger.info(
      f"HTF trend: {'BULL' if alignment.higher_timeframe_trend > 0 else 'BEAR' if alignment.higher_timeframe_trend < 0 else 'RANGE'}")

    logger.info(f"\nBullish TF: {[tf.value for tf in alignment.bullish_timeframes]}")
    logger.info(f"Bearish TF: {[tf.value for tf in alignment.bearish_timeframes]}")
    logger.info(f"Neutral TF: {[tf.value for tf in alignment.neutral_timeframes]}")

    # Confluence zones
    if alignment.confluence_zones:
      logger.info("\nCONFLUENCE ZONES:")
      for zone in alignment.confluence_zones[:3]:  # Топ 3
        logger.info(f"  Level: {zone.price_level:.2f}")
        logger.info(f"  Type: {zone.confluence_type}")
        logger.info(f"  Confirming TF: {[tf.value for tf in zone.timeframes_confirming]}")
        logger.info(f"  Strength: {zone.strength:.2%}")
        logger.info("")

    # Divergences
    if alignment.divergence_type.value != "no_divergence":
      logger.warning(f"⚠️ DIVERGENCE DETECTED")
      logger.warning(f"Type: {alignment.divergence_type.value}")
      logger.warning(f"Severity: {alignment.divergence_severity:.2%}")
      logger.warning(f"Details: {alignment.divergence_details}")

    # Recommendations
    logger.info("\nRECOMMENDATIONS:")
    logger.info(f"Action: {alignment.recommended_action.value}")
    logger.info(f"Confidence: {alignment.recommended_confidence:.2%}")
    logger.info(f"Position multiplier: {alignment.position_size_multiplier:.2f}x")


async def example_risk_management():
  """
  Пример 5: Использование MTF для риск-менеджмента.
  """
  logger.info("=" * 60)
  logger.info("ПРИМЕР 5: MTF-based Risk Management")
  logger.info("=" * 60)

  strategy_manager = ExtendedStrategyManager(ExtendedStrategyManagerConfig())

  mtf_config = MTFManagerConfig(
    enabled=True,
    synthesizer_config=SynthesizerConfig(
      mode=SynthesisMode.TOP_DOWN,
      enable_dynamic_position_sizing=True,
      base_position_size=1.0,
      max_position_multiplier=1.5,
      min_position_multiplier=0.3,
      use_higher_tf_for_stops=True,
      stop_loss_timeframe=Timeframe.M15,
      atr_multiplier_for_stops=2.0
    )
  )

  mtf_manager = MultiTimeframeManager(strategy_manager, mtf_config)

  symbol = "BTCUSDT"
  await mtf_manager.initialize_symbol(symbol)

  mtf_signal = await mtf_manager.analyze_symbol(symbol)

  if mtf_signal:
    # Расчет реальных параметров позиции
    base_position_usd = 1000.0  # Базовый размер позиции

    actual_position_size = base_position_usd * mtf_signal.recommended_position_size_multiplier

    logger.info("=" * 60)
    logger.info("ПАРАМЕТРЫ ПОЗИЦИИ")
    logger.info("=" * 60)
    logger.info(f"Базовый размер: ${base_position_usd:.2f}")
    logger.info(f"Multiplier: {mtf_signal.recommended_position_size_multiplier:.2f}x")
    logger.info(f"Фактический размер: ${actual_position_size:.2f}")
    logger.info(f"Entry price: {mtf_signal.signal.price:.2f}")
    logger.info(f"Stop-loss: {mtf_signal.recommended_stop_loss_price:.2f}")
    logger.info(f"Take-profit: {mtf_signal.recommended_take_profit_price:.2f}")

    # Расчет R:R
    if mtf_signal.recommended_stop_loss_price:
      stop_distance = abs(mtf_signal.signal.price - mtf_signal.recommended_stop_loss_price)
      tp_distance = abs(
        mtf_signal.recommended_take_profit_price - mtf_signal.signal.price) if mtf_signal.recommended_take_profit_price else 0

      risk_reward = tp_distance / stop_distance if stop_distance > 0 else 0

      logger.info(f"\nStop distance: {stop_distance:.2f} ({stop_distance / mtf_signal.signal.price * 100:.2f}%)")
      logger.info(f"TP distance: {tp_distance:.2f} ({tp_distance / mtf_signal.signal.price * 100:.2f}%)")
      logger.info(f"Risk:Reward ratio: 1:{risk_reward:.2f}")

    logger.info(f"\nRisk level: {mtf_signal.risk_level}")
    logger.info(f"Signal quality: {mtf_signal.signal_quality:.2%}")


async def example_monitoring_and_health():
  """
  Пример 6: Мониторинг и health check MTF системы.
  """
  logger.info("=" * 60)
  logger.info("ПРИМЕР 6: Мониторинг и Health Check")
  logger.info("=" * 60)

  strategy_manager = ExtendedStrategyManager(ExtendedStrategyManagerConfig())
  mtf_manager = MultiTimeframeManager(strategy_manager, MTFManagerConfig())

  # Инициализируем несколько символов
  symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

  for symbol in symbols:
    await mtf_manager.initialize_symbol(symbol)

  # Health check
  health = mtf_manager.get_health_status()

  logger.info("HEALTH STATUS:")
  logger.info(f"System healthy: {'✅' if health['healthy'] else '❌'}")
  logger.info(f"Components:")
  for component, status in health['components'].items():
    logger.info(f"  {component}: {'✅' if status else '❌'}")
  logger.info(f"Initialized symbols: {health['initialized_symbols']}")

  if health['data_issues']:
    logger.warning("Data issues detected:")
    for issue in health['data_issues']:
      logger.warning(f"  - {issue}")

  # Полная статистика
  stats = mtf_manager.get_statistics()

  logger.info("\n" + "=" * 60)
  logger.info("ПОЛНАЯ СТАТИСТИКА")
  logger.info("=" * 60)

  logger.info("\nManager:")
  for key, value in stats['manager'].items():
    logger.info(f"  {key}: {value}")

  logger.info("\nCoordinator:")
  for key, value in stats['coordinator'].items():
    logger.info(f"  {key}: {value}")

  logger.info("\nAnalyzer:")
  for key, value in stats['analyzer'].items():
    logger.info(f"  {key}: {value}")

  logger.info("\nAligner:")
  for key, value in stats['aligner'].items():
    logger.info(f"  {key}: {value}")

  logger.info("\nSynthesizer:")
  for key, value in stats['synthesizer'].items():
    logger.info(f"  {key}: {value}")


async def main():
  """Запуск всех примеров."""
  logger.info("🚀 Запуск примеров Multi-Timeframe Analysis")
  logger.info("")

  try:
    # Базовый анализ
    await example_basic_mtf_analysis()
    await asyncio.sleep(2)

    # Consensus режим
    await example_consensus_mode()
    await asyncio.sleep(2)

    # Confluence режим
    await example_confluence_mode()
    await asyncio.sleep(2)

    # Alignment анализ
    await example_alignment_analysis()
    await asyncio.sleep(2)

    # Risk management
    await example_risk_management()
    await asyncio.sleep(2)

    # Мониторинг
    await example_monitoring_and_health()

  except Exception as e:
    logger.error(f"Ошибка в примерах: {e}", exc_info=True)

  logger.info("")
  logger.info("✅ Все примеры завершены")


if __name__ == "__main__":
  asyncio.run(main())