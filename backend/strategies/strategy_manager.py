"""
Strategy Manager - объединение всех торговых стратегий.

Функциональность:
- Управление множественными стратегиями
- Объединение сигналов (consensus)
- Приоритизация стратегий
- Конфликт-резолюция
- Статистика по стратегиям

Путь: backend/strategies/strategy_manager.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from core.logger import get_logger
from models.signal import TradingSignal, SignalType, SignalStrength, SignalSource
from strategy.candle_manager import Candle

# Импорт всех стратегий
from strategies.momentum_strategy import MomentumStrategy, MomentumConfig
from strategies.sar_wave_strategy import SARWaveStrategy, SARWaveConfig
from strategies.supertrend_strategy import SuperTrendStrategy, SuperTrendConfig
from strategies.volume_profile_strategy import VolumeProfileStrategy, VolumeProfileConfig

logger = get_logger(__name__)


class StrategyPriority(Enum):
  """Приоритет стратегий."""
  HIGH = 3
  MEDIUM = 2
  LOW = 1


@dataclass
class StrategyManagerConfig:
  """Конфигурация Strategy Manager."""
  # Режим объединения сигналов
  consensus_mode: str = "weighted"  # "weighted", "majority", "unanimous"

  # Минимальные требования
  min_strategies_for_signal: int = 2  # Минимум стратегий должны согласиться
  min_consensus_confidence: float = 0.6  # Минимальная уверенность для consensus

  # Веса стратегий (для weighted режима)
  strategy_weights: Dict[str, float] = None

  # Приоритеты
  strategy_priorities: Dict[str, StrategyPriority] = None

  # Конфликт-резолюция
  conflict_resolution: str = "highest_confidence"  # "highest_confidence", "priority", "cancel"

  def __post_init__(self):
    if self.strategy_weights is None:
      self.strategy_weights = {
        'momentum': 0.25,
        'sar_wave': 0.25,
        'supertrend': 0.25,
        'volume_profile': 0.25
      }

    if self.strategy_priorities is None:
      self.strategy_priorities = {
        'momentum': StrategyPriority.HIGH,
        'sar_wave': StrategyPriority.MEDIUM,
        'supertrend': StrategyPriority.HIGH,
        'volume_profile': StrategyPriority.MEDIUM
      }


@dataclass
class StrategyResult:
  """Результат от одной стратегии."""
  strategy_name: str
  signal: Optional[TradingSignal]
  priority: StrategyPriority
  weight: float
  execution_time_ms: float


@dataclass
class ConsensusSignal:
  """Объединенный сигнал от нескольких стратегий."""
  final_signal: TradingSignal
  contributing_strategies: List[str]
  agreement_count: int
  disagreement_count: int
  consensus_confidence: float
  strategy_results: List[StrategyResult]


class StrategyManager:
  """
  Менеджер торговых стратегий.

  Управляет множественными стратегиями и объединяет их сигналы.
  """

  def __init__(self, config: StrategyManagerConfig):
    """
    Инициализация менеджера.

    Args:
        config: Конфигурация менеджера
    """
    self.config = config

    # Инициализация стратегий
    self.strategies: Dict[str, any] = {}

    # Momentum Strategy
    self.strategies['momentum'] = MomentumStrategy(MomentumConfig())

    # SAR Wave Strategy
    self.strategies['sar_wave'] = SARWaveStrategy(SARWaveConfig())

    # SuperTrend Strategy
    self.strategies['supertrend'] = SuperTrendStrategy(SuperTrendConfig())

    # Volume Profile Strategy
    self.strategies['volume_profile'] = VolumeProfileStrategy(VolumeProfileConfig())

    # Статистика
    self.total_analyses = 0
    self.signals_generated = 0
    self.consensus_achieved = 0
    self.conflicts_resolved = 0

    logger.info(
      f"Инициализирован StrategyManager: "
      f"strategies={list(self.strategies.keys())}, "
      f"consensus_mode={config.consensus_mode}"
    )

  def analyze_all_strategies(
      self,
      symbol: str,
      candles: List[Candle],
      current_price: float
  ) -> List[StrategyResult]:
    """
    Запустить все стратегии для анализа.

    Returns:
        Список результатов от каждой стратегии
    """
    import time

    results = []

    for strategy_name, strategy in self.strategies.items():
      start_time = time.time()

      try:
        signal = strategy.analyze(symbol, candles, current_price)
        execution_time = (time.time() - start_time) * 1000  # ms

        result = StrategyResult(
          strategy_name=strategy_name,
          signal=signal,
          priority=self.config.strategy_priorities.get(
            strategy_name,
            StrategyPriority.MEDIUM
          ),
          weight=self.config.strategy_weights.get(strategy_name, 0.25),
          execution_time_ms=execution_time
        )

        results.append(result)

        if signal:
          logger.debug(
            f"[{strategy_name}] Signal: {signal.signal_type.value}, "
            f"confidence={signal.confidence:.2f}"
          )

      except Exception as e:
        logger.error(f"Ошибка в стратегии {strategy_name}: {e}", exc_info=True)
        continue

    self.total_analyses += 1

    return results

  def build_consensus(
      self,
      symbol: str,
      strategy_results: List[StrategyResult],
      current_price: float
  ) -> Optional[ConsensusSignal]:
    """
    Построить consensus сигнал из результатов стратегий.

    Returns:
        ConsensusSignal или None
    """
    # Фильтруем результаты с сигналами
    results_with_signals = [r for r in strategy_results if r.signal is not None]

    if len(results_with_signals) < self.config.min_strategies_for_signal:
      logger.debug(
        f"Недостаточно стратегий для consensus: "
        f"{len(results_with_signals)} < {self.config.min_strategies_for_signal}"
      )
      return None

    # Группируем по типу сигнала
    buy_signals = [r for r in results_with_signals if r.signal.signal_type == SignalType.BUY]
    sell_signals = [r for r in results_with_signals if r.signal.signal_type == SignalType.SELL]

    # Определяем consensus направление
    if self.config.consensus_mode == "majority":
      consensus_signal = self._majority_consensus(
        buy_signals,
        sell_signals,
        strategy_results
      )

    elif self.config.consensus_mode == "weighted":
      consensus_signal = self._weighted_consensus(
        buy_signals,
        sell_signals,
        strategy_results
      )

    elif self.config.consensus_mode == "unanimous":
      consensus_signal = self._unanimous_consensus(
        buy_signals,
        sell_signals,
        strategy_results
      )

    else:
      logger.error(f"Неизвестный consensus_mode: {self.config.consensus_mode}")
      return None

    if consensus_signal:
      self.signals_generated += 1
      if consensus_signal.agreement_count == len(results_with_signals):
        self.consensus_achieved += 1
      else:
        self.conflicts_resolved += 1

    return consensus_signal

  def _majority_consensus(
      self,
      buy_signals: List[StrategyResult],
      sell_signals: List[StrategyResult],
      all_results: List[StrategyResult]
  ) -> Optional[ConsensusSignal]:
    """Majority voting consensus."""
    if len(buy_signals) > len(sell_signals):
      signal_type = SignalType.BUY
      contributing = buy_signals
      agreement = len(buy_signals)
      disagreement = len(sell_signals)
    elif len(sell_signals) > len(buy_signals):
      signal_type = SignalType.SELL
      contributing = sell_signals
      agreement = len(sell_signals)
      disagreement = len(buy_signals)
    else:
      # Равное количество - конфликт
      return self._resolve_conflict(buy_signals, sell_signals, all_results)

    # Средняя уверенность
    avg_confidence = float(np.mean([r.signal.confidence for r in contributing]))

    if avg_confidence < self.config.min_consensus_confidence:
      return None

    # Создаем consensus сигнал
    final_signal = TradingSignal(
      symbol=contributing[0].signal.symbol,
      signal_type=signal_type,
      source=SignalSource.STRATEGY,
      strength=self._determine_strength(avg_confidence),
      price=contributing[0].signal.price,
      confidence=avg_confidence,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason=f"Majority consensus: {agreement} strategies agree",
      metadata={
        'consensus_mode': 'majority',
        'contributing_strategies': [r.strategy_name for r in contributing],
        'agreement_count': agreement,
        'disagreement_count': disagreement
      }
    )

    return ConsensusSignal(
      final_signal=final_signal,
      contributing_strategies=[r.strategy_name for r in contributing],
      agreement_count=agreement,
      disagreement_count=disagreement,
      consensus_confidence=avg_confidence,
      strategy_results=all_results
    )

  def _weighted_consensus(
      self,
      buy_signals: List[StrategyResult],
      sell_signals: List[StrategyResult],
      all_results: List[StrategyResult]
  ) -> Optional[ConsensusSignal]:
    """Weighted voting consensus."""
    # Вычисляем взвешенные голоса
    buy_weight = sum(r.weight * r.signal.confidence for r in buy_signals)
    sell_weight = sum(r.weight * r.signal.confidence for r in sell_signals)

    if buy_weight > sell_weight:
      signal_type = SignalType.BUY
      contributing = buy_signals
      consensus_confidence = buy_weight / sum(r.weight for r in buy_signals)
    elif sell_weight > buy_weight:
      signal_type = SignalType.SELL
      contributing = sell_signals
      consensus_confidence = sell_weight / sum(r.weight for r in sell_signals)
    else:
      return self._resolve_conflict(buy_signals, sell_signals, all_results)

    if consensus_confidence < self.config.min_consensus_confidence:
      return None

    final_signal = TradingSignal(
      symbol=contributing[0].signal.symbol,
      signal_type=signal_type,
      source=SignalSource.STRATEGY,
      strength=self._determine_strength(consensus_confidence),
      price=contributing[0].signal.price,
      confidence=consensus_confidence,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason=f"Weighted consensus: {len(contributing)} strategies (weight={buy_weight if signal_type == SignalType.BUY else sell_weight:.2f})",
      metadata={
        'consensus_mode': 'weighted',
        'contributing_strategies': [r.strategy_name for r in contributing],
        'buy_weight': buy_weight,
        'sell_weight': sell_weight
      }
    )

    return ConsensusSignal(
      final_signal=final_signal,
      contributing_strategies=[r.strategy_name for r in contributing],
      agreement_count=len(contributing),
      disagreement_count=len(buy_signals if signal_type == SignalType.SELL else sell_signals),
      consensus_confidence=consensus_confidence,
      strategy_results=all_results
    )

  def _unanimous_consensus(
      self,
      buy_signals: List[StrategyResult],
      sell_signals: List[StrategyResult],
      all_results: List[StrategyResult]
  ) -> Optional[ConsensusSignal]:
    """Unanimous consensus - все стратегии должны согласиться."""
    results_with_signals = [r for r in all_results if r.signal is not None]

    # Все должны быть BUY или все SELL
    if len(buy_signals) == len(results_with_signals):
      signal_type = SignalType.BUY
      contributing = buy_signals
    elif len(sell_signals) == len(results_with_signals):
      signal_type = SignalType.SELL
      contributing = sell_signals
    else:
      # Нет единогласия
      return None

    avg_confidence = float(np.mean([r.signal.confidence for r in contributing]))

    if avg_confidence < self.config.min_consensus_confidence:
      return None

    final_signal = TradingSignal(
      symbol=contributing[0].signal.symbol,
      signal_type=signal_type,
      source=SignalSource.STRATEGY,
      strength=SignalStrength.STRONG,
      price=contributing[0].signal.price,
      confidence=avg_confidence,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason=f"Unanimous consensus: ALL {len(contributing)} strategies agree",
      metadata={
        'consensus_mode': 'unanimous',
        'contributing_strategies': [r.strategy_name for r in contributing]
      }
    )

    return ConsensusSignal(
      final_signal=final_signal,
      contributing_strategies=[r.strategy_name for r in contributing],
      agreement_count=len(contributing),
      disagreement_count=0,
      consensus_confidence=avg_confidence,
      strategy_results=all_results
    )

  def _resolve_conflict(
      self,
      buy_signals: List[StrategyResult],
      sell_signals: List[StrategyResult],
      all_results: List[StrategyResult]
  ) -> Optional[ConsensusSignal]:
    """Разрешить конфликт между стратегиями."""
    if self.config.conflict_resolution == "cancel":
      logger.info("Конфликт стратегий: отмена сигнала")
      return None

    elif self.config.conflict_resolution == "highest_confidence":
      # Выбираем сигнал с наибольшей уверенностью
      all_signals = buy_signals + sell_signals
      best = max(all_signals, key=lambda r: r.signal.confidence)

      logger.info(
        f"Конфликт разрешен: выбран {best.strategy_name} "
        f"с confidence={best.signal.confidence:.2f}"
      )

      return ConsensusSignal(
        final_signal=best.signal,
        contributing_strategies=[best.strategy_name],
        agreement_count=1,
        disagreement_count=len(all_signals) - 1,
        consensus_confidence=best.signal.confidence,
        strategy_results=all_results
      )

    elif self.config.conflict_resolution == "priority":
      # Выбираем по приоритету
      all_signals = buy_signals + sell_signals
      best = max(all_signals, key=lambda r: r.priority.value)

      logger.info(
        f"Конфликт разрешен: выбран {best.strategy_name} "
        f"с приоритетом={best.priority.name}"
      )

      return ConsensusSignal(
        final_signal=best.signal,
        contributing_strategies=[best.strategy_name],
        agreement_count=1,
        disagreement_count=len(all_signals) - 1,
        consensus_confidence=best.signal.confidence,
        strategy_results=all_results
      )

    return None

  def _determine_strength(self, confidence: float) -> SignalStrength:
    """Определить силу сигнала по confidence."""
    if confidence >= 0.8:
      return SignalStrength.STRONG
    elif confidence >= 0.6:
      return SignalStrength.MEDIUM
    else:
      return SignalStrength.WEAK

  def analyze_with_consensus(
      self,
      symbol: str,
      candles: List[Candle],
      current_price: float
  ) -> Optional[ConsensusSignal]:
    """
    Полный анализ: запустить все стратегии и построить consensus.

    Returns:
        ConsensusSignal или None
    """
    # Запускаем все стратегии
    strategy_results = self.analyze_all_strategies(symbol, candles, current_price)

    # Строим consensus
    consensus = self.build_consensus(symbol, strategy_results, current_price)

    if consensus:
      logger.info(
        f"✅ CONSENSUS SIGNAL [{symbol}]: {consensus.final_signal.signal_type.value}, "
        f"confidence={consensus.consensus_confidence:.2f}, "
        f"strategies={consensus.contributing_strategies}, "
        f"agreement={consensus.agreement_count}/{consensus.agreement_count + consensus.disagreement_count}"
      )

    return consensus

  def get_statistics(self) -> Dict:
    """Получить статистику менеджера."""
    consensus_rate = (
      self.consensus_achieved / self.signals_generated
      if self.signals_generated > 0
      else 0.0
    )

    # Статистика по каждой стратегии
    strategy_stats = {}
    for name, strategy in self.strategies.items():
      strategy_stats[name] = strategy.get_statistics()

    return {
      'total_analyses': self.total_analyses,
      'signals_generated': self.signals_generated,
      'consensus_achieved': self.consensus_achieved,
      'conflicts_resolved': self.conflicts_resolved,
      'consensus_rate': consensus_rate,
      'strategies': strategy_stats
    }


# Пример использования
if __name__ == "__main__":
  from strategy.candle_manager import Candle
  from datetime import datetime
  import random

  # Конфигурация
  config = StrategyManagerConfig(
    consensus_mode="weighted",
    min_strategies_for_signal=2,
    min_consensus_confidence=0.6
  )

  manager = StrategyManager(config)

  # Генерируем тестовые свечи
  base_price = 50000.0
  candles = []

  for i in range(150):
    trend = i * 10
    noise = random.uniform(-100, 100)
    price = base_price + trend + noise

    candle = Candle(
      timestamp=int(datetime.now().timestamp() * 1000) + i * 60000,
      open=price - 5,
      high=price + abs(random.uniform(20, 50)),
      low=price - abs(random.uniform(20, 50)),
      close=price,
      volume=1000 + random.uniform(-200, 200)
    )
    candles.append(candle)

  # Анализируем с consensus
  consensus = manager.analyze_with_consensus(
    "BTCUSDT",
    candles,
    candles[-1].close
  )

  if consensus:
    print(f"\nConsensus Signal:")
    print(f"  Type: {consensus.final_signal.signal_type.value}")
    print(f"  Confidence: {consensus.consensus_confidence:.2f}")
    print(f"  Contributing strategies: {consensus.contributing_strategies}")
    print(f"  Agreement: {consensus.agreement_count}/{consensus.agreement_count + consensus.disagreement_count}")
    print(f"  Reason: {consensus.final_signal.reason}")
  else:
    print("\nNo consensus achieved")

  # Статистика
  stats = manager.get_statistics()
  print(f"\nManager Statistics:")
  print(f"  Total analyses: {stats['total_analyses']}")
  print(f"  Signals generated: {stats['signals_generated']}")
  print(f"  Consensus rate: {stats['consensus_rate']:.2%}")

  print(f"\nStrategy Statistics:")
  for name, strategy_stats in stats['strategies'].items():
    print(f"  {name}:")
    for key, value in strategy_stats.items():
      print(f"    {key}: {value}")