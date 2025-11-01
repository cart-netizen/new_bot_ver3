# backend/strategies/adaptive/__init__.py
"""
Adaptive consensus components for dynamic strategy optimization.
"""

from backend.strategies.adaptive.strategy_performance_tracker import (
  StrategyPerformanceTracker,
  PerformanceTrackerConfig,
  SignalOutcome,
  StrategyMetrics
)

from backend.strategies.adaptive.market_regime_detector import (
  MarketRegimeDetector,
  RegimeDetectorConfig,
  MarketRegime,
  TrendRegime,
  VolatilityRegime,
  LiquidityRegime
)

from backend.strategies.adaptive.weight_optimizer import (
  WeightOptimizer,
  WeightOptimizerConfig,
  OptimizationMethod,
  WeightUpdate
)

from backend.strategies.adaptive.adaptive_consensus_manager import (
  AdaptiveConsensusManager,
  AdaptiveConsensusConfig,
  ConsensusQuality
)

__all__ = [
  # Performance Tracker
  'StrategyPerformanceTracker',
  'PerformanceTrackerConfig',
  'SignalOutcome',
  'StrategyMetrics',

  # Regime Detector
  'MarketRegimeDetector',
  'RegimeDetectorConfig',
  'MarketRegime',
  'TrendRegime',
  'VolatilityRegime',
  'LiquidityRegime',

  # Weight Optimizer
  'WeightOptimizer',
  'WeightOptimizerConfig',
  'OptimizationMethod',
  'WeightUpdate',

  # Adaptive Consensus Manager
  'AdaptiveConsensusManager',
  'AdaptiveConsensusConfig',
  'ConsensusQuality'
]