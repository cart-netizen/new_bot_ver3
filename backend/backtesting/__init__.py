"""
Professional Backtesting Framework.

Components:
- BacktestEngine: Core event-driven backtesting engine
- OrderFillModel: Realistic order execution simulation
- FeeModel: Transaction cost modeling
- BacktestMetrics: Comprehensive performance metrics
- WalkForwardOptimizer: Walk-forward optimization
- MonteCarloSimulator: Monte Carlo simulations

Path: backend/backtesting/__init__.py
"""

from .engine import BacktestEngine, BacktestConfig
from .execution import OrderFillModel, FeeModel, MarketImpactModel
from .metrics import BacktestMetrics, TradeAnalyzer
from .optimization import WalkForwardOptimizer, ParameterOptimizer
from .simulation import MonteCarloSimulator
from .data import BacktestDataLoader, BacktestFeatureLoader
from .validation import OOSValidator, StrategyValidator

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'OrderFillModel',
    'FeeModel',
    'MarketImpactModel',
    'BacktestMetrics',
    'TradeAnalyzer',
    'WalkForwardOptimizer',
    'ParameterOptimizer',
    'MonteCarloSimulator',
    'BacktestDataLoader',
    'BacktestFeatureLoader',
    'OOSValidator',
    'StrategyValidator',
]
