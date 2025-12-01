"""
Backtesting Module - оценка ML моделей на исторических данных.

Модули:
- backtest_evaluator: Walk-forward бэктестинг и симуляция P&L
"""

from backend.ml_engine.backtesting.backtest_evaluator import (
    BacktestConfig,
    BacktestResults,
    BacktestEvaluator,
    run_backtest_from_checkpoint
)

__all__ = [
    'BacktestConfig',
    'BacktestResults',
    'BacktestEvaluator',
    'run_backtest_from_checkpoint'
]
