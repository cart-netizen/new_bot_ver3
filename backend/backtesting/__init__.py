"""
Модуль бэктестинга торговых стратегий.

Обеспечивает event-driven бэктестинг с реалистичной симуляцией биржи.
"""

from backend.backtesting.core.data_handler import HistoricalDataHandler
from backend.backtesting.core.simulated_exchange import SimulatedExchange
from backend.backtesting.core.backtesting_engine import BacktestingEngine

__all__ = [
    'HistoricalDataHandler',
    'SimulatedExchange',
    'BacktestingEngine',
]
