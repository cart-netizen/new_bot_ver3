"""
Simple Backtesting Example.

Демонстрирует базовое использование backtesting framework с простой стратегией.

Usage:
    python example_backtest_simple.py

Requirements:
    - Historical data (parquet или CSV)
    - Backtesting modules installed

Path: example_backtest_simple.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from backend.backtesting import BacktestEngine, BacktestConfig
from backend.core.logger import get_logger

logger = get_logger(__name__)


class SimpleMovingAverageCrossStrategy:
    """
    Простая стратегия: пересечение SMA.

    Signals:
    - BUY: Fast SMA crosses above Slow SMA
    - SELL: Fast SMA crosses below Slow SMA
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period

        # State tracking
        self.prev_fast = None
        self.prev_slow = None

    def generate_signals(self, bar: pd.Series, timestamp: pd.Timestamp):
        """
        Generate trading signal based on SMA crossover.

        Args:
            bar: Current bar with OHLCV data and indicators
            timestamp: Current timestamp

        Returns:
            Signal object or None
        """
        # Check if we have SMA indicators
        if 'sma_fast' not in bar or 'sma_slow' not in bar:
            return None

        fast_sma = bar['sma_fast']
        slow_sma = bar['sma_slow']

        # Skip if NaN
        if pd.isna(fast_sma) or pd.isna(slow_sma):
            return None

        # Check for crossover
        signal = None

        if self.prev_fast is not None and self.prev_slow is not None:
            # Golden cross (bullish)
            if self.prev_fast <= self.prev_slow and fast_sma > slow_sma:
                signal = SimpleSignal(
                    action='buy',
                    stop_loss=bar['close'] * 0.98,  # 2% stop loss
                    take_profit=bar['close'] * 1.05  # 5% take profit
                )
                logger.info(f"📈 BUY signal: SMA cross at {bar['close']:.2f}")

            # Death cross (bearish)
            elif self.prev_fast >= self.prev_slow and fast_sma < slow_sma:
                signal = SimpleSignal(
                    action='sell',
                    stop_loss=bar['close'] * 1.02,  # 2% stop loss
                    take_profit=bar['close'] * 0.95  # 5% take profit
                )
                logger.info(f"📉 SELL signal: SMA cross at {bar['close']:.2f}")

        # Update state
        self.prev_fast = fast_sma
        self.prev_slow = slow_sma

        return signal


class SimpleSignal:
    """Simple signal object."""

    def __init__(self, action: str, stop_loss: float = None, take_profit: float = None):
        self.action = action
        self.stop_loss = stop_loss
        self.take_profit = take_profit


def prepare_data(csv_path: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Load and prepare data for backtesting.

    Args:
        csv_path: Path to CSV file with OHLCV data
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV + indicators
    """

    # Load data
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
    else:
        # Generate synthetic data for demo
        logger.warning("No CSV provided, generating synthetic data...")
        df = generate_synthetic_data()

    # Filter by date
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    # Calculate indicators
    df['sma_fast'] = df['close'].rolling(window=10).mean()
    df['sma_slow'] = df['close'].rolling(window=30).mean()

    # Drop NaN rows
    df = df.dropna()

    logger.info(f"Data prepared: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df


def generate_synthetic_data(n_bars: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic price data для демонстрации.

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)

    # Generate dates
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='1H')

    # Generate price with trend + noise
    trend = np.linspace(30000, 35000, n_bars)
    noise = np.random.randn(n_bars) * 200
    close_prices = trend + noise

    # Generate OHLC
    data = {
        'open': close_prices + np.random.randn(n_bars) * 50,
        'high': close_prices + abs(np.random.randn(n_bars)) * 100,
        'low': close_prices - abs(np.random.randn(n_bars)) * 100,
        'close': close_prices,
        'volume': np.random.uniform(10, 100, n_bars)
    }

    df = pd.DataFrame(data, index=dates)
    df.index.name = 'timestamp'

    return df


def run_simple_backtest():
    """
    Run simple backtest example.
    """
    logger.info("=" * 80)
    logger.info("SIMPLE BACKTESTING EXAMPLE")
    logger.info("=" * 80)

    # 1. Prepare data
    data = prepare_data()

    # 2. Create backtest configuration
    config = BacktestConfig(
        initial_capital=10000.0,
        slippage_pct=0.0005,  # 0.05% slippage
        maker_fee=0.0002,  # 0.02%
        taker_fee=0.0006,  # 0.06%
        max_position_size_pct=0.95,  # Use max 95% of capital
        enable_daily_loss_killer=True,
        max_daily_loss_pct=0.15  # 15% daily loss limit
    )

    # 3. Create strategy
    strategy = SimpleMovingAverageCrossStrategy(fast_period=10, slow_period=30)

    # 4. Create engine and run backtest
    engine = BacktestEngine(config)
    results = engine.run(strategy, data, verbose=True)

    # 5. Print results
    print("\n")
    print(results.summary())

    # 6. Detailed analysis
    if results.metrics:
        print("\n📊 DETAILED METRICS:")
        print(f"   Sharpe Ratio:     {results.metrics.sharpe_ratio:.2f}")
        print(f"   Sortino Ratio:    {results.metrics.sortino_ratio:.2f}")
        print(f"   Calmar Ratio:     {results.metrics.calmar_ratio:.2f}")
        print(f"   Omega Ratio:      {results.metrics.omega_ratio:.2f}")
        print(f"   VaR (95%):        {results.metrics.var_95:.4f}")
        print(f"   CVaR (95%):       {results.metrics.cvar_95:.4f}")
        print(f"   Skewness:         {results.metrics.skewness:.2f}")
        print(f"   Kurtosis:         {results.metrics.kurtosis:.2f}")

    # 7. Trade analysis
    if len(results.trades) > 0:
        print("\n📈 TOP 5 WINNING TRADES:")
        top_wins = results.trades.nlargest(5, 'pnl')
        for idx, trade in top_wins.iterrows():
            print(
                f"   {trade['entry_time']:%Y-%m-%d %H:%M} | "
                f"{trade['side'].upper():4s} | "
                f"PnL: ${trade['pnl']:>8.2f} ({trade['pnl_pct']:>6.2%}) | "
                f"Duration: {trade['duration']:.1f}h"
            )

        print("\n📉 TOP 5 LOSING TRADES:")
        top_losses = results.trades.nsmallest(5, 'pnl')
        for idx, trade in top_losses.iterrows():
            print(
                f"   {trade['entry_time']:%Y-%m-%d %H:%M} | "
                f"{trade['side'].upper():4s} | "
                f"PnL: ${trade['pnl']:>8.2f} ({trade['pnl_pct']:>6.2%}) | "
                f"Duration: {trade['duration']:.1f}h"
            )

    # 8. Save results
    results.trades.to_csv('backtest_trades.csv', index=False)
    results.equity_curve.to_csv('backtest_equity.csv')

    logger.info("✅ Results saved to backtest_trades.csv and backtest_equity.csv")

    # 9. Plot (optional)
    try:
        results.plot_equity_curve()
        results.plot_drawdown()
    except Exception as e:
        logger.warning(f"Plotting failed: {e} (matplotlib may not be available)")

    return results


if __name__ == "__main__":
    results = run_simple_backtest()
