"""
Event-Driven Backtesting Engine.

Professional backtesting engine с:
- Realistic order execution
- Transaction costs
- Market impact modeling
- Comprehensive metrics
- Multi-symbol support

Path: backend/backtesting/engine.py
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from backend.core.logger import get_logger
from backend.models.orderbook import OrderBookSnapshot

logger = get_logger(__name__)


class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    # Capital
    initial_capital: float = 10000.0
    max_position_size_pct: float = 0.95  # Max 95% capital per position

    # Execution
    slippage_pct: float = 0.0005  # 0.05% slippage
    slippage_vol_impact: float = 0.0001  # Additional per 1 BTC

    # Fees (Bybit)
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0006  # 0.06%

    # Timeframe
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Risk management
    max_daily_loss_pct: float = 0.15  # 15% daily loss limit
    enable_daily_loss_killer: bool = True

    # Partial fills
    allow_partial_fills: bool = True
    max_fill_ratio: float = 0.3  # Max 30% of orderbook level

    # Latency
    latency_ms: int = 50
    latency_std_ms: int = 20


@dataclass
class BacktestPosition:
    """Open position in backtest."""

    symbol: str
    side: str
    entry_price: float
    size: float
    entry_time: pd.Timestamp
    entry_fee: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        if self.side == "buy":
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized PnL percentage."""
        pnl = self.unrealized_pnl(current_price)
        return pnl / (self.entry_price * self.size)


@dataclass
class BacktestTrade:
    """Closed trade record."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_fee: float
    exit_fee: float
    pnl: float
    pnl_pct: float
    duration: timedelta
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit', 'daily_loss_killer'


@dataclass
class BacktestState:
    """Current backtest state."""

    cash: float
    positions: Dict[str, BacktestPosition] = field(default_factory=dict)
    closed_trades: List[BacktestTrade] = field(default_factory=list)

    # Daily tracking
    daily_starting_equity: float = 0.0
    daily_pnl: float = 0.0

    # Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    def equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate total equity."""
        unrealized_pnl = sum(
            pos.unrealized_pnl(current_prices.get(symbol, pos.entry_price))
            for symbol, pos in self.positions.items()
        )
        return self.cash + unrealized_pnl

    def daily_loss_pct(self, current_prices: Dict[str, float]) -> float:
        """Calculate daily loss percentage."""
        current_equity = self.equity(current_prices)
        return (current_equity - self.daily_starting_equity) / self.daily_starting_equity


class BacktestEngine:
    """
    Professional Event-Driven Backtesting Engine.

    Features:
    - Realistic order execution с slippage
    - Transaction costs (maker/taker fees)
    - Market impact modeling
    - Risk management (daily loss killer)
    - Multi-symbol support
    - Comprehensive trade logging

    Usage:
        config = BacktestConfig(initial_capital=10000)
        engine = BacktestEngine(config)
        results = engine.run(strategy, data)
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.state = BacktestState(cash=config.initial_capital)

        # Equity curve tracking
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []

        # Event log
        self.event_log: List[dict] = []

    def run(
        self,
        strategy,
        data: pd.DataFrame,
        verbose: bool = True
    ) -> 'BacktestResults':
        """
        Run backtest.

        Args:
            strategy: Strategy implementing BacktestableStrategy interface
            data: DataFrame with OHLCV data (and features if needed)
            verbose: Log progress

        Returns:
            BacktestResults object
        """
        if verbose:
            logger.info("=" * 80)
            logger.info("STARTING BACKTEST")
            logger.info("=" * 80)
            logger.info(f"Initial Capital: ${self.config.initial_capital:,.2f}")
            logger.info(f"Period: {data.index[0]} to {data.index[-1]}")
            logger.info(f"Data points: {len(data):,}")

        # Filter by date if specified
        if self.config.start_date:
            data = data[data.index >= self.config.start_date]
        if self.config.end_date:
            data = data[data.index <= self.config.end_date]

        # Initialize
        self.state.daily_starting_equity = self.config.initial_capital
        self._reset_daily_stats(data.index[0])

        # Event-driven simulation
        for timestamp, row in data.iterrows():
            self._process_bar(timestamp, row, strategy)

        # Close all positions at end
        self._close_all_positions(data.index[-1], data.iloc[-1]['close'], "backtest_end")

        if verbose:
            logger.info("=" * 80)
            logger.info("BACKTEST COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Final Equity: ${self.state.equity({'symbol': row['close']}):.2f}")
            logger.info(f"Total Trades: {self.state.total_trades}")
            logger.info(f"Win Rate: {self.state.winning_trades / max(self.state.total_trades, 1) * 100:.1f}%")

        return self._create_results()

    def _process_bar(self, timestamp: pd.Timestamp, bar: pd.Series, strategy):
        """Process single bar."""

        current_price = bar['close']
        symbol = bar.get('symbol', 'BTCUSDT')  # Default if not specified

        # 1. Update equity curve
        current_equity = self.state.equity({symbol: current_price})
        self.equity_curve.append((timestamp, current_equity))

        # 2. Check daily reset
        if self._is_new_day(timestamp):
            self._reset_daily_stats(timestamp)

        # 3. Check daily loss killer
        if self.config.enable_daily_loss_killer:
            daily_loss = self.state.daily_loss_pct({symbol: current_price})
            if daily_loss < -self.config.max_daily_loss_pct:
                logger.warning(
                    f"⚠️ Daily loss limit hit: {daily_loss:.2%} < {-self.config.max_daily_loss_pct:.2%}"
                )
                self._close_all_positions(timestamp, current_price, "daily_loss_killer")
                return

        # 4. Check stop loss / take profit
        self._check_exit_conditions(timestamp, symbol, current_price)

        # 5. Generate signal from strategy
        signal = strategy.generate_signals(bar, timestamp)

        # 6. Execute signal
        if signal:
            self._execute_signal(timestamp, symbol, signal, current_price, bar)

    def _execute_signal(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        signal,
        current_price: float,
        bar: pd.Series
    ):
        """Execute trading signal."""

        # Check if already have position
        if symbol in self.state.positions:
            # Check if signal says to close
            if signal.action == 'close':
                self._close_position(timestamp, symbol, current_price, "signal")
            return

        # Open new position
        if signal.action in ['buy', 'sell']:
            self._open_position(timestamp, symbol, signal, current_price, bar)

    def _open_position(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        signal,
        current_price: float,
        bar: pd.Series
    ):
        """Open new position."""

        side = signal.action  # 'buy' or 'sell'

        # Calculate position size
        max_position_value = self.state.cash * self.config.max_position_size_pct
        position_size = max_position_value / current_price

        # Simulate order execution (with slippage)
        filled_price, filled_size, is_filled = self._simulate_fill(
            side, position_size, current_price
        )

        if not is_filled or filled_size == 0:
            logger.debug(f"Order not filled: {symbol} {side} {position_size:.4f} @ {current_price:.2f}")
            return

        # Calculate fee (market order = taker)
        notional = filled_price * filled_size
        fee = notional * self.config.taker_fee

        # Check if enough cash
        if side == "buy":
            cost = notional + fee
            if cost > self.state.cash:
                logger.debug(f"Insufficient cash: need ${cost:.2f}, have ${self.state.cash:.2f}")
                return
            self.state.cash -= cost
        else:
            # Short selling - remove cash for fee only
            self.state.cash -= fee

        # Create position
        position = BacktestPosition(
            symbol=symbol,
            side=side,
            entry_price=filled_price,
            size=filled_size,
            entry_time=timestamp,
            entry_fee=fee,
            stop_loss=signal.stop_loss if hasattr(signal, 'stop_loss') else None,
            take_profit=signal.take_profit if hasattr(signal, 'take_profit') else None
        )

        self.state.positions[symbol] = position

        logger.debug(
            f"✅ OPEN {side.upper()}: {symbol} {filled_size:.4f} @ ${filled_price:.2f}, "
            f"fee=${fee:.2f}"
        )

        self.event_log.append({
            'timestamp': timestamp,
            'event': 'position_open',
            'symbol': symbol,
            'side': side,
            'price': filled_price,
            'size': filled_size,
            'fee': fee
        })

    def _close_position(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        current_price: float,
        reason: str
    ):
        """Close existing position."""

        if symbol not in self.state.positions:
            return

        position = self.state.positions[symbol]

        # Simulate order execution (opposite side)
        close_side = "sell" if position.side == "buy" else "buy"
        filled_price, filled_size, is_filled = self._simulate_fill(
            close_side, position.size, current_price
        )

        if not is_filled:
            logger.warning(f"Failed to close position: {symbol}")
            return

        # Calculate fee
        notional = filled_price * filled_size
        exit_fee = notional * self.config.taker_fee

        # Calculate PnL
        if position.side == "buy":
            pnl = (filled_price - position.entry_price) * filled_size - position.entry_fee - exit_fee
            self.state.cash += notional - exit_fee
        else:
            pnl = (position.entry_price - filled_price) * filled_size - position.entry_fee - exit_fee
            self.state.cash += (position.entry_price * position.size) + pnl - exit_fee

        pnl_pct = pnl / (position.entry_price * position.size)

        # Create trade record
        trade = BacktestTrade(
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=filled_price,
            size=filled_size,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_fee=position.entry_fee,
            exit_fee=exit_fee,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration=timestamp - position.entry_time,
            exit_reason=reason
        )

        self.state.closed_trades.append(trade)
        self.state.total_trades += 1

        if pnl > 0:
            self.state.winning_trades += 1
        else:
            self.state.losing_trades += 1

        self.state.daily_pnl += pnl

        # Remove position
        del self.state.positions[symbol]

        logger.debug(
            f"❌ CLOSE {position.side.upper()}: {symbol} {filled_size:.4f} @ ${filled_price:.2f}, "
            f"PnL=${pnl:.2f} ({pnl_pct:.2%}), reason={reason}"
        )

        self.event_log.append({
            'timestamp': timestamp,
            'event': 'position_close',
            'symbol': symbol,
            'side': close_side,
            'price': filled_price,
            'size': filled_size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        })

    def _simulate_fill(
        self,
        side: str,
        size: float,
        price: float
    ) -> Tuple[float, float, bool]:
        """
        Simulate order fill with slippage.

        Returns:
            (filled_price, filled_size, is_filled)
        """
        # Apply slippage
        slippage = self.config.slippage_pct + (size * self.config.slippage_vol_impact)

        if side == "buy":
            filled_price = price * (1 + slippage)
        else:
            filled_price = price * (1 - slippage)

        # Assume market orders always fill (simplified)
        filled_size = size
        is_filled = True

        return filled_price, filled_size, is_filled

    def _check_exit_conditions(self, timestamp: pd.Timestamp, symbol: str, current_price: float):
        """Check stop loss / take profit."""

        if symbol not in self.state.positions:
            return

        position = self.state.positions[symbol]

        # Check stop loss
        if position.stop_loss:
            if position.side == "buy" and current_price <= position.stop_loss:
                self._close_position(timestamp, symbol, current_price, "stop_loss")
                return
            elif position.side == "sell" and current_price >= position.stop_loss:
                self._close_position(timestamp, symbol, current_price, "stop_loss")
                return

        # Check take profit
        if position.take_profit:
            if position.side == "buy" and current_price >= position.take_profit:
                self._close_position(timestamp, symbol, current_price, "take_profit")
                return
            elif position.side == "sell" and current_price <= position.take_profit:
                self._close_position(timestamp, symbol, current_price, "take_profit")
                return

    def _close_all_positions(self, timestamp: pd.Timestamp, price: float, reason: str):
        """Close all open positions."""
        for symbol in list(self.state.positions.keys()):
            self._close_position(timestamp, symbol, price, reason)

    def _is_new_day(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is a new day."""
        if not self.equity_curve:
            return True
        last_timestamp = self.equity_curve[-1][0]
        return timestamp.date() != last_timestamp.date()

    def _reset_daily_stats(self, timestamp: pd.Timestamp):
        """Reset daily statistics."""
        current_equity = self.state.equity({})  # No positions at day start
        self.state.daily_starting_equity = current_equity
        self.state.daily_pnl = 0.0

    def _create_results(self) -> 'BacktestResults':
        """Create BacktestResults object."""
        from .metrics import BacktestMetrics

        # Convert to DataFrames
        trades_df = pd.DataFrame([
            {
                'symbol': t.symbol,
                'side': t.side,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'size': t.size,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'duration': t.duration.total_seconds() / 3600,  # hours
                'exit_reason': t.exit_reason
            }
            for t in self.state.closed_trades
        ])

        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_df = equity_df.set_index('timestamp')

        # Calculate metrics
        if len(trades_df) > 0:
            metrics = BacktestMetrics.from_trades(trades_df, equity_df['equity'])
        else:
            # No trades
            metrics = None

        return BacktestResults(
            config=self.config,
            trades=trades_df,
            equity_curve=equity_df,
            metrics=metrics,
            event_log=self.event_log
        )


@dataclass
class BacktestResults:
    """Backtesting results."""

    config: BacktestConfig
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: Optional['BacktestMetrics']
    event_log: List[dict]

    def summary(self) -> str:
        """Generate summary report."""
        if self.metrics is None:
            return "No trades executed."

        report = []
        report.append("=" * 80)
        report.append("BACKTEST RESULTS SUMMARY")
        report.append("=" * 80)
        report.append("")
        report.append(f"Total Return:    {self.metrics.total_return:>10.2%}")
        report.append(f"Annual Return:   {self.metrics.annual_return:>10.2%}")
        report.append(f"Volatility:      {self.metrics.volatility:>10.2%}")
        report.append(f"Sharpe Ratio:    {self.metrics.sharpe_ratio:>10.2f}")
        report.append(f"Sortino Ratio:   {self.metrics.sortino_ratio:>10.2f}")
        report.append(f"Max Drawdown:    {self.metrics.max_drawdown:>10.2%}")
        report.append("")
        report.append(f"Total Trades:    {self.metrics.total_trades:>10}")
        report.append(f"Win Rate:        {self.metrics.win_rate:>10.2%}")
        report.append(f"Profit Factor:   {self.metrics.profit_factor:>10.2f}")
        report.append(f"Expectancy:      {self.metrics.expectancy:>10.4f}")
        report.append("=" * 80)

        return "\n".join(report)

    def plot_equity_curve(self):
        """Plot equity curve."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, self.equity_curve['equity'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_drawdown(self):
        """Plot drawdown."""
        import matplotlib.pyplot as plt

        equity = self.equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max

        plt.figure(figsize=(12, 6))
        plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        plt.plot(drawdown.index, drawdown, color='red')
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
