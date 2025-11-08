"""
Backtesting Performance Metrics.

Comprehensive metrics для анализа результатов бэктестинга:
- Returns & risk metrics
- Drawdown analysis
- Trade statistics
- Risk-adjusted returns
- Statistical measures

Path: backend/backtesting/metrics.py
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BacktestMetrics:
    """Comprehensive backtesting performance metrics."""

    # Returns
    total_return: float
    annual_return: float
    monthly_returns: pd.Series

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int  # days

    # Win rate
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Profit metrics
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float

    # Trade duration
    avg_trade_duration: float
    max_trade_duration: float

    # Risk-adjusted
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional VaR
    omega_ratio: float

    # Statistical
    skewness: float
    kurtosis: float

    @classmethod
    def from_trades(cls, trades: pd.DataFrame, equity_curve: pd.Series):
        """
        Calculate all metrics from trades and equity curve.

        Args:
            trades: DataFrame with columns [pnl, duration, ...]
            equity_curve: Series with equity over time

        Returns:
            BacktestMetrics instance
        """
        # Returns
        returns = equity_curve.pct_change().dropna()

        # Period analysis
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        days = max(days, 1)  # Avoid division by zero

        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annual_return = (1 + total_return) ** (365 / days) - 1

        # Monthly returns
        monthly_equity = equity_curve.resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()

        # Volatility & Sharpe
        volatility = returns.std() * np.sqrt(365) if len(returns) > 1 else 0
        risk_free_rate = 0.03  # 3% annual risk-free rate
        sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Sortino (uses only downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 1 else 0
        sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0

        # Drawdown analysis
        cumulative = equity_curve
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0

        # Calmar ratio (return / max_dd)
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        # Trade statistics
        winning = trades[trades['pnl'] > 0]
        losing = trades[trades['pnl'] <= 0]

        total_trades = len(trades)
        winning_trades = len(winning)
        losing_trades = len(losing)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_win = winning['pnl'].mean() if len(winning) > 0 else 0
        avg_loss = losing['pnl'].mean() if len(losing) > 0 else 0

        # Profit factor (gross profit / gross loss)
        total_wins = winning['pnl'].sum() if len(winning) > 0 else 0
        total_losses = abs(losing['pnl'].sum()) if len(losing) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

        # Trade duration
        avg_duration = trades['duration'].mean() if 'duration' in trades.columns else 0
        max_duration = trades['duration'].max() if 'duration' in trades.columns else 0

        # VaR and CVaR (95% confidence)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else 0

        # Omega ratio (prob-weighted gains / losses)
        threshold = 0.0
        gains = returns[returns > threshold].sum() if (returns > threshold).any() else 0
        losses = abs(returns[returns <= threshold].sum()) if (returns <= threshold).any() else 0
        omega = gains / losses if losses > 0 else 0

        # Statistical measures
        skewness = returns.skew() if len(returns) > 2 else 0
        kurtosis = returns.kurtosis() if len(returns) > 3 else 0

        return cls(
            total_return=total_return,
            annual_return=annual_return,
            monthly_returns=monthly_returns,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            drawdown_duration=days,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_trade_duration=avg_duration,
            max_trade_duration=max_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            omega_ratio=omega,
            skewness=skewness,
            kurtosis=kurtosis
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'avg_drawdown': self.avg_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'avg_trade_duration': self.avg_trade_duration,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'omega_ratio': self.omega_ratio,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis
        }


class TradeAnalyzer:
    """Deep analysis of individual trades."""

    def analyze_trades(self, trades: pd.DataFrame) -> dict:
        """
        Comprehensive trade analysis.

        Returns dict with:
        - time_of_day: Performance by hour
        - day_of_week: Performance by weekday
        - consecutive_analysis: Streaks analysis
        - monthly_performance: Performance by month
        """
        analysis = {}

        if len(trades) == 0:
            return analysis

        # Ensure datetime index
        if 'entry_time' in trades.columns:
            trades = trades.copy()
            trades['hour'] = pd.to_datetime(trades['entry_time']).dt.hour
            trades['day_of_week'] = pd.to_datetime(trades['entry_time']).dt.dayofweek
            trades['month'] = pd.to_datetime(trades['entry_time']).dt.month

            # Time of day analysis
            analysis['time_of_day'] = self._analyze_time_patterns(trades)

            # Day of week analysis
            analysis['day_of_week'] = self._analyze_day_patterns(trades)

            # Monthly analysis
            analysis['monthly'] = self._analyze_monthly(trades)

        # Consecutive wins/losses
        analysis['streaks'] = self._analyze_streaks(trades)

        # Exit reason breakdown
        if 'exit_reason' in trades.columns:
            analysis['exit_reasons'] = trades['exit_reason'].value_counts().to_dict()

        return analysis

    def _analyze_time_patterns(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Performance by hour of day."""
        hourly = trades.groupby('hour').agg({
            'pnl': ['mean', 'sum', 'count'],
        }).round(4)

        hourly.columns = ['avg_pnl', 'total_pnl', 'trade_count']
        hourly['win_rate'] = trades.groupby('hour').apply(
            lambda x: (x['pnl'] > 0).mean()
        )

        return hourly

    def _analyze_day_patterns(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Performance by day of week."""
        daily = trades.groupby('day_of_week').agg({
            'pnl': ['mean', 'sum', 'count'],
        }).round(4)

        daily.columns = ['avg_pnl', 'total_pnl', 'trade_count']
        daily['win_rate'] = trades.groupby('day_of_week').apply(
            lambda x: (x['pnl'] > 0).mean()
        )

        # Map to day names
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily.index = daily.index.map(lambda x: day_names[x] if x < 7 else 'Unknown')

        return daily

    def _analyze_monthly(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Performance by month."""
        monthly = trades.groupby('month').agg({
            'pnl': ['mean', 'sum', 'count'],
        }).round(4)

        monthly.columns = ['avg_pnl', 'total_pnl', 'trade_count']
        monthly['win_rate'] = trades.groupby('month').apply(
            lambda x: (x['pnl'] > 0).mean()
        )

        return monthly

    def _analyze_streaks(self, trades: pd.DataFrame) -> dict:
        """Analyze winning/losing streaks."""
        if len(trades) == 0:
            return {}

        wins = (trades['pnl'] > 0).astype(int).values

        # Find streaks
        streaks = []
        current_streak = 1
        streak_type = wins[0]

        for i in range(1, len(wins)):
            if wins[i] == streak_type:
                current_streak += 1
            else:
                streaks.append((streak_type, current_streak))
                current_streak = 1
                streak_type = wins[i]

        # Add final streak
        streaks.append((streak_type, current_streak))

        # Separate win/loss streaks
        win_streaks = [s[1] for s in streaks if s[0] == 1]
        loss_streaks = [s[1] for s in streaks if s[0] == 0]

        return {
            'max_win_streak': max(win_streaks) if win_streaks else 0,
            'max_loss_streak': max(loss_streaks) if loss_streaks else 0,
            'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
            'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
            'total_win_streaks': len(win_streaks),
            'total_loss_streaks': len(loss_streaks)
        }
