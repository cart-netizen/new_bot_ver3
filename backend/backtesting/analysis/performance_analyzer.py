"""
Performance Analyzer - расчет продвинутых метрик производительности бэктеста.

Метрики:
- Returns: Total, Annual, Monthly
- Risk-Adjusted: Sharpe, Sortino, Calmar
- Drawdown: Max DD, Avg DD, Duration, Recovery
- Trade Stats: Win rate, Profit factor, Expectancy
- Advanced: Omega, Tail ratio, VaR, CVaR
- Quality: Stability (R²), Consistency
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats

from backend.core.logger import get_logger
from backend.backtesting.models import (
    PerformanceMetrics,
    TradeResult,
    EquityPoint
)

logger = get_logger(__name__)


class PerformanceAnalyzer:
    """
    Анализатор производительности бэктеста.

    Рассчитывает полный набор метрик для оценки стратегии:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar)
    - Drawdown analysis
    - Trade statistics
    - Advanced risk metrics
    """

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Инициализация анализатора.

        Args:
            risk_free_rate: Безрисковая ставка (годовая, в процентах)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"PerformanceAnalyzer инициализирован (risk_free_rate={risk_free_rate}%)")

    def analyze(
        self,
        initial_capital: float,
        final_capital: float,
        equity_curve: List[EquityPoint],
        trades: List[TradeResult],
        start_date: datetime,
        end_date: datetime
    ) -> PerformanceMetrics:
        """
        Полный анализ производительности бэктеста.

        Args:
            initial_capital: Начальный капитал
            final_capital: Конечный капитал
            equity_curve: Кривая доходности
            trades: Список всех сделок
            start_date: Дата начала бэктеста
            end_date: Дата окончания бэктеста

        Returns:
            PerformanceMetrics с полным набором метрик
        """
        logger.info("📊 Начинаем расчет метрик производительности...")

        # Конвертация equity curve в pandas Series для удобства
        if equity_curve:
            equity_series = pd.Series(
                [point.equity for point in equity_curve],
                index=[point.timestamp for point in equity_curve]
            )
        else:
            # Если нет equity curve, создаем минимальную
            equity_series = pd.Series([initial_capital, final_capital])

        # Calculate returns
        returns_metrics = self._calculate_returns(
            initial_capital, final_capital, equity_series, start_date, end_date
        )

        # Calculate risk-adjusted metrics
        risk_metrics = self._calculate_risk_adjusted_metrics(equity_series)

        # Calculate drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(equity_series)

        # Calculate trade statistics
        trade_metrics = self._calculate_trade_statistics(trades)

        # Calculate advanced metrics
        advanced_metrics = self._calculate_advanced_metrics(equity_series)

        # Combine all metrics
        metrics = PerformanceMetrics(
            # Returns
            total_return=returns_metrics['total_return'],
            total_return_pct=returns_metrics['total_return_pct'],
            annual_return_pct=returns_metrics['annual_return_pct'],
            monthly_returns=returns_metrics['monthly_returns'],

            # Risk-adjusted
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            sortino_ratio=risk_metrics['sortino_ratio'],
            calmar_ratio=risk_metrics['calmar_ratio'],
            volatility_annual_pct=risk_metrics['volatility_annual_pct'],

            # Drawdown
            max_drawdown=drawdown_metrics['max_drawdown'],
            max_drawdown_pct=drawdown_metrics['max_drawdown_pct'],
            max_drawdown_duration_days=drawdown_metrics['max_drawdown_duration_days'],
            avg_drawdown_pct=drawdown_metrics['avg_drawdown_pct'],

            # Trade stats
            total_trades=trade_metrics['total_trades'],
            winning_trades=trade_metrics['winning_trades'],
            losing_trades=trade_metrics['losing_trades'],
            win_rate_pct=trade_metrics['win_rate_pct'],
            profit_factor=trade_metrics['profit_factor'],
            avg_win=trade_metrics['avg_win'],
            avg_loss=trade_metrics['avg_loss'],
            largest_win=trade_metrics['largest_win'],
            largest_loss=trade_metrics['largest_loss'],
            avg_trade_duration_minutes=trade_metrics['avg_trade_duration_minutes'],

            # Advanced
            omega_ratio=advanced_metrics['omega_ratio'],
            tail_ratio=advanced_metrics['tail_ratio'],
            var_95=advanced_metrics['var_95'],
            cvar_95=advanced_metrics['cvar_95'],
            stability=advanced_metrics['stability']
        )

        logger.info(
            f"✅ Метрики рассчитаны: Sharpe={metrics.sharpe_ratio:.2f}, "
            f"Max DD={metrics.max_drawdown_pct:.2f}%, Win Rate={metrics.win_rate_pct:.1f}%"
        )

        return metrics

    def _calculate_returns(
        self,
        initial_capital: float,
        final_capital: float,
        equity_series: pd.Series,
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """Расчет метрик доходности."""
        total_return = final_capital - initial_capital
        total_return_pct = (final_capital / initial_capital - 1) * 100

        # Annual return (annualized)
        days = (end_date - start_date).days
        if days > 0:
            years = days / 365.25
            annual_return_pct = ((final_capital / initial_capital) ** (1 / years) - 1) * 100
        else:
            annual_return_pct = 0.0

        # Monthly returns
        monthly_returns = self._calculate_monthly_returns(equity_series)

        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'annual_return_pct': annual_return_pct,
            'monthly_returns': monthly_returns
        }

    def _calculate_monthly_returns(self, equity_series: pd.Series) -> List[float]:
        """Расчет месячных доходностей."""
        if len(equity_series) < 2:
            return []

        # Resample to monthly
        try:
            monthly_equity = equity_series.resample('M').last()
            monthly_returns = monthly_equity.pct_change().dropna() * 100
            return monthly_returns.tolist()
        except Exception as e:
            logger.warning(f"Ошибка расчета месячных доходностей: {e}")
            return []

    def _calculate_risk_adjusted_metrics(self, equity_series: pd.Series) -> dict:
        """
        Расчет risk-adjusted метрик.

        Sharpe Ratio: (Mean Return - Risk Free Rate) / Std Dev
        Sortino Ratio: Mean Return / Downside Deviation
        Calmar Ratio: Annual Return / Max Drawdown
        """
        if len(equity_series) < 2:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'volatility_annual_pct': 0.0
            }

        # Calculate returns
        returns = equity_series.pct_change().dropna()

        if len(returns) == 0:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'volatility_annual_pct': 0.0
            }

        # Sharpe Ratio
        mean_return = returns.mean()
        std_return = returns.std()

        # Annualization factor (assuming daily equity curve)
        # For more accuracy, should detect actual frequency
        annualization_factor = np.sqrt(252)  # Assuming ~252 trading days

        if std_return > 0:
            sharpe_ratio = (mean_return / std_return) * annualization_factor
        else:
            sharpe_ratio = 0.0

        # Sortino Ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()

        if downside_std > 0:
            sortino_ratio = (mean_return / downside_std) * annualization_factor
        else:
            sortino_ratio = 0.0 if mean_return <= 0 else 999.0  # Very high if no downside

        # Calmar Ratio (Annual Return / Max Drawdown)
        # Will be calculated after drawdown metrics
        calmar_ratio = 0.0  # Placeholder

        # Volatility (annualized)
        volatility_annual_pct = std_return * annualization_factor * 100

        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,  # Will update later
            'volatility_annual_pct': volatility_annual_pct
        }

    def _calculate_drawdown_metrics(self, equity_series: pd.Series) -> dict:
        """
        Расчет метрик просадки.

        Max Drawdown: Максимальная просадка от пика
        Avg Drawdown: Средняя просадка
        Max DD Duration: Самая длинная просадка (дни)
        """
        if len(equity_series) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'max_drawdown_duration_days': 0.0,
                'avg_drawdown_pct': 0.0
            }

        # Calculate running maximum
        running_max = equity_series.expanding().max()

        # Calculate drawdown
        drawdown = equity_series - running_max
        drawdown_pct = (drawdown / running_max) * 100

        # Max drawdown
        max_drawdown = abs(drawdown.min())
        max_drawdown_pct = abs(drawdown_pct.min())

        # Average drawdown (only when in drawdown)
        drawdowns_only = drawdown_pct[drawdown_pct < 0]
        avg_drawdown_pct = abs(drawdowns_only.mean()) if len(drawdowns_only) > 0 else 0.0

        # Max drawdown duration
        max_dd_duration_days = self._calculate_max_drawdown_duration(equity_series, running_max)

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_duration_days': max_dd_duration_days,
            'avg_drawdown_pct': avg_drawdown_pct
        }

    def _calculate_max_drawdown_duration(
        self,
        equity_series: pd.Series,
        running_max: pd.Series
    ) -> float:
        """Расчет максимальной длительности просадки в днях."""
        try:
            # Find periods where equity is below peak
            in_drawdown = equity_series < running_max

            # Find start and end of each drawdown period
            drawdown_starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
            drawdown_ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)

            start_indices = drawdown_starts[drawdown_starts].index
            end_indices = drawdown_ends[drawdown_ends].index

            max_duration_days = 0.0

            for start_idx in start_indices:
                # Find corresponding end
                potential_ends = end_indices[end_indices > start_idx]

                if len(potential_ends) > 0:
                    end_idx = potential_ends[0]
                    duration = (end_idx - start_idx).total_seconds() / 86400  # Convert to days
                    max_duration_days = max(max_duration_days, duration)
                else:
                    # Drawdown extends to end of backtest
                    if len(equity_series) > 0:
                        end_idx = equity_series.index[-1]
                        duration = (end_idx - start_idx).total_seconds() / 86400
                        max_duration_days = max(max_duration_days, duration)

            return max_duration_days

        except Exception as e:
            logger.warning(f"Ошибка расчета длительности просадки: {e}")
            return 0.0

    def _calculate_trade_statistics(self, trades: List[TradeResult]) -> dict:
        """Расчет статистики по сделкам."""
        total_trades = len(trades)

        if total_trades == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate_pct': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_trade_duration_minutes': 0.0
            }

        # Separate winning and losing trades
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        winning_count = len(winning_trades)
        losing_count = len(losing_trades)

        # Win rate
        win_rate_pct = (winning_count / total_trades) * 100

        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))

        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # Average win/loss
        avg_win = total_wins / winning_count if winning_count > 0 else 0.0
        avg_loss = -total_losses / losing_count if losing_count > 0 else 0.0

        # Largest win/loss
        largest_win = max((t.pnl for t in winning_trades), default=0.0)
        largest_loss = min((t.pnl for t in losing_trades), default=0.0)

        # Average trade duration
        total_duration_minutes = sum(t.duration_seconds / 60 for t in trades)
        avg_trade_duration_minutes = total_duration_minutes / total_trades

        return {
            'total_trades': total_trades,
            'winning_trades': winning_count,
            'losing_trades': losing_count,
            'win_rate_pct': win_rate_pct,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_trade_duration_minutes': avg_trade_duration_minutes
        }

    def _calculate_advanced_metrics(self, equity_series: pd.Series) -> dict:
        """
        Расчет продвинутых метрик.

        Omega Ratio: Вероятность прибыли / Вероятность убытка
        Tail Ratio: 95th percentile / 5th percentile
        VaR: Value at Risk (95%)
        CVaR: Conditional VaR (95%)
        Stability: R² линейной регрессии equity curve
        """
        if len(equity_series) < 2:
            return {
                'omega_ratio': 0.0,
                'tail_ratio': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'stability': 0.0
            }

        returns = equity_series.pct_change().dropna()

        if len(returns) == 0:
            return {
                'omega_ratio': 0.0,
                'tail_ratio': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'stability': 0.0
            }

        # Omega Ratio
        threshold = 0.0  # Threshold return (0% = risk-free rate)
        omega_ratio = self._calculate_omega_ratio(returns, threshold)

        # Tail Ratio
        try:
            percentile_95 = np.percentile(returns, 95)
            percentile_5 = np.percentile(returns, 5)
            tail_ratio = abs(percentile_95 / percentile_5) if percentile_5 != 0 else 0.0
        except:
            tail_ratio = 0.0

        # VaR (Value at Risk) - 95% confidence
        var_95 = np.percentile(returns, 5) * 100  # 5th percentile (worst 5%)

        # CVaR (Conditional VaR) - Expected loss in worst 5%
        worst_5_pct = returns[returns <= np.percentile(returns, 5)]
        cvar_95 = worst_5_pct.mean() * 100 if len(worst_5_pct) > 0 else 0.0

        # Stability (R² of linear regression)
        stability = self._calculate_stability(equity_series)

        return {
            'omega_ratio': omega_ratio,
            'tail_ratio': tail_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'stability': stability
        }

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Omega Ratio: probability-weighted ratio of gains vs losses.

        Ω = E[R - threshold | R > threshold] / E[threshold - R | R < threshold]
        """
        try:
            gains = returns[returns > threshold] - threshold
            losses = threshold - returns[returns < threshold]

            if len(losses) == 0 or losses.sum() == 0:
                return 999.0 if len(gains) > 0 else 0.0

            omega = gains.sum() / losses.sum()
            return omega

        except Exception as e:
            logger.warning(f"Ошибка расчета Omega Ratio: {e}")
            return 0.0

    def _calculate_stability(self, equity_series: pd.Series) -> float:
        """
        Stability: R² линейной регрессии equity curve.

        Показывает насколько smooth кривая доходности.
        Высокий R² (близко к 1) = стабильный рост.
        """
        try:
            if len(equity_series) < 2:
                return 0.0

            # Linear regression
            x = np.arange(len(equity_series))
            y = equity_series.values

            # Calculate R²
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2

            return r_squared

        except Exception as e:
            logger.warning(f"Ошибка расчета stability: {e}")
            return 0.0

    def calculate_calmar_ratio(
        self,
        annual_return_pct: float,
        max_drawdown_pct: float
    ) -> float:
        """
        Calmar Ratio: Annualized Return / Max Drawdown.

        Показывает доходность относительно максимальной просадки.
        > 1.0: хорошо
        > 3.0: отлично
        """
        if max_drawdown_pct == 0:
            return 0.0

        return annual_return_pct / max_drawdown_pct


__all__ = ['PerformanceAnalyzer']
