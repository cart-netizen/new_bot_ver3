"""
Advanced Backtesting Metrics Calculator

Расчет расширенных метрик производительности:
- Risk-adjusted returns (Sortino, Calmar, Omega)
- Consistency metrics (Profit Factor, Expectancy)
- Drawdown analysis (Ulcer Index, Recovery Factor)
- Trade distribution metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeData:
    """Данные одной сделки для расчета метрик"""
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    duration_seconds: float


@dataclass
class EquityData:
    """Данные equity curve для расчета метрик"""
    timestamp: datetime
    equity: float
    drawdown: float
    drawdown_pct: float
    total_return_pct: float


@dataclass
class AdvancedMetrics:
    """Расширенные метрики бэктеста"""

    # Risk-Adjusted Returns
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Consistency Metrics
    profit_factor: float
    expectancy: float
    kelly_criterion: float
    monthly_win_rate: float

    # Drawdown Analysis
    avg_drawdown: float
    avg_drawdown_pct: float
    avg_drawdown_duration_days: float
    max_drawdown_duration_days: float
    recovery_factor: float
    ulcer_index: float

    # Trade Distribution
    win_loss_ratio: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins_max: int
    consecutive_losses_max: int

    # Market Exposure
    market_exposure_pct: float
    avg_trade_duration_hours: float

    # Returns Distribution
    returns_skewness: float
    returns_kurtosis: float
    tail_ratio: float  # 95th percentile / 5th percentile

    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в словарь"""
        return {
            # Risk-Adjusted Returns
            "sortino_ratio": round(self.sortino_ratio, 3),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "omega_ratio": round(self.omega_ratio, 3),

            # Consistency Metrics
            "profit_factor": round(self.profit_factor, 3),
            "expectancy": round(self.expectancy, 3),
            "kelly_criterion": round(self.kelly_criterion, 3),
            "monthly_win_rate": round(self.monthly_win_rate, 2),

            # Drawdown Analysis
            "avg_drawdown": round(self.avg_drawdown, 2),
            "avg_drawdown_pct": round(self.avg_drawdown_pct, 2),
            "avg_drawdown_duration_days": round(self.avg_drawdown_duration_days, 2),
            "max_drawdown_duration_days": round(self.max_drawdown_duration_days, 2),
            "recovery_factor": round(self.recovery_factor, 3),
            "ulcer_index": round(self.ulcer_index, 3),

            # Trade Distribution
            "win_loss_ratio": round(self.win_loss_ratio, 3),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "consecutive_wins_max": self.consecutive_wins_max,
            "consecutive_losses_max": self.consecutive_losses_max,

            # Market Exposure
            "market_exposure_pct": round(self.market_exposure_pct, 2),
            "avg_trade_duration_hours": round(self.avg_trade_duration_hours, 2),

            # Returns Distribution
            "returns_skewness": round(self.returns_skewness, 3),
            "returns_kurtosis": round(self.returns_kurtosis, 3),
            "tail_ratio": round(self.tail_ratio, 3)
        }


class AdvancedMetricsCalculator:
    """Калькулятор расширенных метрик"""

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Args:
            risk_free_rate: Безрисковая ставка (годовая), default 0%
        """
        self.risk_free_rate = risk_free_rate

    def calculate(
        self,
        trades: List[TradeData],
        equity_curve: List[EquityData],
        initial_capital: float,
        final_capital: float,
        total_pnl: float,
        max_drawdown: float,
        start_date: datetime,
        end_date: datetime
    ) -> AdvancedMetrics:
        """
        Рассчитать все расширенные метрики

        Args:
            trades: Список сделок
            equity_curve: Equity curve точки
            initial_capital: Начальный капитал
            final_capital: Конечный капитал
            total_pnl: Общий PnL
            max_drawdown: Максимальная просадка
            start_date: Дата начала
            end_date: Дата окончания

        Returns:
            AdvancedMetrics объект со всеми метриками
        """

        # Подготовка данных
        returns = [t.pnl_pct / 100 for t in trades]  # Конвертировать в decimals
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        # Risk-Adjusted Returns
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(total_pnl, initial_capital, max_drawdown)
        omega_ratio = self._calculate_omega_ratio(returns)

        # Consistency Metrics
        profit_factor = self._calculate_profit_factor(winning_trades, losing_trades)
        expectancy = self._calculate_expectancy(trades)
        kelly_criterion = self._calculate_kelly_criterion(winning_trades, losing_trades, trades)
        monthly_win_rate = self._calculate_monthly_win_rate(trades)

        # Drawdown Analysis
        drawdown_metrics = self._analyze_drawdowns(equity_curve)
        recovery_factor = total_pnl / max_drawdown if max_drawdown > 0 else 0
        ulcer_index = self._calculate_ulcer_index(equity_curve)

        # Trade Distribution
        win_loss_ratio = self._calculate_win_loss_ratio(winning_trades, losing_trades)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = abs(min([t.pnl for t in losing_trades])) if losing_trades else 0
        consecutive_metrics = self._calculate_consecutive_streaks(trades)

        # Market Exposure
        market_exposure_pct = self._calculate_market_exposure(trades, start_date, end_date)
        avg_trade_duration_hours = self._calculate_avg_trade_duration(trades)

        # Returns Distribution
        returns_skewness = self._calculate_skewness(returns) if len(returns) > 2 else 0
        returns_kurtosis = self._calculate_kurtosis(returns) if len(returns) > 3 else 0
        tail_ratio = self._calculate_tail_ratio(returns) if len(returns) > 20 else 1.0

        return AdvancedMetrics(
            # Risk-Adjusted Returns
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,

            # Consistency Metrics
            profit_factor=profit_factor,
            expectancy=expectancy,
            kelly_criterion=kelly_criterion,
            monthly_win_rate=monthly_win_rate,

            # Drawdown Analysis
            avg_drawdown=drawdown_metrics["avg_drawdown"],
            avg_drawdown_pct=drawdown_metrics["avg_drawdown_pct"],
            avg_drawdown_duration_days=drawdown_metrics["avg_duration_days"],
            max_drawdown_duration_days=drawdown_metrics["max_duration_days"],
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,

            # Trade Distribution
            win_loss_ratio=win_loss_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins_max=consecutive_metrics["max_wins"],
            consecutive_losses_max=consecutive_metrics["max_losses"],

            # Market Exposure
            market_exposure_pct=market_exposure_pct,
            avg_trade_duration_hours=avg_trade_duration_hours,

            # Returns Distribution
            returns_skewness=returns_skewness,
            returns_kurtosis=returns_kurtosis,
            tail_ratio=tail_ratio
        )

    def _calculate_sortino_ratio(self, returns: List[float], target_return: float = 0.0) -> float:
        """
        Sortino Ratio - учитывает только downside volatility

        Args:
            returns: Список returns (в decimals)
            target_return: Целевая доходность

        Returns:
            Sortino ratio
        """
        if not returns or len(returns) < 2:
            return 0.0

        excess_returns = np.array(returns) - target_return
        avg_return = np.mean(excess_returns)

        # Downside deviation (только отрицательные returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if avg_return > 0 else 0.0

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return 0.0

        # Annualize
        sortino = (avg_return / downside_std) * np.sqrt(252)  # Assuming daily returns

        return float(sortino)

    def _calculate_calmar_ratio(self, total_pnl: float, initial_capital: float, max_drawdown: float) -> float:
        """
        Calmar Ratio = Annualized Return / Max Drawdown

        Args:
            total_pnl: Общий PnL
            initial_capital: Начальный капитал
            max_drawdown: Максимальная просадка (в USDT)

        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0.0

        total_return = total_pnl / initial_capital
        max_dd_pct = max_drawdown / initial_capital

        if max_dd_pct == 0:
            return 0.0

        calmar = total_return / max_dd_pct

        return float(calmar)

    def _calculate_omega_ratio(self, returns: List[float], threshold: float = 0.0) -> float:
        """
        Omega Ratio = Probability-weighted gains / Probability-weighted losses

        Args:
            returns: Список returns (в decimals)
            threshold: Threshold return

        Returns:
            Omega ratio
        """
        if not returns:
            return 0.0

        returns_array = np.array(returns)
        gains = returns_array[returns_array > threshold] - threshold
        losses = threshold - returns_array[returns_array < threshold]

        sum_gains = np.sum(gains)
        sum_losses = np.sum(losses)

        if sum_losses == 0:
            return float('inf') if sum_gains > 0 else 0.0

        omega = sum_gains / sum_losses

        return float(omega)

    def _calculate_profit_factor(self, winning_trades: List[TradeData], losing_trades: List[TradeData]) -> float:
        """
        Profit Factor = Gross Profit / Gross Loss

        Args:
            winning_trades: Прибыльные сделки
            losing_trades: Убыточные сделки

        Returns:
            Profit factor
        """
        if not winning_trades and not losing_trades:
            return 0.0

        gross_profit = sum([t.pnl for t in winning_trades])
        gross_loss = abs(sum([t.pnl for t in losing_trades]))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _calculate_expectancy(self, trades: List[TradeData]) -> float:
        """
        Expectancy = Average PnL per trade

        Args:
            trades: Все сделки

        Returns:
            Expectancy
        """
        if not trades:
            return 0.0

        return np.mean([t.pnl for t in trades])

    def _calculate_kelly_criterion(
        self,
        winning_trades: List[TradeData],
        losing_trades: List[TradeData],
        all_trades: List[TradeData]
    ) -> float:
        """
        Kelly Criterion = W - ((1 - W) / R)
        где W = win rate, R = avg win / avg loss

        Args:
            winning_trades: Прибыльные сделки
            losing_trades: Убыточные сделки
            all_trades: Все сделки

        Returns:
            Kelly percentage (оптимальный размер позиции)
        """
        if not all_trades:
            return 0.0

        win_rate = len(winning_trades) / len(all_trades)

        if not winning_trades or not losing_trades:
            return 0.0

        avg_win = np.mean([t.pnl for t in winning_trades])
        avg_loss = abs(np.mean([t.pnl for t in losing_trades]))

        if avg_loss == 0:
            return 0.0

        win_loss_ratio = avg_win / avg_loss

        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Clamp between 0 and 1
        kelly = max(0.0, min(1.0, kelly))

        return kelly

    def _calculate_monthly_win_rate(self, trades: List[TradeData]) -> float:
        """
        Процент прибыльных месяцев

        Args:
            trades: Все сделки

        Returns:
            Monthly win rate %
        """
        if not trades:
            return 0.0

        # Группировать по месяцам
        monthly_pnl = {}
        for trade in trades:
            month_key = trade.exit_time.strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += trade.pnl

        if not monthly_pnl:
            return 0.0

        winning_months = sum(1 for pnl in monthly_pnl.values() if pnl > 0)

        return (winning_months / len(monthly_pnl)) * 100

    def _analyze_drawdowns(self, equity_curve: List[EquityData]) -> Dict[str, float]:
        """
        Анализ всех просадок

        Args:
            equity_curve: Equity curve точки

        Returns:
            Dict с метриками просадок
        """
        if not equity_curve:
            return {
                "avg_drawdown": 0.0,
                "avg_drawdown_pct": 0.0,
                "avg_duration_days": 0.0,
                "max_duration_days": 0.0
            }

        # Найти все периоды просадок
        drawdown_periods = []
        in_drawdown = False
        current_dd_start = None
        current_dd_max = 0
        current_dd_max_pct = 0

        for point in equity_curve:
            if point.drawdown > 0:
                if not in_drawdown:
                    # Начало новой просадки
                    in_drawdown = True
                    current_dd_start = point.timestamp
                    current_dd_max = point.drawdown
                    current_dd_max_pct = point.drawdown_pct
                else:
                    # Продолжение просадки
                    current_dd_max = max(current_dd_max, point.drawdown)
                    current_dd_max_pct = max(current_dd_max_pct, point.drawdown_pct)
            else:
                if in_drawdown:
                    # Конец просадки
                    drawdown_periods.append({
                        "start": current_dd_start,
                        "end": point.timestamp,
                        "max_dd": current_dd_max,
                        "max_dd_pct": current_dd_max_pct,
                        "duration": (point.timestamp - current_dd_start).total_seconds() / 86400  # days
                    })
                    in_drawdown = False

        # Если закончились в просадке
        if in_drawdown and equity_curve:
            drawdown_periods.append({
                "start": current_dd_start,
                "end": equity_curve[-1].timestamp,
                "max_dd": current_dd_max,
                "max_dd_pct": current_dd_max_pct,
                "duration": (equity_curve[-1].timestamp - current_dd_start).total_seconds() / 86400
            })

        if not drawdown_periods:
            return {
                "avg_drawdown": 0.0,
                "avg_drawdown_pct": 0.0,
                "avg_duration_days": 0.0,
                "max_duration_days": 0.0
            }

        avg_dd = np.mean([dd["max_dd"] for dd in drawdown_periods])
        avg_dd_pct = np.mean([dd["max_dd_pct"] for dd in drawdown_periods])
        avg_duration = np.mean([dd["duration"] for dd in drawdown_periods])
        max_duration = max([dd["duration"] for dd in drawdown_periods])

        return {
            "avg_drawdown": float(avg_dd),
            "avg_drawdown_pct": float(avg_dd_pct),
            "avg_duration_days": float(avg_duration),
            "max_duration_days": float(max_duration)
        }

    def _calculate_ulcer_index(self, equity_curve: List[EquityData]) -> float:
        """
        Ulcer Index - мера глубины и длительности просадок

        Args:
            equity_curve: Equity curve точки

        Returns:
            Ulcer Index
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0

        # Squared drawdowns
        squared_dds = [point.drawdown_pct ** 2 for point in equity_curve]

        # Mean of squared drawdowns
        mean_squared = np.mean(squared_dds)

        # Ulcer Index = sqrt(mean squared DD)
        ulcer = np.sqrt(mean_squared)

        return float(ulcer)

    def _calculate_win_loss_ratio(self, winning_trades: List[TradeData], losing_trades: List[TradeData]) -> float:
        """
        Win/Loss Ratio = Average Win / Average Loss

        Args:
            winning_trades: Прибыльные сделки
            losing_trades: Убыточные сделки

        Returns:
            Win/Loss ratio
        """
        if not winning_trades or not losing_trades:
            return 0.0

        avg_win = np.mean([t.pnl for t in winning_trades])
        avg_loss = abs(np.mean([t.pnl for t in losing_trades]))

        if avg_loss == 0:
            return 0.0

        return avg_win / avg_loss

    def _calculate_consecutive_streaks(self, trades: List[TradeData]) -> Dict[str, int]:
        """
        Максимальные серии побед/поражений подряд

        Args:
            trades: Все сделки

        Returns:
            Dict с max_wins и max_losses
        """
        if not trades:
            return {"max_wins": 0, "max_losses": 0}

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return {
            "max_wins": max_wins,
            "max_losses": max_losses
        }

    def _calculate_market_exposure(self, trades: List[TradeData], start_date: datetime, end_date: datetime) -> float:
        """
        Процент времени в рынке

        Args:
            trades: Все сделки
            start_date: Начало периода
            end_date: Конец периода

        Returns:
            Market exposure %
        """
        if not trades:
            return 0.0

        total_duration = (end_date - start_date).total_seconds()

        if total_duration == 0:
            return 0.0

        # Сумма времени в сделках
        trade_duration = sum([t.duration_seconds for t in trades])

        exposure = (trade_duration / total_duration) * 100

        return min(100.0, exposure)  # Cap at 100%

    def _calculate_avg_trade_duration(self, trades: List[TradeData]) -> float:
        """
        Средняя длительность сделки в часах

        Args:
            trades: Все сделки

        Returns:
            Средняя длительность в часах
        """
        if not trades:
            return 0.0

        avg_seconds = np.mean([t.duration_seconds for t in trades])

        return avg_seconds / 3600  # Convert to hours

    def _calculate_skewness(self, returns: List[float]) -> float:
        """
        Skewness - асимметрия распределения returns

        Args:
            returns: Список returns

        Returns:
            Skewness
        """
        if len(returns) < 3:
            return 0.0

        returns_array = np.array(returns)
        mean = np.mean(returns_array)
        std = np.std(returns_array, ddof=1)

        if std == 0:
            return 0.0

        n = len(returns)
        skew = (n / ((n - 1) * (n - 2))) * np.sum(((returns_array - mean) / std) ** 3)

        return float(skew)

    def _calculate_kurtosis(self, returns: List[float]) -> float:
        """
        Kurtosis - "толщина хвостов" распределения

        Args:
            returns: Список returns

        Returns:
            Excess kurtosis (0 = normal distribution)
        """
        if len(returns) < 4:
            return 0.0

        returns_array = np.array(returns)
        mean = np.mean(returns_array)
        std = np.std(returns_array, ddof=1)

        if std == 0:
            return 0.0

        n = len(returns)
        kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((returns_array - mean) / std) ** 4)
        kurt -= (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))  # Excess kurtosis

        return float(kurt)

    def _calculate_tail_ratio(self, returns: List[float]) -> float:
        """
        Tail Ratio = 95th percentile / |5th percentile|

        Args:
            returns: Список returns

        Returns:
            Tail ratio
        """
        if len(returns) < 20:
            return 1.0

        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)

        if p5 == 0:
            return 0.0

        return abs(p95 / p5)
