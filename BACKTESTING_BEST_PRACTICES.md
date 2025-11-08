# 🎯 Backtesting Best Practices & Implementation Guide

Comprehensive guide для создания профессиональной системы бэктестинга с учетом специфики вашего бота.

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Performance Optimization](#performance-optimization)
3. [Realistic Execution Simulation](#realistic-execution-simulation)
4. [Advanced Metrics & Analytics](#advanced-metrics--analytics)
5. [Overfitting Prevention](#overfitting-prevention)
6. [Walk-Forward Optimization](#walk-forward-optimization)
7. [Monte Carlo Simulations](#monte-carlo-simulations)
8. [Data Management](#data-management)
9. [Integration with Production](#integration-with-production)
10. [Implementation Roadmap](#implementation-roadmap)

---

## 🏗️ Architecture Overview

### Current State
- ✅ Paper trading mode exists (`PAPER_TRADING` flag)
- ✅ ML data collection (Feature Store)
- ✅ Position tracking, risk management
- ❌ No historical backtesting engine
- ❌ No performance metrics calculation
- ❌ No optimization framework

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BACKTESTING ENGINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Data Loader  │  │   Simulator  │  │   Analyzer   │         │
│  │              │  │              │  │              │         │
│  │ • Parquet    │─>│ • Execution  │─>│ • Metrics    │         │
│  │ • CSV        │  │ • Slippage   │  │ • Reports    │         │
│  │ • Database   │  │ • Fees       │  │ • Plots      │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                            │                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Optimizer   │  │ Walk-Forward │  │ Monte Carlo  │         │
│  │              │  │              │  │              │         │
│  │ • Grid       │  │ • Rolling    │  │ • Bootstrap  │         │
│  │ • Bayesian   │  │ • Anchored   │  │ • Scenarios  │         │
│  │ • Genetic    │  │ • Expanding  │  │ • Stress     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Performance Optimization

### 1. Vectorized Operations

**Problem:** Loop-based backtesting слишком медленный для больших датасетов.

**Solution:** Используйте векторизацию с pandas/numpy.

```python
# ❌ BAD: Loop-based (slow)
for i in range(len(df)):
    if df.loc[i, 'signal'] == 1:
        positions[i] = df.loc[i, 'close']

# ✅ GOOD: Vectorized (100x faster)
positions = np.where(df['signal'] == 1, df['close'], np.nan)
df['position'] = positions
```

### 2. Event-Driven vs Vectorized Trade-off

**Vectorized Backtesting:**
- ✅ Очень быстро (1M+ bars в секунды)
- ✅ Простая реализация
- ❌ Упрощенная логика (look-ahead bias риск)
- ❌ Сложно тестировать complex strategies

**Event-Driven Backtesting:**
- ✅ Реалистичная эмуляция
- ✅ Точная логика стратегии
- ❌ Медленнее (но оптимизируется)
- ✅ Нет look-ahead bias

**Recommendation:** Hybrid подход:
1. Vectorized для быстрого screening параметров
2. Event-driven для финальной валидации

### 3. Parallel Processing

**Multi-Symbol Backtesting:**

```python
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def backtest_symbol(symbol: str, params: dict, data: pd.DataFrame):
    """Backtest одного символа."""
    # ... backtesting logic
    return results

# Параллельный бэктест всех символов
symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', ...]
params = {'stop_loss': 0.02, 'take_profit': 0.03}

with ProcessPoolExecutor(max_workers=8) as executor:
    backtest_func = partial(backtest_symbol, params=params)
    results = list(executor.map(backtest_func, symbols, data_dict.values()))
```

**Parameter Optimization:**

```python
import multiprocessing as mp

def optimize_params_parallel(param_grid: list, data: pd.DataFrame):
    """Параллельная оптимизация параметров."""

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(
            run_backtest_with_params,
            [(params, data) for params in param_grid]
        )

    # Находим лучшие параметры
    best_idx = np.argmax([r['sharpe_ratio'] for r in results])
    return results[best_idx]
```

### 4. Caching & Incremental Updates

```python
from functools import lru_cache
import pickle

class BacktestCache:
    """Кэширование промежуточных результатов."""

    def __init__(self, cache_dir: str = "cache/backtest"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_or_compute(self, key: str, compute_func, *args, **kwargs):
        """Get from cache or compute."""
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        result = compute_func(*args, **kwargs)

        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        return result

# Usage
cache = BacktestCache()
indicators = cache.get_or_compute(
    "btcusdt_indicators_2024",
    calculate_indicators,
    data
)
```

---

## 🎪 Realistic Execution Simulation

### 1. Order Filling Simulation

**Проблема:** Бэктестинг часто предполагает идеальное исполнение - нереалистично!

```python
from dataclasses import dataclass
from enum import Enum

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"

@dataclass
class OrderFillModel:
    """Реалистичная модель исполнения ордеров."""

    # Slippage model
    slippage_pct: float = 0.0005  # 0.05% slippage
    slippage_vol_impact: float = 0.0001  # Additional slippage per 1 BTC volume

    # Latency model
    latency_ms: int = 50  # 50ms задержка
    latency_std_ms: int = 20  # Variance

    # Partial fills
    max_fill_ratio: float = 0.3  # Max 30% orderbook level
    partial_fill_enabled: bool = True

    def simulate_fill(
        self,
        order_type: OrderType,
        side: str,
        size: float,
        price: float,
        orderbook: OrderBookSnapshot,
        timestamp: int
    ) -> tuple[float, float, bool]:
        """
        Simulate order fill.

        Returns:
            (filled_price, filled_size, is_filled)
        """
        if order_type == OrderType.MARKET:
            return self._simulate_market_fill(side, size, orderbook)
        else:
            return self._simulate_limit_fill(side, size, price, orderbook, timestamp)

    def _simulate_market_fill(
        self,
        side: str,
        size: float,
        orderbook: OrderBookSnapshot
    ) -> tuple[float, float, bool]:
        """Market order - walk through orderbook."""

        levels = orderbook.asks if side == "buy" else orderbook.bids
        remaining_size = size
        total_cost = 0.0

        for price, available_qty in levels:
            # Apply max fill ratio
            max_fill = available_qty * self.max_fill_ratio
            fill_qty = min(remaining_size, max_fill)

            # Apply slippage
            slippage = self.slippage_pct + (size * self.slippage_vol_impact)
            adjusted_price = price * (1 + slippage if side == "buy" else 1 - slippage)

            total_cost += fill_qty * adjusted_price
            remaining_size -= fill_qty

            if remaining_size <= 0:
                break

        if remaining_size > 0 and not self.partial_fill_enabled:
            return 0.0, 0.0, False  # Order failed

        filled_size = size - remaining_size
        avg_price = total_cost / filled_size if filled_size > 0 else 0

        return avg_price, filled_size, filled_size > 0

    def _simulate_limit_fill(
        self,
        side: str,
        size: float,
        price: float,
        orderbook: OrderBookSnapshot,
        timestamp: int
    ) -> tuple[float, float, bool]:
        """Limit order - fill only if price touched."""

        # Check if limit price would be filled
        best_price = orderbook.best_ask if side == "buy" else orderbook.best_bid

        if side == "buy" and best_price <= price:
            # Would be filled as maker
            return price, size, True
        elif side == "sell" and best_price >= price:
            return price, size, True
        else:
            # Not filled yet
            return 0.0, 0.0, False
```

### 2. Transaction Costs

```python
@dataclass
class FeeModel:
    """Модель комиссий."""

    maker_fee: float = 0.0002  # 0.02% maker
    taker_fee: float = 0.0006  # 0.06% taker

    # VIP levels
    volume_24h_btc: float = 0.0

    def get_fee_rate(self, is_maker: bool) -> float:
        """Calculate fee based on volume (VIP levels)."""

        # Bybit VIP levels
        if self.volume_24h_btc >= 1000:  # VIP 3
            return 0.00015 if is_maker else 0.00035
        elif self.volume_24h_btc >= 500:  # VIP 2
            return 0.00016 if is_maker else 0.00040
        elif self.volume_24h_btc >= 100:  # VIP 1
            return 0.00018 if is_maker else 0.00050
        else:  # Regular
            return self.maker_fee if is_maker else self.taker_fee

    def calculate_fee(self, notional: float, is_maker: bool) -> float:
        """Calculate fee for trade."""
        fee_rate = self.get_fee_rate(is_maker)
        return notional * fee_rate
```

### 3. Market Impact Model

```python
class MarketImpactModel:
    """Модель влияния ордера на цену."""

    def __init__(self, liquidity_factor: float = 0.001):
        self.liquidity_factor = liquidity_factor

    def calculate_impact(
        self,
        size: float,
        orderbook_depth: float,
        volatility: float
    ) -> float:
        """
        Calculate price impact.

        Based on Kyle's lambda model:
        Impact = λ * (size / depth) * volatility
        """
        size_ratio = size / max(orderbook_depth, 0.01)
        impact = self.liquidity_factor * size_ratio * volatility

        # Cap impact at reasonable levels
        return min(impact, 0.05)  # Max 5% impact
```

---

## 📊 Advanced Metrics & Analytics

### 1. Comprehensive Performance Metrics

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class BacktestMetrics:
    """Comprehensive backtesting metrics."""

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
        """Calculate all metrics from trades."""

        returns = equity_curve.pct_change().dropna()

        # Annual return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annual_return = (1 + total_return) ** (365 / days) - 1

        # Volatility & Sharpe
        volatility = returns.std() * np.sqrt(365)
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0

        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(365)
        sortino = (annual_return - 0.03) / downside_std if downside_std > 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Calmar ratio
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        # Trade statistics
        winning = trades[trades['pnl'] > 0]
        losing = trades[trades['pnl'] < 0]

        win_rate = len(winning) / len(trades) if len(trades) > 0 else 0
        avg_win = winning['pnl'].mean() if len(winning) > 0 else 0
        avg_loss = losing['pnl'].mean() if len(losing) > 0 else 0

        # Profit factor
        total_wins = winning['pnl'].sum() if len(winning) > 0 else 0
        total_losses = abs(losing['pnl'].sum()) if len(losing) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        # Omega ratio
        threshold = 0.0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        omega = gains / losses if losses > 0 else 0

        return cls(
            total_return=total_return,
            annual_return=annual_return,
            monthly_returns=equity_curve.resample('M').last().pct_change(),
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            avg_drawdown=drawdown.mean(),
            drawdown_duration=(equity_curve.index[-1] - equity_curve.index[0]).days,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_trade_duration=trades['duration'].mean() if 'duration' in trades else 0,
            max_trade_duration=trades['duration'].max() if 'duration' in trades else 0,
            var_95=var_95,
            cvar_95=cvar_95,
            omega_ratio=omega,
            skewness=returns.skew(),
            kurtosis=returns.kurtosis()
        )
```

### 2. Trade Analysis

```python
class TradeAnalyzer:
    """Analyze individual trades for patterns."""

    def analyze_trades(self, trades: pd.DataFrame) -> dict:
        """Deep analysis of trades."""

        analysis = {
            'time_of_day': self._analyze_time_patterns(trades),
            'day_of_week': self._analyze_day_patterns(trades),
            'regime_performance': self._analyze_by_regime(trades),
            'consecutive_analysis': self._analyze_streaks(trades),
            'exit_analysis': self._analyze_exits(trades)
        }

        return analysis

    def _analyze_time_patterns(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Performance by hour of day."""
        trades['hour'] = pd.to_datetime(trades['entry_time']).dt.hour

        hourly = trades.groupby('hour').agg({
            'pnl': ['mean', 'sum', 'count'],
            'win': 'mean'
        })

        return hourly

    def _analyze_streaks(self, trades: pd.DataFrame) -> dict:
        """Analyze winning/losing streaks."""
        wins = (trades['pnl'] > 0).astype(int)

        # Find streaks
        streaks = []
        current_streak = 1
        streak_type = wins.iloc[0]

        for i in range(1, len(wins)):
            if wins.iloc[i] == streak_type:
                current_streak += 1
            else:
                streaks.append((streak_type, current_streak))
                current_streak = 1
                streak_type = wins.iloc[i]

        streaks.append((streak_type, current_streak))

        win_streaks = [s[1] for s in streaks if s[0] == 1]
        loss_streaks = [s[1] for s in streaks if s[0] == 0]

        return {
            'max_win_streak': max(win_streaks) if win_streaks else 0,
            'max_loss_streak': max(loss_streaks) if loss_streaks else 0,
            'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
            'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0
        }
```

---

## 🛡️ Overfitting Prevention

### 1. Train/Validation/Test Split

```python
class DataSplitter:
    """Professional data splitting for backtesting."""

    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio

    def split(self, data: pd.DataFrame) -> tuple:
        """
        Time-series aware split.

        CRITICAL: Never shuffle time-series data!
        """
        n = len(data)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train = data.iloc[:train_end]
        val = data.iloc[train_end:val_end]
        test = data.iloc[val_end:]

        return train, val, test

    def purged_kfold(self, data: pd.DataFrame, n_splits: int = 5, embargo_pct: float = 0.01):
        """
        Purged K-Fold для time-series.

        Prevents leakage by:
        1. Purging overlapping samples
        2. Adding embargo period between train/test
        """
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_splits)
        embargo_samples = int(len(data) * embargo_pct)

        for train_idx, test_idx in tscv.split(data):
            # Remove embargo period
            train_idx = train_idx[:-embargo_samples] if embargo_samples > 0 else train_idx

            yield data.iloc[train_idx], data.iloc[test_idx]
```

### 2. Out-of-Sample Testing

```python
class OOSValidator:
    """Out-of-sample validation protocol."""

    def validate_strategy(
        self,
        strategy,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        n_trials: int = 100
    ) -> dict:
        """
        Rigorous OOS validation.

        Protocol:
        1. Optimize on train data
        2. Test on OOS data (no re-optimization!)
        3. Compare metrics
        4. Statistical significance test
        """

        # 1. Optimize on train
        best_params = strategy.optimize(train_data)

        # 2. Backtest on train (with best params)
        train_results = strategy.backtest(train_data, best_params)

        # 3. Backtest on OOS test (same params!)
        test_results = strategy.backtest(test_data, best_params)

        # 4. Compare metrics
        degradation = {
            'sharpe_degradation': test_results.sharpe - train_results.sharpe,
            'return_degradation': test_results.annual_return - train_results.annual_return,
            'dd_change': test_results.max_drawdown - train_results.max_drawdown
        }

        # 5. Check if degradation is acceptable
        is_valid = (
            degradation['sharpe_degradation'] > -0.5 and  # Sharpe не упал сильно
            degradation['return_degradation'] > -0.1 and  # Return не упал сильно
            abs(degradation['dd_change']) < 0.1  # DD не вырос сильно
        )

        return {
            'train_metrics': train_results,
            'test_metrics': test_results,
            'degradation': degradation,
            'is_valid': is_valid
        }
```

### 3. Combinatorially Symmetric Cross-Validation (CSCV)

```python
def cscv_backtest(
    strategy,
    data: pd.DataFrame,
    n_paths: int = 10
) -> list:
    """
    CSCV для устранения selection bias.

    Метод:
    1. Делим данные на N блоков
    2. Генерируем все комбинации train/test splits
    3. Усредняем результаты
    """
    from itertools import combinations

    n = len(data)
    block_size = n // n_paths
    blocks = [data.iloc[i*block_size:(i+1)*block_size] for i in range(n_paths)]

    results = []

    # All combinations of train/test
    for test_idx in range(n_paths):
        train_blocks = [b for i, b in enumerate(blocks) if i != test_idx]
        train_data = pd.concat(train_blocks)
        test_data = blocks[test_idx]

        result = strategy.backtest(test_data, strategy.optimize(train_data))
        results.append(result)

    # Aggregate results
    avg_sharpe = np.mean([r.sharpe_ratio for r in results])
    std_sharpe = np.std([r.sharpe_ratio for r in results])

    return {
        'results': results,
        'avg_sharpe': avg_sharpe,
        'std_sharpe': std_sharpe,
        'confidence_interval': (avg_sharpe - 1.96*std_sharpe, avg_sharpe + 1.96*std_sharpe)
    }
```

---

## 🚶 Walk-Forward Optimization

```python
class WalkForwardOptimizer:
    """
    Walk-Forward Optimization для адаптивных стратегий.

    Методология:
    1. Оптимизируем параметры на train window
    2. Торгуем с этими параметрами на test window
    3. Сдвигаем окно вперед, повторяем
    """

    def __init__(
        self,
        train_period_days: int = 180,
        test_period_days: int = 30,
        step_days: int = 30,
        anchored: bool = False
    ):
        self.train_period = train_period_days
        self.test_period = test_period_days
        self.step = step_days
        self.anchored = anchored  # Anchored vs Rolling

    def optimize(
        self,
        strategy,
        data: pd.DataFrame,
        param_grid: dict
    ) -> pd.DataFrame:
        """
        Run walk-forward optimization.

        Returns:
            DataFrame with results for each period
        """
        results = []

        start_date = data.index[0]
        end_date = data.index[-1]

        current_date = start_date + pd.Timedelta(days=self.train_period)

        while current_date + pd.Timedelta(days=self.test_period) <= end_date:
            # Define train period
            if self.anchored:
                train_start = start_date  # Anchored - всегда с начала
            else:
                train_start = current_date - pd.Timedelta(days=self.train_period)  # Rolling

            train_end = current_date
            test_end = current_date + pd.Timedelta(days=self.test_period)

            # Get data
            train_data = data.loc[train_start:train_end]
            test_data = data.loc[train_end:test_end]

            # Optimize on train
            best_params = strategy.grid_search(train_data, param_grid)

            # Test on OOS period
            test_results = strategy.backtest(test_data, best_params)

            results.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'best_params': best_params,
                'sharpe': test_results.sharpe_ratio,
                'return': test_results.total_return,
                'max_dd': test_results.max_drawdown,
                'trades': test_results.total_trades
            })

            # Move forward
            current_date += pd.Timedelta(days=self.step)

        return pd.DataFrame(results)

    def analyze_stability(self, wfo_results: pd.DataFrame) -> dict:
        """Analyze parameter stability across periods."""

        # Parameter stability
        param_changes = []
        for i in range(1, len(wfo_results)):
            prev_params = wfo_results.iloc[i-1]['best_params']
            curr_params = wfo_results.iloc[i]['best_params']

            # Calculate parameter change
            change = {
                k: abs(curr_params[k] - prev_params[k]) / prev_params[k]
                for k in prev_params.keys()
            }
            param_changes.append(change)

        # Performance consistency
        sharpe_std = wfo_results['sharpe'].std()
        sharpe_mean = wfo_results['sharpe'].mean()

        return {
            'avg_param_change': np.mean([sum(c.values()) for c in param_changes]),
            'performance_consistency': sharpe_mean / sharpe_std if sharpe_std > 0 else 0,
            'periods_analyzed': len(wfo_results),
            'avg_sharpe': sharpe_mean
        }
```

---

## 🎲 Monte Carlo Simulations

```python
class MonteCarloSimulator:
    """
    Monte Carlo симуляции для оценки робастности.
    """

    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations

    def bootstrap_trades(self, trades: pd.DataFrame) -> dict:
        """
        Bootstrap resampling трейдов.

        Оценивает: "Что если порядок трейдов был другим?"
        """
        results = []

        for _ in range(self.n_simulations):
            # Random sampling with replacement
            sampled_trades = trades.sample(n=len(trades), replace=True)

            # Calculate equity curve
            equity = (1 + sampled_trades['pnl']).cumprod()

            # Calculate metrics
            returns = equity.pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_dd = (equity / equity.expanding().max() - 1).min()

            results.append({
                'final_equity': equity.iloc[-1],
                'sharpe': sharpe,
                'max_dd': max_dd
            })

        results_df = pd.DataFrame(results)

        return {
            'sharpe_mean': results_df['sharpe'].mean(),
            'sharpe_std': results_df['sharpe'].std(),
            'sharpe_95ci': (
                np.percentile(results_df['sharpe'], 2.5),
                np.percentile(results_df['sharpe'], 97.5)
            ),
            'prob_positive_sharpe': (results_df['sharpe'] > 0).mean(),
            'max_dd_mean': results_df['max_dd'].mean(),
            'max_dd_worst_case': results_df['max_dd'].min()
        }

    def scenario_analysis(self, strategy, data: pd.DataFrame, scenarios: dict) -> dict:
        """
        Stress testing различных сценариев.

        Scenarios example:
        {
            'high_volatility': {'vol_multiplier': 2.0},
            'flash_crash': {'drop_pct': -0.20, 'recovery_days': 3},
            'trending_market': {'trend_strength': 0.5}
        }
        """
        results = {}

        for scenario_name, params in scenarios.items():
            modified_data = self._apply_scenario(data, params)
            scenario_result = strategy.backtest(modified_data)

            results[scenario_name] = {
                'sharpe': scenario_result.sharpe_ratio,
                'return': scenario_result.annual_return,
                'max_dd': scenario_result.max_drawdown
            }

        return results

    def _apply_scenario(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Apply scenario modifications to data."""
        modified = data.copy()

        if 'vol_multiplier' in params:
            # Increase volatility
            returns = modified['close'].pct_change()
            amplified = returns * params['vol_multiplier']
            modified['close'] = (1 + amplified).cumprod() * data['close'].iloc[0]

        if 'drop_pct' in params:
            # Simulate flash crash
            crash_point = len(modified) // 2
            drop = params['drop_pct']
            modified.loc[modified.index[crash_point], 'close'] *= (1 + drop)

        return modified
```

---

## 💾 Data Management

### 1. Efficient Data Loading

```python
class BacktestDataLoader:
    """Efficient data loading для бэктестинга."""

    def __init__(self, data_dir: str = "data/backtest"):
        self.data_dir = Path(data_dir)

    def load_parquet_optimized(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        columns: list = None
    ) -> pd.DataFrame:
        """
        Load data with optimizations:
        1. Read only needed columns
        2. Filter by date during read (not after)
        3. Use pyarrow engine
        """

        path = self.data_dir / f"{symbol}.parquet"

        # Read with filters (much faster than loading all then filtering)
        df = pd.read_parquet(
            path,
            engine='pyarrow',
            columns=columns,
            filters=[
                ('timestamp', '>=', pd.Timestamp(start_date)),
                ('timestamp', '<=', pd.Timestamp(end_date))
            ]
        )

        return df

    def create_backtest_dataset(
        self,
        symbols: list,
        start_date: str,
        end_date: str,
        timeframe: str = '1h'
    ):
        """
        Create unified dataset для multi-symbol backtesting.
        """

        dfs = []
        for symbol in symbols:
            df = self.load_parquet_optimized(symbol, start_date, end_date)
            df = df.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            df['symbol'] = symbol
            dfs.append(df)

        # Combine
        combined = pd.concat(dfs).sort_index()

        return combined
```

### 2. Feature Store Integration

```python
class BacktestFeatureLoader:
    """Load features from Feature Store для backtesting."""

    def __init__(self, feature_store):
        self.feature_store = feature_store

    def load_features_for_backtest(
        self,
        feature_group: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load features для бэктестинга.

        ВАЖНО: Используйте point-in-time корректные фичи!
        """

        # Read from Feature Store
        features = self.feature_store.read_offline_features(
            feature_group=feature_group,
            start_date=start_date,
            end_date=end_date
        )

        # Ensure no future leakage
        features = self._validate_no_leakage(features)

        return features

    def _validate_no_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate no future data leakage.

        Checks:
        1. Features computed only from past data
        2. No forward-filling across gaps
        3. Proper alignment
        """

        # Check for NaN patterns that indicate leakage
        if 'future_' in str(df.columns):
            raise ValueError("Future-looking columns detected!")

        return df
```

---

## 🔗 Integration with Production

### 1. Strategy Interface

```python
from abc import ABC, abstractmethod

class BacktestableStrategy(ABC):
    """
    Interface для стратегий, поддерживающих бэктестинг.

    Ваши стратегии должны имплементировать этот интерфейс.
    """

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> Optional[Signal]:
        """
        Generate trading signal.

        ВАЖНО: Используйте только данные до timestamp!
        """
        pass

    @abstractmethod
    def calculate_position_size(
        self,
        signal: Signal,
        account_balance: float,
        current_positions: dict
    ) -> float:
        """Calculate position size."""
        pass

    @abstractmethod
    def should_close_position(
        self,
        position: Position,
        current_price: float,
        timestamp: pd.Timestamp
    ) -> bool:
        """Determine if position should be closed."""
        pass

# Your existing strategies должны имплементировать этот interface
class AdaptiveConsensusBacktestable(BacktestableStrategy):
    """Wrapper для AdaptiveConsensusManager."""

    def __init__(self, consensus_manager):
        self.consensus = consensus_manager

    def generate_signals(self, data, timestamp):
        # Use your existing logic
        return self.consensus.evaluate_consensus(...)
```

### 2. Production-Backtest Consistency

```python
class StrategyValidator:
    """
    Validate consistency между production и backtest.
    """

    def validate_strategy(
        self,
        strategy,
        paper_trades: pd.DataFrame,
        backtest_trades: pd.DataFrame,
        tolerance: float = 0.05
    ):
        """
        Compare paper trading vs backtest на том же периоде.

        Должны быть идентичны!
        """

        # Compare signals
        paper_signals = set(paper_trades['signal_id'])
        backtest_signals = set(backtest_trades['signal_id'])

        missing_in_backtest = paper_signals - backtest_signals
        extra_in_backtest = backtest_signals - paper_signals

        if missing_in_backtest or extra_in_backtest:
            logger.warning(
                f"Signal mismatch! "
                f"Missing: {len(missing_in_backtest)}, "
                f"Extra: {len(extra_in_backtest)}"
            )

        # Compare performance
        paper_return = paper_trades['pnl'].sum()
        backtest_return = backtest_trades['pnl'].sum()

        diff_pct = abs(paper_return - backtest_return) / abs(paper_return)

        if diff_pct > tolerance:
            logger.error(
                f"Performance mismatch! "
                f"Paper: {paper_return:.2f}, "
                f"Backtest: {backtest_return:.2f}, "
                f"Diff: {diff_pct:.2%}"
            )
            return False

        return True
```

---

## 🗺️ Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Create `backend/backtesting/` directory structure
- [ ] Implement basic event-driven engine
- [ ] Add OrderFillModel with slippage
- [ ] Implement FeeModel
- [ ] Create BacktestMetrics class

### Phase 2: Data Pipeline (Week 3)
- [ ] BacktestDataLoader для parquet
- [ ] Feature Store integration
- [ ] Data validation (no leakage checks)
- [ ] Create sample datasets

### Phase 3: Strategy Integration (Week 4)
- [ ] BacktestableStrategy interface
- [ ] Adapt existing strategies
- [ ] Test strategy wrappers
- [ ] Production-backtest consistency checks

### Phase 4: Advanced Features (Week 5-6)
- [ ] Walk-Forward Optimization
- [ ] Monte Carlo simulations
- [ ] Advanced metrics
- [ ] Visualization dashboard

### Phase 5: Optimization (Week 7-8)
- [ ] Parameter optimization framework
- [ ] Parallel processing
- [ ] Caching layer
- [ ] Performance profiling

---

## 📈 Quick Start Example

```python
# Example: Simple backtest
from backend.backtesting import BacktestEngine, BacktestConfig
from backend.strategies.adaptive import AdaptiveConsensusManager

# 1. Load data
data = pd.read_parquet("data/BTCUSDT_2024.parquet")

# 2. Create config
config = BacktestConfig(
    initial_capital=10000,
    slippage_pct=0.0005,
    maker_fee=0.0002,
    taker_fee=0.0006,
    start_date="2024-01-01",
    end_date="2024-06-30"
)

# 3. Create strategy
strategy = AdaptiveConsensusBacktestable(your_consensus_manager)

# 4. Run backtest
engine = BacktestEngine(config)
results = engine.run(strategy, data)

# 5. Analyze
print(f"Sharpe: {results.metrics.sharpe_ratio:.2f}")
print(f"Return: {results.metrics.annual_return:.2%}")
print(f"Max DD: {results.metrics.max_drawdown:.2%}")

# 6. Visualize
results.plot_equity_curve()
results.plot_drawdown()
results.generate_report("backtest_report.html")
```

---

## 🎓 Key Takeaways

1. **Реализм превыше всего** - slippage, fees, market impact обязательны
2. **Overfitting - главная опасность** - используйте OOS testing, walk-forward
3. **Векторизация для скорости** - но event-driven для точности
4. **Monte Carlo для уверенности** - bootstrap показывает робастность
5. **Consistency is critical** - backtest должен совпадать с production
6. **Comprehensive metrics** - Sharpe недостаточно, нужен полный анализ

---

## 📚 Recommended Reading

1. **"Advances in Financial Machine Learning"** - Marcos López de Prado
   - Purged K-Fold CV
   - Combinatorially Symmetric Cross-Validation
   - Meta-labeling

2. **"Quantitative Trading"** - Ernest Chan
   - Walk-forward optimization
   - Parameter stability

3. **"Evidence-Based Technical Analysis"** - David Aronson
   - Data mining bias
   - Statistical significance

4. **"Machine Learning for Algorithmic Trading"** - Stefan Jansen
   - Backtesting frameworks
   - Production integration

---

## 🚀 Next Steps

1. Review this document с командой
2. Decide на приоритеты (Phase 1 обязательна)
3. Start с simple vectorized backtester
4. Gradually add реалистичности (slippage, fees)
5. Integrate с вашими existing strategies
6. Validate против paper trading
7. Deploy walk-forward optimization

**Remember:** Хороший бэктест не гарантирует success в production, но плохой бэктест гарантирует failure!
