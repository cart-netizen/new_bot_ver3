# 🎯 Backtesting Framework

Professional event-driven backtesting framework для тестирования торговых стратегий.

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install pandas numpy matplotlib scikit-learn
```

### Basic Usage

```python
from backend.backtesting import BacktestEngine, BacktestConfig

# 1. Load data
data = pd.read_parquet("data/BTCUSDT_2024.parquet")

# 2. Configure backtest
config = BacktestConfig(
    initial_capital=10000,
    slippage_pct=0.0005,
    maker_fee=0.0002,
    taker_fee=0.0006
)

# 3. Create strategy (your strategy class)
strategy = YourStrategy()

# 4. Run backtest
engine = BacktestEngine(config)
results = engine.run(strategy, data)

# 5. Analyze results
print(results.summary())
results.plot_equity_curve()
```

## 📁 Framework Structure

```
backend/backtesting/
├── __init__.py           # Public API
├── engine.py             # Core backtesting engine
├── metrics.py            # Performance metrics
├── execution.py          # Order execution simulation (TODO)
├── optimization.py       # Parameter optimization (TODO)
├── simulation.py         # Monte Carlo simulations (TODO)
├── data.py              # Data loaders (TODO)
├── validation.py        # OOS validation (TODO)
└── README.md            # This file
```

## 🎓 Key Features

### ✅ Implemented

1. **Event-Driven Engine**
   - Realistic bar-by-bar simulation
   - No look-ahead bias
   - Proper order execution flow

2. **Realistic Execution**
   - Slippage modeling
   - Transaction fees (maker/taker)
   - Market impact simulation

3. **Risk Management**
   - Stop loss / take profit
   - Daily loss killer
   - Position sizing

4. **Comprehensive Metrics**
   - Returns: Total, Annual, Monthly
   - Risk: Volatility, Sharpe, Sortino, Calmar
   - Drawdown: Max, Average, Duration
   - Trade stats: Win rate, Profit factor, Expectancy
   - Risk-adjusted: VaR, CVaR, Omega ratio
   - Statistical: Skewness, Kurtosis

5. **Trade Analysis**
   - Individual trade breakdown
   - Top winners/losers
   - Exit reason analysis

### 🚧 TODO (Phase 2+)

1. **Advanced Execution** (`execution.py`)
   - Orderbook-level simulation
   - Partial fills
   - Limit order logic

2. **Optimization** (`optimization.py`)
   - Grid search
   - Bayesian optimization
   - Genetic algorithms
   - Walk-forward optimization

3. **Monte Carlo** (`simulation.py`)
   - Bootstrap resampling
   - Scenario analysis
   - Stress testing

4. **Data Management** (`data.py`)
   - Efficient data loading
   - Feature Store integration
   - Multi-symbol handling

5. **Validation** (`validation.py`)
   - OOS testing
   - Cross-validation
   - Production consistency checks

## 📊 Metrics Explanation

### Returns Metrics

- **Total Return**: Overall % gain/loss
- **Annual Return**: Annualized return
- **Monthly Returns**: Month-by-month breakdown

### Risk Metrics

- **Volatility**: Standard deviation of returns (annualized)
- **Sharpe Ratio**: Risk-adjusted return = (Return - RiskFree) / Volatility
  - > 1.0 = Good
  - > 2.0 = Very Good
  - > 3.0 = Excellent

- **Sortino Ratio**: Like Sharpe, but only penalizes downside volatility
  - Better measure for asymmetric strategies

- **Calmar Ratio**: Return / Max Drawdown
  - Measures return per unit of drawdown risk

### Drawdown Metrics

- **Max Drawdown**: Largest peak-to-trough decline
- **Avg Drawdown**: Average of all drawdown periods
- **Drawdown Duration**: Days in drawdown

### Trade Statistics

- **Win Rate**: % of profitable trades
- **Profit Factor**: Gross Profit / Gross Loss
  - > 1.0 = Profitable
  - > 2.0 = Strong
  - > 3.0 = Excellent

- **Expectancy**: Average $ profit per trade
  - Accounts for win rate and avg win/loss

### Risk-Adjusted Metrics

- **VaR (95%)**: Value at Risk - worst expected loss at 95% confidence
- **CVaR (95%)**: Conditional VaR - average loss beyond VaR
- **Omega Ratio**: Probability-weighted gains / losses
  - > 1.0 = More gains than losses

### Statistical Metrics

- **Skewness**: Asymmetry of returns distribution
  - Positive = More upside outliers (good!)
  - Negative = More downside outliers (bad!)

- **Kurtosis**: "Fat tails" measure
  - > 3 = More outliers than normal distribution

## 🎯 Strategy Interface

Your strategy must implement:

```python
class YourStrategy:
    def generate_signals(self, bar: pd.Series, timestamp: pd.Timestamp):
        """
        Generate trading signal.

        Args:
            bar: Current bar with OHLCV + indicators
            timestamp: Current timestamp

        Returns:
            Signal object with:
            - action: 'buy', 'sell', or 'close'
            - stop_loss: Optional stop loss price
            - take_profit: Optional take profit price

        IMPORTANT: Only use data up to timestamp!
        No future peeking!
        """
        # Your logic here
        if some_condition:
            return Signal(
                action='buy',
                stop_loss=price * 0.98,
                take_profit=price * 1.05
            )
        return None
```

## 📈 Example Strategies

See `example_backtest_simple.py` for working examples:

1. **Simple Moving Average Cross**
   - Fast SMA crosses Slow SMA
   - Stop loss: 2%
   - Take profit: 5%

2. **RSI Mean Reversion** (TODO)
3. **Bollinger Band Breakout** (TODO)
4. **Multi-Timeframe Trend** (TODO)

## ⚠️ Important Caveats

### Look-Ahead Bias

**CRITICAL:** Ensure your strategy only uses data available at `timestamp`!

```python
# ❌ BAD - uses future data!
def generate_signals(self, bar, timestamp):
    future_price = self.data.loc[timestamp + timedelta(hours=1), 'close']
    if bar['close'] < future_price:
        return 'buy'

# ✅ GOOD - only past data
def generate_signals(self, bar, timestamp):
    sma = self.data.loc[:timestamp, 'close'].rolling(20).mean().iloc[-1]
    if bar['close'] > sma:
        return 'buy'
```

### Overfitting

- **Don't**: Optimize on all data, then claim success
- **Do**: Use train/validation/test split
- **Do**: Walk-forward optimization
- **Do**: Out-of-sample testing

### Survivorship Bias

- **Don't**: Only test on current top symbols
- **Do**: Include delisted/failed symbols
- **Do**: Test on all symbols from historical universe

### Transaction Costs

- **Always** include slippage and fees
- **Default settings** are realistic for Bybit
- Adjust if you have VIP status or different exchange

## 🔧 Configuration Options

```python
BacktestConfig(
    # Capital
    initial_capital=10000.0,
    max_position_size_pct=0.95,

    # Execution
    slippage_pct=0.0005,  # 0.05%
    slippage_vol_impact=0.0001,  # Per 1 BTC

    # Fees
    maker_fee=0.0002,  # 0.02%
    taker_fee=0.0006,  # 0.06%

    # Risk
    max_daily_loss_pct=0.15,  # 15%
    enable_daily_loss_killer=True,

    # Fills
    allow_partial_fills=True,
    max_fill_ratio=0.3  # 30% of level
)
```

## 📚 Next Steps

1. ✅ Run `example_backtest_simple.py` to see it in action
2. 📖 Read `BACKTESTING_BEST_PRACTICES.md` for comprehensive guide
3. 🔨 Adapt your existing strategies to BacktestableStrategy interface
4. 🧪 Run backtests on historical data
5. 📊 Analyze metrics and optimize
6. ✅ Validate with walk-forward and OOS testing
7. 🚀 Deploy to production

## 🆘 Troubleshooting

### No trades executed

- Check if strategy generates signals: add logging
- Verify data has required indicators
- Check capital is sufficient for position sizing

### Performance doesn't match production

- Verify slippage/fees match real costs
- Check for look-ahead bias
- Ensure same data preprocessing
- Compare signals between backtest and paper trading

### Backtest too slow

- Use vectorized operations where possible
- Reduce data frequency (e.g., 1H instead of 1m)
- Profile code to find bottlenecks
- Consider parallel processing (Phase 2)

## 💡 Tips

1. **Start simple** - Test with basic strategy first
2. **Validate data** - Ensure OHLC is correct, no gaps
3. **Log everything** - Add logging to understand what's happening
4. **Compare to B&H** - Your strategy should beat buy-and-hold
5. **Multiple timeframes** - Test on different periods
6. **Multiple symbols** - Ensure strategy generalizes
7. **Document assumptions** - Write down what you're testing

## 📞 Support

- See `BACKTESTING_BEST_PRACTICES.md` for detailed guide
- Check `example_backtest_simple.py` for working code
- Refer to inline code documentation
- Review metrics.py for metric formulas

---

**Remember:** Good backtest ≠ guaranteed profit, but bad backtest = guaranteed loss!
