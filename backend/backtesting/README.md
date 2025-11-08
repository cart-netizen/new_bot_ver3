# Backtesting System - Comprehensive Documentation

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tier 1 Features](#tier-1-features)
  - [Advanced Metrics](#advanced-metrics)
  - [Walk-Forward Analysis](#walk-forward-analysis)
  - [Template System](#template-system)
  - [Data Quality Checker](#data-quality-checker)
- [API Reference](#api-reference)
- [Frontend Integration](#frontend-integration)
- [Configuration Reference](#configuration-reference)
- [Examples](#examples)
- [Best Practices](#best-practices)

---

## Overview

This backtesting system provides a comprehensive framework for testing trading strategies on historical data. It includes advanced features for risk assessment, overfitting prevention, configuration management, and data validation.

### Key Capabilities

- **Event-driven architecture** - Realistic simulation of live trading
- **Advanced metrics** - 25+ risk-adjusted and consistency metrics
- **Walk-Forward Analysis** - Prevent overfitting with IS/OOS validation
- **Template System** - Save and reuse configurations
- **Data Quality** - Automatic validation before backtesting
- **ML Integration** - Optional machine learning model support
- **Order Book Simulation** - Realistic market microstructure
- **Multi-strategy consensus** - Combine multiple strategies

---

## Features

### Phase 1-3 Features (Already Implemented)

- ✅ Core backtesting engine with event-driven architecture
- ✅ Order book simulation with depth and spread modeling
- ✅ ML model integration for prediction enhancement
- ✅ Parameter optimization with grid search
- ✅ Multiple strategy support with consensus modes
- ✅ Advanced risk management (stop loss, take profit, trailing stops)
- ✅ Realistic commission and slippage models
- ✅ Trade execution with market/limit order types

### Tier 1 Features (Latest)

- ✅ **Advanced Metrics Calculator** - 25+ metrics for comprehensive analysis
- ✅ **Walk-Forward Analysis** - Robust validation methodology
- ✅ **Template System** - CRUD operations for configuration management
- ✅ **Data Quality Checker** - Pre-backtest data validation

---

## Architecture

```
backend/backtesting/
├── core/
│   ├── backtesting_engine.py      # Main engine with event loop
│   ├── portfolio.py                # Portfolio management
│   └── order_executor.py           # Order execution logic
├── metrics/
│   └── advanced_metrics.py         # Advanced metrics calculator (NEW)
├── analysis/
│   └── walk_forward.py             # Walk-Forward analysis (NEW)
├── data/
│   ├── data_provider.py            # Market data provider
│   └── quality_checker.py          # Data quality validation (NEW)
├── models/
│   └── __init__.py                 # Data models (extended)
├── strategies/
│   └── ...                         # Strategy implementations
└── infrastructure/
    └── repositories/
        └── templates/
            └── template_repository.py  # Template storage (NEW)

backend/api/
└── templates_api.py                # REST API for templates (NEW)

frontend/src/
├── components/backtesting/
│   ├── AdvancedMetricsGrid.tsx     # UI for advanced metrics (NEW)
│   └── TemplateLibrary.tsx         # UI for template management (NEW)
└── api/
    └── templates.api.ts            # Frontend API client (NEW)
```

---

## Installation

### Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start the backend server
uvicorn backend.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## Quick Start

### 1. Create a Backtest via API

```python
import requests

config = {
    "name": "My First Backtest",
    "description": "Testing momentum strategy",
    "symbol": "BTCUSDT",
    "start_date": "2024-01-01T00:00:00",
    "end_date": "2024-01-31T23:59:59",
    "initial_capital": 10000,
    "candle_interval": "1h",
    "enabled_strategies": ["momentum", "supertrend"],
    "consensus_mode": "weighted",
    "position_size_pct": 10.0,
    "stop_loss_pct": 2.0,
    "take_profit_pct": 4.0
}

response = requests.post("http://localhost:8000/api/backtesting/run", json=config)
backtest_id = response.json()["backtest_id"]

# Get results
results = requests.get(f"http://localhost:8000/api/backtesting/{backtest_id}?include_trades=true")
print(results.json())
```

### 2. Use the Frontend UI

1. Navigate to `http://localhost:3000/backtesting`
2. Click "Создать тест"
3. Configure your backtest settings
4. Click "Запустить бэктест"
5. View results in the "Расширенные метрики" tab

---

## Tier 1 Features

### Advanced Metrics

The Advanced Metrics Calculator provides 25+ metrics beyond basic P&L analysis.

#### Available Metrics

**Risk-Adjusted Returns:**
- **Sortino Ratio** - Like Sharpe, but only penalizes downside volatility
- **Calmar Ratio** - Return / Max Drawdown ratio
- **Omega Ratio** - Probability-weighted ratio of gains vs losses
- **Skewness** - Asymmetry of return distribution
- **Kurtosis** - Tail risk indicator
- **Tail Ratio** - Right tail (95%) / Left tail (5%)

**Consistency Metrics:**
- **Expectancy** - Average expected profit per trade
- **Kelly Criterion** - Optimal position size percentage
- **Profit Factor** - Gross profit / Gross loss
- **Monthly Win Rate** - Percentage of profitable months
- **Win/Loss Ratio** - Average win / Average loss
- **Consecutive Wins/Losses** - Maximum streaks

**Drawdown Analysis:**
- **Average Drawdown** - Mean of all drawdowns
- **Recovery Factor** - Net profit / Max drawdown
- **Ulcer Index** - Depth and duration of drawdowns
- **Average Drawdown Duration** - Time to recover

**Market Exposure:**
- **Market Exposure %** - Percentage of time in market
- **Average Trade Duration** - Mean time in positions

#### Usage Example

```python
from backend.backtesting.metrics.advanced_metrics import AdvancedMetricsCalculator, TradeData, EquityData

# Prepare trade data
trade_data = [
    TradeData(
        entry_time=datetime(2024, 1, 1, 10, 0),
        exit_time=datetime(2024, 1, 1, 12, 0),
        pnl=150.0,
        pnl_pct=1.5,
        duration_seconds=7200
    ),
    # ... more trades
]

# Prepare equity curve data
equity_data = [
    EquityData(
        timestamp=datetime(2024, 1, 1, 0, 0),
        equity=10000,
        drawdown=0,
        drawdown_pct=0,
        total_return_pct=0
    ),
    # ... more points
]

# Calculate metrics
calculator = AdvancedMetricsCalculator(risk_free_rate=0.02)
metrics = calculator.calculate(
    trades=trade_data,
    equity_curve=equity_data,
    initial_capital=10000,
    final_capital=12000,
    total_pnl=2000,
    max_drawdown=500,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)

print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
print(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")
print(f"Kelly Criterion: {metrics.kelly_criterion:.2%}")
print(f"Expectancy: ${metrics.expectancy:.2f}")
```

#### Frontend Display

The AdvancedMetricsGrid component displays metrics in 4 organized tabs:

```typescript
import { AdvancedMetricsGrid } from '../components/backtesting/AdvancedMetricsGrid';

<AdvancedMetricsGrid metrics={backtestResults.metrics} />
```

---

### Walk-Forward Analysis

Walk-Forward Analysis prevents overfitting by dividing data into In-Sample (IS) and Out-of-Sample (OOS) periods.

#### How It Works

1. **Window Generation** - Split data into IS/OOS windows
2. **Optimization** - Optimize parameters on IS data
3. **Validation** - Test on OOS data
4. **Iteration** - Move window forward and repeat
5. **Aggregation** - Analyze IS vs OOS performance degradation

#### Configuration

```python
from backend.backtesting.analysis.walk_forward import WalkForwardConfig, WalkForwardAnalyzer

config = WalkForwardConfig(
    in_sample_days=180,          # 6 months for optimization
    out_of_sample_days=60,       # 2 months for validation
    reoptimize_every_days=30,    # Re-optimize every month
    anchor_mode="rolling",       # "rolling" or "expanding"
    optimization_metric="sharpe_ratio",
    param_ranges={
        "stop_loss_pct": [1.0, 2.0, 3.0],
        "take_profit_pct": [2.0, 4.0, 6.0]
    }
)

analyzer = WalkForwardAnalyzer(config)
```

#### Usage Example

```python
# Generate windows
windows = analyzer.generate_windows(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1)
)

# Run backtests for each window (IS and OOS)
for window in windows:
    # Run on IS period
    is_result = run_backtest(
        start=window.is_start,
        end=window.is_end,
        params=optimal_params  # from optimization
    )
    window.is_metrics = is_result.metrics

    # Run on OOS period
    oos_result = run_backtest(
        start=window.oos_start,
        end=window.oos_end,
        params=is_result.optimal_params
    )
    window.oos_metrics = oos_result.metrics

# Analyze results
wfa_result = analyzer.analyze_results(windows)

print(f"Average IS Sharpe: {wfa_result.avg_is_sharpe:.2f}")
print(f"Average OOS Sharpe: {wfa_result.avg_oos_sharpe:.2f}")
print(f"Degradation: {wfa_result.is_oos_degradation_pct:.1f}%")
print(f"OOS Win Rate: {wfa_result.oos_win_rate:.1f}%")
print(f"Is Robust: {wfa_result.is_strategy_robust}")

for rec in wfa_result.recommendations:
    print(f"- {rec}")
```

#### Interpretation

- **Degradation < 10%**: Excellent - strategy generalizes well
- **Degradation 10-30%**: Acceptable - some overfitting present
- **Degradation 30-50%**: Warning - significant overfitting
- **Degradation > 50%**: Critical - likely overfitted to historical data

- **OOS Win Rate > 70%**: High consistency
- **OOS Win Rate 40-70%**: Moderate consistency
- **OOS Win Rate < 40%**: Low consistency

---

### Template System

Save and reuse backtest configurations with the template system.

#### API Endpoints

```
POST   /api/backtesting/templates          - Create template
GET    /api/backtesting/templates          - List templates
GET    /api/backtesting/templates/{id}     - Get template
PUT    /api/backtesting/templates/{id}     - Update template
DELETE /api/backtesting/templates/{id}     - Delete template
```

#### Creating a Template

```python
import requests

template_data = {
    "name": "Conservative Momentum",
    "description": "Low-risk momentum strategy with tight stops",
    "config": {
        "enabled_strategies": ["momentum"],
        "position_size_pct": 5.0,
        "stop_loss_pct": 1.0,
        "take_profit_pct": 2.0,
        "consensus_mode": "majority",
        "min_strategies_for_signal": 1
    },
    "tags": ["conservative", "momentum", "low-risk"],
    "is_public": True
}

response = requests.post(
    "http://localhost:8000/api/backtesting/templates",
    json=template_data
)
template_id = response.json()["template_id"]
```

#### Loading a Template

```python
# Get template
template = requests.get(
    f"http://localhost:8000/api/backtesting/templates/{template_id}"
).json()

# Use config for backtest
backtest_config = {
    **template["config"],
    "name": "Test with Conservative Template",
    "symbol": "ETHUSDT",
    "start_date": "2024-01-01T00:00:00",
    "end_date": "2024-01-31T23:59:59",
    "initial_capital": 10000
}

response = requests.post(
    "http://localhost:8000/api/backtesting/run",
    json=backtest_config
)
```

#### Frontend Usage

```typescript
import { TemplateLibrary } from '../components/backtesting/TemplateLibrary';

function BacktestForm() {
  const [config, setConfig] = useState({});

  const handleLoadTemplate = (templateConfig: any) => {
    setConfig({ ...config, ...templateConfig });
  };

  return (
    <>
      <TemplateLibrary
        onLoadTemplate={handleLoadTemplate}
        currentConfig={config}
      />
      {/* Rest of form */}
    </>
  );
}
```

#### Features

- **Search** - Find templates by name
- **Tag Filtering** - Filter by tags (e.g., "momentum", "conservative")
- **Public/Private** - Share templates or keep private
- **Usage Tracking** - See how many times a template was used
- **One-Click Load** - Apply template to current configuration

---

### Data Quality Checker

Validates market data before backtesting to ensure reliable results.

#### Checks Performed

1. **Duplicate Detection** - Identifies duplicate timestamps
2. **Gap Detection** - Finds missing candles in time series
3. **Zero Volume** - Flags candles with no trading volume
4. **OHLC Consistency** - Validates Open/High/Low/Close relationships
5. **Price Spikes** - Detects statistical outliers (anomalies)

#### Usage Example

```python
from backend.backtesting.data.quality_checker import DataQualityChecker

checker = DataQualityChecker()

# Check data quality
report = checker.check_data_quality(
    candles=candle_list,
    expected_interval_minutes=60  # 1-hour candles
)

print(f"Total candles: {report.total_candles}")
print(f"Errors: {report.errors_count}")
print(f"Warnings: {report.warnings_count}")
print(f"Gaps: {report.gaps_count}")
print(f"Duplicates: {report.duplicates_count}")
print(f"Price spikes: {report.price_spikes_count}")
print(f"Suitable for backtest: {report.is_suitable_for_backtest}")

for rec in report.recommendations:
    print(f"- {rec}")
```

#### Example Output

```
Total candles: 720
Errors: 2
Warnings: 5
Gaps: 3
Duplicates: 0
Price spikes: 2
Suitable for backtest: True

Recommendations:
- Found 3 gaps in data (max gap: 3 hours). Consider filling gaps with interpolation.
- Found 2 price spikes (outliers). Verify data quality or filter anomalies.
- Data quality is acceptable for backtesting, but review warnings.
```

#### Integration with Backtesting

```python
# Check data quality before running backtest
data_provider = BinanceDataProvider()
candles = await data_provider.fetch_candles(
    symbol="BTCUSDT",
    interval="1h",
    start_time=start_date,
    end_time=end_date
)

checker = DataQualityChecker()
quality_report = checker.check_data_quality(candles, expected_interval_minutes=60)

if not quality_report.is_suitable_for_backtest:
    raise ValueError(f"Data quality check failed: {quality_report.recommendations}")

# Proceed with backtest
engine = BacktestingEngine(config, candles)
result = await engine.run()
```

---

## API Reference

### Backtesting Endpoints

#### `POST /api/backtesting/run`

Create and start a new backtest.

**Request Body:**
```json
{
  "name": "My Backtest",
  "description": "Optional description",
  "symbol": "BTCUSDT",
  "start_date": "2024-01-01T00:00:00",
  "end_date": "2024-01-31T23:59:59",
  "initial_capital": 10000,
  "candle_interval": "1h",
  "enabled_strategies": ["momentum", "supertrend"],
  "consensus_mode": "weighted",
  "position_size_pct": 10.0,
  "stop_loss_pct": 2.0,
  "take_profit_pct": 4.0
}
```

**Response:**
```json
{
  "success": true,
  "backtest_id": "uuid",
  "message": "Backtest started successfully"
}
```

#### `GET /api/backtesting/{backtest_id}`

Get backtest details and results.

**Query Parameters:**
- `include_trades` (bool): Include trade list
- `include_equity` (bool): Include equity curve

**Response:**
```json
{
  "id": "uuid",
  "name": "My Backtest",
  "status": "completed",
  "symbol": "BTCUSDT",
  "start_date": "2024-01-01T00:00:00",
  "end_date": "2024-01-31T23:59:59",
  "total_pnl": 1500.50,
  "total_pnl_pct": 15.05,
  "metrics": {
    "sharpe_ratio": 1.85,
    "sortino_ratio": 2.15,
    "calmar_ratio": 3.20,
    "max_drawdown": 500,
    "expectancy": 25.50,
    "win_rate": 65.5,
    "profit_factor": 2.1
  },
  "trades": [...],
  "equity_curve": [...]
}
```

#### `GET /api/backtesting/list`

List all backtests.

**Query Parameters:**
- `status` (optional): Filter by status (running, completed, failed, cancelled)
- `limit` (int): Number of results (default: 50)
- `offset` (int): Pagination offset

#### `DELETE /api/backtesting/{backtest_id}`

Delete a backtest.

#### `POST /api/backtesting/{backtest_id}/cancel`

Cancel a running backtest.

### Template Endpoints

#### `POST /api/backtesting/templates`

Create a new template.

**Request Body:**
```json
{
  "name": "Conservative Strategy",
  "description": "Low-risk approach",
  "config": { ... },
  "tags": ["conservative", "low-risk"],
  "is_public": true
}
```

#### `GET /api/backtesting/templates`

List templates.

**Query Parameters:**
- `tags` (string): Comma-separated tags for filtering

#### `GET /api/backtesting/templates/{template_id}`

Get template by ID (increments usage count).

#### `PUT /api/backtesting/templates/{template_id}`

Update template.

#### `DELETE /api/backtesting/templates/{template_id}`

Delete template.

---

## Frontend Integration

### Components

#### AdvancedMetricsGrid

Displays advanced metrics in organized tabs.

```tsx
import { AdvancedMetricsGrid } from '@/components/backtesting/AdvancedMetricsGrid';

<AdvancedMetricsGrid metrics={backtestResults.metrics} />
```

**Features:**
- 4 tabs: Risk-Adjusted, Consistency, Drawdown Analysis, Market Exposure
- Color-coded metric cards (green/red based on thresholds)
- Tooltips explaining each metric
- Responsive grid layout

#### TemplateLibrary

Manage backtest templates with full CRUD operations.

```tsx
import { TemplateLibrary } from '@/components/backtesting/TemplateLibrary';

<TemplateLibrary
  onLoadTemplate={(config) => setFormConfig(config)}
  currentConfig={formConfig}
/>
```

**Features:**
- Search by name
- Filter by tags
- Create/Edit/Delete templates
- One-click load
- Public/Private visibility

#### BacktestingSettings

Comprehensive settings form for backtest configuration.

```tsx
import { BacktestingSettings } from '@/components/backtesting/BacktestingSettings';

<BacktestingSettings
  config={config}
  onChange={setConfig}
/>
```

**Sections:**
- Basic Settings (symbol, dates, capital)
- Strategy Selection
- Consensus Configuration
- Risk Management
- Advanced Features (ML, OrderBook, Caching)

---

## Configuration Reference

### BacktestConfig

Full configuration object for backtests:

```typescript
interface BacktestConfig {
  // Basic Settings
  name: string;
  description?: string;
  symbol: string;
  start_date: string;  // ISO format
  end_date: string;    // ISO format
  initial_capital: number;
  candle_interval: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';

  // Fees
  commission_rate: number;      // Combined rate (deprecated, use maker/taker)
  maker_commission: number;     // Maker fee (default: 0.0002)
  taker_commission: number;     // Taker fee (default: 0.001)

  // Slippage
  slippage_model: 'fixed' | 'dynamic';
  slippage_pct: number;
  simulate_latency: boolean;

  // Strategies
  enabled_strategies: string[];  // e.g., ['momentum', 'supertrend']

  // Consensus
  consensus_mode: 'majority' | 'unanimous' | 'weighted' | 'any';
  min_strategies_for_signal: number;
  min_consensus_confidence: number;  // 0.0 - 1.0

  // Position Sizing
  position_size_pct: number;
  position_size_mode: 'fixed_percent' | 'kelly' | 'risk_based';
  max_open_positions: number;

  // Risk Management
  stop_loss_pct: number;
  take_profit_pct: number;
  use_trailing_stop: boolean;
  trailing_stop_activation_pct: number;
  trailing_stop_distance_pct: number;
  risk_per_trade_pct: number;

  // Advanced Features
  use_orderbook_data: boolean;
  orderbook_num_levels: number;
  orderbook_base_spread_bps: number;
  use_market_trades: boolean;
  trades_per_volume_unit: number;
  use_ml_model: boolean;
  ml_server_url: string;

  // Performance
  use_cache: boolean;
  warmup_period_bars: number;

  // Logging
  verbose: boolean;
  log_trades: boolean;
}
```

### Strategy Weights

When using `consensus_mode: "weighted"`, configure strategy weights:

```python
strategy_weights = {
    "momentum": 1.5,      # Higher weight
    "supertrend": 1.0,
    "sar_wave": 1.0,
    "volume_profile": 0.8  # Lower weight
}
```

---

## Examples

### Example 1: Basic Backtest

```python
import requests

config = {
    "name": "Simple Momentum Test",
    "symbol": "BTCUSDT",
    "start_date": "2024-01-01T00:00:00",
    "end_date": "2024-01-31T23:59:59",
    "initial_capital": 10000,
    "candle_interval": "1h",
    "enabled_strategies": ["momentum"],
    "position_size_pct": 10,
    "stop_loss_pct": 2.0,
    "take_profit_pct": 4.0
}

response = requests.post("http://localhost:8000/api/backtesting/run", json=config)
backtest_id = response.json()["backtest_id"]

# Wait for completion
import time
while True:
    result = requests.get(f"http://localhost:8000/api/backtesting/{backtest_id}").json()
    if result["status"] == "completed":
        break
    time.sleep(2)

print(f"Total P&L: ${result['total_pnl']:.2f}")
print(f"Sharpe Ratio: {result['metrics']['sharpe_ratio']:.2f}")
```

### Example 2: Multi-Strategy with Consensus

```python
config = {
    "name": "Multi-Strategy Consensus",
    "symbol": "ETHUSDT",
    "start_date": "2024-01-01T00:00:00",
    "end_date": "2024-03-31T23:59:59",
    "initial_capital": 50000,
    "candle_interval": "4h",
    "enabled_strategies": ["momentum", "supertrend", "sar_wave", "volume_profile"],
    "consensus_mode": "weighted",
    "min_strategies_for_signal": 2,
    "min_consensus_confidence": 0.6,
    "position_size_pct": 15,
    "stop_loss_pct": 3.0,
    "take_profit_pct": 6.0,
    "use_trailing_stop": True,
    "trailing_stop_activation_pct": 3.0,
    "trailing_stop_distance_pct": 1.5
}
```

### Example 3: Advanced Features (ML + OrderBook)

```python
config = {
    "name": "ML-Enhanced OrderBook Test",
    "symbol": "BTCUSDT",
    "start_date": "2024-01-01T00:00:00",
    "end_date": "2024-01-31T23:59:59",
    "initial_capital": 100000,
    "candle_interval": "1m",
    "enabled_strategies": ["momentum", "supertrend"],
    "consensus_mode": "majority",

    # ML Model
    "use_ml_model": True,
    "ml_server_url": "http://localhost:8001",

    # OrderBook Simulation
    "use_orderbook_data": True,
    "orderbook_num_levels": 20,
    "orderbook_base_spread_bps": 2.0,

    # Market Trades
    "use_market_trades": True,
    "trades_per_volume_unit": 100,

    # Risk Management
    "position_size_mode": "kelly",
    "max_open_positions": 5,
    "risk_per_trade_pct": 1.0
}
```

### Example 4: Template Workflow

```python
# Step 1: Create template
template = {
    "name": "Aggressive Scalping",
    "description": "High-frequency scalping with tight risk",
    "config": {
        "candle_interval": "1m",
        "enabled_strategies": ["momentum"],
        "position_size_pct": 20,
        "stop_loss_pct": 0.5,
        "take_profit_pct": 1.0,
        "max_open_positions": 10
    },
    "tags": ["scalping", "aggressive", "high-frequency"],
    "is_public": False
}

# Save template
response = requests.post("http://localhost:8000/api/backtesting/templates", json=template)
template_id = response.json()["template_id"]

# Step 2: Use template for multiple backtests
for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
    # Load template
    template_config = requests.get(
        f"http://localhost:8000/api/backtesting/templates/{template_id}"
    ).json()["config"]

    # Customize for each symbol
    backtest_config = {
        **template_config,
        "name": f"Scalping Test - {symbol}",
        "symbol": symbol,
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2024-01-31T23:59:59",
        "initial_capital": 10000
    }

    # Run backtest
    requests.post("http://localhost:8000/api/backtesting/run", json=backtest_config)
```

### Example 5: Walk-Forward Analysis Workflow

```python
from backend.backtesting.analysis.walk_forward import WalkForwardConfig, WalkForwardAnalyzer
from backend.backtesting.core.backtesting_engine import BacktestingEngine

# Configure WFA
wfa_config = WalkForwardConfig(
    in_sample_days=90,
    out_of_sample_days=30,
    reoptimize_every_days=30,
    anchor_mode="rolling",
    optimization_metric="sharpe_ratio"
)

analyzer = WalkForwardAnalyzer(wfa_config)

# Generate windows
windows = analyzer.generate_windows(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1)
)

# Run backtests for each window
for window in windows:
    # In-Sample (optimization)
    is_engine = BacktestingEngine(
        config={
            **base_config,
            "start_date": window.is_start,
            "end_date": window.is_end
        },
        candles=get_candles(window.is_start, window.is_end)
    )
    is_result = await is_engine.run()
    window.is_metrics = {
        "sharpe_ratio": is_result.metrics.sharpe_ratio,
        "total_return_pct": is_result.metrics.total_return_pct
    }

    # Out-of-Sample (validation)
    oos_engine = BacktestingEngine(
        config={
            **base_config,
            "start_date": window.oos_start,
            "end_date": window.oos_end
        },
        candles=get_candles(window.oos_start, window.oos_end)
    )
    oos_result = await oos_engine.run()
    window.oos_metrics = {
        "sharpe_ratio": oos_result.metrics.sharpe_ratio,
        "total_return_pct": oos_result.metrics.total_return_pct
    }

# Analyze WFA results
wfa_result = analyzer.analyze_results(windows)

print(f"Strategy Robust: {wfa_result.is_strategy_robust}")
print(f"IS→OOS Degradation: {wfa_result.is_oos_degradation_pct:.1f}%")
for rec in wfa_result.recommendations:
    print(f"- {rec}")
```

---

## Best Practices

### 1. Data Quality First

Always validate data before backtesting:

```python
from backend.backtesting.data.quality_checker import DataQualityChecker

checker = DataQualityChecker()
report = checker.check_data_quality(candles, expected_interval_minutes=60)

if not report.is_suitable_for_backtest:
    # Fix data issues or use different data source
    raise ValueError("Data quality insufficient")
```

### 2. Use Walk-Forward Analysis

Prevent overfitting by validating on OOS data:

```python
# Don't rely on single backtest
# Do use WFA to ensure robustness
if wfa_result.is_oos_degradation_pct > 30:
    print("Warning: Significant overfitting detected")
```

### 3. Conservative Position Sizing

Start with conservative position sizes:

```python
config = {
    "position_size_pct": 5.0,  # Start small
    "max_open_positions": 3,
    "risk_per_trade_pct": 1.0
}
```

### 4. Realistic Costs

Include realistic commission and slippage:

```python
config = {
    "maker_commission": 0.0002,  # 0.02%
    "taker_commission": 0.001,   # 0.1%
    "slippage_model": "dynamic",
    "slippage_pct": 0.1
}
```

### 5. Multiple Time Periods

Test across different market conditions:

```python
periods = [
    ("2023-01-01", "2023-06-30"),  # Bull market
    ("2023-07-01", "2023-12-31"),  # Bear market
    ("2024-01-01", "2024-06-30")   # Sideways
]

for start, end in periods:
    # Run backtest for each period
    ...
```

### 6. Use Templates for Consistency

Save proven configurations as templates:

```python
# Save successful configuration
if backtest_result.metrics.sharpe_ratio > 2.0:
    template = {
        "name": "High Sharpe Momentum",
        "config": backtest_config,
        "tags": ["proven", "high-sharpe"]
    }
    save_template(template)
```

### 7. Monitor Advanced Metrics

Don't rely solely on P&L:

```python
# Check multiple metrics
if (metrics.sortino_ratio > 1.5 and
    metrics.calmar_ratio > 2.0 and
    metrics.profit_factor > 1.5 and
    metrics.monthly_win_rate > 60):
    print("Strategy shows strong performance")
```

### 8. Incremental Testing

Start simple, add complexity gradually:

```python
# Phase 1: Single strategy
config_v1 = {"enabled_strategies": ["momentum"]}

# Phase 2: Add consensus
config_v2 = {
    "enabled_strategies": ["momentum", "supertrend"],
    "consensus_mode": "majority"
}

# Phase 3: Add advanced features
config_v3 = {
    **config_v2,
    "use_ml_model": True,
    "use_orderbook_data": True
}
```

### 9. Document Assumptions

Always document what you're testing:

```python
config = {
    "name": "Momentum Strategy - Bull Market Test",
    "description": (
        "Testing momentum strategy in bull market conditions (Jan-Jun 2024). "
        "Assumptions: 10bps slippage, 0.1% taker fee, no latency simulation."
    ),
    # ... rest of config
}
```

### 10. Use Version Control for Configs

Save configurations with version control:

```python
# Save config to file
import json
with open(f"configs/backtest_v1.json", "w") as f:
    json.dump(config, f, indent=2)

# Commit to git
# git add configs/backtest_v1.json
# git commit -m "Add momentum strategy config v1"
```

---

## Troubleshooting

### Issue: Data Quality Errors

**Symptom:** Backtest fails with "Data quality check failed"

**Solution:**
```python
# Check the report
report = checker.check_data_quality(candles, 60)
print(report.recommendations)

# Fix common issues:
# 1. Fill gaps with interpolation
# 2. Remove duplicates
# 3. Filter price spikes
```

### Issue: Low Sharpe Ratio on OOS

**Symptom:** Good IS performance, poor OOS performance

**Solution:**
- Reduce parameter complexity
- Increase IS sample size
- Add regularization
- Simplify strategy logic

### Issue: Template Not Loading

**Symptom:** Template library shows no templates

**Solution:**
```bash
# Check database connection
# Check if templates exist
curl http://localhost:8000/api/backtesting/templates

# Create sample template
curl -X POST http://localhost:8000/api/backtesting/templates \
  -H "Content-Type: application/json" \
  -d '{"name":"Test","config":{},"tags":[]}'
```

---

## Performance Optimization

### 1. Enable Caching

```python
config = {
    "use_cache": True,
    "warmup_period_bars": 100
}
```

### 2. Reduce Candle Resolution

Use larger intervals for faster backtests:

```python
# Faster: 1h candles
config = {"candle_interval": "1h"}

# Slower but more accurate: 1m candles
config = {"candle_interval": "1m"}
```

### 3. Disable Verbose Logging

```python
config = {
    "verbose": False,
    "log_trades": False
}
```

### 4. Limit Strategy Count

```python
# Faster
config = {"enabled_strategies": ["momentum"]}

# Slower
config = {"enabled_strategies": ["momentum", "supertrend", "sar_wave", "volume_profile"]}
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Add tests for new features
2. Update documentation
3. Follow existing code style
4. Create detailed commit messages

---

## License

[Your License Here]

---

## Support

For issues and questions:
- GitHub Issues: [Your Repo]
- Documentation: [Your Docs]
- Email: [Your Email]

---

**Last Updated:** 2024-11-08
**Version:** 1.0.0 (Tier 1 Complete)
