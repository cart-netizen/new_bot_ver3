"""
Модели данных для бэктестинга.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


# ==================== ENUMS ====================

class SlippageModel(str, Enum):
    """Модель проскальзывания."""
    FIXED = "fixed"  # Фиксированное проскальзывание
    VOLUME_BASED = "volume_based"  # На основе объема в стакане
    PERCENTAGE = "percentage"  # Процент от цены


# ==================== CONFIG DATACLASSES ====================

@dataclass
class ExchangeConfig:
    """Конфигурация симуляции биржи."""
    # Комиссии
    commission_rate: float = 0.001  # 0.1% (maker/taker одинаковые)
    maker_commission: float = 0.0002  # 0.02% maker
    taker_commission: float = 0.0006  # 0.06% taker

    # Slippage
    slippage_model: SlippageModel = SlippageModel.FIXED
    slippage_pct: float = 0.01  # 0.01% фиксированное проскальзывание

    # Симуляция задержки
    simulate_latency: bool = False
    latency_mean_ms: float = 50.0
    latency_std_ms: float = 20.0

    # Вероятность отклонения ордера
    order_reject_probability: float = 0.0  # 0% - все ордера успешны


@dataclass
class StrategyConfig:
    """Конфигурация стратегий для бэктеста."""
    # Включенные стратегии
    enabled_strategies: List[str] = field(default_factory=lambda: [
        'momentum', 'sar_wave', 'supertrend', 'volume_profile'
    ])

    # Параметры стратегий
    strategy_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Пример: {'momentum': {'period': 14, 'threshold': 0.7}}

    # Consensus настройки
    consensus_mode: str = "weighted"  # weighted, majority, unanimous
    min_strategies_for_signal: int = 2
    min_consensus_confidence: float = 0.6

    # Веса стратегий (для weighted mode)
    strategy_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class RiskConfig:
    """Конфигурация risk management для бэктеста."""
    # Размер позиции
    position_size_pct: float = 10.0  # 10% капитала на позицию
    position_size_mode: str = "percentage"  # percentage, fixed, kelly

    # Лимиты
    max_open_positions: int = 3
    max_position_size_usdt: Optional[float] = None

    # Stop Loss / Take Profit
    stop_loss_pct: float = 2.0  # 2% stop loss
    take_profit_pct: float = 4.0  # 4% take profit
    use_trailing_stop: bool = True
    trailing_stop_activation_pct: float = 1.0  # Активация после 1% прибыли
    trailing_stop_distance_pct: float = 0.5  # Trailing на 0.5% от пика

    # Риск на сделку
    risk_per_trade_pct: float = 1.0  # Максимум 1% капитала на сделку


@dataclass
class BacktestConfig:
    """Главная конфигурация бэктеста."""
    # Основные параметры
    name: str
    description: Optional[str] = None
    symbol: str = "BTCUSDT"
    start_date: datetime = None
    end_date: datetime = None
    initial_capital: float = 10000.0

    # Конфигурации компонентов
    exchange_config: ExchangeConfig = field(default_factory=ExchangeConfig)
    strategy_config: StrategyConfig = field(default_factory=StrategyConfig)
    risk_config: RiskConfig = field(default_factory=RiskConfig)

    # Параметры данных
    candle_interval: str = "1m"  # 1m, 5m, 15m, 1h
    use_orderbook_data: bool = True
    orderbook_sampling_interval_ms: int = 500  # Каждые 500ms

    # Оптимизации
    warmup_period_bars: int = 100  # Количество свечей для прогрева индикаторов
    save_equity_interval_minutes: int = 60  # Сохранять equity каждые 60 минут

    # Отладка
    verbose: bool = False
    log_trades: bool = True


# ==================== RESULT DATACLASSES ====================

@dataclass
class TradeResult:
    """Результат одной сделки."""
    symbol: str
    side: str  # "Buy" or "Sell"
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    duration_seconds: float
    exit_reason: str  # "TP", "SL", "SIGNAL", "END_OF_BACKTEST"

    # Метрики
    max_favorable_excursion: float
    max_adverse_excursion: float

    # Контекст
    entry_signal: Optional[Dict] = None
    exit_signal: Optional[Dict] = None


@dataclass
class EquityPoint:
    """Точка на кривой доходности."""
    timestamp: datetime
    sequence: int
    equity: float
    cash: float
    positions_value: float
    drawdown: float
    drawdown_pct: float
    total_return: float
    total_return_pct: float
    open_positions_count: int


@dataclass
class PerformanceMetrics:
    """Метрики производительности бэктеста."""
    # Returns
    total_return: float
    total_return_pct: float
    annual_return_pct: float
    monthly_returns: List[float] = field(default_factory=list)

    # Risk-Adjusted Returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility_annual_pct: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0
    avg_drawdown_pct: float = 0.0

    # Trade Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration_minutes: float = 0.0

    # Advanced Metrics
    omega_ratio: float = 0.0
    tail_ratio: float = 0.0
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional VaR (95%)

    # Quality Metrics
    stability: float = 0.0  # R² линейной регрессии equity curve

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для сохранения в БД."""
        return {
            'returns': {
                'total_return': self.total_return,
                'total_return_pct': self.total_return_pct,
                'annual_return_pct': self.annual_return_pct,
                'monthly_returns': self.monthly_returns
            },
            'risk': {
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio,
                'volatility_annual': self.volatility_annual_pct
            },
            'drawdown': {
                'max_drawdown_pct': self.max_drawdown_pct,
                'max_drawdown_duration_days': self.max_drawdown_duration_days,
                'avg_drawdown_pct': self.avg_drawdown_pct
            },
            'trade_stats': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate_pct': self.win_rate_pct,
                'profit_factor': self.profit_factor,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'largest_win': self.largest_win,
                'largest_loss': self.largest_loss,
                'avg_trade_duration_minutes': self.avg_trade_duration_minutes
            },
            'advanced': {
                'omega_ratio': self.omega_ratio,
                'tail_ratio': self.tail_ratio,
                'var_95': self.var_95,
                'cvar_95': self.cvar_95,
                'stability': self.stability
            }
        }


@dataclass
class BacktestResult:
    """Результат выполнения бэктеста."""
    backtest_id: str
    config: BacktestConfig

    # Результаты
    final_capital: float
    total_pnl: float
    total_pnl_pct: float

    # Метрики
    metrics: PerformanceMetrics

    # Данные
    trades: List[TradeResult]
    equity_curve: List[EquityPoint]

    # Временные метки
    started_at: datetime
    completed_at: datetime
    duration_seconds: float

    # Статус
    success: bool
    error_message: Optional[str] = None


__all__ = [
    'SlippageModel',
    'ExchangeConfig',
    'StrategyConfig',
    'RiskConfig',
    'BacktestConfig',
    'TradeResult',
    'EquityPoint',
    'PerformanceMetrics',
    'BacktestResult',
]
