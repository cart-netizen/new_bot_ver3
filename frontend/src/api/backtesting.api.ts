// frontend/src/api/backtesting.api.ts

import apiClient from './client';

/**
 * ============================================================
 * TYPES & INTERFACES (ОБНОВЛЕНО для Фазы 1-3)
 * ============================================================
 */

export interface BacktestConfig {
  name: string;
  description?: string;
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  candle_interval: string;

  // Exchange Config
  commission_rate: number;
  maker_commission?: number;
  taker_commission?: number;
  slippage_model: 'fixed' | 'volume_based' | 'percentage';
  slippage_pct: number;
  simulate_latency: boolean;

  // Strategy Config
  enabled_strategies: string[];
  strategy_params?: Record<string, Record<string, any>>;
  consensus_mode: string;
  min_strategies_for_signal: number;
  min_consensus_confidence: number;
  strategy_weights?: Record<string, number>;

  // Risk Config
  position_size_pct: number;
  position_size_mode: string;
  max_open_positions: number;
  stop_loss_pct: number;
  take_profit_pct: number;
  use_trailing_stop: boolean;
  trailing_stop_activation_pct: number;
  trailing_stop_distance_pct: number;
  risk_per_trade_pct: number;

  // OrderBook & Market Trades (НОВОЕ - Фаза 1)
  use_orderbook_data: boolean;
  orderbook_num_levels?: number;  // 10-50, default 20
  orderbook_base_spread_bps?: number;  // Базовый спред в basis points, default 2.0
  use_market_trades?: boolean;
  trades_per_volume_unit?: number;  // Trades на единицу объема, default 100

  // ML Model Integration (НОВОЕ - Фаза 2)
  use_ml_model?: boolean;
  ml_server_url?: string;  // URL ML сервера, default "http://localhost:8001"
  ml_model_name?: string;  // Имя модели (опционально)
  ml_model_version?: string;  // Версия модели (опционально)

  // Optimization (НОВОЕ - Фаза 3)
  use_cache?: boolean;  // Использовать кэширование данных, default true
  warmup_period_bars: number;
  skip_orderbook_generation_every_n?: number;  // Генерировать orderbook каждые N свечей
  skip_trades_generation_every_n?: number;  // Генерировать trades каждые N свечей

  // Debug
  verbose: boolean;
  log_trades?: boolean;
}

export interface BacktestRun {
  id: string;
  name: string;
  description?: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_capital?: number;
  total_pnl?: number;
  total_pnl_pct?: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  progress?: number;
  error_message?: string;
  metrics?: PerformanceMetrics;
}

export interface PerformanceMetrics {
  returns: {
    total_return: number;
    total_return_pct: number;
    annual_return_pct: number;
    monthly_returns: number[];
  };
  risk: {
    sharpe_ratio: number;
    sortino_ratio: number;
    calmar_ratio: number;
    volatility_annual: number;
  };
  drawdown: {
    max_drawdown_pct: number;
    max_drawdown_duration_days: number;
    avg_drawdown_pct: number;
  };
  trade_stats: {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate_pct: number;
    profit_factor: number;
    avg_win: number;
    avg_loss: number;
    largest_win: number;
    largest_loss: number;
    avg_trade_duration_minutes: number;
  };
  advanced: {
    omega_ratio: number;
    tail_ratio: number;
    var_95: number;
    cvar_95: number;
    stability: number;
  };
}

// Информация о консенсусе стратегий (НОВОЕ - Фаза 3)
export interface ConsensusInfo {
  mode: string;  // 'weighted', 'majority', 'unanimous'
  strategies_voted: string[];  // Все проголосовавшие стратегии
  strategies_buy: string[];  // Стратегии за BUY
  strategies_sell: string[];  // Стратегии за SELL
  agreement_count: number;  // Количество согласных
  disagreement_count: number;  // Количество несогласных
  consensus_confidence: number;  // Уверенность консенсуса (0-1)
  final_confidence: number;  // Итоговая уверенность сигнала (0-1)
  candle_strategies: number;  // Количество candle стратегий
  orderbook_strategies: number;  // Количество orderbook стратегий
  hybrid_strategies: number;  // Количество hybrid стратегий
}

export interface Trade {
  symbol: string;
  side: string;
  entry_time: string;
  exit_time: string;
  entry_price: number;
  exit_price: number;
  quantity: number;
  pnl: number;
  pnl_pct: number;
  commission: number;
  duration_seconds: number;
  exit_reason: string;
  max_favorable_excursion?: number;
  max_adverse_excursion?: number;
  entry_signal?: any;  // Информация о сигнале входа
  exit_signal?: any;  // Информация о сигнале выхода
  consensus_info?: ConsensusInfo;  // НОВОЕ: Детали консенсуса
}

export interface EquityPoint {
  timestamp: string;
  sequence: number;
  equity: number;
  cash: number;
  positions_value: number;
  drawdown: number;
  drawdown_pct: number;
  total_return: number;
  total_return_pct: number;
  open_positions_count: number;
}

export interface BacktestStatistics {
  total_backtests: number;
  completed_backtests: number;
  running_backtests: number;
  failed_backtests: number;
  avg_total_return_pct: number;
  avg_sharpe_ratio: number;
  avg_max_drawdown_pct: number;
  avg_win_rate_pct: number;
  best_backtest?: any;
  worst_backtest?: any;
}

/**
 * ============================================================
 * API FUNCTIONS
 * ============================================================
 */

/**
 * Создать и запустить новый бэктест
 */
export async function createBacktest(config: BacktestConfig): Promise<{ id: string; name: string; status: string; message: string; created_at: string }> {
  const response = await apiClient.post('/api/backtesting/runs', config);
  return response.data;
}

/**
 * Получить список бэктестов с фильтрацией
 */
export async function listBacktests(params?: {
  status?: string;
  symbol?: string;
  page?: number;
  page_size?: number;
}): Promise<{ runs: BacktestRun[]; total: number; page: number; page_size: number }> {
  const response = await apiClient.get('/api/backtesting/runs', { params });
  return response.data;
}

/**
 * Получить детальную информацию о бэктесте
 */
export async function getBacktest(
  id: string,
  include_trades = false,
  include_equity = false
): Promise<BacktestRun & { trades?: Trade[]; equity_curve?: EquityPoint[] }> {
  const response = await apiClient.get(`/api/backtesting/runs/${id}`, {
    params: { include_trades, include_equity }
  });
  return response.data;
}

/**
 * Получить сделки бэктеста
 */
export async function getBacktestTrades(id: string, limit = 100): Promise<{ backtest_id: string; trades: Trade[]; total: number }> {
  const response = await apiClient.get(`/api/backtesting/runs/${id}/trades`, {
    params: { limit }
  });
  return response.data;
}

/**
 * Получить equity curve бэктеста
 */
export async function getEquityCurve(
  id: string,
  sampling_interval_minutes = 60
): Promise<{ backtest_id: string; equity_curve: EquityPoint[]; total_points: number }> {
  const response = await apiClient.get(`/api/backtesting/runs/${id}/equity-curve`, {
    params: { sampling_interval_minutes }
  });
  return response.data;
}

/**
 * Отменить выполняющийся бэктест
 */
export async function cancelBacktest(id: string): Promise<{ success: boolean; backtest_id: string; message: string }> {
  const response = await apiClient.post(`/api/backtesting/runs/${id}/cancel`);
  return response.data;
}

/**
 * Удалить бэктест
 */
export async function deleteBacktest(id: string): Promise<{ success: boolean; backtest_id: string; message: string }> {
  const response = await apiClient.delete(`/api/backtesting/runs/${id}`);
  return response.data;
}

/**
 * Получить агрегированную статистику
 */
export async function getStatistics(): Promise<BacktestStatistics> {
  const response = await apiClient.get('/api/backtesting/statistics');
  return response.data;
}

/**
 * Health check
 */
export async function healthCheck(): Promise<{ status: string; service: string; timestamp: string; running_backtests: number }> {
  const response = await apiClient.get('/api/backtesting/health');
  return response.data;
}

/**
 * Получить дефолтные настройки для конфигурации (НОВОЕ)
 */
export async function getDefaultConfig(): Promise<Partial<BacktestConfig>> {
  const response = await apiClient.get('/api/backtesting/config/defaults');
  return response.data;
}

/**
 * Валидировать конфигурацию бэктеста (НОВОЕ)
 */
export async function validateConfig(config: Partial<BacktestConfig>): Promise<{ valid: boolean; errors?: string[] }> {
  const response = await apiClient.post('/api/backtesting/config/validate', config);
  return response.data;
}
