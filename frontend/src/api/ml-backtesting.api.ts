// frontend/src/api/ml-backtesting.api.ts

import apiClient from './client';

/**
 * ============================================================
 * TYPES & INTERFACES
 * ============================================================
 */

export interface MLBacktestConfig {
  name: string;
  description?: string;

  // Model
  model_checkpoint: string;
  model_version?: string;

  // Data source
  data_source: 'holdout' | 'custom' | 'feature_store';
  symbol?: string;
  start_date?: string;
  end_date?: string;
  holdout_set_id?: string;

  // Walk-forward
  use_walk_forward: boolean;
  n_periods: number;
  retrain_each_period: boolean;

  // Trading simulation
  initial_capital: number;
  position_size: number;
  commission: number;
  slippage: number;

  // Confidence filtering
  use_confidence_filter: boolean;
  min_confidence: number;
  confidence_mode: 'threshold' | 'dynamic' | 'percentile';

  // Inference
  sequence_length: number;
  batch_size: number;
  device: 'auto' | 'cuda' | 'cpu';
}

export interface ClassificationMetrics {
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  precision_per_class: Record<string, number>;  // SELL, HOLD, BUY
  recall_per_class: Record<string, number>;
  f1_per_class: Record<string, number>;
  support_per_class?: Record<string, number>;
}

export interface TradingMetrics {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_pnl: number;
  total_pnl_percent: number;
  max_drawdown: number;
  sharpe_ratio: number;
  profit_factor: number;
  final_capital: number;
}

export interface PeriodResult {
  period: number;
  start_idx: number;
  end_idx: number;
  samples: number;
  accuracy: number;
  f1_macro: number;
  precision_macro?: number;
  recall_macro?: number;
  pnl_percent?: number;
  win_rate?: number;
  class_distribution: Record<string, number>;
}

export interface MLBacktestRun {
  id: string;
  name: string;
  description?: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

  // Model
  model_checkpoint: string;
  model_version?: string;
  model_architecture?: string;

  // Data
  data_source: string;
  symbol?: string;
  start_date?: string;
  end_date?: string;

  // Config
  use_walk_forward?: boolean;
  n_periods?: number;
  initial_capital?: number;
  position_size?: number;
  commission?: number;
  slippage?: number;
  use_confidence_filter?: boolean;
  min_confidence?: number;
  sequence_length?: number;
  batch_size?: number;
  device?: string;

  // Classification results
  total_samples?: number;
  accuracy?: number;
  precision_macro?: number;
  recall_macro?: number;
  f1_macro?: number;
  precision_per_class?: Record<string, number>;
  recall_per_class?: Record<string, number>;
  f1_per_class?: Record<string, number>;
  support_per_class?: Record<string, number>;
  confusion_matrix?: number[][];

  // Trading results
  total_trades?: number;
  winning_trades?: number;
  losing_trades?: number;
  win_rate?: number;
  total_pnl?: number;
  total_pnl_percent?: number;
  max_drawdown?: number;
  sharpe_ratio?: number;
  profit_factor?: number;
  final_capital?: number;

  // Walk-forward
  period_results?: PeriodResult[];

  // Meta
  created_at: string;
  started_at?: string;
  completed_at?: string;
  duration_seconds?: number;
  progress_pct?: number;
  error_message?: string;
}

export interface Prediction {
  timestamp: string;
  sequence: number;
  predicted_class: number;
  actual_class: number;
  confidence: number;
  prob_sell?: number;
  prob_hold?: number;
  prob_buy?: number;
  trade_executed: boolean;
  trade_pnl?: number;
  period?: number;
}

export interface ConfusionMatrix {
  matrix: number[][];
  labels: string[];
  normalized?: number[][];
  total_samples: number;
  correct_predictions: number;
  accuracy: number;
}

export interface ModelInfo {
  checkpoint_path: string;
  version?: string;
  architecture?: string;
  created_at?: string;
  metrics?: Record<string, number>;
  stage?: string;
}

export interface MLBacktestStatistics {
  total_backtests: number;
  completed_backtests: number;
  running_backtests: number;
  failed_backtests: number;
  avg_accuracy: number;
  avg_f1_macro: number;
  avg_sharpe_ratio: number;
  avg_win_rate: number;
  best_backtest?: {
    id: string;
    name: string;
    sharpe_ratio: number;
    accuracy?: number;
  };
  worst_backtest?: {
    id: string;
    name: string;
    sharpe_ratio: number;
    accuracy?: number;
  };
}

/**
 * ============================================================
 * API FUNCTIONS
 * ============================================================
 */

/**
 * Create and start ML backtest
 */
export async function createMLBacktest(
  config: MLBacktestConfig
): Promise<{ id: string; name: string; status: string; message: string; created_at: string }> {
  const response = await apiClient.post('/api/ml-backtesting/runs', config);
  return response.data;
}

/**
 * List ML backtests with filtering
 */
export async function listMLBacktests(params?: {
  status?: string;
  page?: number;
  page_size?: number;
}): Promise<{ runs: MLBacktestRun[]; total: number; page: number; page_size: number }> {
  const response = await apiClient.get('/api/ml-backtesting/runs', { params });
  return response.data;
}

/**
 * Get ML backtest details
 */
export async function getMLBacktest(
  id: string,
  include_predictions = false
): Promise<MLBacktestRun & { predictions?: Prediction[] }> {
  const response = await apiClient.get(`/api/ml-backtesting/runs/${id}`, {
    params: { include_predictions }
  });
  return response.data;
}

/**
 * Get ML backtest predictions
 */
export async function getMLBacktestPredictions(
  id: string,
  limit = 1000,
  period?: number
): Promise<{ backtest_id: string; predictions: Prediction[]; total: number }> {
  const response = await apiClient.get(`/api/ml-backtesting/runs/${id}/predictions`, {
    params: { limit, period }
  });
  return response.data;
}

/**
 * Get confusion matrix
 */
export async function getConfusionMatrix(id: string): Promise<ConfusionMatrix> {
  const response = await apiClient.get(`/api/ml-backtesting/runs/${id}/confusion-matrix`);
  return response.data;
}

/**
 * Get walk-forward period results
 */
export async function getWalkForwardPeriods(id: string): Promise<{
  backtest_id: string;
  use_walk_forward: boolean;
  n_periods: number;
  periods: PeriodResult[];
}> {
  const response = await apiClient.get(`/api/ml-backtesting/runs/${id}/periods`);
  return response.data;
}

/**
 * Cancel running ML backtest
 */
export async function cancelMLBacktest(
  id: string
): Promise<{ success: boolean; backtest_id: string; message: string }> {
  const response = await apiClient.post(`/api/ml-backtesting/runs/${id}/cancel`);
  return response.data;
}

/**
 * Delete ML backtest
 */
export async function deleteMLBacktest(
  id: string
): Promise<{ success: boolean; backtest_id: string; message: string }> {
  const response = await apiClient.delete(`/api/ml-backtesting/runs/${id}`);
  return response.data;
}

/**
 * List available models
 */
export async function listAvailableModels(): Promise<{ models: ModelInfo[]; total: number }> {
  const response = await apiClient.get('/api/ml-backtesting/models');
  return response.data;
}

/**
 * Get aggregate statistics
 */
export async function getMLBacktestStatistics(): Promise<MLBacktestStatistics> {
  const response = await apiClient.get('/api/ml-backtesting/statistics');
  return response.data;
}

/**
 * Health check
 */
export async function healthCheck(): Promise<{
  status: string;
  service: string;
  timestamp: string;
  running_backtests: number;
}> {
  const response = await apiClient.get('/api/ml-backtesting/health');
  return response.data;
}

/**
 * ============================================================
 * HELPER FUNCTIONS
 * ============================================================
 */

/**
 * Get class name from class ID
 */
export function getClassName(classId: number): string {
  return { 0: 'SELL', 1: 'HOLD', 2: 'BUY' }[classId] || 'UNKNOWN';
}

/**
 * Get class color for styling
 */
export function getClassColor(className: string): string {
  return {
    'SELL': 'text-red-400',
    'HOLD': 'text-yellow-400',
    'BUY': 'text-green-400'
  }[className] || 'text-gray-400';
}

/**
 * Format percentage
 */
export function formatPercent(value: number | undefined | null, decimals = 2): string {
  if (value === undefined || value === null) return 'N/A';
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format accuracy (already as decimal)
 */
export function formatAccuracy(value: number | undefined | null, decimals = 2): string {
  if (value === undefined || value === null) return 'N/A';
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Default config for creating ML backtest
 */
export function getDefaultMLBacktestConfig(): MLBacktestConfig {
  return {
    name: '',
    description: '',
    model_checkpoint: '',
    model_version: undefined,
    data_source: 'holdout',
    symbol: undefined,
    start_date: undefined,
    end_date: undefined,
    holdout_set_id: undefined,
    use_walk_forward: true,
    n_periods: 5,
    retrain_each_period: false,
    initial_capital: 10000,
    position_size: 0.1,
    commission: 0.001,
    slippage: 0.0005,
    use_confidence_filter: true,
    min_confidence: 0.6,
    confidence_mode: 'threshold',
    sequence_length: 60,
    batch_size: 128,
    device: 'auto'
  };
}

/**
 * ============================================================
 * PBO ANALYSIS TYPES & API
 * ============================================================
 */

export interface PBOAnalysis {
  pbo: number;
  pbo_adjusted: number;
  is_overfit: boolean;
  confidence_level: number;
  n_combinations: number;
  is_sharpe_ratios: number[];
  oos_sharpe_ratios: number[];
  rank_correlation: number;
  best_is_idx: number;
  best_is_sharpe: number;
  best_is_oos_sharpe: number;
  best_is_oos_rank: number;
  interpretation: string;
  risk_level: 'low' | 'moderate' | 'high' | 'very_high';
}

/**
 * Get PBO analysis for backtest
 */
export async function getPBOAnalysis(id: string): Promise<PBOAnalysis> {
  const response = await apiClient.get(`/api/ml-backtesting/runs/${id}/pbo-analysis`);
  return response.data;
}

/**
 * ============================================================
 * MONTE CARLO TYPES & API
 * ============================================================
 */

export interface MonteCarloRequest {
  n_simulations: number;
  confidence_levels?: number[];
}

export interface MonteCarloResult {
  n_simulations: number;
  final_equity: {
    mean: number;
    std: number;
    min: number;
    max: number;
    percentile_5: number;
    percentile_25: number;
    percentile_50: number;
    percentile_75: number;
    percentile_95: number;
  };
  max_drawdown: {
    mean: number;
    std: number;
    worst_case_95: number;
  };
  probability_of_profit: number;
  probability_of_ruin: number;
  var_95: number;
  cvar_95: number;
  equity_paths: number[][];
  percentile_paths: {
    p5: number[];
    p25: number[];
    p50: number[];
    p75: number[];
    p95: number[];
  };
}

/**
 * Run Monte Carlo simulation
 */
export async function runMonteCarloSimulation(
  id: string,
  params: MonteCarloRequest = { n_simulations: 1000 }
): Promise<MonteCarloResult> {
  const response = await apiClient.post(`/api/ml-backtesting/runs/${id}/monte-carlo`, params);
  return response.data;
}

/**
 * ============================================================
 * MODEL COMPARISON TYPES & API
 * ============================================================
 */

export interface ModelComparisonRequest {
  backtest_ids: string[];
}

export interface ModelComparisonResult {
  models: Array<{
    id: string;
    name: string;
    model_architecture?: string;
  }>;
  comparison_table: Array<{
    id: string;
    name: string;
    model_architecture: string;
    accuracy?: number;
    f1_macro?: number;
    sharpe_ratio?: number;
    win_rate?: number;
    max_drawdown?: number;
    total_pnl_percent?: number;
  }>;
  best_model: {
    id: string;
    name: string;
    composite_score: number;
    accuracy?: number;
    sharpe_ratio?: number;
  };
  rankings: Record<string, string[]>;
  statistical_tests?: {
    paired_t_tests?: Array<{
      model_a: string;
      model_b: string;
      t_statistic: number;
      p_value: number;
      significant: boolean;
    }>;
  };
}

/**
 * Compare multiple models
 */
export async function compareModels(
  backtest_ids: string[]
): Promise<ModelComparisonResult> {
  const response = await apiClient.post('/api/ml-backtesting/compare', { backtest_ids });
  return response.data;
}

/**
 * ============================================================
 * REGIME ANALYSIS TYPES & API
 * ============================================================
 */

export interface RegimeMetrics {
  accuracy: number;
  avg_confidence: number;
  n_samples: number;
  pnl_estimate: number;
  win_rate: number;
}

export interface RegimeData {
  regime: string;
  display_name: string;
  accuracy: number;
  avg_confidence: number;
  n_samples: number;
  pnl_estimate: number;
  win_rate: number;
}

export interface RegimeAnalysis {
  regimes: RegimeData[];
  overall_regime_distribution: Record<string, number>;
  best_regime: string;
  worst_regime: string;
  regime_metrics: Record<string, RegimeMetrics>;
}

/**
 * Get regime analysis for backtest
 */
export async function getRegimeAnalysis(id: string): Promise<RegimeAnalysis> {
  const response = await apiClient.get(`/api/ml-backtesting/runs/${id}/regime-analysis`);
  return response.data;
}

/**
 * ============================================================
 * EQUITY CURVE TYPES & API
 * ============================================================
 */

export interface EquityCurvePoint {
  x: number;
  equity: number;
  drawdown: number;
}

export interface EquityCurveData {
  backtest_id: string;
  initial_capital: number;
  final_capital: number;
  total_return_pct: number;
  max_drawdown_pct: number;
  n_points: number;
  data: EquityCurvePoint[];
}

/**
 * Get equity curve data for charts
 */
export async function getEquityCurve(
  id: string,
  sampling = 100
): Promise<EquityCurveData> {
  const response = await apiClient.get(`/api/ml-backtesting/runs/${id}/equity-curve`, {
    params: { sampling }
  });
  return response.data;
}

/**
 * ============================================================
 * RISK LEVEL HELPERS
 * ============================================================
 */

export function getRiskLevelColor(level: string): string {
  switch (level) {
    case 'low':
      return 'text-green-400 bg-green-500/10';
    case 'moderate':
      return 'text-yellow-400 bg-yellow-500/10';
    case 'high':
      return 'text-orange-400 bg-orange-500/10';
    case 'very_high':
      return 'text-red-400 bg-red-500/10';
    default:
      return 'text-gray-400 bg-gray-500/10';
  }
}

export function getRiskLevelLabel(level: string): string {
  switch (level) {
    case 'low':
      return 'Low Risk';
    case 'moderate':
      return 'Moderate Risk';
    case 'high':
      return 'High Risk';
    case 'very_high':
      return 'Very High Risk';
    default:
      return 'Unknown';
  }
}

export function getRegimeColor(regime: string): string {
  switch (regime) {
    case 'trending_up':
      return 'text-green-400';
    case 'trending_down':
      return 'text-red-400';
    case 'ranging':
      return 'text-blue-400';
    case 'high_volatility':
      return 'text-purple-400';
    default:
      return 'text-gray-400';
  }
}
