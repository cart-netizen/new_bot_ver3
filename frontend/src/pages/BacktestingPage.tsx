// frontend/src/pages/BacktestingPage.tsx

import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Play,
  RefreshCw,
  Trash2,
  XCircle,
  ChevronRight,
  Plus,
  TrendingUp,
  Calendar,
  DollarSign,
  Settings,
  BarChart3,
  Activity,
  Target
} from 'lucide-react';
import { toast } from 'sonner';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { MetricsGrid } from '../components/backtesting/MetricsGrid';
import { TradesList } from '../components/backtesting/TradesList';
import { EquityCurveChart } from '../components/backtesting/EquityCurveChart';
import { cn } from '../utils/helpers';
import * as backtestingApi from '../api/backtesting.api';

export function BacktestingPage() {
  const [selectedView, setSelectedView] = useState<'list' | 'create' | 'results'>('list');
  const [selectedBacktestId, setSelectedBacktestId] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Fetch backtests list
  const { data: backtestsData, isLoading: isLoadingList, refetch: refetchList } = useQuery({
    queryKey: ['backtests'],
    queryFn: () => backtestingApi.listBacktests(),
    refetchInterval: 5000 // Refresh every 5s for running backtests
  });

  // Fetch selected backtest details
  const { data: backtestDetails, isLoading: isLoadingDetails } = useQuery({
    queryKey: ['backtest', selectedBacktestId],
    queryFn: () => backtestingApi.getBacktest(selectedBacktestId!, true, true),
    enabled: !!selectedBacktestId && selectedView === 'results'
  });

  // Create backtest mutation
  const createMutation = useMutation({
    mutationFn: backtestingApi.createBacktest,
    onSuccess: () => {
      toast.success('Backtest started successfully!');
      queryClient.invalidateQueries({ queryKey: ['backtests'] });
      setSelectedView('list');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to create backtest');
    }
  });

  // Delete backtest mutation
  const deleteMutation = useMutation({
    mutationFn: backtestingApi.deleteBacktest,
    onSuccess: () => {
      toast.success('Backtest deleted');
      queryClient.invalidateQueries({ queryKey: ['backtests'] });
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to delete backtest');
    }
  });

  // Cancel backtest mutation
  const cancelMutation = useMutation({
    mutationFn: backtestingApi.cancelBacktest,
    onSuccess: () => {
      toast.success('Backtest cancelled');
      queryClient.invalidateQueries({ queryKey: ['backtests'] });
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to cancel backtest');
    }
  });

  const handleViewResults = (id: string) => {
    setSelectedBacktestId(id);
    setSelectedView('results');
  };

  const handleBackToList = () => {
    setSelectedBacktestId(null);
    setSelectedView('list');
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Backtesting</h1>
          <p className="text-gray-400 mt-1">Test strategies on historical data</p>
        </div>
        <div className="flex gap-2">
          {selectedView === 'list' && (
            <>
              <Button onClick={() => refetchList()} variant="outline">
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
              <Button onClick={() => setSelectedView('create')}>
                <Plus className="h-4 w-4 mr-2" />
                New Backtest
              </Button>
            </>
          )}
          {selectedView === 'create' && (
            <Button onClick={handleBackToList} variant="outline">
              Back to List
            </Button>
          )}
          {selectedView === 'results' && (
            <Button onClick={handleBackToList} variant="outline">
              Back to List
            </Button>
          )}
        </div>
      </div>

      {/* Main Content */}
      {selectedView === 'list' && (
        <BacktestsList
          backtests={backtestsData?.runs || []}
          isLoading={isLoadingList}
          onViewResults={handleViewResults}
          onDelete={(id) => deleteMutation.mutate(id)}
          onCancel={(id) => cancelMutation.mutate(id)}
        />
      )}

      {selectedView === 'create' && (
        <BacktestForm
          onSubmit={(config) => createMutation.mutate(config)}
          isSubmitting={createMutation.isPending}
        />
      )}

      {selectedView === 'results' && backtestDetails && (
        <BacktestResults
          backtest={backtestDetails}
          isLoading={isLoadingDetails}
        />
      )}
    </div>
  );
}

// ============================================================
// Backtests List Component
// ============================================================

interface BacktestsListProps {
  backtests: backtestingApi.BacktestRun[];
  isLoading: boolean;
  onViewResults: (id: string) => void;
  onDelete: (id: string) => void;
  onCancel: (id: string) => void;
}

function BacktestsList({ backtests, isLoading, onViewResults, onDelete, onCancel }: BacktestsListProps) {
  if (isLoading) {
    return (
      <Card className="p-8 text-center">
        <RefreshCw className="h-8 w-8 animate-spin text-gray-400 mx-auto mb-2" />
        <p className="text-gray-400">Loading backtests...</p>
      </Card>
    );
  }

  if (backtests.length === 0) {
    return (
      <Card className="p-12 text-center">
        <BarChart3 className="h-12 w-12 text-gray-600 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-white mb-2">No backtests yet</h3>
        <p className="text-gray-400">Create your first backtest to get started</p>
      </Card>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-400 bg-green-500/10';
      case 'running':
        return 'text-blue-400 bg-blue-500/10';
      case 'failed':
        return 'text-red-400 bg-red-500/10';
      case 'cancelled':
        return 'text-yellow-400 bg-yellow-500/10';
      default:
        return 'text-gray-400 bg-gray-500/10';
    }
  };

  return (
    <div className="grid grid-cols-1 gap-4">
      {backtests.map((backtest) => (
        <Card key={backtest.id} className="p-6 hover:border-gray-700 transition-colors">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <h3 className="text-lg font-semibold text-white">{backtest.name}</h3>
                <span className={cn("text-xs font-medium px-2 py-1 rounded", getStatusColor(backtest.status))}>
                  {backtest.status.toUpperCase()}
                </span>
                {backtest.status === 'running' && backtest.progress !== undefined && (
                  <span className="text-sm text-gray-400">
                    {backtest.progress.toFixed(1)}%
                  </span>
                )}
              </div>
              {backtest.description && (
                <p className="text-sm text-gray-400 mb-3">{backtest.description}</p>
              )}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-gray-400">Symbol</p>
                  <p className="text-white font-medium">{backtest.symbol}</p>
                </div>
                <div>
                  <p className="text-gray-400">Period</p>
                  <p className="text-white font-medium">
                    {new Date(backtest.start_date).toLocaleDateString()} - {new Date(backtest.end_date).toLocaleDateString()}
                  </p>
                </div>
                <div>
                  <p className="text-gray-400">Initial Capital</p>
                  <p className="text-white font-medium">${backtest.initial_capital.toFixed(2)}</p>
                </div>
                {backtest.total_pnl !== undefined && backtest.total_pnl !== null && (
                  <div>
                    <p className="text-gray-400">Total PnL</p>
                    <p className={cn(
                      "font-medium",
                      backtest.total_pnl >= 0 ? "text-green-400" : "text-red-400"
                    )}>
                      ${backtest.total_pnl.toFixed(2)} ({backtest.total_pnl_pct?.toFixed(2)}%)
                    </p>
                  </div>
                )}
              </div>
            </div>
            <div className="flex gap-2 ml-4">
              {backtest.status === 'completed' && (
                <Button onClick={() => onViewResults(backtest.id)} size="sm">
                  <ChevronRight className="h-4 w-4 mr-1" />
                  View Results
                </Button>
              )}
              {backtest.status === 'running' && (
                <Button onClick={() => onCancel(backtest.id)} variant="outline" size="sm">
                  <XCircle className="h-4 w-4 mr-1" />
                  Cancel
                </Button>
              )}
              {['completed', 'failed', 'cancelled'].includes(backtest.status) && (
                <Button onClick={() => onDelete(backtest.id)} variant="outline" size="sm">
                  <Trash2 className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
}

// ============================================================
// Backtest Form Component
// ============================================================

interface BacktestFormProps {
  onSubmit: (config: backtestingApi.BacktestConfig) => void;
  isSubmitting: boolean;
}

function BacktestForm({ onSubmit, isSubmitting }: BacktestFormProps) {
  const [formData, setFormData] = useState<backtestingApi.BacktestConfig>({
    name: '',
    description: '',
    symbol: 'BTCUSDT',
    start_date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    end_date: new Date().toISOString().split('T')[0],
    initial_capital: 10000,
    candle_interval: '1',
    commission_rate: 0.0006,
    maker_commission: 0.0002,
    taker_commission: 0.0006,
    slippage_model: 'fixed',
    slippage_pct: 0.01,
    simulate_latency: false,
    enabled_strategies: ['momentum', 'sar_wave', 'supertrend', 'volume_profile'],
    consensus_mode: 'weighted',
    min_strategies_for_signal: 2,
    min_consensus_confidence: 0.6,
    position_size_pct: 10,
    position_size_mode: 'percentage',
    max_open_positions: 3,
    stop_loss_pct: 2.0,
    take_profit_pct: 4.0,
    use_trailing_stop: true,
    trailing_stop_activation_pct: 1.0,
    trailing_stop_distance_pct: 0.5,
    risk_per_trade_pct: 1.0,
    use_orderbook_data: false,
    warmup_period_bars: 100,
    verbose: false
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Basic Settings */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Settings className="h-5 w-5" />
          Basic Settings
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Input
            label="Name *"
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            required
          />
          <Input
            label="Symbol"
            value={formData.symbol}
            onChange={(e) => setFormData({ ...formData, symbol: e.target.value })}
          />
          <Input
            label="Start Date *"
            type="date"
            value={formData.start_date}
            onChange={(e) => setFormData({ ...formData, start_date: e.target.value })}
            required
          />
          <Input
            label="End Date *"
            type="date"
            value={formData.end_date}
            onChange={(e) => setFormData({ ...formData, end_date: e.target.value })}
            required
          />
          <Input
            label="Initial Capital ($)"
            type="number"
            value={formData.initial_capital}
            onChange={(e) => setFormData({ ...formData, initial_capital: parseFloat(e.target.value) })}
          />
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Candle Interval</label>
            <select
              value={formData.candle_interval}
              onChange={(e) => setFormData({ ...formData, candle_interval: e.target.value })}
              className="w-full bg-gray-800 border border-gray-700 text-white rounded-lg px-4 py-2"
            >
              <option value="1">1 minute</option>
              <option value="5">5 minutes</option>
              <option value="15">15 minutes</option>
              <option value="60">1 hour</option>
            </select>
          </div>
        </div>
        <div className="mt-4">
          <label className="block text-sm font-medium text-gray-300 mb-1">Description</label>
          <textarea
            value={formData.description}
            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
            rows={2}
            className="w-full bg-gray-800 border border-gray-700 text-white rounded-lg px-4 py-2"
          />
        </div>
      </Card>

      {/* Risk Management */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Target className="h-5 w-5" />
          Risk Management
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Input
            label="Position Size (%)"
            type="number"
            step="0.1"
            value={formData.position_size_pct}
            onChange={(e) => setFormData({ ...formData, position_size_pct: parseFloat(e.target.value) })}
          />
          <Input
            label="Max Open Positions"
            type="number"
            value={formData.max_open_positions}
            onChange={(e) => setFormData({ ...formData, max_open_positions: parseInt(e.target.value) })}
          />
          <Input
            label="Risk Per Trade (%)"
            type="number"
            step="0.1"
            value={formData.risk_per_trade_pct}
            onChange={(e) => setFormData({ ...formData, risk_per_trade_pct: parseFloat(e.target.value) })}
          />
          <Input
            label="Stop Loss (%)"
            type="number"
            step="0.1"
            value={formData.stop_loss_pct}
            onChange={(e) => setFormData({ ...formData, stop_loss_pct: parseFloat(e.target.value) })}
          />
          <Input
            label="Take Profit (%)"
            type="number"
            step="0.1"
            value={formData.take_profit_pct}
            onChange={(e) => setFormData({ ...formData, take_profit_pct: parseFloat(e.target.value) })}
          />
          <div className="flex items-center gap-2 pt-6">
            <input
              type="checkbox"
              id="trailing_stop"
              checked={formData.use_trailing_stop}
              onChange={(e) => setFormData({ ...formData, use_trailing_stop: e.target.checked })}
              className="w-4 h-4"
            />
            <label htmlFor="trailing_stop" className="text-sm text-gray-300">Use Trailing Stop</label>
          </div>
        </div>
      </Card>

      {/* Exchange Settings */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Exchange Settings
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Input
            label="Commission Rate (%)"
            type="number"
            step="0.0001"
            value={formData.commission_rate * 100}
            onChange={(e) => setFormData({ ...formData, commission_rate: parseFloat(e.target.value) / 100 })}
          />
          <Input
            label="Slippage (%)"
            type="number"
            step="0.001"
            value={formData.slippage_pct}
            onChange={(e) => setFormData({ ...formData, slippage_pct: parseFloat(e.target.value) })}
          />
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Slippage Model</label>
            <select
              value={formData.slippage_model}
              onChange={(e) => setFormData({ ...formData, slippage_model: e.target.value as any })}
              className="w-full bg-gray-800 border border-gray-700 text-white rounded-lg px-4 py-2"
            >
              <option value="fixed">Fixed</option>
              <option value="volume_based">Volume Based</option>
              <option value="percentage">Percentage</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Submit */}
      <div className="flex justify-end gap-4">
        <Button
          type="submit"
          disabled={isSubmitting}
        >
          {isSubmitting ? (
            <>
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              Creating...
            </>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" />
              Run Backtest
            </>
          )}
        </Button>
      </div>
    </form>
  );
}

// ============================================================
// Backtest Results Component
// ============================================================

interface BacktestResultsProps {
  backtest: backtestingApi.BacktestRun & {
    trades?: backtestingApi.Trade[];
    equity_curve?: backtestingApi.EquityPoint[];
  };
  isLoading: boolean;
}

function BacktestResults({ backtest, isLoading }: BacktestResultsProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'trades' | 'equity'>('overview');

  if (isLoading) {
    return (
      <Card className="p-8 text-center">
        <RefreshCw className="h-8 w-8 animate-spin text-gray-400 mx-auto mb-2" />
        <p className="text-gray-400">Loading results...</p>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="p-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-white">{backtest.name}</h2>
            {backtest.description && (
              <p className="text-gray-400 mt-1">{backtest.description}</p>
            )}
          </div>
          <span className="text-xs font-medium px-3 py-1 rounded bg-green-500/10 text-green-400">
            COMPLETED
          </span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-gray-400">Symbol</p>
            <p className="text-white font-medium">{backtest.symbol}</p>
          </div>
          <div>
            <p className="text-gray-400">Period</p>
            <p className="text-white font-medium">
              {new Date(backtest.start_date).toLocaleDateString()} - {new Date(backtest.end_date).toLocaleDateString()}
            </p>
          </div>
          <div>
            <p className="text-gray-400">Duration</p>
            <p className="text-white font-medium">
              {backtest.completed_at && backtest.started_at
                ? `${Math.round((new Date(backtest.completed_at).getTime() - new Date(backtest.started_at).getTime()) / 1000)}s`
                : 'N/A'}
            </p>
          </div>
          <div>
            <p className="text-gray-400">Trades</p>
            <p className="text-white font-medium">{backtest.trades?.length || 0}</p>
          </div>
        </div>
      </Card>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-800">
        {(['overview', 'trades', 'equity'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={cn(
              "px-4 py-2 font-medium capitalize transition-colors",
              activeTab === tab
                ? "text-white border-b-2 border-primary"
                : "text-gray-400 hover:text-white"
            )}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && backtest.metrics && (
        <MetricsGrid
          metrics={backtest.metrics}
          initialCapital={backtest.initial_capital}
          finalCapital={backtest.final_capital}
        />
      )}

      {activeTab === 'trades' && backtest.trades && (
        <TradesList trades={backtest.trades} />
      )}

      {activeTab === 'equity' && backtest.equity_curve && (
        <EquityCurveChart
          equityCurve={backtest.equity_curve}
          initialCapital={backtest.initial_capital}
        />
      )}
    </div>
  );
}
