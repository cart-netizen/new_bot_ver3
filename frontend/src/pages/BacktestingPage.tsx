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
import { Tooltip } from '../components/ui/Tooltip';
import { MetricsGrid } from '../components/backtesting/MetricsGrid';
import { TradesList } from '../components/backtesting/TradesList';
import { EquityCurveChart } from '../components/backtesting/EquityCurveChart';
import { BacktestingSettings } from '../components/backtesting/BacktestingSettings';
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
      {/* Заголовок */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Бэктестинг</h1>
          <p className="text-gray-400 mt-1">Тестирование торговых стратегий на исторических данных</p>
        </div>
        <div className="flex gap-2">
          {selectedView === 'list' && (
            <>
              <Button onClick={() => refetchList()} variant="outline">
                <RefreshCw className="h-4 w-4 mr-2" />
                Обновить
              </Button>
              <Button onClick={() => setSelectedView('create')}>
                <Plus className="h-4 w-4 mr-2" />
                Создать тест
              </Button>
            </>
          )}
          {selectedView === 'create' && (
            <Button onClick={handleBackToList} variant="outline">
              Назад к списку
            </Button>
          )}
          {selectedView === 'results' && (
            <Button onClick={handleBackToList} variant="outline">
              Назад к списку
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
        <p className="text-gray-400">Загрузка тестов...</p>
      </Card>
    );
  }

  if (backtests.length === 0) {
    return (
      <Card className="p-12 text-center">
        <BarChart3 className="h-12 w-12 text-gray-600 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-white mb-2">Пока нет тестов</h3>
        <p className="text-gray-400">Создайте свой первый бэктест для начала работы</p>
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
                  <p className="text-gray-400">Торговая пара</p>
                  <p className="text-white font-medium">{backtest.symbol}</p>
                </div>
                <div>
                  <p className="text-gray-400">Период</p>
                  <p className="text-white font-medium">
                    {new Date(backtest.start_date).toLocaleDateString('ru-RU')} - {new Date(backtest.end_date).toLocaleDateString('ru-RU')}
                  </p>
                </div>
                <div>
                  <p className="text-gray-400">Начальный капитал</p>
                  <p className="text-white font-medium">${backtest.initial_capital.toFixed(2)}</p>
                </div>
                {backtest.total_pnl !== undefined && backtest.total_pnl !== null && (
                  <div>
                    <p className="text-gray-400">Общая прибыль</p>
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
                  Смотреть результаты
                </Button>
              )}
              {backtest.status === 'running' && (
                <Button onClick={() => onCancel(backtest.id)} variant="outline" size="sm">
                  <XCircle className="h-4 w-4 mr-1" />
                  Отменить
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
  const [formData, setFormData] = useState<Partial<backtestingApi.BacktestConfig>>({
    name: '',
    description: '',
    symbol: 'BTCUSDT',
    start_date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().slice(0, 16),
    end_date: new Date().toISOString().slice(0, 16),
    initial_capital: 10000,
    candle_interval: '1m',
    commission_rate: 0.001,
    maker_commission: 0.0002,
    taker_commission: 0.001,
    slippage_model: 'fixed',
    slippage_pct: 0.1,
    simulate_latency: false,
    enabled_strategies: ['momentum', 'sar_wave', 'supertrend', 'volume_profile'],
    consensus_mode: 'weighted',
    min_strategies_for_signal: 2,
    min_consensus_confidence: 0.5,
    position_size_pct: 10,
    position_size_mode: 'fixed_percent',
    max_open_positions: 3,
    stop_loss_pct: 2.0,
    take_profit_pct: 4.0,
    use_trailing_stop: false,
    trailing_stop_activation_pct: 2.0,
    trailing_stop_distance_pct: 1.0,
    risk_per_trade_pct: 1.0,
    use_orderbook_data: false,
    orderbook_num_levels: 20,
    orderbook_base_spread_bps: 2.0,
    use_market_trades: false,
    trades_per_volume_unit: 100,
    use_ml_model: false,
    ml_server_url: 'http://localhost:8001',
    use_cache: true,
    warmup_period_bars: 100,
    verbose: false,
    log_trades: false
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Validate required fields
    if (!formData.name || !formData.start_date || !formData.end_date) {
      toast.error('Заполните обязательные поля: название, даты начала и окончания');
      return;
    }

    onSubmit(formData as backtestingApi.BacktestConfig);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Use the comprehensive BacktestingSettings component */}
      <BacktestingSettings
        config={formData}
        onChange={setFormData}
      />

      {/* Кнопка запуска */}
      <Card className="p-6">
        <div className="flex justify-end gap-4">
          <Button
            type="submit"
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Создание...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Запустить бэктест
              </>
            )}
          </Button>
        </div>
      </Card>
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
        <p className="text-gray-400">Загрузка результатов...</p>
      </Card>
    );
  }

  const tabNames = {
    overview: 'Обзор',
    trades: 'Сделки',
    equity: 'Кривая доходности'
  };

  return (
    <div className="space-y-6">
      {/* Заголовок */}
      <Card className="p-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-white">{backtest.name}</h2>
            {backtest.description && (
              <p className="text-gray-400 mt-1">{backtest.description}</p>
            )}
          </div>
          <span className="text-xs font-medium px-3 py-1 rounded bg-green-500/10 text-green-400">
            ЗАВЕРШЁН
          </span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-gray-400">Торговая пара</p>
            <p className="text-white font-medium">{backtest.symbol}</p>
          </div>
          <div>
            <p className="text-gray-400">Период</p>
            <p className="text-white font-medium">
              {new Date(backtest.start_date).toLocaleDateString('ru-RU')} - {new Date(backtest.end_date).toLocaleDateString('ru-RU')}
            </p>
          </div>
          <div>
            <p className="text-gray-400">Длительность выполнения</p>
            <p className="text-white font-medium">
              {backtest.completed_at && backtest.started_at
                ? `${Math.round((new Date(backtest.completed_at).getTime() - new Date(backtest.started_at).getTime()) / 1000)} сек`
                : 'N/A'}
            </p>
          </div>
          <div>
            <p className="text-gray-400">Количество сделок</p>
            <p className="text-white font-medium">{backtest.trades?.length || 0}</p>
          </div>
        </div>
      </Card>

      {/* Вкладки */}
      <div className="flex gap-2 border-b border-gray-800">
        {(['overview', 'trades', 'equity'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={cn(
              "px-4 py-2 font-medium transition-colors",
              activeTab === tab
                ? "text-white border-b-2 border-primary"
                : "text-gray-400 hover:text-white"
            )}
          >
            {tabNames[tab]}
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
