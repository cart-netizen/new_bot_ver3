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
      {/* Основные настройки */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Settings className="h-5 w-5" />
          Основные настройки
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Название теста *
              <Tooltip content="Краткое название для вашего бэктеста. Помогает идентифицировать тест среди других. Например: 'BTC Momentum Strategy' или 'ETH волновой тест'" />
            </label>
            <Input
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Торговая пара
              <Tooltip content="Торговая пара для тестирования (например, BTCUSDT, ETHUSDT). Убедитесь, что по этой паре есть исторические данные за выбранный период." />
            </label>
            <Input
              value={formData.symbol}
              onChange={(e) => setFormData({ ...formData, symbol: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Дата начала *
              <Tooltip content="С какой даты начать тестирование стратегии. Рекомендуется выбирать период не менее 1 месяца для получения статистически значимых результатов." />
            </label>
            <Input
              type="date"
              value={formData.start_date}
              onChange={(e) => setFormData({ ...formData, start_date: e.target.value })}
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Дата окончания *
              <Tooltip content="До какой даты проводить тестирование. Должна быть позже даты начала. Чем дольше период, тем точнее оценка эффективности стратегии." />
            </label>
            <Input
              type="date"
              value={formData.end_date}
              onChange={(e) => setFormData({ ...formData, end_date: e.target.value })}
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Начальный капитал ($)
              <Tooltip content="Стартовая сумма для торговли в долларах (USDT). Типичные значения: от $1,000 до $100,000. Влияет на размеры позиций и общую прибыль." />
            </label>
            <Input
              type="number"
              value={formData.initial_capital}
              onChange={(e) => setFormData({ ...formData, initial_capital: parseFloat(e.target.value) })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Интервал свечей
              <Tooltip content="Временной интервал одной свечи. 1 минута - для скальпинга, 5-15 минут - для краткосрочной торговли, 1 час - для среднесрочных стратегий." />
            </label>
            <select
              value={formData.candle_interval}
              onChange={(e) => setFormData({ ...formData, candle_interval: e.target.value })}
              className="w-full bg-gray-800 border border-gray-700 text-white rounded-lg px-4 py-2"
            >
              <option value="1">1 минута</option>
              <option value="5">5 минут</option>
              <option value="15">15 минут</option>
              <option value="60">1 час</option>
            </select>
          </div>
        </div>
        <div className="mt-4">
          <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
            Описание
            <Tooltip content="Необязательное описание теста. Можете указать цель теста, особенности настроек или ожидаемые результаты." />
          </label>
          <textarea
            value={formData.description}
            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
            rows={2}
            className="w-full bg-gray-800 border border-gray-700 text-white rounded-lg px-4 py-2"
            placeholder="Например: Тестирование момент-стратегии на волатильном рынке"
          />
        </div>
      </Card>

      {/* Управление рисками */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Target className="h-5 w-5" />
          Управление рисками
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Размер позиции (%)
              <Tooltip content="Какой процент от капитала использовать на одну сделку. Диапазон: 1-100%. Рекомендуется: 5-20%. Например, при 10% и капитале $10,000 одна позиция будет $1,000." />
            </label>
            <Input
              type="number"
              step="0.1"
              value={formData.position_size_pct}
              onChange={(e) => setFormData({ ...formData, position_size_pct: parseFloat(e.target.value) })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Макс. открытых позиций
              <Tooltip content="Максимальное количество одновременно открытых сделок. Диапазон: 1-10. При значении 3 можно иметь не более 3 активных позиций. Защищает от переторговли." />
            </label>
            <Input
              type="number"
              value={formData.max_open_positions}
              onChange={(e) => setFormData({ ...formData, max_open_positions: parseInt(e.target.value) })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Риск на сделку (%)
              <Tooltip content="Максимальный процент капитала, который можно потерять в одной сделке. Диапазон: 0.5-5%. Рекомендуется: 1-2%. При 1% и капитале $10,000 максимальный убыток - $100." />
            </label>
            <Input
              type="number"
              step="0.1"
              value={formData.risk_per_trade_pct}
              onChange={(e) => setFormData({ ...formData, risk_per_trade_pct: parseFloat(e.target.value) })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Стоп-лосс (%)
              <Tooltip content="На сколько процентов может упасть цена от входа, прежде чем позиция закроется автоматически. Диапазон: 0.5-10%. Защищает от больших убытков." />
            </label>
            <Input
              type="number"
              step="0.1"
              value={formData.stop_loss_pct}
              onChange={(e) => setFormData({ ...formData, stop_loss_pct: parseFloat(e.target.value) })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Тейк-профит (%)
              <Tooltip content="На сколько процентов должна вырасти цена, чтобы зафиксировать прибыль. Диапазон: 1-20%. Обычно в 1.5-3 раза больше стоп-лосса для положительного соотношения риск/прибыль." />
            </label>
            <Input
              type="number"
              step="0.1"
              value={formData.take_profit_pct}
              onChange={(e) => setFormData({ ...formData, take_profit_pct: parseFloat(e.target.value) })}
            />
          </div>
          <div className="flex items-center gap-2 pt-6">
            <input
              type="checkbox"
              id="trailing_stop"
              checked={formData.use_trailing_stop}
              onChange={(e) => setFormData({ ...formData, use_trailing_stop: e.target.checked })}
              className="w-4 h-4"
            />
            <label htmlFor="trailing_stop" className="text-sm text-gray-300 flex items-center gap-2">
              Трейлинг-стоп
              <Tooltip content="Автоматически подтягивает стоп-лосс за ценой при движении в прибыль. Позволяет фиксировать больше прибыли на трендовых движениях, защищая от разворотов." />
            </label>
          </div>
        </div>
      </Card>

      {/* Настройки биржи */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Настройки биржи
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Комиссия биржи (%)
              <Tooltip content="Комиссия, которую взимает биржа с каждой сделки. Для Binance обычно 0.06% (0.0006). Влияет на итоговую прибыльность - учитывайте её в расчётах." />
            </label>
            <Input
              type="number"
              step="0.0001"
              value={formData.commission_rate * 100}
              onChange={(e) => setFormData({ ...formData, commission_rate: parseFloat(e.target.value) / 100 })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Проскальзывание (%)
              <Tooltip content="Разница между ожидаемой и фактической ценой исполнения ордера. Диапазон: 0.001-0.1%. Обычно 0.01-0.05%. Учитывает реальные рыночные условия при быстрой торговле." />
            </label>
            <Input
              type="number"
              step="0.001"
              value={formData.slippage_pct}
              onChange={(e) => setFormData({ ...formData, slippage_pct: parseFloat(e.target.value) })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Модель проскальзывания
              <Tooltip content="Как рассчитывать проскальзывание:\n• Фиксированное - постоянное значение\n• На основе объёма - зависит от размера ордера\n• Процентное - процент от цены входа" />
            </label>
            <select
              value={formData.slippage_model}
              onChange={(e) => setFormData({ ...formData, slippage_model: e.target.value as any })}
              className="w-full bg-gray-800 border border-gray-700 text-white rounded-lg px-4 py-2"
            >
              <option value="fixed">Фиксированное</option>
              <option value="volume_based">На основе объёма</option>
              <option value="percentage">Процентное</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Кнопка запуска */}
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
