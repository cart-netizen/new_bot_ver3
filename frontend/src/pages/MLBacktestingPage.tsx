// frontend/src/pages/MLBacktestingPage.tsx

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Play,
  RefreshCw,
  Trash2,
  XCircle,
  ChevronRight,
  Plus,
  Brain,
  Target,
  TrendingUp,
  BarChart3,
  Activity,
  Grid3X3,
  LineChart,
  AlertCircle,
  Shuffle,
  Zap,
  Shield
} from 'lucide-react';
import { toast } from 'sonner';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Tooltip } from '../components/ui/Tooltip';
import { cn } from '../utils/helpers';
import * as mlBacktestingApi from '../api/ml-backtesting.api';

// Chart Components
import { EquityCurveChart } from '../components/ml-backtesting/EquityCurveChart';
import { MonteCarloChart } from '../components/ml-backtesting/MonteCarloChart';
import { WalkForwardChart } from '../components/ml-backtesting/WalkForwardChart';
import { PBOAnalysisCard } from '../components/ml-backtesting/PBOAnalysisCard';
import { RegimeAnalysisCard } from '../components/ml-backtesting/RegimeAnalysisCard';
import { ConfusionMatrixHeatmap } from '../components/ml-backtesting/ConfusionMatrixHeatmap';

interface ApiError {
  response?: {
    data?: {
      detail?: string;
    };
  };
}

export function MLBacktestingPage() {
  const [selectedView, setSelectedView] = useState<'list' | 'create' | 'results'>('list');
  const [selectedBacktestId, setSelectedBacktestId] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Fetch backtests list
  const { data: backtestsData, isLoading: isLoadingList, refetch: refetchList } = useQuery({
    queryKey: ['ml-backtests'],
    queryFn: () => mlBacktestingApi.listMLBacktests(),
    refetchInterval: 5000
  });

  // Fetch selected backtest details
  const { data: backtestDetails, isLoading: isLoadingDetails } = useQuery({
    queryKey: ['ml-backtest', selectedBacktestId],
    queryFn: () => mlBacktestingApi.getMLBacktest(selectedBacktestId!, true),
    enabled: !!selectedBacktestId && selectedView === 'results'
  });

  // Create mutation
  const createMutation = useMutation({
    mutationFn: mlBacktestingApi.createMLBacktest,
    onSuccess: () => {
      toast.success('ML Backtest started successfully!');
      queryClient.invalidateQueries({ queryKey: ['ml-backtests'] });
      setSelectedView('list');
    },
    onError: (error: ApiError) => {
      toast.error(error.response?.data?.detail || 'Failed to create ML backtest');
    }
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: mlBacktestingApi.deleteMLBacktest,
    onSuccess: () => {
      toast.success('ML Backtest deleted');
      queryClient.invalidateQueries({ queryKey: ['ml-backtests'] });
    },
    onError: (error: ApiError) => {
      toast.error(error.response?.data?.detail || 'Failed to delete ML backtest');
    }
  });

  // Cancel mutation
  const cancelMutation = useMutation({
    mutationFn: mlBacktestingApi.cancelMLBacktest,
    onSuccess: () => {
      toast.success('ML Backtest cancelled');
      queryClient.invalidateQueries({ queryKey: ['ml-backtests'] });
    },
    onError: (error: ApiError) => {
      toast.error(error.response?.data?.detail || 'Failed to cancel ML backtest');
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
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Brain className="h-8 w-8 text-purple-400" />
            ML Бэктестинг
          </h1>
          <p className="text-gray-400 mt-1">
            Тестирование ML моделей на исторических данных с walk-forward валидацией
          </p>
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
                Новый тест
              </Button>
            </>
          )}
          {selectedView !== 'list' && (
            <Button onClick={handleBackToList} variant="outline">
              Назад к списку
            </Button>
          )}
        </div>
      </div>

      {/* Main Content */}
      {selectedView === 'list' && (
        <MLBacktestsList
          backtests={backtestsData?.runs || []}
          isLoading={isLoadingList}
          onViewResults={handleViewResults}
          onDelete={(id) => deleteMutation.mutate(id)}
          onCancel={(id) => cancelMutation.mutate(id)}
        />
      )}

      {selectedView === 'create' && (
        <MLBacktestForm
          onSubmit={(config) => createMutation.mutate(config)}
          isSubmitting={createMutation.isPending}
        />
      )}

      {selectedView === 'results' && backtestDetails && (
        <MLBacktestResults
          backtest={backtestDetails}
          isLoading={isLoadingDetails}
        />
      )}
    </div>
  );
}

// ============================================================
// ML Backtests List Component
// ============================================================

interface MLBacktestsListProps {
  backtests: mlBacktestingApi.MLBacktestRun[];
  isLoading: boolean;
  onViewResults: (id: string) => void;
  onDelete: (id: string) => void;
  onCancel: (id: string) => void;
}

function MLBacktestsList({ backtests, isLoading, onViewResults, onDelete, onCancel }: MLBacktestsListProps) {
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
        <Brain className="h-12 w-12 text-gray-600 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-white mb-2">Нет ML бэктестов</h3>
        <p className="text-gray-400">Создайте первый тест для оценки ML модели</p>
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
        <Card key={backtest.id} className="p-6 hover:border-purple-500/30 transition-colors">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <h3 className="text-lg font-semibold text-white">{backtest.name}</h3>
                <span className={cn("text-xs font-medium px-2 py-1 rounded", getStatusColor(backtest.status))}>
                  {backtest.status.toUpperCase()}
                </span>
                {backtest.status === 'running' && backtest.progress_pct !== undefined && (
                  <span className="text-sm text-gray-400">
                    {backtest.progress_pct.toFixed(1)}%
                  </span>
                )}
              </div>
              {backtest.description && (
                <p className="text-sm text-gray-400 mb-3">{backtest.description}</p>
              )}

              <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
                <div>
                  <p className="text-gray-400">Модель</p>
                  <p className="text-white font-medium truncate">
                    {backtest.model_architecture || backtest.model_checkpoint.split('/').pop()}
                  </p>
                </div>
                <div>
                  <p className="text-gray-400">Accuracy</p>
                  <p className="text-white font-medium">
                    {backtest.accuracy ? `${(backtest.accuracy * 100).toFixed(1)}%` : 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-gray-400">F1 (macro)</p>
                  <p className="text-white font-medium">
                    {backtest.f1_macro ? `${(backtest.f1_macro * 100).toFixed(1)}%` : 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-gray-400">Sharpe</p>
                  <p className={cn(
                    "font-medium",
                    backtest.sharpe_ratio && backtest.sharpe_ratio > 0 ? "text-green-400" : "text-red-400"
                  )}>
                    {backtest.sharpe_ratio?.toFixed(2) || 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-gray-400">P&L</p>
                  <p className={cn(
                    "font-medium",
                    backtest.total_pnl_percent && backtest.total_pnl_percent >= 0 ? "text-green-400" : "text-red-400"
                  )}>
                    {backtest.total_pnl_percent !== undefined
                      ? `${(backtest.total_pnl_percent * 100).toFixed(1)}%`
                      : 'N/A'}
                  </p>
                </div>
              </div>
            </div>

            <div className="flex gap-2 ml-4">
              {backtest.status === 'completed' && (
                <Button onClick={() => onViewResults(backtest.id)} className="px-3 py-1.5 text-sm">
                  <ChevronRight className="h-4 w-4 mr-1" />
                  Результаты
                </Button>
              )}
              {backtest.status === 'running' && (
                <Button onClick={() => onCancel(backtest.id)} variant="outline" className="px-3 py-1.5 text-sm">
                  <XCircle className="h-4 w-4 mr-1" />
                  Отменить
                </Button>
              )}
              {['completed', 'failed', 'cancelled'].includes(backtest.status) && (
                <Button onClick={() => onDelete(backtest.id)} variant="outline" className="px-3 py-1.5 text-sm">
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
// ML Backtest Form Component
// ============================================================

interface MLBacktestFormProps {
  onSubmit: (config: mlBacktestingApi.MLBacktestConfig) => void;
  isSubmitting: boolean;
}

function MLBacktestForm({ onSubmit, isSubmitting }: MLBacktestFormProps) {
  const [formData, setFormData] = useState<mlBacktestingApi.MLBacktestConfig>(
    mlBacktestingApi.getDefaultMLBacktestConfig()
  );

  // Holdout set creation state
  const [selectedFolders, setSelectedFolders] = useState<string[]>([]);
  const [holdoutName, setHoldoutName] = useState('');
  const [isCreatingHoldout, setIsCreatingHoldout] = useState(false);
  const queryClient = useQueryClient();

  // Fetch available models
  const { data: modelsData } = useQuery({
    queryKey: ['ml-models'],
    queryFn: mlBacktestingApi.listAvailableModels
  });

  // Fetch data folders for holdout creation
  const { data: foldersData, isLoading: isLoadingFolders } = useQuery({
    queryKey: ['data-folders'],
    queryFn: mlBacktestingApi.listDataFolders,
    enabled: formData.data_source === 'holdout'
  });

  // Fetch available holdout sets
  const { data: holdoutSetsData, isLoading: isLoadingHoldoutSets, refetch: refetchHoldoutSets } = useQuery({
    queryKey: ['holdout-sets'],
    queryFn: mlBacktestingApi.listHoldoutSets,
    enabled: formData.data_source === 'holdout'
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.name) {
      toast.error('Введите название теста');
      return;
    }
    if (!formData.model_checkpoint) {
      toast.error('Выберите модель');
      return;
    }
    if (formData.data_source === 'holdout' && !formData.holdout_set_id) {
      toast.error('Выберите или создайте Holdout Set');
      return;
    }

    onSubmit(formData);
  };

  const updateForm = (updates: Partial<mlBacktestingApi.MLBacktestConfig>) => {
    setFormData(prev => ({ ...prev, ...updates }));
  };

  // Toggle folder selection
  const toggleFolder = (path: string) => {
    setSelectedFolders(prev =>
      prev.includes(path)
        ? prev.filter(p => p !== path)
        : [...prev, path]
    );
  };

  // Create holdout set
  const handleCreateHoldout = async () => {
    if (!holdoutName.trim()) {
      toast.error('Введите название для Holdout Set');
      return;
    }
    if (selectedFolders.length === 0) {
      toast.error('Выберите хотя бы одну папку с данными');
      return;
    }

    setIsCreatingHoldout(true);
    try {
      const result = await mlBacktestingApi.createHoldoutSet({
        name: holdoutName,
        source_paths: selectedFolders,
        sequence_length: formData.sequence_length
      });

      toast.success(`Holdout Set создан: ${result.samples.toLocaleString()} samples`);

      // Refresh holdout sets list
      await refetchHoldoutSets();

      // Auto-select newly created holdout set
      updateForm({ holdout_set_id: result.path });

      // Reset form
      setSelectedFolders([]);
      setHoldoutName('');
    } catch (error) {
      const apiError = error as ApiError;
      toast.error(apiError.response?.data?.detail || 'Ошибка создания Holdout Set');
    } finally {
      setIsCreatingHoldout(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Basic Info */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
          <Brain className="h-5 w-5 text-purple-400" />
          Основная информация
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Название теста *
            </label>
            <Input
              value={formData.name}
              onChange={(e) => updateForm({ name: e.target.value })}
              placeholder="ML Backtest v2.0"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Описание
            </label>
            <Input
              value={formData.description || ''}
              onChange={(e) => updateForm({ description: e.target.value })}
              placeholder="Тестирование модели на holdout данных"
            />
          </div>
        </div>
      </Card>

      {/* Model Selection */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
          <Target className="h-5 w-5 text-purple-400" />
          Выбор модели
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Checkpoint модели *
            </label>
            <select
              value={formData.model_checkpoint}
              onChange={(e) => updateForm({ model_checkpoint: e.target.value })}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
              required
            >
              <option value="">Выберите модель...</option>
              {modelsData?.models.map((model, idx) => (
                <option key={idx} value={model.checkpoint_path}>
                  {model.checkpoint_path.split('/').pop()}
                  {model.architecture && ` (${model.architecture})`}
                </option>
              ))}
              <option value="custom">Указать путь вручную...</option>
            </select>
          </div>
          {formData.model_checkpoint === 'custom' && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Путь к checkpoint
              </label>
              <Input
                value=""
                onChange={(e) => updateForm({ model_checkpoint: e.target.value })}
                placeholder="/path/to/model.pt"
              />
            </div>
          )}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Источник данных
              <Tooltip content="Holdout Set — данные, отложенные при обучении (рекомендуется для честного теста).
Feature Store — последние 30 дней из Feature Store.
Custom Data — загрузить свои данные (в разработке)." />
            </label>
            <select
              value={formData.data_source}
              onChange={(e) => updateForm({ data_source: e.target.value as mlBacktestingApi.MLBacktestConfig['data_source'] })}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            >
              <option value="holdout">Holdout Set</option>
              <option value="feature_store">Feature Store</option>
              <option value="custom">Custom Data</option>
            </select>
          </div>
        </div>

        {/* Holdout Set Management - показывается только при выборе holdout */}
        {formData.data_source === 'holdout' && (
          <div className="mt-6 space-y-6">
            {/* Выбор существующего Holdout Set */}
            <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
              <h3 className="text-lg font-medium text-white mb-3 flex items-center gap-2">
                <Shield className="h-5 w-5 text-green-400" />
                Выбрать Holdout Set
              </h3>

              {isLoadingHoldoutSets ? (
                <div className="text-gray-400 text-sm">Загрузка...</div>
              ) : holdoutSetsData?.holdout_sets && holdoutSetsData.holdout_sets.length > 0 ? (
                <div className="space-y-2">
                  <select
                    value={formData.holdout_set_id || ''}
                    onChange={(e) => updateForm({ holdout_set_id: e.target.value || undefined })}
                    className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  >
                    <option value="">Выберите Holdout Set...</option>
                    {holdoutSetsData.holdout_sets.map((set) => (
                      <option key={set.path} value={set.path}>
                        {set.name} ({set.sample_count?.toLocaleString() || '?'} samples, {set.size_mb} MB)
                      </option>
                    ))}
                  </select>
                  {formData.holdout_set_id && (
                    <p className="text-xs text-gray-500">
                      Путь: {formData.holdout_set_id}
                    </p>
                  )}
                </div>
              ) : (
                <div className="text-gray-400 text-sm">
                  Нет доступных Holdout Set. Создайте новый ниже.
                </div>
              )}
            </div>

            {/* Создание нового Holdout Set */}
            <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
              <h3 className="text-lg font-medium text-white mb-3 flex items-center gap-2">
                <Plus className="h-5 w-5 text-purple-400" />
                Создать новый Holdout Set
              </h3>

              {/* Название */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Название Holdout Set
                </label>
                <Input
                  value={holdoutName}
                  onChange={(e) => setHoldoutName(e.target.value)}
                  placeholder="my_holdout_set"
                />
              </div>

              {/* Выбор папок с данными */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Выберите папки с данными
                  <span className="text-gray-500 font-normal ml-2">
                    (выбрано: {selectedFolders.length})
                  </span>
                </label>

                {isLoadingFolders ? (
                  <div className="text-gray-400 text-sm">Загрузка папок...</div>
                ) : foldersData?.folders && foldersData.folders.length > 0 ? (
                  <div className="max-h-48 overflow-y-auto space-y-2 bg-gray-900 p-3 rounded-lg border border-gray-600">
                    {foldersData.folders.map((folder) => (
                      <label
                        key={folder.path}
                        className={cn(
                          "flex items-center gap-3 p-2 rounded cursor-pointer transition-colors",
                          selectedFolders.includes(folder.path)
                            ? "bg-purple-900/30 border border-purple-500"
                            : "hover:bg-gray-800 border border-transparent"
                        )}
                      >
                        <input
                          type="checkbox"
                          checked={selectedFolders.includes(folder.path)}
                          onChange={() => toggleFolder(folder.path)}
                          className="rounded border-gray-600 bg-gray-800 text-purple-500 focus:ring-purple-500"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium text-white truncate">
                            {folder.name}
                          </div>
                          <div className="text-xs text-gray-400">
                            {folder.files_count} файлов • {folder.size_mb} MB
                          </div>
                        </div>
                      </label>
                    ))}
                  </div>
                ) : (
                  <div className="text-gray-400 text-sm p-3 bg-gray-900 rounded-lg border border-gray-600">
                    Папки с parquet файлами не найдены в data/
                  </div>
                )}
              </div>

              {/* Кнопка создания */}
              <Button
                type="button"
                onClick={handleCreateHoldout}
                disabled={isCreatingHoldout || !holdoutName.trim() || selectedFolders.length === 0}
                className="w-full"
              >
                {isCreatingHoldout ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Создание...
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4 mr-2" />
                    Создать Holdout Set
                  </>
                )}
              </Button>
            </div>
          </div>
        )}
      </Card>

      {/* Walk-Forward Settings */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
          <LineChart className="h-5 w-5 text-purple-400" />
          Walk-Forward валидация
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="use_walk_forward"
              checked={formData.use_walk_forward}
              onChange={(e) => updateForm({ use_walk_forward: e.target.checked })}
              className="rounded border-gray-700 bg-gray-800 text-purple-500 focus:ring-purple-500"
            />
            <label htmlFor="use_walk_forward" className="text-sm text-gray-300 flex items-center gap-1">
              Использовать Walk-Forward
              <Tooltip content="Walk-Forward валидация разбивает данные на последовательные периоды и тестирует модель на каждом периоде отдельно. Это более реалистичный тест, имитирующий реальную торговлю во времени. Рекомендуется включить." />
            </label>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Количество периодов
              <Tooltip content="На сколько частей разбить данные. Больше периодов = более детальный анализ, но меньше данных в каждом периоде. Рекомендуется: 5-10 периодов." />
            </label>
            <Input
              type="number"
              min={2}
              max={20}
              value={formData.n_periods}
              onChange={(e) => updateForm({ n_periods: parseInt(e.target.value) })}
              disabled={!formData.use_walk_forward}
            />
          </div>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="retrain_each_period"
              checked={formData.retrain_each_period}
              onChange={(e) => updateForm({ retrain_each_period: e.target.checked })}
              disabled={!formData.use_walk_forward}
              className="rounded border-gray-700 bg-gray-800 text-purple-500 focus:ring-purple-500"
            />
            <label htmlFor="retrain_each_period" className="text-sm text-gray-300 flex items-center gap-1">
              Переобучать каждый период
              <Tooltip content="Если включено, модель будет переобучаться на каждом периоде с использованием данных предыдущих периодов. Более реалистично, но значительно дольше. В разработке." />
            </label>
          </div>
        </div>
      </Card>

      {/* Confidence Filtering */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5 text-purple-400" />
          Фильтрация по Confidence
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="use_confidence_filter"
              checked={formData.use_confidence_filter}
              onChange={(e) => updateForm({ use_confidence_filter: e.target.checked })}
              className="rounded border-gray-700 bg-gray-800 text-purple-500 focus:ring-purple-500"
            />
            <label htmlFor="use_confidence_filter" className="text-sm text-gray-300 flex items-center gap-1">
              Фильтровать по confidence
              <Tooltip content="Открывать сделки только когда модель уверена в предсказании выше заданного порога. Снижает количество сделок, но повышает их качество. Рекомендуется включить." />
            </label>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Минимальный confidence
              <Tooltip content="Порог уверенности модели (0.5-0.99). Модель откроет сделку только если уверенность выше этого значения. Рекомендуется: 0.6-0.7 для баланса между количеством и качеством сигналов." />
            </label>
            <Input
              type="number"
              min={0.5}
              max={0.99}
              step={0.05}
              value={formData.min_confidence}
              onChange={(e) => updateForm({ min_confidence: parseFloat(e.target.value) })}
              disabled={!formData.use_confidence_filter}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Режим фильтрации
              <Tooltip content="Фиксированный порог — простое отсечение по заданному значению.
Динамический — порог адаптируется к текущей волатильности.
Перцентиль — отбираются только топ-N% самых уверенных сигналов.
Рекомендуется: Фиксированный порог для начала." />
            </label>
            <select
              value={formData.confidence_mode}
              onChange={(e) => updateForm({ confidence_mode: e.target.value as mlBacktestingApi.MLBacktestConfig['confidence_mode'] })}
              disabled={!formData.use_confidence_filter}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-purple-500 focus:border-purple-500 disabled:opacity-50"
            >
              <option value="threshold">Фиксированный порог</option>
              <option value="dynamic">Динамический</option>
              <option value="percentile">Перцентиль</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Trading Simulation */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-purple-400" />
          Симуляция торговли
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Начальный капитал ($)
              <Tooltip content="Виртуальный начальный капитал для симуляции. Используется для расчёта PnL и Sharpe Ratio. Рекомендуется: 10000$ для удобного масштабирования." />
            </label>
            <Input
              type="number"
              min={100}
              step={100}
              value={formData.initial_capital}
              onChange={(e) => updateForm({ initial_capital: parseFloat(e.target.value) })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Размер позиции (%)
              <Tooltip content="Какую долю капитала использовать в каждой сделке (0.01 = 1%, 0.1 = 10%, 1.0 = 100%). Меньше = консервативнее, больше = агрессивнее. Рекомендуется: 0.1 (10%)." />
            </label>
            <Input
              type="number"
              min={0.01}
              max={1}
              step={0.01}
              value={formData.position_size}
              onChange={(e) => updateForm({ position_size: parseFloat(e.target.value) })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Комиссия (%)
              <Tooltip content="Комиссия биржи за сделку (0.001 = 0.1%). Для Binance: Maker 0.02-0.1%, Taker 0.04-0.1%. Рекомендуется: 0.001 (0.1%)." />
            </label>
            <Input
              type="number"
              min={0}
              max={0.1}
              step={0.0001}
              value={formData.commission}
              onChange={(e) => updateForm({ commission: parseFloat(e.target.value) })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Slippage (%)
              <Tooltip content="Проскальзывание — разница между ожидаемой и реальной ценой исполнения. На волатильном рынке может достигать 0.05-0.1%. Рекомендуется: 0.0005 (0.05%)." />
            </label>
            <Input
              type="number"
              min={0}
              max={0.1}
              step={0.0001}
              value={formData.slippage}
              onChange={(e) => updateForm({ slippage: parseFloat(e.target.value) })}
            />
          </div>
        </div>
      </Card>

      {/* Inference Settings */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-purple-400" />
          Настройки инференса
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Длина последовательности
              <Tooltip content="Сколько последних свечей/точек данных подаётся на вход модели. Должно совпадать со значением при обучении модели. Обычно: 60 (последние 60 минут/свечей)." />
            </label>
            <Input
              type="number"
              min={10}
              max={200}
              value={formData.sequence_length}
              onChange={(e) => updateForm({ sequence_length: parseInt(e.target.value) })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Batch size
              <Tooltip content="Количество примеров, обрабатываемых за раз. Больше = быстрее (если хватает памяти GPU), но требует больше VRAM. Рекомендуется: 128 для GPU, 32-64 для CPU." />
            </label>
            <Input
              type="number"
              min={16}
              max={512}
              value={formData.batch_size}
              onChange={(e) => updateForm({ batch_size: parseInt(e.target.value) })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              Device
              <Tooltip content="Auto — автоматически выберет GPU если доступен, иначе CPU.
CUDA — принудительно использовать GPU (ошибка если нет).
CPU — только процессор (медленнее, но стабильнее).
Рекомендуется: Auto." />
            </label>
            <select
              value={formData.device}
              onChange={(e) => updateForm({ device: e.target.value as mlBacktestingApi.MLBacktestConfig['device'] })}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            >
              <option value="auto">Auto (GPU если доступен)</option>
              <option value="cuda">CUDA (GPU)</option>
              <option value="cpu">CPU</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Submit */}
      <Card className="p-6">
        <div className="flex justify-end gap-4">
          <Button type="submit" disabled={isSubmitting}>
            {isSubmitting ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Запуск...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Запустить ML бэктест
              </>
            )}
          </Button>
        </div>
      </Card>
    </form>
  );
}

// ============================================================
// ML Backtest Results Component
// ============================================================

interface MLBacktestResultsProps {
  backtest: mlBacktestingApi.MLBacktestRun & { predictions?: mlBacktestingApi.Prediction[] };
  isLoading: boolean;
}

type ResultsTab = 'overview' | 'classification' | 'trading' | 'walk_forward' | 'equity' | 'monte_carlo' | 'pbo' | 'regimes' | 'predictions';

function MLBacktestResults({ backtest, isLoading }: MLBacktestResultsProps) {
  const [activeTab, setActiveTab] = useState<ResultsTab>('overview');

  if (isLoading) {
    return (
      <Card className="p-8 text-center">
        <RefreshCw className="h-8 w-8 animate-spin text-gray-400 mx-auto mb-2" />
        <p className="text-gray-400">Загрузка результатов...</p>
      </Card>
    );
  }

  const tabConfig: { key: ResultsTab; label: string; icon: typeof TrendingUp }[] = [
    { key: 'overview', label: 'Обзор', icon: Grid3X3 },
    { key: 'classification', label: 'Классификация', icon: Target },
    { key: 'trading', label: 'Trading', icon: TrendingUp },
    { key: 'walk_forward', label: 'Walk-Forward', icon: LineChart },
    { key: 'equity', label: 'Equity Curve', icon: Activity },
    { key: 'monte_carlo', label: 'Monte Carlo', icon: Shuffle },
    { key: 'pbo', label: 'PBO Analysis', icon: Shield },
    { key: 'regimes', label: 'Regimes', icon: Zap },
    { key: 'predictions', label: 'Predictions', icon: BarChart3 },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="p-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              <Brain className="h-6 w-6 text-purple-400" />
              {backtest.name}
            </h2>
            {backtest.description && (
              <p className="text-gray-400 mt-1">{backtest.description}</p>
            )}
          </div>
          <span className="text-xs font-medium px-3 py-1 rounded bg-green-500/10 text-green-400">
            COMPLETED
          </span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
          <div>
            <p className="text-gray-400">Модель</p>
            <p className="text-white font-medium">{backtest.model_architecture || 'N/A'}</p>
          </div>
          <div>
            <p className="text-gray-400">Samples</p>
            <p className="text-white font-medium">{backtest.total_samples?.toLocaleString() || 'N/A'}</p>
          </div>
          <div>
            <p className="text-gray-400">Длительность</p>
            <p className="text-white font-medium">
              {backtest.duration_seconds ? `${backtest.duration_seconds.toFixed(1)}s` : 'N/A'}
            </p>
          </div>
          <div>
            <p className="text-gray-400">Walk-Forward</p>
            <p className="text-white font-medium">
              {backtest.use_walk_forward ? `${backtest.n_periods} периодов` : 'Отключен'}
            </p>
          </div>
          <div>
            <p className="text-gray-400">Confidence Filter</p>
            <p className="text-white font-medium">
              {backtest.use_confidence_filter ? `>= ${(backtest.min_confidence || 0.6) * 100}%` : 'Отключен'}
            </p>
          </div>
        </div>
      </Card>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-800 overflow-x-auto pb-1">
        {tabConfig.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={cn(
              "flex items-center gap-1.5 px-3 py-2 font-medium transition-colors whitespace-nowrap rounded-t-lg text-sm",
              activeTab === key
                ? "text-purple-400 bg-purple-500/10 border-b-2 border-purple-400"
                : "text-gray-400 hover:text-white hover:bg-gray-800/50"
            )}
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && <OverviewTab backtest={backtest} />}
      {activeTab === 'classification' && <ClassificationTab backtest={backtest} />}
      {activeTab === 'trading' && <TradingTab backtest={backtest} />}
      {activeTab === 'walk_forward' && <WalkForwardTab backtest={backtest} />}
      {activeTab === 'equity' && <EquityCurveTab backtestId={backtest.id} />}
      {activeTab === 'monte_carlo' && <MonteCarloTab backtestId={backtest.id} />}
      {activeTab === 'pbo' && <PBOTab backtestId={backtest.id} />}
      {activeTab === 'regimes' && <RegimesTab backtestId={backtest.id} />}
      {activeTab === 'predictions' && <PredictionsTab backtest={backtest} />}
    </div>
  );
}

// ============================================================
// Tab Components
// ============================================================

function OverviewTab({ backtest }: { backtest: mlBacktestingApi.MLBacktestRun }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Classification Summary */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Grid3X3 className="h-5 w-5 text-blue-400" />
          Classification Метрики
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center p-4 bg-gray-800/50 rounded-lg">
            <p className="text-3xl font-bold text-white">
              {backtest.accuracy ? `${(backtest.accuracy * 100).toFixed(1)}%` : 'N/A'}
            </p>
            <p className="text-sm text-gray-400">Accuracy</p>
          </div>
          <div className="text-center p-4 bg-gray-800/50 rounded-lg">
            <p className="text-3xl font-bold text-white">
              {backtest.f1_macro ? `${(backtest.f1_macro * 100).toFixed(1)}%` : 'N/A'}
            </p>
            <p className="text-sm text-gray-400">F1 (macro)</p>
          </div>
          <div className="text-center p-4 bg-gray-800/50 rounded-lg">
            <p className="text-3xl font-bold text-white">
              {backtest.precision_macro ? `${(backtest.precision_macro * 100).toFixed(1)}%` : 'N/A'}
            </p>
            <p className="text-sm text-gray-400">Precision (macro)</p>
          </div>
          <div className="text-center p-4 bg-gray-800/50 rounded-lg">
            <p className="text-3xl font-bold text-white">
              {backtest.recall_macro ? `${(backtest.recall_macro * 100).toFixed(1)}%` : 'N/A'}
            </p>
            <p className="text-sm text-gray-400">Recall (macro)</p>
          </div>
        </div>
      </Card>

      {/* Trading Summary */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-green-400" />
          Trading Метрики
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center p-4 bg-gray-800/50 rounded-lg">
            <p className={cn(
              "text-3xl font-bold",
              backtest.total_pnl_percent && backtest.total_pnl_percent >= 0 ? "text-green-400" : "text-red-400"
            )}>
              {backtest.total_pnl_percent !== undefined
                ? `${(backtest.total_pnl_percent * 100).toFixed(1)}%`
                : 'N/A'}
            </p>
            <p className="text-sm text-gray-400">Total P&L</p>
          </div>
          <div className="text-center p-4 bg-gray-800/50 rounded-lg">
            <p className={cn(
              "text-3xl font-bold",
              backtest.sharpe_ratio && backtest.sharpe_ratio > 1 ? "text-green-400" :
              backtest.sharpe_ratio && backtest.sharpe_ratio > 0 ? "text-yellow-400" : "text-red-400"
            )}>
              {backtest.sharpe_ratio?.toFixed(2) || 'N/A'}
            </p>
            <p className="text-sm text-gray-400">Sharpe Ratio</p>
          </div>
          <div className="text-center p-4 bg-gray-800/50 rounded-lg">
            <p className="text-3xl font-bold text-white">
              {backtest.win_rate !== undefined ? `${(backtest.win_rate * 100).toFixed(1)}%` : 'N/A'}
            </p>
            <p className="text-sm text-gray-400">Win Rate</p>
          </div>
          <div className="text-center p-4 bg-gray-800/50 rounded-lg">
            <p className="text-3xl font-bold text-red-400">
              {backtest.max_drawdown !== undefined ? `${(backtest.max_drawdown * 100).toFixed(1)}%` : 'N/A'}
            </p>
            <p className="text-sm text-gray-400">Max Drawdown</p>
          </div>
        </div>
      </Card>
    </div>
  );
}

function ClassificationTab({ backtest }: { backtest: mlBacktestingApi.MLBacktestRun }) {
  return (
    <div className="space-y-6">
      {/* Confusion Matrix Heatmap */}
      {backtest.confusion_matrix && (
        <ConfusionMatrixHeatmap
          matrix={backtest.confusion_matrix}
          labels={['SELL', 'HOLD', 'BUY']}
          title="Confusion Matrix"
          showPercentages={true}
        />
      )}

      {/* Macro Metrics Summary */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Macro Metrics Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 bg-gray-800/50 rounded-lg text-center">
            <p className="text-2xl font-bold text-purple-400">
              {backtest.accuracy ? `${(backtest.accuracy * 100).toFixed(1)}%` : 'N/A'}
            </p>
            <p className="text-sm text-gray-400">Accuracy</p>
          </div>
          <div className="p-4 bg-gray-800/50 rounded-lg text-center">
            <p className="text-2xl font-bold text-cyan-400">
              {backtest.precision_macro ? `${(backtest.precision_macro * 100).toFixed(1)}%` : 'N/A'}
            </p>
            <p className="text-sm text-gray-400">Precision (macro)</p>
          </div>
          <div className="p-4 bg-gray-800/50 rounded-lg text-center">
            <p className="text-2xl font-bold text-green-400">
              {backtest.recall_macro ? `${(backtest.recall_macro * 100).toFixed(1)}%` : 'N/A'}
            </p>
            <p className="text-sm text-gray-400">Recall (macro)</p>
          </div>
          <div className="p-4 bg-gray-800/50 rounded-lg text-center">
            <p className="text-2xl font-bold text-yellow-400">
              {backtest.f1_macro ? `${(backtest.f1_macro * 100).toFixed(1)}%` : 'N/A'}
            </p>
            <p className="text-sm text-gray-400">F1 (macro)</p>
          </div>
        </div>
      </Card>
    </div>
  );
}

function TradingTab({ backtest }: { backtest: mlBacktestingApi.MLBacktestRun }) {
  return (
    <div className="space-y-6">
      {/* Trading Stats */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Trading Statistics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 bg-gray-800/50 rounded-lg">
            <p className="text-2xl font-bold text-white">{backtest.total_trades || 0}</p>
            <p className="text-sm text-gray-400">Total Trades</p>
          </div>
          <div className="p-4 bg-gray-800/50 rounded-lg">
            <p className="text-2xl font-bold text-green-400">{backtest.winning_trades || 0}</p>
            <p className="text-sm text-gray-400">Winning</p>
          </div>
          <div className="p-4 bg-gray-800/50 rounded-lg">
            <p className="text-2xl font-bold text-red-400">{backtest.losing_trades || 0}</p>
            <p className="text-sm text-gray-400">Losing</p>
          </div>
          <div className="p-4 bg-gray-800/50 rounded-lg">
            <p className="text-2xl font-bold text-white">
              {backtest.profit_factor?.toFixed(2) || 'N/A'}
            </p>
            <p className="text-sm text-gray-400">Profit Factor</p>
          </div>
        </div>
      </Card>

      {/* Capital */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Capital</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="p-4 bg-gray-800/50 rounded-lg">
            <p className="text-2xl font-bold text-white">
              ${backtest.initial_capital?.toLocaleString() || 'N/A'}
            </p>
            <p className="text-sm text-gray-400">Initial Capital</p>
          </div>
          <div className="p-4 bg-gray-800/50 rounded-lg">
            <p className={cn(
              "text-2xl font-bold",
              (backtest.final_capital || 0) >= (backtest.initial_capital || 0)
                ? "text-green-400" : "text-red-400"
            )}>
              ${backtest.final_capital?.toLocaleString() || 'N/A'}
            </p>
            <p className="text-sm text-gray-400">Final Capital</p>
          </div>
          <div className="p-4 bg-gray-800/50 rounded-lg">
            <p className={cn(
              "text-2xl font-bold",
              (backtest.total_pnl || 0) >= 0 ? "text-green-400" : "text-red-400"
            )}>
              ${backtest.total_pnl?.toFixed(2) || 'N/A'}
            </p>
            <p className="text-sm text-gray-400">Total P&L</p>
          </div>
        </div>
      </Card>
    </div>
  );
}

function WalkForwardTab({ backtest }: { backtest: mlBacktestingApi.MLBacktestRun }) {
  if (!backtest.period_results || backtest.period_results.length === 0) {
    return (
      <Card className="p-8 text-center">
        <AlertCircle className="h-12 w-12 text-gray-600 mx-auto mb-4" />
        <p className="text-gray-400">Walk-Forward не был включен для этого теста</p>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Walk-Forward Chart */}
      <WalkForwardChart
        periodResults={backtest.period_results}
        threshold={backtest.min_confidence || 0.6}
      />

      {/* Summary Statistics */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Walk-Forward Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 bg-gray-800/50 rounded-lg text-center">
            <p className="text-2xl font-bold text-purple-400">{backtest.n_periods || backtest.period_results.length}</p>
            <p className="text-sm text-gray-400">Total Periods</p>
          </div>
          <div className="p-4 bg-gray-800/50 rounded-lg text-center">
            <p className="text-2xl font-bold text-cyan-400">
              {backtest.period_results.reduce((a, b) => a + b.samples, 0).toLocaleString()}
            </p>
            <p className="text-sm text-gray-400">Total Samples</p>
          </div>
          <div className="p-4 bg-gray-800/50 rounded-lg text-center">
            <p className="text-2xl font-bold text-green-400">
              {(backtest.period_results.reduce((a, b) => a + b.accuracy, 0) / backtest.period_results.length * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-gray-400">Avg Accuracy</p>
          </div>
          <div className="p-4 bg-gray-800/50 rounded-lg text-center">
            <p className="text-2xl font-bold text-yellow-400">
              {(backtest.period_results.reduce((a, b) => a + b.f1_macro, 0) / backtest.period_results.length * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-gray-400">Avg F1 Score</p>
          </div>
        </div>
      </Card>
    </div>
  );
}

// New Tab Components for Advanced Analysis

function EquityCurveTab({ backtestId }: { backtestId: string }) {
  return (
    <div className="space-y-6">
      <EquityCurveChart backtestId={backtestId} />
    </div>
  );
}

function MonteCarloTab({ backtestId }: { backtestId: string }) {
  return (
    <div className="space-y-6">
      <MonteCarloChart backtestId={backtestId} />
    </div>
  );
}

function PBOTab({ backtestId }: { backtestId: string }) {
  return (
    <div className="space-y-6">
      <PBOAnalysisCard backtestId={backtestId} />
    </div>
  );
}

function RegimesTab({ backtestId }: { backtestId: string }) {
  return (
    <div className="space-y-6">
      <RegimeAnalysisCard backtestId={backtestId} />
    </div>
  );
}

function PredictionsTab({ backtest }: { backtest: mlBacktestingApi.MLBacktestRun & { predictions?: mlBacktestingApi.Prediction[] } }) {
  const predictions = backtest.predictions || [];

  if (predictions.length === 0) {
    return (
      <Card className="p-8 text-center">
        <AlertCircle className="h-12 w-12 text-gray-600 mx-auto mb-4" />
        <p className="text-gray-400">Нет данных о предсказаниях</p>
      </Card>
    );
  }

  // Calculate confidence distribution
  const confBins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  const confDistribution = confBins.slice(0, -1).map((min, idx) => {
    const max = confBins[idx + 1];
    const count = predictions.filter(p => p.confidence >= min && p.confidence < max).length;
    return { range: `${(min * 100).toFixed(0)}-${(max * 100).toFixed(0)}%`, count };
  });

  return (
    <div className="space-y-6">
      {/* Confidence Distribution */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Распределение Confidence</h3>
        <div className="grid grid-cols-5 gap-2">
          {confDistribution.map((bin, idx) => (
            <div key={idx} className="text-center p-3 bg-gray-800/50 rounded-lg">
              <p className="text-lg font-bold text-white">{bin.count}</p>
              <p className="text-xs text-gray-400">{bin.range}</p>
            </div>
          ))}
        </div>
      </Card>

      {/* Sample Predictions */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Примеры предсказаний (первые 100)</h3>
        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-gray-900">
              <tr className="border-b border-gray-700">
                <th className="text-left p-2 text-gray-400">#</th>
                <th className="text-center p-2 text-gray-400">Predicted</th>
                <th className="text-center p-2 text-gray-400">Actual</th>
                <th className="text-center p-2 text-gray-400">Confidence</th>
                <th className="text-center p-2 text-gray-400">Correct</th>
              </tr>
            </thead>
            <tbody>
              {predictions.slice(0, 100).map((pred, idx) => {
                const isCorrect = pred.predicted_class === pred.actual_class;
                return (
                  <tr key={idx} className="border-b border-gray-800">
                    <td className="p-2 text-gray-500">{pred.sequence}</td>
                    <td className={cn(
                      "text-center p-2 font-medium",
                      pred.predicted_class === 0 ? 'text-red-400' :
                      pred.predicted_class === 1 ? 'text-yellow-400' : 'text-green-400'
                    )}>
                      {mlBacktestingApi.getClassName(pred.predicted_class)}
                    </td>
                    <td className={cn(
                      "text-center p-2 font-medium",
                      pred.actual_class === 0 ? 'text-red-400' :
                      pred.actual_class === 1 ? 'text-yellow-400' : 'text-green-400'
                    )}>
                      {mlBacktestingApi.getClassName(pred.actual_class)}
                    </td>
                    <td className="text-center p-2 text-white">
                      {(pred.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="text-center p-2">
                      {isCorrect ? (
                        <span className="text-green-400">Yes</span>
                      ) : (
                        <span className="text-red-400">No</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

export default MLBacktestingPage;
