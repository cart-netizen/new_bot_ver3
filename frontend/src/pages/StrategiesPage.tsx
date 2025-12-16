// frontend/src/pages/StrategiesPage.tsx

import { useState, useEffect, useCallback } from 'react';
import {
  Brain,
  Settings,
  ToggleLeft,
  ToggleRight,
  Sliders,
  RefreshCw,
  Save,
  AlertTriangle,
  CheckCircle,
  XCircle,
  TrendingUp,
  TrendingDown,
  Minus,
  BarChart3,
  Activity,
  Users,
  Target,
  Zap,
  Info
} from 'lucide-react';
import { cn } from '../utils/helpers';

/**
 * ============================================================
 * TYPES & INTERFACES
 * ============================================================
 */

interface EnsembleModel {
  name: string;
  display_name: string;
  weight: number;
  enabled: boolean;
  performance_score: number;
  is_registered: boolean;
  description: string;
}

interface EnsembleStatus {
  enabled: boolean;
  strategy: string;
  models: Record<string, {
    weight: number;
    enabled: boolean;
    performance_score: number;
  }>;
  stats: {
    total_predictions: number;
    unanimous_count: number;
    majority_count: number;
    conflict_count: number;
    trades_signaled: number;
    registered_models: string[];
  };
  config: {
    min_confidence_for_trade: number;
    unanimous_threshold: number;
    conflict_resolution: string;
    enable_adaptive_weights: boolean;
  };
}

interface StrategyInfo {
  current_strategy: string;
  available_strategies: string[];
  strategy_descriptions: Record<string, string>;
}

interface PerformanceStats {
  model_performance: Record<string, {
    weight: number;
    enabled: boolean;
    performance_score: number;
  }>;
  prediction_stats: {
    total_predictions: number;
    unanimous_ratio: number;
    majority_ratio: number;
    conflict_ratio: number;
    trade_ratio: number;
  };
}

/**
 * ============================================================
 * TOOLTIP COMPONENT
 * ============================================================
 */

interface TooltipProps {
  content: string;
  children: React.ReactNode;
}

function Tooltip({ content, children }: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div
      className="relative inline-block"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      {isVisible && (
        <div className="absolute z-50 px-3 py-2 text-sm text-white bg-gray-900 rounded-lg shadow-lg whitespace-nowrap bottom-full left-1/2 transform -translate-x-1/2 mb-2 border border-gray-700 max-w-xs">
          {content}
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1">
            <div className="border-4 border-transparent border-t-gray-900" />
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * ============================================================
 * MAIN COMPONENT
 * ============================================================
 */

export function StrategiesPage() {
  // ========== STATE ==========
  const [ensembleStatus, setEnsembleStatus] = useState<EnsembleStatus | null>(null);
  const [models, setModels] = useState<EnsembleModel[]>([]);
  const [strategyInfo, setStrategyInfo] = useState<StrategyInfo | null>(null);
  const [performanceStats, setPerformanceStats] = useState<PerformanceStats | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);

  // Config state
  const [selectedStrategy, setSelectedStrategy] = useState<string>('weighted_voting');
  const [configChanges, setConfigChanges] = useState<{
    min_confidence_for_trade?: number;
    unanimous_threshold?: number;
    enable_adaptive_weights?: boolean;
  }>({});

  /**
   * ============================================================
   * API FUNCTIONS
   * ============================================================
   */

  const fetchEnsembleStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/ensemble/status');
      if (!response.ok) throw new Error('Failed to fetch ensemble status');
      const data = await response.json();
      setEnsembleStatus(data);
      setSelectedStrategy(data.strategy);
    } catch (err) {
      console.error('Failed to fetch ensemble status:', err);
      setError('Failed to load ensemble status');
    }
  }, []);

  const fetchModels = useCallback(async () => {
    try {
      const response = await fetch('/api/ensemble/models');
      if (!response.ok) throw new Error('Failed to fetch models');
      const data = await response.json();
      setModels(data.models || []);
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  }, []);

  const fetchStrategyInfo = useCallback(async () => {
    try {
      const response = await fetch('/api/ensemble/strategy');
      if (!response.ok) throw new Error('Failed to fetch strategy info');
      const data = await response.json();
      setStrategyInfo(data);
    } catch (err) {
      console.error('Failed to fetch strategy info:', err);
    }
  }, []);

  const fetchPerformanceStats = useCallback(async () => {
    try {
      const response = await fetch('/api/ensemble/performance/stats');
      if (!response.ok) throw new Error('Failed to fetch performance stats');
      const data = await response.json();
      setPerformanceStats(data);
    } catch (err) {
      console.error('Failed to fetch performance stats:', err);
    }
  }, []);

  const refreshAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    await Promise.all([
      fetchEnsembleStatus(),
      fetchModels(),
      fetchStrategyInfo(),
      fetchPerformanceStats()
    ]);
    setLoading(false);
  }, [fetchEnsembleStatus, fetchModels, fetchStrategyInfo, fetchPerformanceStats]);

  // Initial load
  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchPerformanceStats();
    }, 30000);
    return () => clearInterval(interval);
  }, [fetchPerformanceStats]);

  /**
   * ============================================================
   * ACTION HANDLERS
   * ============================================================
   */

  const handleToggleModel = async (modelName: string, enabled: boolean) => {
    try {
      const response = await fetch('/api/ensemble/models/enable', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: modelName, enabled })
      });

      if (!response.ok) throw new Error('Failed to update model');

      setSuccess(`Model ${modelName} ${enabled ? 'enabled' : 'disabled'}`);
      setTimeout(() => setSuccess(null), 3000);

      await fetchModels();
      await fetchEnsembleStatus();
    } catch (err) {
      setError('Failed to toggle model');
    }
  };

  const handleUpdateWeight = async (modelName: string, weight: number) => {
    try {
      const response = await fetch('/api/ensemble/models/weight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: modelName, weight })
      });

      if (!response.ok) throw new Error('Failed to update weight');

      setSuccess(`Weight updated for ${modelName}`);
      setTimeout(() => setSuccess(null), 3000);

      await fetchModels();
      await fetchEnsembleStatus();
    } catch (err) {
      setError('Failed to update weight');
    }
  };

  const handleChangeStrategy = async (strategy: string) => {
    try {
      const response = await fetch('/api/ensemble/strategy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy })
      });

      if (!response.ok) throw new Error('Failed to update strategy');

      setSelectedStrategy(strategy);
      setSuccess(`Strategy changed to ${strategy}`);
      setTimeout(() => setSuccess(null), 3000);

      await fetchEnsembleStatus();
    } catch (err) {
      setError('Failed to change strategy');
    }
  };

  const handleSaveConfig = async () => {
    if (Object.keys(configChanges).length === 0) return;

    setSaving(true);
    try {
      const response = await fetch('/api/ensemble/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configChanges)
      });

      if (!response.ok) throw new Error('Failed to save config');

      // Save to disk
      await fetch('/api/ensemble/config/save', { method: 'POST' });

      setSuccess('Configuration saved');
      setTimeout(() => setSuccess(null), 3000);
      setConfigChanges({});

      await fetchEnsembleStatus();
    } catch (err) {
      setError('Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  /**
   * ============================================================
   * RENDER FUNCTIONS
   * ============================================================
   */

  const renderModelCard = (model: EnsembleModel) => {
    const isActive = model.enabled && model.is_registered;

    return (
      <div
        key={model.name}
        className={cn(
          'bg-surface border rounded-lg p-6 transition-all duration-200',
          isActive ? 'border-primary/50' : 'border-gray-800',
          !model.is_registered && 'opacity-60'
        )}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={cn(
              'p-2 rounded-lg',
              isActive ? 'bg-primary/20' : 'bg-gray-800'
            )}>
              <Brain className={cn(
                'h-6 w-6',
                isActive ? 'text-primary' : 'text-gray-500'
              )} />
            </div>
            <div>
              <h3 className="font-semibold text-white">{model.display_name}</h3>
              <p className="text-sm text-gray-400">{model.description}</p>
            </div>
          </div>

          {/* Toggle */}
          <button
            onClick={() => handleToggleModel(model.name, !model.enabled)}
            disabled={!model.is_registered}
            className={cn(
              'transition-colors',
              !model.is_registered && 'cursor-not-allowed'
            )}
          >
            {model.enabled ? (
              <ToggleRight className="h-8 w-8 text-primary" />
            ) : (
              <ToggleLeft className="h-8 w-8 text-gray-500" />
            )}
          </button>
        </div>

        {/* Status Badge */}
        <div className="flex items-center gap-2 mb-4">
          {model.is_registered ? (
            <span className="px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded-full flex items-center gap-1">
              <CheckCircle className="h-3 w-3" />
              Registered
            </span>
          ) : (
            <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 text-xs rounded-full flex items-center gap-1">
              <AlertTriangle className="h-3 w-3" />
              Not Trained
            </span>
          )}

          {model.enabled && (
            <span className="px-2 py-1 bg-blue-500/20 text-blue-400 text-xs rounded-full">
              Active
            </span>
          )}
        </div>

        {/* Weight Slider */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm text-gray-400">Weight</label>
            <span className="text-sm font-mono text-white">
              {(model.weight * 100).toFixed(0)}%
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={model.weight * 100}
            onChange={(e) => handleUpdateWeight(model.name, parseInt(e.target.value) / 100)}
            disabled={!model.is_registered}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary disabled:opacity-50 disabled:cursor-not-allowed"
          />
        </div>

        {/* Performance Score */}
        <div className="mt-4 pt-4 border-t border-gray-800">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400">Performance Score</span>
            <span className={cn(
              'text-sm font-semibold',
              model.performance_score >= 0.7 ? 'text-green-400' :
              model.performance_score >= 0.5 ? 'text-yellow-400' : 'text-red-400'
            )}>
              {(model.performance_score * 100).toFixed(1)}%
            </span>
          </div>
          <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className={cn(
                'h-full rounded-full transition-all',
                model.performance_score >= 0.7 ? 'bg-green-500' :
                model.performance_score >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'
              )}
              style={{ width: `${model.performance_score * 100}%` }}
            />
          </div>
        </div>
      </div>
    );
  };

  const renderStrategySelector = () => {
    if (!strategyInfo) return null;

    const strategyIcons: Record<string, React.ReactNode> = {
      weighted_voting: <Sliders className="h-5 w-5" />,
      unanimous: <Users className="h-5 w-5" />,
      majority: <BarChart3 className="h-5 w-5" />,
      confidence_based: <Target className="h-5 w-5" />,
      adaptive: <Zap className="h-5 w-5" />
    };

    return (
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-6">
          <Settings className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold">Consensus Strategy</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {strategyInfo.available_strategies.map(strategy => (
            <button
              key={strategy}
              onClick={() => handleChangeStrategy(strategy)}
              className={cn(
                'p-4 rounded-lg border transition-all text-left',
                selectedStrategy === strategy
                  ? 'border-primary bg-primary/10'
                  : 'border-gray-700 hover:border-gray-600 bg-gray-800/50'
              )}
            >
              <div className="flex items-center gap-3 mb-2">
                <div className={cn(
                  'p-2 rounded-lg',
                  selectedStrategy === strategy ? 'bg-primary/20 text-primary' : 'bg-gray-700 text-gray-400'
                )}>
                  {strategyIcons[strategy] || <Settings className="h-5 w-5" />}
                </div>
                <span className={cn(
                  'font-medium capitalize',
                  selectedStrategy === strategy ? 'text-primary' : 'text-white'
                )}>
                  {strategy.replace(/_/g, ' ')}
                </span>
              </div>
              <p className="text-sm text-gray-400">
                {strategyInfo.strategy_descriptions[strategy]}
              </p>
              {selectedStrategy === strategy && (
                <div className="mt-2 flex items-center gap-1 text-primary text-sm">
                  <CheckCircle className="h-4 w-4" />
                  Active
                </div>
              )}
            </button>
          ))}
        </div>
      </div>
    );
  };

  const renderConfigPanel = () => {
    if (!ensembleStatus) return null;

    return (
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Sliders className="h-6 w-6 text-primary" />
            <h2 className="text-xl font-semibold">Configuration</h2>
          </div>
          <button
            onClick={handleSaveConfig}
            disabled={saving || Object.keys(configChanges).length === 0}
            className={cn(
              'px-4 py-2 rounded-lg flex items-center gap-2 transition-colors',
              Object.keys(configChanges).length > 0
                ? 'bg-primary hover:bg-primary/80 text-white'
                : 'bg-gray-700 text-gray-400 cursor-not-allowed'
            )}
          >
            <Save className="h-4 w-4" />
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Min Confidence */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm text-gray-400 flex items-center gap-2">
                Min Confidence for Trade
                <Tooltip content="Minimum confidence level required to signal a trade">
                  <Info className="h-4 w-4 text-gray-500" />
                </Tooltip>
              </label>
              <span className="text-sm font-mono text-white">
                {((configChanges.min_confidence_for_trade ?? ensembleStatus.config.min_confidence_for_trade) * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={(configChanges.min_confidence_for_trade ?? ensembleStatus.config.min_confidence_for_trade) * 100}
              onChange={(e) => setConfigChanges(prev => ({
                ...prev,
                min_confidence_for_trade: parseInt(e.target.value) / 100
              }))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary"
            />
          </div>

          {/* Unanimous Threshold */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm text-gray-400 flex items-center gap-2">
                Unanimous Threshold
                <Tooltip content="Agreement threshold for unanimous strategy">
                  <Info className="h-4 w-4 text-gray-500" />
                </Tooltip>
              </label>
              <span className="text-sm font-mono text-white">
                {((configChanges.unanimous_threshold ?? ensembleStatus.config.unanimous_threshold) * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="50"
              max="100"
              value={(configChanges.unanimous_threshold ?? ensembleStatus.config.unanimous_threshold) * 100}
              onChange={(e) => setConfigChanges(prev => ({
                ...prev,
                unanimous_threshold: parseInt(e.target.value) / 100
              }))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary"
            />
          </div>

          {/* Adaptive Weights Toggle */}
          <div className="col-span-full flex items-center justify-between p-4 bg-gray-800/50 rounded-lg">
            <div>
              <p className="text-white font-medium">Adaptive Weights</p>
              <p className="text-sm text-gray-400">
                Automatically adjust model weights based on recent performance
              </p>
            </div>
            <button
              onClick={() => setConfigChanges(prev => ({
                ...prev,
                enable_adaptive_weights: !(configChanges.enable_adaptive_weights ?? ensembleStatus.config.enable_adaptive_weights)
              }))}
            >
              {(configChanges.enable_adaptive_weights ?? ensembleStatus.config.enable_adaptive_weights) ? (
                <ToggleRight className="h-8 w-8 text-primary" />
              ) : (
                <ToggleLeft className="h-8 w-8 text-gray-500" />
              )}
            </button>
          </div>
        </div>
      </div>
    );
  };

  const renderStatistics = () => {
    if (!performanceStats) return null;

    const stats = performanceStats.prediction_stats;

    const statCards = [
      {
        label: 'Total Predictions',
        value: stats.total_predictions,
        icon: Activity,
        color: 'text-blue-400'
      },
      {
        label: 'Unanimous Decisions',
        value: `${(stats.unanimous_ratio * 100).toFixed(1)}%`,
        icon: Users,
        color: 'text-green-400'
      },
      {
        label: 'Majority Decisions',
        value: `${(stats.majority_ratio * 100).toFixed(1)}%`,
        icon: BarChart3,
        color: 'text-yellow-400'
      },
      {
        label: 'Conflicts',
        value: `${(stats.conflict_ratio * 100).toFixed(1)}%`,
        icon: AlertTriangle,
        color: 'text-red-400'
      },
      {
        label: 'Trade Signals',
        value: `${(stats.trade_ratio * 100).toFixed(1)}%`,
        icon: TrendingUp,
        color: 'text-primary'
      }
    ];

    return (
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-6">
          <BarChart3 className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold">Ensemble Statistics</h2>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {statCards.map(stat => (
            <div key={stat.label} className="p-4 bg-gray-800/50 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <stat.icon className={cn('h-5 w-5', stat.color)} />
              </div>
              <p className="text-2xl font-bold text-white">{stat.value}</p>
              <p className="text-sm text-gray-400">{stat.label}</p>
            </div>
          ))}
        </div>
      </div>
    );
  };

  /**
   * ============================================================
   * MAIN RENDER
   * ============================================================
   */

  if (loading && !ensembleStatus) {
    return (
      <div className="p-6 flex items-center justify-center min-h-[400px]">
        <div className="flex items-center gap-3 text-gray-400">
          <RefreshCw className="h-6 w-6 animate-spin" />
          <span>Loading ensemble configuration...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">Ensemble Strategy Management</h1>
          <p className="text-gray-400">
            Configure multi-model ensemble for improved prediction accuracy
          </p>
        </div>
        <button
          onClick={refreshAll}
          disabled={loading}
          className="p-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <RefreshCw className={cn('h-5 w-5 text-gray-400', loading && 'animate-spin')} />
        </button>
      </div>

      {/* Alerts */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4 flex items-start gap-3">
          <XCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-red-400 font-medium">Error</p>
            <p className="text-red-300 text-sm">{error}</p>
          </div>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300">
            <XCircle className="h-5 w-5" />
          </button>
        </div>
      )}

      {success && (
        <div className="bg-green-500/10 border border-green-500/50 rounded-lg p-4 flex items-center gap-3">
          <CheckCircle className="h-5 w-5 text-green-400" />
          <p className="text-green-400">{success}</p>
        </div>
      )}

      {/* Models Grid */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Brain className="h-6 w-6 text-primary" />
            <h2 className="text-xl font-semibold">Model Management</h2>
          </div>
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <span>{models.filter(m => m.enabled && m.is_registered).length}</span>
            <span>/</span>
            <span>{models.length} models active</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {models.map(model => renderModelCard(model))}
        </div>

        {models.length === 0 && (
          <div className="text-center py-12">
            <Brain className="h-16 w-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 text-lg mb-2">No models configured</p>
            <p className="text-gray-500 text-sm">Train models in the ML Management page</p>
          </div>
        )}
      </div>

      {/* Strategy Selector */}
      {renderStrategySelector()}

      {/* Configuration */}
      {renderConfigPanel()}

      {/* Statistics */}
      {renderStatistics()}
    </div>
  );
}
