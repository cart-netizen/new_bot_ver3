// frontend/src/pages/MLManagementPage.tsx

import { useState, useEffect } from 'react';
import {
  Brain,
  Rocket,
  RefreshCw,
  TrendingUp,
  Download,
  Upload,
  PlayCircle,
  PauseCircle,
  Settings,
  BarChart3,
  Activity,
  Zap,
  Clock,
  CheckCircle,
  XCircle,
  Info,
  ExternalLink,
  Database,
  Gauge
} from 'lucide-react';
import { cn } from '../utils/helpers';

/**
 * ============================================================
 * TYPES & INTERFACES
 * ============================================================
 */

interface TrainingParams {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  early_stopping_patience: number;
  export_onnx: boolean;
  auto_promote: boolean;
  min_accuracy: number;
  data_source: 'feature_store' | 'legacy';
  data_path?: string;
}

interface TrainingStatus {
  is_training: boolean;
  current_job?: {
    job_id: string;
    status: string;
    started_at: string;
    progress: {
      current_epoch: number;
      total_epochs: number;
      current_loss: number;
      best_val_accuracy: number;
    };
  };
  last_completed?: {
    job_id: string;
    status: string;
    completed_at: string;
    result: {
      success: boolean;
      version?: string;
      test_metrics?: {
        accuracy: number;
        precision: number;
        recall: number;
        f1: number;
      };
      promoted_to_production?: boolean;
    };
  };
}

interface Model {
  name: string;
  version: string;
  stage: string;
  created_at: string;
  description?: string;
  metrics: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1?: number;
  };
}

interface RetrainingStatus {
  is_running: boolean;
  config: {
    enable_scheduled: boolean;
    retraining_interval_hours: number;
    enable_drift_trigger: boolean;
    drift_threshold: number;
    enable_performance_trigger: boolean;
    performance_threshold: number;
    auto_promote_to_production: boolean;
  };
  last_training_time?: string;
  last_drift_check_time?: string;
  last_performance_check_time?: string;
}

interface MLflowRun {
  run_id: string;
  experiment_id: string;
  status: string;
  start_time: number;
  end_time?: number;
  metrics?: {
    [key: string]: number;
  };
  params?: {
    [key: string]: string;
  };
}

type TabType = 'training' | 'models' | 'retraining' | 'mlflow' | 'statistics';

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
        <div className="absolute z-50 px-3 py-2 text-sm text-white bg-gray-900 rounded-lg shadow-lg whitespace-nowrap bottom-full left-1/2 transform -translate-x-1/2 mb-2 border border-gray-700">
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

export function MLManagementPage() {
  // ========== STATE ==========
  const [activeTab, setActiveTab] = useState<TabType>('training');

  // Training state
  const [trainingParams, setTrainingParams] = useState<TrainingParams>({
    epochs: 50,
    batch_size: 64,
    learning_rate: 0.001,
    early_stopping_patience: 20,
    export_onnx: true,
    auto_promote: true,
    min_accuracy: 0.80,
    data_source: 'feature_store',
    data_path: undefined
  });

  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    is_training: false
  });

  // Models state
  const [models, setModels] = useState<Model[]>([]);
  const [modelsFilter, setModelsFilter] = useState<string>('all');

  // Retraining state
  const [retrainingStatus, setRetrainingStatus] = useState<RetrainingStatus | null>(null);

  // MLflow state
  const [mlflowRuns, setMlflowRuns] = useState<MLflowRun[]>([]);
  const [bestRun, setBestRun] = useState<any>(null);

  // General state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pollingInterval, setPollingInterval] = useState<number | null>(null);

  /**
   * ============================================================
   * API FUNCTIONS
   * ============================================================
   */

  // Fetch training status
  const fetchTrainingStatus = async () => {
    try {
      const response = await fetch('/api/ml-management/training/status');
      const data = await response.json();
      setTrainingStatus(data);

      // If training is complete, refresh models
      if (!data.is_training && pollingInterval) {
        clearInterval(pollingInterval);
        setPollingInterval(null);
        fetchModels();
      }
    } catch (err) {
      console.error('Failed to fetch training status:', err);
    }
  };

  // Fetch models list
  const fetchModels = async (stage?: string) => {
    try {
      const url = stage && stage !== 'all'
        ? `/api/ml-management/models?stage=${stage}`
        : '/api/ml-management/models';

      const response = await fetch(url);
      const data = await response.json();
      setModels(data.models || []);
    } catch (err) {
      console.error('Failed to fetch models:', err);
      setError('Failed to load models');
    }
  };

  // Start training
  const handleStartTraining = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/ml-management/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainingParams)
      });

      const result = await response.json();

      if (result.job_id) {
        // Start polling for status
        const interval = window.setInterval(() => fetchTrainingStatus(), 2000);
        setPollingInterval(interval);
      } else {
        setError('Failed to start training');
      }
    } catch (err) {
      setError('Failed to start training');
      console.error('Training failed:', err);
    } finally {
      setLoading(false);
    }
  };

  // Promote model
  const handlePromoteModel = async (name: string, version: string, stage: string) => {
    if (!confirm(`Promote ${name} v${version} to ${stage}?`)) {
      return;
    }

    try {
      const response = await fetch(
        `/api/ml-management/models/${name}/${version}/promote?stage=${stage}`,
        { method: 'POST' }
      );

      const result = await response.json();

      if (result.success) {
        fetchModels(modelsFilter);
      } else {
        setError('Failed to promote model');
      }
    } catch (err) {
      console.error('Promotion failed:', err);
      setError('Failed to promote model');
    }
  };

  // Fetch retraining status
  const fetchRetrainingStatus = async () => {
    try {
      const response = await fetch('/api/ml-management/retraining/status');
      const data = await response.json();
      setRetrainingStatus(data);
    } catch (err) {
      console.error('Failed to fetch retraining status:', err);
    }
  };

  // Start retraining pipeline
  const handleStartRetraining = async () => {
    try {
      const response = await fetch('/api/ml-management/retraining/start', {
        method: 'POST'
      });

      const result = await response.json();

      if (result.success) {
        fetchRetrainingStatus();
      }
    } catch (err) {
      console.error('Failed to start retraining:', err);
      setError('Failed to start auto-retraining');
    }
  };

  // Stop retraining pipeline
  const handleStopRetraining = async () => {
    try {
      const response = await fetch('/api/ml-management/retraining/stop', {
        method: 'POST'
      });

      const result = await response.json();

      if (result.success) {
        fetchRetrainingStatus();
      }
    } catch (err) {
      console.error('Failed to stop retraining:', err);
      setError('Failed to stop auto-retraining');
    }
  };

  // Trigger manual retraining
  const handleTriggerRetraining = async () => {
    try {
      const response = await fetch('/api/ml-management/retraining/trigger?trigger=manual', {
        method: 'POST'
      });

      const result = await response.json();

      if (result.success) {
        setActiveTab('training');
        fetchTrainingStatus();
      }
    } catch (err) {
      console.error('Failed to trigger retraining:', err);
      setError('Failed to trigger retraining');
    }
  };

  // Fetch MLflow runs
  const fetchMLflowRuns = async () => {
    try {
      const response = await fetch('/api/ml-management/mlflow/runs?limit=10');
      const data = await response.json();
      setMlflowRuns(data.runs || []);
    } catch (err) {
      console.error('Failed to fetch MLflow runs:', err);
    }
  };

  // Fetch best MLflow run
  const fetchBestRun = async () => {
    try {
      const response = await fetch('/api/ml-management/mlflow/best-run?metric=val_accuracy');
      const data = await response.json();
      setBestRun(data);
    } catch (err) {
      console.error('Failed to fetch best run:', err);
    }
  };

  /**
   * ============================================================
   * EFFECTS
   * ============================================================
   */

  // Initial fetch
  useEffect(() => {
    fetchTrainingStatus();
    fetchModels();
    fetchRetrainingStatus();
  }, []);

  // Fetch data when tab changes
  useEffect(() => {
    if (activeTab === 'models') {
      fetchModels(modelsFilter);
    } else if (activeTab === 'retraining') {
      fetchRetrainingStatus();
    } else if (activeTab === 'mlflow') {
      fetchMLflowRuns();
      fetchBestRun();
    }
  }, [activeTab, modelsFilter]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);

  /**
   * ============================================================
   * RENDER FUNCTIONS - TRAINING TAB
   * ============================================================
   */

  const renderTrainingTab = () => (
    <div className="space-y-6">
      {/* Training Status */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-6">
          <Activity className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold">Training Status</h2>
        </div>

        {!trainingStatus.is_training ? (
          <div className="text-center py-8">
            <p className="text-gray-400 mb-2">No training in progress</p>
            {trainingStatus.last_completed && (
              <div className="mt-4 p-4 bg-gray-800 rounded-lg inline-block text-left">
                <p className="text-sm text-gray-400 mb-2">Last Training:</p>
                <div className="space-y-1 text-sm">
                  <p className="text-white">
                    <span className="text-gray-400">Job ID:</span>{' '}
                    <span className="font-mono">{trainingStatus.last_completed.job_id}</span>
                  </p>
                  <p className="text-white">
                    <span className="text-gray-400">Status:</span>{' '}
                    {trainingStatus.last_completed.status === 'completed' ? (
                      <span className="text-green-400">✓ Completed</span>
                    ) : (
                      <span className="text-red-400">✗ Failed</span>
                    )}
                  </p>
                  {trainingStatus.last_completed.result?.test_metrics && (
                    <p className="text-white">
                      <span className="text-gray-400">Accuracy:</span>{' '}
                      <span className="text-green-400">
                        {trainingStatus.last_completed.result.test_metrics.accuracy.toFixed(4)}
                      </span>
                    </p>
                  )}
                  {trainingStatus.last_completed.result?.promoted_to_production && (
                    <p className="text-green-400 font-semibold mt-2">🚀 Promoted to Production</p>
                  )}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Job ID</p>
                <p className="font-mono text-white">{trainingStatus.current_job?.job_id}</p>
              </div>
              <span className="px-3 py-1 bg-primary/20 text-primary rounded-full text-sm animate-pulse">
                Training...
              </span>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">
                  Epoch {trainingStatus.current_job?.progress.current_epoch} /{' '}
                  {trainingStatus.current_job?.progress.total_epochs}
                </span>
                <span className="text-white">
                  {trainingStatus.current_job &&
                    Math.round(
                      (trainingStatus.current_job.progress.current_epoch /
                        trainingStatus.current_job.progress.total_epochs) *
                        100
                    )}
                  %
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-primary to-blue-500 rounded-full h-3 transition-all duration-500"
                  style={{
                    width: `${
                      trainingStatus.current_job
                        ? (trainingStatus.current_job.progress.current_epoch /
                            trainingStatus.current_job.progress.total_epochs) *
                          100
                        : 0
                    }%`
                  }}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-gray-800 rounded-lg">
                <p className="text-sm text-gray-400 mb-1">Current Loss</p>
                <p className="text-lg font-semibold text-white">
                  {trainingStatus.current_job?.progress.current_loss.toFixed(4) || 'N/A'}
                </p>
              </div>
              <div className="p-3 bg-gray-800 rounded-lg">
                <p className="text-sm text-gray-400 mb-1">Best Val Accuracy</p>
                <p className="text-lg font-semibold text-green-400">
                  {trainingStatus.current_job?.progress.best_val_accuracy.toFixed(4) || 'N/A'}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Data Source Configuration */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-6">
          <Database className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold">Data Source</h2>
        </div>

        <div className="space-y-4">
          {/* Source Type Selector */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-3">
              Select Data Source
            </label>
            <div className="grid grid-cols-2 gap-4">
              <button
                onClick={() => setTrainingParams({ ...trainingParams, data_source: 'feature_store', data_path: undefined })}
                disabled={trainingStatus.is_training}
                className={cn(
                  'p-4 rounded-lg border-2 transition-all',
                  trainingParams.data_source === 'feature_store'
                    ? 'border-primary bg-primary/10 text-white'
                    : 'border-gray-700 bg-gray-800 text-gray-400 hover:border-gray-600',
                  trainingStatus.is_training && 'opacity-50 cursor-not-allowed'
                )}
              >
                <div className="flex items-center gap-3 mb-2">
                  <Database className="h-5 w-5" />
                  <span className="font-semibold">Feature Store</span>
                </div>
                <p className="text-xs text-gray-400">
                  Modern parquet format with named features
                </p>
                <p className="text-xs text-gray-500 mt-1 font-mono">
                  data/feature_store/offline/
                </p>
              </button>

              <button
                onClick={() => setTrainingParams({ ...trainingParams, data_source: 'legacy', data_path: undefined })}
                disabled={trainingStatus.is_training}
                className={cn(
                  'p-4 rounded-lg border-2 transition-all',
                  trainingParams.data_source === 'legacy'
                    ? 'border-primary bg-primary/10 text-white'
                    : 'border-gray-700 bg-gray-800 text-gray-400 hover:border-gray-600',
                  trainingStatus.is_training && 'opacity-50 cursor-not-allowed'
                )}
              >
                <div className="flex items-center gap-3 mb-2">
                  <Database className="h-5 w-5" />
                  <span className="font-semibold">Legacy Files</span>
                </div>
                <p className="text-xs text-gray-400">
                  NumPy .npy files (backward compatibility)
                </p>
                <p className="text-xs text-gray-500 mt-1 font-mono">
                  data/ml_training/
                </p>
              </button>
            </div>
          </div>

          {/* Custom Path Input */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Custom Data Path (Optional)
              <Info className="inline-block ml-1 h-3 w-3 text-gray-500" />
            </label>
            <input
              type="text"
              placeholder={
                trainingParams.data_source === 'feature_store'
                  ? 'e.g., C:\\Users\\1q\\PycharmProjects\\Bot_ver3_stakan_new\\data\\feature_store'
                  : 'e.g., C:\\Users\\1q\\PycharmProjects\\Bot_ver3_stakan_new\\data\\ml_training'
              }
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white font-mono text-sm focus:outline-none focus:border-primary"
              value={trainingParams.data_path || ''}
              onChange={e =>
                setTrainingParams({ ...trainingParams, data_path: e.target.value || undefined })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-2">
              Leave empty to use default location. Provide absolute path for custom location.
            </p>
          </div>

          {/* Current Path Display */}
          <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
            <div className="flex items-start gap-3">
              <Info className="h-4 w-4 text-blue-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm text-gray-400 mb-1">Active Data Path:</p>
                <p className="text-xs font-mono text-white break-all">
                  {trainingParams.data_path ||
                    (trainingParams.data_source === 'feature_store'
                      ? 'data/feature_store/offline/'
                      : 'data/ml_training/')}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Training Configuration */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-6">
          <Rocket className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold">Training Configuration</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {/* Epochs */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Epochs
              <Info className="inline-block ml-1 h-3 w-3 text-gray-500" />
            </label>
            <input
              type="number"
              min="1"
              max="500"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.epochs}
              onChange={e =>
                setTrainingParams({ ...trainingParams, epochs: parseInt(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
          </div>

          {/* Batch Size */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Batch Size
              <Info className="inline-block ml-1 h-3 w-3 text-gray-500" />
            </label>
            <input
              type="number"
              min="8"
              max="512"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.batch_size}
              onChange={e =>
                setTrainingParams({ ...trainingParams, batch_size: parseInt(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
          </div>

          {/* Learning Rate */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Learning Rate
              <Info className="inline-block ml-1 h-3 w-3 text-gray-500" />
            </label>
            <input
              type="number"
              step="0.0001"
              min="0.00001"
              max="0.1"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.learning_rate}
              onChange={e =>
                setTrainingParams({ ...trainingParams, learning_rate: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
          </div>

          {/* Early Stopping Patience */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Early Stopping Patience
              <Tooltip content="Number of epochs without improvement before stopping. Set to 0 to disable early stopping.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              min="0"
              max="100"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.early_stopping_patience}
              onChange={e =>
                setTrainingParams({ ...trainingParams, early_stopping_patience: parseInt(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              {trainingParams.early_stopping_patience === 0
                ? 'Early stopping disabled - will train for all epochs'
                : `Will stop if no improvement for ${trainingParams.early_stopping_patience} epochs`}
            </p>
          </div>

          {/* Min Accuracy for Promotion */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Min Accuracy for Promotion
              <Info className="inline-block ml-1 h-3 w-3 text-gray-500" />
            </label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.min_accuracy}
              onChange={e =>
                setTrainingParams({ ...trainingParams, min_accuracy: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
          </div>
        </div>

        {/* Options */}
        <div className="space-y-3 mb-6">
          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              className="w-5 h-5 rounded border-gray-700 text-primary focus:ring-primary"
              checked={trainingParams.export_onnx}
              onChange={e =>
                setTrainingParams({ ...trainingParams, export_onnx: e.target.checked })
              }
              disabled={trainingStatus.is_training}
            />
            <span className="text-sm text-gray-300 group-hover:text-white transition-colors">
              Export to ONNX format (for optimized inference)
            </span>
          </label>

          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              className="w-5 h-5 rounded border-gray-700 text-primary focus:ring-primary"
              checked={trainingParams.auto_promote}
              onChange={e =>
                setTrainingParams({ ...trainingParams, auto_promote: e.target.checked })
              }
              disabled={trainingStatus.is_training}
            />
            <span className="text-sm text-gray-300 group-hover:text-white transition-colors">
              Auto-promote to production (if accuracy threshold is met)
            </span>
          </label>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3">
          <Tooltip content="Start model training with current parameters">
            <button
              onClick={handleStartTraining}
              disabled={loading || trainingStatus.is_training}
              className={cn(
                'flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors',
                loading || trainingStatus.is_training
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  : 'bg-primary text-white hover:bg-primary/90'
              )}
            >
              <Rocket className="h-5 w-5" />
              {trainingStatus.is_training ? 'Training in Progress...' : 'Start Training'}
            </button>
          </Tooltip>

          <Tooltip content="Refresh training status">
            <button
              onClick={fetchTrainingStatus}
              className="px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
            >
              <RefreshCw className="h-5 w-5 text-gray-400" />
            </button>
          </Tooltip>
        </div>
      </div>
    </div>
  );

  /**
   * ============================================================
   * RENDER FUNCTIONS - MODELS TAB
   * ============================================================
   */

  const renderModelsTab = () => (
    <div className="space-y-6">
      {/* Models Header */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Database className="h-6 w-6 text-primary" />
            <h2 className="text-xl font-semibold">Model Registry</h2>
          </div>

          <div className="flex items-center gap-3">
            {/* Filter */}
            <select
              className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={modelsFilter}
              onChange={e => {
                setModelsFilter(e.target.value);
                fetchModels(e.target.value);
              }}
            >
              <option value="all">All Stages</option>
              <option value="production">Production</option>
              <option value="staging">Staging</option>
              <option value="archived">Archived</option>
            </select>

            <Tooltip content="Refresh models list">
              <button
                onClick={() => fetchModels(modelsFilter)}
                className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
              >
                <RefreshCw className="h-5 w-5 text-gray-400" />
              </button>
            </Tooltip>
          </div>
        </div>
      </div>

      {/* Models Table */}
      <div className="bg-surface border border-gray-800 rounded-lg overflow-hidden">
        {models.length === 0 ? (
          <div className="p-12 text-center">
            <Database className="h-16 w-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 text-lg mb-2">No models found</p>
            <p className="text-gray-500 text-sm">Train your first model to get started</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-800">
                <tr>
                  <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">Name</th>
                  <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">Version</th>
                  <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">Stage</th>
                  <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">Accuracy</th>
                  <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">
                    Precision
                  </th>
                  <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">Recall</th>
                  <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">F1 Score</th>
                  <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">Created</th>
                  <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {models.map(model => (
                  <tr
                    key={`${model.name}_${model.version}`}
                    className="hover:bg-gray-800/50 transition-colors"
                  >
                    <td className="py-4 px-6">
                      <div className="flex items-center gap-2">
                        <Brain className="h-4 w-4 text-primary" />
                        <span className="text-white font-medium">{model.name}</span>
                      </div>
                    </td>
                    <td className="py-4 px-6">
                      <span className="font-mono text-sm text-gray-400">{model.version}</span>
                    </td>
                    <td className="py-4 px-6">
                      <span
                        className={cn(
                          'px-3 py-1 rounded-full text-xs font-semibold',
                          model.stage === 'production'
                            ? 'bg-green-500/20 text-green-400'
                            : model.stage === 'staging'
                            ? 'bg-yellow-500/20 text-yellow-400'
                            : 'bg-gray-500/20 text-gray-400'
                        )}
                      >
                        {model.stage.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-4 px-6">
                      <span className="text-white font-semibold">
                        {model.metrics.accuracy?.toFixed(4) || 'N/A'}
                      </span>
                    </td>
                    <td className="py-4 px-6">
                      <span className="text-gray-300">
                        {model.metrics.precision?.toFixed(4) || 'N/A'}
                      </span>
                    </td>
                    <td className="py-4 px-6">
                      <span className="text-gray-300">
                        {model.metrics.recall?.toFixed(4) || 'N/A'}
                      </span>
                    </td>
                    <td className="py-4 px-6">
                      <span className="text-gray-300">{model.metrics.f1?.toFixed(4) || 'N/A'}</span>
                    </td>
                    <td className="py-4 px-6">
                      <span className="text-sm text-gray-400">
                        {new Date(model.created_at).toLocaleDateString()}
                      </span>
                    </td>
                    <td className="py-4 px-6">
                      <div className="flex items-center gap-2">
                        {model.stage !== 'production' && (
                          <Tooltip content="Promote to production">
                            <button
                              onClick={() => handlePromoteModel(model.name, model.version, 'production')}
                              className="p-2 hover:bg-gray-700 rounded-lg transition-colors group"
                            >
                              <Upload className="h-4 w-4 text-green-400 group-hover:text-green-300" />
                            </button>
                          </Tooltip>
                        )}

                        {model.stage === 'production' && (
                          <Tooltip content="Move to staging">
                            <button
                              onClick={() => handlePromoteModel(model.name, model.version, 'staging')}
                              className="p-2 hover:bg-gray-700 rounded-lg transition-colors group"
                            >
                              <Download className="h-4 w-4 text-yellow-400 group-hover:text-yellow-300" />
                            </button>
                          </Tooltip>
                        )}

                        <Tooltip content="Download model">
                          <button className="p-2 hover:bg-gray-700 rounded-lg transition-colors group">
                            <Download className="h-4 w-4 text-blue-400 group-hover:text-blue-300" />
                          </button>
                        </Tooltip>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );

  /**
   * ============================================================
   * RENDER FUNCTIONS - AUTO-RETRAINING TAB
   * ============================================================
   */

  const renderRetrainingTab = () => (
    <div className="space-y-6">
      {/* Pipeline Status */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Zap className="h-6 w-6 text-primary" />
            <h2 className="text-xl font-semibold">Auto-Retraining Pipeline</h2>
          </div>

          {retrainingStatus && (
            <span
              className={cn(
                'px-3 py-1 rounded-full text-sm font-semibold',
                retrainingStatus.is_running
                  ? 'bg-green-500/20 text-green-400'
                  : 'bg-gray-500/20 text-gray-400'
              )}
            >
              {retrainingStatus.is_running ? '● Running' : '○ Stopped'}
            </span>
          )}
        </div>

        {retrainingStatus && (
          <div className="space-y-4">
            {/* Status Info */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-gray-800 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="h-4 w-4 text-gray-400" />
                  <p className="text-sm text-gray-400">Last Training</p>
                </div>
                <p className="text-white font-mono text-sm">
                  {retrainingStatus.last_training_time
                    ? new Date(retrainingStatus.last_training_time).toLocaleString()
                    : 'Never'}
                </p>
              </div>

              <div className="p-4 bg-gray-800 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="h-4 w-4 text-gray-400" />
                  <p className="text-sm text-gray-400">Last Drift Check</p>
                </div>
                <p className="text-white font-mono text-sm">
                  {retrainingStatus.last_drift_check_time
                    ? new Date(retrainingStatus.last_drift_check_time).toLocaleString()
                    : 'Never'}
                </p>
              </div>

              <div className="p-4 bg-gray-800 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Gauge className="h-4 w-4 text-gray-400" />
                  <p className="text-sm text-gray-400">Last Performance Check</p>
                </div>
                <p className="text-white font-mono text-sm">
                  {retrainingStatus.last_performance_check_time
                    ? new Date(retrainingStatus.last_performance_check_time).toLocaleString()
                    : 'Never'}
                </p>
              </div>
            </div>

            {/* Configuration */}
            <div className="p-4 bg-gray-800 rounded-lg">
              <p className="text-sm font-medium text-gray-400 mb-3">Pipeline Configuration</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Scheduled Retraining</span>
                  {retrainingStatus.config.enable_scheduled ? (
                    <CheckCircle className="h-5 w-5 text-green-400" />
                  ) : (
                    <XCircle className="h-5 w-5 text-gray-600" />
                  )}
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">
                    Interval: {retrainingStatus.config.retraining_interval_hours}h
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Drift Detection</span>
                  {retrainingStatus.config.enable_drift_trigger ? (
                    <CheckCircle className="h-5 w-5 text-green-400" />
                  ) : (
                    <XCircle className="h-5 w-5 text-gray-600" />
                  )}
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">
                    Threshold: {retrainingStatus.config.drift_threshold}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Performance Monitoring</span>
                  {retrainingStatus.config.enable_performance_trigger ? (
                    <CheckCircle className="h-5 w-5 text-green-400" />
                  ) : (
                    <XCircle className="h-5 w-5 text-gray-600" />
                  )}
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">
                    Threshold: {retrainingStatus.config.performance_threshold}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Control Panel */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-6">
          <Settings className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold">Pipeline Controls</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {retrainingStatus && !retrainingStatus.is_running ? (
            <Tooltip content="Start the auto-retraining pipeline (runs in background)">
              <button
                onClick={handleStartRetraining}
                className="flex items-center justify-center gap-2 px-6 py-4 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition-colors"
              >
                <PlayCircle className="h-5 w-5" />
                Start Pipeline
              </button>
            </Tooltip>
          ) : (
            <Tooltip content="Stop the auto-retraining pipeline">
              <button
                onClick={handleStopRetraining}
                className="flex items-center justify-center gap-2 px-6 py-4 bg-red-600 hover:bg-red-700 rounded-lg font-medium transition-colors"
              >
                <PauseCircle className="h-5 w-5" />
                Stop Pipeline
              </button>
            </Tooltip>
          )}

          <Tooltip content="Manually trigger retraining immediately">
            <button
              onClick={handleTriggerRetraining}
              className="flex items-center justify-center gap-2 px-6 py-4 bg-primary hover:bg-primary/90 rounded-lg font-medium transition-colors"
            >
              <Rocket className="h-5 w-5" />
              Trigger Now
            </button>
          </Tooltip>

          <Tooltip content="Refresh pipeline status">
            <button
              onClick={fetchRetrainingStatus}
              className="flex items-center justify-center gap-2 px-6 py-4 bg-gray-800 hover:bg-gray-700 rounded-lg font-medium transition-colors"
            >
              <RefreshCw className="h-5 w-5" />
              Refresh Status
            </button>
          </Tooltip>
        </div>
      </div>
    </div>
  );

  /**
   * ============================================================
   * RENDER FUNCTIONS - MLFLOW TAB
   * ============================================================
   */

  const renderMLflowTab = () => (
    <div className="space-y-6">
      {/* Best Run */}
      {bestRun && (
        <div className="bg-surface border border-gray-800 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <TrendingUp className="h-6 w-6 text-primary" />
            <h2 className="text-xl font-semibold">Best Run</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="p-4 bg-gradient-to-br from-green-500/20 to-green-600/10 rounded-lg border border-green-500/30">
              <p className="text-sm text-gray-400 mb-1">Run ID</p>
              <p className="font-mono text-sm text-white">{bestRun.run_id?.substring(0, 8)}...</p>
            </div>
            {bestRun.metrics && Object.entries(bestRun.metrics).map(([key, value]: [string, any]) => (
              <div key={key} className="p-4 bg-gray-800 rounded-lg">
                <p className="text-sm text-gray-400 mb-1">{key}</p>
                <p className="text-lg font-semibold text-green-400">
                  {typeof value === 'number' ? value.toFixed(4) : value}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Runs */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <BarChart3 className="h-6 w-6 text-primary" />
            <h2 className="text-xl font-semibold">Recent Runs</h2>
          </div>

          <div className="flex items-center gap-3">
            <Tooltip content="Open MLflow UI">
              <a
                href="http://localhost:5000"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
              >
                <ExternalLink className="h-4 w-4 text-gray-400" />
                <span className="text-sm text-gray-300">Open MLflow UI</span>
              </a>
            </Tooltip>

            <Tooltip content="Refresh runs">
              <button
                onClick={() => {
                  fetchMLflowRuns();
                  fetchBestRun();
                }}
                className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
              >
                <RefreshCw className="h-5 w-5 text-gray-400" />
              </button>
            </Tooltip>
          </div>
        </div>

        {mlflowRuns.length === 0 ? (
          <div className="text-center py-12">
            <BarChart3 className="h-16 w-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400">No MLflow runs found</p>
          </div>
        ) : (
          <div className="space-y-3">
            {mlflowRuns.map(run => (
              <div
                key={run.run_id}
                className="p-4 bg-gray-800 hover:bg-gray-750 rounded-lg transition-colors"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-sm text-gray-400">
                      {run.run_id.substring(0, 8)}...
                    </span>
                    <span
                      className={cn(
                        'px-2 py-1 rounded text-xs font-semibold',
                        run.status === 'FINISHED'
                          ? 'bg-green-500/20 text-green-400'
                          : run.status === 'RUNNING'
                          ? 'bg-blue-500/20 text-blue-400'
                          : 'bg-red-500/20 text-red-400'
                      )}
                    >
                      {run.status}
                    </span>
                  </div>
                  <span className="text-sm text-gray-500">
                    {new Date(run.start_time).toLocaleString()}
                  </span>
                </div>

                {run.metrics && (
                  <div className="flex gap-4 text-sm">
                    {Object.entries(run.metrics).slice(0, 4).map(([key, value]: [string, any]) => (
                      <div key={key}>
                        <span className="text-gray-400">{key}: </span>
                        <span className="text-white font-semibold">
                          {typeof value === 'number' ? value.toFixed(4) : value}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  /**
   * ============================================================
   * RENDER FUNCTIONS - STATISTICS TAB
   * ============================================================
   */

  const renderStatisticsTab = () => (
    <div className="space-y-6">
      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-surface border border-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <Database className="h-8 w-8 text-primary" />
            <TrendingUp className="h-5 w-5 text-green-400" />
          </div>
          <p className="text-3xl font-bold text-white mb-1">{models.length}</p>
          <p className="text-sm text-gray-400">Total Models</p>
        </div>

        <div className="bg-surface border border-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <CheckCircle className="h-8 w-8 text-green-400" />
          </div>
          <p className="text-3xl font-bold text-white mb-1">
            {models.filter(m => m.stage === 'production').length}
          </p>
          <p className="text-sm text-gray-400">Production Models</p>
        </div>

        <div className="bg-surface border border-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <Activity className="h-8 w-8 text-yellow-400" />
          </div>
          <p className="text-3xl font-bold text-white mb-1">
            {models.filter(m => m.stage === 'staging').length}
          </p>
          <p className="text-sm text-gray-400">Staging Models</p>
        </div>

        <div className="bg-surface border border-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <Zap className="h-8 w-8 text-primary" />
          </div>
          <p className="text-3xl font-bold text-white mb-1">
            {retrainingStatus?.is_running ? 'ON' : 'OFF'}
          </p>
          <p className="text-sm text-gray-400">Auto-Retraining</p>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-6">
          <Gauge className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold">Model Performance</h2>
        </div>

        {models.length > 0 ? (
          <div className="space-y-4">
            {models
              .filter(m => m.stage === 'production')
              .map(model => (
                <div key={`${model.name}_${model.version}`} className="p-4 bg-gray-800 rounded-lg">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <p className="font-semibold text-white">{model.name}</p>
                      <p className="text-sm text-gray-400 font-mono">{model.version}</p>
                    </div>
                    <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-xs font-semibold">
                      PRODUCTION
                    </span>
                  </div>

                  <div className="grid grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-gray-400 mb-1">Accuracy</p>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-green-400 rounded-full h-2"
                            style={{ width: `${(model.metrics.accuracy || 0) * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-semibold text-white">
                          {((model.metrics.accuracy || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <div>
                      <p className="text-sm text-gray-400 mb-1">Precision</p>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-blue-400 rounded-full h-2"
                            style={{ width: `${(model.metrics.precision || 0) * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-semibold text-white">
                          {((model.metrics.precision || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <div>
                      <p className="text-sm text-gray-400 mb-1">Recall</p>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-yellow-400 rounded-full h-2"
                            style={{ width: `${(model.metrics.recall || 0) * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-semibold text-white">
                          {((model.metrics.recall || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <div>
                      <p className="text-sm text-gray-400 mb-1">F1 Score</p>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-purple-400 rounded-full h-2"
                            style={{ width: `${(model.metrics.f1 || 0) * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-semibold text-white">
                          {((model.metrics.f1 || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <Gauge className="h-16 w-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400">No production models available</p>
          </div>
        )}
      </div>
    </div>
  );

  /**
   * ============================================================
   * MAIN RENDER
   * ============================================================
   */

  const tabs: { id: TabType; label: string; icon: any }[] = [
    { id: 'training', label: 'Training', icon: Rocket },
    { id: 'models', label: 'Models', icon: Database },
    { id: 'retraining', label: 'Auto-Retraining', icon: Zap },
    { id: 'mlflow', label: 'MLflow', icon: BarChart3 },
    { id: 'statistics', label: 'Statistics', icon: Gauge }
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold mb-2">ML Model Management</h1>
        <p className="text-gray-400">
          Comprehensive machine learning lifecycle management platform
        </p>
      </div>

      {/* Error Alert */}
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

      {/* Tabs */}
      <div className="border-b border-gray-800">
        <div className="flex gap-1">
          {tabs.map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  'flex items-center gap-2 px-6 py-3 font-medium transition-colors relative',
                  activeTab === tab.id
                    ? 'text-primary'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800'
                )}
              >
                <Icon className="h-5 w-5" />
                <span>{tab.label}</span>
                {activeTab === tab.id && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab Content */}
      <div>
        {activeTab === 'training' && renderTrainingTab()}
        {activeTab === 'models' && renderModelsTab()}
        {activeTab === 'retraining' && renderRetrainingTab()}
        {activeTab === 'mlflow' && renderMLflowTab()}
        {activeTab === 'statistics' && renderStatisticsTab()}
      </div>
    </div>
  );
}
