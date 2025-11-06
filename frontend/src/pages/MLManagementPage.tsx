// frontend/src/pages/MLManagementPage.tsx

import { useState, useEffect } from 'react';
import { Brain, Rocket, RefreshCw, TrendingUp, Download, Upload } from 'lucide-react';
import { cn } from '../utils/helpers';

/**
 * Типы для ML Management
 */
interface TrainingParams {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  export_onnx: boolean;
  auto_promote: boolean;
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
      best_val_accuracy: number;
    };
  };
}

interface Model {
  name: string;
  version: string;
  stage: string;
  created_at: string;
  metrics: {
    accuracy?: number;
    precision?: number;
    recall?: number;
  };
}

/**
 * Страница управления ML моделями.
 * Позволяет обучать модели, просматривать их список и управлять deployment.
 */
export function MLManagementPage() {
  // State
  const [trainingParams, setTrainingParams] = useState<TrainingParams>({
    epochs: 50,
    batch_size: 64,
    learning_rate: 0.001,
    export_onnx: true,
    auto_promote: true
  });

  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    is_training: false
  });

  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(false);
  const [pollingInterval, setPollingInterval] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fetch training status
  const fetchTrainingStatus = async () => {
    try {
      const response = await fetch('/api/ml-management/training/status');
      const data = await response.json();
      setTrainingStatus(data);

      // If training is complete, show notification and refresh models
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
  const fetchModels = async () => {
    try {
      const response = await fetch('/api/ml-management/models');
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
      }
    } catch (err) {
      setError('Failed to start training');
      console.error('Training failed:', err);
    } finally {
      setLoading(false);
    }
  };

  // Promote model
  const handlePromoteModel = async (name: string, version: string) => {
    if (!confirm(`Promote ${name} v${version} to production?`)) {
      return;
    }

    try {
      const response = await fetch(
        `/api/ml-management/models/${name}/${version}/promote?stage=production`,
        { method: 'POST' }
      );

      const result = await response.json();

      if (result.success) {
        fetchModels();
      }
    } catch (err) {
      console.error('Promotion failed:', err);
      setError('Failed to promote model');
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchTrainingStatus();
    fetchModels();

    // Cleanup
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, []);

  // Render training status card
  const renderTrainingStatus = () => {
    const { is_training, current_job } = trainingStatus;

    if (!is_training) {
      return (
        <div className="bg-surface border border-gray-800 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <Brain className="h-6 w-6 text-primary" />
            <h2 className="text-xl font-semibold">Training Status</h2>
          </div>
          <div className="text-center py-8">
            <p className="text-gray-400">No training in progress</p>
          </div>
        </div>
      );
    }

    const progress = current_job
      ? (current_job.progress.current_epoch / current_job.progress.total_epochs) * 100
      : 0;

    return (
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-4">
          <Brain className="h-6 w-6 text-primary animate-pulse" />
          <h2 className="text-xl font-semibold">Training Status</h2>
          <span className="ml-auto px-3 py-1 bg-primary/20 text-primary rounded-full text-sm">
            Training...
          </span>
        </div>

        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-gray-400">Job ID:</span>
              <span className="text-white font-mono">{current_job?.job_id}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Started:</span>
              <span className="text-white">{current_job?.started_at}</span>
            </div>
          </div>

          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-gray-400">
                Epoch {current_job?.progress.current_epoch} / {current_job?.progress.total_epochs}
              </span>
              <span className="text-white">{Math.round(progress)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-primary rounded-full h-2 transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Best Val Accuracy:</span>
            <span className="text-green-400 font-semibold">
              {current_job?.progress.best_val_accuracy.toFixed(4)}
            </span>
          </div>
        </div>
      </div>
    );
  };

  // Render training form
  const renderTrainingForm = () => (
    <div className="bg-surface border border-gray-800 rounded-lg p-6">
      <div className="flex items-center gap-3 mb-6">
        <Rocket className="h-6 w-6 text-primary" />
        <h2 className="text-xl font-semibold">Quick Train</h2>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Epochs
          </label>
          <input
            type="number"
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
            value={trainingParams.epochs}
            onChange={e =>
              setTrainingParams({ ...trainingParams, epochs: parseInt(e.target.value) })
            }
            disabled={trainingStatus.is_training}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Batch Size
          </label>
          <input
            type="number"
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
            value={trainingParams.batch_size}
            onChange={e =>
              setTrainingParams({ ...trainingParams, batch_size: parseInt(e.target.value) })
            }
            disabled={trainingStatus.is_training}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Learning Rate
          </label>
          <input
            type="number"
            step="0.0001"
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
            value={trainingParams.learning_rate}
            onChange={e =>
              setTrainingParams({ ...trainingParams, learning_rate: parseFloat(e.target.value) })
            }
            disabled={trainingStatus.is_training}
          />
        </div>

        <div className="space-y-2">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              className="w-4 h-4 rounded border-gray-700 text-primary focus:ring-primary"
              checked={trainingParams.export_onnx}
              onChange={e =>
                setTrainingParams({ ...trainingParams, export_onnx: e.target.checked })
              }
              disabled={trainingStatus.is_training}
            />
            <span className="text-sm text-gray-400">Export to ONNX</span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              className="w-4 h-4 rounded border-gray-700 text-primary focus:ring-primary"
              checked={trainingParams.auto_promote}
              onChange={e =>
                setTrainingParams({ ...trainingParams, auto_promote: e.target.checked })
              }
              disabled={trainingStatus.is_training}
            />
            <span className="text-sm text-gray-400">Auto-promote to production</span>
          </label>
        </div>

        <button
          onClick={handleStartTraining}
          disabled={loading || trainingStatus.is_training}
          className={cn(
            'w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-medium transition-colors',
            loading || trainingStatus.is_training
              ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
              : 'bg-primary text-white hover:bg-primary/90'
          )}
        >
          <Rocket className="h-5 w-5" />
          {trainingStatus.is_training ? 'Training...' : 'Start Training'}
        </button>
      </div>
    </div>
  );

  // Render models table
  const renderModelsTable = () => (
    <div className="bg-surface border border-gray-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <TrendingUp className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold">Models</h2>
        </div>
        <button
          onClick={fetchModels}
          className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
          title="Refresh"
        >
          <RefreshCw className="h-5 w-5 text-gray-400" />
        </button>
      </div>

      {models.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-gray-400">No models found</p>
          <p className="text-gray-500 text-sm mt-2">Train your first model to get started</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Name</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Version</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Stage</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Accuracy</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Created</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map(model => (
                <tr key={`${model.name}_${model.version}`} className="border-b border-gray-800 hover:bg-gray-800/50">
                  <td className="py-3 px-4 text-sm text-white">{model.name}</td>
                  <td className="py-3 px-4 text-sm font-mono text-gray-400">{model.version}</td>
                  <td className="py-3 px-4">
                    <span
                      className={cn(
                        'px-2 py-1 rounded-full text-xs font-medium',
                        model.stage === 'production'
                          ? 'bg-green-500/20 text-green-400'
                          : model.stage === 'staging'
                          ? 'bg-yellow-500/20 text-yellow-400'
                          : 'bg-gray-500/20 text-gray-400'
                      )}
                    >
                      {model.stage}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-sm text-white">
                    {model.metrics.accuracy?.toFixed(4) || 'N/A'}
                  </td>
                  <td className="py-3 px-4 text-sm text-gray-400">
                    {new Date(model.created_at).toLocaleString()}
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      {model.stage !== 'production' && (
                        <button
                          onClick={() => handlePromoteModel(model.name, model.version)}
                          className="p-1.5 hover:bg-gray-700 rounded transition-colors"
                          title="Promote to production"
                        >
                          <Upload className="h-4 w-4 text-green-400" />
                        </button>
                      )}
                      <button
                        className="p-1.5 hover:bg-gray-700 rounded transition-colors"
                        title="Download"
                      >
                        <Download className="h-4 w-4 text-blue-400" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">ML Model Management</h1>
        <p className="text-gray-400">
          Train, deploy, and manage machine learning models
        </p>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {renderTrainingStatus()}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">{renderTrainingForm()}</div>
        <div className="lg:col-span-2">{renderModelsTable()}</div>
      </div>
    </div>
  );
}
