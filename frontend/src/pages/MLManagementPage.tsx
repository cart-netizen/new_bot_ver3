// frontend/src/pages/MLManagementPage.tsx

import { useState, useEffect, useCallback, useRef } from 'react';
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
  Gauge,
  Shield,
  Target,
  AlertTriangle,
  Wifi,
  WifiOff
} from 'lucide-react';
import { cn } from '../utils/helpers';
import { useEnsembleWebSocket } from '../hooks/useEnsembleWebSocket';
import { TrainingProgress } from '../services/ensemble-websocket.service';
import { EnsembleRealTimeStatus } from '../components/ensemble/EnsembleRealTimeStatus';

/**
 * ============================================================
 * TYPES & INTERFACES
 * ============================================================
 */

interface TrainingParams {
  // ===== ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ V2 =====
  epochs: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  early_stopping_patience: number;

  // ===== НОВЫЕ ПАРАМЕТРЫ V2 =====
  // Scheduler
  lr_scheduler: string;
  scheduler_T_0: number;
  scheduler_T_mult: number;

  // Regularization
  dropout: number;
  label_smoothing: number;

  // Data Augmentation
  use_augmentation: boolean;
  gaussian_noise_std: number;

  // Class Balancing
  use_focal_loss: boolean;
  focal_gamma: number;
  use_class_weights: boolean;
  use_oversampling: boolean;
  oversample_ratio: number;
  use_undersampling: boolean;

  // ===== INDUSTRY STANDARD FEATURES =====
  // Purging & Embargo (предотвращение data leakage)
  use_purging: boolean;
  use_embargo: boolean;
  embargo_pct: number;

  // Rolling Normalization (для нестационарных данных)
  use_rolling_normalization: boolean;
  rolling_window_size: number;

  // Labeling Method
  labeling_method: 'fixed_threshold' | 'triple_barrier';

  // Triple Barrier Parameters (используются только если labeling_method='triple_barrier')
  tb_tp_multiplier: number;  // Take Profit = ATR * multiplier
  tb_sl_multiplier: number;  // Stop Loss = ATR * multiplier
  tb_max_holding_period: number;  // Max bars to hold position

  // ===== CPCV (Combinatorial Purged Cross-Validation) =====
  use_cpcv: boolean;
  cpcv_n_splits: number;  // Number of groups
  cpcv_n_test_splits: number;  // Number of test groups per combination
  calculate_pbo: boolean;  // Calculate Probability of Backtest Overfitting

  // ===== СТАНДАРТНЫЕ ПАРАМЕТРЫ =====
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
      train_loss?: number;
      val_loss?: number;
      train_acc?: number;
      val_acc?: number;
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

type TabType = 'training' | 'mpd_training' | 'tlob_training' | 'models' | 'retraining' | 'mlflow' | 'statistics' | 'layering' | 'optimization';

// Model Registry sub-tab types
type ModelSubTabType = 'cnn_lstm' | 'mpd_transformer' | 'tlob';

/**
 * ============================================================
 * HELPER FUNCTIONS
 * ============================================================
 */

// Factorial for calculating CPCV combinations C(n, k) = n! / (k! * (n-k)!)
function factorial(n: number): number {
  if (n <= 1) return 1;
  let result = 1;
  for (let i = 2; i <= n; i++) {
    result *= i;
  }
  return result;
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
  const [modelSubTab, setModelSubTab] = useState<ModelSubTabType>('cnn_lstm');

  // Training state
  const [trainingParams, setTrainingParams] = useState<TrainingParams>({
    // ===== ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ V2 =====
    epochs: 150,                      // v2: 150 (было 50)
    batch_size: 128,                  // v2: 128 (было 256, уменьшено для GPU 12GB)
    learning_rate: 0.00005,           // v2: 5e-5 (было 0.001)
    weight_decay: 0.01,               // v2: 0.01 (НОВОЕ)
    early_stopping_patience: 20,

    // ===== НОВЫЕ ПАРАМЕТРЫ V2 =====
    // Scheduler
    lr_scheduler: 'CosineAnnealingWarmRestarts',
    scheduler_T_0: 10,
    scheduler_T_mult: 2,

    // Regularization
    dropout: 0.4,                     // v2: 0.4 (было 0.3)
    label_smoothing: 0.1,             // v2: 0.1 (НОВОЕ)

    // Data Augmentation
    use_augmentation: true,
    gaussian_noise_std: 0.01,

    // Class Balancing - ТОЛЬКО Focal Loss по умолчанию (избежание перекомпенсации)
    use_focal_loss: true,             // Рекомендуется - основной метод балансировки
    focal_gamma: 2.5,                 // v2: 2.5 (фокус на сложных примерах)
    use_class_weights: false,         // Отключено - конфликтует с Focal Loss
    use_oversampling: false,          // Отключено - вызывает перекомпенсацию
    oversample_ratio: 0.5,
    use_undersampling: false,         // Отключено - не рекомендуется

    // ===== INDUSTRY STANDARD FEATURES =====
    // Purging & Embargo (предотвращение data leakage)
    use_purging: true,                // Рекомендуется: включено
    use_embargo: true,                // Рекомендуется: включено
    embargo_pct: 0.02,                // Рекомендуется: 2% от данных

    // Rolling Normalization
    use_rolling_normalization: false, // По умолчанию выключено
    rolling_window_size: 500,         // Рекомендуется: 500

    // Labeling Method
    labeling_method: 'fixed_threshold', // По умолчанию: фиксированный порог

    // Triple Barrier Parameters (López de Prado)
    tb_tp_multiplier: 1.5,            // Take Profit = 1.5 * ATR
    tb_sl_multiplier: 1.0,            // Stop Loss = 1.0 * ATR
    tb_max_holding_period: 24,        // 24 bars (при 1-minute = 24 минуты)

    // ===== CPCV (Combinatorial Purged Cross-Validation) =====
    use_cpcv: false,                  // По умолчанию выключено
    cpcv_n_splits: 6,                 // Рекомендуется: 6 групп
    cpcv_n_test_splits: 2,            // Рекомендуется: 2 тестовых группы
    calculate_pbo: false,             // Расчёт PBO после обучения

    // ===== СТАНДАРТНЫЕ ПАРАМЕТРЫ =====
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

  // WebSocket state for real-time training updates
  const [wsConnected, setWsConnected] = useState(false);

  // WebSocket handler for CNN-LSTM training progress
  const handleTrainingProgress = useCallback((data: TrainingProgress) => {
    // Only handle CNN-LSTM training updates
    if (data.model_type === 'cnn_lstm') {
      console.log('[CNN-LSTM Training] WebSocket update:', data);

      if (data.status === 'started') {
        setTrainingStatus(prev => ({
          ...prev,
          is_training: true,
          current_job: {
            job_id: data.task_id,
            status: 'running',
            started_at: new Date().toISOString(),
            progress: {
              current_epoch: 0,
              total_epochs: data.total_epochs,
              current_loss: 0,
              best_val_accuracy: 0,
              train_loss: 0,
              val_loss: 0,
              train_acc: 0,
              val_acc: 0
            }
          }
        }));
      } else if (data.status === 'training') {
        setTrainingStatus(prev => ({
          ...prev,
          is_training: true,
          current_job: prev.current_job ? {
            ...prev.current_job,
            progress: {
              current_epoch: data.epoch,
              total_epochs: data.total_epochs,
              current_loss: data.metrics.val_loss || 0,
              best_val_accuracy: data.metrics.val_acc || data.metrics.best_val_accuracy || 0,
              train_loss: data.metrics.train_loss,
              val_loss: data.metrics.val_loss,
              train_acc: data.metrics.train_acc,
              val_acc: data.metrics.val_acc
            }
          } : prev.current_job
        }));
      } else if (data.status === 'completed') {
        setTrainingStatus(prev => ({
          ...prev,
          is_training: false,
          last_completed: {
            job_id: data.task_id,
            status: 'completed',
            completed_at: new Date().toISOString(),
            result: {
              success: true,
              test_metrics: {
                accuracy: data.metrics.test_accuracy || data.metrics.best_val_accuracy || 0,
                precision: 0,
                recall: 0,
                f1: 0
              },
              promoted_to_production: data.metrics.promoted_to_production || false
            }
          }
        }));
        // Refresh training status from server to get full details
        fetchTrainingStatus();
      } else if (data.status === 'failed') {
        setTrainingStatus(prev => ({
          ...prev,
          is_training: false,
          last_completed: {
            job_id: data.task_id,
            status: 'failed',
            completed_at: new Date().toISOString(),
            result: {
              success: false
            }
          }
        }));
        if (data.metrics.error) {
          setError(String(data.metrics.error));
        }
      }
    }
  }, []);

  // WebSocket connection
  const { isConnected } = useEnsembleWebSocket({
    subscriptions: ['training'],
    autoConnect: true,
    onConnect: () => {
      console.log('[MLManagement] WebSocket connected');
      setWsConnected(true);
    },
    onDisconnect: () => {
      console.log('[MLManagement] WebSocket disconnected');
      setWsConnected(false);
    },
    onTrainingProgress: handleTrainingProgress
  });

  // Layering model state
  const [layeringStatus, setLayeringStatus] = useState<any>(null);
  const [dataStatus, setDataStatus] = useState<any>(null);
  const [layeringMetrics, setLayeringMetrics] = useState<any>(null);
  const [isLayeringTraining, setIsLayeringTraining] = useState(false);
  const [layeringTrainingOutput, setLayeringTrainingOutput] = useState<string>('');

  // MPD Transformer training state
  const [mpdTrainingStatus, setMpdTrainingStatus] = useState<{
    is_training: boolean;
    task_id?: string;
    current_epoch: number;
    total_epochs: number;
    progress: number;
    status: string;
    error?: string;
  }>({
    is_training: false,
    current_epoch: 0,
    total_epochs: 0,
    progress: 0,
    status: 'idle'
  });

  const [mpdTrainingParams, setMpdTrainingParams] = useState({
    epochs: 150,
    learning_rate: 0.0001,
    batch_size: 32,
    symbols: ['BTCUSDT'],
    days: 30,
    embed_dim: 256,
    num_layers: 6,
    num_heads: 8,
    dropout: 0.1,
    use_amp: true,
    data_path: 'D:\\PYTHON\\Bot_ver3_stakan_new\\data'
  });

  // TLOB Transformer training state
  const [tlobTrainingStatus, setTlobTrainingStatus] = useState<{
    is_training: boolean;
    task_id?: string;
    current_epoch: number;
    total_epochs: number;
    progress: number;
    status: string;
    error?: string;
  }>({
    is_training: false,
    current_epoch: 0,
    total_epochs: 0,
    progress: 0,
    status: 'idle'
  });

  const [tlobTrainingParams, setTlobTrainingParams] = useState({
    epochs: 150,
    learning_rate: 0.0001,
    batch_size: 16,
    symbols: ['BTCUSDT'],
    days: 30,
    num_levels: 20,
    sequence_length: 100,
    num_temporal_layers: 4,
    dropout: 0.1,
    use_amp: true,
    data_path: 'D:\\PYTHON\\Bot_ver3_stakan_new\\data'
  });

  // Available symbols from API (for MPD/TLOB multi-symbol training)
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([]);
  const [symbolGroups, setSymbolGroups] = useState<{[key: string]: string[]}>({});
  const [loadingSymbols, setLoadingSymbols] = useState(false);
  const [showMpdSymbolSelector, setShowMpdSymbolSelector] = useState(false);
  const [showTlobSymbolSelector, setShowTlobSymbolSelector] = useState(false);

  // Hyperparameter Optimization state
  const [optimizationStatus, setOptimizationStatus] = useState<{
    is_running: boolean;
    can_resume: boolean;
    current_mode?: string;
    current_group?: string;
    trials_completed: number;
    total_trials: number;
    best_metric?: number;
    elapsed_time?: string;
    estimated_remaining?: string;
    current_trial_params?: Record<string, any>;
    results_path?: string;
    data_source_path?: string;
  }>({
    is_running: false,
    can_resume: false,
    trials_completed: 0,
    total_trials: 0
  });

  const [optimizationConfig, setOptimizationConfig] = useState({
    mode: 'full' as 'full' | 'quick' | 'group' | 'resume' | 'fine_tune',
    target_group: 'learning_rate' as string,
    epochs_per_trial: 4,
    max_trials_per_group: 15,
    max_total_hours: 24,
    primary_metric: 'val_f1',
    data_source: 'feature_store' as 'feature_store' | 'legacy',
    enable_pruning: true,
    use_warm_start: true
  });

  const [optimizationHistory, setOptimizationHistory] = useState<Array<{
    trial_id: number;
    params: Record<string, any>;
    metrics: Record<string, number>;
    group: string;
    status: string;
    duration_minutes: number;
  }>>([]);

  const [bestParams, setBestParams] = useState<Record<string, any> | null>(null);
  const [optimizationPollingInterval, setOptimizationPollingInterval] = useState<number | null>(null);

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

  const handleDownloadModel = async (name: string, version: string) => {
    try {
      const response = await fetch(
        `/api/ml-management/models/${name}/${version}/download`
      );

      if (!response.ok) {
        throw new Error('Download failed');
      }

      // Create blob and download
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${name}_${version}.h5`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Download failed:', err);
      setError('Failed to download model');
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

  // Fetch layering model status
  const fetchLayeringStatus = async () => {
    try {
      const response = await fetch('/api/ml-management/layering/status');
      const data = await response.json();
      setLayeringStatus(data);
    } catch (err) {
      console.error('Failed to fetch layering status:', err);
    }
  };

  // Fetch layering data status
  const fetchDataStatus = async () => {
    try {
      const response = await fetch('/api/ml-management/layering/data-status');
      const data = await response.json();
      setDataStatus(data);
    } catch (err) {
      console.error('Failed to fetch data status:', err);
    }
  };

  // Fetch layering metrics
  const fetchLayeringMetrics = async () => {
    try {
      const response = await fetch('/api/ml-management/layering/metrics');
      const data = await response.json();
      setLayeringMetrics(data);
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
    }
  };

  // Check layering data
  const handleCheckData = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/ml-management/layering/check-data', {
        method: 'POST'
      });
      const result = await response.json();

      if (result.success) {
        alert('Data check completed!\n\n' + result.output);
        fetchDataStatus();
      } else {
        alert('Data check failed:\n' + result.error);
      }
    } catch (err) {
      console.error('Data check failed:', err);
      setError('Failed to check data');
    } finally {
      setLoading(false);
    }
  };

  // Train layering model
  const handleTrainLayeringModel = async () => {
    if (!confirm('Start training Layering ML model? This will take 2-10 minutes.')) {
      return;
    }

    try {
      setIsLayeringTraining(true);
      setLayeringTrainingOutput('Training started...\n');
      setLoading(true);

      const response = await fetch('/api/ml-management/layering/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ use_improved: true, timeout: 600 })
      });

      const result = await response.json();

      if (result.success) {
        setLayeringTrainingOutput(result.output || 'Training completed successfully!');
        alert('Training completed successfully!');
        fetchLayeringStatus();
        fetchLayeringMetrics();
      } else {
        setLayeringTrainingOutput(result.output + '\n\nError: ' + result.error);
        alert('Training failed:\n' + result.error);
      }
    } catch (err) {
      console.error('Training failed:', err);
      setError('Failed to train model');
      setLayeringTrainingOutput('Training failed: ' + err);
    } finally {
      setIsLayeringTraining(false);
      setLoading(false);
    }
  };

  // ========== HYPERPARAMETER OPTIMIZATION API FUNCTIONS ==========

  // Fetch optimization status
  const fetchOptimizationStatus = async () => {
    try {
      const response = await fetch('/api/hyperopt/status');
      const data = await response.json();
      setOptimizationStatus(data);

      // Stop polling if optimization is complete
      if (!data.is_running && optimizationPollingInterval) {
        clearInterval(optimizationPollingInterval);
        setOptimizationPollingInterval(null);
        fetchOptimizationHistory();
        fetchBestParams();
      }
    } catch (err) {
      console.error('Failed to fetch optimization status:', err);
    }
  };

  // Fetch optimization history
  const fetchOptimizationHistory = async () => {
    try {
      const response = await fetch('/api/hyperopt/history?limit=50');
      const data = await response.json();
      setOptimizationHistory(data.trials || []);
    } catch (err) {
      console.error('Failed to fetch optimization history:', err);
    }
  };

  // Fetch best parameters
  const fetchBestParams = async () => {
    try {
      const response = await fetch('/api/hyperopt/best-params');
      const data = await response.json();
      if (data.success && data.best_params) {
        setBestParams(data.best_params);
      }
    } catch (err) {
      console.error('Failed to fetch best params:', err);
    }
  };

  // Fetch available symbols for MPD/TLOB training
  const fetchAvailableSymbols = async () => {
    try {
      setLoadingSymbols(true);
      // Fixed: use correct ensemble API endpoint
      const response = await fetch('/api/ensemble/training/available-symbols');
      const data = await response.json();
      // API returns: { all_symbols: [...], preset_groups: {...}, ... }
      setAvailableSymbols(data.all_symbols || []);
      setSymbolGroups(data.preset_groups || {});
    } catch (err) {
      console.error('Failed to fetch available symbols:', err);
    } finally {
      setLoadingSymbols(false);
    }
  };

  // Toggle symbol selection helper
  const toggleSymbol = (
    symbols: string[],
    symbol: string,
    setParams: (params: any) => void,
    currentParams: any
  ) => {
    const newSymbols = symbols.includes(symbol)
      ? symbols.filter(s => s !== symbol)
      : [...symbols, symbol];
    setParams({ ...currentParams, symbols: newSymbols });
  };

  // Select symbol group helper
  const selectSymbolGroup = (
    groupName: string,
    setParams: (params: any) => void,
    currentParams: any
  ) => {
    const groupSymbols = symbolGroups[groupName] || [];
    if (groupSymbols.length > 0) {
      setParams({ ...currentParams, symbols: groupSymbols });
    }
  };

  // Select all symbols
  const selectAllSymbols = (
    setParams: (params: any) => void,
    currentParams: any
  ) => {
    setParams({ ...currentParams, symbols: [...availableSymbols] });
  };

  // Clear all symbols
  const clearAllSymbols = (
    setParams: (params: any) => void,
    currentParams: any
  ) => {
    setParams({ ...currentParams, symbols: [] });
  };

  // Start optimization
  const handleStartOptimization = async () => {
    const modeDescriptions: Record<string, string> = {
      'full': 'Полная оптимизация (все группы параметров последовательно)',
      'quick': 'Быстрая оптимизация (только learning_rate и regularization)',
      'group': `Оптимизация группы: ${optimizationConfig.target_group}`,
      'resume': 'Возобновление с последней позиции',
      'fine_tune': 'Тонкая настройка лучших параметров'
    };

    if (!confirm(`Запустить оптимизацию?\n\nРежим: ${modeDescriptions[optimizationConfig.mode]}\nЭпох на пробу: ${optimizationConfig.epochs_per_trial}\nМакс. часов: ${optimizationConfig.max_total_hours}\n\nКаждая проба занимает ~${optimizationConfig.epochs_per_trial * 12} минут.`)) {
      return;
    }

    try {
      setLoading(true);
      const response = await fetch('/api/hyperopt/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(optimizationConfig)
      });

      const result = await response.json();

      if (result.success) {
        // Start polling for status
        const interval = setInterval(fetchOptimizationStatus, 10000) as unknown as number;
        setOptimizationPollingInterval(interval);
        fetchOptimizationStatus();
      } else {
        setError(result.error || 'Failed to start optimization');
      }
    } catch (err) {
      console.error('Failed to start optimization:', err);
      setError('Failed to start optimization');
    } finally {
      setLoading(false);
    }
  };

  // Stop optimization
  const handleStopOptimization = async () => {
    if (!confirm('Остановить оптимизацию? Прогресс будет сохранён для возобновления.')) {
      return;
    }

    try {
      setLoading(true);
      const response = await fetch('/api/hyperopt/stop', {
        method: 'POST'
      });

      const result = await response.json();

      if (result.success) {
        if (optimizationPollingInterval) {
          clearInterval(optimizationPollingInterval);
          setOptimizationPollingInterval(null);
        }
        fetchOptimizationStatus();
        fetchOptimizationHistory();
      } else {
        setError(result.error || 'Failed to stop optimization');
      }
    } catch (err) {
      console.error('Failed to stop optimization:', err);
      setError('Failed to stop optimization');
    } finally {
      setLoading(false);
    }
  };

  // Resume optimization
  const handleResumeOptimization = async () => {
    if (!confirm('Возобновить оптимизацию с последней сохранённой позиции?')) {
      return;
    }

    try {
      setLoading(true);
      const response = await fetch('/api/hyperopt/resume', {
        method: 'POST'
      });

      const result = await response.json();

      if (result.success) {
        const interval = setInterval(fetchOptimizationStatus, 10000) as unknown as number;
        setOptimizationPollingInterval(interval);
        fetchOptimizationStatus();
      } else {
        setError(result.error || 'Failed to resume optimization');
      }
    } catch (err) {
      console.error('Failed to resume optimization:', err);
      setError('Failed to resume optimization');
    } finally {
      setLoading(false);
    }
  };

  // Apply best params to training config
  const handleApplyBestParams = () => {
    if (!bestParams) return;

    if (!confirm('Применить лучшие найденные параметры к конфигурации обучения?')) {
      return;
    }

    setTrainingParams(prev => ({
      ...prev,
      ...bestParams
    }));

    setActiveTab('training');
    alert('Параметры применены! Перейдите к вкладке "Training" для начала обучения.');
  };

  /**
   * ============================================================
   * EFFECTS
   * ============================================================
   */

  // Initial fetch
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    fetchTrainingStatus();
    fetchModels();
    fetchRetrainingStatus();
    fetchAvailableSymbols();
  }, []);

  // Fetch data when tab changes
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (activeTab === 'models') {
      fetchModels(modelsFilter);
    } else if (activeTab === 'retraining') {
      fetchRetrainingStatus();
    } else if (activeTab === 'mlflow') {
      fetchMLflowRuns();
      fetchBestRun();
    } else if (activeTab === 'layering') {
      fetchLayeringStatus();
      fetchDataStatus();
      fetchLayeringMetrics();
    } else if (activeTab === 'optimization') {
      fetchOptimizationStatus();
      fetchOptimizationHistory();
      fetchBestParams();
    }
  }, [activeTab, modelsFilter]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
      if (optimizationPollingInterval) {
        clearInterval(optimizationPollingInterval);
      }
    };
  }, [pollingInterval, optimizationPollingInterval]);

  /**
   * ============================================================
   * RENDER FUNCTIONS - TRAINING TAB
   * ============================================================
   */

  const renderTrainingTab = () => (
    <div className="space-y-6">
      {/* Training Status */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Activity className="h-6 w-6 text-primary" />
            <h2 className="text-xl font-semibold">Training Status</h2>
          </div>
          {/* WebSocket Connection Status */}
          <div className="flex items-center gap-2">
            {wsConnected ? (
              <>
                <Wifi className="w-4 h-4 text-green-500" />
                <span className="text-xs text-green-500">Real-time</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-gray-500" />
                <span className="text-xs text-gray-500">Polling</span>
              </>
            )}
          </div>
        </div>

        {!trainingStatus.is_training ? (
          <div className="text-center py-8">
            <p className="text-gray-400 mb-2">No training in progress</p>
            {trainingStatus.last_completed && (
              <div className="mt-4 p-4 bg-gray-800 rounded-lg inline-block text-left max-w-md">
                <p className="text-sm text-gray-400 mb-3">Last Training Result:</p>
                <div className="space-y-2 text-sm">
                  <p className="text-white">
                    <span className="text-gray-400">Job ID:</span>{' '}
                    <span className="font-mono">{trainingStatus.last_completed.job_id}</span>
                  </p>
                  <p className="text-white">
                    <span className="text-gray-400">Status:</span>{' '}
                    {trainingStatus.last_completed.status === 'completed' ? (
                      <span className="text-green-400 flex items-center gap-1 inline-flex">
                        <CheckCircle className="w-4 h-4" /> Completed
                      </span>
                    ) : (
                      <span className="text-red-400 flex items-center gap-1 inline-flex">
                        <XCircle className="w-4 h-4" /> Failed
                      </span>
                    )}
                  </p>
                  {trainingStatus.last_completed.completed_at && (
                    <p className="text-white">
                      <span className="text-gray-400">Completed at:</span>{' '}
                      <span>{new Date(trainingStatus.last_completed.completed_at).toLocaleString()}</span>
                    </p>
                  )}
                  {trainingStatus.last_completed.result?.test_metrics && (
                    <div className="mt-3 p-3 bg-gray-700 rounded-lg">
                      <p className="text-gray-400 text-xs mb-2">Final Metrics:</p>
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <span className="text-gray-400">Accuracy:</span>{' '}
                          <span className="text-green-400 font-semibold">
                            {(trainingStatus.last_completed.result.test_metrics.accuracy * 100).toFixed(2)}%
                          </span>
                        </div>
                        {trainingStatus.last_completed.result.test_metrics.precision > 0 && (
                          <div>
                            <span className="text-gray-400">Precision:</span>{' '}
                            <span className="text-blue-400">
                              {(trainingStatus.last_completed.result.test_metrics.precision * 100).toFixed(2)}%
                            </span>
                          </div>
                        )}
                        {trainingStatus.last_completed.result.test_metrics.recall > 0 && (
                          <div>
                            <span className="text-gray-400">Recall:</span>{' '}
                            <span className="text-blue-400">
                              {(trainingStatus.last_completed.result.test_metrics.recall * 100).toFixed(2)}%
                            </span>
                          </div>
                        )}
                        {trainingStatus.last_completed.result.test_metrics.f1 > 0 && (
                          <div>
                            <span className="text-gray-400">F1:</span>{' '}
                            <span className="text-blue-400">
                              {(trainingStatus.last_completed.result.test_metrics.f1 * 100).toFixed(2)}%
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  {trainingStatus.last_completed.result?.promoted_to_production && (
                    <div className="mt-3 p-2 bg-green-500/20 border border-green-500/30 rounded-lg">
                      <p className="text-green-400 font-semibold flex items-center gap-2">
                        <Rocket className="w-4 h-4" /> Promoted to Production
                      </p>
                    </div>
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
              <span className="px-3 py-1 bg-primary/20 text-primary rounded-full text-sm animate-pulse flex items-center gap-2">
                <RefreshCw className="w-4 h-4 animate-spin" />
                Training...
              </span>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">
                  Epoch {trainingStatus.current_job?.progress.current_epoch} /{' '}
                  {trainingStatus.current_job?.progress.total_epochs}
                </span>
                <span className="text-white font-semibold">
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

            {/* Training Metrics Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <div className="p-3 bg-gray-800 rounded-lg">
                <p className="text-xs text-gray-400 mb-1">Train Loss</p>
                <p className="text-lg font-semibold text-white">
                  {trainingStatus.current_job?.progress.train_loss?.toFixed(4) || 'N/A'}
                </p>
              </div>
              <div className="p-3 bg-gray-800 rounded-lg">
                <p className="text-xs text-gray-400 mb-1">Val Loss</p>
                <p className="text-lg font-semibold text-white">
                  {trainingStatus.current_job?.progress.val_loss?.toFixed(4) || 'N/A'}
                </p>
              </div>
              <div className="p-3 bg-gray-800 rounded-lg">
                <p className="text-xs text-gray-400 mb-1">Train Acc</p>
                <p className="text-lg font-semibold text-blue-400">
                  {trainingStatus.current_job?.progress.train_acc
                    ? `${(trainingStatus.current_job.progress.train_acc * 100).toFixed(1)}%`
                    : 'N/A'}
                </p>
              </div>
              <div className="p-3 bg-gray-800 rounded-lg">
                <p className="text-xs text-gray-400 mb-1">Val Acc</p>
                <p className="text-lg font-semibold text-green-400">
                  {trainingStatus.current_job?.progress.val_acc
                    ? `${(trainingStatus.current_job.progress.val_acc * 100).toFixed(1)}%`
                    : 'N/A'}
                </p>
              </div>
            </div>

            {/* Best Metrics */}
            <div className="p-3 bg-primary/10 border border-primary/30 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Best Val Accuracy</span>
                <span className="text-lg font-bold text-primary">
                  {trainingStatus.current_job?.progress.best_val_accuracy
                    ? `${(trainingStatus.current_job.progress.best_val_accuracy * 100).toFixed(2)}%`
                    : 'N/A'}
                </span>
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
              <Tooltip content="Количество эпох обучения. Одна эпоха = один проход по всем данным. Рекомендуется: 150. Больше = дольше обучение.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
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
            <p className="text-xs text-gray-500 mt-1">
              v2 рекомендуется: 150 (было: 50 в v1)
            </p>
          </div>

          {/* Batch Size */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Batch Size
              <Tooltip content="Размер пакета данных. Влияет на стабильность градиентов и скорость. Рекомендуется: 128-256 в зависимости от GPU памяти. Больше = стабильнее, но требует больше памяти.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
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
            <p className="text-xs text-gray-500 mt-1">
              v2: 128 для GPU 12GB (256 требует 16GB+)
            </p>
          </div>

          {/* Learning Rate */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Learning Rate
              <Tooltip content="Скорость обучения. Определяет размер шага оптимизации. Рекомендуется: 0.00005 (5e-5). Для финансовых данных нужен маленький LR.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              step="0.00001"
              min="0.00001"
              max="0.1"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.learning_rate}
              onChange={e =>
                setTrainingParams({ ...trainingParams, learning_rate: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2 рекомендуется: 0.00005 (было: 0.001 в v1) - КРИТИЧНО!
            </p>
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

          {/* Weight Decay */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Weight Decay (L2 Regularization)
              <Tooltip content="Регуляризация L2. Контролирует переобучение. Рекомендуется: 0.01. Больше = сильнее регуляризация.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              step="0.001"
              min="0"
              max="1"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.weight_decay}
              onChange={e =>
                setTrainingParams({ ...trainingParams, weight_decay: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2 рекомендуется: 0.01 (было: ~0 в v1)
            </p>
          </div>

          {/* Dropout */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Dropout
              <Tooltip content="Вероятность отключения нейронов. Предотвращает переобучение. Рекомендуется: 0.4. Выше = сильнее регуляризация.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="0.9"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.dropout}
              onChange={e =>
                setTrainingParams({ ...trainingParams, dropout: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2 рекомендуется: 0.4 (было: 0.3 в v1)
            </p>
          </div>

          {/* Label Smoothing */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Label Smoothing
              <Tooltip content="Смягчение меток. Предотвращает излишнюю уверенность модели. Рекомендуется: 0.1. Диапазон: 0-0.3.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="0.5"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.label_smoothing}
              onChange={e =>
                setTrainingParams({ ...trainingParams, label_smoothing: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2: новый параметр, рекомендуется 0.1
            </p>
          </div>

          {/* Focal Loss Gamma */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Focal Loss Gamma
              <Tooltip content="Параметр фокусировки на сложных примерах. Рекомендуется: 2.5. Больше = больше фокус на hard examples.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="5"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.focal_gamma}
              onChange={e =>
                setTrainingParams({ ...trainingParams, focal_gamma: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2 рекомендуется: 2.5 (было: 2.0 в v1)
            </p>
          </div>

          {/* Gaussian Noise Std */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Gaussian Noise Std
              <Tooltip content="Стандартное отклонение гауссовского шума для аугментации. Рекомендуется: 0.01. Добавляет робастность к шуму.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              step="0.001"
              min="0"
              max="0.1"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.gaussian_noise_std}
              onChange={e =>
                setTrainingParams({ ...trainingParams, gaussian_noise_std: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2: новый параметр, рекомендуется 0.01
            </p>
          </div>

          {/* Oversample Ratio */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Oversample Ratio
              <Tooltip content="Коэффициент оверсэмплинга для редких классов. Рекомендуется: 0.5. Помогает при дисбалансе классов.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.oversample_ratio}
              onChange={e =>
                setTrainingParams({ ...trainingParams, oversample_ratio: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2: новый параметр, рекомендуется 0.5
            </p>
          </div>

          {/* LR Scheduler T_0 */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Scheduler T_0 (Period)
              <Tooltip content="Период первого цикла для CosineAnnealing scheduler. Рекомендуется: 10. Определяет частоту перезапуска.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              min="1"
              max="100"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.scheduler_T_0}
              onChange={e =>
                setTrainingParams({ ...trainingParams, scheduler_T_0: parseInt(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2: новый параметр, рекомендуется 10
            </p>
          </div>

          {/* LR Scheduler T_mult */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Scheduler T_mult (Multiplier)
              <Tooltip content="Множитель периода для CosineAnnealing. Рекомендуется: 2. Увеличивает период на каждом цикле.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              min="1"
              max="10"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.scheduler_T_mult}
              onChange={e =>
                setTrainingParams({ ...trainingParams, scheduler_T_mult: parseInt(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2: новый параметр, рекомендуется 2
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

          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              className="w-5 h-5 rounded border-gray-700 text-primary focus:ring-primary"
              checked={trainingParams.use_augmentation}
              onChange={e =>
                setTrainingParams({ ...trainingParams, use_augmentation: e.target.checked })
              }
              disabled={trainingStatus.is_training}
            />
            <Tooltip content="Включить аугментацию данных (Gaussian noise). Повышает робастность модели к шуму.">
              <span className="text-sm text-gray-300 group-hover:text-white transition-colors cursor-help">
                Enable Data Augmentation (рекомендуется для v2)
              </span>
            </Tooltip>
          </label>

          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              className="w-5 h-5 rounded border-gray-700 text-primary focus:ring-primary"
              checked={trainingParams.use_focal_loss}
              onChange={e =>
                setTrainingParams({ ...trainingParams, use_focal_loss: e.target.checked })
              }
              disabled={trainingStatus.is_training}
            />
            <Tooltip content="Использовать Focal Loss вместо CrossEntropy. Лучше справляется с дисбалансом классов.">
              <span className="text-sm text-gray-300 group-hover:text-white transition-colors cursor-help">
                Use Focal Loss (рекомендуется для v2)
              </span>
            </Tooltip>
          </label>

          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              className="w-5 h-5 rounded border-gray-700 text-primary focus:ring-primary"
              checked={trainingParams.use_class_weights}
              onChange={e =>
                setTrainingParams({ ...trainingParams, use_class_weights: e.target.checked })
              }
              disabled={trainingStatus.is_training}
            />
            <Tooltip content="Веса классов в loss функции. НЕ рекомендуется с Focal Loss - вызывает перекомпенсацию.">
              <span className="text-sm text-gray-300 group-hover:text-white transition-colors cursor-help">
                Use Class Weights (не рекомендуется с Focal Loss)
              </span>
            </Tooltip>
          </label>

          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              className="w-5 h-5 rounded border-gray-700 text-primary focus:ring-primary"
              checked={trainingParams.use_oversampling}
              onChange={e =>
                setTrainingParams({ ...trainingParams, use_oversampling: e.target.checked })
              }
              disabled={trainingStatus.is_training}
            />
            <Tooltip content="Дублирование редких классов. НЕ рекомендуется с Focal Loss - вызывает перекомпенсацию.">
              <span className="text-sm text-gray-300 group-hover:text-white transition-colors cursor-help">
                Use Oversampling (не рекомендуется с Focal Loss)
              </span>
            </Tooltip>
          </label>

          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              className="w-5 h-5 rounded border-gray-700 text-primary focus:ring-primary"
              checked={trainingParams.use_undersampling}
              onChange={e =>
                setTrainingParams({ ...trainingParams, use_undersampling: e.target.checked })
              }
              disabled={trainingStatus.is_training}
            />
            <Tooltip content="Удаление примеров мажорного класса. НЕ рекомендуется - теряются данные.">
              <span className="text-sm text-gray-300 group-hover:text-white transition-colors cursor-help">
                Use Undersampling (не рекомендуется)
              </span>
            </Tooltip>
          </label>
        </div>

        {/* ===== INDUSTRY STANDARD FEATURES ===== */}
        <div className="bg-gradient-to-r from-green-900/20 to-emerald-900/20 border border-green-800/50 rounded-lg p-4 mt-6">
          <div className="flex items-center gap-2 mb-4">
            <Shield className="h-5 w-5 text-green-400" />
            <h3 className="text-lg font-semibold text-green-400">Industry Standard Features</h3>
            <Tooltip content="Методы из академических исследований для предотвращения переобучения и data leakage">
              <Info className="h-4 w-4 text-green-500 cursor-help" />
            </Tooltip>
          </div>

          {/* Purging & Embargo Section */}
          <div className="mb-4">
            <p className="text-sm text-gray-400 mb-3">
              <strong>Purging & Embargo</strong> — предотвращение утечки данных между train/val/test
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <label className="flex items-center gap-3 cursor-pointer group">
                <input
                  type="checkbox"
                  className="w-5 h-5 rounded border-gray-700 text-green-500 focus:ring-green-500"
                  checked={trainingParams.use_purging}
                  onChange={e =>
                    setTrainingParams({ ...trainingParams, use_purging: e.target.checked })
                  }
                  disabled={trainingStatus.is_training}
                />
                <Tooltip content="Удаляет samples на границах train/val/test для предотвращения data leakage. Критически важно для временных рядов!">
                  <span className="text-sm text-gray-300 group-hover:text-white transition-colors cursor-help">
                    ✓ Enable Purging (рекомендуется)
                  </span>
                </Tooltip>
              </label>

              <label className="flex items-center gap-3 cursor-pointer group">
                <input
                  type="checkbox"
                  className="w-5 h-5 rounded border-gray-700 text-green-500 focus:ring-green-500"
                  checked={trainingParams.use_embargo}
                  onChange={e =>
                    setTrainingParams({ ...trainingParams, use_embargo: e.target.checked })
                  }
                  disabled={trainingStatus.is_training}
                />
                <Tooltip content="Добавляет gap между sets для учёта автокорреляции. Предотвращает look-ahead bias.">
                  <span className="text-sm text-gray-300 group-hover:text-white transition-colors cursor-help">
                    ✓ Enable Embargo (рекомендуется)
                  </span>
                </Tooltip>
              </label>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">
                  Embargo %
                  <Tooltip content="Процент данных для embargo gap. Рекомендуется: 2%. Больше = безопаснее, но меньше данных для обучения.">
                    <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
                  </Tooltip>
                </label>
                <input
                  type="number"
                  step="0.005"
                  min="0.01"
                  max="0.1"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-green-500"
                  value={trainingParams.embargo_pct}
                  onChange={e =>
                    setTrainingParams({ ...trainingParams, embargo_pct: parseFloat(e.target.value) })
                  }
                  disabled={trainingStatus.is_training}
                />
              </div>
            </div>
          </div>

          {/* Rolling Normalization Section */}
          <div className="mb-4 pt-4 border-t border-gray-700">
            <p className="text-sm text-gray-400 mb-3">
              <strong>Rolling Normalization</strong> — адаптивная нормализация для нестационарных данных
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <label className="flex items-center gap-3 cursor-pointer group">
                <input
                  type="checkbox"
                  className="w-5 h-5 rounded border-gray-700 text-green-500 focus:ring-green-500"
                  checked={trainingParams.use_rolling_normalization}
                  onChange={e =>
                    setTrainingParams({ ...trainingParams, use_rolling_normalization: e.target.checked })
                  }
                  disabled={trainingStatus.is_training}
                />
                <Tooltip content="Использует скользящее окно для расчёта статистик. Помогает при изменении распределения данных со временем.">
                  <span className="text-sm text-gray-300 group-hover:text-white transition-colors cursor-help">
                    Enable Rolling Normalization (экспериментально)
                  </span>
                </Tooltip>
              </label>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">
                  Rolling Window Size
                  <Tooltip content="Размер скользящего окна для статистик. Рекомендуется: 500. Больше окно = стабильнее, но медленнее адаптация.">
                    <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
                  </Tooltip>
                </label>
                <input
                  type="number"
                  step="50"
                  min="100"
                  max="2000"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-green-500"
                  value={trainingParams.rolling_window_size}
                  onChange={e =>
                    setTrainingParams({ ...trainingParams, rolling_window_size: parseInt(e.target.value) })
                  }
                  disabled={trainingStatus.is_training || !trainingParams.use_rolling_normalization}
                />
              </div>
            </div>
          </div>

          {/* Labeling Method Section */}
          <div className="pt-4 border-t border-gray-700">
            <p className="text-sm text-gray-400 mb-3">
              <strong>Labeling Method</strong> — метод разметки данных
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">
                  Метод разметки
                  <Tooltip content="fixed_threshold: простой порог движения цены. triple_barrier: адаптивный метод с учётом волатильности (López de Prado).">
                    <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
                  </Tooltip>
                </label>
                <select
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-green-500"
                  value={trainingParams.labeling_method}
                  onChange={e =>
                    setTrainingParams({ ...trainingParams, labeling_method: e.target.value as 'fixed_threshold' | 'triple_barrier' })
                  }
                  disabled={trainingStatus.is_training}
                >
                  <option value="fixed_threshold">Fixed Threshold (стандартный)</option>
                  <option value="triple_barrier">Triple Barrier (López de Prado)</option>
                </select>
              </div>

              {trainingParams.labeling_method === 'triple_barrier' && (
                <div className="flex items-center p-3 bg-blue-900/30 border border-blue-700/50 rounded-lg">
                  <Target className="h-5 w-5 text-blue-400 mr-2 flex-shrink-0" />
                  <span className="text-xs text-blue-400">
                    Triple Barrier автоматически применится при обучении.<br />
                    Настройте параметры ниже.
                  </span>
                </div>
              )}
            </div>

            {/* Triple Barrier Parameters - показываем только если выбран triple_barrier */}
            {trainingParams.labeling_method === 'triple_barrier' && (
              <div className="mt-4 p-4 bg-blue-900/10 border border-blue-800/30 rounded-lg">
                <p className="text-sm text-blue-400 mb-3 font-medium">
                  <Target className="inline h-4 w-4 mr-1" />
                  Triple Barrier Parameters
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Take Profit Multiplier */}
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">
                      Take Profit (ATR ×)
                      <Tooltip content="Множитель ATR для Take Profit. Например: 1.5 = TP на расстоянии 1.5 × ATR. Рекомендуется: 1.5-2.0">
                        <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
                      </Tooltip>
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      min="0.5"
                      max="5.0"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
                      value={trainingParams.tb_tp_multiplier}
                      onChange={e =>
                        setTrainingParams({ ...trainingParams, tb_tp_multiplier: parseFloat(e.target.value) })
                      }
                      disabled={trainingStatus.is_training}
                    />
                    <p className="text-xs text-gray-500 mt-1">Рекомендуется: 1.5</p>
                  </div>

                  {/* Stop Loss Multiplier */}
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">
                      Stop Loss (ATR ×)
                      <Tooltip content="Множитель ATR для Stop Loss. Например: 1.0 = SL на расстоянии 1.0 × ATR. Рекомендуется: 1.0-1.5">
                        <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
                      </Tooltip>
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      min="0.5"
                      max="5.0"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
                      value={trainingParams.tb_sl_multiplier}
                      onChange={e =>
                        setTrainingParams({ ...trainingParams, tb_sl_multiplier: parseFloat(e.target.value) })
                      }
                      disabled={trainingStatus.is_training}
                    />
                    <p className="text-xs text-gray-500 mt-1">Рекомендуется: 1.0</p>
                  </div>

                  {/* Max Holding Period */}
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">
                      Max Holding (bars)
                      <Tooltip content="Максимальное время удержания позиции в барах. Если TP/SL не сработал за это время - выход по времени. Рекомендуется: 24 для 1-min data.">
                        <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
                      </Tooltip>
                    </label>
                    <input
                      type="number"
                      step="1"
                      min="6"
                      max="100"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
                      value={trainingParams.tb_max_holding_period}
                      onChange={e =>
                        setTrainingParams({ ...trainingParams, tb_max_holding_period: parseInt(e.target.value) })
                      }
                      disabled={trainingStatus.is_training}
                    />
                    <p className="text-xs text-gray-500 mt-1">Рекомендуется: 24</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* CPCV Section */}
          <div className="pt-4 border-t border-gray-700">
            <p className="text-sm text-gray-400 mb-3">
              <strong>CPCV & PBO</strong> — Combinatorial Purged Cross-Validation и Probability of Backtest Overfitting
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <label className="flex items-center gap-3 cursor-pointer group">
                <input
                  type="checkbox"
                  className="w-5 h-5 rounded border-gray-700 text-green-500 focus:ring-green-500"
                  checked={trainingParams.use_cpcv}
                  onChange={e =>
                    setTrainingParams({ ...trainingParams, use_cpcv: e.target.checked })
                  }
                  disabled={trainingStatus.is_training}
                />
                <Tooltip content="CPCV создаёт все комбинации train/test сплитов без data leakage. Более надёжная оценка модели, но требует больше времени.">
                  <span className="text-sm text-gray-300 group-hover:text-white transition-colors cursor-help">
                    Enable CPCV (рекомендуется для финальной оценки)
                  </span>
                </Tooltip>
              </label>

              <label className="flex items-center gap-3 cursor-pointer group">
                <input
                  type="checkbox"
                  className="w-5 h-5 rounded border-gray-700 text-green-500 focus:ring-green-500"
                  checked={trainingParams.calculate_pbo}
                  onChange={e =>
                    setTrainingParams({ ...trainingParams, calculate_pbo: e.target.checked })
                  }
                  disabled={trainingStatus.is_training || !trainingParams.use_cpcv}
                />
                <Tooltip content="PBO оценивает вероятность переобучения на бэктесте. Значение <0.5 = хорошо, >0.5 = вероятно переобучение. Требует включенного CPCV.">
                  <span className={cn(
                    "text-sm transition-colors cursor-help",
                    trainingParams.use_cpcv ? "text-gray-300 group-hover:text-white" : "text-gray-500"
                  )}>
                    Calculate PBO (Probability of Backtest Overfitting)
                  </span>
                </Tooltip>
              </label>
            </div>

            {/* CPCV Parameters - показываем только если CPCV включен */}
            {trainingParams.use_cpcv && (
              <div className="mt-4 p-4 bg-purple-900/10 border border-purple-800/30 rounded-lg">
                <p className="text-sm text-purple-400 mb-3 font-medium">
                  <Gauge className="inline h-4 w-4 mr-1" />
                  CPCV Parameters
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* N Splits */}
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">
                      Number of Groups (N)
                      <Tooltip content="Количество групп для разбиения данных. Больше групп = больше комбинаций = надёжнее, но дольше. Рекомендуется: 6">
                        <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
                      </Tooltip>
                    </label>
                    <input
                      type="number"
                      step="1"
                      min="3"
                      max="12"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-purple-500"
                      value={trainingParams.cpcv_n_splits}
                      onChange={e =>
                        setTrainingParams({ ...trainingParams, cpcv_n_splits: parseInt(e.target.value) })
                      }
                      disabled={trainingStatus.is_training}
                    />
                    <p className="text-xs text-gray-500 mt-1">Рекомендуется: 6</p>
                  </div>

                  {/* N Test Splits */}
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-1">
                      Test Groups per Combo (K)
                      <Tooltip content="Количество тестовых групп в каждой комбинации. При N=6, K=2 получаем C(6,2)=15 комбинаций. Рекомендуется: 2">
                        <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
                      </Tooltip>
                    </label>
                    <input
                      type="number"
                      step="1"
                      min="1"
                      max="4"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-purple-500"
                      value={trainingParams.cpcv_n_test_splits}
                      onChange={e =>
                        setTrainingParams({ ...trainingParams, cpcv_n_test_splits: parseInt(e.target.value) })
                      }
                      disabled={trainingStatus.is_training}
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Комбинаций: {Math.round(
                        factorial(trainingParams.cpcv_n_splits) /
                        (factorial(trainingParams.cpcv_n_test_splits) * factorial(trainingParams.cpcv_n_splits - trainingParams.cpcv_n_test_splits))
                      )}
                    </p>
                  </div>
                </div>

                {/* Info about what CPCV does */}
                <div className="mt-3 p-2 bg-gray-800/50 rounded text-xs text-gray-400">
                  <strong>CPCV</strong> разбивает данные на {trainingParams.cpcv_n_splits} групп и создаёт все комбинации
                  {' '}train/test сплитов с purging для предотвращения data leakage.
                  {trainingParams.calculate_pbo && (
                    <span className="text-purple-400">
                      {' '}После обучения будет рассчитан <strong>PBO</strong> для оценки риска переобучения.
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
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
   * RENDER FUNCTIONS - MODELS TAB (with sub-tabs)
   * ============================================================
   */

  // Model type configuration
  const modelTypeConfig: Record<ModelSubTabType, { name: string; icon: any; color: string; filter: string }> = {
    cnn_lstm: { name: 'CNN-LSTM v2', icon: Rocket, color: 'text-blue-400', filter: 'hybrid_cnn_lstm' },
    mpd_transformer: { name: 'MPD Transformer', icon: Brain, color: 'text-purple-400', filter: 'mpd_transformer' },
    tlob: { name: 'TLOB Transformer', icon: Activity, color: 'text-green-400', filter: 'tlob' }
  };

  // Filter models by type
  const getFilteredModels = (modelType: ModelSubTabType) => {
    const filter = modelTypeConfig[modelType].filter;
    return models.filter(m => m.name.includes(filter) || m.name === filter);
  };

  // Render model table for a specific type
  const renderModelTable = (modelType: ModelSubTabType) => {
    const filteredModels = getFilteredModels(modelType);
    const config = modelTypeConfig[modelType];

    if (filteredModels.length === 0) {
      return (
        <div className="p-12 text-center">
          <config.icon className={cn('h-16 w-16 mx-auto mb-4', config.color, 'opacity-50')} />
          <p className="text-gray-400 text-lg mb-2">No {config.name} models found</p>
          <p className="text-gray-500 text-sm">
            Train your first {config.name} model to get started
          </p>
        </div>
      );
    }

    return (
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-800">
            <tr>
              <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">Version</th>
              <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">Stage</th>
              <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">Accuracy</th>
              <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">Precision</th>
              <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">Recall</th>
              <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">F1 Score</th>
              <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">Created</th>
              <th className="text-left py-4 px-6 text-sm font-medium text-gray-400">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {filteredModels.map(model => (
              <tr
                key={`${model.name}_${model.version}`}
                className="hover:bg-gray-800/50 transition-colors"
              >
                <td className="py-4 px-6">
                  <span className="font-mono text-sm text-white">{model.version}</span>
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
                      <button
                        onClick={() => handleDownloadModel(model.name, model.version)}
                        className="p-2 hover:bg-gray-700 rounded-lg transition-colors group"
                      >
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
    );
  };

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
            {/* Stage Filter */}
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

        {/* Model Type Sub-tabs */}
        <div className="flex gap-2 mt-6 border-t border-gray-700 pt-6">
          {(Object.keys(modelTypeConfig) as ModelSubTabType[]).map(type => {
            const config = modelTypeConfig[type];
            const Icon = config.icon;
            const count = getFilteredModels(type).length;

            return (
              <button
                key={type}
                onClick={() => setModelSubTab(type)}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg transition-all',
                  modelSubTab === type
                    ? 'bg-primary/20 text-primary border border-primary/50'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700 border border-transparent'
                )}
              >
                <Icon className={cn('h-4 w-4', modelSubTab === type ? 'text-primary' : config.color)} />
                <span>{config.name}</span>
                <span className={cn(
                  'px-2 py-0.5 text-xs rounded-full',
                  modelSubTab === type
                    ? 'bg-primary/30 text-primary'
                    : 'bg-gray-700 text-gray-400'
                )}>
                  {count}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Model Type Info Card */}
      <div className={cn(
        'bg-surface border rounded-lg p-4 flex items-start gap-4',
        modelSubTab === 'cnn_lstm' ? 'border-blue-500/30' :
        modelSubTab === 'mpd_transformer' ? 'border-purple-500/30' : 'border-green-500/30'
      )}>
        {modelSubTab === 'cnn_lstm' && (
          <>
            <Rocket className="h-8 w-8 text-blue-400 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-white">CNN-LSTM v2</h3>
              <p className="text-sm text-gray-400 mt-1">
                Hybrid model combining Convolutional Neural Networks for feature extraction with LSTM for temporal dependencies.
                Trained on 112 features (50 LOB + 25 Candle + 37 Indicators).
              </p>
            </div>
          </>
        )}
        {modelSubTab === 'mpd_transformer' && (
          <>
            <Brain className="h-8 w-8 text-purple-400 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-white">MPD Transformer</h3>
              <p className="text-sm text-gray-400 mt-1">
                Vision Transformer (ViT) adapted for financial time series. Converts 112 features × 60 timesteps
                to 2D patches and processes them with multi-head attention.
              </p>
            </div>
          </>
        )}
        {modelSubTab === 'tlob' && (
          <>
            <Activity className="h-8 w-8 text-green-400 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-white">TLOB Transformer</h3>
              <p className="text-sm text-gray-400 mt-1">
                Specialized model for raw Limit Order Book data. Combines Spatial CNN for cross-level patterns
                with Temporal Transformer for sequential dependencies. Multi-horizon predictions.
              </p>
            </div>
          </>
        )}
      </div>

      {/* Models Table */}
      <div className="bg-surface border border-gray-800 rounded-lg overflow-hidden">
        {renderModelTable(modelSubTab)}
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
   * RENDER FUNCTIONS - LAYERING MODEL TAB
   * ============================================================
   */

  const renderLayeringTab = () => (
      <div className="space-y-6">
        {/* Model Status Card */}
        <div className="bg-surface border border-gray-800 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-6">
            <Shield className="h-6 w-6 text-primary" />
            <h2 className="text-xl font-semibold">Layering Model Status</h2>
          </div>

          {layeringStatus ? (
            layeringStatus.loaded ? (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-5 w-5 text-green-400" />
                  <span className="text-green-400 font-semibold">Model Loaded</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 bg-gray-800 rounded-lg">
                    <p className="text-sm text-gray-400 mb-1">Version</p>
                    <p className="text-white font-semibold">
                      {layeringStatus.model_info?.version || 'N/A'}
                    </p>
                  </div>

                  <div className="p-4 bg-gray-800 rounded-lg">
                    <p className="text-sm text-gray-400 mb-1">Training Samples</p>
                    <p className="text-white font-semibold">
                      {layeringStatus.model_info?.training_samples?.toLocaleString() || 'N/A'}
                    </p>
                  </div>

                  <div className="p-4 bg-gray-800 rounded-lg">
                    <p className="text-sm text-gray-400 mb-1">Optimal Threshold</p>
                    <p className="text-white font-semibold">
                      {layeringStatus.model_info?.optimal_threshold?.toFixed(3) || 'N/A'}
                    </p>
                  </div>

                  <div className="p-4 bg-gray-800 rounded-lg">
                    <p className="text-sm text-gray-400 mb-1">Trained At</p>
                    <p className="text-white text-sm font-mono">
                      {layeringStatus.model_info?.trained_at
                        ? new Date(layeringStatus.model_info.trained_at).toLocaleString()
                        : 'N/A'}
                    </p>
                  </div>
                </div>

                <div className="p-4 bg-gray-800 rounded-lg">
                  <p className="text-sm text-gray-400 mb-2">Model Path</p>
                  <p className="text-xs font-mono text-white break-all">
                    {layeringStatus.model_info?.file_path || 'N/A'}
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <AlertTriangle className="h-16 w-16 text-yellow-400 mx-auto mb-4" />
                <p className="text-yellow-400 mb-2">Model Not Loaded</p>
                <p className="text-gray-400 text-sm mb-4">{layeringStatus.message}</p>
                <button
                  onClick={handleTrainLayeringModel}
                  disabled={isLayeringTraining || loading}
                  className="px-6 py-2 bg-primary hover:bg-primary/90 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Train Model
                </button>
              </div>
            )
          ) : (
            <div className="text-center py-8">
              <RefreshCw className="h-8 w-8 text-gray-400 mx-auto mb-2 animate-spin" />
              <p className="text-gray-400">Loading status...</p>
            </div>
          )}
        </div>

        {/* Data Status Card */}
        <div className="bg-surface border border-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <Database className="h-6 w-6 text-primary" />
              <h2 className="text-xl font-semibold">Training Data Status</h2>
            </div>
            <button
              onClick={handleCheckData}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
            >
              <RefreshCw className="h-4 w-4" />
              Check Data
            </button>
          </div>

          {dataStatus ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-gray-800 rounded-lg">
                  <p className="text-sm text-gray-400 mb-1">Total Collected</p>
                  <p className="text-2xl font-bold text-white">
                    {dataStatus.total_collected?.toLocaleString() || '0'}
                  </p>
                </div>

                <div className="p-4 bg-gray-800 rounded-lg">
                  <p className="text-sm text-gray-400 mb-1">Labeled Samples</p>
                  <p className="text-2xl font-bold text-green-400">
                    {dataStatus.total_labeled?.toLocaleString() || '0'}
                  </p>
                </div>

                <div className="p-4 bg-gray-800 rounded-lg">
                  <p className="text-sm text-gray-400 mb-1">Files Count</p>
                  <p className="text-2xl font-bold text-white">
                    {dataStatus.files_count || '0'}
                  </p>
                </div>
              </div>

              <div className={cn(
                'p-4 rounded-lg border-2',
                dataStatus.ready_for_training
                  ? 'bg-green-500/10 border-green-500/30'
                  : 'bg-yellow-500/10 border-yellow-500/30'
              )}>
                <div className="flex items-center gap-3">
                  {dataStatus.ready_for_training ? (
                    <CheckCircle className="h-5 w-5 text-green-400" />
                  ) : (
                    <AlertTriangle className="h-5 w-5 text-yellow-400" />
                  )}
                  <div>
                    <p className={cn(
                      'font-semibold',
                      dataStatus.ready_for_training ? 'text-green-400' : 'text-yellow-400'
                    )}>
                      {dataStatus.ready_for_training ? 'Ready for Training' : 'Collecting Data'}
                    </p>
                    <p className="text-sm text-gray-400">
                      {dataStatus.ready_for_training
                        ? `${dataStatus.total_labeled} labeled samples available`
                        : `Need ${dataStatus.minimum_required - (dataStatus.total_labeled || 0)} more labeled samples`}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <RefreshCw className="h-8 w-8 text-gray-400 mx-auto mb-2 animate-spin" />
              <p className="text-gray-400">Loading data status...</p>
            </div>
          )}
        </div>

        {/* Model Metrics Card */}
        {layeringMetrics && layeringMetrics.available && layeringMetrics.metrics && (
          <div className="bg-surface border border-gray-800 rounded-lg p-6">
            <div className="flex items-center gap-3 mb-6">
              <Target className="h-6 w-6 text-primary" />
              <h2 className="text-xl font-semibold">Model Metrics</h2>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 bg-gray-800 rounded-lg">
                <p className="text-sm text-gray-400 mb-2">Accuracy</p>
                <div className="flex items-baseline gap-2">
                  <p className="text-2xl font-bold text-white">
                    {(layeringMetrics.metrics.accuracy * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="mt-2 w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-green-400 rounded-full h-2"
                    style={{ width: `${layeringMetrics.metrics.accuracy * 100}%` }}
                  />
                </div>
              </div>

              <div className="p-4 bg-gray-800 rounded-lg">
                <p className="text-sm text-gray-400 mb-2">Precision</p>
                <div className="flex items-baseline gap-2">
                  <p className="text-2xl font-bold text-white">
                    {(layeringMetrics.metrics.precision * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="mt-2 w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-400 rounded-full h-2"
                    style={{ width: `${layeringMetrics.metrics.precision * 100}%` }}
                  />
                </div>
              </div>

              <div className="p-4 bg-gray-800 rounded-lg">
                <p className="text-sm text-gray-400 mb-2">Recall</p>
                <div className="flex items-baseline gap-2">
                  <p className="text-2xl font-bold text-white">
                    {(layeringMetrics.metrics.recall * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="mt-2 w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-yellow-400 rounded-full h-2"
                    style={{ width: `${layeringMetrics.metrics.recall * 100}%` }}
                  />
                </div>
              </div>

              <div className="p-4 bg-gray-800 rounded-lg">
                <p className="text-sm text-gray-400 mb-2">F1 Score</p>
                <div className="flex items-baseline gap-2">
                  <p className="text-2xl font-bold text-white">
                    {(layeringMetrics.metrics.f1_score * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="mt-2 w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-purple-400 rounded-full h-2"
                    style={{ width: `${layeringMetrics.metrics.f1_score * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {layeringMetrics.top_features && layeringMetrics.top_features.length > 0 && (
              <div className="mt-6 p-4 bg-gray-800 rounded-lg">
                <p className="text-sm font-medium text-gray-400 mb-3">Top 5 Important Features</p>
                <div className="space-y-2">
                  {layeringMetrics.top_features.slice(0, 5).map((feature: string, index: number) => (
                    <div key={index} className="flex items-center gap-3">
                      <span className="text-xs font-mono text-gray-500 w-6">{index + 1}.</span>
                      <span className="text-sm text-white font-mono flex-1">{feature}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Training Actions */}
        <div className="bg-surface border border-gray-800 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-6">
            <Rocket className="h-6 w-6 text-primary" />
            <h2 className="text-xl font-semibold">Training Actions</h2>
          </div>

          <div className="space-y-4">
            <div className="flex gap-4">
              <button
                onClick={handleTrainLayeringModel}
                disabled={isLayeringTraining || loading || !dataStatus?.ready_for_training}
                className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-primary hover:bg-primary/90 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Rocket className="h-5 w-5" />
                {isLayeringTraining ? 'Training in Progress...' : 'Train Improved Model'}
              </button>

              <Tooltip content="Refresh all status">
                <button
                  onClick={() => {
                    fetchLayeringStatus();
                    fetchDataStatus();
                    fetchLayeringMetrics();
                  }}
                  className="px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
                >
                  <RefreshCw className="h-5 w-5 text-gray-400" />
                </button>
              </Tooltip>
            </div>

            {!dataStatus?.ready_for_training && (
              <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <div className="flex items-start gap-3">
                  <Info className="h-5 w-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm text-yellow-400 font-medium">Not enough labeled data</p>
                    <p className="text-xs text-gray-400 mt-1">
                      Need at least {dataStatus?.minimum_required || 100} labeled samples for training.
                      Currently have {dataStatus?.total_labeled || 0}.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {layeringTrainingOutput && (
              <div className="p-4 bg-gray-900 rounded-lg border border-gray-700">
                <p className="text-sm text-gray-400 mb-2">Training Output:</p>
                <pre className="text-xs font-mono text-white overflow-x-auto max-h-64 overflow-y-auto whitespace-pre-wrap">
                  {layeringTrainingOutput}
                </pre>
              </div>
            )}
          </div>
        </div>
      </div>
    );

  /**
   * ============================================================
   * RENDER FUNCTIONS - OPTIMIZATION TAB
   * ============================================================
   */

  const renderOptimizationTab = () => (
    <div className="space-y-6">
      {/* Status Banner */}
      <div className={cn(
        'p-4 rounded-lg border',
        optimizationStatus.is_running
          ? 'bg-blue-500/10 border-blue-500/50'
          : optimizationStatus.can_resume
          ? 'bg-yellow-500/10 border-yellow-500/50'
          : 'bg-gray-800 border-gray-700'
      )}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {optimizationStatus.is_running ? (
              <RefreshCw className="h-6 w-6 text-blue-400 animate-spin" />
            ) : optimizationStatus.can_resume ? (
              <PauseCircle className="h-6 w-6 text-yellow-400" />
            ) : (
              <Target className="h-6 w-6 text-gray-400" />
            )}
            <div>
              <p className="font-medium">
                {optimizationStatus.is_running
                  ? `Оптимизация запущена: ${optimizationStatus.current_mode || 'full'}`
                  : optimizationStatus.can_resume
                  ? 'Оптимизация приостановлена'
                  : 'Оптимизация не запущена'}
              </p>
              {optimizationStatus.is_running && (
                <p className="text-sm text-gray-400">
                  Группа: {optimizationStatus.current_group || 'N/A'} |
                  Пробы: {optimizationStatus.trials_completed}/{optimizationStatus.total_trials} |
                  Время: {optimizationStatus.elapsed_time || '0:00'}
                  {optimizationStatus.estimated_remaining && ` (осталось: ${optimizationStatus.estimated_remaining})`}
                </p>
              )}
            </div>
          </div>
          <div className="flex gap-2">
            {optimizationStatus.is_running ? (
              <button
                onClick={handleStopOptimization}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors disabled:opacity-50"
              >
                <PauseCircle className="h-4 w-4" />
                Остановить
              </button>
            ) : optimizationStatus.can_resume ? (
              <>
                <button
                  onClick={handleResumeOptimization}
                  disabled={loading}
                  className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50"
                >
                  <PlayCircle className="h-4 w-4" />
                  Продолжить
                </button>
                <button
                  onClick={handleStartOptimization}
                  disabled={loading}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50"
                >
                  <Rocket className="h-4 w-4" />
                  Новый запуск
                </button>
              </>
            ) : (
              <button
                onClick={handleStartOptimization}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary/80 text-white rounded-lg transition-colors disabled:opacity-50"
              >
                <Rocket className="h-4 w-4" />
                Запустить оптимизацию
              </button>
            )}
          </div>
        </div>

        {/* Progress bar */}
        {optimizationStatus.is_running && optimizationStatus.total_trials > 0 && (
          <div className="mt-4">
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 rounded-full h-2 transition-all duration-500"
                style={{ width: `${(optimizationStatus.trials_completed / optimizationStatus.total_trials) * 100}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>{Math.round((optimizationStatus.trials_completed / optimizationStatus.total_trials) * 100)}%</span>
              {optimizationStatus.best_metric && (
                <span>Лучший результат: {(optimizationStatus.best_metric * 100).toFixed(2)}%</span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Data Paths Info */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-4">
          <Database className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-semibold">Пути данных</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-gray-800 rounded-lg">
            <Tooltip content="Путь к данным, используемым для обучения модели при оптимизации">
              <div className="flex items-center gap-2 text-sm text-gray-400 mb-2">
                <Download className="h-4 w-4" />
                <span>Источник данных для обучения</span>
                <Info className="h-3 w-3" />
              </div>
            </Tooltip>
            <p className="text-white font-mono text-sm break-all">
              {optimizationStatus.data_source_path || optimizationConfig.data_source === 'feature_store'
                ? 'data/feature_store/'
                : 'data/collected/'}
            </p>
          </div>
          <div className="p-4 bg-gray-800 rounded-lg">
            <Tooltip content="Путь для сохранения результатов оптимизации и состояния">
              <div className="flex items-center gap-2 text-sm text-gray-400 mb-2">
                <Upload className="h-4 w-4" />
                <span>Результаты оптимизации</span>
                <Info className="h-3 w-3" />
              </div>
            </Tooltip>
            <p className="text-white font-mono text-sm break-all">
              {optimizationStatus.results_path || 'data/hyperopt/'}
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuration */}
        <div className="bg-surface border border-gray-800 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-6">
            <Settings className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-semibold">Конфигурация оптимизации</h3>
          </div>

          <div className="space-y-4">
            {/* Mode Selection */}
            <div>
              <Tooltip content="Режим оптимизации определяет стратегию поиска гиперпараметров">
                <label className="flex items-center gap-2 text-sm font-medium text-gray-400 mb-2">
                  Режим оптимизации
                  <Info className="h-3 w-3" />
                </label>
              </Tooltip>
              <select
                value={optimizationConfig.mode}
                onChange={(e) => setOptimizationConfig(prev => ({ ...prev, mode: e.target.value as typeof prev.mode }))}
                disabled={optimizationStatus.is_running}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50"
              >
                <option value="full">Полная оптимизация (все группы параметров)</option>
                <option value="quick">Быстрая оптимизация (только ключевые параметры)</option>
                <option value="group">Оптимизация одной группы</option>
                <option value="resume">Возобновление предыдущего запуска</option>
                <option value="fine_tune">Тонкая настройка лучших параметров</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {optimizationConfig.mode === 'full' && 'Последовательно оптимизирует все группы: learning_rate → regularization → class_balance → augmentation → scheduler → triple_barrier'}
                {optimizationConfig.mode === 'quick' && 'Оптимизирует только learning_rate и regularization (~60% влияния на качество)'}
                {optimizationConfig.mode === 'group' && 'Оптимизирует только выбранную группу параметров'}
                {optimizationConfig.mode === 'resume' && 'Продолжает с последней сохранённой позиции'}
                {optimizationConfig.mode === 'fine_tune' && 'Сужает диапазон поиска вокруг лучших найденных значений'}
              </p>
            </div>

            {/* Target Group (only for group mode) */}
            {optimizationConfig.mode === 'group' && (
              <div>
                <Tooltip content="Группа параметров для оптимизации">
                  <label className="flex items-center gap-2 text-sm font-medium text-gray-400 mb-2">
                    Целевая группа
                    <Info className="h-3 w-3" />
                  </label>
                </Tooltip>
                <select
                  value={optimizationConfig.target_group}
                  onChange={(e) => setOptimizationConfig(prev => ({ ...prev, target_group: e.target.value }))}
                  disabled={optimizationStatus.is_running}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50"
                >
                  <option value="learning_rate">Learning Rate (~40% влияния)</option>
                  <option value="regularization">Regularization (~25% влияния)</option>
                  <option value="class_balance">Class Balance (~15% влияния)</option>
                  <option value="augmentation">Augmentation (~10% влияния)</option>
                  <option value="scheduler">Scheduler (~5% влияния)</option>
                  <option value="triple_barrier">Triple Barrier (~5% влияния)</option>
                </select>
              </div>
            )}

            {/* Epochs per trial */}
            <div>
              <Tooltip content="Количество эпох для каждой пробы. Больше эпох = точнее оценка, но дольше (~12 мин/эпоха)">
                <label className="flex items-center gap-2 text-sm font-medium text-gray-400 mb-2">
                  Эпох на пробу
                  <Info className="h-3 w-3" />
                </label>
              </Tooltip>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="2"
                  max="10"
                  value={optimizationConfig.epochs_per_trial}
                  onChange={(e) => setOptimizationConfig(prev => ({ ...prev, epochs_per_trial: parseInt(e.target.value) }))}
                  disabled={optimizationStatus.is_running}
                  className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
                />
                <span className="text-white font-medium w-12 text-center">{optimizationConfig.epochs_per_trial}</span>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Время на пробу: ~{optimizationConfig.epochs_per_trial * 12} минут
              </p>
            </div>

            {/* Max trials per group */}
            <div>
              <Tooltip content="Максимальное количество проб для каждой группы параметров">
                <label className="flex items-center gap-2 text-sm font-medium text-gray-400 mb-2">
                  Максимум проб на группу
                  <Info className="h-3 w-3" />
                </label>
              </Tooltip>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="5"
                  max="30"
                  value={optimizationConfig.max_trials_per_group}
                  onChange={(e) => setOptimizationConfig(prev => ({ ...prev, max_trials_per_group: parseInt(e.target.value) }))}
                  disabled={optimizationStatus.is_running}
                  className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
                />
                <span className="text-white font-medium w-12 text-center">{optimizationConfig.max_trials_per_group}</span>
              </div>
            </div>

            {/* Max total hours */}
            <div>
              <Tooltip content="Максимальное время работы оптимизации в часах">
                <label className="flex items-center gap-2 text-sm font-medium text-gray-400 mb-2">
                  Лимит времени (часы)
                  <Info className="h-3 w-3" />
                </label>
              </Tooltip>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="1"
                  max="72"
                  value={optimizationConfig.max_total_hours}
                  onChange={(e) => setOptimizationConfig(prev => ({ ...prev, max_total_hours: parseInt(e.target.value) }))}
                  disabled={optimizationStatus.is_running}
                  className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
                />
                <span className="text-white font-medium w-16 text-center">{optimizationConfig.max_total_hours}ч</span>
              </div>
            </div>

            {/* Primary Metric */}
            <div>
              <Tooltip content="Основная метрика для оптимизации">
                <label className="flex items-center gap-2 text-sm font-medium text-gray-400 mb-2">
                  Целевая метрика
                  <Info className="h-3 w-3" />
                </label>
              </Tooltip>
              <select
                value={optimizationConfig.primary_metric}
                onChange={(e) => setOptimizationConfig(prev => ({ ...prev, primary_metric: e.target.value }))}
                disabled={optimizationStatus.is_running}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50"
              >
                <option value="val_f1">F1-Score (рекомендуется)</option>
                <option value="val_accuracy">Accuracy</option>
                <option value="val_precision">Precision</option>
                <option value="val_recall">Recall</option>
              </select>
            </div>

            {/* Advanced options */}
            <div className="border-t border-gray-700 pt-4 mt-4">
              <p className="text-sm font-medium text-gray-400 mb-3">Дополнительные опции</p>
              <div className="space-y-3">
                <Tooltip content="Раннее отсечение неудачных проб для экономии времени (~60% экономии)">
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={optimizationConfig.enable_pruning}
                      onChange={(e) => setOptimizationConfig(prev => ({ ...prev, enable_pruning: e.target.checked }))}
                      disabled={optimizationStatus.is_running}
                      className="w-4 h-4 text-primary bg-gray-800 border-gray-600 rounded focus:ring-primary disabled:opacity-50"
                    />
                    <span className="text-sm text-gray-300">Включить раннее отсечение (Pruning)</span>
                    <Info className="h-3 w-3 text-gray-500" />
                  </label>
                </Tooltip>

                <Tooltip content="Начать с рекомендованных значений вместо случайных">
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={optimizationConfig.use_warm_start}
                      onChange={(e) => setOptimizationConfig(prev => ({ ...prev, use_warm_start: e.target.checked }))}
                      disabled={optimizationStatus.is_running}
                      className="w-4 h-4 text-primary bg-gray-800 border-gray-600 rounded focus:ring-primary disabled:opacity-50"
                    />
                    <span className="text-sm text-gray-300">Тёплый старт (Warm Start)</span>
                    <Info className="h-3 w-3 text-gray-500" />
                  </label>
                </Tooltip>
              </div>
            </div>
          </div>
        </div>

        {/* Best Parameters */}
        <div className="bg-surface border border-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <TrendingUp className="h-5 w-5 text-green-400" />
              <h3 className="text-lg font-semibold">Лучшие найденные параметры</h3>
            </div>
            {bestParams && (
              <button
                onClick={handleApplyBestParams}
                className="flex items-center gap-2 px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors"
              >
                <CheckCircle className="h-4 w-4" />
                Применить
              </button>
            )}
          </div>

          {bestParams ? (
            <div className="space-y-3">
              {Object.entries(bestParams).map(([key, value]) => (
                <div key={key} className="flex justify-between items-center py-2 border-b border-gray-700 last:border-0">
                  <span className="text-gray-400 text-sm">{key}</span>
                  <span className="text-white font-mono text-sm">
                    {typeof value === 'number' ? value.toFixed(6) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Target className="h-12 w-12 text-gray-600 mx-auto mb-3" />
              <p className="text-gray-400">Запустите оптимизацию для поиска лучших параметров</p>
            </div>
          )}
        </div>
      </div>

      {/* Trial History */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Clock className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-semibold">История проб</h3>
          </div>
          <button
            onClick={fetchOptimizationHistory}
            className="flex items-center gap-2 text-gray-400 hover:text-white text-sm"
          >
            <RefreshCw className="h-4 w-4" />
            Обновить
          </button>
        </div>

        {optimizationHistory.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-400 border-b border-gray-700">
                  <th className="pb-3 pr-4">#</th>
                  <th className="pb-3 pr-4">Группа</th>
                  <th className="pb-3 pr-4">Статус</th>
                  <th className="pb-3 pr-4">F1</th>
                  <th className="pb-3 pr-4">Accuracy</th>
                  <th className="pb-3 pr-4">Время (мин)</th>
                  <th className="pb-3">Параметры</th>
                </tr>
              </thead>
              <tbody>
                {optimizationHistory.slice(0, 20).map((trial) => (
                  <tr key={trial.trial_id} className="border-b border-gray-800 hover:bg-gray-800/50">
                    <td className="py-3 pr-4 text-gray-400">{trial.trial_id}</td>
                    <td className="py-3 pr-4">
                      <span className="px-2 py-1 bg-blue-500/20 text-blue-400 text-xs rounded">
                        {trial.group}
                      </span>
                    </td>
                    <td className="py-3 pr-4">
                      <span className={cn(
                        'px-2 py-1 text-xs rounded',
                        trial.status === 'COMPLETE' ? 'bg-green-500/20 text-green-400' :
                        trial.status === 'PRUNED' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      )}>
                        {trial.status}
                      </span>
                    </td>
                    <td className="py-3 pr-4 font-mono">
                      {trial.metrics?.val_f1 ? (trial.metrics.val_f1 * 100).toFixed(2) + '%' : '-'}
                    </td>
                    <td className="py-3 pr-4 font-mono">
                      {trial.metrics?.val_accuracy ? (trial.metrics.val_accuracy * 100).toFixed(2) + '%' : '-'}
                    </td>
                    <td className="py-3 pr-4 text-gray-400">
                      {trial.duration_minutes?.toFixed(1) || '-'}
                    </td>
                    <td className="py-3">
                      <Tooltip content={JSON.stringify(trial.params, null, 2)}>
                        <span className="text-gray-400 cursor-help flex items-center gap-1">
                          <Info className="h-3 w-3" />
                          {Object.keys(trial.params || {}).length} params
                        </span>
                      </Tooltip>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {optimizationHistory.length > 20 && (
              <p className="text-center text-gray-400 text-sm mt-4">
                Показаны последние 20 из {optimizationHistory.length} проб
              </p>
            )}
          </div>
        ) : (
          <div className="text-center py-8">
            <Activity className="h-12 w-12 text-gray-600 mx-auto mb-3" />
            <p className="text-gray-400">Нет истории оптимизации</p>
            <p className="text-gray-500 text-sm mt-1">Запустите оптимизацию для начала поиска</p>
          </div>
        )}
      </div>

      {/* Current Trial Details */}
      {optimizationStatus.is_running && optimizationStatus.current_trial_params && (
        <div className="bg-surface border border-blue-500/50 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <RefreshCw className="h-5 w-5 text-blue-400 animate-spin" />
            <h3 className="text-lg font-semibold">Текущая проба</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(optimizationStatus.current_trial_params).map(([key, value]) => (
              <div key={key} className="p-3 bg-gray-800 rounded-lg">
                <p className="text-xs text-gray-400 mb-1">{key}</p>
                <p className="text-white font-mono text-sm">
                  {typeof value === 'number' ? value.toFixed(6) : String(value)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  /**
   * ============================================================
   * RENDER FUNCTIONS - MPD TRANSFORMER TRAINING TAB
   * ============================================================
   */

  // Handle MPD training start
  const handleStartMPDTraining = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/ensemble/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_type: 'mpd_transformer',
          epochs: mpdTrainingParams.epochs,
          learning_rate: mpdTrainingParams.learning_rate,
          symbols: mpdTrainingParams.symbols,
          days: mpdTrainingParams.days
        })
      });

      const result = await response.json();

      if (result.success) {
        setMpdTrainingStatus(prev => ({
          ...prev,
          is_training: true,
          task_id: result.task_id,
          status: 'running',
          total_epochs: mpdTrainingParams.epochs
        }));

        // Start polling for status
        const pollStatus = async () => {
          const statusResponse = await fetch(`/api/ensemble/training/status/${result.task_id}`);
          const statusData = await statusResponse.json();

          setMpdTrainingStatus(prev => ({
            ...prev,
            current_epoch: statusData.current_epoch || 0,
            progress: statusData.progress || 0,
            status: statusData.status,
            error: statusData.error
          }));

          if (statusData.status === 'running') {
            setTimeout(pollStatus, 2000);
          } else {
            setMpdTrainingStatus(prev => ({
              ...prev,
              is_training: false
            }));
          }
        };

        pollStatus();
      } else {
        setError(result.message || 'Failed to start training');
      }
    } catch (err) {
      setError('Failed to start MPD training');
      console.error('MPD Training failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const renderMPDTrainingTab = () => (
    <div className="space-y-6">
      {/* Training Status */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-6">
          <Brain className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold">MPD Transformer Training</h2>
          {mpdTrainingStatus.is_training && (
            <span className="px-3 py-1 bg-primary/20 text-primary rounded-full text-sm animate-pulse">
              Training...
            </span>
          )}
        </div>

        {mpdTrainingStatus.is_training ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Task ID</p>
                <p className="font-mono text-white">{mpdTrainingStatus.task_id}</p>
              </div>
              <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm">
                {mpdTrainingStatus.status}
              </span>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">
                  Epoch {mpdTrainingStatus.current_epoch} / {mpdTrainingStatus.total_epochs}
                </span>
                <span className="text-white">{mpdTrainingStatus.progress}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-primary to-blue-500 rounded-full h-3 transition-all duration-500"
                  style={{ width: `${mpdTrainingStatus.progress}%` }}
                />
              </div>
            </div>

            {mpdTrainingStatus.error && (
              <div className="p-4 bg-red-500/10 border border-red-500/50 rounded-lg">
                <p className="text-red-400">{mpdTrainingStatus.error}</p>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-6">
            <p className="text-gray-400 mb-2">Configure and start MPD Transformer training</p>
            <p className="text-sm text-gray-500">
              Vision Transformer model for financial time series analysis
            </p>
          </div>
        )}
      </div>

      {/* Data Source */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-6">
          <Database className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold">Data Source</h2>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Data Path
            </label>
            <input
              type="text"
              placeholder="D:\PYTHON\Bot_ver3_stakan_new\data"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white font-mono text-sm focus:outline-none focus:border-primary"
              value={mpdTrainingParams.data_path}
              onChange={e => setMpdTrainingParams({ ...mpdTrainingParams, data_path: e.target.value })}
              disabled={mpdTrainingStatus.is_training}
            />
          </div>

          {/* Symbol Selection */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="block text-sm font-medium text-gray-400">
                Symbols ({mpdTrainingParams.symbols.length} selected)
              </label>
              <button
                onClick={() => setShowMpdSymbolSelector(!showMpdSymbolSelector)}
                className="text-sm text-primary hover:text-primary/80"
                disabled={mpdTrainingStatus.is_training}
              >
                {showMpdSymbolSelector ? 'Hide' : 'Show'} Selector
              </button>
            </div>

            {/* Selected symbols preview */}
            <div className="flex flex-wrap gap-1 min-h-[36px] bg-gray-800 border border-gray-700 rounded-lg px-3 py-2">
              {mpdTrainingParams.symbols.length === 0 ? (
                <span className="text-gray-500 text-sm">No symbols selected</span>
              ) : mpdTrainingParams.symbols.length <= 5 ? (
                mpdTrainingParams.symbols.map(s => (
                  <span key={s} className="px-2 py-0.5 bg-primary/20 text-primary text-xs rounded">
                    {s.replace('USDT', '')}
                  </span>
                ))
              ) : (
                <span className="text-gray-400 text-sm">
                  {mpdTrainingParams.symbols.slice(0, 3).map(s => s.replace('USDT', '')).join(', ')}
                  ... +{mpdTrainingParams.symbols.length - 3} more
                </span>
              )}
            </div>

            {/* Symbol selector panel */}
            {showMpdSymbolSelector && (
              <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-4">
                {/* Preset Groups */}
                <div className="flex flex-wrap gap-2">
                  <span className="text-xs text-gray-500 mr-2">Presets:</span>
                  {Object.keys(symbolGroups).map(group => (
                    <button
                      key={group}
                      onClick={() => selectSymbolGroup(group, setMpdTrainingParams, mpdTrainingParams)}
                      className="px-2 py-1 text-xs bg-gray-800 hover:bg-gray-700 rounded border border-gray-600 text-gray-300"
                      disabled={mpdTrainingStatus.is_training}
                    >
                      {group.replace('_', ' ')} ({symbolGroups[group]?.length || 0})
                    </button>
                  ))}
                  <button
                    onClick={() => selectAllSymbols(setMpdTrainingParams, mpdTrainingParams)}
                    className="px-2 py-1 text-xs bg-green-900/50 hover:bg-green-900 rounded border border-green-700 text-green-400"
                    disabled={mpdTrainingStatus.is_training}
                  >
                    All ({availableSymbols.length})
                  </button>
                  <button
                    onClick={() => clearAllSymbols(setMpdTrainingParams, mpdTrainingParams)}
                    className="px-2 py-1 text-xs bg-red-900/50 hover:bg-red-900 rounded border border-red-700 text-red-400"
                    disabled={mpdTrainingStatus.is_training}
                  >
                    Clear
                  </button>
                </div>

                {/* Symbol checkboxes */}
                <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-1 max-h-48 overflow-y-auto">
                  {loadingSymbols ? (
                    <span className="text-gray-500 col-span-full">Loading symbols...</span>
                  ) : availableSymbols.length === 0 ? (
                    <span className="text-gray-500 col-span-full">No data available. Collect data first.</span>
                  ) : (
                    availableSymbols.map(symbol => (
                      <label
                        key={symbol}
                        className={cn(
                          "flex items-center gap-1 px-2 py-1 rounded cursor-pointer text-xs",
                          mpdTrainingParams.symbols.includes(symbol)
                            ? "bg-primary/20 text-primary"
                            : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                        )}
                      >
                        <input
                          type="checkbox"
                          checked={mpdTrainingParams.symbols.includes(symbol)}
                          onChange={() => toggleSymbol(mpdTrainingParams.symbols, symbol, setMpdTrainingParams, mpdTrainingParams)}
                          className="sr-only"
                          disabled={mpdTrainingStatus.is_training}
                        />
                        {symbol.replace('USDT', '')}
                      </label>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Days of Data */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Days of Data</label>
            <input
              type="number"
              min="1"
              max="365"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={mpdTrainingParams.days}
              onChange={e => setMpdTrainingParams({ ...mpdTrainingParams, days: parseInt(e.target.value) })}
              disabled={mpdTrainingStatus.is_training}
            />
          </div>
        </div>
      </div>

      {/* Training Configuration */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-6">
          <Settings className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold">Training Configuration</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Epochs</label>
            <input
              type="number"
              min="1"
              max="500"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={mpdTrainingParams.epochs}
              onChange={e => setMpdTrainingParams({ ...mpdTrainingParams, epochs: parseInt(e.target.value) })}
              disabled={mpdTrainingStatus.is_training}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Batch Size</label>
            <input
              type="number"
              min="8"
              max="128"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={mpdTrainingParams.batch_size}
              onChange={e => setMpdTrainingParams({ ...mpdTrainingParams, batch_size: parseInt(e.target.value) })}
              disabled={mpdTrainingStatus.is_training}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Learning Rate</label>
            <input
              type="number"
              step="0.00001"
              min="0.00001"
              max="0.01"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={mpdTrainingParams.learning_rate}
              onChange={e => setMpdTrainingParams({ ...mpdTrainingParams, learning_rate: parseFloat(e.target.value) })}
              disabled={mpdTrainingStatus.is_training}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Embed Dimension</label>
            <select
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={mpdTrainingParams.embed_dim}
              onChange={e => setMpdTrainingParams({ ...mpdTrainingParams, embed_dim: parseInt(e.target.value) })}
              disabled={mpdTrainingStatus.is_training}
            >
              <option value="128">128</option>
              <option value="256">256</option>
              <option value="512">512</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Transformer Layers</label>
            <select
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={mpdTrainingParams.num_layers}
              onChange={e => setMpdTrainingParams({ ...mpdTrainingParams, num_layers: parseInt(e.target.value) })}
              disabled={mpdTrainingStatus.is_training}
            >
              <option value="4">4</option>
              <option value="6">6</option>
              <option value="8">8</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Attention Heads</label>
            <select
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={mpdTrainingParams.num_heads}
              onChange={e => setMpdTrainingParams({ ...mpdTrainingParams, num_heads: parseInt(e.target.value) })}
              disabled={mpdTrainingStatus.is_training}
            >
              <option value="4">4</option>
              <option value="8">8</option>
              <option value="16">16</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Dropout</label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="0.5"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={mpdTrainingParams.dropout}
              onChange={e => setMpdTrainingParams({ ...mpdTrainingParams, dropout: parseFloat(e.target.value) })}
              disabled={mpdTrainingStatus.is_training}
            />
          </div>
        </div>

        {/* Start Training Button */}
        <button
          onClick={handleStartMPDTraining}
          disabled={mpdTrainingStatus.is_training || loading}
          className={cn(
            'w-full py-4 rounded-lg font-semibold flex items-center justify-center gap-3 transition-all',
            mpdTrainingStatus.is_training || loading
              ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-primary to-blue-500 text-white hover:opacity-90'
          )}
        >
          {mpdTrainingStatus.is_training ? (
            <>
              <RefreshCw className="h-5 w-5 animate-spin" />
              Training in Progress...
            </>
          ) : (
            <>
              <PlayCircle className="h-5 w-5" />
              Start MPD Transformer Training
            </>
          )}
        </button>
      </div>

      {/* Model Info */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-4">
          <Info className="h-6 w-6 text-blue-400" />
          <h2 className="text-xl font-semibold">MPD Transformer Info</h2>
        </div>
        <div className="space-y-2 text-sm text-gray-400">
          <p><span className="text-gray-300">Architecture:</span> Vision Transformer (ViT) adapted for time series</p>
          <p><span className="text-gray-300">Input:</span> 112 features × 60 timesteps converted to 2D patches</p>
          <p><span className="text-gray-300">Output:</span> Direction (3 classes), Confidence, Expected Return</p>
          <p><span className="text-gray-300">GPU Memory:</span> ~4-6 GB for training</p>
        </div>
      </div>
    </div>
  );

  /**
   * ============================================================
   * RENDER FUNCTIONS - TLOB TRANSFORMER TRAINING TAB
   * ============================================================
   */

  // Handle TLOB training start
  const handleStartTLOBTraining = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/ensemble/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_type: 'tlob',
          epochs: tlobTrainingParams.epochs,
          learning_rate: tlobTrainingParams.learning_rate,
          symbols: tlobTrainingParams.symbols,
          days: tlobTrainingParams.days
        })
      });

      const result = await response.json();

      if (result.success) {
        setTlobTrainingStatus(prev => ({
          ...prev,
          is_training: true,
          task_id: result.task_id,
          status: 'running',
          total_epochs: tlobTrainingParams.epochs
        }));

        // Start polling for status
        const pollStatus = async () => {
          const statusResponse = await fetch(`/api/ensemble/training/status/${result.task_id}`);
          const statusData = await statusResponse.json();

          setTlobTrainingStatus(prev => ({
            ...prev,
            current_epoch: statusData.current_epoch || 0,
            progress: statusData.progress || 0,
            status: statusData.status,
            error: statusData.error
          }));

          if (statusData.status === 'running') {
            setTimeout(pollStatus, 2000);
          } else {
            setTlobTrainingStatus(prev => ({
              ...prev,
              is_training: false
            }));
          }
        };

        pollStatus();
      } else {
        setError(result.message || 'Failed to start training');
      }
    } catch (err) {
      setError('Failed to start TLOB training');
      console.error('TLOB Training failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const renderTLOBTrainingTab = () => (
    <div className="space-y-6">
      {/* Training Status */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-6">
          <Activity className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold">TLOB Transformer Training</h2>
          {tlobTrainingStatus.is_training && (
            <span className="px-3 py-1 bg-primary/20 text-primary rounded-full text-sm animate-pulse">
              Training...
            </span>
          )}
        </div>

        {tlobTrainingStatus.is_training ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Task ID</p>
                <p className="font-mono text-white">{tlobTrainingStatus.task_id}</p>
              </div>
              <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm">
                {tlobTrainingStatus.status}
              </span>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">
                  Epoch {tlobTrainingStatus.current_epoch} / {tlobTrainingStatus.total_epochs}
                </span>
                <span className="text-white">{tlobTrainingStatus.progress}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-green-500 to-teal-500 rounded-full h-3 transition-all duration-500"
                  style={{ width: `${tlobTrainingStatus.progress}%` }}
                />
              </div>
            </div>

            {tlobTrainingStatus.error && (
              <div className="p-4 bg-red-500/10 border border-red-500/50 rounded-lg">
                <p className="text-red-400">{tlobTrainingStatus.error}</p>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-6">
            <p className="text-gray-400 mb-2">Configure and start TLOB Transformer training</p>
            <p className="text-sm text-gray-500">
              Specialized model for Limit Order Book data analysis
            </p>
          </div>
        )}
      </div>

      {/* Important Notice */}
      <div className="bg-yellow-500/10 border border-yellow-500/50 rounded-lg p-4 flex items-start gap-3">
        <AlertTriangle className="h-5 w-5 text-yellow-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-yellow-400 font-medium">Raw LOB Data Required</p>
          <p className="text-yellow-300 text-sm mt-1">
            TLOB Transformer requires raw order book data. Make sure the Raw LOB Collector has been running
            and collecting data before training. Without sufficient LOB data, training will fail.
          </p>
        </div>
      </div>

      {/* Data Source */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-6">
          <Database className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold">Data Source</h2>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Data Path (Raw LOB)
            </label>
            <input
              type="text"
              placeholder="D:\PYTHON\Bot_ver3_stakan_new\data"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white font-mono text-sm focus:outline-none focus:border-primary"
              value={tlobTrainingParams.data_path}
              onChange={e => setTlobTrainingParams({ ...tlobTrainingParams, data_path: e.target.value })}
              disabled={tlobTrainingStatus.is_training}
            />
          </div>

          {/* Symbol Selection */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="block text-sm font-medium text-gray-400">
                Symbols ({tlobTrainingParams.symbols.length} selected)
                <span className="ml-2 text-xs text-green-400">[Normalized for multi-symbol]</span>
              </label>
              <button
                onClick={() => setShowTlobSymbolSelector(!showTlobSymbolSelector)}
                className="text-sm text-primary hover:text-primary/80"
                disabled={tlobTrainingStatus.is_training}
              >
                {showTlobSymbolSelector ? 'Hide' : 'Show'} Selector
              </button>
            </div>

            {/* Selected symbols preview */}
            <div className="flex flex-wrap gap-1 min-h-[36px] bg-gray-800 border border-gray-700 rounded-lg px-3 py-2">
              {tlobTrainingParams.symbols.length === 0 ? (
                <span className="text-gray-500 text-sm">No symbols selected</span>
              ) : tlobTrainingParams.symbols.length <= 5 ? (
                tlobTrainingParams.symbols.map(s => (
                  <span key={s} className="px-2 py-0.5 bg-primary/20 text-primary text-xs rounded">
                    {s.replace('USDT', '')}
                  </span>
                ))
              ) : (
                <span className="text-gray-400 text-sm">
                  {tlobTrainingParams.symbols.slice(0, 3).map(s => s.replace('USDT', '')).join(', ')}
                  ... +{tlobTrainingParams.symbols.length - 3} more
                </span>
              )}
            </div>

            {/* Symbol selector panel */}
            {showTlobSymbolSelector && (
              <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-4">
                {/* Info about normalization */}
                <div className="text-xs text-gray-500 bg-gray-800 p-2 rounded">
                  <Info className="inline h-3 w-3 mr-1" />
                  TLOB uses normalized data: prices in basis points (relative to mid-price), volumes log-transformed.
                  This enables training on multiple symbols with different price scales.
                </div>

                {/* Preset Groups */}
                <div className="flex flex-wrap gap-2">
                  <span className="text-xs text-gray-500 mr-2">Presets:</span>
                  {Object.keys(symbolGroups).map(group => (
                    <button
                      key={group}
                      onClick={() => selectSymbolGroup(group, setTlobTrainingParams, tlobTrainingParams)}
                      className="px-2 py-1 text-xs bg-gray-800 hover:bg-gray-700 rounded border border-gray-600 text-gray-300"
                      disabled={tlobTrainingStatus.is_training}
                    >
                      {group.replace('_', ' ')} ({symbolGroups[group]?.length || 0})
                    </button>
                  ))}
                  <button
                    onClick={() => selectAllSymbols(setTlobTrainingParams, tlobTrainingParams)}
                    className="px-2 py-1 text-xs bg-green-900/50 hover:bg-green-900 rounded border border-green-700 text-green-400"
                    disabled={tlobTrainingStatus.is_training}
                  >
                    All ({availableSymbols.length})
                  </button>
                  <button
                    onClick={() => clearAllSymbols(setTlobTrainingParams, tlobTrainingParams)}
                    className="px-2 py-1 text-xs bg-red-900/50 hover:bg-red-900 rounded border border-red-700 text-red-400"
                    disabled={tlobTrainingStatus.is_training}
                  >
                    Clear
                  </button>
                </div>

                {/* Symbol checkboxes */}
                <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-1 max-h-48 overflow-y-auto">
                  {loadingSymbols ? (
                    <span className="text-gray-500 col-span-full">Loading symbols...</span>
                  ) : availableSymbols.length === 0 ? (
                    <span className="text-gray-500 col-span-full">No data available. Collect Raw LOB data first.</span>
                  ) : (
                    availableSymbols.map(symbol => (
                      <label
                        key={symbol}
                        className={cn(
                          "flex items-center gap-1 px-2 py-1 rounded cursor-pointer text-xs",
                          tlobTrainingParams.symbols.includes(symbol)
                            ? "bg-primary/20 text-primary"
                            : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                        )}
                      >
                        <input
                          type="checkbox"
                          checked={tlobTrainingParams.symbols.includes(symbol)}
                          onChange={() => toggleSymbol(tlobTrainingParams.symbols, symbol, setTlobTrainingParams, tlobTrainingParams)}
                          className="sr-only"
                          disabled={tlobTrainingStatus.is_training}
                        />
                        {symbol.replace('USDT', '')}
                      </label>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Days of Data */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Days of Data</label>
            <input
              type="number"
              min="1"
              max="365"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={tlobTrainingParams.days}
              onChange={e => setTlobTrainingParams({ ...tlobTrainingParams, days: parseInt(e.target.value) })}
              disabled={tlobTrainingStatus.is_training}
            />
          </div>
        </div>
      </div>

      {/* Training Configuration */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-6">
          <Settings className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold">Training Configuration</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Epochs</label>
            <input
              type="number"
              min="1"
              max="500"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={tlobTrainingParams.epochs}
              onChange={e => setTlobTrainingParams({ ...tlobTrainingParams, epochs: parseInt(e.target.value) })}
              disabled={tlobTrainingStatus.is_training}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Batch Size</label>
            <input
              type="number"
              min="4"
              max="64"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={tlobTrainingParams.batch_size}
              onChange={e => setTlobTrainingParams({ ...tlobTrainingParams, batch_size: parseInt(e.target.value) })}
              disabled={tlobTrainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">TLOB needs more memory, use smaller batch</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Learning Rate</label>
            <input
              type="number"
              step="0.00001"
              min="0.00001"
              max="0.01"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={tlobTrainingParams.learning_rate}
              onChange={e => setTlobTrainingParams({ ...tlobTrainingParams, learning_rate: parseFloat(e.target.value) })}
              disabled={tlobTrainingStatus.is_training}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Order Book Levels</label>
            <select
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={tlobTrainingParams.num_levels}
              onChange={e => setTlobTrainingParams({ ...tlobTrainingParams, num_levels: parseInt(e.target.value) })}
              disabled={tlobTrainingStatus.is_training}
            >
              <option value="10">10 levels</option>
              <option value="20">20 levels</option>
              <option value="50">50 levels</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Sequence Length</label>
            <select
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={tlobTrainingParams.sequence_length}
              onChange={e => setTlobTrainingParams({ ...tlobTrainingParams, sequence_length: parseInt(e.target.value) })}
              disabled={tlobTrainingStatus.is_training}
            >
              <option value="50">50 snapshots</option>
              <option value="100">100 snapshots</option>
              <option value="200">200 snapshots</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Temporal Layers</label>
            <select
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={tlobTrainingParams.num_temporal_layers}
              onChange={e => setTlobTrainingParams({ ...tlobTrainingParams, num_temporal_layers: parseInt(e.target.value) })}
              disabled={tlobTrainingStatus.is_training}
            >
              <option value="2">2</option>
              <option value="4">4</option>
              <option value="6">6</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Dropout</label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="0.5"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={tlobTrainingParams.dropout}
              onChange={e => setTlobTrainingParams({ ...tlobTrainingParams, dropout: parseFloat(e.target.value) })}
              disabled={tlobTrainingStatus.is_training}
            />
          </div>
        </div>

        {/* Start Training Button */}
        <button
          onClick={handleStartTLOBTraining}
          disabled={tlobTrainingStatus.is_training || loading}
          className={cn(
            'w-full py-4 rounded-lg font-semibold flex items-center justify-center gap-3 transition-all',
            tlobTrainingStatus.is_training || loading
              ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-green-500 to-teal-500 text-white hover:opacity-90'
          )}
        >
          {tlobTrainingStatus.is_training ? (
            <>
              <RefreshCw className="h-5 w-5 animate-spin" />
              Training in Progress...
            </>
          ) : (
            <>
              <PlayCircle className="h-5 w-5" />
              Start TLOB Transformer Training
            </>
          )}
        </button>
      </div>

      {/* Model Info */}
      <div className="bg-surface border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-4">
          <Info className="h-6 w-6 text-blue-400" />
          <h2 className="text-xl font-semibold">TLOB Transformer Info</h2>
        </div>
        <div className="space-y-2 text-sm text-gray-400">
          <p><span className="text-gray-300">Architecture:</span> Spatial CNN + Temporal Transformer</p>
          <p><span className="text-gray-300">Input:</span> Raw LOB data (levels × 4: bid_price, bid_vol, ask_price, ask_vol)</p>
          <p><span className="text-gray-300">Output:</span> Multi-horizon predictions (10, 30, 50 steps)</p>
          <p><span className="text-gray-300">GPU Memory:</span> ~6-8 GB for training</p>
          <p><span className="text-gray-300">Data Required:</span> Raw order book snapshots from LOB Collector</p>
        </div>
      </div>
    </div>
  );

  /**
   * ============================================================
   * MAIN RENDER
   * ============================================================
   */

  const tabs: { id: TabType; label: string; icon: any }[] = [
    { id: 'training', label: 'CNN-LSTM Training', icon: Rocket },
    { id: 'mpd_training', label: 'MPD Training', icon: Brain },
    { id: 'tlob_training', label: 'TLOB Training', icon: Activity },
    { id: 'models', label: 'Model Registry', icon: Database },
    { id: 'retraining', label: 'Auto-Retraining', icon: Zap },
    { id: 'mlflow', label: 'MLflow', icon: BarChart3 },
    { id: 'statistics', label: 'Statistics', icon: Gauge },
    { id: 'layering', label: 'Layering Model', icon: Shield },
    { id: 'optimization', label: 'Optimization', icon: Target }
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

      {/* Tab Content with Sidebar for Training Tabs */}
      <div className={cn(
        'grid gap-6',
        ['training', 'mpd_training', 'tlob_training'].includes(activeTab)
          ? 'lg:grid-cols-[1fr,350px]'
          : 'grid-cols-1'
      )}>
        {/* Main Content */}
        <div>
          {activeTab === 'training' && renderTrainingTab()}
          {activeTab === 'mpd_training' && renderMPDTrainingTab()}
          {activeTab === 'tlob_training' && renderTLOBTrainingTab()}
          {activeTab === 'models' && renderModelsTab()}
          {activeTab === 'retraining' && renderRetrainingTab()}
          {activeTab === 'mlflow' && renderMLflowTab()}
          {activeTab === 'statistics' && renderStatisticsTab()}
          {activeTab === 'layering' && renderLayeringTab()}
          {activeTab === 'optimization' && renderOptimizationTab()}
        </div>

        {/* Real-Time Status Sidebar (only on training tabs) */}
        {['training', 'mpd_training', 'tlob_training'].includes(activeTab) && (
          <div className="hidden lg:block">
            <div className="sticky top-6">
              <EnsembleRealTimeStatus />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
