// frontend/src/store/ensembleStore.ts

import { create } from 'zustand';
import {
  TrainingProgress,
  EnsemblePrediction,
  StatusChange,
  HyperoptProgress,
} from '../services/ensemble-websocket.service';

/**
 * Состояние модели в ensemble.
 */
export interface ModelState {
  name: string;
  displayName: string;
  weight: number;
  enabled: boolean;
  performanceScore: number;
  isRegistered: boolean;
  description: string;
}

/**
 * Состояние задачи обучения.
 */
export interface TrainingTask {
  taskId: string;
  modelType: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  currentEpoch: number;
  totalEpochs: number;
  metrics: {
    trainLoss?: number;
    trainAcc?: number;
    valLoss?: number;
    valAcc?: number;
    bestValLoss?: number;
    bestAccuracy?: number;
  };
  startedAt?: string;
  completedAt?: string;
  error?: string;
}

/**
 * Состояние оптимизации гиперпараметров.
 */
export interface HyperoptState {
  jobId: string | null;
  status: 'idle' | 'running' | 'completed' | 'failed';
  mode: string | null;
  currentTrial: number;
  totalTrials: number;
  progressPct: number;
  currentGroup: string | null;
  bestValue: number | null;
  bestParams: Record<string, unknown> | null;
  elapsedTime: string | null;
  error: string | null;
}

/**
 * Store для управления состоянием Ensemble системы.
 */
interface EnsembleStore {
  // WebSocket состояние
  isConnected: boolean;
  setConnected: (connected: boolean) => void;

  // Модели
  models: ModelState[];
  setModels: (models: ModelState[]) => void;
  updateModelState: (modelType: string, updates: Partial<ModelState>) => void;

  // Стратегия
  currentStrategy: string;
  setStrategy: (strategy: string) => void;

  // Конфигурация
  config: {
    minConfidenceForTrade: number;
    unanimousThreshold: number;
    conflictResolution: string;
    enableAdaptiveWeights: boolean;
  };
  setConfig: (config: Partial<EnsembleStore['config']>) => void;

  // Обучение
  trainingTasks: Record<string, TrainingTask>;
  updateTrainingTask: (taskId: string, updates: Partial<TrainingTask>) => void;
  addTrainingTask: (task: TrainingTask) => void;
  removeTrainingTask: (taskId: string) => void;

  // Предсказания
  lastPrediction: EnsemblePrediction | null;
  predictionHistory: EnsemblePrediction[];
  addPrediction: (prediction: EnsemblePrediction) => void;
  clearPredictionHistory: () => void;

  // Hyperopt
  hyperopt: HyperoptState;
  updateHyperopt: (updates: Partial<HyperoptState>) => void;
  resetHyperopt: () => void;

  // Статистика
  stats: {
    totalPredictions: number;
    unanimousCount: number;
    majorityCount: number;
    conflictCount: number;
    tradesSignaled: number;
  };
  setStats: (stats: Partial<EnsembleStore['stats']>) => void;

  // Обработка WebSocket событий
  handleTrainingProgress: (data: TrainingProgress) => void;
  handlePrediction: (data: EnsemblePrediction) => void;
  handleStatusChange: (data: StatusChange) => void;
  handleHyperoptProgress: (data: HyperoptProgress) => void;
}

const initialHyperoptState: HyperoptState = {
  jobId: null,
  status: 'idle',
  mode: null,
  currentTrial: 0,
  totalTrials: 0,
  progressPct: 0,
  currentGroup: null,
  bestValue: null,
  bestParams: null,
  elapsedTime: null,
  error: null,
};

export const useEnsembleStore = create<EnsembleStore>((set, get) => ({
  // WebSocket состояние
  isConnected: false,
  setConnected: (connected) => set({ isConnected: connected }),

  // Модели
  models: [],
  setModels: (models) => set({ models }),
  updateModelState: (modelType, updates) =>
    set((state) => ({
      models: state.models.map((m) =>
        m.name === modelType ? { ...m, ...updates } : m
      ),
    })),

  // Стратегия
  currentStrategy: 'weighted_voting',
  setStrategy: (strategy) => set({ currentStrategy: strategy }),

  // Конфигурация
  config: {
    minConfidenceForTrade: 0.6,
    unanimousThreshold: 0.9,
    conflictResolution: 'hold',
    enableAdaptiveWeights: true,
  },
  setConfig: (config) =>
    set((state) => ({ config: { ...state.config, ...config } })),

  // Обучение
  trainingTasks: {},
  updateTrainingTask: (taskId, updates) =>
    set((state) => ({
      trainingTasks: {
        ...state.trainingTasks,
        [taskId]: { ...state.trainingTasks[taskId], ...updates },
      },
    })),
  addTrainingTask: (task) =>
    set((state) => ({
      trainingTasks: { ...state.trainingTasks, [task.taskId]: task },
    })),
  removeTrainingTask: (taskId) =>
    set((state) => {
      const { [taskId]: _, ...rest } = state.trainingTasks;
      return { trainingTasks: rest };
    }),

  // Предсказания
  lastPrediction: null,
  predictionHistory: [],
  addPrediction: (prediction) =>
    set((state) => ({
      lastPrediction: prediction,
      predictionHistory: [prediction, ...state.predictionHistory].slice(0, 100), // Keep last 100
    })),
  clearPredictionHistory: () =>
    set({ lastPrediction: null, predictionHistory: [] }),

  // Hyperopt
  hyperopt: initialHyperoptState,
  updateHyperopt: (updates) =>
    set((state) => ({ hyperopt: { ...state.hyperopt, ...updates } })),
  resetHyperopt: () => set({ hyperopt: initialHyperoptState }),

  // Статистика
  stats: {
    totalPredictions: 0,
    unanimousCount: 0,
    majorityCount: 0,
    conflictCount: 0,
    tradesSignaled: 0,
  },
  setStats: (stats) =>
    set((state) => ({ stats: { ...state.stats, ...stats } })),

  // Обработка WebSocket событий
  handleTrainingProgress: (data) => {
    const { task_id, model_type, epoch, total_epochs, metrics, status } = data;
    const progressPct = total_epochs > 0 ? (epoch / total_epochs) * 100 : 0;

    // Преобразование статусов бэкенда в фронтенд статусы
    // 'started', 'training', 'loading_data' -> 'running'
    const normalizedStatus =
      ['started', 'training', 'loading_data', 'running'].includes(status)
        ? 'running'
        : status;

    set((state) => ({
      trainingTasks: {
        ...state.trainingTasks,
        [task_id]: {
          taskId: task_id,
          modelType: model_type,
          status: normalizedStatus as TrainingTask['status'],
          progress: progressPct,
          currentEpoch: epoch,
          totalEpochs: total_epochs,
          metrics: {
            trainLoss: metrics.train_loss,
            trainAcc: metrics.train_acc,
            valLoss: metrics.val_loss,
            valAcc: metrics.val_acc,
            bestValLoss: metrics.best_val_loss,
            bestAccuracy: metrics.best_accuracy,
          },
          startedAt: state.trainingTasks[task_id]?.startedAt,
          completedAt: status === 'completed' || status === 'failed'
            ? new Date().toISOString()
            : undefined,
          error: status === 'failed' ? (metrics as any).error : undefined,
        },
      },
    }));
  },

  handlePrediction: (data) => {
    get().addPrediction(data);
  },

  handleStatusChange: (data) => {
    const { model_type, change_type, new_value } = data;

    if (change_type === 'enabled') {
      get().updateModelState(model_type, { enabled: new_value as boolean });
    } else if (change_type === 'weight') {
      get().updateModelState(model_type, { weight: new_value as number });
    } else if (change_type === 'performance') {
      get().updateModelState(model_type, { performanceScore: new_value as number });
    }
  },

  handleHyperoptProgress: (data) => {
    switch (data.type) {
      case 'hyperopt_started':
        set({
          hyperopt: {
            ...initialHyperoptState,
            jobId: data.job_id,
            status: 'running',
            mode: data.mode || null,
            totalTrials: data.total_trials || 0,
          },
        });
        break;

      case 'hyperopt_progress':
        set((state) => ({
          hyperopt: {
            ...state.hyperopt,
            currentTrial: data.current_trial || 0,
            totalTrials: data.total_trials || state.hyperopt.totalTrials,
            progressPct: data.progress_pct || 0,
            currentGroup: data.current_group || null,
            bestValue: data.best_value ?? state.hyperopt.bestValue,
            elapsedTime: data.elapsed_time || null,
          },
        }));
        break;

      case 'hyperopt_completed':
        set((state) => ({
          hyperopt: {
            ...state.hyperopt,
            status: 'completed',
            progressPct: 100,
            bestParams: data.best_params || null,
            bestValue: data.best_value ?? state.hyperopt.bestValue,
            elapsedTime: data.elapsed_time || null,
          },
        }));
        break;

      case 'hyperopt_failed':
        set((state) => ({
          hyperopt: {
            ...state.hyperopt,
            status: 'failed',
            error: data.error || 'Unknown error',
          },
        }));
        break;
    }
  },
}));

export default useEnsembleStore;
