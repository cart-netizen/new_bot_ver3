// frontend/src/components/ensemble/EnsembleRealTimeStatus.tsx

import { useEffect } from 'react';
import {
  Activity,
  Wifi,
  WifiOff,
  TrendingUp,
  TrendingDown,
  Minus,
  Brain,
  Zap,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
} from 'lucide-react';
import { useEnsembleWebSocket } from '../../hooks/useEnsembleWebSocket';
import { useEnsembleStore } from '../../store/ensembleStore';
import { cn } from '../../utils/helpers';

/**
 * Компонент отображения real-time статуса Ensemble.
 * Подключается к WebSocket и показывает:
 * - Статус подключения
 * - Прогресс обучения моделей
 * - Последние предсказания
 * - Прогресс Hyperopt
 */
export function EnsembleRealTimeStatus() {
  const {
    isConnected,
    setConnected,
    trainingTasks,
    lastPrediction,
    hyperopt,
    handleTrainingProgress,
    handlePrediction,
    handleStatusChange,
    handleHyperoptProgress,
  } = useEnsembleStore();

  // Подключаемся к WebSocket
  const { connect, disconnect } = useEnsembleWebSocket({
    subscriptions: ['training', 'predictions', 'hyperopt', 'status'],
    autoConnect: true,
    onConnect: () => setConnected(true),
    onDisconnect: () => setConnected(false),
    onTrainingProgress: handleTrainingProgress,
    onPrediction: handlePrediction,
    onStatusChange: handleStatusChange,
    onHyperoptProgress: handleHyperoptProgress,
  });

  // Активные задачи обучения
  const activeTrainingTasks = Object.values(trainingTasks).filter(
    (t) => t.status === 'running'
  );

  // Иконка направления предсказания
  const DirectionIcon = lastPrediction?.direction === 'BUY'
    ? TrendingUp
    : lastPrediction?.direction === 'SELL'
    ? TrendingDown
    : Minus;

  const directionColor = lastPrediction?.direction === 'BUY'
    ? 'text-green-500'
    : lastPrediction?.direction === 'SELL'
    ? 'text-red-500'
    : 'text-gray-500';

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-blue-400" />
          <h3 className="text-white font-medium">Ensemble Real-Time</h3>
        </div>
        <div className="flex items-center gap-2">
          {isConnected ? (
            <>
              <Wifi className="w-4 h-4 text-green-500" />
              <span className="text-xs text-green-500">Connected</span>
            </>
          ) : (
            <>
              <WifiOff className="w-4 h-4 text-red-500" />
              <span className="text-xs text-red-500">Disconnected</span>
            </>
          )}
        </div>
      </div>

      {/* Active Training Tasks */}
      {activeTrainingTasks.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm text-gray-400 flex items-center gap-1">
            <Brain className="w-4 h-4" />
            Training in Progress
          </h4>
          {activeTrainingTasks.map((task) => (
            <div
              key={task.taskId}
              className="bg-gray-700/50 rounded-lg p-3 space-y-2"
            >
              <div className="flex items-center justify-between">
                <span className="text-sm text-white font-medium">
                  {task.modelType.toUpperCase()}
                </span>
                <span className="text-xs text-gray-400">
                  Epoch {task.currentEpoch}/{task.totalEpochs}
                </span>
              </div>
              {/* Progress bar */}
              <div className="w-full bg-gray-600 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${task.progress}%` }}
                />
              </div>
              {/* Metrics */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Train Loss:</span>
                  <span className="text-white">
                    {task.metrics.trainLoss?.toFixed(4) || '-'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Val Loss:</span>
                  <span className="text-white">
                    {task.metrics.valLoss?.toFixed(4) || '-'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Train Acc:</span>
                  <span className="text-white">
                    {task.metrics.trainAcc
                      ? `${(task.metrics.trainAcc * 100).toFixed(1)}%`
                      : '-'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Val Acc:</span>
                  <span className="text-white">
                    {task.metrics.valAcc
                      ? `${(task.metrics.valAcc * 100).toFixed(1)}%`
                      : '-'}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Hyperopt Progress */}
      {hyperopt.status === 'running' && (
        <div className="space-y-2">
          <h4 className="text-sm text-gray-400 flex items-center gap-1">
            <Zap className="w-4 h-4" />
            Hyperopt Optimization
          </h4>
          <div className="bg-gray-700/50 rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-white font-medium">
                {hyperopt.mode?.toUpperCase() || 'FULL'} Mode
              </span>
              <span className="text-xs text-gray-400">
                Trial {hyperopt.currentTrial}/{hyperopt.totalTrials}
              </span>
            </div>
            {/* Progress bar */}
            <div className="w-full bg-gray-600 rounded-full h-2">
              <div
                className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${hyperopt.progressPct}%` }}
              />
            </div>
            {hyperopt.currentGroup && (
              <div className="text-xs text-gray-400">
                Current: {hyperopt.currentGroup}
              </div>
            )}
            {hyperopt.bestValue !== null && (
              <div className="text-xs">
                <span className="text-gray-400">Best Value: </span>
                <span className="text-green-400">
                  {hyperopt.bestValue.toFixed(4)}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Hyperopt Completed */}
      {hyperopt.status === 'completed' && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-green-500" />
            <span className="text-sm text-green-400">
              Hyperopt Completed
            </span>
          </div>
          {hyperopt.bestValue !== null && (
            <div className="text-xs text-gray-400 mt-1">
              Best Value: {hyperopt.bestValue.toFixed(4)}
            </div>
          )}
        </div>
      )}

      {/* Last Prediction */}
      {lastPrediction && (
        <div className="space-y-2">
          <h4 className="text-sm text-gray-400 flex items-center gap-1">
            <TrendingUp className="w-4 h-4" />
            Last Prediction
          </h4>
          <div className="bg-gray-700/50 rounded-lg p-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <DirectionIcon className={cn('w-5 h-5', directionColor)} />
                <span className={cn('text-lg font-bold', directionColor)}>
                  {lastPrediction.direction}
                </span>
              </div>
              <div className="text-right">
                <div className="text-sm text-white">
                  {(lastPrediction.confidence * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-400">Confidence</div>
              </div>
            </div>
            <div className="mt-2 flex items-center gap-2">
              {lastPrediction.should_trade ? (
                <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded">
                  Trade Signal
                </span>
              ) : (
                <span className="text-xs bg-gray-500/20 text-gray-400 px-2 py-1 rounded">
                  No Trade
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* No activity message */}
      {activeTrainingTasks.length === 0 &&
        hyperopt.status === 'idle' &&
        !lastPrediction && (
          <div className="text-center text-gray-500 py-4">
            <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No active tasks</p>
            <p className="text-xs">
              Start training or make predictions to see real-time updates
            </p>
          </div>
        )}
    </div>
  );
}

export default EnsembleRealTimeStatus;
