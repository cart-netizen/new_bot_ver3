// frontend/src/hooks/useEnsembleWebSocket.ts

import { useEffect, useCallback, useRef } from 'react';
import {
  ensembleWsService,
  EnsembleEventType,
  TrainingProgress,
  EnsemblePrediction,
  StatusChange,
  StrategyChange,
  ConfigUpdate,
  PerformanceUpdate,
  HyperoptProgress,
} from '../services/ensemble-websocket.service';

/**
 * Опции для хука useEnsembleWebSocket.
 */
export interface UseEnsembleWebSocketOptions {
  /** Подписки на типы событий */
  subscriptions?: EnsembleEventType[];
  /** Авто-подключение при монтировании */
  autoConnect?: boolean;
  /** Callbacks для различных событий */
  onTrainingProgress?: (data: TrainingProgress) => void;
  onPrediction?: (data: EnsemblePrediction) => void;
  onStatusChange?: (data: StatusChange) => void;
  onStrategyChange?: (data: StrategyChange) => void;
  onConfigUpdate?: (data: ConfigUpdate) => void;
  onPerformanceUpdate?: (data: PerformanceUpdate) => void;
  onHyperoptProgress?: (data: HyperoptProgress) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: string) => void;
}

/**
 * Хук для подключения к Ensemble WebSocket.
 *
 * @example
 * ```tsx
 * const { isConnected, connect, disconnect } = useEnsembleWebSocket({
 *   subscriptions: ['training', 'predictions'],
 *   onTrainingProgress: (data) => {
 *     console.log('Training progress:', data);
 *   },
 *   onPrediction: (data) => {
 *     console.log('New prediction:', data);
 *   },
 * });
 * ```
 */
export function useEnsembleWebSocket(options: UseEnsembleWebSocketOptions = {}) {
  const {
    subscriptions = ['all'],
    autoConnect = true,
    onTrainingProgress,
    onPrediction,
    onStatusChange,
    onStrategyChange,
    onConfigUpdate,
    onPerformanceUpdate,
    onHyperoptProgress,
    onConnect,
    onDisconnect,
    onError,
  } = options;

  const isConnectedRef = useRef(false);
  const handlersRef = useRef({
    onTrainingProgress,
    onPrediction,
    onStatusChange,
    onStrategyChange,
    onConfigUpdate,
    onPerformanceUpdate,
    onHyperoptProgress,
    onConnect,
    onDisconnect,
    onError,
  });

  // Обновляем handlers при изменении
  useEffect(() => {
    handlersRef.current = {
      onTrainingProgress,
      onPrediction,
      onStatusChange,
      onStrategyChange,
      onConfigUpdate,
      onPerformanceUpdate,
      onHyperoptProgress,
      onConnect,
      onDisconnect,
      onError,
    };
  }, [
    onTrainingProgress,
    onPrediction,
    onStatusChange,
    onStrategyChange,
    onConfigUpdate,
    onPerformanceUpdate,
    onHyperoptProgress,
    onConnect,
    onDisconnect,
    onError,
  ]);

  const connect = useCallback(() => {
    ensembleWsService.connect(
      {
        onTrainingProgress: (data) => handlersRef.current.onTrainingProgress?.(data),
        onPrediction: (data) => handlersRef.current.onPrediction?.(data),
        onStatusChange: (data) => handlersRef.current.onStatusChange?.(data),
        onStrategyChange: (data) => handlersRef.current.onStrategyChange?.(data),
        onConfigUpdate: (data) => handlersRef.current.onConfigUpdate?.(data),
        onPerformanceUpdate: (data) => handlersRef.current.onPerformanceUpdate?.(data),
        onHyperoptProgress: (data) => handlersRef.current.onHyperoptProgress?.(data),
        onConnect: () => {
          isConnectedRef.current = true;
          handlersRef.current.onConnect?.();
        },
        onDisconnect: () => {
          isConnectedRef.current = false;
          handlersRef.current.onDisconnect?.();
        },
        onError: (error) => handlersRef.current.onError?.(error),
      },
      subscriptions
    );
  }, [subscriptions]);

  const disconnect = useCallback(() => {
    ensembleWsService.disconnect();
    isConnectedRef.current = false;
  }, []);

  const subscribe = useCallback((events: EnsembleEventType[]) => {
    ensembleWsService.subscribe(events);
  }, []);

  const unsubscribe = useCallback((events: EnsembleEventType[]) => {
    ensembleWsService.unsubscribe(events);
  }, []);

  const requestStatus = useCallback(() => {
    ensembleWsService.requestStatus();
  }, []);

  const isConnected = useCallback(() => {
    return ensembleWsService.isConnected();
  }, []);

  // Авто-подключение при монтировании
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      // Отключаемся при размонтировании только если авто-подключение было включено
      if (autoConnect) {
        disconnect();
      }
    };
  }, [autoConnect, connect, disconnect]);

  return {
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    requestStatus,
    isConnected,
  };
}

export default useEnsembleWebSocket;
