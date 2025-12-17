// frontend/src/services/ensemble-websocket.service.ts

import { APP_CONFIG } from '../config/app.config';

/**
 * Типы WebSocket сообщений от ensemble API.
 */
export type EnsembleWSMessageType =
  | 'training_progress'
  | 'prediction'
  | 'status_change'
  | 'strategy_changed'
  | 'config_updated'
  | 'performance_updated'
  | 'hyperopt_started'
  | 'hyperopt_progress'
  | 'hyperopt_completed'
  | 'hyperopt_failed'
  | 'subscription_confirmed'
  | 'unsubscription_confirmed'
  | 'current_status'
  | 'pong'
  | 'error';

/**
 * Типы событий для подписки.
 */
export type EnsembleEventType = 'training' | 'predictions' | 'status' | 'hyperopt' | 'all';

/**
 * Прогресс обучения модели.
 */
export interface TrainingProgress {
  type: 'training_progress';
  task_id: string;
  model_type: string;
  epoch: number;
  total_epochs: number;
  progress_pct: number;
  metrics: {
    train_loss?: number;
    train_acc?: number;
    val_loss?: number;
    val_acc?: number;
    best_val_loss?: number;
    best_accuracy?: number;
    [key: string]: number | undefined;
  };
  status: 'started' | 'training' | 'completed' | 'failed';
}

/**
 * Предсказание ensemble.
 */
export interface EnsemblePrediction {
  type: 'prediction';
  direction: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  model_predictions: Record<string, {
    direction: string;
    confidence: number;
    probabilities: number[];
  }>;
  should_trade: boolean;
}

/**
 * Изменение статуса модели.
 */
export interface StatusChange {
  type: 'status_change';
  model_type: string;
  change_type: 'enabled' | 'weight' | 'performance';
  old_value: unknown;
  new_value: unknown;
}

/**
 * Изменение стратегии.
 */
export interface StrategyChange {
  type: 'strategy_changed';
  old_strategy: string;
  new_strategy: string;
}

/**
 * Обновление конфигурации.
 */
export interface ConfigUpdate {
  type: 'config_updated';
  changed_fields: Record<string, { old: unknown; new: unknown }>;
  full_config: Record<string, unknown>;
}

/**
 * Обновление производительности.
 */
export interface PerformanceUpdate {
  type: 'performance_updated';
  model_type: string;
  was_correct: boolean;
  profit_loss: number;
  old_score: number;
  new_score: number;
}

/**
 * Прогресс оптимизации гиперпараметров.
 */
export interface HyperoptProgress {
  type: 'hyperopt_started' | 'hyperopt_progress' | 'hyperopt_completed' | 'hyperopt_failed';
  job_id: string;
  mode?: string;
  current_trial?: number;
  total_trials?: number;
  progress_pct?: number;
  current_group?: string;
  best_value?: number;
  current_value?: number;
  best_params?: Record<string, unknown>;
  elapsed_time?: string;
  time_estimate?: Record<string, unknown>;
  error?: string;
  error_type?: string;
}

/**
 * Общее сообщение WebSocket.
 */
export interface EnsembleWSMessage {
  event_type: EnsembleEventType;
  timestamp: string;
  type: EnsembleWSMessageType;
  [key: string]: unknown;
}

/**
 * Обработчики для различных типов сообщений.
 */
export interface EnsembleMessageHandlers {
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
 * Сервис для WebSocket соединения с Ensemble API.
 * Обеспечивает real-time обновления:
 * - Прогресс обучения моделей
 * - Предсказания ensemble
 * - Изменения статуса/весов моделей
 * - Прогресс оптимизации гиперпараметров
 */
export class EnsembleWebSocketService {
  private ws: WebSocket | null = null;
  private handlers: EnsembleMessageHandlers = {};
  private subscriptions: EnsembleEventType[] = ['all'];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnecting = false;
  private shouldReconnect = true;
  private pingInterval: ReturnType<typeof setInterval> | null = null;

  /**
   * Получить URL для WebSocket подключения к ensemble API.
   */
  private getWsUrl(): string {
    // Преобразуем HTTP URL в WebSocket URL
    const baseUrl = APP_CONFIG.apiUrl.replace(/^http/, 'ws');
    return `${baseUrl}/ensemble/ws`;
  }

  /**
   * Подключение к Ensemble WebSocket.
   */
  connect(handlers: EnsembleMessageHandlers, subscriptions: EnsembleEventType[] = ['all']): void {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      console.log('[Ensemble WS] Already connected or connecting');
      return;
    }

    this.isConnecting = true;
    this.handlers = handlers;
    this.subscriptions = subscriptions;
    this.shouldReconnect = true;

    const wsUrl = this.getWsUrl();
    console.log('[Ensemble WS] Connecting to:', wsUrl);

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('[Ensemble WS] Connected');
        this.isConnecting = false;
        this.reconnectAttempts = 0;

        // Подписываемся на события
        if (subscriptions.length > 0 && !subscriptions.includes('all')) {
          this.subscribe(subscriptions);
        }

        // Запускаем ping для поддержания соединения
        this.startPing();

        handlers.onConnect?.();
      };

      this.ws.onmessage = (event) => {
        try {
          const message: EnsembleWSMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('[Ensemble WS] Failed to parse message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('[Ensemble WS] Error:', error);
        this.isConnecting = false;
      };

      this.ws.onclose = (event) => {
        console.log('[Ensemble WS] Disconnected:', event.code, event.reason);
        this.isConnecting = false;
        this.stopPing();

        handlers.onDisconnect?.();

        // Автоматическое переподключение
        if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

          console.log(`[Ensemble WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

          setTimeout(() => {
            this.connect(handlers, subscriptions);
          }, delay);
        }
      };
    } catch (error) {
      console.error('[Ensemble WS] Connection failed:', error);
      this.isConnecting = false;
    }
  }

  /**
   * Обработка входящих сообщений.
   */
  private handleMessage(message: EnsembleWSMessage): void {
    console.log('[Ensemble WS] Message:', message.type);

    switch (message.type) {
      case 'training_progress':
        this.handlers.onTrainingProgress?.(message as unknown as TrainingProgress);
        break;

      case 'prediction':
        this.handlers.onPrediction?.(message as unknown as EnsemblePrediction);
        break;

      case 'status_change':
        this.handlers.onStatusChange?.(message as unknown as StatusChange);
        break;

      case 'strategy_changed':
        this.handlers.onStrategyChange?.(message as unknown as StrategyChange);
        break;

      case 'config_updated':
        this.handlers.onConfigUpdate?.(message as unknown as ConfigUpdate);
        break;

      case 'performance_updated':
        this.handlers.onPerformanceUpdate?.(message as unknown as PerformanceUpdate);
        break;

      case 'hyperopt_started':
      case 'hyperopt_progress':
      case 'hyperopt_completed':
      case 'hyperopt_failed':
        this.handlers.onHyperoptProgress?.(message as unknown as HyperoptProgress);
        break;

      case 'subscription_confirmed':
        console.log('[Ensemble WS] Subscription confirmed:', (message as any).events);
        break;

      case 'current_status':
        console.log('[Ensemble WS] Current status received');
        break;

      case 'pong':
        // Heartbeat response
        break;

      case 'error':
        console.error('[Ensemble WS] Server error:', (message as any).message);
        this.handlers.onError?.((message as any).message);
        break;

      default:
        console.warn('[Ensemble WS] Unknown message type:', message.type);
    }
  }

  /**
   * Отправка сообщения на сервер.
   */
  private send(data: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('[Ensemble WS] Cannot send message: not connected');
    }
  }

  /**
   * Подписка на события.
   */
  subscribe(events: EnsembleEventType[]): void {
    this.send({ action: 'subscribe', events });
  }

  /**
   * Отписка от событий.
   */
  unsubscribe(events: EnsembleEventType[]): void {
    this.send({ action: 'unsubscribe', events });
  }

  /**
   * Запрос текущего статуса.
   */
  requestStatus(): void {
    this.send({ action: 'get_status' });
  }

  /**
   * Запуск периодического ping.
   */
  private startPing(): void {
    this.stopPing();
    this.pingInterval = setInterval(() => {
      this.send({ action: 'ping' });
    }, 30000);
  }

  /**
   * Остановка ping.
   */
  private stopPing(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /**
   * Отключение от WebSocket.
   */
  disconnect(): void {
    console.log('[Ensemble WS] Disconnecting...');
    this.shouldReconnect = false;
    this.stopPing();

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  /**
   * Проверка состояния подключения.
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
export const ensembleWsService = new EnsembleWebSocketService();
