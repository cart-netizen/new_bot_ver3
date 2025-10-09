// frontend/src/services/websocket.service.ts

import { APP_CONFIG } from '../config/app.config';
import type { OrderBook, OrderBookMetrics } from '../types/orderbook.types';
import type { TradingSignal } from '../types/signal.types';

/**
 * Типы WebSocket сообщений от бэкенда.
 */
type WSMessageType =
  | 'bot_status'
  | 'orderbook_update'
  | 'metrics_update'
  | 'trading_signal'
  | 'execution_update'
  | 'error';

interface WSMessage {
  type: WSMessageType;
  symbol?: string;
  orderbook?: OrderBook;
  metrics?: OrderBookMetrics;
  signal?: TradingSignal;
  execution?: ExecutionData;
  message?: string;
  status?: string;
  details?: Record<string, unknown>;
}

/**
 * Данные статуса бота.
 */
interface BotStatusData {
  status: string;
  details?: Record<string, unknown>;
}

/**
 * Данные исполнения ордера.
 */
interface ExecutionData {
  order_id: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  status: string;
  timestamp: number;
}

/**
 * Обработчики для различных типов сообщений.
 */
interface MessageHandlers {
  onBotStatus?: (data: BotStatusData) => void;
  onOrderBookUpdate?: (symbol: string, orderbook: OrderBook) => void;
  onMetricsUpdate?: (symbol: string, metrics: OrderBookMetrics) => void;
  onTradingSignal?: (signal: TradingSignal) => void;
  onExecutionUpdate?: (data: ExecutionData) => void;
  onError?: (error: string) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

/**
 * Сервис для управления WebSocket соединением.
 * Обеспечивает надежное подключение с автоматическим переподключением.
 */
export class WebSocketService {
  private ws: WebSocket | null = null;
  private handlers: MessageHandlers = {};
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnecting = false;
  private shouldReconnect = true;
  private pingInterval: ReturnType<typeof setInterval> | null = null;

  /**
   * Подключение к WebSocket серверу.
   */
  connect(token: string, handlers: MessageHandlers): void {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      console.log('[WS] Already connected or connecting');
      return;
    }

    this.isConnecting = true;
    this.handlers = handlers;
    this.shouldReconnect = true;

    console.log('[WS] Connecting to:', APP_CONFIG.wsUrl);

    try {
      this.ws = new WebSocket(APP_CONFIG.wsUrl);

      this.ws.onopen = () => {
        console.log('[WS] Connected');
        this.isConnecting = false;
        this.reconnectAttempts = 0;

        // Отправляем токен для аутентификации
        this.send({ type: 'auth', token });

        // Запускаем ping для поддержания соединения
        this.startPing();

        handlers.onConnect?.();
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WSMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('[WS] Failed to parse message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('[WS] Error:', error);
        this.isConnecting = false;
      };

      this.ws.onclose = (event) => {
        console.log('[WS] Disconnected:', event.code, event.reason);
        this.isConnecting = false;
        this.stopPing();

        handlers.onDisconnect?.();

        // Автоматическое переподключение
        if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

          console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

          setTimeout(() => {
            this.connect(token, handlers);
          }, delay);
        }
      };
    } catch (error) {
      console.error('[WS] Connection failed:', error);
      this.isConnecting = false;
    }
  }

  /**
   * Обработка входящих сообщений.
   */
  private handleMessage(message: WSMessage): void {
    console.log('[WS] Message received:', message.type);

    switch (message.type) {
      case 'bot_status':
        if (message.status) {
          this.handlers.onBotStatus?.({
            status: message.status,
            details: message.details,
          });
        }
        break;

      case 'orderbook_update':
        if (message.symbol && message.orderbook) {
          this.handlers.onOrderBookUpdate?.(message.symbol, message.orderbook);
        }
        break;

      case 'metrics_update':
        if (message.symbol && message.metrics) {
          this.handlers.onMetricsUpdate?.(message.symbol, message.metrics);
        }
        break;

      case 'trading_signal':
        if (message.signal) {
          this.handlers.onTradingSignal?.(message.signal);
        }
        break;

      case 'execution_update':
        if (message.execution) {
          this.handlers.onExecutionUpdate?.(message.execution);
        }
        break;

      case 'error':
        if (message.message) {
          console.error('[WS] Server error:', message.message);
          this.handlers.onError?.(message.message);
        }
        break;

      default:
        console.warn('[WS] Unknown message type:', message.type);
    }
  }

  /**
   * Отправка сообщения на сервер.
   */
  private send(data: { type: string; token?: string }): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('[WS] Cannot send message: not connected');
    }
  }

  /**
   * Запуск периодического ping для поддержания соединения.
   */
  private startPing(): void {
    this.stopPing();
    this.pingInterval = setInterval(() => {
      this.send({ type: 'ping' });
    }, 30000); // Каждые 30 секунд
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
    console.log('[WS] Disconnecting...');
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
export const wsService = new WebSocketService();