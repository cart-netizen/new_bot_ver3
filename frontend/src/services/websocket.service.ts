// frontend/src/services/websocket.service.ts
/**
 * WebSocket сервис для real-time коммуникации с сервером.
 *
 * ИСПРАВЛЕНО:
 * - Правильная типизация MessageHandlers
 * - Убраны старые обработчики (onOrderBookUpdate, onMetricsUpdate, onSignalUpdate)
 * - Используется только onMessage для всех типов сообщений
 */

/**
 * Обработчики WebSocket событий.
 */
interface MessageHandlers {
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  onMessage?: (event: MessageEvent) => void; // Главный обработчик всех сообщений
}

/**
 * Конфигурация переподключения.
 */
interface ReconnectConfig {
  maxAttempts: number;
  delay: number;
  backoffMultiplier: number;
}

/**
 * WebSocket Service для управления соединением.
 */
class WebSocketService {
  private ws: WebSocket | null = null;
  private handlers: MessageHandlers = {};
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private reconnectBackoff = 1.5;
  private reconnectTimeoutId: number | null = null;
  private token: string | null = null;
  private isManualDisconnect = false;

  /**
   * Проверка состояния подключения.
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Подключение к WebSocket серверу.
   */
  connect(token: string, handlers: MessageHandlers, config?: Partial<ReconnectConfig>) {
    // Сохраняем токен и обработчики
    this.token = token;
    this.handlers = handlers;

    // Применяем конфигурацию переподключения
    if (config) {
      if (config.maxAttempts) this.maxReconnectAttempts = config.maxAttempts;
      if (config.delay) this.reconnectDelay = config.delay;
      if (config.backoffMultiplier) this.reconnectBackoff = config.backoffMultiplier;
    }

    this.isManualDisconnect = false;
    this._connect();
  }

  /**
   * Внутренний метод подключения.
   */
  private _connect() {
    try {
      // Определяем URL WebSocket
      const wsUrl = this._getWebSocketUrl();

      console.log('[WebSocket] Connecting to:', wsUrl);

      // Создаем новое соединение
      this.ws = new WebSocket(wsUrl);

      // Обработчик успешного подключения
      this.ws.onopen = () => {
        console.log('[WebSocket] Connected successfully');
        this.reconnectAttempts = 0;

        // Аутентификация через WebSocket (если требуется)
        if (this.token) {
          this.send({
            type: 'auth',
            token: this.token,
          });
        }

        // Вызываем обработчик подключения
        if (this.handlers.onConnect) {
          this.handlers.onConnect();
        }
      };

      // Обработчик входящих сообщений
      this.ws.onmessage = (event: MessageEvent) => {
        // Передаем событие в главный обработчик
        if (this.handlers.onMessage) {
          this.handlers.onMessage(event);
        }
      };

      // Обработчик ошибок
      this.ws.onerror = (error: Event) => {
        console.error('[WebSocket] Error:', error);

        if (this.handlers.onError) {
          this.handlers.onError(error);
        }
      };

      // Обработчик закрытия соединения
      this.ws.onclose = (event: CloseEvent) => {
        console.log('[WebSocket] Connection closed:', event.code, event.reason);

        this.ws = null;

        if (this.handlers.onDisconnect) {
          this.handlers.onDisconnect();
        }

        // Автоматическое переподключение (если не ручное отключение)
        if (!this.isManualDisconnect) {
          this._scheduleReconnect();
        }
      };
    } catch (error) {
      console.error('[WebSocket] Connection error:', error);

      if (this.handlers.onError) {
        this.handlers.onError(error as Event);
      }

      // Пытаемся переподключиться
      if (!this.isManualDisconnect) {
        this._scheduleReconnect();
      }
    }
  }

  /**
   * Планирование переподключения.
   */
  private _scheduleReconnect() {
    // Проверяем лимит попыток
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WebSocket] Max reconnect attempts reached');
      return;
    }

    // Увеличиваем счетчик попыток
    this.reconnectAttempts++;

    // Рассчитываем задержку с экспоненциальным backoff
    const delay = this.reconnectDelay * Math.pow(this.reconnectBackoff, this.reconnectAttempts - 1);

    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    // Планируем переподключение
    this.reconnectTimeoutId = window.setTimeout(() => {
      this._connect();
    }, delay);
  }

  /**
   * Отправка сообщения на сервер.
   */
  send(data: any) {
    if (!this.isConnected()) {
      console.warn('[WebSocket] Cannot send message - not connected');
      return false;
    }

    try {
      const message = typeof data === 'string' ? data : JSON.stringify(data);
      this.ws!.send(message);
      return true;
    } catch (error) {
      console.error('[WebSocket] Error sending message:', error);
      return false;
    }
  }

  /**
   * Отключение от сервера.
   */
  disconnect() {
    console.log('[WebSocket] Disconnecting...');

    this.isManualDisconnect = true;

    // Отменяем запланированное переподключение
    if (this.reconnectTimeoutId !== null) {
      clearTimeout(this.reconnectTimeoutId);
      this.reconnectTimeoutId = null;
    }

    // Закрываем соединение
    if (this.ws) {
      this.ws.close(1000, 'Manual disconnect');
      this.ws = null;
    }

    this.reconnectAttempts = 0;
  }

  /**
   * Получение URL WebSocket сервера.
   */
  private _getWebSocketUrl(): string {
    // Определяем протокол (ws или wss)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

    // Определяем хост
    const host = import.meta.env.VITE_WS_URL ||
                 import.meta.env.VITE_API_URL?.replace(/^https?:\/\//, '') ||
                 'localhost:8000';

    // Формируем полный URL
    return `${protocol}//${host}/ws`;
  }

  /**
   * Отправка ping для поддержания соединения.
   */
  ping() {
    return this.send({
      type: 'ping',
      timestamp: Date.now(),
    });
  }
}

// Экспортируем единственный экземпляр сервиса
export const wsService = new WebSocketService();