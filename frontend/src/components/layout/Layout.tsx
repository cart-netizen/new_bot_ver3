/**
 * Главный Layout компонент с WebSocket интеграцией.
 *
 * ИСПРАВЛЕНО:
 * - Правильная обработка WebSocket сообщений
 * - Парсинг JSON из строк
 * - Корректная типизация
 */

import { useEffect, useCallback, useRef } from 'react';
import { Outlet } from 'react-router-dom';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { useAuthStore } from '../../store/authStore';
import { useMarketStore } from '../../store/marketStore';
import { useTradingStore } from '../../store/tradingStore';
import { useScreenerStore } from '../../store/screenerStore';
import { wsService } from '../../services/websocket.service';
import { toast } from 'sonner';
import { MemoryMonitor } from '../dev/MemoryMonitor';

// Импортируем предоставленные интерфейсы (предполагаем, что они доступны; если нет, добавьте import)
export interface OrderBookMetrics {
  symbol: string;
  timestamp: number;
  datetime: string;

  // Ценовые метрики
  prices: {
    best_bid: number | null;
    best_ask: number | null;
    spread: number | null;
    mid_price: number | null;
  };

  // Объемные метрики
  volumes: {
    total_bid: number;
    total_ask: number;
    bid_depth_5: number;
    ask_depth_5: number;
    bid_depth_10: number;
    ask_depth_10: number;
  };

  // Дисбаланс спроса/предложения
  imbalance: {
    overall: number;      // 0.0 (все продажи) - 0.5 (баланс) - 1.0 (все покупки)
    depth_5: number;
    depth_10: number;
  };

  // VWAP метрики
  vwap: {
    bid: number | null;
    ask: number | null;
    mid: number | null;
  };

  // Кластеры объема
  clusters: {
    largest_bid: {
      price: number | null;
      volume: number;
    };
    largest_ask: {
      price: number | null;
      volume: number;
    };
  };
}

export interface OrderBook {
  symbol: string;
  bids: [number, number][];  // [price, quantity]
  asks: [number, number][];  // [price, quantity]
  timestamp: number;
  best_bid: number | null;
  best_ask: number | null;
  spread: number | null;
  mid_price: number | null;
  update_id?: number;
  sequence_id?: number;
}

export interface TradingSignal {
  symbol: string;
  signal_type: SignalType;
  strength: SignalStrength;
  timestamp: number;
  price: number;
  confidence: number;
  metrics: {
    imbalance: number | null;
  };
  reason: string;
  status: {
    is_valid: boolean;
  };
}

// Предполагаем, что SignalType и SignalStrength - это enums или строки; определите их здесь, если нужно
type SignalType = string; // Замените на реальный enum, напр. 'buy' | 'sell'
type SignalStrength = number | string; // Замените на реальный тип

/**
 * Базовый тип WebSocket сообщения.
 */
interface WebSocketMessage {
  type: string;
  timestamp?: string | number;
}

/**
 * Данные пары для скринера из WebSocket.
 */
interface ScreenerPairWebSocketData {
  symbol: string;
  lastPrice: number;
  volume24h: number;
  price24hPcnt: number;
  highPrice24h: number;
  lowPrice24h: number;
  prevPrice24h: number;
}

/**
 * Сообщение с полными данными скринера.
 */
interface ScreenerDataMessage extends WebSocketMessage {
  type: 'screener_data';
  pairs: ScreenerPairWebSocketData[];
  total: number;
  min_volume: number;
}

/**
 * Сообщение с обновлением одной пары скринера.
 */
interface ScreenerUpdateMessage extends WebSocketMessage {
  type: 'screener_update';
  data: ScreenerPairWebSocketData;
}

/**
 * Сообщение с тикером (только цена).
 */
interface TickerMessage extends WebSocketMessage {
  type: 'ticker';
  data: {
    symbol: string;
    lastPrice: number;
  };
}

/**
 * Сообщение с обновлением orderbook.
 */
interface OrderBookUpdateMessage extends WebSocketMessage {
  type: 'orderbook_update';
  data: Pick<OrderBook, 'symbol' | 'bids' | 'asks' | 'timestamp'>;
}

/**
 * Сообщение с обновлением метрик.
 */
interface MetricsUpdateMessage extends WebSocketMessage {
  type: 'metrics_update';
  data: OrderBookMetrics;
}

/**
 * Сообщение с торговым сигналом.
 */
interface SignalUpdateMessage extends WebSocketMessage {
  type: 'signal_update';
  data: Omit<TradingSignal, 'signal_type'> & { type: string };
}

/**
 * Сообщение с обновлением ордера.
 */
interface OrderUpdateMessage extends WebSocketMessage {
  type: 'order_update';
  data: {
    order_id: string;
    [key: string]: unknown;
  };
}

/**
 * Сообщение с обновлением позиции.
 */
interface PositionUpdateMessage extends WebSocketMessage {
  type: 'position_update';
  data: {
    symbol: string;
    [key: string]: unknown;
  };
}

/**
 * Сообщение об ошибке.
 */
interface ErrorMessage extends WebSocketMessage {
  type: 'error';
  message: string;
}

/**
 * Сообщение о подключении.
 */
interface ConnectedMessage extends WebSocketMessage {
  type: 'connected';
  message: string;
}

/**
 * Pong ответ.
 */
interface PongMessage extends WebSocketMessage {
  type: 'pong';
}

/**
 * Объединенный тип всех WebSocket сообщений.
 */
type WebSocketMessageType =
  | ScreenerDataMessage
  | ScreenerUpdateMessage
  | TickerMessage
  | OrderBookUpdateMessage
  | MetricsUpdateMessage
  | SignalUpdateMessage
  | OrderUpdateMessage
  | PositionUpdateMessage
  | ErrorMessage
  | ConnectedMessage
  | PongMessage;

/**
 * Главный Layout компонент.
 */
export function Layout() {
  const { token, isAuthenticated } = useAuthStore();
  const { updateOrderBook, updateMetrics, setConnected: setMarketConnected } = useMarketStore();
  const { addSignal } = useTradingStore();
  const {
    updatePairData,
    updatePairPrice,
    setConnected: setScreenerConnected
  } = useScreenerStore();

  const isConnectingRef = useRef(false);
  const connectionAttemptRef = useRef(0);

  /**
   * Подключение к WebSocket с обработчиками событий.
   */
  const connectWebSocket = useCallback(() => {
    if (isConnectingRef.current) {
      console.log('[Layout] Already connecting, skipping duplicate call');
      return;
    }

    if (!token) {
      console.error('[Layout] No auth token available');
      return;
    }

    const tokenParts = token.split('.');
    if (tokenParts.length !== 3) {
      console.error('[Layout] Invalid token format:', token.substring(0, 20) + '...');
      toast.error('Невалидный токен. Пожалуйста, войдите заново.');
      return;
    }

    isConnectingRef.current = true;
    connectionAttemptRef.current += 1;
    const currentAttempt = connectionAttemptRef.current;

    console.log(`[Layout] Connecting to WebSocket (attempt ${currentAttempt})...`);

    wsService.connect(token, {
      // ==================== ОБРАБОТЧИКИ СОБЫТИЙ ====================

      onConnect: () => {
        console.log('[Layout] WebSocket connected successfully');
        setMarketConnected(true);
        setScreenerConnected(true);
        isConnectingRef.current = false;
        toast.success('Подключено к серверу');
      },

      onDisconnect: () => {
        console.log('[Layout] WebSocket disconnected');
        setMarketConnected(false);
        setScreenerConnected(false);
        isConnectingRef.current = false;
      },

      onError: (error: Event) => {
        console.error('[Layout] WebSocket error:', error);
        isConnectingRef.current = false;
        toast.error('Ошибка подключения к серверу');
      },

      // ==================== ОБРАБОТКА СООБЩЕНИЙ ====================
      /**
       * Главный обработчик всех WebSocket сообщений.
       * WebSocket отправляет данные как строку (JSON), которую нужно парсить.
       */
      onMessage: (event: MessageEvent) => {
  try {
    // Парсим JSON из строки
    const message = JSON.parse(event.data);

    console.log('[Layout] WebSocket message type:', message.type);

    // Обработка по типу сообщения
    switch (message.type) {
      // ==================== Screener Full Data (NEW) ====================
      case 'screener_data': {
        const pairs = message.pairs || [];

        console.log(`[Layout] Received screener_data: ${pairs.length} pairs`);

        // Обновляем все пары одним пакетом
        pairs.forEach((pair: ScreenerPairWebSocketData) => {
          updatePairData(pair.symbol, {
            lastPrice: pair.lastPrice,
            volume24h: pair.volume24h,
            price24hPcnt: pair.price24hPcnt,
            highPrice24h: pair.highPrice24h,
            lowPrice24h: pair.lowPrice24h,
            prevPrice24h: pair.prevPrice24h,
          });
        });

        // Устанавливаем статус подключения
        setScreenerConnected(true);

        // Логируем только первые 5 раз
        if (pairs.length > 0) {
          const firstPair = pairs[0];
          console.log('[Layout] Sample pair:', firstPair.symbol, '@', firstPair.lastPrice);
        }

        break;
      }

      // ==================== OrderBook Updates ====================
      case 'orderbook_update': {
        const data = message.data;
        if (data?.symbol && data?.bids && data?.asks) {
          updateOrderBook(data.symbol, data);
        }
        break;
      }

      // ==================== Metrics Updates ====================
      case 'metrics_update': {
        const data = message.data;
        if (data?.symbol && data?.timestamp) {
          updateMetrics(data.symbol, data);
        }
        break;
      }

      // ==================== Signal Updates ====================
      case 'signal_update': {
        const data = message.data;
        if (data?.symbol && data?.type) {
          addSignal(data);
        }
        break;
      }

      // ==================== Screener Single Update (Legacy) ====================
      case 'screener_update': {
        const data = message.data;
        if (data?.symbol && data?.volume24h !== undefined) {
          updatePairData(data.symbol, {
            lastPrice: data.lastPrice,
            volume24h: data.volume24h,
            price24hPcnt: data.price24hPcnt,
            highPrice24h: data.highPrice24h,
            lowPrice24h: data.lowPrice24h,
            prevPrice24h: data.prevPrice24h,
          });
        }
        break;
      }

      // ==================== Ticker Updates (Price only) ====================
      case 'ticker': {
        const data = message.data;
        if (data?.symbol && data?.lastPrice !== undefined) {
          updatePairPrice(data.symbol, data.lastPrice);
        }
        break;
      }

      // ==================== Order Updates ====================
      case 'order_update': {
        const data = message.data;
        console.log('[Layout] Order update:', data?.order_id);
        // Обработка обновлений ордеров (если нужно)
        break;
      }

      // ==================== Position Updates ====================
      case 'position_update': {
        const data = message.data;
        console.log('[Layout] Position update:', data?.symbol);
        // Обработка обновлений позиций (если нужно)
        break;
      }

      // ==================== Connected Confirmation ====================
      case 'connected': {
        console.log('[Layout] WebSocket connected confirmation:', message.message);
        toast.success('Подключено к серверу');
        break;
      }

      // ==================== Pong Response ====================
      case 'pong': {
        console.log('[Layout] Pong received');
        break;
      }

      // ==================== Error Messages ====================
      case 'error': {
        console.error('[Layout] Server error:', message.message);
        toast.error(`Ошибка сервера: ${message.message}`);
        break;
      }

      // ==================== Unknown Message Type ====================
      default: {
        console.warn('[Layout] Unknown message type:', message.type);
        break;
      }
    }

  } catch (error) {
    console.error('[Layout] Error parsing WebSocket message:', error);
  }
}
    });
  }, [
    token,
    updateOrderBook,
    updateMetrics,
    addSignal,
    updatePairData,
    updatePairPrice,
    setMarketConnected,
    setScreenerConnected
  ]);

  /**
   * Эффект для управления WebSocket соединением.
   */
  useEffect(() => {
    if (!isAuthenticated || !token) {
      console.log('[Layout] Not authenticated, skipping WebSocket connection');
      return;
    }

    if (wsService.isConnected()) {
      console.log('[Layout] WebSocket already connected');
      return;
    }

    connectWebSocket();

    return () => {
      if (wsService.isConnected()) {
        console.log('[Layout] Disconnecting WebSocket on unmount');
        wsService.disconnect();
      }
    };
  }, [isAuthenticated, token, connectWebSocket]);

  return (
    <div className="flex min-h-screen bg-background">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <Header />

        {/* Page Content */}
        <main className="flex-1 overflow-auto p-6">
          <Outlet />
        </main>
      </div>

      {/* Memory Monitor (dev mode только) */}
      {import.meta.env.DEV && <MemoryMonitor />}
    </div>
  );
}