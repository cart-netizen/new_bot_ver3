// frontend/src/components/layout/Layout.tsx
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

      onError: (error: any) => {
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

          console.log('[Layout] WebSocket message:', message.type);

          // Обработка по типу сообщения
          switch (message.type) {
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

            // ==================== Screener Updates ====================
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
              console.log('[Layout] Order update received:', data);
              // Здесь можно добавить обработку через ordersStore
              // Например: updateOrder(data.order_id, data);
              break;
            }

            // ==================== Connected ====================
            case 'connected': {
              console.log('[Layout] WebSocket handshake completed');
              break;
            }

            // ==================== Pong ====================
            case 'pong': {
              // Ответ на ping - ничего не делаем
              break;
            }

            default: {
              console.warn('[Layout] Unknown message type:', message.type);
            }
          }
        } catch (error) {
          console.error('[Layout] Error parsing WebSocket message:', error);
        }
      },
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