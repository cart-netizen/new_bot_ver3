// frontend/src/components/layout/Layout.tsx
// ИСПРАВЛЕННАЯ ВЕРСИЯ с правильной инициализацией WebSocket

import { useEffect, useCallback, useRef } from 'react';
import { Outlet } from 'react-router-dom';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { useAuthStore } from '../../store/authStore';
import { useMarketStore } from '../../store/marketStore';
import { useTradingStore } from '../../store/tradingStore';
import { wsService } from '../../services/websocket.service';
import { toast } from 'sonner';

/**
 * Главный Layout компонент.
 * Управляет WebSocket соединением на глобальном уровне.
 */
export function Layout() {
  const { token, isAuthenticated } = useAuthStore();
  const { updateOrderBook, updateMetrics, setConnected } = useMarketStore();
  const { addSignal } = useTradingStore();

  // Ref для отслеживания, был ли WebSocket уже подключен
  const wsConnectedRef = useRef(false);

  /**
   * Подключение к WebSocket с обработчиками событий.
   */
  const connectWebSocket = useCallback(() => {
    // КРИТИЧЕСКАЯ ПРОВЕРКА: Токен должен быть валидным JWT
    if (!token) {
      console.error('[Layout] No auth token available');
      return;
    }

    // Проверяем формат токена (JWT должен иметь 3 части, разделенные точкой)
    const tokenParts = token.split('.');
    if (tokenParts.length !== 3) {
      console.error('[Layout] Invalid token format:', token.substring(0, 20) + '...');
      console.error('[Layout] Token должен быть JWT с 3 частями');
      toast.error('Невалидный токен. Пожалуйста, войдите заново.');

      // Очищаем невалидный токен
      localStorage.removeItem('auth-token');
      return;
    }

    console.log('[Layout] Token valid, connecting to WebSocket...');
    console.log('[Layout] Token preview:', token.substring(0, 20) + '...' + token.substring(token.length - 20));

    wsService.connect(token, {
      // Успешное подключение
      onConnect: () => {
        console.log('[Layout] ✅ WebSocket connected successfully');
        setConnected(true);
        wsConnectedRef.current = true;
        toast.success('Подключение установлено');
      },

      // Отключение
      onDisconnect: () => {
        console.log('[Layout] ❌ WebSocket disconnected');
        setConnected(false);
        wsConnectedRef.current = false;
        toast.error('Соединение потеряно');
      },

      // Обновление статуса бота
      onBotStatus: (data) => {
        console.log('[Layout] Bot status update:', data.status);
      },

      // Обновление стакана
      onOrderBookUpdate: (symbol, orderbook) => {
        updateOrderBook(symbol, orderbook);
      },

      // Обновление метрик
      onMetricsUpdate: (symbol, metricsData) => {
        updateMetrics(symbol, metricsData);
      },

      // Новый торговый сигнал
      onTradingSignal: (signal) => {
        addSignal(signal);
        toast.info(`Сигнал: ${signal.signal_type} ${signal.symbol}`, {
          description: signal.reason,
        });
      },

      // Ошибка
      onError: (error) => {
        console.error('[Layout] WebSocket error:', error);

        // Если ошибка аутентификации - очищаем токен
        if (error.includes('токен') || error.includes('token') || error.includes('аутентификац')) {
          console.error('[Layout] Authentication error detected, clearing token');
          localStorage.removeItem('auth-token');
          toast.error('Ошибка аутентификации. Пожалуйста, войдите заново.');
        } else {
          toast.error(`Ошибка: ${error}`);
        }
      },
    });
  }, [token, setConnected, updateOrderBook, updateMetrics, addSignal]);

  /**
   * Эффект для инициализации WebSocket.
   * Подключается только если:
   * 1. Пользователь аутентифицирован
   * 2. Есть валидный токен
   * 3. WebSocket еще не подключен
   */
  useEffect(() => {
    console.log('[Layout] Effect triggered');
    console.log('[Layout] isAuthenticated:', isAuthenticated);
    console.log('[Layout] token exists:', !!token);
    console.log('[Layout] wsConnectedRef:', wsConnectedRef.current);

    // Проверяем предусловия
    if (!isAuthenticated) {
      console.warn('[Layout] User not authenticated, skipping WebSocket connection');
      return;
    }

    if (!token) {
      console.warn('[Layout] No token available, skipping WebSocket connection');
      return;
    }

    // Если уже подключен - не переподключаемся
    if (wsConnectedRef.current && wsService.isConnected()) {
      console.log('[Layout] WebSocket already connected, skipping');
      return;
    }

    // Подключаемся
    console.log('[Layout] Initializing WebSocket connection...');
    connectWebSocket();

    // Cleanup при размонтировании
    return () => {
      console.log('[Layout] Unmounting - disconnecting WebSocket');
      wsService.disconnect();
      wsConnectedRef.current = false;
    };
  }, [isAuthenticated, token, connectWebSocket]);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Header />
      <div className="flex-1 flex">
        <Sidebar />
        <main className="flex-1 p-6 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}