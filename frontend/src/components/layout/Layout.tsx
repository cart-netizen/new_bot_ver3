// frontend/src/components/layout/Layout.tsx

import { useEffect, useCallback } from 'react';
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
 * WebSocket подключается один раз при монтировании Layout и остается активным
 * при переключении между страницами Dashboard/Market/Trading.
 */
export function Layout() {
  const { token } = useAuthStore();
  const { updateOrderBook, updateMetrics, setConnected } = useMarketStore();
  const { addSignal } = useTradingStore();

  /**
   * Подключение к WebSocket с обработчиками событий.
   * Выполняется один раз при монтировании Layout.
   */
  const connectWebSocket = useCallback(() => {
    if (!token) {
      console.error('[Layout] No auth token available');
      return;
    }

    console.log('[Layout] Connecting to WebSocket...');

    wsService.connect(token, {
      // Успешное подключение
      onConnect: () => {
        console.log('[Layout] WebSocket connected');
        setConnected(true);
        toast.success('Подключение установлено');
      },

      // Отключение
      onDisconnect: () => {
        console.log('[Layout] WebSocket disconnected');
        setConnected(false);
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
        toast.error(`Ошибка: ${error}`);
      },
    });
  }, [token, setConnected, updateOrderBook, updateMetrics, addSignal]);

  /**
   * Инициализация WebSocket при монтировании Layout.
   * Отключение при размонтировании (выход из приложения).
   */
  useEffect(() => {
    console.log('[Layout] Mounting - initializing WebSocket');
    connectWebSocket();

    // Cleanup: отключаем WebSocket только при размонтировании Layout
    // (т.е. при выходе пользователя из приложения)
    return () => {
      console.log('[Layout] Unmounting - disconnecting WebSocket');
      wsService.disconnect();
    };
  }, [connectWebSocket]);

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