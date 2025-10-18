// frontend/src/components/layout/Layout.tsx
// ИСПРАВЛЕННАЯ ВЕРСИЯ - правильная работа с React Strict Mode

import { useEffect, useCallback, useRef } from 'react';
import { Outlet } from 'react-router-dom';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { useAuthStore } from '../../store/authStore';
import { useMarketStore } from '../../store/marketStore';
import { useTradingStore } from '../../store/tradingStore';
import { wsService } from '../../services/websocket.service';
import { toast } from 'sonner';
import {useScreenerStore} from "@/store/screenerStore";

/**
 * Главный Layout компонент.
 * Управляет WebSocket соединением на глобальном уровне.
 *
 * ВАЖНО: Правильно обрабатывает React Strict Mode,
 * который в dev режиме монтирует/размонтирует компоненты дважды.
 */
const lastNotificationTime = new Map<string, number>();
const NOTIFICATION_COOLDOWN = 3 * 60 * 1000; // 3 минуты в миллисекундах

// Функция проверки throttling
const shouldShowNotification = (symbol: string): boolean => {
  const now = Date.now();
  const lastTime = lastNotificationTime.get(symbol) || 0;

  if (now - lastTime < NOTIFICATION_COOLDOWN) {
    return false; // Пропускаем
  }

  lastNotificationTime.set(symbol, now);
  return true; // Показываем
};

export function Layout() {
  const { token, isAuthenticated } = useAuthStore();
  const { updateOrderBook, updateMetrics, setConnected } = useMarketStore();
  const { addSignal } = useTradingStore();
  const { updatePairs } = useScreenerStore();
  // КРИТИЧЕСКИ ВАЖНО: Используем ref для предотвращения двойного подключения
  const isConnectingRef = useRef(false);
  const connectionAttemptRef = useRef(0);

  /**
   * Подключение к WebSocket с обработчиками событий.
   */
  const connectWebSocket = useCallback(() => {
    // Проверка: уже подключаемся
    if (isConnectingRef.current) {
      console.log('[Layout] Already connecting, skipping duplicate call');
      return;
    }

    // Проверка: токен должен быть валидным JWT
    if (!token) {
      console.error('[Layout] No auth token available');
      return;
    }

    // Проверяем формат токена (JWT должен иметь 3 части, разделенные точкой)
    const tokenParts = token.split('.');
    if (tokenParts.length !== 3) {
      console.error('[Layout] Invalid token format:', token.substring(0, 20) + '...');
      toast.error('Невалидный токен. Пожалуйста, войдите заново.');
      localStorage.removeItem('auth-token');
      return;
    }

    // Проверка: WebSocket уже подключен
    if (wsService.isConnected()) {
      console.log('[Layout] WebSocket already connected');
      setConnected(true);
      return;
    }

    // Устанавливаем флаг подключения
    isConnectingRef.current = true;
    connectionAttemptRef.current++;

    const attemptId = connectionAttemptRef.current;
    console.log(`[Layout] 🔌 Connection attempt #${attemptId}`);
    console.log('[Layout] Token valid, connecting to WebSocket...');
    console.log('[Layout] Token preview:', token.substring(0, 20) + '...' + token.substring(token.length - 20));

    wsService.connect(token, {
      // Успешное подключение
      onConnect: () => {
        console.log(`[Layout] ✅ WebSocket connected (attempt #${attemptId})`);
        setConnected(true);
        isConnectingRef.current = false;
        toast.success('Подключение установлено');
      },

      // Отключение
      onDisconnect: () => {
        console.log(`[Layout] ❌ WebSocket disconnected (attempt #${attemptId})`);
        setConnected(false);
        isConnectingRef.current = false;
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
      // НОВЫЙ ОБРАБОТЧИК для screener_update
      onScreenerUpdate: (data) => {
        console.log('[Layout] Screener update, pairs:', data.pairs?.length);
        updatePairs(data.pairs);
      },

      // Новый торговый сигнал
      onTradingSignal: (signal) => {
        addSignal(signal); // Всегда добавляем сигнал в стор

        // Показываем уведомление только если прошло 3 минуты
        if (shouldShowNotification(signal.symbol)) {
          toast.info(`Сигнал: ${signal.signal_type} ${signal.symbol}`, {
            description: signal.reason,
          });
        }
      },

      // Ошибка
      onError: (error) => {
        console.error('[Layout] WebSocket error:', error);
        isConnectingRef.current = false;

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
  }, [token, setConnected, updateOrderBook, updateMetrics, addSignal, updatePairs]);

  /**
   * Эффект для инициализации WebSocket.
   *
   * ВАЖНО: Правильно обрабатывает React Strict Mode:
   * - В dev режиме React монтирует компонент → вызывает cleanup → монтирует снова
   * - Мы используем isConnectingRef для предотвращения двойного подключения
   * - Cleanup отключает WebSocket только при РЕАЛЬНОМ размонтировании
   */
  useEffect(() => {
    console.log('[Layout] Effect triggered');
    console.log('[Layout] isAuthenticated:', isAuthenticated);
    console.log('[Layout] token exists:', !!token);
    console.log('[Layout] isConnecting:', isConnectingRef.current);
    console.log('[Layout] wsService.isConnected():', wsService.isConnected());

    // Проверяем предусловия
    if (!isAuthenticated) {
      console.warn('[Layout] User not authenticated, skipping WebSocket connection');
      return;
    }

    if (!token) {
      console.warn('[Layout] No token available, skipping WebSocket connection');
      return;
    }

    // КРИТИЧЕСКИ ВАЖНО: Добавляем небольшую задержку для React Strict Mode
    // Это позволяет предыдущему cleanup завершиться перед новым подключением
    const timeoutId = setTimeout(() => {
      console.log('[Layout] Delayed initialization after React Strict Mode cleanup');
      connectWebSocket();
    }, 100); // 100ms задержка

    // Cleanup при размонтировании
    return () => {
      console.log('[Layout] Cleanup triggered');

      // Отменяем отложенное подключение, если компонент размонтируется до его выполнения
      clearTimeout(timeoutId);

      // ВАЖНО: Отключаем WebSocket только если это НЕ React Strict Mode remount
      // React Strict Mode вызывает cleanup → effect снова очень быстро (< 50ms)
      // Реальное размонтирование происходит при выходе пользователя

      // Даем WebSocket время подключиться перед отключением
      const disconnectTimeoutId = setTimeout(() => {
        // Проверяем, что компонент действительно размонтирован (прошло время)
        // и это не просто React Strict Mode remount
        if (wsService.isConnected()) {
          console.log('[Layout] Cleanup: Disconnecting WebSocket (real unmount)');
          wsService.disconnect();
          setConnected(false);
          isConnectingRef.current = false;
        } else {
          console.log('[Layout] Cleanup: WebSocket not connected, skipping disconnect');
        }
      }, 200); // Даем 200ms на случай React Strict Mode

      return () => {
        clearTimeout(disconnectTimeoutId);
      };
    };
  }, [isAuthenticated, token, connectWebSocket, setConnected]);

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