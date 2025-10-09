import { useEffect, useRef, useState, useCallback } from 'react';
import { APP_CONFIG } from '../config/app.config.ts';
import { useAuthStore } from '../store/authStore.ts';
import { useMarketStore } from '../store/marketStore.ts';
import { useTradingStore } from '../store/tradingStore.ts';
import { useBotStore } from '../store/botStore.ts';
import { toast } from 'sonner';

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const token = useAuthStore((state) => state.token);
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  const updateOrderBook = useMarketStore((state) => state.updateOrderBook);
  const updateMetrics = useMarketStore((state) => state.updateMetrics);
  const addSignal = useTradingStore((state) => state.addSignal);
  const updateStatus = useBotStore((state) => state.updateStatus);

  const connect = useCallback(() => {
    if (!isAuthenticated || !token) return;

    const ws = new WebSocket(APP_CONFIG.wsUrl);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      ws.send(JSON.stringify({ type: 'authenticate', token }));
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        switch (message.type) {
          case 'authenticated':
            toast.success('WebSocket подключен');
            break;
          case 'orderbook_update':
            if (message.data?.symbol) {
              updateOrderBook(message.data.symbol, message.data);
            }
            break;
          case 'metrics_update':
            if (message.data?.symbol) {
              updateMetrics(message.data.symbol, message.data);
            }
            break;
          case 'signal':
            if (message.data) {
              addSignal(message.data);
              toast.info(`Сигнал: ${message.data.signal_type} ${message.data.symbol}`);
            }
            break;
          case 'bot_status':
            if (message.data?.status) {
              updateStatus(message.data.status);
            }
            break;
        }
      } catch (error) {
        console.error('WebSocket message error:', error);
      }
    };

    ws.onerror = () => {
      console.error('WebSocket error');
      setIsConnected(false);
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      setIsConnected(false);
      wsRef.current = null;
    };

    wsRef.current = ws;
  }, [isAuthenticated, token, updateOrderBook, updateMetrics, addSignal, updateStatus]);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { isConnected };
}