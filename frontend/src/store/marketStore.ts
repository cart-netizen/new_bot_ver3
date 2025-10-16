// frontend/src/store/marketStore.ts

import { create } from 'zustand';
import type { OrderBook, OrderBookMetrics } from '../types/orderbook.types';

/**
 * Состояние рыночных данных.
 * Хранит данные по всем торговым парам в реальном времени.
 */
interface MarketState {
  // Данные стаканов
  orderbooks: Record<string, OrderBook>;

  // Метрики по парам (текущие)
  metrics: Record<string, OrderBookMetrics>;

  // Выбранная пара для детального просмотра
  selectedSymbol: string | null;

  // Статус подключения WebSocket
  isConnected: boolean;

  // Методы обновления данных
  updateOrderBook: (symbol: string, data: OrderBook) => void;
  updateMetrics: (symbol: string, data: OrderBookMetrics) => void;
  setSelectedSymbol: (symbol: string | null) => void;
  setConnected: (connected: boolean) => void;
  reset: () => void;
}

/**
 * Zustand store для рыночных данных.
 * Обновляется через WebSocket в реальном времени.
 */
export const useMarketStore = create<MarketState>((set) => ({
  orderbooks: {},
  metrics: {},
  selectedSymbol: null,
  isConnected: false,

  /**
   * Обновление данных стакана для конкретной пары.
   */
  updateOrderBook: (symbol, data) =>
    set((state) => ({
      orderbooks: {
        ...state.orderbooks,
        [symbol]: data,
      },
    })),

  /**
   * Обновление метрик для конкретной пары.
   */
  updateMetrics: (symbol, data) =>
    set((state) => ({
      metrics: {
        ...state.metrics,
        [symbol]: data,
      },
    })),

  /**
   * Установка выбранной пары для детального просмотра.
   */
  setSelectedSymbol: (symbol) => set({ selectedSymbol: symbol }),

  /**
   * Обновление статуса WebSocket подключения.
   */
  setConnected: (connected) => set({ isConnected: connected }),

  /**
   * Сброс всех данных к начальному состоянию.
   */
  reset: () =>
    set({
      orderbooks: {},
      metrics: {},
      selectedSymbol: null,
      isConnected: false,
    }),
}));