// frontend/src/store/marketStore.ts

import { create } from 'zustand';
import type { OrderBook, OrderBookMetrics } from '../types/orderbook.types';

/**
 * Максимальное количество точек истории для каждой пары.
 * Храним последние N точек для построения графика.
 */
const MAX_HISTORY_POINTS = 100;

/**
 * Состояние рыночных данных.
 * Хранит данные по всем торговым парам в реальном времени.
 */
interface MarketState {
  // Данные стаканов
  orderbooks: Record<string, OrderBook>;

  // Метрики по парам (текущие)
  metrics: Record<string, OrderBookMetrics>;

  // История метрик для графиков
  metricsHistory: Record<string, OrderBookMetrics[]>;

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
  metricsHistory: {},
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
   * Также добавляет точку в историю для графиков.
   */
  updateMetrics: (symbol, data) =>
    set((state) => {
      // Получаем текущую историю для символа
      const currentHistory = state.metricsHistory[symbol] || [];

      // Добавляем новую точку и ограничиваем размер истории
      const newHistory = [...currentHistory, data].slice(-MAX_HISTORY_POINTS);

      return {
        metrics: {
          ...state.metrics,
          [symbol]: data,
        },
        metricsHistory: {
          ...state.metricsHistory,
          [symbol]: newHistory,
        },
      };
    }),

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
      metricsHistory: {},
      selectedSymbol: null,
      isConnected: false,
    }),
}));