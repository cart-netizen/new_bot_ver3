import { create } from 'zustand';
import type { OrderBook } from '../types/orderbook.types.ts';
import type { OrderBookMetrics } from '../types/metrics.types.ts';

interface MarketState {
  orderbooks: Record<string, OrderBook>;
  metrics: Record<string, OrderBookMetrics>;
  selectedSymbol: string | null;
  updateOrderBook: (symbol: string, data: OrderBook) => void;
  updateMetrics: (symbol: string, data: OrderBookMetrics) => void;
  setSelectedSymbol: (symbol: string) => void;
}

export const useMarketStore = create<MarketState>((set) => ({
  orderbooks: {},
  metrics: {},
  selectedSymbol: null,

  updateOrderBook: (symbol, data) =>
    set((state) => ({
      orderbooks: {
        ...state.orderbooks,
        [symbol]: data,
      },
    })),

  updateMetrics: (symbol, data) =>
    set((state) => ({
      metrics: {
        ...state.metrics,
        [symbol]: data,
      },
    })),

  setSelectedSymbol: (symbol) => set({ selectedSymbol: symbol }),
}));