import apiClient from './client';
import { API_ENDPOINTS } from '../config/app.config.ts';
import type { OrderBook } from '../types/orderbook.types.ts';
import type { OrderBookMetrics } from '../types/metrics.types.ts';

export const marketApi = {
  getPairs: async (): Promise<{ pairs: string[]; count: number }> => {
    const response = await apiClient.get(API_ENDPOINTS.data.pairs);
    return response.data;
  },

  getOrderBook: async (symbol: string): Promise<OrderBook> => {
    const response = await apiClient.get<OrderBook>(
      API_ENDPOINTS.data.orderbook(symbol)
    );
    return response.data;
  },

  getMetrics: async (symbol: string): Promise<OrderBookMetrics> => {
    const response = await apiClient.get<OrderBookMetrics>(
      API_ENDPOINTS.data.metrics(symbol)
    );
    return response.data;
  },

  getAllMetrics: async (): Promise<Record<string, OrderBookMetrics>> => {
    const response = await apiClient.get<Record<string, OrderBookMetrics>>(
      API_ENDPOINTS.data.allMetrics
    );
    return response.data;
  },
};