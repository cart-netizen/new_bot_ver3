import apiClient from './client';
import { API_ENDPOINTS } from '../config/app.config.ts';
import type { TradingSignal } from '../types/signal.types.ts';

export const tradingApi = {
  getSignals: async (params?: {
    symbol?: string;
    limit?: number;
  }): Promise<TradingSignal[]> => {
    const response = await apiClient.get<TradingSignal[]>(
      API_ENDPOINTS.trading.signals,
      { params }
    );
    return response.data;
  },
};