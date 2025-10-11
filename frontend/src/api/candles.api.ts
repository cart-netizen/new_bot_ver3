// frontend/src/api/candles.api.ts

import apiClient from './client';
import type { CandlesResponse } from '../types/candle.types';
import { CandleInterval } from '../types/candle.types';

/**
 * API для работы со свечами.
 */
export const candlesApi = {
  /**
   * Получение свечей для торговой пары.
   *
   * @param symbol - Торговая пара (например, BTCUSDT)
   * @param interval - Интервал свечи (по умолчанию 1 минута)
   * @param limit - Количество свечей (по умолчанию 100)
   * @returns Promise с данными свечей
   */
  async getCandles(
    symbol: string,
    interval: CandleInterval = CandleInterval.MIN_1,
    limit: number = 100
  ): Promise<CandlesResponse> {
    const response = await apiClient.get<CandlesResponse>(
      `/data/candles/${symbol}`,
      {
        params: {
          interval,
          limit,
        },
      }
    );
    return response.data;
  },
};