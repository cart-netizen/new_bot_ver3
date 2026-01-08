/**
 * API для работы с позициями и ордерами.
 * frontend/src/api/positions.api.ts
 */

import apiClient from './client';
import type {
  PositionsResponse,
  OrdersResponse,
  ClosePositionRequest,
  ClosePositionResponse,
  CancelOrderResponse,
} from '../types/position.types';

/**
 * API endpoints для позиций и ордеров.
 */
export const positionsApi = {
  /**
   * Получить открытые позиции с биржи.
   */
  getPositions: async (): Promise<PositionsResponse> => {
    const response = await apiClient.get<PositionsResponse>('/trading/positions/exchange');
    return response.data;
  },

  /**
   * Получить открытые ордера с биржи.
   */
  getOpenOrders: async (symbol?: string): Promise<OrdersResponse> => {
    const params = symbol ? { symbol } : {};
    const response = await apiClient.get<OrdersResponse>('/trading/orders/open', { params });
    return response.data;
  },

  /**
   * Закрыть позицию по рыночной цене.
   */
  closePosition: async (request: ClosePositionRequest): Promise<ClosePositionResponse> => {
    const response = await apiClient.post<ClosePositionResponse>('/trading/position/close', request);
    return response.data;
  },

  /**
   * Отменить ордер.
   */
  cancelOrder: async (symbol: string, orderId: string): Promise<CancelOrderResponse> => {
    const response = await apiClient.post<CancelOrderResponse>(
      '/trading/order/cancel',
      null,
      { params: { symbol, order_id: orderId } }
    );
    return response.data;
  },
};

export default positionsApi;
