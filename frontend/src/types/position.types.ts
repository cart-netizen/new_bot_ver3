/**
 * Типы для позиций и ордеров с Bybit.
 * frontend/src/types/position.types.ts
 */

/**
 * Позиция с биржи Bybit.
 */
export interface Position {
  // Основная информация
  symbol: string;
  side: 'Buy' | 'Sell';  // Buy = Long, Sell = Short
  size: number;
  avgPrice: number;
  positionValue: number;

  // Плечо и маржа
  leverage: string;
  positionIM: number;  // Initial Margin
  positionMM: number;  // Maintenance Margin

  // Цены
  markPrice: number;
  liqPrice: number;

  // P&L
  unrealisedPnl: number;
  cumRealisedPnl: number;
  roePercent: number;

  // TP/SL
  takeProfit: string;
  stopLoss: string;
  trailingStop: string;
  tpslMode: string;

  // Метаданные
  positionIdx: number;
  tradeMode: number;
  positionStatus: string;
  updatedTime: string;
  createdTime: string;
}

/**
 * Открытый ордер с биржи Bybit.
 */
export interface OpenOrder {
  orderId: string;
  orderLinkId: string;
  symbol: string;
  side: 'Buy' | 'Sell';
  orderType: 'Limit' | 'Market';
  price: number;
  qty: number;
  cumExecQty: number;
  cumExecValue: number;
  orderStatus: string;
  timeInForce: string;
  stopLoss: string;
  takeProfit: string;
  reduceOnly: boolean;
  createdTime: string;
  updatedTime: string;
}

/**
 * Ответ API с позициями.
 */
export interface PositionsResponse {
  positions: Position[];
  count: number;
  timestamp: number;
  error?: string;
}

/**
 * Ответ API с ордерами.
 */
export interface OrdersResponse {
  orders: OpenOrder[];
  count: number;
  timestamp: number;
  error?: string;
}

/**
 * Запрос на закрытие позиции.
 */
export interface ClosePositionRequest {
  symbol: string;
  percent: 25 | 50 | 75 | 100;
}

/**
 * Ответ на закрытие позиции.
 */
export interface ClosePositionResponse {
  success: boolean;
  orderId: string;
  symbol: string;
  closedSize: number;
  closedPercent: number;
  message: string;
}

/**
 * Запрос на отмену ордера.
 */
export interface CancelOrderRequest {
  symbol: string;
  orderId: string;
}

/**
 * Ответ на отмену ордера.
 */
export interface CancelOrderResponse {
  success: boolean;
  orderId: string;
  symbol: string;
  message: string;
}
