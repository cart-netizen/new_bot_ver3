// ==================== frontend/src/types/index.ts ====================
/**
 * ОБНОВИТЬ: Добавить экспорт типов ордеров
 */

// Существующие типы
export type { OrderBook, OrderBookMetrics, OrderBookLevel, SymbolStatus } from './orderbook.types';
export type { TradingSignal, SignalType, SignalStrength } from './signal.types';
export type { Position, PositionSide, PositionStatus } from './position.types';
export type {
  ScreenerPairData,
  PricePoint,
  PriceChange,
  SortField as ScreenerSortField,
  SortDirection as ScreenerSortDirection,
  ScreenerPairsResponse,
  ScreenerUpdateMessage,
  TickerMessage,
  ScreenerMemoryStats,
} from './screener.types';

// Новые типы ордеров
export type {
  Order,
  OrderDetail,
  OrderType,
  OrderSide,
  OrderStatus,
  OrderFill,
  CreateOrderParams,
  CloseOrderParams,
  CreateOrderResponse,
  CloseOrderResponse,
  OrderFilters,
  OrdersStats,
} from './orders.types'; // ДОБАВИТЬ