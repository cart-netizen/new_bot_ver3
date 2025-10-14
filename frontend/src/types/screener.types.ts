// frontend/src/types/screener.types.ts
/**
 * Типы данных для функционала скринера.
 *
 * Используется для:
 * - Store управления данными
 * - API запросов и ответов
 * - Компонентов отображения
 */

/**
 * Точка данных цены для расчета динамики.
 */
export interface PricePoint {
  price: number;
  timestamp: number;
}

/**
 * Изменение цены по таймфрейму.
 */
export interface PriceChange {
  timeframe: string; // '1m', '3m', '5m', '15m'
  change_percent: number;
  previous_price: number | null;
  current_price: number;
}

/**
 * Данные торговой пары в скринере.
 */
export interface ScreenerPairData {
  symbol: string;
  lastPrice: number;
  price24hPcnt: number; // Изменение за 24ч в процентах
  volume24h: number;    // Объем за 24ч в USDT
  highPrice24h: number;
  lowPrice24h: number;
  prevPrice24h: number;

  // Динамика по таймфреймам
  changes: {
    '1m': PriceChange | null;
    '3m': PriceChange | null;
    '5m': PriceChange | null;
    '15m': PriceChange | null;
  };

  // История цен для расчета динамики
  priceHistory: PricePoint[];

  // Метаданные
  lastUpdate: number;
  isActive: boolean;
}

/**
 * Параметры сортировки.
 */
export type SortField = 'symbol' | 'lastPrice' | 'price24hPcnt' | 'volume24h';
export type SortDirection = 'asc' | 'desc';

/**
 * Ответ API со списком пар скринера.
 */
export interface ScreenerPairsResponse {
  pairs: Array<{
    symbol: string;
    lastPrice: number;
    price24hPcnt: number;
    volume24h: number;
    highPrice24h: number;
    lowPrice24h: number;
    prevPrice24h: number;
    turnover24h: number;
  }>;
  total: number;
  timestamp: number;
  min_volume: number;
}

/**
 * WebSocket сообщение с обновлением данных скринера.
 */
export interface ScreenerUpdateMessage {
  type: 'screener_update';
  data: {
    symbol: string;
    lastPrice: number;
    volume24h: number;
    price24hPcnt: number;
    highPrice24h: number;
    lowPrice24h: number;
    prevPrice24h: number;
  };
}

/**
 * WebSocket сообщение с тиком цены.
 */
export interface TickerMessage {
  type: 'ticker';
  data: {
    symbol: string;
    lastPrice: number;
    timestamp?: number;
  };
}

/**
 * Статистика памяти скринера.
 */
export interface ScreenerMemoryStats {
  totalPairs: number;
  activePairs: number;
  totalPricePoints: number;
  lastCleanup: number;
}