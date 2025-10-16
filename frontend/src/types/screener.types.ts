// frontend/src/types/screener.types.ts

/**
 * Данные торговой пары в скринере.
 */
export interface ScreenerPair {
  symbol: string;
  last_price: number;
  volume_24h: number;
  price_change_24h_percent: number;
  high_24h: number;
  low_24h: number;
  price_change_5m: number | null;
  price_change_15m: number | null;
  price_change_1h: number | null;
  price_change_4h: number | null;
  last_update: number;
  is_selected: boolean;
}

/**
 * Ответ API со списком пар.
 */
export interface ScreenerPairsResponse {
  pairs: ScreenerPair[];
  total: number;
  timestamp: number;
}

/**
 * Параметры сортировки.
 */
export type SortField = 'symbol' | 'price' | 'volume' | 'change_24h';
export type SortOrder = 'asc' | 'desc';