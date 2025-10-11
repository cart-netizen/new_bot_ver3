// frontend/src/types/candle.types.ts

/**
 * Свеча (Candlestick) - OHLCV данные за период времени.
 */
export interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  turnover: number;
}

/**
 * Интервалы свечей.
 */
export enum CandleInterval {
  MIN_1 = '1',
  MIN_3 = '3',
  MIN_5 = '5',
  MIN_15 = '15',
  MIN_30 = '30',
  HOUR_1 = '60',
  HOUR_2 = '120',
  HOUR_4 = '240',
  HOUR_6 = '360',
  HOUR_12 = '720',
  DAY_1 = 'D',
  WEEK_1 = 'W',
  MONTH_1 = 'M',
}

/**
 * Ответ API для получения свечей.
 */
export interface CandlesResponse {
  symbol: string;
  interval: string;
  candles: Candle[];
  count: number;
}