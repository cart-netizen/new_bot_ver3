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

  // Все временные интервалы для динамики цены
  price_change_1m: number | null;
  price_change_2m: number | null;
  price_change_5m: number | null;
  price_change_15m: number | null;
  price_change_30m: number | null;
  price_change_1h: number | null;
  price_change_4h: number | null;
  price_change_8h: number | null;
  price_change_12h: number | null;
  price_change_24h: number | null;

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
 * Параметры сортировки - все возможные столбцы.
 */
export type SortField =
  | 'symbol'
  | 'price'
  | 'volume'
  | 'change_24h'
  | 'change_1m'
  | 'change_2m'
  | 'change_5m'
  | 'change_15m'
  | 'change_30m'
  | 'change_1h'
  | 'change_4h'
  | 'change_8h'
  | 'change_12h'
  | 'change_24h_interval';

export type SortOrder = 'asc' | 'desc';

/**
 * Настройки скринера.
 */
export interface ScreenerSettings {
  minVolume: number;        // Минимальный объем торгов (USDT)
  topN: number;             // Топ N пар из отфильтрованного списка
  refreshInterval: number;  // Частота обновления (секунды)
  alertThreshold: number;   // Пороговое значение для алертов (%)
}

/**
 * Алерт для торговой пары.
 */
export interface PairAlert {
  symbol: string;
  timestamp: number;
  field: string;           // Какой интервал превысил порог
  value: number;           // Значение изменения цены (%)
  threshold: number;       // Пороговое значение
}