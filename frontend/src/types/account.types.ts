// frontend/src/types/account.types.ts

/**
 * Баланс по одной валюте.
 */
export interface AssetBalance {
  asset: string;
  free: number;
  locked: number;
  total: number;
}

/**
 * Полный баланс аккаунта.
 */
export interface AccountBalance {
  balances: Record<string, AssetBalance>;
  total_usdt: number;
  timestamp: number;
}

/**
 * Точка данных для графика истории баланса.
 */
export interface BalanceHistoryPoint {
  timestamp: number;
  balance: number;
  datetime: string;
}

/**
 * История изменений баланса.
 */
export interface BalanceHistory {
  points: BalanceHistoryPoint[];
  period: '1h' | '24h' | '7d' | '30d';
}

/**
 * Статистика по балансу.
 */
export interface BalanceStats {
  initial_balance: number;
  current_balance: number;
  total_pnl: number;
  total_pnl_percentage: number;
  daily_pnl: number;
  daily_pnl_percentage: number;
  best_day: number;
  worst_day: number;
}