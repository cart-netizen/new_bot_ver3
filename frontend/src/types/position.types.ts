// frontend/src/types/position.types.ts

/**
 * Сторона позиции.
 */
export type PositionSide = 'LONG' | 'SHORT';

/**
 * Статус позиции.
 */
export type PositionStatus = 'OPEN' | 'CLOSED' | 'LIQUIDATED';

/**
 * Позиция.
 */
export interface Position {
  position_id: string;
  symbol: string;
  side: PositionSide;
  status: PositionStatus;
  entry_price: number;
  current_price: number;
  quantity: number;
  leverage: number;
  unrealized_pnl: number;
  realized_pnl: number;
  margin_used: number;
  liquidation_price: number;
  created_at: string;
  updated_at: string;
}