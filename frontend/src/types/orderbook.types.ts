
export interface OrderBook {
  symbol: string;
  bids: [number, number][];
  asks: [number, number][];
  timestamp: number;
  best_bid: number | null;
  best_ask: number | null;
  spread: number | null;
  mid_price: number | null;
}