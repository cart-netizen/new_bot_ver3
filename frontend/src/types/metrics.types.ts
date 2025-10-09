export interface OrderBookMetrics {
  symbol: string;
  timestamp: number;
  best_bid: number | null;
  best_ask: number | null;
  spread: number | null;
  mid_price: number | null;
  imbalance: number | null;
  total_bid_volume: number;
  total_ask_volume: number;
  vwap_bid: number | null;
  vwap_ask: number | null;
}