export interface LoginRequest {
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

export interface ConfigResponse {
  trading_pairs: string[];
  bybit_mode: string;
  orderbook_depth: number;
  imbalance_buy_threshold: number;
  imbalance_sell_threshold: number;
  max_open_positions: number;
  max_exposure_usdt: number;
}