// frontend/src/types/orderbook.types.ts

/**
 * Уровень в стакане ордеров.
 * Представляет одну ценовую точку с объемом.
 */
export interface OrderBookLevel {
  price: number;
  quantity: number;
}

/**
 * Снимок стакана ордеров.
 * Содержит полное состояние книги ордеров для торговой пары.
 */
export interface OrderBook {
  symbol: string;
  bids: [number, number][];  // [price, quantity]
  asks: [number, number][];  // [price, quantity]
  timestamp: number;
  best_bid: number | null;
  best_ask: number | null;
  spread: number | null;
  mid_price: number | null;
  update_id?: number;
  sequence_id?: number;
}

/**
 * Метрики стакана для анализа.
 * Содержит рассчитанные показатели для принятия торговых решений.
 */
export interface OrderBookMetrics {
  symbol: string;
  timestamp: number;
  datetime: string;

  // Ценовые метрики
  prices: {
    best_bid: number | null;
    best_ask: number | null;
    spread: number | null;
    mid_price: number | null;
  };

  // Объемные метрики
  volumes: {
    total_bid: number;
    total_ask: number;
    bid_depth_5: number;
    ask_depth_5: number;
    bid_depth_10: number;
    ask_depth_10: number;
  };

  // Дисбаланс спроса/предложения
  imbalance: {
    overall: number;      // 0.0 (все продажи) - 0.5 (баланс) - 1.0 (все покупки)
    depth_5: number;
    depth_10: number;
  };

  // VWAP метрики
  vwap: {
    bid: number | null;
    ask: number | null;
    mid: number | null;
  };

  // Кластеры объема
  clusters: {
    largest_bid: {
      price: number | null;
      volume: number;
    };
    largest_ask: {
      price: number | null;
      volume: number;
    };
  };
}

/**
 * Статус торговой пары.
 */
export interface SymbolStatus {
  symbol: string;
  active: boolean;
  last_update: number;
  signals_count: number;
  imbalance: number;
  spread: number;
}