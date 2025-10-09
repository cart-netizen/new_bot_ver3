export enum SignalType {
  BUY = 'BUY',
  SELL = 'SELL',
}

export enum SignalStrength {
  WEAK = 'WEAK',
  MEDIUM = 'MEDIUM',
  STRONG = 'STRONG',
}

export interface TradingSignal {
  symbol: string;
  signal_type: SignalType;
  strength: SignalStrength;
  timestamp: number;
  price: number;
  confidence: number;
  metrics: {
    imbalance: number | null;
  };
  reason: string;
  status: {
    is_valid: boolean;
  };
}