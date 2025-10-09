export const SignalType = {
  BUY: 'BUY',
  SELL: 'SELL',
} as const;

export type SignalType = typeof SignalType[keyof typeof SignalType];

export const SignalStrength = {
  WEAK: 'WEAK',
  MEDIUM: 'MEDIUM',
  STRONG: 'STRONG',
} as const;

export type SignalStrength = typeof SignalStrength[keyof typeof SignalStrength];

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