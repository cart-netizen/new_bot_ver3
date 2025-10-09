import { create } from 'zustand';
import type { TradingSignal } from '../types/signal.types.ts';

interface TradingState {
  signals: TradingSignal[];
  addSignal: (signal: TradingSignal) => void;
}

export const useTradingStore = create<TradingState>((set) => ({
  signals: [],

  addSignal: (signal) =>
    set((state) => ({
      signals: [signal, ...state.signals].slice(0, 100),
    })),
}));