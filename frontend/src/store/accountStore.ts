// frontend/src/store/accountStore.ts

import { create } from 'zustand';
import type { AccountBalance, BalanceHistory, BalanceStats } from '../types/account.types';
import apiClient from '../api/client';

/**
 * Состояние аккаунта.
 */
interface AccountState {
  // Данные
  balance: AccountBalance | null;
  balanceHistory: BalanceHistory | null;
  balanceStats: BalanceStats | null;

  // Загрузка
  isLoadingBalance: boolean;
  isLoadingHistory: boolean;
  isLoadingStats: boolean;

  // Ошибки
  error: string | null;

  // Методы
  fetchBalance: () => Promise<void>;
  fetchBalanceHistory: (period?: '1h' | '24h' | '7d' | '30d') => Promise<void>;
  fetchBalanceStats: () => Promise<void>;
  reset: () => void;
}

/**
 * Zustand store для данных аккаунта.
 */
export const useAccountStore = create<AccountState>((set) => ({
  balance: null,
  balanceHistory: null,
  balanceStats: null,
  isLoadingBalance: false,
  isLoadingHistory: false,
  isLoadingStats: false,
  error: null,

  /**
   * Получение баланса аккаунта.
   */
  fetchBalance: async () => {
    set({ isLoadingBalance: true, error: null });
    try {
      const response = await apiClient.get<{ balance: AccountBalance }>('/trading/balance');
      set({ balance: response.data.balance, isLoadingBalance: false });
    } catch (error) {
      console.error('[AccountStore] Failed to fetch balance:', error);
      set({
        error: 'Не удалось загрузить баланс',
        isLoadingBalance: false
      });
    }
  },

  /**
   * Получение истории баланса.
   */
  fetchBalanceHistory: async (period = '24h') => {
    set({ isLoadingHistory: true, error: null });
    try {
      const response = await apiClient.get<BalanceHistory>(
        `/trading/balance/history?period=${period}`
      );
      set({ balanceHistory: response.data, isLoadingHistory: false });
    } catch (error) {
      console.error('[AccountStore] Failed to fetch balance history:', error);
      set({
        error: 'Не удалось загрузить историю баланса',
        isLoadingHistory: false
      });
    }
  },

  /**
   * Получение статистики баланса.
   */
  fetchBalanceStats: async () => {
    set({ isLoadingStats: true, error: null });
    try {
      const response = await apiClient.get<BalanceStats>('/trading/balance/stats');
      set({ balanceStats: response.data, isLoadingStats: false });
    } catch (error) {
      console.error('[AccountStore] Failed to fetch balance stats:', error);
      set({
        error: 'Не удалось загрузить статистику',
        isLoadingStats: false
      });
    }
  },

  /**
   * Сброс состояния.
   */
  reset: () => set({
    balance: null,
    balanceHistory: null,
    balanceStats: null,
    isLoadingBalance: false,
    isLoadingHistory: false,
    isLoadingStats: false,
    error: null,
  }),
}));