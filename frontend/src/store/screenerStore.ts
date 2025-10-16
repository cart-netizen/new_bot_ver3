// frontend/src/store/screenerStore.ts

import { create } from 'zustand';
import apiClient from '../api/client';
import type { ScreenerPair, SortField, SortOrder } from '../types/screener.types';

interface ScreenerStore {
  // Состояние
  pairs: ScreenerPair[];
  selectedPairs: string[];
  sortField: SortField;
  sortOrder: SortOrder;
  isLoading: boolean;

  // Действия
  fetchPairs: () => Promise<void>;
  updatePairs: (pairs: ScreenerPair[]) => void;
  togglePairSelection: (symbol: string) => Promise<void>;
  setSorting: (field: SortField, order: SortOrder) => void;
  getSortedPairs: () => ScreenerPair[];
}

/**
 * Store для управления скринером торговых пар.
 */
export const useScreenerStore = create<ScreenerStore>((set, get) => ({
  // Начальное состояние
  pairs: [],
  selectedPairs: [],
  sortField: 'volume',
  sortOrder: 'desc',
  isLoading: false,

  /**
   * Загрузка списка пар из API.
   */
  fetchPairs: async () => {
    set({ isLoading: true });

    try {
      const response = await apiClient.get('/screener/pairs');
      const pairs = response.data.pairs || [];

      set({
        pairs,
        selectedPairs: pairs.filter((p: ScreenerPair) => p.is_selected).map((p: ScreenerPair) => p.symbol),
        isLoading: false,
      });
    } catch (error) {
      console.error('[ScreenerStore] Failed to fetch pairs:', error);
      set({ isLoading: false });
    }
  },

  /**
   * Обновление списка пар (из WebSocket).
   */
  updatePairs: (pairs: ScreenerPair[]) => {
    set({
      pairs,
      selectedPairs: pairs.filter(p => p.is_selected).map(p => p.symbol),
    });
  },

  /**
   * Переключение выбора пары для графиков.
   */
  togglePairSelection: async (symbol: string) => {
    try {
      await apiClient.post(`/screener/pair/${symbol}/toggle`);

      // Оптимистичное обновление
      set(state => {
        const pairs = state.pairs.map(p =>
          p.symbol === symbol
            ? { ...p, is_selected: !p.is_selected }
            : p
        );

        return {
          pairs,
          selectedPairs: pairs.filter(p => p.is_selected).map(p => p.symbol),
        };
      });
    } catch (error) {
      console.error(`[ScreenerStore] Failed to toggle ${symbol}:`, error);
    }
  },

  /**
   * Установка параметров сортировки.
   */
  setSorting: (field: SortField, order: SortOrder) => {
    set({ sortField: field, sortOrder: order });
  },

  /**
   * Получение отсортированного списка пар.
   */
  getSortedPairs: () => {
    const { pairs, sortField, sortOrder } = get();

    const sorted = [...pairs].sort((a, b) => {
      let aValue: number | string;
      let bValue: number | string;

      switch (sortField) {
        case 'symbol':
          aValue = a.symbol;
          bValue = b.symbol;
          break;
        case 'price':
          aValue = a.last_price;
          bValue = b.last_price;
          break;
        case 'volume':
          aValue = a.volume_24h;
          bValue = b.volume_24h;
          break;
        case 'change_24h':
          aValue = a.price_change_24h_percent;
          bValue = b.price_change_24h_percent;
          break;
        default:
          return 0;
      }

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortOrder === 'asc'
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }

      return sortOrder === 'asc'
        ? (aValue as number) - (bValue as number)
        : (bValue as number) - (aValue as number);
    });

    return sorted;
  },
}));