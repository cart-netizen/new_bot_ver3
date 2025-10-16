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
  error: string | null;

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
  error: null,

  /**
   * Загрузка списка пар из API.
   */
  fetchPairs: async () => {
    console.log('[ScreenerStore] Starting fetchPairs...');
    console.log('[ScreenerStore] Current state before fetch:', {
      pairsCount: get().pairs.length,
      isLoading: get().isLoading
    });

    set({ isLoading: true, error: null });

    try {
      console.log('[ScreenerStore] Making API request to /screener/pairs');
      const response = await apiClient.get('/screener/pairs');

      console.log('[ScreenerStore] Raw response:', response);
      console.log('[ScreenerStore] Response status:', response.status);
      console.log('[ScreenerStore] Response data type:', typeof response.data);
      console.log('[ScreenerStore] Response data keys:', Object.keys(response.data || {}));

      // ИСПРАВЛЕНО: Проверяем структуру ответа без throw
      if (!response.data) {
        console.error('[ScreenerStore] ❌ Response data is undefined');
        set({
          isLoading: false,
          error: 'Пустой ответ от сервера'
        });
        return;
      }

      if (!response.data.pairs) {
        console.error('[ScreenerStore] ❌ Response.data.pairs is undefined!', response.data);
        set({
          isLoading: false,
          error: 'Неверная структура ответа (отсутствует поле pairs)'
        });
        return;
      }

      if (!Array.isArray(response.data.pairs)) {
        console.error('[ScreenerStore] ❌ Response.data.pairs is not an array!', typeof response.data.pairs);
        set({
          isLoading: false,
          error: 'Неверный формат данных (pairs должен быть массивом)'
        });
        return;
      }

      const pairs = response.data.pairs;
      console.log('[ScreenerStore] Extracted pairs:', {
        count: pairs.length,
        isArray: Array.isArray(pairs),
        sample: pairs.slice(0, 2)
      });

      const selectedPairs = pairs
        .filter((p: ScreenerPair) => p.is_selected)
        .map((p: ScreenerPair) => p.symbol);

      console.log('[ScreenerStore] Setting new state:', {
        pairsCount: pairs.length,
        selectedCount: selectedPairs.length
      });

      set({
        pairs,
        selectedPairs,
        isLoading: false,
        error: null
      });

      // Проверяем, что state действительно обновился
      const newState = get();
      console.log('[ScreenerStore] State after update:', {
        pairsCount: newState.pairs.length,
        isLoading: newState.isLoading,
        error: newState.error
      });

      console.log('[ScreenerStore] ✅ State updated successfully');
    } catch (error) {
      // ИСПРАВЛЕНО: Правильная типизация error
      console.error('[ScreenerStore] ❌ Failed to fetch pairs:', error);

      let errorMessage = 'Неизвестная ошибка';

      if (error instanceof Error) {
        errorMessage = error.message;
        console.error('[ScreenerStore] Error message:', error.message);
        console.error('[ScreenerStore] Error stack:', error.stack);
      }

      // Проверяем, есть ли response в ошибке (axios error)
      if (typeof error === 'object' && error !== null && 'response' in error) {
        const axiosError = error as { response?: { data?: { detail?: string }; status?: number } };
        console.error('[ScreenerStore] Axios error details:', {
          status: axiosError.response?.status,
          detail: axiosError.response?.data?.detail
        });
        errorMessage = axiosError.response?.data?.detail || errorMessage;
      }

      set({
        isLoading: false,
        error: errorMessage
      });
    }
  },

  /**
   * Обновление списка пар (из WebSocket).
   */
  updatePairs: (pairs: ScreenerPair[]) => {
    console.log('[ScreenerStore] Updating pairs from WebSocket:', pairs.length);
    set({
      pairs,
      selectedPairs: pairs.filter(p => p.is_selected).map(p => p.symbol),
    });
  },

  /**
   * Переключение выбора пары для графиков.
   */
  togglePairSelection: async (symbol: string) => {
    console.log('[ScreenerStore] Toggling selection for:', symbol);

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

      console.log('[ScreenerStore] Toggle successful');
    } catch (error) {
      console.error(`[ScreenerStore] Failed to toggle ${symbol}:`, error);

      if (error instanceof Error) {
        console.error('[ScreenerStore] Toggle error:', error.message);
      }
    }
  },

  /**
   * Установка параметров сортировки.
   */
  setSorting: (field: SortField, order: SortOrder) => {
    console.log('[ScreenerStore] Setting sort:', { field, order });
    set({ sortField: field, sortOrder: order });
  },

  /**
   * Получение отсортированного списка пар.
   */
  getSortedPairs: () => {
    const state = get();
    const { pairs, sortField, sortOrder } = state;

    if (pairs.length === 0) {
      console.log('[ScreenerStore] No pairs to sort');
      return [];
    }

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
        case 'change_24h':
          aValue = a.price_change_24h_percent;
          bValue = b.price_change_24h_percent;
          break;
        case 'volume':
          aValue = a.volume_24h;
          bValue = b.volume_24h;
          break;
        default:
          return 0;
      }

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortOrder === 'asc'
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }

      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
      }

      return 0;
    });

    console.log('[ScreenerStore] Sorted pairs:', {
      count: sorted.length,
      sortField,
      sortOrder
    });

    return sorted;
  },
}));