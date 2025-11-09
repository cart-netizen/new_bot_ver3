// frontend/src/store/screenerStore.ts

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import apiClient from '../api/client';
import type { ScreenerPair, SortField, SortOrder, ScreenerSettings, PairAlert } from '../types/screener.types';

interface ScreenerStore {
  // Состояние данных
  pairs: ScreenerPair[];
  selectedPairs: string[];
  sortField: SortField;
  sortOrder: SortOrder;
  isLoading: boolean;
  error: string | null;

  // Настройки скринера
  settings: ScreenerSettings;

  // Алерты
  alerts: Map<string, PairAlert>;

  // Действия
  fetchPairs: () => Promise<void>;
  updatePairs: (pairs: ScreenerPair[]) => void;
  togglePairSelection: (symbol: string) => Promise<void>;
  setSorting: (field: SortField, order: SortOrder) => void;
  getSortedPairs: () => ScreenerPair[];

  // Настройки
  updateSettings: (settings: Partial<ScreenerSettings>) => void;

  // Алерты
  checkAlerts: () => void;
  dismissAlert: (symbol: string) => void;
  getAlertedPairs: () => ScreenerPair[];
}

/**
 * Store для управления скринером торговых пар.
 * ОБНОВЛЕНО: Добавлены настройки, алерты, оптимизация.
 */
export const useScreenerStore = create<ScreenerStore>()(
  persist(
    (set, get) => ({
      // Начальное состояние
      pairs: [],
      selectedPairs: [],
      sortField: 'volume',
      sortOrder: 'desc',
      isLoading: false,
      error: null,

      // Настройки по умолчанию
      settings: {
        minVolume: 4_000_000,      // 4M USDT
        topN: 100,                  // Топ 100 пар
        refreshInterval: 5,         // 5 секунд
        alertThreshold: 5,          // 5% изменение
      },

      // Алерты (используем Map для быстрого доступа)
      alerts: new Map<string, PairAlert>(),

      /**
       * Загрузка списка пар из API.
       */
      fetchPairs: async () => {
        const currentSettings = get().settings;

        console.log('[ScreenerStore] Starting fetchPairs with settings:', currentSettings);
        set({ isLoading: true, error: null });

        try {
          const response = await apiClient.get('/screener/pairs', {
            params: {
              min_volume: currentSettings.minVolume,
              sort_by: 'volume',
              sort_order: 'desc',
            }
          });

          console.log('[ScreenerStore] Raw API response:', response);
          console.log('[ScreenerStore] Response data:', response.data);

          if (!response.data?.pairs || !Array.isArray(response.data.pairs)) {
            console.error('[ScreenerStore] Invalid response structure:', {
              hasPairs: !!response.data?.pairs,
              isArray: Array.isArray(response.data?.pairs),
              responseData: response.data
            });
            throw new Error('Неверная структура ответа');
          }

          let pairs = response.data.pairs;
          console.log('[ScreenerStore] Received pairs count:', pairs.length);
          if (pairs.length > 0) {
            console.log('[ScreenerStore] First pair sample:', pairs[0]);
          }

          // Применяем лимит топ N
          if (currentSettings.topN > 0 && pairs.length > currentSettings.topN) {
            pairs = pairs.slice(0, currentSettings.topN);
          }

          const selectedPairs = pairs
            .filter((p: ScreenerPair) => p.is_selected)
            .map((p: ScreenerPair) => p.symbol);

          set({
            pairs,
            selectedPairs,
            isLoading: false,
            error: null
          });

          // Проверяем алерты после загрузки
          get().checkAlerts();

          console.log('[ScreenerStore] ✅ Loaded', pairs.length, 'pairs');
        } catch (error) {
          console.error('[ScreenerStore] ❌ Failed to fetch pairs:', error);

          let errorMessage = 'Неизвестная ошибка';
          if (error instanceof Error) {
            errorMessage = error.message;
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
        const currentSettings = get().settings;

        // Применяем фильтрацию и лимит
        let filteredPairs = pairs.filter(p => p.volume_24h >= currentSettings.minVolume);

        // Сортируем по объему для топ N
        filteredPairs.sort((a, b) => b.volume_24h - a.volume_24h);

        if (currentSettings.topN > 0 && filteredPairs.length > currentSettings.topN) {
          filteredPairs = filteredPairs.slice(0, currentSettings.topN);
        }

        set({
          pairs: filteredPairs,
          selectedPairs: filteredPairs.filter(p => p.is_selected).map(p => p.symbol),
        });

        // Проверяем алерты
        get().checkAlerts();
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
        const state = get();
        const { pairs, sortField, sortOrder } = state;

        if (pairs.length === 0) {
          return [];
        }

        const sorted = [...pairs].sort((a, b) => {
          let aValue: number | string | null;
          let bValue: number | string | null;

          // Маппинг полей
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
            case 'change_1m':
              aValue = a.price_change_1m;
              bValue = b.price_change_1m;
              break;
            case 'change_2m':
              aValue = a.price_change_2m;
              bValue = b.price_change_2m;
              break;
            case 'change_5m':
              aValue = a.price_change_5m;
              bValue = b.price_change_5m;
              break;
            case 'change_15m':
              aValue = a.price_change_15m;
              bValue = b.price_change_15m;
              break;
            case 'change_30m':
              aValue = a.price_change_30m;
              bValue = b.price_change_30m;
              break;
            case 'change_1h':
              aValue = a.price_change_1h;
              bValue = b.price_change_1h;
              break;
            case 'change_4h':
              aValue = a.price_change_4h;
              bValue = b.price_change_4h;
              break;
            case 'change_8h':
              aValue = a.price_change_8h;
              bValue = b.price_change_8h;
              break;
            case 'change_12h':
              aValue = a.price_change_12h;
              bValue = b.price_change_12h;
              break;
            case 'change_24h_interval':
              aValue = a.price_change_24h;
              bValue = b.price_change_24h;
              break;
            default:
              return 0;
          }

          // Обработка null значений (null всегда в конце)
          if (aValue === null && bValue === null) return 0;
          if (aValue === null) return sortOrder === 'asc' ? 1 : -1;
          if (bValue === null) return sortOrder === 'asc' ? -1 : 1;

          // Сортировка строк
          if (typeof aValue === 'string' && typeof bValue === 'string') {
            return sortOrder === 'asc'
              ? aValue.localeCompare(bValue)
              : bValue.localeCompare(aValue);
          }

          // Сортировка чисел
          if (typeof aValue === 'number' && typeof bValue === 'number') {
            return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
          }

          return 0;
        });

        return sorted;
      },

      /**
       * Обновление настроек.
       */
      updateSettings: (newSettings: Partial<ScreenerSettings>) => {
        set(state => ({
          settings: { ...state.settings, ...newSettings }
        }));
        // Перезагружаем данные с новыми настройками
        get().fetchPairs();
      },

      /**
       * Проверка алертов для всех пар.
       * FIX: Auto-cleanup old alerts to prevent memory leak
       */
      checkAlerts: () => {
        const { pairs, settings, alerts } = get();
        const threshold = settings.alertThreshold;
        const now = Date.now();
        const ALERT_TTL = 5 * 60 * 1000; // 5 минут TTL для алертов

        // FIX: Start with empty Map and only keep valid alerts
        const newAlerts = new Map<string, PairAlert>();
        const currentSymbols = new Set(pairs.map(p => p.symbol));

        // Check current pairs for new alerts
        pairs.forEach(pair => {
          // Проверяем все интервалы
          const intervals = [
            { field: 'change_1m', value: pair.price_change_1m },
            { field: 'change_2m', value: pair.price_change_2m },
            { field: 'change_5m', value: pair.price_change_5m },
            { field: 'change_15m', value: pair.price_change_15m },
            { field: 'change_30m', value: pair.price_change_30m },
            { field: 'change_1h', value: pair.price_change_1h },
            { field: 'change_4h', value: pair.price_change_4h },
            { field: 'change_8h', value: pair.price_change_8h },
            { field: 'change_12h', value: pair.price_change_12h },
            { field: 'change_24h', value: pair.price_change_24h },
          ];

          let hasActiveAlert = false;
          for (const interval of intervals) {
            if (interval.value !== null && Math.abs(interval.value) >= threshold) {
              hasActiveAlert = true;

              // Preserve existing alert or create new one
              const existingAlert = alerts.get(pair.symbol);
              if (existingAlert && (now - existingAlert.timestamp) < ALERT_TTL) {
                // Keep existing alert if it's still fresh
                newAlerts.set(pair.symbol, existingAlert);
              } else {
                // Create new alert
                newAlerts.set(pair.symbol, {
                  symbol: pair.symbol,
                  timestamp: now,
                  field: interval.field,
                  value: interval.value,
                  threshold,
                });
              }
              break; // Один алерт на пару
            }
          }

          // FIX: If pair no longer meets threshold, preserve alert for TTL
          if (!hasActiveAlert) {
            const existingAlert = alerts.get(pair.symbol);
            if (existingAlert && (now - existingAlert.timestamp) < ALERT_TTL) {
              // Keep recently dismissed alerts visible for a short time
              newAlerts.set(pair.symbol, existingAlert);
            }
          }
        });

        // FIX: Don't keep alerts for pairs that are no longer in the list
        // (automatically cleaned up by not adding them to newAlerts)

        set({ alerts: newAlerts });
      },

      /**
       * Отклонить алерт для пары.
       */
      dismissAlert: (symbol: string) => {
        set(state => {
          const newAlerts = new Map(state.alerts);
          newAlerts.delete(symbol);
          return { alerts: newAlerts };
        });
      },

      /**
       * Получить пары с активными алертами.
       */
      getAlertedPairs: () => {
        const { pairs, alerts } = get();
        const alertedSymbols = new Set(alerts.keys());

        return pairs.filter(pair => alertedSymbols.has(pair.symbol));
      },
    }),
    {
      name: 'screener-settings', // Сохраняем только настройки
      partialize: (state) => ({ settings: state.settings }), // Храним только настройки
    }
  )
);
