// frontend/src/store/marketStore.ts
// ОПТИМИЗИРОВАННАЯ ВЕРСИЯ с управлением памятью

import { create } from 'zustand';
import type { OrderBook, OrderBookMetrics } from '../types/orderbook.types';

/**
 * Конфигурация управления памятью.
 * КРИТИЧЕСКИ ВАЖНО: Ограничиваем размер хранимых данных.
 */
const MEMORY_CONFIG = {
  // Максимальное количество исторических snapshot'ов orderbook на символ
  MAX_ORDERBOOK_HISTORY: 1,  // Храним только последний snapshot

  // Максимальное количество метрик на символ
  MAX_METRICS_HISTORY: 10,  // Последние 10 обновлений метрик

  // Максимальное количество активных символов
  MAX_ACTIVE_SYMBOLS: 100,

  // Время жизни неактивного символа (мс)
  INACTIVE_SYMBOL_TTL: 5 * 60 * 1000,  // 5 минут

  // Интервал очистки памяти (мс)
  CLEANUP_INTERVAL: 60 * 1000,  // Каждую минуту
} as const;

/**
 * Структура для хранения истории метрик с timestamp.
 */
interface MetricsHistory {
  metrics: OrderBookMetrics;
  timestamp: number;
}

/**
 * Структура для отслеживания активности символа.
 */
interface SymbolActivity {
  lastUpdate: number;
  updateCount: number;
}

/**
 * Состояние рыночных данных с оптимизированным управлением памятью.
 */
interface MarketState {
  // Данные стаканов (только последние snapshot'ы)
  orderbooks: Record<string, OrderBook>;

  // История метрик по парам (ограниченная)
  metricsHistory: Record<string, MetricsHistory[]>;

  // Последние метрики для быстрого доступа
  currentMetrics: Record<string, OrderBookMetrics>;

  // Отслеживание активности символов
  symbolActivity: Record<string, SymbolActivity>;

  // Выбранная пара для детального просмотра
  selectedSymbol: string | null;

  // Статус подключения WebSocket
  isConnected: boolean;

  // Статистика памяти
  memoryStats: {
    totalSymbols: number;
    totalOrderbooks: number;
    totalMetrics: number;
    lastCleanup: number;
  };

  // Методы обновления данных
  updateOrderBook: (symbol: string, data: OrderBook) => void;
  updateMetrics: (symbol: string, data: OrderBookMetrics) => void;
  setSelectedSymbol: (symbol: string | null) => void;
  setConnected: (connected: boolean) => void;

  // Методы управления памятью
  cleanupMemory: () => void;
  getMemoryUsage: () => number;
  reset: () => void;
}

/**
 * Zustand store для рыночных данных с оптимизированным управлением памятью.
 *
 * Ключевые оптимизации:
 * 1. Ограничение истории метрик
 * 2. Хранение только последних orderbook snapshot'ов
 * 3. Автоматическая очистка неактивных символов
 * 4. Периодический сборщик мусора
 */
export const useMarketStore = create<MarketState>((set, get) => {
  // Таймер для периодической очистки памяти
  // ИСПРАВЛЕНО: Используем number вместо NodeJS.Timeout (browser environment)
  let cleanupIntervalId: number | null = null;

  // Запускаем периодическую очистку при создании store
  if (typeof window !== 'undefined') {
    cleanupIntervalId = window.setInterval(() => {
      const state = get();
      state.cleanupMemory();
    }, MEMORY_CONFIG.CLEANUP_INTERVAL);
  }

  return {
    orderbooks: {},
    metricsHistory: {},
    currentMetrics: {},
    symbolActivity: {},
    selectedSymbol: null,
    isConnected: false,
    memoryStats: {
      totalSymbols: 0,
      totalOrderbooks: 0,
      totalMetrics: 0,
      lastCleanup: Date.now(),
    },

    /**
     * Обновление данных стакана для конкретной пары.
     * ОПТИМИЗИРОВАНО: Храним только последний snapshot без истории.
     */
    updateOrderBook: (symbol, data) => {
      const now = Date.now();

      set((state) => {
        // Обновляем активность символа
        const newActivity = {
          ...state.symbolActivity,
          [symbol]: {
            lastUpdate: now,
            updateCount: (state.symbolActivity[symbol]?.updateCount || 0) + 1,
          },
        };

        // КРИТИЧНО: Заменяем старый orderbook новым (не храним историю)
        const newOrderbooks = {
          ...state.orderbooks,
          [symbol]: data,
        };

        // Проверяем лимит активных символов
        const totalSymbols = Object.keys(newOrderbooks).length;
        if (totalSymbols > MEMORY_CONFIG.MAX_ACTIVE_SYMBOLS) {
          console.warn(
            `[MarketStore] Превышен лимит активных символов: ${totalSymbols}/${MEMORY_CONFIG.MAX_ACTIVE_SYMBOLS}`
          );
        }

        return {
          orderbooks: newOrderbooks,
          symbolActivity: newActivity,
          memoryStats: {
            ...state.memoryStats,
            totalSymbols,
            totalOrderbooks: Object.keys(newOrderbooks).length,
          },
        };
      });
    },

    /**
     * Обновление метрик для конкретной пары.
     * ОПТИМИЗИРОВАНО: Храним только последние N метрик с автоочисткой.
     */
    updateMetrics: (symbol, data) => {
      const now = Date.now();

      set((state) => {
        // Получаем текущую историю метрик для символа
        const currentHistory = state.metricsHistory[symbol] || [];

        // Добавляем новую метрику с timestamp
        const newHistoryEntry: MetricsHistory = {
          metrics: data,
          timestamp: now,
        };

        // КРИТИЧНО: Ограничиваем размер истории
        const updatedHistory = [newHistoryEntry, ...currentHistory]
          .slice(0, MEMORY_CONFIG.MAX_METRICS_HISTORY);

        // Обновляем активность символа
        const newActivity = {
          ...state.symbolActivity,
          [symbol]: {
            lastUpdate: now,
            updateCount: (state.symbolActivity[symbol]?.updateCount || 0) + 1,
          },
        };

        // Считаем общее количество метрик
        const totalMetrics = Object.values({
          ...state.metricsHistory,
          [symbol]: updatedHistory,
        }).reduce((sum, history) => sum + history.length, 0);

        return {
          metricsHistory: {
            ...state.metricsHistory,
            [symbol]: updatedHistory,
          },
          currentMetrics: {
            ...state.currentMetrics,
            [symbol]: data,
          },
          symbolActivity: newActivity,
          memoryStats: {
            ...state.memoryStats,
            totalMetrics,
          },
        };
      });
    },

    /**
     * Установка выбранной пары для детального просмотра.
     */
    setSelectedSymbol: (symbol) => set({ selectedSymbol: symbol }),

    /**
     * Обновление статуса WebSocket подключения.
     */
    setConnected: (connected) => set({ isConnected: connected }),

    /**
     * Периодическая очистка памяти.
     * Удаляет данные для неактивных символов.
     */
    cleanupMemory: () => {
      const now = Date.now();

      set((state) => {
        const inactiveSymbols: string[] = [];

        // Находим неактивные символы
        Object.entries(state.symbolActivity).forEach(([symbol, activity]) => {
          if (now - activity.lastUpdate > MEMORY_CONFIG.INACTIVE_SYMBOL_TTL) {
            // Не удаляем выбранный символ
            if (symbol !== state.selectedSymbol) {
              inactiveSymbols.push(symbol);
            }
          }
        });

        if (inactiveSymbols.length === 0) {
          // Обновляем только timestamp очистки
          return {
            memoryStats: {
              ...state.memoryStats,
              lastCleanup: now,
            },
          };
        }

        console.info(
          `[MarketStore] Очистка памяти: удаляем данные для ${inactiveSymbols.length} неактивных символов`,
          inactiveSymbols
        );

        // Удаляем данные неактивных символов
        const newOrderbooks = { ...state.orderbooks };
        const newMetricsHistory = { ...state.metricsHistory };
        const newCurrentMetrics = { ...state.currentMetrics };
        const newSymbolActivity = { ...state.symbolActivity };

        inactiveSymbols.forEach((symbol) => {
          delete newOrderbooks[symbol];
          delete newMetricsHistory[symbol];
          delete newCurrentMetrics[symbol];
          delete newSymbolActivity[symbol];
        });

        // Пересчитываем статистику
        const totalMetrics = Object.values(newMetricsHistory).reduce(
          (sum, history) => sum + history.length,
          0
        );

        return {
          orderbooks: newOrderbooks,
          metricsHistory: newMetricsHistory,
          currentMetrics: newCurrentMetrics,
          symbolActivity: newSymbolActivity,
          memoryStats: {
            totalSymbols: Object.keys(newOrderbooks).length,
            totalOrderbooks: Object.keys(newOrderbooks).length,
            totalMetrics,
            lastCleanup: now,
          },
        };
      });
    },

    /**
     * Примерный расчет использования памяти (в байтах).
     * Для мониторинга и отладки.
     */
    getMemoryUsage: () => {
      const state = get();

      // Примерный размер одного orderbook (bid/ask по 50 уровней)
      const orderbookSize = 50 * 2 * (8 + 8) * 2; // 50 levels * 2 sides * 2 numbers * 8 bytes

      // Примерный размер одной метрики
      const metricsSize = 1024; // ~1KB на метрику

      const totalOrderbooksMemory =
        Object.keys(state.orderbooks).length * orderbookSize;

      const totalMetricsMemory =
        state.memoryStats.totalMetrics * metricsSize;

      return totalOrderbooksMemory + totalMetricsMemory;
    },

    /**
     * Сброс всех данных к начальному состоянию.
     */
    reset: () => {
      // Очищаем таймер
      // ИСПРАВЛЕНО: Используем window.clearInterval для browser environment
      if (cleanupIntervalId !== null) {
        window.clearInterval(cleanupIntervalId);
        cleanupIntervalId = null;
      }

      set({
        orderbooks: {},
        metricsHistory: {},
        currentMetrics: {},
        symbolActivity: {},
        selectedSymbol: null,
        isConnected: false,
        memoryStats: {
          totalSymbols: 0,
          totalOrderbooks: 0,
          totalMetrics: 0,
          lastCleanup: Date.now(),
        },
      });
    },
  };
});

// Хук для мониторинга памяти (для dev mode)
export const useMarketStoreMemoryMonitor = () => {
  const memoryStats = useMarketStore((state) => state.memoryStats);
  const getMemoryUsage = useMarketStore((state) => state.getMemoryUsage);

  return {
    ...memoryStats,
    estimatedMemoryUsage: getMemoryUsage(),
  };
};