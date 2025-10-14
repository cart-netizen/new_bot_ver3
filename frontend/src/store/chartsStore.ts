// frontend/src/store/chartsStore.ts
/**
 * Store для управления отображаемыми графиками.
 *
 * Функционал:
 * - Управление списком выбранных пар для отображения
 * - Максимум 12 графиков одновременно (4 ряда по 3)
 * - Хранение исторических данных свечей
 * - Автоматическая очистка неактивных графиков
 * - Memory-optimized подход
 */

import { create } from 'zustand';

/**
 * Конфигурация управления памятью для графиков.
 */
const CHARTS_MEMORY_CONFIG = {
  // Максимальное количество одновременно отображаемых графиков
  MAX_CHARTS: 12, // 4 ряда по 3 графика

  // Максимальное количество свечей для каждого графика
  MAX_CANDLES_PER_CHART: 200, // ~16 часов на 5-минутном таймфрейме

  // Интервал обновления графиков (мс)
  UPDATE_INTERVAL: 15 * 1000, // 15 секунд

  // Таймфрейм свечей (минуты)
  TIMEFRAME: 5,

  // Интервал очистки старых данных (мс)
  CLEANUP_INTERVAL: 5 * 60 * 1000, // 5 минут
} as const;

/**
 * Свеча (OHLCV).
 */
export interface Candle {
  time: number; // timestamp в секундах
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Данные графика для торговой пары.
 */
interface ChartData {
  symbol: string;
  candles: Candle[];
  lastUpdate: number;
  isLoading: boolean;
  error: string | null;
}

/**
 * Состояние Store для графиков.
 */
interface ChartsState {
  // Выбранные пары для отображения (порядок важен)
  selectedSymbols: string[];

  // Данные свечей по парам
  chartsData: Record<string, ChartData>;

  // Интервал таймер для обновления
  updateIntervalId: number | null;

  // Статус загрузки
  isGlobalLoading: boolean;

  // Actions
  addSymbol: (symbol: string) => void;
  removeSymbol: (symbol: string) => void;
  toggleSymbol: (symbol: string) => void;
  clearAllSymbols: () => void;
  updateChartData: (symbol: string, candles: Candle[]) => void;
  setChartLoading: (symbol: string, loading: boolean) => void;
  setChartError: (symbol: string, error: string | null) => void;
  startAutoUpdate: (updateFunction: () => void) => void;
  stopAutoUpdate: () => void;
  cleanupOldData: () => void;
  reset: () => void;

  // Getters
  getChartData: (symbol: string) => ChartData | null;
  canAddMoreCharts: () => boolean;
}

/**
 * Zustand Store для графиков.
 */
export const useChartsStore = create<ChartsState>((set, get) => {
  // Периодическая очистка старых данных
  setInterval(() => {
    get().cleanupOldData();
  }, CHARTS_MEMORY_CONFIG.CLEANUP_INTERVAL);

  return {
    // Initial state
    selectedSymbols: [],
    chartsData: {},
    updateIntervalId: null,
    isGlobalLoading: false,

    /**
     * Добавление символа в список отображаемых графиков.
     */
    addSymbol: (symbol) => set((state) => {
      // Проверка лимита
      if (state.selectedSymbols.length >= CHARTS_MEMORY_CONFIG.MAX_CHARTS) {
        console.warn(`[ChartsStore] Достигнут лимит графиков (${CHARTS_MEMORY_CONFIG.MAX_CHARTS})`);
        return state;
      }

      // Проверка на дубликат
      if (state.selectedSymbols.includes(symbol)) {
        console.warn(`[ChartsStore] График для ${symbol} уже добавлен`);
        return state;
      }

      console.log(`[ChartsStore] Добавление графика для ${symbol}`);

      // Инициализируем данные графика
      const newChartData: ChartData = {
        symbol,
        candles: [],
        lastUpdate: Date.now(),
        isLoading: true,
        error: null,
      };

      return {
        selectedSymbols: [...state.selectedSymbols, symbol],
        chartsData: {
          ...state.chartsData,
          [symbol]: newChartData,
        },
      };
    }),

    /**
     * Удаление символа из списка отображаемых графиков.
     */
    removeSymbol: (symbol) => set((state) => {
      console.log(`[ChartsStore] Удаление графика для ${symbol}`);

      const updatedSymbols = state.selectedSymbols.filter(s => s !== symbol);
      const updatedChartsData = { ...state.chartsData };
      delete updatedChartsData[symbol];

      return {
        selectedSymbols: updatedSymbols,
        chartsData: updatedChartsData,
      };
    }),

    /**
     * Переключение символа (добавить/удалить).
     */
    toggleSymbol: (symbol) => {
      const state = get();

      if (state.selectedSymbols.includes(symbol)) {
        state.removeSymbol(symbol);
      } else {
        state.addSymbol(symbol);
      }
    },

    /**
     * Очистка всех графиков.
     */
    clearAllSymbols: () => set({
      selectedSymbols: [],
      chartsData: {},
    }),

    /**
     * Обновление данных свечей для графика.
     */
    updateChartData: (symbol, candles) => set((state) => {
      const existingData = state.chartsData[symbol];

      if (!existingData) {
        console.warn(`[ChartsStore] График для ${symbol} не найден`);
        return state;
      }

      // Ограничиваем количество свечей
      const limitedCandles = candles.slice(-CHARTS_MEMORY_CONFIG.MAX_CANDLES_PER_CHART);

      return {
        chartsData: {
          ...state.chartsData,
          [symbol]: {
            ...existingData,
            candles: limitedCandles,
            lastUpdate: Date.now(),
            isLoading: false,
            error: null,
          },
        },
      };
    }),

    /**
     * Установка статуса загрузки для графика.
     */
    setChartLoading: (symbol, loading) => set((state) => {
      const existingData = state.chartsData[symbol];

      if (!existingData) {
        return state;
      }

      return {
        chartsData: {
          ...state.chartsData,
          [symbol]: {
            ...existingData,
            isLoading: loading,
          },
        },
      };
    }),

    /**
     * Установка ошибки для графика.
     */
    setChartError: (symbol, error) => set((state) => {
      const existingData = state.chartsData[symbol];

      if (!existingData) {
        return state;
      }

      return {
        chartsData: {
          ...state.chartsData,
          [symbol]: {
            ...existingData,
            error,
            isLoading: false,
          },
        },
      };
    }),

    /**
     * Запуск автоматического обновления графиков.
     */
    startAutoUpdate: (updateFunction) => {
      const state = get();

      // Останавливаем предыдущий таймер, если есть
      if (state.updateIntervalId !== null) {
        clearInterval(state.updateIntervalId);
      }

      console.log(`[ChartsStore] Запуск автообновления (каждые ${CHARTS_MEMORY_CONFIG.UPDATE_INTERVAL / 1000}с)`);

      // Первое обновление сразу
      updateFunction();

      // Затем периодическое обновление
      const intervalId = window.setInterval(
        updateFunction,
        CHARTS_MEMORY_CONFIG.UPDATE_INTERVAL
      );

      set({ updateIntervalId: intervalId });
    },

    /**
     * Остановка автоматического обновления.
     */
    stopAutoUpdate: () => {
      const state = get();

      if (state.updateIntervalId !== null) {
        console.log('[ChartsStore] Остановка автообновления');
        clearInterval(state.updateIntervalId);
        set({ updateIntervalId: null });
      }
    },

    /**
     * Очистка старых данных.
     */
    cleanupOldData: () => set((state) => {
      const now = Date.now();
      const maxAge = 60 * 60 * 1000; // 1 час

      let removedCount = 0;
      const updatedChartsData = { ...state.chartsData };

      for (const [symbol, data] of Object.entries(updatedChartsData)) {
        const age = now - data.lastUpdate;

        if (age > maxAge) {
          delete updatedChartsData[symbol];
          removedCount++;
        }
      }

      if (removedCount > 0) {
        console.log(`[ChartsStore] Очищено ${removedCount} устаревших графиков`);
      }

      return { chartsData: updatedChartsData };
    }),

    /**
     * Полный сброс состояния.
     */
    reset: () => {
      const state = get();
      state.stopAutoUpdate();

      set({
        selectedSymbols: [],
        chartsData: {},
        updateIntervalId: null,
        isGlobalLoading: false,
      });
    },

    /**
     * Получение данных графика.
     */
    getChartData: (symbol) => {
      const state = get();
      return state.chartsData[symbol] || null;
    },

    /**
     * Проверка возможности добавления графика.
     */
    canAddMoreCharts: () => {
      const state = get();
      return state.selectedSymbols.length < CHARTS_MEMORY_CONFIG.MAX_CHARTS;
    },
  };
});

// Экспорт конфигурации для использования в компонентах
export { CHARTS_MEMORY_CONFIG };