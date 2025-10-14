// frontend/src/store/screenerStore.ts
/**
 * Store для управления данными скринера.
 *
 * Функционал:
 * - Хранение данных торговых пар с фильтрацией по volume > 4M USDT
 * - Расчет динамики по таймфреймам (1m, 3m, 5m, 15m)
 * - Сортировка по различным параметрам
 * - Оптимизированное управление памятью
 *
 * Архитектура:
 * - Использует MEMORY_CONFIG для предотвращения переполнения
 * - Автоматическая очистка старых данных
 * - WebSocket интеграция для real-time обновлений
 */

import { create } from 'zustand';

/**
 * Конфигурация управления памятью для скринера.
 */
const SCREENER_MEMORY_CONFIG = {
  // Максимальное количество пар в скринере
  MAX_SYMBOLS: 100,

  // Минимальный объем за 24ч для отображения (USDT)
  MIN_VOLUME_24H: 4_000_000,

  // Количество исторических снапшотов для расчета динамики
  MAX_PRICE_HISTORY: 100, // Для расчета изменений по таймфреймам

  // Интервал очистки неактивных данных (мс)
  CLEANUP_INTERVAL: 60 * 1000, // 1 минута

  // Время жизни неактивной пары (мс)
  INACTIVE_TTL: 10 * 60 * 1000, // 10 минут
} as const;

/**
 * Точка данных цены для расчета динамики.
 */
interface PricePoint {
  price: number;
  timestamp: number;
}

/**
 * Изменение цены по таймфрейму.
 */
interface PriceChange {
  timeframe: string; // '1m', '3m', '5m', '15m'
  change_percent: number;
  previous_price: number | null;
  current_price: number;
}

/**
 * Данные торговой пары в скринере.
 */
export interface ScreenerPairData {
  symbol: string;
  lastPrice: number;
  price24hPcnt: number; // Изменение за 24ч в процентах
  volume24h: number;    // Объем за 24ч в USDT
  highPrice24h: number;
  lowPrice24h: number;
  prevPrice24h: number;

  // Динамика по таймфреймам
  changes: {
    '1m': PriceChange | null;
    '3m': PriceChange | null;
    '5m': PriceChange | null;
    '15m': PriceChange | null;
  };

  // История цен для расчета динамики
  priceHistory: PricePoint[];

  // Метаданные
  lastUpdate: number;
  isActive: boolean;
}

/**
 * Параметры сортировки.
 */
export type SortField = 'symbol' | 'lastPrice' | 'price24hPcnt' | 'volume24h';
export type SortDirection = 'asc' | 'desc';

/**
 * Состояние Store для скринера.
 */
interface ScreenerState {
  // Данные пар (только те, что проходят фильтр по volume)
  pairs: Record<string, ScreenerPairData>;

  // Фильтрация и сортировка
  filterText: string;
  sortField: SortField;
  sortDirection: SortDirection;

  // Статус подключения
  isConnected: boolean;

  // Статистика памяти
  memoryStats: {
    totalPairs: number;
    activePairs: number;
    totalPricePoints: number;
    lastCleanup: number;
  };

  // Actions
  updatePairData: (symbol: string, data: Partial<ScreenerPairData>) => void;
  updatePairPrice: (symbol: string, price: number) => void;
  setFilterText: (text: string) => void;
  setSorting: (field: SortField, direction: SortDirection) => void;
  setConnected: (connected: boolean) => void;
  cleanupMemory: () => void;
  reset: () => void;

  // Selectors
  getFilteredPairs: () => ScreenerPairData[];
  getSortedPairs: () => ScreenerPairData[];
}

/**
 * Расчет изменения цены за таймфрейм.
 */
function calculatePriceChange(
  currentPrice: number,
  priceHistory: PricePoint[],
  timeframeMinutes: number
): PriceChange | null {
  const now = Date.now();
  const timeframeMs = timeframeMinutes * 60 * 1000;
  const targetTimestamp = now - timeframeMs;

  // Находим ближайшую цену к таргетному времени
  let previousPrice: number | null = null;
  let minTimeDiff = Infinity;

  for (const point of priceHistory) {
    const timeDiff = Math.abs(point.timestamp - targetTimestamp);

    // Допуск ±10% от таймфрейма
    if (timeDiff < timeframeMs * 0.1 && timeDiff < minTimeDiff) {
      minTimeDiff = timeDiff;
      previousPrice = point.price;
    }
  }

  if (previousPrice === null) {
    return null;
  }

  const changePercent = ((currentPrice - previousPrice) / previousPrice) * 100;

  return {
    timeframe: `${timeframeMinutes}m`,
    change_percent: changePercent,
    previous_price: previousPrice,
    current_price: currentPrice,
  };
}

/**
 * Zustand Store для скринера с оптимизированным управлением памятью.
 */
export const useScreenerStore = create<ScreenerState>((set, get) => {
  // Периодическая очистка памяти
  setInterval(() => {
    get().cleanupMemory();
  }, SCREENER_MEMORY_CONFIG.CLEANUP_INTERVAL);

  return {
    // Initial state
    pairs: {},
    filterText: '',
    sortField: 'volume24h',
    sortDirection: 'desc',
    isConnected: false,
    memoryStats: {
      totalPairs: 0,
      activePairs: 0,
      totalPricePoints: 0,
      lastCleanup: Date.now(),
    },

    /**
     * Обновление данных торговой пары.
     */
    updatePairData: (symbol, data) => set((state) => {
      const existingPair = state.pairs[symbol];
      const now = Date.now();

      // Создаем или обновляем данные пары
      const updatedPair: ScreenerPairData = {
        symbol,
        lastPrice: data.lastPrice ?? existingPair?.lastPrice ?? 0,
        price24hPcnt: data.price24hPcnt ?? existingPair?.price24hPcnt ?? 0,
        volume24h: data.volume24h ?? existingPair?.volume24h ?? 0,
        highPrice24h: data.highPrice24h ?? existingPair?.highPrice24h ?? 0,
        lowPrice24h: data.lowPrice24h ?? existingPair?.lowPrice24h ?? 0,
        prevPrice24h: data.prevPrice24h ?? existingPair?.prevPrice24h ?? 0,
        changes: existingPair?.changes ?? {
          '1m': null,
          '3m': null,
          '5m': null,
          '15m': null,
        },
        priceHistory: existingPair?.priceHistory ?? [],
        lastUpdate: now,
        isActive: true,
      };

      // Фильтр по минимальному объему
      if (updatedPair.volume24h < SCREENER_MEMORY_CONFIG.MIN_VOLUME_24H) {
        // Не добавляем пару, если объем слишком мал
        return state;
      }

      // Обновляем историю цен при изменении цены
      if (data.lastPrice && data.lastPrice !== existingPair?.lastPrice) {
        updatedPair.priceHistory = [
          ...updatedPair.priceHistory,
          { price: data.lastPrice, timestamp: now },
        ].slice(-SCREENER_MEMORY_CONFIG.MAX_PRICE_HISTORY); // Ограничиваем размер

        // Пересчитываем динамику по таймфреймам
        updatedPair.changes = {
          '1m': calculatePriceChange(data.lastPrice, updatedPair.priceHistory, 1),
          '3m': calculatePriceChange(data.lastPrice, updatedPair.priceHistory, 3),
          '5m': calculatePriceChange(data.lastPrice, updatedPair.priceHistory, 5),
          '15m': calculatePriceChange(data.lastPrice, updatedPair.priceHistory, 15),
        };
      }

      // Проверка лимита количества пар
      const currentPairsCount = Object.keys(state.pairs).length;
      if (currentPairsCount >= SCREENER_MEMORY_CONFIG.MAX_SYMBOLS && !existingPair) {
        console.warn(`[ScreenerStore] Достигнут лимит пар (${SCREENER_MEMORY_CONFIG.MAX_SYMBOLS})`);
        return state;
      }

      return {
        pairs: {
          ...state.pairs,
          [symbol]: updatedPair,
        },
      };
    }),

    /**
     * Быстрое обновление только цены (для WebSocket тиков).
     */
    updatePairPrice: (symbol, price) => set((state) => {
      const pair = state.pairs[symbol];
      if (!pair) return state;

      const now = Date.now();

      // Обновляем историю цен
      const updatedHistory = [
        ...pair.priceHistory,
        { price, timestamp: now },
      ].slice(-SCREENER_MEMORY_CONFIG.MAX_PRICE_HISTORY);

      // Пересчитываем динамику
      const updatedChanges = {
        '1m': calculatePriceChange(price, updatedHistory, 1),
        '3m': calculatePriceChange(price, updatedHistory, 3),
        '5m': calculatePriceChange(price, updatedHistory, 5),
        '15m': calculatePriceChange(price, updatedHistory, 15),
      };

      return {
        pairs: {
          ...state.pairs,
          [symbol]: {
            ...pair,
            lastPrice: price,
            priceHistory: updatedHistory,
            changes: updatedChanges,
            lastUpdate: now,
          },
        },
      };
    }),

    /**
     * Установка текста фильтра.
     */
    setFilterText: (text) => set({ filterText: text }),

    /**
     * Установка параметров сортировки.
     */
    setSorting: (field, direction) => set({
      sortField: field,
      sortDirection: direction
    }),

    /**
     * Установка статуса подключения.
     */
    setConnected: (connected) => set({ isConnected: connected }),

    /**
     * Очистка памяти от неактивных пар.
     */
    cleanupMemory: () => set((state) => {
      const now = Date.now();
      const updatedPairs: Record<string, ScreenerPairData> = {};

      let removedCount = 0;

      for (const [symbol, pair] of Object.entries(state.pairs)) {
        const inactiveTime = now - pair.lastUpdate;

        // Удаляем пару, если она неактивна слишком долго
        if (inactiveTime > SCREENER_MEMORY_CONFIG.INACTIVE_TTL) {
          removedCount++;
          continue;
        }

        // Удаляем пару, если объем упал ниже минимума
        if (pair.volume24h < SCREENER_MEMORY_CONFIG.MIN_VOLUME_24H) {
          removedCount++;
          continue;
        }

        updatedPairs[symbol] = pair;
      }

      if (removedCount > 0) {
        console.log(`[ScreenerStore] Очищено ${removedCount} неактивных пар`);
      }

      // Подсчет статистики
      const activePairs = Object.values(updatedPairs).filter(p => p.isActive).length;
      const totalPricePoints = Object.values(updatedPairs).reduce(
        (sum, p) => sum + p.priceHistory.length,
        0
      );

      return {
        pairs: updatedPairs,
        memoryStats: {
          totalPairs: Object.keys(updatedPairs).length,
          activePairs,
          totalPricePoints,
          lastCleanup: now,
        },
      };
    }),

    /**
     * Полный сброс состояния.
     */
    reset: () => set({
      pairs: {},
      filterText: '',
      sortField: 'volume24h',
      sortDirection: 'desc',
      isConnected: false,
      memoryStats: {
        totalPairs: 0,
        activePairs: 0,
        totalPricePoints: 0,
        lastCleanup: Date.now(),
      },
    }),

    /**
     * Получение отфильтрованных пар.
     */
    getFilteredPairs: () => {
      const state = get();
      const pairs = Object.values(state.pairs);

      if (!state.filterText) {
        return pairs;
      }

      const filterLower = state.filterText.toLowerCase();
      return pairs.filter(pair =>
        pair.symbol.toLowerCase().includes(filterLower)
      );
    },

    /**
     * Получение отсортированных пар.
     */
    getSortedPairs: () => {
      const state = get();
      const filteredPairs = state.getFilteredPairs();

      return [...filteredPairs].sort((a, b) => {
        const field = state.sortField;
        const direction = state.sortDirection === 'asc' ? 1 : -1;

        let compareResult = 0;

        switch (field) {
          case 'symbol':
            compareResult = a.symbol.localeCompare(b.symbol);
            break;
          case 'lastPrice':
            compareResult = a.lastPrice - b.lastPrice;
            break;
          case 'price24hPcnt':
            compareResult = a.price24hPcnt - b.price24hPcnt;
            break;
          case 'volume24h':
            compareResult = a.volume24h - b.volume24h;
            break;
        }

        return compareResult * direction;
      });
    },
  };
});