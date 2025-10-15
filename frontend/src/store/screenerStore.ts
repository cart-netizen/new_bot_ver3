// frontend/src/store/screenerStore.ts
/**
 * Store для управления данными скринера.
 *
 * Функционал:
 * - Хранение данных торговых пар с фильтрацией по volume > 4M USDT
 * - Расчет динамики по таймфреймам (1m, 3m, 5m, 15m)
 * - Сортировка по различным параметрам
 * - Оптимизированное управление памятью
 * - REST API fallback для начальной загрузки
 *
 * Обновлено: Добавлена загрузка через REST API
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
  MAX_PRICE_HISTORY: 100,

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
 * Статистика памяти скринера.
 */
interface ScreenerMemoryStats {
  totalPairs: number;
  activePairs: number;
  totalPricePoints: number;
  lastCleanup: number;
}

/**
 * Формат данных пары из REST API.
 */
interface ScreenerPairApiResponse {
  symbol: string;
  lastPrice: number;
  price24hPcnt: number;
  volume24h: number;
  highPrice24h: number;
  lowPrice24h: number;
  prevPrice24h: number;
  turnover24h?: number;
}

/**
 * Формат ответа REST API /api/screener/pairs.
 */
interface ScreenerApiResponse {
  pairs: ScreenerPairApiResponse[];
  total: number;
  timestamp: number;
  min_volume: number;
}

/**
 * Состояние Store для скринера.
 */
interface ScreenerStore {
  // Данные
  pairs: Record<string, ScreenerPairData>;
  isConnected: boolean;
  isLoading: boolean;

  // Сортировка
  sortField: SortField;
  sortDirection: SortDirection;

  // Статистика памяти
  memoryStats: ScreenerMemoryStats;

  // Методы управления данными
  updatePairData: (symbol: string, data: Partial<ScreenerPairData>) => void;
  updatePairPrice: (symbol: string, price: number) => void;
  removePair: (symbol: string) => void;
  clearAllPairs: () => void;

  // Методы управления состоянием
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;

  // Сортировка
  setSorting: (field: SortField, direction: SortDirection) => void;

  // Получение отсортированных данных
  getSortedPairs: () => ScreenerPairData[];

  // Управление памятью
  cleanupMemory: () => void;

  // REST API загрузка
  loadInitialData: () => Promise<void>;
}

/**
 * Расчет изменения цены по таймфрейму.
 */
function calculateTimeframeChange(
  priceHistory: PricePoint[],
  currentPrice: number,
  timeframeMinutes: number
): PriceChange | null {
  if (priceHistory.length === 0) return null;

  const now = Date.now();
  const targetTime = now - timeframeMinutes * 60 * 1000;

  // Ищем ближайшую цену к целевому времени
  let closestPoint: PricePoint | null = null;
  let minDiff = Infinity;

  for (const point of priceHistory) {
    const diff = Math.abs(point.timestamp - targetTime);
    if (diff < minDiff) {
      minDiff = diff;
      closestPoint = point;
    }
  }

  if (!closestPoint) return null;

  const changePercent = ((currentPrice - closestPoint.price) / closestPoint.price) * 100;

  return {
    timeframe: `${timeframeMinutes}m`,
    change_percent: changePercent,
    previous_price: closestPoint.price,
    current_price: currentPrice,
  };
}

/**
 * Store для управления данными скринера.
 */
export const useScreenerStore = create<ScreenerStore>((set, get) => ({
  // ==================== НАЧАЛЬНОЕ СОСТОЯНИЕ ====================
  pairs: {},
  isConnected: false,
  isLoading: false,
  sortField: 'volume24h',
  sortDirection: 'desc',
  memoryStats: {
    totalPairs: 0,
    activePairs: 0,
    totalPricePoints: 0,
    lastCleanup: Date.now(),
  },

  // ==================== УПРАВЛЕНИЕ ДАННЫМИ ====================

  /**
   * Обновление данных торговой пары.
   */
  updatePairData: (symbol: string, data: Partial<ScreenerPairData>) => {
    const state = get();
    const now = Date.now();

    // Проверка лимита пар
    if (!state.pairs[symbol] && Object.keys(state.pairs).length >= SCREENER_MEMORY_CONFIG.MAX_SYMBOLS) {
      console.warn(`[ScreenerStore] Достигнут лимит пар (${SCREENER_MEMORY_CONFIG.MAX_SYMBOLS})`);
      return;
    }

    // Фильтр по объему
    if (data.volume24h !== undefined && data.volume24h < SCREENER_MEMORY_CONFIG.MIN_VOLUME_24H) {
      // Удаляем если объем стал меньше минимума
      if (state.pairs[symbol]) {
        const newPairs = { ...state.pairs };
        delete newPairs[symbol];
        set({ pairs: newPairs });
      }
      return;
    }

    const existingPair = state.pairs[symbol];

    // Обновляем или создаем пару
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

    // Добавляем точку в историю цен (если цена обновилась)
    if (data.lastPrice !== undefined && data.lastPrice > 0) {
      updatedPair.priceHistory.push({
        price: data.lastPrice,
        timestamp: now,
      });

      // Ограничиваем размер истории
      if (updatedPair.priceHistory.length > SCREENER_MEMORY_CONFIG.MAX_PRICE_HISTORY) {
        updatedPair.priceHistory = updatedPair.priceHistory.slice(-SCREENER_MEMORY_CONFIG.MAX_PRICE_HISTORY);
      }

      // Рассчитываем изменения по таймфреймам
      updatedPair.changes = {
        '1m': calculateTimeframeChange(updatedPair.priceHistory, updatedPair.lastPrice, 1),
        '3m': calculateTimeframeChange(updatedPair.priceHistory, updatedPair.lastPrice, 3),
        '5m': calculateTimeframeChange(updatedPair.priceHistory, updatedPair.lastPrice, 5),
        '15m': calculateTimeframeChange(updatedPair.priceHistory, updatedPair.lastPrice, 15),
      };
    }

    // Обновляем store
    set({
      pairs: {
        ...state.pairs,
        [symbol]: updatedPair,
      },
    });
  },

  /**
   * Обновление только цены пары (быстрое обновление).
   */
  updatePairPrice: (symbol: string, price: number) => {
    const state = get();
    const pair = state.pairs[symbol];

    if (!pair) return;

    get().updatePairData(symbol, { lastPrice: price });
  },

  /**
   * Удаление пары.
   */
  removePair: (symbol: string) => {
    const state = get();
    const newPairs = { ...state.pairs };
    delete newPairs[symbol];
    set({ pairs: newPairs });
  },

  /**
   * Очистка всех пар.
   */
  clearAllPairs: () => {
    set({ pairs: {} });
  },

  // ==================== УПРАВЛЕНИЕ СОСТОЯНИЕМ ====================

  /**
   * Установка статуса подключения.
   */
  setConnected: (connected: boolean) => {
    set({ isConnected: connected });
  },

  /**
   * Установка статуса загрузки.
   */
  setLoading: (loading: boolean) => {
    set({ isLoading: loading });
  },

  // ==================== СОРТИРОВКА ====================

  /**
   * Установка параметров сортировки.
   */
  setSorting: (field: SortField, direction: SortDirection) => {
    set({ sortField: field, sortDirection: direction });
  },

  /**
   * Получение отсортированного списка пар.
   */
  getSortedPairs: () => {
    const state = get();
    const pairsArray = Object.values(state.pairs);

    return pairsArray.sort((a, b) => {
      let aValue: number | string;
      let bValue: number | string;

      switch (state.sortField) {
        case 'symbol':
          aValue = a.symbol;
          bValue = b.symbol;
          break;
        case 'lastPrice':
          aValue = a.lastPrice;
          bValue = b.lastPrice;
          break;
        case 'price24hPcnt':
          aValue = a.price24hPcnt;
          bValue = b.price24hPcnt;
          break;
        case 'volume24h':
          aValue = a.volume24h;
          bValue = b.volume24h;
          break;
        default:
          aValue = 0;
          bValue = 0;
      }

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return state.sortDirection === 'asc'
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }

      return state.sortDirection === 'asc'
        ? (aValue as number) - (bValue as number)
        : (bValue as number) - (aValue as number);
    });
  },

  // ==================== УПРАВЛЕНИЕ ПАМЯТЬЮ ====================

  /**
   * Очистка неактивных пар и оптимизация памяти.
   */
  cleanupMemory: () => {
    const state = get();
    const now = Date.now();

    console.log('[ScreenerStore] Running memory cleanup...');

    // Находим неактивные пары
    const activePairs: Record<string, ScreenerPairData> = {};
    let removedCount = 0;

    Object.entries(state.pairs).forEach(([symbol, pair]) => {
      const timeSinceUpdate = now - pair.lastUpdate;

      if (timeSinceUpdate < SCREENER_MEMORY_CONFIG.INACTIVE_TTL) {
        activePairs[symbol] = pair;
      } else {
        removedCount++;
      }
    });

    // Подсчитываем статистику
    const totalPricePoints = Object.values(activePairs).reduce(
      (sum, pair) => sum + pair.priceHistory.length,
      0
    );

    const memoryStats: ScreenerMemoryStats = {
      totalPairs: Object.keys(activePairs).length,
      activePairs: Object.values(activePairs).filter(p => p.isActive).length,
      totalPricePoints,
      lastCleanup: now,
    };

    if (removedCount > 0) {
      console.log(`[ScreenerStore] Removed ${removedCount} inactive pairs`);
    }

    console.log(`[ScreenerStore] Memory stats:`, memoryStats);

    set({
      pairs: activePairs,
      memoryStats,
    });
  },

  // ==================== REST API ЗАГРУЗКА ====================

  /**
   * Загрузка начальных данных через REST API.
   *
   * Используется как fallback при старте приложения
   * или если WebSocket медленно подключается.
   */
  loadInitialData: async () => {
    const state = get();

    console.log('[ScreenerStore] Loading initial data via REST API...');
    state.setLoading(true);

    try {
      // Получаем токен
      const token = localStorage.getItem('auth_token');

      if (!token) {
        console.warn('[ScreenerStore] No auth token found, skipping initial load');
        state.setLoading(false);
        return;
      }

      // Запрос к API
      const response = await fetch('http://localhost:8000/api/screener/pairs', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ScreenerApiResponse = await response.json();

      console.log(`[ScreenerStore] Loaded ${data.pairs?.length || 0} pairs via REST`);

      // Обновляем store
      if (data.pairs && Array.isArray(data.pairs)) {
        data.pairs.forEach((pair: ScreenerPairApiResponse) => {
          state.updatePairData(pair.symbol, {
            lastPrice: pair.lastPrice,
            volume24h: pair.volume24h,
            price24hPcnt: pair.price24hPcnt,
            highPrice24h: pair.highPrice24h,
            lowPrice24h: pair.lowPrice24h,
            prevPrice24h: pair.prevPrice24h,
          });
        });

        // Устанавливаем статус подключения
        state.setConnected(true);

        console.log('[ScreenerStore] Initial data loaded successfully');
      }

    } catch (error) {
      console.error('[ScreenerStore] Failed to load initial data:', error);

      // НЕ устанавливаем connected=false, чтобы дать шанс WebSocket
      // state.setConnected(false);
    } finally {
      state.setLoading(false);
    }
  },
}));