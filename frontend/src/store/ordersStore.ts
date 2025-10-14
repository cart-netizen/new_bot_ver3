// frontend/src/store/ordersStore.ts
/**
 * Store для управления ордерами.
 *
 * Функционал:
 * - Хранение списка ордеров
 * - Фильтрация и сортировка
 * - Детальная информация об ордере
 * - Расчет PnL в реальном времени
 * - Интеграция с API для закрытия ордеров
 */

import { create } from 'zustand';
import type { Order, OrderDetail, OrderFilters, OrdersStats, OrderStatus, OrderSide } from '../types/orders.types';

/**
 * Параметры сортировки.
 */
export type OrderSortField = 'created_at' | 'symbol' | 'side' | 'quantity' | 'price' | 'current_pnl';
export type SortDirection = 'asc' | 'desc';

/**
 * Состояние Store для ордеров.
 */
interface OrdersState {
  // Данные ордеров
  orders: Order[];

  // Детальная информация о выбранном ордере
  selectedOrderDetail: OrderDetail | null;

  // Фильтры
  filters: OrderFilters;

  // Сортировка
  sortField: OrderSortField;
  sortDirection: SortDirection;

  // Статус загрузки
  isLoading: boolean;
  isClosing: boolean; // Закрытие ордера

  // Ошибки
  error: string | null;

  // Actions
  setOrders: (orders: Order[]) => void;
  addOrder: (order: Order) => void;
  updateOrder: (orderId: string, updates: Partial<Order>) => void;
  removeOrder: (orderId: string) => void;
  setSelectedOrderDetail: (detail: OrderDetail | null) => void;
  setFilters: (filters: Partial<OrderFilters>) => void;
  setSorting: (field: OrderSortField, direction: SortDirection) => void;
  setLoading: (loading: boolean) => void;
  setClosing: (closing: boolean) => void;
  setError: (error: string | null) => void;
  clearFilters: () => void;
  reset: () => void;

  // Getters
  getFilteredOrders: () => Order[];
  getSortedOrders: () => Order[];
  getActiveOrders: () => Order[];
  getOrderById: (orderId: string) => Order | null;
  getOrdersStats: () => OrdersStats;
}

/**
 * Фильтрация ордеров.
 */
function filterOrders(orders: Order[], filters: OrderFilters): Order[] {
  return orders.filter(order => {
    // Фильтр по символу
    if (filters.symbol && !order.symbol.toLowerCase().includes(filters.symbol.toLowerCase())) {
      return false;
    }

    // Фильтр по стороне
    if (filters.side && order.side !== filters.side) {
      return false;
    }

    // Фильтр по статусу
    if (filters.status && order.status !== filters.status) {
      return false;
    }

    // Фильтр по стратегии
    if (filters.strategy && order.strategy !== filters.strategy) {
      return false;
    }

    // Фильтр по дате (от)
    if (filters.date_from) {
      const orderDate = new Date(order.created_at);
      const fromDate = new Date(filters.date_from);
      if (orderDate < fromDate) {
        return false;
      }
    }

    // Фильтр по дате (до)
    if (filters.date_to) {
      const orderDate = new Date(order.created_at);
      const toDate = new Date(filters.date_to);
      if (orderDate > toDate) {
        return false;
      }
    }

    return true;
  });
}

/**
 * Сортировка ордеров.
 */
function sortOrders(
  orders: Order[],
  field: OrderSortField,
  direction: SortDirection
): Order[] {
  return [...orders].sort((a, b) => {
    let compareResult = 0;

    switch (field) {
      case 'created_at':
        compareResult = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
        break;
      case 'symbol':
        compareResult = a.symbol.localeCompare(b.symbol);
        break;
      case 'side':
        compareResult = a.side.localeCompare(b.side);
        break;
      case 'quantity':
        compareResult = a.quantity - b.quantity;
        break;
      case 'price':
        compareResult = (a.price || 0) - (b.price || 0);
        break;
      default:
        compareResult = 0;
    }

    return direction === 'asc' ? compareResult : -compareResult;
  });
}

/**
 * Проверка активности ордера.
 */
function isActiveOrder(order: Order): boolean {
  return ['PENDING', 'PLACED', 'PARTIALLY_FILLED'].includes(order.status);
}

/**
 * Zustand Store для ордеров.
 */
export const useOrdersStore = create<OrdersState>((set, get) => ({
  // Initial state
  orders: [],
  selectedOrderDetail: null,
  filters: {},
  sortField: 'created_at',
  sortDirection: 'desc',
  isLoading: false,
  isClosing: false,
  error: null,

  /**
   * Установка списка ордеров.
   */
  setOrders: (orders) => set({ orders, error: null }),

  /**
   * Добавление ордера.
   */
  addOrder: (order) => set((state) => ({
    orders: [order, ...state.orders],
  })),

  /**
   * Обновление ордера.
   */
  updateOrder: (orderId, updates) => set((state) => ({
    orders: state.orders.map(order =>
      order.order_id === orderId ? { ...order, ...updates } : order
    ),
  })),

  /**
   * Удаление ордера.
   */
  removeOrder: (orderId) => set((state) => ({
    orders: state.orders.filter(order => order.order_id !== orderId),
  })),

  /**
   * Установка детальной информации об ордере.
   */
  setSelectedOrderDetail: (detail) => set({ selectedOrderDetail: detail }),

  /**
   * Установка фильтров.
   */
  setFilters: (filters) => set((state) => ({
    filters: { ...state.filters, ...filters },
  })),

  /**
   * Установка параметров сортировки.
   */
  setSorting: (field, direction) => set({
    sortField: field,
    sortDirection: direction
  }),

  /**
   * Установка статуса загрузки.
   */
  setLoading: (loading) => set({ isLoading: loading }),

  /**
   * Установка статуса закрытия.
   */
  setClosing: (closing) => set({ isClosing: closing }),

  /**
   * Установка ошибки.
   */
  setError: (error) => set({ error }),

  /**
   * Очистка фильтров.
   */
  clearFilters: () => set({ filters: {} }),

  /**
   * Полный сброс состояния.
   */
  reset: () => set({
    orders: [],
    selectedOrderDetail: null,
    filters: {},
    sortField: 'created_at',
    sortDirection: 'desc',
    isLoading: false,
    isClosing: false,
    error: null,
  }),

  /**
   * Получение отфильтрованных ордеров.
   */
  getFilteredOrders: () => {
    const state = get();
    return filterOrders(state.orders, state.filters);
  },

  /**
   * Получение отсортированных ордеров.
   */
  getSortedOrders: () => {
    const state = get();
    const filtered = state.getFilteredOrders();
    return sortOrders(filtered, state.sortField, state.sortDirection);
  },

  /**
   * Получение только активных ордеров.
   */
  getActiveOrders: () => {
    const state = get();
    return state.orders.filter(isActiveOrder);
  },

  /**
   * Получение ордера по ID.
   */
  getOrderById: (orderId) => {
    const state = get();
    return state.orders.find(order => order.order_id === orderId) || null;
  },

  /**
   * Получение статистики по ордерам.
   */
  getOrdersStats: () => {
    const state = get();
    const orders = state.orders;

    const stats: OrdersStats = {
      total: orders.length,
      active: orders.filter(o => isActiveOrder(o)).length,
      filled: orders.filter(o => o.status === 'FILLED').length,
      cancelled: orders.filter(o => o.status === 'CANCELLED').length,
      total_volume: orders.reduce((sum, o) => sum + o.quantity * (o.price || 0), 0),
      total_pnl: 0, // Будет рассчитываться на backend
    };

    return stats;
  },
}));