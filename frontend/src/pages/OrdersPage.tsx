// frontend/src/pages/OrdersPage.tsx
/**
 * Страница управления ордерами.
 *
 * Функционал:
 * - Отображение списка открытых ордеров
 * - Фильтрация и сортировка
 * - Детальный просмотр с графиком
 * - Расчет PnL в реальном времени
 * - Закрытие ордеров с подтверждением
 */

import {useCallback, useEffect, useState} from 'react';
import { OrdersTable } from '../components/orders/OrdersTable';
import { OrderDetailModal } from '../components/orders/OrderDetailModal';
import { useOrdersStore } from '../store/ordersStore';
import { apiService } from '../services/api.service';
import { toast } from 'sonner';
import { FileText, TrendingUp, TrendingDown, Activity, AlertCircle, RefreshCw } from 'lucide-react';
import type { Order, OrderStatus, OrderSide } from '../types/orders.types';

/**
 * Компонент страницы ордеров.
 */
export function OrdersPage() {
  const {
    orders,
    selectedOrderDetail,
    isLoading,
    isClosing,
    filters,
    setOrders,
    setSelectedOrderDetail,
    setLoading,
    setClosing,
    setFilters,
    getActiveOrders,
    getOrdersStats,
  } = useOrdersStore();

  const [isRefreshing, setIsRefreshing] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  /**
   * Загрузка списка ордеров.
   */
  const loadOrders = useCallback(async () => {
  try {
    setLoading(true);
    console.log('[OrdersPage] Загрузка ордеров...');

    const response = await apiService.get('/api/trading/orders', {
      params: {
        status: 'active',
      },
    });

    if (response && response.orders) {
      setOrders(response.orders);
      console.log(`[OrdersPage] Загружено ${response.orders.length} ордеров`);
    }
  } catch (error: unknown) {
    console.error('[OrdersPage] Ошибка загрузки ордеров:', error);

    const errorMessage =
      error instanceof Error
        ? error.message
        : 'Ошибка загрузки ордеров';

    toast.error('Ошибка загрузки ордеров', {
      description: errorMessage,
    });
  } finally {
    setLoading(false);
  }
}, [setLoading, setOrders]);  // Добавить зависимости из store

  /**
   * Загрузка детальной информации об ордере.
   */
  const loadOrderDetail = async (order: Order) => {
    try {
      setDetailError(null);
      console.log(`[OrdersPage] Загрузка деталей для ордера ${order.order_id}`);

      const response = await apiService.get(`/api/trading/orders/${order.order_id}`);

      if (response && response.order) {
        setSelectedOrderDetail(response.order);
      }
    } catch (error: unknown) {
  console.error('[OrdersPage] Ошибка загрузки ордеров:', error);

  const errorMessage =
    error instanceof Error
      ? error.message
      : 'Ошибка загрузки ордеров';

  const detailMessage =
    typeof error === 'object' &&
    error !== null &&
    'response' in error &&
    typeof error.response === 'object' &&
    error.response !== null &&
    'data' in error.response &&
    typeof error.response.data === 'object' &&
    error.response.data !== null &&
    'detail' in error.response.data
      ? String(error.response.data.detail)
      : errorMessage;

  toast.error('Ошибка загрузки ордеров', {
    description: detailMessage,
  });
}
  };

  /**
   * Закрытие ордера.
   */
  const closeOrder = async (orderId: string) => {
    try {
      setClosing(true);
      console.log(`[OrdersPage] Закрытие ордера ${orderId}`);

      const response = await apiService.post(`/api/trading/orders/${orderId}/close`, {
        reason: 'manual_close',
      });

      if (response && response.success) {
        toast.success('Ордер успешно закрыт', {
          description: `PnL: ${response.final_pnl.toFixed(2)} USDT`,
        });

        // Обновляем список ордеров
        await loadOrders();

        // Закрываем модальное окно
        setSelectedOrderDetail(null);
      }
    } catch (error: unknown) {
  console.error('[OrdersPage] Ошибка загрузки ордеров:', error);

  const errorMessage =
    error instanceof Error
      ? error.message
      : 'Ошибка загрузки ордеров';

  const detailMessage =
    typeof error === 'object' &&
    error !== null &&
    'response' in error &&
    typeof error.response === 'object' &&
    error.response !== null &&
    'data' in error.response &&
    typeof error.response.data === 'object' &&
    error.response.data !== null &&
    'detail' in error.response.data
      ? String(error.response.data.detail)
      : errorMessage;

  toast.error('Ошибка загрузки ордеров', {
    description: detailMessage,
  });
}
  };

  /**
   * Обработка клика по ордеру.
   */
  const handleOrderClick = (order: Order) => {
    loadOrderDetail(order);
  };

  /**
   * Обработка закрытия модального окна.
   */
  const handleCloseModal = () => {
    setSelectedOrderDetail(null);
    setDetailError(null);
  };

  /**
   * Обновление списка ордеров.
   */
  const handleRefresh = async () => {
    setIsRefreshing(true);
    await loadOrders();
    setIsRefreshing(false);
    toast.success('Список ордеров обновлен');
  };

  /**
   * Начальная загрузка при монтировании.
   */
  useEffect(() => {
  loadOrders();

  const intervalId = setInterval(() => {
    loadOrders();
  }, 30 * 1000);

  return () => {
    clearInterval(intervalId);
  };
}, [loadOrders]);  // Теперь ESLint доволен

  /**
   * Получение статистики.
   */
  const stats = getOrdersStats();
  const activeOrders = getActiveOrders();

  /**
   * Рендер состояния загрузки.
   */
  if (isLoading && orders.length === 0) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold">Ордера</h1>

        <div className="bg-surface rounded-lg border border-gray-800 p-12 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-gray-400">Загрузка ордеров...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Заголовок */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <FileText className="h-8 w-8 text-primary" />
            Ордера
          </h1>
          <p className="text-gray-400 mt-1">
            Управление открытыми ордерами
          </p>
        </div>

        {/* Кнопка обновления */}
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          Обновить
        </button>
      </div>

      {/* Статистика */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Всего ордеров */}
        <div className="bg-surface border border-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm text-gray-400">Всего ордеров</div>
            <Activity className="h-5 w-5 text-gray-500" />
          </div>
          <div className="text-2xl font-bold text-white">{stats.total}</div>
        </div>

        {/* Активные */}
        <div className="bg-surface border border-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm text-gray-400">Активные</div>
            <TrendingUp className="h-5 w-5 text-blue-500" />
          </div>
          <div className="text-2xl font-bold text-blue-500">{stats.active}</div>
        </div>

        {/* Исполненные */}
        <div className="bg-surface border border-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm text-gray-400">Исполненные</div>
            <TrendingUp className="h-5 w-5 text-success" />
          </div>
          <div className="text-2xl font-bold text-success">{stats.filled}</div>
        </div>

        {/* Отмененные */}
        <div className="bg-surface border border-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm text-gray-400">Отмененные</div>
            <TrendingDown className="h-5 w-5 text-gray-500" />
          </div>
          <div className="text-2xl font-bold text-gray-500">{stats.cancelled}</div>
        </div>
      </div>

      {/* Фильтры */}
      <div className="bg-surface border border-gray-800 rounded-lg p-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Фильтр по символу */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">Торговая пара</label>
            <input
              type="text"
              placeholder="Например: BTCUSDT"
              value={filters.symbol || ''}
              onChange={(e) => setFilters({ symbol: e.target.value || undefined })}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            />
          </div>

          {/* Фильтр по стороне */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">Сторона</label>
            <select
              value={filters.side || ''}
              onChange={(e) => setFilters({ side: (e.target.value || undefined) as OrderSide | undefined })}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            >
              <option value="">Все</option>
              <option value="BUY">Покупка</option>
              <option value="SELL">Продажа</option>
            </select>
          </div>

          {/* Фильтр по статусу */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">Статус</label>
            <select
              value={filters.status || ''}
              onChange={(e) => setFilters({ status: (e.target.value || undefined) as OrderStatus | undefined })}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            >
              <option value="">Все</option>
              <option value="PENDING">Ожидает</option>
              <option value="PLACED">Размещен</option>
              <option value="PARTIALLY_FILLED">Частично</option>
              <option value="FILLED">Исполнен</option>
              <option value="CANCELLED">Отменен</option>
              <option value="REJECTED">Отклонен</option>
            </select>
          </div>

          {/* Фильтр по стратегии */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">Стратегия</label>
            <input
              type="text"
              placeholder="Название стратегии"
              value={filters.strategy || ''}
              onChange={(e) => setFilters({ strategy: e.target.value || undefined })}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            />
          </div>
        </div>
      </div>

      {/* Информационная панель */}
      {activeOrders.length === 0 && orders.length > 0 && (
        <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-sm font-semibold text-blue-500 mb-1">
                Нет активных ордеров
              </h3>
              <p className="text-sm text-gray-400">
                Все ордера исполнены или отменены. Для просмотра истории используйте фильтры.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Таблица ордеров */}
      <OrdersTable onOrderClick={handleOrderClick} />

      {/* Модальное окно детального просмотра */}
      {selectedOrderDetail && (
        <OrderDetailModal
          orderDetail={selectedOrderDetail}
          error={detailError}
          onClose={handleCloseModal}
          onCloseOrder={closeOrder}
          isClosing={isClosing}
        />
      )}

      {/* Дополнительная информация */}
      <div className="bg-gray-800/30 border border-gray-700 rounded-lg p-4">
        <div className="text-sm text-gray-400 space-y-2">
          <p>
            <strong className="text-white">Совет:</strong> Кликните на любой ордер для просмотра детальной информации, графика и расчета PnL в реальном времени.
          </p>
          <p>
            <strong className="text-white">Автообновление:</strong> Список ордеров автоматически обновляется каждые 30 секунд.
          </p>
        </div>
      </div>
    </div>
  );
}