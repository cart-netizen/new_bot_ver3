// frontend/src/components/orders/OrdersTable.tsx
/**
 * Таблица открытых ордеров.
 *
 * Функционал:
 * - Отображение списка ордеров
 * - Сортировка по колонкам
 * - Фильтрация
 * - Клик по ордеру для детального просмотра
 * - Цветовая индикация статусов
 */

import { useState } from 'react';
import { useOrdersStore, type OrderSortField, type SortDirection } from '../../store/ordersStore';
import type { Order, OrderStatus, OrderSide } from '../../types/orders.types';
import { ArrowUpDown, ArrowUp, ArrowDown, Eye, TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '../../utils/helpers';

/**
 * Форматирование даты и времени.
 */
function formatDateTime(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Форматирование цены.
 */
function formatPrice(price: number | null): string {
  if (price === null) return '—';

  if (price >= 1000) {
    return `$${price.toFixed(2)}`;
  }
  if (price >= 1) {
    return `$${price.toFixed(4)}`;
  }
  return `$${price.toFixed(6)}`;
}

/**
 * Форматирование количества.
 */
function formatQuantity(quantity: number): string {
  return quantity.toFixed(4);
}

/**
 * Получение цвета для статуса.
 */
function getStatusColor(status: OrderStatus): string {
  switch (status) {
    case 'PENDING':
      return 'text-yellow-500 bg-yellow-500/10';
    case 'PLACED':
      return 'text-blue-500 bg-blue-500/10';
    case 'PARTIALLY_FILLED':
      return 'text-cyan-500 bg-cyan-500/10';
    case 'FILLED':
      return 'text-success bg-success/10';
    case 'CANCELLED':
      return 'text-gray-500 bg-gray-500/10';
    case 'REJECTED':
      return 'text-destructive bg-destructive/10';
    default:
      return 'text-gray-500 bg-gray-500/10';
  }
}

/**
 * Получение текста статуса на русском.
 */
function getStatusText(status: OrderStatus): string {
  const statusMap: Record<OrderStatus, string> = {
    'PENDING': 'Ожидает',
    'PLACED': 'Размещен',
    'PARTIALLY_FILLED': 'Частично',
    'FILLED': 'Исполнен',
    'CANCELLED': 'Отменен',
    'REJECTED': 'Отклонен',
  };

  return statusMap[status] || status;
}

/**
 * Получение текста стороны на русском.
 */
function getSideText(side: OrderSide): string {
  return side === 'BUY' ? 'Покупка' : 'Продажа';
}

/**
 * Компонент заголовка колонки с сортировкой.
 */
interface SortableHeaderProps {
  field: OrderSortField;
  label: string;
  currentField: OrderSortField;
  currentDirection: SortDirection;
  onSort: (field: OrderSortField) => void;
}

function SortableHeader({
  field,
  label,
  currentField,
  currentDirection,
  onSort
}: SortableHeaderProps) {
  const isActive = currentField === field;

  return (
    <th
      className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:text-white transition-colors select-none"
      onClick={() => onSort(field)}
    >
      <div className="flex items-center gap-2">
        <span>{label}</span>
        {isActive ? (
          currentDirection === 'asc' ? (
            <ArrowUp className="h-4 w-4 text-primary" />
          ) : (
            <ArrowDown className="h-4 w-4 text-primary" />
          )
        ) : (
          <ArrowUpDown className="h-4 w-4 opacity-50" />
        )}
      </div>
    </th>
  );
}

/**
 * Props компонента.
 */
interface OrdersTableProps {
  /**
   * Callback при клике на ордер для детального просмотра.
   */
  onOrderClick: (order: Order) => void;
}

/**
 * Компонент таблицы ордеров.
 */
export function OrdersTable({ onOrderClick }: OrdersTableProps) {
  const {
    sortField,
    sortDirection,
    setSorting,
    getSortedOrders,
  } = useOrdersStore();

  const [hoveredOrderId, setHoveredOrderId] = useState<string | null>(null);

  /**
   * Обработка клика по заголовку для сортировки.
   */
  const handleSort = (field: OrderSortField) => {
    if (sortField === field) {
      setSorting(field, sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSorting(field, 'desc');
    }
  };

  const orders = getSortedOrders();

  /**
   * Рендер пустого состояния.
   */
  if (orders.length === 0) {
    return (
      <div className="bg-surface rounded-lg border border-gray-800 p-12 text-center">
        <div className="text-gray-500 mb-2">
          <Eye className="h-12 w-12 mx-auto mb-3 opacity-50" />
        </div>
        <h3 className="text-lg font-semibold text-white mb-2">
          Нет открытых ордеров
        </h3>
        <p className="text-sm text-gray-400">
          Открытые ордера появятся здесь после их создания
        </p>
      </div>
    );
  }

  /**
   * Рендер таблицы.
   */
  return (
    <div className="bg-surface rounded-lg border border-gray-800 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-900">
            <tr>
              <SortableHeader
                field="created_at"
                label="Дата/Время"
                currentField={sortField}
                currentDirection={sortDirection}
                onSort={handleSort}
              />
              <SortableHeader
                field="symbol"
                label="Пара"
                currentField={sortField}
                currentDirection={sortDirection}
                onSort={handleSort}
              />
              <SortableHeader
                field="side"
                label="Сторона"
                currentField={sortField}
                currentDirection={sortDirection}
                onSort={handleSort}
              />
              <SortableHeader
                field="quantity"
                label="Количество"
                currentField={sortField}
                currentDirection={sortDirection}
                onSort={handleSort}
              />
              <SortableHeader
                field="price"
                label="Цена входа"
                currentField={sortField}
                currentDirection={sortDirection}
                onSort={handleSort}
              />
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                TP / SL
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                Статус
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase tracking-wider">
                Действия
              </th>
            </tr>
          </thead>

          <tbody className="divide-y divide-gray-800">
            {orders.map((order) => {
              const isHovered = hoveredOrderId === order.order_id;
              const isBuy = order.side === 'BUY';

              return (
                <tr
                  key={order.order_id}
                  className={cn(
                    'hover:bg-gray-800/50 transition-colors cursor-pointer',
                    isHovered && 'bg-gray-800/50'
                  )}
                  onMouseEnter={() => setHoveredOrderId(order.order_id)}
                  onMouseLeave={() => setHoveredOrderId(null)}
                  onClick={() => onOrderClick(order)}
                >
                  {/* Дата/Время */}
                  <td className="px-4 py-3 text-sm text-gray-300 whitespace-nowrap">
                    {formatDateTime(order.created_at)}
                  </td>

                  {/* Символ */}
                  <td className="px-4 py-3 text-sm font-medium text-white whitespace-nowrap">
                    {order.symbol}
                  </td>

                  {/* Сторона */}
                  <td className="px-4 py-3 text-sm whitespace-nowrap">
                    <div className={cn(
                      'flex items-center gap-1',
                      isBuy ? 'text-success' : 'text-destructive'
                    )}>
                      {isBuy ? (
                        <TrendingUp className="h-3 w-3" />
                      ) : (
                        <TrendingDown className="h-3 w-3" />
                      )}
                      <span className="font-medium">{getSideText(order.side)}</span>
                    </div>
                  </td>

                  {/* Количество */}
                  <td className="px-4 py-3 text-sm text-gray-300 whitespace-nowrap">
                    {formatQuantity(order.quantity)}
                  </td>

                  {/* Цена входа */}
                  <td className="px-4 py-3 text-sm text-gray-300 whitespace-nowrap">
                    {formatPrice(order.price)}
                  </td>

                  {/* TP / SL */}
                  <td className="px-4 py-3 text-xs whitespace-nowrap">
                    <div className="space-y-1">
                      <div className="text-success">
                        TP: {formatPrice(order.take_profit)}
                      </div>
                      <div className="text-destructive">
                        SL: {formatPrice(order.stop_loss)}
                      </div>
                    </div>
                  </td>

                  {/* Статус */}
                  <td className="px-4 py-3 whitespace-nowrap">
                    <span className={cn(
                      'px-2 py-1 rounded-full text-xs font-medium',
                      getStatusColor(order.status)
                    )}>
                      {getStatusText(order.status)}
                    </span>
                  </td>

                  {/* Действия */}
                  <td className="px-4 py-3 text-center whitespace-nowrap">
                    <button
                      className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                      onClick={(e) => {
                        e.stopPropagation();
                        onOrderClick(order);
                      }}
                      title="Подробнее"
                    >
                      <Eye className="h-4 w-4 text-gray-400 hover:text-white" />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Футер со статистикой */}
      <div className="px-4 py-3 border-t border-gray-800 bg-gray-900/50 flex items-center justify-between">
        <div className="text-sm text-gray-400">
          Всего ордеров: <span className="font-semibold text-white">{orders.length}</span>
        </div>
        <div className="text-xs text-gray-500">
          Кликните на ордер для подробной информации
        </div>
      </div>
    </div>
  );
}