/**
 * Страница ордеров и позиций.
 * Отображает открытые позиции и ордера с биржи Bybit.
 * frontend/src/pages/OrdersPage.tsx
 */

import { useState, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { positionsApi } from '../api/positions.api';
import type { Position, OpenOrder } from '../types/position.types';

/**
 * Форматирование числа с разделителями.
 */
const formatNumber = (num: number, decimals: number = 2): string => {
  return num.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
};

/**
 * Форматирование цены.
 */
const formatPrice = (price: number): string => {
  if (price >= 1000) return formatNumber(price, 2);
  if (price >= 1) return formatNumber(price, 4);
  return formatNumber(price, 6);
};

/**
 * Форматирование времени.
 */
const formatTime = (timestamp: string): string => {
  if (!timestamp) return '-';
  const date = new Date(parseInt(timestamp));
  return date.toLocaleString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
};

/**
 * Компонент модального окна закрытия позиции.
 */
interface CloseModalProps {
  position: Position;
  onClose: () => void;
  onConfirm: (percent: 25 | 50 | 75 | 100) => void;
  isLoading: boolean;
}

function ClosePositionModal({ position, onClose, onConfirm, isLoading }: CloseModalProps) {
  const [selectedPercent, setSelectedPercent] = useState<25 | 50 | 75 | 100>(100);

  const percentOptions: (25 | 50 | 75 | 100)[] = [25, 50, 75, 100];

  const closeSize = position.size * selectedPercent / 100;
  const estimatedPnl = position.unrealisedPnl * selectedPercent / 100;

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-surface border border-gray-700 rounded-xl p-6 w-full max-w-md mx-4">
        <h3 className="text-xl font-bold mb-4">Закрыть позицию</h3>

        {/* Информация о позиции */}
        <div className="bg-gray-800/50 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-400">Символ</span>
            <span className="font-mono font-bold">{position.symbol}</span>
          </div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-400">Направление</span>
            <span className={position.side === 'Buy' ? 'text-green-400' : 'text-red-400'}>
              {position.side === 'Buy' ? 'LONG' : 'SHORT'}
            </span>
          </div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-400">Размер</span>
            <span className="font-mono">{formatNumber(position.size, 4)}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400">Текущий P&L</span>
            <span className={position.unrealisedPnl >= 0 ? 'text-green-400' : 'text-red-400'}>
              {position.unrealisedPnl >= 0 ? '+' : ''}{formatNumber(position.unrealisedPnl)} USDT
            </span>
          </div>
        </div>

        {/* Выбор процента */}
        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-2">Закрыть процент позиции</label>
          <div className="grid grid-cols-4 gap-2">
            {percentOptions.map((percent) => (
              <button
                key={percent}
                onClick={() => setSelectedPercent(percent)}
                className={`py-2 px-3 rounded-lg font-bold transition-colors ${
                  selectedPercent === percent
                    ? 'bg-primary text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {percent}%
              </button>
            ))}
          </div>
        </div>

        {/* Предварительный расчёт */}
        <div className="bg-gray-800/50 rounded-lg p-4 mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-400">Закрываемый размер</span>
            <span className="font-mono">{formatNumber(closeSize, 4)}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400">Ожидаемый P&L</span>
            <span className={estimatedPnl >= 0 ? 'text-green-400' : 'text-red-400'}>
              {estimatedPnl >= 0 ? '+' : ''}{formatNumber(estimatedPnl)} USDT
            </span>
          </div>
        </div>

        {/* Кнопки */}
        <div className="flex gap-3">
          <button
            onClick={onClose}
            disabled={isLoading}
            className="flex-1 py-3 px-4 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            Отмена
          </button>
          <button
            onClick={() => onConfirm(selectedPercent)}
            disabled={isLoading}
            className="flex-1 py-3 px-4 bg-red-600 hover:bg-red-500 rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <span className="animate-spin">⏳</span>
                Закрытие...
              </>
            ) : (
              `Закрыть ${selectedPercent}%`
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Компонент карточки позиции.
 */
interface PositionCardProps {
  position: Position;
  onCloseClick: (position: Position) => void;
}

function PositionCard({ position, onCloseClick }: PositionCardProps) {
  const isLong = position.side === 'Buy';
  const isProfitable = position.unrealisedPnl >= 0;

  return (
    <div className="bg-surface border border-gray-800 rounded-xl p-4 hover:border-gray-700 transition-colors">
      {/* Заголовок */}
      <div className="flex justify-between items-start mb-4">
        <div className="flex items-center gap-3">
          <span className="text-xl font-bold font-mono">{position.symbol}</span>
          <span className={`px-2 py-1 rounded text-xs font-bold ${
            isLong ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
          }`}>
            {isLong ? 'LONG' : 'SHORT'} {position.leverage}x
          </span>
        </div>
        <div className={`text-right ${isProfitable ? 'text-green-400' : 'text-red-400'}`}>
          <div className="text-lg font-bold">
            {isProfitable ? '+' : ''}{formatNumber(position.unrealisedPnl)} USDT
          </div>
          <div className="text-sm">
            ROE: {isProfitable ? '+' : ''}{formatNumber(position.roePercent)}%
          </div>
        </div>
      </div>

      {/* Основная информация */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div>
          <div className="text-xs text-gray-500 mb-1">Размер</div>
          <div className="font-mono">{formatNumber(position.size, 4)}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500 mb-1">Цена входа</div>
          <div className="font-mono">{formatPrice(position.avgPrice)}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500 mb-1">Марк. цена</div>
          <div className="font-mono">{formatPrice(position.markPrice)}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500 mb-1">Ликвидация</div>
          <div className="font-mono text-orange-400">
            {position.liqPrice > 0 ? formatPrice(position.liqPrice) : '-'}
          </div>
        </div>
      </div>

      {/* Маржа и TP/SL */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div>
          <div className="text-xs text-gray-500 mb-1">Маржа (IM)</div>
          <div className="font-mono">{formatNumber(position.positionIM)} USDT</div>
        </div>
        <div>
          <div className="text-xs text-gray-500 mb-1">Стоимость</div>
          <div className="font-mono">{formatNumber(position.positionValue)} USDT</div>
        </div>
        <div>
          <div className="text-xs text-gray-500 mb-1">Take Profit</div>
          <div className="font-mono text-green-400">
            {position.takeProfit ? formatPrice(parseFloat(position.takeProfit)) : '-'}
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-500 mb-1">Stop Loss</div>
          <div className="font-mono text-red-400">
            {position.stopLoss ? formatPrice(parseFloat(position.stopLoss)) : '-'}
          </div>
        </div>
      </div>

      {/* Кнопка закрытия */}
      <button
        onClick={() => onCloseClick(position)}
        className="w-full py-2 px-4 bg-red-600/20 hover:bg-red-600/40 border border-red-600/50 rounded-lg text-red-400 font-medium transition-colors"
      >
        Закрыть позицию
      </button>
    </div>
  );
}

/**
 * Компонент таблицы ордеров.
 */
interface OrdersTableProps {
  orders: OpenOrder[];
  onCancelOrder: (symbol: string, orderId: string) => void;
  isCancelling: boolean;
}

function OrdersTable({ orders, onCancelOrder, isCancelling }: OrdersTableProps) {
  if (orders.length === 0) {
    return (
      <div className="bg-surface border border-gray-800 rounded-xl p-8 text-center">
        <p className="text-gray-400">Нет открытых ордеров</p>
      </div>
    );
  }

  return (
    <div className="bg-surface border border-gray-800 rounded-xl overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-800 bg-gray-900/50">
              <th className="text-left py-3 px-4 text-xs text-gray-500 font-medium">Символ</th>
              <th className="text-left py-3 px-4 text-xs text-gray-500 font-medium">Тип</th>
              <th className="text-left py-3 px-4 text-xs text-gray-500 font-medium">Сторона</th>
              <th className="text-right py-3 px-4 text-xs text-gray-500 font-medium">Цена</th>
              <th className="text-right py-3 px-4 text-xs text-gray-500 font-medium">Кол-во</th>
              <th className="text-right py-3 px-4 text-xs text-gray-500 font-medium">Исполнено</th>
              <th className="text-left py-3 px-4 text-xs text-gray-500 font-medium">Статус</th>
              <th className="text-left py-3 px-4 text-xs text-gray-500 font-medium">Время</th>
              <th className="text-center py-3 px-4 text-xs text-gray-500 font-medium">Действие</th>
            </tr>
          </thead>
          <tbody>
            {orders.map((order) => (
              <tr key={order.orderId} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                <td className="py-3 px-4 font-mono font-bold">{order.symbol}</td>
                <td className="py-3 px-4">
                  <span className={`px-2 py-0.5 rounded text-xs ${
                    order.orderType === 'Limit' ? 'bg-blue-500/20 text-blue-400' : 'bg-purple-500/20 text-purple-400'
                  }`}>
                    {order.orderType}
                  </span>
                </td>
                <td className="py-3 px-4">
                  <span className={order.side === 'Buy' ? 'text-green-400' : 'text-red-400'}>
                    {order.side === 'Buy' ? 'BUY' : 'SELL'}
                  </span>
                </td>
                <td className="py-3 px-4 text-right font-mono">{formatPrice(order.price)}</td>
                <td className="py-3 px-4 text-right font-mono">{formatNumber(order.qty, 4)}</td>
                <td className="py-3 px-4 text-right font-mono text-gray-400">
                  {formatNumber(order.cumExecQty, 4)}
                </td>
                <td className="py-3 px-4">
                  <span className="px-2 py-0.5 rounded text-xs bg-yellow-500/20 text-yellow-400">
                    {order.orderStatus}
                  </span>
                </td>
                <td className="py-3 px-4 text-gray-400 text-sm">
                  {formatTime(order.createdTime)}
                </td>
                <td className="py-3 px-4 text-center">
                  <button
                    onClick={() => onCancelOrder(order.symbol, order.orderId)}
                    disabled={isCancelling}
                    className="px-3 py-1 bg-red-600/20 hover:bg-red-600/40 border border-red-600/50 rounded text-red-400 text-sm transition-colors disabled:opacity-50"
                  >
                    Отменить
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/**
 * Основной компонент страницы ордеров.
 */
export function OrdersPage() {
  const queryClient = useQueryClient();
  const [selectedPosition, setSelectedPosition] = useState<Position | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Запрос позиций
  const {
    data: positionsData,
    isLoading: isLoadingPositions,
    error: positionsError,
    refetch: refetchPositions,
  } = useQuery({
    queryKey: ['positions'],
    queryFn: positionsApi.getPositions,
    refetchInterval: autoRefresh ? 5000 : false,
  });

  // Запрос ордеров
  const {
    data: ordersData,
    isLoading: isLoadingOrders,
    error: ordersError,
    refetch: refetchOrders,
  } = useQuery({
    queryKey: ['openOrders'],
    queryFn: () => positionsApi.getOpenOrders(),
    refetchInterval: autoRefresh ? 5000 : false,
  });

  // Мутация закрытия позиции
  const closePositionMutation = useMutation({
    mutationFn: (params: { symbol: string; percent: 25 | 50 | 75 | 100 }) =>
      positionsApi.closePosition(params),
    onSuccess: (data) => {
      toast.success(data.message);
      setSelectedPosition(null);
      queryClient.invalidateQueries({ queryKey: ['positions'] });
      queryClient.invalidateQueries({ queryKey: ['openOrders'] });
    },
    onError: (error: Error) => {
      toast.error(`Ошибка закрытия: ${error.message}`);
    },
  });

  // Мутация отмены ордера
  const cancelOrderMutation = useMutation({
    mutationFn: (params: { symbol: string; orderId: string }) =>
      positionsApi.cancelOrder(params.symbol, params.orderId),
    onSuccess: (data) => {
      toast.success(data.message);
      queryClient.invalidateQueries({ queryKey: ['openOrders'] });
    },
    onError: (error: Error) => {
      toast.error(`Ошибка отмены: ${error.message}`);
    },
  });

  // Обработчики
  const handleClosePosition = useCallback((position: Position) => {
    setSelectedPosition(position);
  }, []);

  const handleConfirmClose = useCallback((percent: 25 | 50 | 75 | 100) => {
    if (selectedPosition) {
      closePositionMutation.mutate({
        symbol: selectedPosition.symbol,
        percent,
      });
    }
  }, [selectedPosition, closePositionMutation]);

  const handleCancelOrder = useCallback((symbol: string, orderId: string) => {
    cancelOrderMutation.mutate({ symbol, orderId });
  }, [cancelOrderMutation]);

  const handleRefresh = useCallback(() => {
    refetchPositions();
    refetchOrders();
  }, [refetchPositions, refetchOrders]);

  // Данные
  const positions = positionsData?.positions || [];
  const orders = ordersData?.orders || [];
  const hasError = positionsData?.error || ordersData?.error;

  // Общий P&L
  const totalPnl = positions.reduce((sum, p) => sum + p.unrealisedPnl, 0);
  const totalMargin = positions.reduce((sum, p) => sum + p.positionIM, 0);

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Заголовок */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold">Позиции и Ордера</h1>
          <p className="text-gray-400 text-sm mt-1">
            Управление открытыми позициями и ордерами на Bybit
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Автообновление */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-primary focus:ring-primary"
            />
            <span className="text-sm text-gray-400">Авто (5с)</span>
          </label>
          {/* Кнопка обновления */}
          <button
            onClick={handleRefresh}
            disabled={isLoadingPositions || isLoadingOrders}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm transition-colors disabled:opacity-50"
          >
            {isLoadingPositions || isLoadingOrders ? '...' : 'Обновить'}
          </button>
        </div>
      </div>

      {/* Предупреждение об ошибке */}
      {hasError && (
        <div className="mb-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-yellow-400">
          <strong>Внимание:</strong> {positionsData?.error || ordersData?.error}
        </div>
      )}

      {/* Сводка */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-surface border border-gray-800 rounded-xl p-4">
          <div className="text-sm text-gray-400 mb-1">Открытых позиций</div>
          <div className="text-2xl font-bold">{positions.length}</div>
        </div>
        <div className="bg-surface border border-gray-800 rounded-xl p-4">
          <div className="text-sm text-gray-400 mb-1">Открытых ордеров</div>
          <div className="text-2xl font-bold">{orders.length}</div>
        </div>
        <div className="bg-surface border border-gray-800 rounded-xl p-4">
          <div className="text-sm text-gray-400 mb-1">Общий P&L</div>
          <div className={`text-2xl font-bold ${totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {totalPnl >= 0 ? '+' : ''}{formatNumber(totalPnl)} USDT
          </div>
        </div>
        <div className="bg-surface border border-gray-800 rounded-xl p-4">
          <div className="text-sm text-gray-400 mb-1">Общая маржа</div>
          <div className="text-2xl font-bold">{formatNumber(totalMargin)} USDT</div>
        </div>
      </div>

      {/* Позиции */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span className="w-2 h-2 bg-green-500 rounded-full"></span>
          Открытые позиции ({positions.length})
        </h2>
        {isLoadingPositions ? (
          <div className="bg-surface border border-gray-800 rounded-xl p-8 text-center">
            <p className="text-gray-400">Загрузка позиций...</p>
          </div>
        ) : positions.length === 0 ? (
          <div className="bg-surface border border-gray-800 rounded-xl p-8 text-center">
            <p className="text-gray-400">Нет открытых позиций</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {positions.map((position) => (
              <PositionCard
                key={`${position.symbol}-${position.positionIdx}`}
                position={position}
                onCloseClick={handleClosePosition}
              />
            ))}
          </div>
        )}
      </div>

      {/* Ордера */}
      <div>
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span className="w-2 h-2 bg-yellow-500 rounded-full"></span>
          Открытые ордера ({orders.length})
        </h2>
        {isLoadingOrders ? (
          <div className="bg-surface border border-gray-800 rounded-xl p-8 text-center">
            <p className="text-gray-400">Загрузка ордеров...</p>
          </div>
        ) : (
          <OrdersTable
            orders={orders}
            onCancelOrder={handleCancelOrder}
            isCancelling={cancelOrderMutation.isPending}
          />
        )}
      </div>

      {/* Модальное окно закрытия позиции */}
      {selectedPosition && (
        <ClosePositionModal
          position={selectedPosition}
          onClose={() => setSelectedPosition(null)}
          onConfirm={handleConfirmClose}
          isLoading={closePositionMutation.isPending}
        />
      )}
    </div>
  );
}
