// frontend/src/components/orders/OrderDetailModal.tsx
/**
 * Модальное окно детального просмотра ордера.
 *
 * Функционал:
 * - Отображение всех параметров ордера
 * - График торговой пары
 * - Расчет PnL в реальном времени
 * - Кнопка закрытия ордера с подтверждением
 */

import React, { useState, useEffect } from 'react';
import {X, TrendingUp, TrendingDown, AlertTriangle, Loader2, AlertCircle} from 'lucide-react';
import { cn } from '../../utils/helpers';
import type { OrderDetail } from '../../types/orders.types';
import { PriceChart } from '../market/PriceChart';

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
 * Форматирование процента.
 */
function formatPercent(value: number): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
}

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
    second: '2-digit',
  });
}

/**
 * Props компонента.
 */
interface OrderDetailModalProps {
  /**
   * Детальная информация об ордере.
   */
  orderDetail: OrderDetail;

  /**
   * Ошибка загрузки деталей (опциональная).
   */
  error?: string | null;  // ← ДОБАВИТЬ ЭТУ СТРОКУ

  /**
   * Callback при закрытии модального окна.
   */
  onClose: () => void;

  /**
   * Callback при закрытии ордера.
   */
  onCloseOrder: (orderId: string) => Promise<void>;

  /**
   * Статус закрытия ордера.
   */
  isClosing?: boolean;
}

/**
 * Компонент модального окна детального просмотра.
 */
export function OrderDetailModal({
  orderDetail,
  error,
  onClose,
  onCloseOrder,
  isClosing = false
}: OrderDetailModalProps) {
  const [showConfirmClose, setShowConfirmClose] = useState(false);
  const [currentPnl] = useState(orderDetail.current_pnl);
  const [currentPnlPercent] = useState(orderDetail.current_pnl_percent);

  const isBuy = orderDetail.side === 'BUY';
  const isActive = ['PENDING', 'PLACED', 'PARTIALLY_FILLED'].includes(orderDetail.status);
  const isProfitable = currentPnl >= 0;

  /**
   * Обновление PnL в реальном времени.
   * TODO: Интегрировать с WebSocket для получения текущей цены.
   */
  useEffect(() => {
    // Здесь должна быть интеграция с marketStore для получения текущей цены
    // и пересчета PnL

    // Пример расчета:
    // const currentPrice = getCurrentPrice(orderDetail.symbol);
    // const pnl = calculatePnL(orderDetail.entry_price, currentPrice, orderDetail.quantity, orderDetail.side);
    // setCurrentPnl(pnl);
    // setCurrentPnlPercent((pnl / orderDetail.position_value) * 100);
  }, [orderDetail.symbol]);

  /**
   * Обработка закрытия ордера.
   */
  const handleCloseOrder = async () => {
    try {
      await onCloseOrder(orderDetail.order_id);
      onClose(); // Закрываем модальное окно после успешного закрытия
    } catch (error) {
      console.error('Error closing order:', error);
      // Ошибка обрабатывается в родительском компоненте
    } finally {
      setShowConfirmClose(false);
    }
  };

  /**
   * Обработка клика по оверлею.
   */
  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  /**
   * Рендер подтверждения закрытия.
   */
  const renderConfirmClose = () => {
    if (!showConfirmClose) return null;

    return (
      <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-[60]" onClick={() => setShowConfirmClose(false)}>
        <div className="bg-surface border border-gray-800 rounded-lg p-6 max-w-md mx-4" onClick={(e) => e.stopPropagation()}>
          <div className="flex items-start gap-3 mb-4">
            <AlertTriangle className="h-6 w-6 text-yellow-500 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">
                Подтвердите закрытие ордера
              </h3>
              <p className="text-sm text-gray-400">
                Вы уверены, что хотите закрыть ордер {orderDetail.symbol}?
              </p>
              <p className="text-sm text-gray-400 mt-2">
                Текущий PnL: <span className={cn(
                  'font-semibold',
                  isProfitable ? 'text-success' : 'text-destructive'
                )}>
                  {formatPrice(currentPnl)} ({formatPercent(currentPnlPercent)})
                </span>
              </p>
            </div>
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => setShowConfirmClose(false)}
              className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
              disabled={isClosing}
            >
              Отмена
            </button>
            <button
              onClick={handleCloseOrder}
              disabled={isClosing}
              className="flex-1 px-4 py-2 bg-destructive hover:bg-destructive/90 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isClosing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Закрытие...
                </>
              ) : (
                'Закрыть ордер'
              )}
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <>
      {/* Оверлей */}
      <div
        className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4 overflow-y-auto"
        onClick={handleOverlayClick}
      >
        {/* Модальное окно */}
        <div
          className="bg-surface border border-gray-800 rounded-lg w-full max-w-6xl my-8"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Заголовок */}
          <div className="flex items-center justify-between p-6 border-b border-gray-800">
            <div className="flex items-center gap-3">
              {isBuy ? (
                <TrendingUp className="h-6 w-6 text-success" />
              ) : (
                <TrendingDown className="h-6 w-6 text-destructive" />
              )}
              <div>
                <h2 className="text-2xl font-bold text-white">
                  {orderDetail.symbol}
                </h2>
                <p className="text-sm text-gray-400">
                  {isBuy ? 'Покупка' : 'Продажа'} • Ордер #{orderDetail.client_order_id}
                </p>
              </div>
            </div>

            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
              title="Закрыть"
            >
              <X className="h-6 w-6 text-gray-400 hover:text-white" />
            </button>
          </div>

          {/* Блок ошибки */}
          {error && (
            <div className="mx-4 mt-4">
              <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="text-sm font-semibold text-destructive mb-1">
                      Ошибка загрузки деталей
                    </h3>
                    <p className="text-sm text-gray-400">{error}</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Контент */}
          <div className="p-6 space-y-6">
            {/* PnL Карточка */}
            <div className={cn(
              'p-6 rounded-lg border-2',
              isProfitable
                ? 'bg-success/10 border-success/30'
                : 'bg-destructive/10 border-destructive/30'
            )}>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <div className="text-sm text-gray-400 mb-1">Текущий PnL</div>
                  <div className={cn(
                    'text-3xl font-bold',
                    isProfitable ? 'text-success' : 'text-destructive'
                  )}>
                    {formatPrice(currentPnl)}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">PnL %</div>
                  <div className={cn(
                    'text-3xl font-bold',
                    isProfitable ? 'text-success' : 'text-destructive'
                  )}>
                    {formatPercent(currentPnlPercent)}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">Текущая цена</div>
                  <div className="text-3xl font-bold text-white">
                    {formatPrice(orderDetail.current_price)}
                  </div>
                </div>
              </div>
            </div>

            {/* Grid с информацией и графиком */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Информация об ордере (1/3) */}
              <div className="lg:col-span-1 space-y-4">
                <div className="bg-gray-800/50 rounded-lg p-4 space-y-3">
                  <h3 className="font-semibold text-white mb-3">Параметры ордера</h3>

                  <div>
                    <div className="text-xs text-gray-500">Цена входа</div>
                    <div className="text-sm text-white font-medium">{formatPrice(orderDetail.entry_price)}</div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-500">Количество</div>
                    <div className="text-sm text-white font-medium">{orderDetail.quantity.toFixed(4)}</div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-500">Take Profit</div>
                    <div className="text-sm text-success font-medium">{formatPrice(orderDetail.take_profit)}</div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-500">Stop Loss</div>
                    <div className="text-sm text-destructive font-medium">{formatPrice(orderDetail.stop_loss)}</div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-500">Плечо</div>
                    <div className="text-sm text-white font-medium">{orderDetail.leverage}x</div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-500">Стоимость позиции</div>
                    <div className="text-sm text-white font-medium">{formatPrice(orderDetail.position_value)}</div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-500">Использ. маржа</div>
                    <div className="text-sm text-white font-medium">{formatPrice(orderDetail.margin_used)}</div>
                  </div>

                  {orderDetail.strategy && (
                    <div>
                      <div className="text-xs text-gray-500">Стратегия</div>
                      <div className="text-sm text-white font-medium">{orderDetail.strategy}</div>
                    </div>
                  )}

                  <div>
                    <div className="text-xs text-gray-500">Создан</div>
                    <div className="text-xs text-white">{formatDateTime(orderDetail.created_at)}</div>
                  </div>
                </div>

                {/* Кнопка закрытия ордера */}
                {isActive && (
                  <button
                    onClick={() => setShowConfirmClose(true)}
                    disabled={isClosing}
                    className="w-full px-4 py-3 bg-destructive hover:bg-destructive/90 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-semibold"
                  >
                    {isClosing ? (
                      <>
                        <Loader2 className="h-5 w-5 animate-spin" />
                        Закрытие ордера...
                      </>
                    ) : (
                      <>
                        <X className="h-5 w-5" />
                        Закрыть ордер
                      </>
                    )}
                  </button>
                )}
              </div>

              {/* График (2/3) */}
              <div className="lg:col-span-2">
                <PriceChart
                  symbol={orderDetail.symbol}
                  loading={false}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Подтверждение закрытия */}
      {renderConfirmClose()}
    </>
  );
}