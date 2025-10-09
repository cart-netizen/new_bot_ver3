// frontend/src/components/market/OrderBookWidget.tsx

import { Card } from '../ui/Card';
import type { OrderBook } from '../../types/orderbook.types';
import { useMemo } from 'react';

interface OrderBookWidgetProps {
  orderbook: OrderBook | null;
  loading?: boolean;
  maxLevels?: number;
}

/**
 * Компонент для отображения стакана ордеров.
 * Визуализирует уровни bid/ask с глубиной рынка.
 */
export function OrderBookWidget({
  orderbook,
  loading = false,
  maxLevels = 10
}: OrderBookWidgetProps) {

  /**
   * Подготовка данных для отображения.
   * Ограничиваем количество уровней и находим максимальный объем.
   */
  const { asks, bids, maxVolume } = useMemo(() => {
    if (!orderbook) {
      return { asks: [], bids: [], maxVolume: 0 };
    }

    const asks = orderbook.asks.slice(0, maxLevels);
    const bids = orderbook.bids.slice(0, maxLevels);

    const allVolumes = [...asks, ...bids].map(([_, qty]) => qty);
    const maxVolume = Math.max(...allVolumes, 1);

    return {
      asks: asks.reverse(), // Переворачиваем asks для отображения сверху вниз
      bids,
      maxVolume,
    };
  }, [orderbook, maxLevels]);

  /**
   * Форматирование цены.
   */
  const formatPrice = (price: number): string => {
    return price.toFixed(8);
  };

  /**
   * Форматирование объема.
   */
  const formatVolume = (volume: number): string => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(2)}M`;
    }
    if (volume >= 1000) {
      return `${(volume / 1000).toFixed(2)}K`;
    }
    return volume.toFixed(4);
  };

  /**
   * Расчет ширины прогресс-бара для визуализации объема.
   */
  const getVolumePercentage = (volume: number): number => {
    return (volume / maxVolume) * 100;
  };

  if (loading) {
    return (
      <Card className="p-4">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded mb-4 w-1/3"></div>
          <div className="space-y-2">
            {[...Array(10)].map((_, i) => (
              <div key={i} className="h-6 bg-gray-700 rounded"></div>
            ))}
          </div>
        </div>
      </Card>
    );
  }

  if (!orderbook) {
    return (
      <Card className="p-4">
        <p className="text-gray-400 text-center">Нет данных стакана</p>
      </Card>
    );
  }

  return (
    <Card className="p-4">
      {/* Заголовок */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Order Book</h3>
        <span className="text-xs text-gray-500">{orderbook.symbol}</span>
      </div>

      {/* Заголовки колонок */}
      <div className="grid grid-cols-3 gap-2 text-xs text-gray-400 mb-2 px-2">
        <div className="text-left">Цена</div>
        <div className="text-right">Объем</div>
        <div className="text-right">Всего</div>
      </div>

      {/* Asks (продажи) - красные */}
      <div className="space-y-1 mb-2">
        {asks.map(([price, quantity], index) => {
          const cumulative = asks
            .slice(0, index + 1)
            .reduce((sum, [_, qty]) => sum + qty, 0);
          const volumePercent = getVolumePercentage(quantity);

          return (
            <div
              key={`ask-${price}`}
              className="relative grid grid-cols-3 gap-2 text-xs py-1 px-2 rounded hover:bg-gray-800/50 transition-colors"
            >
              {/* Фоновый индикатор объема */}
              <div
                className="absolute right-0 top-0 bottom-0 bg-destructive/10 rounded"
                style={{ width: `${volumePercent}%` }}
              />

              {/* Данные */}
              <div className="relative z-10 text-left text-destructive font-mono">
                {formatPrice(price)}
              </div>
              <div className="relative z-10 text-right font-mono">
                {formatVolume(quantity)}
              </div>
              <div className="relative z-10 text-right text-gray-500 font-mono text-xs">
                {formatVolume(cumulative)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Спред */}
      <div className="flex items-center justify-between py-3 px-2 my-2 bg-gray-800/50 rounded">
        <span className="text-xs text-gray-400">Спред</span>
        <div className="flex items-center gap-3">
          <span className="text-sm font-mono">
            {orderbook.spread ? orderbook.spread.toFixed(8) : '-'}
          </span>
          {orderbook.spread && orderbook.best_bid && (
            <span className="text-xs text-gray-500">
              ({((orderbook.spread / orderbook.best_bid) * 100).toFixed(3)}%)
            </span>
          )}
        </div>
      </div>

      {/* Bids (покупки) - зеленые */}
      <div className="space-y-1 mt-2">
        {bids.map(([price, quantity], index) => {
          const cumulative = bids
            .slice(0, index + 1)
            .reduce((sum, [_, qty]) => sum + qty, 0);
          const volumePercent = getVolumePercentage(quantity);

          return (
            <div
              key={`bid-${price}`}
              className="relative grid grid-cols-3 gap-2 text-xs py-1 px-2 rounded hover:bg-gray-800/50 transition-colors"
            >
              {/* Фоновый индикатор объема */}
              <div
                className="absolute right-0 top-0 bottom-0 bg-success/10 rounded"
                style={{ width: `${volumePercent}%` }}
              />

              {/* Данные */}
              <div className="relative z-10 text-left text-success font-mono">
                {formatPrice(price)}
              </div>
              <div className="relative z-10 text-right font-mono">
                {formatVolume(quantity)}
              </div>
              <div className="relative z-10 text-right text-gray-500 font-mono text-xs">
                {formatVolume(cumulative)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Футер с информацией */}
      <div className="mt-4 pt-3 border-t border-gray-800 text-xs text-gray-500 text-center">
        Обновлено: {new Date(orderbook.timestamp).toLocaleTimeString('ru-RU')}
      </div>
    </Card>
  );
}