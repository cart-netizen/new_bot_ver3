// frontend/src/components/market/MetricsCard.tsx

import { Card } from '../ui/Card';
import type { OrderBookMetrics } from '@/types/orderbook.types';
import { ArrowUp, ArrowDown, Minus } from 'lucide-react';

interface MetricsCardProps {
  metrics: OrderBookMetrics | null;
  loading?: boolean;
}

/**
 * Компонент для отображения метрик торговой пары.
 * Показывает ключевые показатели: имбаланс, объемы, спред.
 */
export function MetricsCard({ metrics, loading = false }: MetricsCardProps) {
  if (loading) {
    return (
      <Card className="p-4">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded mb-4 w-1/3"></div>
          <div className="space-y-3">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-4 bg-gray-700 rounded"></div>
            ))}
          </div>
        </div>
      </Card>
    );
  }

  if (!metrics) {
    return (
      <Card className="p-4">
        <p className="text-gray-400 text-center">Нет данных</p>
      </Card>
    );
  }

  /**
   * Определение цвета для индикатора имбаланса.
   * Зеленый - покупатели, красный - продавцы, серый - баланс.
   */
  const getImbalanceColor = (imbalance: number): string => {
    if (imbalance > 0.6) return 'text-success';
    if (imbalance < 0.4) return 'text-destructive';
    return 'text-gray-400';
  };

  /**
   * Получение иконки для имбаланса.
   */
  const getImbalanceIcon = (imbalance: number) => {
    if (imbalance > 0.6) return <ArrowUp className="h-4 w-4" />;
    if (imbalance < 0.4) return <ArrowDown className="h-4 w-4" />;
    return <Minus className="h-4 w-4" />;
  };

  /**
   * Форматирование числа с разделителями тысяч.
   */
  const formatNumber = (num: number | null, decimals = 2): string => {
    if (num === null) return '-';
    return num.toLocaleString('ru-RU', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  };

  /**
   * Форматирование процентов для имбаланса.
   */
  const formatImbalance = (imbalance: number): string => {
    return `${(imbalance * 100).toFixed(1)}%`;
  };

  return (
    <Card className="p-4">
      {/* Заголовок */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">{metrics.symbol}</h3>
        <span className="text-xs text-gray-500">
          {new Date(metrics.timestamp).toLocaleTimeString('ru-RU')}
        </span>
      </div>

      {/* Основные метрики */}
      <div className="space-y-3">
        {/* Цены */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-gray-400 mb-1">Best Bid</p>
            <p className="text-sm font-mono text-success">
              {formatNumber(metrics.prices.best_bid, 8)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-400 mb-1">Best Ask</p>
            <p className="text-sm font-mono text-destructive">
              {formatNumber(metrics.prices.best_ask, 8)}
            </p>
          </div>
        </div>

        {/* Спред */}
        <div>
          <p className="text-xs text-gray-400 mb-1">Спред</p>
          <p className="text-sm font-mono">
            {formatNumber(metrics.prices.spread, 8)}
          </p>
        </div>

        {/* Имбаланс */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-gray-400">Имбаланс</p>
            <div className={`flex items-center gap-1 ${getImbalanceColor(metrics.imbalance.overall)}`}>
              {getImbalanceIcon(metrics.imbalance.overall)}
              <span className="text-sm font-semibold">
                {formatImbalance(metrics.imbalance.overall)}
              </span>
            </div>
          </div>

          {/* Визуальная шкала имбаланса */}
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                metrics.imbalance.overall > 0.6 ? 'bg-success' :
                metrics.imbalance.overall < 0.4 ? 'bg-destructive' :
                'bg-gray-400'
              }`}
              style={{ width: `${metrics.imbalance.overall * 100}%` }}
            />
          </div>

          {/* Детальный имбаланс */}
          <div className="grid grid-cols-2 gap-2 mt-2 text-xs">
            <div className="text-gray-500">
              Depth 5: {formatImbalance(metrics.imbalance.depth_5)}
            </div>
            <div className="text-gray-500">
              Depth 10: {formatImbalance(metrics.imbalance.depth_10)}
            </div>
          </div>
        </div>

        {/* Объемы */}
        <div>
          <p className="text-xs text-gray-400 mb-2">Объемы</p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-gray-500">Bid: </span>
              <span className="text-success font-mono">
                {formatNumber(metrics.volumes.total_bid, 0)}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Ask: </span>
              <span className="text-destructive font-mono">
                {formatNumber(metrics.volumes.total_ask, 0)}
              </span>
            </div>
          </div>
        </div>

        {/* Кластеры (если есть) */}
        {(metrics.clusters.largest_bid.volume > 0 || metrics.clusters.largest_ask.volume > 0) && (
          <div>
            <p className="text-xs text-gray-400 mb-2">Крупнейшие кластеры</p>
            <div className="space-y-1 text-xs">
              {metrics.clusters.largest_bid.volume > 0 && (
                <div className="text-success">
                  <span className="font-mono">
                    {formatNumber(metrics.clusters.largest_bid.price, 8)}
                  </span>
                  {' '}
                  <span className="text-gray-500">
                    ({formatNumber(metrics.clusters.largest_bid.volume, 0)})
                  </span>
                </div>
              )}
              {metrics.clusters.largest_ask.volume > 0 && (
                <div className="text-destructive">
                  <span className="font-mono">
                    {formatNumber(metrics.clusters.largest_ask.price, 8)}
                  </span>
                  {' '}
                  <span className="text-gray-500">
                    ({formatNumber(metrics.clusters.largest_ask.volume, 0)})
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}