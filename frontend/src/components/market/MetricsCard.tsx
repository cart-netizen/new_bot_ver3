// frontend/src/components/market/MetricsCard.tsx
// ОБНОВЛЕННАЯ ВЕРСИЯ для работы с новой структурой marketStore

import { Card } from '../ui/Card';
import type { OrderBookMetrics } from '../../types/orderbook.types';
import { formatPrice, formatVolume } from '../../utils/format';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface MetricsCardProps {
  metrics: OrderBookMetrics;
}

/**
 * Компонент отображения метрик стакана.
 * ОБНОВЛЕНО: Работает с новой оптимизированной структурой данных.
 */
export function MetricsCard({ metrics }: MetricsCardProps) {
  // Определение тренда на основе imbalance
  const getTrendIcon = () => {
    const imbalance = metrics.imbalance?.overall ?? 0.5;

    if (imbalance > 0.6) {
      return <TrendingUp className="w-4 h-4 text-success" />;
    } else if (imbalance < 0.4) {
      return <TrendingDown className="w-4 h-4 text-danger" />;
    }
    return <Minus className="w-4 h-4 text-gray-400" />;
  };

  // Цвет для imbalance
  const getImbalanceColor = (value: number) => {
    if (value > 0.6) return 'text-success';
    if (value < 0.4) return 'text-danger';
    return 'text-gray-400';
  };

  const imbalanceOverall = metrics.imbalance?.overall ?? 0.5;
  const imbalanceDepth5 = metrics.imbalance?.depth_5 ?? 0.5;

  return (
    <Card className="p-4">
      {/* Заголовок */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">{metrics.symbol}</h3>
        {getTrendIcon()}
      </div>

      {/* Ценовые метрики */}
      <div className="space-y-2 mb-4">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Best Bid</span>
          <span className="text-success font-mono">
            {metrics.prices?.best_bid
              ? formatPrice(metrics.prices.best_bid, 2)
              : '-'}
          </span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Best Ask</span>
          <span className="text-danger font-mono">
            {metrics.prices?.best_ask
              ? formatPrice(metrics.prices.best_ask, 2)
              : '-'}
          </span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Mid Price</span>
          <span className="text-white font-mono font-semibold">
            {metrics.prices?.mid_price
              ? formatPrice(metrics.prices.mid_price, 2)
              : '-'}
          </span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Spread</span>
          <span className="text-gray-300 font-mono">
            {metrics.prices?.spread
              ? formatPrice(metrics.prices.spread, 4)
              : '-'}
          </span>
        </div>
      </div>

      {/* Объемные метрики */}
      <div className="space-y-2 mb-4 pt-4 border-t border-gray-700">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Total Bid Volume</span>
          <span className="text-success font-mono">
            {metrics.volumes?.total_bid
              ? formatVolume(metrics.volumes.total_bid)
              : '-'}
          </span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Total Ask Volume</span>
          <span className="text-danger font-mono">
            {metrics.volumes?.total_ask
              ? formatVolume(metrics.volumes.total_ask)
              : '-'}
          </span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Bid Depth (5)</span>
          <span className="text-gray-300 font-mono">
            {metrics.volumes?.bid_depth_5
              ? formatVolume(metrics.volumes.bid_depth_5)
              : '-'}
          </span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Ask Depth (5)</span>
          <span className="text-gray-300 font-mono">
            {metrics.volumes?.ask_depth_5
              ? formatVolume(metrics.volumes.ask_depth_5)
              : '-'}
          </span>
        </div>
      </div>

      {/* Дисбаланс */}
      <div className="space-y-2 pt-4 border-t border-gray-700">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Imbalance (Overall)</span>
          <span className={`font-mono font-semibold ${getImbalanceColor(imbalanceOverall)}`}>
            {(imbalanceOverall * 100).toFixed(1)}%
          </span>
        </div>

        {/* Визуальный индикатор imbalance */}
        <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
          <div className="h-full flex">
            {/* Bid side (green) */}
            <div
              className="bg-success transition-all"
              style={{ width: `${imbalanceOverall * 100}%` }}
            />
            {/* Ask side (red) */}
            <div
              className="bg-danger transition-all"
              style={{ width: `${(1 - imbalanceOverall) * 100}%` }}
            />
          </div>
        </div>

        <div className="flex justify-between text-xs text-gray-500">
          <span>← Ask pressure</span>
          <span>Bid pressure →</span>
        </div>

        <div className="flex justify-between text-sm mt-2">
          <span className="text-gray-400">Imbalance (Depth 5)</span>
          <span className={`font-mono ${getImbalanceColor(imbalanceDepth5)}`}>
            {(imbalanceDepth5 * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* VWAP метрики */}
      {metrics.vwap && (
        <div className="space-y-2 pt-4 border-t border-gray-700 mt-4">
          <div className="text-xs text-gray-500 mb-2">VWAP</div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Bid</span>
            <span className="text-success font-mono">
              {metrics.vwap.bid ? formatPrice(metrics.vwap.bid, 2) : '-'}
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Ask</span>
            <span className="text-danger font-mono">
              {metrics.vwap.ask ? formatPrice(metrics.vwap.ask, 2) : '-'}
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Mid</span>
            <span className="text-white font-mono">
              {metrics.vwap.mid ? formatPrice(metrics.vwap.mid, 2) : '-'}
            </span>
          </div>
        </div>
      )}

      {/* Timestamp */}
      <div className="mt-4 pt-4 border-t border-gray-700 text-xs text-gray-500">
        {new Date(metrics.timestamp).toLocaleTimeString()}
      </div>
    </Card>
  );
}