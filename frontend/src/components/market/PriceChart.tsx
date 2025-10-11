// frontend/src/components/market/PriceChart.tsx

import { useMemo } from 'react';
import { Card } from '../ui/Card';
import type { OrderBookMetrics } from '@/types/orderbook.types';
import {

  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart,
} from 'recharts';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface PriceChartProps {
  symbol: string;
  metricsHistory: OrderBookMetrics[];
  loading?: boolean;
}

/**
 * Компонент для отображения графика цены торговой пары.
 * Показывает динамику best bid/ask и mid price.
 */
export function PriceChart({ symbol, metricsHistory, loading = false }: PriceChartProps) {
  /**
   * Подготовка данных для графика.
   * Преобразуем историю метрик в формат для recharts.
   */
  const chartData = useMemo(() => {
    if (!metricsHistory || metricsHistory.length === 0) {
      return [];
    }

    return metricsHistory
      .filter((m) => m.prices.best_bid && m.prices.best_ask)
      .map((metric) => ({
        timestamp: metric.timestamp,
        time: new Date(metric.timestamp).toLocaleTimeString('ru-RU', {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }),
        bestBid: metric.prices.best_bid || 0,
        bestAsk: metric.prices.best_ask || 0,
        midPrice: metric.prices.mid_price || 0,
        spread: metric.prices.spread || 0,
      }));
  }, [metricsHistory]);

  /**
   * Расчёт изменения цены за период.
   */
  const priceChange = useMemo(() => {
    if (chartData.length < 2) {
      return { amount: 0, percentage: 0 };
    }

    const first = chartData[0]?.midPrice || 0;
    const last = chartData[chartData.length - 1]?.midPrice || 0;
    const amount = last - first;
    const percentage = first > 0 ? (amount / first) * 100 : 0;

    return { amount, percentage };
  }, [chartData]);

  const isPositive = priceChange.amount >= 0;

  /**
   * Форматирование цены для отображения.
   */
  const formatPrice = (value: number): string => {
    return value.toFixed(2);
  };

  if (loading) {
    return (
      <Card className="p-4">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded mb-4 w-1/3"></div>
          <div className="h-64 bg-gray-700 rounded"></div>
        </div>
      </Card>
    );
  }

  if (chartData.length === 0) {
    return (
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">График: {symbol}</h3>
        <div className="h-64 flex items-center justify-center">
          <p className="text-gray-400">Накапливаем данные для графика...</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-4">
      {/* Заголовок с изменением цены */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold">{symbol}</h3>
          <p className="text-xs text-gray-500">
            {chartData.length} точек данных
          </p>
        </div>

        {/* Индикатор изменения */}
        <div className="flex items-center gap-2">
          {isPositive ? (
            <TrendingUp className="h-5 w-5 text-success" />
          ) : (
            <TrendingDown className="h-5 w-5 text-destructive" />
          )}
          <div className="text-right">
            <div
              className={`text-lg font-semibold ${
                isPositive ? 'text-success' : 'text-destructive'
              }`}
            >
              {isPositive ? '+' : ''}
              {priceChange.amount.toFixed(2)}
            </div>
            <div
              className={`text-sm ${
                isPositive ? 'text-success' : 'text-destructive'
              }`}
            >
              ({isPositive ? '+' : ''}
              {priceChange.percentage.toFixed(2)}%)
            </div>
          </div>
        </div>
      </div>

      {/* График */}
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData}>
          {/* Сетка */}
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />

          {/* Оси */}
          <XAxis
            dataKey="time"
            stroke="#9CA3AF"
            style={{ fontSize: '11px' }}
            interval="preserveStartEnd"
          />
          <YAxis
            stroke="#9CA3AF"
            style={{ fontSize: '11px' }}
            domain={['auto', 'auto']}
            tickFormatter={formatPrice}
          />

          {/* Tooltip */}
          <Tooltip
            contentStyle={{
              backgroundColor: '#1F2937',
              border: '1px solid #374151',
              borderRadius: '8px',
              color: '#fff',
            }}
            formatter={(value: number, name: string) => {
              const label =
                name === 'bestBid'
                  ? 'Best Bid'
                  : name === 'bestAsk'
                  ? 'Best Ask'
                  : name === 'midPrice'
                  ? 'Mid Price'
                  : name;
              return [formatPrice(value), label];
            }}
            labelStyle={{ color: '#9CA3AF', marginBottom: '8px' }}
          />

          {/* Легенда */}
          <Legend
            wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}
            iconType="line"
          />

          {/* Области между bid и ask */}
          <defs>
            <linearGradient id="colorBid" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10B981" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorAsk" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#EF4444" stopOpacity={0} />
            </linearGradient>
          </defs>

          {/* Линии */}
          <Area
            type="monotone"
            dataKey="bestBid"
            stroke="#10B981"
            strokeWidth={2}
            fill="url(#colorBid)"
            name="Best Bid"
            dot={false}
          />

          <Area
            type="monotone"
            dataKey="bestAsk"
            stroke="#EF4444"
            strokeWidth={2}
            fill="url(#colorAsk)"
            name="Best Ask"
            dot={false}
          />

          <Line
            type="monotone"
            dataKey="midPrice"
            stroke="#8B5CF6"
            strokeWidth={2}
            name="Mid Price"
            dot={false}
            strokeDasharray="5 5"
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* Информация о последней точке */}
      {chartData.length > 0 && (
        <div className="mt-4 pt-3 border-t border-gray-800 grid grid-cols-3 gap-4 text-xs">
          <div>
            <p className="text-gray-400 mb-1">Best Bid</p>
            <p className="font-mono text-success">
              {formatPrice(chartData[chartData.length - 1]?.bestBid || 0)}
            </p>
          </div>
          <div>
            <p className="text-gray-400 mb-1">Mid Price</p>
            <p className="font-mono text-purple-400">
              {formatPrice(chartData[chartData.length - 1]?.midPrice || 0)}
            </p>
          </div>
          <div>
            <p className="text-gray-400 mb-1">Best Ask</p>
            <p className="font-mono text-destructive">
              {formatPrice(chartData[chartData.length - 1]?.bestAsk || 0)}
            </p>
          </div>
        </div>
      )}
    </Card>
  );
}