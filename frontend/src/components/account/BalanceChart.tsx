// frontend/src/components/account/BalanceChart.tsx

import { useMemo, useState } from 'react';
import { Card } from '../ui/Card';
import type { BalanceHistory } from '../../types/account.types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp } from 'lucide-react';

interface BalanceChartProps {
  history: BalanceHistory | null;
  loading?: boolean;
  onPeriodChange?: (period: '1h' | '24h' | '7d' | '30d') => void;
}

/**
 * Компонент для отображения графика изменения баланса.
 */
export function BalanceChart({ history, loading = false, onPeriodChange }: BalanceChartProps) {
  const [selectedPeriod, setSelectedPeriod] = useState<'1h' | '24h' | '7d' | '30d'>('24h');

  /**
   * Обработка выбора периода.
   */
  const handlePeriodChange = (period: '1h' | '24h' | '7d' | '30d') => {
    setSelectedPeriod(period);
    onPeriodChange?.(period);
  };

  /**
   * Подготовка данных для графика.
   */
  const chartData = useMemo(() => {
    if (!history || !history.points || history.points.length === 0) {
      return [];
    }

    return history.points.map((point) => ({
      timestamp: point.timestamp || Date.now(),
      balance: point.balance || 0,
      datetime: point.datetime || new Date(point.timestamp || Date.now()).toLocaleString('ru-RU', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }),
    }));
  }, [history]);

  /**
   * Расчет изменения баланса.
   */
  const balanceChange = useMemo(() => {
    if (chartData.length < 2) {
      return { amount: 0, percentage: 0 };
    }

    const first = chartData[0]?.balance || 0;
    const last = chartData[chartData.length - 1]?.balance || 0;
    const amount = last - first;
    const percentage = first > 0 ? (amount / first) * 100 : 0;

    return { amount, percentage };
  }, [chartData]);

  const isPositive = balanceChange.amount >= 0;

  /**
   * Кнопки периодов.
   */
  const periods: Array<{ value: '1h' | '24h' | '7d' | '30d'; label: string }> = [
    { value: '1h', label: '1ч' },
    { value: '24h', label: '24ч' },
    { value: '7d', label: '7д' },
    { value: '30d', label: '30д' },
  ];

  if (loading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-700 rounded w-1/3"></div>
          <div className="h-64 bg-gray-700 rounded"></div>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6">
      {/* Заголовок */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">Динамика Баланса</h2>
        </div>

        {/* Селектор периода */}
        <div className="flex gap-2">
          {periods.map((period) => (
            <button
              key={period.value}
              onClick={() => handlePeriodChange(period.value)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                selectedPeriod === period.value
                  ? 'bg-primary text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {period.label}
            </button>
          ))}
        </div>
      </div>

      {/* Информация об изменении */}
      {chartData.length >= 2 && (
        <div className="mb-4">
          <p className="text-sm text-gray-400 mb-1">Изменение за период</p>
          <div className="flex items-baseline gap-2">
            <span className={`text-2xl font-bold ${
              isPositive ? 'text-success' : 'text-destructive'
            }`}>
              {isPositive ? '+' : ''}{balanceChange.amount.toFixed(2)} USDT
            </span>
            <span className={`text-lg ${
              isPositive ? 'text-success' : 'text-destructive'
            }`}>
              ({isPositive ? '+' : ''}{balanceChange.percentage.toFixed(2)}%)
            </span>
          </div>
        </div>
      )}

      {/* График */}
      {chartData.length > 0 ? (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="datetime"
              stroke="#9CA3AF"
              style={{ fontSize: '12px' }}
            />
            <YAxis
              stroke="#9CA3AF"
              style={{ fontSize: '12px' }}
              domain={['dataMin - 100', 'dataMax + 100']}
              tickFormatter={(value) => `$${value.toFixed(0)}`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#fff',
              }}
              formatter={(value: number) => [`$${value.toFixed(2)}`, 'Баланс']}
              labelStyle={{ color: '#9CA3AF' }}
            />
            <Line
              type="monotone"
              dataKey="balance"
              stroke={isPositive ? '#10B981' : '#EF4444'}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className="h-64 flex items-center justify-center">
          <p className="text-gray-400">Нет данных для отображения</p>
        </div>
      )}
    </Card>
  );
}