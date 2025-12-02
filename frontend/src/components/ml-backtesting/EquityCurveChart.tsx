// frontend/src/components/ml-backtesting/EquityCurveChart.tsx

import { useQuery } from '@tanstack/react-query';
import {

  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart
} from 'recharts';
import { RefreshCw } from 'lucide-react';
import { Card } from '../ui/Card';
import * as mlBacktestingApi from '../../api/ml-backtesting.api';

interface EquityCurveChartProps {
  backtestId: string;
  showDrawdown?: boolean;
}

export function EquityCurveChart({ backtestId, showDrawdown = true }: EquityCurveChartProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['equity-curve', backtestId],
    queryFn: () => mlBacktestingApi.getEquityCurve(backtestId, 200)
  });

  if (isLoading) {
    return (
      <Card className="p-6">
        <div className="h-64 flex items-center justify-center">
          <RefreshCw className="h-8 w-8 animate-spin text-gray-400" />
        </div>
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card className="p-6">
        <div className="h-64 flex items-center justify-center text-gray-400">
          Failed to load equity curve
        </div>
      </Card>
    );
  }

  const formatCurrency = (value: number) => `$${value.toLocaleString()}`;
  const formatPercent = (value: number) => `${value.toFixed(1)}%`;

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Equity Curve</h3>
        <div className="flex gap-4 text-sm">
          <div>
            <span className="text-gray-400">Initial: </span>
            <span className="text-white font-medium">{formatCurrency(data.initial_capital)}</span>
          </div>
          <div>
            <span className="text-gray-400">Final: </span>
            <span className={data.final_capital >= data.initial_capital ? "text-green-400" : "text-red-400"}>
              {formatCurrency(data.final_capital)}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Return: </span>
            <span className={data.total_return_pct >= 0 ? "text-green-400" : "text-red-400"}>
              {data.total_return_pct >= 0 ? '+' : ''}{formatPercent(data.total_return_pct)}
            </span>
          </div>
        </div>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data.data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="x"
              stroke="#9CA3AF"
              fontSize={12}
              tickFormatter={(v) => `${v}`}
            />
            <YAxis
              yAxisId="equity"
              stroke="#9CA3AF"
              fontSize={12}
              tickFormatter={formatCurrency}
              domain={['auto', 'auto']}
            />
            {showDrawdown && (
              <YAxis
                yAxisId="drawdown"
                orientation="right"
                stroke="#9CA3AF"
                fontSize={12}
                tickFormatter={formatPercent}
                domain={[0, 'auto']}
              />
            )}
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px'
              }}
              labelStyle={{ color: '#9CA3AF' }}
              formatter={(value: number, name: string) => {
                if (name === 'equity') return [formatCurrency(value), 'Equity'];
                if (name === 'drawdown') return [formatPercent(value), 'Drawdown'];
                return [value, name];
              }}
            />
            <Legend />
            <Line
              yAxisId="equity"
              type="monotone"
              dataKey="equity"
              stroke="#8B5CF6"
              strokeWidth={2}
              dot={false}
              name="Equity"
            />
            {showDrawdown && (
              <Area
                yAxisId="drawdown"
                type="monotone"
                dataKey="drawdown"
                fill="#EF4444"
                fillOpacity={0.3}
                stroke="#EF4444"
                strokeWidth={1}
                name="Drawdown"
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4 text-center">
        <div className="p-3 bg-gray-800/50 rounded-lg">
          <p className="text-2xl font-bold text-red-400">{formatPercent(data.max_drawdown_pct)}</p>
          <p className="text-sm text-gray-400">Max Drawdown</p>
        </div>
        <div className="p-3 bg-gray-800/50 rounded-lg">
          <p className="text-2xl font-bold text-white">{data.n_points}</p>
          <p className="text-sm text-gray-400">Data Points</p>
        </div>
        <div className="p-3 bg-gray-800/50 rounded-lg">
          <p className={`text-2xl font-bold ${data.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {data.total_return_pct >= 0 ? '+' : ''}{formatPercent(data.total_return_pct)}
          </p>
          <p className="text-sm text-gray-400">Total Return</p>
        </div>
      </div>
    </Card>
  );
}

export default EquityCurveChart;
