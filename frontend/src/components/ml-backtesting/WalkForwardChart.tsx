// frontend/src/components/ml-backtesting/WalkForwardChart.tsx

import {

  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,

  Line,
  ComposedChart,
  ReferenceLine
} from 'recharts';
import { TrendingUp, TrendingDown, Target } from 'lucide-react';
import { Card } from '../ui/Card';
import { cn } from '../../utils/helpers';
import type { PeriodResult } from '../../api/ml-backtesting.api';

interface WalkForwardChartProps {
  periodResults: PeriodResult[];
  threshold?: number;
}

export function WalkForwardChart({ periodResults, threshold = 0.6 }: WalkForwardChartProps) {
  if (!periodResults || periodResults.length === 0) {
    return (
      <Card className="p-6">
        <div className="h-64 flex items-center justify-center text-gray-400">
          No walk-forward data available
        </div>
      </Card>
    );
  }

  // Prepare data for chart
  const chartData = periodResults.map((period) => ({
    period: `P${period.period}`,
    periodNum: period.period,
    accuracy: period.accuracy * 100,
    f1_macro: period.f1_macro * 100,
    pnl_percent: (period.pnl_percent || 0) * 100,
    win_rate: (period.win_rate || 0) * 100,
    samples: period.samples
  }));

  // Calculate statistics
  const avgAccuracy = chartData.reduce((a, b) => a + b.accuracy, 0) / chartData.length;
  const avgF1 = chartData.reduce((a, b) => a + b.f1_macro, 0) / chartData.length;
  const minAccuracy = Math.min(...chartData.map(d => d.accuracy));
  const maxAccuracy = Math.max(...chartData.map(d => d.accuracy));
  const stabilityScore = 1 - (maxAccuracy - minAccuracy) / avgAccuracy;

  const formatPercent = (value: number) => `${value.toFixed(1)}%`;

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Walk-Forward Performance</h3>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">Stability Score:</span>
          <span className={cn(
            "text-sm font-medium px-2 py-0.5 rounded",
            stabilityScore > 0.8 ? "text-green-400 bg-green-500/10" :
            stabilityScore > 0.6 ? "text-yellow-400 bg-yellow-500/10" :
            "text-red-400 bg-red-500/10"
          )}>
            {(stabilityScore * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Main Chart */}
      <div className="h-64 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="period" stroke="#9CA3AF" fontSize={12} />
            <YAxis
              yAxisId="percent"
              stroke="#9CA3AF"
              fontSize={12}
              tickFormatter={(v) => `${v}%`}
              domain={[0, 100]}
            />
            <YAxis
              yAxisId="pnl"
              orientation="right"
              stroke="#9CA3AF"
              fontSize={12}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px'
              }}
              formatter={(value: number, name: string) => [formatPercent(value), name]}
            />
            <Legend />
            <ReferenceLine
              yAxisId="percent"
              y={threshold * 100}
              stroke="#F59E0B"
              strokeDasharray="5 5"
              label={{ value: 'Threshold', position: 'right', fill: '#F59E0B', fontSize: 10 }}
            />
            <Bar
              yAxisId="percent"
              dataKey="accuracy"
              fill="#8B5CF6"
              name="Accuracy"
              radius={[4, 4, 0, 0]}
            />
            <Bar
              yAxisId="percent"
              dataKey="f1_macro"
              fill="#06B6D4"
              name="F1 Score"
              radius={[4, 4, 0, 0]}
            />
            <Line
              yAxisId="pnl"
              type="monotone"
              dataKey="pnl_percent"
              stroke="#10B981"
              strokeWidth={2}
              dot={{ fill: '#10B981', r: 4 }}
              name="P&L %"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Statistics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 bg-gray-800/50 rounded-lg text-center">
          <div className="flex items-center justify-center gap-2 mb-1">
            <Target className="h-4 w-4 text-purple-400" />
            <span className="text-purple-400 font-bold text-xl">{formatPercent(avgAccuracy)}</span>
          </div>
          <p className="text-xs text-gray-400">Avg Accuracy</p>
        </div>

        <div className="p-4 bg-gray-800/50 rounded-lg text-center">
          <span className="text-cyan-400 font-bold text-xl">{formatPercent(avgF1)}</span>
          <p className="text-xs text-gray-400">Avg F1 Score</p>
        </div>

        <div className="p-4 bg-gray-800/50 rounded-lg text-center">
          <div className="flex items-center justify-center gap-2 mb-1">
            {minAccuracy >= threshold * 100 ? (
              <TrendingUp className="h-4 w-4 text-green-400" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-400" />
            )}
            <span className={cn(
              "font-bold text-xl",
              minAccuracy >= threshold * 100 ? "text-green-400" : "text-red-400"
            )}>
              {formatPercent(minAccuracy)}
            </span>
          </div>
          <p className="text-xs text-gray-400">Min Accuracy</p>
        </div>

        <div className="p-4 bg-gray-800/50 rounded-lg text-center">
          <span className="text-green-400 font-bold text-xl">{formatPercent(maxAccuracy)}</span>
          <p className="text-xs text-gray-400">Max Accuracy</p>
        </div>
      </div>

      {/* Period Details Table */}
      <div className="mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left p-2 text-gray-400">Period</th>
              <th className="text-center p-2 text-gray-400">Samples</th>
              <th className="text-center p-2 text-gray-400">Accuracy</th>
              <th className="text-center p-2 text-gray-400">F1</th>
              <th className="text-center p-2 text-gray-400">Win Rate</th>
              <th className="text-center p-2 text-gray-400">P&L</th>
            </tr>
          </thead>
          <tbody>
            {periodResults.map((period) => (
              <tr key={period.period} className="border-b border-gray-800">
                <td className="p-2 text-white font-medium">Period {period.period}</td>
                <td className="text-center p-2 text-gray-300">{period.samples.toLocaleString()}</td>
                <td className={cn(
                  "text-center p-2 font-medium",
                  period.accuracy >= threshold ? "text-green-400" : "text-red-400"
                )}>
                  {formatPercent(period.accuracy * 100)}
                </td>
                <td className="text-center p-2 text-cyan-400">{formatPercent(period.f1_macro * 100)}</td>
                <td className="text-center p-2 text-white">
                  {period.win_rate !== undefined ? formatPercent(period.win_rate * 100) : 'N/A'}
                </td>
                <td className={cn(
                  "text-center p-2 font-medium",
                  (period.pnl_percent || 0) >= 0 ? "text-green-400" : "text-red-400"
                )}>
                  {period.pnl_percent !== undefined
                    ? `${(period.pnl_percent * 100) >= 0 ? '+' : ''}${formatPercent(period.pnl_percent * 100)}`
                    : 'N/A'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

export default WalkForwardChart;
