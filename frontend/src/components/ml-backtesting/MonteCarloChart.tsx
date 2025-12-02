// frontend/src/components/ml-backtesting/MonteCarloChart.tsx

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';
import { RefreshCw, Play, TrendingUp, AlertTriangle } from 'lucide-react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import * as mlBacktestingApi from '../../api/ml-backtesting.api';

interface MonteCarloChartProps {
  backtestId: string;
}

export function MonteCarloChart({ backtestId }: MonteCarloChartProps) {
  const [nSimulations, setNSimulations] = useState(1000);

  const mutation = useMutation({
    mutationFn: () => mlBacktestingApi.runMonteCarloSimulation(backtestId, { n_simulations: nSimulations })
  });

  const formatCurrency = (value: number) => `$${value.toLocaleString()}`;
  const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;

  // Prepare chart data from percentile paths
  const chartData = mutation.data?.percentile_paths?.p50?.map((_, i) => ({
    x: i,
    p5: mutation.data?.percentile_paths?.p5?.[i] || 0,
    p25: mutation.data?.percentile_paths?.p25?.[i] || 0,
    p50: mutation.data?.percentile_paths?.p50?.[i] || 0,
    p75: mutation.data?.percentile_paths?.p75?.[i] || 0,
    p95: mutation.data?.percentile_paths?.p95?.[i] || 0
  })) || [];

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Monte Carlo Simulation</h3>
        <div className="flex items-center gap-3">
          <select
            value={nSimulations}
            onChange={(e) => setNSimulations(parseInt(e.target.value))}
            className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
          >
            <option value={500}>500 sims</option>
            <option value={1000}>1,000 sims</option>
            <option value={2000}>2,000 sims</option>
            <option value={5000}>5,000 sims</option>
          </select>
          <Button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            className="px-3 py-1.5 text-sm"
          >
            {mutation.isPending ? (
              <RefreshCw className="h-4 w-4 animate-spin mr-1" />
            ) : (
              <Play className="h-4 w-4 mr-1" />
            )}
            Run
          </Button>
        </div>
      </div>

      {!mutation.data && !mutation.isPending && (
        <div className="h-64 flex items-center justify-center text-gray-400">
          Click "Run" to generate Monte Carlo simulation
        </div>
      )}

      {mutation.isPending && (
        <div className="h-64 flex items-center justify-center">
          <RefreshCw className="h-8 w-8 animate-spin text-purple-400" />
          <span className="ml-2 text-gray-400">Running {nSimulations} simulations...</span>
        </div>
      )}

      {mutation.data && (
        <>
          {/* Chart */}
          <div className="h-64 mb-6">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="x" stroke="#9CA3AF" fontSize={12} />
                <YAxis stroke="#9CA3AF" fontSize={12} tickFormatter={formatCurrency} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number, name: string) => [formatCurrency(value), name]}
                />
                <Area
                  type="monotone"
                  dataKey="p95"
                  fill="#10B981"
                  fillOpacity={0.1}
                  stroke="#10B981"
                  strokeWidth={1}
                  name="95th Percentile"
                />
                <Area
                  type="monotone"
                  dataKey="p75"
                  fill="#10B981"
                  fillOpacity={0.2}
                  stroke="#10B981"
                  strokeWidth={1}
                  name="75th Percentile"
                />
                <Line
                  type="monotone"
                  dataKey="p50"
                  stroke="#8B5CF6"
                  strokeWidth={2}
                  dot={false}
                  name="Median"
                />
                <Area
                  type="monotone"
                  dataKey="p25"
                  fill="#EF4444"
                  fillOpacity={0.2}
                  stroke="#EF4444"
                  strokeWidth={1}
                  name="25th Percentile"
                />
                <Area
                  type="monotone"
                  dataKey="p5"
                  fill="#EF4444"
                  fillOpacity={0.1}
                  stroke="#EF4444"
                  strokeWidth={1}
                  name="5th Percentile"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Statistics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-gray-800/50 rounded-lg text-center">
              <div className="flex items-center justify-center gap-2 mb-1">
                <TrendingUp className="h-4 w-4 text-green-400" />
                <span className="text-green-400 font-bold text-xl">
                  {formatPercent(mutation.data.probability_of_profit)}
                </span>
              </div>
              <p className="text-xs text-gray-400">Prob. of Profit</p>
            </div>

            <div className="p-4 bg-gray-800/50 rounded-lg text-center">
              <div className="flex items-center justify-center gap-2 mb-1">
                <AlertTriangle className="h-4 w-4 text-red-400" />
                <span className="text-red-400 font-bold text-xl">
                  {formatPercent(mutation.data.probability_of_ruin)}
                </span>
              </div>
              <p className="text-xs text-gray-400">Prob. of Ruin</p>
            </div>

            <div className="p-4 bg-gray-800/50 rounded-lg text-center">
              <span className="text-yellow-400 font-bold text-xl">
                {formatPercent(mutation.data.var_95)}
              </span>
              <p className="text-xs text-gray-400">VaR (95%)</p>
            </div>

            <div className="p-4 bg-gray-800/50 rounded-lg text-center">
              <span className="text-orange-400 font-bold text-xl">
                {formatPercent(mutation.data.cvar_95)}
              </span>
              <p className="text-xs text-gray-400">CVaR (95%)</p>
            </div>
          </div>

          {/* Final Equity Distribution */}
          <div className="mt-4 p-4 bg-gray-800/30 rounded-lg">
            <h4 className="text-sm font-medium text-white mb-3">Final Equity Distribution</h4>
            <div className="grid grid-cols-5 gap-2 text-center text-sm">
              <div>
                <p className="text-red-400 font-medium">{formatCurrency(mutation.data.final_equity.percentile_5)}</p>
                <p className="text-xs text-gray-500">5th %ile</p>
              </div>
              <div>
                <p className="text-orange-400 font-medium">{formatCurrency(mutation.data.final_equity.percentile_25)}</p>
                <p className="text-xs text-gray-500">25th %ile</p>
              </div>
              <div>
                <p className="text-purple-400 font-bold">{formatCurrency(mutation.data.final_equity.percentile_50)}</p>
                <p className="text-xs text-gray-500">Median</p>
              </div>
              <div>
                <p className="text-blue-400 font-medium">{formatCurrency(mutation.data.final_equity.percentile_75)}</p>
                <p className="text-xs text-gray-500">75th %ile</p>
              </div>
              <div>
                <p className="text-green-400 font-medium">{formatCurrency(mutation.data.final_equity.percentile_95)}</p>
                <p className="text-xs text-gray-500">95th %ile</p>
              </div>
            </div>
          </div>
        </>
      )}
    </Card>
  );
}

export default MonteCarloChart;
