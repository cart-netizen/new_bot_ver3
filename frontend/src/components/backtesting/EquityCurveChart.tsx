// frontend/src/components/backtesting/EquityCurveChart.tsx

import { useMemo } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { Card } from '../ui/Card';
import { cn } from '../../utils/helpers';
import { EquityPoint } from '../../api/backtesting.api';

interface EquityCurveChartProps {
  equityCurve: EquityPoint[];
  initialCapital: number;
}

export function EquityCurveChart({ equityCurve, initialCapital }: EquityCurveChartProps) {
  const { chartData, stats } = useMemo(() => {
    if (!equityCurve.length) {
      return {
        chartData: { points: [], minY: 0, maxY: 0, width: 800, height: 300 },
        stats: { peak: 0, valley: 0, current: initialCapital }
      };
    }

    const values = equityCurve.map(p => p.equity);
    const minY = Math.min(...values, initialCapital);
    const maxY = Math.max(...values, initialCapital);
    const range = maxY - minY || 1;

    const width = 800;
    const height = 300;
    const padding = 20;

    const points = equityCurve.map((point, index) => {
      const x = padding + (index / (equityCurve.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((point.equity - minY) / range) * (height - 2 * padding);
      return { x, y, equity: point.equity, timestamp: point.timestamp };
    });

    return {
      chartData: { points, minY, maxY, width, height },
      stats: {
        peak: Math.max(...values),
        valley: Math.min(...values),
        current: values[values.length - 1]
      }
    };
  }, [equityCurve, initialCapital]);

  const { points, minY, maxY, width, height } = chartData;

  const pathD = points.length > 0
    ? `M ${points.map(p => `${p.x},${p.y}`).join(' L ')}`
    : '';

  const formatMoney = (value: number) => `$${value.toFixed(2)}`;
  const calculateChange = (from: number, to: number) => ((to - from) / from) * 100;

  const isPositive = stats.current >= initialCapital;
  const totalChange = calculateChange(initialCapital, stats.current);

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white mb-1">Equity Curve</h3>
          <p className="text-sm text-gray-400">Portfolio value over time</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="text-sm text-gray-400">Current</p>
            <p className={cn(
              "text-xl font-bold",
              isPositive ? "text-green-400" : "text-red-400"
            )}>
              {formatMoney(stats.current)}
            </p>
          </div>
          <div className={cn(
            "flex items-center gap-1 px-3 py-1 rounded-lg",
            isPositive ? "bg-green-500/10 text-green-400" : "bg-red-500/10 text-red-400"
          )}>
            {isPositive ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
            <span className="font-medium">{isPositive ? '+' : ''}{totalChange.toFixed(2)}%</span>
          </div>
        </div>
      </div>

      {/* Chart */}
      {points.length > 0 ? (
        <div className="relative">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full"
            style={{ maxHeight: '400px' }}
          >
            {/* Grid lines */}
            <g className="opacity-10">
              {[0, 0.25, 0.5, 0.75, 1].map((ratio, i) => {
                const y = height - 20 - ratio * (height - 40);
                return (
                  <line
                    key={i}
                    x1="20"
                    y1={y}
                    x2={width - 20}
                    y2={y}
                    stroke="currentColor"
                    className="text-gray-400"
                    strokeWidth="1"
                  />
                );
              })}
            </g>

            {/* Baseline (initial capital) */}
            <line
              x1="20"
              y1={height - 20 - ((initialCapital - minY) / (maxY - minY)) * (height - 40)}
              x2={width - 20}
              y2={height - 20 - ((initialCapital - minY) / (maxY - minY)) * (height - 40)}
              stroke="currentColor"
              className="text-gray-600"
              strokeWidth="1"
              strokeDasharray="4 4"
            />

            {/* Equity curve */}
            <path
              d={pathD}
              fill="none"
              stroke="currentColor"
              className={isPositive ? "text-green-400" : "text-red-400"}
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Area fill */}
            {points.length > 0 && (
              <path
                d={`${pathD} L ${points[points.length - 1].x},${height - 20} L 20,${height - 20} Z`}
                fill="currentColor"
                className={isPositive ? "text-green-400" : "text-red-400"}
                opacity="0.1"
              />
            )}

            {/* Y-axis labels */}
            <g className="text-xs text-gray-400">
              {[maxY, (maxY + minY) / 2, minY].map((value, i) => {
                const y = 20 + i * ((height - 40) / 2);
                return (
                  <text
                    key={i}
                    x="10"
                    y={y}
                    textAnchor="end"
                    dominantBaseline="middle"
                    className="fill-current"
                  >
                    {formatMoney(value)}
                  </text>
                );
              })}
            </g>
          </svg>
        </div>
      ) : (
        <div className="flex items-center justify-center h-64 text-gray-500">
          No equity curve data available
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mt-6 pt-6 border-t border-gray-800">
        <div>
          <p className="text-sm text-gray-400 mb-1">Initial</p>
          <p className="text-lg font-semibold text-white">{formatMoney(initialCapital)}</p>
        </div>
        <div>
          <p className="text-sm text-gray-400 mb-1">Peak</p>
          <p className="text-lg font-semibold text-green-400">{formatMoney(stats.peak)}</p>
        </div>
        <div>
          <p className="text-sm text-gray-400 mb-1">Valley</p>
          <p className="text-lg font-semibold text-red-400">{formatMoney(stats.valley)}</p>
        </div>
      </div>
    </Card>
  );
}
