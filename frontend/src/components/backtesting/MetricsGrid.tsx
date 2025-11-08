// frontend/src/components/backtesting/MetricsGrid.tsx

import type { LucideIcon } from 'lucide-react';
import { TrendingUp, TrendingDown, Activity, Target, AlertTriangle, BarChart3, DollarSign,
  // Percent
} from 'lucide-react';
import { Card } from '../ui/Card';
import { cn } from '../../utils/helpers';
import type { PerformanceMetrics } from '../../api/backtesting.api';

interface MetricsGridProps {
  metrics: PerformanceMetrics;
  initialCapital: number;
  finalCapital?: number;
}

interface MetricCardData {
  label: string;
  value: string | number;
  change?: number;
  icon: LucideIcon;
  color: 'green' | 'red' | 'blue' | 'yellow' | 'purple';
  description?: string;
}

export function MetricsGrid({ metrics, initialCapital, finalCapital }: MetricsGridProps) {
  const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  const formatNumber = (value: number, decimals = 2) => value.toFixed(decimals);
  const formatMoney = (value: number) => `$${value.toFixed(2)}`;

  const metricsData: MetricCardData[] = [
    // Returns
    {
      label: 'Total Return',
      value: formatMoney(metrics.returns.total_return),
      change: metrics.returns.total_return_pct,
      icon: DollarSign,
      color: metrics.returns.total_return >= 0 ? 'green' : 'red',
      description: `${formatPercent(metrics.returns.total_return_pct)}`
    },
    {
      label: 'Annual Return',
      value: formatPercent(metrics.returns.annual_return_pct),
      icon: TrendingUp,
      color: metrics.returns.annual_return_pct >= 0 ? 'green' : 'red',
      description: 'Annualized'
    },
    // Risk-Adjusted
    {
      label: 'Sharpe Ratio',
      value: formatNumber(metrics.risk.sharpe_ratio),
      icon: Activity,
      color: metrics.risk.sharpe_ratio >= 1 ? 'green' : metrics.risk.sharpe_ratio >= 0 ? 'yellow' : 'red',
      description: 'Risk-adjusted return'
    },
    {
      label: 'Sortino Ratio',
      value: formatNumber(metrics.risk.sortino_ratio),
      icon: Target,
      color: metrics.risk.sortino_ratio >= 1 ? 'green' : metrics.risk.sortino_ratio >= 0 ? 'yellow' : 'red',
      description: 'Downside risk'
    },
    {
      label: 'Calmar Ratio',
      value: formatNumber(metrics.risk.calmar_ratio),
      icon: BarChart3,
      color: metrics.risk.calmar_ratio >= 1 ? 'green' : metrics.risk.calmar_ratio >= 0 ? 'yellow' : 'red',
      description: 'Return / Max DD'
    },
    // Drawdown
    {
      label: 'Max Drawdown',
      value: formatPercent(metrics.drawdown.max_drawdown_pct),
      icon: TrendingDown,
      color: 'red',
      description: `${formatNumber(metrics.drawdown.max_drawdown_duration_days)} days`
    },
    {
      label: 'Avg Drawdown',
      value: formatPercent(metrics.drawdown.avg_drawdown_pct),
      icon: AlertTriangle,
      color: 'yellow',
      description: 'Average'
    },
    // Trade Stats
    {
      label: 'Win Rate',
      value: formatPercent(metrics.trade_stats.win_rate_pct),
      icon: Target,
      color: metrics.trade_stats.win_rate_pct >= 50 ? 'green' : 'red',
      description: `${metrics.trade_stats.winning_trades}/${metrics.trade_stats.total_trades} trades`
    },
    {
      label: 'Profit Factor',
      value: formatNumber(metrics.trade_stats.profit_factor),
      icon: TrendingUp,
      color: metrics.trade_stats.profit_factor >= 1.5 ? 'green' : metrics.trade_stats.profit_factor >= 1 ? 'yellow' : 'red',
      description: 'Gross profit / Gross loss'
    },
    {
      label: 'Total Trades',
      value: metrics.trade_stats.total_trades,
      icon: BarChart3,
      color: 'blue',
      description: `W: ${metrics.trade_stats.winning_trades}, L: ${metrics.trade_stats.losing_trades}`
    },
    {
      label: 'Avg Win',
      value: formatMoney(metrics.trade_stats.avg_win),
      icon: TrendingUp,
      color: 'green',
      description: 'Average winning trade'
    },
    {
      label: 'Avg Loss',
      value: formatMoney(Math.abs(metrics.trade_stats.avg_loss)),
      icon: TrendingDown,
      color: 'red',
      description: 'Average losing trade'
    },
    // Advanced
    {
      label: 'Omega Ratio',
      value: formatNumber(metrics.advanced.omega_ratio),
      icon: Activity,
      color: metrics.advanced.omega_ratio >= 1 ? 'green' : 'red',
      description: 'Upside/Downside probability'
    },
    {
      label: 'Tail Ratio',
      value: formatNumber(metrics.advanced.tail_ratio),
      icon: AlertTriangle,
      color: metrics.advanced.tail_ratio >= 1 ? 'green' : 'red',
      description: '95th/5th percentile'
    },
    {
      label: 'VaR (95%)',
      value: formatPercent(metrics.advanced.var_95),
      icon: TrendingDown,
      color: 'red',
      description: 'Value at Risk'
    },
    {
      label: 'Stability',
      value: formatPercent(metrics.advanced.stability * 100),
      icon: Activity,
      color: metrics.advanced.stability >= 0.8 ? 'green' : metrics.advanced.stability >= 0.5 ? 'yellow' : 'red',
      description: 'R² of equity curve'
    },
  ];

  const getColorClasses = (color: string) => {
    switch (color) {
      case 'green':
        return 'text-green-400 bg-green-500/10';
      case 'red':
        return 'text-red-400 bg-red-500/10';
      case 'blue':
        return 'text-blue-400 bg-blue-500/10';
      case 'yellow':
        return 'text-yellow-400 bg-yellow-500/10';
      case 'purple':
        return 'text-purple-400 bg-purple-500/10';
      default:
        return 'text-gray-400 bg-gray-500/10';
    }
  };

  return (
    <div className="space-y-6">
      {/* Summary Card */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Performance Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-1">
            <p className="text-sm text-gray-400">Initial Capital</p>
            <p className="text-2xl font-bold text-white">{formatMoney(initialCapital)}</p>
          </div>
          <div className="space-y-1">
            <p className="text-sm text-gray-400">Final Capital</p>
            <p className="text-2xl font-bold text-white">{formatMoney(finalCapital || initialCapital)}</p>
          </div>
          <div className="space-y-1">
            <p className="text-sm text-gray-400">Net Profit</p>
            <p className={cn(
              "text-2xl font-bold",
              metrics.returns.total_return >= 0 ? "text-green-400" : "text-red-400"
            )}>
              {formatMoney(metrics.returns.total_return)} ({formatPercent(metrics.returns.total_return_pct)})
            </p>
          </div>
        </div>
      </Card>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metricsData.map((metric, index) => {
          const Icon = metric.icon;
          const colorClasses = getColorClasses(metric.color);

          return (
            <Card key={index} className="p-4 hover:border-gray-700 transition-colors">
              <div className="flex items-start justify-between mb-3">
                <div className={cn("p-2 rounded-lg", colorClasses)}>
                  <Icon className="h-5 w-5" />
                </div>
                {metric.change !== undefined && (
                  <span className={cn(
                    "text-xs font-medium px-2 py-1 rounded",
                    metric.change >= 0 ? "text-green-400 bg-green-500/10" : "text-red-400 bg-red-500/10"
                  )}>
                    {formatPercent(metric.change)}
                  </span>
                )}
              </div>
              <div className="space-y-1">
                <p className="text-sm text-gray-400">{metric.label}</p>
                <p className="text-2xl font-bold text-white">{metric.value}</p>
                {metric.description && (
                  <p className="text-xs text-gray-500">{metric.description}</p>
                )}
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
