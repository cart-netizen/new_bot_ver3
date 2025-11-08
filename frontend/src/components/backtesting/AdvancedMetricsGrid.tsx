// frontend/src/components/backtesting/AdvancedMetricsGrid.tsx

import { useState } from 'react';
import type { LucideIcon } from 'lucide-react';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Target,
  AlertTriangle,
  BarChart3,
  DollarSign,
  Percent,
  Award,
  Clock,
  Layers,
  PieChart
} from 'lucide-react';
import { Card } from '../ui/Card';
import { cn } from '../../utils/helpers';
import type { PerformanceMetrics } from '../../api/backtesting.api';

interface AdvancedMetrics {
  // Risk-Adjusted
  sortino_ratio: number;
  calmar_ratio: number;
  omega_ratio: number;

  // Consistency
  profit_factor: number;
  expectancy: number;
  kelly_criterion: number;
  monthly_win_rate: number;
  win_loss_ratio: number;
  consecutive_wins_max: number;
  consecutive_losses_max: number;

  // Drawdown Extended
  avg_drawdown: number;
  avg_drawdown_pct: number;
  avg_drawdown_duration_days: number;
  recovery_factor: number;
  ulcer_index: number;

  // Market Exposure
  market_exposure_pct: number;
  avg_trade_duration_hours: number;

  // Distribution
  returns_skewness: number;
  returns_kurtosis: number;
  tail_ratio: number;
}

interface AdvancedMetricsGridProps {
  metrics: PerformanceMetrics;
}

export function AdvancedMetricsGrid({ metrics }: AdvancedMetricsGridProps) {
  const [activeTab, setActiveTab] = useState<'risk' | 'consistency' | 'drawdown' | 'exposure'>('risk');

  const formatNumber = (value: number, decimals = 2) => value?.toFixed(decimals) || '0.00';
  const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${formatNumber(value, 2)}%`;

  // Extract advanced metrics
  const advanced = metrics?.advanced || {};
  const drawdown = metrics?.drawdown || {};
  const tradeStats = metrics?.trade_stats || {};

  const tabs = [
    { id: 'risk', label: 'Risk-Adjusted', icon: Activity },
    { id: 'consistency', label: 'Consistency', icon: Target },
    { id: 'drawdown', label: 'Drawdown Analysis', icon: TrendingDown },
    { id: 'exposure', label: 'Market Exposure', icon: Clock }
  ] as const;

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-blue-400" />
          Расширенные метрики
        </h3>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 border-b border-gray-800">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "flex items-center gap-2 px-4 py-2 font-medium transition-colors border-b-2",
                activeTab === tab.id
                  ? "text-white border-blue-500"
                  : "text-gray-400 border-transparent hover:text-white"
              )}
            >
              <Icon className="h-4 w-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab Content */}
      {activeTab === 'risk' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <MetricCard
            label="Sortino Ratio"
            value={formatNumber(advanced.sortino_ratio, 3)}
            description="Downside volatility only"
            icon={Activity}
            color={getColorByValue(advanced.sortino_ratio, 1, 0)}
            tooltip="Measures risk-adjusted returns using only downside volatility. >1.0 is good, >2.0 is excellent."
          />
          <MetricCard
            label="Calmar Ratio"
            value={formatNumber(advanced.calmar_ratio, 3)}
            description="Return / Max DD"
            icon={TrendingUp}
            color={getColorByValue(advanced.calmar_ratio, 1, 0)}
            tooltip="Annual return divided by maximum drawdown. Higher is better. >3.0 is excellent."
          />
          <MetricCard
            label="Omega Ratio"
            value={formatNumber(advanced.omega_ratio, 3)}
            description="Probability-weighted"
            icon={PieChart}
            color={getColorByValue(advanced.omega_ratio, 1, 0)}
            tooltip="Ratio of probability-weighted gains to losses. >1.0 means more gains than losses."
          />
          <MetricCard
            label="Skewness"
            value={formatNumber(advanced.returns_skewness, 3)}
            description="Distribution asymmetry"
            icon={Layers}
            color={advanced.returns_skewness > 0 ? 'green' : 'red'}
            tooltip="Positive skew = more large gains. Negative skew = more large losses."
          />
          <MetricCard
            label="Kurtosis"
            value={formatNumber(advanced.returns_kurtosis, 3)}
            description="Tail thickness"
            icon={AlertTriangle}
            color={Math.abs(advanced.returns_kurtosis) > 3 ? 'yellow' : 'blue'}
            tooltip="Measures tail risk. High kurtosis = more extreme events."
          />
          <MetricCard
            label="Tail Ratio"
            value={formatNumber(advanced.tail_ratio, 2)}
            description="95th / 5th percentile"
            icon={BarChart3}
            color="blue"
            tooltip="Ratio of 95th to 5th percentile returns. Measures symmetry of extremes."
          />
        </div>
      )}

      {activeTab === 'consistency' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <MetricCard
            label="Expectancy"
            value={`$${formatNumber(advanced.expectancy, 2)}`}
            description="Expected value per trade"
            icon={DollarSign}
            color={advanced.expectancy > 0 ? 'green' : 'red'}
            tooltip="Average profit/loss per trade. Positive = profitable system."
          />
          <MetricCard
            label="Kelly Criterion"
            value={formatPercent(advanced.kelly_criterion * 100)}
            description="Optimal position size"
            icon={Award}
            color="blue"
            tooltip="Theoretically optimal position size based on win rate and win/loss ratio."
          />
          <MetricCard
            label="Monthly Win Rate"
            value={formatPercent(advanced.monthly_win_rate)}
            description="% profitable months"
            icon={Target}
            color={getColorByValue(advanced.monthly_win_rate, 60, 40)}
            tooltip="Percentage of months with positive returns. Indicates consistency."
          />
          <MetricCard
            label="Win/Loss Ratio"
            value={formatNumber(tradeStats.win_loss_ratio, 2)}
            description="Avg win / Avg loss"
            icon={TrendingUp}
            color={getColorByValue(tradeStats.win_loss_ratio, 1.5, 1)}
            tooltip="Average winning trade divided by average losing trade. >1.0 is good."
          />
          <MetricCard
            label="Max Consecutive Wins"
            value={tradeStats.consecutive_wins_max}
            description="Winning streak"
            icon={Award}
            color="green"
            tooltip="Longest streak of consecutive winning trades."
          />
          <MetricCard
            label="Max Consecutive Losses"
            value={tradeStats.consecutive_losses_max}
            description="Losing streak"
            icon={AlertTriangle}
            color="red"
            tooltip="Longest streak of consecutive losing trades. Monitor for psychological impact."
          />
        </div>
      )}

      {activeTab === 'drawdown' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <MetricCard
            label="Average Drawdown"
            value={`$${formatNumber(drawdown.avg_drawdown, 2)}`}
            description={formatPercent(drawdown.avg_drawdown_pct)}
            icon={TrendingDown}
            color="yellow"
            tooltip="Average size of all drawdowns during the backtest period."
          />
          <MetricCard
            label="Avg DD Duration"
            value={`${formatNumber(drawdown.avg_drawdown_duration_days, 1)} days`}
            description="Average recovery time"
            icon={Clock}
            color="yellow"
            tooltip="Average time to recover from drawdowns."
          />
          <MetricCard
            label="Max DD Duration"
            value={`${formatNumber(drawdown.max_drawdown_duration_days, 1)} days`}
            description="Longest recovery time"
            icon={AlertTriangle}
            color="red"
            tooltip="Maximum time spent in drawdown. Important for psychological endurance."
          />
          <MetricCard
            label="Recovery Factor"
            value={formatNumber(drawdown.recovery_factor, 2)}
            description="Net profit / Max DD"
            icon={TrendingUp}
            color={getColorByValue(drawdown.recovery_factor, 3, 1)}
            tooltip="How many times net profit covers max drawdown. >3 is excellent."
          />
          <MetricCard
            label="Ulcer Index"
            value={formatNumber(advanced.ulcer_index, 2)}
            description="DD severity measure"
            icon={Activity}
            color={advanced.ulcer_index < 5 ? 'green' : advanced.ulcer_index < 10 ? 'yellow' : 'red'}
            tooltip="Measures depth and duration of drawdowns. Lower is better."
          />
        </div>
      )}

      {activeTab === 'exposure' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <MetricCard
            label="Market Exposure"
            value={formatPercent(advanced.market_exposure_pct)}
            description="% of time in market"
            icon={Clock}
            color="blue"
            tooltip="Percentage of backtest period with open positions."
          />
          <MetricCard
            label="Avg Trade Duration"
            value={`${formatNumber(advanced.avg_trade_duration_hours, 1)}h`}
            description="Average hold time"
            icon={Activity}
            color="blue"
            tooltip="Average time each trade is held before closing."
          />
        </div>
      )}
    </Card>
  );
}

// Helper Components
interface MetricCardProps {
  label: string;
  value: string | number;
  description: string;
  icon: LucideIcon;
  color: 'green' | 'red' | 'blue' | 'yellow' | 'purple';
  tooltip?: string;
}

function MetricCard({ label, value, description, icon: Icon, color, tooltip }: MetricCardProps) {
  const colorClasses = {
    green: 'bg-green-500/10 text-green-400 border-green-500/20',
    red: 'bg-red-500/10 text-red-400 border-red-500/20',
    blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    yellow: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
    purple: 'bg-purple-500/10 text-purple-400 border-purple-500/20'
  };

  const iconColorClasses = {
    green: 'text-green-400',
    red: 'text-red-400',
    blue: 'text-blue-400',
    yellow: 'text-yellow-400',
    purple: 'text-purple-400'
  };

  return (
    <div
      className={cn(
        "p-4 rounded-lg border transition-all hover:scale-105",
        colorClasses[color]
      )}
      title={tooltip}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1">
          <p className="text-xs font-medium text-gray-400 uppercase tracking-wide">
            {label}
          </p>
        </div>
        <Icon className={cn("h-5 w-5", iconColorClasses[color])} />
      </div>
      <div className="mt-2">
        <p className="text-2xl font-bold text-white">{value}</p>
        <p className="text-sm text-gray-400 mt-1">{description}</p>
      </div>
    </div>
  );
}

function getColorByValue(
  value: number,
  goodThreshold: number,
  okThreshold: number
): 'green' | 'yellow' | 'red' {
  if (value >= goodThreshold) return 'green';
  if (value >= okThreshold) return 'yellow';
  return 'red';
}
