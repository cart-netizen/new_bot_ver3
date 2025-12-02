// frontend/src/components/ml-backtesting/RegimeAnalysisCard.tsx

import { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  PieChart,
  Pie,
  Cell,
  Legend
} from 'recharts';
import { TrendingUp, TrendingDown, Activity, Zap, RefreshCw } from 'lucide-react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { cn } from '../../utils/helpers';
import { getRegimeAnalysis, type RegimeAnalysis } from '../../api/ml-backtesting.api';

interface RegimeAnalysisCardProps {
  backtestId: string;
}

const REGIME_CONFIG: Record<string, {
  icon: typeof TrendingUp;
  color: string;
  bgColor: string;
  chartColor: string;
  label: string;
}> = {
  trending_up: {
    icon: TrendingUp,
    color: 'text-green-400',
    bgColor: 'bg-green-500/10',
    chartColor: '#10B981',
    label: 'Trending Up'
  },
  trending_down: {
    icon: TrendingDown,
    color: 'text-red-400',
    bgColor: 'bg-red-500/10',
    chartColor: '#EF4444',
    label: 'Trending Down'
  },
  ranging: {
    icon: Activity,
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/10',
    chartColor: '#3B82F6',
    label: 'Ranging'
  },
  high_volatility: {
    icon: Zap,
    color: 'text-purple-400',
    bgColor: 'bg-purple-500/10',
    chartColor: '#8B5CF6',
    label: 'High Volatility'
  }
};

export function RegimeAnalysisCard({ backtestId }: RegimeAnalysisCardProps) {
  const [data, setData] = useState<RegimeAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getRegimeAnalysis(backtestId);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load regime analysis');
    } finally {
      setLoading(false);
    }
  };

  const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;

  if (!data) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-white">Market Regime Analysis</h3>
            <p className="text-sm text-gray-400 mt-1">
              Model performance breakdown by market conditions
            </p>
          </div>
          <Button onClick={loadAnalysis} disabled={loading}>
            {loading ? 'Loading...' : 'Load Analysis'}
          </Button>
        </div>

        {error && (
          <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
            {error}
          </div>
        )}

        {!error && !loading && (
          <div className="h-48 flex items-center justify-center text-gray-400 border border-dashed border-gray-700 rounded-lg">
            <div className="text-center">
              <Activity className="h-8 w-8 mx-auto mb-2 text-gray-500" />
              <p>Click to analyze performance across market regimes</p>
            </div>
          </div>
        )}
      </Card>
    );
  }

  // Prepare data for charts
  const barChartData = data.regimes.map(regime => ({
    name: REGIME_CONFIG[regime.regime]?.label || regime.display_name,
    accuracy: regime.accuracy * 100,
    win_rate: regime.win_rate * 100,
    samples: regime.n_samples,
    pnl: regime.pnl_estimate * 100,
    regime: regime.regime
  }));

  const radarData = data.regimes.map(regime => ({
    regime: REGIME_CONFIG[regime.regime]?.label || regime.display_name,
    Accuracy: regime.accuracy * 100,
    'Win Rate': regime.win_rate * 100,
    Confidence: regime.avg_confidence * 100,
    fullMark: 100
  }));

  const pieData = Object.entries(data.overall_regime_distribution).map(([regime, count]) => ({
    name: REGIME_CONFIG[regime]?.label || regime,
    value: count,
    color: REGIME_CONFIG[regime]?.chartColor || '#6B7280'
  }));

  const bestRegimeConfig = REGIME_CONFIG[data.best_regime];
  const worstRegimeConfig = REGIME_CONFIG[data.worst_regime];

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white">Market Regime Analysis</h3>
          <p className="text-sm text-gray-400 mt-1">
            Performance breakdown by market conditions
          </p>
        </div>
        <Button variant="outline" onClick={loadAnalysis} disabled={loading}>
          <RefreshCw className={cn("h-4 w-4 mr-1", loading && "animate-spin")} />
          Refresh
        </Button>
      </div>

      {/* Best & Worst Regime Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className={cn("p-4 rounded-lg border", bestRegimeConfig?.bgColor || 'bg-gray-800/50', 'border-green-500/30')}>
          <div className="flex items-center gap-2 mb-2">
            {bestRegimeConfig && <bestRegimeConfig.icon className={cn("h-5 w-5", bestRegimeConfig.color)} />}
            <span className="text-sm font-medium text-gray-400">Best Performance</span>
          </div>
          <p className={cn("text-lg font-bold", bestRegimeConfig?.color || 'text-white')}>
            {bestRegimeConfig?.label || data.best_regime}
          </p>
          {data.regime_metrics[data.best_regime] && (
            <p className="text-xs text-gray-400 mt-1">
              {formatPercent(data.regime_metrics[data.best_regime].accuracy)} accuracy
            </p>
          )}
        </div>

        <div className={cn("p-4 rounded-lg border", worstRegimeConfig?.bgColor || 'bg-gray-800/50', 'border-red-500/30')}>
          <div className="flex items-center gap-2 mb-2">
            {worstRegimeConfig && <worstRegimeConfig.icon className={cn("h-5 w-5", worstRegimeConfig.color)} />}
            <span className="text-sm font-medium text-gray-400">Worst Performance</span>
          </div>
          <p className={cn("text-lg font-bold", worstRegimeConfig?.color || 'text-white')}>
            {worstRegimeConfig?.label || data.worst_regime}
          </p>
          {data.regime_metrics[data.worst_regime] && (
            <p className="text-xs text-gray-400 mt-1">
              {formatPercent(data.regime_metrics[data.worst_regime].accuracy)} accuracy
            </p>
          )}
        </div>
      </div>

      {/* Regime Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {data.regimes.map((regime) => {
          const config = REGIME_CONFIG[regime.regime];
          const Icon = config?.icon || Activity;

          return (
            <div
              key={regime.regime}
              className={cn(
                "p-4 rounded-lg border",
                config?.bgColor || 'bg-gray-800/50',
                'border-gray-700'
              )}
            >
              <div className="flex items-center gap-2 mb-3">
                <Icon className={cn("h-5 w-5", config?.color || 'text-gray-400')} />
                <span className={cn("text-sm font-medium", config?.color || 'text-gray-400')}>
                  {config?.label || regime.display_name}
                </span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Accuracy</span>
                  <span className="text-white font-medium">{formatPercent(regime.accuracy)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Win Rate</span>
                  <span className="text-white font-medium">{formatPercent(regime.win_rate)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Confidence</span>
                  <span className="text-white font-medium">{formatPercent(regime.avg_confidence)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Samples</span>
                  <span className="text-white font-medium">{regime.n_samples.toLocaleString()}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Performance Bar Chart */}
        <div className="lg:col-span-2">
          <h4 className="text-sm font-medium text-gray-400 mb-3">Performance by Regime</h4>
          <div className="h-64 bg-gray-800/30 rounded-lg p-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barChartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#9CA3AF" fontSize={10} />
                <YAxis stroke="#9CA3AF" fontSize={10} tickFormatter={(v) => `${v}%`} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number, name: string) => [`${value.toFixed(1)}%`, name]}
                />
                <Legend />
                <Bar dataKey="accuracy" name="Accuracy" radius={[4, 4, 0, 0]}>
                  {barChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={REGIME_CONFIG[entry.regime]?.chartColor || '#6B7280'} />
                  ))}
                </Bar>
                <Bar dataKey="win_rate" name="Win Rate" fill="#F59E0B" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Distribution Pie Chart */}
        <div>
          <h4 className="text-sm font-medium text-gray-400 mb-3">Regime Distribution</h4>
          <div className="h-64 bg-gray-800/30 rounded-lg p-2">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={70}
                  paddingAngle={2}
                  label={({ name, percent }) => `${name}: ${(Number(percent ?? 0) * 100).toFixed(0)}%`}
                  labelLine={false}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Radar Chart */}
      <div className="mt-6">
        <h4 className="text-sm font-medium text-gray-400 mb-3">Multi-Metric Comparison</h4>
        <div className="h-72 bg-gray-800/30 rounded-lg p-2">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis dataKey="regime" stroke="#9CA3AF" fontSize={10} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#9CA3AF" fontSize={10} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Radar name="Accuracy" dataKey="Accuracy" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.3} />
              <Radar name="Win Rate" dataKey="Win Rate" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.3} />
              <Radar name="Confidence" dataKey="Confidence" stroke="#06B6D4" fill="#06B6D4" fillOpacity={0.3} />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Insights */}
      <div className="mt-6 p-4 bg-gray-800/30 rounded-lg">
        <h4 className="text-sm font-medium text-gray-400 mb-2">Key Insights</h4>
        <ul className="text-xs text-gray-500 space-y-1">
          <li>
            Model performs best in <span className={bestRegimeConfig?.color}>{bestRegimeConfig?.label || data.best_regime}</span> conditions
          </li>
          <li>
            Consider avoiding trades during <span className={worstRegimeConfig?.color}>{worstRegimeConfig?.label || data.worst_regime}</span> periods
          </li>
          <li>
            Regime detection can be used to dynamically adjust position sizing
          </li>
        </ul>
      </div>
    </Card>
  );
}

export default RegimeAnalysisCard;
