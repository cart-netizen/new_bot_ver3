// frontend/src/components/ml-backtesting/PBOAnalysisCard.tsx

import { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ReferenceLine,
  Cell
} from 'recharts';
import { AlertTriangle, CheckCircle, Info, TrendingUp } from 'lucide-react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { cn } from '../../utils/helpers';
import { getPBOAnalysis, type PBOAnalysis } from '../../api/ml-backtesting.api';

interface PBOAnalysisCardProps {
  backtestId: string;
}

export function PBOAnalysisCard({ backtestId }: PBOAnalysisCardProps) {
  const [data, setData] = useState<PBOAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getPBOAnalysis(backtestId);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run PBO analysis');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-400';
      case 'moderate': return 'text-yellow-400';
      case 'high': return 'text-orange-400';
      case 'very_high': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getRiskBgColor = (level: string) => {
    switch (level) {
      case 'low': return 'bg-green-500/10 border-green-500/30';
      case 'moderate': return 'bg-yellow-500/10 border-yellow-500/30';
      case 'high': return 'bg-orange-500/10 border-orange-500/30';
      case 'very_high': return 'bg-red-500/10 border-red-500/30';
      default: return 'bg-gray-500/10 border-gray-500/30';
    }
  };

  const getRiskLabel = (level: string) => {
    switch (level) {
      case 'low': return 'Low Overfit Risk';
      case 'moderate': return 'Moderate Overfit Risk';
      case 'high': return 'High Overfit Risk';
      case 'very_high': return 'Very High Overfit Risk';
      default: return 'Unknown';
    }
  };

  if (!data) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-white">Probability of Backtest Overfitting (PBO)</h3>
            <p className="text-sm text-gray-400 mt-1">
              Lopez de Prado methodology for detecting strategy overfitting
            </p>
          </div>
          <Button onClick={runAnalysis} disabled={loading}>
            {loading ? 'Analyzing...' : 'Run PBO Analysis'}
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
              <Info className="h-8 w-8 mx-auto mb-2 text-gray-500" />
              <p>Click "Run PBO Analysis" to evaluate overfitting probability</p>
              <p className="text-xs mt-1">Uses Combinatorial Purged Cross-Validation (CPCV)</p>
            </div>
          </div>
        )}
      </Card>
    );
  }

  // Prepare scatter data for IS vs OOS Sharpe
  const scatterData = data.is_sharpe_ratios.map((is_sharpe, idx) => ({
    is_sharpe,
    oos_sharpe: data.oos_sharpe_ratios[idx],
    idx
  }));

  // Prepare histogram data for OOS distribution
  const histogramBins = 20;
  const oosMin = Math.min(...data.oos_sharpe_ratios);
  const oosMax = Math.max(...data.oos_sharpe_ratios);
  const binWidth = (oosMax - oosMin) / histogramBins;

  const histogramData: { bin: string; count: number; isNegative: boolean }[] = [];
  for (let i = 0; i < histogramBins; i++) {
    const binStart = oosMin + i * binWidth;
    const binEnd = binStart + binWidth;
    const count = data.oos_sharpe_ratios.filter(s => s >= binStart && s < binEnd).length;
    histogramData.push({
      bin: binStart.toFixed(2),
      count,
      isNegative: binStart < 0
    });
  }

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white">Probability of Backtest Overfitting (PBO)</h3>
          <p className="text-sm text-gray-400 mt-1">
            Based on {data.n_combinations} CPCV combinations
          </p>
        </div>
        <Button variant="secondary" size="sm" onClick={runAnalysis} disabled={loading}>
          Refresh
        </Button>
      </div>

      {/* Main PBO Result */}
      <div className={cn(
        "p-6 rounded-lg border mb-6",
        getRiskBgColor(data.risk_level)
      )}>
        <div className="flex items-start gap-4">
          {data.is_overfit ? (
            <AlertTriangle className={cn("h-8 w-8 flex-shrink-0", getRiskColor(data.risk_level))} />
          ) : (
            <CheckCircle className="h-8 w-8 text-green-400 flex-shrink-0" />
          )}
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <span className={cn("text-3xl font-bold", getRiskColor(data.risk_level))}>
                {(data.pbo * 100).toFixed(1)}%
              </span>
              <span className={cn(
                "px-3 py-1 rounded-full text-sm font-medium",
                getRiskBgColor(data.risk_level),
                getRiskColor(data.risk_level)
              )}>
                {getRiskLabel(data.risk_level)}
              </span>
            </div>
            <p className="text-gray-300">{data.interpretation}</p>
          </div>
        </div>
      </div>

      {/* Statistics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="p-4 bg-gray-800/50 rounded-lg">
          <p className="text-xs text-gray-400 mb-1">PBO (Raw)</p>
          <p className="text-xl font-bold text-white">{(data.pbo * 100).toFixed(1)}%</p>
        </div>
        <div className="p-4 bg-gray-800/50 rounded-lg">
          <p className="text-xs text-gray-400 mb-1">PBO (Adjusted)</p>
          <p className="text-xl font-bold text-white">{(data.pbo_adjusted * 100).toFixed(1)}%</p>
        </div>
        <div className="p-4 bg-gray-800/50 rounded-lg">
          <p className="text-xs text-gray-400 mb-1">Rank Correlation</p>
          <p className={cn(
            "text-xl font-bold",
            data.rank_correlation > 0.3 ? "text-green-400" :
            data.rank_correlation > 0 ? "text-yellow-400" : "text-red-400"
          )}>
            {data.rank_correlation.toFixed(3)}
          </p>
        </div>
        <div className="p-4 bg-gray-800/50 rounded-lg">
          <p className="text-xs text-gray-400 mb-1">Confidence Level</p>
          <p className="text-xl font-bold text-purple-400">{(data.confidence_level * 100).toFixed(0)}%</p>
        </div>
      </div>

      {/* Best In-Sample Performance */}
      <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg mb-6">
        <div className="flex items-center gap-2 mb-2">
          <TrendingUp className="h-5 w-5 text-blue-400" />
          <span className="text-sm font-medium text-blue-400">Best In-Sample Configuration</span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-xs text-gray-400">IS Sharpe</p>
            <p className="text-lg font-bold text-white">{data.best_is_sharpe.toFixed(3)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">OOS Sharpe</p>
            <p className={cn(
              "text-lg font-bold",
              data.best_is_oos_sharpe >= 0 ? "text-green-400" : "text-red-400"
            )}>
              {data.best_is_oos_sharpe.toFixed(3)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-400">OOS Rank</p>
            <p className="text-lg font-bold text-white">
              {data.best_is_oos_rank} / {data.n_combinations}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-400">Degradation</p>
            <p className={cn(
              "text-lg font-bold",
              data.best_is_sharpe > 0 && data.best_is_oos_sharpe < data.best_is_sharpe
                ? "text-orange-400" : "text-green-400"
            )}>
              {data.best_is_sharpe > 0
                ? `${(((data.best_is_sharpe - data.best_is_oos_sharpe) / data.best_is_sharpe) * 100).toFixed(0)}%`
                : 'N/A'}
            </p>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* IS vs OOS Scatter Plot */}
        <div>
          <h4 className="text-sm font-medium text-gray-400 mb-3">In-Sample vs Out-of-Sample Sharpe</h4>
          <div className="h-64 bg-gray-800/30 rounded-lg p-2">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="is_sharpe"
                  type="number"
                  name="IS Sharpe"
                  stroke="#9CA3AF"
                  fontSize={10}
                  label={{ value: 'In-Sample Sharpe', position: 'bottom', fill: '#9CA3AF', fontSize: 10 }}
                />
                <YAxis
                  dataKey="oos_sharpe"
                  type="number"
                  name="OOS Sharpe"
                  stroke="#9CA3AF"
                  fontSize={10}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number) => value.toFixed(3)}
                />
                <ReferenceLine y={0} stroke="#EF4444" strokeDasharray="3 3" />
                <ReferenceLine x={0} stroke="#EF4444" strokeDasharray="3 3" />
                <Scatter data={scatterData} fill="#8B5CF6">
                  {scatterData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.oos_sharpe >= 0 ? '#10B981' : '#EF4444'}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* OOS Sharpe Distribution */}
        <div>
          <h4 className="text-sm font-medium text-gray-400 mb-3">OOS Sharpe Distribution</h4>
          <div className="h-64 bg-gray-800/30 rounded-lg p-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogramData} margin={{ top: 10, right: 10, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="bin"
                  stroke="#9CA3AF"
                  fontSize={10}
                  interval={Math.floor(histogramBins / 5)}
                />
                <YAxis stroke="#9CA3AF" fontSize={10} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                />
                <ReferenceLine x="0.00" stroke="#F59E0B" strokeDasharray="5 5" />
                <Bar dataKey="count" name="Frequency">
                  {histogramData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.isNegative ? '#EF4444' : '#10B981'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Interpretation Guide */}
      <div className="mt-6 p-4 bg-gray-800/30 rounded-lg">
        <h4 className="text-sm font-medium text-gray-400 mb-2">Interpretation Guide</h4>
        <ul className="text-xs text-gray-500 space-y-1">
          <li><span className="text-green-400">PBO &lt; 20%:</span> Low probability of overfitting - strategy likely robust</li>
          <li><span className="text-yellow-400">PBO 20-40%:</span> Moderate risk - use caution, consider additional validation</li>
          <li><span className="text-orange-400">PBO 40-60%:</span> High risk - likely overfit, reconsider strategy</li>
          <li><span className="text-red-400">PBO &gt; 60%:</span> Very high risk - strategy almost certainly overfit</li>
        </ul>
      </div>
    </Card>
  );
}

export default PBOAnalysisCard;
