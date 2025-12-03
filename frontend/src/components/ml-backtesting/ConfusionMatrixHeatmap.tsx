// frontend/src/components/ml-backtesting/ConfusionMatrixHeatmap.tsx

import { useMemo } from 'react';
import { Card } from '../ui/Card';
import { Tooltip } from '../ui/Tooltip';
import { cn } from '../../utils/helpers';

interface ConfusionMatrixHeatmapProps {
  matrix: number[][];
  labels?: string[];
  title?: string;
  showPercentages?: boolean;
}

const DEFAULT_LABELS = ['SELL', 'HOLD', 'BUY'];

export function ConfusionMatrixHeatmap({
  matrix,
  labels = DEFAULT_LABELS,
  title = 'Confusion Matrix',
  showPercentages = true
}: ConfusionMatrixHeatmapProps) {
  // Calculate normalized values and stats
  const { normalizedMatrix, total, correct, perClassMetrics } = useMemo(() => {
    const total = matrix.flat().reduce((a, b) => a + b, 0);
    let correct = 0;

    // Normalize and count correct
    const normalized = matrix.map((row, i) => {
      const rowSum = row.reduce((a, b) => a + b, 0);
      correct += row[i] || 0;
      return row.map(val => rowSum > 0 ? val / rowSum : 0);
    });

    // Calculate per-class metrics
    const metrics = labels.map((label, i) => {
      const tp = matrix[i]?.[i] || 0;
      const rowSum = matrix[i]?.reduce((a, b) => a + b, 0) || 0;
      const colSum = matrix.reduce((sum, row) => sum + (row[i] || 0), 0);

      const precision = colSum > 0 ? tp / colSum : 0;
      const recall = rowSum > 0 ? tp / rowSum : 0;
      const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

      return { label, precision, recall, f1, support: rowSum };
    });

    return { normalizedMatrix: normalized, total, correct, perClassMetrics: metrics };
  }, [matrix, labels]);

  const accuracy = total > 0 ? correct / total : 0;

  // Get color based on normalized value (0-1)
  const getHeatColor = (value: number, isDiagonal: boolean) => {
    if (isDiagonal) {
      // Diagonal (correct predictions) - green scale
      if (value >= 0.8) return 'bg-green-600';
      if (value >= 0.6) return 'bg-green-500';
      if (value >= 0.4) return 'bg-green-600/60';
      if (value >= 0.2) return 'bg-green-700/40';
      return 'bg-gray-800';
    } else {
      // Off-diagonal (errors) - red scale
      if (value >= 0.3) return 'bg-red-600';
      if (value >= 0.2) return 'bg-red-500/80';
      if (value >= 0.1) return 'bg-red-600/50';
      if (value >= 0.05) return 'bg-red-700/30';
      return 'bg-gray-800';
    }
  };

  const getLabelColor = (label: string) => {
    switch (label) {
      case 'SELL': return 'text-red-400';
      case 'HOLD': return 'text-yellow-400';
      case 'BUY': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  if (!matrix || matrix.length === 0) {
    return (
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>
        <div className="h-48 flex items-center justify-center text-gray-400">
          No confusion matrix data available
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          {title}
          <Tooltip content="Confusion Matrix (матрица ошибок) — визуализация того, как модель классифицирует данные.

Строки — реальные классы (Actual)
Столбцы — предсказанные классы (Predicted)

Диагональ (зелёные ячейки) — правильные предсказания.
Остальные ячейки (красные) — ошибки.

Идеальная матрица: все значения на диагонали, остальные = 0.

Анализируйте:
• Какие классы путает модель (большие числа вне диагонали)
• Есть ли bias к какому-то классу" />
        </h3>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-gray-400">
            Accuracy: <span className="text-white font-medium">{(accuracy * 100).toFixed(1)}%</span>
          </span>
          <span className="text-gray-400">
            Samples: <span className="text-white font-medium">{total.toLocaleString()}</span>
          </span>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Confusion Matrix Grid */}
        <div className="flex-1">
          <div className="flex">
            {/* Y-axis label */}
            <div className="flex items-center justify-center w-8 -rotate-90">
              <span className="text-xs text-gray-500 whitespace-nowrap">Actual</span>
            </div>

            <div className="flex-1">
              {/* X-axis labels */}
              <div className="flex mb-1 ml-16">
                <div className="flex-1 text-center text-xs text-gray-500 mb-1">Predicted</div>
              </div>
              <div className="flex mb-2 ml-16">
                {labels.map(label => (
                  <div key={label} className="flex-1 text-center">
                    <span className={cn("text-xs font-medium", getLabelColor(label))}>
                      {label}
                    </span>
                  </div>
                ))}
              </div>

              {/* Matrix cells */}
              {matrix.map((row, i) => (
                <div key={i} className="flex items-center mb-1">
                  {/* Row label */}
                  <div className="w-16 text-right pr-2">
                    <span className={cn("text-xs font-medium", getLabelColor(labels[i]))}>
                      {labels[i]}
                    </span>
                  </div>

                  {/* Cells */}
                  {row.map((value, j) => {
                    const isDiagonal = i === j;
                    const normalized = normalizedMatrix[i]?.[j] || 0;

                    return (
                      <div
                        key={j}
                        className={cn(
                          "flex-1 aspect-square m-0.5 rounded flex flex-col items-center justify-center transition-all",
                          getHeatColor(normalized, isDiagonal),
                          "hover:ring-2 hover:ring-white/30"
                        )}
                        title={`${labels[i]} -> ${labels[j]}: ${value} (${(normalized * 100).toFixed(1)}%)`}
                      >
                        <span className="text-white font-bold text-sm md:text-base">
                          {value.toLocaleString()}
                        </span>
                        {showPercentages && (
                          <span className="text-white/70 text-xs">
                            {(normalized * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="flex items-center justify-center gap-6 mt-4">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-500 rounded" />
              <span className="text-xs text-gray-400">Correct</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-red-500/60 rounded" />
              <span className="text-xs text-gray-400">Errors</span>
            </div>
          </div>
        </div>

        {/* Per-Class Metrics */}
        <div className="lg:w-72">
          <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-1">
            Per-Class Metrics
            <Tooltip content="Метрики качества для каждого класса отдельно.

Precision — из всех предсказаний этого класса, сколько было верных?
Recall — из всех реальных случаев этого класса, сколько модель нашла?
F1 — баланс между Precision и Recall.

Важно для торговли:
• Высокий Precision для BUY/SELL = меньше ложных входов
• Высокий Recall = меньше пропущенных возможностей

Support = количество реальных примеров этого класса." />
          </h4>
          <div className="space-y-3">
            {perClassMetrics.map((metric) => (
              <div key={metric.label} className="p-3 bg-gray-800/50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className={cn("font-medium", getLabelColor(metric.label))}>
                    {metric.label}
                  </span>
                  <span className="text-xs text-gray-500">
                    {metric.support.toLocaleString()} samples
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div>
                    <span className="text-gray-500 block">Precision</span>
                    <span className="text-white font-medium">
                      {(metric.precision * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500 block">Recall</span>
                    <span className="text-white font-medium">
                      {(metric.recall * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500 block">F1</span>
                    <span className="text-white font-medium">
                      {(metric.f1 * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Mini progress bars */}
                <div className="mt-2 space-y-1">
                  <div className="flex items-center gap-2">
                    <div className="w-12 text-xs text-gray-500">P</div>
                    <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-purple-500 rounded-full"
                        style={{ width: `${metric.precision * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-12 text-xs text-gray-500">R</div>
                    <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-cyan-500 rounded-full"
                        style={{ width: `${metric.recall * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-12 text-xs text-gray-500">F1</div>
                    <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-green-500 rounded-full"
                        style={{ width: `${metric.f1 * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Macro Averages */}
          <div className="mt-4 p-3 bg-gray-800/80 rounded-lg border border-gray-700">
            <h5 className="text-xs font-medium text-gray-400 mb-2 flex items-center gap-1">
              Macro Average
              <Tooltip content="Macro Average — среднее арифметическое метрик по всем классам.

Каждый класс имеет равный вес независимо от количества примеров.

Важно: если классы несбалансированы (например, HOLD много, а BUY/SELL мало), macro average покажет честную картину для редких классов.

Сравнивайте с weighted average, который учитывает размер классов." />
            </h5>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <span className="text-gray-500 block">Precision</span>
                <span className="text-purple-400 font-medium">
                  {(perClassMetrics.reduce((a, b) => a + b.precision, 0) / perClassMetrics.length * 100).toFixed(1)}%
                </span>
              </div>
              <div>
                <span className="text-gray-500 block">Recall</span>
                <span className="text-cyan-400 font-medium">
                  {(perClassMetrics.reduce((a, b) => a + b.recall, 0) / perClassMetrics.length * 100).toFixed(1)}%
                </span>
              </div>
              <div>
                <span className="text-gray-500 block">F1</span>
                <span className="text-green-400 font-medium">
                  {(perClassMetrics.reduce((a, b) => a + b.f1, 0) / perClassMetrics.length * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}

export default ConfusionMatrixHeatmap;
