import { Card } from '../ui/Card.tsx';
import type {OrderBookMetrics} from '../../types/metrics.types.ts';
import { formatPrice, formatVolume, formatPercent } from '../utils/format.ts;

interface MetricsCardProps {
  metrics: OrderBookMetrics;
}

export function MetricsCard({ metrics }: MetricsCardProps) {
  return (
    <Card className="p-4">
      <h3 className="text-lg font-semibold mb-3">{metrics.symbol}</h3>

      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-400">Mid Price:</span>
          <span className="font-mono">{formatPrice(metrics.mid_price, 2)}</span>
        </div>

        <div className="flex justify-between">
          <span className="text-gray-400">Spread:</span>
          <span className="font-mono">{formatPrice(metrics.spread, 4)}</span>
        </div>

        <div className="flex justify-between">
          <span className="text-gray-400">Imbalance:</span>
          <span className={`font-mono ${
            (metrics.imbalance || 0) > 0.5 ? 'text-success' : 'text-danger'
          }`}>
            {metrics.imbalance ? formatPercent(metrics.imbalance) : 'N/A'}
          </span>
        </div>

        <div className="pt-2 border-t border-gray-800">
          <div className="flex justify-between">
            <span className="text-gray-400">Bid Volume:</span>
            <span className="text-success font-mono">
              {formatVolume(metrics.total_bid_volume)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Ask Volume:</span>
            <span className="text-danger font-mono">
              {formatVolume(metrics.total_ask_volume)}
            </span>
          </div>
        </div>
      </div>
    </Card>
  );
}