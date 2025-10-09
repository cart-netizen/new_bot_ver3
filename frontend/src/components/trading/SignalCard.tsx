import { Card } from '../ui/Card.tsx';
import { TrendingUp, TrendingDown } from 'lucide-react';
import {type TradingSignal, SignalType } from '../../types/signal.types.ts';
import { formatPrice, formatPercent, formatRelativeTime } from '../../utils/format.ts';

interface SignalCardProps {
  signal: TradingSignal;
}

export function SignalCard({ signal }: SignalCardProps) {
  const isBuy = signal.signal_type === SignalType.BUY;

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`p-2 rounded-lg ${isBuy ? 'bg-success/20' : 'bg-danger/20'}`}>
            {isBuy ? (
              <TrendingUp className="h-5 w-5 text-success" />
            ) : (
              <TrendingDown className="h-5 w-5 text-danger" />
            )}
          </div>
          <div>
            <h3 className="font-semibold">{signal.symbol}</h3>
            <p className="text-xs text-gray-400">
              {formatRelativeTime(signal.timestamp)}
            </p>
          </div>
        </div>
        <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
          isBuy ? 'bg-success/20 text-success' : 'bg-danger/20 text-danger'
        }`}>
          {signal.signal_type}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <p className="text-gray-400">Цена</p>
          <p className="font-mono">{formatPrice(signal.price, 2)}</p>
        </div>
        <div>
          <p className="text-gray-400">Уверенность</p>
          <p className="font-semibold text-primary">{formatPercent(signal.confidence)}</p>
        </div>
      </div>

      {signal.reason && (
        <p className="mt-3 text-xs text-gray-400 italic">{signal.reason}</p>
      )}
    </Card>
  );
}