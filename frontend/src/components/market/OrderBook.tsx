import { Card } from '../ui/Card.tsx';
import { useMarketStore } from '../../store/marketStore.ts';
import { formatPrice, formatVolume } from '../../utils/format.ts';

interface OrderBookProps {
  symbol: string;
}

export function OrderBook({ symbol }: OrderBookProps) {
  const orderbook = useMarketStore((state) => state.orderbooks[symbol]);

  if (!orderbook) {
    return (
      <Card className="p-6">
        <p className="text-gray-400">Загрузка...</p>
      </Card>
    );
  }

  const maxVolume = Math.max(
    ...orderbook.bids.map((b) => b[1]),
    ...orderbook.asks.map((a) => a[1])
  );

  return (
    <Card className="p-4">
      <h3 className="text-lg font-semibold mb-4">{symbol}</h3>

      <div className="grid grid-cols-2 gap-4 text-sm text-gray-400 mb-2">
        <div>Цена</div>
        <div className="text-right">Объем</div>
      </div>

      {/* Asks */}
      <div className="space-y-1 mb-4">
        {orderbook.asks.slice(0, 10).reverse().map(([price, volume], idx) => (
          <div key={idx} className="relative">
            <div
              className="absolute right-0 h-full bg-danger/20"
              style={{ width: `${(volume / maxVolume) * 100}%` }}
            />
            <div className="relative grid grid-cols-2 gap-4 py-1 px-2">
              <span className="text-danger font-mono">{formatPrice(price, 2)}</span>
              <span className="text-right font-mono">{formatVolume(volume)}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Spread */}
      <div className="py-2 border-y border-gray-800 text-center mb-4">
        <div className="text-white font-medium">{formatPrice(orderbook.mid_price, 2)}</div>
        <div className="text-xs text-gray-400">
          Spread: {formatPrice(orderbook.spread, 4)}
        </div>
      </div>

      {/* Bids */}
      <div className="space-y-1">
        {orderbook.bids.slice(0, 10).map(([price, volume], idx) => (
          <div key={idx} className="relative">
            <div
              className="absolute left-0 h-full bg-success/20"
              style={{ width: `${(volume / maxVolume) * 100}%` }}
            />
            <div className="relative grid grid-cols-2 gap-4 py-1 px-2">
              <span className="text-success font-mono">{formatPrice(price, 2)}</span>
              <span className="text-right font-mono">{formatVolume(volume)}</span>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}