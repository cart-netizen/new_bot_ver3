import { useState } from 'react';
import { OrderBook } from '../components/market/OrderBook.tsx';
import { MetricsCard } from '../components/market/MetricsCard.tsx';
import { useBotStore } from '../store/botStore.ts';
import { useMarketStore } from '../store/marketStore.ts';

export function MarketPage() {
  const symbols = useBotStore((state) => state.symbols);
  const metrics = useMarketStore((state) => state.metrics);
  const [selectedSymbol, setSelectedSymbol] = useState<string>(symbols[0] || '');

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Рыночные Данные</h1>

      {symbols.length > 0 && (
        <>
          <div className="flex gap-2 flex-wrap">
            {symbols.map((symbol) => (
              <button
                key={symbol}
                onClick={() => setSelectedSymbol(symbol)}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  selectedSymbol === symbol
                    ? 'bg-primary text-white'
                    : 'bg-surface text-gray-400 hover:bg-gray-800'
                }`}
              >
                {symbol}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {selectedSymbol && <OrderBook symbol={selectedSymbol} />}

            <div className="space-y-4">
              {selectedSymbol && metrics[selectedSymbol] && (
                <MetricsCard metrics={metrics[selectedSymbol]} />
              )}
            </div>
          </div>

          <div>
            <h2 className="text-xl font-semibold mb-4">Все Метрики</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.values(metrics).map((m) => (
                <MetricsCard key={m.symbol} metrics={m} />
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}