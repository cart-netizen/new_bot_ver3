import { SignalCard } from '../components/trading/SignalCard.tsx';
import { useTradingStore } from '../store/tradingStore.ts';

export function TradingPage() {
  const signals = useTradingStore((state) => state.signals);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Торговые Сигналы</h1>

      {signals.length === 0 ? (
        <div className="bg-surface p-8 rounded-lg border border-gray-800 text-center">
          <p className="text-gray-400">Пока нет торговых сигналов</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {signals.map((signal, idx) => (
            <SignalCard key={idx} signal={signal} />
          ))}
        </div>
      )}
    </div>
  );
}