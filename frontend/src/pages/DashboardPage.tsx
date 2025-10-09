import { useEffect } from 'react';
import { BotControls } from '../components/bot/BotControls';
import { useBotStore } from '../store/botStore';

export function DashboardPage() {
  const { fetchStatus, fetchConfig, symbols } = useBotStore();

  useEffect(() => {
    fetchStatus();
    fetchConfig();
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Dashboard</h1>
      <BotControls />

      {symbols.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {symbols.map((symbol) => (
            <div key={symbol} className="bg-surface p-4 rounded-lg border border-gray-800">
              <h3 className="font-semibold">{symbol}</h3>
              <p className="text-sm text-gray-400">Активен</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}