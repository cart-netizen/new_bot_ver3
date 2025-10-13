// frontend/src/pages/MarketPage.tsx
// ИСПРАВЛЕНО: Используется currentMetrics вместо metrics

import { useState } from 'react';
import { OrderBook } from '../components/market/OrderBook';
import { MetricsCard } from '../components/market/MetricsCard';
import { useBotStore } from '../store/botStore';
import { useMarketStore } from '../store/marketStore';

/**
 * Страница рыночных данных.
 * Отображает стаканы и метрики по всем торговым парам.
 *
 * ИСПРАВЛЕНО:
 * - Использует currentMetrics вместо metrics (новое API)
 * - Безопасная обработка отсутствующих данных
 */
export function MarketPage() {
  const symbols = useBotStore((state) => state.symbols);

  // ИСПРАВЛЕНО: Используем currentMetrics вместо metrics
  const currentMetrics = useMarketStore((state) => state.currentMetrics);

  const [selectedSymbol, setSelectedSymbol] = useState<string>(symbols[0] || '');

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Рыночные Данные</h1>

      {symbols.length > 0 ? (
        <>
          {/* Селектор символов */}
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

          {/* Детали выбранного символа */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Стакан */}
            {selectedSymbol && <OrderBook symbol={selectedSymbol} />}

            {/* Метрики выбранного символа */}
            <div className="space-y-4">
              {/* ИСПРАВЛЕНО: Используем currentMetrics */}
              {selectedSymbol && currentMetrics[selectedSymbol] ? (
                <MetricsCard metrics={currentMetrics[selectedSymbol]} />
              ) : (
                <div className="bg-surface p-6 rounded-lg border border-gray-800">
                  <p className="text-gray-400 text-center">
                    Ожидание данных для {selectedSymbol}...
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Все метрики */}
          <div>
            <h2 className="text-xl font-semibold mb-4">Все Метрики</h2>

            {/* ИСПРАВЛЕНО: Используем currentMetrics и безопасную обработку */}
            {Object.keys(currentMetrics).length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.values(currentMetrics).map((m) => (
                  <MetricsCard key={m.symbol} metrics={m} />
                ))}
              </div>
            ) : (
              <div className="bg-surface p-8 rounded-lg border border-gray-800 text-center">
                <p className="text-gray-400">
                  Нет доступных метрик. Запустите бота для начала сбора данных.
                </p>
              </div>
            )}
          </div>
        </>
      ) : (
        <div className="bg-surface p-8 rounded-lg border border-gray-800 text-center">
          <p className="text-gray-400">
            Символы не настроены. Проверьте конфигурацию бота.
          </p>
        </div>
      )}
    </div>
  );
}