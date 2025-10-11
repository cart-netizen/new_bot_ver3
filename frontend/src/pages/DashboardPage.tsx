// frontend/src/pages/DashboardPage.tsx

import { useEffect, useState } from 'react';
import { BotControls } from '../components/bot/BotControls';
import { MetricsCard } from '../components/market/MetricsCard';
import { OrderBookWidget } from '../components/market/OrderBookWidget';
import { PriceChart } from '../components/market/PriceChart';
import { SignalsTable } from '../components/market/SignalsTable';
import { useBotStore } from '../store/botStore';
import { useMarketStore } from '../store/marketStore';
import { useTradingStore } from '../store/tradingStore';
import { Activity, Wifi, WifiOff } from 'lucide-react';
import { toast } from 'sonner';

/**
 * Главная страница дашборда.
 * Отображает управление ботом, метрики по парам, стакан и сигналы в реальном времени.
 *
 * Примечание: WebSocket управляется на уровне Layout, поэтому соединение
 * остается активным при переключении между страницами.
 */
export function DashboardPage() {
  const { fetchStatus, fetchConfig, symbols } = useBotStore();
  const {
    orderbooks,
    metrics,
    metricsHistory,
    selectedSymbol,
    isConnected,
    setSelectedSymbol,
  } = useMarketStore();
  const { signals } = useTradingStore();

  const [isInitializing, setIsInitializing] = useState(true);

  /**
   * Инициализация: загрузка данных бота.
   * WebSocket уже подключен на уровне Layout.
   */
  useEffect(() => {
    const initialize = async () => {
      try {
        console.log('[Dashboard] Initializing...');

        // Загружаем статус и конфигурацию бота
        await fetchStatus();
        await fetchConfig();

        console.log('[Dashboard] Initialized successfully');
      } catch (error) {
        console.error('[Dashboard] Initialization error:', error);
        toast.error('Ошибка инициализации');
      } finally {
        setIsInitializing(false);
      }
    };

    initialize();

    // WebSocket НЕ отключается при размонтировании DashboardPage,
    // так как он управляется на уровне Layout
  }, [fetchStatus, fetchConfig]);

  /**
   * Выбор пары для детального просмотра.
   */
  const handleSymbolSelect = (symbol: string) => {
    setSelectedSymbol(symbol === selectedSymbol ? null : symbol);
  };

  /**
   * Получение выбранной пары или первой доступной.
   */
  const currentSymbol = selectedSymbol || symbols[0];

  if (isInitializing) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Activity className="h-12 w-12 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-gray-400">Загрузка...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Заголовок с индикатором подключения */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Dashboard</h1>

        {/* Индикатор WebSocket соединения */}
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${
            isConnected
              ? 'bg-success/10 text-success'
              : 'bg-destructive/10 text-destructive'
          }`}
        >
          {isConnected ? (
            <Wifi className="h-4 w-4" />
          ) : (
            <WifiOff className="h-4 w-4" />
          )}
          <span className="text-sm font-medium">
            {isConnected ? 'Подключено' : 'Отключено'}
          </span>
        </div>
      </div>

      {/* Управление ботом */}
      <BotControls />

      {/* Список торговых пар */}
      {symbols.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold mb-4">Торговые пары</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {symbols.map((symbol) => {
              const symbolMetrics = metrics[symbol];

              return (
                <button
                  key={symbol}
                  onClick={() => handleSymbolSelect(symbol)}
                  className={`p-4 rounded-lg border transition-all text-left ${
                    selectedSymbol === symbol
                      ? 'border-primary bg-primary/10'
                      : 'border-gray-800 bg-surface hover:border-gray-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold">{symbol}</h3>
                    {symbolMetrics && (
                      <div
                        className={`flex items-center gap-1 text-xs ${
                          symbolMetrics.imbalance.overall > 0.6
                            ? 'text-success'
                            : symbolMetrics.imbalance.overall < 0.4
                            ? 'text-destructive'
                            : 'text-gray-400'
                        }`}
                      >
                        <span>
                          {(symbolMetrics.imbalance.overall * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                  </div>

                  {symbolMetrics ? (
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Mid Price:</span>
                        <span className="font-mono">
                          {symbolMetrics.prices.mid_price
                            ? symbolMetrics.prices.mid_price.toFixed(2)
                            : '-'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Spread:</span>
                        <span className="font-mono text-xs">
                          {symbolMetrics.prices.spread
                            ? symbolMetrics.prices.spread.toFixed(8)
                            : '-'}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">Ожидание данных...</p>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Детальная информация по выбранной паре */}
      {currentSymbol && (
        <div>
          <h2 className="text-xl font-semibold mb-4">
            Детали: {currentSymbol}
          </h2>

          {/* НОВЫЙ LAYOUT: 3 колонки - Метрики | График | Order Book (1/3 ширины) */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            {/* Колонка 1: Метрики (1/3) */}
            <div className="lg:col-span-1">
              <MetricsCard
                metrics={metrics[currentSymbol] || null}
                loading={!metrics[currentSymbol]}
              />
            </div>

            {/* Колонка 2: График цены (1/3) */}
            <div className="lg:col-span-1">
              <PriceChart
                symbol={currentSymbol}
                metricsHistory={metricsHistory[currentSymbol] || []}
                loading={!metricsHistory[currentSymbol]}
              />
            </div>

            {/* Колонка 3: Order Book (1/3) */}
            <div className="lg:col-span-1">
              <OrderBookWidget
                orderbook={orderbooks[currentSymbol] || null}
                loading={!orderbooks[currentSymbol]}
              />
            </div>
          </div>
        </div>
      )}

      {/* Таблица торговых сигналов */}
      <div>
        <h2 className="text-xl font-semibold mb-4">Торговые Сигналы</h2>
        <SignalsTable signals={signals} />
      </div>
    </div>
  );
}