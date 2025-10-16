// frontend/src/pages/DashboardPage.tsx

import { useEffect, useState } from 'react';
import { BotControls } from '../components/bot/BotControls';
import { MetricsCard } from '../components/market/MetricsCard';
import { OrderBookWidget } from '../components/market/OrderBookWidget';
import { PriceChart } from '../components/market/PriceChart';
import { SignalsTable } from '../components/market/SignalsTable';
import { TradingPairsList } from '../components/screener/TradingPairsList';
import { useBotStore } from '../store/botStore';
import { useMarketStore } from '../store/marketStore';
import { useTradingStore } from '../store/tradingStore';
import { Activity, Wifi, WifiOff } from 'lucide-react';
import { toast } from 'sonner';

/**
 * Главная страница дашборда.
 * Отображает управление ботом, метрики, стакан и сигналы.
 * Layout: TradingPairsList -> Детали + График -> Order Book.
 */
export function DashboardPage() {
  const { fetchStatus, fetchConfig, symbols } = useBotStore();
  const {
    orderbooks,
    metrics,
    selectedSymbol,
    isConnected,
    setSelectedSymbol,
  } = useMarketStore();
  const { signals } = useTradingStore();

  const [isInitializing, setIsInitializing] = useState(true);

  /**
   * Инициализация: загрузка данных бота.
   */
  useEffect(() => {
    const initialize = async () => {
      try {
        console.log('[Dashboard] Initializing...');
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
  }, [fetchStatus, fetchConfig]);

  /**
   * Обработчик клика на торговую пару.
   * Устанавливает выбранную пару для детального просмотра.
   */
  const handlePairClick = (symbol: string) => {
    console.log('[Dashboard] Pair clicked:', symbol);
    console.log('[Dashboard] Current selectedSymbol:', selectedSymbol);
    console.log('[Dashboard] New selectedSymbol:', symbol);
    setSelectedSymbol(symbol);
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
    <div className="flex h-screen overflow-hidden">
      {/* 1. TradingPairsList - вертикальный список слева */}
      <div className="w-[360px] flex-shrink-0">
        <TradingPairsList
          onPairClick={handlePairClick}
          selectedSymbol={selectedSymbol}
        />
      </div>

      {/* 2-3. Основной контент (Детали + График + Order Book) */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-6 space-y-6">
          {/* Заголовок с индикатором подключения */}
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold">Dashboard</h1>

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

          {/* Детальная информация по выбранной паре */}
          {currentSymbol && (
            <div className="flex gap-6">
              {/* 2. Детали + График (вертикально) */}
              <div className="flex-1 space-y-6">
                {/* Заголовок */}
                <h2 className="text-xl font-semibold">
                  Детали: {currentSymbol}
                </h2>

                {/* Метрики */}
                <div>
                  <h3 className="text-lg font-medium mb-3">Метрики</h3>
                  <MetricsCard
                    metrics={metrics[currentSymbol] || null}
                    loading={!metrics[currentSymbol]}
                  />
                </div>

                {/* График */}
                <div>
                  <h3 className="text-lg font-medium mb-3">График цены</h3>
                  <PriceChart
                    symbol={currentSymbol}
                    loading={false}
                  />
                </div>
              </div>

              {/* 3. Order Book (уменьшен на 30%) */}
              <div className="w-[224px] flex-shrink-0">
                <h3 className="text-lg font-medium mb-3">Стакан</h3>
                <OrderBookWidget
                  orderbook={orderbooks[currentSymbol] || null}
                  loading={!orderbooks[currentSymbol]}
                />
              </div>
            </div>
          )}

          {/* Таблица торговых сигналов */}
          <div>
            <h2 className="text-xl font-semibold mb-4">Торговые Сигналы</h2>
            <SignalsTable signals={signals} />
          </div>
        </div>
      </div>
    </div>
  );
}