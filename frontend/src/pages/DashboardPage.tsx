// frontend/src/pages/DashboardPage.tsx
/**
 * Главная страница дашборда с оптимизированным layout.
 *
 * ОБНОВЛЕНО:
 * - Добавлен TradingPairsList компонент (примыкает к Sidebar)
 * - Новые пропорции: TradingPairDetail (1/5), Chart (3/5), OrderBook (1/5)
 * - Независимый скролл для TradingPairsList
 * - Сохранен существующий функционал и дизайн
 */

import { useEffect, useState } from 'react';
import { BotControls } from '../components/bot/BotControls';
import { MetricsCard } from '../components/market/MetricsCard';
import { OrderBookWidget } from '../components/market/OrderBookWidget';
import { PriceChart } from '../components/market/PriceChart';
import { SignalsTable } from '../components/market/SignalsTable';
import { TradingPairsList } from '../components/market/TradingPairsList';
import { useBotStore } from '../store/botStore';
import { useMarketStore } from '../store/marketStore';
import { useTradingStore } from '../store/tradingStore';
import { Activity, Wifi, WifiOff } from 'lucide-react';
import { toast } from 'sonner';

/**
 * Главная страница дашборда.
 */
export function DashboardPage() {
  const { fetchStatus, fetchConfig, symbols } = useBotStore();
  const {
    orderbooks,
    currentMetrics,
    selectedSymbol: storeSelectedSymbol,
    isConnected,
    // setSelectedSymbol: _setStoreSelectedSymbol,
    memoryStats,
  } = useMarketStore();
  const { signals } = useTradingStore();

  const [isInitializing, setIsInitializing] = useState(true);

  // Локальное состояние для выбранной пары (для Dashboard)
  // Независимо от глобального selectedSymbol
  const [localSelectedSymbol, setLocalSelectedSymbol] = useState<string | null>(null);

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
   * Логирование статистики памяти (dev mode).
   */
  useEffect(() => {
    if (import.meta.env.DEV && !isInitializing) {
      console.log('[Dashboard] Memory stats:', memoryStats);
    }
  }, [memoryStats, isInitializing]);

  /**
   * Обработка выбора пары из списка.
   */
  const handleSelectPair = (symbol: string) => {
    console.log('[Dashboard] Selected pair:', symbol);
    setLocalSelectedSymbol(symbol);
    // Опционально: можно также обновить глобальный state
    // setStoreSelectedSymbol(symbol);
  };

  /**
   * Получение текущей выбранной пары.
   * Приоритет: localSelectedSymbol > storeSelectedSymbol > первый символ из списка
   */
  const currentSymbol = localSelectedSymbol || storeSelectedSymbol || symbols[0] || 'BTCUSDT';

  /**
   * Рендер состояния загрузки.
   */
  if (isInitializing) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse space-y-4">
          <div className="h-10 bg-gray-700 rounded w-1/4"></div>
          <div className="h-64 bg-gray-700 rounded"></div>
          <div className="grid grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-48 bg-gray-700 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    // Изменяем контейнер на flex для горизонтального расположения
    <div className="flex h-full">
      {/* ==================== НОВЫЙ БЛОК: Список торговых пар ==================== */}
      <TradingPairsList
        onSelectPair={handleSelectPair}
        selectedSymbol={localSelectedSymbol}
      />

      {/* ==================== ОСНОВНОЙ КОНТЕНТ ==================== */}
      <div className="flex-1 overflow-auto">
        <div className="p-6 space-y-6">
          {/* Заголовок и статус */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">Dashboard</h1>
              <p className="text-gray-400 mt-1">
                Мониторинг торговых пар в реальном времени
              </p>
            </div>

            {/* Статус подключения */}
            <div className="flex items-center gap-3">
              <Activity className="h-5 w-5 text-gray-400" />
              {isConnected ? (
                <div className="flex items-center gap-2">
                  <Wifi className="h-5 w-5 text-success" />
                  <span className="text-sm text-success font-medium">Подключено</span>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <WifiOff className="h-5 w-5 text-destructive" />
                  <span className="text-sm text-destructive font-medium">Отключено</span>
                </div>
              )}
            </div>
          </div>

          {/* Управление ботом */}
          <BotControls />

          {/* ==================== ОБНОВЛЕННЫЙ LAYOUT: 1/5 - 3/5 - 1/5 ==================== */}
          {currentSymbol && (
            <div>
              {/* Заголовок секции */}
              <div className="mb-4">
                <h2 className="text-xl font-semibold">
                  Детали: <span className="text-primary">{currentSymbol}</span>
                </h2>
              </div>

              {/* Grid с новыми пропорциями */}
              <div className="grid grid-cols-5 gap-4">
                {/* Колонка 1: Trading Pair Detail (1/5 = 20%) */}
                <div className="col-span-1">
                  {currentMetrics[currentSymbol] ? (
                    <MetricsCard metrics={currentMetrics[currentSymbol]} />
                  ) : (
                    <div className="bg-surface p-6 rounded-lg border border-gray-800">
                      <div className="animate-pulse">
                        <div className="h-6 bg-gray-700 rounded mb-4 w-2/3"></div>
                        <div className="space-y-3">
                          {[1, 2, 3, 4].map((i) => (
                            <div key={i} className="h-4 bg-gray-700 rounded"></div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Колонка 2: Price Chart (3/5 = 60%) */}
                <div className="col-span-3">
                  <PriceChart
                    symbol={currentSymbol}
                    loading={false}
                  />
                </div>

                {/* Колонка 3: Order Book (1/5 = 20%) */}
                <div className="col-span-1">
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
      </div>
    </div>
  );
}