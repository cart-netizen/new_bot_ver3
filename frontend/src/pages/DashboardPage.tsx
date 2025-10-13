// frontend/src/pages/DashboardPage.tsx
// ФИНАЛЬНАЯ ВЕРСИЯ с исправлениями утечки памяти и ошибок TypeScript

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
 * ОБНОВЛЕНО:
 * - Использует оптимизированный marketStore с управлением памятью
 * - Исправлены TypeScript ошибки
 * - Исправлены ESLint предупреждения
 * - Сохранен оригинальный дизайн и layout
 */
export function DashboardPage() {
  const { fetchStatus, fetchConfig, symbols } = useBotStore();
  const {
    orderbooks,
    currentMetrics,  // ← ИСПРАВЛЕНО: используем currentMetrics вместо metrics
    selectedSymbol,
    isConnected,
    setSelectedSymbol,
    memoryStats,     // ← НОВОЕ: статистика памяти
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
  }, [fetchStatus, fetchConfig]);  // ИСПРАВЛЕНО: Убрали memoryStats из dependencies

  /**
   * Отдельный эффект для логирования статистики памяти (dev mode).
   * НОВОЕ: Разделен от основного эффекта для правильных dependencies.
   */
  useEffect(() => {
    if (import.meta.env.DEV && !isInitializing) {
      console.log('[Dashboard] Memory stats:', memoryStats);
    }
  }, [memoryStats, isInitializing]);

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
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-gray-400">Загрузка данных...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Заголовок и статус подключения */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Торговый Дашборд</h1>
        <div className="flex items-center gap-2">
          {isConnected ? (
            <>
              <Wifi className="w-5 h-5 text-success" />
              <span className="text-sm text-success">Подключено</span>
            </>
          ) : (
            <>
              <WifiOff className="w-5 h-5 text-destructive" />
              <span className="text-sm text-destructive">Отключено</span>
            </>
          )}
        </div>
      </div>

      {/* Управление ботом */}
      <BotControls />

      {/* НОВОЕ: Статистика памяти (только dev mode) */}
      {import.meta.env.DEV && (
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-3">
          <div className="flex items-center gap-6 text-xs text-gray-400">
            <div>
              <span className="font-semibold">Активных символов:</span>{' '}
              <span className="text-white">{memoryStats.totalSymbols}</span>
            </div>
            <div>
              <span className="font-semibold">Orderbooks:</span>{' '}
              <span className="text-white">{memoryStats.totalOrderbooks}</span>
            </div>
            <div>
              <span className="font-semibold">Метрик:</span>{' '}
              <span className="text-white">{memoryStats.totalMetrics}</span>
            </div>
            <div>
              <span className="font-semibold">Последняя очистка:</span>{' '}
              <span className="text-gray-300">
                {new Date(memoryStats.lastCleanup).toLocaleTimeString()}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Селектор символов */}
      {symbols.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold mb-4">Торговые Пары</h2>
          <div className="flex flex-wrap gap-2">
            {symbols.map((symbol) => {
              // ИСПРАВЛЕНО: Используем currentMetrics вместо metrics
              const symbolMetrics = currentMetrics[symbol];
              const isSelected = symbol === selectedSymbol;

              return (
                <button
                  key={symbol}
                  onClick={() => handleSymbolSelect(symbol)}
                  className={`
                    relative px-4 py-3 rounded-lg border transition-all
                    ${
                      isSelected
                        ? 'bg-primary/20 border-primary text-primary'
                        : 'bg-surface border-gray-700 text-gray-300 hover:border-gray-600 hover:bg-gray-800'
                    }
                  `}
                >
                  <div className="flex items-center gap-3">
                    <span className="font-semibold">{symbol}</span>

                    {/* Индикатор imbalance */}
                    {symbolMetrics && symbolMetrics.imbalance && (
                      <div
                        className={`text-xs font-mono ${
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

          {/* LAYOUT: 3 колонки равной ширины - Метрики | График | Order Book */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            {/* Колонка 1: Метрики (1/3) */}
            {/* ИСПРАВЛЕНО: Убран проп loading, если TypeScript его не видит */}
            <div className="lg:col-span-1">
              {currentMetrics[currentSymbol] ? (
                <MetricsCard metrics={currentMetrics[currentSymbol]} />
              ) : (
                <div className="bg-surface p-6 rounded-lg border border-gray-800">
                  <div className="animate-pulse">
                    <div className="h-6 bg-gray-700 rounded mb-4 w-1/3"></div>
                    <div className="space-y-3">
                      {[1, 2, 3, 4].map((i) => (
                        <div key={i} className="h-4 bg-gray-700 rounded"></div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Колонка 2: График цены со свечами (1/3) */}
            <div className="lg:col-span-1">
              <PriceChart
                symbol={currentSymbol}
                loading={false}
              />
            </div>

            {/* Колонка 3: Order Book (1/3) */}
            {/* ИСПРАВЛЕНО: Передаем orderbook напрямую, не symbol */}
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