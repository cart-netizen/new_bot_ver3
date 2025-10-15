// frontend/src/pages/ScreenerPage.tsx
/**
 * Страница скринера торговых пар.
 *
 * Функционал:
 * - Отображение скринера с real-time данными
 * - Автоматическое обновление через WebSocket
 * - Фильтрация пар с volume > 4M USDT
 * - Расчет динамики по таймфреймам
 * - Начальная загрузка через REST API
 *
 * Интеграция:
 * - Использует screenerStore для управления данными
 * - Подключается к WebSocket через Layout
 * - Memory-optimized подход
 *
 * Обновлено: Добавлена начальная загрузка через REST API
 */

import { useEffect, useState } from 'react';
import { ScreenerTable } from '../components/screener/ScreenerTable';
import { useScreenerStore } from '../store/screenerStore';
import { AlertCircle, Info, Activity, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

/**
 * Компонент страницы скринера.
 */
export function ScreenerPage() {
  const {
    isConnected,
    isLoading,
    memoryStats,
    loadInitialData,
    cleanupMemory
  } = useScreenerStore();

  const [isInitializing, setIsInitializing] = useState(true);

  /**
   * Инициализация при монтировании.
   */
  useEffect(() => {
    console.log('[ScreenerPage] Initializing...');

    // Принудительная очистка памяти при входе на страницу
    cleanupMemory();

    // Загружаем начальные данные через REST API
    loadInitialData()
      .then(() => {
        console.log('[ScreenerPage] Initial data loaded successfully');
        setIsInitializing(false);
      })
      .catch((error) => {
        console.error('[ScreenerPage] Failed to load initial data:', error);
        setIsInitializing(false);
      });

    // Периодическая очистка памяти (каждые 2 минуты)
    const cleanupInterval = setInterval(() => {
      console.log('[ScreenerPage] Running periodic cleanup...');
      cleanupMemory();
    }, 2 * 60 * 1000);

    return () => {
      clearInterval(cleanupInterval);
      console.log('[ScreenerPage] Unmounting, cleaning up...');
    };
  }, [cleanupMemory, loadInitialData]);

  /**
   * Мониторинг статистики памяти (только в dev mode).
   */
  useEffect(() => {
    if (import.meta.env.DEV && !isInitializing) {
      console.log('[ScreenerPage] Memory stats:', memoryStats);

      // Предупреждение при достижении 80% лимита пар
      if (memoryStats.totalPairs > 80) {
        toast.warning('Приближение к лимиту количества пар в скринере', {
          description: `${memoryStats.totalPairs} из 100 пар`,
        });
      }
    }
  }, [memoryStats, isInitializing]);

  /**
   * Отображение загрузки при инициализации.
   */
  if (isInitializing || isLoading) {
    return (
      <div className="p-6">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-white mb-2">Скринер Торговых Пар</h1>
          <p className="text-gray-400">
            Анализ рынка в реальном времени
          </p>
        </div>

        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <Loader2 className="animate-spin h-12 w-12 text-primary mx-auto mb-4" />
            <p className="text-gray-400 mb-2">Загрузка данных скринера...</p>
            <p className="text-sm text-gray-500">
              Подключение к серверу и получение данных
            </p>
          </div>
        </div>
      </div>
    );
  }

  /**
   * Отображение предупреждения при отсутствии подключения.
   */
  const renderConnectionWarning = () => {
    if (isConnected) {
      return null;
    }

    return (
      <div className="mb-6 bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-yellow-500 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-yellow-500 mb-1">
              Отсутствует подключение к WebSocket
            </h3>
            <p className="text-sm text-gray-400">
              Данные скринера обновляются через WebSocket.
              Пожалуйста, проверьте подключение к серверу.
            </p>
          </div>
        </div>
      </div>
    );
  };

  /**
   * Отображение информационного блока.
   */
  const renderInfoPanel = () => {
    return (
      <div className="mb-6 bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-blue-500 mb-2">
              О скринере
            </h3>
            <ul className="text-sm text-gray-400 space-y-1">
              <li>• Отображаются только пары с объёмом торгов {'>'} 4,000,000 USDT за 24 часа</li>
              <li>• Динамика рассчитывается по таймфреймам: 1m, 3m, 5m, 15m</li>
              <li>• Обновление данных происходит в реальном времени через WebSocket</li>
              <li>• Используйте сортировку для поиска интересных торговых возможностей</li>
              <li>• Максимальное количество отслеживаемых пар: 100</li>
            </ul>
          </div>
        </div>
      </div>
    );
  };

  /**
   * Отображение статистики памяти (только в dev mode).
   */
  const renderMemoryStats = () => {
    if (!import.meta.env.DEV) {
      return null;
    }

    return (
      <div className="mb-6 bg-gray-800/50 border border-gray-700 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Activity className="h-5 w-5 text-gray-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <h3 className="text-sm font-semibold text-gray-300 mb-2">
              Статистика памяти (Dev Mode)
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-gray-500">Всего пар</p>
                <p className="text-white font-medium">{memoryStats.totalPairs}</p>
              </div>
              <div>
                <p className="text-gray-500">Активных</p>
                <p className="text-white font-medium">{memoryStats.activePairs}</p>
              </div>
              <div>
                <p className="text-gray-500">История цен</p>
                <p className="text-white font-medium">{memoryStats.totalPricePoints} точек</p>
              </div>
              <div>
                <p className="text-gray-500">Посл. очистка</p>
                <p className="text-white font-medium">
                  {new Date(memoryStats.lastCleanup).toLocaleTimeString()}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  /**
   * Отображение статуса подключения.
   */
  const renderConnectionStatus = () => {
    return (
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className={`h-2 w-2 rounded-full ${
              isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
            }`}
          />
          <span className="text-sm text-gray-400">
            {isConnected ? 'Подключено к серверу' : 'Нет подключения'}
          </span>
        </div>

        <div className="text-sm text-gray-500">
          Обновление в реальном времени
        </div>
      </div>
    );
  };

  return (
    <div className="p-6">
      {/* Заголовок */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-2">
          Скринер Торговых Пар
        </h1>
        <p className="text-gray-400">
          Анализ рынка криптовалют в реальном времени
        </p>
      </div>

      {/* Статус подключения */}
      {renderConnectionStatus()}

      {/* Предупреждение о подключении */}
      {renderConnectionWarning()}

      {/* Информационная панель */}
      {renderInfoPanel()}

      {/* Статистика памяти (dev mode) */}
      {renderMemoryStats()}

      {/* Таблица скринера */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
        <ScreenerTable />
      </div>

      {/* Подсказка внизу */}
      <div className="mt-6 text-center">
        <p className="text-sm text-gray-500">
          💡 Совет: Кликните по заголовку колонки для сортировки данных
        </p>
      </div>
    </div>
  );
}