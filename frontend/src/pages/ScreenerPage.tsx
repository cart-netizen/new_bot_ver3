// frontend/src/pages/ScreenerPage.tsx
/**
 * Страница скринера торговых пар.
 *
 * Функционал:
 * - Отображение скринера с real-time данными
 * - Автоматическое обновление через WebSocket
 * - Фильтрация пар с volume > 4M USDT
 * - Расчет динамики по таймфреймам
 *
 * Интеграция:
 * - Использует screenerStore для управления данными
 * - Подключается к WebSocket через Layout
 * - Memory-optimized подход
 */

import { useEffect, useState } from 'react';
import { ScreenerTable } from '../components/screener/ScreenerTable';
import { useScreenerStore } from '../store/screenerStore';
import { AlertCircle, Info } from 'lucide-react';
import { toast } from 'sonner';

/**
 * Компонент страницы скринера.
 */
export function ScreenerPage() {
  const { isConnected, memoryStats, cleanupMemory } = useScreenerStore();
  const [isInitializing, setIsInitializing] = useState(true);

  /**
   * Инициализация при монтировании.
   */
  useEffect(() => {
    console.log('[ScreenerPage] Initializing...');

    // Принудительная очистка памяти при входе на страницу
    cleanupMemory();

    // Завершаем инициализацию
    setIsInitializing(false);

    // Периодическая очистка памяти (каждые 2 минуты)
    const cleanupInterval = setInterval(() => {
      cleanupMemory();
    }, 2 * 60 * 1000);

    return () => {
      clearInterval(cleanupInterval);
      console.log('[ScreenerPage] Unmounting, cleaning up...');
    };
  }, [cleanupMemory]);

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
   * Отображение предупреждения при отсутствии подключения.
   */
  const renderConnectionWarning = () => {
    if (isConnected || isInitializing) {
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
        <h3 className="text-sm font-semibold text-gray-400 mb-2">
          Статистика памяти (Dev Mode)
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <div className="text-gray-500">Всего пар</div>
            <div className="text-white font-semibold">{memoryStats.totalPairs}</div>
          </div>
          <div>
            <div className="text-gray-500">Активных</div>
            <div className="text-white font-semibold">{memoryStats.activePairs}</div>
          </div>
          <div>
            <div className="text-gray-500">Точек данных</div>
            <div className="text-white font-semibold">{memoryStats.totalPricePoints}</div>
          </div>
          <div>
            <div className="text-gray-500">Последняя очистка</div>
            <div className="text-white font-semibold">
              {new Date(memoryStats.lastCleanup).toLocaleTimeString('ru-RU')}
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (isInitializing) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold">Скринер</h1>

        {/* Skeleton loader */}
        <div className="bg-surface rounded-lg border border-gray-800 p-6">
          <div className="animate-pulse space-y-4">
            <div className="h-6 bg-gray-700 rounded w-1/3"></div>
            <div className="space-y-3">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="h-12 bg-gray-700 rounded"></div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Заголовок */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Скринер</h1>

        {/* Статус подключения */}
        <div className="flex items-center gap-2 text-sm">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-success animate-pulse' : 'bg-gray-500'}`} />
          <span className="text-gray-400">
            {isConnected ? 'Подключено' : 'Отключено'}
          </span>
        </div>
      </div>

      {/* Предупреждение о подключении */}
      {renderConnectionWarning()}

      {/* Информационная панель */}
      {renderInfoPanel()}

      {/* Статистика памяти (dev mode) */}
      {renderMemoryStats()}

      {/* Таблица скринера */}
      <ScreenerTable />
    </div>
  );
}