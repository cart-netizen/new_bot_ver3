// frontend/src/components/dev/MemoryMonitor.tsx
// Компонент для отладки и мониторинга использования памяти

import { useEffect, useState, useReducer } from 'react';
import { useMarketStoreMemoryMonitor, useMarketStore } from '../../store/marketStore';
import { Card } from '../ui/Card';
import { Activity, Database, Trash2 } from 'lucide-react';

/**
 * Компонент мониторинга памяти для development режима.
 * Показывает статистику использования памяти и позволяет вручную очищать данные.
 *
 * ИСПОЛЬЗОВАНИЕ:
 * Добавить в Layout.tsx или DashboardPage.tsx только в dev режиме:
 *
 * {import.meta.env.DEV && <MemoryMonitor />}
 */
export function MemoryMonitor() {
  const memoryStats = useMarketStoreMemoryMonitor();
  const cleanupMemory = useMarketStore((state) => state.cleanupMemory);
  const [isVisible, setIsVisible] = useState(false);

  // Обновляем статистику каждые 5 секунд
  // ИСПРАВЛЕНО: Используем useReducer для принудительного ре-рендера
  const [, forceUpdate] = useReducer((x) => x + 1, 0);
  useEffect(() => {
    const interval = window.setInterval(() => {
      forceUpdate();
    }, 5000);

    return () => window.clearInterval(interval);
  }, []);

  // Форматирование размера памяти
  const formatMemory = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  // Форматирование времени
  const formatTime = (timestamp: number): string => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };

  // Рассчитываем процент использования
  const calculateUsagePercent = (current: number, max: number): number => {
    return Math.min((current / max) * 100, 100);
  };

  if (!isVisible) {
    return (
      <button
        onClick={() => setIsVisible(true)}
        className="fixed bottom-4 right-4 p-3 bg-gray-800 rounded-full shadow-lg hover:bg-gray-700 transition-colors z-50"
        title="Показать мониторинг памяти"
      >
        <Activity className="w-5 h-5 text-blue-400" />
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 w-80 z-50">
      <Card className="p-4 bg-gray-900 border-gray-700 shadow-2xl">
        {/* Заголовок */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Database className="w-5 h-5 text-blue-400" />
            <h3 className="text-sm font-semibold text-white">Memory Monitor</h3>
          </div>
          <button
            onClick={() => setIsVisible(false)}
            className="text-gray-400 hover:text-white transition-colors"
          >
            ×
          </button>
        </div>

        {/* Статистика */}
        <div className="space-y-3 text-xs">
          {/* Общая память */}
          <div>
            <div className="flex justify-between text-gray-400 mb-1">
              <span>Estimated Memory</span>
              <span className="font-mono text-blue-400">
                {formatMemory(memoryStats.estimatedMemoryUsage)}
              </span>
            </div>
          </div>

          {/* Активные символы */}
          <div>
            <div className="flex justify-between text-gray-400 mb-1">
              <span>Active Symbols</span>
              <span className="font-mono text-white">
                {memoryStats.totalSymbols}
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-1.5">
              <div
                className="bg-blue-500 h-1.5 rounded-full transition-all"
                style={{
                  width: `${calculateUsagePercent(memoryStats.totalSymbols, 100)}%`,
                }}
              />
            </div>
          </div>

          {/* Orderbooks */}
          <div>
            <div className="flex justify-between text-gray-400 mb-1">
              <span>Orderbooks</span>
              <span className="font-mono text-white">
                {memoryStats.totalOrderbooks}
              </span>
            </div>
          </div>

          {/* Metrics */}
          <div>
            <div className="flex justify-between text-gray-400 mb-1">
              <span>Metrics Snapshots</span>
              <span className="font-mono text-white">
                {memoryStats.totalMetrics}
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-1.5">
              <div
                className="bg-green-500 h-1.5 rounded-full transition-all"
                style={{
                  width: `${calculateUsagePercent(memoryStats.totalMetrics, 1000)}%`,
                }}
              />
            </div>
          </div>

          {/* Последняя очистка */}
          <div className="pt-2 border-t border-gray-700">
            <div className="flex justify-between text-gray-400">
              <span>Last Cleanup</span>
              <span className="font-mono text-gray-300">
                {formatTime(memoryStats.lastCleanup)}
              </span>
            </div>
          </div>

          {/* Кнопка ручной очистки */}
          <button
            onClick={() => {
              cleanupMemory();
              console.info('[MemoryMonitor] Manual cleanup triggered');
            }}
            className="w-full mt-3 flex items-center justify-center gap-2 px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors text-xs font-medium"
          >
            <Trash2 className="w-4 h-4" />
            Force Cleanup
          </button>
        </div>

        {/* Dev mode indicator */}
        <div className="mt-3 pt-3 border-t border-gray-700 text-xs text-gray-500 text-center">
          Development Mode Only
        </div>
      </Card>
    </div>
  );
}