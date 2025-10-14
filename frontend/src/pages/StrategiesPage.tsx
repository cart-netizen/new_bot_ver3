// ==================== frontend/src/pages/StrategiesPage.tsx ====================
/**
 * Страница управления стратегиями.
 *
 * TODO:
 * - Список стратегий с переключателями
 * - Статистика по каждой стратегии
 * - Управление параметрами
 */

import { Construction } from 'lucide-react';

export function StrategiesPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Стратегии</h1>

      <div className="bg-surface rounded-lg border border-gray-800 p-12 text-center">
        <Construction className="h-16 w-16 text-gray-600 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-white mb-2">
          В разработке
        </h2>
        <p className="text-gray-400 mb-4">
          Страница управления стратегиями находится в процессе разработки
        </p>
        <div className="text-sm text-gray-500 space-y-1">
          <p>• Список стратегий с переключателями вкл/выкл</p>
          <p>• Статистика: сигналы, сделки, прибыль/убыток</p>
          <p>• Управление параметрами стратегий</p>
          <p>• История performance</p>
        </div>
      </div>
    </div>
  );
}