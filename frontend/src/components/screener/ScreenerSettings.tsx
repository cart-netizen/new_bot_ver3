// frontend/src/components/screener/ScreenerSettings.tsx

import React, { useState } from 'react';
import { Settings } from 'lucide-react';
import type { ScreenerSettings as ScreenerSettingsType } from '../../types/screener.types';

interface ScreenerSettingsProps {
  settings: ScreenerSettingsType;
  onUpdate: (settings: Partial<ScreenerSettingsType>) => void;
}

/**
 * Панель настроек скринера.
 * Позволяет настроить фильтрацию, лимиты и частоту обновления.
 */
export const ScreenerSettings = React.memo(({ settings, onUpdate }: ScreenerSettingsProps) => {
  const [localSettings, setLocalSettings] = useState(settings);
  const [isExpanded, setIsExpanded] = useState(false);

  const handleApply = () => {
    onUpdate(localSettings);
    setIsExpanded(false);
  };

  const handleReset = () => {
    const defaultSettings: ScreenerSettingsType = {
      minVolume: 4_000_000,
      topN: 100,
      refreshInterval: 5,
      alertThreshold: 5,
    };
    setLocalSettings(defaultSettings);
    onUpdate(defaultSettings);
  };

  return (
    <div className="bg-surface border border-gray-800 rounded-lg">
      {/* Заголовок */}
      <div
        className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-800/30 transition"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Settings className="h-5 w-5 text-primary" />
          <h2 className="text-lg font-semibold">Настройки скринера</h2>
        </div>
        <span className="text-xs text-gray-400">
          {isExpanded ? 'Свернуть' : 'Развернуть'}
        </span>
      </div>

      {/* Панель настроек */}
      {isExpanded && (
        <div className="p-4 border-t border-gray-800 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Минимальный объем */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Мин. объем торгов (USDT)
              </label>
              <input
                type="number"
                value={localSettings.minVolume}
                onChange={(e) => setLocalSettings({
                  ...localSettings,
                  minVolume: Number(e.target.value)
                })}
                className="w-full px-3 py-2 bg-background border border-gray-700 rounded-lg focus:outline-none focus:border-primary text-sm"
                min={0}
                step={1000000}
              />
              <p className="text-xs text-gray-400 mt-1">
                Текущее: {(localSettings.minVolume / 1_000_000).toFixed(1)}M
              </p>
            </div>

            {/* Топ N пар */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Топ N пар
              </label>
              <input
                type="number"
                value={localSettings.topN}
                onChange={(e) => setLocalSettings({
                  ...localSettings,
                  topN: Number(e.target.value)
                })}
                className="w-full px-3 py-2 bg-background border border-gray-700 rounded-lg focus:outline-none focus:border-primary text-sm"
                min={1}
                max={500}
              />
              <p className="text-xs text-gray-400 mt-1">
                Показать лучшие {localSettings.topN} пар
              </p>
            </div>

            {/* Частота обновления */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Частота обновления (сек)
              </label>
              <input
                type="number"
                value={localSettings.refreshInterval}
                onChange={(e) => setLocalSettings({
                  ...localSettings,
                  refreshInterval: Number(e.target.value)
                })}
                className="w-full px-3 py-2 bg-background border border-gray-700 rounded-lg focus:outline-none focus:border-primary text-sm"
                min={1}
                max={60}
              />
              <p className="text-xs text-gray-400 mt-1">
                Обновление каждые {localSettings.refreshInterval}с
              </p>
            </div>

            {/* Порог алерта */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Порог алерта (%)
              </label>
              <input
                type="number"
                value={localSettings.alertThreshold}
                onChange={(e) => setLocalSettings({
                  ...localSettings,
                  alertThreshold: Number(e.target.value)
                })}
                className="w-full px-3 py-2 bg-background border border-gray-700 rounded-lg focus:outline-none focus:border-primary text-sm"
                min={0.1}
                max={100}
                step={0.5}
              />
              <p className="text-xs text-gray-400 mt-1">
                Алерт при изменении ≥ {localSettings.alertThreshold}%
              </p>
            </div>
          </div>

          {/* Кнопки */}
          <div className="flex gap-3">
            <button
              onClick={handleApply}
              className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/80 transition text-sm font-medium"
            >
              Применить
            </button>
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition text-sm font-medium"
            >
              Сбросить
            </button>
            <button
              onClick={() => setIsExpanded(false)}
              className="px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition text-sm font-medium"
            >
              Отмена
            </button>
          </div>
        </div>
      )}
    </div>
  );
});

ScreenerSettings.displayName = 'ScreenerSettings';
