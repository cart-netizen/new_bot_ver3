// frontend/src/pages/ScreenerPage.tsx

import { useEffect, useMemo, useCallback } from 'react';
import { useScreenerStore } from '../store/screenerStore';
import { ScreenerTable } from '../components/screener/ScreenerTable';
import { ScreenerSettings } from '../components/screener/ScreenerSettings';
import { Activity, AlertCircle, TrendingUp } from 'lucide-react';
import type { SortField } from '../types/screener.types';

/**
 * Страница скринера торговых пар.
 * Отображает расширенную таблицу с динамикой цены по всем таймфреймам.
 * Фильтрация по объему и лимит топ N пар.
 * Система алертов для отслеживания памп.
 */
export function ScreenerPage() {
  const {
    pairs,
    sortField,
    sortOrder,
    isLoading,
    error,
    settings,
    alerts,
    fetchPairs,
    setSorting,
    updateSettings,
    dismissAlert,
    getSortedPairs,
    getAlertedPairs,
  } = useScreenerStore();

  /**
   * Загрузка пар при монтировании компонента.
   */
  useEffect(() => {
    console.log('[ScreenerPage] Component mounted');
    if (pairs.length === 0 && !isLoading) {
      console.log('[ScreenerPage] Fetching pairs...');
      fetchPairs();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Автоматическое обновление с настраиваемым интервалом.
   */
  useEffect(() => {
    const intervalMs = settings.refreshInterval * 1000;

    console.log(`[ScreenerPage] Setting up auto-refresh every ${settings.refreshInterval}s`);

    const intervalId = setInterval(() => {
      console.log('[ScreenerPage] Auto-refreshing pairs...');
      fetchPairs();
    }, intervalMs);

    return () => {
      console.log('[ScreenerPage] Clearing auto-refresh');
      clearInterval(intervalId);
    };
  }, [settings.refreshInterval, fetchPairs]);

  /**
   * Обработчик сортировки.
   */
  const handleSort = useCallback((field: SortField) => {
    if (sortField === field) {
      // Переключаем порядок
      setSorting(field, sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      // Новое поле - начинаем с desc
      setSorting(field, 'desc');
    }
  }, [sortField, sortOrder, setSorting]);

  /**
   * Получаем отсортированные пары с алертами сверху.
   */
  const displayPairs = useMemo(() => {
    const alertedPairs = getAlertedPairs();
    const alertedSymbols = new Set(alertedPairs.map(p => p.symbol));

    // Остальные пары (без алертов)
    const normalPairs = getSortedPairs().filter(p => !alertedSymbols.has(p.symbol));

    // Алерты всегда сверху (не подчиняются сортировке)
    return [...alertedPairs, ...normalPairs];
  }, [pairs, sortField, sortOrder, alerts]); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Set для быстрой проверки алертов.
   */
  const alertedSymbols = useMemo(() => {
    return new Set(Array.from(alerts.keys()));
  }, [alerts]);

  if (isLoading && pairs.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Activity className="h-12 w-12 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-gray-400">Загрузка скринера...</p>
        </div>
      </div>
    );
  }

  if (error && pairs.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 mx-auto mb-4 text-destructive" />
          <p className="text-destructive mb-4">Ошибка: {error}</p>
          <button
            onClick={() => fetchPairs()}
            className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/80 transition"
          >
            Повторить попытку
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Заголовок */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Скринер торговых пар</h1>
          <p className="text-gray-400 mt-1">
            Динамика изменения цены по всем таймфреймам
          </p>
        </div>

        {/* Статистика */}
        <div className="flex gap-4">
          <div className="px-4 py-2 bg-surface border border-gray-800 rounded-lg">
            <p className="text-xs text-gray-400">Всего пар</p>
            <p className="text-xl font-bold text-primary">{displayPairs.length}</p>
          </div>

          {alerts.size > 0 && (
            <div className="px-4 py-2 bg-destructive/10 border border-destructive/50 rounded-lg">
              <p className="text-xs text-destructive">Активных алертов</p>
              <p className="text-xl font-bold text-destructive">{alerts.size}</p>
            </div>
          )}

          <div className="px-4 py-2 bg-surface border border-gray-800 rounded-lg">
            <p className="text-xs text-gray-400">Обновление</p>
            <p className="text-xl font-bold text-success">{settings.refreshInterval}с</p>
          </div>
        </div>
      </div>

      {/* Настройки */}
      <ScreenerSettings
        settings={settings}
        onUpdate={updateSettings}
      />

      {/* Информация об алертах */}
      {alerts.size > 0 && (
        <div className="bg-destructive/10 border border-destructive/50 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <TrendingUp className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold text-destructive">Обнаружены ПАМП!</h3>
              <p className="text-sm text-gray-300 mt-1">
                {alerts.size} {alerts.size === 1 ? 'пара превысила' : 'пар превысили'} пороговое значение {settings.alertThreshold}%.
                Эти пары выделены красным и закреплены сверху списка.
                Нажмите на крестик рядом с символом, чтобы отменить алерт.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Таблица */}
      <div className="bg-surface border border-gray-800 rounded-lg overflow-hidden">
        <div className="max-h-[calc(100vh-320px)] overflow-y-auto">
          <ScreenerTable
            pairs={displayPairs}
            alertedSymbols={alertedSymbols}
            sortField={sortField}
            sortOrder={sortOrder}
            onSort={handleSort}
            onDismissAlert={dismissAlert}
          />
        </div>
      </div>

      {/* Индикатор загрузки внизу */}
      {isLoading && (
        <div className="fixed bottom-4 right-4 bg-primary text-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-2">
          <Activity className="h-4 w-4 animate-spin" />
          <span className="text-sm">Обновление...</span>
        </div>
      )}
    </div>
  );
}
