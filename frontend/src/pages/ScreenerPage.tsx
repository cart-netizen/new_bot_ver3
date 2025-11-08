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
  } = useScreenerStore();

  /**
   * Загрузка пар при монтировании компонента.
   */
  useEffect(() => {
    if (pairs.length === 0 && !isLoading) {
      fetchPairs();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Автоматическое обновление с настраиваемым интервалом.
   */
  useEffect(() => {
    const intervalMs = settings.refreshInterval * 1000;

    const intervalId = setInterval(() => {
      fetchPairs();
    }, intervalMs);

    return () => {
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
    if (pairs.length === 0) return [];

    // Разделяем пары на алертные и обычные
    const alertedSymbols = new Set(alerts.keys());
    const alertedPairs = pairs.filter(p => alertedSymbols.has(p.symbol));
    const normalPairs = pairs.filter(p => !alertedSymbols.has(p.symbol));

    // Сортируем обычные пары
    const sortedNormalPairs = [...normalPairs].sort((a, b) => {
      let aValue: number | string | null;
      let bValue: number | string | null;

      // Маппинг полей для сортировки
      switch (sortField) {
        case 'symbol':
          aValue = a.symbol;
          bValue = b.symbol;
          break;
        case 'price':
          aValue = a.last_price;
          bValue = b.last_price;
          break;
        case 'volume':
          aValue = a.volume_24h;
          bValue = b.volume_24h;
          break;
        case 'change_24h':
          aValue = a.price_change_24h_percent;
          bValue = b.price_change_24h_percent;
          break;
        case 'change_1m':
          aValue = a.price_change_1m;
          bValue = b.price_change_1m;
          break;
        case 'change_2m':
          aValue = a.price_change_2m;
          bValue = b.price_change_2m;
          break;
        case 'change_5m':
          aValue = a.price_change_5m;
          bValue = b.price_change_5m;
          break;
        case 'change_15m':
          aValue = a.price_change_15m;
          bValue = b.price_change_15m;
          break;
        case 'change_30m':
          aValue = a.price_change_30m;
          bValue = b.price_change_30m;
          break;
        case 'change_1h':
          aValue = a.price_change_1h;
          bValue = b.price_change_1h;
          break;
        case 'change_4h':
          aValue = a.price_change_4h;
          bValue = b.price_change_4h;
          break;
        case 'change_8h':
          aValue = a.price_change_8h;
          bValue = b.price_change_8h;
          break;
        case 'change_12h':
          aValue = a.price_change_12h;
          bValue = b.price_change_12h;
          break;
        case 'change_24h_interval':
          aValue = a.price_change_24h;
          bValue = b.price_change_24h;
          break;
        default:
          return 0;
      }

      // Обработка null значений
      if (aValue === null && bValue === null) return 0;
      if (aValue === null) return sortOrder === 'asc' ? 1 : -1;
      if (bValue === null) return sortOrder === 'asc' ? -1 : 1;

      // Сортировка строк
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortOrder === 'asc'
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }

      // Сортировка чисел
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
      }

      return 0;
    });

    // Алерты всегда сверху (не подчиняются сортировке)
    return [...alertedPairs, ...sortedNormalPairs];
  }, [pairs, sortField, sortOrder, alerts]);

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
