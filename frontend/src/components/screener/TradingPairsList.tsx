// frontend/src/components/screener/TradingPairsList.tsx

import { useEffect, useMemo } from 'react';
import { useScreenerStore } from '../../store/screenerStore';
import { ArrowUp, ArrowDown, ArrowUpDown, Plus, Check } from 'lucide-react';
import type { SortField } from '../../types/screener.types';

/**
 * Узкий вертикальный список торговых пар (как sidebar).
 * Отображается между Sidebar и основным контентом.
 */
export function TradingPairsList() {
  const {
    pairs,
    sortField,
    sortOrder,
    isLoading,
    fetchPairs,
    togglePairSelection,
    setSorting,
    getSortedPairs,
  } = useScreenerStore();

  useEffect(() => {
    fetchPairs();
  }, [fetchPairs]);

  const sortedPairs = useMemo(() => getSortedPairs(), [pairs, sortField, sortOrder, getSortedPairs]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSorting(field, sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSorting(field, 'desc');
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) {
      return <ArrowUpDown className="h-3 w-3 opacity-30" />;
    }
    return sortOrder === 'asc'
      ? <ArrowUp className="h-3 w-3" />
      : <ArrowDown className="h-3 w-3" />;
  };

  const formatVolume = (value: number): string => {
    if (value >= 1_000_000) {
      return `${(value / 1_000_000).toFixed(1)}M`;
    }
    return `${(value / 1000).toFixed(0)}K`;
  };

  const formatPrice = (value: number): string => {
    if (value >= 1000) {
      return value.toLocaleString('en-US', { maximumFractionDigits: 1 });
    } else if (value >= 1) {
      return value.toFixed(2);
    } else {
      return value.toFixed(4);
    }
  };

  return (
    <div className="h-screen flex flex-col border-r border-gray-800 bg-surface">
      {/* Заголовок */}
      <div className="px-3 py-4 border-b border-gray-800">
        <h2 className="text-sm font-semibold">Торговые пары</h2>
        <p className="text-xs text-gray-400 mt-1">
          {sortedPairs.length} пар
        </p>
      </div>

      {/* Таблица с независимым скроллингом */}
      <div className="flex-1 overflow-y-auto">
        {isLoading && pairs.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <p className="text-xs text-gray-400">Загрузка...</p>
          </div>
        ) : sortedPairs.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <p className="text-xs text-gray-400">Нет данных</p>
          </div>
        ) : (
          <table className="w-full">
            <thead className="sticky top-0 bg-surface border-b border-gray-800 z-10">
              <tr className="text-xs">
                {/* Пара */}
                <th
                  className="px-2 py-2 text-left font-medium cursor-pointer hover:bg-gray-800 transition"
                  onClick={() => handleSort('symbol')}
                >
                  <div className="flex items-center gap-1">
                    <span>Пара</span>
                    <SortIcon field="symbol" />
                  </div>
                </th>

                {/* Цена */}
                <th
                  className="px-2 py-2 text-right font-medium cursor-pointer hover:bg-gray-800 transition"
                  onClick={() => handleSort('price')}
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>Цена</span>
                    <SortIcon field="price" />
                  </div>
                </th>

                {/* 24ч % */}
                <th
                  className="px-2 py-2 text-right font-medium cursor-pointer hover:bg-gray-800 transition"
                  onClick={() => handleSort('change_24h')}
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>%</span>
                    <SortIcon field="change_24h" />
                  </div>
                </th>

                {/* Объем */}
                <th
                  className="px-2 py-2 text-right font-medium cursor-pointer hover:bg-gray-800 transition"
                  onClick={() => handleSort('volume')}
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>Vol</span>
                    <SortIcon field="volume" />
                  </div>
                </th>

                {/* Кнопка */}
                <th className="px-2 py-2 w-10">
                  <span className="sr-only">Выбор</span>
                </th>
              </tr>
            </thead>

            <tbody>
              {sortedPairs.map((pair) => (
                <tr
                  key={pair.symbol}
                  className="border-b border-gray-800 hover:bg-gray-800/30 transition text-xs"
                >
                  {/* Пара */}
                  <td className="px-2 py-1.5 font-mono font-medium">
                    <div className="truncate max-w-[80px]" title={pair.symbol}>
                      {pair.symbol.replace('USDT', '')}
                    </div>
                  </td>

                  {/* Цена */}
                  <td className="px-2 py-1.5 text-right font-mono">
                    <div className="truncate">
                      {formatPrice(pair.last_price)}
                    </div>
                  </td>

                  {/* 24ч % */}
                  <td className="px-2 py-1.5 text-right font-mono">
                    <span
                      className={
                        pair.price_change_24h_percent > 0
                          ? 'text-success'
                          : pair.price_change_24h_percent < 0
                          ? 'text-destructive'
                          : ''
                      }
                    >
                      {pair.price_change_24h_percent > 0 && '+'}
                      {pair.price_change_24h_percent.toFixed(1)}
                    </span>
                  </td>

                  {/* Объем */}
                  <td className="px-2 py-1.5 text-right font-mono text-gray-400">
                    {formatVolume(pair.volume_24h)}
                  </td>

                  {/* Кнопка */}
                  <td className="px-2 py-1.5 text-center">
                    <button
                      onClick={() => togglePairSelection(pair.symbol)}
                      className={`p-1 rounded transition ${
                        pair.is_selected
                          ? 'bg-primary text-white'
                          : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                      }`}
                      title={pair.is_selected ? 'Убрать' : 'Добавить'}
                    >
                      {pair.is_selected ? (
                        <Check className="h-3 w-3" />
                      ) : (
                        <Plus className="h-3 w-3" />
                      )}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}