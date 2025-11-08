// frontend/src/components/screener/ScreenerTable.tsx

import React, { useMemo, useCallback } from 'react';
import { ArrowUp, ArrowDown, ArrowUpDown, X } from 'lucide-react';
import type { ScreenerPair, SortField, SortOrder } from '../../types/screener.types';

interface ScreenerTableProps {
  pairs: ScreenerPair[];
  alertedSymbols: Set<string>;
  sortField: SortField;
  sortOrder: SortOrder;
  onSort: (field: SortField) => void;
  onDismissAlert: (symbol: string) => void;
}

/**
 * Компонент таблицы скринера с оптимизацией.
 * Мемоизирован для предотвращения лишних ре-рендеров.
 */
export const ScreenerTable = React.memo(({
  pairs,
  alertedSymbols,
  sortField,
  sortOrder,
  onSort,
  onDismissAlert,
}: ScreenerTableProps) => {
  console.log('[ScreenerTable] Rendering with', pairs.length, 'pairs');

  /**
   * Форматирование объема (в миллионах).
   */
  const formatVolume = useCallback((value: number): string => {
    return `${(value / 1_000_000).toFixed(2)}M`;
  }, []);

  /**
   * Форматирование процентов с цветом.
   */
  const formatPercent = useCallback((value: number | null): {text: string, color: string} => {
    if (value === null) {
      return { text: '-', color: 'text-gray-500' };
    }

    const text = `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
    const color = value > 0 ? 'text-success' : value < 0 ? 'text-destructive' : 'text-gray-400';

    return { text, color };
  }, []);

  /**
   * Компонент иконки сортировки.
   */
  const SortIcon = useCallback(({ field }: { field: SortField }) => {
    if (sortField !== field) {
      return <ArrowUpDown className="h-3 w-3 opacity-30" />;
    }
    return sortOrder === 'asc'
      ? <ArrowUp className="h-3 w-3" />
      : <ArrowDown className="h-3 w-3" />;
  }, [sortField, sortOrder]);

  /**
   * Заголовок столбца с сортировкой.
   */
  const SortableHeader = useCallback(({
    field,
    label,
    align = 'left'
  }: {
    field: SortField;
    label: string;
    align?: 'left' | 'right';
  }) => {
    return (
      <th
        className={`px-3 py-3 text-xs font-medium cursor-pointer hover:bg-gray-800/50 transition ${
          align === 'right' ? 'text-right' : 'text-left'
        }`}
        onClick={() => onSort(field)}
      >
        <div className={`flex items-center gap-1 ${align === 'right' ? 'justify-end' : ''}`}>
          <span>{label}</span>
          <SortIcon field={field} />
        </div>
      </th>
    );
  }, [onSort, SortIcon]);

  /**
   * Строка таблицы с мемоизацией.
   */
  const TableRow = React.memo(({
    pair,
    isAlerted
  }: {
    pair: ScreenerPair;
    isAlerted: boolean;
  }) => {
    const handleDismiss = useCallback((e: React.MouseEvent) => {
      e.stopPropagation();
      onDismissAlert(pair.symbol);
    }, [pair.symbol]);

    return (
      <tr
        className={`border-b border-gray-800 transition text-xs ${
          isAlerted
            ? 'bg-destructive/20 hover:bg-destructive/25 border-destructive/50'
            : 'hover:bg-gray-800/30'
        }`}
      >
        {/* Symbol */}
        <td className="px-3 py-2 font-mono font-medium sticky left-0 bg-surface">
          <div className="flex items-center gap-2">
            {isAlerted && (
              <button
                onClick={handleDismiss}
                className="p-0.5 hover:bg-gray-700 rounded transition"
                title="Отменить алерт"
              >
                <X className="h-3 w-3 text-destructive" />
              </button>
            )}
            <span className="truncate max-w-[100px]" title={pair.symbol}>
              {pair.symbol.replace('USDT', '')}
            </span>
          </div>
        </td>

        {/* Price */}
        <td className="px-3 py-2 text-right font-mono">
          ${pair.last_price.toFixed(pair.last_price >= 1 ? 2 : 4)}
        </td>

        {/* Volume */}
        <td className="px-3 py-2 text-right font-mono text-gray-400">
          {formatVolume(pair.volume_24h)}
        </td>

        {/* 1m */}
        <td className={`px-3 py-2 text-right font-mono ${formatPercent(pair.price_change_1m).color}`}>
          {formatPercent(pair.price_change_1m).text}
        </td>

        {/* 2m */}
        <td className={`px-3 py-2 text-right font-mono ${formatPercent(pair.price_change_2m).color}`}>
          {formatPercent(pair.price_change_2m).text}
        </td>

        {/* 5m */}
        <td className={`px-3 py-2 text-right font-mono ${formatPercent(pair.price_change_5m).color}`}>
          {formatPercent(pair.price_change_5m).text}
        </td>

        {/* 15m */}
        <td className={`px-3 py-2 text-right font-mono ${formatPercent(pair.price_change_15m).color}`}>
          {formatPercent(pair.price_change_15m).text}
        </td>

        {/* 30m */}
        <td className={`px-3 py-2 text-right font-mono ${formatPercent(pair.price_change_30m).color}`}>
          {formatPercent(pair.price_change_30m).text}
        </td>

        {/* 1h */}
        <td className={`px-3 py-2 text-right font-mono ${formatPercent(pair.price_change_1h).color}`}>
          {formatPercent(pair.price_change_1h).text}
        </td>

        {/* 4h */}
        <td className={`px-3 py-2 text-right font-mono ${formatPercent(pair.price_change_4h).color}`}>
          {formatPercent(pair.price_change_4h).text}
        </td>

        {/* 8h */}
        <td className={`px-3 py-2 text-right font-mono ${formatPercent(pair.price_change_8h).color}`}>
          {formatPercent(pair.price_change_8h).text}
        </td>

        {/* 12h */}
        <td className={`px-3 py-2 text-right font-mono ${formatPercent(pair.price_change_12h).color}`}>
          {formatPercent(pair.price_change_12h).text}
        </td>

        {/* 24h */}
        <td className={`px-3 py-2 text-right font-mono ${formatPercent(pair.price_change_24h).color}`}>
          {formatPercent(pair.price_change_24h).text}
        </td>
      </tr>
    );
  });

  TableRow.displayName = 'TableRow';

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead className="sticky top-0 bg-surface border-b-2 border-gray-700 z-10">
          <tr className="text-xs">
            <th className="px-3 py-3 text-left font-medium sticky left-0 bg-surface">
              <div className="flex items-center gap-1">
                <span>Пара</span>
              </div>
            </th>
            <SortableHeader field="price" label="Цена" align="right" />
            <SortableHeader field="volume" label="Объем 24ч" align="right" />
            <SortableHeader field="change_1m" label="1 мин" align="right" />
            <SortableHeader field="change_2m" label="2 мин" align="right" />
            <SortableHeader field="change_5m" label="5 мин" align="right" />
            <SortableHeader field="change_15m" label="15 мин" align="right" />
            <SortableHeader field="change_30m" label="30 мин" align="right" />
            <SortableHeader field="change_1h" label="1 час" align="right" />
            <SortableHeader field="change_4h" label="4 часа" align="right" />
            <SortableHeader field="change_8h" label="8 часов" align="right" />
            <SortableHeader field="change_12h" label="12 часов" align="right" />
            <SortableHeader field="change_24h_interval" label="24 часа" align="right" />
          </tr>
        </thead>

        <tbody>
          {pairs.map((pair) => (
            <TableRow
              key={pair.symbol}
              pair={pair}
              isAlerted={alertedSymbols.has(pair.symbol)}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
});

ScreenerTable.displayName = 'ScreenerTable';
