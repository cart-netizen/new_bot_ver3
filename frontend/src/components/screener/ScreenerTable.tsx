// frontend/src/components/screener/ScreenerTable.tsx

import React, { useCallback } from 'react';
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
 * Форматирование объема (в миллионах).
 */
const formatVolume = (value: number | null | undefined): string => {
  if (typeof value !== 'number' || isNaN(value)) {
    return '-';
  }
  return `${(value / 1_000_000).toFixed(2)}M`;
};

/**
 * Форматирование цены.
 */
const formatPrice = (value: number | null | undefined): string => {
  if (typeof value !== 'number' || isNaN(value)) {
    return '-';
  }
  return `$${value.toFixed(value >= 1 ? 2 : 4)}`;
};

/**
 * Форматирование процентов с цветом.
 */
const formatPercent = (value: number | null | undefined): {text: string, color: string} => {
  if (typeof value !== 'number' || isNaN(value)) {
    return { text: '-', color: 'text-gray-500' };
  }

  const text = `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
  const color = value > 0 ? 'text-success' : value < 0 ? 'text-destructive' : 'text-gray-400';

  return { text, color };
};

/**
 * Строка таблицы.
 */
const TableRow = React.memo(({
  pair,
  isAlerted,
  onDismissAlert,
}: {
  pair: ScreenerPair;
  isAlerted: boolean;
  onDismissAlert: (symbol: string) => void;
}) => {
  const handleDismiss = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onDismissAlert(pair.symbol);
  }, [pair.symbol, onDismissAlert]);

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
          <span className="truncate max-w-[100px]" title={pair.symbol || 'N/A'}>
            {pair.symbol ? pair.symbol.replace('USDT', '') : 'N/A'}
          </span>
        </div>
      </td>

      {/* Price */}
      <td className="px-3 py-2 text-right font-mono">
        {formatPrice(pair.last_price)}
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

/**
 * Компонент иконки сортировки.
 */
const SortIcon = React.memo(({
  field,
  currentSortField,
  sortOrder
}: {
  field: SortField;
  currentSortField: SortField;
  sortOrder: SortOrder;
}) => {
  if (currentSortField !== field) {
    return <ArrowUpDown className="h-3 w-3 opacity-30" />;
  }
  return sortOrder === 'asc'
    ? <ArrowUp className="h-3 w-3" />
    : <ArrowDown className="h-3 w-3" />;
});

SortIcon.displayName = 'SortIcon';

/**
 * Заголовок столбца с сортировкой.
 */
const SortableHeader = React.memo(({
  field,
  label,
  align = 'left',
  currentSortField,
  sortOrder,
  onSort,
}: {
  field: SortField;
  label: string;
  align?: 'left' | 'right';
  currentSortField: SortField;
  sortOrder: SortOrder;
  onSort: (field: SortField) => void;
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
        <SortIcon field={field} currentSortField={currentSortField} sortOrder={sortOrder} />
      </div>
    </th>
  );
});

SortableHeader.displayName = 'SortableHeader';

/**
 * Компонент таблицы скринера с оптимизацией.
 */
export const ScreenerTable = React.memo(({
  pairs,
  alertedSymbols,
  sortField,
  sortOrder,
  onSort,
  onDismissAlert,
}: ScreenerTableProps) => {
  // Безопасная обработка пустого массива
  if (!pairs || pairs.length === 0) {
    return (
      <div className="p-8 text-center text-gray-400">
        <p>Нет данных для отображения</p>
      </div>
    );
  }

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
            <SortableHeader field="price" label="Цена" align="right" currentSortField={sortField} sortOrder={sortOrder} onSort={onSort} />
            <SortableHeader field="volume" label="Объем 24ч" align="right" currentSortField={sortField} sortOrder={sortOrder} onSort={onSort} />
            <SortableHeader field="change_1m" label="1 мин" align="right" currentSortField={sortField} sortOrder={sortOrder} onSort={onSort} />
            <SortableHeader field="change_2m" label="2 мин" align="right" currentSortField={sortField} sortOrder={sortOrder} onSort={onSort} />
            <SortableHeader field="change_5m" label="5 мин" align="right" currentSortField={sortField} sortOrder={sortOrder} onSort={onSort} />
            <SortableHeader field="change_15m" label="15 мин" align="right" currentSortField={sortField} sortOrder={sortOrder} onSort={onSort} />
            <SortableHeader field="change_30m" label="30 мин" align="right" currentSortField={sortField} sortOrder={sortOrder} onSort={onSort} />
            <SortableHeader field="change_1h" label="1 час" align="right" currentSortField={sortField} sortOrder={sortOrder} onSort={onSort} />
            <SortableHeader field="change_4h" label="4 часа" align="right" currentSortField={sortField} sortOrder={sortOrder} onSort={onSort} />
            <SortableHeader field="change_8h" label="8 часов" align="right" currentSortField={sortField} sortOrder={sortOrder} onSort={onSort} />
            <SortableHeader field="change_12h" label="12 часов" align="right" currentSortField={sortField} sortOrder={sortOrder} onSort={onSort} />
            <SortableHeader field="change_24h_interval" label="24 часа" align="right" currentSortField={sortField} sortOrder={sortOrder} onSort={onSort} />
          </tr>
        </thead>

        <tbody>
          {pairs.map((pair) => (
            <TableRow
              key={pair.symbol}
              pair={pair}
              isAlerted={alertedSymbols.has(pair.symbol)}
              onDismissAlert={onDismissAlert}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
});

ScreenerTable.displayName = 'ScreenerTable';