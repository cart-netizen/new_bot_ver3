// frontend/src/components/screener/TradingPairsList.tsx

import { useEffect, useMemo } from 'react';
import { useScreenerStore } from '../../store/screenerStore';
import { ArrowUp, ArrowDown, ArrowUpDown, Plus, Check } from 'lucide-react';
import type { SortField } from '../../types/screener.types';

interface TradingPairsListProps {
  /**
   * Callback при клике на торговую пару.
   */
  onPairClick?: (symbol: string) => void;

  /**
   * Текущая выбранная пара (для подсветки).
   */
  selectedSymbol?: string | null;
}

/**
 * Узкий вертикальный список торговых пар.
 */
export function TradingPairsList({ onPairClick, selectedSymbol }: TradingPairsListProps) {
  const {
    pairs,
    sortField,
    sortOrder,
    isLoading,
    error,
    fetchPairs,
    togglePairSelection,
    setSorting,
  } = useScreenerStore();

  /**
   * Загрузка пар при монтировании компонента.
   */
  useEffect(() => {
    console.log('[TradingPairsList] Component mounted');
    console.log('[TradingPairsList] Current pairs count:', pairs.length);

    // КРИТИЧНО: Загружаем пары только если их еще нет
    if (pairs.length === 0 && !isLoading) {
      console.log('[TradingPairsList] Fetching pairs...');
      fetchPairs();
    } else {
      console.log('[TradingPairsList] Pairs already loaded, skipping fetch');
    }
  }, []); // Пустой массив - вызывается только при монтировании

  /**
   * Логирование состояния при изменении pairs.
   */
  useEffect(() => {
    console.log('[TradingPairsList] Pairs updated:', {
      count: pairs.length,
      isLoading,
      error,
      sample: pairs.slice(0, 2)
    });
  }, [pairs, isLoading, error]);

  /**
   * ИСПРАВЛЕНО: useMemo с правильными зависимостями.
   * Пересчитывается при изменении pairs, sortField или sortOrder.
   */
  const sortedPairs = useMemo(() => {
    console.log('[TradingPairsList] Recalculating sorted pairs...');

    if (pairs.length === 0) {
      console.log('[TradingPairsList] No pairs to sort');
      return [];
    }

    const sorted = [...pairs].sort((a, b) => {
      let aValue: number | string;
      let bValue: number | string;

      switch (sortField) {
        case 'symbol':
          aValue = a.symbol;
          bValue = b.symbol;
          break;
        case 'price':
          aValue = a.last_price;
          bValue = b.last_price;
          break;
        case 'change_24h':
          aValue = a.price_change_24h_percent;
          bValue = b.price_change_24h_percent;
          break;
        case 'volume':
          aValue = a.volume_24h;
          bValue = b.volume_24h;
          break;
        default:
          return 0;
      }

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortOrder === 'asc'
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }

      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
      }

      return 0;
    });

    console.log('[TradingPairsList] Sorted pairs:', {
      count: sorted.length,
      sortField,
      sortOrder,
      first3: sorted.slice(0, 3).map(p => p.symbol)
    });

    return sorted;
  }, [pairs, sortField, sortOrder]); // ИСПРАВЛЕНО: Правильные зависимости

  const handleSort = (field: SortField) => {
    console.log('[TradingPairsList] Sorting by:', field);
    if (sortField === field) {
      setSorting(field, sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSorting(field, 'desc');
    }
  };

  /**
   * Обработчик клика на строку торговой пары.
   */
  const handleRowClick = (symbol: string) => {
    console.log('[TradingPairsList] Row clicked:', symbol);
    if (onPairClick) {
      onPairClick(symbol);
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
        {/* Индикатор загрузки в заголовке */}
        {isLoading && (
          <p className="text-xs text-blue-400 mt-1">Загрузка...</p>
        )}
        {/* Отображение ошибки в заголовке */}
        {error && (
          <p className="text-xs text-red-400 mt-1">Ошибка загрузки</p>
        )}
      </div>

      {/* Таблица с независимым скроллингом */}
      <div className="flex-1 overflow-y-auto">
        {isLoading && pairs.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full mx-auto mb-2"></div>
              <p className="text-xs text-gray-400">Загрузка пар...</p>
            </div>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <p className="text-xs text-red-400 mb-2">Ошибка: {error}</p>
              <button
                onClick={() => {
                  console.log('[TradingPairsList] Manual retry clicked');
                  fetchPairs();
                }}
                className="text-xs text-primary hover:text-primary/80 underline"
              >
                Повторить загрузку
              </button>
            </div>
          </div>
        ) : sortedPairs.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <p className="text-xs text-gray-400 mb-2">Нет данных</p>
              <button
                onClick={() => {
                  console.log('[TradingPairsList] Manual load clicked');
                  fetchPairs();
                }}
                className="text-xs text-primary hover:text-primary/80 underline"
              >
                Загрузить пары
              </button>
            </div>
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
                  onClick={() => handleRowClick(pair.symbol)}
                  className={`border-b border-gray-800 transition cursor-pointer text-xs ${
                    selectedSymbol === pair.symbol
                      ? 'bg-primary/20 hover:bg-primary/25'
                      : 'hover:bg-gray-800/30'
                  }`}
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
                      onClick={(e) => {
                        e.stopPropagation();
                        togglePairSelection(pair.symbol);
                      }}
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