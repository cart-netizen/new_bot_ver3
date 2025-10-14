// frontend/src/components/market/TradingPairsList.tsx
/**
 * Компонент вертикального списка торговых пар.
 *
 * Функционал:
 * - Независимый вертикальный скролл
 * - Фильтрация по volume > 4M USDT
 * - Колонки: Пара, Цена, Изм. 24H %, Объём 24H, Кнопка выбора
 * - Сортировка по каждому столбцу
 * - Кнопка выбора пары для отображения графика
 *
 * Дизайн:
 * - Примыкает к Sidebar
 * - Не зависит от скролла остальной страницы
 * - Компактный и эффективный
 */

import { useState, useEffect, useMemo } from 'react';
import { useScreenerStore, type SortField, type SortDirection } from '../../store/screenerStore';
import { ArrowUpDown, ArrowUp, ArrowDown, TrendingUp, TrendingDown, Eye } from 'lucide-react';
import { cn } from '../../utils/helpers';

/**
 * Props компонента.
 */
interface TradingPairsListProps {
  /**
   * Callback при выборе торговой пары.
   */
  onSelectPair: (symbol: string) => void;

  /**
   * Текущая выбранная пара (для единичного выбора, например на Dashboard).
   * Если null - режим множественного выбора (для страницы Графики).
   */
  selectedSymbol?: string | null;

  /**
   * Массив выбранных символов (для множественного выбора).
   * Используется на странице Графики.
   */
  selectedSymbols?: string[];

  /**
   * Минимальная высота компонента.
   */
  minHeight?: string;
}

/**
 * Форматирование числа с сокращениями (K, M, B).
 */
function formatNumber(value: number, decimals: number = 2): string {
  if (value >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(decimals)}B`;
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(decimals)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(decimals)}K`;
  }
  return value.toFixed(decimals);
}

/**
 * Форматирование цены.
 */
function formatPrice(price: number): string {
  if (price >= 1000) {
    return price.toFixed(2);
  }
  if (price >= 1) {
    return price.toFixed(4);
  }
  return price.toFixed(6);
}

/**
 * Форматирование процентного изменения.
 */
function formatPercentage(value: number): string {
  const formatted = Math.abs(value).toFixed(2);
  const sign = value >= 0 ? '+' : '-';
  return `${sign}${formatted}%`;
}

/**
 * Компонент заголовка колонки с сортировкой.
 */
interface HeaderCellProps {
  field: SortField;
  label: string;
  currentField: SortField;
  currentDirection: SortDirection;
  onSort: (field: SortField) => void;
  align?: 'left' | 'center' | 'right';
}

function HeaderCell({
  field,
  label,
  currentField,
  currentDirection,
  onSort,
  align = 'left'
}: HeaderCellProps) {
  const isActive = currentField === field;

  const alignClass = {
    left: 'justify-start',
    center: 'justify-center',
    right: 'justify-end',
  }[align];

  return (
    <th
      className="px-2 py-2 cursor-pointer hover:bg-gray-800/50 transition-colors select-none group"
      onClick={() => onSort(field)}
    >
      <div className={cn('flex items-center gap-1 text-[10px] font-medium text-gray-400 uppercase tracking-wider', alignClass)}>
        <span className="group-hover:text-white transition-colors">{label}</span>
        {isActive ? (
          currentDirection === 'asc' ? (
            <ArrowUp className="h-3 w-3 text-primary" />
          ) : (
            <ArrowDown className="h-3 w-3 text-primary" />
          )
        ) : (
          <ArrowUpDown className="h-3 w-3 opacity-0 group-hover:opacity-50 transition-opacity" />
        )}
      </div>
    </th>
  );
}

/**
 * Компонент списка торговых пар.
 */
export function TradingPairsList({
  onSelectPair,
  selectedSymbol,
  minHeight = 'calc(100vh - 64px)' // 64px - высота header
}: TradingPairsListProps) {
  const {
    pairs,
    sortField,
    sortDirection,
    isConnected,
    setSorting,
    getSortedPairs,
  } = useScreenerStore();

  const [filterText, setFilterText] = useState('');
  const [flashedRows, setFlashedRows] = useState<Record<string, 'green' | 'red'>>({});

  /**
   * Обработка клика по заголовку для сортировки.
   */
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSorting(field, sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSorting(field, 'desc');
    }
  };

  /**
   * Фильтрация пар по тексту поиска.
   */
  const filteredPairs = useMemo(() => {
    const sorted = getSortedPairs();

    if (!filterText) {
      return sorted;
    }

    const filterLower = filterText.toLowerCase();
    return sorted.filter(pair =>
      pair.symbol.toLowerCase().includes(filterLower)
    );
  }, [filterText, getSortedPairs]);

  /**
   * Эффект для flash-анимации при изменении цены.
   */
  useEffect(() => {
    const previousPrices: Record<string, number> = {};

    Object.values(pairs).forEach(pair => {
      const prevPrice = previousPrices[pair.symbol];

      if (prevPrice && prevPrice !== pair.lastPrice) {
        const flashColor = pair.lastPrice > prevPrice ? 'green' : 'red';

        setFlashedRows(prev => ({
          ...prev,
          [pair.symbol]: flashColor,
        }));

        setTimeout(() => {
          setFlashedRows(prev => {
            const updated = { ...prev };
            delete updated[pair.symbol];
            return updated;
          });
        }, 1000);
      }

      previousPrices[pair.symbol] = pair.lastPrice;
    });
  }, [pairs]);

  return (
    <div
      className="flex flex-col bg-surface border-r border-gray-800"
      style={{
        width: '280px', // Фиксированная ширина
        minHeight,
        maxHeight: minHeight,
      }}
    >
      {/* Заголовок с поиском */}
      <div className="flex-shrink-0 p-3 border-b border-gray-800 space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white">Пары</h3>
          <div className={cn(
            'w-2 h-2 rounded-full',
            isConnected ? 'bg-success animate-pulse' : 'bg-gray-500'
          )} />
        </div>

        {/* Поле поиска */}
        <input
          type="text"
          placeholder="Поиск..."
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          className="w-full px-2 py-1.5 text-xs bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-primary focus:border-transparent transition-all"
        />

        {/* Счетчик пар */}
        <div className="text-[10px] text-gray-500">
          Отображено: <span className="text-white font-medium">{filteredPairs.length}</span>
        </div>
      </div>

      {/* Таблица с независимым скроллом */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-gray-900 z-10">
            <tr>
              <HeaderCell
                field="symbol"
                label="Пара"
                currentField={sortField}
                currentDirection={sortDirection}
                onSort={handleSort}
                align="left"
              />
              <HeaderCell
                field="lastPrice"
                label="Цена"
                currentField={sortField}
                currentDirection={sortDirection}
                onSort={handleSort}
                align="right"
              />
              <HeaderCell
                field="price24hPcnt"
                label="24H%"
                currentField={sortField}
                currentDirection={sortDirection}
                onSort={handleSort}
                align="right"
              />
              <HeaderCell
                field="volume24h"
                label="Объём"
                currentField={sortField}
                currentDirection={sortDirection}
                onSort={handleSort}
                align="right"
              />
              <th className="px-2 py-2 text-[10px] font-medium text-gray-400 uppercase tracking-wider text-center">
                <Eye className="h-3 w-3 mx-auto" />
              </th>
            </tr>
          </thead>

          <tbody>
            {filteredPairs.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-2 py-8 text-center text-gray-500 text-xs">
                  {filterText ? 'Нет пар, соответствующих фильтру' : 'Загрузка...'}
                </td>
              </tr>
            ) : (
              filteredPairs.map((pair) => {
                const flashColor = flashedRows[pair.symbol];
                const isSelected = selectedSymbol === pair.symbol;
                const isPositive = pair.price24hPcnt >= 0;

                return (
                    <tr
                        key={pair.symbol}
                        className={cn(
                            'hover:bg-gray-800/50 transition-colors border-b border-gray-800/50',
                            isSelected && 'bg-primary/10',
                            flashColor === 'green' && 'animate-flash-green',
                            flashColor === 'red' && 'animate-flash-red'
                        )}
                    >
                      {/* Символ */}
                      <td className="px-2 py-2 font-medium text-white whitespace-nowrap">
                        {pair.symbol.replace('USDT', '')}
                        <span className="text-gray-500 text-[9px]">USDT</span>
                      </td>

                      {/* Цена */}
                      <td className="px-2 py-2 text-gray-300 text-right whitespace-nowrap">
                        {formatPrice(pair.lastPrice)}
                      </td>

                      {/* Изменение 24H */}
                      <td className={cn(
                          'px-2 py-2 text-right font-medium whitespace-nowrap',
                          isPositive ? 'text-success' : 'text-destructive'
                      )}>
                        <div className="flex items-center justify-end gap-0.5">
                          {isPositive ? (
                              <TrendingUp className="h-2.5 w-2.5"/>
                          ) : (
                              <TrendingDown className="h-2.5 w-2.5"/>
                          )}
                          <span>{formatPercentage(pair.price24hPcnt)}</span>
                        </div>
                      </td>

                      {/* Объём */}
                      <td className="px-2 py-2 text-gray-400 text-right whitespace-nowrap">
                        ${formatNumber(pair.volume24h, 1)}
                      </td>

                      {/* Кнопка выбора */}
                      <td className="px-2 py-2 text-center">
                        <button
                            onClick={() => onSelectPair(pair.symbol)}
                            className={cn(
                                'p-1 rounded hover:bg-gray-700 transition-colors',
                                // Проверяем оба режима: единичный и множественный выбор
                                (selectedSymbol === pair.symbol || selectedSymbol?.includes(pair.symbol)) &&
                                'bg-primary hover:bg-primary/90 text-white'
                            )}
                            title={selectedSymbol?.includes(pair.symbol) ? "Убрать с графиков" : "Показать график"}
                        >
                          <Eye className="h-3 w-3"/>
                        </button>
                      </td>
                    </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Футер с информацией */}
      <div className="flex-shrink-0 px-3 py-2 border-t border-gray-800 bg-gray-900/50">
        <div className="text-[9px] text-gray-500">
          Min volume: 4M USDT
        </div>
      </div>
    </div>
  );
}