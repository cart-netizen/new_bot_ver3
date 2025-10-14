// frontend/src/components/screener/ScreenerTable.tsx
/**
 * Таблица скринера торговых пар.
 *
 * Функционал:
 * - Отображение пар с volume > 4M USDT
 * - Динамика изменения цены по таймфреймам (1m, 3m, 5m, 15m)
 * - Сортировка по всем колонкам
 * - Фильтрация по названию пары
 * - Real-time обновление через WebSocket
 *
 * Дизайн:
 * - Адаптирован из существующего скринера в репозитории
 * - Использует Tailwind CSS для стилизации
 * - Анимация изменения цен (flash green/red)
 */

import { useState, useEffect, useRef } from 'react';
import { useScreenerStore, type SortField, type SortDirection } from '../../store/screenerStore';
import { ArrowUpDown, ArrowUp, ArrowDown, TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '../../utils/helpers';

/**
 * Форматирование числа с разделителями тысяч.
 */
function formatNumber(value: number, decimals: number = 2): string {
  return value.toLocaleString('ru-RU', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
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
interface SortableHeaderProps {
  field: SortField;
  label: string;
  currentField: SortField;
  currentDirection: SortDirection;
  onSort: (field: SortField) => void;
}

function SortableHeader({
  field,
  label,
  currentField,
  currentDirection,
  onSort
}: SortableHeaderProps) {
  const isActive = currentField === field;

  return (
    <th
      className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:text-white transition-colors select-none"
      onClick={() => onSort(field)}
    >
      <div className="flex items-center gap-2">
        <span>{label}</span>
        {isActive ? (
          currentDirection === 'asc' ? (
            <ArrowUp className="h-4 w-4 text-primary" />
          ) : (
            <ArrowDown className="h-4 w-4 text-primary" />
          )
        ) : (
          <ArrowUpDown className="h-4 w-4 opacity-50" />
        )}
      </div>
    </th>
  );
}

/**
 * Компонент ячейки с изменением цены по таймфрейму.
 */
interface TimeframeChangeCellProps {
  change: number | null;
  loading?: boolean;
}

function TimeframeChangeCell({ change, loading }: TimeframeChangeCellProps) {
  if (loading || change === null) {
    return (
      <td className="px-4 py-3 text-sm text-gray-500 text-center">
        —
      </td>
    );
  }

  const isPositive = change >= 0;
  const Icon = isPositive ? TrendingUp : TrendingDown;

  return (
    <td className={cn(
      'px-4 py-3 text-sm font-medium text-center',
      isPositive ? 'text-success' : 'text-destructive'
    )}>
      <div className="flex items-center justify-center gap-1">
        <Icon className="h-3 w-3" />
        <span>{formatPercentage(change)}</span>
      </div>
    </td>
  );
}

/**
 * Компонент таблицы скринера.
 */
export function ScreenerTable() {
  const {
    pairs,
    filterText,
    sortField,
    sortDirection,
    isConnected,
    setFilterText,
    setSorting,
    getSortedPairs,
  } = useScreenerStore();

  const [flashedCells, setFlashedCells] = useState<Record<string, 'green' | 'red'>>({});
  const previousPricesRef = useRef<Record<string, number>>({});

  /**
   * Обработка клика по заголовку для сортировки.
   */
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      // Переключаем направление
      setSorting(field, sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      // Новое поле, сортируем по убыванию
      setSorting(field, 'desc');
    }
  };

  /**
   * Эффект для flash-анимации при изменении цены.
   */
  useEffect(() => {
    const currentPairs = getSortedPairs();

    currentPairs.forEach(pair => {
      const previousPrice = previousPricesRef.current[pair.symbol];

      if (previousPrice && previousPrice !== pair.lastPrice) {
        const flashColor = pair.lastPrice > previousPrice ? 'green' : 'red';

        setFlashedCells(prev => ({
          ...prev,
          [pair.symbol]: flashColor,
        }));

        // Убираем flash через 1 секунду
        setTimeout(() => {
          setFlashedCells(prev => {
            const updated = { ...prev };
            delete updated[pair.symbol];
            return updated;
          });
        }, 1000);
      }

      previousPricesRef.current[pair.symbol] = pair.lastPrice;
    });
  }, [pairs, getSortedPairs]);

  const sortedPairs = getSortedPairs();

  return (
    <div className="bg-surface rounded-lg border border-gray-800 overflow-hidden">
      {/* Заголовок с фильтром и статусом */}
      <div className="p-4 border-b border-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-semibold text-white">
            Скринер Фьючерсов Bybit
          </h2>
          <div className="flex items-center gap-2">
            <div className={cn(
              'w-2 h-2 rounded-full',
              isConnected ? 'bg-success animate-pulse' : 'bg-gray-500'
            )} />
            <span className="text-sm text-gray-400">
              {isConnected ? 'Подключено' : 'Отключено'}
            </span>
          </div>
        </div>

        {/* Фильтр */}
        <input
          type="text"
          placeholder="Поиск по тикеру..."
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all"
        />
      </div>

      {/* Таблица */}
      <div className="overflow-x-auto max-h-[calc(100vh-200px)] overflow-y-auto">
        <table className="w-full">
          <thead className="bg-gray-900 sticky top-0 z-10">
            <tr>
              <SortableHeader
                field="symbol"
                label="Пара"
                currentField={sortField}
                currentDirection={sortDirection}
                onSort={handleSort}
              />
              <SortableHeader
                field="lastPrice"
                label="Цена"
                currentField={sortField}
                currentDirection={sortDirection}
                onSort={handleSort}
              />
              <SortableHeader
                field="price24hPcnt"
                label="Изм. 24Ч"
                currentField={sortField}
                currentDirection={sortDirection}
                onSort={handleSort}
              />
              <SortableHeader
                field="volume24h"
                label="Объём 24Ч"
                currentField={sortField}
                currentDirection={sortDirection}
                onSort={handleSort}
              />
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase tracking-wider">
                1m
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase tracking-wider">
                3m
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase tracking-wider">
                5m
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase tracking-wider">
                15m
              </th>
            </tr>
          </thead>

          <tbody className="divide-y divide-gray-800">
            {sortedPairs.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-4 py-12 text-center text-gray-500">
                  {filterText ? 'Нет пар, соответствующих фильтру' : 'Ожидание данных...'}
                </td>
              </tr>
            ) : (
              sortedPairs.map((pair) => {
                const flashColor = flashedCells[pair.symbol];
                const rowClassName = cn(
                  'hover:bg-gray-800/50 transition-colors',
                  flashColor === 'green' && 'animate-flash-green',
                  flashColor === 'red' && 'animate-flash-red'
                );

                return (
                  <tr key={pair.symbol} className={rowClassName}>
                    {/* Символ */}
                    <td className="px-4 py-3 text-sm font-medium text-white whitespace-nowrap">
                      {pair.symbol}
                    </td>

                    {/* Цена */}
                    <td className="px-4 py-3 text-sm text-gray-300 whitespace-nowrap">
                      ${formatNumber(pair.lastPrice, 2)}
                    </td>

                    {/* Изменение за 24ч */}
                    <td className={cn(
                      'px-4 py-3 text-sm font-medium whitespace-nowrap',
                      pair.price24hPcnt >= 0 ? 'text-success' : 'text-destructive'
                    )}>
                      {formatPercentage(pair.price24hPcnt)}
                    </td>

                    {/* Объём за 24ч */}
                    <td className="px-4 py-3 text-sm text-gray-300 whitespace-nowrap">
                      ${formatNumber(pair.volume24h, 0)}
                    </td>

                    {/* Динамика по таймфреймам */}
                    <TimeframeChangeCell
                      change={pair.changes['1m']?.change_percent ?? null}
                    />
                    <TimeframeChangeCell
                      change={pair.changes['3m']?.change_percent ?? null}
                    />
                    <TimeframeChangeCell
                      change={pair.changes['5m']?.change_percent ?? null}
                    />
                    <TimeframeChangeCell
                      change={pair.changes['15m']?.change_percent ?? null}
                    />
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Футер со статистикой */}
      <div className="p-4 border-t border-gray-800 flex items-center justify-between bg-gray-900/50">
        <div className="text-sm text-gray-400">
          Отображено пар: <span className="font-semibold text-white">{sortedPairs.length}</span>
        </div>
        <div className="text-xs text-gray-500">
          Обновление в реальном времени через WebSocket
        </div>
      </div>
    </div>
  );
}