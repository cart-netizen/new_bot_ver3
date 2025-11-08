// frontend/src/pages/ChartsPage.tsx

import { useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useScreenerStore } from '../store/screenerStore';
import { PriceChart } from '../components/market/PriceChart';
import { X, TrendingUp, Activity } from 'lucide-react';

/**
 * Страница графиков.
 * Отображает графики выбранных торговых пар (по 3 в ряд).
 * Таймфрейм: 1 минута.
 * Обновление: каждые 15 секунд (автоматически в PriceChart).
 */
export function ChartsPage() {
  const { pairs, fetchPairs, togglePairSelection, isLoading } = useScreenerStore();

  /**
   * Загрузка пар при монтировании, если их еще нет.
   */
  useEffect(() => {
    if (pairs.length === 0 && !isLoading) {
      fetchPairs();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Получаем только выбранные пары.
   */
  const selectedPairs = useMemo(() => {
    return pairs.filter(p => p.is_selected);
  }, [pairs]);

  /**
   * Обработчик удаления пары из графиков.
   */
  const handleRemovePair = (symbol: string) => {
    togglePairSelection(symbol);
  };

  // Состояние загрузки
  if (isLoading && pairs.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Activity className="h-12 w-12 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-gray-400">Загрузка данных...</p>
        </div>
      </div>
    );
  }

  // Пустое состояние - нет выбранных пар
  if (selectedPairs.length === 0) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold">Графики</h1>
            <p className="text-gray-400 mt-1">
              Выбранные торговые пары отображаются здесь
            </p>
          </div>
        </div>

        <div className="bg-surface border border-gray-800 rounded-lg p-12 text-center">
          <TrendingUp className="h-16 w-16 mx-auto mb-4 text-gray-600" />
          <h2 className="text-xl font-semibold mb-2">Нет выбранных пар</h2>
          <p className="text-gray-400 max-w-md mx-auto mb-6">
            Перейдите на страницу <span className="text-primary font-medium">Dashboard</span> или{' '}
            <span className="text-primary font-medium">Скринер</span>, чтобы выбрать торговые пары
            для отображения графиков. Нажмите на кнопку{' '}
            <span className="text-primary font-medium">Plus</span> рядом с нужной парой.
          </p>
          <div className="flex gap-3 justify-center">
            <Link
              to="/dashboard"
              className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/80 transition"
            >
              Перейти к Dashboard
            </Link>
            <Link
              to="/screener"
              className="px-4 py-2 bg-surface border border-gray-700 rounded-lg hover:bg-gray-800 transition"
            >
              Перейти к Скринеру
            </Link>
          </div>
        </div>
      </div>
    );
  }

  // Отображение графиков
  return (
    <div className="p-6">
      {/* Заголовок */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold">Графики</h1>
          <p className="text-gray-400 mt-1">
            Отображено {selectedPairs.length} {selectedPairs.length === 1 ? 'график' : 'графиков'}
          </p>
        </div>

        {/* Статистика */}
        <div className="px-4 py-2 bg-surface border border-gray-800 rounded-lg">
          <p className="text-xs text-gray-400">Выбранных пар</p>
          <p className="text-xl font-bold text-primary">{selectedPairs.length}</p>
        </div>
      </div>

      {/* Сетка графиков (по 3 в ряд) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {selectedPairs.map((pair) => (
          <ChartCard
            key={pair.symbol}
            symbol={pair.symbol}
            onRemove={handleRemovePair}
          />
        ))}
      </div>
    </div>
  );
}

/**
 * Компонент карточки графика с кнопкой удаления.
 */
function ChartCard({
  symbol,
  onRemove,
}: {
  symbol: string;
  onRemove: (symbol: string) => void;
}) {
  return (
    <div className="relative bg-surface border border-gray-800 rounded-lg overflow-hidden">
      {/* Кнопка удаления (крест) */}
      <button
        onClick={() => onRemove(symbol)}
        className="absolute top-3 right-3 z-10 p-1.5 bg-destructive/10 hover:bg-destructive/20 border border-destructive/50 rounded-lg transition group"
        title={`Убрать ${symbol} из графиков`}
      >
        <X className="h-4 w-4 text-destructive group-hover:scale-110 transition-transform" />
      </button>

      {/* График */}
      <div className="p-2">
        <PriceChart symbol={symbol} />
      </div>
    </div>
  );
}
