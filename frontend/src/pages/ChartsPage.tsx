// frontend/src/pages/ChartsPage.tsx
/**
 * Страница с графиками торговых пар.
 *
 * Функционал:
 * - Отображение 3 графиков в ряд
 * - Таймфрейм: 5 минут
 * - Обновление каждые 15 секунд
 * - Интеграция с TradingPairsList
 * - До 12 графиков одновременно (4 ряда)
 */

import { useEffect } from 'react';
import { ChartsGrid } from '../components/charts/ChartsGrid';
import { TradingPairsList } from '../components/market/TradingPairsList';
import { useChartsStore, CHARTS_MEMORY_CONFIG } from '../store/chartsStore';
import { BarChart3, Info } from 'lucide-react';
import { toast } from 'sonner';

/**
 * Компонент страницы графиков.
 */
export function ChartsPage() {
  const {
    selectedSymbols,
    toggleSymbol,
    canAddMoreCharts,
    reset
  } = useChartsStore();

  /**
   * Обработка выбора пары из списка.
   * Добавляет/удаляет график.
   */
  const handleSelectPair = (symbol: string) => {
    const isAlreadySelected = selectedSymbols.includes(symbol);

    if (isAlreadySelected) {
      // Удаляем график
      toggleSymbol(symbol);
      toast.info(`График ${symbol} удален`);
    } else {
      // Проверяем лимит
      if (!canAddMoreCharts()) {
        toast.warning('Достигнут лимит графиков', {
          description: `Максимум ${CHARTS_MEMORY_CONFIG.MAX_CHARTS} графиков одновременно`,
        });
        return;
      }

      // Добавляем график
      toggleSymbol(symbol);
      toast.success(`График ${symbol} добавлен`);
    }
  };

  /**
   * Cleanup при размонтировании.
   */
  useEffect(() => {
    return () => {
      // Не сбрасываем состояние при размонтировании,
      // чтобы сохранить выбранные графики при переходе между страницами
      // reset();
    };
  }, []);

  return (
    // Flex контейнер для горизонтального расположения
    <div className="flex h-full">
      {/* ==================== Список торговых пар ==================== */}
      <TradingPairsList
        onSelectPair={handleSelectPair}
        selectedSymbol={null} // Здесь не нужен единичный выбор
        selectedSymbols={selectedSymbols} // ДОБАВИТЬ эту строку

      />

      {/* ==================== Основной контент ==================== */}
      <div className="flex-1 overflow-auto">
        <div className="p-6 space-y-6">
          {/* Заголовок */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold flex items-center gap-3">
                <BarChart3 className="h-8 w-8 text-primary" />
                Графики
              </h1>
              <p className="text-gray-400 mt-1">
                Мониторинг графиков торговых пар в реальном времени
              </p>
            </div>

            {/* Счетчик выбранных графиков */}
            <div className="text-right">
              <div className="text-2xl font-bold text-white">
                {selectedSymbols.length} <span className="text-gray-500">/ {CHARTS_MEMORY_CONFIG.MAX_CHARTS}</span>
              </div>
              <div className="text-xs text-gray-500">Графиков</div>
            </div>
          </div>

          {/* Информационная панель */}
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <Info className="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-sm font-semibold text-blue-500 mb-2">
                  Как использовать
                </h3>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• Выберите торговые пары в списке слева, нажав на кнопку с иконкой глаза</li>
                  <li>• Графики отображаются по 3 в ряд на 5-минутном таймфрейме</li>
                  <li>• Автоматическое обновление происходит каждые 15 секунд</li>
                  <li>• Максимум {CHARTS_MEMORY_CONFIG.MAX_CHARTS} графиков одновременно</li>
                  <li>• Удалить график можно кнопкой ×  в правом верхнем углу графика</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Сетка графиков */}
          <ChartsGrid />

          {/* Дополнительная информация */}
          {selectedSymbols.length > 0 && (
            <div className="bg-gray-800/30 border border-gray-700 rounded-lg p-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="text-gray-500 mb-1">Таймфрейм</div>
                  <div className="text-white font-semibold">5 минут</div>
                </div>
                <div>
                  <div className="text-gray-500 mb-1">Интервал обновления</div>
                  <div className="text-white font-semibold">
                    {CHARTS_MEMORY_CONFIG.UPDATE_INTERVAL / 1000} секунд
                  </div>
                </div>
                <div>
                  <div className="text-gray-500 mb-1">Активных графиков</div>
                  <div className="text-white font-semibold">
                    {selectedSymbols.length} из {CHARTS_MEMORY_CONFIG.MAX_CHARTS}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}