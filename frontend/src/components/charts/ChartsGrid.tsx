// frontend/src/components/charts/ChartsGrid.tsx
/**
 * Компонент сетки графиков.
 *
 * Функционал:
 * - Отображение 3 графиков в ряд
 * - Автоматическая загрузка данных свечей
 * - Обновление каждые 15 секунд
 * - Responsive layout
 */

import { useEffect, useCallback } from 'react';
import { ChartWidget } from './ChartWidget';
import { useChartsStore, CHARTS_MEMORY_CONFIG } from '../../store/chartsStore';
import { apiService } from '../../services/api.service';
import { toast } from 'sonner';
import { LayoutGrid, Plus } from 'lucide-react';

/**
 * Компонент сетки графиков.
 */
export function ChartsGrid() {
  const {
    selectedSymbols,
    chartsData,
    removeSymbol,
    updateChartData,
    setChartLoading,
    setChartError,
    startAutoUpdate,
    stopAutoUpdate,
    canAddMoreCharts,
  } = useChartsStore();

  /**
   * Загрузка данных свечей для символа.
   */
  const loadChartData = useCallback(async (symbol: string) => {
    try {
      setChartLoading(symbol, true);

      console.log(`[ChartsGrid] Загрузка данных для ${symbol}`);

      // Получаем свечи с API
      // Таймфрейм: 5 минут, лимит: 200 свечей (~16 часов)
      const response = await apiService.get(`/api/market/klines/${symbol}`, {
        params: {
          interval: '5', // 5 минут
          limit: CHARTS_MEMORY_CONFIG.MAX_CANDLES_PER_CHART,
        },
      });

      if (!response || !response.candles) {
        throw new Error('Нет данных свечей');
      }

      // Конвертируем данные в нужный формат
      const candles = response.candles.map((kline: any) => ({
        time: Math.floor(kline.timestamp / 1000), // конвертируем в секунды
        open: parseFloat(kline.open),
        high: parseFloat(kline.high),
        low: parseFloat(kline.low),
        close: parseFloat(kline.close),
        volume: parseFloat(kline.volume),
      }));

      // Сортируем по времени
      candles.sort((a, b) => a.time - b.time);

      updateChartData(symbol, candles);

      console.log(`[ChartsGrid] Загружено ${candles.length} свечей для ${symbol}`);
    } catch (error: any) {
      console.error(`[ChartsGrid] Ошибка загрузки данных для ${symbol}:`, error);

      const errorMessage = error.response?.data?.detail || error.message || 'Ошибка загрузки данных';
      setChartError(symbol, errorMessage);

      toast.error(`Ошибка загрузки графика ${symbol}`, {
        description: errorMessage,
      });
    }
  }, [updateChartData, setChartLoading, setChartError]);

  /**
   * Загрузка данных для всех выбранных символов.
   */
  const loadAllCharts = useCallback(() => {
    console.log(`[ChartsGrid] Обновление ${selectedSymbols.length} графиков`);

    selectedSymbols.forEach(symbol => {
      loadChartData(symbol);
    });
  }, [selectedSymbols, loadChartData]);

  /**
   * Эффект для начальной загрузки данных.
   */
  useEffect(() => {
    if (selectedSymbols.length === 0) {
      return;
    }

    // Загружаем данные для всех символов
    loadAllCharts();
  }, [selectedSymbols, loadAllCharts]);

  /**
   * Эффект для автоматического обновления.
   */
  useEffect(() => {
    if (selectedSymbols.length === 0) {
      stopAutoUpdate();
      return;
    }

    // Запускаем автообновление
    startAutoUpdate(loadAllCharts);

    return () => {
      stopAutoUpdate();
    };
  }, [selectedSymbols.length, loadAllCharts, startAutoUpdate, stopAutoUpdate]);

  /**
   * Обработка удаления графика.
   */
  const handleRemoveChart = (symbol: string) => {
    console.log(`[ChartsGrid] Удаление графика ${symbol}`);
    removeSymbol(symbol);
    toast.info(`График ${symbol} удален`);
  };

  /**
   * Рендер пустого состояния.
   */
  if (selectedSymbols.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-[400px] bg-surface rounded-lg border border-gray-800">
        <div className="text-center px-6 py-12">
          <LayoutGrid className="h-16 w-16 text-gray-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">
            Графики не выбраны
          </h3>
          <p className="text-gray-400 mb-4 max-w-md">
            Выберите торговые пары в списке слева, нажав на кнопку с иконкой глаза.
            Можно добавить до {CHARTS_MEMORY_CONFIG.MAX_CHARTS} графиков.
          </p>
          <div className="flex items-center justify-center gap-2 text-sm text-gray-500">
            <Plus className="h-4 w-4" />
            <span>Максимум {CHARTS_MEMORY_CONFIG.MAX_CHARTS} графиков</span>
          </div>
        </div>
      </div>
    );
  }

  /**
   * Рендер сетки графиков.
   */
  return (
    <div className="space-y-4">
      {/* Информационная панель */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-400">
            Отображено графиков: <span className="font-semibold text-white">{selectedSymbols.length}</span> из {CHARTS_MEMORY_CONFIG.MAX_CHARTS}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Обновление каждые {CHARTS_MEMORY_CONFIG.UPDATE_INTERVAL / 1000} секунд
          </p>
        </div>

        {!canAddMoreCharts() && (
          <div className="text-xs text-yellow-500 bg-yellow-500/10 px-3 py-1.5 rounded-lg border border-yellow-500/30">
            Достигнут лимит графиков
          </div>
        )}
      </div>

      {/* Сетка графиков: 3 в ряд */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        {selectedSymbols.map((symbol) => {
          const chartData = chartsData[symbol];

          return (
            <ChartWidget
              key={symbol}
              symbol={symbol}
              candles={chartData?.candles || []}
              isLoading={chartData?.isLoading || false}
              error={chartData?.error || null}
              onRemove={handleRemoveChart}
              height={300}
            />
          );
        })}
      </div>

      {/* Подсказка для добавления графиков */}
      {canAddMoreCharts() && (
        <div className="text-center py-6 bg-gray-800/30 rounded-lg border border-gray-700 border-dashed">
          <p className="text-sm text-gray-400">
            Вы можете добавить еще {CHARTS_MEMORY_CONFIG.MAX_CHARTS - selectedSymbols.length} график(ов)
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Используйте кнопку <Plus className="h-3 w-3 inline" /> в списке торговых пар
          </p>
        </div>
      )}
    </div>
  );
}