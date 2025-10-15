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
import type { CandlesResponse } from '../../types/candle.types';
import { AxiosError } from 'axios';

/**
 * Тип для сырых данных свечи от API (формат Bybit).
 * Массив: [timestamp, open, high, low, close, volume, turnover]
 */
interface RawKlineData {
  timestamp: number;
  open: string | number;
  high: string | number;
  low: string | number;
  close: string | number;
  volume: string | number;
  turnover?: string | number;
}

/**
 * Тип для данных свечи для графика (Lightweight Charts).
 */
interface ChartCandle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

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
      const response = await apiService.get(
        `/api/market/klines/${symbol}`,
        {
          params: {
            interval: '5', // 5 минут
            limit: CHARTS_MEMORY_CONFIG.MAX_CANDLES_PER_CHART,
          },
        }
      ) as CandlesResponse;

      if (!response || !response.candles) {
        throw new Error('Нет данных свечей');
      }

      // Конвертируем данные в нужный формат для графика
      const candles: ChartCandle[] = response.candles.map((kline: RawKlineData) => ({
        time: Math.floor(kline.timestamp / 1000), // конвертируем в секунды
        open: parseFloat(String(kline.open)),
        high: parseFloat(String(kline.high)),
        low: parseFloat(String(kline.low)),
        close: parseFloat(String(kline.close)),
        volume: parseFloat(String(kline.volume)),
      }));

      // Сортируем по времени (явная типизация параметров)
      candles.sort((a: ChartCandle, b: ChartCandle) => a.time - b.time);

      updateChartData(symbol, candles);

      console.log(`[ChartsGrid] Загружено ${candles.length} свечей для ${symbol}`);
    } catch (error: unknown) {
      console.error(`[ChartsGrid] Ошибка загрузки данных для ${symbol}:`, error);

      // Безопасная обработка ошибки
      let errorMessage = 'Ошибка загрузки данных';

      if (error instanceof AxiosError) {
        // Обработка ошибки Axios
        errorMessage = error.response?.data?.detail || error.message;
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }

      setChartError(symbol, errorMessage);

      toast.error(`Ошибка загрузки графика ${symbol}`, {
        description: errorMessage,
      });
    } finally {
      setChartLoading(symbol, false);
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
    if (selectedSymbols.length > 0) {
      loadAllCharts();
    }
  }, [selectedSymbols, loadAllCharts]);

  /**
   * Эффект для автообновления.
   */
  useEffect(() => {
    if (selectedSymbols.length === 0) {
      stopAutoUpdate();
      return;
    }

    // Запускаем автообновление с функцией загрузки
    startAutoUpdate(loadAllCharts);

    return () => {
      stopAutoUpdate();
    };
  }, [selectedSymbols.length, startAutoUpdate, stopAutoUpdate, loadAllCharts]);

  /**
   * Рендер пустого состояния.
   */
  if (selectedSymbols.length === 0) {
    return (
      <div className="bg-surface rounded-lg border border-gray-800 p-12 text-center">
        <LayoutGrid className="h-16 w-16 text-gray-600 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-white mb-2">
          Графики не выбраны
        </h3>
        <p className="text-gray-400 mb-4">
          Добавьте графики из списка инструментов
        </p>
        <button
          className="inline-flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary/90 text-white rounded-lg transition-colors"
          onClick={() => toast.info('Используйте список инструментов для добавления графиков')}
        >
          <Plus className="h-5 w-5" />
          Добавить график
        </button>
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
        <div className="text-sm text-gray-400">
          Отображено графиков: <span className="text-white font-semibold">{selectedSymbols.length}</span>
          {!canAddMoreCharts() && (
            <span className="text-yellow-500 ml-2">
              (достигнут максимум {CHARTS_MEMORY_CONFIG.MAX_CHARTS})
            </span>
          )}
        </div>
        <div className="text-xs text-gray-500">
          Автообновление каждые 15 секунд
        </div>
      </div>

      {/* Сетка графиков (3 колонки) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        {selectedSymbols.map(symbol => {
          const chartData = chartsData[symbol];

          return (
            <ChartWidget
              key={symbol}
              symbol={symbol}
              candles={chartData?.candles || []}
              isLoading={chartData?.isLoading || false}
              error={chartData?.error || null}
              onRemove={() => removeSymbol(symbol)}
              height={300}
            />
          );
        })}
      </div>

      {/* Совет */}
      <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
        <div className="text-sm text-gray-400">
          <strong className="text-blue-500">Совет:</strong> Графики автоматически обновляются.
          Для удаления графика нажмите кнопку "×" в правом верхнем углу.
        </div>
      </div>
    </div>
  );
}