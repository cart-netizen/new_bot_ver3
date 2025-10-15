// frontend/src/components/charts/ChartWidget.tsx
/**
 * Виджет графика для торговой пары.
 *
 * Использует TradingView Lightweight Charts для отображения.
 *
 * Функционал:
 * - Отображение свечей на 5-минутном таймфрейме
 * - Автоматическое обновление каждые 15 секунд
 * - Индикация загрузки и ошибок
 * - Кнопка удаления графика
 * - Responsive дизайн
 */

import { useEffect, useRef, useState } from 'react';
import { createChart, CandlestickSeries } from 'lightweight-charts';
import type { IChartApi, ISeriesApi, UTCTimestamp } from 'lightweight-charts';
import { X, TrendingUp, Loader2, AlertCircle } from 'lucide-react';
import { cn } from '../../utils/helpers';
import type { Candle } from '../../store/chartsStore';

/**
 * Props компонента.
 */
interface ChartWidgetProps {
  /**
   * Символ торговой пары.
   */
  symbol: string;

  /**
   * Данные свечей.
   */
  candles: Candle[];

  /**
   * Статус загрузки.
   */
  isLoading?: boolean;

  /**
   * Сообщение об ошибке.
   */
  error?: string | null;

  /**
   * Callback при удалении графика.
   */
  onRemove: (symbol: string) => void;

  /**
   * Высота графика (px).
   */
  height?: number;
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
 * Компонент виджета графика.
 */
export function ChartWidget({
  symbol,
  candles,
  isLoading = false,
  error = null,
  onRemove,
  height = 300,
}: ChartWidgetProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [priceChange, setPriceChange] = useState<number>(0);

  /**
   * Инициализация графика.
   */
  useEffect(() => {
    if (!chartContainerRef.current || candles.length === 0) {
      return;
    }

    // Создаем график
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height,
      layout: {
        background: { color: '#1F2937' },
        textColor: '#D1D5DB',
      },
      grid: {
        vertLines: { color: '#374151' },
        horzLines: { color: '#374151' },
      },
      crosshair: {
        mode: 1, // Normal
      },
      rightPriceScale: {
        borderColor: '#4B5563',
      },
      timeScale: {
        borderColor: '#4B5563',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // Создаем серию свечей
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10B981',
      downColor: '#EF4444',
      borderUpColor: '#10B981',
      borderDownColor: '#EF4444',
      wickUpColor: '#10B981',
      wickDownColor: '#EF4444',
    });

    // Конвертируем данные в формат Lightweight Charts
    const chartData = candles.map(candle => ({
      time: candle.time as UTCTimestamp,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    }));

    // Устанавливаем данные
    candlestickSeries.setData(chartData);

    // Подгоняем видимую область под данные
    chart.timeScale().fitContent();

    // Сохраняем ссылки
    chartRef.current = chart;
    seriesRef.current = candlestickSeries;

    // Обновляем текущую цену и изменение
    if (candles.length > 0) {
      const lastCandle = candles[candles.length - 1];
      const firstCandle = candles[0];

      setCurrentPrice(lastCandle.close);

      if (firstCandle.open > 0) {
        const change = ((lastCandle.close - firstCandle.open) / firstCandle.open) * 100;
        setPriceChange(change);
      }
    }

    // Cleanup при размонтировании
    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [candles, height]);

  /**
   * Обработка изменения размера окна.
   */
  useEffect(() => {
    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  /**
   * Обновление данных при изменении свечей.
   */
  useEffect(() => {
    if (seriesRef.current && candles.length > 0) {
      const chartData = candles.map(candle => ({
        time: candle.time as UTCTimestamp,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
      }));

      seriesRef.current.setData(chartData);

      // Обновляем текущую цену
      const lastCandle = candles[candles.length - 1];
      const firstCandle = candles[0];

      setCurrentPrice(lastCandle.close);

      if (firstCandle.open > 0) {
        const change = ((lastCandle.close - firstCandle.open) / firstCandle.open) * 100;
        setPriceChange(change);
      }
    }
  }, [candles]);

  /**
   * Рендер состояния загрузки.
   */
  if (isLoading && candles.length === 0) {
    return (
      <div className="bg-surface rounded-lg border border-gray-800 overflow-hidden">
        {/* Заголовок */}
        <div className="flex items-center justify-between p-3 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-gray-400" />
            <h3 className="font-semibold text-white text-sm">{symbol}</h3>
          </div>
          <button
            onClick={() => onRemove(symbol)}
            className="p-1 hover:bg-gray-700 rounded transition-colors"
            title="Удалить график"
          >
            <X className="h-4 w-4 text-gray-400 hover:text-white" />
          </button>
        </div>

        {/* Загрузка */}
        <div
          className="flex items-center justify-center bg-gray-900"
          style={{ height: `${height}px` }}
        >
          <div className="text-center">
            <Loader2 className="h-8 w-8 text-primary animate-spin mx-auto mb-2" />
            <p className="text-sm text-gray-400">Загрузка данных...</p>
          </div>
        </div>
      </div>
    );
  }

  /**
   * Рендер состояния ошибки.
   */
  if (error) {
    return (
      <div className="bg-surface rounded-lg border border-gray-800 overflow-hidden">
        {/* Заголовок */}
        <div className="flex items-center justify-between p-3 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-gray-400" />
            <h3 className="font-semibold text-white text-sm">{symbol}</h3>
          </div>
          <button
            onClick={() => onRemove(symbol)}
            className="p-1 hover:bg-gray-700 rounded transition-colors"
            title="Удалить график"
          >
            <X className="h-4 w-4 text-gray-400 hover:text-white" />
          </button>
        </div>

        {/* Ошибка */}
        <div
          className="flex items-center justify-center bg-gray-900"
          style={{ height: `${height}px` }}
        >
          <div className="text-center px-4">
            <AlertCircle className="h-8 w-8 text-destructive mx-auto mb-2" />
            <p className="text-sm text-gray-400">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  /**
   * Рендер графика.
   */
  return (
    <div className="bg-surface rounded-lg border border-gray-800 overflow-hidden">
      {/* Заголовок с информацией */}
      <div className="flex items-center justify-between p-3 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <TrendingUp className="h-4 w-4 text-gray-400" />
          <div>
            <h3 className="font-semibold text-white text-sm">{symbol}</h3>
            {currentPrice !== null && (
              <div className="flex items-center gap-2 text-xs">
                <span className="text-gray-400">${formatPrice(currentPrice)}</span>
                <span className={cn(
                  'font-medium',
                  priceChange >= 0 ? 'text-success' : 'text-destructive'
                )}>
                  {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Кнопка удаления */}
        <button
          onClick={() => onRemove(symbol)}
          className="p-1 hover:bg-gray-700 rounded transition-colors"
          title="Удалить график"
        >
          <X className="h-4 w-4 text-gray-400 hover:text-white" />
        </button>
      </div>

      {/* График */}
      <div
        ref={chartContainerRef}
        className="bg-gray-900"
        style={{ height: `${height}px` }}
      />

      {/* Футер с информацией */}
      <div className="px-3 py-2 border-t border-gray-800 bg-gray-900/50">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <span>5m таймфрейм</span>
          <span>{candles.length} свечей</span>
        </div>
      </div>
    </div>
  );
}