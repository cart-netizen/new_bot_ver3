// frontend/src/components/market/PriceChart.tsx

import { useEffect, useState, useMemo, useCallback, useRef } from 'react';
import { Card } from '../ui/Card';
import { candlesApi } from '../../api/candles.api';
import type { Candle } from '../../types/candle.types';
import { CandleInterval } from '../../types/candle.types';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { TrendingUp, TrendingDown, RefreshCw } from 'lucide-react';

interface PriceChartProps {
  symbol: string;
  loading?: boolean;
}

/**
 * Компонент для отображения графика цены торговой пары.
 * Использует минутные свечи от биржи.
 * Автообновление: каждые 5 секунд.
 */
export function PriceChart({ symbol, loading: externalLoading = false }: PriceChartProps) {
  const [candles, setCandles] = useState<Candle[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Используем ref для предотвращения множественных запросов
  const isLoadingRef = useRef(false);
  const intervalRef = useRef<number | null>(null);

  /**
   * Загрузка свечей с биржи.
   */
  const fetchCandles = useCallback(async (showLoadingIndicator = true) => {
    // Предотвращаем множественные одновременные запросы
    if (isLoadingRef.current) {
      console.log(`[PriceChart] Запрос для ${symbol} уже выполняется, пропускаем`);
      return;
    }

    try {
      isLoadingRef.current = true;

      // Показываем индикатор загрузки только при первом запросе
      if (showLoadingIndicator) {
        setLoading(true);
      }
      setError(null);

      console.log(`[PriceChart] Загрузка свечей для ${symbol}...`);

      const response = await candlesApi.getCandles(
        symbol,
        CandleInterval.MIN_1, // Минутный таймфрейм
        100 // Последние 100 свечей
      );

      setCandles(response.candles);
      setLastUpdate(new Date());
      console.log(`[PriceChart] ✓ Загружено ${response.candles.length} свечей для ${symbol}`);

    } catch (err) {
      console.error(`[PriceChart] ✗ Ошибка загрузки свечей для ${symbol}:`, err);

      // Показываем ошибку только если данных еще нет
      setError('Не удалось загрузить данные');
    } finally {
      setLoading(false);
      isLoadingRef.current = false;
    }
  }, [symbol]); // Убрали candles.length из зависимостей

  /**
   * Загрузка данных при монтировании и изменении символа.
   */
  useEffect(() => {
    console.log(`[PriceChart] useEffect triggered for ${symbol}`);

    // Очищаем предыдущий интервал
    if (intervalRef.current !== null) {
      console.log(`[PriceChart] Clearing previous interval`);
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Сбрасываем состояние при смене символа
    setCandles([]);
    setError(null);
    setLoading(true);

    // Загружаем данные для нового символа (с индикатором загрузки)
    fetchCandles(true);

    // Автообновление каждые 5 секунд (как в скринере)
    console.log(`[PriceChart] Setting up 5s interval for ${symbol}`);
    intervalRef.current = window.setInterval(() => {
      console.log(`[PriceChart] Auto-refresh triggered for ${symbol}`);
      // При автообновлении не показываем loader
      fetchCandles(false);
    }, 5000); // 5 секунд

    // Cleanup
    return () => {
      console.log(`[PriceChart] Cleanup for ${symbol}`);
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [symbol, fetchCandles]); // Добавили fetchCandles в зависимости

  /**
   * Ручное обновление данных.
   */
  const handleManualRefresh = useCallback(() => {
    console.log(`[PriceChart] Ручное обновление для ${symbol}`);
    fetchCandles(false);
  }, [fetchCandles, symbol]);

  /**
   * Подготовка данных для графика.
   */
  const chartData = useMemo(() => {
    if (!candles || candles.length === 0) {
      return [];
    }

    return candles.map((candle) => ({
      timestamp: candle.timestamp,
      time: new Date(candle.timestamp).toLocaleTimeString('ru-RU', {
        hour: '2-digit',
        minute: '2-digit',
      }),
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
      volume: candle.volume,
    }));
  }, [candles]);

  /**
   * Расчёт изменения цены за период.
   */
  const priceChange = useMemo(() => {
    if (chartData.length < 2) {
      return { amount: 0, percentage: 0 };
    }

    const first = chartData[0]?.open || 0;
    const last = chartData[chartData.length - 1]?.close || 0;
    const amount = last - first;
    const percentage = first > 0 ? (amount / first) * 100 : 0;

    return { amount, percentage };
  }, [chartData]);

  const isPositive = priceChange.amount >= 0;


  /**
   * Форматирование цены для отображения.
   */
  const formatPrice = (value: number): string => {
    if (value === 0) return '0';

    // Для больших цен (>= 1) - до 2 знаков
    if (value >= 1) {
      return value.toLocaleString('en-US', {
        minimumFractionDigits: 0,
        maximumFractionDigits: 2
      });
    }

    // Для малых цен (< 1) - до 6 знаков, убираем trailing zeros
    return value.toLocaleString('en-US', {
      minimumFractionDigits: 0,
      maximumFractionDigits: 6
    });
  };

  if (loading || externalLoading) {
    return (
      <Card className="p-4">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded mb-4 w-1/3"></div>
          <div className="h-64 bg-gray-700 rounded"></div>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">График: {symbol}</h3>
        <div className="h-64 flex flex-col items-center justify-center gap-4">
          <p className="text-destructive">{error}</p>
          <button
            onClick={handleManualRefresh}
            disabled={isLoadingRef.current}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RefreshCw className={`h-4 w-4 ${isLoadingRef.current ? 'animate-spin' : ''}`} />
            {isLoadingRef.current ? 'Загрузка...' : 'Повторить'}
          </button>
        </div>
      </Card>
    );
  }

  if (chartData.length === 0) {
    return (
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">График: {symbol}</h3>
        <div className="h-64 flex items-center justify-center">
          <p className="text-gray-400">Загрузка данных...</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-4">
      {/* Заголовок с изменением цены */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold">{symbol}</h3>
          <p className="text-xs text-gray-500">
            {chartData.length} свечей (1m) • {lastUpdate.toLocaleTimeString('ru-RU')}
          </p>
        </div>

        {/* Индикатор изменения */}
        <div className="flex items-center gap-2">
          {isPositive ? (
            <TrendingUp className="h-5 w-5 text-success" />
          ) : (
            <TrendingDown className="h-5 w-5 text-destructive" />
          )}
          <div className="text-right">
            <div
              className={`text-lg font-semibold ${
                isPositive ? 'text-success' : 'text-destructive'
              }`}
            >
              {isPositive ? '+' : ''}
              {priceChange.amount.toFixed(4)}
            </div>
            <div
              className={`text-sm ${
                isPositive ? 'text-success' : 'text-destructive'
              }`}
            >
              ({isPositive ? '+' : ''}
              {priceChange.percentage.toFixed(4)}%)
            </div>
          </div>
        </div>

        {/* Кнопка обновления */}
        <button
          onClick={handleManualRefresh}
          disabled={isLoadingRef.current}
          className={`p-2 rounded transition-colors ${
            isLoadingRef.current 
              ? 'opacity-50 cursor-not-allowed' 
              : 'hover:bg-gray-800'
          }`}
          title="Обновить данные"
        >
          <RefreshCw className={`h-4 w-4 text-gray-400 ${isLoadingRef.current ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* График */}
      <ResponsiveContainer width="100%" height={300} key={`chart-${symbol}`}>
        <LineChart data={chartData}>
          {/* Сетка */}
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />

          {/* Оси */}
          <XAxis
            dataKey="time"
            stroke="#9CA3AF"
            style={{ fontSize: '11px' }}
            interval="preserveStartEnd"
          />
          <YAxis
            stroke="#9CA3AF"
            style={{ fontSize: '11px' }}
            domain={['auto', 'auto']}
            tickFormatter={formatPrice}
          />

          {/* Tooltip */}
          <Tooltip
            contentStyle={{
              backgroundColor: '#1F2937',
              border: '1px solid #374151',
              borderRadius: '8px',
              color: '#fff',
            }}
            formatter={(value: number) => {
              return [formatPrice(value), 'Close'];
            }}
            labelStyle={{ color: '#9CA3AF', marginBottom: '8px' }}
          />

          {/* Легенда */}
          <Legend
            wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}
            iconType="line"
          />

          {/* Линия Close - основная цена закрытия */}
          <Line
            type="monotone"
            dataKey="close"
            stroke="#8B5CF6"
            strokeWidth={2}
            name="Close"
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Информация о последней свече */}
      {chartData.length > 0 && (
        <div className="mt-4 pt-3 border-t border-gray-800 flex items-center justify-between text-xs">
          <div>
            <p className="text-gray-400 mb-1">Последняя цена (Close)</p>
            <p className="font-mono text-purple-400 text-lg">
              {formatPrice(chartData[chartData.length - 1]?.close || 0)}
            </p>
          </div>
          <div className="text-right">
            <p className="text-gray-400 mb-1">Свечей на графике</p>
            <p className="font-mono">{chartData.length}</p>
          </div>
        </div>
      )}
    </Card>
  );
}