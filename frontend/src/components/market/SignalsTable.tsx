// frontend/src/components/market/SignalsTable.tsx

import { Card } from '../ui/Card';
import type { TradingSignal } from '@/types/signal.types';
import { TrendingUp, TrendingDown, Zap } from 'lucide-react';

interface SignalsTableProps {
  signals: TradingSignal[];
  maxSignals?: number;
}

/**
 * Компонент для отображения таблицы торговых сигналов.
 * Показывает последние сигналы от стратегий в реальном времени.
 */
export function SignalsTable({ signals, maxSignals = 20 }: SignalsTableProps) {
  const displaySignals = signals.slice(0, maxSignals);

  /**
   * Получение цвета для типа сигнала.
   */
  const getSignalColor = (type: string): string => {
    return type === 'BUY' ? 'text-success' : 'text-destructive';
  };

  /**
   * Получение иконки для типа сигнала.
   */
  const getSignalIcon = (type: string) => {
    return type === 'BUY' ?
      <TrendingUp className="h-4 w-4" /> :
      <TrendingDown className="h-4 w-4" />;
  };

  /**
   * Получение цвета для силы сигнала.
   */
  const getStrengthColor = (strength: string): string => {
    switch (strength) {
      case 'STRONG':
        return 'text-yellow-500';
      case 'MEDIUM':
        return 'text-blue-500';
      case 'WEAK':
        return 'text-gray-400';
      default:
        return 'text-gray-400';
    }
  };

  /**
   * Получение бейджа для силы сигнала.
   */
  const getStrengthBadge = (strength: string): React.ReactElement => {
    const colors = {
      STRONG: 'bg-yellow-500/20 text-yellow-500 border-yellow-500/30',
      MEDIUM: 'bg-blue-500/20 text-blue-500 border-blue-500/30',
      WEAK: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
    };

    const labels = {
      STRONG: 'Сильный',
      MEDIUM: 'Средний',
      WEAK: 'Слабый',
    };

    return (
      <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${colors[strength as keyof typeof colors]}`}>
        {labels[strength as keyof typeof labels]}
      </span>
    );
  };

  /**
   * Форматирование времени.
   */
  const formatTime = (timestamp: number): string => {
    return new Date(timestamp).toLocaleTimeString('ru-RU');
  };

  /**
   * Форматирование цены.
   */
  const formatPrice = (price: number): string => {
    return price.toFixed(8);
  };

  /**
   * Форматирование confidence.
   */
  const formatConfidence = (confidence: number): string => {
    return `${(confidence * 100).toFixed(1)}%`;
  };

  if (displaySignals.length === 0) {
    return (
      <Card className="p-6">
        <div className="text-center text-gray-400">
          <Zap className="h-12 w-12 mx-auto mb-3 opacity-50" />
          <p className="text-sm">Пока нет торговых сигналов</p>
          <p className="text-xs mt-1">Сигналы появятся после запуска бота</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="overflow-hidden">
      {/* Заголовок */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-yellow-500" />
            <h3 className="text-lg font-semibold">Торговые Сигналы</h3>
          </div>
          <span className="text-xs text-gray-500">
            {signals.length} всего
          </span>
        </div>
      </div>

      {/* Таблица */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-800/50 text-xs text-gray-400">
            <tr>
              <th className="px-4 py-3 text-left font-medium">Время</th>
              <th className="px-4 py-3 text-left font-medium">Пара</th>
              <th className="px-4 py-3 text-left font-medium">Сигнал</th>
              <th className="px-4 py-3 text-right font-medium">Цена</th>
              <th className="px-4 py-3 text-center font-medium">Сила</th>
              <th className="px-4 py-3 text-right font-medium">Уверенность</th>
              <th className="px-4 py-3 text-left font-medium">Причина</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {displaySignals.map((signal, index) => (
              <tr
                key={`${signal.symbol}-${signal.timestamp}-${index}`}
                className="hover:bg-gray-800/30 transition-colors"
              >
                {/* Время */}
                <td className="px-4 py-3 text-xs text-gray-500 whitespace-nowrap">
                  {formatTime(signal.timestamp)}
                </td>

                {/* Пара */}
                <td className="px-4 py-3 text-sm font-medium">
                  {signal.symbol}
                </td>

                {/* Сигнал */}
                <td className="px-4 py-3">
                  <div className={`flex items-center gap-2 ${getSignalColor(signal.signal_type)}`}>
                    {getSignalIcon(signal.signal_type)}
                    <span className="text-sm font-semibold">
                      {signal.signal_type}
                    </span>
                  </div>
                </td>

                {/* Цена */}
                <td className="px-4 py-3 text-sm font-mono text-right">
                  {formatPrice(signal.price)}
                </td>

                {/* Сила */}
                <td className="px-4 py-3 text-center">
                  {getStrengthBadge(signal.strength)}
                </td>

                {/* Уверенность */}
                <td className="px-4 py-3 text-sm text-right">
                  <div className="flex items-center justify-end gap-2">
                    <span className={getStrengthColor(signal.strength)}>
                      {formatConfidence(signal.confidence)}
                    </span>
                    {/* Прогресс-бар уверенности */}
                    <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all ${
                          signal.confidence > 0.8 ? 'bg-yellow-500' :
                          signal.confidence > 0.6 ? 'bg-blue-500' :
                          'bg-gray-500'
                        }`}
                        style={{ width: `${signal.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </td>

                {/* Причина */}
                <td className="px-4 py-3 text-xs text-gray-400 max-w-xs truncate">
                  {signal.reason}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Футер */}
      {signals.length > maxSignals && (
        <div className="p-3 bg-gray-800/30 text-center text-xs text-gray-500">
          Показано {maxSignals} из {signals.length} сигналов
        </div>
      )}
    </Card>
  );
}