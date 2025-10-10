// frontend/src/components/account/BalanceCard.tsx

import { Card } from '../ui/Card';
import type { AccountBalance, BalanceStats } from '../../types/account.types';
import { TrendingUp, TrendingDown, Wallet, DollarSign } from 'lucide-react';

interface BalanceCardProps {
  balance: AccountBalance | null;
  stats: BalanceStats | null;
  loading?: boolean;
}

/**
 * Компонент для отображения баланса аккаунта.
 * Показывает общий баланс, PnL и статистику.
 */
export function BalanceCard({ balance, stats, loading = false }: BalanceCardProps) {
  /**
   * Форматирование числа с разделителями.
   * Добавлена проверка на null/undefined.
   */
  const formatNumber = (num: number | null | undefined, decimals = 2): string => {
    if (num === null || num === undefined || isNaN(num)) {
      return '0.00';
    }
    return num.toLocaleString('ru-RU', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  };

  /**
   * Форматирование процента.
   * Добавлена проверка на null/undefined.
   */
  const formatPercentage = (num: number | null | undefined): string => {
    if (num === null || num === undefined || isNaN(num)) {
      return '+0.00%';
    }
    const sign = num >= 0 ? '+' : '';
    return `${sign}${num.toFixed(2)}%`;
  };

  if (loading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-gray-700 rounded w-1/3"></div>
          <div className="h-12 bg-gray-700 rounded w-1/2"></div>
          <div className="grid grid-cols-2 gap-4">
            <div className="h-20 bg-gray-700 rounded"></div>
            <div className="h-20 bg-gray-700 rounded"></div>
          </div>
        </div>
      </Card>
    );
  }

  if (!balance) {
    return (
      <Card className="p-6">
        <p className="text-gray-400 text-center">Нет данных о балансе</p>
      </Card>
    );
  }

  const totalBalance = balance.total_usdt || 0;
  const dailyPnl = stats?.daily_pnl || 0;
  const dailyPnlPercentage = stats?.daily_pnl_percentage || 0;
  const totalPnl = stats?.total_pnl || 0;
  const totalPnlPercentage = stats?.total_pnl_percentage || 0;

  const isDailyPositive = dailyPnl >= 0;
  const isTotalPositive = totalPnl >= 0;

  return (
    <Card className="p-6">
      {/* Заголовок */}
      <div className="flex items-center gap-2 mb-4">
        <Wallet className="h-5 w-5 text-primary" />
        <h2 className="text-xl font-semibold">Баланс Аккаунта</h2>
      </div>

      {/* Общий баланс */}
      <div className="mb-6">
        <p className="text-gray-400 text-sm mb-1">Общий баланс</p>
        <div className="flex items-baseline gap-2">
          <DollarSign className="h-6 w-6 text-gray-400" />
          <span className="text-4xl font-bold">
            {formatNumber(totalBalance, 2)}
          </span>
          <span className="text-gray-400 text-xl">USDT</span>
        </div>
      </div>

      {/* Статистика */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Дневной PnL */}
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              {isDailyPositive ? (
                <TrendingUp className="h-4 w-4 text-success" />
              ) : (
                <TrendingDown className="h-4 w-4 text-destructive" />
              )}
              <p className="text-sm text-gray-400">За сегодня</p>
            </div>
            <div className="space-y-1">
              <p className={`text-2xl font-bold ${
                isDailyPositive ? 'text-success' : 'text-destructive'
              }`}>
                {formatNumber(dailyPnl, 2)} USDT
              </p>
              <p className={`text-sm ${
                isDailyPositive ? 'text-success' : 'text-destructive'
              }`}>
                {formatPercentage(dailyPnlPercentage)}
              </p>
            </div>
          </div>

          {/* Общий PnL */}
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              {isTotalPositive ? (
                <TrendingUp className="h-4 w-4 text-success" />
              ) : (
                <TrendingDown className="h-4 w-4 text-destructive" />
              )}
              <p className="text-sm text-gray-400">Всего</p>
            </div>
            <div className="space-y-1">
              <p className={`text-2xl font-bold ${
                isTotalPositive ? 'text-success' : 'text-destructive'
              }`}>
                {formatNumber(totalPnl, 2)} USDT
              </p>
              <p className={`text-sm ${
                isTotalPositive ? 'text-success' : 'text-destructive'
              }`}>
                {formatPercentage(totalPnlPercentage)}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Активы */}
      {balance.balances && Object.keys(balance.balances).length > 0 && (
        <div className="mt-6">
          <p className="text-sm text-gray-400 mb-3">Активы</p>
          <div className="space-y-2">
            {Object.entries(balance.balances).map(([asset, data]) => (
              <div
                key={asset}
                className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                    <span className="text-sm font-bold text-primary">
                      {asset.substring(0, 2)}
                    </span>
                  </div>
                  <div>
                    <p className="font-medium">{asset}</p>
                    <p className="text-xs text-gray-500">
                      Доступно: {formatNumber(data.free, 4)}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-semibold">{formatNumber(data.total, 4)}</p>
                  {data.locked > 0 && (
                    <p className="text-xs text-warning">
                      Заблокировано: {formatNumber(data.locked, 4)}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Сообщение если нет активов */}
      {balance.balances && Object.keys(balance.balances).length === 0 && (
        <div className="mt-6 text-center p-4 bg-gray-800/30 rounded-lg">
          <p className="text-gray-400 text-sm">Нет активов для отображения</p>
        </div>
      )}
    </Card>
  );
}