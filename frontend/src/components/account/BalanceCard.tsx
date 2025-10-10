// frontend/src/components/account/BalanceCard.tsx
// Исправленная версия с компактным основным блоком и всеми оригинальными элементами

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
 * Основной блок сделан компактным для единообразия с другими блоками.
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

  /**
   * Загрузочный плейсхолдер
   */
  if (loading) {
    return (
      <Card className="p-4">
        <div className="animate-pulse space-y-3">
          <div className="h-5 bg-gray-700 rounded w-1/3"></div>
          <div className="h-8 bg-gray-700 rounded w-2/3"></div>
          <div className="h-4 bg-gray-700 rounded w-1/2"></div>
        </div>
      </Card>
    );
  }

  /**
   * Состояние когда нет данных
   */
  if (!balance) {
    return (
      <Card className="p-4">
        <div className="text-center text-gray-400">
          <Wallet className="h-8 w-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">Нет данных о балансе</p>
        </div>
      </Card>
    );
  }

  // Получаем данные с безопасной обработкой
  const totalUsdt = balance?.total_usdt ?? 0;
  const usdtBalance = balance?.balances?.['USDT'];
  const freeUsdt = usdtBalance?.free ?? 0;
  // const lockedUsdt = usdtBalance?.locked ?? 0;

  // PnL данные из stats
  const dailyPnl = stats?.daily_pnl ?? 0;
  const dailyPnlPercentage = stats?.daily_pnl_percentage ?? 0;
  const totalPnl = stats?.total_pnl ?? 0;
  const totalPnlPercentage = stats?.total_pnl_percentage ?? 0;

  const isDailyPositive = dailyPnl >= 0;
  const isTotalPositive = totalPnl >= 0;

  return (
    <div className="space-y-4">
      {/* ОСНОВНОЙ КОМПАКТНЫЙ БЛОК - Баланс Аккаунта */}
      <Card className="p-4">
        <div className="flex items-center gap-2 mb-2">
          <Wallet className="h-5 w-5 text-primary" />
          <h2 className="text-lg font-semibold">Баланс Аккаунта</h2>
        </div>

        <div className="space-y-1">
          <p className="text-2xl font-bold text-white">
            {formatNumber(totalUsdt, 2)} USDT
          </p>
          {usdtBalance && (
            <p className="text-sm text-gray-400">
              Доступно: {formatNumber(freeUsdt, 4)}
            </p>
          )}
        </div>
      </Card>

      {/* PnL ИНФОРМАЦИЯ */}
      {stats && (
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <DollarSign className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-semibold">Прибыль/Убыток</h3>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* Дневной PnL */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                {isDailyPositive ? (
                  <TrendingUp className="h-4 w-4 text-success" />
                ) : (
                  <TrendingDown className="h-4 w-4 text-destructive" />
                )}
                <p className="text-sm text-gray-400">За день</p>
              </div>
              <div className="space-y-1">
                <p className={`text-xl font-bold ${
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
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                {isTotalPositive ? (
                  <TrendingUp className="h-4 w-4 text-success" />
                ) : (
                  <TrendingDown className="h-4 w-4 text-destructive" />
                )}
                <p className="text-sm text-gray-400">Всего</p>
              </div>
              <div className="space-y-1">
                <p className={`text-xl font-bold ${
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
        </Card>
      )}

      {/* АКТИВЫ - Детализация по валютам */}
      {/*{balance.balances && Object.keys(balance.balances).length > 0 && (*/}
      {/*  <Card className="p-6">*/}
      {/*    <h3 className="text-lg font-semibold mb-4">Активы</h3>*/}
      {/*    <div className="space-y-3">*/}
      {/*      {Object.entries(balance.balances).map(([asset, data]) => (*/}
      {/*        <div*/}
      {/*          key={asset}*/}
      {/*          className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg hover:bg-gray-800/50 transition-colors"*/}
      {/*        >*/}
      {/*          <div className="flex items-center gap-3">*/}
      {/*            <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">*/}
      {/*              <span className="text-sm font-bold text-primary">*/}
      {/*                {asset.substring(0, 2)}*/}
      {/*              </span>*/}
      {/*            </div>*/}
      {/*            <div>*/}
      {/*              <p className="font-medium text-base">{asset}</p>*/}
      {/*              <p className="text-xs text-gray-400">*/}
      {/*                Доступно: {formatNumber(data.free, 4)}*/}
      {/*              </p>*/}
      {/*            </div>*/}
      {/*          </div>*/}
      {/*          <div className="text-right">*/}
      {/*            <p className="font-semibold text-base">*/}
      {/*              {formatNumber(data.total, 4)}*/}
      {/*            </p>*/}
      {/*            {data.locked > 0 && (*/}
      {/*              <p className="text-xs text-warning">*/}
      {/*                Заблокировано: {formatNumber(data.locked, 4)}*/}
      {/*              </p>*/}
      {/*            )}*/}
      {/*          </div>*/}
      {/*        </div>*/}
      {/*      ))}*/}
      {/*    </div>*/}
      {/*  </Card>*/}
      {/*)}*/}

      {/* СООБЩЕНИЕ если нет активов */}
      {balance.balances && Object.keys(balance.balances).length === 0 && (
        <Card className="p-6">
          <div className="text-center text-gray-400">
            <Wallet className="h-12 w-12 mx-auto mb-3 opacity-30" />
            <p className="text-sm">Нет активов для отображения</p>
            <p className="text-xs text-gray-500 mt-1">
              Пополните баланс для начала торговли
            </p>
          </div>
        </Card>
      )}
    </div>
  );
}