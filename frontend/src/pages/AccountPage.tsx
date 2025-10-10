// frontend/src/pages/AccountPage.tsx
// Обновленная страница с компактным отображением баланса

import { useEffect, useState } from 'react';
import { useAccountStore } from '../store/accountStore';
import { BalanceCard } from '../components/account/BalanceCard';
import { BalanceChart } from '../components/account/BalanceChart';
import { RefreshCw, TrendingUp, TrendingDown } from 'lucide-react';
import { toast } from 'sonner';

/**
 * Страница личного кабинета.
 * Отображает баланс аккаунта и график динамики баланса.
 */
export function AccountPage() {
  const {
    balance,
    balanceHistory,
    balanceStats,
    isLoadingBalance,
    isLoadingHistory,
    isLoadingStats,

    fetchBalance,
    fetchBalanceHistory,
    fetchBalanceStats,
  } = useAccountStore();

  const [selectedPeriod, setSelectedPeriod] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [isRefreshing, setIsRefreshing] = useState(false);

  /**
   * Загрузка всех данных при монтировании.
   */
  useEffect(() => {
    const loadData = async () => {
      console.log('[AccountPage] Loading data...');
      try {
        await Promise.all([
          fetchBalance(),
          fetchBalanceHistory(selectedPeriod),
          fetchBalanceStats(),
        ]);
        console.log('[AccountPage] Data loaded successfully');
      } catch (error) {
        console.error('[AccountPage] Failed to load data:', error);
      }
    };

    loadData();
  }, [fetchBalance, fetchBalanceHistory, fetchBalanceStats, selectedPeriod]);

  /**
   * Обработка изменения периода графика.
   */
  const handlePeriodChange = async (period: '1h' | '24h' | '7d' | '30d') => {
    setSelectedPeriod(period);
    try {
      await fetchBalanceHistory(period);
    } catch (error) {
      console.error('[AccountPage] Failed to load history:', error);
      toast.error('Не удалось загрузить историю');
    }
  };

  /**
   * Обновление всех данных.
   */
  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      await Promise.all([
        fetchBalance(),
        fetchBalanceHistory(selectedPeriod),
        fetchBalanceStats(),
      ]);
      toast.success('Данные обновлены');
    } catch (error) {
      console.error('[AccountPage] Failed to refresh:', error);
      toast.error('Не удалось обновить данные');
    } finally {
      setIsRefreshing(false);
    }
  };

  /**
   * Форматирование числа.
   */
  const formatNumber = (num: number, decimals = 2): string => {
    return num.toLocaleString('ru-RU', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  };

  return (
    <div className="space-y-6">
      {/* Заголовок с кнопкой обновления */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Личный Кабинет</h1>
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary/90 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg transition-colors"
        >
          <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          <span>Обновить</span>
        </button>
      </div>

      {/* Сетка с балансом и статистикой - одинаковая высота блоков */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Баланс Аккаунта - компактный блок */}
        <BalanceCard
          balance={balance}
          stats={balanceStats}
          loading={isLoadingBalance}
        />

        {/* Начальный баланс */}
        <div className="bg-surface p-4 rounded-lg border border-gray-800">
          <p className="text-sm text-gray-400 mb-2">Начальный баланс</p>
          {isLoadingStats ? (
            <div className="animate-pulse">
              <div className="h-8 bg-gray-700 rounded w-2/3"></div>
            </div>
          ) : (
            <p className="text-2xl font-bold text-white">
              ${balanceStats ? formatNumber(balanceStats.initial_balance) : '0.00'}
            </p>
          )}
        </div>

        {/* Общий PnL */}
        <div className="bg-surface p-4 rounded-lg border border-gray-800">
          <p className="text-sm text-gray-400 mb-2">Общий PnL</p>
          {isLoadingStats ? (
            <div className="animate-pulse">
              <div className="h-8 bg-gray-700 rounded w-2/3"></div>
            </div>
          ) : balanceStats ? (
            <div className="space-y-1">
              <p className={`text-2xl font-bold ${
                balanceStats.total_pnl >= 0 ? 'text-success' : 'text-destructive'
              }`}>
                {balanceStats.total_pnl >= 0 ? '+' : ''}{formatNumber(balanceStats.total_pnl)} USDT
              </p>
              <p className={`text-sm ${
                balanceStats.total_pnl >= 0 ? 'text-success' : 'text-destructive'
              }`}>
                {balanceStats.total_pnl >= 0 ? '+' : ''}{formatNumber(balanceStats.total_pnl_percentage, 2)}%
              </p>
            </div>
          ) : (
            <p className="text-2xl font-bold text-white">$0.00</p>
          )}
        </div>
      </div>

      {/* График */}
      <BalanceChart
        history={balanceHistory}
        loading={isLoadingHistory}
        onPeriodChange={handlePeriodChange}
      />

      {/* Дополнительная статистика */}
      {balanceStats && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Лучший день */}
          <div className="bg-surface p-4 rounded-lg border border-gray-800">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="h-5 w-5 text-success" />
              <p className="text-sm text-gray-400">Лучший день</p>
            </div>
            <p className="text-2xl font-bold text-success">
              +${formatNumber(balanceStats.best_day)}
            </p>
          </div>

          {/* Худший день */}
          <div className="bg-surface p-4 rounded-lg border border-gray-800">
            <div className="flex items-center gap-2 mb-2">
              <TrendingDown className="h-5 w-5 text-destructive" />
              <p className="text-sm text-gray-400">Худший день</p>
            </div>
            <p className="text-2xl font-bold text-destructive">
              ${formatNumber(balanceStats.worst_day)}
            </p>
          </div>
        </div>
      )}

      {/* Детализация по активам */}
      {balance && balance.balances && Object.keys(balance.balances).length > 0 && (
        <div className="bg-surface p-6 rounded-lg border border-gray-800">
          <h2 className="text-xl font-semibold mb-4">Активы</h2>
          <div className="space-y-3">
            {Object.entries(balance.balances).map(([asset, data]) => (
              <div
                key={asset}
                className="flex items-center justify-between p-4 bg-gray-800/30 rounded-lg hover:bg-gray-800/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                    <span className="text-sm font-bold text-primary">
                      {asset.substring(0, 2)}
                    </span>
                  </div>
                  <div>
                    <p className="font-semibold text-lg">{asset}</p>
                    <p className="text-sm text-gray-400">
                      Доступно: {formatNumber(data.free, 4)}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-lg font-semibold">{formatNumber(data.total, 4)}</p>
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
    </div>
  );
}