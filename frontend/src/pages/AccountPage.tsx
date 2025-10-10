// frontend/src/pages/AccountPage.tsx

import { useEffect, useState } from 'react';
import { useAccountStore } from '../store/accountStore';
import { BalanceCard } from '../components/account/BalanceCard';
import { BalanceChart } from '../components/account/BalanceChart';
import { Activity, RefreshCw } from 'lucide-react';
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
    error,
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
        // Не показываем toast здесь, ошибки уже обработаны в store
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

  const isLoading = isLoadingBalance || isLoadingHistory || isLoadingStats;

  // Показываем индикатор загрузки только при первом рендере
  if (isLoading && !balance && !balanceHistory && !balanceStats) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Activity className="h-12 w-12 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-gray-400">Загрузка данных аккаунта...</p>
        </div>
      </div>
    );
  }

  // Показываем сообщение об ошибке если не удалось загрузить данные
  if (!isLoading && !balance && error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center max-w-md">
          <div className="bg-destructive/20 text-destructive p-4 rounded-lg mb-4">
            <p className="font-semibold mb-2">Не удалось загрузить данные</p>
            <p className="text-sm">{error}</p>
          </div>
          <button
            onClick={handleRefresh}
            className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors"
          >
            Попробовать снова
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Заголовок */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Личный Кабинет</h1>

        {/* Кнопка обновления */}
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700
                     transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          <span>Обновить</span>
        </button>
      </div>

      {/* Баланс */}
      <BalanceCard
        balance={balance}
        stats={balanceStats}
        loading={isLoadingBalance}
      />

      {/* График */}
      <BalanceChart
        history={balanceHistory}
        loading={isLoadingHistory}
        onPeriodChange={handlePeriodChange}
      />

      {/* Дополнительная информация */}
      {balanceStats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Начальный баланс */}
          <div className="bg-surface p-4 rounded-lg border border-gray-800">
            <p className="text-sm text-gray-400 mb-1">Начальный баланс</p>
            <p className="text-2xl font-bold">
              ${balanceStats.initial_balance.toFixed(2)}
            </p>
          </div>

          {/* Лучший день */}
          <div className="bg-surface p-4 rounded-lg border border-gray-800">
            <p className="text-sm text-gray-400 mb-1">Лучший день</p>
            <p className="text-2xl font-bold text-success">
              +${balanceStats.best_day.toFixed(2)}
            </p>
          </div>

          {/* Худший день */}
          <div className="bg-surface p-4 rounded-lg border border-gray-800">
            <p className="text-sm text-gray-400 mb-1">Худший день</p>
            <p className="text-2xl font-bold text-destructive">
              ${balanceStats.worst_day.toFixed(2)}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}