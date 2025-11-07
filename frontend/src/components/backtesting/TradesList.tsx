// frontend/src/components/backtesting/TradesList.tsx

import { useState } from 'react';
import { ArrowUpRight, ArrowDownRight, Clock, DollarSign, TrendingUp, TrendingDown } from 'lucide-react';
import { Card } from '../ui/Card';
import { cn } from '../../utils/helpers';
import type { Trade } from '../../api/backtesting.api';

interface TradesListProps {
  trades: Trade[];
}

export function TradesList({ trades }: TradesListProps) {
  const [sortBy, setSortBy] = useState<'time' | 'pnl' | 'duration'>('time');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  const formatDateTime = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatMoney = (value: number) => `$${Math.abs(value).toFixed(2)}`;
  const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ${hours % 24}h`;
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    return `${minutes}m`;
  };

  const sortedTrades = [...trades].sort((a, b) => {
    let comparison = 0;

    switch (sortBy) {
      case 'time':
        comparison = new Date(a.entry_time).getTime() - new Date(b.entry_time).getTime();
        break;
      case 'pnl':
        comparison = a.pnl - b.pnl;
        break;
      case 'duration':
        comparison = a.duration_seconds - b.duration_seconds;
        break;
    }

    return sortOrder === 'asc' ? comparison : -comparison;
  });

  const handleSort = (field: typeof sortBy) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  const winningTrades = trades.filter(t => t.pnl > 0);
  const losingTrades = trades.filter(t => t.pnl < 0);

  return (
    <div className="space-y-4">
      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center gap-2 text-sm text-gray-400 mb-1">
            <DollarSign className="h-4 w-4" />
            <span>Total Trades</span>
          </div>
          <p className="text-2xl font-bold text-white">{trades.length}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 text-sm text-gray-400 mb-1">
            <TrendingUp className="h-4 w-4 text-green-400" />
            <span>Winning</span>
          </div>
          <p className="text-2xl font-bold text-green-400">{winningTrades.length}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 text-sm text-gray-400 mb-1">
            <TrendingDown className="h-4 w-4 text-red-400" />
            <span>Losing</span>
          </div>
          <p className="text-2xl font-bold text-red-400">{losingTrades.length}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 text-sm text-gray-400 mb-1">
            <Clock className="h-4 w-4" />
            <span>Avg Duration</span>
          </div>
          <p className="text-2xl font-bold text-white">
            {formatDuration(trades.reduce((sum, t) => sum + t.duration_seconds, 0) / trades.length || 0)}
          </p>
        </Card>
      </div>

      {/* Trades Table */}
      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-800/50 border-b border-gray-800">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Side
                </th>
                <th
                  className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:text-white"
                  onClick={() => handleSort('time')}
                >
                  Entry / Exit {sortBy === 'time' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Price
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Quantity
                </th>
                <th
                  className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:text-white"
                  onClick={() => handleSort('pnl')}
                >
                  PnL {sortBy === 'pnl' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th
                  className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:text-white"
                  onClick={() => handleSort('duration')}
                >
                  Duration {sortBy === 'duration' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Exit Reason
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {sortedTrades.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                    No trades to display
                  </td>
                </tr>
              ) : (
                sortedTrades.map((trade, index) => {
                  const isProfit = trade.pnl > 0;
                  const isBuy = trade.side.toLowerCase() === 'buy';

                  return (
                    <tr key={index} className="hover:bg-gray-800/30 transition-colors">
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          {isBuy ? (
                            <>
                              <ArrowUpRight className="h-4 w-4 text-green-400" />
                              <span className="text-sm font-medium text-green-400">BUY</span>
                            </>
                          ) : (
                            <>
                              <ArrowDownRight className="h-4 w-4 text-red-400" />
                              <span className="text-sm font-medium text-red-400">SELL</span>
                            </>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="space-y-1">
                          <p className="text-sm text-white">{formatDateTime(trade.entry_time)}</p>
                          <p className="text-xs text-gray-500">{formatDateTime(trade.exit_time)}</p>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <div className="space-y-1">
                          <p className="text-sm text-white">${trade.entry_price.toFixed(2)}</p>
                          <p className="text-xs text-gray-500">${trade.exit_price.toFixed(2)}</p>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <p className="text-sm text-white">{trade.quantity.toFixed(4)}</p>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <div className="space-y-1">
                          <p className={cn(
                            "text-sm font-medium",
                            isProfit ? "text-green-400" : "text-red-400"
                          )}>
                            {isProfit ? '+' : ''}{formatMoney(trade.pnl)}
                          </p>
                          <p className={cn(
                            "text-xs",
                            isProfit ? "text-green-400/60" : "text-red-400/60"
                          )}>
                            {formatPercent(trade.pnl_pct)}
                          </p>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <p className="text-sm text-white">{formatDuration(trade.duration_seconds)}</p>
                      </td>
                      <td className="px-4 py-3">
                        <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded bg-gray-800 text-gray-300">
                          {trade.exit_reason}
                        </span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
