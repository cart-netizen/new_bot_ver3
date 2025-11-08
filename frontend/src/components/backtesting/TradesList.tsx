// frontend/src/components/backtesting/TradesList.tsx (ОБНОВЛЕНО - Фаза 3)

import { useState } from 'react';
import {
  ArrowUpRight,
  ArrowDownRight,
  Clock,
  DollarSign,
  TrendingUp,
  TrendingDown,
  ChevronDown,
  ChevronRight,
  Users,
  // Target,
  // Brain
} from 'lucide-react';
import { Card } from '../ui/Card';
import { cn } from '../../utils/helpers';
import type { Trade, ConsensusInfo } from '../../api/backtesting.api';

interface TradesListProps {
  trades: Trade[];
}

function ConsensusDetails({ consensus }: { consensus: ConsensusInfo }) {
  return (
    <div className="p-4 bg-gray-800/30 border-t border-gray-800 space-y-3">
      <div className="flex items-center gap-2 text-sm font-medium text-gray-300">
        <Users className="h-4 w-4" />
        <span>Детали консенсуса</span>
      </div>

      {/* Режим консенсуса */}
      <div className="flex items-center gap-2 text-sm">
        <span className="text-gray-400">Режим:</span>
        <span className="px-2 py-0.5 text-xs font-medium rounded bg-blue-500/20 text-blue-300">
          {consensus.mode}
        </span>
      </div>

      {/* Стратегии - BUY */}
      {consensus.strategies_buy && consensus.strategies_buy.length > 0 && (
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <TrendingUp className="h-3 w-3 text-green-400" />
            <span>Стратегии за BUY ({consensus.strategies_buy.length})</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {consensus.strategies_buy.map((strategy, idx) => (
              <span
                key={idx}
                className="px-2 py-0.5 text-xs font-medium rounded bg-green-500/20 text-green-300"
              >
                {strategy}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Стратегии - SELL */}
      {consensus.strategies_sell && consensus.strategies_sell.length > 0 && (
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <TrendingDown className="h-3 w-3 text-red-400" />
            <span>Стратегии за SELL ({consensus.strategies_sell.length})</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {consensus.strategies_sell.map((strategy, idx) => (
              <span
                key={idx}
                className="px-2 py-0.5 text-xs font-medium rounded bg-red-500/20 text-red-300"
              >
                {strategy}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Статистика */}
      <div className="grid grid-cols-3 gap-3 pt-2 border-t border-gray-700/50">
        <div className="space-y-1">
          <p className="text-xs text-gray-400">Уверенность</p>
          <p className="text-sm font-medium text-white">
            {(consensus.final_confidence * 100).toFixed(0)}%
          </p>
        </div>
        <div className="space-y-1">
          <p className="text-xs text-gray-400">Согласных</p>
          <p className="text-sm font-medium text-green-400">
            {consensus.agreement_count}
          </p>
        </div>
        <div className="space-y-1">
          <p className="text-xs text-gray-400">Несогласных</p>
          <p className="text-sm font-medium text-red-400">
            {consensus.disagreement_count}
          </p>
        </div>
      </div>

      {/* Разбивка по типам стратегий */}
      {(consensus.candle_strategies > 0 || consensus.orderbook_strategies > 0 || consensus.hybrid_strategies > 0) && (
        <div className="flex gap-3 pt-2 border-t border-gray-700/50 text-xs">
          {consensus.candle_strategies > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="text-gray-400">Candle:</span>
              <span className="text-white font-medium">{consensus.candle_strategies}</span>
            </div>
          )}
          {consensus.orderbook_strategies > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="text-gray-400">OrderBook:</span>
              <span className="text-white font-medium">{consensus.orderbook_strategies}</span>
            </div>
          )}
          {consensus.hybrid_strategies > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="text-gray-400">Hybrid:</span>
              <span className="text-white font-medium">{consensus.hybrid_strategies}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function TradesList({ trades }: TradesListProps) {
  const [sortBy, setSortBy] = useState<'time' | 'pnl' | 'duration'>('time');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [expandedTrades, setExpandedTrades] = useState<Set<number>>(new Set());

  const formatDateTime = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString('ru-RU', {
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

    if (days > 0) return `${days}д ${hours % 24}ч`;
    if (hours > 0) return `${hours}ч ${minutes % 60}м`;
    return `${minutes}м`;
  };

  const toggleExpand = (index: number) => {
    const newExpanded = new Set(expandedTrades);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedTrades(newExpanded);
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
            <span>Всего сделок</span>
          </div>
          <p className="text-2xl font-bold text-white">{trades.length}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 text-sm text-gray-400 mb-1">
            <TrendingUp className="h-4 w-4 text-green-400" />
            <span>Прибыльных</span>
          </div>
          <p className="text-2xl font-bold text-green-400">{winningTrades.length}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 text-sm text-gray-400 mb-1">
            <TrendingDown className="h-4 w-4 text-red-400" />
            <span>Убыточных</span>
          </div>
          <p className="text-2xl font-bold text-red-400">{losingTrades.length}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 text-sm text-gray-400 mb-1">
            <Clock className="h-4 w-4" />
            <span>Ср. длительность</span>
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
                <th className="w-10 px-4 py-3"></th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Направление
                </th>
                <th
                  className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:text-white"
                  onClick={() => handleSort('time')}
                >
                  Вход / Выход {sortBy === 'time' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Цена
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Объем
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
                  Длительность {sortBy === 'duration' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Выход
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {sortedTrades.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-4 py-8 text-center text-gray-500">
                    Нет сделок для отображения
                  </td>
                </tr>
              ) : (
                sortedTrades.map((trade, index) => {
                  const isProfit = trade.pnl > 0;
                  const isBuy = trade.side.toLowerCase() === 'buy';
                  const isExpanded = expandedTrades.has(index);
                  const hasConsensus = !!trade.consensus_info;

                  return (
                    <>
                      <tr
                        key={index}
                        className={cn(
                          "transition-colors",
                          hasConsensus ? "hover:bg-gray-800/50 cursor-pointer" : "hover:bg-gray-800/30"
                        )}
                        onClick={() => hasConsensus && toggleExpand(index)}
                      >
                        <td className="px-4 py-3">
                          {hasConsensus && (
                            <button className="text-gray-400 hover:text-white">
                              {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                            </button>
                          )}
                        </td>
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
                            {hasConsensus && (
                              <span title="Консенсус доступен">
                                <Users className="h-3 w-3 text-blue-400" />
                              </span>
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
                      {/* Expanded Row - Consensus Details */}
                      {isExpanded && hasConsensus && (
                        <tr key={`${index}-details`}>
                          <td colSpan={8} className="p-0">
                            <ConsensusDetails consensus={trade.consensus_info!} />
                          </td>
                        </tr>
                      )}
                    </>
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
