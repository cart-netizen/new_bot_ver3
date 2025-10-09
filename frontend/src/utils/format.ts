import { formatDistanceToNow } from 'date-fns';
import { ru } from 'date-fns/locale';

export function formatPrice(price: number | null, decimals: number = 2): string {
  if (price === null) return 'N/A';
  return price.toFixed(decimals);
}

export function formatVolume(volume: number, decimals: number = 2): string {
  if (volume >= 1_000_000) {
    return `${(volume / 1_000_000).toFixed(2)}M`;
  }
  if (volume >= 1_000) {
    return `${(volume / 1_000).toFixed(2)}K`;
  }
  return volume.toFixed(decimals);
}

export function formatPercent(value: number, decimals: number = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatRelativeTime(timestamp: number): string {
  return formatDistanceToNow(new Date(timestamp), {
    addSuffix: true,
    locale: ru,
  });
}