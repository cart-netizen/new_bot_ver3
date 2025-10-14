// frontend/src/components/layout/Sidebar.tsx
/**
 * Боковая панель навигации с расширенным функционалом.
 *
 * ИСПРАВЛЕНО: Добавлена правильная TypeScript типизация
 */

import { Link, useLocation } from 'react-router-dom';
import {
  User,
  Home,
  BarChart3,
  TrendingUp,
  LineChart,      // Графики
  FileText,       // Ордера
  Search,         // Скринер
  Settings        // Стратегии
} from 'lucide-react';
import { cn } from '../../utils/helpers';

/**
 * Тип элемента навигации.
 */
interface NavItem {
  path: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  new?: boolean; // Опциональное свойство для пометки новых пунктов
}

/**
 * Элементы навигации.
 */
const NAV_ITEMS: NavItem[] = [
  {
    path: '/account',
    label: 'Личный Кабинет',
    icon: User
  },
  {
    path: '/dashboard',
    label: 'Dashboard',
    icon: Home
  },
  {
    path: '/market',
    label: 'Рынок',
    icon: BarChart3
  },
  {
    path: '/trading',
    label: 'Торговля',
    icon: TrendingUp
  },
  // ==================== НОВЫЕ ПУНКТЫ ====================
  {
    path: '/charts',
    label: 'Графики',
    icon: LineChart,
    new: true  // Пометка для визуализации новых пунктов
  },
  {
    path: '/orders',
    label: 'Ордера',
    icon: FileText,
    new: true
  },
  {
    path: '/screener',
    label: 'Скринер',
    icon: Search,
    new: true
  },
  {
    path: '/strategies',
    label: 'Стратегии',
    icon: Settings,
    new: true
  },
];

/**
 * Компонент боковой панели навигации.
 */
export function Sidebar() {
  const location = useLocation();

  return (
    <aside className="w-64 border-r border-gray-800 bg-surface">
      <nav className="p-4 space-y-2">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;

          return (
            <Link
              key={item.path}
              to={item.path}
              className={cn(
                'flex items-center gap-3 px-4 py-3 rounded-lg transition-colors relative',
                isActive
                  ? 'bg-primary text-white'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-white'
              )}
            >
              <Icon className="h-5 w-5" />
              <span>{item.label}</span>

              {/* Индикатор "новый пункт" (опционально, можно убрать) */}
              {item.new && !isActive && (
                <span className="absolute right-3 top-1/2 -translate-y-1/2">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                  </span>
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Разделитель и информация */}
      <div className="px-4 py-6 border-t border-gray-800 mt-4">
        <div className="text-xs text-gray-500 space-y-1">
          <p className="font-semibold">Торговый Бот v1.0</p>
          <p>Real-time данные</p>
          <p className="text-gray-600">Bybit Integration</p>
        </div>
      </div>
    </aside>
  );
}