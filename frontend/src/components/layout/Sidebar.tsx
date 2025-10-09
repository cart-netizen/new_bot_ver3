import { Link, useLocation } from 'react-router-dom';
import { Home, BarChart3, TrendingUp } from 'lucide-react';
import { cn } from '../../utils/helpers';

const NAV_ITEMS = [
  { path: '/dashboard', label: 'Dashboard', icon: Home },
  { path: '/market', label: 'Рынок', icon: BarChart3 },
  { path: '/trading', label: 'Торговля', icon: TrendingUp },
];

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
                'flex items-center gap-3 px-4 py-3 rounded-lg transition-colors',
                isActive
                  ? 'bg-primary text-white'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-white'
              )}
            >
              <Icon className="h-5 w-5" />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}