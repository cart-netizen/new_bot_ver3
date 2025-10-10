// frontend/src/components/layout/Header.tsx

import { LogOut } from 'lucide-react';
import { Button } from '../ui/Button';
import { useAuthStore } from '../../store/authStore';
import { useMarketStore } from '../../store/marketStore';
import { useNavigate } from 'react-router-dom';

/**
 * Header компонент приложения.
 * Показывает статус WebSocket соединения и кнопку выхода.
 */
export function Header() {
  const logout = useAuthStore((state) => state.logout);
  const navigate = useNavigate();

  // Получаем статус соединения из store (обновляется из Layout)
  const isConnected = useMarketStore((state) => state.isConnected);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <header className="h-16 border-b border-gray-800 bg-surface px-6 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <h1 className="text-xl font-bold">Trading Bot</h1>

        {/* Индикатор WebSocket соединения */}
        <div className="flex items-center gap-2 text-sm">
          <div
            className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-success' : 'bg-danger'
            }`}
          />
          <span className="text-gray-400">
            {isConnected ? 'Подключено' : 'Отключено'}
          </span>
        </div>
      </div>

      <Button variant="outline" onClick={handleLogout}>
        <LogOut className="h-4 w-4" />
      </Button>
    </header>
  );
}