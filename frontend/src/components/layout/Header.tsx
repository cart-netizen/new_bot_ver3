import { LogOut } from 'lucide-react';
import { Button } from '../ui/Button';
import { useAuthStore } from '@/store/authStore';
import { useNavigate } from 'react-router-dom';
import { useWebSocket } from '@/hooks/useWebSocket';

export function Header() {
  const logout = useAuthStore((state) => state.logout);
  const navigate = useNavigate();
  const { isConnected } = useWebSocket();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <header className="h-16 border-b border-gray-800 bg-surface px-6 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <h1 className="text-xl font-bold">Trading Bot</h1>
        <div className="flex items-center gap-2 text-sm">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-success' : 'bg-danger'}`} />
          <span className="text-gray-400">{isConnected ? 'Подключено' : 'Отключено'}</span>
        </div>
      </div>
      <Button variant="outline" onClick={handleLogout}>
        <LogOut className="h-4 w-4" />
      </Button>
    </header>
  );
}