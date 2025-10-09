import type { FormEvent } from 'react';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { useAuthStore } from '../../store/authStore';
import { toast } from 'sonner';

export function LoginForm() {
  const [password, setPassword] = useState('');
  const { login, isLoading } = useAuthStore();
  const navigate = useNavigate();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      await login({ password });
      toast.success('Вход выполнен');
      navigate('/dashboard');
    } catch  {
      toast.error('Неверный пароль');
    }
  };

  return (
    <Card className="w-full max-w-md p-8">
      <h1 className="text-2xl font-bold text-center mb-6">Trading Bot</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-2">Пароль</label>
          <Input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Введите пароль"
            required
          />
        </div>
        <Button type="submit" disabled={isLoading} className="w-full">
          {isLoading ? 'Вход...' : 'Войти'}
        </Button>
      </form>
    </Card>
  );
}