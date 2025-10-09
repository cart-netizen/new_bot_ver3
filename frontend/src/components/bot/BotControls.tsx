import { Play, Square } from 'lucide-react';
import { Button } from '../ui/Button.tsx';
import { Card } from '../ui/Card.tsx';
import { useBotStore } from '../../store/botStore.ts';
import { BotStatus } from '../../types/common.types.ts';
import { toast } from 'sonner';

export function BotControls() {
  const { status, startBot, stopBot, isLoading } = useBotStore();

  const handleStart = async () => {
    try {
      await startBot();
      toast.success('Бот запускается...');
    } catch {
      toast.error('Ошибка запуска');
    }
  };

  const handleStop = async () => {
    try {
      await stopBot();
      toast.success('Бот останавливается...');
    } catch {
      toast.error('Ошибка остановки');
    }
  };

  const isRunning = status === BotStatus.RUNNING;

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold mb-2">Управление Ботом</h2>
          <div className="flex items-center gap-2">
            <span className="text-gray-400">Статус:</span>
            <span className={`font-semibold ${
              isRunning ? 'text-success' : 'text-gray-400'
            }`}>
              {status}
            </span>
          </div>
        </div>

        <div>
          {!isRunning ? (
            <Button onClick={handleStart} disabled={isLoading}>
              <Play className="h-4 w-4 mr-2" />
              Запустить
            </Button>
          ) : (
            <Button variant="destructive" onClick={handleStop} disabled={isLoading}>
              <Square className="h-4 w-4 mr-2" />
              Остановить
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}