// frontend/src/components/bot/BotControls.tsx
import { Play, Square, Loader2 } from 'lucide-react';
import { Button } from '../ui/Button';
import { Card } from '../ui/Card';
import { useBotStore } from '@/store/botStore';
import { BotStatus } from '@/types/common.types';
import { toast } from 'sonner';

/**
 * Компонент управления ботом.
 * Отображает текущий статус бота и кнопки запуска/остановки.
 */
export function BotControls() {
  const { status, startBot, stopBot, isLoading } = useBotStore();

  /**
   * Обработчик запуска бота.
   */
  const handleStart = async () => {
    try {
      await startBot();
      toast.success('Бот запускается...');
    } catch {
      toast.error('Ошибка запуска');
    }
  };

  /**
   * Обработчик остановки бота.
   */
  const handleStop = async () => {
    try {
      await stopBot();
      toast.success('Бот останавливается...');
    } catch {
      toast.error('Ошибка остановки');
    }
  };

  // Определение состояний для визуальной обратной связи
  const isRunning = status === BotStatus.RUNNING;
  const isStopped = status === BotStatus.STOPPED;
  const isStarting = status === BotStatus.STARTING;
  const isStopping = status === BotStatus.STOPPING;

  /**
   * Получение цвета статуса для визуализации.
   */
  const getStatusColor = () => {
    switch (status) {
      case BotStatus.RUNNING:
        return 'text-success';
      case BotStatus.STARTING:
        return 'text-yellow-500';
      case BotStatus.STOPPING:
        return 'text-orange-500';
      case BotStatus.ERROR:
        return 'text-destructive';
      default:
        return 'text-gray-400';
    }
  };

  /**
   * Получение текста статуса на русском.
   */
  const getStatusText = () => {
    switch (status) {
      case BotStatus.RUNNING:
        return 'Работает';
      case BotStatus.STOPPED:
        return 'Остановлен';
      case BotStatus.STARTING:
        return 'Запускается...';
      case BotStatus.STOPPING:
        return 'Останавливается...';
      case BotStatus.ERROR:
        return 'Ошибка';
      default:
        return status;
    }
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between">
        {/* Информация о статусе */}
        <div>
          <h2 className="text-xl font-semibold mb-2">Управление Ботом</h2>
          <div className="flex items-center gap-2">
            <span className="text-gray-400">Статус:</span>
            <div className="flex items-center gap-2">
              {/* Индикатор загрузки для переходных состояний */}
              {(isStarting || isStopping) && (
                <Loader2 className="h-4 w-4 animate-spin text-yellow-500" />
              )}
              <span className={`font-semibold ${getStatusColor()}`}>
                {getStatusText()}
              </span>
            </div>
          </div>
        </div>

        {/* Кнопки управления */}
        <div className="flex items-center gap-3">
          {/* Кнопка запуска */}
          <Button
            onClick={handleStart}
            disabled={isLoading || isRunning || isStarting}
            className="min-w-[120px]"
          >
            <Play className="h-4 w-4 mr-2" />
            Запустить
          </Button>

          {/* Кнопка остановки */}
          <Button
            variant="destructive"
            onClick={handleStop}
            disabled={isLoading || isStopped || isStopping}
            className="min-w-[120px]"
          >
            <Square className="h-4 w-4 mr-2" />
            Остановить
          </Button>
        </div>
      </div>
    </Card>
  );
}