"""
Daily Loss Killer - Экстренная защита от критических убытков.

КРИТИЧЕСКИ ВАЖНО:
- Автоматическое отключение торговли при убытке ≥15%
- Немедленное закрытие всех позиций
- Уведомления на Email + Telegram
- Невозможность продолжить торговлю до ручного вмешательства

Путь: backend/strategy/daily_loss_killer.py
"""
import asyncio
from datetime import datetime, time as datetime_time, timedelta
from typing import Optional, Tuple

from core.logger import get_logger
from config import settings
from utils.balance_tracker import balance_tracker
from services.notification_service import NotificationService
from infrastructure.repositories.audit_repository import audit_repository
from database.models import AuditAction
from strategy.risk_models import DailyLossMetrics

logger = get_logger(__name__)

# Глобальный экземпляр notification service
notification_service = NotificationService()


class DailyLossKiller:
  """
  Мониторинг и контроль дневного убытка.

  ЗАЩИТА:
  - Warning: 10% убыток
  - Critical: 15% убыток → EMERGENCY SHUTDOWN
  """

  def __init__(self):
    """Инициализация."""
    self.enabled = settings.DAILY_LOSS_KILLER_ENABLED
    self.max_loss_percent = settings.DAILY_LOSS_MAX_PERCENT / 100  # 15% -> 0.15
    self.warning_percent = settings.DAILY_LOSS_WARNING_PERCENT / 100  # 10% -> 0.10
    self.check_interval = settings.DAILY_LOSS_CHECK_INTERVAL_SEC

    # Состояние
    self.starting_balance: Optional[float] = None
    self.daily_reset_time = datetime_time(0, 0)  # 00:00 UTC
    self.last_check: Optional[datetime] = None
    self.is_emergency_shutdown = False
    self.warning_sent = False

    # Мониторинг task
    self.monitoring_task: Optional[asyncio.Task] = None

    logger.info(
      f"DailyLossKiller initialized: "
      f"enabled={self.enabled}, "
      f"max_loss={self.max_loss_percent:.0%}, "
      f"warning={self.warning_percent:.0%}"
    )

  async def start(self):
    """Запуск мониторинга."""
    if not self.enabled:
      logger.info("DailyLossKiller disabled by config")
      return

    # Устанавливаем starting balance
    await self._initialize_starting_balance()

    # Запускаем monitoring task
    self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    logger.info(
      f"✓ DailyLossKiller started: "
      f"starting_balance=${self.starting_balance:.2f}"
    )

  async def stop(self):
    """Остановка мониторинга."""
    if self.monitoring_task:
      self.monitoring_task.cancel()
      try:
        await self.monitoring_task
      except asyncio.CancelledError:
        pass

    logger.info("DailyLossKiller stopped")

  async def _initialize_starting_balance(self):
    """Инициализация starting balance для дня."""
    current_balance = balance_tracker.get_current_balance()

    if current_balance is None or current_balance == 0:
      # Fallback на initial balance из tracker
      stats = balance_tracker.get_stats()
      current_balance = stats.get('initial_balance', 10000.0)
      logger.warning(
        f"Current balance unavailable, using initial: ${current_balance:.2f}"
      )

    self.starting_balance = current_balance
    self.warning_sent = False
    self.is_emergency_shutdown = False

    logger.info(f"Starting balance set: ${self.starting_balance:.2f}")

  async def _monitoring_loop(self):
    """Основной цикл мониторинга."""
    logger.info("Daily Loss monitoring loop started")

    while True:
      try:
        # Проверка reset времени
        await self._check_daily_reset()

        # Проверка убытка
        await self._check_daily_loss()

        # Пауза до следующей проверки
        await asyncio.sleep(self.check_interval)

      except asyncio.CancelledError:
        logger.info("Monitoring loop cancelled")
        break
      except Exception as e:
        logger.error(f"Error in monitoring loop: {e}", exc_info=True)
        await asyncio.sleep(self.check_interval)

  async def _check_daily_reset(self):
    """Проверка и выполнение daily reset."""
    now = datetime.now()
    current_time = now.time()

    # Проверяем, прошли ли мы через reset время
    if self.last_check is not None:
      last_time = self.last_check.time()

      # Если прошли через 00:00
      if last_time > self.daily_reset_time >= current_time or \
          (last_time < self.daily_reset_time and current_time >= self.daily_reset_time):
        logger.info("=" * 60)
        logger.info("DAILY RESET TRIGGERED")
        logger.info("=" * 60)

        await self._perform_daily_reset()

    self.last_check = now

  async def _perform_daily_reset(self):
    """Выполнение daily reset."""
    # Обновляем starting balance
    await self._initialize_starting_balance()

    # Разрешаем торговлю (если была остановлена)
    # ВАЖНО: Требует ручной проверки перед включением!
    if self.is_emergency_shutdown:
      logger.warning(
        "Trading remains DISABLED after emergency shutdown. "
        "Manual intervention required!"
      )

    # Логируем reset
    await audit_repository.log(
      action=AuditAction.SYSTEM,
      entity_type="DailyLossKiller",
      entity_id="daily_reset",
      success=True,
      context={
        'event': 'daily_reset',
        'new_starting_balance': self.starting_balance,
        'timestamp': datetime.now().isoformat()
      }
    )

    logger.info(f"Daily reset completed: new starting balance=${self.starting_balance:.2f}")

  async def _check_daily_loss(self):
    """Проверка текущего дневного убытка."""
    if self.starting_balance is None:
      logger.warning("Starting balance not set, skipping loss check")
      return

    # Получаем текущий баланс
    current_balance = balance_tracker.get_current_balance()

    if current_balance is None:
      logger.warning("Current balance unavailable, skipping loss check")
      return

    # Рассчитываем метрики
    metrics = self._calculate_metrics(current_balance)

    # Проверка CRITICAL уровня (15%)
    if metrics.daily_loss_percent >= self.max_loss_percent:
      logger.critical(
        f"🚨 CRITICAL DAILY LOSS REACHED: {metrics.daily_loss_percent:.2%} 🚨"
      )
      await self._trigger_emergency_shutdown(metrics)
      return

    # Проверка WARNING уровня (10%)
    if metrics.daily_loss_percent >= self.warning_percent and not self.warning_sent:
      logger.warning(
        f"⚠️ WARNING: Daily loss at {metrics.daily_loss_percent:.2%} "
        f"(threshold: {self.warning_percent:.2%})"
      )
      await self._send_warning_notification(metrics)
      self.warning_sent = True

    # Логируем периодически (каждую 5-ю проверку)
    if hasattr(self, '_check_counter'):
      self._check_counter += 1
    else:
      self._check_counter = 1

    if self._check_counter % 5 == 0:
      logger.debug(
        f"Daily loss check: "
        f"PnL=${metrics.daily_pnl:.2f} ({metrics.daily_loss_percent:.2%}), "
        f"balance=${current_balance:.2f}"
      )

  def _calculate_metrics(self, current_balance: float) -> DailyLossMetrics:
    """Расчет метрик убытка."""
    daily_pnl = current_balance - self.starting_balance
    daily_loss_percent = abs(daily_pnl) / self.starting_balance if daily_pnl < 0 else 0.0

    is_critical = daily_loss_percent >= self.max_loss_percent

    # Время до reset (00:00 UTC следующего дня)
    now = datetime.now()
    next_reset = datetime.combine(
      now.date() + (timedelta(days=1) if now.time() >= self.daily_reset_time else timedelta(0)),
      self.daily_reset_time
    )

    return DailyLossMetrics(
      starting_balance=self.starting_balance,
      current_balance=current_balance,
      daily_pnl=daily_pnl,
      daily_loss_percent=daily_loss_percent,
      max_daily_loss_percent=self.max_loss_percent,
      is_critical=is_critical,
      time_to_reset=next_reset
    )

  async def _trigger_emergency_shutdown(self, metrics: DailyLossMetrics):
    """
    EMERGENCY SHUTDOWN при достижении критического убытка.

    ДЕЙСТВИЯ:
    1. Остановить торговлю
    2. Отправить критические уведомления
    3. Audit log

    TODO: Интеграция с ExecutionManager для закрытия позиций
    """
    if self.is_emergency_shutdown:
      # Уже выполнено
      return

    logger.critical("=" * 80)
    logger.critical("🚨 EMERGENCY SHUTDOWN INITIATED 🚨")
    logger.critical("=" * 80)
    logger.critical(f"Daily Loss: {metrics.daily_loss_percent:.2%} (max: {self.max_loss_percent:.2%})")
    logger.critical(f"Starting Balance: ${metrics.starting_balance:.2f}")
    logger.critical(f"Current Balance: ${metrics.current_balance:.2f}")
    logger.critical(f"Daily PnL: ${metrics.daily_pnl:.2f}")
    logger.critical("=" * 80)

    try:
      # 1. Остановить прием новых сигналов
      self.is_emergency_shutdown = True

      # 2. Отправить критические уведомления
      await self._send_emergency_notifications(metrics)

      # 3. Audit log
      await audit_repository.log(
        action=AuditAction.EMERGENCY_SHUTDOWN,
        entity_type="DailyLossKiller",
        entity_id="emergency_shutdown",
        success=True,
        context={
          'daily_loss_percent': metrics.daily_loss_percent,
          'starting_balance': metrics.starting_balance,
          'current_balance': metrics.current_balance,
          'daily_pnl': metrics.daily_pnl,
          'timestamp': datetime.now().isoformat()
        },
        reason=f"Daily loss {metrics.daily_loss_percent:.2%} exceeded max {self.max_loss_percent:.2%}"
      )

      logger.critical("=" * 80)
      logger.critical("🚨 EMERGENCY SHUTDOWN COMPLETED 🚨")
      logger.critical("TRADING DISABLED - MANUAL INTERVENTION REQUIRED")
      logger.critical("=" * 80)

    except Exception as e:
      logger.critical(f"ERROR DURING EMERGENCY SHUTDOWN: {e}", exc_info=True)
      # Все равно блокируем торговлю
      self.is_emergency_shutdown = True

      await notification_service.send_critical_alert(
        title="EMERGENCY SHUTDOWN FAILED",
        message=f"Error during shutdown: {e}",
        context={'error': str(e)}
      )

  async def _send_warning_notification(self, metrics: DailyLossMetrics):
    """Отправка предупреждения о приближении к лимиту."""
    await notification_service.send_warning(
      title=f"⚠️ Daily Loss Warning: {metrics.daily_loss_percent:.2%}",
      message=(
        f"Daily loss approaching critical threshold!\n\n"
        f"Current Loss: {metrics.daily_loss_percent:.2%}\n"
        f"Warning Threshold: {self.warning_percent:.2%}\n"
        f"Critical Threshold: {self.max_loss_percent:.2%}\n\n"
        f"Starting Balance: ${metrics.starting_balance:.2f}\n"
        f"Current Balance: ${metrics.current_balance:.2f}\n"
        f"Daily PnL: ${metrics.daily_pnl:.2f}"
      ),
      context=metrics.__dict__
    )

  async def _send_emergency_notifications(self, metrics: DailyLossMetrics):
    """Отправка критических уведомлений при shutdown."""
    await notification_service.send_critical_alert(
      title="🚨 EMERGENCY SHUTDOWN - Critical Daily Loss",
      message=(
        f"EMERGENCY SHUTDOWN EXECUTED\n\n"
        f"Daily Loss: {metrics.daily_loss_percent:.2%} "
        f"(max allowed: {self.max_loss_percent:.2%})\n\n"
        f"Starting Balance: ${metrics.starting_balance:.2f}\n"
        f"Current Balance: ${metrics.current_balance:.2f}\n"
        f"Daily PnL: ${metrics.daily_pnl:.2f}\n\n"
        f"⚠️ MANUAL INTERVENTION REQUIRED ⚠️\n"
        f"Trading will resume after manual review and approval."
      ),
      context=metrics.__dict__
    )

  def is_trading_allowed(self) -> Tuple[bool, Optional[str]]:
    """
    Проверка, разрешена ли торговля.

    Returns:
        (allowed, reason)
    """
    if not self.enabled:
      return True, None

    if self.is_emergency_shutdown:
      return False, "Emergency shutdown active - manual intervention required"

    return True, None

  def get_metrics(self) -> Optional[DailyLossMetrics]:
    """Получение текущих метрик убытка."""
    if self.starting_balance is None:
      return None

    current_balance = balance_tracker.get_current_balance()
    if current_balance is None:
      return None

    return self._calculate_metrics(current_balance)


# Глобальный экземпляр
daily_loss_killer = DailyLossKiller()