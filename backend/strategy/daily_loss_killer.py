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
from typing import Optional, Tuple, Dict, TYPE_CHECKING

from backend.core.logger import get_logger
from backend.config import settings
from backend.utils.balance_tracker import balance_tracker
from backend.services.notification_service import NotificationService
from backend.infrastructure.repositories.audit_repository import audit_repository
from backend.database.models import AuditAction
from backend.strategy.risk_models import DailyLossMetrics

# Type-only imports to avoid circular dependency
if TYPE_CHECKING:
  from backend.execution.execution_manager import ExecutionManager

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

  def __init__(self, execution_manager: Optional['ExecutionManager'] = None):
    """
    Инициализация Daily Loss Killer.

    Args:
        execution_manager: Менеджер исполнения для закрытия позиций (опционально)
    """
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

    # ExecutionManager для закрытия позиций при emergency shutdown
    self.execution_manager = execution_manager

    logger.info(
      f"DailyLossKiller initialized: "
      f"enabled={self.enabled}, "
      f"max_loss={self.max_loss_percent:.0%}, "
      f"warning={self.warning_percent:.0%}, "
      f"has_execution_manager={execution_manager is not None}"
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

  async def _get_current_market_price(self, symbol: str) -> Optional[float]:
    """
    Получение текущей рыночной цены для закрытия позиции.

    Использует REST API для получения последней цены.

    Args:
        symbol: Торговая пара (e.g., "BTCUSDT")

    Returns:
        float: Текущая цена или None при ошибке
    """
    try:
      from backend.exchange.rest_client import rest_client

      # Получаем ticker с биржи
      ticker = await rest_client.get_ticker(symbol=symbol)

      if not ticker:
        logger.error(f"Failed to get ticker for {symbol}")
        return None

      # Bybit API возвращает: {"result": {"list": [{"lastPrice": "..."}]}}
      result = ticker.get("result", {})

      if isinstance(result, dict):
        ticker_list = result.get("list", [])
        if ticker_list and len(ticker_list) > 0:
          last_price = float(ticker_list[0].get("lastPrice", 0))
          logger.debug(f"{symbol} | Current market price: ${last_price:.2f}")
          return last_price

      logger.error(f"Unexpected ticker format for {symbol}: {ticker}")
      return None

    except Exception as e:
      logger.error(f"Error getting market price for {symbol}: {e}", exc_info=True)
      return None

  async def _close_all_positions(self) -> Dict[str, any]:
    """
    Закрытие ВСЕХ открытых позиций при emergency shutdown.

    Алгоритм:
    1. Получить список активных позиций из БД
    2. Для каждой позиции:
       - Получить текущую рыночную цену
       - Закрыть через ExecutionManager
       - Логировать результат
    3. Вернуть статистику: успешно/неудачно

    Returns:
        Dict с результатами: {
            'total': int,
            'closed': int,
            'failed': int,
            'failed_symbols': List[str],
            'closed_positions': List[str]
        }
    """
    if not self.execution_manager:
      logger.critical(
        "ExecutionManager not initialized! "
        "Cannot close positions during emergency shutdown."
      )
      return {
        'total': 0,
        'closed': 0,
        'failed': 0,
        'failed_symbols': [],
        'closed_positions': [],
        'error': 'ExecutionManager not available'
      }

    logger.critical("=" * 80)
    logger.critical("🚨 CLOSING ALL OPEN POSITIONS 🚨")
    logger.critical("=" * 80)

    try:
      # ==========================================
      # ШАГ 1: ПОЛУЧЕНИЕ АКТИВНЫХ ПОЗИЦИЙ
      # ==========================================
      from backend.infrastructure.repositories.position_repository import position_repository

      active_positions = await position_repository.get_active_positions()

      if not active_positions:
        logger.info("No active positions to close")
        return {
          'total': 0,
          'closed': 0,
          'failed': 0,
          'failed_symbols': [],
          'closed_positions': []
        }

      logger.critical(f"Found {len(active_positions)} active positions to close")

      # ==========================================
      # ШАГ 2: ЗАКРЫТИЕ КАЖДОЙ ПОЗИЦИИ
      # ==========================================
      results = {
        'total': len(active_positions),
        'closed': 0,
        'failed': 0,
        'failed_symbols': [],
        'closed_positions': []
      }

      for position in active_positions:
        symbol = position.symbol
        position_id = str(position.id)

        logger.critical(
          f"Closing position: {symbol} | "
          f"ID: {position_id} | "
          f"Side: {position.side.value} | "
          f"Entry: ${position.entry_price:.2f}"
        )

        try:
          # Получаем текущую цену
          current_price = await self._get_current_market_price(symbol)

          if current_price is None:
            logger.error(
              f"Failed to get market price for {symbol}, "
              f"using entry price as fallback"
            )
            current_price = position.entry_price

          # Закрываем позицию через ExecutionManager
          close_result = await self.execution_manager.close_position(
            position_id=position_id,
            exit_price=current_price,
            exit_reason="Emergency shutdown: Critical daily loss"
          )

          if close_result and close_result.get('status') == 'success':
            results['closed'] += 1
            results['closed_positions'].append(position_id)

            realized_pnl = close_result.get('realized_pnl', 0.0)

            logger.critical(
              f"✓ Position closed: {symbol} | "
              f"Exit: ${current_price:.2f} | "
              f"PnL: ${realized_pnl:+.2f}"
            )
          else:
            results['failed'] += 1
            results['failed_symbols'].append(symbol)

            logger.error(
              f"✗ Failed to close position: {symbol} | "
              f"Result: {close_result}"
            )

        except Exception as e:
          results['failed'] += 1
          results['failed_symbols'].append(symbol)

          logger.error(
            f"✗ Error closing position {symbol}: {e}",
            exc_info=True
          )

      # ==========================================
      # ШАГ 3: SUMMARY
      # ==========================================
      logger.critical("=" * 80)
      logger.critical(f"POSITIONS CLOSURE SUMMARY:")
      logger.critical(f"  Total: {results['total']}")
      logger.critical(f"  Closed: {results['closed']}")
      logger.critical(f"  Failed: {results['failed']}")
      if results['failed_symbols']:
        logger.critical(f"  Failed symbols: {', '.join(results['failed_symbols'])}")
      logger.critical("=" * 80)

      return results

    except Exception as e:
      logger.critical(
        f"CRITICAL ERROR during position closure: {e}",
        exc_info=True
      )

      return {
        'total': 0,
        'closed': 0,
        'failed': 0,
        'failed_symbols': [],
        'closed_positions': [],
        'error': str(e)
      }

  async def _trigger_emergency_shutdown(self, metrics: DailyLossMetrics):
    """
    EMERGENCY SHUTDOWN при достижении критического убытка.

    ДЕЙСТВИЯ:
    1. Остановить торговлю
    2. Закрыть все открытые позиции через ExecutionManager
    3. Отправить критические уведомления
    4. Audit log
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

      # 2. Закрыть все открытые позиции через ExecutionManager
      closure_results = await self._close_all_positions()

      # 3. Отправить критические уведомления
      await self._send_emergency_notifications(metrics, closure_results)

      # 4. Audit log
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
          'positions_closure': closure_results,
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

  async def _send_emergency_notifications(
    self,
    metrics: DailyLossMetrics,
    closure_results: Optional[Dict] = None
  ):
    """
    Отправка критических уведомлений при shutdown.

    Args:
        metrics: Метрики убытка
        closure_results: Результаты закрытия позиций (опционально)
    """
    # Формируем сообщение о закрытии позиций
    positions_message = ""
    if closure_results:
      positions_message = (
        f"\n\n📊 POSITIONS CLOSED:\n"
        f"Total: {closure_results.get('total', 0)}\n"
        f"Successfully closed: {closure_results.get('closed', 0)}\n"
        f"Failed to close: {closure_results.get('failed', 0)}"
      )

      if closure_results.get('failed_symbols'):
        positions_message += f"\n⚠️ Failed symbols: {', '.join(closure_results['failed_symbols'])}"

      if closure_results.get('error'):
        positions_message += f"\n❌ Error: {closure_results['error']}"

    await notification_service.send_critical_alert(
      title="🚨 EMERGENCY SHUTDOWN - Critical Daily Loss",
      message=(
        f"EMERGENCY SHUTDOWN EXECUTED\n\n"
        f"Daily Loss: {metrics.daily_loss_percent:.2%} "
        f"(max allowed: {self.max_loss_percent:.2%})\n\n"
        f"Starting Balance: ${metrics.starting_balance:.2f}\n"
        f"Current Balance: ${metrics.current_balance:.2f}\n"
        f"Daily PnL: ${metrics.daily_pnl:.2f}"
        f"{positions_message}\n\n"
        f"⚠️ MANUAL INTERVENTION REQUIRED ⚠️\n"
        f"Trading will resume after manual review and approval."
      ),
      context={
        **metrics.__dict__,
        'positions_closure': closure_results
      }
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

  def get_statistics(self) -> Dict:
    """
    Получение статистики DailyLossKiller для мониторинга.

    Returns:
        Dict с текущим состоянием
    """
    current_balance = balance_tracker.get_current_balance()
    metrics = self._calculate_metrics(current_balance) if current_balance and self.starting_balance else None

    return {
      'enabled': self.enabled,
      'is_emergency_shutdown': self.is_emergency_shutdown,
      'starting_balance': self.starting_balance,
      'current_balance': current_balance,
      'daily_pnl': metrics.daily_pnl if metrics else None,
      'daily_loss_percent': metrics.daily_loss_percent if metrics else None,
      'max_loss_percent': self.max_loss_percent,
      'warning_percent': self.warning_percent,
      'warning_sent': self.warning_sent,
      'is_allowed': not self.is_emergency_shutdown if self.enabled else True,
      'has_execution_manager': self.execution_manager is not None
    }


# Глобальный экземпляр
daily_loss_killer = DailyLossKiller()