# backend/utils/balance_tracker.py
"""
Модуль отслеживания баланса.
Периодически сохраняет баланс и ведет историю.
"""

import asyncio
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

from backend.core.logger import get_logger
from backend.utils.helpers import get_timestamp_ms

logger = get_logger(__name__)


class BalanceTracker:
  """Трекер баланса аккаунта."""

  def __init__(self, save_interval: int = 60):
    """
    Инициализация трекера баланса.

    Args:
        save_interval: Интервал сохранения баланса в секундах (по умолчанию 60 = 1 минута)
    """
    self.save_interval = save_interval
    self.is_running = False
    self.tracking_task: Optional[asyncio.Task] = None

    # История баланса в памяти (последние 30 дней)
    self.balance_history: deque = deque(maxlen=30 * 24 * 60)  # 30 дней по минутам

    # Путь к файлу с историей
    self.history_file = Path("data/balance_history.json")
    self.history_file.parent.mkdir(exist_ok=True)

    # Начальный баланс для расчета статистики
    self.initial_balance: Optional[float] = None

    # Загружаем сохраненную историю
    self._load_history()

    logger.info(f"Инициализирован трекер баланса (интервал: {save_interval}с)")

  def _load_history(self):
    """Загрузка истории из файла."""
    try:
      if self.history_file.exists():
        with open(self.history_file, 'r') as f:
          data = json.load(f)
          self.balance_history = deque(data.get("history", []), maxlen=30 * 24 * 60)
          self.initial_balance = data.get("initial_balance")
        logger.info(f"Загружено {len(self.balance_history)} записей истории")
      else:
        logger.info("Файл истории не найден, создаем новый")
    except Exception as e:
      logger.error(f"Ошибка загрузки истории: {e}")
      self.balance_history = deque(maxlen=30 * 24 * 60)

  def _save_history(self):
    """Сохранение истории в файл."""
    try:
      data = {
        "initial_balance": self.initial_balance,
        "history": list(self.balance_history),
        "last_updated": datetime.now().isoformat()
      }
      with open(self.history_file, 'w') as f:
        json.dump(data, f, indent=2)
      logger.debug(f"История сохранена ({len(self.balance_history)} записей)")
    except Exception as e:
      logger.error(f"Ошибка сохранения истории: {e}")

  async def start(self):
    """Запуск трекера баланса."""
    if self.is_running:
      logger.warning("Трекер баланса уже запущен")
      return

    self.is_running = True
    logger.info("Запуск трекера баланса")

    # Запускаем задачу отслеживания
    self.tracking_task = asyncio.create_task(self._tracking_loop())

  async def stop(self):
    """Остановка трекера баланса."""
    if not self.is_running:
      logger.warning("Трекер баланса уже остановлен")
      return

    logger.info("Остановка трекера баланса")
    self.is_running = False

    # Отменяем задачу
    if self.tracking_task and not self.tracking_task.done():
      self.tracking_task.cancel()
      try:
        await self.tracking_task
      except asyncio.CancelledError:
        pass

    # Сохраняем финальное состояние
    self._save_history()

  async def _tracking_loop(self):
    """Цикл отслеживания баланса."""
    logger.info("Запущен цикл отслеживания баланса")

    while self.is_running:
      try:
        # Получаем текущий баланс
        from backend.exchange.rest_client import rest_client
        from backend.config import settings

        # Проверяем что API ключи настроены
        if not settings.BYBIT_API_KEY or not settings.BYBIT_API_SECRET:
          logger.warning("API ключи не настроены, пропускаем сохранение баланса")
          await asyncio.sleep(self.save_interval)
          continue

        try:
          balance_data = await rest_client.get_wallet_balance()
          total_balance = self._calculate_total_balance(balance_data)

          # Сохраняем в историю
          timestamp = get_timestamp_ms()
          self.balance_history.append({
            "timestamp": timestamp,
            "balance": total_balance,
            "datetime": datetime.fromtimestamp(timestamp / 1000).isoformat()
          })

          from backend.main import bot_controller
          if bot_controller and bot_controller.risk_manager:
            bot_controller.risk_manager.update_available_balance(total_balance)

          # Устанавливаем начальный баланс при первой записи
          if self.initial_balance is None:
            self.initial_balance = total_balance
            logger.info(f"Установлен начальный баланс: ${total_balance:.2f}")

          # Сохраняем в файл каждые 10 записей
          if len(self.balance_history) % 10 == 0:
            self._save_history()

          logger.debug(f"Баланс сохранен: ${total_balance:.2f}")

        except ValueError as e:
          # API ключи не настроены
          logger.warning(f"Не удалось получить баланс: {e}")
        except Exception as e:
          logger.error(f"Ошибка получения баланса: {e}")

        # Ждем интервал
        await asyncio.sleep(self.save_interval)

      except asyncio.CancelledError:
        logger.info("Цикл отслеживания баланса отменен")
        break
      except Exception as e:
        logger.error(f"Ошибка в цикле отслеживания баланса: {e}")
        await asyncio.sleep(self.save_interval)

  def _calculate_total_balance(self, balance_data: Dict) -> float:
    """
    Расчет общего баланса в USDT.

    Args:
        balance_data: Данные баланса от Bybit

    Returns:
        float: Общий баланс в USDT
    """
    try:
      result = balance_data.get("result", {})
      wallet_list = result.get("list", [])

      total_usdt = 0.0

      for wallet in wallet_list:
        coins = wallet.get("coin", [])
        for coin in coins:
          # Получаем баланс в USDT или эквивалент
          if coin.get("coin") == "USDT":
            wallet_balance = float(coin.get("walletBalance", 0))
            total_usdt += wallet_balance

      return round(total_usdt, 2)

    except Exception as e:
      logger.error(f"Ошибка расчета общего баланса: {e}")
      return 0.0

  def get_current_balance(self) -> Optional[float]:
    """
    Получение текущего баланса из истории.

    Returns:
        Optional[float]: Текущий баланс или None
    """
    if self.balance_history:
      return self.balance_history[-1]["balance"]
    return None

  def get_history(self, period: str = "24h") -> List[Dict]:
    """
    Получение истории баланса за период.

    Args:
        period: Период ('1h', '24h', '7d', '30d')

    Returns:
        List[Dict]: История баланса
    """
    if not self.balance_history:
      return []

    # Определяем временной диапазон
    now = datetime.now()
    period_map = {
      "1h": timedelta(hours=1),
      "24h": timedelta(hours=24),
      "7d": timedelta(days=7),
      "30d": timedelta(days=30)
    }

    delta = period_map.get(period, timedelta(hours=24))
    cutoff_time = (now - delta).timestamp() * 1000

    # Фильтруем записи
    filtered = [
      record for record in self.balance_history
      if record["timestamp"] >= cutoff_time
    ]

    return filtered

  def get_stats(self) -> Dict:
    """
    Получение статистики по балансу.

    Returns:
        Dict: Статистика баланса
    """
    if not self.balance_history or self.initial_balance is None:
      return {
        "initial_balance": 0.0,
        "current_balance": 0.0,
        "total_pnl": 0.0,
        "total_pnl_percentage": 0.0,
        "daily_pnl": 0.0,
        "daily_pnl_percentage": 0.0,
        "best_day": 0.0,
        "worst_day": 0.0
      }

    current_balance = self.balance_history[-1]["balance"]

    # Общий PnL
    total_pnl = current_balance - self.initial_balance
    total_pnl_percentage = (total_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0.0

    # Дневной PnL
    daily_history = self.get_history("24h")
    if len(daily_history) >= 2:
      daily_start = daily_history[0]["balance"]
      daily_pnl = current_balance - daily_start
      daily_pnl_percentage = (daily_pnl / daily_start * 100) if daily_start > 0 else 0.0
    else:
      daily_pnl = 0.0
      daily_pnl_percentage = 0.0

    # Лучший и худший день
    best_day, worst_day = self._calculate_best_worst_days()

    return {
      "initial_balance": round(self.initial_balance, 2),
      "current_balance": round(current_balance, 2),
      "total_pnl": round(total_pnl, 2),
      "total_pnl_percentage": round(total_pnl_percentage, 2),
      "daily_pnl": round(daily_pnl, 2),
      "daily_pnl_percentage": round(daily_pnl_percentage, 2),
      "best_day": round(best_day, 2),
      "worst_day": round(worst_day, 2)
    }

  def _calculate_best_worst_days(self) -> tuple[float, float]:
    """
    Расчет лучшего и худшего дня.

    Returns:
        tuple: (best_day, worst_day)
    """
    if len(self.balance_history) < 2:
      return (0.0, 0.0)

    # Группируем по дням
    daily_changes = {}
    prev_balance = None

    for record in self.balance_history:
      date = datetime.fromtimestamp(record["timestamp"] / 1000).date()
      balance = record["balance"]

      if prev_balance is not None:
        change = balance - prev_balance
        if date not in daily_changes:
          daily_changes[date] = 0.0
        daily_changes[date] += change

      prev_balance = balance

    if not daily_changes:
      return (0.0, 0.0)

    best_day = max(daily_changes.values())
    worst_day = min(daily_changes.values())

    return (best_day, worst_day)


# Глобальный экземпляр трекера
balance_tracker = BalanceTracker(save_interval=60)