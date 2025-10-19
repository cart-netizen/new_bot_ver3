"""
Тесты для Daily Loss Killer.

Покрытие:
- Проверка расчета метрик убытка
- Тригер WARNING при 10% убытке
- Тригер EMERGENCY SHUTDOWN при 15% убытке
- Daily reset в полночь
- Проверка is_trading_allowed()

Путь: backend/tests/test_daily_loss_killer.py
"""

import sys
from pathlib import Path

# Определяем путь к backend директории
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from strategy.daily_loss_killer import DailyLossKiller
from strategy.risk_models import DailyLossMetrics


class TestDailyLossKiller:
  """Тесты для Daily Loss Killer."""

  def setup_method(self):
    """Setup перед каждым тестом."""
    self.killer = DailyLossKiller()
    self.killer.enabled = True
    self.killer.starting_balance = 10000.0

  def test_initialization(self):
    """Тест инициализации."""
    killer = DailyLossKiller()

    assert killer.max_loss_percent > 0
    assert killer.warning_percent > 0
    assert killer.warning_percent < killer.max_loss_percent
    assert killer.is_emergency_shutdown is False

  def test_calculate_metrics_no_loss(self):
    """Тест расчета метрик без убытка."""
    current_balance = 10000.0
    metrics = self.killer._calculate_metrics(current_balance)

    assert metrics.starting_balance == 10000.0
    assert metrics.current_balance == 10000.0
    assert metrics.daily_pnl == 0.0
    assert metrics.daily_loss_percent == 0.0
    assert metrics.is_critical is False

  def test_calculate_metrics_profit(self):
    """Тест расчета метрик с прибылью."""
    current_balance = 11000.0  # +10%
    metrics = self.killer._calculate_metrics(current_balance)

    assert metrics.daily_pnl == 1000.0
    assert metrics.daily_loss_percent == 0.0  # Нет убытка
    assert metrics.is_critical is False

  def test_calculate_metrics_warning_loss(self):
    """Тест расчета метрик при WARNING уровне (10%)."""
    current_balance = 9000.0  # -10%
    metrics = self.killer._calculate_metrics(current_balance)

    assert metrics.daily_pnl == -1000.0
    assert abs(metrics.daily_loss_percent - 0.10) < 0.001  # 10%

    # Для warning (10%) - еще не critical
    if self.killer.max_loss_percent > 0.10:
      assert metrics.is_critical is False

  def test_calculate_metrics_critical_loss(self):
    """Тест расчета метрик при CRITICAL уровне (15%)."""
    current_balance = 8500.0  # -15%
    metrics = self.killer._calculate_metrics(current_balance)

    assert metrics.daily_pnl == -1500.0
    assert abs(metrics.daily_loss_percent - 0.15) < 0.001  # 15%
    assert metrics.is_critical is True

  def test_is_trading_allowed_normal(self):
    """Тест is_trading_allowed() в нормальном состоянии."""
    self.killer.is_emergency_shutdown = False

    allowed, reason = self.killer.is_trading_allowed()

    assert allowed is True
    assert reason is None

  def test_is_trading_allowed_shutdown(self):
    """Тест is_trading_allowed() при emergency shutdown."""
    self.killer.is_emergency_shutdown = True

    allowed, reason = self.killer.is_trading_allowed()

    assert allowed is False
    assert reason is not None
    assert "emergency" in reason.lower()

  def test_is_trading_allowed_disabled(self):
    """Тест is_trading_allowed() когда killer выключен."""
    self.killer.enabled = False
    self.killer.is_emergency_shutdown = True  # Даже при shutdown

    allowed, reason = self.killer.is_trading_allowed()

    # Если killer disabled, торговля всегда разрешена
    assert allowed is True

  def test_get_metrics_normal(self):
    """Тест get_metrics() в нормальном состоянии."""
    self.killer.starting_balance = 10000.0

    with patch('utils.balance_tracker.balance_tracker.get_current_balance', return_value=9500.0):
      metrics = self.killer.get_metrics()

    assert metrics is not None
    assert metrics.current_balance == 9500.0
    assert metrics.daily_pnl == -500.0
    assert metrics.daily_loss_percent == 0.05  # 5%

  def test_get_metrics_no_starting_balance(self):
    """Тест get_metrics() без starting balance."""
    self.killer.starting_balance = None

    metrics = self.killer.get_metrics()

    assert metrics is None

  def test_get_metrics_no_current_balance(self):
    """Тест get_metrics() когда текущий баланс недоступен."""
    self.killer.starting_balance = 10000.0

    with patch('utils.balance_tracker.balance_tracker.get_current_balance', return_value=None):
      metrics = self.killer.get_metrics()

    assert metrics is None

  @pytest.mark.asyncio
  async def test_warning_notification_sent(self):
    """Тест отправки WARNING уведомления."""
    self.killer.starting_balance = 10000.0
    self.killer.warning_sent = False

    with patch('utils.balance_tracker.balance_tracker.get_current_balance', return_value=9000.0):
      with patch.object(self.killer, '_send_warning_notification', new_callable=AsyncMock) as mock_warning:
        await self.killer._check_daily_loss()

        # Должно быть вызвано при 10% убытке
        mock_warning.assert_called_once()
        assert self.killer.warning_sent is True

  @pytest.mark.asyncio
  async def test_emergency_shutdown_triggered(self):
    """Тест тригера emergency shutdown."""
    self.killer.starting_balance = 10000.0
    self.killer.is_emergency_shutdown = False

    with patch('utils.balance_tracker.balance_tracker.get_current_balance', return_value=8500.0):  # -15%
      with patch.object(self.killer, '_trigger_emergency_shutdown', new_callable=AsyncMock) as mock_shutdown:
        await self.killer._check_daily_loss()

        # Должен быть вызван при 15% убытке
        mock_shutdown.assert_called_once()

  @pytest.mark.asyncio
  async def test_emergency_shutdown_only_once(self):
    """Тест что emergency shutdown выполняется только один раз."""
    self.killer.starting_balance = 10000.0
    self.killer.is_emergency_shutdown = True  # Уже выполнено

    metrics = DailyLossMetrics(
      starting_balance=10000.0,
      current_balance=8500.0,
      daily_pnl=-1500.0,
      daily_loss_percent=0.15,
      max_daily_loss_percent=0.15,
      is_critical=True,
      time_to_reset=datetime.now() + timedelta(days=1)
    )

    with patch.object(self.killer, '_send_emergency_notifications', new_callable=AsyncMock) as mock_notify:
      await self.killer._trigger_emergency_shutdown(metrics)

      # Не должно быть вызвано, т.к. shutdown уже активен
      mock_notify.assert_not_called()


class TestDailyLossKillerIntegration:
  """Интеграционные тесты."""

  @pytest.mark.asyncio
  async def test_full_warning_cycle(self):
    """Тест полного цикла: нормально -> warning -> восстановление."""
    killer = DailyLossKiller()
    killer.enabled = True
    killer.starting_balance = 10000.0
    killer.warning_sent = False

    # 1. Нормальное состояние (9500 = -5%)
    with patch('utils.balance_tracker.balance_tracker.get_current_balance', return_value=9500.0):
      await killer._check_daily_loss()

      assert killer.warning_sent is False
      assert killer.is_emergency_shutdown is False

    # 2. WARNING состояние (9000 = -10%)
    with patch('utils.balance_tracker.balance_tracker.get_current_balance', return_value=9000.0):
      with patch.object(killer, '_send_warning_notification', new_callable=AsyncMock):
        await killer._check_daily_loss()

        assert killer.warning_sent is True
        assert killer.is_emergency_shutdown is False

    # 3. Восстановление (9800 = -2%)
    with patch('utils.balance_tracker.balance_tracker.get_current_balance', return_value=9800.0):
      await killer._check_daily_loss()

      # Warning остается True (не сбрасывается до daily reset)
      assert killer.warning_sent is True
      assert killer.is_emergency_shutdown is False

  @pytest.mark.asyncio
  async def test_full_shutdown_cycle(self):
    """Тест полного цикла: нормально -> critical -> shutdown."""
    killer = DailyLossKiller()
    killer.enabled = True
    killer.starting_balance = 10000.0

    # 1. CRITICAL loss (8500 = -15%)
    with patch('utils.balance_tracker.balance_tracker.get_current_balance', return_value=8500.0):
      with patch.object(killer, '_send_emergency_notifications', new_callable=AsyncMock):
        with patch('infrastructure.repositories.audit_repository.audit_repository.log', new_callable=AsyncMock):
          await killer._check_daily_loss()

          assert killer.is_emergency_shutdown is True

    # 2. Проверка is_trading_allowed()
    allowed, reason = killer.is_trading_allowed()

    assert allowed is False
    assert "emergency" in reason.lower()