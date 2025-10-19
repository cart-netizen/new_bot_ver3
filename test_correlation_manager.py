"""
Тесты для CorrelationManager.

Покрытие:
- Расчет корреляций
- Группировка символов
- Проверка лимитов
- Обновление при открытии/закрытии позиций

Путь: backend/tests/test_correlation_manager.py
"""
import pytest
import numpy as np
from datetime import datetime

from strategy.correlation_manager import (
  CorrelationCalculator,
  CorrelationGroupManager,
  CorrelationManager
)
from strategy.risk_models import CorrelationGroup


class TestCorrelationCalculator:
  """Тесты для CorrelationCalculator."""

  def setup_method(self):
    """Setup перед каждым тестом."""
    self.calculator = CorrelationCalculator()

  def test_calculate_returns(self):
    """Тест расчета returns."""
    prices = np.array([100.0, 102.0, 101.0, 105.0])

    returns = self.calculator.calculate_returns(prices)

    # Ожидаемые returns:
    # (102-100)/100 = 0.02
    # (101-102)/102 ≈ -0.0098
    # (105-101)/101 ≈ 0.0396

    assert len(returns) == 3
    assert abs(returns[0] - 0.02) < 0.0001
    assert returns[1] < 0  # Отрицательный return
    assert returns[2] > 0  # Положительный return

  def test_calculate_returns_empty(self):
    """Тест с пустым массивом."""
    prices = np.array([])
    returns = self.calculator.calculate_returns(prices)
    assert len(returns) == 0

  def test_calculate_correlation_perfect_positive(self):
    """Тест идеальной положительной корреляции."""
    # Оба актива растут одинаково
    returns_a = np.array([0.01, 0.02, 0.03, 0.01])
    returns_b = np.array([0.01, 0.02, 0.03, 0.01])

    correlation = self.calculator.calculate_correlation(returns_a, returns_b)

    # Должна быть близка к 1.0
    assert abs(correlation - 1.0) < 0.01

  def test_calculate_correlation_perfect_negative(self):
    """Тест идеальной отрицательной корреляции."""
    # Активы движутся в противоположных направлениях
    returns_a = np.array([0.01, 0.02, 0.03, 0.01])
    returns_b = np.array([-0.01, -0.02, -0.03, -0.01])

    correlation = self.calculator.calculate_correlation(returns_a, returns_b)

    # Должна быть близка к -1.0
    assert abs(correlation - (-1.0)) < 0.01

  def test_calculate_correlation_no_correlation(self):
    """Тест отсутствия корреляции."""
    # Случайные независимые движения
    np.random.seed(42)
    returns_a = np.random.randn(100)
    returns_b = np.random.randn(100)

    correlation = self.calculator.calculate_correlation(returns_a, returns_b)

    # Корреляция должна быть близка к 0
    assert abs(correlation) < 0.3

  def test_calculate_correlation_different_lengths(self):
    """Тест с массивами разной длины."""
    returns_a = np.array([0.01, 0.02, 0.03])
    returns_b = np.array([0.01, 0.02, 0.03, 0.01, 0.02])

    # Должно использовать последние 3 элемента из returns_b
    correlation = self.calculator.calculate_correlation(returns_a, returns_b)

    # Корреляция должна быть рассчитана
    assert -1.0 <= correlation <= 1.0


class TestCorrelationGroupManager:
  """Тесты для CorrelationGroupManager."""

  def setup_method(self):
    """Setup."""
    self.manager = CorrelationGroupManager(correlation_threshold=0.7)

  def test_create_groups_high_correlation(self):
    """Тест создания групп с высокой корреляцией."""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # BTC-ETH: 0.85 (высокая)
    # BTC-SOL: 0.78 (высокая)
    # ETH-SOL: 0.80 (высокая)
    correlation_matrix = {
      ("BTCUSDT", "ETHUSDT"): 0.85,
      ("BTCUSDT", "SOLUSDT"): 0.78,
      ("ETHUSDT", "SOLUSDT"): 0.80
    }

    self.manager.create_groups_from_matrix(symbols, correlation_matrix)

    # Все 3 должны быть в одной группе
    assert len(self.manager.groups) == 1

    group = list(self.manager.groups.values())[0]
    assert len(group.symbols) == 3
    assert set(group.symbols) == set(symbols)

  def test_create_groups_mixed_correlation(self):
    """Тест с mixed корреляцией."""
    symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "UNIUSDT"]

    # BTC-ETH: 0.85 (группа 1)
    # DOGE-UNI: 0.75 (группа 2)
    # BTC-DOGE: 0.3 (нет группы)
    correlation_matrix = {
      ("BTCUSDT", "ETHUSDT"): 0.85,
      ("BTCUSDT", "DOGEUSDT"): 0.3,
      ("BTCUSDT", "UNIUSDT"): 0.2,
      ("DOGEUSDT", "UNIUSDT"): 0.75,
      ("ETHUSDT", "DOGEUSDT"): 0.25,
      ("ETHUSDT", "UNIUSDT"): 0.15
    }

    self.manager.create_groups_from_matrix(symbols, correlation_matrix)

    # Должно быть 2 группы
    assert len(self.manager.groups) == 2

    # Проверяем что символы правильно распределены
    all_grouped_symbols = []
    for group in self.manager.groups.values():
      all_grouped_symbols.extend(group.symbols)

    assert "BTCUSDT" in all_grouped_symbols
    assert "ETHUSDT" in all_grouped_symbols
    assert "DOGEUSDT" in all_grouped_symbols
    assert "UNIUSDT" in all_grouped_symbols

  def test_create_groups_no_correlation(self):
    """Тест когда нет корреляций."""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # Все корреляции низкие
    correlation_matrix = {
      ("BTCUSDT", "ETHUSDT"): 0.3,
      ("BTCUSDT", "SOLUSDT"): 0.2,
      ("ETHUSDT", "SOLUSDT"): 0.1
    }

    self.manager.create_groups_from_matrix(symbols, correlation_matrix)

    # Групп не должно быть
    assert len(self.manager.groups) == 0

  def test_get_group_for_symbol(self):
    """Тест получения группы для символа."""
    symbols = ["BTCUSDT", "ETHUSDT"]
    correlation_matrix = {
      ("BTCUSDT", "ETHUSDT"): 0.85
    }

    self.manager.create_groups_from_matrix(symbols, correlation_matrix)

    # Получаем группу для BTC
    group_btc = self.manager.get_group_for_symbol("BTCUSDT")
    assert group_btc is not None
    assert "BTCUSDT" in group_btc.symbols

    # Для несуществующего символа
    group_none = self.manager.get_group_for_symbol("XRPUSDT")
    assert group_none is None

  def test_update_group_position_opened(self):
    """Тест обновления при открытии позиции."""
    symbols = ["BTCUSDT", "ETHUSDT"]
    correlation_matrix = {
      ("BTCUSDT", "ETHUSDT"): 0.85
    }

    self.manager.create_groups_from_matrix(symbols, correlation_matrix)

    # Открываем позицию
    self.manager.update_group_position_opened("BTCUSDT", 1000.0)

    group = self.manager.get_group_for_symbol("BTCUSDT")
    assert group.active_positions == 1
    assert group.total_exposure_usdt == 1000.0

    # Открываем еще одну
    self.manager.update_group_position_opened("ETHUSDT", 500.0)

    assert group.active_positions == 2
    assert group.total_exposure_usdt == 1500.0

  def test_update_group_position_closed(self):
    """Тест обновления при закрытии позиции."""
    symbols = ["BTCUSDT", "ETHUSDT"]
    correlation_matrix = {
      ("BTCUSDT", "ETHUSDT"): 0.85
    }

    self.manager.create_groups_from_matrix(symbols, correlation_matrix)

    # Открываем 2 позиции
    self.manager.update_group_position_opened("BTCUSDT", 1000.0)
    self.manager.update_group_position_opened("ETHUSDT", 500.0)

    group = self.manager.get_group_for_symbol("BTCUSDT")
    assert group.active_positions == 2

    # Закрываем одну
    self.manager.update_group_position_closed("BTCUSDT", 1000.0)

    assert group.active_positions == 1
    assert group.total_exposure_usdt == 500.0


class TestCorrelationManager:
  """Тесты для CorrelationManager (unit)."""

  def setup_method(self):
    """Setup."""
    # Создаем manager с мок-данными
    self.manager = CorrelationManager()
    self.manager.enabled = True

  def test_can_open_position_no_group(self):
    """Тест когда символ не в группе."""
    # Символ не коррелирует ни с кем
    can_open, reason = self.manager.can_open_position("XRPUSDT", 1000.0)

    assert can_open is True
    assert reason is None

  def test_can_open_position_within_limit(self):
    """Тест открытия в пределах лимита."""
    # Создаем тестовую группу
    group = CorrelationGroup(
      group_id="test_group",
      symbols=["BTCUSDT", "ETHUSDT"],
      avg_correlation=0.85,
      active_positions=0,
      total_exposure_usdt=0.0
    )

    self.manager.group_manager.groups["test_group"] = group
    self.manager.group_manager.symbol_to_group["BTCUSDT"] = "test_group"

    # Лимит = 1, текущих позиций = 0
    self.manager.max_positions_per_group = 1

    can_open, reason = self.manager.can_open_position("BTCUSDT", 1000.0)

    assert can_open is True
    assert reason is None

  def test_can_open_position_exceeds_limit(self):
    """Тест превышения лимита."""
    # Создаем тестовую группу с уже открытой позицией
    group = CorrelationGroup(
      group_id="test_group",
      symbols=["BTCUSDT", "ETHUSDT"],
      avg_correlation=0.85,
      active_positions=1,  # Уже 1 позиция
      total_exposure_usdt=1000.0
    )

    self.manager.group_manager.groups["test_group"] = group
    self.manager.group_manager.symbol_to_group["ETHUSDT"] = "test_group"

    # Лимит = 1, текущих = 1
    self.manager.max_positions_per_group = 1

    can_open, reason = self.manager.can_open_position("ETHUSDT", 500.0)

    assert can_open is False
    assert reason is not None
    assert "лимит" in reason.lower()

  def test_notify_position_opened(self):
    """Тест уведомления об открытии."""
    group = CorrelationGroup(
      group_id="test_group",
      symbols=["BTCUSDT"],
      avg_correlation=0.0,
      active_positions=0,
      total_exposure_usdt=0.0
    )

    self.manager.group_manager.groups["test_group"] = group
    self.manager.group_manager.symbol_to_group["BTCUSDT"] = "test_group"

    # Уведомляем
    self.manager.notify_position_opened("BTCUSDT", 1000.0)

    # Проверяем обновление
    assert group.active_positions == 1
    assert group.total_exposure_usdt == 1000.0

  def test_notify_position_closed(self):
    """Тест уведомления о закрытии."""
    group = CorrelationGroup(
      group_id="test_group",
      symbols=["BTCUSDT"],
      avg_correlation=0.0,
      active_positions=1,
      total_exposure_usdt=1000.0
    )

    self.manager.group_manager.groups["test_group"] = group
    self.manager.group_manager.symbol_to_group["BTCUSDT"] = "test_group"

    # Уведомляем
    self.manager.notify_position_closed("BTCUSDT", 1000.0)

    # Проверяем обновление
    assert group.active_positions == 0
    assert group.total_exposure_usdt == 0.0

  def test_get_statistics(self):
    """Тест получения статистики."""
    group = CorrelationGroup(
      group_id="test_group",
      symbols=["BTCUSDT", "ETHUSDT"],
      avg_correlation=0.85,
      active_positions=2,
      total_exposure_usdt=1500.0
    )

    self.manager.group_manager.groups["test_group"] = group

    stats = self.manager.get_statistics()

    assert stats["enabled"] is True
    assert stats["total_groups"] == 1
    assert stats["groups_with_positions"] == 1
    assert stats["total_active_positions"] == 2
    assert stats["total_exposure_usdt"] == 1500.0