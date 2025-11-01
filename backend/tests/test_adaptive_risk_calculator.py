"""
Тесты для Adaptive Risk Calculator.

Покрытие:
- Fixed mode
- Adaptive mode с различными корректировками
- Kelly Criterion mode
- Volatility adjustment
- Win rate adjustment
- ML confidence adjustment
- Запись и использование истории трейдов

Путь: backend/tests/test_adaptive_risk_calculator.py
"""

import sys
from datetime import datetime
from pathlib import Path

# Определяем путь к backend директории
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

import pytest
import numpy as np
from unittest.mock import Mock, patch



from backend.strategy.adaptive_risk_calculator import AdaptiveRiskCalculator
from backend.models.signal import TradingSignal, SignalType, SignalStrength, SignalSource


class TestAdaptiveRiskCalculator:
  """Тесты для Adaptive Risk Calculator."""

  def setup_method(self):
    """Setup перед каждым тестом."""
    self.calculator = AdaptiveRiskCalculator()
    self.calculator.mode = "adaptive"  # По умолчанию adaptive
    self.calculator.trade_history = []  # Очищаем историю

  # ============================================================
  # ТЕСТЫ РЕЖИМОВ
  # ============================================================

  def test_fixed_mode(self):
    """Тест Fixed mode - всегда базовый риск."""
    self.calculator.mode = "fixed"
    self.calculator.base_percent = 0.02  # 2%

    risk = self.calculator._calculate_fixed()

    assert risk == 0.02

  def test_kelly_mode_insufficient_trades(self):
    """Тест Kelly mode с недостаточной историей."""
    self.calculator.mode = "kelly"
    self.calculator.kelly_min_trades = 30
    self.calculator.base_percent = 0.02

    # Только 10 трейдов (< 30 минимум)
    self.calculator.trade_history = [(True, 100)] * 10

    risk = self.calculator._calculate_kelly()

    # Должен вернуть base_percent
    assert risk == 0.02

  def test_kelly_mode_sufficient_trades(self):
    """Тест Kelly mode с достаточной историей."""
    self.calculator.mode = "kelly"
    self.calculator.kelly_min_trades = 30
    self.calculator.kelly_fraction = 0.25

    # 40 трейдов: 60% win rate, avg_win=200, avg_loss=100
    wins = [(True, 200)] * 24  # 60%
    losses = [(False, -100)] * 16  # 40%
    self.calculator.trade_history = wins + losses

    risk = self.calculator._calculate_kelly()

    # Kelly formula: (p * b - q) / b
    # p=0.6, q=0.4, b=200/100=2
    # f = (0.6 * 2 - 0.4) / 2 = 0.4
    # fractional = 0.4 * 0.25 = 0.1 = 10%

    assert risk > 0  # Должен быть положительный
    assert risk >= self.calculator.base_percent * 0.5  # Минимум 50% от base

  # ============================================================
  # ТЕСТЫ КОРРЕКТИРОВОК
  # ============================================================

  def test_volatility_adjustment_high(self):
    """Тест корректировки на высокую волатильность."""
    self.calculator.volatility_baseline = 0.02  # 2% baseline

    # Высокая волатильность (4%) -> должен снизить риск
    adj = self.calculator._get_volatility_adj(0.04)

    assert adj < 1.0  # Снижение риска
    assert adj == 0.5  # 0.02 / 0.04 = 0.5

  def test_volatility_adjustment_low(self):
    """Тест корректировки на низкую волатильность."""
    self.calculator.volatility_baseline = 0.02  # 2% baseline

    # Низкая волатильность (1%) -> должен увеличить риск
    adj = self.calculator._get_volatility_adj(0.01)

    assert adj > 1.0  # Увеличение риска
    assert adj <= 1.5  # Ограничено сверху

  def test_win_rate_adjustment_high(self):
    """Тест корректировки на высокий win rate."""
    self.calculator.win_rate_baseline = 0.55  # 55% baseline

    # 70% win rate -> должен увеличить риск
    self.calculator.trade_history = [(True, 100)] * 70 + [(False, -100)] * 30

    adj = self.calculator._get_win_rate_adj()

    assert adj > 1.0  # Увеличение риска
    # 0.7 / 0.55 = 1.27
    assert abs(adj - 1.27) < 0.01

  def test_win_rate_adjustment_low(self):
    """Тест корректировки на низкий win rate."""
    self.calculator.win_rate_baseline = 0.55  # 55% baseline

    # 40% win rate -> должен снизить риск
    self.calculator.trade_history = [(True, 100)] * 40 + [(False, -100)] * 60

    adj = self.calculator._get_win_rate_adj()

    assert adj < 1.0  # Снижение риска
    # 0.4 / 0.55 = 0.727, но ограничено [0.6, 1.4]
    assert adj >= 0.6

  def test_ml_confidence_adjustment_very_high(self):
    """Тест корректировки на очень высокую ML confidence."""
    adj = self.calculator._get_ml_confidence_adj(0.95)

    assert adj == 1.3  # +30% risk

  def test_ml_confidence_adjustment_high(self):
    """Тест корректировки на высокую ML confidence."""
    adj = self.calculator._get_ml_confidence_adj(0.85)

    assert adj == 1.15  # +15% risk

  def test_ml_confidence_adjustment_medium(self):
    """Тест корректировки на среднюю ML confidence."""
    adj = self.calculator._get_ml_confidence_adj(0.75)

    assert adj == 1.0  # Без изменений

  def test_ml_confidence_adjustment_low(self):
    """Тест корректировки на низкую ML confidence."""
    adj = self.calculator._get_ml_confidence_adj(0.65)

    assert adj == 0.85  # -15% risk

  # ============================================================
  # ТЕСТЫ ПОЛНОГО РАСЧЕТА
  # ============================================================

  def test_calculate_basic(self):
    """Тест базового расчета без корректировок."""
    self.calculator.mode = "fixed"
    self.calculator.base_percent = 0.02  # 2%

    signal = TradingSignal(
      symbol="BTCUSDT",
      signal_type=SignalType.BUY,
      strength=SignalStrength.MEDIUM,
      confidence=0.7,
      price=50000.0,
      source=SignalSource.STRATEGY,
      timestamp=int(datetime.now().timestamp() * 1000)
    )

    balance = 10000.0
    stop_loss_price = 49000.0  # 1000 USDT risk per BTC

    risk_params = self.calculator.calculate(
      signal=signal,
      balance=balance,
      stop_loss_price=stop_loss_price
    )

    # При 2% риске: max_risk = 10000 * 0.02 = 200 USDT
    # Risk per unit = 50000 - 49000 = 1000 USDT
    # Max quantity = 200 / 1000 = 0.2 BTC
    # Position size = 0.2 * 50000 = 10000 USDT
    # Но ограничено max_percent (5%) = 500 USDT

    assert risk_params.final_risk_percent == 0.02
    assert risk_params.max_position_usdt > 0
    assert risk_params.max_position_usdt <= balance * 0.05  # Не больше 5%

  def test_calculate_with_volatility(self):
    """Тест расчета с корректировкой на волатильность."""
    self.calculator.mode = "adaptive"
    self.calculator.base_percent = 0.02
    self.calculator.volatility_scaling = True
    self.calculator.volatility_baseline = 0.02

    signal = TradingSignal(
      symbol="BTCUSDT",
      signal_type=SignalType.BUY,
      strength=SignalStrength.MEDIUM,
      confidence=0.7,
      price=50000.0,
      source=SignalSource.STRATEGY,
      timestamp=int(datetime.now().timestamp() * 1000)
    )

    # Высокая волатильность -> снижение риска
    risk_params = self.calculator.calculate(
      signal=signal,
      balance=10000.0,
      stop_loss_price=49000.0,
      current_volatility=0.04  # 4% (в 2 раза выше baseline)
    )

    # final_risk должен быть меньше base из-за high volatility
    assert risk_params.final_risk_percent < self.calculator.base_percent
    assert risk_params.volatility_adjustment == 0.5

  def test_calculate_with_correlation_penalty(self):
    """Тест расчета с correlation penalty."""
    self.calculator.mode = "adaptive"
    self.calculator.base_percent = 0.02
    self.calculator.correlation_penalty = True

    signal = TradingSignal(
      symbol="BTCUSDT",
      signal_type=SignalType.BUY,
      strength=SignalStrength.MEDIUM,
      confidence=0.7,
      price=50000.0,
      source=SignalSource.STRATEGY,
      timestamp=int(datetime.now().timestamp() * 1000)
    )

    # Correlation factor = 0.7 (30% penalty)
    risk_params = self.calculator.calculate(
      signal=signal,
      balance=10000.0,
      stop_loss_price=49000.0,
      correlation_factor=0.7
    )

    # final_risk должен быть снижен correlation penalty
    expected_risk = self.calculator.base_percent * 0.7
    assert abs(risk_params.final_risk_percent - expected_risk) < 0.001
    assert risk_params.correlation_adjustment == 0.7

  def test_calculate_with_all_adjustments(self):
    """Тест расчета со всеми корректировками."""
    self.calculator.mode = "adaptive"
    self.calculator.base_percent = 0.02
    self.calculator.volatility_scaling = True
    self.calculator.win_rate_scaling = True
    self.calculator.correlation_penalty = True

    # Добавляем историю для win_rate adjustment
    self.calculator.trade_history = [(True, 100)] * 60 + [(False, -100)] * 40  # 60% win rate

    signal = TradingSignal(
      symbol="BTCUSDT",
      signal_type=SignalType.BUY,
      strength=SignalStrength.STRONG,
      confidence=0.8,
      price=50000.0,
      source=SignalSource.STRATEGY,
      timestamp=int(datetime.now().timestamp() * 1000)
    )

    risk_params = self.calculator.calculate(
      signal=signal,
      balance=10000.0,
      stop_loss_price=49000.0,
      current_volatility=0.02,  # Baseline volatility
      correlation_factor=0.9,  # Небольшой penalty
      ml_confidence=0.85  # High ML confidence
    )

    # Проверяем что все корректировки применены
    assert risk_params.volatility_adjustment == 1.0  # Baseline vol
    assert risk_params.correlation_adjustment == 0.9
    assert risk_params.final_risk_percent > 0

  # ============================================================
  # ТЕСТЫ ЗАПИСИ ИСТОРИИ
  # ============================================================

  def test_record_trade(self):
    """Тест записи результата трейда."""
    assert len(self.calculator.trade_history) == 0

    self.calculator.record_trade(True, 150.0)

    assert len(self.calculator.trade_history) == 1
    assert self.calculator.trade_history[0] == (True, 150.0)

  def test_record_trade_history_limit(self):
    """Тест лимита истории трейдов."""
    # Записываем 600 трейдов (лимит 500)
    for i in range(600):
      self.calculator.record_trade(True, 100.0)

    # Должно остаться только 500
    assert len(self.calculator.trade_history) == 500

  # ============================================================
  # ТЕСТЫ СТАТИСТИКИ
  # ============================================================

  def test_get_statistics_empty(self):
    """Тест статистики без истории."""
    stats = self.calculator.get_statistics()

    assert stats['total_trades'] == 0
    assert stats['win_rate'] == 0.0
    assert stats['avg_win'] == 0.0
    assert stats['avg_loss'] == 0.0
    assert stats['payoff_ratio'] == 0.0

  def test_get_statistics_with_trades(self):
    """Тест статистики с историей."""
    # 60% win rate, avg_win=200, avg_loss=100
    wins = [(True, 200)] * 6
    losses = [(False, -100)] * 4
    self.calculator.trade_history = wins + losses

    stats = self.calculator.get_statistics()

    assert stats['total_trades'] == 10
    assert stats['win_rate'] == 0.6
    assert stats['avg_win'] == 200.0
    assert stats['avg_loss'] == -100.0
    assert stats['payoff_ratio'] == 2.0  # 200 / 100


class TestAdaptiveRiskCalculatorIntegration:
  """Интеграционные тесты."""

  def test_full_adaptive_workflow(self):
    """Тест полного workflow: расчет -> запись -> следующий расчет."""
    calculator = AdaptiveRiskCalculator()
    calculator.mode = "adaptive"
    calculator.base_percent = 0.02
    calculator.win_rate_scaling = True
    calculator.kelly_min_trades = 10

    signal = TradingSignal(
      symbol="BTCUSDT",
      signal_type=SignalType.BUY,
      strength=SignalStrength.MEDIUM,
      confidence=0.7,
      price=50000.0,
      source=SignalSource.STRATEGY  ,
      timestamp=int(datetime.now().timestamp() * 1000)
    )

    # 1. Первый расчет (без истории)
    risk_params_1 = calculator.calculate(
      signal=signal,
      balance=10000.0,
      stop_loss_price=49000.0
    )

    assert risk_params_1.final_risk_percent == calculator.base_percent

    # 2. Записываем историю успешных трейдов
    for _ in range(15):
      calculator.record_trade(True, 200.0)
    for _ in range(5):
      calculator.record_trade(False, -100.0)

    # 3. Второй расчет (с хорошей историей - 75% win rate)
    risk_params_2 = calculator.calculate(
      signal=signal,
      balance=10000.0,
      stop_loss_price=49000.0
    )

    # Риск должен увеличиться из-за high win rate
    assert risk_params_2.final_risk_percent > risk_params_1.final_risk_percent

    # 4. Проверяем статистику
    stats = calculator.get_statistics()
    assert stats['win_rate'] == 0.75
    assert stats['total_trades'] == 20