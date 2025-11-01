"""
Тесты для UnifiedSLTPCalculator.

Покрытие:
- ML-based расчет
- ATR-based fallback
- Fixed fallback
- Валидации и ограничения
- Корректировки на market regime

Путь: backend/tests/test_sltp_calculator.py
"""
import sys
from pathlib import Path

# Определяем путь к backend директории
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

import pytest
from datetime import datetime

from backend.strategy.sltp_calculator import UnifiedSLTPCalculator
from backend.strategy.risk_models import MarketRegime
from backend.models.signal import TradingSignal, SignalType, SignalSource, SignalStrength


class TestUnifiedSLTPCalculator:
  """Тесты для SL/TP калькулятора."""

  def setup_method(self):
    """Setup перед каждым тестом."""
    self.calculator = UnifiedSLTPCalculator()
    self.entry_price = 50000.0

    # Создаем тестовый сигнал
    self.signal = TradingSignal(
      symbol="BTCUSDT",
      signal_type=SignalType.BUY,
      source=SignalSource.STRATEGY,
      strength=SignalStrength.STRONG,
      price=self.entry_price,
      confidence=0.8,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason="Test signal"
    )

  def test_ml_based_calculation_long(self):
    """Тест ML-based расчета для long позиции."""
    ml_result = {
      'predicted_mae': 0.012,  # -1.2% ожидаемый максимальный откат
      'predicted_return': 0.028,  # +2.8% ожидаемая прибыль
      'confidence': 0.85
    }

    result = self.calculator.calculate(
      signal=self.signal,
      entry_price=self.entry_price,
      ml_result=ml_result
    )

    # Проверки
    assert result.calculation_method == "ml"
    assert result.stop_loss < self.entry_price, "SL должен быть ниже entry для long"
    assert result.take_profit > self.entry_price, "TP должен быть выше entry для long"
    assert result.risk_reward_ratio >= 2.0, f"Min R/R должен быть 2.0, получен {result.risk_reward_ratio}"

    # Проверка max SL (3%)
    sl_percent = abs((self.entry_price - result.stop_loss) / self.entry_price)
    assert sl_percent <= 0.03, f"SL превышает max 3%: {sl_percent:.2%}"

    # Проверка reasoning
    assert 'predicted_mae' in result.reasoning
    assert 'confidence' in result.reasoning
    assert result.confidence == 0.85

  def test_ml_based_calculation_short(self):
    """Тест ML-based расчета для short позиции."""
    # Создаем short сигнал
    short_signal = TradingSignal(
      symbol="BTCUSDT",
      signal_type=SignalType.SELL,
      source=SignalSource.STRATEGY,
      strength=SignalStrength.STRONG,
      price=self.entry_price,
      confidence=0.8,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason="Test short signal"
    )

    ml_result = {
      'predicted_mae': 0.015,
      'predicted_return': 0.03,
      'confidence': 0.8
    }

    result = self.calculator.calculate(
      signal=short_signal,
      entry_price=self.entry_price,
      ml_result=ml_result
    )

    assert result.calculation_method == "ml"
    assert result.stop_loss > self.entry_price, "SL должен быть выше entry для short"
    assert result.take_profit < self.entry_price, "TP должен быть ниже entry для short"
    assert result.risk_reward_ratio >= 2.0

  def test_atr_based_fallback(self):
    """Тест ATR fallback при отсутствии ML."""
    atr = 750.0  # ATR = $750

    result = self.calculator.calculate(
      signal=self.signal,
      entry_price=self.entry_price,
      atr=atr
    )

    assert result.calculation_method == "atr"
    assert result.confidence == 0.7, "ATR должен иметь среднюю уверенность"
    assert 'atr' in result.reasoning
    assert result.reasoning['atr'] == atr

  def test_fixed_fallback(self):
    """Тест fixed fallback (emergency)."""
    result = self.calculator.calculate(
      signal=self.signal,
      entry_price=self.entry_price
    )

    assert result.calculation_method == "fixed"
    assert result.risk_reward_ratio == 2.0, "Fixed должен использовать min R/R"
    assert result.confidence == 0.5, "Fixed имеет низкую уверенность"
    assert 'note' in result.reasoning
    assert result.reasoning['note'] == "Emergency fallback"

  def test_max_sl_enforcement(self):
    """Тест ограничения max SL."""
    ml_result = {
      'predicted_mae': 0.05,  # 5% - больше max 3%
      'predicted_return': 0.10,
      'confidence': 0.9
    }

    result = self.calculator.calculate(
      signal=self.signal,
      entry_price=self.entry_price,
      ml_result=ml_result
    )

    sl_percent = abs((self.entry_price - result.stop_loss) / self.entry_price)
    assert sl_percent <= 0.03, f"SL должен быть capped at 3%, получен {sl_percent:.2%}"

  def test_min_rr_enforcement(self):
    """Тест соблюдения минимального R/R."""
    ml_result = {
      'predicted_mae': 0.02,  # 2%
      'predicted_return': 0.03,  # 3% - даст R/R = 1.5, меньше min
      'confidence': 0.75
    }

    result = self.calculator.calculate(
      signal=self.signal,
      entry_price=self.entry_price,
      ml_result=ml_result
    )

    assert result.risk_reward_ratio >= 2.0, \
      f"R/R должен быть >= 2.0, получен {result.risk_reward_ratio}"

  def test_regime_adjustment_trending(self):
    """Тест корректировки для trending market."""
    ml_result = {
      'predicted_mae': 0.012,
      'predicted_return': 0.025,
      'confidence': 0.8
    }

    result = self.calculator.calculate(
      signal=self.signal,
      entry_price=self.entry_price,
      ml_result=ml_result,
      market_regime=MarketRegime.STRONG_TREND
    )

    # В trending должны быть шире stops и дальше targets
    assert 'regime_adjustment' in result.reasoning
    assert result.reasoning['regime_adjustment']['sl_mult'] > 1.0, \
      "В тренде SL должен быть шире"
    assert result.reasoning['regime_adjustment']['tp_mult'] > 1.0, \
      "В тренде TP должен быть дальше"

  def test_regime_adjustment_ranging(self):
    """Тест корректировки для ranging market."""
    ml_result = {
      'predicted_mae': 0.012,
      'predicted_return': 0.025,
      'confidence': 0.8
    }

    result = self.calculator.calculate(
      signal=self.signal,
      entry_price=self.entry_price,
      ml_result=ml_result,
      market_regime=MarketRegime.RANGING
    )

    # В ranging должны быть уже stops и closer targets
    assert result.reasoning['regime_adjustment']['sl_mult'] < 1.0, \
      "В ranging SL должен быть уже"
    assert result.reasoning['regime_adjustment']['tp_mult'] < 1.0, \
      "В ranging TP должен быть ближе"

  def test_high_confidence_multiplier(self):
    """Тест множителя при высокой уверенности."""
    ml_result_high_conf = {
      'predicted_mae': 0.012,
      'predicted_return': 0.025,
      'confidence': 0.95  # Очень высокая уверенность
    }

    result_high = self.calculator.calculate(
      signal=self.signal,
      entry_price=self.entry_price,
      ml_result=ml_result_high_conf
    )

    ml_result_low_conf = {
      'predicted_mae': 0.012,
      'predicted_return': 0.025,
      'confidence': 0.65  # Низкая уверенность
    }

    result_low = self.calculator.calculate(
      signal=self.signal,
      entry_price=self.entry_price,
      ml_result=ml_result_low_conf
    )

    # При высокой уверенности TP должен быть дальше
    assert abs(result_high.take_profit - self.entry_price) > \
           abs(result_low.take_profit - self.entry_price), \
      "Высокая уверенность должна давать больший TP"

  def test_invalid_ml_result(self):
    """Тест обработки невалидного ML результата."""
    invalid_ml_result = {
      'confidence': 0.8
      # Отсутствуют predicted_mae и predicted_return
    }

    # Должен использовать fallback
    result = self.calculator.calculate(
      signal=self.signal,
      entry_price=self.entry_price,
      ml_result=invalid_ml_result,
      atr=750.0
    )

    assert result.calculation_method in ["atr", "fixed"], \
      "При невалидном ML должен использовать fallback"

  def test_slippage_buffer_included(self):
    """Тест что slippage buffer включен в SL."""
    ml_result = {
      'predicted_mae': 0.01,  # 1%
      'predicted_return': 0.03,
      'confidence': 0.8
    }

    result = self.calculator.calculate(
      signal=self.signal,
      entry_price=self.entry_price,
      ml_result=ml_result
    )

    # Slippage buffer = 0.2%, поэтому SL должен быть > 1%
    sl_percent = abs((self.entry_price - result.stop_loss) / self.entry_price)
    assert sl_percent > 0.01, "SL должен включать slippage buffer"
    assert sl_percent <= 0.015, "SL не должен превышать mae + buffer значительно"

  def test_trailing_start_calculation(self):
    """Тест расчета trailing start."""
    ml_result = {
      'predicted_mae': 0.012,
      'predicted_return': 0.03,
      'confidence': 0.8
    }

    result = self.calculator.calculate(
      signal=self.signal,
      entry_price=self.entry_price,
      ml_result=ml_result
    )

    # Trailing должен начинаться при 60% пути к TP
    assert 0 < result.trailing_start_profit < 0.03, \
      f"Trailing start должен быть между 0 и TP: {result.trailing_start_profit}"


# Тесты для граничных случаев
class TestSLTPCalculatorEdgeCases:
  """Тесты граничных случаев."""

  def setup_method(self):
    """Setup."""
    self.calculator = UnifiedSLTPCalculator()

  def test_zero_atr(self):
    """Тест с нулевым ATR (не должно падать)."""
    signal = TradingSignal(
      symbol="BTCUSDT",
      signal_type=SignalType.BUY,
      source=SignalSource.STRATEGY,
      strength=SignalStrength.MEDIUM,
      price=50000.0,
      confidence=0.7,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason="Test"
    )

    # С нулевым ATR должен использовать fixed fallback
    result = self.calculator.calculate(
      signal=signal,
      entry_price=50000.0,
      atr=0.0
    )

    assert result.calculation_method == "fixed"

  def test_very_low_price(self):
    """Тест с очень низкой ценой."""
    signal = TradingSignal(
      symbol="PEPEUSDT",
      signal_type=SignalType.BUY,
      source=SignalSource.STRATEGY,
      strength=SignalStrength.WEAK,
      price=0.000001,
      confidence=0.6,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason="Low price test"
    )

    result = self.calculator.calculate(
      signal=signal,
      entry_price=0.000001
    )

    # Должен работать даже с микро-ценами
    assert result.stop_loss > 0
    assert result.take_profit > result.stop_loss

  def test_all_fallbacks_fail_gracefully(self):
    """Тест что при любых условиях возвращается валидный результат."""
    signal = TradingSignal(
      symbol="BTCUSDT",
      signal_type=SignalType.BUY,
      source=SignalSource.STRATEGY,
      strength=SignalStrength.WEAK,
      price=50000.0,
      confidence=0.5,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason="Ultimate fallback test"
    )

    # Без ML, без ATR - должен вернуть fixed
    result = self.calculator.calculate(
      signal=signal,
      entry_price=50000.0
    )

    assert result is not None
    assert result.calculation_method == "fixed"
    assert result.stop_loss != result.take_profit
    assert result.risk_reward_ratio > 0