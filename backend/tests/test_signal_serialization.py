"""
Тесты JSON сериализации TradingSignal.
"""

import sys
from pathlib import Path

# Определяем путь к backend директории
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))
import json
import pytest
from datetime import datetime

from backend.models.signal import TradingSignal, SignalType, SignalStrength, SignalSource


def test_trading_signal_to_dict():
  """Тест метода to_dict()"""
  signal = TradingSignal(
    symbol="BTCUSDT",
    signal_type=SignalType.BUY,
    strength=SignalStrength.STRONG,
    source=SignalSource.ML_VALIDATED,
    timestamp=int(datetime.now().timestamp() * 1000),
    price=50000.0,
    confidence=0.85,
    imbalance=0.65,
    reason="Strong buy signal",
    metadata={"test": "data"}
  )

  # Конвертируем в dict
  signal_dict = signal.to_dict()

  # Проверяем тип
  assert isinstance(signal_dict, dict)

  # Проверяем что Enum сконвертированы в строки
  assert isinstance(signal_dict["signal_type"], str)
  assert signal_dict["signal_type"] == "BUY"

  assert isinstance(signal_dict["strength"], str)
  assert signal_dict["strength"] == "STRONG"

  assert isinstance(signal_dict["source"], str)
  assert signal_dict["source"] == "ml_validated"

  print(f"✓ Signal dict: {signal_dict}")


def test_trading_signal_json_serialization():
  """Тест JSON сериализации"""
  signal = TradingSignal(
    symbol="ETHUSDT",
    signal_type=SignalType.SELL,
    strength=SignalStrength.MEDIUM,
    source=SignalSource.IMBALANCE,
    timestamp=int(datetime.now().timestamp() * 1000),
    price=3000.0,
    confidence=0.72
  )

  # Конвертируем в dict
  signal_dict = signal.to_dict()

  # Пробуем сериализовать в JSON
  try:
    json_str = json.dumps(signal_dict)
    print(f"✓ JSON serialization successful: {len(json_str)} bytes")

    # Десериализуем обратно
    parsed = json.loads(json_str)
    assert parsed["symbol"] == "ETHUSDT"
    assert parsed["signal_type"] == "SELL"

  except TypeError as e:
    pytest.fail(f"JSON serialization failed: {e}")


def test_trading_signal_with_none_values():
  """Тест сериализации с None значениями"""
  signal = TradingSignal(
    symbol="ADAUSDT",
    signal_type=SignalType.BUY,
    strength=SignalStrength.WEAK,
    source=SignalSource.CLUSTER,
    timestamp=int(datetime.now().timestamp() * 1000),
    price=0.45,
    confidence=0.55,
    imbalance=None,  # None значение
    volume_delta=None,
    cluster_info=None
  )

  signal_dict = signal.to_dict()

  # Проверяем что None корректно обрабатывается
  assert signal_dict["metrics"]["imbalance"] is None
  assert signal_dict["metrics"]["volume_delta"] is None

  # JSON сериализация должна работать
  try:
    json_str = json.dumps(signal_dict)
    print(f"✓ JSON with None values: {len(json_str)} bytes")
  except TypeError as e:
    pytest.fail(f"JSON serialization with None failed: {e}")


if __name__ == "__main__":
  test_trading_signal_to_dict()
  test_trading_signal_json_serialization()
  test_trading_signal_with_none_values()
  print("\n✅ Все тесты пройдены!")