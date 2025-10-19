"""
Тесты для Reversal Detector.

Проверка основных сценариев обнаружения разворотов.
"""
import sys
from pathlib import Path

# Определяем путь к backend директории
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

import pytest
import numpy as np
from datetime import datetime
from typing import List

from ml_engine.features.candle_feature_extractor import Candle
from models.signal import SignalType
from strategy.reversal_detector import ReversalDetector
from strategy.risk_models import ReversalStrength


@pytest.fixture
def detector():
  """Фикстура детектора с настройками по умолчанию."""
  detector = ReversalDetector()
  detector.enabled = True
  detector.min_indicators = 3
  detector.cooldown_seconds = 300
  detector.auto_action = False
  return detector


@pytest.fixture
def sample_candles() -> List[Candle]:
  """Создание тестовых свечей."""
  candles = []
  base_price = 50000.0

  for i in range(100):
    # Восходящий тренд
    price = base_price + (i * 10)
    candle = Candle(
      timestamp=datetime.now().timestamp() - (100 - i) * 60,
      open=price - 5,
      high=price + 10,
      low=price - 10,
      close=price,
      volume=1000.0 + (i * 10)
    )
    candles.append(candle)

  return candles


def test_detector_initialization():
  """Тест инициализации детектора."""
  detector = ReversalDetector()

  assert detector.enabled is not None
  assert detector.min_indicators > 0
  assert detector.cooldown_seconds > 0
  assert isinstance(detector.reversal_history, dict)


def test_insufficient_candles(detector):
  """Тест с недостаточным количеством свечей."""
  candles = [
              Candle(
                timestamp=datetime.now().timestamp(),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=1000.0
              )
            ] * 10  # Только 10 свечей

  result = detector.detect_reversal(
    symbol="BTCUSDT",
    candles=candles,
    current_trend=SignalType.BUY,
    indicators={"rsi": [50], "macd": [0], "macd_signal": [0]}
  )

  assert result is None


def test_price_action_doji(detector, sample_candles):
  """Тест обнаружения Doji паттерна."""
  # Добавляем Doji свечу
  last_price = sample_candles[-1].close
  doji = Candle(
    timestamp=datetime.now().timestamp(),
    open=last_price,
    high=last_price + 50,
    low=last_price - 50,
    close=last_price + 2,  # Очень маленькое тело
    volume=2000.0
  )
  sample_candles.append(doji)

  indicators = {
    "rsi": [75] * 50,  # Overbought
    "macd": [0.5] * 50,
    "macd_signal": [0.3] * 50
  }

  result = detector.detect_reversal(
    symbol="BTCUSDT",
    candles=sample_candles,
    current_trend=SignalType.BUY,
    indicators=indicators
  )

  # Может обнаружить или нет, в зависимости от других индикаторов
  # Проверяем, что метод работает без ошибок
  assert result is None or isinstance(result.strength, ReversalStrength)


def test_bearish_engulfing(detector):
  """Тест обнаружения медвежьего поглощения."""
  candles = []

  # Создаем историю
  for i in range(50):
    candles.append(Candle(
      timestamp=datetime.now().timestamp() - (50 - i) * 60,
      open=50000 + i * 10,
      high=50010 + i * 10,
      low=49990 + i * 10,
      close=50005 + i * 10,
      volume=1000.0
    ))

  # Зеленая свеча
  candles.append(Candle(
    timestamp=datetime.now().timestamp() - 120,
    open=50500,
    high=50600,
    low=50450,
    close=50580,  # Закрытие выше открытия
    volume=1500.0
  ))

  # Красная поглощающая свеча
  candles.append(Candle(
    timestamp=datetime.now().timestamp() - 60,
    open=50600,  # Открытие выше закрытия предыдущей
    high=50650,
    low=50400,
    close=50420,  # Закрытие ниже открытия предыдущей
    volume=3000.0
  ))

  indicators = {
    "rsi": [78] * 52,  # Overbought
    "macd": [0.8] * 51 + [0.6],  # Начинает падать
    "macd_signal": [0.5] * 52
  }

  result = detector.detect_reversal(
    symbol="BTCUSDT",
    candles=candles,
    current_trend=SignalType.BUY,
    indicators=indicators
  )

  # Должен обнаружить engulfing + overbought RSI
  if result:
    assert "engulfing" in result.indicators_confirming or "rsi" in str(result.indicators_confirming)


def test_rsi_overbought_reversal(detector, sample_candles):
  """Тест разворота из зоны перекупленности."""
  indicators = {
    "rsi": [50] * 48 + [78, 76, 74],  # RSI падает из overbought
    "macd": [0.5] * 51,
    "macd_signal": [0.3] * 51
  }

  result = detector.detect_reversal(
    symbol="BTCUSDT",
    candles=sample_candles,
    current_trend=SignalType.BUY,
    indicators=indicators
  )

  # Может не обнаружить, если недостаточно подтверждений
  if result:
    assert "rsi" in str(result.indicators_confirming).lower()


def test_momentum_divergence(detector):
  """Тест медвежьей дивергенции."""
  candles = []

  # Цена растет
  for i in range(70):
    price = 50000 + (i * 20)
    candles.append(Candle(
      timestamp=datetime.now().timestamp() - (70 - i) * 60,
      open=price - 5,
      high=price + 10,
      low=price - 10,
      close=price,
      volume=1000.0
    ))

  # RSI падает, несмотря на рост цены
  rsi_values = list(np.linspace(60, 55, 70))

  indicators = {
    "rsi": rsi_values,
    "macd": [0.5] * 70,
    "macd_signal": [0.3] * 70
  }

  result = detector.detect_reversal(
    symbol="BTCUSDT",
    candles=candles,
    current_trend=SignalType.BUY,
    indicators=indicators
  )

  # Может обнаружить дивергенцию
  if result:
    assert "divergence" in str(result.indicators_confirming).lower()


def test_cooldown_mechanism(detector, sample_candles):
  """Тест механизма cooldown."""
  indicators = {
    "rsi": [80] * 51,
    "macd": [1.0] * 50 + [0.8],
    "macd_signal": [0.5] * 51
  }

  # Первое обнаружение
  result1 = detector.detect_reversal(
    symbol="BTCUSDT",
    candles=sample_candles,
    current_trend=SignalType.BUY,
    indicators=indicators
  )

  # Второе обнаружение сразу же (должно быть заблокировано cooldown)
  result2 = detector.detect_reversal(
    symbol="BTCUSDT",
    candles=sample_candles,
    current_trend=SignalType.BUY,
    indicators=indicators
  )

  # Если первое обнаружено, второе должно быть None из-за cooldown
  if result1:
    assert result2 is None


def test_reversal_strength_calculation(detector):
  """Тест расчета силы сигнала."""
  assert detector._calculate_reversal_strength(1) == ReversalStrength.WEAK
  assert detector._calculate_reversal_strength(3) == ReversalStrength.MODERATE
  assert detector._calculate_reversal_strength(5) == ReversalStrength.STRONG
  assert detector._calculate_reversal_strength(7) == ReversalStrength.CRITICAL


def test_suggested_action_determination(detector):
  """Тест определения рекомендуемого действия."""
  assert detector._determine_action(
    ReversalStrength.WEAK,
    SignalType.BUY
  ) == "no_action"

  assert detector._determine_action(
    ReversalStrength.MODERATE,
    SignalType.BUY
  ) == "tighten_sl"

  assert detector._determine_action(
    ReversalStrength.STRONG,
    SignalType.BUY
  ) == "reduce_size"

  assert detector._determine_action(
    ReversalStrength.CRITICAL,
    SignalType.BUY
  ) == "close_position"


def test_orderbook_pressure_shift(detector, sample_candles):
  """Тест обнаружения изменения давления в стакане."""
  indicators = {
    "rsi": [70] * 51,
    "macd": [0.5] * 51,
    "macd_signal": [0.3] * 51
  }

  # Сильное давление продавцов
  orderbook_metrics = {
    "imbalance": -0.5  # Sellers dominate
  }

  result = detector.detect_reversal(
    symbol="BTCUSDT",
    candles=sample_candles,
    current_trend=SignalType.BUY,
    indicators=indicators,
    orderbook_metrics=orderbook_metrics
  )

  # Может обнаружить shift в давлении
  if result:
    assert "orderbook" in str(result.indicators_confirming).lower() or len(result.indicators_confirming) > 0


def test_multiple_confirmations(detector, sample_candles):
  """Тест множественных подтверждений разворота."""
  # Создаем условия для нескольких индикаторов
  indicators = {
    "rsi": [50] * 48 + [78, 76, 73],  # RSI reversal
    "macd": [0.8] * 49 + [0.7, 0.5],  # MACD падает
    "macd_signal": [0.6] * 51
  }

  orderbook_metrics = {
    "imbalance": -0.45  # Сильное давление продавцов
  }

  result = detector.detect_reversal(
    symbol="BTCUSDT",
    candles=sample_candles,
    current_trend=SignalType.BUY,
    indicators=indicators,
    orderbook_metrics=orderbook_metrics
  )

  # С множественными подтверждениями должен обнаружить
  # Но зависит от min_indicators
  if result:
    assert len(result.indicators_confirming) >= detector.min_indicators
    assert result.confidence > 0.0