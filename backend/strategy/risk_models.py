"""
Модели данных для расширенного Risk Manager.

Путь: backend/strategy/risk_models.py
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
from datetime import datetime
from enum import Enum


class MarketRegime(Enum):
  """Режимы рынка для адаптации риск-параметров."""
  STRONG_TREND = "strong_trend"
  MILD_TREND = "mild_trend"
  RANGING = "ranging"
  HIGH_VOLATILITY = "high_volatility"
  DISTRIBUTION = "distribution"  # Крупные продавцы
  ACCUMULATION = "accumulation"  # Крупные покупатели


class ReversalStrength(Enum):
  """Сила сигнала разворота тренда."""
  WEAK = "weak"
  MODERATE = "moderate"
  STRONG = "strong"
  CRITICAL = "critical"


@dataclass
class SLTPCalculation:
  """
  Результат расчета Stop Loss и Take Profit.

  Attributes:
      stop_loss: Цена stop loss
      take_profit: Цена take profit
      risk_reward_ratio: Соотношение риск/прибыль
      trailing_start_profit: При какой прибыли (%) начинать trailing
      calculation_method: Метод расчета (ml/atr/fixed)
      reasoning: Детали расчета для логирования
      confidence: Уверенность в расчете (0.0-1.0)
  """
  stop_loss: float
  take_profit: float
  risk_reward_ratio: float
  trailing_start_profit: float
  calculation_method: Literal["ml", "atr", "fixed"]
  reasoning: Dict
  confidence: float

  def __post_init__(self):
    """Валидация после создания."""
    assert 0.0 <= self.confidence <= 1.0, "Confidence должен быть в [0, 1]"
    assert self.risk_reward_ratio > 0, "R/R ratio должен быть положительным"


@dataclass
class CorrelationGroup:
  """
  Группа коррелирующих символов для управления exposure.

  Attributes:
      group_id: Уникальный ID группы
      symbols: Список символов в группе
      avg_correlation: Средняя корреляция между символами
      active_positions: Количество открытых позиций
      total_exposure_usdt: Общая экспозиция группы в USDT
  """
  group_id: str
  symbols: List[str]
  avg_correlation: float
  active_positions: int
  total_exposure_usdt: float


@dataclass
class DailyLossMetrics:
  """
  Метрики дневного убытка для Daily Loss Killer.

  Attributes:
      starting_balance: Баланс на начало дня
      current_balance: Текущий баланс
      daily_pnl: Прибыль/убыток за день
      daily_loss_percent: Процент убытка от стартового баланса
      max_daily_loss_percent: Максимально допустимый убыток (из конфига)
      is_critical: Критический ли уровень убытка
      time_to_reset: Время до сброса счетчика (UTC midnight)
  """
  starting_balance: float
  current_balance: float
  daily_pnl: float
  daily_loss_percent: float
  max_daily_loss_percent: float
  is_critical: bool
  time_to_reset: datetime


@dataclass
class RiskPerTradeParams:
  """
  Параметры риска на одну сделку (Adaptive).

  Attributes:
      base_risk_percent: Базовый % риска от баланса
      kelly_fraction: Kelly Criterion fraction
      volatility_adjustment: Корректировка на волатильность
      correlation_adjustment: Корректировка на корреляцию позиций
      final_risk_percent: Итоговый % риска на сделку
      max_position_usdt: Максимальный размер позиции в USDT
  """
  base_risk_percent: float
  kelly_fraction: float
  volatility_adjustment: float
  correlation_adjustment: float
  final_risk_percent: float
  max_position_usdt: float


@dataclass
class ReversalSignal:
  """
  Сигнал разворота тренда для защиты позиций.

  Attributes:
      symbol: Торговая пара
      detected_at: Время обнаружения
      strength: Сила разворота
      indicators_confirming: Список подтверждающих индикаторов
      confidence: Уверенность в развороте (0.0-1.0)
      suggested_action: Рекомендуемое действие
      reason: Причина обнаружения разворота
  """
  symbol: str
  detected_at: datetime
  strength: ReversalStrength
  indicators_confirming: List[str]
  confidence: float
  suggested_action: Literal["close_position", "reduce_size", "tighten_sl", "no_action"]
  reason: str


@dataclass
class TrailingStopState:
  """
  Состояние trailing stop для активной позиции.

  Attributes:
      position_id: ID позиции
      symbol: Торговая пара
      entry_price: Цена входа
      current_price: Текущая цена
      highest_price: Максимальная цена для long
      lowest_price: Минимальная цена для short
      current_stop_loss: Текущий уровень stop loss
      trailing_distance_percent: Дистанция trailing в %
      activation_profit_percent: При какой прибыли активировать
      is_active: Активен ли trailing
      updated_at: Время последнего обновления
  """
  position_id: str
  symbol: str
  entry_price: float
  current_price: float
  highest_price: float
  lowest_price: float
  current_stop_loss: float
  trailing_distance_percent: float
  activation_profit_percent: float
  is_active: bool
  updated_at: datetime


@dataclass
class MLRiskAdjustments:
  """
  ML-based корректировки риска для позиции.

  Attributes:
      position_size_multiplier: Множитель размера (0.5x - 2.5x)
      stop_loss_price: ML-предсказанный stop loss
      take_profit_price: ML-предсказанный take profit
      ml_confidence: Уверенность ML модели
      expected_return: Ожидаемая доходность
      market_regime: Определенный режим рынка
      manipulation_risk_score: Оценка риска манипуляций (0-1)
      feature_quality: Качество признаков (0-1)
      allow_entry: Разрешить ли вход в позицию
      rejection_reason: Причина отказа (если allow_entry=False)
  """
  position_size_multiplier: float
  stop_loss_price: float
  take_profit_price: float
  ml_confidence: float
  expected_return: float
  market_regime: MarketRegime
  manipulation_risk_score: float
  feature_quality: float
  allow_entry: bool
  rejection_reason: Optional[str] = None

  def __post_init__(self):
    """Валидация после создания."""
    assert 0.5 <= self.position_size_multiplier <= 2.5, \
      "Position size multiplier должен быть в [0.5, 2.5]"
    assert 0.0 <= self.ml_confidence <= 1.0, \
      "ML confidence должен быть в [0, 1]"
    assert 0.0 <= self.manipulation_risk_score <= 1.0, \
      "Manipulation risk должен быть в [0, 1]"
    assert 0.0 <= self.feature_quality <= 1.0, \
      "Feature quality должен быть в [0, 1]"