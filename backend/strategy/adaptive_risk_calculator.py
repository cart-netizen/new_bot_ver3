"""
Адаптивный расчет риска на сделку.

РЕЖИМЫ:
1. Fixed - Фиксированный %
2. Adaptive - Динамический с корректировками
3. Kelly Criterion - На основе win rate и payoff ratio

КОРРЕКТИРОВКИ:
- Volatility adjustment (inverse scaling)
- Win rate adjustment
- Correlation penalty
- ML confidence boost

Путь: backend/strategy/adaptive_risk_calculator.py
"""
from typing import Optional
import numpy as np

from core.logger import get_logger
from config import settings
from models.signal import TradingSignal
from strategy.risk_models import RiskPerTradeParams

logger = get_logger(__name__)


class AdaptiveRiskCalculator:
  """
  Расчет адаптивного риска на сделку.

  Поддерживает три режима:
  - Fixed: Фиксированный процент
  - Adaptive: Динамический с множественными корректировками
  - Kelly: Kelly Criterion с fractional Kelly для консерватизма
  """

  def __init__(self):
    """Инициализация."""
    self.mode = settings.RISK_PER_TRADE_MODE
    self.base_percent = settings.RISK_PER_TRADE_BASE_PERCENT / 100  # 2% -> 0.02
    self.max_percent = settings.RISK_PER_TRADE_MAX_PERCENT / 100  # 5% -> 0.05

    # Kelly Criterion параметры
    self.kelly_fraction = settings.RISK_KELLY_FRACTION
    self.kelly_min_trades = settings.RISK_KELLY_MIN_TRADES

    # Volatility scaling
    self.volatility_scaling = settings.RISK_VOLATILITY_SCALING
    self.volatility_baseline = settings.RISK_VOLATILITY_BASELINE

    # Win rate scaling
    self.win_rate_scaling = settings.RISK_WIN_RATE_SCALING
    self.win_rate_baseline = settings.RISK_WIN_RATE_BASELINE

    # Correlation penalty
    self.correlation_penalty = settings.RISK_CORRELATION_PENALTY

    # История трейдов для Kelly и Adaptive
    self.trade_history = []  # [(win: bool, pnl: float)]

    logger.info(
      f"AdaptiveRiskCalculator initialized: "
      f"mode={self.mode}, "
      f"base={self.base_percent:.1%}, "
      f"max={self.max_percent:.1%}"
    )

  def calculate(
      self,
      signal: TradingSignal,
      balance: float,
      stop_loss_price: float,
      current_volatility: Optional[float] = None,
      correlation_factor: Optional[float] = None,
      ml_confidence: Optional[float] = None
  ) -> RiskPerTradeParams:
    """
    Расчет параметров риска для сделки.

    Args:
        signal: Торговый сигнал
        balance: Доступный баланс
        stop_loss_price: Цена stop loss
        current_volatility: Текущая волатильность (опционально)
        correlation_factor: Фактор корреляции 0.0-1.0 (опционально)
        ml_confidence: ML уверенность 0.0-1.0 (опционально)

    Returns:
        RiskPerTradeParams: Параметры риска
    """
    # Выбираем метод расчета базового риска
    if self.mode == "fixed":
      risk_percent = self._calculate_fixed()

    elif self.mode == "kelly":
      risk_percent = self._calculate_kelly()

    else:  # adaptive (default)
      risk_percent = self._calculate_adaptive(
        signal,
        current_volatility,
        correlation_factor,
        ml_confidence
      )

    # Ограничиваем максимумом
    risk_percent = min(risk_percent, self.max_percent)

    # Рассчитываем размер позиции на основе risk и stop loss distance
    entry_price = signal.price
    risk_per_unit = abs(entry_price - stop_loss_price)

    if risk_per_unit > 0:
      # Максимальный риск в USDT
      max_risk_usdt = balance * risk_percent

      # Количество по формуле: max_risk / риск_на_единицу
      max_quantity = max_risk_usdt / risk_per_unit

      # Размер позиции в USDT
      max_position_usdt = max_quantity * entry_price
    else:
      # Fallback если stop loss == entry (не должно случаться)
      logger.warning(
        f"{signal.symbol} | Stop loss == entry price, "
        f"using fallback position sizing"
      )
      max_position_usdt = balance * risk_percent

    # Не превышаем максимальный процент от баланса
    max_position_usdt = min(max_position_usdt, balance * self.max_percent)

    logger.debug(
      f"{signal.symbol} | Risk calculation: "
      f"final_risk={risk_percent:.2%}, "
      f"position=${max_position_usdt:.2f}"
    )

    return RiskPerTradeParams(
      base_risk_percent=self.base_percent,
      kelly_fraction=self.kelly_fraction if self.mode == "kelly" else 0.0,
      volatility_adjustment=self._get_volatility_adj(current_volatility) if current_volatility else 1.0,
      correlation_adjustment=correlation_factor if correlation_factor else 1.0,
      final_risk_percent=risk_percent,
      max_position_usdt=max_position_usdt
    )

  def _calculate_fixed(self) -> float:
    """Fixed риск - всегда базовый процент."""
    return self.base_percent

  def _calculate_kelly(self) -> float:
    """
    Kelly Criterion расчет.

    Formula: f = (p * b - q) / b
    где:
        f = fraction of capital to bet
        p = probability of win (win rate)
        q = probability of loss (1 - p)
        b = ratio of win to loss (avg_win / avg_loss)
    """
    if len(self.trade_history) < self.kelly_min_trades:
      logger.debug(
        f"Insufficient trades for Kelly: "
        f"{len(self.trade_history)} < {self.kelly_min_trades}, "
        f"using base_percent"
      )
      return self.base_percent

    # Считаем win rate
    wins = [t for t in self.trade_history if t[0]]
    losses = [t for t in self.trade_history if not t[0]]

    if not wins or not losses:
      logger.debug("No wins or losses in history, using base_percent")
      return self.base_percent

    win_rate = len(wins) / len(self.trade_history)

    # Считаем avg win / avg loss
    avg_win = np.mean([t[1] for t in wins])
    avg_loss = abs(np.mean([t[1] for t in losses]))

    if avg_loss == 0:
      logger.warning("Average loss is 0, cannot calculate Kelly")
      return self.base_percent

    payoff_ratio = avg_win / avg_loss

    # Kelly formula
    kelly_fraction_full = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio

    # Применяем fractional Kelly (консервативный подход)
    kelly_fraction_adj = kelly_fraction_full * self.kelly_fraction

    # Ограничиваем снизу (минимум 50% от base)
    kelly_fraction_adj = max(kelly_fraction_adj, self.base_percent * 0.5)

    logger.debug(
      f"Kelly calculation: "
      f"win_rate={win_rate:.2%}, "
      f"payoff_ratio={payoff_ratio:.2f}, "
      f"full_kelly={kelly_fraction_full:.2%}, "
      f"fractional={kelly_fraction_adj:.2%}"
    )

    return kelly_fraction_adj

  def _calculate_adaptive(
      self,
      signal: TradingSignal,
      current_volatility: Optional[float],
      correlation_factor: Optional[float],
      ml_confidence: Optional[float]
  ) -> float:
    """
    Adaptive расчет с множественными корректировками.

    Базовый риск корректируется на основе:
    1. Текущей волатильности (inverse scaling)
    2. Win rate истории
    3. Корреляции с открытыми позициями
    4. ML confidence
    """
    risk = self.base_percent

    logger.debug(f"Adaptive risk base: {risk:.2%}")

    # Корректировка 1: Volatility (inverse scaling)
    if self.volatility_scaling and current_volatility is not None:
      vol_adj = self._get_volatility_adj(current_volatility)
      risk *= vol_adj
      logger.debug(f"  → After volatility adj ({vol_adj:.2f}x): {risk:.2%}")

    # Корректировка 2: Win Rate (если есть достаточно истории)
    if self.win_rate_scaling and len(self.trade_history) >= 10:
      win_rate_adj = self._get_win_rate_adj()
      risk *= win_rate_adj
      logger.debug(f"  → After win rate adj ({win_rate_adj:.2f}x): {risk:.2%}")

    # Корректировка 3: Correlation penalty
    if self.correlation_penalty and correlation_factor is not None:
      risk *= correlation_factor
      logger.debug(f"  → After correlation adj ({correlation_factor:.2f}x): {risk:.2%}")

    # Корректировка 4: ML Confidence boost
    if ml_confidence is not None:
      ml_adj = self._get_ml_confidence_adj(ml_confidence)
      risk *= ml_adj
      logger.debug(f"  → After ML confidence adj ({ml_adj:.2f}x): {risk:.2%}")

    return risk

  def _get_volatility_adj(self, current_volatility: float) -> float:
    """
    Корректировка на волатильность (inverse scaling).

    Логика:
    - High volatility → Reduce risk (adj < 1.0)
    - Low volatility → Increase risk (adj > 1.0)

    Args:
        current_volatility: Текущая волатильность

    Returns:
        float: Множитель корректировки [0.5, 1.5]
    """
    if current_volatility <= 0:
      return 1.0

    # Inverse relationship: baseline / current
    adj = self.volatility_baseline / current_volatility

    # Ограничиваем диапазон [0.5, 1.5]
    adj = max(0.5, min(1.5, adj))

    return adj

  def _get_win_rate_adj(self) -> float:
    """
    Корректировка на win rate.

    Логика:
    - Win rate > baseline → Increase risk (adj > 1.0)
    - Win rate < baseline → Reduce risk (adj < 1.0)

    Returns:
        float: Множитель корректировки [0.6, 1.4]
    """
    wins = sum(1 for t in self.trade_history if t[0])
    win_rate = wins / len(self.trade_history)

    # Ratio к baseline
    adj = win_rate / self.win_rate_baseline

    # Ограничиваем диапазон [0.6, 1.4]
    adj = max(0.6, min(1.4, adj))

    return adj

  def _get_ml_confidence_adj(self, ml_confidence: float) -> float:
    """
    Корректировка на ML confidence.

    Логика:
    - Very high confidence (>0.9) → +30% risk
    - High confidence (>0.8) → +15% risk
    - Medium confidence (>0.7) → no change
    - Low confidence (<0.7) → -15% risk

    Args:
        ml_confidence: ML уверенность 0.0-1.0

    Returns:
        float: Множитель корректировки [0.85, 1.3]
    """
    if ml_confidence >= 0.9:
      return 1.3
    elif ml_confidence >= 0.8:
      return 1.15
    elif ml_confidence >= 0.7:
      return 1.0
    else:
      return 0.85

  def record_trade(self, is_win: bool, pnl: float):
    """
    Запись результата сделки для статистики.

    Args:
        is_win: Прибыльная ли сделка (True/False)
        pnl: P&L в USDT
    """
    self.trade_history.append((is_win, pnl))

    # Ограничиваем историю (последние 500 трейдов)
    if len(self.trade_history) > 500:
      self.trade_history = self.trade_history[-500:]

    logger.debug(
      f"Trade recorded: win={is_win}, pnl={pnl:.2f}, "
      f"total_history={len(self.trade_history)}"
    )

  def get_statistics(self) -> dict:
    """
    Получение статистики трейдов.

    Returns:
        dict: Статистика с win_rate, avg_win, avg_loss, payoff_ratio
    """
    if not self.trade_history:
      return {
        'total_trades': 0,
        'win_rate': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'payoff_ratio': 0.0
      }

    wins = [t for t in self.trade_history if t[0]]
    losses = [t for t in self.trade_history if not t[0]]

    avg_win = np.mean([t[1] for t in wins]) if wins else 0.0
    avg_loss = np.mean([t[1] for t in losses]) if losses else 0.0

    return {
      'total_trades': len(self.trade_history),
      'win_rate': len(wins) / len(self.trade_history),
      'avg_win': avg_win,
      'avg_loss': avg_loss,
      'payoff_ratio': (
        avg_win / abs(avg_loss)
        if wins and losses and avg_loss != 0
        else 0.0
      )
    }


# Глобальный экземпляр
adaptive_risk_calculator = AdaptiveRiskCalculator()