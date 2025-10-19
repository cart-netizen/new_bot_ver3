"""
Модуль управления рисками с поддержкой кредитного плеча.
Контроль лимитов и проверка торговых операций перед исполнением.

Изменения:
1. Добавлен параметр leverage в конфигурацию
2. calculate_position_size теперь учитывает leverage
3. Автоматическое увеличение размера позиции до минимального с учетом плеча
4. Детальное логирование расчётов
"""

from typing import Dict, Optional
from dataclasses import dataclass

from core.logger import get_logger
from core.exceptions import RiskManagementError
from models.signal import TradingSignal, SignalType
from config import settings
from strategy.adaptive_risk_calculator import adaptive_risk_calculator
from strategy.correlation_manager import correlation_manager
from strategy.daily_loss_killer import daily_loss_killer

logger = get_logger(__name__)


@dataclass
class RiskLimits:
  """Лимиты риск-менеджмента."""

  max_open_positions: int
  max_exposure_usdt: float
  min_order_size_usdt: float
  default_leverage: int = 10  # Добавлено: кредитное плечо по умолчанию

  def to_dict(self) -> Dict:
    """Преобразование в словарь."""
    return {
      "max_open_positions": self.max_open_positions,
      "max_exposure_usdt": self.max_exposure_usdt,
      "min_order_size_usdt": self.min_order_size_usdt,
      "default_leverage": self.default_leverage,
    }


@dataclass
class RiskMetrics:
  """Текущие метрики риска."""

  open_positions_count: int = 0
  total_exposure_usdt: float = 0.0
  available_exposure_usdt: float = 0.0
  largest_position_size: float = 0.0

  def to_dict(self) -> Dict:
    """Преобразование в словарь."""
    return {
      "open_positions_count": self.open_positions_count,
      "total_exposure_usdt": self.total_exposure_usdt,
      "available_exposure_usdt": self.available_exposure_usdt,
      "largest_position_size": self.largest_position_size,
    }


class RiskManager:
  """Менеджер управления рисками."""

  def __init__(self, default_leverage: int = 10, initial_balance: Optional[float] = None):
    """
    Инициализация риск-менеджера.

    Args:
        default_leverage: Кредитное плечо по умолчанию
        initial_balance: Начальный баланс (если None, будет запрошен позже)
    """
    # Загружаем лимиты из конфигурации
    self.limits = RiskLimits(
      max_open_positions=settings.MAX_OPEN_POSITIONS,
      max_exposure_usdt=settings.MAX_EXPOSURE_USDT,  # Это МАКСИМУМ, не текущий баланс!
      min_order_size_usdt=settings.MIN_ORDER_SIZE_USDT,
      default_leverage=default_leverage
    )

    # КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Инициализация с РЕАЛЬНЫМ балансом
    if initial_balance is not None:
      actual_available = initial_balance
    else:

      # Если баланс не передан, используем дефолтное значение ВРЕМЕННО
      # Он ДОЛЖЕН быть обновлён через update_available_balance()
      actual_available = 0.0
      logger.warning(
        "⚠️ Risk Manager инициализирован БЕЗ баланса! "
        "Вызовите update_available_balance() перед использованием!"
      )

    # Текущие метрики с РЕАЛЬНЫМ балансом
    self.metrics = RiskMetrics(
      available_exposure_usdt=actual_available
    )

    # Трекинг открытых позиций
    self.open_positions: Dict[str, Dict] = {}

    # ========== НОВОЕ: Интеграция CorrelationManager ==========
    self.correlation_manager = correlation_manager

    self.maintenance_margin_rate = 0.2  # 20% для Bybit
    self.min_notional_value = 5.0  # Минимум notional в USDT
    self.default_leverage = 10

    logger.info(
      f"Risk Manager инициализирован с CorrelationManager: "
      f"enabled={self.correlation_manager.enabled}"
    )

    logger.info(
      f"🛡️ Инициализирован Risk Manager: "
      f"max_positions={self.limits.max_open_positions}, "
      f"max_exposure_limit={self.limits.max_exposure_usdt} USDT, "
      f"current_available={self.metrics.available_exposure_usdt:.2f} USDT, "
      f"min_order_size={self.limits.min_order_size_usdt} USDT, "
      f"default_leverage={self.limits.default_leverage}x"
    )

  def update_available_balance(self, new_balance: float):
    """
    Обновление доступного баланса из реального источника.

    Должен вызываться:
    - При старте бота (после запроса баланса с биржи)
    - Периодически из balance_tracker
    - После закрытия позиций

    Args:
        new_balance: Новый доступный баланс в USDT
    """
    old_balance = self.metrics.available_exposure_usdt

    # Вычитаем текущую экспозицию из нового баланса
    self.metrics.available_exposure_usdt = max(
      0.0,
      new_balance - self.metrics.total_exposure_usdt
    )

    logger.info(
      f"💰 Баланс обновлён: {old_balance:.2f} → {self.metrics.available_exposure_usdt:.2f} USDT "
      f"(total_balance={new_balance:.2f}, locked={self.metrics.total_exposure_usdt:.2f})"
    )

  def validate_signal(
        self,
        signal: TradingSignal,
        position_size_usdt: float,
        leverage: Optional[int] = None  # ← ДОБАВЛЕН параметр
    ) -> tuple[bool, Optional[str]]:
      """
      Валидация торгового сигнала перед исполнением.

      Args:
          signal: Торговый сигнал
          position_size_usdt: Размер позиции в USDT (С УЧЕТОМ leverage!)
          leverage: Кредитное плечо (опционально)

      Returns:
          tuple[bool, Optional[str]]: (валидность, причина отклонения)
      """
      try:
        # Используем переданное плечо или дефолтное
        if leverage is None:
          leverage = self.limits.default_leverage

        # ============================================
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Вычисляем required margin
        # ============================================
        required_margin = position_size_usdt / leverage

        # ============================================
        # ПРОВЕРКА 0.0: DAILY LOSS KILLER (КРИТИЧНО!)
        # ============================================
        is_allowed, shutdown_reason = daily_loss_killer.is_trading_allowed()

        if not is_allowed:
          logger.critical(
            f"{signal.symbol} | 🚨 TRADING BLOCKED: {shutdown_reason}"
          )
          return False, shutdown_reason

        logger.debug(f"{signal.symbol} | ✓ Daily Loss Killer: торговля разрешена")



        # ============================================
        # ПРОВЕРКА 0.1: ЖЁСТКИЙ ЛИМИТ ПОЗИЦИЙ (ДВОЙНАЯ ЗАЩИТА)
        # ============================================
        if self.metrics.open_positions_count >= self.limits.max_open_positions:
          reason = (
            f"🛑 ДОСТИГНУТ ЛИМИТ: {self.metrics.open_positions_count}/"
            f"{self.limits.max_open_positions} позиций. "
            f"Открытые: {list(self.open_positions.keys())}"
          )
          logger.error(f"{signal.symbol} | {reason}")
          return False, reason

        # Проверка: уже есть позиция по этой паре?
        if signal.symbol in self.open_positions:
          reason = f"Позиция по {signal.symbol} уже открыта"
          logger.warning(f"{signal.symbol} | {reason}")
          return False, reason

        logger.debug(
          f"{signal.symbol} | Валидация: "
          f"position_size={position_size_usdt:.2f} USDT, "
          f"leverage={leverage}x, "
          f"required_margin={required_margin:.2f} USDT, "
          f"available={self.metrics.available_exposure_usdt:.2f} USDT"
        )

        # ============================================
        # ПРОВЕРКА 1: МИНИМАЛЬНЫЙ РАЗМЕР ОРДЕРА
        # ============================================
        # Проверяем position_size (с leverage), т.к. это то что идёт на биржу
        if position_size_usdt < self.limits.min_order_size_usdt:
          reason = (
            f"Размер позиции {position_size_usdt:.2f} USDT меньше минимального "
            f"{self.limits.min_order_size_usdt} USDT"
          )
          logger.warning(f"{signal.symbol} | {reason}")
          return False, reason

        # ============================================
        # ПРОВЕРКА 2: ДОСТУПНАЯ ЭКСПОЗИЦИЯ (MARGIN)
        # ============================================
        # ПРАВИЛЬНО: Сравниваем required_margin с available_exposure
        if required_margin > self.metrics.available_exposure_usdt:
          reason = (
            f"Недостаточно margin: "
            f"требуется {required_margin:.2f} USDT, "
            f"доступно {self.metrics.available_exposure_usdt:.2f} USDT "
            f"(position_size={position_size_usdt:.2f} USDT с leverage {leverage}x)"
          )
          logger.warning(f"{signal.symbol} | {reason}")
          return False, reason

        # ============================================
        # ПРОВЕРКА 3: МАКСИМАЛЬНОЕ КОЛИЧЕСТВО ПОЗИЦИЙ
        # ============================================
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
          # Проверяем только если это НОВАЯ позиция (не закрытие существующей)
          if signal.symbol not in self.open_positions:
            current_count = self.metrics.open_positions_count
            max_count = self.limits.max_open_positions

            if current_count >= max_count:
              reason = (
                f"Достигнут лимит открытых позиций: "
                f"{current_count}/{max_count}. "
                f"Открытые пары: {list(self.open_positions.keys())}"
              )
              logger.warning(f"{signal.symbol} | ⛔ {reason}")
              return False, reason

            logger.debug(
              f"{signal.symbol} | Проверка лимита позиций: "
              f"{current_count + 1}/{max_count} (после открытия)"
            )

        # ============================================
        # ПРОВЕРКА 4: АКТУАЛЬНОСТЬ СИГНАЛА
        # ============================================
        if not signal.is_valid:
          reason = f"Сигнал устарел (возраст {signal.age_seconds:.1f}с)"
          logger.warning(f"{signal.symbol} | {reason}")
          return False, reason

        # ========== НОВОЕ: Проверка корреляционных лимитов ==========
        can_open_corr, corr_reason = self.correlation_manager.can_open_position(
          symbol=signal.symbol,
          position_size_usdt=position_size_usdt
        )

        if not can_open_corr:
          logger.warning(
            f"{signal.symbol} | Отклонено из-за корреляции: {corr_reason}"
          )
          return False, corr_reason

        logger.debug(
          f"{signal.symbol} | ✓ Сигнал прошел валидацию: "
          f"position={position_size_usdt:.2f} USDT, "
          f"margin={required_margin:.2f} USDT"
        )
        return True, None

      except Exception as e:
        logger.error(f"{signal.symbol} | Ошибка валидации сигнала: {e}")
        raise RiskManagementError(f"Failed to validate signal: {str(e)}")

  def calculate_position_size(
      self,
      signal: TradingSignal,
      available_balance: float,
      stop_loss_price: float,
      leverage: Optional[int] = None,
      current_volatility: Optional[float] = None,
      ml_confidence: Optional[float] = None
  ) -> float:
    """
    Расчет размера позиции с ПРАВИЛЬНОЙ логикой Risk%.

    ПРАВИЛЬНАЯ ЛОГИКА:
    Risk% = процент баланса используемый как маржа (initial margin)
    position_base = balance * risk% (initial margin)
    position_notional = position_base * leverage (размер позиции в USDT)

    ПРИМЕР:
    balance = 3.76 USDT
    risk% = 2%
    position_base = 3.76 * 0.02 = 0.0752 USDT (initial margin)
    position_notional = 0.0752 * 10 = 0.752 USDT
    min_notional = 5 USDT
    → Final position = 5.0 USDT (после применения минимума)

    Args:
        signal: Торговый сигнал
        available_balance: Доступный баланс в USDT
        stop_loss_price: Цена stop loss
        leverage: Кредитное плечо (опционально)
        current_volatility: Текущая волатильность (опционально)
        ml_confidence: ML уверенность (опционально)

    Returns:
        float: Размер позиции в USDT (notional value)
               0.0 если невозможно открыть
    """
    if leverage is None:
      leverage = self.limits.default_leverage

    symbol = signal.symbol
    entry_price = signal.price

    logger.info(
      f"{symbol} | Начало расчета размера позиции: "
      f"balance={available_balance:.2f} USDT, leverage={leverage}x"
    )

    # ============================================
    # ЭТАП 1: РАСЧЕТ CORRELATION FACTOR
    # ============================================

    correlation_factor = 1.0

    try:
      # Получаем группу для символа
      group = self.correlation_manager.group_manager.get_group_for_symbol(
        signal.symbol
      )

      if group and group.active_positions > 0:
        # Если в группе уже есть позиции - применяем штраф
        # Чем больше позиций в коррелирующей группе, тем меньше risk
        correlation_factor = 1.0 / (1.0 + group.active_positions * 0.3)

        logger.debug(
          f"{symbol} | Correlation penalty applied: "
          f"group={group.group_id}, "
          f"active_positions={group.active_positions}, "
          f"factor={correlation_factor:.2f}"
        )
    except Exception as e:
      logger.warning(
        f"{symbol} | Error checking correlation: {e}"
      )
      correlation_factor = 1.0

    # ============================================
    # ЭТАП 2: ПОЛУЧЕНИЕ RISK% (adaptive)
    # ============================================

    # Adaptive Risk Calculator возвращает final_risk_percent
    # Это процент баланса для использования как маржи
    risk_params = adaptive_risk_calculator.calculate(
      signal=signal,
      balance=available_balance,
      stop_loss_price=stop_loss_price,
      current_volatility=current_volatility,
      correlation_factor=correlation_factor,
      ml_confidence=ml_confidence
    )

    risk_percent = risk_params.final_risk_percent  # например 0.02 = 2%

    logger.debug(
      f"{symbol} | Adaptive risk calculated: "
      f"risk%={risk_percent:.2%}, "
      f"volatility_adj={risk_params.volatility_adjustment:.2f}, "
      f"correlation_adj={risk_params.correlation_adjustment:.2f}"
    )

    # ============================================
    # ЭТАП 3: РАСЧЕТ POSITION_BASE (initial margin)
    # ============================================

    # position_base = сумма баланса для initial margin
    position_base = available_balance * risk_percent

    logger.debug(
      f"{symbol} | position_base (initial margin): "
      f"{position_base:.4f} USDT"
    )

    # ============================================
    # ЭТАП 4: РАСЧЕТ NOTIONAL POSITION
    # ============================================

    # Размер позиции на бирже = маржа * leverage
    position_notional = position_base * leverage

    logger.debug(
      f"{symbol} | position_notional (без минимума): "
      f"{position_notional:.4f} USDT"
    )

    # ============================================
    # ЭТАП 5: ПРИМЕНЕНИЕ МИНИМУМА
    # ============================================

    min_notional = self.limits.min_order_size_usdt

    if position_notional < min_notional:
      logger.info(
        f"{symbol} | Position {position_notional:.4f} USDT "
        f"< minimum {min_notional:.2f} USDT, "
        f"увеличиваем до минимума"
      )

      position_notional = min_notional

      # Пересчитываем требуемую маржу после увеличения до минимума
      position_base = position_notional / leverage

      logger.debug(
        f"{symbol} | После минимума: "
        f"notional={position_notional:.2f} USDT, "
        f"required_margin={position_base:.4f} USDT"
      )

    # ============================================
    # ЭТАП 6: РАСЧЕТ ТРЕБУЕМОГО БАЛАНСА
    # ============================================

    # Initial margin (уже рассчитана)
    initial_margin = position_base

    # Maintenance margin (20% от initial margin для Bybit)
    maintenance_margin_rate = 0.2
    maintenance_margin = initial_margin * maintenance_margin_rate

    # Total required = initial + maintenance
    total_required = initial_margin + maintenance_margin

    logger.debug(
      f"{symbol} | Margin breakdown:\n"
      f"  Initial: {initial_margin:.4f} USDT\n"
      f"  Maintenance: {maintenance_margin:.4f} USDT\n"
      f"  Total: {total_required:.4f} USDT"
    )

    # ============================================
    # ЭТАП 7: ПРОВЕРКА ДОСТАТОЧНОСТИ БАЛАНСА
    # ============================================

    if total_required > available_balance:
      shortage = total_required - available_balance

      logger.error(
        f"{symbol} | ❌ НЕДОСТАТОЧНО БАЛАНСА:\n"
        f"  Position size: {position_notional:.2f} USDT\n"
        f"  Leverage: {leverage}x\n"
        f"  Risk%: {risk_percent:.2%}\n"
        f"\n"
        f"  MARGIN REQUIREMENTS:\n"
        f"  Initial margin: {initial_margin:.4f} USDT\n"
        f"  Maintenance margin: {maintenance_margin:.4f} USDT\n"
        f"  Total required: {total_required:.4f} USDT\n"
        f"\n"
        f"  BALANCE:\n"
        f"  Available: {available_balance:.4f} USDT\n"
        f"  Shortage: {shortage:.4f} USDT\n"
        f"\n"
        f"РЕШЕНИЯ:\n"
        f"  1. Пополнить баланс минимум на {shortage:.4f} USDT\n"
        f"  2. Уменьшить min_order_size_usdt в конфиге\n"
        f"  3. Увеличить leverage (НЕ рекомендуется)"
      )

      return 0.0  # REJECT

    # ============================================
    # ЭТАП 8: ПРИМЕНЕНИЕ МАКСИМУМА (опционально)
    # ============================================

    # Максимальный размер от buying power
    # Обычно 100% = весь баланс можно использовать
    max_buying_power_percent = 90/100  # 100%

    # Решаем уравнение: total_required <= balance * max%
    # (position/leverage) * 1.2 <= balance * max%
    # position <= balance * max% * leverage / 1.2

    max_position = (available_balance * max_buying_power_percent * leverage) / 1.2

    if position_notional > max_position:
      logger.warning(
        f"{symbol} | Position {position_notional:.2f} USDT "
        f"> maximum {max_position:.2f} USDT, capping"
      )

      position_notional = max_position

      # Пересчитываем маржу после capping
      initial_margin = position_notional / leverage
      maintenance_margin = initial_margin * maintenance_margin_rate
      total_required = initial_margin + maintenance_margin

      # Повторная проверка баланса
      if total_required > available_balance:
        logger.error(
          f"{symbol} | ❌ После capping всё равно "
          f"недостаточно баланса"
        )
        return 0.0

    # ============================================
    # ЭТАП 9: ФИНАЛЬНАЯ ПРОВЕРКА МИНИМУМА
    # ============================================

    # После всех корректировок проверяем что >= минимума
    if position_notional < min_notional:
      logger.error(
        f"{symbol} | ❌ После применения ограничений "
        f"размер {position_notional:.2f} USDT < "
        f"минимума {min_notional:.2f} USDT"
      )
      return 0.0

    # ============================================
    # ЭТАП 10: ФИНАЛЬНОЕ ЛОГИРОВАНИЕ
    # ============================================

    remaining = available_balance - total_required
    remaining_percent = (remaining / available_balance) * 100

    logger.info(
      f"{symbol} | ✅ РАЗМЕР ПОЗИЦИИ РАССЧИТАН:\n"
      f"  Position size: {position_notional:.2f} USDT\n"
      f"  Leverage: {leverage}x\n"
      f"  Risk%: {risk_percent:.2%}\n"
      f"\n"
      f"  ADAPTIVE ADJUSTMENTS:\n"
      f"  Volatility adj: {risk_params.volatility_adjustment:.2f}x\n"
      f"  Correlation adj: {risk_params.correlation_adjustment:.2f}x\n"
      f"\n"
      f"  MARGIN REQUIREMENTS:\n"
      f"  Initial margin: {initial_margin:.4f} USDT\n"
      f"  Maintenance margin: {maintenance_margin:.4f} USDT\n"
      f"  Total required: {total_required:.4f} USDT\n"
      f"\n"
      f"  BALANCE:\n"
      f"  Available: {available_balance:.4f} USDT\n"
      f"  Remaining: {remaining:.4f} USDT ({remaining_percent:.1f}%)"
    )

    return position_notional



  # ==============================================================
  # НОВЫЙ МЕТОД: Запись результата трейда
  # ==============================================================

  def record_trade_result(self, is_win: bool, pnl: float):
    """
    Запись результата закрытого трейда для статистики.

    Используется для:
    - Kelly Criterion расчетов
    - Adaptive win rate adjustment

    Args:
        is_win: Прибыльный ли трейд
        pnl: P&L в USDT
    """
    adaptive_risk_calculator.record_trade(is_win, pnl)

    logger.debug(
      f"Trade result recorded: win={is_win}, pnl={pnl:.2f} USDT"
    )

  # ==============================================================
  # НОВЫЙ МЕТОД: Получение статистики Adaptive Risk
  # ==============================================================

  def get_adaptive_risk_statistics(self) -> dict:
    """
    Получение статистики Adaptive Risk Calculator.

    Returns:
        dict: Статистика с win_rate, payoff_ratio и т.д.
    """
    return adaptive_risk_calculator.get_statistics()

  def register_position_opened(
      self,
      symbol: str,
      side: SignalType,
      size_usdt: float,
      entry_price: float,
      leverage: int = None
  ):
    """
    Регистрация открытой позиции.

    Args:
        symbol: Торговая пара
        side: Сторона (BUY/SELL)
        size_usdt: Размер позиции в USDT (с учетом leverage)
        entry_price: Цена входа
        leverage: Кредитное плечо позиции
    """
    if leverage is None:
      leverage = self.limits.default_leverage

    actual_margin = size_usdt / leverage

    self.open_positions[symbol] = {
      "side": side.value,
      "size_usdt": size_usdt,
      "entry_price": entry_price,
      "leverage": leverage,
      "actual_margin": actual_margin,
    }

    # Обновляем метрики
    self.metrics.open_positions_count = len(self.open_positions)
    self.metrics.total_exposure_usdt += actual_margin
    self.metrics.available_exposure_usdt = (
        self.limits.max_exposure_usdt - self.metrics.total_exposure_usdt
    )

    if size_usdt > self.metrics.largest_position_size:
      self.metrics.largest_position_size = size_usdt

    self.correlation_manager.notify_position_opened(
      symbol=symbol,
      exposure_usdt=size_usdt
    )

    logger.debug(
      f"{symbol} | Позиция зарегистрирована в CorrelationManager"
    )

    logger.info(
      f"{symbol} | ✓ Позиция зарегистрирована: "
      f"{side.value} {size_usdt:.2f} USDT @ {entry_price:.8f} "
      f"(leverage={leverage}x, margin={actual_margin:.2f} USDT)"
    )
    logger.info(
      f"📊 Открытые позиции: {self.metrics.open_positions_count}/"
      f"{self.limits.max_open_positions} | "
      f"Margin: {self.metrics.total_exposure_usdt:.2f}/"
      f"{self.limits.max_exposure_usdt:.2f} USDT | "
      f"Пары: {list(self.open_positions.keys())}"
    )

  def register_position_closed(self, symbol: str):
    """
    Регистрация закрытой позиции.

    Args:
        symbol: Торговая пара
    """
    if symbol in self.open_positions:
      position = self.open_positions.pop(symbol)
      actual_margin = position["actual_margin"]

      # Обновляем метрики
      self.metrics.open_positions_count = len(self.open_positions)
      self.metrics.total_exposure_usdt -= actual_margin
      self.metrics.available_exposure_usdt = (
          self.limits.max_exposure_usdt - self.metrics.total_exposure_usdt
      )

      # Пересчитываем largest_position_size
      if self.open_positions:
        self.metrics.largest_position_size = max(
          pos["size_usdt"] for pos in self.open_positions.values()
        )
      else:
        self.metrics.largest_position_size = 0.0

      self.correlation_manager.notify_position_closed(
        symbol=symbol,
        exposure_usdt=position["size_usdt"]
      )

      logger.debug(
        f"{symbol} | Закрытие позиции зарегистрировано в CorrelationManager"
      )

      logger.info(
        f"{symbol} | ✓ Позиция закрыта: освобождено {actual_margin:.2f} USDT margin"
      )
      logger.info(
        f"📊 Открытые позиции: {self.metrics.open_positions_count}/"
        f"{self.limits.max_open_positions} | "
        f"Margin: {self.metrics.total_exposure_usdt:.2f}/"
        f"{self.limits.max_exposure_usdt:.2f} USDT | "
        f"Пары: {list(self.open_positions.keys())}"
      )
    else:
      logger.warning(
        f"{symbol} | ⚠️ Попытка закрыть незарегистрированную позицию"
      )

  def can_open_new_position(self, symbol: str) -> tuple[bool, Optional[str]]:
    """
    Проверка возможности открыть новую позицию.

    Args:
        symbol: Торговая пара

    Returns:
        tuple[bool, Optional[str]]: (можно_открыть, причина_отказа)
    """
    # Проверка 1: Уже есть открытая позиция по этой паре?
    if symbol in self.open_positions:
      return False, f"Позиция по {symbol} уже открыта"

    # Проверка 2: Достигнут лимит позиций?
    if self.metrics.open_positions_count >= self.limits.max_open_positions:
      return False, (
        f"Достигнут лимит позиций: "
        f"{self.metrics.open_positions_count}/{self.limits.max_open_positions}"
      )

    # Проверка 3: Есть доступный margin?
    if self.metrics.available_exposure_usdt < self.limits.min_order_size_usdt / self.limits.default_leverage:
      return False, (
        f"Недостаточно margin: доступно {self.metrics.available_exposure_usdt:.2f} USDT"
      )

    return True, None

  def get_position(self, symbol: str) -> Optional[Dict]:
    """
    Получение информации о позиции.

    Args:
        symbol: Торговая пара

    Returns:
        Dict: Информация о позиции или None
    """
    return self.open_positions.get(symbol)

  def get_all_positions(self) -> Dict[str, Dict]:
    """
    Получение всех открытых позиций.

    Returns:
        Dict[str, Dict]: Словарь позиций
    """
    return self.open_positions.copy()

  def get_risk_status(self) -> Dict:
    """
    Получение текущего статуса риска.

    Returns:
        Dict: Статус риска
    """
    return {
      "limits": self.limits.to_dict(),
      "metrics": self.metrics.to_dict(),
      "positions": self.open_positions,
      "utilization": {
        "positions": f"{self.metrics.open_positions_count}/{self.limits.max_open_positions}",
        "exposure": f"{self.metrics.total_exposure_usdt:.2f}/{self.limits.max_exposure_usdt:.2f} USDT",
        "exposure_percent": (
          (self.metrics.total_exposure_usdt / self.limits.max_exposure_usdt) * 100
          if self.limits.max_exposure_usdt > 0 else 0
        )
      }
    }

  def update_leverage(self, new_leverage: int):
    """
    Обновление кредитного плеча по умолчанию.

    Args:
        new_leverage: Новое значение кредитного плеча
    """
    old_leverage = self.limits.default_leverage
    self.limits.default_leverage = new_leverage

    logger.info(
      f"✓ Кредитное плечо обновлено: {old_leverage}x -> {new_leverage}x"
    )