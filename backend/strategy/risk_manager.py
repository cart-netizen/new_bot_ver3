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
        available_balance: float,  # РЕАЛЬНЫЙ баланс передается явно!
        leverage: Optional[int] = None
    ) -> float:
      """
      Расчет размера позиции с учетом РЕАЛЬНОГО баланса.

      КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Теперь НЕ используем self.metrics.available_exposure_usdt
      в расчётах, а используем переданный available_balance!
      """
      if leverage is None:
        leverage = self.limits.default_leverage

      logger.debug(
        f"{signal.symbol} | Расчет размера позиции: "
        f"real_balance={available_balance:.2f} USDT, "
        f"leverage={leverage}x"
      )

      # ШАГ 1: Базовый размер - 5% от РЕАЛЬНОГО ДОСТУПНОГО баланса
      base_size = available_balance * 0.05

      logger.debug(
        f"{signal.symbol} | Базовый размер (5% от real balance): "
        f"{base_size:.2f} USDT"
      )

      # ШАГ 2: Корректируем на основе силы сигнала
      strength_multiplier = {
        "STRONG": 1.0,
        "MEDIUM": 0.7,
        "WEAK": 0.5
      }.get(signal.strength.value, 0.5)

      position_size_before_leverage = base_size * strength_multiplier

      logger.debug(
        f"{signal.symbol} | Размер с учетом силы ({signal.strength.value}): "
        f"{position_size_before_leverage:.2f} USDT (mult={strength_multiplier})"
      )

      # ШАГ 3: Применяем кредитное плечо
      position_size_with_leverage = position_size_before_leverage * leverage

      logger.debug(
        f"{signal.symbol} | Размер с плечом {leverage}x: "
        f"{position_size_with_leverage:.2f} USDT"
      )

      # ШАГ 4: Проверяем минимальный размер
      if position_size_with_leverage < self.limits.min_order_size_usdt:
        logger.warning(
          f"{signal.symbol} | Размер {position_size_with_leverage:.2f} USDT "
          f"< минимального {self.limits.min_order_size_usdt} USDT"
        )

        position_size_with_leverage = self.limits.min_order_size_usdt

        logger.info(
          f"{signal.symbol} | ✓ Размер увеличен до минимального: "
          f"{position_size_with_leverage:.2f} USDT"
        )

      # ШАГ 5: Ограничиваем максимумами
      # Не можем открыть больше реального баланса с плечом
      max_by_balance = available_balance * leverage
      position_size_with_leverage = min(position_size_with_leverage, max_by_balance)

      # Расчет фактического margin
      actual_margin = position_size_with_leverage / leverage

      # Проверяем что хватает margin
      if actual_margin > available_balance:
        logger.error(
          f"{signal.symbol} | ❌ Недостаточно margin: "
          f"требуется {actual_margin:.2f} USDT, доступно {available_balance:.2f} USDT"
        )
        # Возвращаем максимально возможный размер
        position_size_with_leverage = available_balance * leverage
        actual_margin = available_balance

      logger.info(
        f"{signal.symbol} | 📊 ФИНАЛЬНЫЙ РАЗМЕР: "
        f"{position_size_with_leverage:.2f} USDT "
        f"(margin: {actual_margin:.2f} USDT, leverage: {leverage}x, "
        f"strength: {signal.strength.value})"
      )

      return position_size_with_leverage

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