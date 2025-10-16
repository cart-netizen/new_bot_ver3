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

  def __init__(self, default_leverage: int = 10):
    """
    Инициализация риск-менеджера.

    Args:
        default_leverage: Кредитное плечо по умолчанию
    """
    # Загружаем лимиты из конфигурации
    self.limits = RiskLimits(
      max_open_positions=settings.MAX_OPEN_POSITIONS,
      max_exposure_usdt=settings.MAX_EXPOSURE_USDT,
      min_order_size_usdt=settings.MIN_ORDER_SIZE_USDT,
      default_leverage=default_leverage
    )

    # Текущие метрики
    self.metrics = RiskMetrics(
      available_exposure_usdt=self.limits.max_exposure_usdt
    )

    # Трекинг открытых позиций
    self.open_positions: Dict[str, Dict] = {}

    logger.info(
      f"🛡️ Инициализирован Risk Manager: "
      f"max_positions={self.limits.max_open_positions}, "
      f"max_exposure={self.limits.max_exposure_usdt} USDT, "
      f"min_order_size={self.limits.min_order_size_usdt} USDT, "
      f"default_leverage={self.limits.default_leverage}x"
    )

  def validate_signal(
      self,
      signal: TradingSignal,
      position_size_usdt: float
  ) -> tuple[bool, Optional[str]]:
    """
    Валидация торгового сигнала перед исполнением.

    Args:
        signal: Торговый сигнал
        position_size_usdt: Размер позиции в USDT (с учетом leverage)

    Returns:
        tuple[bool, Optional[str]]: (валидность, причина отклонения)
    """
    try:
      # Проверка минимального размера ордера
      if position_size_usdt < self.limits.min_order_size_usdt:
        reason = (
          f"Размер позиции {position_size_usdt:.2f} USDT меньше минимального "
          f"{self.limits.min_order_size_usdt} USDT"
        )
        logger.warning(f"{signal.symbol} | {reason}")
        return False, reason

      # Проверка максимального количества позиций
      if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
        if signal.symbol not in self.open_positions:
          # Новая позиция
          if self.metrics.open_positions_count >= self.limits.max_open_positions:
            reason = (
              f"Достигнут лимит открытых позиций: "
              f"{self.metrics.open_positions_count}/{self.limits.max_open_positions}"
            )
            logger.warning(f"{signal.symbol} | {reason}")
            return False, reason

      # Проверка максимальной экспозиции
      if position_size_usdt > self.metrics.available_exposure_usdt:
        reason = (
          f"Недостаточно доступной экспозиции: "
          f"требуется {position_size_usdt:.2f} USDT, "
          f"доступно {self.metrics.available_exposure_usdt:.2f} USDT"
        )
        logger.warning(f"{signal.symbol} | {reason}")
        return False, reason

      # Проверка актуальности сигнала
      if not signal.is_valid:
        reason = f"Сигнал устарел (возраст {signal.age_seconds:.1f}с)"
        logger.warning(f"{signal.symbol} | {reason}")
        return False, reason

      logger.debug(
        f"{signal.symbol} | ✓ Сигнал прошел валидацию: "
        f"size={position_size_usdt:.2f} USDT"
      )
      return True, None

    except Exception as e:
      logger.error(f"{signal.symbol} | Ошибка валидации сигнала: {e}")
      raise RiskManagementError(f"Failed to validate signal: {str(e)}")

  def calculate_position_size(
      self,
      signal: TradingSignal,
      available_balance: float,
      leverage: Optional[int] = None
  ) -> float:
    """
    Расчет размера позиции на основе риска с учетом кредитного плеча.

    Логика:
    1. Рассчитываем базовый размер позиции (% от доступной экспозиции)
    2. Корректируем на основе силы сигнала
    3. Применяем кредитное плечо для увеличения позиции
    4. Проверяем и корректируем до минимального размера если нужно
    5. Ограничиваем максимумами

    Args:
        signal: Торговый сигнал
        available_balance: Доступный баланс в USDT
        leverage: Кредитное плечо (опционально, используется default)

    Returns:
        float: Размер позиции в USDT (с учетом leverage)
    """
    # Используем переданное плечо или дефолтное
    if leverage is None:
      leverage = self.limits.default_leverage

    logger.debug(
      f"{signal.symbol} | Расчет размера позиции: "
      f"balance={available_balance:.2f} USDT, "
      f"leverage={leverage}x"
    )

    # ШАГ 1: Базовый размер позиции (5% от доступной экспозиции)
    base_size = self.metrics.available_exposure_usdt * 0.05

    logger.debug(
      f"{signal.symbol} | Базовый размер (5% от exposure): "
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
      f"{signal.symbol} | Размер с учетом силы сигнала "
      f"({signal.strength.value}): {position_size_before_leverage:.2f} USDT "
      f"(multiplier={strength_multiplier})"
    )

    # ШАГ 3: Применяем кредитное плечо
    # Фактический размер позиции на бирже будет больше за счет плеча
    position_size_with_leverage = position_size_before_leverage * leverage

    logger.debug(
      f"{signal.symbol} | Размер с кредитным плечом {leverage}x: "
      f"{position_size_with_leverage:.2f} USDT"
    )

    # ШАГ 4: Проверяем минимальный размер и корректируем если нужно
    if position_size_with_leverage < self.limits.min_order_size_usdt:
      logger.warning(
        f"{signal.symbol} | Размер позиции "
        f"{position_size_with_leverage:.2f} USDT меньше минимального "
        f"{self.limits.min_order_size_usdt} USDT"
      )

      # Корректируем до минимального размера
      position_size_with_leverage = self.limits.min_order_size_usdt

      logger.info(
        f"{signal.symbol} | ✓ Размер позиции увеличен до минимального: "
        f"{position_size_with_leverage:.2f} USDT"
      )

    # ШАГ 5: Ограничиваем максимумами
    # Не можем открыть позицию больше доступного баланса (с учетом leverage)
    max_position_by_balance = available_balance * leverage
    position_size_with_leverage = min(
      position_size_with_leverage,
      max_position_by_balance
    )

    # Не можем превысить доступную экспозицию
    position_size_with_leverage = min(
      position_size_with_leverage,
      self.metrics.available_exposure_usdt
    )

    # Расчет фактического используемого маржина (без leverage)
    actual_margin_used = position_size_with_leverage / leverage

    logger.info(
      f"{signal.symbol} | 📊 ФИНАЛЬНЫЙ РАЗМЕР ПОЗИЦИИ: "
      f"{position_size_with_leverage:.2f} USDT "
      f"(маржин: {actual_margin_used:.2f} USDT, "
      f"leverage: {leverage}x, "
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

    # Фактический используемый маржин
    actual_margin = size_usdt / leverage

    self.open_positions[symbol] = {
      "side": side.value,
      "size_usdt": size_usdt,
      "entry_price": entry_price,
      "leverage": leverage,
      "actual_margin": actual_margin,
    }

    # Обновляем метрики
    # В метриках экспозиции учитываем только фактический используемый маржин
    self.metrics.open_positions_count = len(self.open_positions)
    self.metrics.total_exposure_usdt += actual_margin
    self.metrics.available_exposure_usdt = (
        self.limits.max_exposure_usdt - self.metrics.total_exposure_usdt
    )

    if size_usdt > self.metrics.largest_position_size:
      self.metrics.largest_position_size = size_usdt

    logger.info(
      f"{symbol} | ✓ Позиция зарегистрирована: "
      f"{side.value} {size_usdt:.2f} USDT @ {entry_price:.8f} "
      f"(leverage={leverage}x, margin={actual_margin:.2f} USDT)"
    )
    logger.info(
      f"📈 Текущая экспозиция (margin): {self.metrics.total_exposure_usdt:.2f}/"
      f"{self.limits.max_exposure_usdt:.2f} USDT "
      f"({self.metrics.open_positions_count} позиций)"
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

      # Обновляем метрики (возвращаем маржин)
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

      logger.info(
        f"{symbol} | ✓ Позиция закрыта: освобождено {actual_margin:.2f} USDT margin"
      )
      logger.info(
        f"📉 Текущая экспозиция (margin): {self.metrics.total_exposure_usdt:.2f}/"
        f"{self.limits.max_exposure_usdt:.2f} USDT "
        f"({self.metrics.open_positions_count} позиций)"
      )

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