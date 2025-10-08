"""
Модуль управления рисками.
Контроль лимитов и проверка торговых операций перед исполнением.
"""

from typing import Dict, Optional
from dataclasses import dataclass

from core.logger import get_logger
from core.exceptions import (
  RiskManagementError
)
from models.signal import TradingSignal, SignalType
from backend.config import settings

logger = get_logger(__name__)


@dataclass
class RiskLimits:
  """Лимиты риск-менеджмента."""

  max_open_positions: int
  max_exposure_usdt: float
  min_order_size_usdt: float

  def to_dict(self) -> Dict:
    """Преобразование в словарь."""
    return {
      "max_open_positions": self.max_open_positions,
      "max_exposure_usdt": self.max_exposure_usdt,
      "min_order_size_usdt": self.min_order_size_usdt,
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

  def __init__(self):
    """Инициализация риск-менеджера."""
    # Загружаем лимиты из конфигурации
    self.limits = RiskLimits(
      max_open_positions=settings.MAX_OPEN_POSITIONS,
      max_exposure_usdt=settings.MAX_EXPOSURE_USDT,
      min_order_size_usdt=settings.MIN_ORDER_SIZE_USDT
    )

    # Текущие метрики
    self.metrics = RiskMetrics(
      available_exposure_usdt=self.limits.max_exposure_usdt
    )

    # Трекинг открытых позиций (ЗАГЛУШКА - в реальности из биржи)
    self.open_positions: Dict[str, Dict] = {}

    logger.info(
      f"Инициализирован риск-менеджер: "
      f"max_positions={self.limits.max_open_positions}, "
      f"max_exposure={self.limits.max_exposure_usdt} USDT"
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
        position_size_usdt: Размер позиции в USDT

    Returns:
        tuple[bool, Optional[str]]: (валидность, причина отклонения)
    """
    try:
      # Проверка минимального размера ордера
      if position_size_usdt < self.limits.min_order_size_usdt:
        reason = (
          f"Размер позиции {position_size_usdt} USDT меньше минимального "
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
          f"требуется {position_size_usdt} USDT, "
          f"доступно {self.metrics.available_exposure_usdt} USDT"
        )
        logger.warning(f"{signal.symbol} | {reason}")
        return False, reason

      # Проверка актуальности сигнала
      if not signal.is_valid:
        reason = f"Сигнал устарел (возраст {signal.age_seconds:.1f}с)"
        logger.warning(f"{signal.symbol} | {reason}")
        return False, reason

      logger.debug(f"{signal.symbol} | Сигнал прошел валидацию")
      return True, None

    except Exception as e:
      logger.error(f"{signal.symbol} | Ошибка валидации сигнала: {e}")
      raise RiskManagementError(f"Failed to validate signal: {str(e)}")

  def calculate_position_size(
      self,
      signal: TradingSignal,
      available_balance: float
  ) -> float:
    """
    Расчет размера позиции на основе риска.

    Args:
        signal: Торговый сигнал
        available_balance: Доступный баланс в USDT

    Returns:
        float: Размер позиции в USDT
    """
    # ЗАГЛУШКА: Простой расчет размера позиции
    # В реальной стратегии здесь будет более сложная логика

    # Базовый размер позиции (5% от доступной экспозиции)
    base_size = self.metrics.available_exposure_usdt * 0.05

    # Корректируем на основе силы сигнала
    strength_multiplier = {
      "STRONG": 1.0,
      "MEDIUM": 0.7,
      "WEAK": 0.5
    }.get(signal.strength.value, 0.5)

    position_size = base_size * strength_multiplier

    # Ограничиваем минимумом и максимумом
    position_size = max(position_size, self.limits.min_order_size_usdt)
    position_size = min(position_size, available_balance)
    position_size = min(position_size, self.metrics.available_exposure_usdt)

    logger.debug(
      f"{signal.symbol} | Рассчитан размер позиции: {position_size:.2f} USDT"
    )

    return position_size

  def register_position_opened(
      self,
      symbol: str,
      side: SignalType,
      size_usdt: float,
      entry_price: float
  ):
    """
    Регистрация открытой позиции.

    Args:
        symbol: Торговая пара
        side: Сторона (BUY/SELL)
        size_usdt: Размер позиции в USDT
        entry_price: Цена входа
    """
    self.open_positions[symbol] = {
      "side": side.value,
      "size_usdt": size_usdt,
      "entry_price": entry_price,
    }

    # Обновляем метрики
    self.metrics.open_positions_count = len(self.open_positions)
    self.metrics.total_exposure_usdt += size_usdt
    self.metrics.available_exposure_usdt = (
        self.limits.max_exposure_usdt - self.metrics.total_exposure_usdt
    )

    if size_usdt > self.metrics.largest_position_size:
      self.metrics.largest_position_size = size_usdt

    logger.info(
      f"{symbol} | Позиция зарегистрирована: "
      f"{side.value} {size_usdt:.2f} USDT @ {entry_price:.8f}"
    )
    logger.info(
      f"Текущая экспозиция: {self.metrics.total_exposure_usdt:.2f}/"
      f"{self.limits.max_exposure_usdt:.2f} USDT"
    )

  def register_position_closed(self, symbol: str):
    """
    Регистрация закрытой позиции.

    Args:
        symbol: Торговая пара
    """
    if symbol in self.open_positions:
      position = self.open_positions.pop(symbol)
      size_usdt = position["size_usdt"]

      # Обновляем метрики
      self.metrics.open_positions_count = len(self.open_positions)
      self.metrics.total_exposure_usdt -= size_usdt
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
        f"{symbol} | Позиция закрыта: освобождено {size_usdt:.2f} USDT"
      )
      logger.info(
        f"Текущая экспозиция: {self.metrics.total_exposure_usdt:.2f}/"
        f"{self.limits.max_exposure_usdt:.2f} USDT"
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