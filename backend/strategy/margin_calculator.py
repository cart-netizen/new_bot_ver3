"""
Helper методы для расчета маржи.
Путь: backend/strategy/margin_calculator.py (новый файл)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MarginRequirements:
  """Требования по марже для позиции."""
  position_size_usdt: float  # Notional value
  initial_margin: float  # Начальная маржа
  maintenance_margin: float  # Поддерживающая маржа
  total_required: float  # Всего требуется баланса
  leverage: int  # Используемое плечо


class MarginCalculator:
  """Калькулятор маржи для Bybit."""

  # Константы Bybit
  MAINTENANCE_MARGIN_RATE = 0.2  # 20%

  @staticmethod
  def calculate_margin_requirements(
      position_size_usdt: float,
      leverage: int
  ) -> MarginRequirements:
    """
    Расчет требований по марже.

    Args:
        position_size_usdt: Размер позиции (notional value)
        leverage: Кредитное плечо

    Returns:
        MarginRequirements: Требования по марже
    """
    # Initial Margin (начальная маржа)
    initial_margin = position_size_usdt / leverage

    # Maintenance Margin (поддерживающая маржа)
    maintenance_margin = initial_margin * MarginCalculator.MAINTENANCE_MARGIN_RATE

    # Total Required
    total_required = initial_margin + maintenance_margin

    return MarginRequirements(
      position_size_usdt=position_size_usdt,
      initial_margin=initial_margin,
      maintenance_margin=maintenance_margin,
      total_required=total_required,
      leverage=leverage
    )

  @staticmethod
  def max_position_from_balance(
      available_balance: float,
      leverage: int,
      safety_buffer: float = 0.05  # 5% резерв
  ) -> float:
    """
    Максимальный размер позиции исходя из баланса.

    Решает уравнение:
    total_required = (position / leverage) * (1 + maintenance_rate)
    position = total_required * leverage / (1 + maintenance_rate)

    Args:
        available_balance: Доступный баланс
        leverage: Кредитное плечо
        safety_buffer: Резерв (например 0.05 = 5%)

    Returns:
        float: Максимальный размер позиции в USDT
    """
    # Баланс с учетом резерва
    usable_balance = available_balance * (1 - safety_buffer)

    # Коэффициент маржи
    margin_multiplier = 1 + MarginCalculator.MAINTENANCE_MARGIN_RATE

    # Максимальная позиция
    max_position = usable_balance * leverage / margin_multiplier

    return max_position

  @staticmethod
  def validate_balance(
      position_size_usdt: float,
      available_balance: float,
      leverage: int
  ) -> tuple[bool, Optional[str]]:
    """
    Проверка достаточности баланса.

    Args:
        position_size_usdt: Размер позиции
        available_balance: Доступный баланс
        leverage: Кредитное плечо

    Returns:
        (is_valid, error_message)
    """
    requirements = MarginCalculator.calculate_margin_requirements(
      position_size_usdt, leverage
    )

    if requirements.total_required > available_balance:
      error_msg = (
        f"Недостаточно баланса: "
        f"требуется {requirements.total_required:.4f} USDT, "
        f"доступно {available_balance:.4f} USDT "
        f"(shortage: {requirements.total_required - available_balance:.4f} USDT)"
      )
      return False, error_msg

    return True, None


# ============================================
# ИСПОЛЬЗОВАНИЕ В RISK_MANAGER
# ============================================

"""
В методе calculate_position_size:

from backend.strategy.margin_calculator import MarginCalculator

# После базового расчета position_size_usdt:

# 1. Проверяем что баланс позволяет открыть позицию
is_valid, error = MarginCalculator.validate_balance(
    position_size_usdt,
    available_balance,
    leverage
)

if not is_valid:
    logger.error(f"{signal.symbol} | {error}")

    # Пытаемся уменьшить до максимально возможного
    max_possible = MarginCalculator.max_position_from_balance(
        available_balance,
        leverage,
        safety_buffer=0.05
    )

    if max_possible >= min_notional_value:
        logger.warning(
            f"{signal.symbol} | Уменьшаем позицию до максимально возможной: "
            f"{max_possible:.2f} USDT"
        )
        position_size_usdt = max_possible
    else:
        logger.error(
            f"{signal.symbol} | Даже минимальная позиция "
            f"{min_notional_value:.2f} USDT невозможна. "
            f"Максимум: {max_possible:.2f} USDT"
        )
        return 0.0

# 2. Получаем детали маржи для логирования
margin = MarginCalculator.calculate_margin_requirements(
    position_size_usdt,
    leverage
)

logger.info(
    f"{signal.symbol} | ✅ Позиция рассчитана: "
    f"{margin.position_size_usdt:.2f} USDT, "
    f"маржа: {margin.total_required:.4f} USDT "
    f"({margin.initial_margin:.4f} initial + "
    f"{margin.maintenance_margin:.4f} maintenance)"
)
"""

# ============================================
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ============================================

if __name__ == "__main__":
  calc = MarginCalculator()

  # Пример 1: Расчет маржи для позиции 5 USDT с leverage 10x
  print("=== Пример 1: Минимальная позиция ===")
  margin = calc.calculate_margin_requirements(5.0, 10)
  print(f"Position: {margin.position_size_usdt} USDT")
  print(f"Initial margin: {margin.initial_margin} USDT")
  print(f"Maintenance margin: {margin.maintenance_margin} USDT")
  print(f"Total required: {margin.total_required} USDT")
  print()

  # Пример 2: Максимальная позиция при балансе 3.76 USDT
  print("=== Пример 2: Максимум при балансе 3.76 USDT ===")
  balance = 3.76
  max_pos = calc.max_position_from_balance(balance, 10, safety_buffer=0.05)
  print(f"Available balance: {balance} USDT")
  print(f"Max position: {max_pos:.2f} USDT")

  margin = calc.calculate_margin_requirements(max_pos, 10)
  print(f"Required margin: {margin.total_required:.4f} USDT")
  print(f"Remaining: {balance - margin.total_required:.4f} USDT")
  print()

  # Пример 3: Валидация баланса
  print("=== Пример 3: Валидация ===")
  is_valid, error = calc.validate_balance(37.6, 3.76, 10)
  print(f"Position 37.6 USDT valid? {is_valid}")
  if error:
    print(f"Error: {error}")
  print()

  is_valid, error = calc.validate_balance(5.0, 3.76, 10)
  print(f"Position 5.0 USDT valid? {is_valid}")
  if error:
    print(f"Error: {error}")