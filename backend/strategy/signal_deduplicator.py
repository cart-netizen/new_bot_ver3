"""
Signal Deduplicator - предотвращает дублирование сигналов для одной пары.

backend/strategy/signal_deduplicator.py
"""

from typing import Dict, Optional
from dataclasses import dataclass
from datetime import timedelta, datetime

from backend.core.logger import get_logger
from backend.models.signal import TradingSignal, SignalType

logger = get_logger(__name__)


@dataclass
class SignalRecord:
  """Запись о последнем сигнале."""
  signal_type: SignalType
  timestamp: datetime
  confidence: float
  price: float


class SignalDeduplicator:
  """
  Дедупликатор торговых сигналов.

  Предотвращает:
  - Дублирование одинаковых сигналов для одной пары
  - Слишком частую генерацию сигналов (cooldown)
  - Противоречивые сигналы (BUY после BUY без CLOSE)
  """

  def __init__(
      self,
      cooldown_seconds: int = 60,  # Минимум 60 сек между сигналами
      allow_same_direction: bool = False,  # Разрешить повторный BUY/SELL
      price_change_threshold: float = 0.001  # 0.1% изменение цены
  ):
    """
    Инициализация дедупликатора.

    Args:
        cooldown_seconds: Минимальное время между сигналами (сек)
        allow_same_direction: Разрешить повторные сигналы в том же направлении
        price_change_threshold: Минимальное изменение цены для нового сигнала
    """
    self.cooldown_seconds = cooldown_seconds
    self.allow_same_direction = allow_same_direction
    self.price_change_threshold = price_change_threshold

    # Хранилище последних сигналов по символам
    self.last_signals: Dict[str, SignalRecord] = {}

    # Статистика
    self.stats = {
      "total_checked": 0,
      "duplicates_blocked": 0,
      "cooldown_blocked": 0,
      "allowed": 0
    }

    logger.info(
      f"✓ SignalDeduplicator инициализирован: "
      f"cooldown={cooldown_seconds}s, "
      f"allow_same_direction={allow_same_direction}"
    )

  def should_process_signal(self, signal: TradingSignal) -> tuple[bool, Optional[str]]:
    """
    Проверка, нужно ли обрабатывать сигнал.

    Args:
        signal: Торговый сигнал

    Returns:
        tuple[bool, Optional[str]]: (разрешить, причина блокировки)
    """
    self.stats["total_checked"] += 1

    symbol = signal.symbol
    current_time = datetime.now()

    # Проверяем есть ли предыдущий сигнал для этой пары
    if symbol not in self.last_signals:
      # Первый сигнал для этой пары - разрешаем
      self._record_signal(signal, current_time)
      self.stats["allowed"] += 1
      logger.debug(f"{symbol} | ✓ Первый сигнал - разрешён")
      return True, None

    last_record = self.last_signals[symbol]
    time_since_last = (current_time - last_record.timestamp).total_seconds()

    # ==========================================
    # ПРОВЕРКА 1: COOLDOWN PERIOD
    # ==========================================
    if time_since_last < self.cooldown_seconds:
      reason = (
        f"Cooldown активен: {time_since_last:.1f}s < {self.cooldown_seconds}s "
        f"с последнего сигнала"
      )
      self.stats["cooldown_blocked"] += 1
      logger.debug(f"{symbol} | ❌ {reason}")
      return False, reason

    # ==========================================
    # ПРОВЕРКА 2: ОДИНАКОВОЕ НАПРАВЛЕНИЕ
    # ==========================================
    if signal.signal_type == last_record.signal_type:
      if not self.allow_same_direction:
        reason = (
          f"Повторный {signal.signal_type.value} сигнал запрещён "
          f"(последний: {last_record.timestamp.strftime('%H:%M:%S')})"
        )
        self.stats["duplicates_blocked"] += 1
        logger.debug(f"{symbol} | ❌ {reason}")
        return False, reason
      else:
        # Проверяем изменение цены
        price_change = abs(signal.price - last_record.price) / last_record.price

        if price_change < self.price_change_threshold:
          reason = (
            f"Цена изменилась незначительно: {price_change * 100:.2f}% < "
            f"{self.price_change_threshold * 100:.2f}%"
          )
          self.stats["duplicates_blocked"] += 1
          logger.debug(f"{symbol} | ❌ {reason}")
          return False, reason

    # ==========================================
    # ПРОВЕРКА 3: CONFIDENCE НЕ ХУЖЕ
    # ==========================================
    if signal.confidence < last_record.confidence * 0.9:  # -10% tolerance
      reason = (
        f"Confidence ниже: {signal.confidence:.2f} < "
        f"{last_record.confidence:.2f}"
      )
      self.stats["duplicates_blocked"] += 1
      logger.debug(f"{symbol} | ❌ {reason}")
      return False, reason

    # ==========================================
    # ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ - РАЗРЕШАЕМ
    # ==========================================
    self._record_signal(signal, current_time)
    self.stats["allowed"] += 1

    logger.info(
      f"{symbol} | ✓ Сигнал разрешён: {signal.signal_type.value} "
      f"(cooldown: {time_since_last:.1f}s, confidence: {signal.confidence:.2f})"
    )

    return True, None

  def _record_signal(self, signal: TradingSignal, timestamp: datetime):
    """Сохранить сигнал в истории."""
    self.last_signals[signal.symbol] = SignalRecord(
      signal_type=signal.signal_type,
      timestamp=timestamp,
      confidence=signal.confidence,
      price=signal.price
    )

  def clear_symbol(self, symbol: str):
    """
    Очистить историю для символа (после закрытия позиции).

    Args:
        symbol: Торговая пара
    """
    if symbol in self.last_signals:
      del self.last_signals[symbol]
      logger.debug(f"{symbol} | История сигналов очищена")

  def get_last_signal(self, symbol: str) -> Optional[SignalRecord]:
    """
    Получить последний сигнал для символа.

    Args:
        symbol: Торговая пара

    Returns:
        Optional[SignalRecord]: Последний сигнал или None
    """
    return self.last_signals.get(symbol)

  def get_statistics(self) -> Dict:
    """Получить статистику дедупликатора."""
    total = self.stats["total_checked"]
    if total == 0:
      return {**self.stats, "block_rate": 0.0}

    blocked = self.stats["duplicates_blocked"] + self.stats["cooldown_blocked"]

    return {
      **self.stats,
      "block_rate": (blocked / total * 100) if total > 0 else 0.0,
      "cooldown_block_rate": (
          self.stats["cooldown_blocked"] / total * 100
      ) if total > 0 else 0.0,
      "duplicate_block_rate": (
          self.stats["duplicates_blocked"] / total * 100
      ) if total > 0 else 0.0
    }

  def cleanup_old_records(self, max_age_hours: int = 24):
    """
    Очистка старых записей.

    Args:
        max_age_hours: Максимальный возраст записей (часы)
    """
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(hours=max_age_hours)

    symbols_to_remove = [
      symbol
      for symbol, record in self.last_signals.items()
      if record.timestamp < cutoff_time
    ]

    for symbol in symbols_to_remove:
      del self.last_signals[symbol]

    if symbols_to_remove:
      logger.info(
        f"Очищено {len(symbols_to_remove)} старых записей "
        f"(старше {max_age_hours}ч)"
      )


# Глобальный экземпляр дедупликатора
signal_deduplicator = SignalDeduplicator(
  cooldown_seconds=60,  # 1 минута между сигналами
  allow_same_direction=False,  # Не разрешать повторные BUY/SELL
  price_change_threshold=0.001  # 0.1% изменение цены
)