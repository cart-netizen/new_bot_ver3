"""
Модели данных для торговых сигналов.
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime


class SignalType(str, Enum):
  """Тип торгового сигнала."""
  BUY = "BUY"
  SELL = "SELL"
  HOLD = "HOLD"


class SignalStrength(str, Enum):
  """Сила торгового сигнала."""
  WEAK = "WEAK"
  MEDIUM = "MEDIUM"
  STRONG = "STRONG"


class SignalSource(str, Enum):
  """Источник торгового сигнала."""
  IMBALANCE = "IMBALANCE"  # Дисбаланс стакана
  CLUSTER = "CLUSTER"  # Кластерный анализ
  VOLUME = "VOLUME"  # Анализ объемов
  SPREAD = "SPREAD"  # Анализ спреда
  COMBINED = "COMBINED"  # Комбинированный анализ
  ML_VALIDATED = "ml_validated"
  STRATEGY = "strategy"

@dataclass
class TradingSignal:
  """Модель торгового сигнала."""

  symbol: str
  signal_type: SignalType
  strength: SignalStrength
  source: SignalSource
  timestamp: int
  price: float
  confidence: float  # 0.0 - 1.0

  # Метрики, на основе которых сгенерирован сигнал
  imbalance: Optional[float] = None
  volume_delta: Optional[float] = None
  cluster_info: Optional[dict] = None

  # Дополнительная информация
  reason: str = ""
  metadata: dict = field(default_factory=dict)

  # Для отслеживания результата
  executed: bool = False
  execution_price: Optional[float] = None
  execution_timestamp: Optional[int] = None

  # ===== НОВЫЕ ПОЛЯ ДЛЯ ML =====
  ml_features_used: bool = False  # Использованы ли ML признаки
  feature_count: Optional[int] = None  # Количество признаков
  ml_confidence_adjustment: Optional[float] = None  # Корректировка confidence от ML

  def __post_init__(self):
    """Валидация после инициализации."""
    if not 0.0 <= self.confidence <= 1.0:
      raise ValueError("Confidence должен быть в диапазоне 0.0-1.0")

  @property
  def age_seconds(self) -> float:
    """Возраст сигнала в секундах."""
    current_timestamp = int(datetime.now().timestamp() * 1000)
    return (current_timestamp - self.timestamp) / 1000

  @property
  def is_valid(self) -> bool:
    """Проверка, что сигнал еще актуален (не старше 60 секунд)."""
    return self.age_seconds < 60

  def to_dict(self) -> dict:
    """
    Преобразование в словарь для API.

    ИСПРАВЛЕНИЕ:
    - Безопасная обработка Enum полей (если уже строка, не вызываем .value)
    - Защита от AttributeError
    """
    return {
      "symbol": self.symbol,
      # ИСПРАВЛЕНИЕ: Проверяем тип перед вызовом .value
      "signal_type": self.signal_type.value if hasattr(self.signal_type, 'value') else self.signal_type,
      "strength": self.strength.value if hasattr(self.strength, 'value') else self.strength,
      "source": self.source.value if hasattr(self.source, 'value') else self.source,
      "timestamp": self.timestamp,
      "datetime": datetime.fromtimestamp(self.timestamp / 1000).isoformat(),
      "price": self.price,
      "confidence": self.confidence,
      "metrics": {
        "imbalance": self.imbalance,
        "volume_delta": self.volume_delta,
        "cluster_info": self.cluster_info,
      },
      "reason": self.reason,
      "metadata": self.metadata,
      "status": {
        "executed": self.executed,
        "execution_price": self.execution_price,
        "execution_timestamp": self.execution_timestamp,
        "age_seconds": self.age_seconds,
        "is_valid": self.is_valid,
      },
      "ml_features_used": self.ml_features_used,
      "feature_count": self.feature_count,
      "ml_confidence_adjustment": self.ml_confidence_adjustment
    }


@dataclass
class SignalStatistics:
  """Статистика торговых сигналов."""

  symbol: str
  total_signals: int = 0
  buy_signals: int = 0
  sell_signals: int = 0
  hold_signals: int = 0

  strong_signals: int = 0
  medium_signals: int = 0
  weak_signals: int = 0

  executed_signals: int = 0
  pending_signals: int = 0

  avg_confidence: float = 0.0

  # Статистика по источникам
  imbalance_signals: int = 0
  cluster_signals: int = 0
  volume_signals: int = 0
  combined_signals: int = 0

  def to_dict(self) -> dict:
    """Преобразование в словарь для API."""
    return {
      "symbol": self.symbol,
      "total": self.total_signals,
      "by_type": {
        "buy": self.buy_signals,
        "sell": self.sell_signals,
        "hold": self.hold_signals,
      },
      "by_strength": {
        "strong": self.strong_signals,
        "medium": self.medium_signals,
        "weak": self.weak_signals,
      },
      "execution": {
        "executed": self.executed_signals,
        "pending": self.pending_signals,
      },
      "avg_confidence": self.avg_confidence,
      "by_source": {
        "imbalance": self.imbalance_signals,
        "cluster": self.cluster_signals,
        "volume": self.volume_signals,
        "combined": self.combined_signals,
      }
    }