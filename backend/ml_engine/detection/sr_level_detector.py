"""
Support/Resistance Level Detection с динамическим трекингом.

Функциональность:
- Обнаружение уровней поддержки и сопротивления
- Динамическое отслеживание силы уровней
- Детекция пробоев (breakouts)
- Кластерный анализ объемов
- Машинное обучение для валидации уровней

Путь: backend/ml_engine/detection/sr_level_detector.py
"""


from typing import Dict, List, Optional, Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.strategy.candle_manager import Candle
else:
    Candle = None

from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

from backend.core.logger import get_logger
from backend.core.periodic_logger import periodic_logger
from backend.models.orderbook import OrderBookSnapshot
from backend.strategy.candle_manager import Candle


logger = get_logger(__name__)


@dataclass
class SRLevelConfig:
  """Конфигурация детектора."""
  # Кластеризация
  price_tolerance_pct: float = 0.001  # 0.1% для группировки уровней
  min_touches: int = 2  # Минимум касаний для валидного уровня

  # Сила уровня
  volume_weight: float = 0.4  # Вес объема в расчете силы
  touch_weight: float = 0.3  # Вес касаний
  recency_weight: float = 0.3  # Вес недавности

  # История
  lookback_candles: int = 200  # Анализ последних 200 свечей
  max_age_hours: int = 24  # Макс возраст уровня

  # Пробои
  breakout_confirmation_candles: int = 2  # Свечей для подтверждения
  breakout_volume_threshold: float = 1.5  # 1.5x средний объем


@dataclass
class SRLevel:
  """Уровень поддержки/сопротивления."""
  price: float
  level_type: str  # "support" или "resistance"
  strength: float  # 0-1

  # Метрики
  touch_count: int
  total_volume: float
  avg_volume: float

  # История касаний
  touch_timestamps: List[int]
  touch_prices: List[float]

  # Актуальность
  first_seen: int  # timestamp ms
  last_seen: int

  # Состояние
  is_broken: bool = False
  breakout_timestamp: Optional[int] = None
  breakout_direction: Optional[str] = None  # "up" или "down"

  def age_hours(self, current_time: int) -> float:
    """Возраст уровня в часах."""
    return (current_time - self.first_seen) / (1000 * 3600)

  def time_since_last_touch(self, current_time: int) -> float:
    """Время с последнего касания в часах."""
    return (current_time - self.last_seen) / (1000 * 3600)


class SRLevelDetector:
  """
  Детектор уровней поддержки и сопротивления.

  Методология:
  1. Анализ исторических свечей для определения экстремумов
  2. Кластеризация близких уровней цены
  3. Подсчет касаний и объемов на уровнях
  4. Вычисление силы уровней
  5. Динамическое обновление при новых данных
  6. Детекция пробоев уровней
  """

  def __init__(self, config: SRLevelConfig):
    """
    Инициализация детектора.

    Args:
        config: Конфигурация
    """
    self.config = config

    # Уровни для каждого символа
    # symbol -> [SRLevel]
    self.levels: Dict[str, List[SRLevel]] = defaultdict(list)

    # История свечей (для анализа)
    # symbol -> deque[Candle]
    self.candle_history: Dict[str, List[Candle]] = defaultdict(list)

    # Статистика
    self.total_levels_detected = 0
    self.total_breakouts_detected = 0

    logger.info(
      f"Инициализирован SRLevelDetector: "
      f"min_touches={config.min_touches}, "
      f"lookback={config.lookback_candles}"
    )

  def update_candles(self, symbol: str, candles: List[Candle]):
    """
    Обновить историю свечей.

    Args:
        symbol: Торговая пара
        candles: Список свечей
    """
    # Ограничиваем размер истории
    max_candles = self.config.lookback_candles
    self.candle_history[symbol] = candles[-max_candles:]

  def detect_levels(self, symbol: str) -> List[SRLevel]:
    """
    Обнаружить S/R уровни для символа.

    Args:
        symbol: Торговая пара

    Returns:
        Список уровней
    """
    candles = self.candle_history.get(symbol, [])

    if len(candles) < 50:
      logger.debug(f"Недостаточно свечей для {symbol}: {len(candles)}")
      return []

    # Шаг 1: Находим экстремумы (highs и lows)
    highs, lows = self._find_extrema(candles)

    # Шаг 2: Кластеризуем уровни
    support_levels = self._cluster_levels(lows, candles, "support")
    resistance_levels = self._cluster_levels(highs, candles, "resistance")

    # Объединяем
    all_levels = support_levels + resistance_levels

    # Шаг 3: Фильтруем слабые уровни
    valid_levels = [
      level for level in all_levels
      if level.touch_count >= self.config.min_touches
    ]

    # Шаг 4: Обновляем сохраненные уровни
    self._update_levels(symbol, valid_levels)

    # Шаг 5: Проверяем пробои
    self._check_breakouts(symbol, candles)

    self.total_levels_detected += len(valid_levels)

    logger.debug(
      f"Обнаружено уровней для {symbol}: "
      f"support={len(support_levels)}, "
      f"resistance={len(resistance_levels)}"
    )

    return self.levels[symbol]

  def _find_extrema(
      self,
      candles: List[Candle]
  ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Найти экстремумы (локальные максимумы и минимумы).

    Returns:
        (highs, lows) - списки (index, price)
    """
    highs_prices = np.array([c.high for c in candles])
    lows_prices = np.array([c.low for c in candles])

    # Находим пики и впадины
    # distance=5 означает минимум 5 свечей между пиками
    high_peaks_array, _ = find_peaks(highs_prices, distance=5)
    low_peaks_array, _ = find_peaks(-lows_prices, distance=5)

    # Конвертируем в списки индексов (явное приведение типа)
    high_peaks: List[int] = [int(idx) for idx in high_peaks_array]  # type: ignore
    low_peaks: List[int] = [int(idx) for idx in low_peaks_array]  # type: ignore

    # Формируем результаты
    highs = [(idx, float(highs_prices[idx])) for idx in high_peaks]
    lows = [(idx, float(lows_prices[idx])) for idx in low_peaks]

    return highs, lows

  def _cluster_levels(
      self,
      extrema: List[Tuple[int, float]],
      candles: List[Candle],
      level_type: str
  ) -> List[SRLevel]:
    """
    Кластеризовать близкие уровни.

    Args:
        extrema: Список (index, price)
        candles: Свечи для извлечения метаданных
        level_type: "support" или "resistance"

    Returns:
        Список уровней
    """
    if not extrema:
      return []

    # Извлекаем цены
    prices = np.array([price for _, price in extrema])

    # DBSCAN для кластеризации
    # eps = price_tolerance в абсолютных единицах
    avg_price = np.mean(prices)
    eps = avg_price * self.config.price_tolerance_pct

    clustering = DBSCAN(eps=eps, min_samples=1)
    labels = clustering.fit_predict(prices.reshape(-1, 1))

    # Группируем по кластерам
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
      if label != -1:  # Игнорируем шум
        idx, price = extrema[i]
        clusters[label].append((idx, price))

    # Создаем уровни
    levels = []
    current_time = int(datetime.now().timestamp() * 1000)

    for cluster_prices in clusters.values():
      # Средняя цена кластера
      cluster_price = float(np.mean([p for _, p in cluster_prices]))

      # Считаем метрики
      touch_count = len(cluster_prices)

      # Объемы на касаниях
      volumes = []
      touch_timestamps = []
      touch_prices = []

      for idx, price in cluster_prices:
        if idx < len(candles):
          candle = candles[idx]
          volumes.append(candle.volume)
          touch_timestamps.append(int(candle.timestamp))
          touch_prices.append(price)

      total_volume = sum(volumes)
      avg_volume = np.mean(volumes) if volumes else 0.0

      # Временные метки
      first_seen = min(touch_timestamps) if touch_timestamps else current_time
      last_seen = max(touch_timestamps) if touch_timestamps else current_time

      # Вычисляем силу
      strength = self._calculate_strength(
        touch_count=touch_count,
        total_volume=total_volume,
        age_hours=(current_time - first_seen) / (1000 * 3600),
        current_time=current_time,
        last_seen=last_seen
      )

      level = SRLevel(
        price=cluster_price,
        level_type=level_type,
        strength=strength,
        touch_count=touch_count,
        total_volume=total_volume,
        avg_volume=avg_volume,
        touch_timestamps=touch_timestamps,
        touch_prices=touch_prices,
        first_seen=first_seen,
        last_seen=last_seen
      )

      levels.append(level)

    return levels

  def _calculate_strength(
      self,
      touch_count: int,
      total_volume: float,
      age_hours: float,
      current_time: int,
      last_seen: int
  ) -> float:
    """
    Вычислить силу уровня (0-1).

    Формула: strength = volume_component * volume_weight +
                       touch_component * touch_weight +
                       recency_component * recency_weight
    """
    # Volume component (нормализуем логарифмически)
    volume_component = min(np.log1p(total_volume) / 10.0, 1.0)

    # Touch component (нормализуем до 10 касаний = 1.0)
    touch_component = min(touch_count / 10.0, 1.0)

    # Recency component (недавние касания = сильнее)
    hours_since_touch = (current_time - last_seen) / (1000 * 3600)
    recency_component = max(1.0 - (hours_since_touch / 24.0), 0.0)

    # Взвешенная сумма
    strength = (
        volume_component * self.config.volume_weight +
        touch_component * self.config.touch_weight +
        recency_component * self.config.recency_weight
    )

    return min(strength, 1.0)

  def _update_levels(self, symbol: str, new_levels: List[SRLevel]):
    """Обновить сохраненные уровни."""
    # Объединяем с существующими уровнями
    existing_levels = self.levels[symbol]

    # Удаляем старые или слабые уровни
    current_time = int(datetime.now().timestamp() * 1000)

    filtered_existing = [
      level for level in existing_levels
      if (
          level.age_hours(current_time) < self.config.max_age_hours
          and level.strength > 0.3
          and not level.is_broken
      )
    ]

    # Добавляем новые уровни
    all_levels = filtered_existing + new_levels

    # Убираем дубликаты (близкие цены)
    unique_levels = []
    for level in all_levels:
      # Проверяем есть ли уже похожий уровень
      is_duplicate = False
      for existing in unique_levels:
        price_diff = abs(level.price - existing.price) / level.price
        if price_diff < self.config.price_tolerance_pct:
          # Дубликат - выбираем сильнейший
          if level.strength > existing.strength:
            unique_levels.remove(existing)
            unique_levels.append(level)
          is_duplicate = True
          break

      if not is_duplicate:
        unique_levels.append(level)

    # Сортируем по силе
    unique_levels.sort(key=lambda x: x.strength, reverse=True)

    # Ограничиваем количество уровней
    self.levels[symbol] = unique_levels[:20]  # Топ 20

  def _check_breakouts(self, symbol: str, candles: List[Candle]):
    """
    Проверить пробои уровней.

    Args:
        symbol: Торговая пара
        candles: Список свечей
    """
    if len(candles) < self.config.breakout_confirmation_candles + 1:
      return

    levels = self.levels[symbol]
    recent_candles = candles[-self.config.breakout_confirmation_candles - 1:]

    # Средний объем для сравнения
    avg_volume = float(np.mean([c.volume for c in candles[-20:]]))

    for level in levels:
      if level.is_broken:
        continue

      # Проверяем пробой
      breakout = self._detect_breakout(
        level,
        recent_candles,
        avg_volume
      )

      if breakout:
        direction, timestamp = breakout
        level.is_broken = True
        level.breakout_timestamp = timestamp
        level.breakout_direction = direction

        self.total_breakouts_detected += 1

        # ============================================
        # ДЕДУПЛИКАЦИЯ ЛОГОВ ПРОБОЕВ
        # ============================================
        # Создаём уникальный ключ для каждого пробоя:
        # symbol + тип уровня + направление + цена (округлённая)
        breakout_key = (
          f"breakout_{symbol}_{level.level_type}_"
          f"{direction}_{level.price:.2f}"
        )

        # Проверяем нужно ли логировать (cooldown 10 секунд)
        should_log, time_since = periodic_logger.should_log_with_cooldown(
          breakout_key,
          cooldown_seconds=20
        )

        if should_log:
          logger.info(
            f"🎯 ПРОБОЙ УРОВНЯ [{symbol}]: "
            f"price={level.price:.2f}, "
            f"type={level.level_type}, "
            f"direction={direction}, "
            f"strength={level.strength:.2f}"
          )
        else:
          # Логируем на DEBUG чтобы не терять информацию полностью
          logger.debug(
            f"🎯 ПРОБОЙ (дубликат) [{symbol}]: "
            f"price={level.price:.2f}, type={level.level_type}, "
            f"direction={direction} "
            f"(пропущен, последний лог {time_since:.1f}s назад)"
          )

  def _detect_breakout(
        self,
        level: SRLevel,
        candles: List[Candle],
        avg_volume: float
    ) -> Optional[Tuple[str, int]]:
      """
      Детектировать пробой уровня.

      Args:
          level: Уровень S/R
          candles: Последние свечи для проверки
          avg_volume: Средний объём

      Returns:
          (direction, timestamp) или None
      """
      if len(candles) < 2:
        return None

      # Последние N свечей для подтверждения
      confirmation_candles = candles[-self.config.breakout_confirmation_candles:]

      # Проверяем direction
      if level.level_type == "resistance":
        # Пробой вверх через сопротивление
        closes_above = all(c.close > level.price for c in confirmation_candles)
        high_volume = any(
          c.volume > avg_volume * self.config.breakout_volume_threshold
          for c in confirmation_candles
        )

        if closes_above and high_volume:
          return ("up", int(confirmation_candles[-1].timestamp))

      elif level.level_type == "support":
        # Пробой вниз через поддержку
        closes_below = all(c.close < level.price for c in confirmation_candles)
        high_volume = any(
          c.volume > avg_volume * self.config.breakout_volume_threshold
          for c in confirmation_candles
        )

        if closes_below and high_volume:
          return ("down", int(confirmation_candles[-1].timestamp))

      return None

  def get_nearest_levels(
      self,
      symbol: str,
      current_price: float,
      max_distance_pct: float = 0.02  # 2%
  ) -> Dict[str, Optional[SRLevel]]:
    """
    Получить ближайшие уровни.

    Returns:
        {"support": level, "resistance": level}
    """
    levels = self.levels.get(symbol, [])

    nearest_support = None
    nearest_resistance = None

    min_support_dist = float('inf')
    min_resistance_dist = float('inf')

    for level in levels:
      if level.is_broken:
        continue

      distance = abs(level.price - current_price) / current_price

      if distance > max_distance_pct:
        continue

      if level.level_type == "support" and level.price < current_price:
        if distance < min_support_dist:
          min_support_dist = distance
          nearest_support = level

      elif level.level_type == "resistance" and level.price > current_price:
        if distance < min_resistance_dist:
          min_resistance_dist = distance
          nearest_resistance = level

    return {
      "support": nearest_support,
      "resistance": nearest_resistance
    }

  def get_statistics(self) -> Dict:
    """Получить статистику детектора."""
    total_levels = sum(len(levels) for levels in self.levels.values())

    active_levels = sum(
      sum(1 for level in levels if not level.is_broken)
      for levels in self.levels.values()
    )

    broken_levels = sum(
      sum(1 for level in levels if level.is_broken)
      for levels in self.levels.values()
    )

    return {
      'symbols_monitored': len(self.levels),
      'total_levels': total_levels,
      'active_levels': active_levels,
      'broken_levels': broken_levels,
      'total_levels_detected': self.total_levels_detected,
      'total_breakouts': self.total_breakouts_detected
    }


# Пример использования
if __name__ == "__main__":
  from backend.strategy.candle_manager import Candle

  config = SRLevelConfig(
    price_tolerance_pct=0.001,
    min_touches=2,
    lookback_candles=200
  )

  detector = SRLevelDetector(config)

  # Создаем тестовые свечи
  np.random.seed(42)
  base_price = 50000.0
  candles = []

  for i in range(200):
    # Симулируем отскоки от уровней 49500 и 50500
    if i % 20 == 0:
      price = 49500 + np.random.randn() * 50
    elif i % 15 == 0:
      price = 50500 + np.random.randn() * 50
    else:
      price = base_price + np.random.randn() * 200

    candle = Candle(
      timestamp=int(datetime.now().timestamp() * 1000) + i * 60000,
      open=price,
      high=price + abs(np.random.randn() * 50),
      low=price - abs(np.random.randn() * 50),
      close=price + np.random.randn() * 30,
      volume=1000 + np.random.randn() * 200
    )
    candles.append(candle)

  # Обновляем детектор
  detector.update_candles("BTCUSDT", candles)

  # Детектируем уровни
  levels = detector.detect_levels("BTCUSDT")

  print(f"Обнаружено уровней: {len(levels)}")

  # Показываем топ-5 сильнейших
  for level in levels[:5]:
    print(f"\n{level.level_type.upper()}: ${level.price:.2f}")
    print(f"  Strength: {level.strength:.2f}")
    print(f"  Touches: {level.touch_count}")
    print(f"  Volume: {level.total_volume:.2f}")
    print(f"  Broken: {level.is_broken}")

  # Получаем ближайшие уровни
  nearest = detector.get_nearest_levels("BTCUSDT", 50000.0)
  print(f"\nБлижайшие уровни к $50,000:")
  if nearest["support"]:
    print(f"  Support: ${nearest['support'].price:.2f}")
  if nearest["resistance"]:
    print(f"  Resistance: ${nearest['resistance'].price:.2f}")