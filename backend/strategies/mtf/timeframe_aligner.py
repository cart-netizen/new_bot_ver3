"""
Timeframe Aligner - проверка согласованности сигналов между таймфреймами.

Функциональность:
- Trend Alignment - проверка согласованности трендов
- Confluence Detection - обнаружение множественных подтверждений
- Divergence Detection - выявление противоречий
- Alignment Scoring - количественная оценка согласованности
- Contextual Filtering - фильтрация на основе контекста

Правила alignment:
1. Higher timeframe определяет направление
2. Confluence усиливает сигнал
3. Divergence отменяет или ослабляет сигнал
4. Alignment score используется для sizing

Путь: backend/strategies/mtf/timeframe_aligner.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from backend.core.logger import get_logger
from backend.models.signal import TradingSignal, SignalType
from backend.strategies.mtf.timeframe_coordinator import Timeframe
from backend.strategies.mtf.timeframe_analyzer import (
  TimeframeAnalysisResult,
  MarketRegime,
  VolatilityRegime
)

logger = get_logger(__name__)


class AlignmentType(Enum):
  """Типы alignment между таймфреймами."""
  STRONG_BULL = "strong_bull"  # Все TF бычьи
  MODERATE_BULL = "moderate_bull"  # Большинство TF бычьи
  WEAK_BULL = "weak_bull"  # Слабый бычий alignment
  NEUTRAL = "neutral"  # Нет четкого alignment
  WEAK_BEAR = "weak_bear"  # Слабый медвежий alignment
  MODERATE_BEAR = "moderate_bear"  # Большинство TF медвежьи
  STRONG_BEAR = "strong_bear"  # Все TF медвежьи


class DivergenceType(Enum):
  """Типы расхождений между таймфреймами."""
  NO_DIVERGENCE = "no_divergence"
  TREND_COUNTER = "trend_counter"  # Сигнал против тренда HTF
  CONFLICTING_TRENDS = "conflicting_trends"  # Разные тренды на TF
  VOLUME_DIVERGENCE = "volume_divergence"  # Расхождение в объемах
  MOMENTUM_DIVERGENCE = "momentum_divergence"  # Расхождение в momentum


@dataclass
class ConfluenceZone:
  """
  Зона confluence - множественные TF подтверждают уровень.

  ОБНОВЛЕНО: Добавлена поддержка Fibonacci retracement уровней для
  более точной идентификации психологически важных зон.
  """
  price_level: float
  timeframes_confirming: List[Timeframe]
  confluence_type: str  # 'support', 'resistance', 'breakout'
  strength: float  # 0.0-1.0

  # Fibonacci metadata (НОВОЕ)
  fib_levels: List[float] = None  # Fibonacci ratios совпадающие с этой зоной [0.382, 0.618, ...]
  has_fib_confluence: bool = False  # True если зона совпадает с Fibonacci уровнем
  fib_timeframes: List[Timeframe] = None  # TF, для которых найдено Fib совпадение

  def __post_init__(self):
    """Инициализация после создания dataclass."""
    if self.fib_levels is None:
      self.fib_levels = []
    if self.fib_timeframes is None:
      self.fib_timeframes = []
    # Автоматически определяем has_fib_confluence
    self.has_fib_confluence = len(self.fib_levels) > 0


@dataclass
class TimeframeAlignment:
  """Результат проверки alignment между таймфреймами."""
  # Общий alignment
  alignment_type: AlignmentType
  alignment_score: float  # 0.0-1.0, где 1.0 = perfect alignment

  # Детали по таймфреймам
  bullish_timeframes: List[Timeframe]
  bearish_timeframes: List[Timeframe]
  neutral_timeframes: List[Timeframe]

  # Тренд от высшего TF
  higher_timeframe_trend: int  # 1 = bull, -1 = bear, 0 = ranging
  higher_timeframe: Timeframe

  # Confluence
  confluence_zones: List[ConfluenceZone]
  has_strong_confluence: bool

  # Divergences
  divergence_type: DivergenceType
  divergence_severity: float  # 0.0-1.0
  divergence_details: str

  # Рекомендации
  recommended_action: SignalType  # BUY, SELL, HOLD
  recommended_confidence: float
  position_size_multiplier: float  # 0.0-1.5 (adjustment based on alignment)

  # Метаданные
  timestamp: int
  analyzed_timeframes: int
  warnings: List[str] = field(default_factory=list)


@dataclass
class AlignmentConfig:
  """Конфигурация TimeframeAligner."""
  # Веса таймфреймов для alignment score
  timeframe_weights: Dict[Timeframe, float] = field(default_factory=lambda: {
    Timeframe.M1: 0.10,
    Timeframe.M5: 0.20,
    Timeframe.M15: 0.30,
    Timeframe.H1: 0.40
  })

  # Primary timeframe (определяет главный тренд)
  primary_timeframe: Timeframe = Timeframe.H1

  # Thresholds
  min_alignment_score: float = 0.65  # Минимальный score для торговли
  strong_alignment_threshold: float = 0.85
  moderate_alignment_threshold: float = 0.70

  # Confluence detection
  confluence_price_tolerance_percent: float = 0.5  # 0.5% tolerance
  min_timeframes_for_confluence: int = 2

  # Divergence handling
  max_divergence_severity: float = 0.3  # Максимально допустимая divergence
  allow_trend_counter_signals: bool = False  # Разрешить сигналы против HTF тренда

  # Position sizing adjustments
  position_size_boost_on_confluence: float = 1.3
  position_size_penalty_on_divergence: float = 0.7


class TimeframeAligner:
  """
  Алайнер таймфреймов - проверка согласованности сигналов.

  Основные функции:
  1. Проверка trend alignment
  2. Детекция confluence zones
  3. Выявление divergences
  4. Расчет alignment score
  5. Генерация рекомендаций
  """

  def __init__(self, config: AlignmentConfig):
    """
    Инициализация алайнера.

    Args:
        config: Конфигурация alignment
    """
    self.config = config

    # Статистика
    self.total_alignments_checked = 0
    self.strong_alignments = 0
    self.divergences_detected = 0
    self.confluence_zones_found = 0

    logger.info(
      f"Инициализирован TimeframeAligner: "
      f"primary_tf={config.primary_timeframe.value}, "
      f"min_alignment_score={config.min_alignment_score}"
    )

  def check_alignment(
      self,
      tf_results: Dict[Timeframe, TimeframeAnalysisResult],
      current_price: float
  ) -> TimeframeAlignment:
    """
    Проверить alignment между всеми таймфреймами.

    Args:
        tf_results: Результаты анализа каждого таймфрейма
        current_price: Текущая цена

    Returns:
        TimeframeAlignment с полной информацией
    """
    from datetime import datetime

    self.total_alignments_checked += 1

    if not tf_results:
      logger.warning("Нет результатов таймфреймов для alignment check")
      return self._create_neutral_alignment(current_price)

    warnings = []

    # Шаг 1: Классификация таймфреймов по направлению
    bullish_tfs, bearish_tfs, neutral_tfs = self._classify_timeframes(tf_results)

    # Шаг 2: Определение тренда от высшего TF
    htf_trend, htf = self._get_higher_timeframe_trend(tf_results)

    # Шаг 3: Расчет alignment score
    alignment_score = self._calculate_alignment_score(
      tf_results, bullish_tfs, bearish_tfs, neutral_tfs
    )

    # Шаг 4: Определение типа alignment
    alignment_type = self._determine_alignment_type(
      alignment_score, bullish_tfs, bearish_tfs, htf_trend
    )

    # Шаг 5: Детекция confluence zones
    confluence_zones = self._detect_confluence_zones(
      tf_results, current_price
    )

    has_strong_confluence = len(confluence_zones) > 0 and any(
      len(z.timeframes_confirming) >= self.config.min_timeframes_for_confluence
      for z in confluence_zones
    )

    if has_strong_confluence:
      self.confluence_zones_found += 1

    # Шаг 6: Детекция divergences
    divergence_type, divergence_severity, divergence_details = \
      self._detect_divergences(
        tf_results, htf_trend, bullish_tfs, bearish_tfs
      )

    if divergence_type != DivergenceType.NO_DIVERGENCE:
      self.divergences_detected += 1

    # Шаг 7: Генерация рекомендаций
    recommended_action, recommended_confidence, position_multiplier = \
      self._generate_recommendations(
        alignment_type,
        alignment_score,
        htf_trend,
        has_strong_confluence,
        divergence_severity
      )

    # Статистика
    if alignment_score >= self.config.strong_alignment_threshold:
      self.strong_alignments += 1

    # Создание результата
    alignment = TimeframeAlignment(
      alignment_type=alignment_type,
      alignment_score=alignment_score,
      bullish_timeframes=bullish_tfs,
      bearish_timeframes=bearish_tfs,
      neutral_timeframes=neutral_tfs,
      higher_timeframe_trend=htf_trend,
      higher_timeframe=htf,
      confluence_zones=confluence_zones,
      has_strong_confluence=has_strong_confluence,
      divergence_type=divergence_type,
      divergence_severity=divergence_severity,
      divergence_details=divergence_details,
      recommended_action=recommended_action,
      recommended_confidence=recommended_confidence,
      position_size_multiplier=position_multiplier,
      timestamp=int(datetime.now().timestamp() * 1000),
      analyzed_timeframes=len(tf_results),
      warnings=warnings
    )

    logger.debug(
      f"Alignment check: type={alignment_type.value}, "
      f"score={alignment_score:.2f}, "
      f"htf_trend={'BULL' if htf_trend > 0 else 'BEAR' if htf_trend < 0 else 'RANGE'}, "
      f"confluence={len(confluence_zones)}, "
      f"divergence={divergence_type.value}"
    )

    return alignment

  def _classify_timeframes(
      self,
      tf_results: Dict[Timeframe, TimeframeAnalysisResult]
  ) -> Tuple[List[Timeframe], List[Timeframe], List[Timeframe]]:
    """
    Классифицировать таймфреймы по направлению сигналов.

    Returns:
        (bullish_tfs, bearish_tfs, neutral_tfs)
    """
    bullish = []
    bearish = []
    neutral = []

    for tf, result in tf_results.items():
      # Используем как сигнал TF, так и regime
      signal_direction = None

      if result.timeframe_signal:
        if result.timeframe_signal.signal_type == SignalType.BUY:
          signal_direction = 1
        elif result.timeframe_signal.signal_type == SignalType.SELL:
          signal_direction = -1

      # Также учитываем regime
      regime_direction = result.regime.trend_direction

      # Комбинируем
      if signal_direction == 1 or (signal_direction is None and regime_direction > 0):
        bullish.append(tf)
      elif signal_direction == -1 or (signal_direction is None and regime_direction < 0):
        bearish.append(tf)
      else:
        neutral.append(tf)

    return bullish, bearish, neutral

  def _get_higher_timeframe_trend(
      self,
      tf_results: Dict[Timeframe, TimeframeAnalysisResult]
  ) -> Tuple[int, Timeframe]:
    """
    Получить тренд от высшего таймфрейма.

    Returns:
        (trend_direction, timeframe)
    """
    # Приоритет: H1 > M15 > M5 > M1
    priority_order = [
      Timeframe.H1, Timeframe.H4, Timeframe.D1,
      Timeframe.M15, Timeframe.M5, Timeframe.M1
    ]

    for tf in priority_order:
      if tf in tf_results:
        result = tf_results[tf]
        return result.regime.trend_direction, tf

    # Fallback
    if tf_results:
      first_tf = list(tf_results.keys())[0]
      return tf_results[first_tf].regime.trend_direction, first_tf

    return 0, self.config.primary_timeframe

  def _calculate_alignment_score(
      self,
      tf_results: Dict[Timeframe, TimeframeAnalysisResult],
      bullish_tfs: List[Timeframe],
      bearish_tfs: List[Timeframe],
      neutral_tfs: List[Timeframe]
  ) -> float:
    """
    Рассчитать alignment score (взвешенный консенсус).

    Score = 1.0 означает perfect alignment (все согласны)
    Score = 0.0 означает полное противоречие

    Returns:
        Alignment score 0.0-1.0
    """
    if not tf_results:
      return 0.0

    # Определяем доминирующее направление
    bullish_weight = sum(
      self.config.timeframe_weights.get(tf, 0.1)
      for tf in bullish_tfs
    )

    bearish_weight = sum(
      self.config.timeframe_weights.get(tf, 0.1)
      for tf in bearish_tfs
    )

    neutral_weight = sum(
      self.config.timeframe_weights.get(tf, 0.1)
      for tf in neutral_tfs
    )

    total_weight = bullish_weight + bearish_weight + neutral_weight

    if total_weight == 0:
      return 0.0

    # Нормализация
    bullish_ratio = bullish_weight / total_weight
    bearish_ratio = bearish_weight / total_weight
    neutral_ratio = neutral_weight / total_weight

    # Score базируется на доминировании одного направления
    # Если все бычьи или все медвежьи = 1.0
    # Если 50/50 = 0.0

    if bullish_ratio > bearish_ratio:
      # Бычий alignment
      # Штраф за bearish и neutral TF
      score = bullish_ratio - (bearish_ratio * 0.5) - (neutral_ratio * 0.3)
    elif bearish_ratio > bullish_ratio:
      # Медвежий alignment
      score = bearish_ratio - (bullish_ratio * 0.5) - (neutral_ratio * 0.3)
    else:
      # Равновесие
      score = 0.0

    # Ограничиваем 0-1
    score = max(0.0, min(1.0, score))

    # Бонус за единогласие высших TF
    htf_bonus = 0.0
    primary_tf = self.config.primary_timeframe

    if primary_tf in tf_results:
      primary_direction = tf_results[primary_tf].regime.trend_direction

      if primary_direction != 0:
        # Проверяем согласие других TF с primary
        agreeing_tfs = [
          tf for tf, res in tf_results.items()
          if np.sign(res.regime.trend_direction) == np.sign(primary_direction)
        ]

        agreement_ratio = len(agreeing_tfs) / len(tf_results)

        if agreement_ratio > 0.75:
          htf_bonus = 0.1 * agreement_ratio

    final_score = min(1.0, score + htf_bonus)

    return final_score

  def _determine_alignment_type(
      self,
      score: float,
      bullish_tfs: List[Timeframe],
      bearish_tfs: List[Timeframe],
      htf_trend: int
  ) -> AlignmentType:
    """Определить тип alignment на основе score и распределения."""
    # Определяем направление
    if len(bullish_tfs) > len(bearish_tfs):
      direction = 1  # Bullish
    elif len(bearish_tfs) > len(bullish_tfs):
      direction = -1  # Bearish
    else:
      return AlignmentType.NEUTRAL

    # Определяем силу
    if direction > 0:
      # Bullish alignment
      if score >= self.config.strong_alignment_threshold:
        return AlignmentType.STRONG_BULL
      elif score >= self.config.moderate_alignment_threshold:
        return AlignmentType.MODERATE_BULL
      else:
        return AlignmentType.WEAK_BULL
    else:
      # Bearish alignment
      if score >= self.config.strong_alignment_threshold:
        return AlignmentType.STRONG_BEAR
      elif score >= self.config.moderate_alignment_threshold:
        return AlignmentType.MODERATE_BEAR
      else:
        return AlignmentType.WEAK_BEAR

  def _calculate_fibonacci_levels(
      self,
      tf_results: Dict[Timeframe, TimeframeAnalysisResult],
      current_price: float
  ) -> Dict[Timeframe, List[Tuple[float, str, float]]]:
    """
    Рассчитать Fibonacci retracement уровни для каждого таймфрейма.

    Fibonacci уровни - психологически важные зоны, используемые институционалами:
    - 0.236 (23.6%) - слабый уровень
    - 0.382 (38.2%) - сильный уровень коррекции
    - 0.5 (50%) - психологический уровень
    - 0.618 (61.8%) - золотое сечение, критический уровень
    - 0.786 (78.6%) - глубокая коррекция

    Args:
        tf_results: Результаты анализа по таймфреймам
        current_price: Текущая цена (для определения направления)

    Returns:
        Dict[Timeframe, List[Tuple[price, type, ratio]]]
        где type = 'fib_support' или 'fib_resistance'
    """
    fib_levels_by_tf: Dict[Timeframe, List[Tuple[float, str, float]]] = {}

    # Стандартные Fibonacci ratios
    FIB_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786]

    for tf, result in tf_results.items():
      fib_levels = []

      # Для расчета Fibonacci нужны swing high и swing low
      swing_high = result.indicators.swing_high
      swing_low = result.indicators.swing_low

      if not swing_high or not swing_low:
        continue

      # Защита от некорректных данных
      if swing_high <= swing_low:
        logger.warning(
          f"[FIB] {tf.value}: invalid swing levels "
          f"(high={swing_high:.2f} <= low={swing_low:.2f}), skipping"
        )
        continue

      # Рассчитываем диапазон движения
      price_range = swing_high - swing_low

      # Определяем тренд на основе текущей цены
      # Если цена ближе к swing_high - коррекция вниз (retracement от uptrend)
      # Если цена ближе к swing_low - коррекция вверх (retracement от downtrend)
      distance_to_high = abs(current_price - swing_high)
      distance_to_low = abs(current_price - swing_low)

      is_uptrend_retracement = distance_to_high < distance_to_low

      # Рассчитываем уровни для каждого Fibonacci ratio
      for ratio in FIB_RATIOS:
        if is_uptrend_retracement:
          # Uptrend correction: уровни рассчитываются от high вниз
          # Формула: high - (range * ratio)
          fib_price = swing_high - (price_range * ratio)
          level_type = 'fib_support'  # В uptrend Fibonacci = support
        else:
          # Downtrend correction: уровни рассчитываются от low вверх
          # Формула: low + (range * ratio)
          fib_price = swing_low + (price_range * ratio)
          level_type = 'fib_resistance'  # В downtrend Fibonacci = resistance

        # Сохраняем уровень с метаданными
        fib_levels.append((fib_price, level_type, ratio))

      if fib_levels:
        fib_levels_by_tf[tf] = fib_levels

        logger.debug(
          f"[FIB] {tf.value}: calculated {len(fib_levels)} levels, "
          f"range={price_range:.2f}, "
          f"trend={'UP' if is_uptrend_retracement else 'DOWN'}, "
          f"swing_high={swing_high:.2f}, swing_low={swing_low:.2f}"
        )

    return fib_levels_by_tf

  def _detect_confluence_zones(
      self,
      tf_results: Dict[Timeframe, TimeframeAnalysisResult],
      current_price: float
  ) -> List[ConfluenceZone]:
    """
    Детектировать зоны confluence - уровни, подтвержденные множественными TF.

    Ищет:
    - Swing highs/lows совпадающие между TF
    - Bollinger bands на схожих уровнях
    - Fibonacci retracement levels (РЕАЛИЗОВАНО)

    Returns:
        Список ConfluenceZone
    """
    confluence_zones = []

    # Собираем ключевые уровни со всех TF
    levels_by_tf: Dict[Timeframe, List[Tuple[float, str]]] = {}

    for tf, result in tf_results.items():
      levels = []

      # Swing highs/lows
      if result.indicators.swing_high:
        levels.append((result.indicators.swing_high, 'resistance'))

      if result.indicators.swing_low:
        levels.append((result.indicators.swing_low, 'support'))

      # Bollinger bands
      if result.indicators.bollinger_upper:
        levels.append((result.indicators.bollinger_upper, 'resistance'))

      if result.indicators.bollinger_lower:
        levels.append((result.indicators.bollinger_lower, 'support'))

      # SMAs (могут быть dynamic S/R)
      if result.indicators.sma_slow:
        level_type = 'support' if current_price > result.indicators.sma_slow else 'resistance'
        levels.append((result.indicators.sma_slow, level_type))

      if levels:
        levels_by_tf[tf] = levels

    # НОВОЕ: Рассчитываем Fibonacci retracement уровни
    fib_levels_by_tf = self._calculate_fibonacci_levels(tf_results, current_price)

    # Добавляем Fibonacci уровни к общим уровням
    for tf, fib_levels in fib_levels_by_tf.items():
      if tf not in levels_by_tf:
        levels_by_tf[tf] = []

      # Добавляем каждый Fibonacci уровень
      for fib_price, fib_type, fib_ratio in fib_levels:
        # Конвертируем fib_type в обычный type (убираем префикс 'fib_')
        simple_type = fib_type.replace('fib_', '')
        levels_by_tf[tf].append((fib_price, simple_type))

    # Ищем пересечения уровней между TF
    tolerance = current_price * (self.config.confluence_price_tolerance_percent / 100)

    # Проверяем все комбинации уровней
    all_levels = []
    for tf, levels in levels_by_tf.items():
      for price, level_type in levels:
        all_levels.append((tf, price, level_type))

    # Группируем близкие уровни
    grouped_levels: Dict[float, List[Tuple[Timeframe, str]]] = {}

    for tf, price, level_type in all_levels:
      # Ищем существующую группу в пределах tolerance
      found_group = False

      for group_price in list(grouped_levels.keys()):
        if abs(price - group_price) <= tolerance:
          grouped_levels[group_price].append((tf, level_type))
          found_group = True
          break

      if not found_group:
        grouped_levels[price] = [(tf, level_type)]

    # Создаем ConfluenceZone для групп с минимум 2 TF
    for price, tf_types in grouped_levels.items():
      if len(tf_types) >= self.config.min_timeframes_for_confluence:
        # Определяем преобладающий тип
        types = [t for _, t in tf_types]
        confluence_type = max(set(types), key=types.count)

        timeframes = [tf for tf, _ in tf_types]

        # Сила confluence = количество подтверждающих TF
        strength = len(timeframes) / len(tf_results)

        # НОВОЕ: Проверяем совпадение с Fibonacci уровнями
        fib_ratios_matched = []
        fib_tfs_matched = []

        for tf, fib_levels in fib_levels_by_tf.items():
          for fib_price, fib_type, fib_ratio in fib_levels:
            # Проверяем совпадение цены с Fibonacci уровнем
            if abs(price - fib_price) <= tolerance:
              # Этот Fibonacci уровень совпадает с confluence zone
              if fib_ratio not in fib_ratios_matched:
                fib_ratios_matched.append(fib_ratio)
              if tf not in fib_tfs_matched:
                fib_tfs_matched.append(tf)

        # Бонус к strength если есть Fibonacci confluence
        if fib_ratios_matched:
          # Golden ratio (0.618) дает больший бонус
          golden_bonus = 0.15 if 0.618 in fib_ratios_matched else 0.0
          fib_bonus = 0.1 * len(fib_ratios_matched) + golden_bonus
          strength = min(1.0, strength + fib_bonus)

        zone = ConfluenceZone(
          price_level=price,
          timeframes_confirming=timeframes,
          confluence_type=confluence_type,
          strength=strength,
          fib_levels=fib_ratios_matched,
          has_fib_confluence=len(fib_ratios_matched) > 0,
          fib_timeframes=fib_tfs_matched
        )

        confluence_zones.append(zone)

    # Сортируем по силе
    confluence_zones.sort(key=lambda z: z.strength, reverse=True)

    if confluence_zones:
      # Подсчет зон с Fibonacci confluence
      fib_confluence_count = sum(1 for z in confluence_zones if z.has_fib_confluence)

      logger.debug(
        f"Обнаружено {len(confluence_zones)} confluence zones "
        f"({fib_confluence_count} с Fibonacci), "
        f"сильнейшая: {confluence_zones[0].price_level:.2f} "
        f"({len(confluence_zones[0].timeframes_confirming)} TF, "
        f"strength={confluence_zones[0].strength:.2f}"
        f"{', Fib=' + str(confluence_zones[0].fib_levels) if confluence_zones[0].has_fib_confluence else ''})"
      )

      # Дополнительное логирование для зон с Fibonacci confluence
      for zone in confluence_zones:
        if zone.has_fib_confluence:
          logger.debug(
            f"[FIB CONFLUENCE] price={zone.price_level:.2f}, "
            f"type={zone.confluence_type}, "
            f"strength={zone.strength:.2f}, "
            f"fib_levels={zone.fib_levels}, "
            f"fib_tfs={[tf.value for tf in zone.fib_timeframes]}"
          )

    return confluence_zones

  def _detect_divergences(
      self,
      tf_results: Dict[Timeframe, TimeframeAnalysisResult],
      htf_trend: int,
      bullish_tfs: List[Timeframe],
      bearish_tfs: List[Timeframe]
  ) -> Tuple[DivergenceType, float, str]:
    """
    Детектировать расхождения между таймфреймами.

    Returns:
        (divergence_type, severity 0-1, details)
    """
    # Проверка 1: Trend Counter Signal
    # Если LTF сигнал против HTF тренда
    if htf_trend != 0:
      counter_trend_tfs = bearish_tfs if htf_trend > 0 else bullish_tfs

      if counter_trend_tfs:
        # Считаем вес counter-trend TF
        counter_weight = sum(
          self.config.timeframe_weights.get(tf, 0.1)
          for tf in counter_trend_tfs
        )

        total_weight = sum(self.config.timeframe_weights.values())
        severity = counter_weight / total_weight

        if severity > 0.3:
          details = (
            f"Counter-trend signals на {len(counter_trend_tfs)} TF "
            f"против HTF {htf_trend}, severity={severity:.2f}"
          )

          return DivergenceType.TREND_COUNTER, severity, details

    # Проверка 2: Conflicting Trends
    # Если равное количество bull и bear TF
    if len(bullish_tfs) > 0 and len(bearish_tfs) > 0:
      bull_weight = sum(
        self.config.timeframe_weights.get(tf, 0.1) for tf in bullish_tfs
      )
      bear_weight = sum(
        self.config.timeframe_weights.get(tf, 0.1) for tf in bearish_tfs
      )

      # Если примерно равные веса - конфликт
      weight_diff = abs(bull_weight - bear_weight)
      total_weight = bull_weight + bear_weight

      if total_weight > 0:
        balance_ratio = weight_diff / total_weight

        if balance_ratio < 0.3:  # Почти равновесие
          severity = 1.0 - balance_ratio
          details = (
            f"Conflicting trends: {len(bullish_tfs)} bull TF vs "
            f"{len(bearish_tfs)} bear TF, balance_ratio={balance_ratio:.2f}"
          )

          return DivergenceType.CONFLICTING_TRENDS, severity, details

    # Проверка 3: Volume Divergence
    # Если price движется вверх, но volume падает (или наоборот)
    volume_divergence_count = 0

    for tf, result in tf_results.items():
      if result.indicators.volume_ratio:
        price_direction = result.regime.trend_direction
        volume_strength = result.indicators.volume_ratio

        # Расхождение: цена растет, volume падает
        if price_direction > 0 and volume_strength < 0.8:
          volume_divergence_count += 1
        # Или наоборот
        elif price_direction < 0 and volume_strength < 0.8:
          volume_divergence_count += 1

    if volume_divergence_count >= 2:
      severity = min(volume_divergence_count / len(tf_results), 1.0)
      details = (
        f"Volume divergence на {volume_divergence_count} TF, "
        f"severity={severity:.2f}"
      )

      return DivergenceType.VOLUME_DIVERGENCE, severity, details

    # Проверка 4: Momentum Divergence
    # Если RSI показывает расхождение между TF
    rsi_values = []
    for tf, result in tf_results.items():
      if result.indicators.rsi:
        rsi_values.append((tf, result.indicators.rsi))

    if len(rsi_values) >= 2:
      # Проверяем есть ли противоположные сигналы от RSI
      overbought = [tf for tf, rsi in rsi_values if rsi > 70]
      oversold = [tf for tf, rsi in rsi_values if rsi < 30]

      if overbought and oversold:
        severity = min((len(overbought) + len(oversold)) / len(rsi_values), 1.0)
        details = (
          f"Momentum divergence: {len(overbought)} overbought TF, "
          f"{len(oversold)} oversold TF"
        )

        return DivergenceType.MOMENTUM_DIVERGENCE, severity, details

    # Нет существенных расхождений
    return DivergenceType.NO_DIVERGENCE, 0.0, "No significant divergences detected"

  def _generate_recommendations(
      self,
      alignment_type: AlignmentType,
      alignment_score: float,
      htf_trend: int,
      has_confluence: bool,
      divergence_severity: float
  ) -> Tuple[SignalType, float, float]:
    """
    Генерировать рекомендации на основе alignment.

    Returns:
        (recommended_action, confidence, position_size_multiplier)
    """
    # Базовое действие на основе alignment
    if alignment_type in [AlignmentType.STRONG_BULL, AlignmentType.MODERATE_BULL]:
      base_action = SignalType.BUY
    elif alignment_type in [AlignmentType.STRONG_BEAR, AlignmentType.MODERATE_BEAR]:
      base_action = SignalType.SELL
    elif alignment_type in [AlignmentType.WEAK_BULL, AlignmentType.WEAK_BEAR]:
      # Слабый alignment - осторожность
      base_action = SignalType.HOLD
    else:
      base_action = SignalType.HOLD

    # Базовая confidence из alignment score
    base_confidence = alignment_score

    # Модификаторы
    confidence_multiplier = 1.0
    position_multiplier = 1.0

    # Confluence boost
    if has_confluence:
      confidence_multiplier *= 1.15
      position_multiplier *= self.config.position_size_boost_on_confluence
      logger.debug("Confluence detected: boosting confidence and position size")

    # Divergence penalty
    if divergence_severity > 0:
      penalty = 1.0 - (divergence_severity * 0.5)  # Максимум -50%
      confidence_multiplier *= penalty
      position_multiplier *= self.config.position_size_penalty_on_divergence
      logger.debug(
        f"Divergence detected (severity={divergence_severity:.2f}): "
        f"reducing confidence and position size"
      )

    # Проверка против HTF тренда
    if not self.config.allow_trend_counter_signals:
      if htf_trend != 0:
        # Отменяем сигнал если он против HTF тренда
        if (base_action == SignalType.BUY and htf_trend < 0) or \
            (base_action == SignalType.SELL and htf_trend > 0):
          logger.debug(
            f"Signal against HTF trend: cancelling "
            f"(action={base_action.value}, htf_trend={htf_trend})"
          )
          base_action = SignalType.HOLD
          confidence_multiplier *= 0.5

    # Финальные значения
    final_confidence = min(base_confidence * confidence_multiplier, 0.95)
    final_position_multiplier = max(0.0, min(position_multiplier, 1.5))

    # Если confidence слишком низкая - отменяем
    if final_confidence < self.config.min_alignment_score:
      logger.debug(
        f"Confidence below threshold: {final_confidence:.2f} < "
        f"{self.config.min_alignment_score}, setting HOLD"
      )
      base_action = SignalType.HOLD
      final_position_multiplier = 0.0

    return base_action, final_confidence, final_position_multiplier

  def _create_neutral_alignment(self, current_price: float) -> TimeframeAlignment:
    """Создать нейтральный alignment (fallback)."""
    from datetime import datetime

    return TimeframeAlignment(
      alignment_type=AlignmentType.NEUTRAL,
      alignment_score=0.0,
      bullish_timeframes=[],
      bearish_timeframes=[],
      neutral_timeframes=[],
      higher_timeframe_trend=0,
      higher_timeframe=self.config.primary_timeframe,
      confluence_zones=[],
      has_strong_confluence=False,
      divergence_type=DivergenceType.NO_DIVERGENCE,
      divergence_severity=0.0,
      divergence_details="No data available",
      recommended_action=SignalType.HOLD,
      recommended_confidence=0.0,
      position_size_multiplier=0.0,
      timestamp=int(datetime.now().timestamp() * 1000),
      analyzed_timeframes=0,
      warnings=["No timeframe data available"]
    )

  def get_statistics(self) -> Dict:
    """Получить статистику алайнера."""
    return {
      'total_alignments_checked': self.total_alignments_checked,
      'strong_alignments': self.strong_alignments,
      'strong_alignment_rate': (
        self.strong_alignments / self.total_alignments_checked
        if self.total_alignments_checked > 0 else 0.0
      ),
      'divergences_detected': self.divergences_detected,
      'divergence_rate': (
        self.divergences_detected / self.total_alignments_checked
        if self.total_alignments_checked > 0 else 0.0
      ),
      'confluence_zones_found': self.confluence_zones_found
    }