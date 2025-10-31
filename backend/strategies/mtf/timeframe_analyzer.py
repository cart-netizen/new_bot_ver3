"""
Timeframe Analyzer - независимый анализ каждого таймфрейма.

Функциональность:
- Запуск всех стратегий на каждом таймфрейме независимо
- Извлечение timeframe-specific features
- Генерация per-timeframe signals с контекстом
- Timeframe-specific indicator calculation
- Regime detection для каждого таймфрейма

Архитектура:
    TimeframeAnalyzer
    ├── _calculate_tf_specific_indicators() - индикаторы для каждого TF
    ├── _detect_tf_regime() - определение режима рынка
    ├── _extract_tf_features() - извлечение признаков
    └── analyze_timeframe() - полный анализ TF

Путь: backend/strategies/mtf/timeframe_analyzer.py
"""

from typing import Dict, List, Optional, Tuple
from typing import TYPE_CHECKING


from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import time

from core.logger import get_logger
from ml_engine.detection.sr_level_detector import SRLevelDetector, SRLevelConfig, SRLevel
from strategies.volume_profile_strategy import VolumeProfile, VolumeProfileAnalyzer
from strategy.candle_manager import Candle
from models.orderbook import OrderBookSnapshot, OrderBookMetrics
from models.signal import TradingSignal, SignalType
from strategies.strategy_manager import ExtendedStrategyManager, StrategyResult
from strategies.mtf.timeframe_coordinator import Timeframe

logger = get_logger(__name__)


class MarketRegime(Enum):
  """Режимы рынка для таймфрейма."""
  STRONG_UPTREND = "strong_uptrend"
  WEAK_UPTREND = "weak_uptrend"
  RANGING = "ranging"
  WEAK_DOWNTREND = "weak_downtrend"
  STRONG_DOWNTREND = "strong_downtrend"


class VolatilityRegime(Enum):
  """Режимы волатильности."""
  HIGH = "high"
  NORMAL = "normal"
  LOW = "low"


@dataclass
class TimeframeIndicators:
  """Индикаторы для конкретного таймфрейма."""
  # Trend indicators
  sma_fast: Optional[float] = None  # Быстрая SMA
  sma_slow: Optional[float] = None  # Медленная SMA
  ema_fast: Optional[float] = None  # Быстрая EMA
  ema_slow: Optional[float] = None  # Медленная EMA

  # Trend strength
  adx: Optional[float] = None  # Average Directional Index
  adx_di_plus: Optional[float] = None  # +DI
  adx_di_minus: Optional[float] = None  # -DI

  # Momentum
  rsi: Optional[float] = None  # Relative Strength Index
  stochastic_k: Optional[float] = None  # Stochastic %K
  stochastic_d: Optional[float] = None  # Stochastic %D
  macd: Optional[float] = None  # MACD line
  macd_signal: Optional[float] = None  # MACD signal line
  macd_histogram: Optional[float] = None  # MACD histogram

  # Volatility
  atr: Optional[float] = None  # Average True Range
  atr_percent: Optional[float] = None  # ATR as % of price
  bollinger_upper: Optional[float] = None
  bollinger_middle: Optional[float] = None
  bollinger_lower: Optional[float] = None
  bollinger_width: Optional[float] = None  # Band width

  # Volume
  volume_sma: Optional[float] = None  # Volume SMA
  volume_ratio: Optional[float] = None  # Current / SMA
  obv: Optional[float] = None  # On-Balance Volume
  vwap: Optional[float] = None  # Volume Weighted Average Price

  # Price structure
  swing_high: Optional[float] = None  # Последний swing high
  swing_low: Optional[float] = None  # Последний swing low

  # Ichimoku (для высших TF)
  ichimoku_conversion: Optional[float] = None  # Tenkan-sen
  ichimoku_base: Optional[float] = None  # Kijun-sen
  ichimoku_span_a: Optional[float] = None  # Senkou Span A
  ichimoku_span_b: Optional[float] = None  # Senkou Span B


@dataclass
class CachedIndicators:
  """
  Кэшированные индикаторы с timestamp валидацией.

  Обеспечивает актуальность закэшированных данных через проверку timestamp
  последней свечи и времени расчета.
  """
  indicators: TimeframeIndicators
  candle_timestamp: int  # timestamp последней свечи, для которой рассчитаны индикаторы
  calculated_at: int  # когда были рассчитаны (system timestamp в ms)

  def is_valid(self, current_candle_timestamp: int, max_age_ms: Optional[int] = None) -> bool:
    """
    Проверить актуальность кэша.

    Args:
        current_candle_timestamp: timestamp последней доступной свечи
        max_age_ms: максимальный возраст кэша в миллисекундах

    Returns:
        True если кэш актуален
    """
    # Проверка 1: timestamp свечи должен совпадать
    if self.candle_timestamp != current_candle_timestamp:
      return False

    # Проверка 2: максимальный возраст (если указан)
    if max_age_ms is not None:
      current_time_ms = int(time.time() * 1000)
      cache_age = current_time_ms - self.calculated_at
      if cache_age > max_age_ms:
        return False

    return True


@dataclass
class CachedRegime:
  """Кэшированная информация о режиме рынка с timestamp валидацией."""
  regime: 'TimeframeRegimeInfo'
  candle_timestamp: int
  calculated_at: int

  def is_valid(self, current_candle_timestamp: int, max_age_ms: Optional[int] = None) -> bool:
    """Проверить актуальность кэша."""
    if self.candle_timestamp != current_candle_timestamp:
      return False

    if max_age_ms is not None:
      current_time_ms = int(time.time() * 1000)
      cache_age = current_time_ms - self.calculated_at
      if cache_age > max_age_ms:
        return False

    return True


@dataclass
class TimeframeRegimeInfo:
  """Информация о режиме рынка на таймфрейме."""
  market_regime: MarketRegime
  volatility_regime: VolatilityRegime

  # Детали тренда
  trend_direction: int  # 1 = up, -1 = down, 0 = ranging
  trend_strength: float  # 0.0-1.0

  # Детали волатильности
  volatility_percentile: float  # Процентиль относительно истории
  normalized_atr: float  # Нормализованный ATR

  # Дополнительные флаги
  is_breakout: bool = False  # Пробой уровня
  is_consolidation: bool = False  # Консолидация
  is_reversal_pattern: bool = False  # Паттерн разворота

  # Confidence в определении режима
  regime_confidence: float = 0.0


@dataclass
class TimeframeAnalysisResult:
  """Результат анализа одного таймфрейма."""
  timeframe: Timeframe
  timestamp: int

  # Данные свечей
  current_price: float
  candles_analyzed: int

  # Индикаторы
  indicators: TimeframeIndicators

  # Режим рынка
  regime: TimeframeRegimeInfo

  # Сигналы от стратегий
  strategy_results: List[StrategyResult]

  # Агрегированный сигнал таймфрейма
  timeframe_signal: Optional[TradingSignal] = None

  # Метаданные
  analysis_duration_ms: float = 0.0
  warnings: List[str] = field(default_factory=list)


class TimeframeAnalyzer:
  """
  Анализатор таймфреймов - независимый анализ каждого TF.

  Для каждого таймфрейма:
  1. Рассчитывает специфичные индикаторы
  2. Определяет рыночный режим
  3. Запускает все стратегии
  4. Генерирует агрегированный сигнал
  """

  def __init__(
      self,
      strategy_manager: ExtendedStrategyManager,
      lookback_periods: Dict[Timeframe, int] = None

  ):
    """
    Инициализация анализатора.

    Args:
        strategy_manager: Менеджер стратегий для запуска
        lookback_periods: Периоды для индикаторов (по таймфреймам)
    """
    self.strategy_manager = strategy_manager

    # Периоды lookback для расчета индикаторов
    # Разные TF требуют разные периоды
    self.lookback_periods = lookback_periods or {
      Timeframe.M1: {
        'sma_fast': 9, 'sma_slow': 21,
        'ema_fast': 12, 'ema_slow': 26,
        'rsi': 14, 'atr': 14,
        'adx': 14, 'stochastic': 14,
        'volume_sma': 20, 'bollinger': 20
      },
      Timeframe.M5: {
        'sma_fast': 20, 'sma_slow': 50,
        'ema_fast': 12, 'ema_slow': 26,
        'rsi': 14, 'atr': 14,
        'adx': 14, 'stochastic': 14,
        'volume_sma': 20, 'bollinger': 20
      },
      Timeframe.M15: {
        'sma_fast': 50, 'sma_slow': 100,
        'ema_fast': 12, 'ema_slow': 26,
        'rsi': 14, 'atr': 14,
        'adx': 14, 'stochastic': 14,
        'volume_sma': 20, 'bollinger': 20
      },
      Timeframe.H1: {
        'sma_fast': 50, 'sma_slow': 200,
        'ema_fast': 12, 'ema_slow': 26,
        'rsi': 14, 'atr': 14,
        'adx': 14, 'stochastic': 14,
        'volume_sma': 20, 'bollinger': 20,
        'ichimoku': True  # Включаем Ichimoku для высших TF
      }
    }

    # Кэш расчетов (для оптимизации) - ОБНОВЛЕНО с timestamp validation
    self._indicators_cache: Dict[Tuple[str, Timeframe], CachedIndicators] = {}
    self._regime_cache: Dict[Tuple[str, Timeframe], CachedRegime] = {}

    # TTL для кэша (максимальный возраст) - защита от устаревших данных
    self.cache_ttl_ms: Dict[Timeframe, int] = {
      Timeframe.M1: 60_000,       # 1 минута
      Timeframe.M5: 300_000,      # 5 минут
      Timeframe.M15: 900_000,     # 15 минут
      Timeframe.H1: 3_600_000,    # 1 час
      Timeframe.H4: 14_400_000,   # 4 часа
      Timeframe.D1: 86_400_000,   # 24 часа
    }

    # Статистика кэша
    self.cache_hits = 0
    self.cache_misses = 0
    self.cache_invalidations = 0

    # Статистика анализа
    self.total_analyses = 0
    self.analyses_by_timeframe = {tf: 0 for tf in Timeframe}

    # ============================================================
    # ИНИЦИАЛИЗАЦИЯ S/R LEVEL DETECTOR
    # ============================================================
    self.sr_detector = SRLevelDetector(
      config=SRLevelConfig(
        price_tolerance_pct=0.001,  # 0.1% для кластеризации
        min_touches=2,  # Минимум 2 касания для валидного уровня
        lookback_candles=200,  # Анализ последних 200 свечей
        max_age_hours=24,  # Уровни старше 24ч игнорируются
        breakout_confirmation_candles=2,  # Подтверждение пробоя
        breakout_volume_threshold=1.5  # 1.5x средний объем
      )
    )

    # ============================================================
    # КЭШИ ДЛЯ VOLUME PROFILE
    # ============================================================
    # symbol -> VolumeProfile
    self.volume_profiles: Dict[str, VolumeProfile] = {}

    # symbol -> timestamp последнего обновления
    self.last_vp_update: Dict[str, int] = {}

    # Интервал обновления Volume Profile (30 минут в ms)
    self.vp_update_interval = 30 * 60 * 1000

    self.ml_validator = None  # Будет установлен из main.py
    self.feature_pipeline = None  # Будет установлен из main.py

    # Статистика
    self.total_analyses = 0
    self.analyses_by_timeframe = {tf: 0 for tf in Timeframe}

    logger.info(
      f"Инициализирован TimeframeAnalyzer с lookback periods для "
      f"{len(self.lookback_periods)} таймфреймов"
    )

  async def analyze_timeframe(
        self,
        symbol: str,
        timeframe: Timeframe,
        candles: List[Candle],
        current_price: float,
        orderbook: Optional[OrderBookSnapshot] = None,
        metrics: Optional[OrderBookMetrics] = None
    ) -> TimeframeAnalysisResult:
      """
      Полный анализ одного таймфрейма с интеграцией S/R, VP и ML.

      РАСШИРЕННЫЙ PIPELINE:
      1. Валидация данных
      2. Расчет индикаторов
      3. Определение режима рынка
      4. ✅ НОВОЕ: S/R Level Detection
      5. ✅ НОВОЕ: Volume Profile Analysis
      6. ✅ НОВОЕ: ML Predictions (опционально)
      7. Запуск всех стратегий (с новыми данными)
      8. Генерация агрегированного сигнала

      Args:
          symbol: Торговая пара
          timeframe: Анализируемый таймфрейм
          candles: Свечи для этого таймфрейма
          current_price: Текущая цена
          orderbook: Стакан (опционально)
          metrics: Метрики стакана (опционально)

      Returns:
          TimeframeAnalysisResult с полным контекстом
      """
      import time
      start_time = time.time()

      warnings = []

      # ============================================================
      # ВАЛИДАЦИЯ ДАННЫХ
      # ============================================================
      if not candles or len(candles) < 50:
        warnings.append(f"Недостаточно свечей: {len(candles)} < 50")
        logger.warning(
          f"[{timeframe.value}] {symbol}: "
          f"Недостаточно данных для анализа"
        )

      try:
        # ============================================================
        # ШАГ 1: РАСЧЕТ ИНДИКАТОРОВ
        # ============================================================
        indicators = self._calculate_tf_specific_indicators(
          symbol, timeframe, candles
        )

        # ============================================================
        # ШАГ 2: ОПРЕДЕЛЕНИЕ РЕЖИМА РЫНКА
        # ============================================================
        regime = self._detect_tf_regime(
          symbol, timeframe, candles, indicators
        )

        # ============================================================
        # ШАГ 3: S/R LEVEL DETECTION (НОВОЕ)
        # ============================================================
        sr_levels: Optional[List[SRLevel]] = None

        try:
          # Обновляем историю свечей в детекторе
          self.sr_detector.update_candles(symbol, candles)

          # Детектируем уровни
          sr_levels = self.sr_detector.detect_levels(symbol)

          if sr_levels:
            logger.debug(
              f"[{timeframe.value}] {symbol}: "
              f"Обнаружено {len(sr_levels)} S/R уровней"
            )

            # Получаем ближайшие уровни для контекста
            nearest = self.sr_detector.get_nearest_levels(
              symbol,
              current_price,
              max_distance_pct=0.02  # В пределах 2%
            )

            if nearest.get("support"):
              support = nearest["support"]
              logger.debug(
                f"[{timeframe.value}] {symbol} - "
                f"Nearest Support: ${support.price:.2f} "
                f"(strength={support.strength:.2f})"
              )

            if nearest.get("resistance"):
              resistance = nearest["resistance"]
              logger.debug(
                f"[{timeframe.value}] {symbol} - "
                f"Nearest Resistance: ${resistance.price:.2f} "
                f"(strength={resistance.strength:.2f})"
              )

        except Exception as e:
          logger.error(
            f"[{timeframe.value}] {symbol} - "
            f"Ошибка S/R Detection: {e}"
          )
          sr_levels = None  # Продолжаем без S/R уровней
          warnings.append(f"S/R Detection failed: {str(e)}")

        # ============================================================
        # ШАГ 4: VOLUME PROFILE ANALYSIS (НОВОЕ)
        # ============================================================
        volume_profile: Optional[Dict] = None

        try:
          # Инициализируем структуры если нужно
          if symbol not in self.volume_profiles:
            self.volume_profiles[symbol] = {}
            self.last_vp_update[symbol] = {}

          # Проверяем нужно ли обновление
          current_time = int(candles[-1].timestamp)
          should_update = (
              timeframe not in self.volume_profiles[symbol] or
              timeframe not in self.last_vp_update[symbol] or
              (current_time - self.last_vp_update[symbol][timeframe]) > self.vp_update_interval
          )

          if should_update:
            # Берем последние 100 свечей для профиля
            profile_candles = candles[-100:]

            # Строим Volume Profile
            vp = VolumeProfileAnalyzer.build_profile(
              candles=profile_candles,
              price_bins=50,  # 50 ценовых уровней
              value_area_percent=0.70  # 70% объема для VA
            )

            # Обнаруживаем HVN/LVN nodes
            hvn_nodes, lvn_nodes = VolumeProfileAnalyzer.detect_nodes(
              profile=vp,
              hvn_percentile=80,  # Top 20% = HVN
              lvn_percentile=20  # Bottom 20% = LVN
            )

            vp.hvn_nodes = hvn_nodes
            vp.lvn_nodes = lvn_nodes

            # Сохраняем
            self.volume_profiles[symbol][timeframe] = vp
            self.last_vp_update[symbol][timeframe] = current_time

            logger.debug(
              f"[{timeframe.value}] {symbol} - "
              f"Volume Profile обновлен: "
              f"POC=${vp.poc_price:.2f}, "
              f"VA=[${vp.value_area_low:.2f}, ${vp.value_area_high:.2f}], "
              f"HVN={len(hvn_nodes)}, LVN={len(lvn_nodes)}"
            )
          else:
            # Используем кэшированный профиль
            vp = self.volume_profiles[symbol].get(timeframe)

          # Конвертируем в dict для передачи в стратегии
          if vp:
            volume_profile = {
              'poc_price': vp.poc_price,
              'poc_volume': vp.poc_volume,
              'value_area_high': vp.value_area_high,
              'value_area_low': vp.value_area_low,
              'hvn_nodes': [
                {
                  'price': node.price,
                  'volume': node.volume,
                  'node_type': node.node_type
                }
                for node in vp.hvn_nodes
              ],
              'lvn_nodes': [
                {
                  'price': node.price,
                  'volume': node.volume,
                  'node_type': node.node_type
                }
                for node in vp.lvn_nodes
              ]
            }

        except Exception as e:
          logger.error(
            f"[{timeframe.value}] {symbol} - "
            f"Ошибка Volume Profile: {e}"
          )
          volume_profile = None  # Продолжаем без VP
          warnings.append(f"Volume Profile failed: {str(e)}")

        # ============================================================
        # ШАГ 5: ML PREDICTION (НОВОЕ, ОПЦИОНАЛЬНО)
        # ============================================================
        ml_prediction: Optional[Dict] = None

        try:
          # Проверяем доступность ML модуля
          if self.ml_validator is not None:
            # Проверяем что ML validator не в fallback режиме
            if hasattr(self.ml_validator, 'model_available') and \
                self.ml_validator.model_available:

              # Извлекаем признаки (если feature_pipeline доступен)
              if self.feature_pipeline is not None:
                feature_vector = await self.feature_pipeline.extract(
                  symbol=symbol,
                  orderbook_snapshot=orderbook,
                  orderbook_metrics=metrics,
                  candles=candles,
                  sr_levels=sr_levels
                )

                # Получаем предсказание
                prediction_response = await self.ml_validator.get_prediction(
                  symbol=symbol,
                  feature_vector=feature_vector
                )

                if prediction_response:
                  ml_prediction = {
                    'prediction': prediction_response.get('direction'),  # 'bullish'/'bearish'
                    'confidence': prediction_response.get('confidence', 0.0),
                    'expected_return': prediction_response.get('expected_return', 0.0),
                    'manipulation_risk': prediction_response.get('manipulation_risk', 0.0)
                  }

                  logger.debug(
                    f"[{timeframe.value}] {symbol} - "
                    f"ML Prediction: {ml_prediction['prediction']} "
                    f"(conf={ml_prediction['confidence']:.2f})"
                  )
              else:
                logger.debug(
                  f"[{timeframe.value}] {symbol} - "
                  f"Feature pipeline недоступен"
                )
            else:
              logger.debug(
                f"[{timeframe.value}] {symbol} - "
                f"ML модель недоступна"
              )
          else:
            logger.debug(
              f"[{timeframe.value}] {symbol} - "
              f"ML validator не инициализирован"
            )

        except Exception as e:
          logger.error(
            f"[{timeframe.value}] {symbol} - "
            f"Ошибка ML Prediction: {e}"
          )
          ml_prediction = None  # Продолжаем без ML
          warnings.append(f"ML Prediction failed: {str(e)}")

        # ============================================================
        # ШАГ 6: ЗАПУСК ВСЕХ СТРАТЕГИЙ (С НОВЫМИ ДАННЫМИ)
        # ============================================================
        strategy_results = self.strategy_manager.analyze_all_strategies(
          symbol=symbol,
          candles=candles,
          current_price=current_price,
          orderbook=orderbook,
          metrics=metrics,
          sr_levels=sr_levels,  # ✅ ИНТЕГРИРОВАНО
          volume_profile=volume_profile,  # ✅ ИНТЕГРИРОВАНО
          ml_prediction=ml_prediction  # ✅ ИНТЕГРИРОВАНО
        )

        # ============================================================
        # ШАГ 7: ГЕНЕРАЦИЯ АГРЕГИРОВАННОГО СИГНАЛА ТАЙМФРЕЙМА
        # ============================================================
        timeframe_signal = self._generate_timeframe_signal(
          symbol, timeframe, strategy_results, regime, current_price
        )

        # Обогащаем сигнал контекстом (если сигнал есть)
        if timeframe_signal and timeframe_signal.metadata is None:
          timeframe_signal.metadata = {}

        if timeframe_signal:
          # Добавляем S/R контекст
          if sr_levels:
            nearest = self.sr_detector.get_nearest_levels(symbol, current_price)
            if nearest.get("support"):
              timeframe_signal.metadata['nearest_support'] = {
                'price': nearest["support"].price,
                'strength': nearest["support"].strength
              }
            if nearest.get("resistance"):
              timeframe_signal.metadata['nearest_resistance'] = {
                'price': nearest["resistance"].price,
                'strength': nearest["resistance"].strength
              }

          # Добавляем VP контекст
          if volume_profile:
            timeframe_signal.metadata['volume_profile'] = {
              'poc_price': volume_profile.get('poc_price'),
              'in_value_area': (
                  volume_profile['value_area_low'] <= current_price <=
                  volume_profile['value_area_high']
              )
            }

          # Добавляем ML контекст
          if ml_prediction:
            timeframe_signal.metadata['ml_prediction'] = ml_prediction

        # ============================================================
        # ФИНАЛИЗАЦИЯ
        # ============================================================
        # Статистика
        self.total_analyses += 1
        self.analyses_by_timeframe[timeframe] += 1

        analysis_duration = (time.time() - start_time) * 1000

        result = TimeframeAnalysisResult(
          timeframe=timeframe,
          timestamp=int(datetime.now().timestamp() * 1000),
          current_price=current_price,
          candles_analyzed=len(candles),
          indicators=indicators,
          regime=regime,
          strategy_results=strategy_results,
          timeframe_signal=timeframe_signal,
          analysis_duration_ms=analysis_duration,
          warnings=warnings
        )

        logger.debug(
          f"✅ [{timeframe.value}] {symbol} анализ завершен: "
          f"regime={regime.market_regime.value}, "
          f"signal={timeframe_signal.signal_type.value if timeframe_signal else 'NONE'}, "
          f"duration={analysis_duration:.1f}ms, "
          f"sr_levels={len(sr_levels) if sr_levels else 0}, "
          f"vp={'yes' if volume_profile else 'no'}, "
          f"ml={'yes' if ml_prediction else 'no'}"
        )

        return result

      except Exception as e:
        logger.error(
          f"Ошибка анализа [{timeframe.value}] {symbol}: {e}",
          exc_info=True
        )

        # Возвращаем пустой результат с ошибкой
        return TimeframeAnalysisResult(
          timeframe=timeframe,
          timestamp=int(datetime.now().timestamp() * 1000),
          current_price=current_price,
          candles_analyzed=len(candles),
          indicators=TimeframeIndicators(),
          regime=TimeframeRegimeInfo(
            market_regime=MarketRegime.RANGING,
            volatility_regime=VolatilityRegime.NORMAL,
            trend_direction=0,
            trend_strength=0.0,
            volatility_percentile=0.5,
            normalized_atr=0.0
          ),
          strategy_results=[],
          warnings=[f"Analysis error: {str(e)}"]
        )

  def get_sr_statistics(self) -> Dict:
    """Получить статистику S/R Detector."""
    return self.sr_detector.get_statistics()

  def get_volume_profile(
      self,
      symbol: str,
      timeframe: Timeframe
  ) -> Optional[VolumeProfile]:
    """Получить Volume Profile для символа и таймфрейма."""
    return self.volume_profiles.get(symbol, {}).get(timeframe)

  def clear_cache(self, symbol: Optional[str] = None):
    """
    Очистить кэш индикаторов и профилей.

    Args:
        symbol: Если указан - очистить только для этого символа
    """
    if symbol:
      # Очистка для конкретного символа
      keys_to_remove = [
        key for key in self._indicators_cache.keys()
        if key[0] == symbol
      ]
      for key in keys_to_remove:
        del self._indicators_cache[key]

      keys_to_remove = [
        key for key in self._regime_cache.keys()
        if key[0] == symbol
      ]
      for key in keys_to_remove:
        del self._regime_cache[key]

      if symbol in self.volume_profiles:
        del self.volume_profiles[symbol]
      if symbol in self.last_vp_update:
        del self.last_vp_update[symbol]
    else:
      # Полная очистка
      self._indicators_cache.clear()
      self._regime_cache.clear()
      self.volume_profiles.clear()
      self.last_vp_update.clear()

    logger.info(f"Кэш очищен" + (f" для {symbol}" if symbol else ""))

  def _calculate_tf_specific_indicators(
      self,
      symbol: str,
      timeframe: Timeframe,
      candles: List[Candle]
  ) -> TimeframeIndicators:
    """
    Рассчитать индикаторы, специфичные для таймфрейма.

    ОБНОВЛЕНО: Добавлена timestamp валидация кэша для гарантии актуальности.

    Разные таймфреймы используют разные периоды и индикаторы:
    - 1m: быстрые EMA, micro-структура
    - 5m: стандартные oscillators
    - 15m: средние MA, Bollinger
    - 1h: долгосрочные MA, Ichimoku

    Args:
        symbol: Торговая пара
        timeframe: Таймфрейм
        candles: Свечи

    Returns:
        TimeframeIndicators с рассчитанными значениями
    """
    # Проверяем кэш с timestamp валидацией
    cache_key = (symbol, timeframe)

    # Валидация кэша: проверяем timestamp последней свечи и TTL
    if candles:
      last_candle_ts = candles[-1].timestamp
      cached = self._indicators_cache.get(cache_key)

      if cached:
        # Получаем TTL для этого таймфрейма
        max_age_ms = self.cache_ttl_ms.get(timeframe)

        # Проверяем актуальность кэша
        if cached.is_valid(last_candle_ts, max_age_ms):
          # Кэш актуален - возвращаем закэшированные индикаторы
          self.cache_hits += 1
          logger.debug(
            f"[CACHE HIT] {symbol} {timeframe.value}: "
            f"indicators from cache (hits={self.cache_hits}, misses={self.cache_misses})"
          )
          return cached.indicators
        else:
          # Кэш устарел
          self.cache_invalidations += 1
          logger.debug(
            f"[CACHE INVALID] {symbol} {timeframe.value}: "
            f"candle_ts={last_candle_ts}, cached_ts={cached.candle_timestamp}, "
            f"age={(int(time.time() * 1000) - cached.calculated_at) / 1000:.1f}s"
          )

      # Кэш отсутствует или устарел
      self.cache_misses += 1

    if not candles or len(candles) < 20:
      logger.warning(
        f"Недостаточно данных для индикаторов: {len(candles)} свечей"
      )
      return TimeframeIndicators()

    indicators = TimeframeIndicators()

    # Извлекаем OHLCV
    closes = np.array([c.close for c in candles])
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    volumes = np.array([c.volume for c in candles])

    periods = self.lookback_periods.get(timeframe, {})

    try:
      # === Trend Indicators ===
      sma_fast_period = periods.get('sma_fast', 20)
      sma_slow_period = periods.get('sma_slow', 50)

      if len(closes) >= sma_fast_period:
        indicators.sma_fast = float(np.mean(closes[-sma_fast_period:]))

      if len(closes) >= sma_slow_period:
        indicators.sma_slow = float(np.mean(closes[-sma_slow_period:]))

      # EMA
      ema_fast_period = periods.get('ema_fast', 12)
      ema_slow_period = periods.get('ema_slow', 26)

      if len(closes) >= ema_fast_period:
        indicators.ema_fast = self._calculate_ema(closes, ema_fast_period)

      if len(closes) >= ema_slow_period:
        indicators.ema_slow = self._calculate_ema(closes, ema_slow_period)

      # === Momentum Indicators ===
      rsi_period = periods.get('rsi', 14)
      if len(closes) >= rsi_period + 1:
        indicators.rsi = self._calculate_rsi(closes, rsi_period)

      # Stochastic
      stoch_period = periods.get('stochastic', 14)
      if len(highs) >= stoch_period:
        indicators.stochastic_k = self._calculate_stochastic_k(
          closes, highs, lows, stoch_period
        )
        # %D - 3-period SMA of %K
        if indicators.stochastic_k is not None:
          # Упрощенно - можно улучшить
          indicators.stochastic_d = indicators.stochastic_k

      # MACD
      if indicators.ema_fast and indicators.ema_slow:
        indicators.macd = indicators.ema_fast - indicators.ema_slow
        # Signal line - EMA(9) of MACD
        # Упрощенно - в production использовать полный расчет
        indicators.macd_signal = indicators.macd * 0.9
        indicators.macd_histogram = indicators.macd - indicators.macd_signal

      # === Volatility Indicators ===
      atr_period = periods.get('atr', 14)
      if len(candles) >= atr_period + 1:
        indicators.atr = self._calculate_atr(candles, atr_period)
        if indicators.atr and closes[-1] > 0:
          indicators.atr_percent = (indicators.atr / closes[-1]) * 100

      # Bollinger Bands
      bb_period = periods.get('bollinger', 20)
      if len(closes) >= bb_period:
        bb_middle = np.mean(closes[-bb_period:])
        bb_std = np.std(closes[-bb_period:])

        indicators.bollinger_middle = float(bb_middle)
        indicators.bollinger_upper = float(bb_middle + 2 * bb_std)
        indicators.bollinger_lower = float(bb_middle - 2 * bb_std)
        indicators.bollinger_width = float(
          (indicators.bollinger_upper - indicators.bollinger_lower) / bb_middle * 100
        )

      # === Volume Indicators ===
      vol_sma_period = periods.get('volume_sma', 20)
      if len(volumes) >= vol_sma_period:
        indicators.volume_sma = float(np.mean(volumes[-vol_sma_period:]))
        if indicators.volume_sma > 0:
          indicators.volume_ratio = float(volumes[-1] / indicators.volume_sma)

      # OBV (On-Balance Volume)
      indicators.obv = self._calculate_obv(candles)

      # VWAP (Volume Weighted Average Price)
      indicators.vwap = self._calculate_vwap(candles)

      # === ADX (для определения силы тренда) ===
      adx_period = periods.get('adx', 14)
      if len(candles) >= adx_period + 1:
        adx_result = self._calculate_adx(candles, adx_period)
        if adx_result:
          indicators.adx = adx_result['adx']
          indicators.adx_di_plus = adx_result['di_plus']
          indicators.adx_di_minus = adx_result['di_minus']

      # === Swing Highs/Lows ===
      swing_lookback = 10
      if len(candles) >= swing_lookback:
        indicators.swing_high = self._find_swing_high(highs, swing_lookback)
        indicators.swing_low = self._find_swing_low(lows, swing_lookback)

      # === Ichimoku (только для высших TF) ===
      if periods.get('ichimoku') and timeframe in [Timeframe.H1, Timeframe.H4, Timeframe.D1]:
        if len(candles) >= 52:  # Минимум для Ichimoku
          ichimoku = self._calculate_ichimoku(candles)
          if ichimoku:
            indicators.ichimoku_conversion = ichimoku['conversion']
            indicators.ichimoku_base = ichimoku['base']
            indicators.ichimoku_span_a = ichimoku['span_a']
            indicators.ichimoku_span_b = ichimoku['span_b']

      # Кэшируем результат с timestamp метаданными
      if candles:
        last_candle_ts = candles[-1].timestamp
        current_time_ms = int(time.time() * 1000)

        self._indicators_cache[cache_key] = CachedIndicators(
          indicators=indicators,
          candle_timestamp=last_candle_ts,
          calculated_at=current_time_ms
        )

        logger.debug(
          f"[CACHE STORE] {symbol} {timeframe.value}: "
          f"indicators cached with ts={last_candle_ts}"
        )

    except Exception as e:
      logger.error(f"Ошибка расчета индикаторов {timeframe.value}: {e}")

    return indicators

  def _detect_tf_regime(
      self,
      symbol: str,
      timeframe: Timeframe,
      candles: List[Candle],
      indicators: TimeframeIndicators
  ) -> TimeframeRegimeInfo:
    """
    Определить режим рынка для таймфрейма.

    Использует:
    - ADX для силы тренда
    - MA для направления
    - ATR для волатильности
    - Price action patterns

    Args:
        symbol: Торговая пара
        timeframe: Таймфрейм
        candles: Свечи
        indicators: Рассчитанные индикаторы

    Returns:
        TimeframeRegimeInfo с режимом рынка
    """
    if not candles or len(candles) < 20:
      return TimeframeRegimeInfo(
        market_regime=MarketRegime.RANGING,
        volatility_regime=VolatilityRegime.NORMAL,
        trend_direction=0,
        trend_strength=0.0,
        volatility_percentile=0.5,
        normalized_atr=0.0,
        regime_confidence=0.0
      )

    current_price = candles[-1].close

    # === Определение Тренда ===
    trend_direction = 0
    trend_strength = 0.0

    # Используем SMA для направления
    if indicators.sma_fast and indicators.sma_slow:
      if indicators.sma_fast > indicators.sma_slow:
        trend_direction = 1  # Uptrend
      elif indicators.sma_fast < indicators.sma_slow:
        trend_direction = -1  # Downtrend

    # Используем ADX для силы тренда
    if indicators.adx:
      # ADX интерпретация:
      # < 20: weak/no trend
      # 20-25: developing trend
      # 25-50: strong trend
      # > 50: very strong trend
      trend_strength = min(indicators.adx / 50.0, 1.0)

    # Определяем market regime
    market_regime = MarketRegime.RANGING

    if indicators.adx:
      if indicators.adx > 25:  # Strong trend
        if trend_direction > 0:
          market_regime = MarketRegime.STRONG_UPTREND
        elif trend_direction < 0:
          market_regime = MarketRegime.STRONG_DOWNTREND
      elif indicators.adx > 15:  # Weak trend
        if trend_direction > 0:
          market_regime = MarketRegime.WEAK_UPTREND
        elif trend_direction < 0:
          market_regime = MarketRegime.WEAK_DOWNTREND
      else:  # ADX < 15
        market_regime = MarketRegime.RANGING

    # === Определение Волатильности ===
    volatility_regime = VolatilityRegime.NORMAL
    volatility_percentile = 0.5
    normalized_atr = 0.0

    if indicators.atr and indicators.atr_percent:
      normalized_atr = indicators.atr_percent

      # Рассчитываем percentile ATR относительно истории
      if len(candles) >= 100:
        # Вычисляем ATR для последних 100 периодов
        historical_atrs = []
        for i in range(len(candles) - 100, len(candles)):
          if i >= 14:  # Минимум для ATR
            atr_val = self._calculate_atr(candles[i - 13:i + 1], 14)
            if atr_val:
              historical_atrs.append(atr_val)

        if historical_atrs:
          # Percentile текущего ATR
          current_atr = indicators.atr
          volatility_percentile = (
              sum(1 for x in historical_atrs if x < current_atr) /
              len(historical_atrs)
          )

          # Классификация
          if volatility_percentile > 0.80:
            volatility_regime = VolatilityRegime.HIGH
          elif volatility_percentile < 0.20:
            volatility_regime = VolatilityRegime.LOW
          else:
            volatility_regime = VolatilityRegime.NORMAL

    # === Детекция паттернов ===
    is_breakout = False
    is_consolidation = False
    is_reversal_pattern = False

    # Breakout detection
    if indicators.bollinger_upper and indicators.bollinger_lower:
      if current_price > indicators.bollinger_upper:
        is_breakout = True
      elif current_price < indicators.bollinger_lower:
        is_breakout = True

    # Consolidation detection (узкие Bollinger Bands)
    if indicators.bollinger_width:
      if indicators.bollinger_width < 2.0:  # Узкие bands
        is_consolidation = True

    # Reversal pattern (простая версия - MACD divergence)
    if indicators.macd_histogram:
      # Если histogram меняет знак - возможный разворот
      # Упрощенная логика
      pass

    # === Confidence в определении режима ===
    regime_confidence = 0.5  # Base

    # Повышаем confidence если есть подтверждение от нескольких индикаторов
    confirmations = 0

    if indicators.adx and indicators.adx > 20:
      confirmations += 1

    if indicators.sma_fast and indicators.sma_slow:
      if abs(indicators.sma_fast - indicators.sma_slow) / indicators.sma_slow > 0.02:
        confirmations += 1

    if indicators.rsi:
      if (trend_direction > 0 and indicators.rsi > 50) or \
          (trend_direction < 0 and indicators.rsi < 50):
        confirmations += 1

    regime_confidence = min(0.5 + (confirmations * 0.15), 0.95)

    regime_info = TimeframeRegimeInfo(
      market_regime=market_regime,
      volatility_regime=volatility_regime,
      trend_direction=trend_direction,
      trend_strength=trend_strength,
      volatility_percentile=volatility_percentile,
      normalized_atr=normalized_atr,
      is_breakout=is_breakout,
      is_consolidation=is_consolidation,
      is_reversal_pattern=is_reversal_pattern,
      regime_confidence=regime_confidence
    )

    logger.debug(
      f"[{timeframe.value}] {symbol} режим: "
      f"{market_regime.value}, volatility={volatility_regime.value}, "
      f"confidence={regime_confidence:.2f}"
    )

    return regime_info

  def _generate_timeframe_signal(
      self,
      symbol: str,
      timeframe: Timeframe,
      strategy_results: List[StrategyResult],
      regime: TimeframeRegimeInfo,
      current_price: float
  ) -> Optional[TradingSignal]:
    """
    Генерировать агрегированный сигнал таймфрейма.

    Комбинирует сигналы от стратегий с учетом рыночного режима.

    Args:
        symbol: Торговая пара
        timeframe: Таймфрейм
        strategy_results: Результаты стратегий
        regime: Режим рынка
        current_price: Текущая цена

    Returns:
        TradingSignal или None
    """
    # Фильтруем только стратегии с сигналами
    results_with_signals = [
      r for r in strategy_results
      if r.signal and r.signal.signal_type != SignalType.HOLD
    ]

    if not results_with_signals:
      return None

    # Подсчитываем голоса
    buy_signals = [r for r in results_with_signals if r.signal.signal_type == SignalType.BUY]
    sell_signals = [r for r in results_with_signals if r.signal.signal_type == SignalType.SELL]

    # Простое большинство
    if len(buy_signals) > len(sell_signals):
      signal_type = SignalType.BUY
      contributing = buy_signals
    elif len(sell_signals) > len(buy_signals):
      signal_type = SignalType.SELL
      contributing = sell_signals
    else:
      # Равенство - нет консенсуса
      return None

    # Вычисляем среднюю confidence
    avg_confidence = np.mean([r.signal.confidence for r in contributing])

    # Модифицируем confidence на основе режима рынка
    regime_modifier = 1.0

    # Если сигнал согласуется с трендом - увеличиваем confidence
    if signal_type == SignalType.BUY and regime.trend_direction > 0:
      regime_modifier = 1.0 + (regime.trend_strength * 0.2)
    elif signal_type == SignalType.SELL and regime.trend_direction < 0:
      regime_modifier = 1.0 + (regime.trend_strength * 0.2)
    # Если против тренда - снижаем
    elif signal_type == SignalType.BUY and regime.trend_direction < 0:
      regime_modifier = 0.8
    elif signal_type == SignalType.SELL and regime.trend_direction > 0:
      regime_modifier = 0.8

    final_confidence = min(avg_confidence * regime_modifier, 0.95)

    # Создаем агрегированный сигнал
    from models.signal import SignalSource, SignalStrength

    signal = TradingSignal(
      symbol=symbol,
      signal_type=signal_type,
      source=SignalSource.STRATEGY,
      strength=SignalStrength.MEDIUM,  # TODO: определять по confidence
      price=current_price,
      confidence=final_confidence,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason=(
        f"[{timeframe.value}] Consensus: {len(contributing)}/{len(results_with_signals)} "
        f"strategies agree, regime={regime.market_regime.value}"
      ),
      metadata={
        'timeframe': timeframe.value,
        'regime': regime.market_regime.value,
        'regime_confidence': regime.regime_confidence,
        'trend_strength': regime.trend_strength,
        'contributing_strategies': [r.strategy_name for r in contributing]
      }
    )

    return signal

  # ==================== Helper Methods ====================

  def _calculate_ema(self, values: np.ndarray, period: int) -> float:
    """Экспоненциальная скользящая средняя."""
    if len(values) < period:
      return float(np.mean(values))

    multiplier = 2 / (period + 1)
    ema = np.mean(values[:period])  # Начальная SMA

    for price in values[period:]:
      ema = (price * multiplier) + (ema * (1 - multiplier))

    return float(ema)

  def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index."""
    if len(closes) < period + 1:
      return 50.0

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
      return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return float(rsi)

  def _calculate_stochastic_k(
      self,
      closes: np.ndarray,
      highs: np.ndarray,
      lows: np.ndarray,
      period: int = 14
  ) -> float:
    """Stochastic %K."""
    if len(closes) < period:
      return 50.0

    lowest_low = np.min(lows[-period:])
    highest_high = np.max(highs[-period:])
    current_close = closes[-1]

    if highest_high == lowest_low:
      return 50.0

    k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100

    return float(k)

  def _calculate_atr(self, candles: List[Candle], period: int = 14) -> Optional[float]:
    """Average True Range."""
    if len(candles) < period + 1:
      return None

    true_ranges = []

    for i in range(1, len(candles)):
      high = candles[i].high
      low = candles[i].low
      prev_close = candles[i - 1].close

      tr = max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
      )

      true_ranges.append(tr)

    if len(true_ranges) < period:
      return None

    atr = np.mean(true_ranges[-period:])

    return float(atr)

  def _calculate_adx(
      self,
      candles: List[Candle],
      period: int = 14
  ) -> Optional[Dict]:
    """Average Directional Index с +DI и -DI."""
    if len(candles) < period + 1:
      return None

    # Расчет +DM и -DM
    plus_dm_values = []
    minus_dm_values = []

    for i in range(1, len(candles)):
      high_diff = candles[i].high - candles[i - 1].high
      low_diff = candles[i - 1].low - candles[i].low

      plus_dm = high_diff if high_diff > low_diff and high_diff > 0 else 0
      minus_dm = low_diff if low_diff > high_diff and low_diff > 0 else 0

      plus_dm_values.append(plus_dm)
      minus_dm_values.append(minus_dm)

    # Сглаживание через SMA (упрощенно)
    smoothed_plus_dm = np.mean(plus_dm_values[-period:])
    smoothed_minus_dm = np.mean(minus_dm_values[-period:])

    # Расчет ATR
    atr = self._calculate_atr(candles, period)
    if not atr or atr == 0:
      return None

    # +DI и -DI
    di_plus = (smoothed_plus_dm / atr) * 100
    di_minus = (smoothed_minus_dm / atr) * 100

    # DX
    di_diff = abs(di_plus - di_minus)
    di_sum = di_plus + di_minus

    if di_sum == 0:
      return None

    dx = (di_diff / di_sum) * 100

    # ADX - сглаживание DX (упрощенно используем текущий DX)
    adx = dx

    return {
      'adx': float(adx),
      'di_plus': float(di_plus),
      'di_minus': float(di_minus)
    }

  def _calculate_obv(self, candles: List[Candle]) -> float:
    """On-Balance Volume."""
    if len(candles) < 2:
      return 0.0

    obv = 0.0

    for i in range(1, len(candles)):
      if candles[i].close > candles[i - 1].close:
        obv += candles[i].volume
      elif candles[i].close < candles[i - 1].close:
        obv -= candles[i].volume

    return float(obv)

  def _calculate_vwap(self, candles: List[Candle]) -> float:
    """Volume Weighted Average Price."""
    if not candles:
      return 0.0

    total_volume = sum(c.volume for c in candles)

    if total_volume == 0:
      return candles[-1].close

    vwap = sum(
      ((c.high + c.low + c.close) / 3) * c.volume
      for c in candles
    ) / total_volume

    return float(vwap)

  def _find_swing_high(self, highs: np.ndarray, lookback: int = 10) -> Optional[float]:
    """Найти последний swing high."""
    if len(highs) < lookback * 2:
      return None

    # Простая логика: локальный максимум в окне lookback
    for i in range(len(highs) - lookback - 1, lookback, -1):
      is_swing_high = all(
        highs[i] > highs[j]
        for j in range(i - lookback, i + lookback + 1)
        if j != i
      )

      if is_swing_high:
        return float(highs[i])

    return None

  def _find_swing_low(self, lows: np.ndarray, lookback: int = 10) -> Optional[float]:
    """Найти последний swing low."""
    if len(lows) < lookback * 2:
      return None

    for i in range(len(lows) - lookback - 1, lookback, -1):
      is_swing_low = all(
        lows[i] < lows[j]
        for j in range(i - lookback, i + lookback + 1)
        if j != i
      )

      if is_swing_low:
        return float(lows[i])

    return None

  def _calculate_ichimoku(self, candles: List[Candle]) -> Optional[Dict]:
    """Ichimoku Cloud (упрощенная версия)."""
    if len(candles) < 52:
      return None

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    highs_9 = [c.high for c in candles[-9:]]
    lows_9 = [c.low for c in candles[-9:]]
    conversion = (max(highs_9) + min(lows_9)) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    highs_26 = [c.high for c in candles[-26:]]
    lows_26 = [c.low for c in candles[-26:]]
    base = (max(highs_26) + min(lows_26)) / 2

    # Senkou Span A (Leading Span A): (Conversion + Base) / 2
    span_a = (conversion + base) / 2

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    highs_52 = [c.high for c in candles[-52:]]
    lows_52 = [c.low for c in candles[-52:]]
    span_b = (max(highs_52) + min(lows_52)) / 2

    return {
      'conversion': float(conversion),
      'base': float(base),
      'span_a': float(span_a),
      'span_b': float(span_b)
    }

  def get_statistics(self) -> Dict:
    """Получить статистику анализатора."""
    return {
      'total_analyses': self.total_analyses,
      'analyses_by_timeframe': {
        tf.value: count
        for tf, count in self.analyses_by_timeframe.items()
      },
      'cache_size': {
        'indicators': len(self._indicators_cache),
        'regime': len(self._regime_cache)
      }
    }