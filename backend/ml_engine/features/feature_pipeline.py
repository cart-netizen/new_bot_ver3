"""
Feature Pipeline - оркестратор всех feature extractors.

Объединяет признаки из OrderBook, Candles и Indicators в единый вектор.
Поддерживает Multi-Channel Representation для ML моделей.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from strategy.trade_manager import TradeManager

if TYPE_CHECKING:
    from ml_engine.detection.sr_level_detector import SRLevel
else:
    SRLevel = None

from dataclasses import dataclass, field
import numpy as np
from sklearn.preprocessing import StandardScaler
import asyncio

from core.logger import get_logger
from core.periodic_logger import periodic_logger
from ml_engine.detection.sr_level_detector import SRLevel
from models.orderbook import OrderBookSnapshot, OrderBookMetrics
from ml_engine.features.orderbook_feature_extractor import (
  OrderBookFeatureExtractor,
  OrderBookFeatures
)
from ml_engine.features.candle_feature_extractor import (
  CandleFeatureExtractor,
  CandleFeatures,
  Candle
)
from ml_engine.features.indicator_feature_extractor import (
  IndicatorFeatureExtractor,
  IndicatorFeatures
)
from ml_engine.features.feature_scaler_manager import (
  FeatureScalerManager,
  ScalerConfig
)
# Для type hints без circular imports
if TYPE_CHECKING:
    from models.orderbook import OrderBookMetrics
    from ml_engine.detection.sr_level_detector import SRLevel
logger = get_logger(__name__)


@dataclass
class FeatureVector:
  """
  Объединенный вектор признаков из всех источников.
  Поддерживает Multi-Channel Representation.
  """

  symbol: str
  timestamp: int

  # Отдельные каналы признаков
  orderbook_features: OrderBookFeatures
  candle_features: CandleFeatures
  indicator_features: IndicatorFeatures

  # Метаданные
  feature_count: int
  version: str = "v1.0.0"

  metadata: Dict = field(default_factory=dict)  # ← Дополнительный контекст

  def to_array(self) -> np.ndarray:
    """
    Преобразует в единый numpy array (concatenated).

    Returns:
        np.ndarray: shape (feature_count,)
    """
    return np.concatenate([
      self.orderbook_features.to_array(),
      self.candle_features.to_array(),
      self.indicator_features.to_array()
    ])

  def to_channels(self) -> Dict[str, np.ndarray]:
    """
    Преобразует в multi-channel representation.

    Returns:
        Dict с отдельными каналами для каждого типа признаков
    """
    return {
      "orderbook": self.orderbook_features.to_array(),
      "candle": self.candle_features.to_array(),
      "indicator": self.indicator_features.to_array()
    }

  def to_dict(self) -> Dict[str, any]:
    """Преобразование в словарь для Redis/JSON"""
    return {
      "symbol": self.symbol,
      "timestamp": self.timestamp,
      "version": self.version,
      "orderbook": self.orderbook_features.to_dict(),
      "candle": self.candle_features.to_dict(),
      "indicator": self.indicator_features.to_dict(),
      "feature_count": self.feature_count
    }

  def get_feature_names(self) -> List[str]:
    """Возвращает имена всех признаков"""
    ob_names = [f"ob_{k}" for k in self.orderbook_features.to_dict().keys()
                if k not in ['symbol', 'timestamp']]
    candle_names = [f"candle_{k}" for k in self.candle_features.to_dict().keys()
                    if k not in ['symbol', 'timestamp']]
    indicator_names = [f"ind_{k}" for k in self.indicator_features.to_dict().keys()
                       if k not in ['symbol', 'timestamp']]

    return ob_names + candle_names + indicator_names


class FeaturePipeline:
  """
  Главный оркестратор извлечения признаков.

  Координирует OrderBookFeatureExtractor, CandleFeatureExtractor
  и IndicatorFeatureExtractor для создания полного FeatureVector.
  """

  def __init__(
      self,
      symbol: str,
      normalize: bool = True,
      cache_enabled: bool = False,
      trade_manager=None
  ):
    """
    Args:
        symbol: Торговая пара
        normalize: Применять ли нормализацию к признакам
        cache_enabled: Использовать ли кэширование (Redis)
        trade_manager: Optional TradeManager для реальных market trades
    """
    self.symbol = symbol
    self.normalize = normalize
    self.cache_enabled = cache_enabled

    # Инициализация extractors
    self.orderbook_extractor = OrderBookFeatureExtractor(symbol, trade_manager=trade_manager)
    self.candle_extractor = CandleFeatureExtractor(symbol)
    self.indicator_extractor = IndicatorFeatureExtractor(symbol)

    # ============================================================================
    # PROFESSIONAL FEATURE SCALER MANAGER
    # Replaces simplified StandardScaler (old lines 145-149)
    # ============================================================================
    self.scaler_manager: Optional[FeatureScalerManager] = None
    if normalize:
      # Create professional multi-channel scaler
      scaler_config = ScalerConfig(
        orderbook_scaler_type="standard",  # StandardScaler for orderbook
        candle_scaler_type="robust",       # RobustScaler for candles (outlier-resistant)
        indicator_scaler_type="minmax",    # MinMaxScaler for bounded indicators
        min_samples_for_fitting=100,
        refit_interval_samples=1000,
        auto_save=True,
        save_interval_samples=500
      )
      self.scaler_manager = FeatureScalerManager(symbol=symbol, config=scaler_config)
      logger.info(
        f"{symbol} | Professional FeatureScalerManager initialized: "
        f"multi-channel, persistent state, auto-save"
      )

    # Кэш последних признаков
    self._last_feature_vector: Optional[FeatureVector] = None
    self._cache: Dict[int, FeatureVector] = {}

    logger.info(
      f"FeaturePipeline инициализирован для {symbol}, "
      f"normalize={normalize}, cache={cache_enabled}"
    )

  async def extract_features(
      self,
      orderbook_snapshot: OrderBookSnapshot,
      candles: List[Candle],
      prev_orderbook: Optional[OrderBookSnapshot] = None,
      prev_candle: Optional[Candle] = None
  ) -> FeatureVector:
    """
    Извлекает все признаки и создает FeatureVector.

    Args:
        orderbook_snapshot: Текущий снимок стакана
        candles: Список свечей для индикаторов
        prev_orderbook: Предыдущий снимок стакана
        prev_candle: Предыдущая свеча

    Returns:
        FeatureVector: Полный вектор признаков
    """
    logger.debug(f"{self.symbol} | Извлечение признаков через pipeline")

    try:
      # Проверяем кэш
      cache_key = orderbook_snapshot.timestamp
      if self.cache_enabled and cache_key in self._cache:
        logger.debug(f"{self.symbol} | Признаки взяты из кэша")
        return self._cache[cache_key]

      # 1. Извлекаем признаки из OrderBook
      orderbook_features = self.orderbook_extractor.extract(
        orderbook_snapshot,
        prev_orderbook
      )
      logger.debug(f"{self.symbol} | OrderBook признаки извлечены")

      # 2. Извлекаем признаки из Candles
      if candles:
        current_candle = candles[-1]
        candle_features = self.candle_extractor.extract(
          current_candle,
          prev_candle
        )
        logger.debug(f"{self.symbol} | Candle признаки извлечены")
      else:
        # Создаем дефолтные если свечей нет
        logger.warning(f"{self.symbol} | Нет свечей, используем дефолт")
        candle_features = self._create_default_candle_features(
          orderbook_snapshot
        )

      # 3. Извлекаем индикаторы
      if len(candles) >= 50:
        indicator_features = self.indicator_extractor.extract(candles)
        logger.debug(f"{self.symbol} | Indicator признаки извлечены")
      else:
        logger.warning(
          f"{self.symbol} | Недостаточно свечей для индикаторов "
          f"({len(candles)} < 50)"
        )
        indicator_features = self.indicator_extractor._create_default_features(
          candles[-1] if candles else None
        )

      # 4. Объединяем в FeatureVector
      feature_vector = FeatureVector(
        symbol=self.symbol,
        timestamp=orderbook_snapshot.timestamp,
        orderbook_features=orderbook_features,
        candle_features=candle_features,
        indicator_features=indicator_features,
        feature_count=50 + 25 + 35,  # 110 признаков
        metadata={}
      )

      # 5. Нормализация (если включена)
      if self.normalize:
        feature_vector = await self._normalize_features(feature_vector)

      # 6. Сохраняем в кэш
      if self.cache_enabled:
        self._cache[cache_key] = feature_vector
        # Ограничиваем размер кэша
        if len(self._cache) > 100:
          oldest_key = min(self._cache.keys())
          del self._cache[oldest_key]

      self._last_feature_vector = feature_vector

      key = f"orderbook_features_{self.symbol}"
      should_log, count = periodic_logger.should_log(key, every_n=500, first_n=1)

      if should_log:
        logger.info(
          f"{self.symbol} | Pipeline завершен: {feature_vector.feature_count} признаков, "
          f"timestamp={feature_vector.timestamp}"
        )
      #
      # logger.info(
      #   f"{self.symbol} | Pipeline завершен: {feature_vector.feature_count} признаков, "
      #   f"timestamp={feature_vector.timestamp}"
      # )

      return feature_vector

    except Exception as e:
      logger.error(f"{self.symbol} | Ошибка в pipeline: {e}")
      raise



  async def _normalize_features(
      self,
      feature_vector: FeatureVector
  ) -> FeatureVector:
    """
    Нормализует признаки используя профессиональный FeatureScalerManager.

    ============================================================================
    PROFESSIONAL FEATURE NORMALIZATION
    Replaces broken normalization (old lines 270-319)
    ============================================================================

    Previous Issue:
    - Normalized data was calculated but NEVER USED
    - Original (raw) feature_vector was returned
    - ML models received unnormalized data

    New Implementation:
    - Uses multi-channel scalers (OrderBook, Candle, Indicator)
    - Creates NEW FeatureVector with normalized values
    - Preserves original values in metadata
    - Proper batch fitting on historical data

    Returns:
        FeatureVector with NORMALIZED features (not original!)
    """
    if not self.scaler_manager:
      logger.debug(f"{self.symbol} | Scaler manager not initialized, skipping normalization")
      return feature_vector

    try:
      # ========================================================================
      # Use professional FeatureScalerManager
      # ========================================================================
      scaled_vector = await self.scaler_manager.scale_features(
        feature_vector=feature_vector,
        update_history=True
      )

      if scaled_vector is None:
        logger.warning(f"{self.symbol} | Scaling failed, returning original vector")
        return feature_vector

      # ========================================================================
      # CRITICAL: Return SCALED vector, not original!
      # ========================================================================
      logger.debug(
        f"{self.symbol} | Features professionally normalized using multi-channel scalers"
      )

      return scaled_vector  # ← Returns normalized data!

    except Exception as e:
      logger.error(f"{self.symbol} | Error in professional normalization: {e}", exc_info=True)
      # Fallback: return original (unnormalized)
      return feature_vector

  def _create_default_candle_features(
      self,
      orderbook: OrderBookSnapshot
  ) -> CandleFeatures:
    """Создает дефолтные candle features на основе orderbook"""
    mid_price = orderbook.mid_price or 0.0

    return CandleFeatures(
      symbol=self.symbol,
      timestamp=orderbook.timestamp,
      open=mid_price,
      high=mid_price,
      low=mid_price,
      close=mid_price,
      volume=0.0,
      typical_price=mid_price,
      returns=0.0,
      log_returns=0.0,
      high_low_range=0.0,
      close_open_diff=0.0,
      upper_shadow=0.0,
      lower_shadow=0.0,
      body_size=0.0,
      realized_volatility=0.0,
      parkinson_volatility=0.0,
      garman_klass_volatility=0.0,
      volume_ma_ratio=1.0,
      volume_change_rate=0.0,
      price_volume_trend=0.0,
      volume_weighted_price=mid_price,
      money_flow=0.0,
      doji_strength=0.0,
      hammer_strength=0.0,
      engulfing_strength=0.0,
      gap_size=0.0
    )

  def get_feature_names(self) -> List[str]:
    """Возвращает список всех имен признаков"""
    if self._last_feature_vector:
      return self._last_feature_vector.get_feature_names()
    return []

  def get_feature_importance(self) -> Dict[str, float]:
    """
    Вычисляет важность признаков на основе вариации.

    ============================================================================
    PROFESSIONAL FEATURE IMPORTANCE
    Replaces empty stub (old lines 364-373)
    ============================================================================

    Uses variance-based importance from FeatureScalerManager.

    Returns:
        Dict[feature_name, importance_score] sorted by importance

    Example:
    --------
    importance = pipeline.get_feature_importance()
    for feature, score in list(importance.items())[:10]:
    print(f"{feature}: {score:.4f}")
    """
    if not self.scaler_manager:
      logger.warning(f"{self.symbol} | Scaler manager not available for feature importance")
      return {}

    return self.scaler_manager.get_feature_importance()

  async def warmup(
      self,
      historical_feature_vectors: List[FeatureVector],
      force_refit: bool = False
  ) -> bool:
    """
    Прогрев pipeline и scaler на исторических данных.

    ============================================================================
    PROFESSIONAL WARMUP
    Replaces empty stub (old lines 375-392)
    ============================================================================

    Previous Issue:
    - Method did nothing
    - No actual warm-up on historical data
    - Scalers not fitted before use

    New Implementation:
    - Fits scalers on historical feature vectors (batch fitting)
    - Requires actual historical data, not just a number
    - Saves scaler state for persistence

    Args:
        historical_feature_vectors: List of FeatureVector objects from history
        force_refit: Force refitting even if already fitted

    Returns:
        True if successful, False otherwise

    Example:
    --------
    # Load last 1000 feature vectors from database/cache
    historical_vectors = load_historical_features(symbol, count=1000)
    success = await pipeline.warmup(historical_vectors)
    if success:
    print("Pipeline ready for live trading")
    """
    logger.info(
      f"{self.symbol} | Warming up pipeline on "
      f"{len(historical_feature_vectors)} historical samples"
    )

    if not self.scaler_manager:
      logger.warning(f"{self.symbol} | No scaler manager, skipping warmup")
      return False

    if not historical_feature_vectors:
      logger.warning(f"{self.symbol} | No historical data provided for warmup")
      return False

    # Use professional FeatureScalerManager warmup
    success = await self.scaler_manager.warmup(
      feature_vectors=historical_feature_vectors,
      force_refit=force_refit
    )

    if success:
      logger.info(f"{self.symbol} | ✅ Pipeline warmed up successfully")
    else:
      logger.warning(f"{self.symbol} | ❌ Pipeline warmup failed")

    return success


class MultiSymbolFeaturePipeline:
  """
  Управляет pipeline для нескольких символов одновременно.
  """

  def __init__(
      self,
      symbols: List[str],
      normalize: bool = True,
      cache_enabled: bool = True,
      trade_managers: Optional[Dict[str, 'TradeManager']] = None
  ):
    """
    Args:
        symbols: Список торговых пар
        normalize: Применять ли нормализацию
        cache_enabled: Использовать ли кэширование
        trade_managers: Optional Dict[symbol, TradeManager] для реальных market trades
    """
    self.symbols = symbols
    self.normalize = normalize
    self.cache_enabled = cache_enabled

    # Создаем pipeline для каждого символа
    self.pipelines: Dict[str, FeaturePipeline] = {}
    for symbol in symbols:
      # Получаем TradeManager для этого символа, если доступен
      trade_manager = trade_managers.get(symbol) if trade_managers else None

      self.pipelines[symbol] = FeaturePipeline(
        symbol=symbol,
        normalize=normalize,
        cache_enabled=cache_enabled,
        trade_manager=trade_manager
      )

    logger.info(
      f"MultiSymbolFeaturePipeline инициализирован для "
      f"{len(symbols)} символов"
    )

  async def extract_features_batch(
      self,
      data: Dict[str, Tuple[OrderBookSnapshot, List[Candle]]]
  ) -> Dict[str, FeatureVector]:
    """
    Извлекает признаки для нескольких символов параллельно.

    Args:
        data: Dict[symbol, (orderbook_snapshot, candles)]

    Returns:
        Dict[symbol, FeatureVector]
    """
    logger.debug(f"Batch extraction для {len(data)} символов")

    # Создаем задачи для параллельного выполнения
    tasks = []
    symbols_order = []

    for symbol, (orderbook, candles) in data.items():
      if symbol in self.pipelines:
        task = self.pipelines[symbol].extract_features(
          orderbook_snapshot=orderbook,
          candles=candles
        )
        tasks.append(task)
        symbols_order.append(symbol)

    # Выполняем параллельно
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Собираем результаты
    feature_vectors: Dict[str, FeatureVector] = {}
    for symbol, result in zip(data.keys(), results):
      if result is not None:
        feature_vectors[symbol] = result

    logger.info(
      f"Batch extraction завершен: {len(feature_vectors)}/{len(data)} успешно"
    )

    return feature_vectors

  async def extract_features_enhanced(
        self,
        symbol: str,
        orderbook_snapshot: OrderBookSnapshot,
        candles: List[Candle],
        orderbook_metrics: Optional['OrderBookMetrics'] = None,
        sr_levels: Optional[List['SRLevel']] = None,
        prev_orderbook: Optional[OrderBookSnapshot] = None,
        prev_candle: Optional[Candle] = None
    ) -> Optional[FeatureVector]:
      """
      Расширенная версия извлечения признаков с дополнительными метриками.

      Извлекает базовые 110 признаков + добавляет enriched metadata:
      - OrderBook метрики (imbalance, spread, vwap)
      - S/R уровни (поддержка/сопротивление)
      - Расстояния до ключевых уровней

      Args:
          symbol: Торговая пара
          orderbook_snapshot: Снимок стакана
          candles: История свечей (минимум 50 для индикаторов)
          orderbook_metrics: Рассчитанные метрики стакана (из OrderBookAnalyzer)
          sr_levels: Список уровней поддержки/сопротивления (из SRLevelDetector)
          prev_orderbook: Предыдущий снимок для временных признаков
          prev_candle: Предыдущая свеча для momentum признаков

      Returns:
          FeatureVector с 110 признаками + enriched metadata или None

      Example:
          feature_vector = await pipeline.extract_features_enhanced(
          ...     symbol="BTCUSDT",
          ...     orderbook_snapshot=snapshot,
          ...     candles=candles,
          ...     orderbook_metrics=metrics,
          ...     sr_levels=levels
          ... )
          print(f"Features: {feature_vector.feature_count}")
          print(f"Metadata: {feature_vector.metadata}")
      """
      # Проверка наличия pipeline для символа
      if symbol not in self.pipelines:
        logger.warning(
          f"{symbol} | Pipeline не найден в MultiSymbolFeaturePipeline"
        )
        return None

      try:
        # ============================================================
        # ШАГ 1: БАЗОВОЕ ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ (110 features)
        # ============================================================

        pipeline = self.pipelines[symbol]
        feature_vector = await pipeline.extract_features(
          orderbook_snapshot=orderbook_snapshot,
          candles=candles,
          prev_orderbook=prev_orderbook,
          prev_candle=prev_candle
        )

        if not feature_vector:
          logger.warning(f"{symbol} | Не удалось извлечь базовые признаки")
          return None

        logger.debug(
          f"{symbol} | Базовое извлечение: {feature_vector.feature_count} признаков"
        )

        # ============================================================
        # ШАГ 2: ENRICHMENT - ORDERBOOK METRICS
        # ============================================================

        if orderbook_metrics:
          try:
            # Извлекаем ключевые метрики
            enriched_metrics = {
              'imbalance': float(orderbook_metrics.imbalance),
              'spread': float(orderbook_metrics.spread) if orderbook_metrics.spread else 0.0,
              'mid_price': float(orderbook_metrics.mid_price) if orderbook_metrics.mid_price else 0.0,
              'total_bid_volume': float(orderbook_metrics.total_bid_volume),
              'total_ask_volume': float(orderbook_metrics.total_ask_volume),
              'total_volume': float(
                orderbook_metrics.total_bid_volume +
                orderbook_metrics.total_ask_volume
              ),
            }

            # VWAP метрики (если доступны)
            if orderbook_metrics.vwap_bid:
              enriched_metrics['vwap_bid'] = float(orderbook_metrics.vwap_bid)

            if orderbook_metrics.vwap_ask:
              enriched_metrics['vwap_ask'] = float(orderbook_metrics.vwap_ask)

            if orderbook_metrics.vwmp:
              enriched_metrics['vwmp'] = float(orderbook_metrics.vwmp)

            # Кластеры объема (если есть)
            if orderbook_metrics.largest_bid_cluster_volume > 0:
              enriched_metrics['bid_cluster_volume'] = float(
                orderbook_metrics.largest_bid_cluster_volume
              )

            if orderbook_metrics.largest_ask_cluster_volume > 0:
              enriched_metrics['ask_cluster_volume'] = float(
                orderbook_metrics.largest_ask_cluster_volume
              )

            # Добавляем в metadata
            feature_vector.metadata['orderbook_metrics'] = enriched_metrics

            logger.debug(
              f"{symbol} | OrderBook metrics enriched: "
              f"imbalance={enriched_metrics['imbalance']:.3f}, "
              f"spread={enriched_metrics['spread']:.8f}"
            )

          except Exception as e:
            logger.error(
              f"{symbol} | Ошибка enrichment orderbook_metrics: {e}",
              exc_info=True
            )

        # ============================================================
        # ШАГ 3: ENRICHMENT - S/R LEVELS
        # ============================================================

        if sr_levels:
          try:
            # Разделяем по типу
            supports = [
              lvl for lvl in sr_levels
              if lvl.level_type == "support" and not lvl.is_broken
            ]
            resistances = [
              lvl for lvl in sr_levels
              if lvl.level_type == "resistance" and not lvl.is_broken
            ]

            # Базовая информация
            sr_context = {
              'num_supports': len(supports),
              'num_resistances': len(resistances),
              'total_levels': len(sr_levels),
              'active_levels': len(supports) + len(resistances)
            }

            # Находим сильнейшие уровни
            if supports:
              strongest_support = max(supports, key=lambda x: x.strength)
              sr_context['strongest_support'] = {
                'price': float(strongest_support.price),
                'strength': float(strongest_support.strength),
                'touch_count': int(strongest_support.touch_count),
                'total_volume': float(strongest_support.total_volume)
              }

              # Расстояние до сильнейшей поддержки
              current_price = orderbook_snapshot.mid_price
              if current_price:
                distance_pct = (
                    (current_price - strongest_support.price) /
                    current_price * 100
                )
                sr_context['strongest_support']['distance_pct'] = float(distance_pct)

            if resistances:
              strongest_resistance = max(resistances, key=lambda x: x.strength)
              sr_context['strongest_resistance'] = {
                'price': float(strongest_resistance.price),
                'strength': float(strongest_resistance.strength),
                'touch_count': int(strongest_resistance.touch_count),
                'total_volume': float(strongest_resistance.total_volume)
              }

              # Расстояние до сильнейшего сопротивления
              current_price = orderbook_snapshot.mid_price
              if current_price:
                distance_pct = (
                    (strongest_resistance.price - current_price) /
                    current_price * 100
                )
                sr_context['strongest_resistance']['distance_pct'] = float(distance_pct)

            # Ближайшие уровни (топ-3 по близости)
            if supports:
              current_price = orderbook_snapshot.mid_price or 0.0
              nearest_supports = sorted(
                supports,
                key=lambda x: abs(x.price - current_price)
              )[:3]

              sr_context['nearest_supports'] = [
                {
                  'price': float(lvl.price),
                  'strength': float(lvl.strength),
                  'distance_pct': float(
                    abs(lvl.price - current_price) / current_price * 100
                  ) if current_price > 0 else 0.0
                }
                for lvl in nearest_supports
              ]

            if resistances:
              current_price = orderbook_snapshot.mid_price or 0.0
              nearest_resistances = sorted(
                resistances,
                key=lambda x: abs(x.price - current_price)
              )[:3]

              sr_context['nearest_resistances'] = [
                {
                  'price': float(lvl.price),
                  'strength': float(lvl.strength),
                  'distance_pct': float(
                    abs(lvl.price - current_price) / current_price * 100
                  ) if current_price > 0 else 0.0
                }
                for lvl in nearest_resistances
              ]

            # Добавляем в metadata
            feature_vector.metadata['sr_levels'] = sr_context

            logger.debug(
              f"{symbol} | S/R levels enriched: "
              f"supports={len(supports)}, resistances={len(resistances)}"
            )

          except Exception as e:
            logger.error(
              f"{symbol} | Ошибка enrichment sr_levels: {e}",
              exc_info=True
            )

        # ============================================================
        # ШАГ 4: ДОПОЛНИТЕЛЬНЫЕ КОНТЕКСТНЫЕ ДАННЫЕ
        # ============================================================

        # Timestamp для tracking
        feature_vector.metadata['extraction_timestamp'] = int(
          datetime.now().timestamp() * 1000
        )

        # Информация о качестве данных
        feature_vector.metadata['data_quality'] = {
          'num_candles': len(candles),
          'has_prev_orderbook': prev_orderbook is not None,
          'has_prev_candle': prev_candle is not None,
          'has_orderbook_metrics': orderbook_metrics is not None,
          'has_sr_levels': sr_levels is not None and len(sr_levels) > 0
        }

        logger.info(
          f"{symbol} | Enhanced extraction complete: "
          f"{feature_vector.feature_count} features + "
          f"{len(feature_vector.metadata)} metadata keys"
        )

        return feature_vector

      except Exception as e:
        logger.error(
          f"{symbol} | Критическая ошибка в extract_features_enhanced: {e}",
          exc_info=True
        )
        return None

  async def extract_features_single(
      self,
      symbol: str,
      orderbook_snapshot: OrderBookSnapshot,
      candles: List[Candle]
  ) -> Optional[FeatureVector]:
    """
    Извлечь признаки для одного символа.

    Args:
        symbol: Торговая пара
        orderbook_snapshot: Snapshot стакана
        candles: История свечей

    Returns:
        FeatureVector или None если символ не найден
    """
    if symbol not in self.pipelines:
      logger.warning(f"Pipeline для {symbol} не найден")
      return None

    pipeline = self.pipelines[symbol]
    return await pipeline.extract_features(orderbook_snapshot, candles)

  def get_pipeline(self, symbol: str) -> Optional[FeaturePipeline]:
    """Возвращает pipeline для конкретного символа"""
    return self.pipelines.get(symbol)

  def add_symbol(self, symbol: str):
    """Добавляет новый символ"""
    if symbol not in self.pipelines:
      self.pipelines[symbol] = FeaturePipeline(
        symbol=symbol,
        normalize=self.normalize,
        cache_enabled=self.cache_enabled
      )
      self.symbols.append(symbol)
      logger.info(f"Добавлен символ {symbol} в pipeline")

  def remove_symbol(self, symbol: str):
    """Удаляет символ"""
    if symbol in self.pipelines:
      del self.pipelines[symbol]
      self.symbols.remove(symbol)
      logger.info(f"Удален символ {symbol} из pipeline")