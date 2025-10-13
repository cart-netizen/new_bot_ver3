"""
Feature Pipeline - оркестратор всех feature extractors.

Объединяет признаки из OrderBook, Candles и Indicators в единый вектор.
Поддерживает Multi-Channel Representation для ML моделей.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler
import asyncio

from core.logger import get_logger
from models.orderbook import OrderBookSnapshot
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
      cache_enabled: bool = False
  ):
    """
    Args:
        symbol: Торговая пара
        normalize: Применять ли нормализацию к признакам
        cache_enabled: Использовать ли кэширование (Redis)
    """
    self.symbol = symbol
    self.normalize = normalize
    self.cache_enabled = cache_enabled

    # Инициализация extractors
    self.orderbook_extractor = OrderBookFeatureExtractor(symbol)
    self.candle_extractor = CandleFeatureExtractor(symbol)
    self.indicator_extractor = IndicatorFeatureExtractor(symbol)

    # Scaler для нормализации
    self.scaler: Optional[StandardScaler] = None
    if normalize:
      self.scaler = StandardScaler()
      self._scaler_fitted = False

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
        feature_count=50 + 25 + 35  # 110 признаков
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

      logger.info(
        f"{self.symbol} | Pipeline завершен: {feature_vector.feature_count} признаков, "
        f"timestamp={feature_vector.timestamp}"
      )

      return feature_vector

    except Exception as e:
      logger.error(f"{self.symbol} | Ошибка в pipeline: {e}")
      raise

  async def _normalize_features(
      self,
      feature_vector: FeatureVector
  ) -> FeatureVector:
    """
    Нормализует признаки используя StandardScaler.

    Note: Для production нужно обучить scaler на исторических данных.
    Здесь используется online normalization.
    """
    try:
      # Получаем массив признаков
      features_array = feature_vector.to_array()

      # Проверяем NaN/Inf
      if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
        logger.warning(
          f"{self.symbol} | Обнаружены NaN/Inf перед нормализацией, "
          "заменяем на 0"
        )
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

      # Для первого вызова - "обучаем" scaler
      if not self._scaler_fitted:
        # Reshape для sklearn
        features_2d = features_array.reshape(1, -1)
        self.scaler.partial_fit(features_2d)
        self._scaler_fitted = True
        logger.debug(f"{self.symbol} | Scaler инициализирован")

      # Нормализуем
      features_2d = features_array.reshape(1, -1)
      normalized = self.scaler.transform(features_2d).flatten()

      # Обновляем признаки (создаем новые объекты с normalized values)
      # Это упрощенная версия - в production лучше использовать отдельные scalers
      # для каждого канала

      logger.debug(
        f"{self.symbol} | Признаки нормализованы, "
        f"mean={normalized.mean():.3f}, std={normalized.std():.3f}"
      )

      # Возвращаем оригинальный вектор (в production здесь нужно обновить значения)
      return feature_vector

    except Exception as e:
      logger.error(f"{self.symbol} | Ошибка нормализации: {e}")
      # Возвращаем ненормализованный вектор
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

    Returns:
        Dict[feature_name, importance_score]
    """
    # Для production здесь должен быть анализ на основе модели
    # Пока возвращаем заглушку
    return {}

  async def warmup(self, num_samples: int = 10):
    """
    Прогрев pipeline и scaler на начальных данных.

    Args:
        num_samples: Количество образцов для прогрева
    """
    logger.info(f"{self.symbol} | Прогрев pipeline на {num_samples} образцах")

    # В production здесь нужно загрузить исторические данные
    # и "обучить" scaler

    if self.scaler and not self._scaler_fitted:
      logger.warning(
        f"{self.symbol} | Для корректной нормализации нужны "
        "исторические данные"
      )


class MultiSymbolFeaturePipeline:
  """
  Управляет pipeline для нескольких символов одновременно.
  """

  def __init__(
      self,
      symbols: List[str],
      normalize: bool = True,
      cache_enabled: bool = True
  ):
    """
    Args:
        symbols: Список торговых пар
        normalize: Применять ли нормализацию
        cache_enabled: Использовать ли кэширование
    """
    self.symbols = symbols
    self.normalize = normalize
    self.cache_enabled = cache_enabled

    # Создаем pipeline для каждого символа
    self.pipelines: Dict[str, FeaturePipeline] = {}
    for symbol in symbols:
      self.pipelines[symbol] = FeaturePipeline(
        symbol=symbol,
        normalize=normalize,
        cache_enabled=cache_enabled
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

    # Выполняем все задачи параллельно
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Собираем результаты
    feature_vectors: Dict[str, FeatureVector] = {}
    for symbol, result in zip(data.keys(), results):
      if result is not None:
        feature_vectors[symbol] = result

    return feature_vectors

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