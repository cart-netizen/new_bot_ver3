"""
Strategy Manager - объединение всех торговых стратегий.

ИСПРАВЛЕНИЕ: Добавлена валидация минимального количества стратегий
в побеждающей группе для всех режимов consensus.

ИСПРАВЛЕНИЕ 2: Добавлено кэширование результатов стратегий для
предотвращения множественных вызовов за один цикл анализа.

Функциональность:
- Управление множественными стратегиями
- Объединение сигналов (consensus)
- Приоритизация стратегий
- Конфликт-резолюция
- Статистика по стратегиям
- Кэширование результатов (НОВОЕ)

Путь: backend/strategies/strategy_manager.py
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time as time_module

from backend.core.logger import get_logger
from backend.models.orderbook import OrderBookSnapshot, OrderBookMetrics
from backend.models.signal import TradingSignal, SignalType, SignalStrength, SignalSource
from backend.strategy.candle_manager import Candle

# Импорт всех стратегий
from backend.strategies.momentum_strategy import MomentumStrategy, MomentumConfig
from backend.strategies.sar_wave_strategy import SARWaveStrategy, SARWaveConfig
from backend.strategies.supertrend_strategy import SuperTrendStrategy, SuperTrendConfig
from backend.strategies.volume_profile_strategy import VolumeProfileStrategy, VolumeProfileConfig

# Импорт новых OrderBook стратегий
from backend.strategies.imbalance_strategy import ImbalanceStrategy, ImbalanceConfig
from backend.strategies.volume_flow_strategy import VolumeFlowStrategy, VolumeFlowConfig
from backend.strategies.liquidity_zone_strategy import LiquidityZoneStrategy, LiquidityZoneConfig
from backend.strategies.smart_money_strategy import SmartMoneyStrategy, SmartMoneyConfig
from backend.strategy.trade_manager import TradeManager

from backend.utils.helpers import safe_enum_value

logger = get_logger(__name__)


class StrategyPriority(Enum):
  """Приоритет стратегий."""
  HIGH = 3
  MEDIUM = 2
  LOW = 1

class StrategyType(Enum):
  """Тип стратегии."""
  CANDLE = "candle"  # Работает только со свечами
  ORDERBOOK = "orderbook"  # Работает только со стаканом
  HYBRID = "hybrid"  # Комбинирует свечи и стакан

@dataclass
class StrategyManagerConfig:
  """Конфигурация Strategy Manager."""
  # Режим объединения сигналов
  consensus_mode: str = "weighted"  # "weighted", "majority", "unanimous"

  # Минимальные требования
  min_strategies_for_signal: int = 2  # Минимум стратегий должны согласиться
  min_consensus_confidence: float = 0.6  # Минимальная уверенность для consensus

  # Веса стратегий (для weighted режима)
  strategy_weights: Dict[str, float] = None

  # Приоритеты
  strategy_priorities: Dict[str, StrategyPriority] = None

  # Конфликт-резолюция
  conflict_resolution: str = "highest_confidence"  # "highest_confidence", "priority", "cancel"

  def __post_init__(self):
    if self.strategy_weights is None:
      self.strategy_weights = {
        'momentum': 0.25,
        'sar_wave': 0.25,
        'supertrend': 0.25,
        'volume_profile': 0.25
      }

    if self.strategy_priorities is None:
      self.strategy_priorities = {
        'momentum': StrategyPriority.HIGH,
        'sar_wave': StrategyPriority.MEDIUM,
        'supertrend': StrategyPriority.HIGH,
        'volume_profile': StrategyPriority.MEDIUM
      }


@dataclass
class ExtendedStrategyManagerConfig:
  """Расширенная конфигурация Strategy Manager."""
  # Режим объединения сигналов
  consensus_mode: str = "weighted"  # "weighted", "majority", "unanimous"

  # Минимальные требования
  min_strategies_for_signal: int = 2
  min_consensus_confidence: float = 0.6

  # Веса стратегий (candle-based)
  candle_strategy_weights: Dict[str, float] = field(default_factory=lambda: {
    'momentum': 0.20,
    'sar_wave': 0.15,
    'supertrend': 0.20,
    'volume_profile': 0.15
  })

  # Веса стратегий (orderbook-based)
  orderbook_strategy_weights: Dict[str, float] = field(default_factory=lambda: {
    'imbalance': 0.10,
    'volume_flow': 0.10,
    'liquidity_zone': 0.10
  })

  # Веса стратегий (hybrid)
  hybrid_strategy_weights: Dict[str, float] = field(default_factory=lambda: {
    'smart_money': 0.15
  })

  # Приоритеты стратегий
  strategy_priorities: Dict[str, StrategyPriority] = field(default_factory=lambda: {
    # Candle strategies
    'momentum': StrategyPriority.HIGH,
    'sar_wave': StrategyPriority.MEDIUM,
    'supertrend': StrategyPriority.HIGH,
    'volume_profile': StrategyPriority.MEDIUM,
    # OrderBook strategies
    'imbalance': StrategyPriority.MEDIUM,
    'volume_flow': StrategyPriority.MEDIUM,
    'liquidity_zone': StrategyPriority.HIGH,
    # Hybrid strategies
    'smart_money': StrategyPriority.HIGH
  })

  # Типы стратегий
  strategy_types: Dict[str, StrategyType] = field(default_factory=lambda: {
    'momentum': StrategyType.CANDLE,
    'sar_wave': StrategyType.CANDLE,
    'supertrend': StrategyType.CANDLE,
    'volume_profile': StrategyType.CANDLE,
    'imbalance': StrategyType.ORDERBOOK,
    'volume_flow': StrategyType.ORDERBOOK,
    'liquidity_zone': StrategyType.ORDERBOOK,
    'smart_money': StrategyType.HYBRID
  })

  # Гибридный consensus режим
  hybrid_consensus_mode: str = 'weighted'  # 'all_agree', 'any_agree', 'weighted'

  # Включение гибридных стратегий
  enable_hybrid_strategies: bool = True
  enable_orderbook_strategies: bool = True

  # Конфликт-резолюция
  conflict_resolution: str = "highest_confidence"

@dataclass
class StrategyResult:
    """Результат от одной стратегии."""
    strategy_name: str
    strategy_type: StrategyType
    signal: Optional[TradingSignal]
    priority: StrategyPriority
    weight: float
    execution_time_ms: float


@dataclass
class CachedStrategyResults:
    """
    Кэш результатов стратегий.

    Предотвращает множественные вызовы analyze_all_strategies
    за один цикл анализа (например, из MTF + Single-TF mode).
    """
    results: List[StrategyResult]
    candle_timestamp: int  # timestamp последней свечи (идентификатор данных)
    calculated_at: int  # когда посчитано (для TTL)
    cache_key: str  # уникальный ключ кэша

    def is_valid(self, current_candle_ts: int, max_age_ms: int = 2000) -> bool:
        """
        Проверить актуальность кэша.

        Кэш валиден если:
        1. Timestamp свечи совпадает (те же входные данные)
        2. Не старше max_age_ms (защита от stale данных)

        Args:
            current_candle_ts: Timestamp текущей последней свечи
            max_age_ms: Максимальный возраст кэша в мс (default 2 сек)

        Returns:
            True если кэш валиден
        """
        # Проверка 1: Данные те же (timestamp свечи совпадает)
        if self.candle_timestamp != current_candle_ts:
            return False

        # Проверка 2: Не слишком старый
        current_time = int(time_module.time() * 1000)
        age_ms = current_time - self.calculated_at

        return age_ms < max_age_ms


@dataclass
class ConsensusSignal:
  """Объединенный сигнал от нескольких стратегий."""
  final_signal: TradingSignal
  contributing_strategies: List[str]
  agreement_count: int
  disagreement_count: int
  consensus_confidence: float
  strategy_results: List[StrategyResult]

  # Разбивка по типам стратегий
  candle_strategies_count: int = 0
  orderbook_strategies_count: int = 0
  hybrid_strategies_count: int = 0

class ExtendedStrategyManager:
  """
  Расширенный менеджер торговых стратегий.

  Управляет:
  - Свечными стратегиями (традиционный технический анализ)
  - OrderBook стратегиями (микроструктура рынка)
  - Гибридными стратегиями (комбинация обоих)
  """

  def __init__(self, config: ExtendedStrategyManagerConfig, trade_managers: Optional[Dict[str, 'TradeManager']] = None):
    """
    Инициализация менеджера.

    Args:
        config: Конфигурация менеджера
        trade_managers: Optional Dict[symbol, TradeManager] для интеграции реальных market trades
    """
    self.config = config
    self.trade_managers = trade_managers or {}

    # Инициализация свечных стратегий
    self.candle_strategies: Dict[str, any] = {}
    self.candle_strategies['momentum'] = MomentumStrategy(MomentumConfig())
    self.candle_strategies['sar_wave'] = SARWaveStrategy(SARWaveConfig())
    self.candle_strategies['supertrend'] = SuperTrendStrategy(SuperTrendConfig())
    self.candle_strategies['volume_profile'] = VolumeProfileStrategy(VolumeProfileConfig())

    # Инициализация OrderBook стратегий (с TradeManager)
    # ПРИМЕЧАНИЕ: trade_manager передается для конкретного символа в analyze_all_strategies
    self.orderbook_strategies: Dict[str, any] = {}
    if config.enable_orderbook_strategies:
      # Передаем None в __init__, т.к. trade_manager специфичен для символа
      # и будет установлен динамически для каждого вызова analyze
      self.orderbook_strategies['imbalance'] = ImbalanceStrategy(ImbalanceConfig())
      self.orderbook_strategies['volume_flow'] = VolumeFlowStrategy(VolumeFlowConfig())
      self.orderbook_strategies['liquidity_zone'] = LiquidityZoneStrategy(LiquidityZoneConfig())

    # Инициализация гибридных стратегий (с TradeManager)
    self.hybrid_strategies: Dict[str, any] = {}
    if config.enable_hybrid_strategies:
      self.hybrid_strategies['smart_money'] = SmartMoneyStrategy(SmartMoneyConfig())

    # Объединенный список всех стратегий
    self.all_strategies = {
      **self.candle_strategies,
      **self.orderbook_strategies,
      **self.hybrid_strategies
    }

    # Статистика
    self.total_analyses = 0
    self.signals_generated = 0
    self.consensus_achieved = 0
    self.conflicts_resolved = 0

    # Кэширование результатов стратегий
    self._strategy_cache: Dict[str, CachedStrategyResults] = {}
    self._cache_hits = 0
    self._cache_misses = 0

    logger.info(
      f"Инициализирован ExtendedStrategyManager: "
      f"candle_strategies={list(self.candle_strategies.keys())}, "
      f"orderbook_strategies={list(self.orderbook_strategies.keys())}, "
      f"hybrid_strategies={list(self.hybrid_strategies.keys())}, "
      f"consensus_mode={config.consensus_mode}"
    )

  def analyze_all_strategies(
      self,
      symbol: str,
      candles: List[Candle],
      current_price: float,
      orderbook: Optional[OrderBookSnapshot] = None,
      metrics: Optional[OrderBookMetrics] = None,
      sr_levels: Optional[List] = None,
      volume_profile: Optional[Dict] = None,
      ml_prediction: Optional[Dict] = None,
      market_trades: Optional[List] = None  # НОВОЕ: Market trades для анализа
  ) -> List[StrategyResult]:
    """
    Запустить ВСЕ стратегии для анализа.

    Роутинг данных:
    - Candle strategies: получают только candles
    - OrderBook strategies: получают candles + orderbook + metrics + market_trades
    - Hybrid strategies: получают всё

    Args:
        symbol: Торговая пара
        candles: История свечей
        current_price: Текущая цена
        orderbook: Снимок стакана (для OrderBook/Hybrid стратегий)
        metrics: Метрики стакана (для OrderBook/Hybrid стратегий)
        sr_levels: S/R уровни (опционально)
        volume_profile: Volume profile (опционально)
        ml_prediction: ML предсказание (опционально)
        market_trades: Список публичных сделок (для анализа потока ордеров)

    Returns:
        Список результатов от каждой стратегии
    """
    import time

    # ========== КЭШИРОВАНИЕ: Проверка кэша ==========
    # Генерируем ключ кэша (symbol + orderbook hash)
    cache_key = self._generate_cache_key(symbol, candles, orderbook)

    # Получаем timestamp последней свечи как идентификатор данных
    candle_timestamp = candles[-1].timestamp if candles else 0

    # Проверяем кэш
    cached = self._strategy_cache.get(cache_key)
    if cached and cached.is_valid(candle_timestamp):
      self._cache_hits += 1
      logger.debug(
        f"[{symbol}] analyze_all_strategies: CACHE HIT "
        f"(key={cache_key[:16]}..., hits={self._cache_hits})"
      )
      return cached.results

    self._cache_misses += 1
    # ========== END КЭШИРОВАНИЕ ==========

    logger.info(
      f"[{symbol}] analyze_all_strategies: "
      f"candle_strategies={len(self.candle_strategies)}, "
      f"orderbook_strategies={len(self.orderbook_strategies)}, "
      f"hybrid_strategies={len(self.hybrid_strategies)}"
    )

    results = []

    # Получаем TradeManager для символа (будет использоваться всеми типами стратегий)
    trade_manager = self.trade_managers.get(symbol)

    # ========== Candle Strategies ==========
    for strategy_name, strategy in self.candle_strategies.items():
      start_time = time.time()

      try:
        # Устанавливаем trade_manager для стратегий, которые его поддерживают (например, MomentumStrategy)
        if hasattr(strategy, 'trade_manager'):
          strategy.trade_manager = trade_manager

        signal = strategy.analyze(symbol, candles, current_price)
        execution_time = (time.time() - start_time) * 1000

        result = StrategyResult(
          strategy_name=strategy_name,
          strategy_type=StrategyType.CANDLE,
          signal=signal,
          priority=self.config.strategy_priorities.get(
            strategy_name, StrategyPriority.MEDIUM
          ),
          weight=self.config.candle_strategy_weights.get(strategy_name, 0.10),
          execution_time_ms=execution_time
        )

        results.append(result)

        if signal:
          logger.info(
            f"[CANDLE/{strategy_name}] {symbol}: "
            f"{safe_enum_value(signal.signal_type)}, "
            f"confidence={signal.confidence:.2f}"
          )
        else:
          logger.debug(f"[CANDLE/{strategy_name}] {symbol}: NO SIGNAL")

      except Exception as e:
        logger.error(f"Ошибка в candle стратегии {strategy_name}: {e}", exc_info=True)
        continue

    # ========== OrderBook Strategies ==========
    if orderbook and metrics and self.config.enable_orderbook_strategies:
      for strategy_name, strategy in self.orderbook_strategies.items():
        start_time = time.time()

        try:
          # Устанавливаем trade_manager для этого символа (уже установлен выше)
          if hasattr(strategy, 'trade_manager'):
            strategy.trade_manager = trade_manager

          # Передаем дополнительные параметры для LiquidityZone
          if strategy_name == 'liquidity_zone':
            signal = strategy.analyze(
              symbol, candles, current_price, orderbook, metrics,
              sr_levels=sr_levels,
              volume_profile=volume_profile
            )
          else:
            signal = strategy.analyze(
              symbol, candles, current_price, orderbook, metrics
            )

          execution_time = (time.time() - start_time) * 1000

          result = StrategyResult(
            strategy_name=strategy_name,
            strategy_type=StrategyType.ORDERBOOK,
            signal=signal,
            priority=self.config.strategy_priorities.get(
              strategy_name, StrategyPriority.MEDIUM
            ),
            weight=self.config.orderbook_strategy_weights.get(strategy_name, 0.10),
            execution_time_ms=execution_time
          )

          results.append(result)

          if signal:
            logger.info(
              f"[ORDERBOOK/{strategy_name}] {symbol}: "
              f"{safe_enum_value(signal.signal_type)}, "
              f"confidence={signal.confidence:.2f}"
            )
          else:
            logger.debug(f"[ORDERBOOK/{strategy_name}] {symbol}: NO SIGNAL")

        except Exception as e:
          logger.error(
            f"Ошибка в orderbook стратегии {strategy_name}: {e}",
            exc_info=True
          )
          continue

    # ========== Hybrid Strategies ==========
    if orderbook and metrics and self.config.enable_hybrid_strategies:
      for strategy_name, strategy in self.hybrid_strategies.items():
        start_time = time.time()

        try:
          # Устанавливаем trade_manager для этого символа (уже установлен выше)
          if hasattr(strategy, 'trade_manager'):
            strategy.trade_manager = trade_manager

          signal = strategy.analyze(
            symbol=symbol,
            candles=candles,
            current_price=current_price,
            orderbook=orderbook,
            metrics=metrics,
            volume_profile=volume_profile,
            ml_prediction=ml_prediction
          )

          execution_time = (time.time() - start_time) * 1000

          result = StrategyResult(
            strategy_name=strategy_name,
            strategy_type=StrategyType.HYBRID,
            signal=signal,
            priority=self.config.strategy_priorities.get(
              strategy_name, StrategyPriority.MEDIUM
            ),
            weight=self.config.hybrid_strategy_weights.get(strategy_name, 0.15),
            execution_time_ms=execution_time
          )

          results.append(result)

          if signal:
            logger.info(
              f"[HYBRID/{strategy_name}] {symbol}: "
              f"{safe_enum_value(signal.signal_type)}, "
              f"confidence={signal.confidence:.2f}"
            )
          else:
            logger.debug(f"[HYBRID/{strategy_name}] {symbol}: NO SIGNAL")

        except Exception as e:
          logger.error(
            f"Ошибка в hybrid стратегии {strategy_name}: {e}",
            exc_info=True
          )
          continue

    self.total_analyses += 1

    signals_count = len([r for r in results if r.signal is not None])
    logger.info(
      f"[{symbol}] analyze_all_strategies завершён: "
      f"total_results={len(results)}, "
      f"with_signals={signals_count}"
    )

    # ========== КЭШИРОВАНИЕ: Сохранение результатов ==========
    self._strategy_cache[cache_key] = CachedStrategyResults(
      results=results,
      candle_timestamp=candle_timestamp,
      calculated_at=int(time_module.time() * 1000),
      cache_key=cache_key
    )

    # Очистка старых записей кэша (храним максимум 50)
    if len(self._strategy_cache) > 50:
      oldest_key = min(
        self._strategy_cache.keys(),
        key=lambda k: self._strategy_cache[k].calculated_at
      )
      del self._strategy_cache[oldest_key]
    # ========== END КЭШИРОВАНИЕ ==========

    return results

  def _generate_cache_key(
      self,
      symbol: str,
      candles: List[Candle],
      orderbook: Optional[OrderBookSnapshot]
  ) -> str:
    """
    Генерация ключа кэша на основе входных данных.

    Ключ включает:
    - symbol
    - timestamp последней свечи
    - hash от orderbook (если есть)
    """
    import hashlib

    # Базовый ключ
    key_parts = [symbol]

    # Timestamp последней свечи
    if candles:
      key_parts.append(str(candles[-1].timestamp))
      # Также включаем close price для дополнительной уникальности
      key_parts.append(f"{candles[-1].close:.8f}")
    else:
      key_parts.append("no_candles")

    # Hash от orderbook
    if orderbook:
      # Используем простой hash от best bid/ask
      ob_data = f"{orderbook.best_bid}:{orderbook.best_ask}:{orderbook.timestamp}"
      key_parts.append(hashlib.md5(ob_data.encode()).hexdigest()[:8])
    else:
      key_parts.append("no_ob")

    return "|".join(key_parts)

  def _detect_volume_explosion(
      self,
      strategy_results: List[StrategyResult]
  ) -> float:
    """
    Детекция volume explosion из результатов стратегий.

    Проверяет metadata от VolumeProfile и других стратегий на наличие volume_ratio.

    Returns:
        float: Максимальный volume_ratio или 1.0 если не найден
    """
    max_volume_ratio = 1.0

    for result in strategy_results:
      if result.signal and result.signal.metadata:
        # Проверяем volume_ratio в metadata
        volume_ratio = result.signal.metadata.get('volume_ratio', 0)
        if volume_ratio > max_volume_ratio:
          max_volume_ratio = volume_ratio

    return max_volume_ratio

  def _get_dynamic_weights(
      self,
      volume_ratio: float,
      original_weights: Dict[str, float]
  ) -> Dict[str, float]:
    """
    Динамические веса стратегий при volume explosion.

    При volume_ratio > 5x увеличиваем веса volume-based стратегий.
    """
    if volume_ratio < 5.0:
      return original_weights.copy()

    adjusted = original_weights.copy()

    # При volume explosion увеличиваем веса volume-based стратегий
    if 'volume_profile' in adjusted:
      # 15% → 30% при explosion
      adjusted['volume_profile'] = min(0.30, adjusted['volume_profile'] * 2)

    if 'liquidity_zone' in adjusted:
      # 10% → 20% при explosion
      adjusted['liquidity_zone'] = min(0.20, adjusted['liquidity_zone'] * 2)

    logger.info(
      f"🚀 VOLUME EXPLOSION ({volume_ratio:.1f}x): "
      f"Adjusted weights: volume_profile={adjusted.get('volume_profile', 0):.0%}, "
      f"liquidity_zone={adjusted.get('liquidity_zone', 0):.0%}"
    )

    return adjusted

  def _get_dynamic_consensus_threshold(
      self,
      volume_ratio: float
  ) -> float:
    """
    Динамический порог консенсуса при volume explosion.

    При volume_ratio > 5x снижаем требования для быстрого входа.
    """
    if volume_ratio >= 8.0:
      # Extreme explosion - очень низкий порог
      return 0.35
    elif volume_ratio >= 5.0:
      # Strong explosion - сниженный порог
      return 0.40
    elif volume_ratio >= 3.0:
      # Moderate spike - немного сниженный порог
      return 0.50
    else:
      return self.config.min_consensus_confidence

  def build_consensus(
      self,
      symbol: str,
      strategy_results: List[StrategyResult],
      current_price: float
  ) -> Optional[ConsensusSignal]:
    """
    Построить consensus сигнал из результатов стратегий.

    Учитывает:
    - Разные типы стратегий (candle/orderbook/hybrid)
    - Веса и приоритеты
    - Конфликты между сигналами
    - НОВОЕ: Dynamic weights и threshold при volume explosion
    """
    # ========== VOLUME EXPLOSION DETECTION ==========
    volume_ratio = self._detect_volume_explosion(strategy_results)
    is_explosion = volume_ratio >= 5.0

    if is_explosion:
      logger.info(
        f"[{symbol}] 🚀 VOLUME EXPLOSION DETECTED: {volume_ratio:.1f}x - "
        f"Activating dynamic weights and threshold"
      )

    # Фильтруем только стратегии с сигналами
    results_with_signals = [r for r in strategy_results if r.signal is not None]

    # ДЕБАГ: Подробное логирование
    logger.info(
      f"[{symbol}] build_consensus: "
      f"total_strategies={len(strategy_results)}, "
      f"with_signals={len(results_with_signals)}"
    )

    if len(strategy_results) > 0:
      for result in strategy_results:
        logger.info(
          f"[{symbol}] Strategy '{result.strategy_name}': "
          f"signal={'YES' if result.signal else 'NO'}"
        )

    if not results_with_signals:
      logger.info(f"[{symbol}] ❌ Нет сигналов от стратегий, возврат None")
      return None

    # Подсчет по типам
    candle_count = len([r for r in results_with_signals if r.strategy_type == StrategyType.CANDLE])
    orderbook_count = len([r for r in results_with_signals if r.strategy_type == StrategyType.ORDERBOOK])
    hybrid_count = len([r for r in results_with_signals if r.strategy_type == StrategyType.HYBRID])

    # Анализ согласованности
    buy_signals = [r for r in results_with_signals if r.signal.signal_type == SignalType.BUY]
    sell_signals = [r for r in results_with_signals if r.signal.signal_type == SignalType.SELL]

    # ========== ПРОВЕРКА КОНФЛИКТА НАПРАВЛЕНИЙ ==========
    # Если есть и BUY и SELL сигналы - это конфликт
    if buy_signals and sell_signals:
      total_with_direction = len(buy_signals) + len(sell_signals)
      buy_ratio = len(buy_signals) / total_with_direction
      sell_ratio = len(sell_signals) / total_with_direction

      # Вычисляем dominance (насколько одно направление доминирует)
      dominance = max(buy_ratio, sell_ratio)

      # Логируем конфликт
      logger.warning(
        f"[{symbol}] ⚠️ Direction conflict: BUY={len(buy_signals)}, SELL={len(sell_signals)}, "
        f"dominance={dominance:.2f}"
      )

      # Если конфликт сильный (dominance < 65%) - отклоняем сигнал
      if dominance < 0.65:
        logger.info(
          f"[{symbol}] ❌ Сигнал отклонён: сильный конфликт направлений "
          f"(BUY={len(buy_signals)}, SELL={len(sell_signals)}, dominance={dominance:.2f} < 0.65)"
        )
        self.conflicts_resolved += 1
        return None

      # Если конфликт средний (65-75%) - продолжаем, но помечаем
      if dominance < 0.75:
        logger.info(
          f"[{symbol}] ⚠️ Средний конфликт направлений, продолжаем с пониженной уверенностью"
        )
    # ========== END ПРОВЕРКА КОНФЛИКТА ==========

    # Проверка минимального количества стратегий
    if len(results_with_signals) < self.config.min_strategies_for_signal:
      logger.info(
        f"[{symbol}] ❌ Недостаточно стратегий с сигналами: "
        f"{len(results_with_signals)}/{self.config.min_strategies_for_signal}, возврат None"
      )
      return None

    # ========== APPLY DYNAMIC WEIGHTS ==========
    # При volume explosion увеличиваем веса volume-based стратегий
    if is_explosion:
      # Собираем все веса
      all_weights = {
        **self.config.candle_strategy_weights,
        **self.config.orderbook_strategy_weights,
        **self.config.hybrid_strategy_weights
      }
      dynamic_weights = self._get_dynamic_weights(volume_ratio, all_weights)

      # Применяем динамические веса к результатам
      for result in results_with_signals:
        if result.strategy_name in dynamic_weights:
          old_weight = result.weight
          result.weight = dynamic_weights[result.strategy_name]
          if old_weight != result.weight:
            logger.debug(
              f"[{symbol}] Dynamic weight: {result.strategy_name} "
              f"{old_weight:.0%} → {result.weight:.0%}"
            )

      # Обновляем buy_signals и sell_signals с новыми весами
      buy_signals = [r for r in results_with_signals if r.signal.signal_type == SignalType.BUY]
      sell_signals = [r for r in results_with_signals if r.signal.signal_type == SignalType.SELL]

    # Динамический порог консенсуса
    dynamic_threshold = self._get_dynamic_consensus_threshold(volume_ratio)
    if dynamic_threshold != self.config.min_consensus_confidence:
      logger.info(
        f"[{symbol}] 🎯 Dynamic consensus threshold: "
        f"{self.config.min_consensus_confidence:.0%} → {dynamic_threshold:.0%}"
      )

    # Определение консенсуса по режиму
    if self.config.consensus_mode == "weighted":
      consensus_signal = self._weighted_consensus(
        buy_signals, sell_signals, symbol, current_price, dynamic_threshold
      )
    elif self.config.consensus_mode == "majority":
      consensus_signal = self._majority_consensus(
        buy_signals, sell_signals, symbol, current_price
      )
    elif self.config.consensus_mode == "unanimous":
      consensus_signal = self._unanimous_consensus(
        results_with_signals, symbol, current_price
      )
    else:
      consensus_signal = self._weighted_consensus(
        buy_signals, sell_signals, symbol, current_price, dynamic_threshold
      )

    if not consensus_signal:
      return None

    # Обогащаем информацией о типах стратегий
    consensus_signal.candle_strategies_count = candle_count
    consensus_signal.orderbook_strategies_count = orderbook_count
    consensus_signal.hybrid_strategies_count = hybrid_count

    self.consensus_achieved += 1

    return consensus_signal

  def _weighted_consensus(
      self,
      buy_signals: List[StrategyResult],
      sell_signals: List[StrategyResult],
      symbol: str,
      current_price: float,
      min_confidence_threshold: Optional[float] = None
  ) -> Optional[ConsensusSignal]:
    """
    Взвешенный consensus на основе весов и confidence.

    Args:
        min_confidence_threshold: Динамический порог (если None, используется config)
    """
    # Используем динамический или стандартный порог
    effective_threshold = min_confidence_threshold or self.config.min_consensus_confidence

    # Взвешенные голоса
    buy_score = sum(r.weight * r.signal.confidence for r in buy_signals)
    sell_score = sum(r.weight * r.signal.confidence for r in sell_signals)

    total_score = buy_score + sell_score

    if total_score == 0:
      return None

    # Определение победителя
    if buy_score > sell_score:
      final_type = SignalType.BUY
      consensus_confidence = buy_score / total_score
      contributing = buy_signals
      agreement_count = len(buy_signals)
      disagreement_count = len(sell_signals)
    else:
      final_type = SignalType.SELL
      consensus_confidence = sell_score / total_score
      contributing = sell_signals
      agreement_count = len(sell_signals)
      disagreement_count = len(buy_signals)

    # Проверка минимальной consensus confidence (используем динамический порог)
    if consensus_confidence < effective_threshold:
      logger.info(
        f"[{symbol}] ❌ Consensus confidence слишком низкая: "
        f"{consensus_confidence:.2f} < {effective_threshold:.2f}, возврат None"
      )
      return None

    # Усредненная confidence от согласных стратегий
    avg_confidence = np.mean([r.signal.confidence for r in contributing])

    # Итоговая confidence: комбинация consensus и средней confidence
    final_confidence = (consensus_confidence + avg_confidence) / 2.0

    # Penalty за конфликт направлений (если есть несогласные стратегии)
    if disagreement_count > 0:
      # Чем больше несогласных, тем больше penalty
      conflict_ratio = disagreement_count / (agreement_count + disagreement_count)
      # Penalty от 0% (нет несогласных) до 25% (50/50 конфликт)
      conflict_penalty = conflict_ratio * 0.5  # Max 25% penalty
      final_confidence *= (1.0 - conflict_penalty)

      logger.debug(
        f"[{symbol}] Applied conflict penalty: {conflict_penalty:.2%}, "
        f"final_confidence={final_confidence:.2f}"
      )

    # Определение силы
    if final_confidence >= 0.8:
      signal_strength = SignalStrength.STRONG
    elif final_confidence >= 0.65:
      signal_strength = SignalStrength.MEDIUM
    else:
      signal_strength = SignalStrength.WEAK

    # Собираем причины
    reasons = []
    for result in contributing[:3]:  # Топ 3
      if result.signal.reason:
        reasons.append(f"{result.strategy_name}: {result.signal.reason[:50]}")

    # Создание финального сигнала
    final_signal = TradingSignal(
      symbol=symbol,
      signal_type=final_type,
      source=SignalSource.CONSENSUS,
      strength=signal_strength,
      price=current_price,
      confidence=final_confidence,
      timestamp=int(datetime.now().timestamp() * 1000),
      reason=f"Consensus ({agreement_count}/{agreement_count + disagreement_count}): " +
             " | ".join(reasons[:2]),
      metadata={
        'consensus_mode': 'weighted',
        'buy_score': buy_score,
        'sell_score': sell_score,
        'contributing_strategies': [r.strategy_name for r in contributing]
      }
    )

    consensus = ConsensusSignal(
      final_signal=final_signal,
      contributing_strategies=[r.strategy_name for r in contributing],
      agreement_count=agreement_count,
      disagreement_count=disagreement_count,
      consensus_confidence=final_confidence,
      strategy_results=contributing
    )

    self.signals_generated += 1

    return consensus

  def _majority_consensus(
      self,
      buy_signals: List[StrategyResult],
      sell_signals: List[StrategyResult],
      symbol: str,
      current_price: float
  ) -> Optional[ConsensusSignal]:
    """Простое большинство голосов."""
    if len(buy_signals) > len(sell_signals):
      final_type = SignalType.BUY
      contributing = buy_signals
      agreement_count = len(buy_signals)
      disagreement_count = len(sell_signals)
    elif len(sell_signals) > len(buy_signals):
      final_type = SignalType.SELL
      contributing = sell_signals
      agreement_count = len(sell_signals)
      disagreement_count = len(buy_signals)
    else:
      # Ничья - нет consensus
      return None

    # Consensus confidence = процент согласия
    total_strategies = agreement_count + disagreement_count
    consensus_confidence = agreement_count / total_strategies

    if consensus_confidence < self.config.min_consensus_confidence:
      return None

    # Средняя confidence
    avg_confidence = np.mean([r.signal.confidence for r in contributing])
    final_confidence = avg_confidence

    signal_strength = SignalStrength.MEDIUM

    final_signal = TradingSignal(
      symbol=symbol,
      signal_type=final_type,
      source=SignalSource.CONSENSUS,
      strength=signal_strength,
      price=current_price,
      confidence=float(final_confidence),
      timestamp=int(datetime.now().timestamp() * 1000),
      reason=f"Majority consensus: {agreement_count}/{total_strategies}",
      metadata={'consensus_mode': 'majority'}
    )

    consensus = ConsensusSignal(
      final_signal=final_signal,
      contributing_strategies=[r.strategy_name for r in contributing],
      agreement_count=agreement_count,
      disagreement_count=disagreement_count,
      consensus_confidence=consensus_confidence,
      strategy_results=contributing
    )

    self.signals_generated += 1

    return consensus

  def _unanimous_consensus(
      self,
      results_with_signals: List[StrategyResult],
      symbol: str,
      current_price: float
  ) -> Optional[ConsensusSignal]:
    """Единогласное согласие всех стратегий."""
    if not results_with_signals:
      return None

    # Проверяем что все сигналы одинаковые
    first_signal_type = results_with_signals[0].signal.signal_type

    if not all(r.signal.signal_type == first_signal_type for r in results_with_signals):
      return None

    # Все согласны!
    avg_confidence = np.mean([r.signal.confidence for r in results_with_signals])

    final_signal = TradingSignal(
      symbol=symbol,
      signal_type=first_signal_type,
      source=SignalSource.CONSENSUS,
      strength=SignalStrength.STRONG,
      price=current_price,
      confidence=float(avg_confidence),
      timestamp=int(datetime.now().timestamp() * 1000),
      reason=f"Unanimous consensus: all {len(results_with_signals)} strategies agree",
      metadata={'consensus_mode': 'unanimous'}
    )

    consensus = ConsensusSignal(
      final_signal=final_signal,
      contributing_strategies=[r.strategy_name for r in results_with_signals],
      agreement_count=len(results_with_signals),
      disagreement_count=0,
      consensus_confidence=1.0,
      strategy_results=results_with_signals
    )

    self.signals_generated += 1

    return consensus

  def analyze_with_consensus(
      self,
      symbol: str,
      candles: List[Candle],
      current_price: float,
      orderbook: Optional[OrderBookSnapshot] = None,
      metrics: Optional[OrderBookMetrics] = None,
      sr_levels: Optional[List] = None,
      volume_profile: Optional[Dict] = None,
      ml_prediction: Optional[Dict] = None,
      market_trades: Optional[List] = None  # НОВОЕ: Market trades
  ) -> Optional[ConsensusSignal]:
    """
    Полный анализ с генерацией consensus сигнала.

    Удобный метод, объединяющий analyze_all_strategies + build_consensus.
    """
    # Шаг 1: Запускаем все стратегии
    results = self.analyze_all_strategies(
      symbol=symbol,
      candles=candles,
      current_price=current_price,
      orderbook=orderbook,
      metrics=metrics,
      sr_levels=sr_levels,
      volume_profile=volume_profile,
      ml_prediction=ml_prediction,
      market_trades=market_trades  # Передаем market_trades
    )

    # Шаг 2: Строим consensus
    consensus = self.build_consensus(symbol, results, current_price)

    if consensus:
      logger.info(
        f"✅ CONSENSUS [{symbol}]: {safe_enum_value(consensus.final_signal.signal_type)}, "
        f"confidence={consensus.consensus_confidence:.2f}, "
        f"agreement={consensus.agreement_count}/"
        f"{consensus.agreement_count + consensus.disagreement_count}, "
        f"strategies={', '.join(consensus.contributing_strategies)}"
      )

    return consensus

  def get_statistics(self) -> Dict:
    """Получить статистику работы менеджера."""
    consensus_rate = (
      self.consensus_achieved / self.signals_generated
      if self.signals_generated > 0 else 0.0
    )

    # Статистика кэша
    total_cache_requests = self._cache_hits + self._cache_misses
    cache_hit_rate = (
      self._cache_hits / total_cache_requests
      if total_cache_requests > 0 else 0.0
    )

    # Статистика по каждой стратегии
    strategy_stats = {}
    for name, strategy in self.all_strategies.items():
      strategy_stats[name] = strategy.get_statistics()

    return {
      'total_analyses': self.total_analyses,
      'signals_generated': self.signals_generated,
      'consensus_achieved': self.consensus_achieved,
      'conflicts_resolved': self.conflicts_resolved,
      'consensus_rate': consensus_rate,
      'candle_strategies_count': len(self.candle_strategies),
      'orderbook_strategies_count': len(self.orderbook_strategies),
      'hybrid_strategies_count': len(self.hybrid_strategies),
      # Статистика кэша
      'cache_hits': self._cache_hits,
      'cache_misses': self._cache_misses,
      'cache_hit_rate': cache_hit_rate,
      'cache_size': len(self._strategy_cache),
      'strategies': strategy_stats
    }
#
# class StrategyManager:
#   """
#   Менеджер торговых стратегий.
#
#   Управляет множественными стратегиями и объединяет их сигналы.
#   """
#
#   def __init__(self, config: StrategyManagerConfig):
#     """
#     Инициализация менеджера.
#
#     Args:
#         config: Конфигурация менеджера
#     """
#     self.config = config
#
#     # Инициализация стратегий
#     self.strategies: Dict[str, any] = {}
#
#     # Momentum Strategy
#     self.strategies['momentum'] = MomentumStrategy(MomentumConfig())
#
#     # SAR Wave Strategy
#     self.strategies['sar_wave'] = SARWaveStrategy(SARWaveConfig())
#
#     # SuperTrend Strategy
#     self.strategies['supertrend'] = SuperTrendStrategy(SuperTrendConfig())
#
#     # Volume Profile Strategy
#     self.strategies['volume_profile'] = VolumeProfileStrategy(VolumeProfileConfig())
#
#     # Статистика
#     self.total_analyses = 0
#     self.signals_generated = 0
#     self.consensus_achieved = 0
#     self.conflicts_resolved = 0
#
#     logger.info(
#       f"Инициализирован StrategyManager: "
#       f"strategies={list(self.strategies.keys())}, "
#       f"consensus_mode={config.consensus_mode}, "
#       f"min_strategies={config.min_strategies_for_signal}"
#     )
#
#   def analyze_all_strategies(
#       self,
#       symbol: str,
#       candles: List[Candle],
#       current_price: float
#   ) -> List[StrategyResult]:
#     """
#     Запустить все стратегии для анализа.
#
#     Returns:
#         Список результатов от каждой стратегии
#     """
#     import time
#
#     results = []
#
#     for strategy_name, strategy in self.strategies.items():
#       start_time = time.time()
#
#       try:
#         signal = strategy.analyze(symbol, candles, current_price)
#         execution_time = (time.time() - start_time) * 1000  # ms
#
#         result = StrategyResult(
#           strategy_name=strategy_name,
#           signal=signal,
#           priority=self.config.strategy_priorities.get(
#             strategy_name,
#             StrategyPriority.MEDIUM
#           ),
#           weight=self.config.strategy_weights.get(strategy_name, 0.25),
#           execution_time_ms=execution_time
#         )
#
#         results.append(result)
#
#         if signal:
#           logger.debug(
#             f"[{strategy_name}] Signal: {safe_enum_value(signal.signal_type)}, "
#             f"confidence={signal.confidence:.2f}"
#           )
#
#       except Exception as e:
#         logger.error(f"Ошибка в стратегии {strategy_name}: {e}", exc_info=True)
#         continue
#
#     self.total_analyses += 1
#
#     return results
#
#   def build_consensus(
#       self,
#       symbol: str,
#       strategy_results: List[StrategyResult],
#       current_price: float
#   ) -> Optional[ConsensusSignal]:
#     """
#     Построить consensus сигнал из результатов стратегий.
#
#     Args:
#         symbol: Торговая пара
#         strategy_results: Результаты от всех стратегий
#         current_price: Текущая цена
#
#     Returns:
#         ConsensusSignal или None
#     """
#     # Фильтруем результаты с сигналами
#     results_with_signals = [r for r in strategy_results if r.signal is not None]
#
#     if len(results_with_signals) < self.config.min_strategies_for_signal:
#       logger.debug(
#         f"Недостаточно стратегий для consensus: "
#         f"{len(results_with_signals)} < {self.config.min_strategies_for_signal}"
#       )
#       return None
#
#     # Группируем по типу сигнала
#     buy_signals = [r for r in results_with_signals if r.signal.signal_type == SignalType.BUY]
#     sell_signals = [r for r in results_with_signals if r.signal.signal_type == SignalType.SELL]
#
#     # Логируем распределение
#     logger.debug(
#       f"[{symbol}] Consensus распределение: "
#       f"BUY={len(buy_signals)}, SELL={len(sell_signals)}, "
#       f"total={len(results_with_signals)}"
#     )
#
#     # Определяем consensus направление
#     if self.config.consensus_mode == "majority":
#       consensus_signal = self._majority_consensus(
#         buy_signals,
#         sell_signals,
#         strategy_results
#       )
#
#     elif self.config.consensus_mode == "weighted":
#       consensus_signal = self._weighted_consensus(
#         buy_signals,
#         sell_signals,
#         strategy_results
#       )
#
#     elif self.config.consensus_mode == "unanimous":
#       consensus_signal = self._unanimous_consensus(
#         buy_signals,
#         sell_signals,
#         strategy_results
#       )
#
#     else:
#       logger.error(f"Неизвестный consensus_mode: {self.config.consensus_mode}")
#       return None
#
#     if consensus_signal:
#       self.signals_generated += 1
#       if consensus_signal.agreement_count == len(results_with_signals):
#         self.consensus_achieved += 1
#       else:
#         self.conflicts_resolved += 1
#
#     return consensus_signal
#
#   def _majority_consensus(
#       self,
#       buy_signals: List[StrategyResult],
#       sell_signals: List[StrategyResult],
#       all_results: List[StrategyResult]
#   ) -> Optional[ConsensusSignal]:
#     """
#     Majority voting consensus.
#
#     ИСПРАВЛЕНИЕ: Добавлена проверка минимального количества в побеждающей группе.
#     """
#     if len(buy_signals) > len(sell_signals):
#       signal_type = SignalType.BUY
#       contributing = buy_signals
#       agreement = len(buy_signals)
#       disagreement = len(sell_signals)
#     elif len(sell_signals) > len(buy_signals):
#       signal_type = SignalType.SELL
#       contributing = sell_signals
#       agreement = len(sell_signals)
#       disagreement = len(buy_signals)
#     else:
#       # Равное количество - конфликт
#       return self._resolve_conflict(buy_signals, sell_signals, all_results)
#
#     # ========================================
#     # ИСПРАВЛЕНИЕ: Проверка минимального количества в побеждающей группе
#     # ========================================
#     if len(contributing) < self.config.min_strategies_for_signal:
#       logger.info(
#         f"Majority consensus отклонен: недостаточно стратегий в побеждающей группе "
#         f"({len(contributing)} < {self.config.min_strategies_for_signal}). "
#         f"BUY={len(buy_signals)}, SELL={len(sell_signals)}"
#       )
#       return None
#
#     # Средняя уверенность
#     avg_confidence = float(np.mean([r.signal.confidence for r in contributing]))
#
#     if avg_confidence < self.config.min_consensus_confidence:
#       logger.debug(
#         f"Majority consensus отклонен: низкая confidence "
#         f"({avg_confidence:.2f} < {self.config.min_consensus_confidence})"
#       )
#       return None
#
#     # Создаем consensus сигнал
#     final_signal = TradingSignal(
#       symbol=contributing[0].signal.symbol,
#       signal_type=signal_type,
#       source=SignalSource.STRATEGY,
#       strength=self._determine_strength(avg_confidence),
#       price=contributing[0].signal.price,
#       confidence=avg_confidence,
#       timestamp=int(datetime.now().timestamp() * 1000),
#       reason=f"Majority consensus: {agreement} strategies agree",
#       metadata={
#         'consensus_mode': 'majority',
#         'contributing_strategies': [r.strategy_name for r in contributing],
#         'agreement_count': agreement,
#         'disagreement_count': disagreement
#       }
#     )
#
#     return ConsensusSignal(
#       final_signal=final_signal,
#       contributing_strategies=[r.strategy_name for r in contributing],
#       agreement_count=agreement,
#       disagreement_count=disagreement,
#       consensus_confidence=avg_confidence,
#       strategy_results=all_results
#     )
#
#   def _weighted_consensus(
#       self,
#       buy_signals: List[StrategyResult],
#       sell_signals: List[StrategyResult],
#       all_results: List[StrategyResult]
#   ) -> Optional[ConsensusSignal]:
#     """
#     Weighted voting consensus.
#
#     ИСПРАВЛЕНИЕ: Добавлена проверка минимального количества в побеждающей группе.
#     """
#     # Вычисляем взвешенные голоса
#     buy_weight = sum(r.weight * r.signal.confidence for r in buy_signals)
#     sell_weight = sum(r.weight * r.signal.confidence for r in sell_signals)
#
#     logger.debug(
#       f"Weighted voting: BUY_weight={buy_weight:.4f} ({len(buy_signals)} strategies), "
#       f"SELL_weight={sell_weight:.4f} ({len(sell_signals)} strategies)"
#     )
#
#     if buy_weight > sell_weight:
#       signal_type = SignalType.BUY
#       contributing = buy_signals
#       winning_weight = buy_weight
#     elif sell_weight > buy_weight:
#       signal_type = SignalType.SELL
#       contributing = sell_signals
#       winning_weight = sell_weight
#     else:
#       logger.debug("Weighted voting: равные веса - переход к conflict resolution")
#       return self._resolve_conflict(buy_signals, sell_signals, all_results)
#
#     # ========================================
#     # ИСПРАВЛЕНИЕ: Проверка минимального количества в побеждающей группе
#     # ========================================
#     if len(contributing) < self.config.min_strategies_for_signal:
#       logger.info(
#         f"Weighted consensus отклонен: недостаточно стратегий в побеждающей группе "
#         f"({len(contributing)} < {self.config.min_strategies_for_signal}). "
#         f"{signal_type.value} group: {[r.strategy_name for r in contributing]}, "
#         f"weight={winning_weight:.4f}"
#       )
#       return None
#
#     # Вычисляем consensus confidence
#     consensus_confidence = winning_weight / sum(r.weight for r in contributing)
#
#     logger.debug(
#       f"Weighted consensus confidence: {consensus_confidence:.4f} "
#       f"(winning_weight={winning_weight:.4f} / sum_weights={sum(r.weight for r in contributing):.4f})"
#     )
#
#     if consensus_confidence < self.config.min_consensus_confidence:
#       logger.info(
#         f"Weighted consensus отклонен: низкая confidence "
#         f"({consensus_confidence:.4f} < {self.config.min_consensus_confidence})"
#       )
#       return None
#
#     final_signal = TradingSignal(
#       symbol=contributing[0].signal.symbol,
#       signal_type=signal_type,
#       source=SignalSource.STRATEGY,
#       strength=self._determine_strength(consensus_confidence),
#       price=contributing[0].signal.price,
#       confidence=consensus_confidence,
#       timestamp=int(datetime.now().timestamp() * 1000),
#       reason=f"Weighted consensus: {len(contributing)} strategies (total_weight={winning_weight:.2f})",
#       metadata={
#         'consensus_mode': 'weighted',
#         'contributing_strategies': [r.strategy_name for r in contributing],
#         'buy_weight': buy_weight,
#         'sell_weight': sell_weight,
#         'winning_weight': winning_weight,
#         'consensus_confidence': consensus_confidence
#       }
#     )
#
#     return ConsensusSignal(
#       final_signal=final_signal,
#       contributing_strategies=[r.strategy_name for r in contributing],
#       agreement_count=len(contributing),
#       disagreement_count=len(buy_signals if signal_type == SignalType.SELL else sell_signals),
#       consensus_confidence=consensus_confidence,
#       strategy_results=all_results
#     )
#
#   def _unanimous_consensus(
#       self,
#       buy_signals: List[StrategyResult],
#       sell_signals: List[StrategyResult],
#       all_results: List[StrategyResult]
#   ) -> Optional[ConsensusSignal]:
#     """
#     Unanimous consensus - все стратегии должны согласиться.
#
#     ИСПРАВЛЕНИЕ: Добавлена проверка минимального количества.
#     """
#     results_with_signals = [r for r in all_results if r.signal is not None]
#
#     # Все должны быть BUY или все SELL
#     if len(buy_signals) == len(results_with_signals):
#       signal_type = SignalType.BUY
#       contributing = buy_signals
#     elif len(sell_signals) == len(results_with_signals):
#       signal_type = SignalType.SELL
#       contributing = sell_signals
#     else:
#       # Нет единогласия
#       logger.debug(
#         f"Unanimous consensus невозможен: BUY={len(buy_signals)}, SELL={len(sell_signals)}"
#       )
#       return None
#
#     # ========================================
#     # ИСПРАВЛЕНИЕ: Проверка минимального количества
#     # ========================================
#     if len(contributing) < self.config.min_strategies_for_signal:
#       logger.info(
#         f"Unanimous consensus отклонен: недостаточно стратегий "
#         f"({len(contributing)} < {self.config.min_strategies_for_signal})"
#       )
#       return None
#
#     avg_confidence = float(np.mean([r.signal.confidence for r in contributing]))
#
#     if avg_confidence < self.config.min_consensus_confidence:
#       logger.debug(
#         f"Unanimous consensus отклонен: низкая confidence "
#         f"({avg_confidence:.2f} < {self.config.min_consensus_confidence})"
#       )
#       return None
#
#     final_signal = TradingSignal(
#       symbol=contributing[0].signal.symbol,
#       signal_type=signal_type,
#       source=SignalSource.STRATEGY,
#       strength=SignalStrength.STRONG,
#       price=contributing[0].signal.price,
#       confidence=avg_confidence,
#       timestamp=int(datetime.now().timestamp() * 1000),
#       reason=f"Unanimous consensus: ALL {len(contributing)} strategies agree",
#       metadata={
#         'consensus_mode': 'unanimous',
#         'contributing_strategies': [r.strategy_name for r in contributing]
#       }
#     )
#
#     return ConsensusSignal(
#       final_signal=final_signal,
#       contributing_strategies=[r.strategy_name for r in contributing],
#       agreement_count=len(contributing),
#       disagreement_count=0,
#       consensus_confidence=avg_confidence,
#       strategy_results=all_results
#     )
#
#   def _resolve_conflict(
#       self,
#       buy_signals: List[StrategyResult],
#       sell_signals: List[StrategyResult],
#       all_results: List[StrategyResult]
#   ) -> Optional[ConsensusSignal]:
#     """Разрешить конфликт между стратегиями."""
#     logger.info(
#       f"Конфликт стратегий: BUY={len(buy_signals)}, SELL={len(sell_signals)}, "
#       f"resolution={self.config.conflict_resolution}"
#     )
#
#     if self.config.conflict_resolution == "cancel":
#       logger.info("Конфликт стратегий: отмена сигнала")
#       return None
#
#     elif self.config.conflict_resolution == "highest_confidence":
#       # Выбираем сигнал с наибольшей уверенностью
#       all_signals = buy_signals + sell_signals
#       best = max(all_signals, key=lambda r: r.signal.confidence)
#
#       logger.info(
#         f"Конфликт разрешен: выбран {best.strategy_name} "
#         f"с confidence={best.signal.confidence:.2f}"
#       )
#
#       return ConsensusSignal(
#         final_signal=best.signal,
#         contributing_strategies=[best.strategy_name],
#         agreement_count=1,
#         disagreement_count=len(all_signals) - 1,
#         consensus_confidence=best.signal.confidence,
#         strategy_results=all_results
#       )
#
#     elif self.config.conflict_resolution == "priority":
#       # Выбираем по приоритету
#       all_signals = buy_signals + sell_signals
#       best = max(all_signals, key=lambda r: r.priority.value)
#
#       logger.info(
#         f"Конфликт разрешен: выбран {best.strategy_name} "
#         f"с приоритетом={best.priority.name}"
#       )
#
#       return ConsensusSignal(
#         final_signal=best.signal,
#         contributing_strategies=[best.strategy_name],
#         agreement_count=1,
#         disagreement_count=len(all_signals) - 1,
#         consensus_confidence=best.signal.confidence,
#         strategy_results=all_results
#       )
#
#     return None
#
#   def _determine_strength(self, confidence: float) -> SignalStrength:
#     """Определить силу сигнала по confidence."""
#     if confidence >= 0.8:
#       return SignalStrength.STRONG
#     elif confidence >= 0.6:
#       return SignalStrength.MEDIUM
#     else:
#       return SignalStrength.WEAK
#
#   def analyze_with_consensus(
#       self,
#       symbol: str,
#       candles: List[Candle],
#       current_price: float
#   ) -> Optional[ConsensusSignal]:
#     """
#     Полный анализ: запустить все стратегии и построить consensus.
#
#     Returns:
#         ConsensusSignal или None
#     """
#     # Запускаем все стратегии
#     strategy_results = self.analyze_all_strategies(symbol, candles, current_price)
#
#     # Строим consensus
#     consensus = self.build_consensus(symbol, strategy_results, current_price)
#
#     if consensus:
#       logger.info(
#         f"✅ CONSENSUS SIGNAL [{symbol}]: {consensus.final_signal.signal_type.value}, "
#         f"confidence={consensus.consensus_confidence:.2f}, "
#         f"strategies={consensus.contributing_strategies}, "
#         f"agreement={consensus.agreement_count}/{consensus.agreement_count + consensus.disagreement_count}"
#       )
#     else:
#       logger.debug(
#         f"❌ NO CONSENSUS [{symbol}]: не достигнут консенсус стратегий"
#       )
#
#     return consensus
#
#   def get_statistics(self) -> Dict:
#     """Получить статистику менеджера."""
#     consensus_rate = (
#       self.consensus_achieved / self.signals_generated
#       if self.signals_generated > 0
#       else 0.0
#     )
#
#     # Статистика по каждой стратегии
#     strategy_stats = {}
#     for name, strategy in self.strategies.items():
#       strategy_stats[name] = strategy.get_statistics()
#
#     return {
#       'total_analyses': self.total_analyses,
#       'signals_generated': self.signals_generated,
#       'consensus_achieved': self.consensus_achieved,
#       'conflicts_resolved': self.conflicts_resolved,
#       'consensus_rate': consensus_rate,
#       'strategies': strategy_stats
#     }