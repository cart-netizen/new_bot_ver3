"""
Ядро торговой стратегии.
Генерация торговых сигналов на основе анализа стакана.
"""

from typing import Dict, List, Optional
from collections import deque

from backend.core.logger import get_logger
from backend.core.exceptions import StrategyError
from backend.ml_engine.features import FeatureVector
from backend.models.orderbook import OrderBookMetrics
from backend.models.signal import (
  TradingSignal,
  SignalType,
  SignalStrength,
  SignalSource,
  SignalStatistics
)
from backend.config import settings
from backend.utils.helpers import safe_enum_value

logger = get_logger(__name__)


class StrategyEngine:
  """Движок торговой стратегии."""

  def __init__(self):
    """Инициализация стратегии."""
    self.signals_history: Dict[str, deque] = {}
    self.signal_statistics: Dict[str, SignalStatistics] = {}
    self.max_history_size = 100

    # Пороги из конфигурации
    self.imbalance_buy_threshold = settings.IMBALANCE_BUY_THRESHOLD
    self.imbalance_sell_threshold = settings.IMBALANCE_SELL_THRESHOLD

    # ===== НОВЫЕ ПАРАМЕТРЫ ДЛЯ ML =====
    self.ml_enabled = True  # Включаем использование ML признаков
    self.ml_weight = 0.5  # Вес ML корректировок (0.0-1.0)

    logger.info(
      f"Инициализирована торговая стратегия: "
      f"buy_threshold={self.imbalance_buy_threshold}, "
      f"sell_threshold={self.imbalance_sell_threshold}, "
      f"ml_enabled={self.ml_enabled}"
    )

  def analyze_and_generate_signal(
        self,
        symbol: str,
        metrics: OrderBookMetrics,
        features: Optional[FeatureVector] = None  # ← НОВЫЙ ПАРАМЕТР
    ) -> Optional[TradingSignal]:
      """
      Анализ метрик и генерация торгового сигнала с учетом ML признаков.

      Args:
          symbol: Торговая пара
          metrics: Базовые метрики стакана
          features: ML признаки (опционально)

      Returns:
          TradingSignal: Торговый сигнал или None
      """
      try:
        # Инициализируем историю и статистику для символа
        if symbol not in self.signals_history:
          self.signals_history[symbol] = deque(maxlen=self.max_history_size)
          self.signal_statistics[symbol] = SignalStatistics(symbol=symbol)

        # Проверяем наличие необходимых данных
        if metrics.best_bid is None or metrics.best_ask is None:
          logger.debug(f"{symbol} | Недостаточно данных для генерации сигнала")
          return None

        # ===== БАЗОВЫЙ АНАЛИЗ ДИСБАЛАНСА =====
        signal = self._analyze_imbalance(symbol, metrics)

        # ===== ОБОГАЩЕНИЕ СИГНАЛА ML ПРИЗНАКАМИ =====
        if signal and features and self.ml_enabled:
          signal = self._enhance_signal_with_ml(signal, features, metrics)

        if signal:
          # Сохраняем сигнал в истории
          self.signals_history[symbol].append(signal)
          self._update_statistics(symbol, signal)

          logger.info(
            f"{symbol} | Сгенерирован сигнал: "
            f"{safe_enum_value(signal.signal_type)} "
            f"(strength={signal.strength.value}, "
            f"confidence={signal.confidence:.2f}, "
            f"ml_enhanced={features is not None})"
          )

        return signal

      except Exception as e:
        logger.error(f"{symbol} | Ошибка генерации сигнала: {e}")
        raise StrategyError(f"Failed to generate signal: {str(e)}")

  def _enhance_signal_with_ml(
      self,
      signal: TradingSignal,
      features: FeatureVector,
      metrics: OrderBookMetrics
  ) -> TradingSignal:
    """
    Обогащение сигнала ML признаками.

    Использует дополнительные признаки для фильтрации и корректировки уверенности.

    Args:
        signal: Базовый сигнал
        features: ML признаки
        metrics: Базовые метрики

    Returns:
        TradingSignal: Обогащенный сигнал
    """
    try:
      # Извлекаем каналы признаков (ВЫЗОВ МЕТОДА!)
      channels = features.to_channels()
      orderbook_features = channels.get("orderbook")
      candle_features = channels.get("candle")
      indicator_features = channels.get("indicator")

      # Начальная confidence
      original_confidence = signal.confidence
      adjusted_confidence = signal.confidence

      # ===== ФИЛЬТР 1: ВОЛАТИЛЬНОСТЬ =====
      # candle_features[16] = realized_volatility
      if candle_features is not None and len(candle_features) > 16:
        volatility = candle_features[16]

        # Если волатильность слишком высокая - снижаем уверенность
        if volatility > 0.05:  # 5%
          adjusted_confidence *= 0.7
          if signal.strength == SignalStrength.STRONG:
            signal.strength = SignalStrength.MEDIUM
          elif signal.strength == SignalStrength.MEDIUM:
            signal.strength = SignalStrength.WEAK

          logger.debug(
            f"{signal.symbol} | Высокая волатильность ({volatility:.4f}), "
            f"снижена уверенность: {original_confidence:.2f} → {adjusted_confidence:.2f}"
          )

      # ===== ФИЛЬТР 2: RSI (ПЕРЕКУПЛЕННОСТЬ/ПЕРЕПРОДАННОСТЬ) =====
      # indicator_features[5] = RSI
      if indicator_features is not None and len(indicator_features) > 5:
        rsi = indicator_features[5]

        # Усиливаем сигналы в экстремальных зонах
        if signal.signal_type == SignalType.BUY and rsi < 30:
          # Перепроданность - усиливаем BUY
          adjusted_confidence *= 1.15
          logger.debug(
            f"{signal.symbol} | RSI={rsi:.2f} (перепроданность), "
            f"усилен BUY: {original_confidence:.2f} → {adjusted_confidence:.2f}"
          )
        elif signal.signal_type == SignalType.SELL and rsi > 70:
          # Перекупленность - усиливаем SELL
          adjusted_confidence *= 1.15
          logger.debug(
            f"{signal.symbol} | RSI={rsi:.2f} (перекупленность), "
            f"усилен SELL: {original_confidence:.2f} → {adjusted_confidence:.2f}"
          )
        # Фильтруем противоречащие сигналы
        elif signal.signal_type == SignalType.BUY and rsi > 70:
          # Попытка купить в перекупленности - снижаем
          adjusted_confidence *= 0.6
          logger.debug(
            f"{signal.symbol} | RSI={rsi:.2f} (перекупленность), "
            f"ослаблен BUY: {original_confidence:.2f} → {adjusted_confidence:.2f}"
          )
        elif signal.signal_type == SignalType.SELL and rsi < 30:
          # Попытка продать в перепроданности - снижаем
          adjusted_confidence *= 0.6
          logger.debug(
            f"{signal.symbol} | RSI={rsi:.2f} (перепроданность), "
            f"ослаблен SELL: {original_confidence:.2f} → {adjusted_confidence:.2f}"
          )

      # ===== ФИЛЬТР 3: MACD ПОДТВЕРЖДЕНИЕ =====
      # indicator_features[2] = MACD
      if indicator_features is not None and len(indicator_features) > 2:
        macd = indicator_features[2]

        # MACD должен подтверждать направление
        if signal.signal_type == SignalType.BUY and macd > 0:
          adjusted_confidence *= 1.1  # Подтверждение
        elif signal.signal_type == SignalType.SELL and macd < 0:
          adjusted_confidence *= 1.1  # Подтверждение
        elif signal.signal_type == SignalType.BUY and macd < -50:
          adjusted_confidence *= 0.8  # Противоречие
        elif signal.signal_type == SignalType.SELL and macd > 50:
          adjusted_confidence *= 0.8  # Противоречие

      # Ограничиваем confidence диапазоном [0.0, 1.0]
      signal.confidence = max(0.0, min(1.0, adjusted_confidence))

      # Корректируем силу сигнала на основе итоговой confidence
      if signal.confidence >= 0.8:
        signal.strength = SignalStrength.STRONG
      elif signal.confidence >= 0.6:
        signal.strength = SignalStrength.MEDIUM
      else:
        signal.strength = SignalStrength.WEAK

      # Добавляем ML метаданные
      signal.ml_features_used = True
      signal.feature_count = features.feature_count

      # Логируем итоговую корректировку
      if abs(signal.confidence - original_confidence) > 0.05:
        logger.info(
          f"{signal.symbol} | ML корректировка: "
          f"{original_confidence:.2f} → {signal.confidence:.2f} "
          f"({signal.strength.value})"
        )

      return signal

    except Exception as e:
      logger.error(f"Ошибка обогащения сигнала ML: {e}")
      # Возвращаем исходный сигнал в случае ошибки
      signal.ml_features_used = False
      return signal

  def _analyze_imbalance(
      self,
      symbol: str,
      metrics: OrderBookMetrics
  ) -> Optional[TradingSignal]:
    """
    Анализ дисбаланса для генерации сигнала.

    Args:
        symbol: Торговая пара
        metrics: Метрики стакана

    Returns:
        TradingSignal: Сигнал или None
    """
    imbalance = metrics.imbalance

    # Проверяем сигнал на покупку
    if imbalance >= self.imbalance_buy_threshold:
      # Определяем силу сигнала
      if imbalance >= 0.85:
        strength = SignalStrength.STRONG
      elif imbalance >= 0.78:
        strength = SignalStrength.MEDIUM
      else:
        strength = SignalStrength.WEAK

      signal = TradingSignal(
        symbol=symbol,
        signal_type=SignalType.BUY,
        strength=strength,
        source=SignalSource.IMBALANCE,
        timestamp=metrics.timestamp,
        price=metrics.mid_price or metrics.best_ask,
        confidence=imbalance,
        imbalance=imbalance,
        reason=f"Дисбаланс {imbalance:.4f} превышает порог покупки {self.imbalance_buy_threshold}"
      )

      return signal

    # Проверяем сигнал на продажу
    elif imbalance <= self.imbalance_sell_threshold:
      # Определяем силу сигнала
      if imbalance <= 0.15:
        strength = SignalStrength.STRONG
      elif imbalance <= 0.22:
        strength = SignalStrength.MEDIUM
      else:
        strength = SignalStrength.WEAK

      signal = TradingSignal(
        symbol=symbol,
        signal_type=SignalType.SELL,
        strength=strength,
        source=SignalSource.IMBALANCE,
        timestamp=metrics.timestamp,
        price=metrics.mid_price or metrics.best_bid,
        confidence=1.0 - imbalance,  # Инвертируем для sell
        imbalance=imbalance,
        reason=f"Дисбаланс {imbalance:.4f} ниже порога продажи {self.imbalance_sell_threshold}"
      )

      return signal

    # Нет сигнала
    return None

  def _update_statistics(self, symbol: str, signal: TradingSignal):
    """
    Обновление статистики сигналов.

    Args:
        symbol: Торговая пара
        signal: Торговый сигнал
    """
    stats = self.signal_statistics[symbol]

    stats.total_signals += 1

    # По типу
    if signal.signal_type == SignalType.BUY:
      stats.buy_signals += 1
    elif signal.signal_type == SignalType.SELL:
      stats.sell_signals += 1
    else:
      stats.hold_signals += 1

    # По силе
    if signal.strength == SignalStrength.STRONG:
      stats.strong_signals += 1
    elif signal.strength == SignalStrength.MEDIUM:
      stats.medium_signals += 1
    else:
      stats.weak_signals += 1

    # По источнику
    if signal.source == SignalSource.IMBALANCE:
      stats.imbalance_signals += 1
    elif signal.source == SignalSource.CLUSTER:
      stats.cluster_signals += 1
    elif signal.source == SignalSource.VOLUME:
      stats.volume_signals += 1
    else:
      stats.combined_signals += 1

    # Статус исполнения
    if signal.executed:
      stats.executed_signals += 1
    else:
      stats.pending_signals += 1

    # Средняя уверенность
    if stats.total_signals > 0:
      # Пересчитываем среднюю уверенность
      total_confidence = (stats.avg_confidence * (stats.total_signals - 1) +
                          signal.confidence)
      stats.avg_confidence = total_confidence / stats.total_signals

  def get_signal_history(
      self,
      symbol: str,
      limit: Optional[int] = None
  ) -> List[TradingSignal]:
    """
    Получение истории сигналов для символа.

    Args:
        symbol: Торговая пара
        limit: Максимальное количество сигналов

    Returns:
        List[TradingSignal]: Список сигналов
    """
    if symbol not in self.signals_history:
      return []

    history = list(self.signals_history[symbol])

    if limit:
      history = history[-limit:]

    return history

  def get_statistics(self, symbol: str) -> Optional[SignalStatistics]:
    """
    Получение статистики сигналов для символа.

    Args:
        symbol: Торговая пара

    Returns:
        SignalStatistics: Статистика или None
    """
    return self.signal_statistics.get(symbol)

  def get_all_statistics(self) -> Dict[str, SignalStatistics]:
    """
    Получение статистики для всех символов.

    Returns:
        Dict[str, SignalStatistics]: Словарь статистики
    """
    return self.signal_statistics.copy()

  def clear_history(self, symbol: Optional[str] = None):
    """
    Очистка истории сигналов.

    Args:
        symbol: Торговая пара (если None, очищается вся история)
    """
    if symbol:
      if symbol in self.signals_history:
        self.signals_history[symbol].clear()
        logger.info(f"История сигналов для {symbol} очищена")
    else:
      self.signals_history.clear()
      logger.info("Вся история сигналов очищена")