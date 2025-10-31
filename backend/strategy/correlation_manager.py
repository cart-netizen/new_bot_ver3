"""
Менеджер корреляций для диверсификации портфеля.

Функциональность:
- Расчет rolling correlation между торговыми парами
- Группировка по корреляции (threshold > 0.7)
- Ограничение позиций в коррелирующих группах
- Интеграция с RiskManager

Путь: backend/strategy/correlation_manager.py
"""
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

from core.logger import get_logger
from config import settings
from strategy.risk_models import CorrelationGroup
from exchange.rest_client import rest_client

logger = get_logger(__name__)


class CorrelationCalculator:
  """
  Расчет корреляций между торговыми парами.

  Использует Pearson correlation coefficient на основе
  исторических price returns (30 дней по умолчанию).
  """

  @staticmethod
  def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """
    Расчет процентных изменений (returns).

    Args:
        prices: Массив цен [price1, price2, ...]

    Returns:
        np.ndarray: Массив returns [r1, r2, ...]
    """
    if len(prices) < 2:
      return np.array([])

    # Return = (price[t] - price[t-1]) / price[t-1]
    returns = np.diff(prices) / prices[:-1]
    return returns

  @staticmethod
  def calculate_correlation(
      returns_a: np.ndarray,
      returns_b: np.ndarray
  ) -> float:
    """
    Расчет Pearson correlation coefficient.

    Args:
        returns_a: Returns для актива A
        returns_b: Returns для актива B

    Returns:
        float: Correlation coefficient [-1, 1]
    """
    if len(returns_a) < 2 or len(returns_b) < 2:
      return 0.0

    if len(returns_a) != len(returns_b):
      # Выравниваем длину массивов
      min_len = min(len(returns_a), len(returns_b))
      returns_a = returns_a[-min_len:]
      returns_b = returns_b[-min_len:]

    try:
      # Используем numpy для расчета корреляции
      correlation_matrix = np.corrcoef(returns_a, returns_b)
      correlation = correlation_matrix[0, 1]

      # Проверка на NaN
      if np.isnan(correlation):
        return 0.0

      return float(correlation)

    except Exception as e:
      logger.warning(f"Ошибка расчета корреляции: {e}")
      return 0.0


class CorrelationGroupManager:
  """
  Управление группами коррелирующих активов.

  Группирует символы с корреляцией выше threshold в единые группы.
  """

  def __init__(self, correlation_threshold: float = 0.7):
    """
    Инициализация менеджера групп.

    Args:
        correlation_threshold: Минимальная корреляция для группировки
    """
    self.correlation_threshold = correlation_threshold
    self.groups: Dict[str, CorrelationGroup] = {}  # group_id -> CorrelationGroup
    self.symbol_to_group: Dict[str, str] = {}  # symbol -> group_id

    logger.info(
      f"CorrelationGroupManager инициализирован: "
      f"threshold={correlation_threshold:.2f}"
    )

  def create_groups_from_matrix(
      self,
      symbols: List[str],
      correlation_matrix: Dict[Tuple[str, str], float]
  ):
    """
    Создание групп на основе correlation matrix.

    Использует простую жадную группировку:
    - Если symbol1 и symbol2 коррелируют > threshold → в одну группу

    Args:
        symbols: Список торговых пар
        correlation_matrix: Словарь {(symbol1, symbol2): correlation}
    """
    logger.info(f"Группировка {len(symbols)} символов по корреляции...")

    # Очищаем старые группы
    self.groups.clear()
    self.symbol_to_group.clear()

    # Множество уже обработанных символов
    processed: Set[str] = set()

    group_counter = 0

    for symbol in symbols:
      if symbol in processed:
        continue

      # Создаем новую группу
      group_id = f"group_{group_counter}"
      group_symbols = {symbol}

      # Ищем все символы, коррелирующие с текущим
      for other_symbol in symbols:
        if other_symbol == symbol or other_symbol in processed:
          continue

        # Проверяем корреляцию
        key = (symbol, other_symbol) if symbol < other_symbol else (other_symbol, symbol)
        correlation = correlation_matrix.get(key, 0.0)

        if abs(correlation) >= self.correlation_threshold:
          group_symbols.add(other_symbol)

      # Если в группе больше 1 символа - создаем
      if len(group_symbols) > 1:
        # Рассчитываем среднюю корреляцию внутри группы
        correlations = []
        for s1 in group_symbols:
          for s2 in group_symbols:
            if s1 < s2:
              key = (s1, s2)
              if key in correlation_matrix:
                correlations.append(abs(correlation_matrix[key]))

        avg_correlation = np.mean(correlations) if correlations else 0.0

        group = CorrelationGroup(
          group_id=group_id,
          symbols=sorted(list(group_symbols)),
          avg_correlation=avg_correlation,
          active_positions=0,
          total_exposure_usdt=0.0
        )

        self.groups[group_id] = group

        # Регистрируем все символы в группе
        for s in group_symbols:
          self.symbol_to_group[s] = group_id
          processed.add(s)

        logger.info(
          f"Создана группа {group_id}: {group.symbols} | "
          f"avg_corr={avg_correlation:.3f}"
        )

        group_counter += 1
      else:
        # Символ не коррелирует ни с кем - помечаем как обработанный
        processed.add(symbol)

    logger.info(
      f"Группировка завершена: создано {len(self.groups)} групп, "
      f"покрыто {len(self.symbol_to_group)}/{len(symbols)} символов"
    )
    # Находим символы, которые не вошли ни в одну группу
    uncovered_symbols = [s for s in symbols if s not in self.symbol_to_group]
    if uncovered_symbols:
      logger.info(
        f"⚠ {len(uncovered_symbols)} символов не вошло в группы. "
        f"Символы: {', '.join(sorted(uncovered_symbols))}"
      )
    else:
      logger.info("✓ Все символы покрыты корреляционными группами")


  def get_group_for_symbol(self, symbol: str) -> Optional[CorrelationGroup]:
    """
    Получить группу для символа.

    Args:
        symbol: Торговая пара

    Returns:
        CorrelationGroup или None если символ не в группе
    """
    group_id = self.symbol_to_group.get(symbol)
    if not group_id:
      return None

    return self.groups.get(group_id)

  def update_group_position_opened(self, symbol: str, exposure_usdt: float):
    """
    Обновление при открытии позиции.

    Args:
        symbol: Торговая пара
        exposure_usdt: Размер позиции в USDT
    """
    group = self.get_group_for_symbol(symbol)
    if not group:
      logger.debug(f"{symbol} не принадлежит ни одной корреляционной группе")
      return

    group.active_positions += 1
    group.total_exposure_usdt += exposure_usdt

    logger.info(
      f"Позиция открыта в группе {group.group_id}: "
      f"{symbol} | positions={group.active_positions}, "
      f"exposure={group.total_exposure_usdt:.2f} USDT"
    )

  def update_group_position_closed(self, symbol: str, exposure_usdt: float):
    """
    Обновление при закрытии позиции.

    Args:
        symbol: Торговая пара
        exposure_usdt: Размер позиции в USDT
    """
    group = self.get_group_for_symbol(symbol)
    if not group:
      return

    group.active_positions = max(0, group.active_positions - 1)
    group.total_exposure_usdt = max(0.0, group.total_exposure_usdt - exposure_usdt)

    logger.info(
      f"Позиция закрыта в группе {group.group_id}: "
      f"{symbol} | positions={group.active_positions}, "
      f"exposure={group.total_exposure_usdt:.2f} USDT"
    )

  def get_all_groups(self) -> List[CorrelationGroup]:
    """Получить все группы."""
    return list(self.groups.values())


class CorrelationManager:
  """
  Главный менеджер корреляций.

  Координирует расчет корреляций, группировку и валидацию.
  """

  def __init__(self):
    """Инициализация."""
    self.enabled = settings.CORRELATION_CHECK_ENABLED
    self.max_threshold = settings.CORRELATION_MAX_THRESHOLD
    self.max_positions_per_group = settings.CORRELATION_MAX_POSITIONS_PER_GROUP
    self.lookback_days = settings.CORRELATION_LOOKBACK_DAYS

    self.calculator = CorrelationCalculator()
    self.group_manager = CorrelationGroupManager(
      correlation_threshold=self.max_threshold
    )

    # Кеш исторических данных
    self.price_cache: Dict[str, List[float]] = {}
    self.last_update: Optional[datetime] = None

    logger.info(
      f"CorrelationManager инициализирован: "
      f"enabled={self.enabled}, "
      f"threshold={self.max_threshold:.2f}, "
      f"max_per_group={self.max_positions_per_group}, "
      f"lookback={self.lookback_days}d"
    )

  async def initialize(self, symbols: List[str]):
    """
    Инициализация корреляций для списка символов.

    Args:
        symbols: Список торговых пар
    """
    if not self.enabled:
      logger.info("Correlation Manager отключен в конфигурации")
      return

    logger.info(f"Инициализация корреляций для {len(symbols)} символов...")

    try:
      # Шаг 1: Загрузить исторические данные
      await self._load_historical_data(symbols)

      # Шаг 2: Рассчитать correlation matrix
      correlation_matrix = self._calculate_correlation_matrix(symbols)

      # Шаг 3: Создать группы
      self.group_manager.create_groups_from_matrix(symbols, correlation_matrix)

      self.last_update = datetime.now()

      logger.info(
        f"✓ Инициализация завершена: "
        f"групп={len(self.group_manager.groups)}, "
        f"покрыто символов={len(self.group_manager.symbol_to_group)}"
      )

    except Exception as e:
      logger.error(f"Ошибка инициализации CorrelationManager: {e}", exc_info=True)

  async def _load_historical_data(self, symbols: List[str]):
    """
    Загрузка исторических данных для расчета корреляций.

    Args:
        symbols: Список торговых пар
    """
    logger.info(f"Загрузка исторических данных ({self.lookback_days} дней)...")

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=self.lookback_days)).timestamp() * 1000)

    for symbol in symbols:
      try:
        # Запрашиваем дневные свечи
        klines = await rest_client.get_kline(
          symbol=symbol,
          interval="D",  # Дневные свечи
          limit=self.lookback_days,
          start=start_time,
          end=end_time
        )

        if not klines:
          logger.warning(f"{symbol} | Нет исторических данных")
          continue

        # Извлекаем close prices
        close_prices = [float(k[4]) for k in klines]  # k[4] = close

        self.price_cache[symbol] = close_prices

        logger.debug(
          f"{symbol} | Загружено {len(close_prices)} свечей"
        )

      except Exception as e:
        logger.error(f"{symbol} | Ошибка загрузки данных: {e}")

    logger.info(
      f"✓ Загружено данных для {len(self.price_cache)}/{len(symbols)} символов"
    )

  def _calculate_correlation_matrix(
      self,
      symbols: List[str]
  ) -> Dict[Tuple[str, str], float]:
    """
    Расчет correlation matrix для всех пар символов.

    Args:
        symbols: Список торговых пар

    Returns:
        Dict: {(symbol1, symbol2): correlation}
    """
    logger.info("Расчет correlation matrix...")

    correlation_matrix = {}

    # Рассчитываем returns для всех символов
    returns_cache = {}
    for symbol in symbols:
      if symbol not in self.price_cache:
        continue

      prices = np.array(self.price_cache[symbol])
      returns = self.calculator.calculate_returns(prices)
      returns_cache[symbol] = returns

    # Рассчитываем корреляцию для каждой пары
    calculated_pairs = 0

    for i, symbol1 in enumerate(symbols):
      if symbol1 not in returns_cache:
        continue

      for symbol2 in symbols[i + 1:]:
        if symbol2 not in returns_cache:
          continue

        returns1 = returns_cache[symbol1]
        returns2 = returns_cache[symbol2]

        correlation = self.calculator.calculate_correlation(
          returns1, returns2
        )

        # Сохраняем в обе стороны для удобства
        key = (symbol1, symbol2)
        correlation_matrix[key] = correlation

        calculated_pairs += 1

        if abs(correlation) >= self.max_threshold:
          logger.debug(
            f"Высокая корреляция: {symbol1} <-> {symbol2} = {correlation:.3f}"
          )

    logger.info(
      f"✓ Рассчитано {calculated_pairs} пар корреляций"
    )

    return correlation_matrix

  def can_open_position(
      self,
      symbol: str,
      position_size_usdt: float
  ) -> Tuple[bool, Optional[str]]:
    """
    Проверка возможности открытия позиции с учетом корреляций.

    Args:
        symbol: Торговая пара
        position_size_usdt: Размер позиции в USDT

    Returns:
        Tuple[bool, Optional[str]]: (можно_открыть, причина_отказа)
    """
    if not self.enabled:
      return True, None

    # Получаем группу для символа
    group = self.group_manager.get_group_for_symbol(symbol)

    if not group:
      # Символ не коррелирует ни с кем - можно открывать
      logger.debug(f"{symbol} | Нет корреляционных ограничений")
      return True, None

    # Проверяем лимит позиций в группе
    if group.active_positions >= self.max_positions_per_group:
      reason = (
        f"Достигнут лимит позиций в группе {group.group_id} "
        f"({group.active_positions}/{self.max_positions_per_group}). "
        f"Коррелирующие активы: {', '.join(group.symbols)}"
      )

      logger.warning(
        f"{symbol} | Позиция отклонена: {reason}"
      )

      return False, reason

    # Можно открывать
    logger.info(
      f"{symbol} | ✓ Проверка корреляции пройдена: "
      f"группа={group.group_id}, "
      f"позиций={group.active_positions}/{self.max_positions_per_group}"
    )

    return True, None

  def notify_position_opened(self, symbol: str, exposure_usdt: float):
    """
    Уведомление об открытии позиции.

    Args:
        symbol: Торговая пара
        exposure_usdt: Размер позиции в USDT
    """
    if not self.enabled:
      return

    self.group_manager.update_group_position_opened(symbol, exposure_usdt)

  def notify_position_closed(self, symbol: str, exposure_usdt: float):
    """
    Уведомление о закрытии позиции.

    Args:
        symbol: Торговая пара
        exposure_usdt: Размер позиции в USDT
    """
    if not self.enabled:
      return

    self.group_manager.update_group_position_closed(symbol, exposure_usdt)

  async def update_correlations(self, symbols: List[str]):
    """
    Обновление корреляций (вызывать периодически, раз в день).

    Args:
        symbols: Список торговых пар
    """
    if not self.enabled:
      return

    logger.info("Обновление корреляций...")

    try:
      # Перезагружаем исторические данные
      await self._load_historical_data(symbols)

      # Пересчитываем correlation matrix
      correlation_matrix = self._calculate_correlation_matrix(symbols)

      # Сохраняем старые группы для сравнения
      old_groups = {
        group_id: set(group.symbols)
        for group_id, group in self.group_manager.groups.items()
      }

      # Перегруппировываем
      self.group_manager.create_groups_from_matrix(symbols, correlation_matrix)

      # Проверяем изменения
      new_groups = {
        group_id: set(group.symbols)
        for group_id, group in self.group_manager.groups.items()
      }

      if old_groups != new_groups:
        logger.warning(
          "⚠️ Структура корреляционных групп изменилась! "
          "Рекомендуется пересмотреть открытые позиции."
        )

      self.last_update = datetime.now()

      logger.info("✓ Корреляции обновлены")

    except Exception as e:
      logger.error(f"Ошибка обновления корреляций: {e}", exc_info=True)

  def get_statistics(self) -> Dict:
    """Получить статистику."""
    groups = self.group_manager.get_all_groups()

    total_active_positions = sum(g.active_positions for g in groups)
    total_exposure = sum(g.total_exposure_usdt for g in groups)

    groups_with_positions = [g for g in groups if g.active_positions > 0]

    return {
      "enabled": self.enabled,
      "total_groups": len(groups),
      "groups_with_positions": len(groups_with_positions),
      "total_active_positions": total_active_positions,
      "total_exposure_usdt": total_exposure,
      "last_update": self.last_update.isoformat() if self.last_update else None,
      "max_positions_per_group": self.max_positions_per_group,
      "correlation_threshold": self.max_threshold
    }

  def get_group_details(self) -> List[Dict]:
    """Получить детали всех групп."""
    groups = self.group_manager.get_all_groups()

    return [
      {
        "group_id": g.group_id,
        "symbols": g.symbols,
        "avg_correlation": g.avg_correlation,
        "active_positions": g.active_positions,
        "total_exposure_usdt": g.total_exposure_usdt
      }
      for g in groups
    ]


# Глобальный экземпляр
correlation_manager = CorrelationManager()