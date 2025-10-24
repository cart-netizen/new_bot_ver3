"""
Timeframe Coordinator - управление свечными данными для множественных таймфреймов.

Функциональность:
- Управление множественными CandleManager для разных таймфреймов
- Синхронизация обновлений данных
- Timeframe aggregation (построение высших TF из низших)
- Валидация и consistency checking
- Efficient caching и update scheduling

Путь: backend/strategies/mtf/timeframe_coordinator.py
"""
import traceback
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import numpy as np

from core.logger import get_logger
from strategy.candle_manager import CandleManager, Candle
from exchange.rest_client import rest_client

logger = get_logger(__name__)


class Timeframe(Enum):
    """Поддерживаемые таймфреймы."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    def to_api_format(self) -> str:
        """
        Конвертация в формат REST API (числовой).

        Returns:
            Числовой формат интервала для Bybit API
        """
        mapping = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "1d": "D",
            "1w": "W",
        }
        return mapping.get(self.value, self.value)

@dataclass
class TimeframeConfig:
    """Конфигурация для конкретного таймфрейма."""
    timeframe: Timeframe
    candles_count: int  # Количество свечей для хранения
    update_interval_seconds: int  # Интервал обновления
    enable_aggregation: bool = False  # Строить из низшего TF
    aggregation_source: Optional[Timeframe] = None  # Источник для агрегации


@dataclass
class MultiTimeframeConfig:
    """Конфигурация Multi-Timeframe Coordinator."""
    # Активные таймфреймы
    active_timeframes: List[Timeframe] = field(default_factory=lambda: [
        Timeframe.M1,
        Timeframe.M5,
        Timeframe.M15,
        Timeframe.H1
    ])
    
    # Количество свечей на таймфрейм
    candles_per_timeframe: Dict[Timeframe, int] = field(default_factory=lambda: {
        Timeframe.M1: 200,   # 3.3 часа
        Timeframe.M5: 200,   # 16.7 часов
        Timeframe.M15: 200,  # 50 часов / ~2 дня
        Timeframe.H1: 200,   # 200 часов / ~8 дней
        Timeframe.H4: 200,   # 800 часов / ~33 дня
        Timeframe.D1: 200    # 200 дней
    })
    
    # Интервалы обновления (секунды)
    update_intervals: Dict[Timeframe, int] = field(default_factory=lambda: {
        Timeframe.M1: 5,      # Каждые 5 секунд
        Timeframe.M5: 30,     # Каждые 30 секунд
        Timeframe.M15: 60,    # Каждую минуту
        Timeframe.H1: 300,    # Каждые 5 минут
        Timeframe.H4: 900,    # Каждые 15 минут
        Timeframe.D1: 3600    # Каждый час
    })
    
    # Aggregation settings
    enable_aggregation: bool = True  # Строить высшие TF из низших
    aggregation_mapping: Dict[Timeframe, Timeframe] = field(default_factory=lambda: {
        Timeframe.M5: Timeframe.M1,   # 5m из 1m
        Timeframe.M15: Timeframe.M5,  # 15m из 5m
        Timeframe.H1: Timeframe.M15   # 1h из 15m
    })
    
    # Primary timeframe для определения тренда
    primary_timeframe: Timeframe = Timeframe.H1
    
    # Execution timeframe для точного входа
    execution_timeframe: Timeframe = Timeframe.M1


class TimeframeCoordinator:
    """
    Координатор множественных таймфреймов.
    
    Управляет CandleManager для каждого таймфрейма:
    - Загрузка исторических данных
    - Синхронизированные обновления
    - Агрегация таймфреймов
    - Валидация данных
    """

    def __init__(self, config: MultiTimeframeConfig):
        """
        Инициализация координатора.

        Args:
            config: Конфигурация MTF
        """
        self.config = config
        
        # CandleManagers для каждого символа и таймфрейма
        # symbol -> timeframe -> CandleManager
        self.candle_managers: Dict[str, Dict[Timeframe, CandleManager]] = {}
        
        # Timestamp последнего обновления
        # symbol -> timeframe -> timestamp
        self.last_update: Dict[str, Dict[Timeframe, int]] = {}
        
        # Флаги инициализации
        # symbol -> timeframe -> bool
        self.initialized: Dict[str, Dict[Timeframe, bool]] = {}
        
        # Статистика
        self.total_updates = 0
        self.aggregations_performed = 0
        self.validation_failures = 0
        
        logger.info(
            f"Инициализирован TimeframeCoordinator: "
            f"timeframes={[tf.value for tf in config.active_timeframes]}, "
            f"primary={config.primary_timeframe.value}, "
            f"execution={config.execution_timeframe.value}"
        )

    @staticmethod
    def _timeframe_to_bybit_interval(timeframe: Timeframe) -> str:
        """
        Конвертирует внутренний Timeframe в формат Bybit API.

        Bybit API принимает числовые значения интервала в минутах:
        - "1" = 1 минута
        - "5" = 5 минут
        - "15" = 15 минут
        - "60" = 1 час
        - "240" = 4 часа
        - "D" = 1 день

        Args:
            timeframe: Внутренний Timeframe enum

        Returns:
            Строка интервала для Bybit API

        Raises:
            ValueError: Если таймфрейм не поддерживается
        """
        # Маппинг таймфреймов
        TIMEFRAME_TO_BYBIT: Dict[Timeframe, str] = {
            Timeframe.M1: "1",  # 1 минута
            Timeframe.M5: "5",  # 5 минут
            Timeframe.M15: "15",  # 15 минут
            Timeframe.H1: "60",  # 1 час = 60 минут
            Timeframe.H4: "240",  # 4 часа = 240 минут
            Timeframe.D1: "D",  # 1 день
        }

        if timeframe not in TIMEFRAME_TO_BYBIT:
            raise ValueError(f"Неподдерживаемый таймфрейм: {timeframe}")

        return TIMEFRAME_TO_BYBIT[timeframe]

    async def initialize_symbol(self, symbol: str) -> bool:
        """Инициализация символа для всех таймфреймов."""

        # ✅ ИСПРАВЛЕНИЕ: Инициализируем вложенные словари ДО начала работы
        if symbol not in self.candle_managers:
            self.candle_managers[symbol] = {}

        if symbol not in self.last_update:
            self.last_update[symbol] = {}

        if symbol not in self.initialized:
            self.initialized[symbol] = {}

        # ✅ ДОБАВИТЬ: Маппинг таймфреймов
        TIMEFRAME_TO_API = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "1d": "D",
            "1w": "W",
        }

        try:
            timeframes = self.config.active_timeframes
            initialized_count = 0
            total_timeframes = len(timeframes)

            for tf in timeframes:
                try:
                    # ✅ ИСПРАВЛЕНО: Конвертация формата
                    api_interval = TIMEFRAME_TO_API.get(tf.value, tf.value)

                    logger.info(
                        f"[{symbol}] {tf.value}: Запрос свечей с биржи "
                        f"(API interval={api_interval})..."
                    )

                    candles = await rest_client.get_kline(
                        symbol=symbol,
                        interval=api_interval,  # ✅ Числовой формат!
                        limit=200
                    )

                    logger.info(
                        f"[{symbol}] {tf.value}: Биржа вернула "
                        f"{len(candles) if candles else 0} свечей"
                    )

                    if not candles or len(candles) == 0:
                        logger.error(
                            f"❌ [{symbol}] {tf.value}: Биржа вернула 0 свечей! "
                            f"Проверьте правильность символа или доступность пары."
                        )
                        # ❌ Отмечаем как не инициализированный
                        self.initialized[symbol][tf] = False
                        continue

                    # Создаем CandleManager
                    candle_manager = CandleManager(
                        symbol=symbol,
                        timeframe=tf.value,  # Сохраняем в оригинальном формате
                        max_candles=200
                    )

                    await candle_manager.load_historical_data(candles)

                    self.candle_managers[symbol][tf] = candle_manager

                    # ✅ ИСПРАВЛЕНИЕ: Отмечаем таймфрейм как инициализированный
                    self.initialized[symbol][tf] = True

                    # ✅ ИСПРАВЛЕНИЕ: Инициализируем timestamp последнего обновления
                    self.last_update[symbol][tf] = int(datetime.now().timestamp())

                    initialized_count += 1

                    logger.info(
                        f"✅ [{symbol}] {tf.value}: Инициализирован с {len(candles)} свечами"
                    )

                except Exception as e:
                    logger.error(
                        f"❌ [{symbol}] {tf.value}: Ошибка инициализации - {e}"
                    )
                    logger.debug(traceback.format_exc())
                    # ❌ Отмечаем как не инициализированный
                    self.initialized[symbol][tf] = False

            # Строгая проверка
            if initialized_count < total_timeframes:
                logger.warning(
                    f"⚠️ [{symbol}]: Частичная инициализация "
                    f"({initialized_count}/{total_timeframes}). "
                    f"Удаляем частичные данные."
                )

                # Очистка частично инициализированных данных
                if symbol in self.candle_managers:
                    del self.candle_managers[symbol]

                if symbol in self.last_update:
                    del self.last_update[symbol]

                if symbol in self.initialized:
                    del self.initialized[symbol]

                return False

            logger.info(
                f"✅ [{symbol}]: Все {total_timeframes} TF инициализированы"
            )
            return True

        except Exception as e:
            logger.error(f"❌ [{symbol}]: Критическая ошибка - {e}")
            logger.debug(traceback.format_exc())

            # Очистка при критической ошибке
            if symbol in self.candle_managers:
                del self.candle_managers[symbol]

            if symbol in self.last_update:
                del self.last_update[symbol]

            if symbol in self.initialized:
                del self.initialized[symbol]

            return False

    async def update_all_timeframes(self, symbol: str) -> Dict[Timeframe, bool]:
        """
        Обновить все таймфреймы для символа.

        Args:
            symbol: Торговая пара

        Returns:
            Dict[Timeframe, успешность обновления]
        """
        if symbol not in self.candle_managers:
            logger.warning(f"{symbol} не инициализирован")
            return {}

        if symbol not in self.last_update:
            logger.warning(f"{symbol} не инициализирован в last_update, создаём")
            self.last_update[symbol] = {}

        if symbol not in self.initialized:
            logger.warning(f"{symbol} не инициализирован в initialized, создаём")
            self.initialized[symbol] = {}

        results = {}
        current_time = int(datetime.now().timestamp())
        
        for timeframe in self.config.active_timeframes:
            # Проверяем нужно ли обновление
            if not self._should_update(symbol, timeframe, current_time):
                results[timeframe] = True  # Не требуется обновление
                continue
            
            try:
                # Проверяем использовать ли агрегацию
                if (self.config.enable_aggregation and
                    timeframe in self.config.aggregation_mapping):

                    # Строим из низшего TF (НЕ async метод!)
                    success =await self._aggregate_from_lower_timeframe(
                        symbol, timeframe
                    )
                else:
                    # Обновляем напрямую из API (async метод!)
                    success = await self._update_from_api(  # ✅ С await - это async
                        symbol, timeframe
                    )
                
                if success:
                    self.last_update[symbol][timeframe] = current_time
                    self.total_updates += 1
                
                results[timeframe] = success
            
            except Exception as e:
                logger.error(
                    f"Ошибка обновления {symbol} {timeframe.value}: {e}"
                )
                results[timeframe] = False
        
        return results

    def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: Optional[int] = None
    ) -> List[Candle]:
        """
        Получить свечи для символа и таймфрейма.

        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            count: Количество свечей (None = все)

        Returns:
            Список свечей
        """
        if (symbol not in self.candle_managers or 
            timeframe not in self.candle_managers[symbol]):
            return []
        
        manager = self.candle_managers[symbol][timeframe]
        candles = manager.get_candles()
        
        if count:
            return candles[-count:]
        
        return candles

    def get_all_timeframes_candles(
        self,
        symbol: str
    ) -> Dict[Timeframe, List[Candle]]:
        """
        Получить свечи для всех таймфреймов символа.

        Args:
            symbol: Торговая пара

        Returns:
            Dict[Timeframe, List[Candle]]
        """
        result = {}
        
        for timeframe in self.config.active_timeframes:
            candles = self.get_candles(symbol, timeframe)
            if candles:
                result[timeframe] = candles
        
        return result

    def is_initialized(self, symbol: str, timeframe: Optional[Timeframe] = None) -> bool:
        """
        Проверить инициализирован ли символ/таймфрейм.

        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм (None = все)

        Returns:
            True если инициализирован
        """
        if symbol not in self.initialized:
            return False
        
        if timeframe:
            return self.initialized[symbol].get(timeframe, False)
        
        # Проверяем все таймфреймы
        return all(
            self.initialized[symbol].get(tf, False)
            for tf in self.config.active_timeframes
        )

    def _should_update(
        self,
        symbol: str,
        timeframe: Timeframe,
        current_time: int
    ) -> bool:
        """
        Проверить нужно ли обновление таймфрейма.
        """
        if timeframe not in self.last_update.get(symbol, {}):
            return True
        
        last_update = self.last_update[symbol][timeframe]
        update_interval = self.config.update_intervals.get(timeframe, 60)
        
        return (current_time - last_update) >= update_interval

    async def _load_historical_candles(
        self,
        candle_manager,  # CandleManager
        symbol: str,
        timeframe: Timeframe,
        count: int
    ) -> bool:
        """
        Загрузить исторические свечи через REST API.

        ✅ ИСПРАВЛЕНО:
        - Правильная конвертация таймфрейма в формат Bybit
        - Конвертация данных в формат list для CandleManager

        Args:
            candle_manager: Менеджер свечей для обновления
            symbol: Торговая пара (BTCUSDT, ETHUSDT, etc.)
            timeframe: Таймфрейм (Timeframe.M1, M5, etc.)
            count: Количество свечей для загрузки

        Returns:
            True если загрузка успешна, False иначе
        """
        global logger
        try:
            # Конвертируем timeframe в формат Bybit API
            bybit_interval = self._timeframe_to_bybit_interval(timeframe)

            from exchange.rest_client import rest_client
            from core.logger import get_logger

            logger = get_logger(__name__)

            # Загружаем через REST API с правильным интервалом
            candles_data = await rest_client.get_kline(
                symbol=symbol,
                interval=bybit_interval,  # "1", "5", "15", "60" etc.
                limit=count,
            )

            if not candles_data:
                logger.warning(f"Нет данных для {symbol} {timeframe.value}")
                return False

            logger.debug(
                f"📊 Получено {len(candles_data)} свечей для {symbol} {timeframe.value} "
                f"(API interval: {bybit_interval})"
            )

            # ✅ ИСПРАВЛЕНИЕ: Bybit возвращает список списков - используем напрямую!
            # Формат: [timestamp, open, high, low, close, volume, turnover]
            for kline in candles_data:
                try:
                    # ✅ ПРАВИЛЬНО: Передаем список напрямую в CandleManager
                    # CandleManager.update_candle ожидает List[timestamp, o, h, l, c, v]
                    await candle_manager.update_candle(
                        candle_data=kline,  # ← Передаем список как есть!
                        is_closed=True  # Исторические свечи всегда закрыты
                    )

                except (IndexError, ValueError, TypeError) as e:
                    logger.warning(
                        f"⚠️ Ошибка парсинга свечи {symbol} {timeframe.value}: {e}"
                    )
                    continue

            # Проверяем успешность загрузки
            loaded_candles = candle_manager.get_candles()
            logger.info(
                f"✅ {symbol} {timeframe.value}: Загружено {len(loaded_candles)} свечей "
                f"(запрошено: {count})"
            )

            return len(loaded_candles) > 0

        except Exception as e:
            logger.error(
                f"❌ Ошибка загрузки {symbol} {timeframe.value}: {e}",
                exc_info=True
            )
            return False

    async def _update_from_api(
        self,
        symbol: str,
        timeframe: Timeframe
    ) -> bool:
        """
        Обновить свечи напрямую из API.

        ✅ ИСПРАВЛЕНО:
        - Правильная конвертация таймфрейма
        - Конвертация dict → list для CandleManager
        """
        global logger
        try:
            from exchange.rest_client import rest_client
            from core.logger import get_logger

            logger = get_logger(__name__)

            # Получаем CandleManager
            if symbol not in self.candle_managers:
                logger.warning(f"{symbol} не найден в candle_managers")
                return False

            if timeframe not in self.candle_managers[symbol]:
                logger.warning(f"{symbol} {timeframe.value} не найден")
                return False

            manager = self.candle_managers[symbol][timeframe]

            # Получаем последнюю свечу для определения start_time
            existing_candles = manager.get_candles()
            if not existing_candles:
                logger.warning(f"Нет существующих свечей для {symbol} {timeframe.value}")
                return False

            # Конвертируем timeframe в формат Bybit
            bybit_interval = self._timeframe_to_bybit_interval(timeframe)

            # Загружаем новые свечи
            candles_data = await rest_client.get_kline(
                symbol=symbol,
                interval=bybit_interval,  # ✅ Правильный формат
                limit=10,
            )

            if not candles_data:
                logger.warning(f"API вернул пустой результат для {symbol} {timeframe.value}")
                return False

            # ✅ ИСПРАВЛЕНИЕ: Обновляем свечи правильным форматом
            updated_count = 0

            for kline in candles_data:
                try:
                    # ✅ ПРАВИЛЬНО: Передаем список напрямую
                    # Формат от Bybit: [timestamp, open, high, low, close, volume, ...]
                    await manager.update_candle(
                        candle_data=kline,  # ← Список, не словарь!
                        is_closed=True
                    )
                    updated_count += 1

                except (IndexError, ValueError, TypeError) as e:
                    logger.warning(
                        f"⚠️ Ошибка парсинга обновления {symbol} {timeframe.value}: {e}"
                    )
                    continue

            logger.debug(
                f"🔄 Обновлено {updated_count}/{len(candles_data)} свечей "
                f"для {symbol} {timeframe.value}"
            )

            return updated_count > 0

        except Exception as e:
            logger.error(
                f"❌ Ошибка обновления {symbol} {timeframe.value}: {e}",
                exc_info=True
            )
            return False

    async def _aggregate_from_lower_timeframe(
        self,
        symbol: str,
        target_timeframe: Timeframe
    ) -> bool:
        """
        Построить свечи высшего таймфрейма из низшего.

        ✅ ИСПРАВЛЕНО: Конвертация Candle объектов в list для update_candle
        """
        try:
            source_timeframe = self.config.aggregation_mapping.get(target_timeframe)

            if not source_timeframe:
                logger.warning(
                    f"Нет маппинга агрегации для {target_timeframe.value}"
                )
                return False

            source_candles = self.get_candles(symbol, source_timeframe)

            if not source_candles:
                logger.warning(
                    f"Нет исходных свечей {symbol} {source_timeframe.value} "
                    f"для агрегации в {target_timeframe.value}"
                )
                return False

            aggregation_factor = self._get_aggregation_factor(
                source_timeframe, target_timeframe
            )

            if aggregation_factor <= 1:
                logger.warning(
                    f"Некорректный коэффициент агрегации: {aggregation_factor}"
                )
                return False

            aggregated_candles = self._aggregate_candles(
                source_candles,
                aggregation_factor
            )

            if not aggregated_candles:
                logger.warning(
                    f"Агрегация не дала результатов для {symbol} "
                    f"{source_timeframe.value} → {target_timeframe.value}"
                )
                return False

            # Создаем manager если не существует
            if target_timeframe not in self.candle_managers[symbol]:
                self.candle_managers[symbol][target_timeframe] = CandleManager(
                    symbol=symbol,
                    timeframe=target_timeframe.value,
                    max_candles=self.config.candles_per_timeframe.get(
                        target_timeframe, 200
                    )
                )

            target_manager = self.candle_managers[symbol][target_timeframe]

            # ✅ ИСПРАВЛЕНИЕ: Конвертируем Candle объект в список
            updated_count = 0

            for candle in aggregated_candles:
                try:
                    # ✅ ПРАВИЛЬНО: Конвертируем в формат [timestamp, o, h, l, c, v]
                    candle_data_list = [
                        candle.timestamp,
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume
                    ]

                    await target_manager.update_candle(
                        candle_data=candle_data_list,  # ← Список, не словарь!
                        is_closed=True
                    )
                    updated_count += 1

                except Exception as e:
                    logger.warning(
                        f"⚠️ Ошибка добавления агрегированной свечи: {e}"
                    )
                    continue

            self.aggregations_performed += 1

            logger.debug(
                f"✅ Агрегация {symbol}: {source_timeframe.value} → {target_timeframe.value}, "
                f"свечей: {updated_count}/{len(aggregated_candles)}"
            )

            return updated_count > 0

        except Exception as e:
            logger.error(
                f"❌ Ошибка агрегации {symbol} {target_timeframe.value}: {e}",
                exc_info=True
            )
            return False

    def _aggregate_candles(
        self,
        source_candles: List[Candle],
        factor: int
    ) -> List[Candle]:
        """
        Агрегировать свечи с коэффициентом factor.
        
        Args:
            source_candles: Исходные свечи
            factor: Коэффициент агрегации (например, 5 для 1m→5m)

        Returns:
            Список агрегированных свечей
        """
        aggregated = []
        
        # Группируем по factor свечей
        for i in range(0, len(source_candles), factor):
            group = source_candles[i:i+factor]
            
            if len(group) < factor:
                # Неполная группа - пропускаем
                continue
            
            # Агрегируем
            aggregated_candle = Candle(
                timestamp=group[0].timestamp,
                open=group[0].open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                volume=sum(c.volume for c in group)
            )
            
            aggregated.append(aggregated_candle)
        
        return aggregated

    def _get_aggregation_factor(
        self,
        source: Timeframe,
        target: Timeframe
    ) -> int:
        """
        Вычислить коэффициент агрегации между таймфреймами.
        
        Returns:
            Количество source свечей для одной target свечи
        """
        source_seconds = self._get_interval_seconds(source)
        target_seconds = self._get_interval_seconds(target)
        
        return target_seconds // source_seconds

    def _get_interval_seconds(self, timeframe: Timeframe) -> int:
        """Получить длительность таймфрейма в секундах."""
        mapping = {
            Timeframe.M1: 60,
            Timeframe.M5: 300,
            Timeframe.M15: 900,
            Timeframe.H1: 3600,
            Timeframe.H4: 14400,
            Timeframe.D1: 86400
        }
        return mapping.get(timeframe, 60)

    def validate_data_consistency(self, symbol: str) -> Dict[str, any]:
        """
        Валидировать консистентность данных между таймфреймами.

        Args:
            symbol: Торговая пара

        Returns:
            Dict с результатами валидации
        """
        results = {
            'valid': True,
            'issues': [],
            'timeframes_checked': 0
        }
        
        for timeframe in self.config.active_timeframes:
            candles = self.get_candles(symbol, timeframe)
            
            if not candles:
                results['issues'].append(f"{timeframe.value}: нет данных")
                results['valid'] = False
                continue
            
            results['timeframes_checked'] += 1
            
            # Проверка 1: Нет gaps в timestamp
            for i in range(1, len(candles)):
                expected_gap = self._get_interval_seconds(timeframe) * 1000
                actual_gap = candles[i].timestamp - candles[i-1].timestamp
                
                if abs(actual_gap - expected_gap) > 1000:  # Tolerance 1s
                    results['issues'].append(
                        f"{timeframe.value}: gap обнаружен на индексе {i}"
                    )
                    results['valid'] = False
                    self.validation_failures += 1
                    break
            
            # Проверка 2: OHLC валидность
            for i, candle in enumerate(candles):
                if not (candle.low <= candle.open <= candle.high and
                        candle.low <= candle.close <= candle.high):
                    results['issues'].append(
                        f"{timeframe.value}: невалидная OHLC на индексе {i}"
                    )
                    results['valid'] = False
                    self.validation_failures += 1
                    break
        
        return results

    def get_statistics(self) -> Dict:
        """Получить статистику координатора."""
        total_candles = 0
        symbols_count = len(self.candle_managers)
        
        for symbol_managers in self.candle_managers.values():
            for manager in symbol_managers.values():
                total_candles += len(manager.get_candles())
        
        return {
            'symbols_tracked': symbols_count,
            'total_updates': self.total_updates,
            'aggregations_performed': self.aggregations_performed,
            'validation_failures': self.validation_failures,
            'total_candles_stored': total_candles,
            'timeframes_active': len(self.config.active_timeframes)
        }
