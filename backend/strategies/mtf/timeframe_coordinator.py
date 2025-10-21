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

    async def initialize_symbol(self, symbol: str) -> bool:
        """
        Инициализировать все таймфреймы для символа.

        Args:
            symbol: Торговая пара

        Returns:
            True если успешно
        """
        if symbol not in self.candle_managers:
            self.candle_managers[symbol] = {}
            self.last_update[symbol] = {}
            self.initialized[symbol] = {}
        
        success_count = 0
        
        for timeframe in self.config.active_timeframes:
            try:
                # Создаем CandleManager
                candle_manager = CandleManager(
                    symbol=symbol,
                    interval=timeframe.value
                )
                
                # Загружаем исторические данные
                candles_count = self.config.candles_per_timeframe.get(timeframe, 200)
                
                success = await self._load_historical_candles(
                    candle_manager,
                    symbol,
                    timeframe,
                    candles_count
                )
                
                if success:
                    self.candle_managers[symbol][timeframe] = candle_manager
                    self.initialized[symbol][timeframe] = True
                    self.last_update[symbol][timeframe] = int(datetime.now().timestamp())
                    success_count += 1
                    
                    logger.info(
                        f"✅ Инициализирован {symbol} {timeframe.value}: "
                        f"{len(candle_manager.get_candles())} свечей"
                    )
                else:
                    logger.error(f"Ошибка загрузки {symbol} {timeframe.value}")
                    self.initialized[symbol][timeframe] = False
            
            except Exception as e:
                logger.error(
                    f"Ошибка инициализации {symbol} {timeframe.value}: {e}",
                    exc_info=True
                )
                self.initialized[symbol][timeframe] = False
        
        return success_count == len(self.config.active_timeframes)

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
                    
                    # Строим из низшего TF
                    success = self._aggregate_from_lower_timeframe(
                        symbol, timeframe
                    )
                else:
                    # Обновляем напрямую из API
                    success = await self._update_from_api(
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
        candle_manager: CandleManager,
        symbol: str,
        timeframe: Timeframe,
        count: int
    ) -> bool:
        """
        Загрузить исторические свечи через REST API.
        """
        try:
            # Вычисляем start_time
            interval_seconds = self._get_interval_seconds(timeframe)
            start_time = int(
                (datetime.now() - timedelta(seconds=interval_seconds * count)).timestamp() * 1000
            )
            
            # Загружаем через REST API
            candles_data = await rest_client.get_klines(
                symbol=symbol,
                interval=timeframe.value,
                limit=count,
                start_time=start_time
            )
            
            if not candles_data:
                return False
            
            # Добавляем в CandleManager
            for candle_data in candles_data:
                candle = Candle(
                    timestamp=candle_data['timestamp'],
                    open=candle_data['open'],
                    high=candle_data['high'],
                    low=candle_data['low'],
                    close=candle_data['close'],
                    volume=candle_data['volume']
                )
                candle_manager.add_candle(candle)
            
            return True
        
        except Exception as e:
            logger.error(
                f"Ошибка загрузки исторических данных {symbol} {timeframe.value}: {e}"
            )
            return False

    async def _update_from_api(
        self,
        symbol: str,
        timeframe: Timeframe
    ) -> bool:
        """
        Обновить свечи напрямую из API.
        """
        try:
            manager = self.candle_managers[symbol][timeframe]
            
            # Получаем последнюю свечу
            existing_candles = manager.get_candles()
            
            if not existing_candles:
                return False
            
            last_candle = existing_candles[-1]
            
            # Запрашиваем новые свечи
            candles_data = await rest_client.get_klines(
                symbol=symbol,
                interval=timeframe.value,
                limit=10,  # Последние 10 свечей
                start_time=last_candle.timestamp
            )
            
            if not candles_data:
                return False
            
            # Обновляем/добавляем свечи
            for candle_data in candles_data:
                candle = Candle(
                    timestamp=candle_data['timestamp'],
                    open=candle_data['open'],
                    high=candle_data['high'],
                    low=candle_data['low'],
                    close=candle_data['close'],
                    volume=candle_data['volume']
                )
                
                # add_candle автоматически обновляет существующие
                manager.add_candle(candle)
            
            return True
        
        except Exception as e:
            logger.error(
                f"Ошибка обновления {symbol} {timeframe.value}: {e}"
            )
            return False

    def _aggregate_from_lower_timeframe(
        self,
        symbol: str,
        target_timeframe: Timeframe
    ) -> bool:
        """
        Построить свечи высшего таймфрейма из низшего.
        
        Например: 5m из 1m (агрегация 5 свечей)
        """
        try:
            source_timeframe = self.config.aggregation_mapping.get(target_timeframe)
            
            if not source_timeframe:
                return False
            
            # Получаем source свечи
            source_candles = self.get_candles(symbol, source_timeframe)
            
            if not source_candles:
                return False
            
            # Вычисляем коэффициент агрегации
            aggregation_factor = self._get_aggregation_factor(
                source_timeframe, target_timeframe
            )
            
            if aggregation_factor <= 1:
                return False
            
            # Группируем свечи
            aggregated_candles = self._aggregate_candles(
                source_candles,
                aggregation_factor
            )
            
            # Получаем target manager
            if target_timeframe not in self.candle_managers[symbol]:
                self.candle_managers[symbol][target_timeframe] = CandleManager(
                    symbol=symbol,
                    interval=target_timeframe.value
                )
            
            target_manager = self.candle_managers[symbol][target_timeframe]
            
            # Добавляем агрегированные свечи
            for candle in aggregated_candles:
                target_manager.add_candle(candle)
            
            self.aggregations_performed += 1
            
            logger.debug(
                f"Агрегация {symbol}: {source_timeframe.value} → {target_timeframe.value}, "
                f"свечей: {len(aggregated_candles)}"
            )
            
            return True
        
        except Exception as e:
            logger.error(
                f"Ошибка агрегации {symbol} {target_timeframe.value}: {e}"
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
