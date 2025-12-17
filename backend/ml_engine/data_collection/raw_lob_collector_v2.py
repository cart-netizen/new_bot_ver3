#!/usr/bin/env python3
"""
Raw LOB Data Collector v2 - Production-grade collector для TLOB Transformer.

Улучшения относительно v1:
1. Адаптивный сбор данных на основе cycle time
2. Агрессивное управление памятью с мониторингом
3. Экстренное сохранение при нехватке памяти
4. Batch-запись для эффективности I/O
5. Дедупликация снимков
6. Graceful degradation при перегрузке

Для обучения TLOB Transformer (50 символов, 11-15 сек/цикл):
- Собираем ~1 снимок на символ за цикл (не 10/сек!)
- Используем deque с жестким maxlen
- Периодическая очистка старых данных

Файл: backend/ml_engine/data_collection/raw_lob_collector_v2.py
"""

import os
import gc
import sys
import asyncio
import weakref
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import deque
from threading import RLock
import numpy as np
import pandas as pd

from backend.core.logger import get_logger
from backend.models.orderbook import OrderBookSnapshot

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RawLOBConfigV2:
    """
    Конфигурация сборщика v2 с учетом реальных условий.

    Расчет для 50 символов:
    - Цикл анализа: 11-15 сек
    - 1 снимок/символ/цикл = 50 снимков/цикл
    - За 5 мин (300 сек) = ~1500 снимков total
    - 1500 / 50 символов = 30 снимков/символ в буфере
    """

    # Количество уровней стакана (с каждой стороны)
    num_levels: int = 20

    # Максимум снимков в памяти НА СИМВОЛ (жесткий лимит)
    max_snapshots_per_symbol: int = 100  # ~15 минут данных при 11-15 сек/цикл

    # Общий лимит памяти для всех буферов (MB)
    max_total_memory_mb: float = 150.0  # ~150 MB для 50 символов

    # Интервал автосохранения (секунды)
    save_interval_seconds: int = 300  # 5 минут

    # Минимум снимков для сохранения файла
    min_snapshots_for_save: int = 10

    # Путь хранения
    storage_path: str = "data/raw_lob"

    # Сжатие parquet
    compression: str = "snappy"

    # Адаптивный сбор: пропускать снимки если cycle_time > threshold
    adaptive_collection: bool = True
    adaptive_skip_threshold_sec: float = 20.0  # Пропускать если цикл > 20 сек

    # Минимальный интервал между снимками одного символа (мс)
    min_snapshot_interval_ms: int = 5000  # 5 сек минимум между снимками

    # Экстренное сохранение при достижении % памяти
    emergency_save_memory_percent: float = 0.85

    # Периодическая очистка старых файлов (дни)
    max_file_age_days: int = 30

    # Включен ли сборщик
    enabled: bool = True

    # Партиционирование по дате
    partition_by_date: bool = True


# ============================================================================
# MEMORY-EFFICIENT SNAPSHOT
# ============================================================================

@dataclass(slots=True)  # slots для экономии памяти
class RawLOBSnapshotV2:
    """
    Оптимизированный по памяти снимок стакана.

    Использует slots для уменьшения overhead.
    Хранит только необходимые данные.
    """

    symbol: str
    timestamp: int  # Unix timestamp ms

    # Numpy arrays (более эффективно чем list)
    bid_prices: np.ndarray   # shape: (num_levels,), dtype=float32
    bid_volumes: np.ndarray  # shape: (num_levels,), dtype=float32
    ask_prices: np.ndarray   # shape: (num_levels,), dtype=float32
    ask_volumes: np.ndarray  # shape: (num_levels,), dtype=float32

    # Метаданные (минимум)
    mid_price: float
    spread: float

    def __sizeof__(self) -> int:
        """Реальный размер объекта в байтах."""
        size = sys.getsizeof(self.symbol)
        size += sys.getsizeof(self.timestamp)
        size += self.bid_prices.nbytes
        size += self.bid_volumes.nbytes
        size += self.ask_prices.nbytes
        size += self.ask_volumes.nbytes
        size += sys.getsizeof(self.mid_price)
        size += sys.getsizeof(self.spread)
        return size

    def to_tensor(self) -> np.ndarray:
        """
        Преобразует в тензор (num_levels, 4).

        Returns:
            np.ndarray: [bid_price, bid_vol, ask_price, ask_vol]
        """
        return np.column_stack([
            self.bid_prices,
            self.bid_volumes,
            self.ask_prices,
            self.ask_volumes
        ])

    def to_normalized_tensor(self, reference_price: Optional[float] = None) -> np.ndarray:
        """
        Нормализованный тензор для ML.

        Нормализация:
        - Цены: относительно mid_price (в basis points)
        - Объемы: log1p трансформация
        """
        ref_price = reference_price or self.mid_price
        if ref_price <= 0:
            ref_price = 1.0

        # Basis points относительно mid_price
        norm_bid_prices = (self.bid_prices - self.mid_price) / ref_price * 10000
        norm_ask_prices = (self.ask_prices - self.mid_price) / ref_price * 10000

        # Log-нормализация объемов
        norm_bid_volumes = np.log1p(self.bid_volumes)
        norm_ask_volumes = np.log1p(self.ask_volumes)

        return np.column_stack([
            norm_bid_prices,
            norm_bid_volumes,
            norm_ask_prices,
            norm_ask_volumes
        ]).astype(np.float32)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flat dict для DataFrame (без вложенных структур)."""
        record = {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'mid_price': self.mid_price,
            'spread': self.spread,
        }

        for i in range(len(self.bid_prices)):
            record[f'bid_price_{i}'] = float(self.bid_prices[i])
            record[f'bid_volume_{i}'] = float(self.bid_volumes[i])
            record[f'ask_price_{i}'] = float(self.ask_prices[i])
            record[f'ask_volume_{i}'] = float(self.ask_volumes[i])

        return record

    @classmethod
    def from_orderbook(
        cls,
        snapshot: OrderBookSnapshot,
        num_levels: int = 20
    ) -> 'RawLOBSnapshotV2':
        """Создает из OrderBookSnapshot."""

        # Pre-allocate arrays
        bid_prices = np.zeros(num_levels, dtype=np.float32)
        bid_volumes = np.zeros(num_levels, dtype=np.float32)
        ask_prices = np.zeros(num_levels, dtype=np.float32)
        ask_volumes = np.zeros(num_levels, dtype=np.float32)

        # Fill from bids (already sorted by price desc)
        for i, (price, volume) in enumerate(snapshot.bids[:num_levels]):
            bid_prices[i] = price
            bid_volumes[i] = volume

        # Fill from asks (already sorted by price asc)
        for i, (price, volume) in enumerate(snapshot.asks[:num_levels]):
            ask_prices[i] = price
            ask_volumes[i] = volume

        return cls(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            ask_prices=ask_prices,
            ask_volumes=ask_volumes,
            mid_price=snapshot.mid_price or 0.0,
            spread=snapshot.spread or 0.0
        )


# ============================================================================
# RAW LOB COLLECTOR V2
# ============================================================================

class RawLOBCollectorV2:
    """
    Production-grade сборщик сырых данных LOB для TLOB Transformer.

    Ключевые особенности:
    1. Жесткие лимиты памяти с мониторингом
    2. Адаптивный сбор на основе cycle time
    3. Дедупликация по timestamp
    4. Экстренное сохранение при нехватке памяти
    5. Thread-safe операции
    6. Graceful degradation

    Использование:
    ```python
    collector = RawLOBCollectorV2(config)
    await collector.initialize()

    # В цикле анализа
    collected = await collector.collect(orderbook_snapshot, cycle_time=12.5)

    # Периодически
    await collector.maybe_save_buffers()

    # При завершении
    await collector.finalize()
    ```
    """

    def __init__(self, config: Optional[RawLOBConfigV2] = None):
        self.config = config or RawLOBConfigV2()

        # Thread-safe lock
        self._lock = RLock()

        # Буферы для каждого символа (deque с maxlen для автоочистки)
        self._buffers: Dict[str, deque] = {}

        # Последние timestamps для дедупликации
        self._last_timestamps: Dict[str, int] = {}

        # Время последнего сохранения (per symbol)
        self._last_save_times: Dict[str, datetime] = {}

        # Статистика
        self._stats = {
            'total_collected': 0,
            'total_saved': 0,
            'total_skipped_duplicate': 0,
            'total_skipped_adaptive': 0,
            'total_skipped_interval': 0,
            'files_written': 0,
            'emergency_saves': 0,
            'memory_warnings': 0,
        }

        # Storage path
        self.storage_path = Path(self.config.storage_path)

        # Estimated memory per snapshot (bytes)
        # 20 levels * 4 floats * 4 bytes = 320 bytes + overhead ~500 bytes
        self._estimated_snapshot_size = 500

        # Tracking symbols
        self._active_symbols: Set[str] = set()

        # Initialized flag
        self._initialized = False

        logger.info(
            f"RawLOBCollectorV2 created: "
            f"levels={self.config.num_levels}, "
            f"max_per_symbol={self.config.max_snapshots_per_symbol}, "
            f"max_memory={self.config.max_total_memory_mb}MB, "
            f"enabled={self.config.enabled}"
        )

    async def initialize(self) -> None:
        """Инициализация хранилища."""
        if not self.config.enabled:
            logger.info("RawLOBCollectorV2 disabled, skipping initialization")
            return

        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            logger.info(f"RawLOBCollectorV2 initialized: {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to initialize RawLOBCollectorV2: {e}")
            raise

    async def collect(
        self,
        snapshot: OrderBookSnapshot,
        cycle_time: Optional[float] = None
    ) -> bool:
        """
        Собирает снимок стакана с адаптивной логикой.

        Args:
            snapshot: OrderBook snapshot для сбора
            cycle_time: Время текущего цикла анализа (секунды)

        Returns:
            bool: True если снимок был сохранен в буфер
        """
        if not self.config.enabled or not self._initialized:
            return False

        symbol = snapshot.symbol
        timestamp = snapshot.timestamp

        # === Адаптивная логика ===
        if self.config.adaptive_collection and cycle_time is not None:
            if cycle_time > self.config.adaptive_skip_threshold_sec:
                self._stats['total_skipped_adaptive'] += 1
                return False

        with self._lock:
            # === Дедупликация по timestamp ===
            if symbol in self._last_timestamps:
                elapsed_ms = timestamp - self._last_timestamps[symbol]

                # Пропускаем дубликаты
                if elapsed_ms <= 0:
                    self._stats['total_skipped_duplicate'] += 1
                    return False

                # Минимальный интервал между снимками
                if elapsed_ms < self.config.min_snapshot_interval_ms:
                    self._stats['total_skipped_interval'] += 1
                    return False

            # === Проверка памяти ===
            if self._should_emergency_save():
                # Экстренное сохранение
                asyncio.create_task(self._emergency_save_all())

            # === Инициализация буфера для символа ===
            if symbol not in self._buffers:
                self._buffers[symbol] = deque(
                    maxlen=self.config.max_snapshots_per_symbol
                )
                self._last_save_times[symbol] = datetime.now()
                self._active_symbols.add(symbol)

            # === Создание и сохранение снимка ===
            try:
                raw_snapshot = RawLOBSnapshotV2.from_orderbook(
                    snapshot,
                    num_levels=self.config.num_levels
                )

                self._buffers[symbol].append(raw_snapshot)
                self._last_timestamps[symbol] = timestamp
                self._stats['total_collected'] += 1

                return True

            except Exception as e:
                logger.warning(f"Failed to create snapshot for {symbol}: {e}")
                return False

    def _should_emergency_save(self) -> bool:
        """Проверяет необходимость экстренного сохранения."""
        current_memory = self._estimate_memory_usage()
        max_memory = self.config.max_total_memory_mb * 1024 * 1024  # bytes
        threshold = max_memory * self.config.emergency_save_memory_percent

        if current_memory > threshold:
            self._stats['memory_warnings'] += 1
            return True
        return False

    def _estimate_memory_usage(self) -> int:
        """Оценивает текущее использование памяти буферами (bytes)."""
        total_snapshots = sum(len(buf) for buf in self._buffers.values())
        return total_snapshots * self._estimated_snapshot_size

    async def maybe_save_buffers(self) -> int:
        """
        Проверяет и сохраняет буферы по расписанию.

        Returns:
            int: Количество сохраненных файлов
        """
        if not self.config.enabled or not self._initialized:
            return 0

        saved_count = 0
        now = datetime.now()

        symbols_to_save = []

        with self._lock:
            for symbol, last_save in list(self._last_save_times.items()):
                elapsed = (now - last_save).total_seconds()

                if elapsed >= self.config.save_interval_seconds:
                    if symbol in self._buffers and len(self._buffers[symbol]) >= self.config.min_snapshots_for_save:
                        symbols_to_save.append(symbol)

        for symbol in symbols_to_save:
            try:
                if await self._save_buffer(symbol):
                    saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save buffer for {symbol}: {e}")

        return saved_count

    async def _save_buffer(self, symbol: str) -> bool:
        """
        Сохраняет буфер символа на диск.

        Args:
            symbol: Символ для сохранения

        Returns:
            bool: True если успешно
        """
        with self._lock:
            if symbol not in self._buffers or not self._buffers[symbol]:
                return False

            # Копируем данные и очищаем буфер
            snapshots = list(self._buffers[symbol])
            self._buffers[symbol].clear()
            self._last_save_times[symbol] = datetime.now()

        if not snapshots:
            return False

        try:
            # Создаем DataFrame
            records = [snap.to_flat_dict() for snap in snapshots]
            df = pd.DataFrame(records)

            # Определяем путь
            if self.config.partition_by_date:
                date_str = datetime.now().strftime("%Y-%m-%d")
                symbol_dir = self.storage_path / symbol / f"date={date_str}"
            else:
                symbol_dir = self.storage_path / symbol

            symbol_dir.mkdir(parents=True, exist_ok=True)

            # Генерируем имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timestamp}.parquet"
            filepath = symbol_dir / filename

            # Сохраняем
            df.to_parquet(
                filepath,
                compression=self.config.compression,
                index=False
            )

            self._stats['total_saved'] += len(snapshots)
            self._stats['files_written'] += 1

            logger.info(
                f"RawLOBCollectorV2: Saved {len(snapshots)} snapshots "
                f"for {symbol} → {filepath.name}"
            )

            # Explicit cleanup
            del records
            del df
            del snapshots

            return True

        except Exception as e:
            logger.error(f"Failed to save RAW LOB for {symbol}: {e}")

            # Возвращаем данные в буфер при ошибке
            with self._lock:
                for snap in snapshots:
                    self._buffers[symbol].appendleft(snap)

            return False

    async def _emergency_save_all(self) -> None:
        """Экстренное сохранение всех буферов."""
        logger.warning("RawLOBCollectorV2: EMERGENCY SAVE triggered (memory threshold)")
        self._stats['emergency_saves'] += 1

        symbols = list(self._buffers.keys())

        for symbol in symbols:
            try:
                await self._save_buffer(symbol)
            except Exception as e:
                logger.error(f"Emergency save failed for {symbol}: {e}")

        # Force garbage collection
        gc.collect()

    async def finalize(self) -> None:
        """
        Финализация: сохранение всех буферов и очистка.

        Вызывать при завершении работы бота.
        """
        if not self._initialized:
            return

        logger.info("RawLOBCollectorV2: Finalizing...")

        # Сохраняем все буферы
        symbols = list(self._buffers.keys())
        saved = 0

        for symbol in symbols:
            try:
                if await self._save_buffer(symbol):
                    saved += 1
            except Exception as e:
                logger.error(f"Finalize save failed for {symbol}: {e}")

        # Очищаем все
        with self._lock:
            self._buffers.clear()
            self._last_timestamps.clear()
            self._last_save_times.clear()
            self._active_symbols.clear()

        logger.info(
            f"RawLOBCollectorV2 finalized: "
            f"saved {saved} files, "
            f"stats={self._stats}"
        )

    def get_sequence(
        self,
        symbol: str,
        length: int = 60
    ) -> Optional[np.ndarray]:
        """
        Получает последовательность снимков для inference.

        Args:
            symbol: Торговая пара
            length: Длина последовательности

        Returns:
            np.ndarray: shape (length, num_levels, 4) или None
        """
        with self._lock:
            if symbol not in self._buffers:
                return None

            buffer = self._buffers[symbol]

            if len(buffer) < length:
                return None

            # Берем последние length снимков
            snapshots = list(buffer)[-length:]

        # Конвертируем в тензор
        tensors = [snap.to_normalized_tensor() for snap in snapshots]
        return np.stack(tensors, axis=0).astype(np.float32)

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает детальную статистику."""
        with self._lock:
            buffer_info = {
                symbol: len(buf)
                for symbol, buf in self._buffers.items()
            }
            total_in_memory = sum(buffer_info.values())

        memory_mb = self._estimate_memory_usage() / (1024 * 1024)

        return {
            **self._stats,
            'active_symbols': len(self._active_symbols),
            'total_in_memory': total_in_memory,
            'estimated_memory_mb': round(memory_mb, 2),
            'buffer_sizes': buffer_info,
            'config': {
                'max_per_symbol': self.config.max_snapshots_per_symbol,
                'max_memory_mb': self.config.max_total_memory_mb,
                'save_interval_sec': self.config.save_interval_seconds,
            }
        }

    def cleanup_old_data(self, symbol: Optional[str] = None) -> int:
        """
        Очищает старые данные из буферов.

        Args:
            symbol: Символ для очистки (None = все)

        Returns:
            int: Количество удаленных снимков
        """
        removed = 0

        with self._lock:
            symbols = [symbol] if symbol else list(self._buffers.keys())

            for sym in symbols:
                if sym in self._buffers:
                    old_len = len(self._buffers[sym])
                    # deque с maxlen автоматически удаляет старые
                    # Но можем принудительно очистить часть
                    if old_len > self.config.max_snapshots_per_symbol // 2:
                        # Оставляем только половину
                        keep = self.config.max_snapshots_per_symbol // 2
                        while len(self._buffers[sym]) > keep:
                            self._buffers[sym].popleft()
                            removed += 1

        if removed > 0:
            logger.debug(f"RawLOBCollectorV2: Cleaned {removed} old snapshots")

        return removed

    async def cleanup_old_files(self, max_age_days: Optional[int] = None) -> int:
        """
        Удаляет старые parquet файлы.

        Args:
            max_age_days: Максимальный возраст файлов

        Returns:
            int: Количество удаленных файлов
        """
        max_age = max_age_days or self.config.max_file_age_days
        cutoff = datetime.now() - timedelta(days=max_age)
        deleted = 0

        try:
            for symbol_dir in self.storage_path.iterdir():
                if not symbol_dir.is_dir():
                    continue

                for item in symbol_dir.rglob("*.parquet"):
                    try:
                        # Извлекаем дату из имени файла
                        file_date_str = item.stem.split('_')[1]
                        file_date = datetime.strptime(file_date_str, "%Y%m%d")

                        if file_date < cutoff:
                            item.unlink()
                            deleted += 1
                    except (IndexError, ValueError):
                        continue

            if deleted > 0:
                logger.info(
                    f"RawLOBCollectorV2: Deleted {deleted} old files "
                    f"(older than {max_age} days)"
                )

        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")

        return deleted


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_raw_lob_collector_v2(
    num_levels: int = 20,
    storage_path: str = "data/raw_lob",
    max_snapshots_per_symbol: int = 100,
    max_total_memory_mb: float = 150.0,
    save_interval_seconds: int = 300,
    enabled: bool = True
) -> RawLOBCollectorV2:
    """
    Создает сконфигурированный RawLOBCollectorV2.

    Args:
        num_levels: Количество уровней стакана
        storage_path: Путь хранения
        max_snapshots_per_symbol: Макс снимков на символ
        max_total_memory_mb: Лимит памяти (MB)
        save_interval_seconds: Интервал сохранения
        enabled: Включен ли сборщик

    Returns:
        RawLOBCollectorV2: Настроенный сборщик
    """
    config = RawLOBConfigV2(
        num_levels=num_levels,
        storage_path=storage_path,
        max_snapshots_per_symbol=max_snapshots_per_symbol,
        max_total_memory_mb=max_total_memory_mb,
        save_interval_seconds=save_interval_seconds,
        enabled=enabled
    )

    return RawLOBCollectorV2(config)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        print("=" * 80)
        print("RAW LOB COLLECTOR V2 TEST")
        print("=" * 80)

        collector = create_raw_lob_collector_v2(
            num_levels=10,
            storage_path="data/test_raw_lob_v2",
            max_snapshots_per_symbol=50,
            save_interval_seconds=10
        )

        await collector.initialize()

        # Симуляция 50 символов
        symbols = [f"TEST{i}USDT" for i in range(50)]

        for cycle in range(5):
            cycle_time = 12.5  # Симуляция 12.5 сек цикла

            for symbol in symbols:
                # Создаем тестовый snapshot
                bids = [(100.0 - j * 0.1, 10.0 + j) for j in range(20)]
                asks = [(100.0 + j * 0.1, 10.0 + j) for j in range(20)]

                snapshot = OrderBookSnapshot(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=1700000000000 + cycle * 12500,
                    update_id=cycle
                )

                await collector.collect(snapshot, cycle_time=cycle_time)

            print(f"Cycle {cycle + 1}: {collector.get_statistics()['total_in_memory']} snapshots in memory")
            await asyncio.sleep(0.1)

        # Сохраняем
        await collector.maybe_save_buffers()

        # Статистика
        print(f"\nFinal stats: {collector.get_statistics()}")

        # Финализация
        await collector.finalize()

        print("\n✅ Test completed!")

    asyncio.run(test())
