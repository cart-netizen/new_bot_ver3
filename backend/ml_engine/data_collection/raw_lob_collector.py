"""
Raw LOB Data Collector - сбор сырых данных стакана для TLOB модели.

Собирает полную структуру стакана (Level 2 данные) для обучения
Transformer-based LOB моделей.

Формат данных:
- Временной ряд снимков стакана
- Каждый снимок: [levels × 4] (bid_price, bid_vol, ask_price, ask_vol)
- Хранение в Parquet для эффективного сжатия

Файл: backend/ml_engine/data_collection/raw_lob_collector.py
"""

import os
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import deque
import numpy as np
import pandas as pd

from backend.core.logger import get_logger
from backend.models.orderbook import OrderBookSnapshot

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RawLOBConfig:
    """Конфигурация сборщика сырых LOB данных."""

    # Количество уровней стакана для сохранения
    num_levels: int = 20  # Уровни с обеих сторон

    # Максимальное количество снимков в памяти (per symbol)
    max_snapshots_in_memory: int = 1000

    # Частота сохранения на диск (секунды)
    save_interval_seconds: int = 300  # Каждые 5 минут

    # Путь для хранения данных
    storage_path: str = "data/raw_lob"

    # Формат хранения
    storage_format: str = "parquet"  # parquet или numpy

    # Сжатие для parquet
    compression: str = "snappy"  # snappy, gzip, lz4

    # Минимальный интервал между снимками (мс)
    min_snapshot_interval_ms: int = 100  # 100ms = 10 снимков/сек

    # Включить сбор
    enabled: bool = True

    # Максимальный размер файла (MB)
    max_file_size_mb: int = 50


# ============================================================================
# RAW LOB SNAPSHOT
# ============================================================================

@dataclass
class RawLOBSnapshot:
    """
    Сырой снимок стакана для TLOB модели.

    Формат данных оптимизирован для Transformer:
    - bids: (num_levels, 2) - [price, volume] для каждого уровня
    - asks: (num_levels, 2) - [price, volume] для каждого уровня
    """

    symbol: str
    timestamp: int  # Unix timestamp in milliseconds

    # Сырые данные стакана (numpy arrays)
    bid_prices: np.ndarray  # shape: (num_levels,)
    bid_volumes: np.ndarray  # shape: (num_levels,)
    ask_prices: np.ndarray  # shape: (num_levels,)
    ask_volumes: np.ndarray  # shape: (num_levels,)

    # Метаданные
    mid_price: float
    spread: float
    update_id: Optional[int] = None

    def to_tensor(self) -> np.ndarray:
        """
        Преобразует в тензор для TLOB модели.

        Returns:
            np.ndarray: shape (num_levels, 4) - [bid_price, bid_vol, ask_price, ask_vol]
        """
        return np.column_stack([
            self.bid_prices,
            self.bid_volumes,
            self.ask_prices,
            self.ask_volumes
        ]).astype(np.float32)

    def to_normalized_tensor(self, price_scale: float = 1.0) -> np.ndarray:
        """
        Нормализованный тензор для ML модели.

        Нормализация:
        - Цены: относительно mid_price (в процентах)
        - Объемы: log1p трансформация

        Args:
            price_scale: Масштаб для цен (обычно mid_price)

        Returns:
            np.ndarray: shape (num_levels, 4) - нормализованные данные
        """
        if price_scale <= 0:
            price_scale = self.mid_price if self.mid_price > 0 else 1.0

        # Нормализация цен относительно mid_price
        norm_bid_prices = (self.bid_prices - self.mid_price) / price_scale * 100
        norm_ask_prices = (self.ask_prices - self.mid_price) / price_scale * 100

        # Log-нормализация объемов
        norm_bid_volumes = np.log1p(self.bid_volumes)
        norm_ask_volumes = np.log1p(self.ask_volumes)

        return np.column_stack([
            norm_bid_prices,
            norm_bid_volumes,
            norm_ask_prices,
            norm_ask_volumes
        ]).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сериализации."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'bid_prices': self.bid_prices.tolist(),
            'bid_volumes': self.bid_volumes.tolist(),
            'ask_prices': self.ask_prices.tolist(),
            'ask_volumes': self.ask_volumes.tolist(),
            'mid_price': self.mid_price,
            'spread': self.spread,
            'update_id': self.update_id
        }

    @classmethod
    def from_orderbook_snapshot(
        cls,
        snapshot: OrderBookSnapshot,
        num_levels: int = 20
    ) -> 'RawLOBSnapshot':
        """
        Создает RawLOBSnapshot из OrderBookSnapshot.

        Args:
            snapshot: Стандартный снимок стакана
            num_levels: Количество уровней для извлечения

        Returns:
            RawLOBSnapshot: Сырой снимок для TLOB
        """
        # Извлекаем данные из bids
        bid_prices = np.zeros(num_levels, dtype=np.float32)
        bid_volumes = np.zeros(num_levels, dtype=np.float32)

        for i, (price, volume) in enumerate(snapshot.bids[:num_levels]):
            bid_prices[i] = price
            bid_volumes[i] = volume

        # Извлекаем данные из asks
        ask_prices = np.zeros(num_levels, dtype=np.float32)
        ask_volumes = np.zeros(num_levels, dtype=np.float32)

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
            spread=snapshot.spread or 0.0,
            update_id=snapshot.update_id
        )


# ============================================================================
# RAW LOB SEQUENCE
# ============================================================================

@dataclass
class RawLOBSequence:
    """
    Последовательность снимков LOB для обучения TLOB.

    Формат для модели:
    - shape: (sequence_length, num_levels, 4)
    - 4 канала: bid_price, bid_volume, ask_price, ask_volume
    """

    symbol: str
    snapshots: List[RawLOBSnapshot]

    @property
    def sequence_length(self) -> int:
        return len(self.snapshots)

    @property
    def num_levels(self) -> int:
        if self.snapshots:
            return len(self.snapshots[0].bid_prices)
        return 0

    def to_tensor(self) -> np.ndarray:
        """
        Преобразует в 3D тензор для TLOB модели.

        Returns:
            np.ndarray: shape (sequence_length, num_levels, 4)
        """
        if not self.snapshots:
            return np.array([])

        tensors = [snap.to_tensor() for snap in self.snapshots]
        return np.stack(tensors, axis=0)

    def to_normalized_tensor(self) -> np.ndarray:
        """
        Нормализованный 3D тензор.

        Нормализация:
        - Цены: относительно mid_price первого снимка
        - Объемы: log1p

        Returns:
            np.ndarray: shape (sequence_length, num_levels, 4)
        """
        if not self.snapshots:
            return np.array([])

        # Используем mid_price первого снимка как масштаб
        price_scale = self.snapshots[0].mid_price

        tensors = [snap.to_normalized_tensor(price_scale) for snap in self.snapshots]
        return np.stack(tensors, axis=0)

    @property
    def timestamps(self) -> List[int]:
        return [snap.timestamp for snap in self.snapshots]

    @property
    def mid_prices(self) -> np.ndarray:
        return np.array([snap.mid_price for snap in self.snapshots])


# ============================================================================
# RAW LOB DATA COLLECTOR
# ============================================================================

class RawLOBCollector:
    """
    Сборщик сырых данных LOB для обучения TLOB модели.

    Основные функции:
    1. Сбор полных снимков стакана в реальном времени
    2. Буферизация в памяти с автоматической записью на диск
    3. Эффективное хранение в Parquet формате
    4. Создание последовательностей для обучения

    Использование:
    ```python
    collector = RawLOBCollector(config)
    await collector.initialize()

    # В цикле обработки данных
    await collector.collect(orderbook_snapshot)

    # Получение последовательности для обучения
    sequence = collector.get_sequence(symbol, length=60)
    tensor = sequence.to_normalized_tensor()  # shape: (60, 20, 4)
    ```
    """

    def __init__(self, config: Optional[RawLOBConfig] = None):
        """
        Args:
            config: Конфигурация сборщика
        """
        self.config = config or RawLOBConfig()

        # Буферы для каждого символа
        self._buffers: Dict[str, deque] = {}

        # Последние временные метки (для throttling)
        self._last_timestamps: Dict[str, int] = {}

        # Статистика
        self._stats = {
            'total_collected': 0,
            'total_saved': 0,
            'files_written': 0
        }

        # Путь хранения
        self.storage_path = Path(self.config.storage_path)

        # Последнее время сохранения
        self._last_save_time: Dict[str, datetime] = {}

        # Lock для потокобезопасности
        self._lock = asyncio.Lock()

        logger.info(
            f"RawLOBCollector initialized: "
            f"levels={self.config.num_levels}, "
            f"storage={self.storage_path}, "
            f"enabled={self.config.enabled}"
        )

    async def initialize(self):
        """Инициализация хранилища."""
        if not self.config.enabled:
            logger.info("RawLOBCollector disabled, skipping initialization")
            return

        try:
            # Создаем директории
            self.storage_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"RawLOBCollector storage initialized: {self.storage_path}")

        except Exception as e:
            logger.error(f"Failed to initialize RawLOBCollector: {e}")
            raise

    async def collect(self, snapshot: OrderBookSnapshot) -> bool:
        """
        Собирает снимок стакана.

        Args:
            snapshot: Снимок стакана для сбора

        Returns:
            bool: True если снимок был сохранен
        """
        if not self.config.enabled:
            return False

        symbol = snapshot.symbol

        # Throttling: проверяем минимальный интервал
        if not self._should_collect(symbol, snapshot.timestamp):
            return False

        async with self._lock:
            # Инициализируем буфер для символа
            if symbol not in self._buffers:
                self._buffers[symbol] = deque(
                    maxlen=self.config.max_snapshots_in_memory
                )
                self._last_save_time[symbol] = datetime.now()

            # Создаем сырой снимок
            raw_snapshot = RawLOBSnapshot.from_orderbook_snapshot(
                snapshot,
                num_levels=self.config.num_levels
            )

            # Добавляем в буфер
            self._buffers[symbol].append(raw_snapshot)
            self._last_timestamps[symbol] = snapshot.timestamp
            self._stats['total_collected'] += 1

            # Проверяем необходимость сохранения
            await self._check_and_save(symbol)

        return True

    def _should_collect(self, symbol: str, timestamp: int) -> bool:
        """Проверяет, нужно ли собирать снимок (throttling)."""
        if symbol not in self._last_timestamps:
            return True

        elapsed = timestamp - self._last_timestamps[symbol]
        return elapsed >= self.config.min_snapshot_interval_ms

    async def _check_and_save(self, symbol: str):
        """Проверяет и сохраняет буфер на диск при необходимости."""
        if symbol not in self._last_save_time:
            return

        elapsed = (datetime.now() - self._last_save_time[symbol]).total_seconds()

        if elapsed >= self.config.save_interval_seconds:
            await self._save_buffer(symbol)
            self._last_save_time[symbol] = datetime.now()

    async def _save_buffer(self, symbol: str):
        """Сохраняет буфер на диск."""
        if symbol not in self._buffers or not self._buffers[symbol]:
            return

        try:
            buffer = list(self._buffers[symbol])

            if not buffer:
                return

            # Создаем DataFrame
            df = self._buffer_to_dataframe(buffer)

            # Генерируем имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timestamp}.parquet"

            # Создаем директорию для символа
            symbol_dir = self.storage_path / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)

            filepath = symbol_dir / filename

            # Сохраняем в Parquet
            df.to_parquet(
                filepath,
                compression=self.config.compression,
                index=False
            )

            self._stats['total_saved'] += len(buffer)
            self._stats['files_written'] += 1

            logger.info(
                f"RawLOBCollector saved {len(buffer)} snapshots "
                f"for {symbol} to {filepath}"
            )

        except Exception as e:
            logger.error(f"Failed to save RawLOB buffer for {symbol}: {e}")

    def _buffer_to_dataframe(
        self,
        buffer: List[RawLOBSnapshot]
    ) -> pd.DataFrame:
        """Преобразует буфер в DataFrame для Parquet."""
        records = []

        for snap in buffer:
            record = {
                'symbol': snap.symbol,
                'timestamp': snap.timestamp,
                'mid_price': snap.mid_price,
                'spread': snap.spread,
                'update_id': snap.update_id
            }

            # Добавляем данные по уровням
            for i in range(len(snap.bid_prices)):
                record[f'bid_price_{i}'] = snap.bid_prices[i]
                record[f'bid_volume_{i}'] = snap.bid_volumes[i]
                record[f'ask_price_{i}'] = snap.ask_prices[i]
                record[f'ask_volume_{i}'] = snap.ask_volumes[i]

            records.append(record)

        return pd.DataFrame(records)

    def get_sequence(
        self,
        symbol: str,
        length: int = 60,
        end_timestamp: Optional[int] = None
    ) -> Optional[RawLOBSequence]:
        """
        Получает последовательность снимков для обучения.

        Args:
            symbol: Торговая пара
            length: Длина последовательности
            end_timestamp: Конечная временная метка (None = последний)

        Returns:
            RawLOBSequence или None если недостаточно данных
        """
        if symbol not in self._buffers:
            return None

        buffer = list(self._buffers[symbol])

        if len(buffer) < length:
            logger.debug(
                f"Insufficient data for {symbol}: "
                f"{len(buffer)} < {length}"
            )
            return None

        # Берем последние length снимков
        if end_timestamp is None:
            snapshots = buffer[-length:]
        else:
            # Находим индекс end_timestamp
            idx = None
            for i, snap in enumerate(buffer):
                if snap.timestamp >= end_timestamp:
                    idx = i
                    break

            if idx is None or idx < length:
                return None

            snapshots = buffer[idx - length:idx]

        return RawLOBSequence(symbol=symbol, snapshots=snapshots)

    def get_latest_snapshot(self, symbol: str) -> Optional[RawLOBSnapshot]:
        """Получает последний снимок для символа."""
        if symbol not in self._buffers or not self._buffers[symbol]:
            return None

        return self._buffers[symbol][-1]

    async def flush_all(self):
        """Сохраняет все буферы на диск."""
        async with self._lock:
            for symbol in list(self._buffers.keys()):
                await self._save_buffer(symbol)

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику сборщика."""
        buffer_sizes = {
            symbol: len(buffer)
            for symbol, buffer in self._buffers.items()
        }

        return {
            **self._stats,
            'buffer_sizes': buffer_sizes,
            'total_in_memory': sum(buffer_sizes.values()),
            'symbols': list(self._buffers.keys())
        }

    async def load_historical(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[RawLOBSnapshot]:
        """
        Загружает исторические данные из Parquet файлов.

        Args:
            symbol: Торговая пара
            start_time: Начало периода
            end_time: Конец периода
            limit: Максимальное количество снимков

        Returns:
            List[RawLOBSnapshot]: Загруженные снимки
        """
        symbol_dir = self.storage_path / symbol

        if not symbol_dir.exists():
            logger.warning(f"No historical data for {symbol}")
            return []

        try:
            # Находим все Parquet файлы
            files = sorted(symbol_dir.glob("*.parquet"))

            if not files:
                return []

            snapshots = []

            for file_path in files:
                df = pd.read_parquet(file_path)

                # Фильтруем по времени
                if start_time:
                    start_ms = int(start_time.timestamp() * 1000)
                    df = df[df['timestamp'] >= start_ms]

                if end_time:
                    end_ms = int(end_time.timestamp() * 1000)
                    df = df[df['timestamp'] <= end_ms]

                # Преобразуем в RawLOBSnapshot
                for _, row in df.iterrows():
                    snap = self._row_to_snapshot(row, symbol)
                    snapshots.append(snap)

                    if limit and len(snapshots) >= limit:
                        break

                if limit and len(snapshots) >= limit:
                    break

            logger.info(f"Loaded {len(snapshots)} historical snapshots for {symbol}")
            return snapshots

        except Exception as e:
            logger.error(f"Failed to load historical data for {symbol}: {e}")
            return []

    def _row_to_snapshot(
        self,
        row: pd.Series,
        symbol: str
    ) -> RawLOBSnapshot:
        """Преобразует строку DataFrame в RawLOBSnapshot."""
        num_levels = self.config.num_levels

        bid_prices = np.array([
            row.get(f'bid_price_{i}', 0.0)
            for i in range(num_levels)
        ], dtype=np.float32)

        bid_volumes = np.array([
            row.get(f'bid_volume_{i}', 0.0)
            for i in range(num_levels)
        ], dtype=np.float32)

        ask_prices = np.array([
            row.get(f'ask_price_{i}', 0.0)
            for i in range(num_levels)
        ], dtype=np.float32)

        ask_volumes = np.array([
            row.get(f'ask_volume_{i}', 0.0)
            for i in range(num_levels)
        ], dtype=np.float32)

        return RawLOBSnapshot(
            symbol=symbol,
            timestamp=int(row['timestamp']),
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            ask_prices=ask_prices,
            ask_volumes=ask_volumes,
            mid_price=float(row.get('mid_price', 0.0)),
            spread=float(row.get('spread', 0.0)),
            update_id=row.get('update_id')
        )

    async def cleanup_old_files(self, max_age_days: int = 30):
        """
        Удаляет старые файлы данных.

        Args:
            max_age_days: Максимальный возраст файлов в днях
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        deleted_count = 0

        try:
            for symbol_dir in self.storage_path.iterdir():
                if not symbol_dir.is_dir():
                    continue

                for file_path in symbol_dir.glob("*.parquet"):
                    # Извлекаем дату из имени файла
                    try:
                        file_date_str = file_path.stem.split('_')[1]
                        file_date = datetime.strptime(file_date_str, "%Y%m%d")

                        if file_date < cutoff:
                            file_path.unlink()
                            deleted_count += 1

                    except (IndexError, ValueError):
                        continue

            if deleted_count > 0:
                logger.info(
                    f"RawLOBCollector cleaned up {deleted_count} old files "
                    f"(older than {max_age_days} days)"
                )

        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_raw_lob_collector(
    num_levels: int = 20,
    storage_path: str = "data/raw_lob",
    enabled: bool = True
) -> RawLOBCollector:
    """
    Создает сконфигурированный RawLOBCollector.

    Args:
        num_levels: Количество уровней стакана
        storage_path: Путь для хранения
        enabled: Включен ли сбор

    Returns:
        RawLOBCollector: Настроенный сборщик
    """
    config = RawLOBConfig(
        num_levels=num_levels,
        storage_path=storage_path,
        enabled=enabled
    )

    return RawLOBCollector(config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        print("=" * 80)
        print("RAW LOB COLLECTOR TEST")
        print("=" * 80)

        # Создаем сборщик
        collector = RawLOBCollector(RawLOBConfig(
            num_levels=10,
            storage_path="data/test_raw_lob",
            save_interval_seconds=60
        ))

        await collector.initialize()

        # Симулируем снимки стакана
        for i in range(100):
            # Создаем тестовый снимок
            bids = [(100.0 - j * 0.1, 10.0 + j) for j in range(20)]
            asks = [(100.0 + j * 0.1, 10.0 + j) for j in range(20)]

            snapshot = OrderBookSnapshot(
                symbol="BTCUSDT",
                bids=bids,
                asks=asks,
                timestamp=1700000000000 + i * 100,
                update_id=i
            )

            await collector.collect(snapshot)

        # Получаем последовательность
        sequence = collector.get_sequence("BTCUSDT", length=60)

        if sequence:
            tensor = sequence.to_normalized_tensor()
            print(f"\n📊 Sequence tensor shape: {tensor.shape}")
            print(f"   Expected: (60, 10, 4)")

        # Статистика
        stats = collector.get_stats()
        print(f"\n📈 Stats: {stats}")

        # Сохраняем все
        await collector.flush_all()

        print("\n✅ Test completed!")

    asyncio.run(main())
