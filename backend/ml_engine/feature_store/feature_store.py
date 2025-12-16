"""
Feature Store - централизованное хранилище фич

Функциональность:
- Online feature serving (real-time predictions)
- Offline feature serving (training)
- Feature versioning and metadata
- Feature consistency между training/serving
- Feature caching для performance
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from collections import OrderedDict

from backend.core.logger import get_logger

logger = get_logger(__name__)

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _resolve_path(path: str) -> Path:
    """
    Resolve a path to absolute.
    If path is relative, resolve it relative to PROJECT_ROOT.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


@dataclass
class FeatureMetadata:
    """Метаданные фичи"""
    name: str
    version: str
    description: str
    feature_type: str  # "orderbook", "candle", "indicator", "technical"
    data_type: str  # "float", "int", "categorical"
    created_at: str
    updated_at: str

    # Feature engineering details
    source: str  # Откуда извлекается фича
    dependencies: List[str]  # Зависимости от других фич

    # Statistics
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None

    # Служебная информация
    is_active: bool = True


class FeatureStore:
    """
    Feature Store для управления фичами

    Архитектура:
    - Offline Store: Parquet files для training
    - Online Store: In-memory cache для serving
    - Metadata Store: JSON для feature definitions
    """

    def __init__(
        self,
        storage_path: str = "data/feature_store",
        cache_ttl_seconds: int = 3600
    ):
        """
        Инициализация Feature Store

        Args:
            storage_path: Путь для хранения фич
            cache_ttl_seconds: TTL для online cache (секунды)
        """
        # Resolve to absolute path
        self.storage_path = _resolve_path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FeatureStore: Using storage path: {self.storage_path}")

        # Directories
        self.offline_dir = self.storage_path / "offline"
        self.online_dir = self.storage_path / "online"
        self.metadata_dir = self.storage_path / "metadata"

        for dir_path in [self.offline_dir, self.online_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Online cache
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.online_cache: OrderedDict[str, Tuple[np.ndarray, datetime]] = OrderedDict()
        self.max_cache_size = 1000

        # Feature metadata
        self.feature_metadata: Dict[str, FeatureMetadata] = {}
        self._load_metadata()

        logger.info(
            f"Feature Store initialized: "
            f"path={storage_path}, "
            f"features={len(self.feature_metadata)}"
        )

    def register_feature(
        self,
        metadata: FeatureMetadata
    ) -> bool:
        """
        Зарегистрировать новую фичу

        Args:
            metadata: Метаданные фичи

        Returns:
            True если успешно
        """
        try:
            # Validate
            if not metadata.name:
                logger.error("Feature name is required")
                return False

            # Check if exists
            feature_key = f"{metadata.name}:{metadata.version}"
            if feature_key in self.feature_metadata:
                logger.warning(f"Feature already registered: {feature_key}")
                return False

            # Save metadata
            metadata.created_at = datetime.now().isoformat()
            metadata.updated_at = metadata.created_at

            self.feature_metadata[feature_key] = metadata

            # Persist to disk
            metadata_path = self.metadata_dir / f"{feature_key}.json"
            with open(metadata_path, 'w') as f:
                # Convert dataclass to dict for JSON serialization
                metadata_dict = asdict(metadata)  # type: ignore[arg-type]
                json.dump(metadata_dict, f, indent=2)

            logger.info(f"Registered feature: {feature_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to register feature: {e}")
            return False

    def write_offline_features(
        self,
        feature_group: str,
        features: pd.DataFrame,
        timestamp_column: str = "timestamp"
    ) -> bool:
        """
        Записать фичи в offline store (для training)

        Args:
            feature_group: Группа фич (например, "orderbook_features")
            features: DataFrame с фичами
            timestamp_column: Название колонки с timestamp

        Returns:
            True если успешно
        """
        try:
            # Validate timestamp column
            if timestamp_column not in features.columns:
                logger.error(f"Timestamp column not found: {timestamp_column}")
                return False

            # Create partition by date
            if len(features) > 0:
                # CRITICAL FIX: Check ALL timestamps, not just the first one
                # If any timestamp is 0, we should use current date
                timestamp_values = features[timestamp_column]
                zero_count = int(timestamp_values.eq(0).sum())
                none_count = int(timestamp_values.isna().sum())

                if zero_count > 0 or none_count > 0:
                    logger.error(
                        f"[FeatureStore] Found INVALID timestamps! "
                        f"Zero timestamps: {zero_count}/{len(features)}, "
                        f"None/NaN timestamps: {none_count}/{len(features)}. "
                        f"First 10 timestamps: {timestamp_values.head(10).tolist()}. "
                        f"Using current date as fallback."
                    )
                    date_str = datetime.now().strftime("%Y-%m-%d")
                else:
                    # All timestamps are valid, get first one
                    first_timestamp = timestamp_values.iloc[0]

                    # DEBUG: Детальное логирование timestamp
                    logger.info(
                        f"[FeatureStore] Processing timestamp: {first_timestamp}, "
                        f"type={type(first_timestamp)}, "
                        f"feature_group={feature_group}, "
                        f"rows={len(features)}"
                    )

                    # Process based on timestamp type
                    # CRITICAL: Include numpy types (numpy.int64, numpy.float64, etc.)
                    if isinstance(first_timestamp, (int, float, np.integer, np.floating)):
                        # CRITICAL: Validate timestamp is not zero or invalid (double-check)
                        if first_timestamp == 0 or pd.isna(first_timestamp):
                            logger.warning(
                                f"[FeatureStore] Invalid timestamp detected (0 or NaN): using current date. "
                                f"This may indicate a bug in data collection. "
                                f"First 5 timestamps: {features[timestamp_column].head().tolist()}"
                            )
                            date_str = datetime.now().strftime("%Y-%m-%d")
                        # Check if timestamp is in milliseconds (>10 digits)
                        # Crypto exchanges (Bybit, Binance) use milliseconds
                        elif first_timestamp > 1e10:  # milliseconds (13 digits)
                            timestamp_seconds = first_timestamp / 1000.0
                            date_str = datetime.fromtimestamp(timestamp_seconds).strftime("%Y-%m-%d")
                            logger.info(
                                f"[FeatureStore] Timestamp in milliseconds: {first_timestamp} -> {date_str}"
                            )
                        else:  # already in seconds (10 digits)
                            # Additional validation: timestamp should be reasonable (after 2020)
                            if first_timestamp < 1577836800:  # 2020-01-01 in seconds
                                logger.warning(
                                    f"[FeatureStore] Suspicious timestamp detected: {first_timestamp} "
                                    f"(before 2020-01-01 or too small). Using current date. "
                                    f"First 5 timestamps: {features[timestamp_column].head().tolist()}"
                                )
                                date_str = datetime.now().strftime("%Y-%m-%d")
                            else:
                                timestamp_seconds = first_timestamp
                                date_str = datetime.fromtimestamp(timestamp_seconds).strftime("%Y-%m-%d")
                                logger.info(
                                    f"[FeatureStore] Timestamp in seconds: {first_timestamp} -> {date_str}"
                                )
                    else:
                        # Timestamp is datetime object or other type
                        # This should rarely happen - log warning if we see unexpected types
                        logger.warning(
                            f"[FeatureStore] Unexpected timestamp type: {type(first_timestamp)}, "
                            f"value={first_timestamp}. Attempting pd.to_datetime conversion."
                        )
                        date_str = pd.to_datetime(first_timestamp).strftime("%Y-%m-%d")
                        logger.info(
                            f"[FeatureStore] Converted to date: {date_str}"
                        )
            else:
                date_str = datetime.now().strftime("%Y-%m-%d")
                logger.warning("[FeatureStore] Empty features DataFrame, using current date")

            # Save as parquet
            partition_dir = self.offline_dir / feature_group / f"date={date_str}"
            partition_dir.mkdir(parents=True, exist_ok=True)

            output_path = partition_dir / f"features_{datetime.now().strftime('%H%M%S')}.parquet"
            features.to_parquet(output_path, index=False)

            logger.info(
                f"Wrote offline features: group={feature_group}, "
                f"rows={len(features)}, path={output_path}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to write offline features: {e}")
            return False

    def read_offline_features(
        self,
        feature_group: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Прочитать фичи из offline store

        Args:
            feature_group: Группа фич
            start_date: Начальная дата (YYYY-MM-DD)
            end_date: Конечная дата (YYYY-MM-DD)
            columns: Список колонок (опционально)

        Returns:
            DataFrame с фичами
        """
        try:
            group_dir = self.offline_dir / feature_group

            if not group_dir.exists():
                logger.warning(f"Feature group not found: {feature_group}")
                return pd.DataFrame()

            # Find all parquet files
            parquet_files = []
            for partition_dir in group_dir.iterdir():
                if not partition_dir.is_dir():
                    continue

                # Extract date from partition name
                if partition_dir.name.startswith("date="):
                    partition_date = partition_dir.name.split("=")[1]

                    # Filter by date range
                    if start_date and partition_date < start_date:
                        continue
                    if end_date and partition_date > end_date:
                        continue

                    # Add all parquet files from this partition
                    parquet_files.extend(list(partition_dir.glob("*.parquet")))

            if not parquet_files:
                logger.warning(
                    f"No parquet files found: group={feature_group}, "
                    f"start={start_date}, end={end_date}"
                )
                return pd.DataFrame()

            # Read and concatenate
            dfs = []
            for file_path in parquet_files:
                df = pd.read_parquet(file_path, columns=columns)
                dfs.append(df)

            result = pd.concat(dfs, ignore_index=True)

            logger.info(
                f"Read offline features: group={feature_group}, "
                f"files={len(parquet_files)}, rows={len(result)}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to read offline features: {e}")
            return pd.DataFrame()

    def write_online_features(
        self,
        entity_id: str,
        features: np.ndarray,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Записать фичи в online store (для serving)

        Args:
            entity_id: ID сущности (например, "BTCUSDT")
            features: Feature vector
            ttl_seconds: TTL для кеша (опционально)

        Returns:
            True если успешно
        """
        try:
            # Add to cache
            ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else self.cache_ttl
            expire_at = datetime.now() + ttl

            self.online_cache[entity_id] = (features, expire_at)

            # Limit cache size (LRU)
            if len(self.online_cache) > self.max_cache_size:
                self.online_cache.popitem(last=False)

            # Persist to disk для восстановления
            online_path = self.online_dir / f"{entity_id}.pkl"
            with open(online_path, 'wb') as f:
                pickle.dump({
                    'features': features,
                    'expire_at': expire_at
                }, f)

            logger.debug(f"Wrote online features: entity={entity_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to write online features: {e}")
            return False

    def read_online_features(
        self,
        entity_id: str
    ) -> Optional[np.ndarray]:
        """
        Прочитать фичи из online store

        Args:
            entity_id: ID сущности

        Returns:
            Feature vector или None
        """
        try:
            # Check cache
            if entity_id in self.online_cache:
                features, expire_at = self.online_cache[entity_id]

                # Check expiration
                if datetime.now() < expire_at:
                    logger.debug(f"Cache hit: entity={entity_id}")
                    return features
                else:
                    # Expired, remove from cache
                    del self.online_cache[entity_id]
                    logger.debug(f"Cache expired: entity={entity_id}")

            # Try to load from disk
            online_path = self.online_dir / f"{entity_id}.pkl"
            if online_path.exists():
                with open(online_path, 'rb') as f:
                    data = pickle.load(f)

                features = data['features']
                expire_at = data['expire_at']

                # Check expiration
                if datetime.now() < expire_at:
                    # Restore to cache
                    self.online_cache[entity_id] = (features, expire_at)
                    logger.debug(f"Restored from disk: entity={entity_id}")
                    return features
                else:
                    # Expired, delete file
                    online_path.unlink()
                    logger.debug(f"Disk cache expired: entity={entity_id}")

            logger.debug(f"Cache miss: entity={entity_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to read online features: {e}")
            return None

    def get_feature_metadata(
        self,
        feature_name: str,
        version: str = "latest"
    ) -> Optional[FeatureMetadata]:
        """
        Получить метаданные фичи

        Args:
            feature_name: Название фичи
            version: Версия (или "latest")

        Returns:
            Метаданные или None
        """
        if version == "latest":
            # Find latest version
            matching_features = [
                (key, meta) for key, meta in self.feature_metadata.items()
                if meta.name == feature_name and meta.is_active
            ]

            if not matching_features:
                return None

            # Sort by version (descending)
            matching_features.sort(key=lambda x: x[1].version, reverse=True)
            return matching_features[0][1]
        else:
            feature_key = f"{feature_name}:{version}"
            return self.feature_metadata.get(feature_key)

    def list_features(
        self,
        feature_type: Optional[str] = None,
        active_only: bool = True
    ) -> List[FeatureMetadata]:
        """
        Список всех фич

        Args:
            feature_type: Фильтр по типу
            active_only: Только активные

        Returns:
            Список метаданных
        """
        features = list(self.feature_metadata.values())

        # Filter
        if feature_type:
            features = [f for f in features if f.feature_type == feature_type]

        if active_only:
            features = [f for f in features if f.is_active]

        return features

    def update_feature_stats(
        self,
        feature_name: str,
        version: str,
        stats: Dict[str, float]
    ) -> bool:
        """
        Обновить статистики фичи

        Args:
            feature_name: Название фичи
            version: Версия
            stats: Словарь со статистиками

        Returns:
            True если успешно
        """
        feature_key = f"{feature_name}:{version}"

        if feature_key not in self.feature_metadata:
            logger.error(f"Feature not found: {feature_key}")
            return False

        metadata = self.feature_metadata[feature_key]

        # Update stats
        if 'min' in stats:
            metadata.min_value = stats['min']
        if 'max' in stats:
            metadata.max_value = stats['max']
        if 'mean' in stats:
            metadata.mean_value = stats['mean']
        if 'std' in stats:
            metadata.std_value = stats['std']

        metadata.updated_at = datetime.now().isoformat()

        # Persist
        metadata_path = self.metadata_dir / f"{feature_key}.json"
        with open(metadata_path, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            metadata_dict = asdict(metadata)  # type: ignore[arg-type]
            json.dump(metadata_dict, f, indent=2)

        logger.info(f"Updated feature stats: {feature_key}")
        return True

    def clear_online_cache(self) -> int:
        """
        Очистить online cache

        Returns:
            Количество удаленных записей
        """
        count = len(self.online_cache)
        self.online_cache.clear()

        logger.info(f"Cleared online cache: {count} entries")
        return count

    def _load_metadata(self) -> None:
        """Загрузить метаданные из диска"""
        try:
            metadata_files = list(self.metadata_dir.glob("*.json"))

            for file_path in metadata_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                metadata = FeatureMetadata(**data)
                feature_key = f"{metadata.name}:{metadata.version}"
                self.feature_metadata[feature_key] = metadata

            logger.info(f"Loaded {len(metadata_files)} feature metadata")

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")

    async def initialize(self) -> bool:
        """
        Асинхронная инициализация Feature Store.

        Выполняет проверку директорий, загружает метаданные,
        очищает просроченный кэш.

        Returns:
            True если инициализация успешна
        """
        try:
            logger.info("Initializing Feature Store...")

            # Ensure directories exist
            for dir_path in [self.offline_dir, self.online_dir, self.metadata_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Reload metadata
            self._load_metadata()

            # Clean expired online cache
            self._clean_expired_cache()

            logger.info(
                f"✓ Feature Store initialized: "
                f"{len(self.feature_metadata)} features registered"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Feature Store: {e}")
            return False

    def _clean_expired_cache(self) -> int:
        """
        Очистить просроченные записи из online cache.

        Returns:
            Количество удалённых записей
        """
        expired_keys = []
        now = datetime.now()

        for entity_id, (features, expire_at) in self.online_cache.items():
            if now >= expire_at:
                expired_keys.append(entity_id)

        for key in expired_keys:
            del self.online_cache[key]

        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    async def get_training_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        feature_group: str = "training_features"
    ) -> Optional[pd.DataFrame]:
        """
        Получить данные для обучения за указанный период.

        Args:
            symbols: Список торговых пар (BTCUSDT, ETHUSDT, ...)
            start_date: Начальная дата
            end_date: Конечная дата
            feature_group: Группа фич для загрузки

        Returns:
            DataFrame с фичами или None если данные не найдены
        """
        try:
            logger.info(
                f"Loading training data: symbols={symbols}, "
                f"period={start_date.date()} to {end_date.date()}"
            )

            # Convert dates to string format for read_offline_features
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Try to read from offline store
            df = self.read_offline_features(
                feature_group=feature_group,
                start_date=start_str,
                end_date=end_str
            )

            if df.empty:
                # Try alternative feature groups
                alternative_groups = [
                    "ml_features",
                    "orderbook_features",
                    "candle_features",
                    "combined_features"
                ]

                for alt_group in alternative_groups:
                    df = self.read_offline_features(
                        feature_group=alt_group,
                        start_date=start_str,
                        end_date=end_str
                    )
                    if not df.empty:
                        logger.info(f"Found data in feature group: {alt_group}")
                        break

            if df.empty:
                logger.warning(
                    f"No training data found for period {start_str} to {end_str}"
                )
                return None

            # Filter by symbols if 'symbol' column exists
            if 'symbol' in df.columns and symbols:
                df = df[df['symbol'].isin(symbols)]

                if df.empty:
                    logger.warning(f"No data found for symbols: {symbols}")
                    return None

            # Sort by timestamp if column exists
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)

            logger.info(
                f"✓ Loaded {len(df)} rows of training data "
                f"({df.shape[1]} features)"
            )

            return df

        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return None


# Singleton instance
_feature_store_instance: Optional[FeatureStore] = None


def get_feature_store(
    storage_path: str = "data/feature_store"
) -> FeatureStore:
    """
    Получить singleton instance Feature Store

    Args:
        storage_path: Путь для хранения

    Returns:
        Feature Store instance
    """
    global _feature_store_instance

    if _feature_store_instance is None:
        _feature_store_instance = FeatureStore(storage_path=storage_path)

    return _feature_store_instance
