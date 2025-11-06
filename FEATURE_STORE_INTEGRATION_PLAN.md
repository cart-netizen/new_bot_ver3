# Feature Store → Sequences Integration Plan

План реализации конвертации данных из Feature Store в sequences для обучения модели в `retraining_pipeline.py`.

---

## 📋 Текущее состояние

**Файл**: `backend/ml_engine/auto_retraining/retraining_pipeline.py:431-438`

```python
# Convert to sequences (simplified - should match your feature pipeline)
# This is a placeholder - implement according to your data structure
logger.info(f"Collected {len(features_df)} samples from Feature Store")

# For now, fallback to legacy loader
# TODO: Implement proper Feature Store → sequences conversion
data_loader = HistoricalDataLoader(self.config.data_config or DataConfig())
return data_loader.load_and_split()
```

**Проблема**:
- Feature Store возвращает `pd.DataFrame`
- Нужно конвертировать в sequences для модели
- Сейчас просто fallback на legacy loader (игнорирует Feature Store данные)

---

## 🎯 Цель

Реализовать полный pipeline:
```
Feature Store (DataFrame)
  → Extract features/labels
  → Create sequences
  → Train/Val/Test split
  → Create DataLoaders
  → Return for training
```

---

## 📊 Анализ структуры данных

### 1. Feature Store Output

**Метод**: `feature_store.read_offline_features()`

**Возвращает**: `pd.DataFrame` со структурой:
```python
# Предполагаемые колонки:
{
    'timestamp': int64,           # Unix timestamp
    'symbol': str,                # Торговая пара (BTCUSDT, etc.)

    # Features (110 фич, как в ModelConfig)
    'bid_price_level_0': float,
    'bid_volume_level_0': float,
    'ask_price_level_0': float,
    ...
    'indicator_rsi': float,
    'indicator_macd': float,
    ...

    # Labels (если есть)
    'future_direction_60s': int,  # 0, 1, 2 (down, neutral, up)
    'future_return_60s': float,   # Expected return
}
```

### 2. Model Input Requirements

**Класс**: `HybridCNNLSTM`

**Ожидает**:
```python
input_shape = (batch_size, sequence_length, input_features)
# sequence_length = 60 (из DataConfig)
# input_features = 110 (из ModelConfig)
```

**Labels**:
```python
labels = (batch_size,)  # Classification: 0, 1, 2
```

### 3. DataLoader Output

**Метод**: `HistoricalDataLoader.create_dataloaders()`

**Возвращает**:
```python
{
    'train': DataLoader,
    'val': DataLoader,
    'test': DataLoader  # optional
}
```

---

## 🏗️ Архитектура решения

### Вариант 1: Новый класс FeatureStoreDataConverter

**Преимущества**:
- Чистая separation of concerns
- Переиспользуемый код
- Легко тестировать

**Недостатки**:
- Дополнительный класс

```python
class FeatureStoreDataConverter:
    """Конвертер данных из Feature Store в sequences для обучения"""

    def __init__(self, data_config: DataConfig):
        self.config = data_config

    def convert_to_sequences(
        self,
        features_df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str,
        timestamp_column: str = 'timestamp'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Конвертировать DataFrame в sequences

        Returns:
            sequences: (N, seq_len, features)
            labels: (N,)
            timestamps: (N,)
        """
```

### Вариант 2: Метод в HistoricalDataLoader

**Преимущества**:
- Всё в одном месте
- Использует существующую логику

**Недостатки**:
- Смешивание ответственностей

```python
class HistoricalDataLoader:
    ...

    def load_from_dataframe(
        self,
        features_df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """Загрузить данные из DataFrame и создать DataLoaders"""
```

### ✅ Рекомендация: Вариант 2

Добавить метод `load_from_dataframe()` в `HistoricalDataLoader`, т.к.:
- Переиспользует существующую логику (create_sequences, train_val_test_split, create_dataloaders)
- Не дублирует код
- Логически относится к DataLoader

---

## 📝 Детальный план реализации

### Шаг 1: Определить структуру Feature Store данных

**Файл**: Проверить реальную структуру `write_offline_features()`

**Задачи**:
1. Посмотреть какие колонки записываются в Feature Store
2. Определить feature_columns (список из 110 фич)
3. Определить label_column (например, `future_direction_60s`)
4. Определить timestamp_column (обычно `timestamp`)

**Результат**: Чёткое понимание схемы данных

---

### Шаг 2: Создать метод `load_from_dataframe()` в HistoricalDataLoader

**Файл**: `backend/ml_engine/training/data_loader.py`

**Сигнатура**:
```python
def load_from_dataframe(
    self,
    features_df: pd.DataFrame,
    feature_columns: List[str],
    label_column: str = 'future_direction_60s',
    timestamp_column: str = 'timestamp',
    symbol_column: Optional[str] = 'symbol',
    apply_resampling: bool = False
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Загрузить данные из DataFrame и создать DataLoaders.

    Используется для загрузки данных из Feature Store.

    Args:
        features_df: DataFrame с фичами из Feature Store
        feature_columns: Список колонок с фичами (110 columns)
        label_column: Колонка с метками (future_direction_60s)
        timestamp_column: Колонка с timestamps
        symbol_column: Колонка с символами (опционально)
        apply_resampling: Применять ли class balancing

    Returns:
        train_loader, val_loader, test_loader
    """
```

**Логика**:
```python
def load_from_dataframe(self, features_df, feature_columns, ...):
    # 1. Валидация
    if features_df.empty:
        raise ValueError("Empty DataFrame")

    # Проверить наличие необходимых колонок
    required_cols = feature_columns + [label_column, timestamp_column]
    missing = set(required_cols) - set(features_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 2. Сортировка по времени (КРИТИЧНО для временных рядов!)
    features_df = features_df.sort_values(timestamp_column).reset_index(drop=True)

    # 3. Извлечение данных
    X = features_df[feature_columns].values  # (N, 110)
    y = features_df[label_column].values     # (N,)
    timestamps = features_df[timestamp_column].values  # (N,)

    logger.info(f"Загружено из DataFrame: {len(X):,} семплов")
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")

    # 4. Class distribution check
    from collections import Counter
    class_dist = Counter(y)
    logger.info(f"Распределение классов: {dict(class_dist)}")

    # 5. Resampling (если включен)
    if apply_resampling and self.balancing_strategy:
        logger.info("Применение class balancing...")
        X, y = self.balancing_strategy.balance_dataset(X, y)

        # Обновить timestamps (простая стратегия)
        if len(timestamps) != len(X):
            if len(timestamps) < len(X):
                # Oversample: дублируем timestamps
                indices = np.random.choice(len(timestamps), len(X), replace=True)
                timestamps = timestamps[indices]
            else:
                # Undersample: обрезаем
                timestamps = timestamps[:len(X)]

    # 6. Создание sequences (переиспользуем существующий метод)
    sequences, seq_labels, seq_timestamps = self.create_sequences(
        X, y, timestamps
    )

    logger.info(f"Создано sequences: {sequences.shape}")

    # 7. Train/Val/Test split (переиспользуем существующий метод)
    train_data, val_data, test_data = self.train_val_test_split(
        sequences, seq_labels, seq_timestamps
    )

    # 8. Create DataLoaders (переиспользуем существующий метод)
    dataloaders = self.create_dataloaders(train_data, val_data, test_data)

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders.get('test', None)

    logger.info(f"✓ DataLoaders созданы из Feature Store данных")
    logger.info(f"  • Train batches: {len(train_loader)}")
    logger.info(f"  • Val batches: {len(val_loader)}")
    if test_loader:
        logger.info(f"  • Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader
```

---

### Шаг 3: Создать конфигурацию колонок

**Файл**: `backend/ml_engine/feature_store/feature_schema.py` (новый файл)

**Цель**: Определить схему данных Feature Store

```python
"""
Feature Store Schema - определяет структуру данных
"""

from typing import List
from dataclasses import dataclass


@dataclass
class FeatureStoreSchema:
    """Схема данных Feature Store"""

    # Meta columns
    timestamp_column: str = 'timestamp'
    symbol_column: str = 'symbol'

    # Label columns
    label_column: str = 'future_direction_60s'
    return_column: str = 'future_return_60s'

    # Feature columns (110 features)
    orderbook_features: List[str] = None  # 50 фич: bid/ask levels
    candle_features: List[str] = None     # 20 фич: OHLCV, derived
    indicator_features: List[str] = None  # 20 фич: RSI, MACD, etc.
    volume_features: List[str] = None     # 10 фич: volume metrics
    spread_features: List[str] = None     # 10 фич: spread metrics

    def __post_init__(self):
        """Generate default feature lists"""
        if self.orderbook_features is None:
            self.orderbook_features = self._generate_orderbook_features()

        if self.candle_features is None:
            self.candle_features = self._generate_candle_features()

        # ... аналогично для остальных

    def _generate_orderbook_features(self) -> List[str]:
        """Generate orderbook feature names"""
        features = []
        for i in range(10):  # 10 levels
            features.extend([
                f'bid_price_level_{i}',
                f'bid_volume_level_{i}',
                f'ask_price_level_{i}',
                f'ask_volume_level_{i}',
            ])
        # + imbalance, weighted mid, etc.
        features.extend([
            'orderbook_imbalance',
            'weighted_mid_price',
            'total_bid_volume',
            'total_ask_volume',
            'bid_ask_spread',
            'spread_bps'
        ])
        return features

    def _generate_candle_features(self) -> List[str]:
        """Generate candle feature names"""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'hl_range', 'oc_range', 'upper_shadow', 'lower_shadow',
            'body_size', 'volume_ma_5', 'volume_ma_20',
            'return_1m', 'return_5m', 'return_15m',
            'volatility_1h', 'volatility_4h', 'volatility_24h',
            'vwap', 'typical_price'
        ]

    def get_all_feature_columns(self) -> List[str]:
        """Get list of all feature columns"""
        return (
            self.orderbook_features +
            self.candle_features +
            self.indicator_features +
            self.volume_features +
            self.spread_features
        )

    def validate_dataframe(self, df) -> bool:
        """Validate that DataFrame has all required columns"""
        required = (
            [self.timestamp_column, self.symbol_column, self.label_column] +
            self.get_all_feature_columns()
        )
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return True


# Global default schema
DEFAULT_SCHEMA = FeatureStoreSchema()
```

---

### Шаг 4: Обновить `_collect_training_data()` в retraining_pipeline

**Файл**: `backend/ml_engine/auto_retraining/retraining_pipeline.py`

**Было**:
```python
# Convert to sequences (simplified - should match your feature pipeline)
# This is a placeholder - implement according to your data structure
logger.info(f"Collected {len(features_df)} samples from Feature Store")

# For now, fallback to legacy loader
# TODO: Implement proper Feature Store → sequences conversion
data_loader = HistoricalDataLoader(self.config.data_config or DataConfig())
return data_loader.load_and_split()
```

**Станет**:
```python
# Convert Feature Store DataFrame to sequences and DataLoaders
logger.info(f"Collected {len(features_df)} samples from Feature Store")

# Import schema
from backend.ml_engine.feature_store.feature_schema import DEFAULT_SCHEMA

# Validate DataFrame
try:
    DEFAULT_SCHEMA.validate_dataframe(features_df)
except ValueError as e:
    logger.error(f"Invalid DataFrame schema: {e}")
    # Fallback to legacy loader
    data_loader = HistoricalDataLoader(self.config.data_config or DataConfig())
    return data_loader.load_and_split()

# Convert to DataLoaders
logger.info("Converting Feature Store data to DataLoaders...")
data_loader = HistoricalDataLoader(self.config.data_config or DataConfig())

train_loader, val_loader, test_loader = data_loader.load_from_dataframe(
    features_df=features_df,
    feature_columns=DEFAULT_SCHEMA.get_all_feature_columns(),
    label_column=DEFAULT_SCHEMA.label_column,
    timestamp_column=DEFAULT_SCHEMA.timestamp_column,
    symbol_column=DEFAULT_SCHEMA.symbol_column,
    apply_resampling=True  # Enable class balancing for retraining
)

logger.info("✓ Feature Store data successfully converted to DataLoaders")
return train_loader, val_loader, test_loader
```

---

### Шаг 5: Обработка edge cases

**1. Multiple symbols в DataFrame**

```python
# В load_from_dataframe():
if symbol_column and symbol_column in features_df.columns:
    # Обработать каждый символ отдельно или вместе?
    symbols = features_df[symbol_column].unique()
    logger.info(f"Found {len(symbols)} symbols: {symbols}")

    # Вариант А: Объединить все символы (рекомендуется)
    # Просто сортируем по timestamp, символы смешиваются

    # Вариант Б: Обучать отдельно для каждого символа
    # Создать sequences для каждого символа отдельно
    # symbol_sequences = {}
    # for symbol in symbols:
    #     symbol_df = features_df[features_df[symbol_column] == symbol]
    #     ...
```

**2. Отсутствующие labels**

```python
# Проверка наличия labels
if label_column not in features_df.columns:
    logger.warning(f"Label column '{label_column}' not found")
    logger.warning("Available columns: {list(features_df.columns)}")
    raise ValueError("Cannot train without labels")

# Проверка на NaN в labels
null_labels = features_df[label_column].isnull().sum()
if null_labels > 0:
    logger.warning(f"Found {null_labels} null labels, dropping them")
    features_df = features_df.dropna(subset=[label_column])
```

**3. Недостаточно данных**

```python
min_samples_required = self.config.sequence_length * 10  # Минимум 10 sequences

if len(features_df) < min_samples_required:
    logger.error(
        f"Insufficient data: {len(features_df)} samples, "
        f"need at least {min_samples_required}"
    )
    # Fallback to legacy loader
    data_loader = HistoricalDataLoader(...)
    return data_loader.load_and_split()
```

**4. Разные feature dimensions**

```python
# Проверить размерность features
expected_features = 110  # Из ModelConfig.input_features
actual_features = len(feature_columns)

if actual_features != expected_features:
    logger.error(
        f"Feature dimension mismatch: expected {expected_features}, "
        f"got {actual_features}"
    )
    raise ValueError("Feature dimension mismatch")
```

---

## 🧪 План тестирования

### Unit Tests

**Файл**: `tests/ml_engine/training/test_dataloader_from_dataframe.py`

```python
def test_load_from_dataframe_basic():
    """Test basic DataFrame to DataLoader conversion"""
    # Create mock DataFrame
    df = create_mock_feature_dataframe(n_samples=1000)

    # Load
    loader = HistoricalDataLoader(DataConfig())
    train, val, test = loader.load_from_dataframe(
        df, feature_columns=MOCK_FEATURES, label_column='label'
    )

    # Assertions
    assert len(train) > 0
    assert len(val) > 0

def test_load_from_dataframe_with_resampling():
    """Test with class balancing"""
    # Create imbalanced DataFrame
    df = create_imbalanced_dataframe()

    # Load with resampling
    train, val, test = loader.load_from_dataframe(
        df, ..., apply_resampling=True
    )

    # Check class distribution is more balanced
    ...

def test_load_from_dataframe_edge_cases():
    """Test edge cases"""
    # Empty DataFrame
    # Missing columns
    # Insufficient data
    # NaN values
    ...
```

### Integration Tests

**Файл**: `tests/ml_engine/integration/test_feature_store_to_training.py`

```python
async def test_full_pipeline():
    """Test Feature Store → Sequences → Training"""
    # 1. Write features to Feature Store
    feature_store = get_feature_store()
    feature_store.write_offline_features(...)

    # 2. Read back
    df = feature_store.read_offline_features(...)

    # 3. Convert to DataLoaders
    loader = HistoricalDataLoader(...)
    train, val, test = loader.load_from_dataframe(df, ...)

    # 4. Train model
    model = HybridCNNLSTM(...)
    trainer = ModelTrainer(model, ...)
    results = trainer.train(train, val)

    # 5. Assertions
    assert results['final_val_accuracy'] > 0.5
```

---

## 📊 Метрики успеха

1. **Корректность данных**:
   - ✅ Shape sequences: (N, 60, 110)
   - ✅ Labels: (N,) с values [0, 1, 2]
   - ✅ Временной порядок сохранён

2. **Производительность**:
   - ✅ Время загрузки < 5 секунд на 100K samples
   - ✅ Memory usage разумный (< 2GB для 1M samples)

3. **Качество модели**:
   - ✅ Model training succeeds without errors
   - ✅ Validation accuracy > baseline (legacy loader)

---

## 🚀 План внедрения

### Фаза 1: Подготовка (1-2 часа)
1. Изучить реальную структуру Feature Store данных
2. Создать `feature_schema.py`
3. Написать unit tests для `load_from_dataframe()`

### Фаза 2: Реализация (2-3 часа)
1. Реализовать `load_from_dataframe()` в `HistoricalDataLoader`
2. Обновить `_collect_training_data()` в `retraining_pipeline.py`
3. Запустить unit tests

### Фаза 3: Интеграционное тестирование (1-2 часа)
1. Сгенерировать тестовые данные в Feature Store
2. Запустить полный pipeline
3. Проверить обучение модели

### Фаза 4: Оптимизация (опционально)
1. Профилирование производительности
2. Кэширование часто используемых схем
3. Параллельная загрузка для multiple symbols

---

## 📝 Checklist реализации

- [ ] Шаг 1: Определить схему Feature Store данных
- [ ] Шаг 2: Создать `feature_schema.py`
- [ ] Шаг 3: Реализовать `load_from_dataframe()` в `HistoricalDataLoader`
- [ ] Шаг 4: Обновить `_collect_training_data()` в `retraining_pipeline.py`
- [ ] Шаг 5: Написать unit tests
- [ ] Шаг 6: Написать integration tests
- [ ] Шаг 7: Протестировать на реальных данных
- [ ] Шаг 8: Обновить документацию
- [ ] Шаг 9: Code review
- [ ] Шаг 10: Merge в main

---

## 🔗 Связанные файлы

- `backend/ml_engine/auto_retraining/retraining_pipeline.py` - использует
- `backend/ml_engine/training/data_loader.py` - добавить метод
- `backend/ml_engine/feature_store/feature_store.py` - источник данных
- `backend/ml_engine/feature_store/feature_schema.py` - создать новый
- `backend/ml_engine/models/hybrid_cnn_lstm.py` - target model

---

## 💡 Альтернативные подходы

### Подход А: Direct conversion (текущий план)
✅ Простой
✅ Переиспользует существующий код
❌ Требует схему данных

### Подход Б: Automatic schema detection
```python
def load_from_dataframe_auto(df):
    # Автоматически определить feature columns
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'symbol', 'future_direction_60s']]
    ...
```
✅ Гибкий
❌ Может захватить лишние колонки
❌ Менее явный

### Подход В: Feature Store preprocessing
```python
class FeatureStore:
    def get_training_data(self, start_date, end_date) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Прямо возвращает DataLoaders"""
        ...
```
✅ Инкапсуляция
❌ Feature Store отвечает за слишком много
❌ Сложнее тестировать

**Рекомендация**: Подход А (direct conversion) - оптимальный баланс

---

## 📚 Дополнительные материалы

- MLflow Integration Guide: `ML_INFRASTRUCTURE_GUIDE.md`
- Feature Store Documentation: `backend/ml_engine/feature_store/README.md`
- Model Training Guide: `backend/ml_engine/training/README.md`

---

**Дата создания**: 2025-11-06
**Статус**: Ready for implementation
**Приоритет**: High
**Effort**: 4-6 hours
