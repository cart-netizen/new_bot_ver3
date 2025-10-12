АРХИТЕКТУРНОЕ РЕШЕНИЕ И ПЛАН РЕАЛИЗАЦИИ
Фаза: ML Integration и Продвинутые Стратегии

I. EXECUTIVE SUMMARY
Текущее Состояние Проекта
✅ РЕАЛИЗОВАНО (Фаза 0 - Критический Фундамент):

Database Layer с PostgreSQL + TimescaleDB
FSM для жизненных циклов Orders и Positions
Idempotency Service с генерацией уникальных Client Order ID
Circuit Breaker Pattern для защиты от каскадных сбоев
Recovery & State Reconciliation при рестартах
Structured Logging с Trace ID Propagation
Advanced Rate Limiting (Token Bucket)
Repositories с Optimistic Locking
Audit Logging всех критических операций
Базовые модели OrderBook (Snapshot, Delta, Metrics)
OrderBookAnalyzer с расчетом базовых метрик стакана
OrderBookManager для управления состоянием стакана

❌ ОТСУТСТВУЕТ (Требует Реализации):

Feature Engineering из стакана (50+ признаков)
ML Infrastructure (Training, Serving, Versioning)
ML модели для валидации сигналов и предсказания движений
Детекторы манипуляций (Spoofing, Layering, RPI-awareness)
Продвинутые торговые стратегии (Momentum, SAR Wave, SuperTrend, Volume Profile)
Support/Resistance Level Detection с динамическим трекингом
Real-time Feature Store для ML inference
Model Drift Detection и автоматическая рекалибровка


II. АРХИТЕКТУРНОЕ ВИДЕНИЕ
Концептуальная Архитектура ML-Enhanced Trading System
На основе анализа научных исследований и best practices, предлагается гибридная многоуровневая архитектура, которая объединяет:

Real-time Feature Pipeline — непрерывное извлечение признаков из стакана, свечей и индикаторов
Multi-Modal ML Engine — использование специализированных моделей для разных типов данных
Ensemble Decision System — комбинирование ML-предсказаний с rule-based стратегиями
Adaptive Risk Management — динамическая корректировка на основе market regime detection

Ключевые Архитектурные Принципы
1. Separation of Concerns через Layered Architecture
┌──────────────────────────────────────────────────────────────┐
│                    STRATEGY LAYER                             │
│   (Momentum, SAR Wave, SuperTrend, Volume Profile)            │
│   ↓ генерируют сигналы на основе ↓                            │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                    ML VALIDATION LAYER                        │
│   (Signal Validator, Confidence Scorer, Regime Detector)      │
│   ↓ фильтрует и оценивает сигналы через ↓                     │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                    ML INFERENCE LAYER                         │
│   (Stockformer/CNN-LSTM для предсказания, <5ms latency)      │
│   ↓ работает на признаках из ↓                                │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING LAYER                  │
│   (50+ признаков: OrderBook + Candles + Indicators)          │
│   ↓ извлекаются из ↓                                          │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                    DATA AGGREGATION LAYER                     │
│   (OrderBook Manager, Candle Builder, Indicator Calculator)  │
│   ↓ получает данные из ↓                                      │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                    DATA FEED LAYER (УЖЕ РЕАЛИЗОВАН)          │
│   (WebSocket Manager, Market Data Streams)                   │
└──────────────────────────────────────────────────────────────┘
2. Multi-Modal Data Integration
Согласно исследованиям, графовое представление является наиболее эффективным подходом для интеграции разнородных данных:
Архитектура Multi-Modal Feature Fusion:
python# Концептуальная схема

OrderBook Features (50+)  ──┐
                            ├──> Feature Fusion Layer ──> Stockformer/CNN-LSTM ──> Prediction
Candle Features (OHLCV)   ──┤
                            │
Technical Indicators (30+) ──┘
Три подхода к интеграции (в порядке сложности):

Simple Concatenation (Начальная реализация)

Все признаки объединяются в единый вектор
Используется в Stockformer (360+ признаков)
✅ Простота реализации
❌ Не учитывает семантику разных типов данных


Multi-Channel Representation (Рекомендуется для первой итерации)

Разные типы данных обрабатываются отдельными модулями
Канал OrderBook: [OFI, Imbalance, VWAP, Depth]
Канал Price: [OHLC, Returns, Volatility]
Канал Indicators: [RSI, MACD, Bollinger Bands]
✅ Учитывает структуру данных
✅ Умеренная сложность


Graph-based Representation (Будущее развитие)

Граф, где узлы = признаки, рёбра = корреляции
Использует Graph Neural Networks (GNN)
✅ Максимальная гибкость
❌ Высокая сложность реализации



3. Model Architecture Selection
На основе Профессионального анализа исследований, рекомендуется следующая архитектура:
РЕКОМЕНДУЕМАЯ АРХИТЕКТУРА: Hybrid CNN-LSTM с Attention
Обоснование:

CNN извлекает локальные паттерны из временных окон (breakouts, clusters)
LSTM/BiLSTM моделирует долгосрочные зависимости
Attention Mechanism фокусируется на критических моментах
Доказанная эффективность на финансовых данных (см. исследования)
Баланс между производительностью и сложностью

Альтернатива для продвинутых сценариев: Stockformer

Multitask Learning (предсказание цены + волатильности + направления)
360+ признаков
Готовый код и инструменты для backtesting
Требует больше вычислительных ресурсов

python# Концептуальная архитектура Hybrid CNN-LSTM

Input: [batch, sequence_length, features]
  ↓
Conv1D Layers (извлечение паттернов)
  ↓ [batch, sequence_length, conv_features]
BiLSTM Layers (временные зависимости)
  ↓ [batch, sequence_length, lstm_hidden]
Attention Layer (фокусировка на важных моментах)
  ↓ [batch, attention_output]
Dense Layers (классификация/регрессия)
  ↓
Output: [batch, predictions]
  - Direction (Buy/Sell/Hold)
  - Confidence (0-1)
  - Expected Return (%)
4. Real-Time Inference Architecture
Критические требования:

Latency inference < 5ms
Throughput > 1000 predictions/sec для 100 пар
Zero-copy операции где возможно
Feature caching для reduce compute

Архитектура Feature Store:
┌─────────────────────────────────────────────────────────┐
│                   Redis Feature Store                    │
│  ┌────────────────────────────────────────────────┐     │
│  │  Key: f"features:{symbol}:{timestamp_ms}"      │     │
│  │  Value: {                                       │     │
│  │    "orderbook": [...50 features...],           │     │
│  │    "candles": [...20 features...],             │     │
│  │    "indicators": [...30 features...],          │     │
│  │    "computed_at": timestamp,                   │     │
│  │    "version": "v1.2.3"                         │     │
│  │  }                                              │     │
│  │  TTL: 60 seconds                                │     │
│  └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
                          ↓
                  (read by ML Inference)
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Model Serving Layer (FastAPI)              │
│  ┌────────────────────────────────────────────────┐     │
│  │  POST /predict                                  │     │
│  │  {                                              │     │
│  │    "symbol": "BTCUSDT",                        │     │
│  │    "features": {...},                          │     │
│  │    "model_version": "v1.2.3"                   │     │
│  │  }                                              │     │
│  │  Response:                                      │     │
│  │  {                                              │     │
│  │    "direction": "BUY",                         │     │
│  │    "confidence": 0.85,                         │     │
│  │    "expected_return": 0.023,                   │     │
│  │    "inference_time_ms": 3.2                    │     │
│  │  }                                              │     │
│  └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘

III. ДЕТАЛЬНАЯ ДЕКОМПОЗИЦИЯ КОМПОНЕНТОВ
1. Feature Engineering Layer
1.1 OrderBookFeatureExtractor
Файл: backend/ml_engine/features/orderbook_feature_extractor.py
Ответственность: Извлечение 50+ признаков из стакана в реальном времени
Признаки (категории):
A. Базовые Микроструктурные Признаки (15)
python1. bid_ask_spread_abs        # Абсолютный спред
2. bid_ask_spread_rel        # Относительный спред (%)
3. mid_price                 # Средняя цена
4. micro_price               # (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
5. vwap_bid_5                # VWAP для 5 уровней bids
6. vwap_ask_5                # VWAP для 5 уровней asks
7. vwap_bid_10               # VWAP для 10 уровней bids
8. vwap_ask_10               # VWAP для 10 уровней asks
9. depth_bid_5               # Совокупный объем на 5 уровнях bids
10. depth_ask_5              # Совокупный объем на 5 уровнях asks
11. depth_bid_10             # Совокупный объем на 10 уровнях bids
12. depth_ask_10             # Совокупный объем на 10 уровнях asks
13. total_bid_volume         # Общий объем всех bids
14. total_ask_volume         # Общий объем всех asks
15. book_depth_ratio         # total_bid_volume / total_ask_volume
B. Признаки Дисбаланса и Давления (10)
python16. imbalance_5              # (bid_vol_5 - ask_vol_5) / (bid_vol_5 + ask_vol_5)
17. imbalance_10             # (bid_vol_10 - ask_vol_10) / (bid_vol_10 + ask_vol_10)
18. imbalance_total          # (total_bid - total_ask) / (total_bid + total_ask)
19. price_pressure           # (bid_vol * ask_price - ask_vol * bid_price) / (bid_vol + ask_vol)
20. volume_delta_5           # Разница объемов на 5 уровнях за последние N секунд
21. order_flow_imbalance     # Кумулятивный OFI за окно
22. bid_intensity            # Количество bid ордеров / средний размер ордера
23. ask_intensity            # Количество ask ордеров / средний размер ордера
24. buy_sell_ratio           # Соотношение агрессивных покупок к продажам
25. smart_money_index        # Индикатор институциональной активности
C. Кластерные и Уровневые Признаки (10)
python26. largest_bid_cluster_price
27. largest_bid_cluster_volume
28. largest_ask_cluster_price
29. largest_ask_cluster_volume
30. num_bid_clusters         # Количество значимых кластеров
31. num_ask_clusters
32. support_level_1          # Ближайший уровень поддержки
33. resistance_level_1       # Ближайший уровень сопротивления
34. distance_to_support      # Расстояние от mid_price до support_level_1
35. distance_to_resistance   # Расстояние от mid_price до resistance_level_1
D. Признаки Ликвидности (8)
python36. liquidity_bid_5          # Объем, необходимый для движения цены на 5 уровней вниз
37. liquidity_ask_5          # Объем, необходимый для движения цены на 5 уровней вверх
38. liquidity_asymmetry      # liquidity_bid / liquidity_ask
39. effective_spread         # Эффективный спред с учетом объемов
40. kyle_lambda              # Коэффициент рыночного влияния
41. amihud_illiquidity       # Мера неликвидности Amihud
42. roll_spread              # Roll measure спреда
43. depth_imbalance_ratio    # Отношение глубины bids к asks на разных уровнях
E. Временные и Динамические Признаки (7+)
python44. level_ttl_avg            # Среднее время жизни уровня (для детекции spoofing)
45. level_ttl_std            # Стандартное отклонение TTL
46. orderbook_volatility     # Волатильность изменений стакана
47. update_frequency         # Частота обновлений стакана (updates/sec)
48. quote_intensity          # Интенсивность котировок
49. trade_arrival_rate       # Частота прихода сделок
50. spread_volatility        # Волатильность спреда

# Rolling window features (добавляют еще 10-20 признаков)
51-70. rolling_mean/std/skew для ключевых метрик (окна: 10s, 30s, 60s)
Оптимизация:

Использование numba для векторных операций
Pre-allocated numpy arrays для zero-copy
Caching промежуточных вычислений
Параллельная обработка для разных символов

1.2 CandleFeatureExtractor
Файл: backend/ml_engine/features/candle_feature_extractor.py
Признаки из свечных данных (20+):
python# Базовые OHLCV
1. open, high, low, close, volume

# Производные метрики
6. returns                    # (close - prev_close) / prev_close
7. log_returns               # ln(close / prev_close)
8. high_low_range            # (high - low) / close
9. close_open_diff           # (close - open) / open
10. upper_shadow             # (high - max(open, close)) / (high - low)
11. lower_shadow             # (min(open, close) - low) / (high - low)
12. body_size                # abs(close - open) / (high - low)

# Волатильность
13. realized_volatility      # std(returns) за окно
14. parkinson_volatility     # High-Low estimator
15. garman_klass_volatility  # OHLC estimator

# Volume features
16. volume_ma_ratio          # volume / MA(volume, 20)
17. volume_change_rate       # (volume - prev_volume) / prev_volume
18. price_volume_trend       # Кумулятивный PVT

# Pattern indicators
19. doji_strength            # Сила паттерна дожи
20. hammer_strength          # Сила паттерна молот
1.3 IndicatorFeatureExtractor
Файл: backend/ml_engine/features/indicator_feature_extractor.py
Технические индикаторы (30+):
python# Trend indicators
1. sma_10, sma_20, sma_50, sma_100, sma_200
6. ema_10, ema_20, ema_50
9. macd, macd_signal, macd_histogram
12. adx, +di, -di

# Momentum indicators
15. rsi_14, rsi_28
17. stochastic_k, stochastic_d
19. williams_r
20. cci
21. momentum_10

# Volatility indicators
22. bollinger_upper, bollinger_middle, bollinger_lower
25. bollinger_width, bollinger_pct
27. atr_14
28. keltner_upper, keltner_lower

# Volume indicators
30. obv (On-Balance Volume)
31. vwap
32. ad_line (Accumulation/Distribution)
33. cmf (Chaikin Money Flow)
34. mfi (Money Flow Index)

# Custom indicators
35. parabolic_sar
36. supertrend
37. ichimoku_components (5 линий)
1.4 FeaturePipeline (Orchestrator)
Файл: backend/ml_engine/features/feature_pipeline.py
Ответственность:

Координация всех extractors
Нормализация и масштабирование
Обработка missing values
Создание rolling windows
Кэширование в Redis Feature Store

Псевдокод:
pythonclass FeaturePipeline:
    """
    Главный оркестратор извлечения признаков
    """
    
    def __init__(
        self,
        orderbook_extractor: OrderBookFeatureExtractor,
        candle_extractor: CandleFeatureExtractor,
        indicator_extractor: IndicatorFeatureExtractor,
        feature_store: RedisFeatureStore,
        scaler: StandardScaler
    ):
        self.orderbook_extractor = orderbook_extractor
        self.candle_extractor = candle_extractor
        self.indicator_extractor = indicator_extractor
        self.feature_store = feature_store
        self.scaler = scaler
    
    async def extract_features(
        self,
        symbol: str,
        orderbook_snapshot: OrderBookSnapshot,
        candles: List[Candle],
        timestamp: int
    ) -> FeatureVector:
        """
        Извлекает все признаки для символа
        
        Возвращает:
            FeatureVector: Нормализованный вектор признаков
        """
        # Проверяем кэш
        cached = await self.feature_store.get(symbol, timestamp)
        if cached:
            return cached
        
        # Параллельное извлечение признаков
        orderbook_features = await self.orderbook_extractor.extract(
            orderbook_snapshot
        )
        
        candle_features = await self.candle_extractor.extract(
            candles
        )
        
        indicator_features = await self.indicator_extractor.extract(
            candles
        )
        
        # Объединение и нормализация
        raw_features = self._combine_features(
            orderbook_features,
            candle_features,
            indicator_features
        )
        
        normalized_features = self.scaler.transform(raw_features)
        
        feature_vector = FeatureVector(
            symbol=symbol,
            timestamp=timestamp,
            features=normalized_features,
            metadata={
                "orderbook_count": len(orderbook_features),
                "candle_count": len(candle_features),
                "indicator_count": len(indicator_features)
            }
        )
        
        # Сохраняем в кэш
        await self.feature_store.set(
            symbol,
            timestamp,
            feature_vector,
            ttl_seconds=60
        )
        
        return feature_vector

2. ML Models Layer
2.1 Model Architecture
Файл: backend/ml_engine/models/hybrid_cnn_lstm.py
Архитектура Hybrid CNN-LSTM:
pythonimport torch
import torch.nn as nn

class HybridCNNLSTM(nn.Module):
    """
    Гибридная модель CNN-LSTM для предсказания направления движения цены
    
    Architecture:
        Input -> Conv1D Blocks -> BiLSTM -> Attention -> Dense -> Output
    """
    
    def __init__(
        self,
        input_features: int = 100,  # Общее количество признаков
        sequence_length: int = 60,   # Длина временного окна (60 секунд)
        conv_filters: List[int] = [64, 128, 256],
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3  # Buy, Sell, Hold
    ):
        super().__init__()
        
        # CNN для извлечения локальных паттернов
        self.conv_blocks = nn.ModuleList()
        in_channels = input_features
        
        for filters in conv_filters:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, filters, kernel_size=3, padding=1),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = filters
        
        # BiLSTM для моделирования временных зависимостей
        self.lstm = nn.LSTM(
            input_size=conv_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Auxiliary outputs для multitask learning
        self.confidence_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Confidence score 0-1
        )
        
        self.return_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Expected return
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch, sequence_length, input_features)
        
        Returns:
            direction_logits: (batch, num_classes)
            confidence: (batch, 1)
            expected_return: (batch, 1)
        """
        # Transpose для Conv1D: (batch, features, sequence)
        x = x.transpose(1, 2)
        
        # CNN blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Transpose обратно для LSTM: (batch, sequence, features)
        x = x.transpose(1, 2)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, sequence, lstm_hidden*2)
        
        # Attention
        attention_weights = torch.softmax(
            self.attention(lstm_out), dim=1
        )  # (batch, sequence, 1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, lstm_hidden*2)
        
        # Outputs
        direction_logits = self.classifier(context)
        confidence = self.confidence_head(context)
        expected_return = self.return_head(context)
        
        return direction_logits, confidence, expected_return
2.2 Model Training Service
Файл: backend/ml_engine/training/model_trainer.py
Ответственность:

Обучение моделей на исторических данных
Walk-forward validation
Hyperparameter tuning
Model versioning через MLflow

Ключевые компоненты:
pythonclass ModelTrainer:
    """
    Сервис для обучения ML моделей
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        data_loader: HistoricalDataLoader,
        mlflow_tracker: MLflowTracker
    ):
        self.model_config = model_config
        self.data_loader = data_loader
        self.mlflow_tracker = mlflow_tracker
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        epochs: int = 100,
        early_stopping_patience: int = 10
    ) -> TrainedModel:
        """
        Обучение модели с early stopping
        """
        # Создаем модель
        model = HybridCNNLSTM(**self.model_config.model_params)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop с логированием в MLflow
        with self.mlflow_tracker.start_run():
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                train_loss, train_acc = self._train_epoch(
                    model, train_data, optimizer, criterion
                )
                
                # Validation
                val_loss, val_acc = self._validate_epoch(
                    model, val_data, criterion
                )
                
                # Log metrics
                self.mlflow_tracker.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, step=epoch)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Сохраняем лучшую модель
                    self._save_checkpoint(model, epoch)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Логируем финальную модель
            model_uri = self.mlflow_tracker.log_model(model)
            
            return TrainedModel(
                model=model,
                model_uri=model_uri,
                version=self.mlflow_tracker.get_run_id(),
                metrics={
                    "final_val_loss": best_val_loss,
                    "final_val_acc": val_acc
                }
            )
    
    def walk_forward_validation(
        self,
        data: pd.DataFrame,
        train_window: int = 30,  # дней
        test_window: int = 7      # дней
    ) -> WalkForwardResults:
        """
        Walk-forward валидация для проверки стабильности модели
        """
        results = []
        
        # Разбиваем данные на окна
        windows = self._create_walk_forward_windows(
            data, train_window, test_window
        )
        
        for idx, (train_data, test_data) in enumerate(windows):
            logger.info(f"Walk-forward fold {idx+1}/{len(windows)}")
            
            # Обучаем модель на training window
            model = self.train(train_data, test_data)
            
            # Тестируем на test window
            metrics = self._evaluate_model(model, test_data)
            
            results.append({
                "fold": idx,
                "train_period": (train_data.index[0], train_data.index[-1]),
                "test_period": (test_data.index[0], test_data.index[-1]),
                "metrics": metrics
            })
        
        return WalkForwardResults(results)
2.3 Model Serving Service
Файл: backend/ml_engine/inference/model_server.py
Ответственность:

Real-time inference с latency < 5ms
Model versioning и A/B testing
Graceful model updates без downtime

Архитектура:
pythonfrom fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from typing import Dict, Optional
import time

app = FastAPI(title="ML Model Serving")

class PredictionRequest(BaseModel):
    symbol: str
    features: Dict[str, float]
    model_version: Optional[str] = "latest"

class PredictionResponse(BaseModel):
    symbol: str
    direction: str  # BUY, SELL, HOLD
    confidence: float
    expected_return: float
    inference_time_ms: float
    model_version: str

class ModelServer:
    """
    Сервер для real-time ML inference
    """
    
    def __init__(self):
        self.models: Dict[str, torch.nn.Module] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.default_version = "latest"
        
        # Загружаем модели при старте
        self._load_models()
    
    def _load_models(self):
        """
        Загрузка всех доступных версий моделей
        """
        # Загружаем из MLflow Model Registry
        model_versions = self._fetch_model_versions_from_registry()
        
        for version_info in model_versions:
            model = self._load_model_from_mlflow(version_info['uri'])
            model.eval()  # Режим inference
            
            # Компилируем модель для ускорения (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                model = torch.compile(model)
            
            self.models[version_info['version']] = model
            self.model_metadata[version_info['version']] = version_info
            
            logger.info(f"Loaded model version {version_info['version']}")
    
    @torch.no_grad()
    async def predict(
        self,
        request: PredictionRequest
    ) -> PredictionResponse:
        """
        Выполняет inference для одного символа
        """
        start_time = time.perf_counter()
        
        # Получаем модель
        model_version = request.model_version or self.default_version
        model = self.models.get(model_version)
        
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model version {model_version} not found"
            )
        
        # Подготовка features
        features_array = self._prepare_features(request.features)
        features_tensor = torch.FloatTensor(features_array).unsqueeze(0)
        
        # Inference
        direction_logits, confidence, expected_return = model(features_tensor)
        
        # Post-processing
        direction_probs = torch.softmax(direction_logits, dim=1)
        predicted_class = torch.argmax(direction_probs, dim=1).item()
        
        direction_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        return PredictionResponse(
            symbol=request.symbol,
            direction=direction_map[predicted_class],
            confidence=confidence.item(),
            expected_return=expected_return.item(),
            inference_time_ms=inference_time_ms,
            model_version=model_version
        )
    
    async def reload_models(self):
        """
        Горячая перезагрузка моделей без downtime
        """
        logger.info("Reloading models...")
        new_models = {}
        
        model_versions = self._fetch_model_versions_from_registry()
        
        for version_info in model_versions:
            if version_info['version'] not in self.models:
                # Загружаем только новые версии
                model = self._load_model_from_mlflow(version_info['uri'])
                model.eval()
                new_models[version_info['version']] = model
        
        # Атомарное обновление
        self.models.update(new_models)
        
        logger.info(f"Models reloaded. Active versions: {list(self.models.keys())}")

# FastAPI endpoints
model_server = ModelServer()

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    return await model_server.predict(request)

@app.post("/reload-models")
async def reload_models_endpoint():
    await model_server.reload_models()
    return {"status": "success", "message": "Models reloaded"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_models": list(model_server.models.keys()),
        "default_version": model_server.default_version
    }

3. Manipulation Detection Layer
3.1 SpoofingDetector
Файл: backend/ml_engine/detectors/spoofing_detector.py
Ответственность: Обнаружение ложных котировок (spoofing)
Алгоритм:
pythonfrom dataclasses import dataclass
from typing import Dict, List, Optional
import time

@dataclass
class SpoofingSignal:
    """Сигнал о возможном spoofing"""
    symbol: str
    timestamp: int
    level_price: float
    level_volume: float
    ttl_ms: int  # Time To Live в миллисекундах
    confidence: float  # 0-1
    side: str  # BID или ASK

class SpoofingDetector:
    """
    Детектор spoofing атак через анализ TTL уровней стакана
    """
    
    def __init__(
        self,
        ttl_threshold_ms: int = 200,  # Уровни исчезающие быстрее считаются подозрительными
        volume_threshold_percentile: float = 90,  # Только крупные уровни
        history_window_seconds: int = 60
    ):
        self.ttl_threshold_ms = ttl_threshold_ms
        self.volume_threshold_percentile = volume_threshold_percentile
        self.history_window_seconds = history_window_seconds
        
        # История уровней для каждого символа
        self.level_history: Dict[str, List[LevelSnapshot]] = {}
    
    async def detect(
        self,
        symbol: str,
        orderbook_snapshot: OrderBookSnapshot,
        prev_snapshot: Optional[OrderBookSnapshot] = None
    ) -> List[SpoofingSignal]:
        """
        Обнаружение spoofing на основе TTL анализа
        """
        if prev_snapshot is None:
            # Первый снимок, нечего сравнивать
            return []
        
        signals = []
        
        # Анализируем BID сторону
        bid_signals = self._analyze_side(
            symbol,
            orderbook_snapshot.bids,
            prev_snapshot.bids,
            side="BID",
            timestamp=orderbook_snapshot.timestamp
        )
        signals.extend(bid_signals)
        
        # Анализируем ASK сторону
        ask_signals = self._analyze_side(
            symbol,
            orderbook_snapshot.asks,
            prev_snapshot.asks,
            side="ASK",
            timestamp=orderbook_snapshot.timestamp
        )
        signals.extend(ask_signals)
        
        return signals
    
    def _analyze_side(
        self,
        symbol: str,
        current_levels: List[Tuple[float, float]],
        prev_levels: List[Tuple[float, float]],
        side: str,
        timestamp: int
    ) -> List[SpoofingSignal]:
        """
        Анализ одной стороны стакана (BID или ASK)
        """
        signals = []
        
        # Создаем словари для быстрого поиска
        current_dict = {price: volume for price, volume in current_levels}
        prev_dict = {price: volume for price, volume in prev_levels}
        
        # Вычисляем порог объема (только крупные ордера интересны)
        all_volumes = [vol for _, vol in current_levels]
        volume_threshold = np.percentile(all_volumes, self.volume_threshold_percentile)
        
        # Ищем уровни которые исчезли
        for price, volume in prev_levels:
            if volume < volume_threshold:
                continue  # Игнорируем мелкие уровни
            
            if price not in current_dict:
                # Уровень исчез
                ttl_ms = self._calculate_ttl(symbol, price, timestamp)
                
                if ttl_ms < self.ttl_threshold_ms:
                    # Подозрительно быстрое исчезновение
                    confidence = self._calculate_confidence(ttl_ms, volume, volume_threshold)
                    
                    signals.append(SpoofingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        level_price=price,
                        level_volume=volume,
                        ttl_ms=ttl_ms,
                        confidence=confidence,
                        side=side
                    ))
                    
                    logger.warning(
                        f"Spoofing detected: {symbol} {side} level {price} "
                        f"(volume={volume:.2f}) disappeared in {ttl_ms}ms"
                    )
        
        return signals
    
    def _calculate_ttl(self, symbol: str, price: float, current_timestamp: int) -> int:
        """
        Вычисляет время жизни уровня
        """
        # Ищем в истории когда уровень появился
        history = self.level_history.get(symbol, [])
        
        for snapshot in reversed(history):
            if price in snapshot.levels:
                # Нашли первое появление
                ttl_ms = current_timestamp - snapshot.timestamp
                return ttl_ms
        
        # Не нашли в истории, считаем как 0
        return 0
    
    def _calculate_confidence(
        self,
        ttl_ms: int,
        volume: float,
        volume_threshold: float
    ) -> float:
        """
        Вычисляет уверенность в spoofing (0-1)
        """
        # Чем быстрее исчез и чем крупнее объем, тем выше confidence
        ttl_factor = max(0, (self.ttl_threshold_ms - ttl_ms) / self.ttl_threshold_ms)
        volume_factor = min(1.0, volume / (volume_threshold * 2))
        
        confidence = (ttl_factor * 0.7) + (volume_factor * 0.3)
        return confidence
3.2 LayeringDetector
Файл: backend/ml_engine/detectors/layering_detector.py
Ответственность: Обнаружение каскадных манипуляций (layering)
Алгоритм:
python@dataclass
class LayeringSignal:
    """Сигнал о возможном layering"""
    symbol: str
    timestamp: int
    side: str  # BID или ASK
    layer_count: int  # Количество слоев
    total_volume: float
    price_range: Tuple[float, float]
    confidence: float

class LayeringDetector:
    """
    Детектор layering через анализ появления/исчезновения групп уровней
    """
    
    def __init__(
        self,
        min_layer_count: int = 5,  # Минимум слоев для подозрения
        price_proximity_percent: float = 0.5,  # Слои в пределах 0.5% считаются одной группой
        simultaneous_threshold_ms: int = 500  # Слои появившиеся в течение 500ms
    ):
        self.min_layer_count = min_layer_count
        self.price_proximity_percent = price_proximity_percent
        self.simultaneous_threshold_ms = simultaneous_threshold_ms
        
        # История событий для pattern matching
        self.event_history: Dict[str, List[OrderBookEvent]] = {}
    
    async def detect(
        self,
        symbol: str,
        orderbook_snapshot: OrderBookSnapshot,
        prev_snapshot: Optional[OrderBookSnapshot] = None
    ) -> List[LayeringSignal]:
        """
        Обнаружение layering паттернов
        """
        if prev_snapshot is None:
            return []
        
        # Детектируем новые уровни
        new_levels = self._find_new_levels(
            orderbook_snapshot,
            prev_snapshot
        )
        
        # Логируем события
        self._log_events(symbol, new_levels, orderbook_snapshot.timestamp)
        
        # Анализируем паттерны в истории событий
        signals = self._analyze_patterns(symbol)
        
        return signals
    
    def _find_new_levels(
        self,
        current: OrderBookSnapshot,
        prev: OrderBookSnapshot
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Находит новые уровни появившиеся в стакане
        """
        prev_bid_dict = {price: vol for price, vol in prev.bids}
        prev_ask_dict = {price: vol for price, vol in prev.asks}
        
        new_bids = [
            (price, vol) for price, vol in current.bids
            if price not in prev_bid_dict
        ]
        
        new_asks = [
            (price, vol) for price, vol in current.asks
            if price not in prev_ask_dict
        ]
        
        return {
            "BID": new_bids,
            "ASK": new_asks
        }
    
    def _analyze_patterns(self, symbol: str) -> List[LayeringSignal]:
        """
        Ищет layering паттерны в истории событий
        
        Pattern: Множество уровней появляется почти одновременно
                 в узком ценовом диапазоне
        """
        signals = []
        
        history = self.event_history.get(symbol, [])
        if len(history) < self.min_layer_count:
            return signals
        
        # Ищем кластеры событий во времени
        time_clusters = self._find_time_clusters(history)
        
        for cluster in time_clusters:
            # Проверяем каждый кластер на layering признаки
            for side in ["BID", "ASK"]:
                side_events = [e for e in cluster if e.side == side]
                
                if len(side_events) >= self.min_layer_count:
                    # Проверяем близость цен
                    prices = [e.price for e in side_events]
                    price_range = (min(prices), max(prices))
                    mid_price = (price_range[0] + price_range[1]) / 2
                    price_spread_pct = (
                        (price_range[1] - price_range[0]) / mid_price * 100
                    )
                    
                    if price_spread_pct <= self.price_proximity_percent:
                        # Найден layering паттерн!
                        total_volume = sum(e.volume for e in side_events)
                        confidence = self._calculate_layering_confidence(
                            len(side_events),
                            price_spread_pct,
                            total_volume
                        )
                        
                        signals.append(LayeringSignal(
                            symbol=symbol,
                            timestamp=cluster[0].timestamp,
                            side=side,
                            layer_count=len(side_events),
                            total_volume=total_volume,
                            price_range=price_range,
                            confidence=confidence
                        ))
                        
                        logger.warning(
                            f"Layering detected: {symbol} {side} "
                            f"{len(side_events)} layers, "
                            f"total_volume={total_volume:.2f}, "
                            f"price_range={price_range}"
                        )
        
        return signals

4. Advanced Trading Strategies Layer
4.1 MomentumStrategy
Файл: backend/strategies/momentum_strategy.py
Логика:

Детектирует momentum на основе признаков из стакана + свечей
Использует ML для валидации силы momentum
Входит только при высокой confidence от ML модели

python@dataclass
class MomentumSignal:
    type: SignalType  # BUY или SELL
    strength: float  # 0-1
    entry_price: float
    take_profit: float
    stop_loss: float
    ml_confidence: float  # Confidence от ML модели
    momentum_score: float  # Momentum из анализа данных

class MomentumStrategy(IStrategy):
    """
    Momentum стратегия с ML валидацией
    """
    
    def __init__(
        self,
        config: MomentumConfig,
        ml_validator: MLSignalValidator,
        feature_pipeline: FeaturePipeline
    ):
        self.config = config
        self.ml_validator = ml_validator
        self.feature_pipeline = feature_pipeline
    
    async def analyze(
        self,
        symbol: str,
        market_data: MarketData
    ) -> Optional[MomentumSignal]:
        """
        Анализ momentum и генерация сигнала
        """
        # 1. Вычисляем momentum score
        momentum_score = self._calculate_momentum(market_data)
        
        if momentum_score < self.config.min_momentum_score:
            return None  # Слабый momentum
        
        # 2. Извлекаем признаки для ML
        features = await self.feature_pipeline.extract_features(
            symbol,
            market_data.orderbook_snapshot,
            market_data.candles,
            market_data.timestamp
        )
        
        # 3. Получаем ML предсказание
        ml_prediction = await self.ml_validator.validate_signal(
            symbol=symbol,
            signal_type="MOMENTUM",
            features=features
        )
        
        if ml_prediction.confidence < self.config.min_ml_confidence:
            return None  # ML не подтверждает сигнал
        
        # 4. Генерируем торговый сигнал
        direction = self._determine_direction(
            momentum_score,
            ml_prediction
        )
        
        if direction is None:
            return None
        
        # 5. Вычисляем entry, TP, SL
        entry_price = market_data.orderbook_snapshot.mid_price
        take_profit, stop_loss = self._calculate_levels(
            entry_price,
            direction,
            momentum_score,
            market_data.volatility
        )
        
        return MomentumSignal(
            type=direction,
            strength=momentum_score,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            ml_confidence=ml_prediction.confidence,
            momentum_score=momentum_score
        )
    
    def _calculate_momentum(self, market_data: MarketData) -> float:
        """
        Вычисляет momentum score из multiple sources
        """
        # Price momentum
        price_momentum = self._calculate_price_momentum(market_data.candles)
        
        # Volume momentum
        volume_momentum = self._calculate_volume_momentum(market_data.candles)
        
        # OrderBook momentum (imbalance direction)
        ob_momentum = self._calculate_orderbook_momentum(
            market_data.orderbook_snapshot
        )
        
        # Weighted average
        momentum_score = (
            price_momentum * 0.4 +
            volume_momentum * 0.3 +
            ob_momentum * 0.3
        )
        
        return np.clip(momentum_score, 0, 1)
4.2 SARWaveStrategy
Файл: backend/strategies/sar_wave_strategy.py
Логика:

Parabolic SAR для определения тренда
Elliott Wave детектор для волновой структуры
Flat market детектор для избегания флэтов
ML валидация перед входом

pythonclass SARWaveStrategy(IStrategy):
    """
    SAR Wave стратегия: комбинация Parabolic SAR + Elliott Waves
    """
    
    def __init__(
        self,
        config: SARWaveConfig,
        wave_detector: ElliottWaveDetector,
        flat_detector: FlatMarketDetector,
        ml_validator: MLSignalValidator
    ):
        self.config = config
        self.wave_detector = wave_detector
        self.flat_detector = flat_detector
        self.ml_validator = ml_validator
    
    async def analyze(
        self,
        symbol: str,
        market_data: MarketData
    ) -> Optional[Signal]:
        """
        Генерация сигналов на основе SAR + Waves
        """
        # 1. Проверяем что рынок НЕ в флэте
        is_flat = await self.flat_detector.is_flat(market_data)
        if is_flat.is_flat and is_flat.strength > 0.7:
            return None  # Избегаем флэтов
        
        # 2. Вычисляем Parabolic SAR
        sar_value = self._calculate_sar(market_data.candles)
        
        # 3. Детектируем волновую структуру
        wave_pattern = await self.wave_detector.detect(market_data)
        
        if wave_pattern.confidence < self.config.min_wave_confidence:
            return None  # Неуверенная волновая структура
        
        # 4. Проверяем SAR crossover
        current_price = market_data.candles[-1].close
        prev_sar = self._get_previous_sar(symbol)
        
        signal_type = None
        
        # Bullish crossover (цена пересекает SAR снизу вверх)
        if current_price > sar_value and market_data.candles[-2].close <= prev_sar:
            if wave_pattern.current_wave in [1, 3, 5]:  # Импульсные волны
                signal_type = SignalType.BUY
        
        # Bearish crossover
        elif current_price < sar_value and market_data.candles[-2].close >= prev_sar:
            if wave_pattern.current_wave in [2, 4]:  # Коррекционные волны
                signal_type = SignalType.SELL
        
        if signal_type is None:
            return None
        
        # 5. ML валидация
        features = await self.feature_pipeline.extract_features(...)
        ml_prediction = await self.ml_validator.validate_signal(...)
        
        if ml_prediction.confidence < self.config.min_ml_confidence:
            return None
        
        # 6. Генерируем сигнал
        return Signal(
            type=signal_type,
            symbol=symbol,
            strategy="SAR_Wave",
            strength=wave_pattern.confidence,
            entry_price=current_price,
            take_profit=self._calculate_tp(current_price, wave_pattern),
            stop_loss=sar_value,  # SAR как stop loss
            metadata={
                "sar_value": sar_value,
                "wave": wave_pattern.current_wave,
                "wave_confidence": wave_pattern.confidence,
                "ml_confidence": ml_prediction.confidence
            }
        )
4.3 SuperTrendStrategy
Использует SuperTrend индикатор в комбинации с volume analysis и ML.
4.4 VolumeProfileStrategy
Анализирует Volume Profile для поиска зон POC (Point of Control) и генерации сигналов.

5. ML Signal Validator
Файл: backend/ml_engine/validators/ml_signal_validator.py
Ответственность:

Валидация сигналов от стратегий через ML модель
Оценка confidence каждого сигнала
Фильтрация ложных сигналов

pythonclass MLSignalValidator:
    """
    Валидатор торговых сигналов через ML модель
    """
    
    def __init__(
        self,
        model_server: ModelServer,
        feature_pipeline: FeaturePipeline,
        config: ValidatorConfig
    ):
        self.model_server = model_server
        self.feature_pipeline = feature_pipeline
        self.config = config
    
    async def validate_signal(
        self,
        symbol: str,
        signal_type: str,  # MOMENTUM, SAR_WAVE, etc.
        features: FeatureVector,
        signal_strength: float = 0.0
    ) -> MLPrediction:
        """
        Валидирует торговый сигнал через ML модель
        
        Returns:
            MLPrediction с confidence score
        """
        # 1. Получаем предсказание от модели
        prediction = await self.model_server.predict(
            PredictionRequest(
                symbol=symbol,
                features=features.to_dict(),
                model_version="latest"
            )
        )
        
        # 2. Проверяем согласованность ML с сигналом
        is_aligned = self._check_alignment(
            signal_type,
            prediction.direction
        )
        
        if not is_aligned:
            # ML не согласна со стратегией
            logger.warning(
                f"{symbol} | ML disagreement: strategy={signal_type}, "
                f"ml={prediction.direction}, confidence={prediction.confidence}"
            )
            
            # Снижаем confidence
            prediction.confidence *= 0.5
        
        # 3. Комбинируем confidence от стратегии и ML
        final_confidence = (
            signal_strength * self.config.strategy_weight +
            prediction.confidence * self.config.ml_weight
        )
        
        return MLPrediction(
            direction=prediction.direction,
            confidence=final_confidence,
            expected_return=prediction.expected_return,
            raw_ml_confidence=prediction.confidence,
            inference_time_ms=prediction.inference_time_ms
        )
    
    def _check_alignment(
        self,
        signal_type: str,
        ml_direction: str
    ) -> bool:
        """
        Проверяет согласованность сигнала стратегии с ML
        """
        signal_direction_map = {
            "BUY": "BUY",
            "SELL": "SELL",
            "LONG": "BUY",
            "SHORT": "SELL"
        }
        
        expected_direction = signal_direction_map.get(signal_type)
        
        return expected_direction == ml_direction or ml_direction == "HOLD"

IV. ПОСЛЕДОВАТЕЛЬНОСТЬ РЕАЛИЗАЦИИ
Фаза 1: Feature Engineering Infrastructure (Неделя 1-2)
Цель: Создать полный pipeline извлечения признаков
Задачи:
День 1-3: OrderBook Feature Extraction

 Реализовать OrderBookFeatureExtractor с 50+ признаками
 Базовые микроструктурные признаки (spread, depth, vwap)
 Признаки дисбаланса и давления (imbalance, pressure, OFI)
 Кластерные признаки (largest clusters, S/R levels)
 Временные признаки (TTL, rolling statistics)
 Unit tests для всех методов экстракции

День 4-5: Candle & Indicator Feature Extraction

 Реализовать CandleFeatureExtractor
 OHLCV производные метрики
 Волатильность (realized, Parkinson, Garman-Klass)
 Volume features
 Реализовать IndicatorFeatureExtractor
 Trend indicators (SMA, EMA, MACD, ADX)
 Momentum indicators (RSI, Stochastic, Williams R)
 Volatility indicators (Bollinger, Keltner, ATR)
 Volume indicators (OBV, VWAP, MFI)

День 6-7: Feature Pipeline Integration

 Реализовать FeaturePipeline orchestrator
 Нормализация и масштабирование (StandardScaler)
 Обработка missing values
 Redis Feature Store для кэширования
 Performance optimization (numba, vectorization)
 Integration tests с real-time данными

Критерий успеха:

Извлечение 100+ признаков для одного символа < 10ms
Zero missing values после preprocessing
Redis cache hit rate > 80%


Фаза 2: ML Model Development (Неделя 3-4)
Цель: Разработать и обучить ML модели
Задачи:
День 1-3: Model Architecture

 Реализовать HybridCNNLSTM модель
 CNN блоки для локальных паттернов
 BiLSTM для временных зависимостей
 Attention mechanism
 Multitask learning heads (direction + confidence + return)
 Model unit tests

День 4-5: Training Infrastructure

 Реализовать ModelTrainer
 Data loading и preprocessing
 Training loop с early stopping
 Walk-forward validation
 MLflow integration для tracking
 Hyperparameter tuning (Optuna)

День 6-7: Model Evaluation & Deployment

 Backtesting на исторических данных
 Метрики оценки (accuracy, precision, recall, F1, Sharpe)
 Model versioning в MLflow Registry
 Экспорт модели для production (ONNX)
 Documentation моделей

Критерий успеха:

Accuracy > 60% на validation set
Sharpe ratio > 1.5 на backtest
Inference latency < 5ms


Фаза 3: ML Serving Infrastructure (Неделя 5)
Цель: Развернуть ML модели для real-time inference
Задачи:
День 1-2: Model Server

 Реализовать ModelServer на FastAPI
 Endpoints для inference
 Model versioning и A/B testing
 Graceful model reload
 Health checks и monitoring

День 3-4: Integration с Trading System

 Реализовать MLSignalValidator
 Интеграция с FeaturePipeline
 Async requests к Model Server
 Error handling и retries
 Fallback strategy при недоступности ML

День 5: Performance Testing

 Load testing (1000+ requests/sec)
 Latency testing (< 5ms target)
 Stress testing с multiple models
 Memory profiling и optimization

Критерий успеха:

Throughput > 1000 predictions/sec
P99 latency < 10ms
Uptime > 99.9%


Фаза 4: Manipulation Detection (Неделя 6)
Цель: Реализовать детекторы манипуляций
Задачи:
День 1-3: Spoofing Detection

 Реализовать SpoofingDetector
 TTL анализ уровней стакана
 Level history tracking
 Confidence scoring
 Alert generation
 Tests с synthetic spoofing data

День 4-5: Layering Detection

 Реализовать LayeringDetector
 Event history tracking
 Pattern recognition (multiple layers)
 Time clustering analysis
 Tests с synthetic layering data

День 6-7: Integration & Testing

 Интеграция детекторов в trading pipeline
 RPI-ордера awareness (Bybit specific)
 Signal filtering на основе манипуляций
 Backtesting с учетом детекции
 Alerts и monitoring

Критерий успеха:

Detection rate > 70% на synthetic data
False positive rate < 10%
Processing latency < 1ms


Фаза 5: Advanced Trading Strategies (Неделя 7-8)
Цель: Реализовать продвинутые стратегии с ML
Задачи:
День 1-2: Momentum Strategy

 Реализовать MomentumStrategy
 Momentum calculation (price + volume + orderbook)
 ML validation integration
 Entry/TP/SL logic
 Backtesting

День 3-4: SAR Wave Strategy

 Реализовать SARWaveStrategy
 Parabolic SAR calculation
 ElliottWaveDetector implementation
 FlatMarketDetector implementation
 ML validation integration
 Backtesting

День 5: SuperTrend Strategy

 Реализовать SuperTrendStrategy
 SuperTrend indicator calculation
 Volume analysis integration
 ML validation
 Backtesting

День 6-7: Volume Profile Strategy

 Реализовать VolumeProfileStrategy
 Volume Profile calculation
 POC detection
 Value Area analysis
 ML validation
 Backtesting

Критерий успеха:

Каждая стратегия: Sharpe ratio > 2.0 на backtest
ML validation улучшает результаты на 20%+
Win rate > 55%


Фаза 6: Support/Resistance & Optimization (Неделя 9)
Цель: Продвинутый анализ уровней и оптимизация
Задачи:
День 1-3: Support/Resistance Detection

 Реализовать SupportResistanceDetector
 Level detection алгоритм
 Dynamic level tracking
 Level strength calculation
 Historical level persistence
 Integration с strategies

День 4-5: System Optimization

 Profiling критических путей
 Numba optimization для hot paths
 Memory optimization
 Redis caching optimization
 Async optimization

День 6-7: Integration Testing

 End-to-end testing полного pipeline
 Paper trading mode
 Live testnet testing
 Performance monitoring
 Bug fixes

Критерий успеха:

Latency обработки < 5ms для 100 пар
Memory footprint < 4GB
CPU usage < 50%


Фаза 7: Model Drift Detection & Auto-Retraining (Неделя 10)
Цель: Мониторинг и автоматическое обновление моделей
Задачи:
День 1-3: Drift Detection

 Реализовать ModelDriftDetector
 Feature distribution monitoring
 Prediction performance tracking
 Statistical tests (KS test, PSI)
 Alert generation

День 4-5: Auto-Retraining Pipeline

 Реализовать AutoRetrainingService
 Scheduled retraining triggers
 Data collection pipeline
 Walk-forward validation
 Automatic model deployment

День 6-7: MLOps & Monitoring

 MLflow production setup
 Model registry management
 A/B testing infrastructure
 Champion/Challenger pattern
 Comprehensive monitoring dashboards

Критерий успеха:

Drift detection latency < 24 hours
Auto-retraining успешно раз в неделю
Zero-downtime model updates


V. ТЕХНОЛОГИЧЕСКИЙ СТЕК
ML/Data Science
yamlCore ML:
  - PyTorch: 2.0+ (главный фреймворк)
  - NumPy: 1.24+
  - Pandas: 2.0+
  - Scikit-learn: 1.3+ (preprocessing, metrics)

Deep Learning:
  - PyTorch Lightning: 2.0+ (training framework)
  - TorchMetrics: (метрики)

Feature Engineering:
  - TA-Lib: 0.4.0+ (технические индикаторы)
  - Numba: 0.57+ (JIT compilation)

ML Operations:
  - MLflow: 2.8+ (tracking, registry)
  - Optuna: 3.4+ (hyperparameter tuning)
  - ONNX Runtime: 1.16+ (production inference)

Model Serving:
  - FastAPI: 0.104+ (API server)
  - Uvicorn: 0.24+ (ASGI server)
  - Pydantic: 2.0+ (validation)
Data Storage & Caching
yamlPrimary Database:
  - PostgreSQL: 16+ (уже реализовано)
  - TimescaleDB: 2.12+ (time-series)

Feature Store:
  - Redis: 7+ (кэш признаков, < 5ms access)
  - TTL: 60 seconds для real-time features

Historical Data:
  - TimescaleDB hypertables
  - Retention: 3 months для training
  - Compression: после 7 дней
Performance Optimization
yamlComputation:
  - Numba: JIT compilation для hot paths
  - NumPy: векторизация операций
  - PyTorch: GPU acceleration (опционально)

Memory:
  - Pre-allocated arrays
  - Zero-copy operations где возможно
  - Object pooling для часто создаваемых объектов

Concurrency:
  - asyncio: для I/O операций
  - ThreadPoolExecutor: для CPU-bound задач
  - ProcessPoolExecutor: для heavy compute

VI. АРХИТЕКТУРНЫЕ ПАТТЕРНЫ
1. Multi-Modal Feature Fusion
python"""
Паттерн для объединения разнородных признаков
"""

class MultiModalFeatureFusion:
    """
    Fusion layer для объединения OrderBook, Candle, Indicator features
    """
    
    def __init__(self):
        # Отдельные encoders для каждого типа данных
        self.orderbook_encoder = nn.Sequential(...)
        self.candle_encoder = nn.Sequential(...)
        self.indicator_encoder = nn.Sequential(...)
        
        # Fusion layer
        self.fusion = nn.Sequential(...)
    
    def forward(
        self,
        orderbook_features,
        candle_features,
        indicator_features
    ):
        # Encode каждый тип отдельно
        ob_encoded = self.orderbook_encoder(orderbook_features)
        candle_encoded = self.candle_encoder(candle_features)
        indicator_encoded = self.indicator_encoder(indicator_features)
        
        # Concatenate
        fused = torch.cat([ob_encoded, candle_encoded, indicator_encoded], dim=-1)
        
        # Final fusion
        output = self.fusion(fused)
        
        return output
2. Ensemble Decision Making
python"""
Комбинирование ML predictions с rule-based strategies
"""

class EnsembleDecisionMaker:
    """
    Ensemble из ML модели и rule-based стратегий
    """
    
    def __init__(
        self,
        ml_validator: MLSignalValidator,
        strategies: List[IStrategy],
        weights: Dict[str, float]
    ):
        self.ml_validator = ml_validator
        self.strategies = strategies
        self.weights = weights
    
    async def make_decision(
        self,
        symbol: str,
        market_data: MarketData
    ) -> Optional[Signal]:
        """
        Генерирует финальное решение на основе ensemble
        """
        # 1. Получаем сигналы от всех стратегий
        strategy_signals = []
        for strategy in self.strategies:
            signal = await strategy.analyze(symbol, market_data)
            if signal:
                strategy_signals.append(signal)
        
        if not strategy_signals:
            return None
        
        # 2. ML валидация каждого сигнала
        validated_signals = []
        for signal in strategy_signals:
            ml_prediction = await self.ml_validator.validate_signal(...)
            if ml_prediction.confidence > self.config.min_confidence:
                validated_signals.append((signal, ml_prediction))
        
        if not validated_signals:
            return None
        
        # 3. Weighted voting
        final_signal = self._weighted_vote(validated_signals)
        
        return final_signal
3. Circuit Breaker для ML Service
python"""
Protection pattern для ML serving
"""

class MLServiceCircuitBreaker:
    """
    Circuit breaker специально для ML сервиса
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 30,
        fallback_strategy: str = "RULE_BASED"
    ):
        self.breaker = CircuitBreaker(...)
        self.fallback_strategy = fallback_strategy
    
    async def call_ml_service(
        self,
        request: PredictionRequest
    ) -> Union[MLPrediction, FallbackPrediction]:
        """
        Вызывает ML сервис через circuit breaker
        """
        try:
            return await self.breaker.call_async(
                self._ml_service_call,
                request
            )
        except CircuitBreakerError:
            # Fallback на rule-based предсказание
            logger.warning("ML service unavailable, using fallback")
            return await self._fallback_prediction(request)
    
    async def _fallback_prediction(
        self,
        request: PredictionRequest
    ) -> FallbackPrediction:
        """
        Fallback стратегия когда ML недоступна
        """
        # Простые правила на основе признаков
        # Например: если imbalance > 0.7 -> BUY с confidence 0.5
        ...

VII. МОНИТОРИНГ И МЕТРИКИ
ML-Specific Metrics
python# Prometheus metrics для ML

from prometheus_client import Histogram, Counter, Gauge

# Inference metrics
ml_inference_latency = Histogram(
    'ml_inference_latency_seconds',
    'ML model inference latency',
    ['model_version', 'symbol']
)

ml_prediction_confidence = Histogram(
    'ml_prediction_confidence',
    'ML prediction confidence distribution',
    ['model_version', 'direction']
)

# Feature metrics
feature_extraction_latency = Histogram(
    'feature_extraction_latency_seconds',
    'Feature extraction latency',
    ['feature_type', 'symbol']
)

feature_cache_hit_rate = Gauge(
    'feature_cache_hit_rate',
    'Redis feature cache hit rate'
)

# Model performance metrics
ml_accuracy = Gauge(
    'ml_prediction_accuracy',
    'Rolling ML prediction accuracy',
    ['model_version', 'time_window']
)

# Drift metrics
feature_drift_score = Gauge(
    'feature_drift_score',
    'Feature distribution drift score',
    ['feature_name', 'model_version']
)

model_performance_drift = Gauge(
    'model_performance_drift',
    'Model performance drift over time',
    ['model_version', 'metric']
)

# Strategy metrics
strategy_signal_count = Counter(
    'strategy_signals_total',
    'Total signals generated by strategy',
    ['strategy_name', 'signal_type', 'ml_validated']
)

strategy_win_rate = Gauge(
    'strategy_win_rate',
    'Strategy win rate',
    ['strategy_name']
)

# Manipulation detection metrics
spoofing_detected = Counter(
    'spoofing_events_total',
    'Total spoofing events detected',
    ['symbol', 'side']
)

layering_detected = Counter(
    'layering_events_total',
    'Total layering events detected',
    ['symbol', 'side']
)
Grafana Dashboards
Dashboard 1: ML Performance

Inference latency (P50, P95, P99)
Prediction confidence distribution
Model accuracy over time
Feature drift scores
Cache hit rates

Dashboard 2: Strategy Performance

Signals generated per strategy
ML validation rate
Win rate per strategy
PnL per strategy
Drawdown metrics

Dashboard 3: Market Manipulation

Spoofing events per symbol
Layering events per symbol
Manipulation confidence scores
Impact on trading decisions


VIII. РИСКИ И МИТИГАЦИЯ
Технические Риски
РискВероятностьВлияниеМитигацияML inference latency > 5msСредняяВысокое1. ONNX optimization<br>2. Feature caching<br>3. Model quantization<br>4. Batch inferenceFeature drift снижает точностьВысокаяВысокое1. Continuous monitoring<br>2. Auto-retraining раз в неделю<br>3. A/B testing новых моделейOverfitting на backtestingВысокаяКритическое1. Walk-forward validation<br>2. Out-of-sample testing<br>3. Multiple time periods<br>4. Paper trading перед productionMemory leaks в feature extractionНизкаяСреднее1. Object pooling<br>2. Regular profiling<br>3. Memory monitoring<br>4. Graceful degradationRedis downtimeНизкаяСреднее1. Redis Cluster<br>2. Fallback на direct compute<br>3. Cache warming
Торговые Риски
РискВероятностьВлияниеМитигацияFalse positive signals от MLСредняяВысокое1. Ensemble с rule-based<br>2. Confidence thresholds<br>3. Position sizing по confidenceManipulation не детектитсяСредняяВысокое1. Conservative thresholds<br>2. Multiple detectors<br>3. Human review периодическиMarket regime changeВысокаяКритическое1. Regime detection<br>2. Auto-pause trading<br>3. Quick retrainingSlippage на быстрых движенияхСредняяСреднее1. Limit orders<br>2. Smart order routing<br>3. Latency optimization

IX. КРИТЕРИИ УСПЕХА ФАЗЫ
Технические KPI
✅ Feature Engineering:

100+ признаков извлекаются < 10ms
Feature cache hit rate > 80%
Zero missing values после preprocessing

✅ ML Models:

Accuracy > 60% на validation
Inference latency < 5ms (P99 < 10ms)
Sharpe ratio > 1.5 на backtest

✅ Infrastructure:

Throughput > 1000 predictions/sec
Uptime ML service > 99.9%
Auto-retraining работает раз в неделю

✅ Strategies:

Каждая стратегия: Sharpe > 2.0
ML validation улучшает results на 20%+
Win rate > 55%

✅ Manipulation Detection:

Detection rate > 70%
False positive rate < 10%
Processing latency < 1ms

Торговые KPI
✅ Backtesting Results:

Общий Sharpe ratio > 2.5
Max drawdown < 15%
Profit factor > 2.0
Win rate > 55%

✅ Paper Trading (1 месяц):

Consistency с backtesting (+/- 10%)
Zero critical bugs
Stable latency

✅ Live Trading (Testnet):

Positive PnL
Max daily loss < 2%
Все safety mechanisms работают


X. ДАЛЬНЕЙШЕЕ РАЗВИТИЕ
После успешной реализации Фазы ML Integration, следующие направления:
Short-term (3-6 месяцев)

Advanced ML Architectures

Stockformer implementation
Graph Neural Networks для multi-asset
Reinforcement Learning для portfolio optimization


Enhanced Manipulation Detection

Wash trading detection
Quote stuffing detection
Advanced RPI-ордера handling


Multi-Exchange Arbitrage

Cross-exchange orderbook analysis
Arbitrage opportunity detection
Smart order routing



Mid-term (6-12 месяцев)

Portfolio Optimization

Modern Portfolio Theory integration
Risk parity strategies
Dynamic allocation


Market Making Strategies

Liquidity provision strategies
Inventory management
Adverse selection mitigation


News & Sentiment Integration

NLP для новостей и социальных медиа
Sentiment analysis
Event-driven trading



Long-term (12+ месяцев)

Autonomous Trading System

Full reinforcement learning agent
Self-optimizing strategies
Minimal human intervention


Multi-Asset Class Expansion

Traditional markets integration
Options and futures
Cross-asset strategies




XI. ЗАКЛЮЧЕНИЕ
Предлагаемая архитектура представляет собой production-ready, scalable, и maintainable решение для интеграции ML в торговую систему.
Ключевые Преимущества Архитектуры
✅ Модульность: Каждый компонент независим и тестируем
✅ Производительность: Optimized для real-time с latency < 5ms
✅ Надежность: Circuit breakers, fallbacks, monitoring
✅ Масштабируемость: Горизонтальное масштабирование
✅ Observability: Comprehensive metrics и logging
✅ Maintainability: Clean code, документация, tests