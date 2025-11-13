"""
Модуль конфигурации приложения.
Загружает настройки из переменных окружения и валидирует их.
"""

import os
from typing import List, Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from dotenv import load_dotenv

# from core.logger import get_logger

# from core.logger import get_logger

# logger = get_logger(__name__)
# Загрузка переменных окружения из .env файла
load_dotenv()

def clean_env_value(value: str) -> str:
  """
  Очистка значения от комментариев и лишних пробелов.

  Args:
      value: Исходное значение из .env

  Returns:
      str: Очищенное значение
  """
  if not value:
    return value

  # Удаляем всё после # (комментарий)
  if '#' in value:
    value = value.split('#')[0]

  # Удаляем лишние пробелы
  value = value.strip()

  return value

class Settings(BaseSettings):
  """Класс настроек приложения с валидацией."""

  STOP_LOSS_PERCENT: float = 0.8
  TAKE_PROFIT_PERCENT: float = 3

  # ===== НАСТРОЙКИ ПРИЛОЖЕНИЯ =====
  APP_NAME: str = Field(default="Scalping Trading Bot")
  APP_VERSION: str = Field(default="1.0.0")
  DEBUG: bool = Field(default=True)
  LOG_LEVEL: str = Field(default="INFO")

  # ===== БЕЗОПАСНОСТЬ =====
  SECRET_KEY: str = Field(
    ...,
    description="Секретный ключ для JWT токенов"
  )
  ALGORITHM: str = Field(default="HS256")
  ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=1440)  # 24 часа
  APP_PASSWORD: str = Field(
    ...,
    description="Пароль для входа в приложение"
  )

  # ===== НАСТРОЙКИ BYBIT API =====
  BYBIT_MODE: Literal["testnet", "mainnet"] = Field(default="testnet")
  BYBIT_API_KEY: str = Field(default="")
  BYBIT_API_SECRET: str = Field(default="")
  BYBIT_MAINNET_API_KEY: str = Field(default="")
  BYBIT_MAINNET_API_SECRET: str = Field(default="")
  BYBIT_TESTNET_URL: str = "https://api-testnet.bybit.com"
  BYBIT_MAINNET_URL: str = "https://api.bybit.com"

  # ===== ТОРГОВЫЕ НАСТРОЙКИ =====
  TRADING_PAIRS: str = Field(default="BTCUSDT,ETHUSDT,SOLUSDT")
  MAX_PAIRS_PER_CONNECTION: int = Field(default=10)
  ORDERBOOK_DEPTH: int = Field(default=200)

  # ===== НАСТРОЙКИ СТРАТЕГИИ =====
  IMBALANCE_BUY_THRESHOLD: float = Field(default=0.75)
  IMBALANCE_SELL_THRESHOLD: float = Field(default=0.25)
  MIN_CLUSTER_VOLUME: float = Field(default=10000)

  # ===== НАСТРОЙКИ РИСК-МЕНЕДЖМЕНТА =====
  MAX_OPEN_POSITIONS: int = Field(default=2)
  MAX_EXPOSURE_USDT: float = Field(default=10000)
  MIN_ORDER_SIZE_USDT: float = Field(default=5)
  MAX_POSITION_SIZE_USDT: float = 1000.0
  IMBALANCE_THRESHOLD: float = 0.7

  # ===== ML CONFIGURATION =====
  ML_SERVER_URL: str = Field(
      default="http://localhost:8001",
      description="URL ML сервера для валидации сигналов"
  )
  ML_MIN_CONFIDENCE: float = Field(
      default=0.6,
      ge=0.0,
      le=1.0,
      description="Минимальная уверенность ML для валидации"
  )
  ML_WEIGHT: float = Field(
      default=0.6,
      ge=0.0,
      le=1.0,
      description="Вес ML в гибридном решении"
  )
  STRATEGY_WEIGHT: float = Field(
      default=0.4,
      ge=0.0,
      le=1.0,
      description="Вес стратегии в гибридном решении")

  ONLY_TRAINING: bool = Field(
      default=False,
      description="Режим только сбора данных для ML (без поиска сигналов). True - только сбор данных, False - полная работа бота"
  )

  # ===== STRATEGY MANAGER CONFIGURATION =====
  CONSENSUS_MODE: Literal["weighted", "majority", "unanimous"] = Field(
      default="weighted",
      description="Режим консенсуса стратегий"
  )
  MIN_STRATEGIES: int = Field(
      default=2,
      ge=1,
      le=10,
      description="Минимальное количество стратегий для консенсуса"
  )
  MIN_CONSENSUS_CONFIDENCE: float = Field(
      default=0.6,
      ge=0.0,
      le=1.0,
      description="Минимальная уверенность для консенсуса"
  )

  # ========================================
  # ✅ ИСПРАВЛЕНО: Adaptive Consensus Settings
  # ========================================

  # Основные настройки
  ENABLE_ADAPTIVE_CONSENSUS: bool = Field(
    default=True,
    description="Включить адаптивный консенсус стратегий"
  )

  ADAPTIVE_MIN_SIGNALS_FOR_EVALUATION: int = Field(default=20)
  ADAPTIVE_WEIGHT_UPDATE_FREQUENCY_SECONDS: int = Field(default=21600)

  # Performance Tracking
  PERFORMANCE_DATA_DIR: str = Field(
    default="data/strategy_performance",
    description="Директория для хранения данных о производительности"
  )
  PERFORMANCE_TRACKING_ENABLED: bool = Field(
    default=True,
    description="Включить отслеживание производительности стратегий"
  )

  # Regime Detection
  REGIME_DETECTION_ENABLED: bool = Field(
    default=True,
    description="Включить детекцию рыночных режимов"
  )
  REGIME_UPDATE_FREQUENCY_SECONDS: int = Field(
    default=300,
    ge=60,
    le=3600,
    description="Частота обновления режима рынка (секунды)"
  )

  # Weight Optimization
  WEIGHT_OPTIMIZATION_ENABLED: bool = Field(
    default=True,
    description="Включить оптимизацию весов стратегий"
  )
  WEIGHT_OPTIMIZATION_METHOD: Literal["PERFORMANCE", "REGIME", "HYBRID", "BAYESIAN"] = Field(
    default="HYBRID",
    description="Метод оптимизации весов"
  )
  WEIGHT_UPDATE_FREQUENCY_SECONDS: int = Field(
    default=21600,  # 6 часов
    ge=3600,
    le=86400,
    description="Частота обновления весов стратегий (секунды)"
  )

  # ==================== MULTI-TIMEFRAME SETTINGS ====================

  # Multi-Timeframe Analysis
  ENABLE_MTF_ANALYSIS: bool = Field(
    default=True,

    description="Включить multi-timeframe анализ"
  )

  MTF_ACTIVE_TIMEFRAMES: str = Field(
    default="1m,5m,15m,1h",

    description="Активные таймфреймы (через запятую)"
  )

  MTF_PRIMARY_TIMEFRAME: str = Field(
    default="1h",

    description="Основной таймфрейм для тренда"
  )

  MTF_EXECUTION_TIMEFRAME: str = Field(
    default="1m",

    description="Таймфрейм для точного входа"
  )

  MTF_SYNTHESIS_MODE: str = Field(
    default="top_down",

    description="Режим синтеза: top_down, consensus, confluence"
  )

  MTF_MIN_QUALITY: float = Field(
    default=0.60,

    description="Минимальное качество MTF сигнала"
  )

  MTF_STAGGERED_UPDATE_INTERVAL: int = Field(default=30)

  # ==================== INTEGRATED ENGINE SETTINGS ====================

  # Integrated Analysis Engine
  INTEGRATED_ANALYSIS_MODE: str = Field(
    default="hybrid",

    description="Режим анализа: single_tf_only, mtf_only, hybrid, adaptive"
  )

  HYBRID_MTF_PRIORITY: float = Field(
    default=0.6,

    description="Вес MTF в hybrid режиме (0-1)"
  )

  HYBRID_MIN_AGREEMENT: bool = Field(
    default=True,

    description="Требовать согласия между single-TF и MTF"
  )

  HYBRID_CONFLICT_RESOLUTION: str = Field(
    default="highest_quality",

    description="Стратегия разрешения конфликтов: mtf, single_tf, highest_quality"
  )

  MIN_COMBINED_QUALITY: float = Field(
    default=0.65,

    description="Минимальное качество интегрированного сигнала"
  )
  MIN_SIGNAL_CONFIDENCE: float = Field(
    default=0.60,
    ge=0.0,
    le=1.0,
    description="Минимальная уверенность сигнала для исполнения"
  )


  # === ДОПОЛНИТЕЛЬНЫЕ ===
  ANALYSIS_WARNING_THRESHOLD: float = Field(default=2.0)
  MIN_CANDLES_FOR_ANALYSIS: int = Field(default=50)
  POSITION_CHECK_INTERVAL: int = Field(default=30)
  RECOVERY_CHECK_INTERVAL: int = Field(default=300)
  AUTO_CLOSE_ON_STOP: bool = Field(default=False)
  ENABLE_NOTIFICATIONS: bool = Field(default=False)
  ENABLE_CRITICAL_ALERTS: bool = Field(default=True)

  CANDLE_LIMIT: int = Field(default=200)  # ✅ ДОБАВИТЬ
  CANDLE_UPDATE_INTERVAL: int = Field(default=5)  # Секунды между обновлениями

  @field_validator("CONSENSUS_MODE", mode="before")
  @classmethod
  def validate_consensus_mode(cls, v):
    """
    Валидация и очистка CONSENSUS_MODE.
    """
    if not v:
      print("⚠️ CONSENSUS_MODE не задан, используется значение по умолчанию: weighted")
      return "weighted"

    # Очистка от комментариев (если есть)
    if isinstance(v, str) and '#' in v:
      v = v.split('#')[0].strip()

    # Проверка допустимых значений
    valid_modes = ["weighted", "majority", "unanimous"]
    if v not in valid_modes:
      error_msg = (
        f"❌ Неизвестный CONSENSUS_MODE: '{v}'. "
        f"Допустимые значения: {', '.join(valid_modes)}"
      )
      print(error_msg)
      raise ValueError(
        f"Invalid CONSENSUS_MODE: '{v}'. "
        f"Must be one of: {', '.join(valid_modes)}"
      )

    print(f"✓ CONSENSUS_MODE: {v}")
    return v

  @field_validator("WEIGHT_OPTIMIZATION_METHOD", mode="before")
  @classmethod
  def validate_optimization_method(cls, v):
    """
    Валидация метода оптимизации весов.
    """
    if not v:
      return "HYBRID"

    # Очистка от комментариев
    if isinstance(v, str) and '#' in v:
      v = v.split('#')[0].strip()

    # Приведение к верхнему регистру
    v = v.upper()

    valid_methods = ["PERFORMANCE", "REGIME", "HYBRID", "BAYESIAN"]
    if v not in valid_methods:
      print(
        f"⚠️ Неизвестный WEIGHT_OPTIMIZATION_METHOD: '{v}'. "
        f"Используется HYBRID"
      )
      return "HYBRID"

    print(f"✓ WEIGHT_OPTIMIZATION_METHOD: {v}")
    return v

  @field_validator("CONSENSUS_MODE", mode="before")
  @classmethod
  def validate_consensus_mode(cls, v):
      """
      Валидация и очистка CONSENSUS_MODE.
      Удаляет комментарии и проверяет допустимые значения.
      """
      if not v:
          print("⚠️ CONSENSUS_MODE не задан, используется значение по умолчанию: weighted")
          return "weighted"

      # Очистка от комментариев
      original_value = v
      v = clean_env_value(str(v))

      if v != original_value:
          print(f"⚠️ CONSENSUS_MODE содержал комментарий: '{original_value}' -> '{v}'")

      # Проверка допустимых значений
      valid_modes = ["weighted", "majority", "unanimous"]
      if v not in valid_modes:
          error_msg = (
              f"❌ Неизвестный CONSENSUS_MODE: '{v}'. "
              f"Допустимые значения: {', '.join(valid_modes)}"
          )
          print(error_msg)
          raise ValueError(
              f"Invalid CONSENSUS_MODE: '{v}'. "
              f"Must be one of: {', '.join(valid_modes)}"
          )

      print(f"✓ CONSENSUS_MODE: {v}")
      return v

  # ===== LEVERAGE CONFIGURATION =====
  DEFAULT_LEVERAGE: int = Field(
    default=10,
    ge=1,
    le=100,
    description="Кредитное плечо по умолчанию"
  )
  MAX_LEVERAGE: int = Field(
    default=50,
    ge=1,
    le=100,
    description="Максимальное допустимое кредитное плечо"
  )

  @field_validator("MAX_LEVERAGE", mode="before")
  @classmethod
  def validate_max_leverage(cls, v, info):
    """Проверка, что MAX_LEVERAGE >= DEFAULT_LEVERAGE"""
    # ИСПРАВЛЕНО: Сначала конвертируем в int
    try:
      v_int = int(v)
    except (TypeError, ValueError):
      raise ValueError(f"MAX_LEVERAGE должен быть целым числом, получено: {v}")

    default_leverage = info.data.get("DEFAULT_LEVERAGE", 10)
    if v_int < default_leverage:
      raise ValueError(
        f"MAX_LEVERAGE ({v_int}) должен быть >= DEFAULT_LEVERAGE ({default_leverage})"
      )

    return v_int

  # ===== НАСТРОЙКИ API СЕРВЕРА =====
  API_HOST: str = Field(default="0.0.0.0")
  API_PORT: int = Field(default=8000)
  CORS_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:5173")

  # ===== НАСТРОЙКИ WEBSOCKET =====
  WS_RECONNECT_TIMEOUT: int = Field(default=5)
  WS_MAX_RECONNECT_ATTEMPTS: int = Field(default=10)
  WS_PING_INTERVAL: int = Field(default=20)

  # WebSocket настройки
  WS_RECONNECT_DELAY: int = 5
  AUTO_RECONCILE_ON_STARTUP: bool = True
  RECONCILE_INTERVAL_MINUTES: int = 60

  # Circuit Breaker настройки
  CIRCUIT_BREAKER_ENABLED: bool = True
  CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
  CIRCUIT_BREAKER_COOLDOWN_SECONDS: int = 60

  # Idempotency настройки
  IDEMPOTENCY_TTL_MINUTES: int = 6

  # Database (PostgreSQL + TimescaleDB)
  DATABASE_URL: str = "postgresql+asyncpg://user:robocop@localhost:5432/trading_bot"

  # Database настройки
  DB_POOL_SIZE: int = 10
  DB_MAX_OVERFLOW: int = 20
  DB_ECHO: bool = False

  # MLflow Tracking (PostgreSQL Backend)
  MLFLOW_TRACKING_URI: str = Field(
    default="postgresql://trading_bot:robocop@localhost:5432/trading_bot",
    description="MLflow Tracking URI (PostgreSQL backend)"
  )
  MLFLOW_ARTIFACT_LOCATION: str = Field(
    default="./mlruns/artifacts",
    description="Path for MLflow artifacts storage"
  )
  MLFLOW_EXPERIMENT_NAME: str = Field(
    default="trading_bot_ml",
    description="Default MLflow experiment name"
  )

  # Memory Profiling (опционально для диагностики)
  ENABLE_MEMORY_PROFILING: bool = Field(
    default=False,  # CRITICAL: Disabled - causes 2-3 minute freezes during cleanup!
    description="Enable memory profiling (adds overhead, use only for debugging)"
  )

  # Redis (для будущего использования)
  REDIS_URL: Optional[str] = None
  REDIS_HOST: str = "localhost"
  REDIS_PORT: int = 6379
  REDIS_DB: int = 0

  # Rate Limiting настройки
  RATE_LIMIT_ENABLED: bool = True
  RATE_LIMIT_REST_PUBLIC: int = 120  # запросов в минуту
  RATE_LIMIT_REST_PRIVATE: int = 120
  RATE_LIMIT_REST_TRADE: int = 100
  RATE_LIMIT_ORDER_PLACEMENT: int = 50

  # ==================== RECOVERY SERVICE ====================
  # ✅ ИСПРАВЛЕНО: Добавлены аннотации типов для всех полей

  # Таймаут для обнаружения зависших ордеров (в минутах)
  # Если ордер находится в статусе PENDING или PLACED дольше этого времени,
  # он считается зависшим
  HANGING_ORDER_TIMEOUT_MINUTES: int = Field(
    default=30,
    description="Timeout для обнаружения зависших ордеров (минуты)"
  )

  # Включить автоматическое восстановление после краша
  # Если True - система автоматически выполнит recover_from_crash() при старте
  ENABLE_AUTO_RECOVERY: bool = Field(
    default=True,
    description="Автоматическое восстановление при старте"
  )

  # Включить проверку зависших ордеров при каждой сверке состояния
  ENABLE_HANGING_ORDER_CHECK: bool = Field(
    default=True,
    description="Проверка зависших ордеров"
  )

  # ===== НОВЫЕ НАСТРОЙКИ =====
  # Автоматическое исправление зависших ордеров
  AUTO_FIX_HANGING_ORDERS: bool = Field(
    default=True,
    description="Автоматически исправлять зависшие ордера при обнаружении"
  )

  # Автоматическое создание позиций
  AUTO_CREATE_POSITIONS_FROM_FILLED: bool = Field(
    default=True,
    description="Автоматически создавать позиции из исполненных ордеров"
  )

  # Автоматически восстанавливать FSM при старте
  # Если True - все активные ордера и позиции получат FSM при старте системы
  ENABLE_FSM_AUTO_RESTORE: bool = Field(
    default=True,
    description="Автоматическое восстановление FSM"
  )

  # Максимальное количество попыток получить данные с биржи при сверке
  MAX_RECONCILIATION_RETRIES: int = Field(
    default=3,
    description="Максимум попыток сверки с биржей"
  )

  # Задержка между попытками сверки (в секундах)
  RECONCILIATION_RETRY_DELAY: int = Field(
    default=2,
    description="Задержка между попытками сверки (секунды)"
  )

  # Логировать детальную информацию о каждом зависшем ордере
  DETAILED_HANGING_ORDER_LOGGING: bool = Field(
    default=True,
    description="Детальное логирование зависших ордеров"
  )

  # ==================== SCREENER SETTINGS ====================
  """
  Настройки скринера торговых пар.
  """

  # Bybit WebSocket URL для тикеров (публичный endpoint)
  BYBIT_WS_URL: str = Field(
    default="wss://stream.bybit.com/v5/public/linear",
    description="Bybit WebSocket v5 URL для публичных данных"
  )

  # Минимальный объем за 24ч для отображения в скринере (USDT)
  SCREENER_MIN_VOLUME: float = Field(
    default=4_000_000.0,
    description="Минимальный объем торгов за 24ч в USDT для фильтрации пар"
  )

  # Максимальное количество пар в скринере
  SCREENER_MAX_PAIRS: int = Field(
    default=200,
    description="Максимальное количество торговых пар в памяти"
  )

  # Интервал broadcast данных скринера (секунды)
  SCREENER_BROADCAST_INTERVAL: float = Field(
    default=2.0,
    description="Интервал отправки данных скринера через WebSocket"
  )

  # Интервал очистки неактивных пар (секунды)
  SCREENER_CLEANUP_INTERVAL: int = Field(
    default=60,
    description="Интервал очистки неактивных пар из памяти"
  )

  # TTL для неактивных пар (секунды)
  SCREENER_INACTIVE_TTL: int = Field(
    default=300,
    description="Время жизни неактивной пары (без обновлений) перед удалением"
  )

  # Включение/отключение скринера
  SCREENER_ENABLED: bool = Field(
    default=True,
    description="Включить/отключить функционал скринера"
  )

  # Логирование статистики скринера (каждые N секунд)
  SCREENER_STATS_LOG_INTERVAL: int = Field(
    default=60,
    description="Интервал логирования статистики скринера"
  )

  # ===== DYNAMIC SYMBOLS SETTINGS =====
  DYNAMIC_SYMBOLS_ENABLED: bool = Field(
    default=True,
    description="Включить динамическое управление списком пар"
  )

  DYNAMIC_MIN_VOLUME: float = Field(
    default=4_000_000.0,
    description="Минимальный объем для отбора пар"
  )

  DYNAMIC_MAX_VOLUME_PAIRS: int = Field(
    default=200,
    description="Максимум пар после фильтра по объему"
  )

  DYNAMIC_TOP_GAINERS: int = Field(
    default=40,
    description="Количество растущих пар"
  )

  DYNAMIC_TOP_LOSERS: int = Field(
    default=20,
    description="Количество падающих пар"
  )

  DYNAMIC_REFRESH_INTERVAL: int = Field(
    default=300,
    description="Интервал обновления списка пар (секунды)"
  )

  SIGNAL_COOLDOWN_SECONDS: int = Field(default=60, ge=10, le=300)
  ALLOW_SAME_DIRECTION_SIGNALS: bool = Field(default=False)

  ALLOW_MULTIPLE_POSITIONS_PER_SYMBOL: bool = Field(
    default=False,
    description="Разрешить несколько позиций по одной паре"
  )

  # =====================================================
  # РАСШИРЕННЫЙ РИСК-МЕНЕДЖМЕНТ
  # =====================================================

  # Унифицированный SL/TP Calculator
  SLTP_CALCULATION_METHOD: str = Field(
    default=os.getenv("SLTP_CALCULATION_METHOD", "ml"),
    description="Метод расчета SL/TP: ml, atr, fixed"
  )
  SLTP_ML_FALLBACK_ENABLED: bool = Field(
    default=os.getenv("SLTP_ML_FALLBACK_ENABLED", "true").lower() == "true",
    description="Использовать ATR fallback при недоступности ML"
  )
  SLTP_MAX_STOP_LOSS_PERCENT: float = Field(
    default=float(os.getenv("SLTP_MAX_STOP_LOSS_PERCENT", "3.0")),
    description="Максимальный stop loss в % (default: 3%)"
  )
  SLTP_ATR_MULTIPLIER_SL: float = Field(
    default=float(os.getenv("SLTP_ATR_MULTIPLIER_SL", "2.0")),
    description="Множитель ATR для stop loss"
  )
  SLTP_ATR_MULTIPLIER_TP: float = Field(
    default=float(os.getenv("SLTP_ATR_MULTIPLIER_TP", "4.0")),
    description="Множитель ATR для take profit"
  )
  SLTP_MIN_RISK_REWARD: float = Field(
    default=float(os.getenv("SLTP_MIN_RISK_REWARD", "2.0")),
    description="Минимальное соотношение риск/прибыль"
  )

  # Корреляция позиций
  CORRELATION_CHECK_ENABLED: bool = Field(
    default=os.getenv("CORRELATION_CHECK_ENABLED", "true").lower() == "true",
    description="Проверять корреляцию между позициями"
  )
  CORRELATION_MAX_THRESHOLD: float = Field(
    default=float(os.getenv("CORRELATION_MAX_THRESHOLD", "0.7")),
    description="Максимальная допустимая корреляция"
  )
  CORRELATION_MAX_POSITIONS_PER_GROUP: int = Field(
    default=int(os.getenv("CORRELATION_MAX_POSITIONS_PER_GROUP", "1")),
    description="Максимум позиций в одной корреляционной группе"
  )
  CORRELATION_LOOKBACK_DAYS: int = Field(
    default=int(os.getenv("CORRELATION_LOOKBACK_DAYS", "30")),
    description="Период расчета корреляции (дни)"
  )

  # ===== ADVANCED CORRELATION SETTINGS =====
  # Методы расчета корреляции
  CORRELATION_USE_ADVANCED: bool = Field(
    default=os.getenv("CORRELATION_USE_ADVANCED", "true").lower() == "true",
    description="Использовать продвинутый анализ корреляций"
  )
  CORRELATION_USE_SPEARMAN: bool = Field(
    default=os.getenv("CORRELATION_USE_SPEARMAN", "true").lower() == "true",
    description="Использовать Spearman rank correlation"
  )
  CORRELATION_USE_DTW: bool = Field(
    default=os.getenv("CORRELATION_USE_DTW", "false").lower() == "true",
    description="Использовать Dynamic Time Warping (требует больше ресурсов)"
  )

  # Rolling windows
  CORRELATION_SHORT_WINDOW: int = Field(
    default=int(os.getenv("CORRELATION_SHORT_WINDOW", "7")),
    ge=2,
    le=30,
    description="Короткое окно для rolling correlation (дни)"
  )
  CORRELATION_MEDIUM_WINDOW: int = Field(
    default=int(os.getenv("CORRELATION_MEDIUM_WINDOW", "14")),
    ge=2,
    le=60,
    description="Среднее окно для rolling correlation (дни)"
  )
  CORRELATION_LONG_WINDOW: int = Field(
    default=int(os.getenv("CORRELATION_LONG_WINDOW", "30")),
    ge=2,
    le=90,
    description="Длинное окно для rolling correlation (дни)"
  )

  # Веса окон
  CORRELATION_SHORT_WEIGHT: float = Field(
    default=float(os.getenv("CORRELATION_SHORT_WEIGHT", "0.5")),
    ge=0.0,
    le=1.0,
    description="Вес короткого окна"
  )
  CORRELATION_MEDIUM_WEIGHT: float = Field(
    default=float(os.getenv("CORRELATION_MEDIUM_WEIGHT", "0.3")),
    ge=0.0,
    le=1.0,
    description="Вес среднего окна"
  )
  CORRELATION_LONG_WEIGHT: float = Field(
    default=float(os.getenv("CORRELATION_LONG_WEIGHT", "0.2")),
    ge=0.0,
    le=1.0,
    description="Вес длинного окна"
  )

  # Методы группировки
  CORRELATION_GROUPING_METHOD: str = Field(
    default=os.getenv("CORRELATION_GROUPING_METHOD", "ensemble"),
    description="Метод группировки: greedy, louvain, hierarchical, ensemble"
  )

  # DTW параметры
  CORRELATION_DTW_MAX_LAG_HOURS: int = Field(
    default=int(os.getenv("CORRELATION_DTW_MAX_LAG_HOURS", "24")),
    description="Максимальный лаг для DTW (часы)"
  )
  CORRELATION_DTW_WINDOW_HOURS: int = Field(
    default=int(os.getenv("CORRELATION_DTW_WINDOW_HOURS", "168")),
    description="Размер окна для DTW (часы, 168 = 7 дней)"
  )

  # Режимы корреляций
  CORRELATION_REGIME_DETECTION: bool = Field(
    default=os.getenv("CORRELATION_REGIME_DETECTION", "true").lower() == "true",
    description="Детектировать режим корреляций и адаптировать параметры"
  )
  CORRELATION_REGIME_LOW_THRESHOLD: float = Field(
    default=float(os.getenv("CORRELATION_REGIME_LOW_THRESHOLD", "0.4")),
    description="Порог для низких корреляций"
  )
  CORRELATION_REGIME_MODERATE_THRESHOLD: float = Field(
    default=float(os.getenv("CORRELATION_REGIME_MODERATE_THRESHOLD", "0.6")),
    description="Порог для умеренных корреляций"
  )
  CORRELATION_REGIME_HIGH_THRESHOLD: float = Field(
    default=float(os.getenv("CORRELATION_REGIME_HIGH_THRESHOLD", "0.75")),
    description="Порог для высоких корреляций"
  )
  CORRELATION_REGIME_CRISIS_THRESHOLD: float = Field(
    default=float(os.getenv("CORRELATION_REGIME_CRISIS_THRESHOLD", "0.85")),
    description="Порог для кризисных корреляций"
  )

  # Volatility clustering
  CORRELATION_VOLATILITY_CLUSTERING: bool = Field(
    default=os.getenv("CORRELATION_VOLATILITY_CLUSTERING", "true").lower() == "true",
    description="Группировать активы по волатильности"
  )
  CORRELATION_VOLATILITY_CLUSTERS: int = Field(
    default=int(os.getenv("CORRELATION_VOLATILITY_CLUSTERS", "3")),
    ge=2,
    le=10,
    description="Количество кластеров волатильности"
  )

  # Daily Loss Killer
  DAILY_LOSS_KILLER_ENABLED: bool = Field(
    default=os.getenv("DAILY_LOSS_KILLER_ENABLED", "true").lower() == "true",
    description="Автоматическое отключение при дневном убытке"
  )
  DAILY_LOSS_MAX_PERCENT: float = Field(
    default=float(os.getenv("DAILY_LOSS_MAX_PERCENT", "15.0")),
    description="Максимальный дневной убыток для emergency shutdown (%)"
  )
  DAILY_LOSS_WARNING_PERCENT: float = Field(
    default=float(os.getenv("DAILY_LOSS_WARNING_PERCENT", "10.0")),
    description="Процент убытка для предупреждения (%)"
  )
  DAILY_LOSS_CHECK_INTERVAL_SEC: int = Field(
    default=int(os.getenv("DAILY_LOSS_CHECK_INTERVAL_SEC", "60")),
    description="Интервал проверки дневного убытка (секунды)"
  )

  # Adaptive Risk per Trade
  RISK_PER_TRADE_MODE: str = Field(
    default=os.getenv("RISK_PER_TRADE_MODE", "adaptive"),
    description="Режим расчета риска: fixed, adaptive, kelly"
  )
  RISK_PER_TRADE_BASE_PERCENT: float = Field(
    default=float(os.getenv("RISK_PER_TRADE_BASE_PERCENT", "2.0")),
    description="Базовый риск на сделку (%)"
  )
  RISK_PER_TRADE_MAX_PERCENT: float = Field(
    default=float(os.getenv("RISK_PER_TRADE_MAX_PERCENT", "5.0")),
    description="Максимальный риск на сделку (%)"
  )
  RISK_KELLY_FRACTION: float = Field(
    default=float(os.getenv("RISK_KELLY_FRACTION", "0.25")),
    description="Kelly Criterion fraction (0.25 = 1/4 Kelly)"
  )
  RISK_VOLATILITY_SCALING: bool = Field(
    default=os.getenv("RISK_VOLATILITY_SCALING", "true").lower() == "true",
    description="Масштабировать риск на основе волатильности"
  )

  # Adaptive Risk - дополнительные параметры
  RISK_KELLY_MIN_TRADES: int = Field(
    default=int(os.getenv("RISK_KELLY_MIN_TRADES", "30")),
    description="Минимум трейдов для использования Kelly Criterion"
  )

  RISK_VOLATILITY_BASELINE: float = Field(
    default=float(os.getenv("RISK_VOLATILITY_BASELINE", "0.02")),
    description="Baseline дневной волатильности для нормализации (2%)"
  )

  RISK_WIN_RATE_SCALING: bool = Field(
    default=os.getenv("RISK_WIN_RATE_SCALING", "true").lower() == "true",
    description="Масштабировать риск на основе win rate"
  )

  RISK_WIN_RATE_BASELINE: float = Field(
    default=float(os.getenv("RISK_WIN_RATE_BASELINE", "0.55")),
    description="Baseline win rate для нормализации (55%)"
  )

  RISK_CORRELATION_PENALTY: bool = Field(
    default=os.getenv("RISK_CORRELATION_PENALTY", "true").lower() == "true",
    description="Применять штраф за корреляцию позиций"
  )

  # Reversal Detector
  REVERSAL_DETECTOR_ENABLED: bool = Field(
    default=os.getenv("REVERSAL_DETECTOR_ENABLED", "true").lower() == "true",
    description="Детектор разворота тренда"
  )
  REVERSAL_MIN_INDICATORS_CONFIRM: int = Field(
    default=int(os.getenv("REVERSAL_MIN_INDICATORS_CONFIRM", "3")),
    description="Минимум индикаторов для подтверждения разворота"
  )
  REVERSAL_COOLDOWN_SECONDS: int = Field(
    default=int(os.getenv("REVERSAL_COOLDOWN_SECONDS", "300")),
    description="Cooldown между проверками разворота (секунды)"
  )
  REVERSAL_AUTO_ACTION: bool = Field(
    default=os.getenv("REVERSAL_AUTO_ACTION", "false").lower() == "true",
    description="Автоматически действовать при развороте"
  )
  # ==================== POSITION MONITOR ====================
  POSITION_MONITOR_ENABLED: bool = True
  POSITION_MONITOR_INTERVAL: float = 2.0  # Секунды между проверками
  POSITION_MONITOR_REVERSAL_CHECK: bool = True
  POSITION_MONITOR_SLTP_CHECK: bool = True

  # Trailing Stop Manager
  TRAILING_STOP_ENABLED: bool = Field(
    default=os.getenv("TRAILING_STOP_ENABLED", "true").lower() == "true",
    description="Автоматический trailing stop"
  )
  TRAILING_STOP_ACTIVATION_PROFIT_PERCENT: float = Field(
    default=float(os.getenv("TRAILING_STOP_ACTIVATION_PROFIT_PERCENT", "1.5")),
    description="При какой прибыли активировать trailing (%)"
  )
  TRAILING_STOP_DISTANCE_PERCENT: float = Field(
    default=float(os.getenv("TRAILING_STOP_DISTANCE_PERCENT", "0.8")),
    description="Дистанция trailing stop от пика (%)"
  )
  TRAILING_STOP_UPDATE_INTERVAL_SEC: int = Field(
    default=int(os.getenv("TRAILING_STOP_UPDATE_INTERVAL_SEC", "5")),
    description="Интервал обновления trailing stop (секунды)"
  )

  # ML Integration для риск-менеджмента
  ML_RISK_INTEGRATION_ENABLED: bool = Field(
    default=os.getenv("ML_RISK_INTEGRATION_ENABLED", "true").lower() == "true",
    description="Использовать ML для корректировки риска"
  )

  # ML Data Collection
  ML_DATA_COLLECTION_ENABLED: bool = Field(
    default=os.getenv("ML_DATA_COLLECTION_ENABLED", "true").lower() == "true",
    description="Включить сбор данных для обучения ML модели (может влиять на производительность)"
  )

  ML_MIN_CONFIDENCE_THRESHOLD: float = Field(
    default=float(os.getenv("ML_MIN_CONFIDENCE_THRESHOLD", "0.70")),
    description="Минимальная уверенность ML для входа"
  )
  ML_REQUIRE_AGREEMENT: bool = Field(
    default=os.getenv("ML_REQUIRE_AGREEMENT", "true").lower() == "true",
    description="Требовать согласия ML с сигналом стратегии"
  )
  ML_POSITION_SIZING: bool = Field(
    default=os.getenv("ML_POSITION_SIZING", "true").lower() == "true",
    description="ML корректировка размера позиции"
  )
  ML_SLTP_CALCULATION: bool = Field(
    default=os.getenv("ML_SLTP_CALCULATION", "true").lower() == "true",
    description="ML расчет SL/TP уровней"
  )
  ML_MANIPULATION_CHECK: bool = Field(
    default=os.getenv("ML_MANIPULATION_CHECK", "true").lower() == "true",
    description="ML проверка на манипуляции"
  )
  ML_REGIME_CHECK: bool = Field(
    default=os.getenv("ML_REGIME_CHECK", "true").lower() == "true",
    description="ML определение режима рынка"
  )

  # Screener Settings (MEMORY FIX)
  SCREENER_MAX_PAIRS: int = Field(
    default=30,
    description="Maximum number of pairs to track in screener (memory optimization)"
  )
  SCREENER_MIN_VOLUME: float = Field(
    default=5_000_000.0,
    description="Minimum 24h volume in USDT to include pair"
  )
  SCREENER_CLEANUP_INTERVAL: int = Field(
    default=60,
    description="Cleanup interval in seconds (MEMORY FIX: 300 → 60)"
  )
  SCREENER_INACTIVE_TTL: int = Field(
    default=120,
    description="Inactive pair TTL in seconds before removal"
  )

  # Notification Settings (заглушки для будущего)
  NOTIFICATION_EMAIL_ENABLED: bool = Field(
    default=os.getenv("NOTIFICATION_EMAIL_ENABLED", "false").lower() == "true",
    description="Email уведомления"
  )
  NOTIFICATION_TELEGRAM_ENABLED: bool = Field(
    default=os.getenv("NOTIFICATION_TELEGRAM_ENABLED", "false").lower() == "true",
    description="Telegram уведомления"
  )

  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    case_sensitive=True,
    extra="allow"
  )

  @field_validator("SLTP_MAX_STOP_LOSS_PERCENT", "DAILY_LOSS_MAX_PERCENT", mode="before")
  @classmethod
  def validate_percent_positive(cls, v):
    """Валидация положительных процентов."""
    # ИСПРАВЛЕНО: Сначала конвертируем в float
    try:
      v_float = float(v)
    except (TypeError, ValueError):
      raise ValueError(f"Процент должен быть числом, получено: {v}")

    if v_float <= 0:
      raise ValueError("Процент должен быть положительным числом")

    return v_float

  @field_validator("SLTP_MIN_RISK_REWARD", mode="before")
  @classmethod
  def validate_risk_reward_ratio(cls, v):
    """Валидация минимального R/R."""
    # ИСПРАВЛЕНО: Сначала конвертируем в float
    try:
      v_float = float(v)
    except (TypeError, ValueError):
      raise ValueError(f"Risk/Reward должен быть числом, получено: {v}")

    if v_float < 1.0:
      raise ValueError("Минимальный R/R должен быть >= 1.0")

    return v_float

  @field_validator("CORRELATION_MAX_THRESHOLD", "ML_MIN_CONFIDENCE_THRESHOLD", mode="before")
  @classmethod
  def validate_threshold_range(cls, v):
    """Валидация порогов в диапазоне [0, 1]."""
    # ИСПРАВЛЕНО: Сначала конвертируем в float
    try:
      v_float = float(v)
    except (TypeError, ValueError):
      raise ValueError(f"Порог должен быть числом, получено: {v}")

    if not 0.0 <= v_float <= 1.0:
      raise ValueError("Порог должен быть в диапазоне [0.0, 1.0]")

    return v_float

  @field_validator("TRADING_PAIRS")
  def validate_trading_pairs(cls, v):
    """Валидация формата торговых пар."""
    if not v or v.strip() == "":
      raise ValueError("TRADING_PAIRS не может быть пустым")
    pairs = [pair.strip() for pair in v.split(",")]
    if len(pairs) == 0:
      raise ValueError("Необходимо указать хотя бы одну торговую пару")
    return v

  @field_validator("ORDERBOOK_DEPTH", mode="before")
  @classmethod
  def validate_orderbook_depth(cls, v):
    """Валидация глубины стакана."""
    # ИСПРАВЛЕНО: Сначала конвертируем в int
    try:
      v_int = int(v)
    except (TypeError, ValueError):
      raise ValueError(f"Глубина стакана должна быть целым числом, получено: {v}")

    valid_depths = [1, 50, 200, 500]
    if v_int not in valid_depths:
      raise ValueError(
        f"ORDERBOOK_DEPTH должен быть одним из: {valid_depths}. "
        f"Получено: {v_int}"
      )

    return v_int

  @field_validator("IMBALANCE_BUY_THRESHOLD", "IMBALANCE_SELL_THRESHOLD")
  def validate_imbalance_thresholds(cls, v):
    """Валидация порогов дисбаланса."""
    if not 0.0 <= v <= 1.0:
      raise ValueError("Пороги дисбаланса должны быть в диапазоне 0.0-1.0")
    return v

  @field_validator("SECRET_KEY")
  def validate_secret_key(cls, v):
    """Валидация секретного ключа."""
    if v == "your_secret_key_here_change_this_to_random_string":
      raise ValueError(
        "Необходимо изменить SECRET_KEY на случайную строку. "
        "Используйте: openssl rand -hex 32"
      )
    if len(v) < 32:
      raise ValueError("SECRET_KEY должен содержать минимум 32 символа")
    return v

  @field_validator("APP_PASSWORD")
  def validate_app_password(cls, v):
    """Валидация пароля приложения."""
    if v == "change_this_password":
      raise ValueError("Необходимо изменить APP_PASSWORD на безопасный пароль")
    if len(v) < 8:
      raise ValueError("APP_PASSWORD должен содержать минимум 8 символов")
    return v

  def get_trading_pairs_list(self) -> List[str]:
    """Возвращает список торговых пар."""
    return [pair.strip() for pair in self.TRADING_PAIRS.split(",")]

  def get_cors_origins_list(self) -> List[str]:
    """Возвращает список разрешенных CORS источников."""
    return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

  def get_bybit_credentials(self) -> tuple[str, str]:
    """Возвращает API ключи в зависимости от режима работы."""
    if self.BYBIT_MODE == "mainnet":
      if not self.BYBIT_MAINNET_API_KEY or not self.BYBIT_MAINNET_API_SECRET:
        raise ValueError(
          "Для режима mainnet необходимо указать BYBIT_MAINNET_API_KEY "
          "и BYBIT_MAINNET_API_SECRET"
        )
      return self.BYBIT_MAINNET_API_KEY, self.BYBIT_MAINNET_API_SECRET
    else:
      if not self.BYBIT_API_KEY or not self.BYBIT_API_SECRET:
        raise ValueError(
          "Для режима testnet необходимо указать BYBIT_API_KEY "
          "и BYBIT_API_SECRET"
        )
      return self.BYBIT_API_KEY, self.BYBIT_API_SECRET

  def __init__(self, **kwargs):
    """Инициализация с дополнительной валидацией."""
    super().__init__(**kwargs)
    self._validate_configuration()

  def _validate_configuration(self):
    """Дополнительная валидация конфигурации после загрузки."""

    # Проверка весов ML и Strategy (с проверкой наличия атрибутов)
    if hasattr(self, 'ML_WEIGHT') and hasattr(self, 'STRATEGY_WEIGHT'):
      weights_sum = self.ML_WEIGHT + self.STRATEGY_WEIGHT
      if not (0.99 <= weights_sum <= 1.01):
        error_msg = (
          f"❌ Сумма ML_WEIGHT ({self.ML_WEIGHT}) и STRATEGY_WEIGHT "
          f"({self.STRATEGY_WEIGHT}) должна быть равна 1.0, текущая: {weights_sum}"
        )
        print(error_msg)
        raise ValueError(
          f"ML_WEIGHT + STRATEGY_WEIGHT must equal 1.0, got {weights_sum}"
        )
    else:
      print("⚠️ ML_WEIGHT или STRATEGY_WEIGHT не заданы в .env, используются значения по умолчанию")

    # Проверка MIN_ORDER_SIZE vs MAX_POSITION_SIZE
    if hasattr(self, 'MIN_ORDER_SIZE_USDT') and hasattr(self, 'MAX_POSITION_SIZE_USDT'):
      if self.MIN_ORDER_SIZE_USDT > self.MAX_POSITION_SIZE_USDT:
        error_msg = (
          f"❌ MIN_ORDER_SIZE_USDT ({self.MIN_ORDER_SIZE_USDT}) "
          f"больше MAX_POSITION_SIZE_USDT ({self.MAX_POSITION_SIZE_USDT})"
        )
        print(error_msg)
        raise ValueError(
          "MIN_ORDER_SIZE_USDT must be <= MAX_POSITION_SIZE_USDT"
        )

    # Проверка MAX_POSITION_SIZE vs MAX_EXPOSURE
    if hasattr(self, 'MAX_POSITION_SIZE_USDT') and hasattr(self, 'MAX_EXPOSURE_USDT'):
      if self.MAX_POSITION_SIZE_USDT > self.MAX_EXPOSURE_USDT:
        print(
          f"⚠️ MAX_POSITION_SIZE_USDT ({self.MAX_POSITION_SIZE_USDT}) "
          f"больше MAX_EXPOSURE_USDT ({self.MAX_EXPOSURE_USDT})"
        )

    # Валидация ML Server URL
    if hasattr(self, 'ML_SERVER_URL'):
      if not self.ML_SERVER_URL.startswith(("http://", "https://")):
        error_msg = f"❌ ML_SERVER_URL должен начинаться с http:// или https://"
        print(error_msg)
        raise ValueError(f"Invalid ML_SERVER_URL: {self.ML_SERVER_URL}")

    print("✅ Конфигурация успешно валидирована")

    # Логируем критичные настройки ПОСЛЕ валидации
    # Импортируем logger ВНУТРИ метода, чтобы избежать circular import
    try:
      from backend.core.logger import get_logger
      logger = get_logger(__name__)
      self._log_critical_settings(logger)
    except ImportError:
      # Если logger ещё не доступен, просто выводим в консоль
      self._log_critical_settings_to_console()

  def _log_critical_settings(self, logger):
    """Логирование критичных настроек через logger."""
    logger.info("=" * 60)
    logger.info("🔧 КРИТИЧНЫЕ НАСТРОЙКИ БОТА:")
    logger.info(f"  • Mode: {getattr(self, 'BYBIT_MODE', 'N/A')}")
    logger.info(f"  • Trading Pairs: {getattr(self, 'TRADING_PAIRS', 'N/A')}")
    logger.info(f"  • Consensus Mode: {getattr(self, 'CONSENSUS_MODE', 'N/A')}")
    logger.info(f"  • Default Leverage: {getattr(self, 'DEFAULT_LEVERAGE', 'N/A')}x")
    logger.info(f"  • Min Order Size: {getattr(self, 'MIN_ORDER_SIZE_USDT', 'N/A')} USDT")
    logger.info(f"  • Max Position Size: {getattr(self, 'MAX_POSITION_SIZE_USDT', 'N/A')} USDT")
    logger.info(f"  • Max Exposure: {getattr(self, 'MAX_EXPOSURE_USDT', 'N/A')} USDT")
    logger.info(f"  • Max Open Positions: {getattr(self, 'MAX_OPEN_POSITIONS', 'N/A')}")
    logger.info(f"  • ML Server: {getattr(self, 'ML_SERVER_URL', 'N/A')}")
    logger.info(
      f"  • ML Weight: {getattr(self, 'ML_WEIGHT', 'N/A')} / Strategy Weight: {getattr(self, 'STRATEGY_WEIGHT', 'N/A')}")
    logger.info("=" * 60)

  def _log_critical_settings_to_console(self):
    """Логирование критичных настроек в консоль (fallback)."""
    print("=" * 60)

  def get_bybit_api_url(self) -> str:
    """
    Получение URL API Bybit в зависимости от режима.

    Returns:
        str: URL API
    """
    if self.BYBIT_MODE == "mainnet":
      return self.BYBIT_MAINNET_URL
    return self.BYBIT_TESTNET_URL

  def is_testnet(self) -> bool:
    """
    Проверка режима testnet.

    Returns:
        bool: True если testnet
    """
    return self.BYBIT_MODE == "testnet"

  def get_redis_url(self) -> str:
    """
    Получение Redis URL.

    Returns:
        str: Redis URL
    """
    if self.REDIS_URL:
      return self.REDIS_URL
    return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"



# Создание глобального экземпляра настроек
try:
  settings = Settings()
except Exception as e:
  print(f"❌ КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАГРУЗКЕ КОНФИГУРАЦИИ: {e}")
  print("Проверьте наличие файла .env и корректность всех параметров")
  import traceback

  traceback.print_exc()
  raise
  print("🔧 КРИТИЧНЫЕ НАСТРОЙКИ БОТА:")
  print(f"  • Mode: {getattr(self, 'BYBIT_MODE', 'N/A')}")
  print(f"  • Trading Pairs: {getattr(self, 'TRADING_PAIRS', 'N/A')}")
  print(f"  • Consensus Mode: {getattr(self, 'CONSENSUS_MODE', 'N/A')}")
  print(f"  • Default Leverage: {getattr(self, 'DEFAULT_LEVERAGE', 'N/A')}x")
  print(f"  • Min Order Size: {getattr(self, 'MIN_ORDER_SIZE_USDT', 'N/A')} USDT")
  print(f"  • Max Position Size: {getattr(self, 'MAX_POSITION_SIZE_USDT', 'N/A')} USDT")
  print(f"  • Max Exposure: {getattr(self, 'MAX_EXPOSURE_USDT', 'N/A')} USDT")
  print(f"  • Max Open Positions: {getattr(self, 'MAX_OPEN_POSITIONS', 'N/A')}")
  print(f"  • ML Server: {getattr(self, 'ML_SERVER_URL', 'N/A')}")
  print(
    f"  • ML Weight: {getattr(self, 'ML_WEIGHT', 'N/A')} / Strategy Weight: {getattr(self, 'STRATEGY_WEIGHT', 'N/A')}")
  print("=" * 60)

