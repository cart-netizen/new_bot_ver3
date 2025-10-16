"""
Модуль конфигурации приложения.
Загружает настройки из переменных окружения и валидирует их.
"""

import os
from typing import List, Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator, field_validator
from dotenv import load_dotenv

from core.logger import get_logger

# from core.logger import get_logger

logger = get_logger(__name__)
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
  MAX_OPEN_POSITIONS: int = Field(default=5)
  MAX_EXPOSURE_USDT: float = Field(default=10000)
  MIN_ORDER_SIZE_USDT: float = Field(default=5)
  MAX_POSITION_SIZE_USDT: float = 1000.0
  IMBALANCE_THRESHOLD: float = 0.7

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

  @field_validator("MAX_LEVERAGE")
  @classmethod
  def validate_max_leverage(cls, v, info):
    """Проверка, что MAX_LEVERAGE >= DEFAULT_LEVERAGE"""
    default_leverage = info.data.get("DEFAULT_LEVERAGE", 10)
    if v < default_leverage:
      raise ValueError(
        f"MAX_LEVERAGE ({v}) должен быть >= DEFAULT_LEVERAGE ({default_leverage})"
      )
    return v

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

  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    case_sensitive=True,
    extra="ignore"
  )

  @field_validator("TRADING_PAIRS")
  def validate_trading_pairs(cls, v):
    """Валидация формата торговых пар."""
    if not v or v.strip() == "":
      raise ValueError("TRADING_PAIRS не может быть пустым")
    pairs = [pair.strip() for pair in v.split(",")]
    if len(pairs) == 0:
      raise ValueError("Необходимо указать хотя бы одну торговую пару")
    return v

  @field_validator("ORDERBOOK_DEPTH")
  def validate_orderbook_depth(cls, v):
    """Валидация глубины стакана."""
    allowed_depths = [1, 50, 200, 500]
    if v not in allowed_depths:
      raise ValueError(f"ORDERBOOK_DEPTH должен быть одним из {allowed_depths}")
    return v

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


    # Проверка весов ML и Strategy
    weights_sum = self.ML_WEIGHT + self.STRATEGY_WEIGHT
    if not (0.99 <= weights_sum <= 1.01):  # Допуск на погрешность округления
      logger.error(
        f"❌ Сумма ML_WEIGHT ({self.ML_WEIGHT}) и STRATEGY_WEIGHT "
        f"({self.STRATEGY_WEIGHT}) должна быть равна 1.0, текущая: {weights_sum}"
      )
      raise ValueError(
        f"ML_WEIGHT + STRATEGY_WEIGHT must equal 1.0, got {weights_sum}"
      )

    # Проверка MIN_ORDER_SIZE vs MAX_POSITION_SIZE
    if self.MIN_ORDER_SIZE_USDT > self.MAX_POSITION_SIZE_USDT:
      logger.error(
        f"❌ MIN_ORDER_SIZE_USDT ({self.MIN_ORDER_SIZE_USDT}) "
        f"больше MAX_POSITION_SIZE_USDT ({self.MAX_POSITION_SIZE_USDT})"
      )
      raise ValueError(
        "MIN_ORDER_SIZE_USDT must be <= MAX_POSITION_SIZE_USDT"
      )

    # Проверка MAX_POSITION_SIZE vs MAX_EXPOSURE
    if self.MAX_POSITION_SIZE_USDT > self.MAX_EXPOSURE_USDT:
      logger.warning(
        f"⚠️ MAX_POSITION_SIZE_USDT ({self.MAX_POSITION_SIZE_USDT}) "
        f"больше MAX_EXPOSURE_USDT ({self.MAX_EXPOSURE_USDT})"
      )

    # Валидация ML Server URL
    if not self.ML_SERVER_URL.startswith(("http://", "https://")):
      logger.error(
        f"❌ ML_SERVER_URL должен начинаться с http:// или https://"
      )
      raise ValueError(
        f"Invalid ML_SERVER_URL: {self.ML_SERVER_URL}"
      )

    logger.info("✅ Конфигурация успешно валидирована")
    self._log_critical_settings()

  def _log_critical_settings(self):
    """Логирование критичных настроек при запуске."""
    logger.info("=" * 60)
    logger.info("🔧 КРИТИЧНЫЕ НАСТРОЙКИ БОТА:")
    logger.info(f"  • Mode: {self.BYBIT_MODE}")
    logger.info(f"  • Trading Pairs: {self.TRADING_PAIRS}")
    logger.info(f"  • Consensus Mode: {self.CONSENSUS_MODE}")
    logger.info(f"  • Default Leverage: {self.DEFAULT_LEVERAGE}x")
    logger.info(f"  • Min Order Size: {self.MIN_ORDER_SIZE_USDT} USDT")
    logger.info(f"  • Max Position Size: {self.MAX_POSITION_SIZE_USDT} USDT")
    logger.info(f"  • Max Exposure: {self.MAX_EXPOSURE_USDT} USDT")
    logger.info(f"  • Max Open Positions: {self.MAX_OPEN_POSITIONS}")
    logger.info(f"  • ML Server: {self.ML_SERVER_URL}")
    logger.info(f"  • ML Weight: {self.ML_WEIGHT} / Strategy Weight: {self.STRATEGY_WEIGHT}")
    logger.info("=" * 60)


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
  print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить конфигурацию: {e}")
  print("Проверьте наличие файла .env и корректность всех параметров")
  raise

