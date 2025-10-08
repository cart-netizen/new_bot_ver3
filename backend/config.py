"""
Модуль конфигурации приложения.
Загружает настройки из переменных окружения и валидирует их.
"""

import os
from typing import List, Literal
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()


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

  # ===== НАСТРОЙКИ API СЕРВЕРА =====
  API_HOST: str = Field(default="0.0.0.0")
  API_PORT: int = Field(default=8000)
  CORS_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:5173")

  # ===== НАСТРОЙКИ WEBSOCKET =====
  WS_RECONNECT_TIMEOUT: int = Field(default=5)
  WS_MAX_RECONNECT_ATTEMPTS: int = Field(default=10)
  WS_PING_INTERVAL: int = Field(default=20)

  @validator("TRADING_PAIRS")
  def validate_trading_pairs(cls, v):
    """Валидация формата торговых пар."""
    if not v or v.strip() == "":
      raise ValueError("TRADING_PAIRS не может быть пустым")
    pairs = [pair.strip() for pair in v.split(",")]
    if len(pairs) == 0:
      raise ValueError("Необходимо указать хотя бы одну торговую пару")
    return v

  @validator("ORDERBOOK_DEPTH")
  def validate_orderbook_depth(cls, v):
    """Валидация глубины стакана."""
    allowed_depths = [1, 50, 200, 500]
    if v not in allowed_depths:
      raise ValueError(f"ORDERBOOK_DEPTH должен быть одним из {allowed_depths}")
    return v

  @validator("IMBALANCE_BUY_THRESHOLD", "IMBALANCE_SELL_THRESHOLD")
  def validate_imbalance_thresholds(cls, v):
    """Валидация порогов дисбаланса."""
    if not 0.0 <= v <= 1.0:
      raise ValueError("Пороги дисбаланса должны быть в диапазоне 0.0-1.0")
    return v

  @validator("SECRET_KEY")
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

  @validator("APP_PASSWORD")
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

  class Config:
    env_file = ".env"
    env_file_encoding = "utf-8"
    case_sensitive = True


# Создание глобального экземпляра настроек
try:
  settings = Settings()
except Exception as e:
  print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить конфигурацию: {e}")
  print("Проверьте наличие файла .env и корректность всех параметров")
  raise