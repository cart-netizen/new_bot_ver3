# backend/app/config.py

import os
from pydantic_settings import BaseSettings
from pydantic import Field

# Определяем путь к .env файлу относительно текущего файла
# Это делает запуск скрипта более предсказуемым из любой директории
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')


class Settings(BaseSettings):
  """
  Класс для хранения и валидации всех настроек приложения.
  Загружает переменные из .env файла.
  """
  # Bybit API
  bybit_api_url: str = Field(..., env="BYBIT_API_URL")
  bybit_futures_ws_url: str = Field(..., env="BYBIT_FUTURES_WS_URL")

  # Логика скринера
  min_daily_volume_usd: int = Field(1000000, env="MIN_DAILY_VOLUME_USD")
  top_n_pairs: int = Field(200, env="TOP_N_PAIRS")

  # Настройки приложения
  log_level: str = Field("INFO", env="LOG_LEVEL")
  data_broadcast_interval_seconds: float = Field(2.0, env="DATA_BROADCAST_INTERVAL_SECONDS")

  class Config:
    env_file = env_path
    env_file_encoding = 'utf-8'


# Создаем единственный экземпляр настроек, который будет использоваться во всем приложении
settings = Settings()