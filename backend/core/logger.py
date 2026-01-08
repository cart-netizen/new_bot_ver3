"""
Модуль настройки логирования для всего приложения.
Обеспечивает структурированное логирование с различными уровнями детализации.
"""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from backend.config import settings


class ColoredFormatter(logging.Formatter):
    """Форматтер с цветным выводом для консоли."""

    # ANSI цветовые коды
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record: logging.LogRecord) -> str:
        """Форматирование записи лога с цветами."""
        # Получаем цвет для уровня логирования
        color = self.COLORS.get(record.levelname, self.RESET)

        # Форматируем timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Форматируем имя модуля
        module_name = record.name

        # Создаем цветное сообщение
        log_message = (
            f"{color}{self.BOLD}[{timestamp}]{self.RESET} "
            f"{color}[{record.levelname:8}]{self.RESET} "
            f"[{module_name:25}] "
            f"{record.getMessage()}"
        )

        # Добавляем информацию об исключении, если есть
        if record.exc_info:
            log_message += '\n' + self.formatException(record.exc_info)

        return log_message


class FileFormatter(logging.Formatter):
    """Форматтер для записи в файл без цветов."""

    def format(self, record: logging.LogRecord) -> str:
        """Форматирование записи лога для файла."""
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        module_name = record.name

        log_message = (
            f"[{timestamp}] "
            f"[{record.levelname:8}] "
            f"[{module_name:25}] "
            f"[{record.funcName}:{record.lineno}] "
            f"{record.getMessage()}"
        )

        if record.exc_info:
            log_message += '\n' + self.formatException(record.exc_info)

        return log_message


def setup_logging(
    log_level: Optional[str] = None,
    log_to_file: bool = True,
    log_dir: str = "logs"
) -> None:
    """
    Настройка системы логирования для всего приложения.

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Флаг записи логов в файл
        log_dir: Директория для сохранения файлов логов
    """
    # Определяем уровень логирования
    if log_level is None:
        log_level = settings.LOG_LEVEL

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Получаем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Очищаем существующие обработчики
    root_logger.handlers.clear()

    # ===== КОНСОЛЬНЫЙ ОБРАБОТЧИК =====
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(console_handler)

    # ===== ФАЙЛОВЫЙ ОБРАБОТЧИК С РОТАЦИЕЙ ПО ДНЯМ =====
    if log_to_file:
        # Создаем директорию для логов
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # Базовое имя файла (без даты - TimedRotatingFileHandler добавит сам)
        log_filepath = log_path / "bot.log"

        # Обработчик для общего лога с ротацией в полночь
        # when='midnight' - ротация в полночь каждого дня
        # backupCount=30 - хранить логи за 30 дней
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_filepath,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        # Формат имени архивных файлов: bot.log.2025-12-03
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(FileFormatter())
        root_logger.addHandler(file_handler)

        # Обработчик для лога ошибок с ротацией
        error_log_filepath = log_path / "bot_errors.log"

        error_file_handler = logging.handlers.TimedRotatingFileHandler(
            error_log_filepath,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        error_file_handler.suffix = "%Y-%m-%d"
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(FileFormatter())
        root_logger.addHandler(error_file_handler)

        # ===== TRADES LOG - ДЕТАЛЬНЫЕ ОТЧЕТЫ О СДЕЛКАХ =====
        _setup_trades_logger(log_path)

    # Отключаем избыточное логирование от сторонних библиотек
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Логируем успешную инициализацию
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Система логирования инициализирована: уровень={log_level.upper()}")
    logger.info(f"Режим работы: {'ОТЛАДКА' if settings.DEBUG else 'PRODUCTION'}")
    logger.info(f"Bybit режим: {settings.BYBIT_MODE.upper()}")
    logger.info(f"Торговые пары: {settings.get_trading_pairs_list()}")
    logger.info("=" * 80)


class TradesFileFormatter(logging.Formatter):
    """Форматтер для trades.log - чистый формат без лишней информации."""

    def format(self, record: logging.LogRecord) -> str:
        """Форматирование записи для файла сделок."""
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return f"[{timestamp}]\n{record.getMessage()}\n"


# Глобальная ссылка на trades logger
_trades_logger: Optional[logging.Logger] = None


def _setup_trades_logger(log_path: Path) -> None:
    """
    Настройка отдельного логгера для детальных отчетов о сделках.

    Записывает в trades.log только реализованные ордера с полным анализом:
    - Индикаторы каждой стратегии
    - ML прогнозы от всех моделей
    - MTF анализ
    - Причины принятия решения
    """
    global _trades_logger

    trades_logger = logging.getLogger("trades")
    trades_logger.setLevel(logging.INFO)
    trades_logger.propagate = False  # Не дублировать в основной лог

    # Очищаем существующие обработчики
    trades_logger.handlers.clear()

    # Файловый обработчик с ротацией по дням
    trades_filepath = log_path / "trades.log"

    trades_handler = logging.handlers.TimedRotatingFileHandler(
        trades_filepath,
        when='midnight',
        interval=1,
        backupCount=90,  # Храним 90 дней для анализа
        encoding='utf-8'
    )
    trades_handler.suffix = "%Y-%m-%d"
    trades_handler.setLevel(logging.INFO)
    trades_handler.setFormatter(TradesFileFormatter())
    trades_logger.addHandler(trades_handler)

    _trades_logger = trades_logger


def get_trades_logger() -> logging.Logger:
    """
    Получение логгера для записи детальных отчетов о сделках.

    Returns:
        logging.Logger: Логгер для trades.log
    """
    global _trades_logger
    if _trades_logger is None:
        # Fallback - создаем базовый логгер если setup_logging еще не вызван
        _trades_logger = logging.getLogger("trades")
        _trades_logger.setLevel(logging.INFO)
    return _trades_logger


def get_logger(name: str) -> logging.Logger:
    """
    Получение логгера для модуля.

    Args:
        name: Имя модуля (обычно __name__)

    Returns:
        logging.Logger: Настроенный логгер
    """
    return logging.getLogger(name)