"""
Модуль настройки логирования для всего приложения.
Обеспечивает структурированное логирование с различными уровнями детализации.
"""

import logging
import sys
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

    # ===== ФАЙЛОВЫЙ ОБРАБОТЧИК =====
    if log_to_file:
        # Создаем директорию для логов
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # Имя файла с текущей датой
        log_filename = f"bot_{datetime.now().strftime('%Y%m%d')}.log"
        log_filepath = log_path / log_filename

        # Обработчик для общего лога
        file_handler = logging.FileHandler(
            log_filepath,
            mode='a',
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(FileFormatter())
        root_logger.addHandler(file_handler)

        # Обработчик для лога ошибок
        error_log_filename = f"bot_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_log_filepath = log_path / error_log_filename

        error_file_handler = logging.FileHandler(
            error_log_filepath,
            mode='a',
            encoding='utf-8'
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(FileFormatter())
        root_logger.addHandler(error_file_handler)

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


def get_logger(name: str) -> logging.Logger:
    """
    Получение логгера для модуля.

    Args:
        name: Имя модуля (обычно __name__)

    Returns:
        logging.Logger: Настроенный логгер
    """
    return logging.getLogger(name)