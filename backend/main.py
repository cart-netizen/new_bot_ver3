"""
Главный файл приложения.
Точка входа и контроллер торгового бота.
"""

import asyncio
import signal
from typing import Dict, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import WebSocket, WebSocketDisconnect

from backend.config import settings
from core.logger import setup_logging, get_logger
from core.exceptions import log_exception
from exchange.rest_client import rest_client
from exchange.websocket_manager import BybitWebSocketManager
from strategy.orderbook_manager import OrderBookManager
from strategy.analyzer import MarketAnalyzer
from strategy.strategy_engine import StrategyEngine
from strategy.risk_manager import RiskManager
from execution.execution_manager import ExecutionManager
from utils.constants import BotStatus
from api.websocket import manager as ws_manager, handle_websocket_messages

# Настройка логирования
setup_logging()
logger = get_logger(__name__)


class BotController:
  """Главный контроллер торгового бота."""

  def __init__(self):
    """Инициализация контроллера."""
    self.status = BotStatus.STOPPED
    self.symbols = settings.get_trading_pairs_list()

    # Компоненты
    self.websocket_manager: Optional[BybitWebSocketManager] = None
    self.orderbook_managers: Dict[str, OrderBookManager] = {}
    self.market_analyzer: Optional[MarketAnalyzer] = None
    self.strategy_engine: Optional[StrategyEngine] = None
    self.risk_manager: Optional[RiskManager] = None
    self.execution_manager: Optional[ExecutionManager] = None

    # Задачи
    self.websocket_task: Optional[asyncio.Task] = None
    self.analysis_task: Optional[asyncio.Task] = None

    logger.info("Инициализирован контроллер бота")

  async def initialize(self):
    """Инициализация всех компонентов бота."""
    try:
      logger.info("=" * 80)
      logger.info("Инициализация компонентов бота")
      logger.info("=" * 80)

      # Инициализируем REST клиент
      await rest_client.initialize()
      logger.info("✓ REST клиент инициализирован")

      # Проверяем подключение к бирже
      server_time = await rest_client.get_server_time()
      logger.info(f"✓ Подключение к Bybit успешно. Серверное время: {server_time}")

      # Создаем менеджеры стакана для каждой пары
      for symbol in self.symbols:
        self.orderbook_managers[symbol] = OrderBookManager(symbol)
      logger.info(f"✓ Создано {len(self.orderbook_managers)} менеджеров стакана")

      # Инициализируем анализатор рынка
      self.market_analyzer = MarketAnalyzer()
      for symbol in self.symbols:
        self.market_analyzer.add_symbol(symbol)
      logger.info("✓ Анализатор рынка инициализирован")

      # Инициализируем стратегию
      self.strategy_engine = StrategyEngine()
      logger.info("✓ Торговая стратегия инициализирована")

      # Инициализируем риск-менеджер
      self.risk_manager = RiskManager()
      logger.info("✓ Риск-менеджер инициализирован")

      # Инициализируем менеджер исполнения
      self.execution_manager = ExecutionManager(self.risk_manager)
      logger.info("✓ Менеджер исполнения инициализирован")

      # Создаем WebSocket менеджер
      self.websocket_manager = BybitWebSocketManager(
        symbols=self.symbols,
        on_message=self._handle_orderbook_message
      )
      logger.info("✓ WebSocket менеджер создан")

      logger.info("=" * 80)
      logger.info("Все компоненты успешно инициализированы")
      logger.info("=" * 80)

    except Exception as e:
      logger.error(f"Ошибка инициализации бота: {e}")
      log_exception(logger, e, "Инициализация бота")
      raise

  async def start(self):
    """Запуск бота."""
    if self.status == BotStatus.RUNNING:
      logger.warning("Бот уже запущен")
      return

    try:
      self.status = BotStatus.STARTING
      logger.info("=" * 80)
      logger.info("ЗАПУСК ТОРГОВОГО БОТА")
      logger.info("=" * 80)

      # Запускаем менеджер исполнения
      await self.execution_manager.start()
      logger.info("✓ Менеджер исполнения запущен")

      # Запускаем WebSocket соединения
      self.websocket_task = asyncio.create_task(
        self.websocket_manager.start()
      )
      logger.info("✓ WebSocket соединения запущены")

      # Запускаем задачу анализа
      self.analysis_task = asyncio.create_task(
        self._analysis_loop()
      )
      logger.info("✓ Цикл анализа запущен")

      self.status = BotStatus.RUNNING
      logger.info("=" * 80)
      logger.info("БОТ УСПЕШНО ЗАПУЩЕН")
      logger.info("=" * 80)

      # Уведомляем фронтенд
      from api.websocket import broadcast_bot_status
      await broadcast_bot_status("running", {
        "symbols": self.symbols,
        "message": "Бот успешно запущен"
      })

    except Exception as e:
      self.status = BotStatus.ERROR
      logger.error(f"Ошибка запуска бота: {e}")
      log_exception(logger, e, "Запуск бота")
      raise

  async def stop(self):
    """Остановка бота."""
    if self.status == BotStatus.STOPPED:
      logger.warning("Бот уже остановлен")
      return

    try:
      self.status = BotStatus.STOPPING
      logger.info("=" * 80)
      logger.info("ОСТАНОВКА ТОРГОВОГО БОТА")
      logger.info("=" * 80)

      # Останавливаем задачу анализа
      if self.analysis_task and not self.analysis_task.done():
        self.analysis_task.cancel()
        try:
          await self.analysis_task
        except asyncio.CancelledError:
          pass
      logger.info("✓ Цикл анализа остановлен")

      # Останавливаем WebSocket соединения
      if self.websocket_manager:
        await self.websocket_manager.stop()
      logger.info("✓ WebSocket соединения закрыты")

      # Останавливаем менеджер исполнения
      if self.execution_manager:
        await self.execution_manager.stop()
      logger.info("✓ Менеджер исполнения остановлен")

      self.status = BotStatus.STOPPED
      logger.info("=" * 80)
      logger.info("БОТ УСПЕШНО ОСТАНОВЛЕН")
      logger.info("=" * 80)

      # Уведомляем фронтенд
      from api.websocket import broadcast_bot_status
      await broadcast_bot_status("stopped", {
        "message": "Бот остановлен"
      })

    except Exception as e:
      self.status = BotStatus.ERROR
      logger.error(f"Ошибка остановки бота: {e}")
      log_exception(logger, e, "Остановка бота")
      raise

  async def _handle_orderbook_message(self, data: dict):
    """
    Обработка сообщения от WebSocket стакана.

    Args:
        data: Данные от WebSocket
    """
    try:
      topic = data.get("topic", "")
      message_type = data.get("type", "")
      message_data = data.get("data", {})

      # Извлекаем символ из топика
      if "orderbook" in topic:
        parts = topic.split(".")
        if len(parts) >= 3:
          symbol = parts[2]

          if symbol not in self.orderbook_managers:
            logger.warning(f"Получены данные для неизвестного символа: {symbol}")
            return

          manager = self.orderbook_managers[symbol]

          # Обрабатываем snapshot или delta
          if message_type == "snapshot":
            manager.apply_snapshot(message_data)
          elif message_type == "delta":
            manager.apply_delta(message_data)

    except Exception as e:
      logger.error(f"Ошибка обработки сообщения стакана: {e}")
      log_exception(logger, e, "Обработка сообщения стакана")

  async def _analysis_loop(self):
    """Цикл анализа и генерации сигналов."""
    logger.info("Запущен цикл анализа")

    while self.status == BotStatus.RUNNING:
      try:
        # Ждем пока все WebSocket соединения установятся
        if not self.websocket_manager.is_all_connected():
          await asyncio.sleep(1)
          continue

        # Анализируем каждую пару
        for symbol in self.symbols:
          try:
            manager = self.orderbook_managers[symbol]

            # Пропускаем если нет данных
            if not manager.snapshot_received:
              continue

            # Анализируем стакан
            metrics = self.market_analyzer.analyze_symbol(symbol, manager)

            # Генерируем торговый сигнал
            signal = self.strategy_engine.analyze_and_generate_signal(
              symbol,
              metrics
            )

            # Если есть сигнал - отправляем на исполнение
            if signal:
              await self.execution_manager.submit_signal(signal)

              # Уведомляем фронтенд
              from api.websocket import broadcast_signal
              await broadcast_signal(signal.to_dict())

          except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")

        # Пауза между циклами анализа
        await asyncio.sleep(0.5)  # 500ms

      except asyncio.CancelledError:
        logger.info("Цикл анализа отменен")
        break
      except Exception as e:
        logger.error(f"Ошибка в цикле анализа: {e}")
        log_exception(logger, e, "Цикл анализа")
        await asyncio.sleep(1)

  def get_status(self) -> dict:
    """
    Получение статуса бота.

    Returns:
        dict: Статус бота
    """
    ws_status = {}
    if self.websocket_manager:
      ws_status = self.websocket_manager.get_connection_statuses()

    return {
      "status": self.status.value,
      "symbols": self.symbols,
      "websocket_connections": ws_status,
      "orderbook_managers": {
        symbol: manager.get_stats()
        for symbol, manager in self.orderbook_managers.items()
      },
      "execution_stats": (
        self.execution_manager.get_statistics()
        if self.execution_manager else {}
      ),
    }


# Глобальный контроллер бота
bot_controller: Optional[BotController] = None


@asynccontextmanager
async def lifespan(app):
  """
  Управление жизненным циклом приложения.

  Args:
      app: FastAPI приложение
  """
  global bot_controller

  # Startup
  logger.info("Запуск приложения")
  try:
    # Создаем и инициализируем контроллер
    bot_controller = BotController()
    await bot_controller.initialize()

    logger.info("Приложение готово к работе")
    yield

  except Exception as e:
    logger.error(f"Критическая ошибка при запуске: {e}")
    log_exception(logger, e, "Запуск приложения")
    raise

  finally:
    # Shutdown
    logger.info("Остановка приложения")

    if bot_controller:
      if bot_controller.status == BotStatus.RUNNING:
        await bot_controller.stop()

      # Закрываем REST клиент
      await rest_client.close()

    logger.info("Приложение остановлено")


# Импортируем FastAPI приложение и добавляем lifespan
from api.app import app

app.router.lifespan_context = lifespan

# Регистрируем роутеры
from api.routes import auth_router, bot_router, data_router, trading_router

app.include_router(auth_router)
app.include_router(bot_router)
app.include_router(data_router)
app.include_router(trading_router)


# WebSocket эндпоинт
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
  """
  WebSocket эндпоинт для фронтенда.

  Args:
      websocket: WebSocket соединение
  """
  await ws_manager.connect(websocket)

  try:
    await handle_websocket_messages(websocket)
  except WebSocketDisconnect:
    logger.info("WebSocket клиент отключен")
  except Exception as e:
    logger.error(f"Ошибка WebSocket: {e}")
  finally:
    ws_manager.disconnect(websocket)


def handle_shutdown_signal(signum, frame):
  """
  Обработчик сигналов завершения.

  Args:
      signum: Номер сигнала
      frame: Фрейм
  """
  logger.info(f"Получен сигнал завершения: {signum}")
  # Uvicorn обработает остановку автоматически


# Регистрируем обработчики сигналов
signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)

if __name__ == "__main__":
  """Точка входа при запуске напрямую."""

  logger.info("=" * 80)
  logger.info(f"Запуск {settings.APP_NAME} v{settings.APP_VERSION}")
  logger.info(f"Режим: {settings.BYBIT_MODE.upper()}")
  logger.info(f"Хост: {settings.API_HOST}:{settings.API_PORT}")
  logger.info("=" * 80)

  # Запускаем Uvicorn сервер
  uvicorn.run(
    "main:app",
    host=settings.API_HOST,
    port=settings.API_PORT,
    reload=settings.DEBUG,
    log_level=settings.LOG_LEVEL.lower(),
  )