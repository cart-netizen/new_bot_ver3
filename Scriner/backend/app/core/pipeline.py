# backend/app/core/pipeline.py

import asyncio
import json
import websockets
import threading

from .logger import log
from ..config import settings
from .bybit_client import BybitRESTClient
from .data_processor import ScreenerDataProcessor
from .websocket_manager import manager


class DataPipeline(threading.Thread):
  def __init__(self):
    super().__init__()
    self.daemon = True
    self.data_processor: ScreenerDataProcessor | None = None
    self.loop = asyncio.new_event_loop()

  async def _bybit_ws_listener_task(self):
    if not self.data_processor: return
    topics = [f"tickers.{symbol}" for symbol in self.data_processor.symbols]
    ws_url = settings.bybit_futures_ws_url
    while True:
      try:
        log.info(f"[Pipeline] Подключение к Bybit WebSocket v5...")
        async with websockets.connect(ws_url, open_timeout=20) as ws:
          await ws.send(json.dumps({"op": "subscribe", "args": topics}))
          log.success(f"[Pipeline] Успешно подписались на {len(topics)} тем.")
          while True:
            try:
              message = await asyncio.wait_for(ws.recv(), timeout=30.0)
              data = json.loads(message)
              if data.get("op") == "ping":
                await ws.send(json.dumps({"op": "pong", "req_id": data.get("req_id")}))
                continue
              if "topic" in data and "tickers" in data["topic"]:
                tick_info = data.get("data", {})
                if tick_info.get("symbol") and tick_info.get("lastPrice"):
                  tick_info["last_price"] = tick_info.pop("lastPrice")
                  self.data_processor.update_with_tick(tick_info)
            except asyncio.TimeoutError:
              await ws.send(json.dumps({"op": "ping"}))
      except Exception as e:
        log.error(f"[Pipeline] Ошибка в WebSocket: {e}. Переподключение через 5с...")
        await asyncio.sleep(5)

  async def _calculation_broadcast_task(self):
    if not self.data_processor:
      log.error("[Pipeline] Процессор данных не инициализирован для расчетов.")
      return

    log.info("[Pipeline] Цикл расчета и трансляции запускается.")
    iteration = 0

    while True:
      try:
        iteration += 1
        log.debug(f"[Pipeline] Итерация расчетов #{iteration}")

        self.data_processor.calculate_all_metrics()
        screener_data = self.data_processor.get_screener_data()

        log.debug(f"[Pipeline] Отправка данных {len(screener_data.get('pairs', []))} пар клиентам.")

        if manager.loop and manager.active_connections:
          asyncio.run_coroutine_threadsafe(manager.broadcast(screener_data), manager.loop)
        else:
          log.debug("[Pipeline] Нет активных соединений для отправки данных.")

      except Exception as e:
        log.error(f"[Pipeline] Ошибка в цикле расчета (итерация {iteration}): {e}", exc_info=True)

      await asyncio.sleep(settings.data_broadcast_interval_seconds)

  # async def _main_pipeline(self):
  #   log.info("[Pipeline] Получение списка торговых пар...")
  #   bybit_client = BybitRESTClient()
  #
  #   try:
  #     initial_result = await bybit_client.get_top_volume_pairs()  # Вернули исходный метод
  #
  #     if not initial_result or not initial_result.get("symbols"):
  #       log.critical("[Pipeline] Не удалось получить список пар. Поток завершает работу.")
  #       return
  #
  #     log.info(f"[Pipeline] Получено {len(initial_result['symbols'])} торговых пар.")
  #
  #     self.data_processor = ScreenerDataProcessor(
  #       symbols=initial_result["symbols"],
  #       initial_data=initial_result["initial_data"]
  #     )
  #
  #     log.info("[Pipeline] Запуск дочерних задач...")
  #     listener_task = self.loop.create_task(self._bybit_ws_listener_task())
  #     calculator_task = self.loop.create_task(self._calculation_broadcast_task())
  #
  #     await asyncio.gather(listener_task, calculator_task)
  #
  #   except Exception as e:
  #     log.error(f"[Pipeline] Критическая ошибка в главном конвейере: {e}", exc_info=True)

  async def _main_pipeline(self):
    log.info("[Pipeline] Получение списка торговых пар...")
    bybit_client = BybitRESTClient()

    try:
      initial_result = await bybit_client.get_top_volume_pairs()

      if not initial_result or not initial_result.get("symbols"):
        log.critical("[Pipeline] Не удалось получить список пар. Поток завершает работу.")
        return

      log.info(f"[Pipeline] Получено {len(initial_result['symbols'])} торговых пар.")

      # Загружаем исторические 5м свечи
      log.info("[Pipeline] Загрузка исторических 5м свечей...")
      historical_5m_data = await bybit_client.get_5m_klines_for_symbols(initial_result["symbols"])

      self.data_processor = ScreenerDataProcessor(
        symbols=initial_result["symbols"],
        initial_data=initial_result["initial_data"],
        historical_5m_data=historical_5m_data
      )

      log.info("[Pipeline] Запуск дочерних задач...")
      listener_task = self.loop.create_task(self._bybit_ws_listener_task())
      calculator_task = self.loop.create_task(self._calculation_broadcast_task())

      await asyncio.gather(listener_task, calculator_task)

    except Exception as e:
      log.error(f"[Pipeline] Критическая ошибка в главном конвейере: {e}", exc_info=True)

  def run(self):
    asyncio.set_event_loop(self.loop)
    self.loop.run_until_complete(self._main_pipeline())


data_pipeline = DataPipeline()
