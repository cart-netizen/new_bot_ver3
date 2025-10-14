# # backend/app/main.py


import asyncio
import os
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse  # Изменено для более надежной отдачи файла

# Используем относительные импорты
from .api.endpoints import router as api_router
from .core.logger import setup_logger, log
from .core.websocket_manager import manager
from .core.pipeline import data_pipeline  # Импортируем наш конвейер

# Инициализация FastAPI приложения
app = FastAPI(title="Bybit Crypto Screener Backend")

# --- Обслуживание фронтенда ---
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'frontend')
html_file_path = os.path.join(frontend_dir, 'frontend.html')

# Считываем содержимое HTML файла один раз при запуске для стабильности
html_content = ""
try:
  with open(html_file_path, "r", encoding="utf-8") as f:
    html_content = f.read()
except FileNotFoundError:
  log.error(f"Файл фронтенда не найден по пути: {html_file_path}")
  html_content = "<h1>Ошибка: Файл frontend.html не найден.</h1>"


@app.get("/", response_class=HTMLResponse)
async def read_root() -> Response:
  """ Отдает главную HTML страницу фронтенда. """
  return HTMLResponse(content=html_content)


# --- События жизненного цикла приложения ---

@app.on_event("startup")
async def startup_event():
  """
  Выполняется при старте приложения.
  """
  setup_logger()
  log.info("--- Запуск бэкенд-сервиса (Архитектура с потоком) ---")

  # Сохраняем основной цикл событий в менеджере для потокобезопасной отправки
  manager.set_loop(asyncio.get_running_loop())

  # Запускаем весь конвейер данных в отдельном фоновом потоке
  data_pipeline.start()

  log.success("Веб-сервер запущен, фоновый поток для данных стартовал.")


# Подключаем роутер с нашим WebSocket эндпоинтом
app.include_router(api_router)
