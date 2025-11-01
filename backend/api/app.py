"""
FastAPI приложение.
Основное приложение REST API для взаимодействия с фронтендом.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
  """
  Создание и настройка FastAPI приложения.

  Returns:
      FastAPI: Настроенное приложение
  """
  # Создаем приложение
  app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API для скальпингового торгового бота на Bybit",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
  )

  # Настраиваем CORS
  app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
  )

  # Обработчики событий приложения
  @app.on_event("startup")
  async def startup_event():
    """Событие при запуске приложения."""
    logger.info("=" * 80)
    logger.info(f"FastAPI приложение запускается: {settings.APP_NAME}")
    logger.info(f"Версия: {settings.APP_VERSION}")
    logger.info(f"Режим отладки: {settings.DEBUG}")
    logger.info(f"CORS источники: {settings.get_cors_origins_list()}")
    logger.info("=" * 80)

  @app.on_event("shutdown")
  async def shutdown_event():
    """Событие при остановке приложения."""
    logger.info("=" * 80)
    logger.info("FastAPI приложение останавливается")
    logger.info("=" * 80)

  # Глобальный обработчик исключений
  @app.exception_handler(Exception)
  async def global_exception_handler(request, exc):
    """
    Глобальный обработчик исключений.

    Args:
        request: HTTP запрос
        exc: Исключение

    Returns:
        JSONResponse: Ответ с ошибкой
    """
    logger.error(f"Необработанное исключение: {exc}", exc_info=True)
    return JSONResponse(
      status_code=500,
      content={
        "error": "Internal Server Error",
        "message": str(exc) if settings.DEBUG else "An error occurred",
      }
    )

  # Корневой эндпоинт
  @app.get("/")
  async def root():
    """Корневой эндпоинт."""
    return {
      "name": settings.APP_NAME,
      "version": settings.APP_VERSION,
      "status": "running",
      "docs": "/docs" if settings.DEBUG else None,
    }

  logger.info("FastAPI приложение создано и настроено")
  return app


# Создаем экземпляр приложения
app = create_app()