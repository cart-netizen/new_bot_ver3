"""
API маршруты.
Определение всех REST API эндпоинтов для взаимодействия с фронтендом.
"""
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from core.logger import get_logger
from core.auth import (
  AuthService,
  require_auth
)
from core.exceptions import AuthenticationError
from models.user import LoginRequest, LoginResponse, ChangePasswordRequest
from config import settings
from utils.balance_tracker import balance_tracker

logger = get_logger(__name__)

# Создаем роутеры
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
bot_router = APIRouter(prefix="/bot", tags=["Bot Control"])
data_router = APIRouter(prefix="/data", tags=["Market Data"])
trading_router = APIRouter(prefix="/trading", tags=["Trading"])


# ===== МОДЕЛИ ОТВЕТОВ =====

class StatusResponse(BaseModel):
  """Модель ответа статуса."""
  status: str
  message: Optional[str] = None


class ConfigResponse(BaseModel):
  """Модель ответа конфигурации."""
  trading_pairs: List[str]
  bybit_mode: str
  orderbook_depth: int
  imbalance_buy_threshold: float
  imbalance_sell_threshold: float
  max_open_positions: int
  max_exposure_usdt: float


# ===== АУТЕНТИФИКАЦИЯ =====

@auth_router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
  """
  Вход в приложение.

  Args:
      request: Данные для входа

  Returns:
      LoginResponse: Токен доступа
  """
  try:
    logger.info("Попытка входа в систему")
    token_data = AuthService.authenticate(request.password)
    logger.info("Успешный вход в систему")
    return LoginResponse(**token_data)
  except AuthenticationError as e:
    logger.warning(f"Неудачная попытка входа: {e}")
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail=str(e)
    )


@auth_router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: dict = Depends(require_auth)
):
  """
  Изменение пароля.

  Args:
      request: Данные для смены пароля
      current_user: Текущий пользователь

  Returns:
      StatusResponse: Статус операции
  """
  try:
    AuthService.change_password(request.old_password, request.new_password)
    logger.info("Пароль успешно изменен")
    return StatusResponse(
      status="success",
      message="Пароль успешно изменен"
    )
  except AuthenticationError as e:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail=str(e)
    )
  except ValueError as e:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail=str(e)
    )


@auth_router.get("/verify")
async def verify_token(current_user: dict = Depends(require_auth)):
  """
  Проверка валидности токена.

  Args:
      current_user: Текущий пользователь

  Returns:
      dict: Информация о пользователе
  """
  return {
    "valid": True,
    "user": current_user["sub"]
  }


# ===== УПРАВЛЕНИЕ БОТОМ =====

@bot_router.get("/status")
async def get_bot_status(current_user: dict = Depends(require_auth)):
  """
  Получение статуса бота.

  Args:
      current_user: Текущий пользователь

  Returns:
      dict: Статус бота
  """
  # Импортируем здесь, чтобы избежать циклической зависимости
  from main import bot_controller

  if bot_controller:
    return bot_controller.get_status()

  return {
    "status": "stopped",
    "message": "Бот не инициализирован"
  }


@bot_router.post("/start")
async def start_bot(current_user: dict = Depends(require_auth)):
  """
  Запуск бота.

  Args:
      current_user: Текущий пользователь

  Returns:
      StatusResponse: Статус операции
  """
  from main import bot_controller

  if not bot_controller:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Бот не инициализирован"
    )

  try:
    await bot_controller.start()
    logger.info("Бот запущен через API")
    return StatusResponse(
      status="success",
      message="Бот успешно запущен"
    )
  except Exception as e:
    logger.error(f"Ошибка запуска бота: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )


@bot_router.post("/stop")
async def stop_bot(current_user: dict = Depends(require_auth)):
  """
  Остановка бота.

  Args:
      current_user: Текущий пользователь

  Returns:
      StatusResponse: Статус операции
  """
  from main import bot_controller

  if not bot_controller:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Бот не инициализирован"
    )

  try:
    await bot_controller.stop()
    logger.info("Бот остановлен через API")
    return StatusResponse(
      status="success",
      message="Бот успешно остановлен"
    )
  except Exception as e:
    logger.error(f"Ошибка остановки бота: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )


@bot_router.get("/config", response_model=ConfigResponse)
async def get_config(current_user: dict = Depends(require_auth)):
  """
  Получение конфигурации бота.

  Args:
      current_user: Текущий пользователь

  Returns:
      ConfigResponse: Конфигурация
  """
  return ConfigResponse(
    trading_pairs=settings.get_trading_pairs_list(),
    bybit_mode=settings.BYBIT_MODE,
    orderbook_depth=settings.ORDERBOOK_DEPTH,
    imbalance_buy_threshold=settings.IMBALANCE_BUY_THRESHOLD,
    imbalance_sell_threshold=settings.IMBALANCE_SELL_THRESHOLD,
    max_open_positions=settings.MAX_OPEN_POSITIONS,
    max_exposure_usdt=settings.MAX_EXPOSURE_USDT
  )


# ===== РЫНОЧНЫЕ ДАННЫЕ =====

@data_router.get("/pairs")
async def get_pairs(current_user: dict = Depends(require_auth)):
  """
  Получение списка торговых пар.

  Args:
      current_user: Текущий пользователь

  Returns:
      dict: Список пар
  """
  return {
    "pairs": settings.get_trading_pairs_list(),
    "count": len(settings.get_trading_pairs_list())
  }


@data_router.get("/orderbook/{symbol}")
async def get_orderbook(
    symbol: str,
    current_user: dict = Depends(require_auth)
):
  """
  Получение стакана для символа.

  Args:
      symbol: Торговая пара
      current_user: Текущий пользователь

  Returns:
      dict: Данные стакана
  """
  from main import bot_controller

  if not bot_controller or not bot_controller.orderbook_managers:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Данные стакана недоступны"
    )

  manager = bot_controller.orderbook_managers.get(symbol)
  if not manager:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Стакан для {symbol} не найден"
    )

  snapshot = manager.get_snapshot()
  if not snapshot:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Нет данных стакана для {symbol}"
    )

  return snapshot.to_dict()


@data_router.get("/metrics/{symbol}")
async def get_metrics(
    symbol: str,
    current_user: dict = Depends(require_auth)
):
  """
  Получение метрик для символа.

  Args:
      symbol: Торговая пара
      current_user: Текущий пользователь

  Returns:
      dict: Метрики
  """
  from main import bot_controller

  if not bot_controller or not bot_controller.market_analyzer:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Анализатор недоступен"
    )

  metrics = bot_controller.market_analyzer.get_latest_metrics(symbol)
  if not metrics:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Метрики для {symbol} не найдены"
    )

  return metrics.to_dict()


@data_router.get("/metrics")
async def get_all_metrics(current_user: dict = Depends(require_auth)):
  """
  Получение метрик для всех символов.

  Args:
      current_user: Текущий пользователь

  Returns:
      dict: Метрики всех символов
  """
  from main import bot_controller

  if not bot_controller or not bot_controller.market_analyzer:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Анализатор недоступен"
    )

  all_metrics = bot_controller.market_analyzer.get_all_metrics()

  return {
    "metrics": {
      symbol: metrics.to_dict()
      for symbol, metrics in all_metrics.items()
    },
    "count": len(all_metrics)
  }


# ===== ТОРГОВЛЯ =====

@trading_router.get("/signals")
async def get_signals(
    symbol: Optional[str] = None,
    limit: int = 50,
    current_user: dict = Depends(require_auth)
):
  """
  Получение торговых сигналов.

  Args:
      symbol: Торговая пара (опционально)
      limit: Максимальное количество сигналов
      current_user: Текущий пользователь

  Returns:
      dict: Торговые сигналы
  """
  from main import bot_controller

  if not bot_controller or not bot_controller.strategy_engine:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Стратегия недоступна"
    )

  if symbol:
    signals = bot_controller.strategy_engine.get_signal_history(symbol, limit)
    return {
      "symbol": symbol,
      "signals": [s.to_dict() for s in signals],
      "count": len(signals)
    }
  else:
    all_stats = bot_controller.strategy_engine.get_all_statistics()
    return {
      "statistics": {
        sym: stats.to_dict()
        for sym, stats in all_stats.items()
      }
    }


@trading_router.get("/balance")
async def get_balance(current_user: dict = Depends(require_auth)):
  """
  Получение баланса аккаунта.
  Возвращает реальный баланс из Bybit API.

  Args:
      current_user: Текущий пользователь

  Returns:
      dict: Баланс аккаунта с детализацией по активам
  """
  logger.info("Запрос баланса через API")

  try:
    # Получаем реальный баланс из Bybit
    from exchange.rest_client import rest_client

    balance_data = await rest_client.get_wallet_balance()
    result = balance_data.get("result", {})
    wallet_list = result.get("list", [])

    # Парсим балансы по активам
    balances = {}
    total_usdt = 0.0

    for wallet in wallet_list:
      account_type = wallet.get("accountType", "UNIFIED")
      coins = wallet.get("coin", [])

      for coin in coins:
        coin_name = coin.get("coin", "")
        wallet_balance = float(coin.get("walletBalance", 0))
        available_balance = float(coin.get("availableToWithdraw", 0))
        locked_balance = wallet_balance - available_balance

        # Добавляем только активы с балансом > 0
        if wallet_balance > 0:
          balances[coin_name] = {
            "asset": coin_name,
            "free": round(available_balance, 8),
            "locked": round(locked_balance, 8),
            "total": round(wallet_balance, 8)
          }

          # Считаем USDT эквивалент
          if coin_name == "USDT":
            total_usdt += wallet_balance

    logger.info(f"Баланс получен: {len(balances)} активов, ${total_usdt:.2f} USDT")

    return {
      "balance": {
        "balances": balances,
        "total_usdt": round(total_usdt, 2),
        "timestamp": int(datetime.now().timestamp() * 1000)
      }
    }

  except ValueError as e:
    # API ключи не настроены
    logger.warning(f"API ключи не настроены: {e}")
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="API ключи Bybit не настроены. Настройте BYBIT_API_KEY и BYBIT_API_SECRET в .env файле."
    )
  except Exception as e:
    logger.error(f"Ошибка получения баланса: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Не удалось получить баланс: {str(e)}"
    )

@trading_router.get("/balance/history")
async def get_balance_history(
    period: str = "24h",
    current_user: dict = Depends(require_auth)
):
  """
  Получение истории изменения баланса.
  Возвращает реальные данные из трекера баланса.

  Args:
      period: Период ('1h', '24h', '7d', '30d')
      current_user: Текущий пользователь

  Returns:
      dict: История баланса с точками данных
  """
  logger.info(f"Запрос истории баланса за период: {period}")

  try:
    # Проверяем валидность периода
    valid_periods = ['1h', '24h', '7d', '30d']
    if period not in valid_periods:
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Неверный период. Используйте: {', '.join(valid_periods)}"
      )

    # Получаем историю из трекера
    history = balance_tracker.get_history(period)

    if not history:
      logger.warning(f"История баланса пуста для периода {period}")
      # Возвращаем пустую историю
      return {
        "points": [],
        "period": period
      }

    logger.info(f"Возвращено {len(history)} точек истории для периода {period}")

    return {
      "points": history,
      "period": period
    }

  except HTTPException:
    raise
  except Exception as e:
    logger.error(f"Ошибка получения истории баланса: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Не удалось получить историю: {str(e)}"
    )

@trading_router.get("/balance/stats")
async def get_balance_stats(current_user: dict = Depends(require_auth)):
  """
  Получение статистики баланса.
  Возвращает реальные данные: начальный баланс, PnL, лучший/худший день.

  Args:
      current_user: Текущий пользователь

  Returns:
      dict: Статистика баланса
  """
  logger.info("Запрос статистики баланса")

  try:
    # Получаем статистику из трекера
    stats = balance_tracker.get_stats()

    logger.info(
      f"Статистика: начальный=${stats['initial_balance']:.2f}, "
      f"текущий=${stats['current_balance']:.2f}, "
      f"PnL=${stats['total_pnl']:.2f} ({stats['total_pnl_percentage']:.2f}%)"
    )

    return stats

  except Exception as e:
    logger.error(f"Ошибка получения статистики баланса: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Не удалось получить статистику: {str(e)}"
    )

@trading_router.get("/positions")
async def get_positions(current_user: dict = Depends(require_auth)):
  """
  Получение открытых позиций.

  Args:
      current_user: Текущий пользователь

  Returns:
      dict: Позиции
  """
  from main import bot_controller

  if not bot_controller or not bot_controller.risk_manager:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Риск-менеджер недоступен"
    )

  positions = bot_controller.risk_manager.get_all_positions()
  return {
    "positions": positions,
    "count": len(positions)
  }


@trading_router.get("/risk-status")
async def get_risk_status(current_user: dict = Depends(require_auth)):
  """
  Получение статуса риска.

  Args:
      current_user: Текущий пользователь

  Returns:
      dict: Статус риска
  """
  from main import bot_controller

  if not bot_controller or not bot_controller.risk_manager:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Риск-менеджер недоступен"
    )

  return bot_controller.risk_manager.get_risk_status()


@trading_router.get("/execution-stats")
async def get_execution_stats(current_user: dict = Depends(require_auth)):
  """
  Получение статистики исполнения.

  Args:
      current_user: Текущий пользователь

  Returns:
      dict: Статистика
  """
  from main import bot_controller

  if not bot_controller or not bot_controller.execution_manager:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Менеджер исполнения недоступен"
    )

  return bot_controller.execution_manager.get_statistics()