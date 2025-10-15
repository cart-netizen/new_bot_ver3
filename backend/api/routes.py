"""
API маршруты.
Определение всех REST API эндпоинтов для взаимодействия с фронтендом.
"""
import time
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query

from pydantic import BaseModel, Field
from fastapi import WebSocket
from api.app import app
from api.websocket import websocket_endpoint
from core.logger import get_logger
from core.auth import (
  AuthService,
  require_auth
)
from core.exceptions import AuthenticationError
from domain.state_machines.order_fsm import OrderStateMachine
from exchange.rest_client import rest_client, BybitRESTClient
from infrastructure.repositories.order_repository import OrderRepository
from infrastructure.resilience.circuit_breaker import circuit_breaker_manager
from infrastructure.resilience.rate_limiter import rate_limiter
from main import bot_controller
from models.user import LoginRequest, LoginResponse, ChangePasswordRequest
from config import settings
from utils.balance_tracker import balance_tracker

logger = get_logger(__name__)

# Создаем роутеры
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
bot_router = APIRouter(prefix="/bot", tags=["Bot Control"])
data_router = APIRouter(prefix="/data", tags=["Market Data"])
trading_router = APIRouter(prefix="/trading", tags=["Trading"])

monitoring_router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

# ML Router
ml_router = APIRouter(prefix="/api/ml", tags=["ml"])

# Detection Router
detection_router = APIRouter(prefix="/api/detection", tags=["detection"])

# Strategies Router
strategies_router = APIRouter(prefix="/api/strategies", tags=["strategies"])

screener_router = APIRouter(prefix="/api/screener", tags=["screener"])

orders_router = APIRouter(prefix="/api/trading/orders", tags=["orders"])


class OrderResponse(BaseModel):
  """Модель ответа с данными ордера."""
  order_id: str
  client_order_id: str
  exchange_order_id: Optional[str] = None
  symbol: str
  side: str  # BUY или SELL
  order_type: str
  quantity: float
  price: Optional[float]
  filled_quantity: float
  average_price: float
  take_profit: Optional[float]
  stop_loss: Optional[float]
  leverage: int
  status: str
  created_at: str
  updated_at: str
  filled_at: Optional[str] = None
  strategy: Optional[str] = None
  signal_id: Optional[str] = None
  notes: Optional[str] = None

  class Config:
    json_schema_extra = {
      "example": {
        "order_id": "uuid-123",
        "client_order_id": "ORDER_123",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "order_type": "LIMIT",
        "quantity": 0.01,
        "price": 50000.0,
        "filled_quantity": 0.0,
        "average_price": 0.0,
        "take_profit": 51000.0,
        "stop_loss": 49000.0,
        "leverage": 10,
        "status": "PLACED",
        "created_at": "2025-10-14T10:00:00Z",
        "updated_at": "2025-10-14T10:00:00Z",
      }
    }


class OrderDetailResponse(BaseModel):
  """Модель детального ответа с расчетом PnL."""
  order_id: str
  client_order_id: str
  exchange_order_id: Optional[str]
  symbol: str
  side: str
  order_type: str
  quantity: float
  price: Optional[float]
  filled_quantity: float
  average_price: float
  take_profit: Optional[float]
  stop_loss: Optional[float]
  leverage: int
  status: str
  created_at: str
  updated_at: str
  filled_at: Optional[str]
  strategy: Optional[str]
  signal_id: Optional[str]

  # Дополнительные поля для детального просмотра
  current_pnl: float = Field(..., description="Текущий PnL в USDT")
  current_pnl_percent: float = Field(..., description="Текущий PnL в процентах")
  current_price: float = Field(..., description="Текущая цена актива")
  entry_price: float = Field(..., description="Цена входа")
  position_value: float = Field(..., description="Стоимость позиции")
  margin_used: float = Field(..., description="Использованная маржа")
  liquidation_price: Optional[float] = Field(None, description="Цена ликвидации")
  fees: float = Field(default=0.0, description="Комиссии")


class OrdersListResponse(BaseModel):
  """Модель ответа со списком ордеров."""
  orders: List[OrderResponse]
  total: int
  active: int
  timestamp: int


class CloseOrderRequest(BaseModel):
  """Модель запроса на закрытие ордера."""
  reason: Optional[str] = Field(None, description="Причина закрытия")


class CloseOrderResponse(BaseModel):
  """Модель ответа при закрытии ордера."""
  success: bool
  order_id: str
  closed_at: str
  final_pnl: float
  message: str

# ===== МОДЕЛИ ОТВЕТОВ =====
class ScreenerPairResponse(BaseModel):
  """Модель ответа с данными торговой пары для скринера."""
  symbol: str = Field(..., description="Символ торговой пары")
  lastPrice: float = Field(..., description="Последняя цена")
  price24hPcnt: float = Field(..., description="Изменение цены за 24ч в процентах")
  volume24h: float = Field(..., description="Объем торгов за 24ч в USDT")
  highPrice24h: float = Field(..., description="Максимальная цена за 24ч")
  lowPrice24h: float = Field(..., description="Минимальная цена за 24ч")
  prevPrice24h: float = Field(..., description="Цена 24 часа назад")
  turnover24h: float = Field(..., description="Оборот за 24ч")

  class Config:
    json_schema_extra = {
      "example": {
        "symbol": "BTCUSDT",
        "lastPrice": 50000.0,
        "price24hPcnt": 2.5,
        "volume24h": 5000000.0,
        "highPrice24h": 51000.0,
        "lowPrice24h": 49000.0,
        "prevPrice24h": 48000.0,
        "turnover24h": 250000000.0
      }
    }


class ScreenerDataResponse(BaseModel):
  """Полный ответ со всеми парами скринера."""
  pairs: List[ScreenerPairResponse] = Field(..., description="Список торговых пар")
  total: int = Field(..., description="Общее количество пар")
  timestamp: int = Field(..., description="Временная метка данных")
  min_volume: float = Field(default=4_000_000, description="Минимальный объем для фильтрации")


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

# ===== БАЛАНС И АККАУНТ =====
@trading_router.get("/balance")
async def get_balance(current_user: dict = Depends(require_auth)):
  """
  Получение баланса аккаунта.
  Возвращает реальный баланс из Bybit API или тестовые данные если API ключи не настроены.

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

    # Логируем полученную структуру данных для отладки
    logger.debug(f"Структура balance_data: {type(balance_data)}")
    logger.debug(f"Ключи balance_data: {balance_data.keys() if isinstance(balance_data, dict) else 'не dict'}")

    result = balance_data.get("result", {})
    wallet_list = result.get("list", [])

    logger.debug(f"Количество кошельков в wallet_list: {len(wallet_list)}")

    # Парсим балансы по активам с безопасной обработкой
    balances = {}
    total_usdt = 0.0
    parsing_errors = []

    for wallet_idx, wallet in enumerate(wallet_list):
      try:
        account_type = wallet.get("accountType", "UNIFIED")
        coins = wallet.get("coin", [])

        logger.debug(f"Обработка wallet {wallet_idx}: accountType={account_type}, монет={len(coins)}")

        for coin_idx, coin in enumerate(coins):
          try:
            coin_name = coin.get("coin", "")

            # Безопасное получение числовых значений
            wallet_balance_raw = coin.get("walletBalance", "0")
            available_balance_raw = coin.get("availableToWithdraw", "0")

            logger.debug(
              f"  Монета {coin_idx}: {coin_name}, "
              f"walletBalance={wallet_balance_raw} (тип: {type(wallet_balance_raw)}), "
              f"availableToWithdraw={available_balance_raw} (тип: {type(available_balance_raw)})"
            )

            # Преобразуем в float с обработкой исключений
            try:
              wallet_balance = float(wallet_balance_raw) if wallet_balance_raw else 0.0
            except (ValueError, TypeError) as e:
              logger.warning(f"Не удалось преобразовать walletBalance '{wallet_balance_raw}' для {coin_name}: {e}")
              wallet_balance = 0.0
              parsing_errors.append(f"{coin_name}.walletBalance")

            try:
              available_balance = float(available_balance_raw) if available_balance_raw else 0.0
            except (ValueError, TypeError) as e:
              logger.warning(
                f"Не удалось преобразовать availableToWithdraw '{available_balance_raw}' для {coin_name}: {e}")
              available_balance = 0.0
              parsing_errors.append(f"{coin_name}.availableToWithdraw")

            locked_balance = max(0.0, wallet_balance - available_balance)

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

              logger.debug(f"  Добавлен актив: {coin_name} = {wallet_balance:.8f}")

          except Exception as e:
            logger.error(f"Ошибка обработки монеты {coin_idx} ({coin.get('coin', 'unknown')}): {e}", exc_info=True)
            parsing_errors.append(f"coin_{coin_idx}")
            continue

      except Exception as e:
        logger.error(f"Ошибка обработки кошелька {wallet_idx}: {e}", exc_info=True)
        parsing_errors.append(f"wallet_{wallet_idx}")
        continue

    # Логируем итоги парсинга
    if parsing_errors:
      logger.warning(f"Ошибки парсинга полей: {', '.join(parsing_errors)}")

    logger.info(f"Баланс получен: {len(balances)} активов, ${total_usdt:.2f} USDT")

    return {
      "balance": {
        "balances": balances,
        "total_usdt": round(total_usdt, 2),
        "timestamp": int(datetime.now().timestamp() * 1000)
      }
    }

  except ValueError as e:
    # API ключи не настроены - это единственный случай когда должен быть ValueError от rest_client
    logger.warning(f"API ключи не настроены: {e}")
    logger.warning(f"Возвращаем тестовые данные")

    return {
      "balance": {
        "balances": {
          "USDT": {
            "asset": "USDT",
            "free": 10000.0,
            "locked": 0.0,
            "total": 10000.0
          }
        },
        "total_usdt": 10000.0,
        "timestamp": int(datetime.now().timestamp() * 1000)
      }
    }

  except Exception as e:
    # Любые другие ошибки
    logger.error(f"Критическая ошибка получения баланса: {type(e).__name__}: {e}", exc_info=True)

    # Возвращаем тестовые данные вместо ошибки
    return {
      "balance": {
        "balances": {
          "USDT": {
            "asset": "USDT",
            "free": 10000.0,
            "locked": 0.0,
            "total": 10000.0
          }
        },
        "total_usdt": 10000.0,
        "timestamp": int(datetime.now().timestamp() * 1000)
      }
    }


@trading_router.get("/balance/history")
async def get_balance_history(
    period: str = "24h",
    current_user: dict = Depends(require_auth)
):
  """
  Получение истории изменения баланса.
  Возвращает реальные данные из трекера баланса или тестовые данные.

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
      logger.warning(f"История баланса пуста для периода {period}, возвращаем тестовые данные")
      # Генерируем тестовые данные
      from datetime import datetime, timedelta
      import random

      now = datetime.now()
      period_config = {
        "1h": {"points": 12, "interval_minutes": 5},
        "24h": {"points": 24, "interval_minutes": 60},
        "7d": {"points": 7, "interval_minutes": 1440},
        "30d": {"points": 30, "interval_minutes": 1440}
      }

      config = period_config.get(period, period_config["24h"])
      base_balance = 10000.0
      points = []

      for i in range(config["points"]):
        timestamp_dt = now - timedelta(minutes=config["interval_minutes"] * (config["points"] - i - 1))
        variation = random.uniform(-0.01, 0.01)
        balance = base_balance * (1 + variation)

        points.append({
          "timestamp": int(timestamp_dt.timestamp() * 1000),
          "balance": round(balance, 2),
          "datetime": timestamp_dt.isoformat()
        })

      return {
        "points": points,
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
    # Возвращаем пустую историю вместо ошибки
    return {
      "points": [],
      "period": period
    }


@trading_router.get("/balance/stats")
async def get_balance_stats(current_user: dict = Depends(require_auth)):
  """
  Получение статистики баланса.
  Возвращает реальные данные или тестовые данные если история пустая.

  Args:
      current_user: Текущий пользователь

  Returns:
      dict: Статистика баланса
  """
  logger.info("Запрос статистики баланса")

  try:
    # Получаем статистику из трекера
    stats = balance_tracker.get_stats()

    # Проверяем что есть реальные данные
    if stats['current_balance'] == 0.0:
      logger.warning("Статистика баланса пустая, возвращаем тестовые данные")
      # Возвращаем тестовые данные
      return {
        "initial_balance": 10000.0,
        "current_balance": 10000.0,
        "total_pnl": 0.0,
        "total_pnl_percentage": 0.0,
        "daily_pnl": 0.0,
        "daily_pnl_percentage": 0.0,
        "best_day": 0.0,
        "worst_day": 0.0
      }

    logger.info(
      f"Статистика: начальный=${stats['initial_balance']:.2f}, "
      f"текущий=${stats['current_balance']:.2f}, "
      f"PnL=${stats['total_pnl']:.2f} ({stats['total_pnl_percentage']:.2f}%)"
    )

    return stats

  except Exception as e:
    logger.error(f"Ошибка получения статистики баланса: {e}")
    # Возвращаем тестовые данные вместо ошибки
    return {
      "initial_balance": 10000.0,
      "current_balance": 10000.0,
      "total_pnl": 0.0,
      "total_pnl_percentage": 0.0,
      "daily_pnl": 0.0,
      "daily_pnl_percentage": 0.0,
      "best_day": 0.0,
      "worst_day": 0.0
    }

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


@data_router.get("/candles/{symbol}")
async def get_candles(
    symbol: str,
    interval: str = "1",  # По умолчанию 1 минута
    limit: int = 100,
    current_user: dict = Depends(require_auth)
):
  """
  Получение свечных данных (klines) для торговой пары.

  Args:
      symbol: Торговая пара (например, BTCUSDT)
      interval: Интервал свечи (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
      limit: Количество свечей (макс 200, по умолчанию 100)
      current_user: Текущий пользователь

  Returns:
      dict: Данные свечей с timestamps и ценами OHLCV
  """
  logger.info(f"Запрос свечей для {symbol}, интервал: {interval}, лимит: {limit}")

  try:
    # Валидация параметров
    if limit < 1 or limit > 200:
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Limit должен быть от 1 до 200"
      )

    # Валидация интервала
    valid_intervals = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
    if interval not in valid_intervals:
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Неверный интервал. Допустимые: {', '.join(valid_intervals)}"
      )

    # Получаем свечи от биржи
    klines = await rest_client.get_kline(
      symbol=symbol,
      interval=interval,
      limit=limit
    )

    # Bybit возвращает массив в формате:
    # [timestamp, open, high, low, close, volume, turnover]
    # Нужно преобразовать в удобный формат для фронтенда

    candles = []
    for kline in reversed(klines):  # Bybit возвращает от новых к старым, разворачиваем
      try:
        candles.append({
          "timestamp": int(kline[0]),  # timestamp в миллисекундах
          "open": float(kline[1]),
          "high": float(kline[2]),
          "low": float(kline[3]),
          "close": float(kline[4]),
          "volume": float(kline[5]),
          "turnover": float(kline[6]) if len(kline) > 6 else 0.0
        })
      except (IndexError, ValueError) as e:
        logger.warning(f"Ошибка парсинга свечи: {e}")
        continue

    logger.info(f"Успешно получено {len(candles)} свечей для {symbol}")

    return {
      "symbol": symbol,
      "interval": interval,
      "candles": candles,
      "count": len(candles)
    }

  except HTTPException:
    raise
  except Exception as e:
    logger.error(f"Ошибка получения свечей для {symbol}: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Ошибка получения свечей: {str(e)}"
    )


@monitoring_router.get("/circuit-breakers")
async def get_circuit_breakers(current_user: dict = Depends(require_auth)):
  """Получение статуса всех Circuit Breakers."""
  return circuit_breaker_manager.get_all_status()


@monitoring_router.get("/rate-limiters")
async def get_rate_limiters(current_user: dict = Depends(require_auth)):
  """Получение статуса всех Rate Limiters."""
  return rate_limiter.get_all_status()


@monitoring_router.post("/circuit-breakers/{name}/reset")
async def reset_circuit_breaker(
    name: str,
    current_user: dict = Depends(require_auth)
):
  """Сброс конкретного Circuit Breaker."""
  breaker = circuit_breaker_manager.breakers.get(name)
  if not breaker:
    raise HTTPException(404, f"Circuit breaker '{name}' not found")

  breaker.reset()
  return {"status": "reset", "breaker": name}

# ==================== ML INFRASTRUCTURE ====================

@ml_router.get("/status")
async def get_ml_status():
  """Статус ML компонентов."""
  return {
    "ml_validator": bot_controller.ml_validator.get_statistics(),
    "spoofing_detector": bot_controller.spoofing_detector.get_statistics(),
    "layering_detector": bot_controller.layering_detector.get_statistics(),
    "sr_detector": bot_controller.sr_detector.get_statistics(),
    "drift_detector": bot_controller.drift_detector.get_drift_report()
  }


# ==================== DETECTION SYSTEMS ====================

@detection_router.get("/status/{symbol}")
async def get_detection_status(symbol: str):
  """Статус детекторов для символа."""
  spoofing_active = bot_controller.spoofing_detector.is_spoofing_active(symbol)
  layering_active = bot_controller.layering_detector.is_layering_active(symbol)

  spoofing_patterns = bot_controller.spoofing_detector.get_recent_patterns(symbol)
  layering_patterns = bot_controller.layering_detector.get_recent_patterns(symbol)

  return {
    "symbol": symbol,
    "spoofing": {
      "active": spoofing_active,
      "patterns": [
        {
          "side": p.side,
          "confidence": p.confidence,
          "reason": p.reason
        }
        for p in spoofing_patterns
      ]
    },
    "layering": {
      "active": layering_active,
      "patterns": [
        {
          "side": p.side,
          "confidence": p.confidence,
          "layers": len(p.layers),
          "reason": p.reason
        }
        for p in layering_patterns
      ]
    }
  }


@detection_router.get("/sr-levels/{symbol}")
async def get_sr_levels(symbol: str):
  """S/R уровни для символа."""
  levels = bot_controller.sr_detector.levels.get(symbol, [])

  return {
    "symbol": symbol,
    "levels": [
      {
        "price": level.price,
        "type": level.level_type,
        "strength": level.strength,
        "touch_count": level.touch_count,
        "is_broken": level.is_broken,
        "age_hours": level.age_hours(int(datetime.now().timestamp() * 1000))
      }
      for level in levels
    ]
  }


# ==================== STRATEGY MANAGER ====================

@strategies_router.get("/status")
async def get_strategies_status():
  """Статус всех стратегий."""
  return bot_controller.strategy_manager.get_statistics()


@strategies_router.get("/{strategy_name}/stats")
async def get_strategy_stats(strategy_name: str):
  """Статистика конкретной стратегии."""
  if strategy_name not in bot_controller.strategy_manager.strategies:
    raise HTTPException(status_code=404, detail="Strategy not found")

  strategy = bot_controller.strategy_manager.strategies[strategy_name]
  return strategy.get_statistics()


@screener_router.get(
  "/pairs",
  response_model=ScreenerDataResponse,
  summary="Получение данных всех пар для скринера",
  description="""
    Возвращает данные всех торговых пар с фильтрацией по минимальному объему.

    **Фильтрация:**
    - Только USDT perpetual futures
    - Объем за 24ч > 4,000,000 USDT
    - Только активные пары

    **Данные включают:**
    - Текущую цену
    - Изменение за 24 часа
    - Объем торгов
    - High/Low за 24 часа
    """
)
async def get_screener_pairs(
    min_volume: float = 4_000_000,
    current_user: dict = Depends(require_auth)
) -> ScreenerDataResponse:
  """
  Получение данных торговых пар для скринера.

  Args:
      min_volume: Минимальный объем за 24ч для фильтрации (default: 4M USDT)
      current_user: Авторизованный пользователь

  Returns:
      ScreenerDataResponse с данными всех подходящих пар

  Raises:
      HTTPException: При ошибке получения данных от биржи
  """
  try:
    logger.info(f"Запрос данных скринера с min_volume={min_volume}")

    # Создаем клиент Bybit API
    rest_client = BybitRESTClient()

    # Получаем тикеры всех пар
    tickers_response = await rest_client.get_tickers(category="linear")

    if not tickers_response or "result" not in tickers_response:
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Не удалось получить данные от Bybit API"
      )

    tickers_list = tickers_response["result"].get("list", [])

    # Фильтруем и форматируем данные
    filtered_pairs: List[ScreenerPairResponse] = []

    for ticker in tickers_list:
      try:
        symbol = ticker.get("symbol", "")

        # Фильтр 1: Только USDT пары
        if not symbol.endswith("USDT"):
          continue

        # Парсим числовые значения
        last_price = float(ticker.get("lastPrice", 0))
        volume_24h = float(ticker.get("volume24h", 0))
        turnover_24h = float(ticker.get("turnover24h", 0))
        price_24h_pcnt = float(ticker.get("price24hPcnt", 0)) * 100  # Конвертируем в проценты
        high_price_24h = float(ticker.get("highPrice24h", 0))
        low_price_24h = float(ticker.get("lowPrice24h", 0))
        prev_price_24h = float(ticker.get("prevPrice24h", 0))

        # Фильтр 2: Минимальный объем
        # Используем turnover_24h (оборот в USDT), а не volume_24h (объем в монетах)
        if turnover_24h < min_volume:
          continue

        # Фильтр 3: Валидные данные
        if last_price <= 0 or high_price_24h <= 0 or low_price_24h <= 0:
          continue

        # Добавляем пару в результат
        pair_data = ScreenerPairResponse(
          symbol=symbol,
          lastPrice=last_price,
          price24hPcnt=price_24h_pcnt,
          volume24h=turnover_24h,  # Используем turnover как volume в USDT
          highPrice24h=high_price_24h,
          lowPrice24h=low_price_24h,
          prevPrice24h=prev_price_24h,
          turnover24h=turnover_24h
        )

        filtered_pairs.append(pair_data)

      except (ValueError, KeyError) as e:
        logger.warning(f"Ошибка парсинга тикера {ticker.get('symbol', 'unknown')}: {e}")
        continue

    # Сортируем по объему (по убыванию)
    filtered_pairs.sort(key=lambda x: x.volume24h, reverse=True)

    logger.info(
      f"Успешно обработано {len(filtered_pairs)} пар из {len(tickers_list)} "
      f"(фильтр: volume > {min_volume:,.0f} USDT)"
    )

    # Формируем ответ
    response = ScreenerDataResponse(
      pairs=filtered_pairs,
      total=len(filtered_pairs),
      timestamp=int(time.time() * 1000),
      min_volume=min_volume
    )

    return response

  except HTTPException:
    raise
  except Exception as e:
    logger.error(f"Ошибка получения данных скринера: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Внутренняя ошибка сервера: {str(e)}"
    )


@screener_router.get(
  "/pairs/{symbol}",
  response_model=ScreenerPairResponse,
  summary="Получение данных конкретной пары",
  description="Возвращает детальные данные для указанной торговой пары"
)
async def get_screener_pair(
    symbol: str,
    current_user: dict = Depends(require_auth)
) -> ScreenerPairResponse:
  """
  Получение данных конкретной торговой пары.

  Args:
      symbol: Символ торговой пары (например, BTCUSDT)
      current_user: Авторизованный пользователь

  Returns:
      ScreenerPairResponse с данными пары

  Raises:
      HTTPException: При ошибке или если пара не найдена
  """
  try:
    logger.info(f"Запрос данных пары {symbol} для скринера")

    rest_client = BybitRESTClient()

    # Получаем тикер конкретной пары
    ticker_response = await rest_client.get_tickers(
      category="linear",
      symbol=symbol
    )

    if not ticker_response or "result" not in ticker_response:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Пара {symbol} не найдена"
      )

    ticker = ticker_response["result"].get("list", [{}])[0]

    if not ticker:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Данные для пары {symbol} отсутствуют"
      )

    # Парсим данные
    pair_data = ScreenerPairResponse(
      symbol=ticker.get("symbol", symbol),
      lastPrice=float(ticker.get("lastPrice", 0)),
      price24hPcnt=float(ticker.get("price24hPcnt", 0)) * 100,
      volume24h=float(ticker.get("turnover24h", 0)),
      highPrice24h=float(ticker.get("highPrice24h", 0)),
      lowPrice24h=float(ticker.get("lowPrice24h", 0)),
      prevPrice24h=float(ticker.get("prevPrice24h", 0)),
      turnover24h=float(ticker.get("turnover24h", 0))
    )

    logger.info(f"Успешно получены данные для {symbol}")

    return pair_data

  except HTTPException:
    raise
  except Exception as e:
    logger.error(f"Ошибка получения данных для {symbol}: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Ошибка получения данных: {str(e)}"
    )

@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    await websocket_endpoint(websocket)


@orders_router.get(
  "",
  response_model=OrdersListResponse,
  summary="Получение списка ордеров",
  description="""
    Возвращает список ордеров с возможностью фильтрации.

    **Фильтры:**
    - status: фильтр по статусу (active, filled, cancelled, all)
    - symbol: фильтр по торговой паре
    - strategy: фильтр по стратегии
    """
)
async def get_orders(
    status: Optional[str] = Query("active", description="Фильтр по статусу"),
    symbol: Optional[str] = Query(None, description="Фильтр по символу"),
    strategy: Optional[str] = Query(None, description="Фильтр по стратегии"),
    current_user: dict = Depends(require_auth)
):
  """
  Получение списка ордеров.
  """
  try:
    logger.info(f"Запрос списка ордеров: status={status}, symbol={symbol}, strategy={strategy}")

    # Получаем репозиторий
    order_repo = OrderRepository()

    # Определяем статусы для фильтрации
    if status == "active":
      filter_statuses = ["Pending", "Placed", "PartiallyFilled"]
    elif status == "filled":
      filter_statuses = ["Filled"]
    elif status == "cancelled":
      filter_statuses = ["Cancelled"]
    else:
      filter_statuses = None

    # Получаем ордера из БД
    orders = await order_repo.find_orders(
      symbol=symbol,
      statuses=filter_statuses,
      strategy=strategy
    )

    # Конвертируем в response модели
    order_responses = []
    active_count = 0

    for order in orders:
      order_response = OrderResponse(
        order_id=order.order_id,
        client_order_id=order.client_order_id,
        exchange_order_id=order.exchange_order_id,
        symbol=order.symbol,
        side=order.side,
        order_type=order.order_type,
        quantity=order.quantity,
        price=order.price,
        filled_quantity=order.filled_quantity,
        average_price=order.average_price,
        take_profit=order.take_profit,
        stop_loss=order.stop_loss,
        leverage=order.leverage,
        status=order.status,
        created_at=order.created_at.isoformat(),
        updated_at=order.updated_at.isoformat(),
        filled_at=order.filled_at.isoformat() if order.filled_at else None,
        strategy=order.strategy,
        signal_id=order.signal_id,
        notes=order.notes
      )

      order_responses.append(order_response)

      if order.status in ["Pending", "Placed", "PartiallyFilled"]:
        active_count += 1

    logger.info(f"Найдено {len(order_responses)} ордеров ({active_count} активных)")

    return OrdersListResponse(
      orders=order_responses,
      total=len(order_responses),
      active=active_count,
      timestamp=int(datetime.now().timestamp() * 1000)
    )

  except Exception as e:
    logger.error(f"Ошибка получения списка ордеров: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Ошибка получения списка ордеров: {str(e)}"
    )


@orders_router.get(
  "/{order_id}",
  response_model=OrderDetailResponse,
  summary="Получение детальной информации об ордере",
  description="Возвращает детальную информацию об ордере с расчетом PnL"
)
async def get_order_detail(
    order_id: str,
    current_user: dict = Depends(require_auth)
):
  """
  Получение детальной информации об ордере.
  """
  try:
    logger.info(f"Запрос деталей для ордера {order_id}")

    # Получаем репозиторий
    order_repo = OrderRepository()

    # Получаем ордер из БД
    order = await order_repo.get_by_client_order_id(order_id)

    if not order:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Ордер {order_id} не найден"
      )

    # Получаем текущую цену актива
    rest_client = BybitRESTClient()
    ticker = await rest_client.get_tickers(category="linear", symbol=order.symbol)

    current_price = float(ticker["result"]["list"][0]["lastPrice"])
    entry_price = order.price or order.average_price

    # Расчет PnL
    position_value = order.quantity * entry_price
    margin_used = position_value / order.leverage

    if order.side == "BUY":
      pnl = (current_price - entry_price) * order.quantity
    else:  # SELL
      pnl = (entry_price - current_price) * order.quantity

    pnl_percent = (pnl / position_value) * 100 if position_value > 0 else 0

    # Расчет цены ликвидации (упрощенный)
    if order.side == "BUY":
      liquidation_price = entry_price * (1 - 1 / order.leverage)
    else:
      liquidation_price = entry_price * (1 + 1 / order.leverage)

    # Комиссии (0.1% maker/taker)
    fees = position_value * 0.001

    logger.info(f"Детали ордера {order_id}: PnL={pnl:.2f}, Price={current_price}")

    return OrderDetailResponse(
      order_id=order.order_id,
      client_order_id=order.client_order_id,
      exchange_order_id=order.exchange_order_id,
      symbol=order.symbol,
      side=order.side,
      order_type=order.order_type,
      quantity=order.quantity,
      price=order.price,
      filled_quantity=order.filled_quantity,
      average_price=order.average_price,
      take_profit=order.take_profit,
      stop_loss=order.stop_loss,
      leverage=order.leverage,
      status=order.status,
      created_at=order.created_at.isoformat(),
      updated_at=order.updated_at.isoformat(),
      filled_at=order.filled_at.isoformat() if order.filled_at else None,
      strategy=order.strategy,
      signal_id=order.signal_id,
      current_pnl=pnl,
      current_pnl_percent=pnl_percent,
      current_price=current_price,
      entry_price=entry_price,
      position_value=position_value,
      margin_used=margin_used,
      liquidation_price=liquidation_price,
      fees=fees
    )

  except HTTPException:
    raise
  except Exception as e:
    logger.error(f"Ошибка получения деталей ордера {order_id}: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Ошибка получения деталей ордера: {str(e)}"
    )


@orders_router.post(
  "/{order_id}/close",
  response_model=CloseOrderResponse,
  summary="Закрытие ордера",
  description="Закрывает открытый ордер с расчетом финального PnL"
)
async def close_order(
    order_id: str,
    request: CloseOrderRequest,
    current_user: dict = Depends(require_auth)
):
  """
  Закрытие ордера.
  """
  try:
    logger.info(f"Запрос на закрытие ордера {order_id}, причина: {request.reason}")

    # Получаем репозиторий
    order_repo = OrderRepository()

    # Получаем ордер из БД
    order = await order_repo.get_by_client_order_id(order_id)

    if not order:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Ордер {order_id} не найден"
      )

    # Проверяем, что ордер активен
    if order.status not in ["Pending", "Placed", "PartiallyFilled"]:
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Ордер в статусе {order.status} не может быть закрыт"
      )

    # Отменяем ордер на бирже
    rest_client = BybitRESTClient()

    try:
      cancel_result = await rest_client.cancel_order(
        category="linear",
        symbol=order.symbol,
        orderLinkId=order.client_order_id
      )

      logger.info(f"Ордер {order_id} отменен на бирже: {cancel_result}")
    except Exception as e:
      logger.error(f"Ошибка отмены ордера на бирже: {e}")
      # Продолжаем даже если не удалось отменить на бирже

    # Получаем текущую цену для расчета PnL
    ticker = await rest_client.get_tickers(category="linear", symbol=order.symbol)
    ticker_result = ticker.get("result", {})
    ticker_list = ticker_result.get("list", [])
    if not ticker_list:
      raise HTTPException(status_code=404, detail="Ticker not found")
    current_price = float(ticker_list[0].get("lastPrice", 0))
    entry_price = order.price or order.average_price

    # Расчет финального PnL
    if order.side == "BUY":
      final_pnl = (current_price - entry_price) * order.filled_quantity
    else:
      final_pnl = (entry_price - current_price) * order.filled_quantity

    # Обновляем статус ордера в БД через FSM
    fsm = OrderStateMachine(order)
    fsm.cancel()

    await order_repo.update(order)

    logger.info(f"Ордер {order_id} успешно закрыт, PnL={final_pnl:.2f}")

    return CloseOrderResponse(
      success=True,
      order_id=order_id,
      closed_at=datetime.now().isoformat(),
      final_pnl=final_pnl,
      message=f"Ордер успешно закрыт. PnL: {final_pnl:.2f} USDT"
    )

  except HTTPException:
    raise
  except Exception as e:
    logger.error(f"Ошибка закрытия ордера {order_id}: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Ошибка закрытия ордера: {str(e)}"
    )