"""
API маршруты.
Определение всех REST API эндпоинтов для взаимодействия с фронтендом.
"""
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.core.logger import get_logger
from backend.core.auth import (
  AuthService,
  require_auth
)
from backend.core.exceptions import AuthenticationError
from backend.exchange.rest_client import rest_client
from backend.infrastructure.resilience.circuit_breaker import circuit_breaker_manager
from backend.infrastructure.resilience.rate_limiter import rate_limiter


from backend.models.user import LoginRequest, LoginResponse, ChangePasswordRequest
from backend.config import settings
from backend.strategy.correlation_manager import correlation_manager
from backend.utils.balance_tracker import balance_tracker

logger = get_logger(__name__)

# Создаем роутеры
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
bot_router = APIRouter(prefix="/bot", tags=["Bot Control"])
data_router = APIRouter(prefix="/data", tags=["Market Data"])
trading_router = APIRouter(prefix="/trading", tags=["Trading"])

monitoring_router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

ml_router = APIRouter(prefix="/api/ml", tags=["ml"])

# Detection Router
detection_router = APIRouter(prefix="/api/detection", tags=["detection"])

# Strategies Router
strategies_router = APIRouter(prefix="/api/strategies", tags=["strategies"])

screener_router = APIRouter(prefix="/screener", tags=["Screener"])

adaptive_router = APIRouter(prefix="/adaptive", tags=["adaptive"])

# ML Management Router - для управления обучением и моделями через фронтенд
from backend.api.ml_management_api import router as ml_management_router

# Layering ML Router - для управления Layering ML моделью
from backend.api.layering_ml_api import router as layering_ml_router

# Hyperparameter Optimization Router - для автоматического поиска оптимальных параметров
from backend.api.hyperopt_api import router as hyperopt_router

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
  from backend.main import bot_controller

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
  from backend.main import bot_controller

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
  from backend.main import bot_controller

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
  from backend.main import bot_controller

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
  from backend.main import bot_controller

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
  from backend.main import bot_controller

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
  from backend.main import bot_controller

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
    from backend.exchange.rest_client import rest_client

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
  from backend.main import bot_controller

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
  from backend.main import bot_controller

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
  from backend.main import bot_controller

  if not bot_controller or not bot_controller.execution_manager:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Менеджер исполнения недоступен"
    )

  return bot_controller.execution_manager.get_statistics()


@trading_router.get("/circuit-breaker/status")
async def get_circuit_breaker_status(current_user: dict = Depends(require_auth)):
  """Статус защиты от слива депозита."""
  from backend.main import bot_controller

  rm = bot_controller.risk_manager
  current = rm.metrics.open_positions_count
  max_allowed = rm.limits.max_open_positions

  return {
    "circuit_breaker_active": current >= max_allowed,
    "current_positions": current,
    "max_positions": max_allowed,
    "remaining_slots": max(0, max_allowed - current),
    "open_positions": [
      {
        "symbol": symbol,
        "side": pos["side"],
        "size_usdt": pos["size_usdt"],
        "margin": pos["actual_margin"]
      }
      for symbol, pos in rm.open_positions.items()
    ],
    "can_open_new": current < max_allowed,
    "warning": "CIRCUIT BREAKER ACTIVE" if current >= max_allowed else None
  }

@trading_router.get("/positions/can-open/{symbol}")
async def can_open_position(
    symbol: str,
    current_user: dict = Depends(require_auth)
):
  """Проверка возможности открыть позицию."""
  from backend.main import bot_controller

  can_open, reason = bot_controller.risk_manager.can_open_new_position(symbol)

  return {
    "symbol": symbol,
    "can_open": can_open,
    "reason": reason,
    "open_positions_count": bot_controller.risk_manager.metrics.open_positions_count,
    "max_positions": bot_controller.risk_manager.limits.max_open_positions,
    "open_positions": list(bot_controller.risk_manager.open_positions.keys())
  }

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
  from backend.main import bot_controller
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
  from backend.main import bot_controller

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
          "spoofing_side": p.spoofing_side,
          "execution_side": p.execution_side,
          "confidence": p.confidence,
          "layers": len(p.layers),
          "total_orders": p.total_orders,
          "cancellation_rate": p.cancellation_rate,
          "spoofing_execution_ratio": p.spoofing_execution_ratio,
          "reason": p.reason
        }
        for p in layering_patterns
      ]
    }
  }


@detection_router.get("/sr-levels/{symbol}")
async def get_sr_levels(symbol: str):
  """S/R уровни для символа."""
  from backend.main import bot_controller
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


# ==================== QUOTE STUFFING DETECTION ====================

@detection_router.get("/quote-stuffing/status/{symbol}")
async def get_quote_stuffing_status(symbol: str, current_user: dict = Depends(require_auth)):
  """
  Статус Quote Stuffing для символа.

  Args:
      symbol: Торговая пара

  Returns:
      dict: Статус и метрики quote stuffing
  """
  from backend.main import bot_controller

  if not hasattr(bot_controller, 'quote_stuffing_detector') or not bot_controller.quote_stuffing_detector:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Quote Stuffing Detector недоступен"
    )

  detector = bot_controller.quote_stuffing_detector

  # Проверяем активность
  is_active = detector.is_stuffing_active(symbol, time_window_seconds=30)

  # Получаем последние события
  recent_events = detector.get_recent_events(symbol, limit=10)

  return {
    "symbol": symbol,
    "is_active": is_active,
    "recent_events": [
      {
        "timestamp": event.timestamp,
        "pattern_type": event.pattern_type,
        "confidence": event.confidence,
        "updates_per_second": event.metrics.get("updates_per_second", 0),
        "cancellation_rate": event.metrics.get("cancellation_rate", 0),
        "avg_order_size_btc": event.metrics.get("avg_order_size_btc", 0),
        "price_range_bps": event.metrics.get("price_range_bps", 0)
      }
      for event in recent_events
    ]
  }


@detection_router.get("/quote-stuffing/statistics")
async def get_quote_stuffing_statistics(current_user: dict = Depends(require_auth)):
  """
  Общая статистика Quote Stuffing Detector.

  Returns:
      dict: Статистика по всем символам
  """
  from backend.main import bot_controller

  if not hasattr(bot_controller, 'quote_stuffing_detector') or not bot_controller.quote_stuffing_detector:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Quote Stuffing Detector недоступен"
    )

  detector = bot_controller.quote_stuffing_detector
  stats = detector.get_statistics()

  return {
    "total_events_detected": stats.get("total_events_detected", 0),
    "symbols_tracked": stats.get("symbols_tracked", 0),
    "active_now": stats.get("active_now", []),
    "detection_rate_24h": stats.get("detection_rate_24h", 0)
  }


# ==================== PATTERN DATABASE ====================

@detection_router.get("/patterns/list")
async def get_pattern_list(
    limit: int = 50,
    sort_by: str = "occurrence_count",
    blacklist_only: bool = False,
    current_user: dict = Depends(require_auth)
):
  """
  Список исторических паттернов.

  Args:
      limit: Максимальное количество
      sort_by: Поле сортировки (occurrence_count, last_seen, success_rate)
      blacklist_only: Показать только blacklist

  Returns:
      dict: Список паттернов
  """
  from backend.main import bot_controller

  if not hasattr(bot_controller, 'pattern_database') or not bot_controller.pattern_database:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Pattern Database недоступна"
    )

  db = bot_controller.pattern_database
  patterns = db.get_all_patterns(limit=limit, sort_by=sort_by, blacklist_only=blacklist_only)

  return {
    "patterns": [
      {
        "pattern_id": p["pattern_id"],
        "first_seen": p["first_seen"],
        "last_seen": p["last_seen"],
        "occurrence_count": p["occurrence_count"],
        "avg_layer_count": p["avg_layer_count"],
        "avg_cancellation_rate": p["avg_cancellation_rate"],
        "avg_volume_btc": p["avg_volume_btc"],
        "symbols": p["symbols"],
        "success_rate": p["success_rate"],
        "risk_level": p["risk_level"],
        "blacklist": bool(p["blacklist"])
      }
      for p in patterns
    ],
    "total": len(patterns)
  }


@detection_router.get("/patterns/statistics")
async def get_pattern_database_statistics(current_user: dict = Depends(require_auth)):
  """
  Статистика Pattern Database.

  Returns:
      dict: Общая статистика базы паттернов
  """
  from backend.main import bot_controller

  if not hasattr(bot_controller, 'pattern_database') or not bot_controller.pattern_database:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Pattern Database недоступна"
    )

  db = bot_controller.pattern_database
  stats = db.get_statistics()

  return {
    "total_patterns": stats.get("total_patterns", 0),
    "blacklisted_patterns": stats.get("blacklisted_patterns", 0),
    "unique_symbols": stats.get("unique_symbols", 0),
    "avg_success_rate": stats.get("avg_success_rate", 0),
    "oldest_pattern_age_hours": stats.get("oldest_pattern_age_hours", 0)
  }


@detection_router.post("/patterns/{pattern_id}/blacklist")
async def toggle_pattern_blacklist(
    pattern_id: str,
    current_user: dict = Depends(require_auth)
):
  """
  Переключить blacklist статус паттерна.

  Args:
      pattern_id: ID паттерна

  Returns:
      dict: Новый статус
  """
  from backend.main import bot_controller

  if not hasattr(bot_controller, 'pattern_database') or not bot_controller.pattern_database:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Pattern Database недоступна"
    )

  db = bot_controller.pattern_database

  # Получаем текущий паттерн
  pattern = db.get_pattern_by_id(pattern_id)

  if not pattern:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Pattern {pattern_id} не найден"
    )

  # Переключаем blacklist
  new_blacklist_status = not bool(pattern["blacklist"])
  db.update_blacklist(pattern_id, new_blacklist_status)

  return {
    "pattern_id": pattern_id,
    "blacklist": new_blacklist_status,
    "message": f"Pattern {'добавлен в' if new_blacklist_status else 'удален из'} blacklist"
  }


# ==================== ML DATA COLLECTOR ====================

@ml_router.get("/data-collector/statistics")
async def get_data_collector_statistics(current_user: dict = Depends(require_auth)):
  """
  Статистика сбора ML данных.

  Returns:
      dict: Статистика data collector
  """
  from backend.main import bot_controller

  if not hasattr(bot_controller, 'layering_data_collector') or not bot_controller.layering_data_collector:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Layering Data Collector недоступен"
    )

  collector = bot_controller.layering_data_collector
  stats = collector.get_statistics()

  return {
    "enabled": collector.enabled,
    "buffer_size": stats.get("buffer_size", 0),
    "auto_save_interval": stats.get("auto_save_interval", 0),
    "total_collected": stats.get("total_collected", 0),
    "labeled_samples": stats.get("labeled_samples", 0),
    "unlabeled_samples": stats.get("unlabeled_samples", 0),
    "data_directory": stats.get("data_directory", ""),
    "files_count": stats.get("files_count", 0)
  }


@ml_router.post("/data-collector/save")
async def save_collected_data(current_user: dict = Depends(require_auth)):
  """
  Принудительное сохранение собранных данных.

  Returns:
      dict: Результат сохранения
  """
  from backend.main import bot_controller

  if not hasattr(bot_controller, 'layering_data_collector') or not bot_controller.layering_data_collector:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Layering Data Collector недоступен"
    )

  collector = bot_controller.layering_data_collector

  try:
    collector.save_to_disk()
    stats = collector.get_statistics()

    return {
      "status": "success",
      "message": f"Сохранено {stats['buffer_size']} samples",
      "buffer_size": stats["buffer_size"]
    }
  except Exception as e:
    logger.error(f"Ошибка сохранения ML данных: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Ошибка сохранения: {str(e)}"
    )


@ml_router.get("/data-collector/labeled-data")
async def get_labeled_data_info(current_user: dict = Depends(require_auth)):
  """
  Информация о labeled данных для обучения.

  Returns:
      dict: Статистика labeled samples
  """
  from backend.main import bot_controller

  if not hasattr(bot_controller, 'layering_data_collector') or not bot_controller.layering_data_collector:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Layering Data Collector недоступен"
    )

  collector = bot_controller.layering_data_collector

  try:
    labeled_df = collector.get_labeled_data()

    if labeled_df.empty:
      return {
        "total_labeled": 0,
        "positive_samples": 0,
        "negative_samples": 0,
        "ready_for_training": False
      }

    positive_count = len(labeled_df[labeled_df['is_true_layering'] == True])
    negative_count = len(labeled_df[labeled_df['is_true_layering'] == False])

    return {
      "total_labeled": len(labeled_df),
      "positive_samples": positive_count,
      "negative_samples": negative_count,
      "ready_for_training": len(labeled_df) >= 100,
      "balance_ratio": positive_count / max(negative_count, 1)
    }
  except Exception as e:
    logger.error(f"Ошибка получения labeled data: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Ошибка: {str(e)}"
    )


# ==================== ADAPTIVE ML MODEL ====================

@ml_router.get("/adaptive-model/status")
async def get_adaptive_model_status(current_user: dict = Depends(require_auth)):
  """
  Статус Adaptive ML Model.

  Returns:
      dict: Статус модели
  """
  from backend.main import bot_controller

  if not hasattr(bot_controller, 'adaptive_layering_model') or not bot_controller.adaptive_layering_model:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Adaptive Layering Model недоступна"
    )

  model = bot_controller.adaptive_layering_model

  return {
    "enabled": model.enabled,
    "is_trained": model.is_trained,
    "model_version": model.model_version if hasattr(model, 'model_version') else None,
    "feature_count": len(model.feature_names),
    "scaler_fitted": model.scaler is not None
  }


@ml_router.get("/adaptive-model/metrics")
async def get_adaptive_model_metrics(current_user: dict = Depends(require_auth)):
  """
  Метрики обученной модели.

  Returns:
      dict: Метрики модели (accuracy, precision, recall, F1, ROC AUC)
  """
  from backend.main import bot_controller

  if not hasattr(bot_controller, 'adaptive_layering_model') or not bot_controller.adaptive_layering_model:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Adaptive Layering Model недоступна"
    )

  model = bot_controller.adaptive_layering_model

  if not model.is_trained:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="Модель еще не обучена"
    )

  metrics = model.get_metrics()

  if not metrics:
    return {
      "available": False,
      "message": "Метрики недоступны - модель загружена из файла"
    }

  return {
    "available": True,
    "accuracy": metrics.accuracy,
    "precision": metrics.precision,
    "recall": metrics.recall,
    "f1_score": metrics.f1_score,
    "roc_auc": metrics.roc_auc,
    "confusion_matrix": metrics.confusion_matrix.tolist() if hasattr(metrics, 'confusion_matrix') else None
  }


@ml_router.get("/adaptive-model/feature-importance")
async def get_feature_importance(
    top_n: int = 10,
    current_user: dict = Depends(require_auth)
):
  """
  Feature importance для модели.

  Args:
      top_n: Количество top features

  Returns:
      dict: Список feature importance
  """
  from backend.main import bot_controller

  if not hasattr(bot_controller, 'adaptive_layering_model') or not bot_controller.adaptive_layering_model:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Adaptive Layering Model недоступна"
    )

  model = bot_controller.adaptive_layering_model

  if not model.is_trained:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="Модель еще не обучена"
    )

  try:
    importance_dict = model.get_feature_importance()

    # Сортируем по важности и берем top N
    sorted_features = sorted(
      importance_dict.items(),
      key=lambda x: x[1],
      reverse=True
    )[:top_n]

    return {
      "top_features": [
        {
          "feature": name,
          "importance": float(importance)
        }
        for name, importance in sorted_features
      ],
      "total_features": len(importance_dict)
    }
  except Exception as e:
    logger.error(f"Ошибка получения feature importance: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Ошибка: {str(e)}"
    )


# ==================== STRATEGY MANAGER ====================

@strategies_router.get("/status")
async def get_strategies_status():
  """Статус всех стратегий."""
  from backend.main import bot_controller
  return bot_controller.strategy_manager.get_statistics()


@strategies_router.get("/{strategy_name}/stats")
async def get_strategy_stats(strategy_name: str):
  """Статистика конкретной стратегии."""
  from backend.main import bot_controller
  if strategy_name not in bot_controller.strategy_manager.all_strategies:
    raise HTTPException(status_code=404, detail="Strategy not found")

  strategy = bot_controller.strategy_manager.all_strategies[strategy_name]
  return strategy.get_statistics()


@screener_router.get("/pairs", operation_id="get_screener_pairs_list")
async def get_screener_pairs(
    min_volume: Optional[float] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = "desc",
    current_user: dict = Depends(require_auth)
):
  """
  Получение списка торговых пар для скринера.

  Args:
      min_volume: Минимальный объем (опционально)
      sort_by: Поле для сортировки (price, volume, change_24h)
      sort_order: Порядок сортировки (asc/desc)
      current_user: Текущий пользователь

  Returns:
      List[dict]: Список торговых пар
  """
  try:
    from backend.main import bot_controller

    if not bot_controller or not hasattr(bot_controller, 'screener_manager') or not bot_controller.screener_manager:
      raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Screener не инициализирован"
      )

    screener = bot_controller.screener_manager
    pairs = screener.get_all_pairs()

    # Фильтрация по объему
    if min_volume is not None:
      pairs = [p for p in pairs if p['volume_24h'] >= min_volume]

    # Сортировка
    if sort_by:
      sort_field_map = {
        "price": "last_price",
        "volume": "volume_24h",
        "change_24h": "price_change_24h_percent"
      }
      field = sort_field_map.get(sort_by, "volume_24h")
      reverse = (sort_order == "desc")
      pairs.sort(key=lambda x: x.get(field, 0), reverse=reverse)

    logger.info(f"Возвращено {len(pairs)} пар из скринера")

    return {
      "pairs": pairs,
      "total": len(pairs),
      "timestamp": int(datetime.now().timestamp() * 1000)
    }

  except Exception as e:
    logger.error(f"Ошибка получения пар скринера: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )


@screener_router.post("/pair/{symbol}/toggle", operation_id="toggle_screener_pair_selection")
async def toggle_pair_selection(
    symbol: str,
    current_user: dict = Depends(require_auth)
):
  """
  Переключение выбора торговой пары для графиков.

  Args:
      symbol: Торговая пара
      current_user: Текущий пользователь

  Returns:
      dict: Результат операции
  """
  try:
    from backend.main import bot_controller

    if not bot_controller or not hasattr(bot_controller, 'screener_manager') or not bot_controller.screener_manager:
      raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Screener не инициализирован"
      )

    screener = bot_controller.screener_manager
    success = screener.toggle_selection(symbol.upper())

    if not success:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Пара {symbol} не найдена в скринере"
      )

    return {
      "success": True,
      "symbol": symbol,
      "selected_pairs": screener.get_selected_pairs()
    }

  except HTTPException:
    raise
  except Exception as e:
    logger.error(f"Ошибка переключения выбора пары: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )


@screener_router.get("/selected", operation_id="get_screener_selected_pairs")
async def get_selected_pairs(current_user: dict = Depends(require_auth)):
  """
  Получение списка выбранных пар для графиков.

  Args:
      current_user: Текущий пользователь

  Returns:
      dict: Список выбранных пар
  """
  try:
    from backend.main import bot_controller

    if not bot_controller or not hasattr(bot_controller, 'screener_manager') or not bot_controller.screener_manager:
      raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Screener не инициализирован"
      )

    screener = bot_controller.screener_manager
    selected = screener.get_selected_pairs()

    return {
      "selected_pairs": selected,
      "count": len(selected)
    }

  except Exception as e:
    logger.error(f"Ошибка получения выбранных пар: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )


@screener_router.get("/stats", operation_id="get_screener_statistics")
async def get_screener_stats(current_user: dict = Depends(require_auth)):
  """
  Получение статистики скринера.

  Args:
      current_user: Текущий пользователь

  Returns:
      dict: Статистика скринера
  """
  try:
    from backend.main import bot_controller

    if not bot_controller or not hasattr(bot_controller, 'screener_manager') or not bot_controller.screener_manager:
      raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Screener не инициализирован"
      )

    screener = bot_controller.screener_manager
    stats = screener.get_stats()

    return stats

  except Exception as e:
    logger.error(f"Ошибка получения статистики скринера: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )


@trading_router.get("/correlation/statistics")
async def get_correlation_statistics(current_user: dict = Depends(require_auth)):
  """
  Получение статистики корреляций.

  Returns:
      dict: Статистика корреляционных групп
  """
  try:
    stats = correlation_manager.get_statistics()
    return stats
  except Exception as e:
    logger.error(f"Ошибка получения статистики корреляций: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )


@trading_router.get("/correlation/groups")
async def get_correlation_groups(current_user: dict = Depends(require_auth)):
  """
  Получение детальной информации о корреляционных группах.

  Returns:
      dict: Список групп с деталями
  """
  try:
    groups = correlation_manager.get_group_details()
    return {
      "groups": groups,
      "total": len(groups)
    }
  except Exception as e:
    logger.error(f"Ошибка получения групп корреляций: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )


@trading_router.get("/risk/adaptive-stats")
async def get_adaptive_risk_stats(current_user: dict = Depends(require_auth)):
  """API endpoint для статистики Adaptive Risk."""

  from backend.main import bot_controller
  from backend.config import settings

  # Получаем статистику
  stats = bot_controller.risk_manager.get_adaptive_risk_statistics()

  return {
    "mode": settings.RISK_PER_TRADE_MODE,
    "config": {
      "base_percent": settings.RISK_PER_TRADE_BASE_PERCENT,
      "max_percent": settings.RISK_PER_TRADE_MAX_PERCENT,
      "kelly_fraction": settings.RISK_KELLY_FRACTION,
      "volatility_scaling": settings.RISK_VOLATILITY_SCALING,
      "win_rate_scaling": settings.RISK_WIN_RATE_SCALING,
      "correlation_penalty": settings.RISK_CORRELATION_PENALTY
    },
    "statistics": {
      "total_trades": stats['total_trades'],
      "win_rate": round(stats['win_rate'], 4),
      "avg_win": round(stats['avg_win'], 2),
      "avg_loss": round(stats['avg_loss'], 2),
      "payoff_ratio": round(stats['payoff_ratio'], 2)
    },
    "kelly_available": stats['total_trades'] >= settings.RISK_KELLY_MIN_TRADES
  }


@trading_router.get("/position-monitor/stats")
async def get_position_monitor_stats(current_user: dict = Depends(require_auth)):
  """Получение статистики Position Monitor."""
  from backend.main import bot_controller

  if not bot_controller or not bot_controller.position_monitor:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Position Monitor недоступен"
    )

  return bot_controller.position_monitor.get_statistics()


@adaptive_router.get("/statistics")
async def get_adaptive_statistics():
  """Статистика Adaptive Consensus."""
  from backend.main import bot_controller

  if not bot_controller.adaptive_consensus_manager:
    return {"enabled": False}

  return {
    "enabled": True,
    "statistics": bot_controller.adaptive_consensus_manager.get_statistics()
  }


@adaptive_router.get("/regime/{symbol}")
async def get_market_regime(symbol: str):
  """Текущий режим рынка для символа."""
  from backend.main import bot_controller

  if not bot_controller.adaptive_consensus_manager:
    return {"error": "Adaptive Consensus disabled"}

  regime_detector = bot_controller.adaptive_consensus_manager.regime_detector

  if not regime_detector:
    return {"error": "Regime Detector not available"}

  regime = regime_detector.get_current_regime(symbol)

  if not regime:
    return {"error": f"No regime data for {symbol}"}

  return {
    "symbol": regime.symbol,
    "trend": regime.trend.value,
    "trend_strength": regime.trend_strength,
    "volatility": regime.volatility.value,
    "liquidity": regime.liquidity.value,
    "adx": regime.adx_value,
    "atr": regime.atr_value,
    "recommended_weights": regime.recommended_strategy_weights
  }


@adaptive_router.get("/performance/{symbol}")
async def get_strategy_performance(symbol: str, time_window: str = "7d"):
  """Performance метрики стратегий для символа."""
  from backend.main import bot_controller

  if not bot_controller.adaptive_consensus_manager:
    return {"error": "Adaptive Consensus disabled"}

  tracker = bot_controller.adaptive_consensus_manager.performance_tracker

  if not tracker:
    return {"error": "Performance Tracker not available"}

  # Получаем метрики для всех стратегий
  metrics = {}

  for strategy_name in bot_controller.strategy_manager.all_strategies.keys():
    strategy_metrics = tracker.get_strategy_metrics(
      strategy_name, symbol, time_window
    )

    if strategy_metrics:
      metrics[strategy_name] = {
        "win_rate": strategy_metrics.win_rate,
        "sharpe_ratio": strategy_metrics.sharpe_ratio,
        "profit_factor": strategy_metrics.profit_factor,
        "performance_score": strategy_metrics.performance_score,
        "total_signals": strategy_metrics.total_signals,
        "closed_signals": strategy_metrics.closed_signals
      }

  return {"symbol": symbol, "time_window": time_window, "metrics": metrics}


@adaptive_router.get("/weights/{symbol}")
async def get_current_weights(symbol: str):
  """Текущие веса стратегий для символа."""
  from backend.main import bot_controller

  if not bot_controller.adaptive_consensus_manager:
    return {"error": "Adaptive Consensus disabled"}

  optimizer = bot_controller.adaptive_consensus_manager.weight_optimizer

  if not optimizer:
    return {"error": "Weight Optimizer not available"}

  current_weights = optimizer.current_weights.get(symbol, {})

  return {"symbol": symbol, "weights": current_weights}


# ==================== POSITIONS & ORDERS (BYBIT) ====================

class ClosePositionRequest(BaseModel):
  """Запрос на закрытие позиции."""
  symbol: str
  percent: int = 100  # 25, 50, 75, 100


@trading_router.get("/positions/exchange")
async def get_exchange_positions(current_user: dict = Depends(require_auth)):
  """
  Получение открытых позиций напрямую с биржи Bybit.

  Returns:
      dict: Список позиций с полной информацией
  """
  logger.info("Запрос позиций с биржи")

  try:
    from backend.exchange.rest_client import rest_client

    response = await rest_client.get_positions()
    positions_list = response.get("result", {}).get("list", [])

    # Фильтруем только позиции с размером > 0
    active_positions = []
    for pos in positions_list:
      size = float(pos.get("size", 0))
      if size > 0:
        # Вычисляем ROE%
        unrealised_pnl = float(pos.get("unrealisedPnl", 0))
        position_im = float(pos.get("positionIM", 0))
        roe_percent = (unrealised_pnl / position_im * 100) if position_im > 0 else 0

        active_positions.append({
          # Основная информация
          "symbol": pos.get("symbol", ""),
          "side": pos.get("side", ""),  # Buy = Long, Sell = Short
          "size": size,
          "avgPrice": float(pos.get("avgPrice", 0)),
          "positionValue": float(pos.get("positionValue", 0)),

          # Плечо и маржа
          "leverage": pos.get("leverage", "1"),
          "positionIM": position_im,  # Initial Margin
          "positionMM": float(pos.get("positionMM", 0)),  # Maintenance Margin

          # Цены
          "markPrice": float(pos.get("markPrice", 0)),
          "liqPrice": float(pos.get("liqPrice", 0)),

          # P&L
          "unrealisedPnl": unrealised_pnl,
          "cumRealisedPnl": float(pos.get("cumRealisedPnl", 0)),
          "roePercent": round(roe_percent, 2),

          # TP/SL
          "takeProfit": pos.get("takeProfit", ""),
          "stopLoss": pos.get("stopLoss", ""),
          "trailingStop": pos.get("trailingStop", ""),
          "tpslMode": pos.get("tpslMode", ""),

          # Метаданные
          "positionIdx": pos.get("positionIdx", 0),
          "tradeMode": pos.get("tradeMode", 0),
          "positionStatus": pos.get("positionStatus", ""),
          "updatedTime": pos.get("updatedTime", ""),
          "createdTime": pos.get("createdTime", ""),
        })

    logger.info(f"Получено {len(active_positions)} активных позиций с биржи")

    return {
      "positions": active_positions,
      "count": len(active_positions),
      "timestamp": int(datetime.now().timestamp() * 1000)
    }

  except ValueError as e:
    error_msg = str(e)
    # Проверяем, действительно ли это ошибка API ключей
    if "API" in error_msg or "ключ" in error_msg.lower() or "key" in error_msg.lower():
      logger.warning(f"API ключи не настроены: {e}")
      return {
        "positions": [],
        "count": 0,
        "timestamp": int(datetime.now().timestamp() * 1000),
        "error": "API ключи не настроены"
      }
    else:
      # Другая ValueError - логируем реальную ошибку
      logger.error(f"ValueError при получении позиций: {e}", exc_info=True)
      return {
        "positions": [],
        "count": 0,
        "timestamp": int(datetime.now().timestamp() * 1000),
        "error": f"Ошибка: {error_msg}"
      }

  except Exception as e:
    logger.error(f"Ошибка получения позиций с биржи: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )


@trading_router.get("/orders/open")
async def get_open_orders(
    symbol: Optional[str] = None,
    current_user: dict = Depends(require_auth)
):
  """
  Получение открытых ордеров с биржи Bybit.

  Args:
      symbol: Торговая пара (опционально)

  Returns:
      dict: Список открытых ордеров
  """
  logger.info(f"Запрос открытых ордеров{f' для {symbol}' if symbol else ''}")

  try:
    from backend.exchange.rest_client import rest_client

    response = await rest_client.get_open_orders(symbol)
    orders_list = response.get("result", {}).get("list", [])

    orders = []
    for order in orders_list:
      orders.append({
        "orderId": order.get("orderId", ""),
        "orderLinkId": order.get("orderLinkId", ""),
        "symbol": order.get("symbol", ""),
        "side": order.get("side", ""),
        "orderType": order.get("orderType", ""),
        "price": float(order.get("price", 0)),
        "qty": float(order.get("qty", 0)),
        "cumExecQty": float(order.get("cumExecQty", 0)),
        "cumExecValue": float(order.get("cumExecValue", 0)),
        "orderStatus": order.get("orderStatus", ""),
        "timeInForce": order.get("timeInForce", ""),
        "stopLoss": order.get("stopLoss", ""),
        "takeProfit": order.get("takeProfit", ""),
        "reduceOnly": order.get("reduceOnly", False),
        "createdTime": order.get("createdTime", ""),
        "updatedTime": order.get("updatedTime", ""),
      })

    logger.info(f"Получено {len(orders)} открытых ордеров")

    return {
      "orders": orders,
      "count": len(orders),
      "timestamp": int(datetime.now().timestamp() * 1000)
    }

  except ValueError as e:
    error_msg = str(e)
    if "API" in error_msg or "ключ" in error_msg.lower() or "key" in error_msg.lower():
      logger.warning(f"API ключи не настроены: {e}")
      return {
        "orders": [],
        "count": 0,
        "timestamp": int(datetime.now().timestamp() * 1000),
        "error": "API ключи не настроены"
      }
    else:
      logger.error(f"ValueError при получении ордеров: {e}", exc_info=True)
      return {
        "orders": [],
        "count": 0,
        "timestamp": int(datetime.now().timestamp() * 1000),
        "error": f"Ошибка: {error_msg}"
      }

  except Exception as e:
    logger.error(f"Ошибка получения ордеров: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )


@trading_router.post("/position/close")
async def close_position(
    request: ClosePositionRequest,
    current_user: dict = Depends(require_auth)
):
  """
  Закрытие позиции по рыночной цене.

  Args:
      request: Параметры закрытия (symbol, percent)

  Returns:
      dict: Результат закрытия
  """
  logger.info(f"Закрытие позиции: {request.symbol}, {request.percent}%")

  try:
    from backend.exchange.rest_client import rest_client

    # Получаем текущую позицию
    positions_response = await rest_client.get_positions(request.symbol)
    positions_list = positions_response.get("result", {}).get("list", [])

    # Ищем позицию с размером > 0
    position = None
    for pos in positions_list:
      if float(pos.get("size", 0)) > 0:
        position = pos
        break

    if not position:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Нет открытой позиции для {request.symbol}"
      )

    # Вычисляем размер для закрытия
    total_size = float(position.get("size", 0))
    close_size = round(total_size * request.percent / 100, 8)

    if close_size <= 0:
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Размер для закрытия слишком мал"
      )

    # Определяем сторону для закрытия (противоположная)
    position_side = position.get("side", "")
    close_side = "Sell" if position_side == "Buy" else "Buy"

    logger.info(
      f"Закрытие {request.percent}% позиции {request.symbol}: "
      f"size={close_size}, side={close_side}"
    )

    # Размещаем рыночный ордер с reduceOnly=True
    from backend.utils.constants import BybitCategory, BybitAPIPaths

    params = {
      "category": BybitCategory.LINEAR.value,
      "symbol": request.symbol,
      "side": close_side,
      "orderType": "Market",
      "qty": str(close_size),
      "reduceOnly": True,
      "timeInForce": "IOC"  # Immediate or Cancel для рыночных
    }

    response = await rest_client._request(
      "POST",
      BybitAPIPaths.PLACE_ORDER,
      params,
      authenticated=True
    )

    order_id = response.get("result", {}).get("orderId", "")

    logger.info(f"Позиция закрыта: orderId={order_id}")

    return {
      "success": True,
      "orderId": order_id,
      "symbol": request.symbol,
      "closedSize": close_size,
      "closedPercent": request.percent,
      "message": f"Закрыто {request.percent}% позиции ({close_size})"
    }

  except HTTPException:
    raise

  except ValueError as e:
    logger.warning(f"API ключи не настроены: {e}")
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="API ключи не настроены"
    )

  except Exception as e:
    logger.error(f"Ошибка закрытия позиции: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )


@trading_router.post("/order/cancel")
async def cancel_order_endpoint(
    symbol: str,
    order_id: str,
    current_user: dict = Depends(require_auth)
):
  """
  Отмена ордера.

  Args:
      symbol: Торговая пара
      order_id: ID ордера

  Returns:
      dict: Результат отмены
  """
  logger.info(f"Отмена ордера: {symbol}, orderId={order_id}")

  try:
    from backend.exchange.rest_client import rest_client

    response = await rest_client.cancel_order(symbol, order_id)

    return {
      "success": True,
      "orderId": order_id,
      "symbol": symbol,
      "message": "Ордер отменен"
    }

  except Exception as e:
    logger.error(f"Ошибка отмены ордера: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=str(e)
    )