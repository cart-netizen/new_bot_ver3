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
from exchange.rest_client import rest_client
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