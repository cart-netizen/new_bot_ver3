"""
Константы для приложения.
"""

from enum import Enum


# ===== BYBIT ENDPOINTS =====

class BybitEndpoints:
  """URLs эндпоинтов Bybit API."""

  # REST API
  MAINNET_REST = "https://api.bybit.com"
  TESTNET_REST = "https://api-testnet.bybit.com"

  # WebSocket API - LINEAR для фьючерсов
  MAINNET_WS_PUBLIC = "wss://stream.bybit.com/v5/public/linear"
  TESTNET_WS_PUBLIC = "wss://stream-testnet.bybit.com/v5/public/linear"

  MAINNET_WS_PRIVATE = "wss://stream.bybit.com/v5/private"
  TESTNET_WS_PRIVATE = "wss://stream-testnet.bybit.com/v5/private"


# ===== API PATHS =====

class BybitAPIPaths:
  """Пути к API эндпоинтам Bybit."""

  # Рыночные данные
  SERVER_TIME = "/v5/market/time"
  ORDERBOOK = "/v5/market/orderbook"
  TICKERS = "/v5/market/tickers"
  KLINE = "/v5/market/kline"
  RECENT_TRADES = "/v5/market/recent-trade"
  INSTRUMENTS_INFO = "/v5/market/instruments-info"

  # Торговля
  PLACE_ORDER = "/v5/order/create"
  AMEND_ORDER = "/v5/order/amend"
  CANCEL_ORDER = "/v5/order/cancel"
  CANCEL_ALL_ORDERS = "/v5/order/cancel-all"
  GET_OPEN_ORDERS = "/v5/order/realtime"
  GET_ORDER_HISTORY = "/v5/order/history"

  # Позиции
  GET_POSITIONS = "/v5/position/list"
  SET_LEVERAGE = "/v5/position/set-leverage"
  SET_TRADING_STOP = "/v5/position/trading-stop"

  # Аккаунт
  GET_WALLET_BALANCE = "/v5/account/wallet-balance"
  GET_TRANSACTION_LOG = "/v5/account/transaction-log"

  # Пользовательские данные
  GET_API_KEY_INFO = "/v5/user/query-api"


# ===== WEBSOCKET TOPICS =====

class BybitWSTopics:
  """WebSocket топики Bybit."""

  # Публичные топики
  ORDERBOOK = "orderbook"
  ORDERBOOK_DEPTH_1 = "orderbook.1"
  ORDERBOOK_DEPTH_50 = "orderbook.50"
  ORDERBOOK_DEPTH_200 = "orderbook.200"
  ORDERBOOK_DEPTH_500 = "orderbook.500"

  TRADES = "publicTrade"
  TICKERS = "tickers"
  KLINE = "kline"

  # Приватные топики
  POSITION = "position"
  EXECUTION = "execution"
  ORDER = "order"
  WALLET = "wallet"

  @staticmethod
  def get_orderbook_topic(symbol: str, depth: int = 200) -> str:
    """
    Получение топика стакана для символа.

    Args:
        symbol: Торговая пара
        depth: Глубина стакана (1, 50, 200, 500)

    Returns:
        str: Название топика
    """
    return f"orderbook.{depth}.{symbol}"

  @staticmethod
  def get_trades_topic(symbol: str) -> str:
    """
    Получение топика сделок для символа.

    Args:
        symbol: Торговая пара

    Returns:
        str: Название топика
    """
    return f"publicTrade.{symbol}"

  @staticmethod
  def get_ticker_topic(symbol: str) -> str:
    """
    Получение топика тикера для символа.

    Args:
        symbol: Торговая пара

    Returns:
        str: Название топика
    """
    return f"tickers.{symbol}"


# ===== ТАЙМАУТЫ И ЛИМИТЫ =====

class Timeouts:
  """Таймауты для различных операций."""

  HTTP_REQUEST = 10  # секунд
  WS_CONNECT = 10  # секунд
  WS_MESSAGE = 5  # секунд
  WS_PING_INTERVAL = 20  # секунд
  WS_PONG_TIMEOUT = 30  # секунд

  # Переподключение
  RECONNECT_DELAY_MIN = 1  # секунд
  RECONNECT_DELAY_MAX = 60  # секунд
  RECONNECT_MAX_ATTEMPTS = 10


class RateLimits:
  """Лимиты запросов."""

  REST_REQUESTS_PER_SECOND = 10
  REST_REQUESTS_PER_MINUTE = 120

  WS_MAX_CONNECTIONS = 10
  WS_MAX_SUBSCRIPTIONS_PER_CONNECTION = 10


# ===== PRECISION =====

class Precision:
  """Точность для различных вычислений."""

  PRICE_DECIMALS = 8
  QUANTITY_DECIMALS = 8
  PERCENTAGE_DECIMALS = 2


# ===== СТАТУСЫ =====

class BotStatus(str, Enum):
  """Статус работы бота."""

  STOPPED = "stopped"
  STARTING = "starting"
  RUNNING = "running"
  STOPPING = "stopping"
  ERROR = "error"


class ConnectionStatus(str, Enum):
  """Статус подключения."""

  DISCONNECTED = "disconnected"
  CONNECTING = "connecting"
  CONNECTED = "connected"
  RECONNECTING = "reconnecting"
  ERROR = "error"


# ===== КАТЕГОРИИ BYBIT =====

class BybitCategory(str, Enum):
  """Категории продуктов Bybit."""

  SPOT = "spot"
  LINEAR = "linear"
  INVERSE = "inverse"
  OPTION = "option"


# ===== ИНТЕРВАЛЫ =====

class KlineInterval(str, Enum):
  """Интервалы для свечей."""

  MIN_1 = "1"
  MIN_3 = "3"
  MIN_5 = "5"
  MIN_15 = "15"
  MIN_30 = "30"
  HOUR_1 = "60"
  HOUR_2 = "120"
  HOUR_4 = "240"
  HOUR_6 = "360"
  HOUR_12 = "720"
  DAY_1 = "D"
  WEEK_1 = "W"
  MONTH_1 = "M"


# ===== СООБЩЕНИЯ =====

class Messages:
  """Стандартные сообщения."""

  # Успех
  SUCCESS_AUTH = "Успешная аутентификация"
  SUCCESS_CONNECTION = "Успешное подключение"
  SUCCESS_SUBSCRIPTION = "Успешная подписка"

  # Ошибки
  ERROR_AUTH = "Ошибка аутентификации"
  ERROR_CONNECTION = "Ошибка подключения"
  ERROR_SUBSCRIPTION = "Ошибка подписки"
  ERROR_INVALID_DATA = "Невалидные данные"
  ERROR_RATE_LIMIT = "Превышен лимит запросов"

  # Предупреждения
  WARNING_RECONNECTING = "Переподключение..."
  WARNING_CONNECTION_LOST = "Соединение потеряно"


# ===== REGEX PATTERNS =====

class Patterns:
  """Регулярные выражения для валидации."""

  SYMBOL = r"^[A-Z]{2,10}USDT$"  # Например: BTCUSDT
  ORDER_ID = r"^[a-f0-9-]{36}$"  # UUID формат
  API_KEY = r"^[A-Za-z0-9]{20,}$"


# ===== ЦВЕТА ДЛЯ ЛОГИРОВАНИЯ =====

class Colors:
  """ANSI цветовые коды."""

  RESET = '\033[0m'
  BOLD = '\033[1m'

  BLACK = '\033[30m'
  RED = '\033[31m'
  GREEN = '\033[32m'
  YELLOW = '\033[33m'
  BLUE = '\033[34m'
  MAGENTA = '\033[35m'
  CYAN = '\033[36m'
  WHITE = '\033[37m'

  BG_BLACK = '\033[40m'
  BG_RED = '\033[41m'
  BG_GREEN = '\033[42m'
  BG_YELLOW = '\033[43m'
  BG_BLUE = '\033[44m'
  BG_MAGENTA = '\033[45m'
  BG_CYAN = '\033[46m'
  BG_WHITE = '\033[47m'

# Константа по умолчанию для всего приложения
DEFAULT_CATEGORY = BybitCategory.LINEAR