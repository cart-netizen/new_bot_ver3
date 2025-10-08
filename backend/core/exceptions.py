"""
Модуль с кастомными исключениями для приложения.
Обеспечивает четкую типизацию ошибок и их обработку.
"""


class TradingBotException(Exception):
  """Базовое исключение для всех ошибок торгового бота."""

  def __init__(self, message: str, details: dict = None):
    self.message = message
    self.details = details or {}
    super().__init__(self.message)

  def __str__(self):
    if self.details:
      return f"{self.message} | Детали: {self.details}"
    return self.message


# ===== ИСКЛЮЧЕНИЯ КОНФИГУРАЦИИ =====

class ConfigurationError(TradingBotException):
  """Ошибка конфигурации приложения."""
  pass


class InvalidCredentialsError(TradingBotException):
  """Ошибка неверных учетных данных."""
  pass


# ===== ИСКЛЮЧЕНИЯ АУТЕНТИФИКАЦИИ =====

class AuthenticationError(TradingBotException):
  """Ошибка аутентификации."""
  pass


class AuthorizationError(TradingBotException):
  """Ошибка авторизации."""
  pass


class TokenExpiredError(TradingBotException):
  """Ошибка истекшего токена."""
  pass


class InvalidTokenError(TradingBotException):
  """Ошибка невалидного токена."""
  pass


# ===== ИСКЛЮЧЕНИЯ БИРЖИ =====

class ExchangeError(TradingBotException):
  """Базовое исключение для ошибок биржи."""
  pass


class ExchangeConnectionError(ExchangeError):
  """Ошибка подключения к бирже."""
  pass


class ExchangeAPIError(ExchangeError):
  """Ошибка API биржи."""

  def __init__(self, message: str, status_code: int = None, response: dict = None):
    self.status_code = status_code
    self.response = response or {}
    details = {
      "status_code": status_code,
      "response": response
    }
    super().__init__(message, details)


class RateLimitError(ExchangeError):
  """Ошибка превышения лимита запросов."""
  pass


class InvalidOrderError(ExchangeError):
  """Ошибка невалидного ордера."""
  pass


class InsufficientBalanceError(ExchangeError):
  """Ошибка недостаточного баланса."""
  pass


# ===== ИСКЛЮЧЕНИЯ WEBSOCKET =====

class WebSocketError(TradingBotException):
  """Базовое исключение для ошибок WebSocket."""
  pass


class WebSocketConnectionError(WebSocketError):
  """Ошибка подключения WebSocket."""
  pass


class WebSocketDisconnectedError(WebSocketError):
  """Ошибка разрыва WebSocket соединения."""
  pass


class WebSocketTimeoutError(WebSocketError):
  """Ошибка таймаута WebSocket."""
  pass


class WebSocketReconnectError(WebSocketError):
  """Ошибка переподключения WebSocket."""
  pass


# ===== ИСКЛЮЧЕНИЯ ДАННЫХ =====

class DataError(TradingBotException):
  """Базовое исключение для ошибок данных."""
  pass


class InvalidDataError(DataError):
  """Ошибка невалидных данных."""
  pass


class DataParsingError(DataError):
  """Ошибка парсинга данных."""
  pass


class OrderBookError(DataError):
  """Ошибка обработки стакана."""
  pass


class OrderBookSyncError(OrderBookError):
  """Ошибка синхронизации стакана."""
  pass


# ===== ИСКЛЮЧЕНИЯ СТРАТЕГИИ =====

class StrategyError(TradingBotException):
  """Базовое исключение для ошибок стратегии."""
  pass


class InvalidSignalError(StrategyError):
  """Ошибка невалидного торгового сигнала."""
  pass


class AnalysisError(StrategyError):
  """Ошибка анализа рыночных данных."""
  pass


# ===== ИСКЛЮЧЕНИЯ РИСК-МЕНЕДЖМЕНТА =====

class RiskManagementError(TradingBotException):
  """Базовое исключение для ошибок риск-менеджмента."""
  pass


class MaxPositionsExceededError(RiskManagementError):
  """Ошибка превышения максимального количества позиций."""
  pass


class MaxExposureExceededError(RiskManagementError):
  """Ошибка превышения максимальной экспозиции."""
  pass


class InvalidPositionSizeError(RiskManagementError):
  """Ошибка невалидного размера позиции."""
  pass


# ===== ИСКЛЮЧЕНИЯ ИСПОЛНЕНИЯ =====

class ExecutionError(TradingBotException):
  """Базовое исключение для ошибок исполнения."""
  pass


class OrderExecutionError(ExecutionError):
  """Ошибка исполнения ордера."""
  pass


class OrderCancellationError(ExecutionError):
  """Ошибка отмены ордера."""
  pass


# ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====

def log_exception(logger, exception: Exception, context: str = "") -> None:
  """
  Логирование исключения с контекстом.

  Args:
      logger: Объект логгера
      exception: Исключение для логирования
      context: Контекст, в котором произошла ошибка
  """
  error_message = f"{context}: {str(exception)}" if context else str(exception)

  if isinstance(exception, TradingBotException):
    if exception.details:
      logger.error(f"{error_message} | Детали: {exception.details}", exc_info=True)
    else:
      logger.error(error_message, exc_info=True)
  else:
    logger.error(error_message, exc_info=True)