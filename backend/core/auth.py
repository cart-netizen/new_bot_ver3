"""
Модуль аутентификации и авторизации пользователей.
Обеспечивает защиту API через JWT токены и хеширование паролей.
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.config import settings
from core.exceptions import (
  AuthenticationError,
  TokenExpiredError,
  InvalidTokenError
)
from core.logger import get_logger

logger = get_logger(__name__)

# Контекст для хеширования паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Схема безопасности для FastAPI
security = HTTPBearer()


class PasswordHasher:
  """Класс для работы с хешированием паролей."""

  @staticmethod
  def hash_password(password: str) -> str:
    """
    Хеширование пароля.

    Args:
        password: Пароль в открытом виде

    Returns:
        str: Хешированный пароль
    """
    return pwd_context.hash(password)

  @staticmethod
  def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Проверка пароля.

    Args:
        plain_password: Пароль в открытом виде
        hashed_password: Хешированный пароль

    Returns:
        bool: True если пароль верный
    """
    return pwd_context.verify(plain_password, hashed_password)


class TokenManager:
  """Класс для работы с JWT токенами."""

  @staticmethod
  def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Создание JWT токена доступа.

    Args:
        data: Данные для включения в токен
        expires_delta: Время жизни токена

    Returns:
        str: JWT токен
    """
    to_encode = data.copy()

    if expires_delta:
      expire = datetime.utcnow() + expires_delta
    else:
      expire = datetime.utcnow() + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
      )

    to_encode.update({
      "exp": expire,
      "iat": datetime.utcnow(),
      "type": "access"
    })

    encoded_jwt = jwt.encode(
      to_encode,
      settings.SECRET_KEY,
      algorithm=settings.ALGORITHM
    )

    logger.info(f"Создан новый токен доступа с истечением: {expire}")
    return encoded_jwt

  @staticmethod
  def verify_token(token: str) -> dict:
    """
    Проверка и декодирование JWT токена.

    Args:
        token: JWT токен

    Returns:
        dict: Декодированные данные токена

    Raises:
        TokenExpiredError: Если токен истек
        InvalidTokenError: Если токен невалидный
    """
    try:
      payload = jwt.decode(
        token,
        settings.SECRET_KEY,
        algorithms=[settings.ALGORITHM]
      )

      # Проверяем тип токена
      if payload.get("type") != "access":
        raise InvalidTokenError("Неверный тип токена")

      logger.debug(f"Токен успешно верифицирован для пользователя: {payload.get('sub')}")
      return payload

    except jwt.ExpiredSignatureError:
      logger.warning("Попытка использования истекшего токена")
      raise TokenExpiredError("Срок действия токена истек")
    except JWTError as e:
      logger.warning(f"Ошибка валидации токена: {e}")
      raise InvalidTokenError(f"Невалидный токен: {str(e)}")


class AuthService:
  """Сервис аутентификации пользователей."""

  # Хешированный пароль хранится в памяти
  # В production среде следует использовать базу данных
  _hashed_password: Optional[str] = None

  @classmethod
  def initialize(cls):
    """Инициализация сервиса аутентификации."""
    cls._hashed_password = PasswordHasher.hash_password(settings.APP_PASSWORD)
    logger.info("Сервис аутентификации инициализирован")

  @classmethod
  def authenticate(cls, password: str) -> dict:
    """
    Аутентификация пользователя по паролю.

    Args:
        password: Пароль пользователя

    Returns:
        dict: Данные токена доступа

    Raises:
        AuthenticationError: Если пароль неверный
    """
    if cls._hashed_password is None:
      cls.initialize()

    if not PasswordHasher.verify_password(password, cls._hashed_password):
      logger.warning("Неудачная попытка входа с неверным паролем")
      raise AuthenticationError("Неверный пароль")

    # Создаем токен
    access_token = TokenManager.create_access_token(
      data={"sub": "bot_user"}
    )

    logger.info("Успешная аутентификация пользователя")

    return {
      "access_token": access_token,
      "token_type": "bearer",
      "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60  # в секундах
    }

  @classmethod
  def change_password(cls, old_password: str, new_password: str) -> bool:
    """
    Изменение пароля.

    Args:
        old_password: Старый пароль
        new_password: Новый пароль

    Returns:
        bool: True если пароль успешно изменен

    Raises:
        AuthenticationError: Если старый пароль неверный
    """
    if cls._hashed_password is None:
      cls.initialize()

    # Проверяем старый пароль
    if not PasswordHasher.verify_password(old_password, cls._hashed_password):
      logger.warning("Неудачная попытка смены пароля с неверным старым паролем")
      raise AuthenticationError("Неверный старый пароль")

    # Валидация нового пароля
    if len(new_password) < 8:
      raise ValueError("Новый пароль должен содержать минимум 8 символов")

    # Устанавливаем новый пароль
    cls._hashed_password = PasswordHasher.hash_password(new_password)

    logger.info("Пароль успешно изменен")
    return True


# ===== ЗАВИСИМОСТИ ДЛЯ FASTAPI =====

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
  """
  Зависимость FastAPI для получения текущего пользователя из токена.

  Args:
      credentials: HTTP авторизационные данные

  Returns:
      dict: Данные пользователя из токена

  Raises:
      HTTPException: Если токен невалидный или истек
  """
  try:
    token = credentials.credentials
    payload = TokenManager.verify_token(token)
    return payload

  except TokenExpiredError:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Срок действия токена истек",
      headers={"WWW-Authenticate": "Bearer"},
    )
  except InvalidTokenError as e:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail=str(e),
      headers={"WWW-Authenticate": "Bearer"},
    )
  except Exception as e:
    logger.error(f"Неожиданная ошибка при проверке токена: {e}")
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Не удалось проверить учетные данные",
      headers={"WWW-Authenticate": "Bearer"},
    )


def require_auth(current_user: dict = Depends(get_current_user)) -> dict:
  """
  Зависимость FastAPI для требования аутентификации.

  Args:
      current_user: Текущий пользователь из токена

  Returns:
      dict: Данные пользователя
  """
  return current_user


# Инициализация при импорте модуля
AuthService.initialize()