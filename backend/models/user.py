"""
Модели данных для пользователей и аутентификации.
"""
from dataclasses import field

from pydantic import BaseModel, Field, validator, field_validator


class LoginRequest(BaseModel):
  """Модель запроса на вход."""

  password: str = Field(
    ...,
    min_length=8,
    description="Пароль для входа в приложение"
  )


class LoginResponse(BaseModel):
  """Модель ответа на успешный вход."""

  access_token: str = Field(..., description="JWT токен доступа")
  token_type: str = Field(default="bearer", description="Тип токена")
  expires_in: int = Field(..., description="Время жизни токена в секундах")


class ChangePasswordRequest(BaseModel):
  """Модель запроса на изменение пароля."""

  old_password: str = Field(
    ...,
    min_length=8,
    description="Текущий пароль"
  )
  new_password: str = Field(
    ...,
    min_length=8,
    description="Новый пароль"
  )

  @field_validator("new_password")
  def passwords_must_differ(cls, v, values):
    """Проверка, что новый пароль отличается от старого."""
    if "old_password" in values and v == values["old_password"]:
      raise ValueError("Новый пароль должен отличаться от старого")
    return v


class TokenPayload(BaseModel):
  """Модель данных JWT токена."""

  sub: str = Field(..., description="Subject (идентификатор пользователя)")
  exp: int = Field(..., description="Expiration time (время истечения)")
  iat: int = Field(..., description="Issued at (время выпуска)")
  type: str = Field(default="access", description="Тип токена")