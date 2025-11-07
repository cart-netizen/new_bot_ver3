# MLflow PostgreSQL Setup - Fix Guide

## Проблема

MLflow и основное приложение используют одну таблицу `alembic_version` в PostgreSQL, что создает конфликт:
```
ERROR: Can't locate revision identified by '003_add_layering_patterns'
```

`003_add_layering_patterns` - это миграция основного приложения, но MLflow пытается её найти в своих миграциях.

## Решение

Использовать **отдельную базу данных** для MLflow, чтобы изолировать миграции.

### Для Windows (PowerShell)

```powershell
# Запустите PowerShell от имени администратора
cd C:\Users\1q\PycharmProjects\Bot_ver3_stakan_new

# Разрешите выполнение скриптов (если нужно)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Запустите скрипт установки
.\setup_mlflow_postgres.ps1
```

**Внимание**: Вам может потребоваться ввести пароль postgres пользователя.

### Для Linux/Mac

```bash
cd /path/to/project
chmod +x setup_mlflow_postgres.sh
./setup_mlflow_postgres.sh
```

### Ручная установка

Если скрипт не работает, выполните вручную:

#### 1. Создайте базу данных MLflow

```sql
-- Подключитесь к PostgreSQL как superuser
psql -U postgres -h localhost

-- Создайте базу данных
CREATE DATABASE mlflow_tracking OWNER trading_bot;
GRANT ALL PRIVILEGES ON DATABASE mlflow_tracking TO trading_bot;

-- Подключитесь к новой базе
\c mlflow_tracking

-- Настройте права
GRANT ALL ON SCHEMA public TO trading_bot;
ALTER SCHEMA public OWNER TO trading_bot;
```

#### 2. Обновите .env файл

Найдите строку:
```env
MLFLOW_TRACKING_URI=postgresql://trading_bot:robocop@localhost:5432/trading_bot
```

Замените на:
```env
MLFLOW_TRACKING_URI=postgresql://trading_bot:robocop@localhost:5432/mlflow_tracking
```

#### 3. Перезапустите приложение

```bash
# Остановите сервер
# Ctrl+C

# Запустите снова
python -m backend.main
```

## Проверка

После настройки в логах должно быть:

```
✓ Connected to MLflow tracking URI: postgresql://trading_bot:robocop@localhost:5432/mlflow_tracking
INFO mlflow.store.db.utils: Creating initial MLflow database tables...
INFO alembic.runtime.migration: Running upgrade  -> 451aebb31d03, add metric step
...
```

**НЕ должно быть**:
```
WARNI Failed to connect to configured MLflow tracking URI
Falling back to SQLite...
```

## Структура баз данных после настройки

```
PostgreSQL
├── trading_bot              # Основное приложение
│   ├── positions
│   ├── orders
│   ├── layering_patterns    # 003_add_layering_patterns миграция
│   ├── alembic_version      # Миграции основного приложения
│   └── ...
│
└── mlflow_tracking          # MLflow (отдельная база!)
    ├── experiments
    ├── runs
    ├── metrics
    ├── params
    ├── models
    ├── alembic_version      # Миграции MLflow (изолированы!)
    └── ...
```

## Преимущества этого подхода

✅ **Полная изоляция** миграций - нет конфликтов alembic_version
✅ **Чистая архитектура** - ML tracking отдельно от бизнес-логики
✅ **Простое обслуживание** - можно удалить/пересоздать MLflow БД независимо
✅ **Безопасность** - можно настроить разные права доступа

## Откат (если что-то пошло не так)

```sql
-- Удалить MLflow базу данных
DROP DATABASE IF EXISTS mlflow_tracking;

-- В .env вернуть старое значение
MLFLOW_TRACKING_URI=postgresql://trading_bot:robocop@localhost:5432/trading_bot

-- Или использовать SQLite (temporary)
MLFLOW_TRACKING_URI=sqlite:///data/mlflow/mlruns.db
```

## Troubleshooting

### "psql command not found"

Убедитесь что PostgreSQL установлен и добавлен в PATH:
- Windows: `C:\Program Files\PostgreSQL\15\bin`
- Linux: обычно уже в PATH после установки

### "FATAL: password authentication failed"

Проверьте пароль postgres пользователя в pg_hba.conf или используйте pgAdmin для создания базы данных вручную.

### Права доступа

Если `trading_bot` пользователь не может создавать таблицы:
```sql
\c mlflow_tracking
GRANT ALL PRIVILEGES ON DATABASE mlflow_tracking TO trading_bot;
GRANT ALL ON SCHEMA public TO trading_bot;
ALTER SCHEMA public OWNER TO trading_bot;
```
