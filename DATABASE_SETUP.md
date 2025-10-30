# Инструкция по настройке PostgreSQL для Trading Bot

## Проблема: Ошибка подключения к базе данных

Если вы видите ошибку:
```
OSError: Multiple exceptions: [Errno 10061] Connect call failed ('::1', 5432, 0, 0), [Errno 10061] Connect call failed ('127.0.0.1', 5432)
```

Это означает, что PostgreSQL не запущен или не установлен на вашем компьютере.

---

## Решение 1: Установка PostgreSQL на Windows

### Шаг 1: Скачать и установить PostgreSQL

1. **Скачайте PostgreSQL:**
   - Перейдите на https://www.postgresql.org/download/windows/
   - Или используйте прямую ссылку: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads
   - Выберите последнюю версию (рекомендуется PostgreSQL 15 или 16)

2. **Запустите установщик:**
   - Установите пароль для пользователя `postgres` (запомните его!)
   - Порт: оставьте `5432` (по умолчанию)
   - Locale: выберите по умолчанию или Russian

3. **Установите компоненты:**
   - PostgreSQL Server
   - pgAdmin 4 (графический интерфейс)
   - Command Line Tools

### Шаг 2: Установить TimescaleDB (опционально, для временных рядов)

Если вам нужна поддержка TimescaleDB:

1. Перейдите на https://docs.timescale.com/install/latest/self-hosted/installation-windows/
2. Следуйте инструкциям для установки расширения TimescaleDB

### Шаг 3: Создать базу данных и пользователя

#### Вариант A: Через pgAdmin (графический интерфейс)

1. Откройте **pgAdmin 4**
2. Подключитесь к серверу (введите пароль postgres)
3. **Создайте пользователя:**
   - Правый клик на `Login/Group Roles` → `Create` → `Login/Group Role`
   - Name: `trading_bot`
   - Password: `robocop`
   - Privileges: отметьте `Can login?`

4. **Создайте базу данных:**
   - Правый клик на `Databases` → `Create` → `Database`
   - Database: `trading_bot`
   - Owner: `trading_bot`

#### Вариант B: Через командную строку (psql)

1. Откройте **командную строку** или **PowerShell**
2. Подключитесь к PostgreSQL:
   ```bash
   psql -U postgres
   ```
3. Введите пароль postgres
4. Выполните команды:
   ```sql
   -- Создать пользователя
   CREATE USER trading_bot WITH PASSWORD 'robocop';

   -- Создать базу данных
   CREATE DATABASE trading_bot OWNER trading_bot;

   -- Дать права
   GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_bot;

   -- Выход
   \q
   ```

### Шаг 4: Проверить, что PostgreSQL запущен

1. Нажмите `Win + R`
2. Введите `services.msc` и нажмите Enter
3. Найдите службу **postgresql-x64-XX** (где XX - версия)
4. Убедитесь, что статус: **Запущена (Running)**
5. Если не запущена - правый клик → **Запустить**

### Шаг 5: Обновить конфигурацию приложения

Проверьте файл `.env` в корне проекта:

```env
# DATABASE (PostgreSQL + TimescaleDB)
DATABASE_URL=postgresql+asyncpg://trading_bot:robocop@localhost:5432/trading_bot
```

Формат: `postgresql+asyncpg://username:password@host:port/database`

---

## Решение 2: Использование Docker (для разработки)

Если вы используете Docker, можно быстро запустить PostgreSQL:

```bash
docker run --name trading-postgres \
  -e POSTGRES_USER=trading_bot \
  -e POSTGRES_PASSWORD=robocop \
  -e POSTGRES_DB=trading_bot \
  -p 5432:5432 \
  -d postgres:15
```

Для TimescaleDB:
```bash
docker run --name trading-timescaledb \
  -e POSTGRES_USER=trading_bot \
  -e POSTGRES_PASSWORD=robocop \
  -e POSTGRES_DB=trading_bot \
  -p 5432:5432 \
  -d timescale/timescaledb:latest-pg15
```

---

## Решение 3: Изменить настройки подключения

Если ваш PostgreSQL работает на другом хосте или порту:

### Другой порт
```env
DATABASE_URL=postgresql+asyncpg://trading_bot:robocop@localhost:5433/trading_bot
```

### Удаленный сервер
```env
DATABASE_URL=postgresql+asyncpg://trading_bot:robocop@192.168.1.100:5432/trading_bot
```

---

## Проверка подключения

После настройки, проверьте подключение:

### Вариант 1: Через psql
```bash
psql -h localhost -p 5432 -U trading_bot -d trading_bot
```

### Вариант 2: Через Python
Создайте файл `test_connection.py`:

```python
import asyncio
import asyncpg

async def test_connection():
    try:
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='trading_bot',
            password='robocop',
            database='trading_bot'
        )

        # Тестовый запрос
        result = await conn.fetchval('SELECT version()')
        print(f"✅ Подключение успешно!")
        print(f"PostgreSQL версия: {result}")

        await conn.close()
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")

asyncio.run(test_connection())
```

Запустите:
```bash
python test_connection.py
```

---

## Частые ошибки и решения

### Ошибка: "password authentication failed"

**Решение:**
- Проверьте правильность пароля в `.env`
- Убедитесь, что пользователь `trading_bot` существует
- Проверьте файл `pg_hba.conf` (должна быть строка для локальных подключений)

### Ошибка: "database does not exist"

**Решение:**
```sql
CREATE DATABASE trading_bot OWNER trading_bot;
```

### Ошибка: "role 'trading_bot' does not exist"

**Решение:**
```sql
CREATE USER trading_bot WITH PASSWORD 'robocop';
```

### PostgreSQL не запускается

**Решение:**
1. Проверьте логи: `C:\Program Files\PostgreSQL\XX\data\log\`
2. Возможно, порт 5432 занят другим приложением
3. Попробуйте переустановить PostgreSQL

---

## После успешной настройки

1. Запустите ваше приложение:
   ```bash
   python backend/main.py
   ```

2. При первом запуске автоматически создадутся таблицы

3. Проверьте логи - должно быть сообщение:
   ```
   ✓ База данных подключена успешно
   ✓ Таблицы созданы успешно
   ```

---

## Дополнительные ресурсы

- [Официальная документация PostgreSQL](https://www.postgresql.org/docs/)
- [TimescaleDB Documentation](https://docs.timescale.com/)
- [SQLAlchemy AsyncPG Documentation](https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#module-sqlalchemy.dialects.postgresql.asyncpg)

---

## Техническая поддержка

Если проблема не решается:

1. Проверьте версию PostgreSQL: `psql --version`
2. Проверьте, слушает ли PostgreSQL на порту 5432:
   ```bash
   netstat -an | findstr :5432
   ```
3. Проверьте логи приложения
4. Убедитесь, что файрвол не блокирует порт 5432
