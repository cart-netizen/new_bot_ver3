# БЫСТРОЕ ИСПРАВЛЕНИЕ PostgreSQL

## Ваша ситуация
```
connection timeout expired
```
PostgreSQL не отвечает после краша. Это легко исправить!

---

## ВАРИАНТ 1: Перезапустить PostgreSQL (2 минуты)

### Самый простой способ - используйте наш скрипт:

1. **Откройте командную строку от имени Администратора:**
   - Нажмите `Win + X`
   - Выберите "Командная строка (администратор)" или "Windows PowerShell (администратор)"

2. **Перейдите в папку проекта:**
   ```cmd
   cd C:\Users\1q\PycharmProjects\Bot_ver3_stakan_new
   ```

3. **Запустите скрипт исправления:**
   ```cmd
   fix_postgres.bat
   ```

4. **Нажмите `Y` когда спросит о перезапуске**

5. **Готово!** Попробуйте подключиться через pgAdmin

---

## ВАРИАНТ 2: Перезапустить вручную (1 минута)

1. Нажмите `Win + R`
2. Введите `services.msc` и нажмите Enter
3. Найдите `postgresql-x64-15` (или другую версию)
4. Правый клик → **Перезапустить**
5. Подождите 10-15 секунд
6. Попробуйте подключиться через pgAdmin

---

## ВАРИАНТ 3: Docker (рекомендуется, если есть Docker) - 30 секунд

Если у вас установлен Docker Desktop:

```bash
# Запустить PostgreSQL в контейнере
docker-compose up -d

# Проверить статус
docker ps

# Готово! PostgreSQL работает на localhost:5432
```

**Преимущества Docker:**
- ✅ Работает изолированно от системы
- ✅ Не конфликтует со сломанным PostgreSQL
- ✅ Легко удалить и пересоздать
- ✅ Включает pgAdmin на http://localhost:5050

**Доступ к pgAdmin в Docker:**
- URL: http://localhost:5050
- Email: admin@trading.local
- Password: admin

Добавить сервер в pgAdmin:
- Host: postgres (или host.docker.internal)
- Port: 5432
- User: trading_bot
- Password: robocop

---

## ПРОВЕРКА: Работает ли PostgreSQL?

### Способ 1: Через командную строку
```cmd
netstat -ano | findstr :5432
```

Должно показать `LISTENING` → PostgreSQL работает

### Способ 2: Через pgAdmin
1. Откройте pgAdmin
2. Попробуйте подключиться к localhost:5432
3. Если подключается → ✅ Работает!

### Способ 3: Через Python
```python
import asyncio
import asyncpg

async def test():
    conn = await asyncpg.connect(
        'postgresql://trading_bot:robocop@localhost:5432/trading_bot'
    )
    print("✅ PostgreSQL работает!")
    await conn.close()

asyncio.run(test())
```

---

## ЕСЛИ НЕ ПОМОГЛО

### Проблема: Служба не запускается

**Проверьте логи:**
```
C:\Program Files\PostgreSQL\15\data\log\
```

Откройте последний файл и найдите строки с `FATAL` или `ERROR`.

### Проблема: Порт 5432 занят

**Найдите, что использует порт:**
```cmd
netstat -ano | findstr :5432
tasklist | findstr <PID>
```

**Убейте процесс (осторожно!):**
```cmd
taskkill /PID <PID> /F
```

### Проблема: Lock file exists

**Удалите файл блокировки:**
```cmd
del "C:\Program Files\PostgreSQL\15\data\postmaster.pid"
```

Затем запустите службу снова.

---

## ПОСЛЕ ИСПРАВЛЕНИЯ

1. **Создайте базу данных (если еще не создали):**

Откройте pgAdmin или psql:
```sql
CREATE USER trading_bot WITH PASSWORD 'robocop';
CREATE DATABASE trading_bot OWNER trading_bot;
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_bot;
```

2. **Проверьте .env файл:**
```env
DATABASE_URL=postgresql+asyncpg://trading_bot:robocop@localhost:5432/trading_bot
```

3. **Запустите приложение:**
```cmd
python backend/main.py
```

Должно появиться:
```
✓ База данных подключена успешно
✓ Таблицы созданы успешно
```

---

## ДОПОЛНИТЕЛЬНАЯ ПОМОЩЬ

- Подробная инструкция: `DATABASE_SETUP.md`
- Решение проблем после краша: `FIX_POSTGRESQL_CRASH.md`
- Docker инструкция: см. `docker-compose.yml`

---

## Резюме

```
┌─────────────────────────────────────┐
│  1. Запустить fix_postgres.bat     │
│  2. Нажать Y для перезапуска        │
│  3. Подождать 10 секунд             │
│  4. Проверить через pgAdmin         │
│  5. Запустить приложение            │
└─────────────────────────────────────┘
```

**Если всё ещё не работает → используйте Docker (Вариант 3)**
