# Исправление проблемы с PostgreSQL после краша

## Проблема
```
connection timeout expired
Multiple connection attempts failed
```

Это означает, что PostgreSQL не отвечает. Возможные причины:
1. Служба PostgreSQL не запущена
2. Служба зависла после краша
3. Порт 5432 занят или заблокирован
4. PostgreSQL поврежден

---

## БЫСТРОЕ РЕШЕНИЕ (попробуйте сначала это)

### Шаг 1: Перезапустить службу PostgreSQL

#### Способ A: Через Службы Windows

1. Нажмите `Win + R`
2. Введите `services.msc` и нажмите Enter
3. Найдите службу **postgresql-x64-XX** (где XX - версия, например 15 или 16)
4. Проверьте статус:
   - Если статус "Запущена" → Правый клик → **Перезапустить**
   - Если статус "Остановлена" → Правый клик → **Запустить**
5. Если ошибка при запуске → переходите к Шагу 2

#### Способ B: Через командную строку (запустите от имени Администратора)

```cmd
REM Остановить службу
net stop postgresql-x64-15

REM Подождать 5 секунд
timeout /t 5

REM Запустить службу
net start postgresql-x64-15
```

Замените `15` на вашу версию PostgreSQL.

### Шаг 2: Проверить, что порт 5432 свободен

Откройте PowerShell или командную строку:

```cmd
netstat -ano | findstr :5432
```

**Что означают результаты:**

- **Нет вывода** → порт свободен, PostgreSQL не запущен
- **LISTENING** → PostgreSQL работает
- **Другой PID** → порт занят другим процессом

Если порт занят другим процессом:
```cmd
REM Узнать, какая программа использует порт
tasklist | findstr <PID>

REM Завершить процесс (только если уверены!)
taskkill /PID <PID> /F
```

---

## ЕСЛИ СЛУЖБА НЕ ЗАПУСКАЕТСЯ

### Шаг 3: Проверить логи PostgreSQL

Логи обычно находятся здесь:
```
C:\Program Files\PostgreSQL\15\data\log\
```

Или:
```
C:\Program Files\PostgreSQL\15\data\pg_log\
```

**Откройте последний файл лога** и найдите ошибки.

### Частые проблемы в логах:

#### Проблема 1: "could not create shared memory segment"
```
FATAL:  could not create shared memory segment: Invalid argument
```

**Решение:**
1. Откройте файл `postgresql.conf`:
   ```
   C:\Program Files\PostgreSQL\15\data\postgresql.conf
   ```
2. Найдите и измените:
   ```
   shared_buffers = 32MB
   ```
3. Перезапустите службу

#### Проблема 2: "database system was interrupted"
```
LOG:  database system was interrupted; last known up at 2025-10-30
```

**Решение:** PostgreSQL восстанавливается после краша
1. Дайте службе время (2-5 минут)
2. Проверьте снова

#### Проблема 3: "data directory is locked"
```
FATAL:  lock file "postmaster.pid" already exists
```

**Решение:**
1. Убедитесь, что PostgreSQL действительно не запущен
2. Удалите файл блокировки:
   ```
   C:\Program Files\PostgreSQL\15\data\postmaster.pid
   ```
3. Запустите службу снова

#### Проблема 4: "could not bind IPv4 address"
```
FATAL:  could not bind IPv4 address "127.0.0.1": Permission denied
```

**Решение:** Порт занят или нужны права администратора
- Закройте другие программы, использующие порт 5432
- Перезагрузите компьютер
- Запустите PostgreSQL с правами администратора

---

## ЕСЛИ НИЧЕГО НЕ ПОМОГАЕТ

### Вариант 1: Полный сброс PostgreSQL

**⚠️ ВНИМАНИЕ: Это удалит все данные в базе!**

1. Остановите службу PostgreSQL
2. Создайте резервную копию папки данных:
   ```cmd
   xcopy "C:\Program Files\PostgreSQL\15\data" "C:\PostgreSQL_backup" /E /I
   ```
3. Удалите папку данных:
   ```cmd
   rmdir /S /Q "C:\Program Files\PostgreSQL\15\data"
   ```
4. Переинициализируйте базу:
   ```cmd
   "C:\Program Files\PostgreSQL\15\bin\initdb.exe" -D "C:\Program Files\PostgreSQL\15\data" -U postgres -A md5 -E UTF8
   ```
5. Запустите службу
6. Создайте пользователя и базу снова:
   ```sql
   CREATE USER trading_bot WITH PASSWORD 'robocop';
   CREATE DATABASE trading_bot OWNER trading_bot;
   ```

### Вариант 2: Переустановить PostgreSQL

1. **Остановите службу PostgreSQL**

2. **Удалите PostgreSQL:**
   - Панель управления → Программы и компоненты
   - Найдите PostgreSQL → Удалить

3. **Удалите остатки:**
   ```
   C:\Program Files\PostgreSQL\
   C:\Users\<ваш_пользователь>\AppData\Roaming\postgresql\
   ```

4. **Перезагрузите компьютер**

5. **Установите PostgreSQL заново:**
   - https://www.enterprisedb.com/downloads/postgres-postgresql-downloads
   - Следуйте инструкциям из `DATABASE_SETUP.md`

### Вариант 3: Использовать Docker (БЫСТРОЕ РЕШЕНИЕ)

Если нужно срочно запустить приложение:

```bash
# Остановить старый контейнер (если есть)
docker stop trading-postgres
docker rm trading-postgres

# Запустить новый PostgreSQL в Docker
docker run --name trading-postgres \
  -e POSTGRES_USER=trading_bot \
  -e POSTGRES_PASSWORD=robocop \
  -e POSTGRES_DB=trading_bot \
  -p 5432:5432 \
  -d postgres:15

# Проверить статус
docker ps

# Проверить логи
docker logs trading-postgres
```

**Преимущества Docker:**
- Запускается за 10 секунд
- Изолирован от системы
- Легко удалить и пересоздать
- Не конфликтует с другими установками

---

## ПРОВЕРКА ПОСЛЕ ИСПРАВЛЕНИЯ

### Тест 1: Проверить службу
```cmd
sc query postgresql-x64-15
```

Должно быть:
```
STATE: 4 RUNNING
```

### Тест 2: Проверить порт
```cmd
netstat -ano | findstr :5432
```

Должно быть:
```
TCP    0.0.0.0:5432           0.0.0.0:0              LISTENING       <PID>
```

### Тест 3: Подключиться через psql
```cmd
psql -h localhost -U postgres -c "SELECT version();"
```

### Тест 4: Подключиться через pgAdmin
1. Откройте pgAdmin
2. Создайте новое подключение:
   - Host: localhost
   - Port: 5432
   - User: postgres
   - Password: (ваш пароль)

### Тест 5: Из Python
Создайте `test_db.py`:
```python
import asyncio
import asyncpg

async def test():
    try:
        conn = await asyncpg.connect(
            'postgresql://postgres:YOUR_PASSWORD@localhost:5432/postgres'
        )
        version = await conn.fetchval('SELECT version()')
        print(f"✅ PostgreSQL работает!")
        print(f"Версия: {version[:50]}...")
        await conn.close()
    except Exception as e:
        print(f"❌ Ошибка: {e}")

asyncio.run(test())
```

Запустите:
```cmd
python test_db.py
```

---

## АЛЬТЕРНАТИВА: Временно использовать SQLite

Если PostgreSQL не работает и нужно срочно тестировать приложение:

### Вариант A: Добавить поддержку SQLite

Измените в `.env`:
```env
# Временно закомментировать PostgreSQL
# DATABASE_URL=postgresql+asyncpg://trading_bot:robocop@localhost:5432/trading_bot

# Использовать SQLite
DATABASE_URL=sqlite+aiosqlite:///./trading_bot.db
```

**⚠️ Важно:** SQLite не поддерживает TimescaleDB и некоторые продвинутые функции PostgreSQL. Это только для разработки/тестирования.

---

## ДИАГНОСТИКА: Команды для сбора информации

Если проблема не решается, соберите информацию:

```cmd
REM 1. Версия PostgreSQL
psql --version

REM 2. Статус службы
sc query postgresql-x64-15

REM 3. Процессы PostgreSQL
tasklist | findstr postgres

REM 4. Что слушает на порту 5432
netstat -ano | findstr :5432

REM 5. Проверка файрвола
netsh advfirewall firewall show rule name=all | findstr 5432

REM 6. Последние 20 строк лога PostgreSQL
powershell "Get-Content 'C:\Program Files\PostgreSQL\15\data\log\*.log' -Tail 20"
```

Отправьте вывод этих команд для дальнейшей диагностики.

---

## Быстрая памятка

```
Служба не запускается   → Проверить логи (data/log/*.log)
Timeout expired        → Проверить, запущена ли служба
Connection refused     → Служба остановлена, запустить
Port already in use    → Найти процесс (netstat), убить его
Lock file exists       → Удалить postmaster.pid
```

---

## Следующие шаги

После того как PostgreSQL заработает:

1. Создайте пользователя и базу (если еще не создали)
2. Обновите `.env` с правильными настройками
3. Запустите ваше приложение
4. Проверьте, что таблицы создаются автоматически

Если проблема сохраняется, пришлите:
- Логи PostgreSQL
- Вывод `sc query postgresql-x64-15`
- Вывод `netstat -ano | findstr :5432`
