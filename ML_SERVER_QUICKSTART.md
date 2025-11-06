# ML Model Server - Quick Start Guide

## Запуск сервера

### Вариант 1: Python скрипт (рекомендуется)

**Windows:**
```bash
# Из корневой директории проекта
python run_ml_server.py
```

**Или двойной клик на:**
```
run_ml_server.bat
```

**Linux/Mac:**
```bash
python3 run_ml_server.py
```

### Вариант 2: Uvicorn напрямую

**Важно:** Запускать из **корневой директории проекта**!

```bash
# Правильно (из корня проекта):
cd C:\Users\1q\PycharmProjects\Bot_ver3_stakan_new
uvicorn backend.ml_engine.inference.model_server_v2:app --host 0.0.0.0 --port 8001

# Неправильно - НЕ работает:
uvicorn ml_engine.inference.model_server_v2:app --host 0.0.0.0 --port 8001
```

### Вариант 3: PyCharm Run Configuration

1. **Run → Edit Configurations**
2. **Add New Configuration → Python**
3. Настройки:
   - **Script path**: `C:\Users\1q\PycharmProjects\Bot_ver3_stakan_new\run_ml_server.py`
   - **Working directory**: `C:\Users\1q\PycharmProjects\Bot_ver3_stakan_new`
   - **Python interpreter**: `.venv` interpreter
4. **Apply → OK**

## Проверка работы

После запуска сервер доступен на:

```
http://localhost:8001
```

### Проверить health:
```bash
curl http://localhost:8001/api/ml/health
```

Или откройте в браузере:
```
http://localhost:8001/docs
```

## Устранение проблем

### Ошибка: `ModuleNotFoundError: No module named 'ml_engine'`

**Причина:** Неправильный путь импорта или запуск не из корневой директории.

**Решение:**
1. Убедитесь, что находитесь в корневой директории проекта
2. Используйте правильный путь: `backend.ml_engine.inference.model_server_v2:app`
3. Или используйте `run_ml_server.py` скрипт (он сам настроит пути)

### Ошибка: `ModuleNotFoundError: No module named 'backend'`

**Причина:** Python не видит корневую директорию в sys.path.

**Решение:**
```bash
# Добавьте корневую директорию в PYTHONPATH (Windows PowerShell):
$env:PYTHONPATH = "C:\Users\1q\PycharmProjects\Bot_ver3_stakan_new"

# Или используйте run_ml_server.py (автоматически настроит)
```

### Ошибка: Port 8001 уже занят

**Решение:**
```bash
# Найти процесс на порту 8001 (Windows):
netstat -ano | findstr :8001

# Убить процесс по PID:
taskkill /PID <PID> /F

# Или использовать другой порт:
uvicorn backend.ml_engine.inference.model_server_v2:app --host 0.0.0.0 --port 8002
```

## API Endpoints

После запуска доступны эндпоинты:

- `POST /api/ml/predict` - Одиночное предсказание
- `POST /api/ml/predict/batch` - Batch предсказания
- `GET /api/ml/models` - Список загруженных моделей
- `POST /api/ml/models/reload` - Перезагрузка модели
- `POST /api/ml/ab-test/create` - Создать A/B тест
- `GET /api/ml/ab-test/{id}/analyze` - Анализ эксперимента
- `GET /api/ml/health` - Health check

**Документация API:**
```
http://localhost:8001/docs
```

## Интеграция с ботом

После запуска ML Server, бот может подключиться через `ModelClient`:

```python
from backend.ml_engine.inference.model_client import get_model_client

client = get_model_client()
result = await client.predict(
    symbol="BTCUSDT",
    features=features_array
)
```

## Логи

Логи сервера сохраняются в:
```
backend/logs/ml_server.log
```

## Development Mode

Для разработки с auto-reload:

```python
# В run_ml_server.py уже включено reload=True

# Или вручную:
uvicorn backend.ml_engine.inference.model_server_v2:app \
    --host 0.0.0.0 \
    --port 8001 \
    --reload \
    --log-level debug
```

## Production Mode

Для production рекомендуется:

```bash
# С несколькими workers
uvicorn backend.ml_engine.inference.model_server_v2:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 4 \
    --log-level info

# Или через gunicorn
gunicorn backend.ml_engine.inference.model_server_v2:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8001
```
