# MLflow PostgreSQL Integration Setup

## ✅ Что изменено

MLflow теперь использует PostgreSQL вместо SQLite как backend store для:
- Experiment tracking (runs, params, metrics, tags)
- Model Registry (model versions, stages, metadata)
- Artifacts остаются на filesystem (`./mlruns/artifacts/`)

## 🔧 Конфигурация

### 1. .env настройки

Добавлены следующие переменные в `.env`:

```bash
# MLflow Tracking (PostgreSQL Backend)
MLFLOW_TRACKING_URI=postgresql://trading_bot:robocop@localhost:5432/trading_bot
MLFLOW_ARTIFACT_LOCATION=./mlruns/artifacts
MLFLOW_EXPERIMENT_NAME=trading_bot_ml
```

### 2. backend/config.py

Добавлены поля:
```python
MLFLOW_TRACKING_URI: str = Field(
    default="postgresql://trading_bot:robocop@localhost:5432/trading_bot",
    description="MLflow Tracking URI (PostgreSQL backend)"
)
MLFLOW_ARTIFACT_LOCATION: str = Field(
    default="./mlruns/artifacts",
    description="Path for MLflow artifacts storage"
)
MLFLOW_EXPERIMENT_NAME: str = Field(
    default="trading_bot_ml",
    description="Default MLflow experiment name"
)
```

### 3. MLflow Tracker

Обновлен `backend/ml_engine/mlflow_integration/mlflow_tracker.py`:
- Использует `config.MLFLOW_TRACKING_URI` по умолчанию
- Подключается к PostgreSQL вместо SQLite
- Все остальное работает без изменений

### 4. Dependencies

Добавлен в `requirements_ml.txt`:
```
psycopg2-binary>=2.9.9  # Required for MLflow PostgreSQL backend
```

## 📦 Установка

### 1. Установить зависимости

```bash
pip install -r requirements_ml.txt
```

### 2. Убедиться, что PostgreSQL запущен

**Windows:**
```bash
sc query postgresql-x64-16
```

**Linux/Mac:**
```bash
systemctl status postgresql
```

### 3. Протестировать подключение

```bash
python test_mlflow_postgres.py
```

Этот скрипт:
- Проверяет конфигурацию
- Создает тестовый эксперимент
- Логирует параметры и метрики
- Верифицирует данные в PostgreSQL

## 🚀 Использование

### Запустить обучение

```bash
python train_model.py --epochs 50
```

Все данные теперь сохраняются в PostgreSQL!

### Запустить MLflow UI

```bash
# Вариант 1: Указать backend явно
mlflow ui --backend-store-uri postgresql://trading_bot:robocop@localhost:5432/trading_bot --port 5000

# Вариант 2: Использовать переменную окружения
export MLFLOW_TRACKING_URI=postgresql://trading_bot:robocop@localhost:5432/trading_bot
mlflow ui --port 5000
```

Откройте http://localhost:5000 и увидите все эксперименты из PostgreSQL.

### Использовать в коде

```python
from backend.ml_engine.mlflow_integration import get_mlflow_tracker

# Автоматически использует PostgreSQL из config
tracker = get_mlflow_tracker()

# Все остальное работает как раньше
run_id = tracker.start_run("my_experiment")
tracker.log_params({"lr": 0.001})
tracker.log_metrics({"accuracy": 0.95})
tracker.end_run()
```

## 🔍 Проверка данных в PostgreSQL

### Подключиться к PostgreSQL

```bash
psql -h localhost -U trading_bot -d trading_bot
```

### Посмотреть таблицы MLflow

```sql
-- Список всех таблиц MLflow
\dt

-- Должны быть таблицы:
-- experiments
-- runs
-- metrics
-- params
-- tags
-- latest_metrics
-- model_versions
-- registered_models
-- и другие...
```

### Посмотреть эксперименты

```sql
SELECT experiment_id, name, lifecycle_stage, artifact_location
FROM experiments
ORDER BY creation_time DESC;
```

### Посмотреть runs

```sql
SELECT run_uuid, experiment_id, status, start_time, end_time
FROM runs
ORDER BY start_time DESC
LIMIT 10;
```

### Посмотреть метрики

```sql
SELECT r.run_uuid, m.key, m.value, m.step
FROM metrics m
JOIN runs r ON m.run_uuid = r.run_uuid
ORDER BY m.timestamp DESC
LIMIT 20;
```

## 🎯 Преимущества PostgreSQL над SQLite

1. **Concurrency**: Множественные процессы могут писать одновременно
2. **Scalability**: Лучшая производительность на большом объеме данных
3. **Reliability**: ACID транзакции, репликация, backup
4. **Production-ready**: Готово для production deployment
5. **Unified Storage**: Все данные проекта в одной БД (trading bot + ML)

## 🐛 Troubleshooting

### Ошибка: "No module named 'psycopg2'"

```bash
pip install psycopg2-binary
```

### Ошибка: "could not connect to server"

PostgreSQL не запущен. Запустите:

**Windows:**
```bash
net start postgresql-x64-16
```

**Linux:**
```bash
sudo systemctl start postgresql
```

### Ошибка: "database does not exist"

База данных создана во время init_database.py. Проверьте:

```bash
psql -h localhost -U trading_bot -l
```

Должна быть база `trading_bot`.

### Ошибка: "permission denied"

Проверьте пароль в .env:
```bash
DATABASE_URL=postgresql+asyncpg://trading_bot:robocop@localhost:5432/trading_bot
MLFLOW_TRACKING_URI=postgresql://trading_bot:robocop@localhost:5432/trading_bot
```

## 📚 Дополнительные ресурсы

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow PostgreSQL Backend](https://mlflow.org/docs/latest/tracking.html#postgresql)
- [ML_INFRASTRUCTURE_GUIDE.md](ML_INFRASTRUCTURE_GUIDE.md) - Полное руководство
- [ML_QUICK_START.md](ML_QUICK_START.md) - Быстрый старт

## ✅ Checklist

После установки проверьте:

- [ ] PostgreSQL запущен и доступен
- [ ] psycopg2-binary установлен (`pip install psycopg2-binary`)
- [ ] .env содержит MLFLOW_TRACKING_URI с PostgreSQL URI
- [ ] test_mlflow_postgres.py успешно выполнен
- [ ] MLflow UI показывает эксперименты (`mlflow ui`)
- [ ] Обучение модели работает (`python train_model.py`)

---

**Все готово! MLflow теперь использует PostgreSQL как backend store.**
