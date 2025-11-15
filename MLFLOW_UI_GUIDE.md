# MLflow UI Guide

## Что такое MLflow UI?

MLflow UI - это веб-интерфейс для просмотра и управления экспериментами машинного обучения:
- 📊 Просмотр всех training runs и метрик
- 📈 Сравнение экспериментов и параметров
- 🔍 Поиск лучших моделей по метрикам
- 📦 Управление версиями моделей
- 📁 Доступ к artifacts (графики, логи)

## Запуск MLflow UI

### Windows:
```bash
# Вариант 1: Двойной клик на файл
start_mlflow_ui.bat

# Вариант 2: Из командной строки
python start_mlflow_ui.py
```

### Linux/macOS:
```bash
python3 start_mlflow_ui.py
```

## Доступ к UI

После запуска откройте браузер:
```
http://localhost:5000
```

Или используйте кнопку "Open MLflow UI" во фронтенде (вкладка MLflow).

## Основные возможности UI

### 1. Просмотр Experiments
- Список всех экспериментов
- Фильтрация по параметрам и метрикам
- Сравнение runs

### 2. Детали Run
Для каждого training run:
- **Parameters**: epochs, batch_size, learning_rate
- **Metrics**: accuracy, precision, recall, F1, loss
- **Artifacts**: модели (.pth, .onnx), графики, конфиги
- **Tags**: version, auto_promoted, data_source

### 3. Model Registry
- Все зарегистрированные модели
- Версии моделей
- Stages: None → Staging → Production → Archived
- История переходов между stages

### 4. Сравнение Runs
- Parallel Coordinates Plot
- Scatter Plot матрицы метрик
- Таблица со всеми параметрами

## Полезные фильтры

### Найти лучшие runs по accuracy:
```
metrics.val_accuracy > 0.85
```

### Найти runs с конкретным data source:
```
tags.data_source = "feature_store"
```

### Runs за последние 7 дней:
```
attributes.start_time > "2025-11-08"
```

## Интеграция с Backend

MLflow UI подключается к той же PostgreSQL базе данных, что и backend:
- **Tracking URI**: `postgresql://trading_bot:robocop@localhost:5432/trading_bot`
- **Artifact Location**: `./mlruns/artifacts/`

Все training runs, запущенные через фронтенд или скрипты, автоматически появляются в MLflow UI.

## Автоматический запуск (опционально)

### Windows - Task Scheduler:
1. Открыть Task Scheduler
2. Create Basic Task
3. Trigger: "At startup"
4. Action: Start program → `start_mlflow_ui.bat`

### Linux - systemd:
```bash
# Создать /etc/systemd/system/mlflow-ui.service
sudo systemctl enable mlflow-ui
sudo systemctl start mlflow-ui
```

## Troubleshooting

### Ошибка: "Connection refused"
- MLflow UI server не запущен
- Запустите `start_mlflow_ui.bat` или `start_mlflow_ui.py`

### Ошибка: "Database connection failed"
- PostgreSQL не запущен
- Проверьте подключение к базе данных
- Убедитесь, что база `trading_bot` существует

### Порт 5000 занят
Измените порт в `start_mlflow_ui.py`:
```python
"--port", "5001"  # Вместо 5000
```

И обновите URL во фронтенде в `MLManagementPage.tsx`:
```typescript
href="http://localhost:5001"  // Вместо 5000
```

## Остановка MLflow UI

Нажмите **Ctrl+C** в окне терминала, где запущен MLflow UI.

## Дополнительные ресурсы

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
