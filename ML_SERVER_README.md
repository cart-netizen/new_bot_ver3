# ML Model Server - Руководство по запуску

## 📋 Что это?

ML Model Server - это **опциональный** REST API сервер для ML предсказаний, который работает на порту 8001 и интегрируется с ML Signal Validator в основном боте.

## ⚙️ Функциональность

- **Real-time ML inference** - предсказания для торговых сигналов
- **A/B Testing** - тестирование нескольких моделей одновременно
- **Model Registry** - автоматическая загрузка production моделей
- **Health checks** - мониторинг состояния сервера
- **Batch predictions** - обработка множественных запросов

## 🚀 Как запустить

### Windows:
```bash
# Двойной клик на файл или запуск из командной строки
start_ml_server.bat
```

### Linux/Mac:
```bash
python start_ml_server.py
```

### Или напрямую:
```bash
python -m uvicorn backend.ml_engine.inference.model_server_v2:app --host 0.0.0.0 --port 8001
```

## 📍 Endpoints

После запуска доступны следующие endpoints:

### Основные
- **Health Check**: `GET http://localhost:8001/health`
- **Predict**: `POST http://localhost:8001/predict`
- **API Docs**: http://localhost:8001/docs (Swagger UI)

### Расширенные (с префиксом /api/ml/)
- **Health**: `GET http://localhost:8001/api/ml/health`
- **Predict**: `POST http://localhost:8001/api/ml/predict`
- **Batch Predict**: `POST http://localhost:8001/api/ml/predict/batch`
- **Model Info**: `GET http://localhost:8001/api/ml/models`
- **Reload Model**: `POST http://localhost:8001/api/ml/models/{name}/reload`

## 📊 Интеграция с ботом

### Автоматическая интеграция
Бот автоматически определяет доступность ML сервера:

```python
# В config.py
ML_SERVER_URL = "http://localhost:8001"  # Адрес ML сервера
ML_MIN_CONFIDENCE = 0.6  # Минимальная уверенность ML
ML_WEIGHT = 0.6  # Вес ML в гибридном решении
```

### Режимы работы:

**ML сервер запущен:**
- ✅ MLSignalValidator использует ML предсказания
- ✅ Hybrid decision: ML (60%) + Strategy (40%)
- ✅ Enhanced validation с ML метриками

**ML сервер не запущен:**
- ✅ Fallback режим - использует только стратегию
- ✅ Торговля продолжается без блокировки
- ✅ Health check каждые 30 секунд (DEBUG logs)

## 📝 Пример запроса

### Health Check
```bash
curl http://localhost:8001/health
```

Ответ:
```json
{
  "status": "healthy",
  "models_loaded": 1,
  "uptime_seconds": 123.45
}
```

### Predict
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "features": [0.1, 0.2, ...],
    "model_version": "latest"
  }'
```

Ответ:
```json
{
  "symbol": "BTCUSDT",
  "prediction": {
    "direction": "BUY",
    "confidence": 0.85,
    "expected_return": 0.012
  },
  "model_name": "hybrid_cnn_lstm",
  "model_version": "20251114_192715",
  "latency_ms": 15.3
}
```

## 🔧 Требования

### Обязательно:
- Python 3.11+
- FastAPI
- PyTorch (для ML моделей)
- uvicorn

### Проверка зависимостей:
```bash
pip install fastapi uvicorn torch numpy
```

## ⚠️ Важные замечания

1. **Опциональность**: ML сервер не обязателен для работы бота
2. **Production модель**: Сервер автоматически загружает модель в PRODUCTION stage из Model Registry
3. **Порт 8001**: Убедитесь, что порт свободен
4. **Не для браузера**: Это REST API, не веб-интерфейс (используйте /docs для UI)

## 🐛 Troubleshooting

### "No module named 'backend'"
```bash
# Запускайте из корневой директории проекта
cd /path/to/new_bot_ver3
python start_ml_server.py
```

### "Port 8001 already in use"
```bash
# Windows: найти процесс
netstat -ano | findstr :8001
taskkill /PID <PID> /F

# Linux: найти процесс
lsof -i :8001
kill -9 <PID>
```

### "No production model found"
```bash
# Обучите модель через ML Management UI или:
python train_model.py
```

## 📚 Дополнительная информация

- Логи сервера: консоль где запущен `start_ml_server.py`
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc
- Model Registry: данные в папке `models/`

## 🎯 Рекомендации

**Для тестирования:**
- Запустите ML сервер в отдельном терминале
- Проверьте health check: http://localhost:8001/health
- Запустите основной бот
- Проверьте логи: `"ML server health check: OK"`

**For production:**
- Используйте process manager (systemd, supervisor, pm2)
- Настройте auto-restart при сбоях
- Мониторьте логи и метрики
- Используйте HTTPS для внешних подключений
