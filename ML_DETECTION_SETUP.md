# ML Detection Components - Setup Guide

## 📋 Содержание

1. [Автоматическая Инициализация](#автоматическая-инициализация)
2. [Установка Зависимостей](#установка-зависимостей)
3. [Структура Данных](#структура-данных)
4. [Первый Запуск](#первый-запуск)
5. [Проверка Работоспособности](#проверка-работоспособности)

---

## ✅ Автоматическая Инициализация

**Хорошая новость:** Все базы данных и директории создаются **автоматически** при первом запуске!

### Что происходит автоматически:

#### 1. **Pattern Database (SQLite)**
```python
# backend/ml_engine/detection/pattern_database.py:101-104
Path(db_path).parent.mkdir(parents=True, exist_ok=True)  # Создаёт data/
self._init_database()  # Создаёт таблицы автоматически
```

**Создаётся:**
- 📁 `data/` директория
- 📊 `data/layering_patterns.db` - SQLite база данных
- 🏗️ Таблицы: `patterns`, `pattern_statistics`
- 📇 Индексы: `idx_fingerprint_hash`, `idx_blacklist`, `idx_last_seen`

#### 2. **ML Data Collector (Parquet)**
```python
# backend/ml_engine/detection/layering_data_collector.py:132
self.data_dir.mkdir(parents=True, exist_ok=True)  # Создаёт директорию
```

**Создаётся:**
- 📁 `data/ml_training/layering/` директория
- 📦 Parquet файлы: `layering_data_YYYYMMDD_HHMMSS.parquet`

#### 3. **Adaptive ML Model**
```python
# backend/ml_engine/detection/adaptive_layering_model.py
# Загружает существующую модель если есть, иначе работает в режиме сбора данных
```

**Создаётся (после обучения):**
- 📁 `data/models/` директория
- 🧠 `data/models/layering_adaptive_v1.pkl` - обученная модель

---

## 📦 Установка Зависимостей

### 1. Обновите зависимости

```bash
cd /home/user/new_bot_ver3
pip install -r requirements.txt
```

### 2. Новая зависимость: pyarrow

**Добавлено в requirements.txt:**
```
pyarrow==18.1.0
```

**Что это даёт:**
- Эффективное хранение ML данных в формате Parquet
- Быстрая загрузка больших датасетов (10x быстрее CSV)
- Сжатие данных (snappy compression)
- Совместимость с pandas, sklearn, PyTorch

### 3. Проверьте установку

```bash
python -c "import pyarrow; import sklearn; import pandas; print('✅ All dependencies OK')"
```

**Ожидаемый вывод:**
```
✅ All dependencies OK
```

---

## 📂 Структура Данных

После первого запуска будет создана следующая структура:

```
backend/
├── data/
│   ├── layering_patterns.db              # SQLite база паттернов
│   │   ├── patterns table                # Исторические паттерны
│   │   └── pattern_statistics table      # Статистика
│   │
│   ├── ml_training/
│   │   └── layering/
│   │       ├── layering_data_20251031_120000.parquet
│   │       ├── layering_data_20251031_130000.parquet
│   │       └── ...                       # ML training samples
│   │
│   └── models/
│       └── layering_adaptive_v1.pkl      # Обученная модель (создаётся после обучения)
```

---

## 🚀 Первый Запуск

### Режим 1: ONLY_TRAINING (сбор данных без торговли)

**В config/.env:**
```bash
TRADING_MODE=ONLY_TRAINING
```

**Что происходит:**
```
1. ✅ SQLite база создаётся автоматически
2. ✅ Директория data/ml_training/layering/ создаётся автоматически
3. 📊 Layering Detector начинает работать
4. 💾 Данные собираются в память (buffer)
5. 💾 Каждые 100 samples → сохранение в Parquet
6. 🧠 ML модель НЕ используется (ещё не обучена)
```

### Режим 2: Full Trading (торговля + сбор данных)

**В config/.env:**
```bash
TRADING_MODE=FULL
```

**Что происходит:**
```
1. ✅ Всё как в ONLY_TRAINING
2. 🤖 Бот торгует по сигналам
3. 📈 Layering Detection работает в production
4. 💾 Данные продолжают собираться
5. 🧠 ML модель используется (если обучена)
```

---

## 🔍 Проверка Работоспособности

### 1. Проверьте создание файлов

```bash
# Запустите бота
python backend/main.py

# В другом терминале проверьте:
ls -lh backend/data/
ls -lh backend/data/ml_training/layering/
```

**Ожидаемый вывод:**
```
backend/data/layering_patterns.db          # 12KB - 1MB
backend/data/ml_training/layering/*.parquet # Появятся после 100+ samples
```

### 2. Проверьте логи

```bash
tail -f logs/bot.log | grep -i "pattern\|collector\|adaptive"
```

**Ожидаемые сообщения:**
```
✅ HistoricalPatternDatabase initialized: data/layering_patterns.db
   Loaded 0 patterns from database

✅ LayeringDataCollector initialized: data/ml_training/layering
   Total samples collected: 0

⚠️  AdaptiveLayeringModel: No trained model found, working in data collection mode
```

### 3. Проверьте через API

```bash
# Проверьте статус Pattern Database
curl -X GET "http://localhost:8000/api/detection/patterns/statistics" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Проверьте Data Collector
curl -X GET "http://localhost:8000/api/ml/data-collector/statistics" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Проверьте ML модель
curl -X GET "http://localhost:8000/api/ml/adaptive-model/status" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Ожидаемый вывод (первый запуск):**
```json
{
  "total_patterns": 0,
  "blacklisted_patterns": 0,
  "unique_symbols": 0,
  "avg_success_rate": 0
}

{
  "enabled": true,
  "buffer_size": 0,
  "total_collected": 0,
  "labeled_samples": 0,
  "unlabeled_samples": 0
}

{
  "enabled": true,
  "is_trained": false,
  "model_version": null,
  "feature_count": 24
}
```

---

## 🎓 Обучение ML Модели

### Когда можно обучать?

**Минимум:** 100 labeled samples
**Рекомендуется:** 500-1000 labeled samples
**Оптимально:** 5000+ labeled samples

### Шаги обучения

#### 1. Соберите данные (1-7 дней)
```bash
# Запустите бота в ONLY_TRAINING или FULL режиме
python backend/main.py

# Подождите пока соберётся 100+ samples
```

#### 2. Проверьте количество данных
```bash
curl -X GET "http://localhost:8000/api/ml/data-collector/labeled-data" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### 3. Запустите обучение
```bash
cd backend
python scripts/train_layering_model.py
```

**Вывод:**
```
📊 Training Layering Detection Model...

Loading training data...
✓ Loaded 523 labeled samples

Training model...
✓ Model trained successfully

Evaluation Metrics:
  Accuracy:  0.876
  Precision: 0.842
  Recall:    0.891
  F1 Score:  0.866
  ROC AUC:   0.923

💾 Model saved to: data/models/layering_adaptive_v1.pkl
```

#### 4. Перезапустите бота
```bash
# Модель загрузится автоматически
python backend/main.py
```

---

## 🛠️ Troubleshooting

### Проблема: "No module named 'pyarrow'"

**Решение:**
```bash
pip install pyarrow==18.1.0
```

### Проблема: "Permission denied: data/layering_patterns.db"

**Решение:**
```bash
chmod -R 755 backend/data/
```

### Проблема: Parquet файлы не создаются

**Причина:** Нет detected layering patterns
**Решение:** Подождите пока detector обнаружит паттерны (это может занять время в зависимости от рыночной активности)

### Проблема: SQLite database locked

**Решение:**
```bash
# Остановите бота
# Проверьте открытые соединения
lsof | grep layering_patterns.db

# Удалите lock файл если есть
rm backend/data/layering_patterns.db-journal
```

---

## 📊 Мониторинг через Frontend

После запуска бота вы можете мониторить ML компоненты через веб-интерфейс:

**API Endpoints:**
- `/api/detection/quote-stuffing/status/{symbol}` - Quote Stuffing
- `/api/detection/patterns/list` - Historical Patterns
- `/api/ml/data-collector/statistics` - Data Collection
- `/api/ml/adaptive-model/metrics` - ML Model Metrics

**WebSocket Events:**
- `layering_detected` - новый паттерн обнаружен
- `quote_stuffing_detected` - HFT manipulation
- `ml_data_collected` - новый training sample

---

## ✅ Checklist для запуска

- [ ] `pip install -r requirements.txt` выполнен
- [ ] `pyarrow` установлен и импортируется
- [ ] `scikit-learn` установлен и импортируется
- [ ] Директория `backend/data/` создана (или создастся автоматически)
- [ ] Бот запущен хотя бы раз
- [ ] Проверены логи на наличие ошибок
- [ ] API endpoints отвечают (через curl или Swagger UI)

---

## 🎯 Итог

### Что делать вам:

1. **Только установить зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Запустить бота:**
   ```bash
   python backend/main.py
   ```

3. **Всё остальное происходит автоматически!**

### Что происходит автоматически:

- ✅ Создание SQLite базы данных
- ✅ Создание таблиц и индексов
- ✅ Создание директорий для Parquet файлов
- ✅ Сбор данных в background
- ✅ Автоматическое сохранение каждые 100 samples
- ✅ Загрузка обученной модели (если существует)

**Никаких ручных действий с базами данных не требуется!**

---

## 📚 Дополнительные Ресурсы

- **Layering Detection Algorithm:** `backend/ml_engine/detection/layering_detector.py`
- **Training Script:** `backend/scripts/train_layering_model.py`
- **API Documentation:** `http://localhost:8000/docs` (при DEBUG=true)
- **Pattern Database Schema:** См. `pattern_database.py:119-149`

---

**Версия:** 1.0
**Дата:** 2025-10-31
**Commit:** ea9df0e
