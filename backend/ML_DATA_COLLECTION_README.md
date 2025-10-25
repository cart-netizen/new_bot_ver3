# ML Data Collection Mode - Инструкция по использованию

## 📋 Описание

Упрощенная версия analysis loop **ТОЛЬКО для сбора данных для обучения ML модели**.

Предназначена для запуска на облачном сервере для экономии ресурсов.
**Торговля ОТКЛЮЧЕНА** - бот только собирает данные для последующего обучения модели.

---

## ✅ Что ВКЛЮЧЕНО

- ✅ Получение orderbook snapshots
- ✅ Получение candles
- ✅ Извлечение ML features (110+ признаков)
- ✅ Детекция манипуляций (spoofing/layering)
- ✅ S/R levels detection
- ✅ ML Data Collection для обучения

## ❌ Что УБРАНО (для экономии ресурсов)

- ❌ Генерация торговых сигналов (IntegratedEngine)
- ❌ Adaptive Consensus Manager
- ❌ Multi-Timeframe Analysis
- ❌ ML Validation сигналов
- ❌ Risk & Quality Checks
- ❌ Execution Manager / размещение ордеров
- ❌ Real-time UI broadcasting
- ❌ Position management
- ❌ Drift monitoring

---

## 🚀 Использование

### Вариант 1: Использовать в main.py (рекомендуется)

Отредактируйте `backend/main.py`:

```python
# В методе run() класса BotController

async def run(self):
    """Запуск бота."""
    try:
        logger.info("Запуск бота...")
        self.status = BotStatus.RUNNING

        # Запускаем WebSocket и другие сервисы
        await self._start_websockets()
        await self.risk_manager.start()

        # ========================================
        # РЕЖИМ СБОРА ML ДАННЫХ (БЕЗ ТОРГОВЛИ)
        # ========================================

        from analysis_loop_ml_data_collection import ml_data_collection_loop

        # Запускаем упрощенный цикл ТОЛЬКО для сбора данных
        await ml_data_collection_loop(
            bot_controller=self,
            symbols=self.symbols,
            analysis_interval=settings.ANALYSIS_INTERVAL
        )

        # ========================================
        # ОБЫЧНЫЙ РЕЖИМ (С ТОРГОВЛЕЙ)
        # ========================================

        # await self._analysis_loop_ml_enhanced()  # Закомментировать для режима сбора данных

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
    finally:
        await self.stop()
```

### Вариант 2: Использовать отдельный запуск

Создайте файл `backend/run_ml_data_collection.py`:

```python
"""
Запуск бота в режиме сбора ML данных (без торговли).
"""

import asyncio
from main import BotController
from analysis_loop_ml_data_collection import ml_data_collection_loop
from config import settings

async def main():
    # Создаем bot controller
    bot = BotController()

    # Инициализируем все необходимые компоненты
    await bot.initialize()

    # Запускаем режим сбора данных
    await ml_data_collection_loop(
        bot_controller=bot,
        symbols=bot.symbols,
        analysis_interval=settings.ANALYSIS_INTERVAL
    )

if __name__ == "__main__":
    asyncio.run(main())
```

Запуск:
```bash
cd backend
python3 run_ml_data_collection.py
```

---

## ⚙️ Настройка конфигурации

### 1. Отключить ненужные компоненты в `.env`:

```bash
# ===== ML DATA COLLECTION MODE =====
# Оставляем только необходимые компоненты

# Обязательные (должны быть ВКЛЮЧЕНЫ)
ENABLE_ML_FEATURES=true
ENABLE_ML_DATA_COLLECTION=true
ENABLE_SPOOFING_DETECTION=true    # Опционально, для меток манипуляций
ENABLE_LAYERING_DETECTION=true    # Опционально, для меток манипуляций
ENABLE_SR_DETECTION=true          # Опционально, для обогащения признаков

# Отключаем торговые компоненты (экономия ресурсов)
ENABLE_TRADING=false              # ❌ Торговля отключена
ENABLE_ADAPTIVE_CONSENSUS=false   # ❌ Не используется
ENABLE_MTF_ANALYSIS=false         # ❌ Не используется
ENABLE_ML_VALIDATION=false        # ❌ Не используется
ENABLE_DRIFT_MONITORING=false     # ❌ Не используется
ENABLE_UI_BROADCAST=false         # ❌ Не используется

# Интервал анализа
ANALYSIS_INTERVAL=60              # Частота сбора данных (секунды)

# ML Data Collector Settings
ML_DATA_SAMPLES_PER_DAY=1000      # Целевое количество samples в день
ML_DATA_MAX_FILE_SIZE_MB=100     # Максимальный размер файла
```

### 2. Минимальная конфигурация символов:

Если нужно снизить нагрузку, уменьшите количество символов в `.env`:

```bash
# Минимальный набор для сбора данных
TRADING_PAIRS=BTCUSDT,ETHUSDT,SOLUSDT
```

---

## 📊 Что собирается

Каждый sample включает:

1. **Timestamp** - время сбора
2. **Symbol** - торговая пара
3. **Features** - 110+ ML признаков:
   - OrderBook features (imbalance, depth, pressure, etc.)
   - Candle features (momentum, trend, volatility, etc.)
   - S/R features (distances, strengths, etc.)
   - Temporal features (изменения между snapshot'ами)
4. **Price** - текущая цена
5. **OrderBook Snapshot** - bid/ask, spread, imbalance
6. **Market Metrics** - volatility, volume, momentum
7. **Manipulations** - флаги spoofing/layering (для меток)

---

## 📁 Где хранятся данные

По умолчанию данные сохраняются в:

```
backend/ml_engine/data_collection/samples/
├── BTCUSDT_2025-01-15.jsonl
├── ETHUSDT_2025-01-15.jsonl
└── SOLUSDT_2025-01-15.jsonl
```

Формат: **JSONL** (JSON Lines) - каждая строка это отдельный sample.

---

## 📈 Мониторинг

### Логи

Каждые 10 циклов выводится статистика:

```
================================================================================
📈 СТАТИСТИКА СБОРА ДАННЫХ (Цикл #10)
   ├─ Циклов анализа: 10
   ├─ ML данных собрано: 150
   ├─ Манипуляций обнаружено: 3
   ├─ Ошибок: 0
   └─ Время цикла: 8.23s
================================================================================
```

### Проверка собранных данных

```python
import json

# Читаем samples
with open('ml_engine/data_collection/samples/BTCUSDT_2025-01-15.jsonl', 'r') as f:
    samples = [json.loads(line) for line in f]

print(f"Всего samples: {len(samples)}")
print(f"Первый sample:")
print(json.dumps(samples[0], indent=2))
```

---

## 🔧 Требования к ресурсам

### Минимальные требования (3 символа):

- **CPU**: 1 core
- **RAM**: 1 GB
- **Disk**: 10 GB (для данных)
- **Network**: Стабильное соединение WebSocket

### Рекомендуемые (10 символов):

- **CPU**: 2 cores
- **RAM**: 2 GB
- **Disk**: 50 GB

---

## ⚠️ Важные замечания

1. **API Keys**: Нужны только для получения orderbook/candles, торговые права НЕ требуются
2. **WebSocket**: Должен быть стабильным, иначе будут пропуски данных
3. **Disk Space**: Следите за местом на диске, данные накапливаются
4. **Backups**: Регулярно копируйте собранные данные на другой сервер

---

## 🛠️ Troubleshooting

### Проблема: Нет данных в файлах

**Решение:**
- Проверьте что `ENABLE_ML_DATA_COLLECTION=true`
- Проверьте логи: `grep "ML Data sample" backend.log`
- Проверьте права на запись в `ml_engine/data_collection/samples/`

### Проблема: Медленное выполнение

**Решение:**
- Уменьшите количество символов в `TRADING_PAIRS`
- Увеличьте `ANALYSIS_INTERVAL` (например, до 120 секунд)
- Отключите необязательные детекторы (spoofing/layering)

### Проблема: Много ошибок "Feature extraction вернул None"

**Решение:**
- Проверьте что WebSocket стабилен
- Проверьте что достаточно свечей (>50) для каждого символа
- Увеличьте задержку перед началом анализа после запуска

---

## 📞 Поддержка

Если возникли проблемы:

1. Проверьте логи: `tail -f backend/logs/backend.log`
2. Проверьте статус компонентов в логах при запуске
3. Убедитесь что все обязательные компоненты инициализированы (✅)

---

## 🎯 Следующие шаги

После сбора достаточного количества данных (рекомендуется 10,000+ samples):

1. Скопируйте данные с сервера локально
2. Используйте `ml_engine/training/` для обучения модели
3. Валидируйте модель на отложенных данных
4. Разверните обученную модель обратно в production

---

**Удачного сбора данных! 🚀**
