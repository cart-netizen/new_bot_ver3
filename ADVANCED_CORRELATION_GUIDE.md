# Руководство по продвинутому анализу корреляций

## Обзор

Продвинутый менеджер корреляций расширяет базовый функционал множественными метриками и методами группировки для более точного управления рисками.

---

## Новые возможности

### 1. Многомерный анализ корреляций

#### 1.1. Pearson Correlation с Rolling Windows

**Что это:**
- Расчет корреляции для 3 временных окон: короткое (7д), среднее (14д), длинное (30д)
- Взвешенное комбинирование для итоговой оценки

**Преимущества:**
- Учитывает краткосрочные изменения рынка
- Более стабильные долгосрочные оценки
- Адаптивность к изменяющимся условиям

**Пример:**
```python
# Конфигурация
CORRELATION_SHORT_WINDOW=7
CORRELATION_MEDIUM_WINDOW=14
CORRELATION_LONG_WINDOW=30

# Веса
CORRELATION_SHORT_WEIGHT=0.5   # Больше вес короткому окну
CORRELATION_MEDIUM_WEIGHT=0.3
CORRELATION_LONG_WEIGHT=0.2
```

**Интерпретация:**
```
BTC-ETH:
  - 7d correlation: 0.92 (очень высокая в последние дни)
  - 14d correlation: 0.85 (высокая в среднесроке)
  - 30d correlation: 0.78 (умеренно-высокая долгосрочно)

→ Weighted score: 0.87 (акцент на краткосрочную динамику)
```

---

#### 1.2. Spearman Rank Correlation

**Что это:**
- Измеряет монотонную зависимость (не обязательно линейную)
- Работает с рангами, а не с абсолютными значениями

**Когда полезно:**
```
Pearson: 0.65 (средняя линейная корреляция)
Spearman: 0.85 (высокая монотонная корреляция)

→ Активы движутся в одном направлении,
  но с разной скоростью/амплитудой
```

**Пример использования:**
```python
CORRELATION_USE_SPEARMAN=true

# Результат:
BTC-SOL:
  Pearson: 0.68
  Spearman: 0.82

→ SOL часто следует за BTC, но с задержкой/усилением
→ Все равно высокий риск концентрации
```

---

#### 1.3. Dynamic Time Warping (DTW)

**Что это:**
- Измеряет схожесть "формы" движения цен
- Позволяет находить корреляцию с временным лагом

**Преимущества:**
```
Обычная корреляция:
BTC: ↑ ↑ ↑ ↓ ↓
ETH: ↑ ↑ ↑ ↓ ↓
Correlation: 0.95 ✅

DTW находит даже:
BTC: ↑ ↑ ↑ ↓ ↓
ETH: → ↑ ↑ ↑ ↓  (сдвиг на 1 период)
DTW similarity: 0.90 ✅
```

**Конфигурация:**
```python
CORRELATION_USE_DTW=true
CORRELATION_DTW_MAX_LAG_HOURS=24    # Максимальный лаг
CORRELATION_DTW_WINDOW_HOURS=168    # Окно анализа (7 дней)
```

**⚠️ Внимание:**
DTW требует больше ресурсов. Используйте только если:
- Торгуете парами с возможным лагом (например, alt следует за BTC)
- Достаточно вычислительных мощностей

---

#### 1.4. Volatility Distance

**Что это:**
- Мера разницы в волатильности между активами

**Зачем:**
```
BTC volatility: 2%
ETH volatility: 2.1%
→ Volatility distance: 0.05 (очень похожие)

BTC volatility: 2%
SHIB volatility: 15%
→ Volatility distance: 0.87 (сильно отличаются)
```

**Применение:**
Активы с похожей волатильностью имеют больший risk overlap.
Учитывается в weighted score.

---

#### 1.5. Return Sign Agreement

**Что это:**
- Процент дней, когда оба актива движутся в одном направлении

**Пример:**
```
День 1: BTC +2%, ETH +1.5% → Согласие ✅
День 2: BTC -1%, ETH -0.5% → Согласие ✅
День 3: BTC +3%, ETH -0.2% → Расхождение ❌
День 4: BTC +1%, ETH +2%   → Согласие ✅

Agreement: 75% (3 из 4 дней)
```

**Интерпретация:**
- >80% - сильное согласие (высокий риск)
- 60-80% - умеренное согласие
- <60% - часто движутся независимо

---

### 2. Продвинутые методы группировки

#### 2.1. Louvain Community Detection (граф-based)

**Как работает:**
```
1. Строится граф:
   - Вершины = торговые пары
   - Ребра = корреляция ≥ threshold

2. Louvain алгоритм автоматически находит "сообщества"
   (плотно связанные группы)

3. Результат: естественные кластеры коррелирующих активов
```

**Преимущества:**
```
Жадная группировка:
  BTC → ETH (0.85) → SOL (0.78)
  Проблема: BTC-SOL могут быть слабо связаны напрямую

Louvain:
  Находит подгруппы:
    Group 1: [BTC, ETH, BNB]  - Major coins
    Group 2: [SOL, AVAX, FTM] - Layer-1 alts

  Даже если BTC-SOL = 0.65, они в разных группах
```

**Конфигурация:**
```python
CORRELATION_GROUPING_METHOD=louvain
```

---

#### 2.2. Hierarchical Clustering

**Как работает:**
```
1. Корреляции → дистанции (distance = 1 - correlation)
2. Агломеративная кластеризация (Ward's linkage)
3. Автоматический выбор количества кластеров
```

**Преимущества:**
- Более стабильные группы
- Хорошо работает с большим количеством активов (100+)
- Создает иерархию (можно выбрать детализацию)

**Конфигурация:**
```python
CORRELATION_GROUPING_METHOD=hierarchical
```

---

#### 2.3. Ensemble Method (консенсус)

**Как работает:**
```
1. Применяется Louvain
2. Применяется Hierarchical
3. Финальная группа = пары, которые в одной группе
   в обоих методах
```

**Преимущества:**
```
Louvain: [BTC, ETH, SOL, BNB, ADA]
Hierarchical: [BTC, ETH, BNB]

Ensemble (консенсус): [BTC, ETH, BNB]

→ Только самые надежные группы
→ Меньше ложных срабатываний
```

**Конфигурация:**
```python
CORRELATION_GROUPING_METHOD=ensemble  # Рекомендуется!
```

---

### 3. Детекция режима корреляций

**Режимы:**

#### 3.1. Low Correlation Regime
```
Средняя корреляция: < 0.4
Рынок: Rotation между секторами
Рекомендации:
  - Threshold: 0.6 (можно мягче)
  - Max positions: 3
  - Больше торговых возможностей
```

#### 3.2. Moderate Correlation Regime
```
Средняя корреляция: 0.4 - 0.6
Рынок: Нормальные условия
Рекомендации:
  - Threshold: 0.7
  - Max positions: 2
  - Сбалансированный подход
```

#### 3.3. High Correlation Regime
```
Средняя корреляция: 0.6 - 0.75
Рынок: Сильный тренд (все растет/падает вместе)
Рекомендации:
  - Threshold: 0.75
  - Max positions: 1
  - Осторожность!
```

#### 3.4. Crisis Correlation Regime
```
Средняя корреляция: > 0.85
Рынок: Кризис, паника, все падает одновременно
Рекомендации:
  - Threshold: 0.85
  - Max positions: 1
  - Максимальная защита
```

**Автоматическая адаптация:**
```python
CORRELATION_REGIME_DETECTION=true

# Пороги можно настроить
CORRELATION_REGIME_LOW_THRESHOLD=0.4
CORRELATION_REGIME_MODERATE_THRESHOLD=0.6
CORRELATION_REGIME_HIGH_THRESHOLD=0.75
CORRELATION_REGIME_CRISIS_THRESHOLD=0.85

# Система автоматически подстроит параметры
```

---

### 4. Кластеризация по волатильности

**Зачем:**
```
Проблема:
  BTC (volatility: 2%) + SHIB (volatility: 15%)
  → Корреляция может быть низкая
  → Но SHIB создает огромный риск из-за волатильности

Решение:
  Группируем активы по уровню волатильности:
    Low vol:  [BTC, ETH, BNB]
    Med vol:  [SOL, AVAX, LINK]
    High vol: [SHIB, PEPE, DOGE]

  Можно ограничить позиции в high vol группе
```

**Конфигурация:**
```python
CORRELATION_VOLATILITY_CLUSTERING=true
CORRELATION_VOLATILITY_CLUSTERS=3  # Low, Med, High
```

---

## Примеры конфигурации

### Консервативная (минимум риска)

```env
# Базовые параметры
CORRELATION_CHECK_ENABLED=true
CORRELATION_MAX_THRESHOLD=0.6  # Низкий порог - больше групп
CORRELATION_MAX_POSITIONS_PER_GROUP=1

# Продвинутый анализ
CORRELATION_USE_ADVANCED=true
CORRELATION_USE_SPEARMAN=true
CORRELATION_USE_DTW=false  # Экономим ресурсы

# Rolling windows (акцент на стабильность)
CORRELATION_SHORT_WINDOW=7
CORRELATION_MEDIUM_WINDOW=14
CORRELATION_LONG_WINDOW=30
CORRELATION_SHORT_WEIGHT=0.2
CORRELATION_MEDIUM_WEIGHT=0.3
CORRELATION_LONG_WEIGHT=0.5  # Больше вес долгосроку

# Группировка
CORRELATION_GROUPING_METHOD=ensemble  # Консенсус = надежнее

# Режимы
CORRELATION_REGIME_DETECTION=true

# Волатильность
CORRELATION_VOLATILITY_CLUSTERING=true
```

**Результат:**
- Максимальная диверсификация
- Меньше торговых возможностей
- Минимальный риск концентрации

---

### Сбалансированная (рекомендуется)

```env
CORRELATION_CHECK_ENABLED=true
CORRELATION_MAX_THRESHOLD=0.7
CORRELATION_MAX_POSITIONS_PER_GROUP=2

CORRELATION_USE_ADVANCED=true
CORRELATION_USE_SPEARMAN=true
CORRELATION_USE_DTW=false

# Rolling windows (баланс)
CORRELATION_SHORT_WINDOW=7
CORRELATION_MEDIUM_WINDOW=14
CORRELATION_LONG_WINDOW=30
CORRELATION_SHORT_WEIGHT=0.5
CORRELATION_MEDIUM_WEIGHT=0.3
CORRELATION_LONG_WEIGHT=0.2

CORRELATION_GROUPING_METHOD=ensemble

CORRELATION_REGIME_DETECTION=true
CORRELATION_VOLATILITY_CLUSTERING=true
```

**Результат:**
- Хорошая диверсификация
- Достаточно торговых возможностей
- Защита от риска

---

### Агрессивная (максимум сделок)

```env
CORRELATION_CHECK_ENABLED=true
CORRELATION_MAX_THRESHOLD=0.8  # Высокий порог - меньше групп
CORRELATION_MAX_POSITIONS_PER_GROUP=3

CORRELATION_USE_ADVANCED=true
CORRELATION_USE_SPEARMAN=true
CORRELATION_USE_DTW=true  # Все методы

# Rolling windows (акцент на краткосрок)
CORRELATION_SHORT_WINDOW=5
CORRELATION_MEDIUM_WINDOW=10
CORRELATION_LONG_WINDOW=20
CORRELATION_SHORT_WEIGHT=0.6
CORRELATION_MEDIUM_WEIGHT=0.3
CORRELATION_LONG_WEIGHT=0.1

CORRELATION_GROUPING_METHOD=louvain  # Быстрее ensemble

CORRELATION_REGIME_DETECTION=true
CORRELATION_VOLATILITY_CLUSTERING=false  # Экономим ресурсы
```

**Результат:**
- Больше торговых возможностей
- Выше риск концентрации
- Подходит для опытных трейдеров

---

## Мониторинг и отладка

### Просмотр групп

```bash
python show_correlation_groups.py
```

**Новый вывод с продвинутыми метриками:**
```
Группа 1: ensemble_0
  Метод группировки:     Ensemble (консенсус)
  Количество пар:        8
  Средняя корреляция:    0.791
  Min/Max корреляция:    0.72 / 0.89
  Avg DTW distance:      0.15
  Cluster quality:       0.82

  Торговые пары:
    BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, ...

  Pearson 7d:   0.85
  Pearson 14d:  0.82
  Pearson 30d:  0.78
  Spearman:     0.88

  Режим корреляций: HIGH_CORRELATION
  → Будьте осторожны, высокий риск!
```

---

## Производительность

**Время инициализации (170 пар):**

| Метод | Время | Требования CPU/RAM |
|-------|-------|---------------------|
| Greedy | ~2 сек | Низкие |
| Louvain | ~4 сек | Средние |
| Hierarchical | ~5 сек | Средние |
| Ensemble | ~10 сек | Высокие |
| + DTW | +15 сек | Очень высокие |

**Рекомендации:**
- **Для production**: Ensemble без DTW (баланс точности и скорости)
- **Для backtesting**: Все методы включая DTW
- **Для быстрого старта**: Louvain

---

## FAQ

### 1. Какой метод группировки лучше?

**Ensemble** - самый надежный, но медленнее.
**Louvain** - хороший баланс точности и скорости.
**Hierarchical** - лучше для большого количества пар (200+).

### 2. Стоит ли включать DTW?

**Да**, если:
- Торгуете альты, которые следуют за BTC с лагом
- Есть мощный сервер
- Важна максимальная точность

**Нет**, если:
- Нужна быстрая инициализация
- Ограниченные ресурсы
- Торгуете только major coins

### 3. Как часто обновлять корреляции?

```python
# Рекомендуется:
- Полное обновление: 1 раз в сутки
- Инкрементальное: каждые 4-6 часов

# При резких изменениях рынка:
- Проверяйте режим корреляций каждый час
- При переходе в Crisis режим - немедленная проверка позиций
```

### 4. Как интерпретировать weighted score?

```
Weighted score комбинирует все метрики:
- 40% Pearson (rolling windows)
- 20% Spearman
- 20% DTW similarity
- 10% Volatility similarity
- 10% Sign agreement

Результат [-1, 1]:
  > 0.8:  Очень высокая корреляция
  0.6-0.8: Высокая корреляция
  0.4-0.6: Средняя корреляция
  < 0.4:  Низкая корреляция
```

---

## Техническая документация

### Установка зависимостей

```bash
pip install numpy scipy scikit-learn networkx
```

Для полной поддержки DTW (опционально):
```bash
pip install dtaidistance
```

### Запуск тестов

```bash
pytest backend/tests/test_advanced_correlation.py -v
```

---

## Миграция с базового менеджера

Существующий код совместим! Просто обновите .env:

```env
# Включаем продвинутый анализ
CORRELATION_USE_ADVANCED=true

# Остальные параметры опциональны (используются defaults)
```

Система автоматически переключится на продвинутый режим.

---

## Поддержка

Вопросы и предложения:
- GitHub Issues
- Документация: `CORRELATION_EXPLAINED.md` (базовый)
- Эта документация: `ADVANCED_CORRELATION_GUIDE.md` (продвинутый)
