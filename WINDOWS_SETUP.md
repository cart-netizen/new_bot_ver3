# 🪟 Windows Setup Guide - Layering ML

## 📋 Быстрый старт для Windows

### 1. Проверка статуса данных

**Из корня проекта (где находится .venv):**
```powershell
# Активируйте виртуальное окружение
.venv\Scripts\activate

# Проверьте статус
python check_layering_data_status.py
```

### 2. Запуск сбора данных

**Вариант 1 - Через PowerShell:**
```powershell
# Активируйте venv (если еще не активировано)
.venv\Scripts\activate

# Запустите бота
python backend/main.py
```

**Вариант 2 - Через .bat файл (если есть):**
```powershell
start_bot.bat
```

### 3. Обучение модели

**Вариант 1 - Простой (рекомендуется):**
```powershell
# Просто запустите wrapper скрипт
python train_layering_model.py
```

**Вариант 2 - Через .bat файл:**
```powershell
# Двойной клик по файлу ИЛИ из командной строки:
train_layering_model.bat
```

**Вариант 3 - Прямой вызов:**
```powershell
# Активируйте venv
.venv\Scripts\activate

# Запустите скрипт обучения
python backend/scripts/train_layering_model.py
```

### 4. Анализ данных

**Детальный анализ:**
```powershell
# Активируйте venv
.venv\Scripts\activate

# Установите зависимости (если еще нет)
pip install pandas pyarrow scikit-learn

# Запустите анализ
python analyze_layering_ml_data.py
```

---

## 🔧 Устранение проблем

### Проблема: "ModuleNotFoundError: No module named 'backend'"

**Решение:**
```powershell
# Убедитесь что вы находитесь в КОРНЕ проекта, а не в backend/
cd C:\Users\1q\PycharmProjects\Bot_ver3_stakan_new

# Активируйте venv
.venv\Scripts\activate

# Запустите из КОРНЯ проекта:
python train_layering_model.py
# ИЛИ
python backend/scripts/train_layering_model.py
```

### Проблема: "No module named 'pandas'"

**Решение:**
```powershell
# Активируйте venv
.venv\Scripts\activate

# Установите зависимости
pip install pandas pyarrow scikit-learn
```

### Проблема: "python: command not found"

**Решение:**
```powershell
# Используйте python3 вместо python
python3 train_layering_model.py

# ИЛИ используйте полный путь к Python
C:\Python311\python.exe train_layering_model.py

# ИЛИ активируйте venv (где python доступен)
.venv\Scripts\activate
python train_layering_model.py
```

### Проблема: Скрипт не может найти данные

**Решение:**
```powershell
# Проверьте что директория существует
dir data\ml_training\layering

# Если не существует - это нормально при первом запуске
# Запустите бота для начала сбора:
python backend/main.py
```

---

## 📁 Структура проекта (важно!)

```
C:\Users\1q\PycharmProjects\Bot_ver3_stakan_new\
│
├── .venv\                              # Виртуальное окружение
│
├── backend\
│   ├── scripts\
│   │   └── train_layering_model.py    # Основной скрипт обучения
│   ├── ml_engine\
│   │   └── detection\
│   │       ├── layering_detector.py
│   │       ├── layering_data_collector.py
│   │       └── adaptive_layering_model.py
│   └── main.py                         # Основной бот
│
├── data\
│   ├── ml_training\
│   │   └── layering\                   # Здесь хранятся .parquet файлы
│   └── models\
│       └── layering_adaptive_v1.pkl    # Обученная модель
│
├── train_layering_model.py            # Wrapper (удобно!)
├── train_layering_model.bat           # Для Windows (двойной клик)
├── check_layering_data_status.py      # Проверка статуса
├── analyze_layering_ml_data.py        # Детальный анализ
│
├── LAYERING_ML_GUIDE.md               # Полная документация
└── LAYERING_ML_QUICKSTART.md          # Быстрый старт
```

---

## ✅ Правильные команды для Windows

### Сценарий 1: Первый запуск (сбор данных)

```powershell
# 1. Откройте PowerShell в корне проекта
cd C:\Users\1q\PycharmProjects\Bot_ver3_stakan_new

# 2. Активируйте venv
.venv\Scripts\activate

# 3. Проверьте статус
python check_layering_data_status.py

# 4. Запустите бота для сбора данных
python backend/main.py

# 5. Подождите 1-7 дней...
```

### Сценарий 2: Обучение модели (после сбора)

```powershell
# 1. Проверьте что данные собраны
python check_layering_data_status.py

# 2. Запустите обучение (САМЫЙ ПРОСТОЙ СПОСОБ)
python train_layering_model.py

# ИЛИ двойной клик по:
# train_layering_model.bat
```

### Сценарий 3: Анализ данных

```powershell
# 1. Активируйте venv
.venv\Scripts\activate

# 2. Установите зависимости (если нужно)
pip install pandas pyarrow scikit-learn

# 3. Запустите анализ
python analyze_layering_ml_data.py
```

---

## 🎯 Рекомендации для Windows пользователей

1. **Всегда запускайте из КОРНЯ проекта**, не из backend/
2. **Активируйте виртуальное окружение** перед запуском
3. **Используйте PowerShell или CMD**, не Git Bash (могут быть проблемы с путями)
4. **Используйте wrapper скрипты** (train_layering_model.py или .bat) для удобства

---

## 📞 Если что-то не работает

**Проверочный список:**
```powershell
# ✓ Вы в корне проекта?
pwd
# Должно быть: C:\Users\1q\PycharmProjects\Bot_ver3_stakan_new

# ✓ Venv активирован?
where python
# Должно показать путь к .venv\Scripts\python.exe

# ✓ Зависимости установлены?
pip list | findstr pandas
pip list | findstr scikit-learn
pip list | findstr pyarrow

# ✓ Файлы существуют?
dir train_layering_model.py
dir backend\scripts\train_layering_model.py
```

Если всё ещё не работает - напишите какая именно ошибка возникает! 🙂
