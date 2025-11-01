# PyCharm Configuration Guide

## Проблема и решение

**Проблема:** При разработке в PyCharm папка `backend/` была помечена как "Source Root", что приводило к импортам без префикса `backend.`:
```python
from core.logger import get_logger  # ❌ PyCharm формат
```

Но при деплое на сервер, backend запускается через `uvicorn backend.main:app`, что требует импортов С префиксом:
```python
from backend.core.logger import get_logger  # ✅ Production формат
```

**Решение:** Проект теперь приведен к production-формату. Все импорты используют префикс `backend.`.

---

## Настройка PyCharm

### 1. Убрать Source Root с папки backend

1. Откройте PyCharm
2. В Project View найдите папку `backend/`
3. **Правой кнопкой мыши** на `backend/` → **Mark Directory as** → **Unmark as Sources Root**
4. Теперь `backend/` должна стать обычной папкой (синий цвет исчезнет)

### 2. Установить корень проекта

1. **Правой кнопкой мыши** на **корень проекта** (`new_bot_ver3/`)
2. **Mark Directory as** → **Sources Root** (если еще не установлено)

### 3. Настроить Run Configuration

#### Для backend (main.py):

1. **Run** → **Edit Configurations...**
2. Добавить новую конфигурацию **Python**:
   - **Name:** `Backend Server`
   - **Script path:** `/path/to/project/backend/main.py`
   - **Working directory:** `/path/to/project` (корень проекта, НЕ `/path/to/project/backend`)
   - **Python interpreter:** Выбрать виртуальное окружение проекта
   - **Environment variables:** (если нужно)

3. Применить и сохранить

#### Для Alembic миграций:

1. **Run** → **Edit Configurations...**
2. Добавить конфигурацию **Python**:
   - **Name:** `Alembic Upgrade`
   - **Script path:** `/path/to/project/backend/database/migrations/env.py`
   - **Parameters:** `upgrade head`
   - **Working directory:** `/path/to/project`
   - **Python interpreter:** Виртуальное окружение

---

## Проверка настройки

### 1. Проверка PYTHONPATH

В терминале PyCharm выполните:

```bash
python -c "import sys; print('\n'.join(sys.path))"
```

Убедитесь, что в списке есть корень проекта `/path/to/project`, но НЕТ `/path/to/project/backend`.

### 2. Проверка импортов

Откройте любой файл в `backend/`, например `backend/main.py`.

Импорты должны быть БЕЗ подчеркивания (не показывать ошибки):
```python
from backend.core.logger import get_logger  # ✅ Должно быть зеленым
from backend.database.connection import db_manager  # ✅ Должно быть зеленым
```

Если PyCharm подчеркивает импорты красным:
1. **File** → **Invalidate Caches / Restart...**
2. Выбрать **Invalidate and Restart**

### 3. Проверка запуска

Запустите backend через Run Configuration:

```bash
# Должно работать без ошибок
python backend/main.py
```

Если возникают ошибки `ModuleNotFoundError`:
- Убедитесь что Working Directory = корень проекта
- Проверьте что папка `backend/` НЕ помечена как Sources Root

---

## Структура проекта

```
new_bot_ver3/                    ← Sources Root (корень проекта)
├── backend/                     ← Обычная папка (НЕ Sources Root)
│   ├── main.py
│   ├── core/
│   ├── api/
│   ├── ml_engine/
│   └── ...
├── frontend/
├── scripts/
├── requirements.txt
└── PYCHARM_SETUP.md
```

---

## FAQ

### Q: PyCharm подчеркивает все импорты красным после изменений

**A:** Выполните **File → Invalidate Caches / Restart** и перезапустите PyCharm.

### Q: При запуске получаю `ModuleNotFoundError: No module named 'backend'`

**A:** Убедитесь что:
1. Working Directory в Run Configuration = корень проекта (не `backend/`)
2. Папка `backend/` НЕ помечена как Sources Root
3. Корень проекта `new_bot_ver3/` помечен как Sources Root

### Q: Автодополнение не работает для backend модулей

**A:**
1. Проверьте что виртуальное окружение активировано
2. Выполните **File → Invalidate Caches / Restart**
3. Убедитесь что Python interpreter настроен на виртуальное окружение проекта

### Q: Нужно ли менять что-то при деплое?

**A:** **НЕТ!** Теперь код полностью готов к production деплою. Скрипт `scripts/fix_imports.sh` больше не нужен. В `deploy_improved.sh` можно удалить шаг 3 (исправление импортов).

---

## Преимущества новой структуры

✅ **Консистентность:** Одинаковые импорты в dev и production
✅ **Простота деплоя:** Не нужно модифицировать код перед деплоем
✅ **Предсказуемость:** Импорты работают одинаково везде
✅ **CI/CD ready:** Код готов к автоматическому деплою

---

## Дополнительные ресурсы

- [PyCharm: Configuring Project Structure](https://www.jetbrains.com/help/pycharm/configuring-project-structure.html)
- [Python: Package Resolution and __init__.py](https://docs.python.org/3/tutorial/modules.html#packages)
- [FastAPI: Project Structure](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
