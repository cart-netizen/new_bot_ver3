#!/bin/bash
# Скрипт для исправления импортов в backend

echo "Исправление импортов в backend..."

# Список ВСЕХ модулей в backend для исправления
modules="api core database domain engine exchange execution infrastructure ml_engine models screener scripts services strategies strategy tasks tests utils config"

# Для каждого модуля заменяем импорты
for module in $modules; do
    echo "Исправление импортов для модуля: $module"

    # Находим все Python файлы в backend и заменяем импорты
    find /home/user/new_bot_ver3/backend -name "*.py" -type f -exec sed -i \
        -e "s/^\(\s*\)from ${module}\./\1from backend.${module}./g" \
        -e "s/^\(\s*\)from ${module} /\1from backend.${module} /g" \
        -e "s/^\(\s*\)import ${module}$/\1import backend.${module} as ${module}/g" \
        {} \;
done

# Специальная обработка для main модуля (с учетом отступов)
echo "Исправление импортов для модуля: main"
find /home/user/new_bot_ver3/backend -name "*.py" -type f -exec sed -i \
    -e "s/^\(\s*\)from main import /\1from backend.main import /g" \
    -e "s/^\(\s*\)from main\./\1from backend.main./g" \
    -e "s/^\(\s*\)import main$/\1import backend.main as main/g" \
    {} \;

echo "✅ Импорты исправлены!"
echo "Проверка количества исправленных файлов..."
grep -r "^from backend\." /home/user/new_bot_ver3/backend --include="*.py" | cut -d: -f1 | sort -u | wc -l
