#!/bin/bash
#
# Скрипт конвертации импортов для production деплоя
# Преобразует импорты из PyCharm Source Root формата в production формат
#
# До:  from core.logger import get_logger
# После: from backend.core.logger import get_logger
#

set -e

PROJECT_DIR="/home/user/new_bot_ver3"
BACKEND_DIR="$PROJECT_DIR/backend"

echo "🔄 Конвертация импортов в production формат..."
echo ""
echo "📂 Backend directory: $BACKEND_DIR"
echo ""

# Список ВСЕХ модулей backend для конвертации
# (взято из deploy скрипта)
modules="api core database domain engine exchange execution infrastructure ml_engine models screener scripts services strategies strategy tasks tests utils config"

# Счетчики для статистики
total_replacements=0

for module in $modules; do
    echo "📦 Обработка модуля: $module"

    # Подсчитываем сколько файлов содержат такие импорты
    count_before=$(find "$BACKEND_DIR" -name "*.py" -type f -exec grep -l "^from ${module}\." {} \; 2>/dev/null | wc -l)
    count_before2=$(find "$BACKEND_DIR" -name "*.py" -type f -exec grep -l "^from ${module} " {} \; 2>/dev/null | wc -l)
    count_before3=$(find "$BACKEND_DIR" -name "*.py" -type f -exec grep -l "^import ${module}$" {} \; 2>/dev/null | wc -l)

    total_before=$((count_before + count_before2 + count_before3))

    if [ $total_before -gt 0 ]; then
        echo "   ✓ Найдено файлов с импортами: $total_before"

        # Применяем замены (только если НЕ уже начинается с backend.)
        find "$BACKEND_DIR" -name "*.py" -type f -exec sed -i \
            -e "s|^\(\s*\)from ${module}\.|\\1from backend.${module}.|g" \
            -e "s|^\(\s*\)from ${module} |\\1from backend.${module} |g" \
            -e "s|^\(\s*\)import ${module}$|\\1import backend.${module} as ${module}|g" \
            {} \;

        total_replacements=$((total_replacements + total_before))
    else
        echo "   - Импорты не найдены"
    fi
done

# Специальная обработка для main модуля
echo ""
echo "📦 Обработка модуля: main"
count_main=$(find "$BACKEND_DIR" -name "*.py" -type f -exec grep -l "^from main import\|^from main\.\|^import main$" {} \; 2>/dev/null | wc -l)

if [ $count_main -gt 0 ]; then
    echo "   ✓ Найдено файлов с импортами: $count_main"

    find "$BACKEND_DIR" -name "*.py" -type f -exec sed -i \
        -e "s|^\(\s*\)from main import |\\1from backend.main import |g" \
        -e "s|^\(\s*\)from main\.|\\1from backend.main.|g" \
        -e "s|^\(\s*\)import main$|\\1import backend.main as main|g" \
        {} \;

    total_replacements=$((total_replacements + count_main))
else
    echo "   - Импорты не найдены"
fi

echo ""
echo "════════════════════════════════════════"
echo "✅ Конвертация завершена!"
echo "════════════════════════════════════════"
echo ""
echo "📊 Статистика:"
echo "   Обработано файлов: ~$total_replacements"
echo ""
echo "🔍 Проверка результата:"
backend_imports=$(grep -r "^from backend\." "$BACKEND_DIR" --include="*.py" 2>/dev/null | wc -l)
echo "   Импорты с 'from backend.': $backend_imports"
echo ""

# Проверка что не осталось старых импортов (исключаем комментарии)
echo "⚠️  Проверка оставшихся импортов без backend. префикса:"
remaining=0
for module in $modules; do
    count=$(grep -r "^from ${module}\." "$BACKEND_DIR" --include="*.py" 2>/dev/null | grep -v "# " | wc -l)
    if [ $count -gt 0 ]; then
        echo "   ⚠️  Модуль '$module': $count импортов БЕЗ backend. префикса"
        remaining=$((remaining + count))
    fi
done

if [ $remaining -eq 0 ]; then
    echo "   ✓ Все импорты успешно сконвертированы!"
else
    echo ""
    echo "   ⚠️  ВНИМАНИЕ: Найдено $remaining импортов требующих проверки"
    echo "   Возможно это импорты из внешних библиотек или комментарии"
fi

echo ""
echo "📝 Следующие шаги:"
echo "   1. Проверьте что backend запускается: python backend/main.py"
echo "   2. Обновите PyCharm: уберите Source Root с папки backend/"
echo "   3. Обновите Run Configuration в PyCharm:"
echo "      Script path: /path/to/project/backend/main.py"
echo "      Working directory: /path/to/project"
echo "      (НЕ /path/to/project/backend)"
echo ""
