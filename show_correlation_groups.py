"""
Скрипт для визуализации корреляционных групп торговых пар.

Использование:
    python show_correlation_groups.py
"""
import asyncio
import sys
from pathlib import Path

# Добавляем backend в путь
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from strategy.correlation_manager import correlation_manager
from config import settings
from core.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Основная функция."""
    print("=" * 80)
    print("CORRELATION GROUPS VIEWER")
    print("=" * 80)
    print()

    # Проверяем, включен ли менеджер корреляций
    if not settings.CORRELATION_CHECK_ENABLED:
        print("❌ Correlation Manager ОТКЛЮЧЕН в конфигурации!")
        print("   Установите CORRELATION_CHECK_ENABLED=true в .env файле")
        return

    print("📊 Настройки:")
    print(f"   • Порог корреляции: {settings.CORRELATION_MAX_THRESHOLD}")
    print(f"   • Макс. позиций на группу: {settings.CORRELATION_MAX_POSITIONS_PER_GROUP}")
    print(f"   • Период расчета: {settings.CORRELATION_LOOKBACK_DAYS} дней")
    print()

    # Получаем список торговых пар
    trading_pairs = settings.get_trading_pairs_list()
    print(f"🔍 Анализ {len(trading_pairs)} торговых пар...")
    print()

    # Инициализируем correlation manager
    try:
        await correlation_manager.initialize(trading_pairs)
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return

    # Получаем статистику
    stats = correlation_manager.get_statistics()

    print("=" * 80)
    print("📈 СТАТИСТИКА")
    print("=" * 80)
    print(f"   Всего групп создано:        {stats['total_groups']}")
    print(f"   Групп с открытыми позициями: {stats['groups_with_positions']}")
    print(f"   Активных позиций:            {stats['total_active_positions']}")
    print(f"   Общая экспозиция:            {stats['total_exposure_usdt']:.2f} USDT")
    print(f"   Покрыто символов:            {len(correlation_manager.group_manager.symbol_to_group)}/{len(trading_pairs)}")
    print(f"   Независимых пар:             {len(trading_pairs) - len(correlation_manager.group_manager.symbol_to_group)}")
    print()

    # Получаем детали групп
    groups = correlation_manager.get_group_details()

    if not groups:
        print("ℹ️  Не создано ни одной корреляционной группы")
        print("   Все торговые пары движутся независимо (корреляция < {})".format(
            settings.CORRELATION_MAX_THRESHOLD
        ))
        return

    print("=" * 80)
    print("📦 КОРРЕЛЯЦИОННЫЕ ГРУППЫ")
    print("=" * 80)
    print()

    # Сортируем группы по размеру (больше пар = важнее)
    groups.sort(key=lambda g: len(g['symbols']), reverse=True)

    for i, group in enumerate(groups, 1):
        symbols = group['symbols']
        avg_corr = group['avg_correlation']
        active_pos = group['active_positions']
        exposure = group['total_exposure_usdt']

        # Заголовок группы
        print(f"{'─' * 80}")
        print(f"Группа {i}: {group['group_id']}")
        print(f"{'─' * 80}")

        # Информация
        print(f"   Количество пар:        {len(symbols)}")
        print(f"   Средняя корреляция:    {avg_corr:.3f}")
        print(f"   Открытых позиций:      {active_pos}/{settings.CORRELATION_MAX_POSITIONS_PER_GROUP}")

        if exposure > 0:
            print(f"   Экспозиция:            {exposure:.2f} USDT")

        print()

        # Список пар
        print("   Торговые пары:")

        # Разбиваем на строки по 5 пар
        for j in range(0, len(symbols), 5):
            chunk = symbols[j:j+5]
            formatted = ", ".join(f"{s:15}" for s in chunk)
            print(f"      {formatted}")

        print()

        # Интерпретация корреляции
        if avg_corr >= 0.9:
            strength = "🔴 ОЧЕНЬ ВЫСОКАЯ"
            desc = "Пары движутся почти идентично. Открытие нескольких позиций = очень высокий риск."
        elif avg_corr >= 0.8:
            strength = "🟠 ВЫСОКАЯ"
            desc = "Пары обычно движутся вместе. Риск концентрации значительный."
        elif avg_corr >= 0.7:
            strength = "🟡 УМЕРЕННАЯ"
            desc = "Пары часто движутся вместе, но иногда расходятся."
        else:
            strength = "🟢 НИЗКАЯ"
            desc = "Пары коррелируют слабо, но попали в группу транзитивно."

        print(f"   Сила корреляции: {strength}")
        print(f"   └─ {desc}")
        print()

    # Показываем независимые пары
    independent_symbols = [
        s for s in trading_pairs
        if s not in correlation_manager.group_manager.symbol_to_group
    ]

    if independent_symbols:
        print("=" * 80)
        print(f"🆓 НЕЗАВИСИМЫЕ ПАРЫ ({len(independent_symbols)})")
        print("=" * 80)
        print()
        print("   Эти пары не коррелируют ни с какими другими (корреляция < {})".format(
            settings.CORRELATION_MAX_THRESHOLD
        ))
        print("   Для них нет ограничений на открытие позиций.")
        print()

        # Список независимых пар
        for j in range(0, len(independent_symbols), 6):
            chunk = independent_symbols[j:j+6]
            formatted = ", ".join(f"{s:15}" for s in chunk)
            print(f"      {formatted}")

        print()

    # Рекомендации
    print("=" * 80)
    print("💡 РЕКОМЕНДАЦИИ")
    print("=" * 80)
    print()

    if len(groups) > 20:
        print("   ✅ Хорошая диверсификация - много маленьких групп")
    elif len(groups) > 10:
        print("   ⚠️  Средняя диверсификация")
    else:
        print("   🔴 Низкая диверсификация - мало групп, много пар в каждой")
        print("      Рассмотрите снижение CORRELATION_MAX_THRESHOLD")

    print()

    if settings.CORRELATION_MAX_POSITIONS_PER_GROUP == 1:
        print("   ✅ Консервативная стратегия (1 позиция на группу)")
    elif settings.CORRELATION_MAX_POSITIONS_PER_GROUP <= 2:
        print("   ⚠️  Сбалансированная стратегия (2 позиции на группу)")
    else:
        print("   🔴 Агрессивная стратегия (3+ позиций на группу)")
        print("      Высокий риск концентрации!")

    print()

    # Самые большие группы (риск концентрации)
    big_groups = [g for g in groups if len(g['symbols']) >= 5]
    if big_groups:
        print("   ⚠️  ВНИМАНИЕ: Обнаружены большие группы (≥5 пар):")
        for group in big_groups[:3]:  # Показываем топ-3
            print(f"      • {group['group_id']}: {len(group['symbols'])} пар, corr={group['avg_correlation']:.3f}")
        print()
        print("      Будьте осторожны - эти группы сильно ограничат торговлю!")
        print()

    print("=" * 80)
    print()
    print("ℹ️  Для изменения настроек редактируйте .env файл:")
    print()
    print("   CORRELATION_CHECK_ENABLED=true              # Включить/выключить")
    print("   CORRELATION_MAX_THRESHOLD=0.7               # Порог группировки")
    print("   CORRELATION_MAX_POSITIONS_PER_GROUP=1       # Лимит позиций")
    print("   CORRELATION_LOOKBACK_DAYS=30                # Период расчета")
    print()
    print("📖 Подробнее: см. CORRELATION_EXPLAINED.md")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        print(f"\n❌ ОШИБКА: {e}")
