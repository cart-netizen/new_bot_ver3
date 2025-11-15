#!/usr/bin/env python3
"""
Анализ РЕАЛЬНОГО времени цикла анализа и пересчет параметров
"""


def analyze_with_real_cycle_time():
    """Анализ с учетом реального времени цикла."""

    print("\n" + "=" * 80)
    print("📊 АНАЛИЗ РЕАЛЬНЫХ ПАРАМЕТРОВ СБОРА ДАННЫХ")
    print("=" * 80 + "\n")

    # === ТЕКУЩИЕ НАСТРОЙКИ ===
    print("🔧 ТЕКУЩИЕ НАСТРОЙКИ:")
    print("-" * 80)

    max_samples_per_file = 100
    collection_interval = 10
    max_buffer_memory_mb = 20
    cleanup_interval_cycles = 1440
    symbols_count = 15

    # КРИТИЧНО: Используем РЕАЛЬНОЕ время цикла из логов
    analysis_interval_config = 0.5  # В настройках
    analysis_interval_real = 3.0    # Реальное (из логов: 2-4.5 сек, среднее ~3)

    print(f"  • Analysis Interval (config): {analysis_interval_config} сек")
    print(f"  • Analysis Interval (REAL):   {analysis_interval_real} сек ⚠️")
    print(f"  • Разница:                     {analysis_interval_real/analysis_interval_config:.1f}x медленнее")
    print(f"  • Collection Interval:         каждые {collection_interval} итераций")
    print(f"  • Max Samples per File:        {max_samples_per_file} семплов")
    print(f"  • Cleanup Interval:            каждые {cleanup_interval_cycles} циклов")
    print(f"  • Торговых пар:                {symbols_count}")
    print()

    # === РАСЧЕТ С РЕАЛЬНЫМ ВРЕМЕНЕМ ===
    print("⏱️  РАСЧЕТ С РЕАЛЬНЫМ ВРЕМЕНЕМ ЦИКЛА:")
    print("-" * 80)

    cycles_per_minute = 60 / analysis_interval_real
    cycles_per_hour = cycles_per_minute * 60

    collections_per_minute = cycles_per_minute / collection_interval
    collections_per_hour = collections_per_minute * 60

    print(f"  • Циклов анализа в минуту:       {cycles_per_minute:.1f}")
    print(f"  • Циклов анализа в час:          {cycles_per_hour:.0f}")
    print(f"  • Сборов данных в минуту:        {collections_per_minute:.1f}")
    print(f"  • Сборов данных в час:           {collections_per_hour:.0f}")
    print()

    # === ВРЕМЯ ДО СОХРАНЕНИЯ ===
    print("💾 ВРЕМЯ ДО СОХРАНЕНИЯ (РЕАЛЬНОЕ):")
    print("-" * 80)

    minutes_to_save = max_samples_per_file / collections_per_minute
    hours_to_save = minutes_to_save / 60

    print(f"  • Время до сохранения:           {minutes_to_save:.1f} минут ({hours_to_save:.2f} часов)")
    print(f"  • Семплов в буфере:              {max_samples_per_file}")

    if minutes_to_save > 20:
        print(f"  ❌ КРИТИЧНО: Слишком долго! Данные не сохраняются!")
    elif minutes_to_save > 10:
        print(f"  ⚠️  ВНИМАНИЕ: Долгое ожидание, риск потери данных")
    else:
        print(f"  ✅ Приемлемое время сохранения")
    print()

    # === CLEANUP ИНТЕРВАЛ ===
    print("🧹 CLEANUP ИНТЕРВАЛ (РЕАЛЬНЫЙ):")
    print("-" * 80)

    cleanup_minutes = (cleanup_interval_cycles * analysis_interval_real) / 60
    cleanup_hours = cleanup_minutes / 60

    samples_at_cleanup = cleanup_interval_cycles / collection_interval

    print(f"  • Cleanup интервал:              {cleanup_minutes:.1f} минут ({cleanup_hours:.2f} часов)")
    print(f"  • Семплов накопится за cleanup:  {samples_at_cleanup:.0f}")
    print(f"  • Целевой порог:                 {max_samples_per_file}")

    if samples_at_cleanup < max_samples_per_file:
        print(f"  ❌ ПРОБЛЕМА: Cleanup НЕ ДОЖДЕТСЯ накопления {max_samples_per_file} семплов!")
        print(f"     Cleanup произойдет через {cleanup_minutes:.0f} мин с {samples_at_cleanup:.0f} семплами")
        print(f"     Автосохранение через {minutes_to_save:.0f} мин с {max_samples_per_file} семплами")
        print(f"     Cleanup успеет первым? {cleanup_minutes < minutes_to_save}")
    elif samples_at_cleanup > max_samples_per_file * 1.2:
        print(f"  ✅ Cleanup происходит ПОСЛЕ автосохранения - норма")
    else:
        print(f"  ℹ️  Cleanup близко к порогу сохранения")
    print()

    # === ПРОБЛЕМЫ ===
    print("❌ ОБНАРУЖЕННЫЕ ПРОБЛЕМЫ:")
    print("-" * 80)

    problems = []

    if minutes_to_save > 30:
        problems.append(f"1. Сохранение каждые {minutes_to_save:.0f} минут - КРИТИЧНО долго!")
        problems.append(f"   Данные не сохраняются за разумное время")

    if cleanup_minutes < minutes_to_save:
        problems.append(f"2. Cleanup ({cleanup_minutes:.0f} мин) происходит ДО автосохранения ({minutes_to_save:.0f} мин)")
        problems.append(f"   Будут сохраняться только {samples_at_cleanup:.0f} семплов вместо {max_samples_per_file}")

    if analysis_interval_real > analysis_interval_config * 2:
        problems.append(f"3. Реальный цикл ({analysis_interval_real}s) >> настройка ({analysis_interval_config}s)")
        problems.append(f"   Все расчеты на основе неверных предположений!")

    if problems:
        for p in problems:
            print(f"  {p}")
    else:
        print("  ✅ Проблем не обнаружено")
    print()

    # === РЕКОМЕНДУЕМЫЕ ИСПРАВЛЕНИЯ ===
    print("💡 РЕКОМЕНДУЕМЫЕ ИСПРАВЛЕНИЯ:")
    print("=" * 80)

    # Целевое время сохранения: 10-15 минут
    target_save_minutes = 12

    # Вариант 1: Уменьшить max_samples_per_file
    recommended_samples_v1 = int(collections_per_minute * target_save_minutes)

    # Вариант 2: Уменьшить collection_interval
    recommended_interval_v2 = int(collection_interval * (minutes_to_save / target_save_minutes))
    if recommended_interval_v2 < 1:
        recommended_interval_v2 = 1

    # Вариант 3: Комбинация
    recommended_samples_v3 = 60
    recommended_interval_v3 = 5

    print("\n🎯 Вариант 1: УМЕНЬШИТЬ max_samples_per_file (рекомендуется)")
    print(f"   max_samples_per_file = {recommended_samples_v1}")
    print(f"   collection_interval = {collection_interval} (без изменений)")
    print(f"   └─ Сохранение каждые ~{target_save_minutes} минут")
    print(f"   └─ Сборов в минуту: {collections_per_minute:.1f}")

    print("\n🎯 Вариант 2: УМЕНЬШИТЬ collection_interval")
    print(f"   max_samples_per_file = {max_samples_per_file} (без изменений)")
    print(f"   collection_interval = {recommended_interval_v2}")
    recommended_coll_per_min_v2 = cycles_per_minute / recommended_interval_v2
    recommended_save_min_v2 = max_samples_per_file / recommended_coll_per_min_v2
    print(f"   └─ Сохранение каждые ~{recommended_save_min_v2:.1f} минут")
    print(f"   └─ Сборов в минуту: {recommended_coll_per_min_v2:.1f}")

    print("\n🎯 Вариант 3: КОМБИНАЦИЯ (оптимально)")
    print(f"   max_samples_per_file = {recommended_samples_v3}")
    print(f"   collection_interval = {recommended_interval_v3}")
    recommended_coll_per_min_v3 = cycles_per_minute / recommended_interval_v3
    recommended_save_min_v3 = recommended_samples_v3 / recommended_coll_per_min_v3
    print(f"   └─ Сохранение каждые ~{recommended_save_min_v3:.1f} минут")
    print(f"   └─ Сборов в минуту: {recommended_coll_per_min_v3:.1f}")
    print(f"   └─ Данных в час: {recommended_coll_per_min_v3 * 60:.0f} семплов/символ")

    print("\n🎯 Вариант 4: АГРЕССИВНЫЙ (если цикл очень медленный)")
    aggressive_samples = 40
    aggressive_interval = 3
    aggressive_coll_per_min = cycles_per_minute / aggressive_interval
    aggressive_save_min = aggressive_samples / aggressive_coll_per_min
    print(f"   max_samples_per_file = {aggressive_samples}")
    print(f"   collection_interval = {aggressive_interval}")
    print(f"   └─ Сохранение каждые ~{aggressive_save_min:.1f} минут")
    print(f"   └─ Сборов в минуту: {aggressive_coll_per_min:.1f}")
    print(f"   └─ Данных в час: {aggressive_coll_per_min * 60:.0f} семплов/символ")

    # === CLEANUP КОРРЕКТИРОВКА ===
    print("\n🧹 CLEANUP INTERVAL (корректировка):")
    print("-" * 80)

    # Cleanup должен быть в 1.5-2x больше времени до автосохранения
    target_cleanup_minutes = target_save_minutes * 1.5
    target_cleanup_cycles = int((target_cleanup_minutes * 60) / analysis_interval_real)

    print(f"  • Текущий: {cleanup_interval_cycles} циклов = {cleanup_minutes:.0f} мин")
    print(f"  • Рекомендуемый: {target_cleanup_cycles} циклов = {target_cleanup_minutes:.0f} мин")
    print(f"  • Причина: Cleanup должен быть в 1.5x больше времени автосохранения")

    print()
    print("=" * 80)
    print("✅ АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 80 + "\n")

    # === ИТОГОВЫЕ РЕКОМЕНДАЦИИ ===
    print("📋 ИТОГОВЫЕ РЕКОМЕНДАЦИИ:")
    print("-" * 80)
    print("\n⚠️  КРИТИЧНО: Реальный цикл анализа (~3 сек) сильно отличается от настройки (0.5 сек)!")
    print(f"   Это привело к проблемам: данные не сохраняются {minutes_to_save:.0f} минут!")
    print()
    print("🔧 РЕКОМЕНДУЕМЫЕ ИЗМЕНЕНИЯ (backend/main.py):")
    print()
    print(f"   # ИСПРАВЛЕНИЕ 1: Уменьшить max_samples_per_file")
    print(f"   max_samples_per_file={recommended_samples_v3}  # БЫЛО: {max_samples_per_file}")
    print()
    print(f"   # ИСПРАВЛЕНИЕ 2: Уменьшить collection_interval")
    print(f"   collection_interval={recommended_interval_v3}  # БЫЛО: {collection_interval}")
    print()
    print(f"   # ИСПРАВЛЕНИЕ 3: Скорректировать cleanup_interval")
    print(f"   cleanup_interval_cycles={target_cleanup_cycles}  # БЫЛО: {cleanup_interval_cycles}")
    print()
    print("📊 РЕЗУЛЬТАТ:")
    print(f"   • Сохранение каждые ~{recommended_save_min_v3:.1f} минут (вместо {minutes_to_save:.0f})")
    print(f"   • Cleanup каждые ~{target_cleanup_minutes:.0f} минут (вместо {cleanup_minutes:.0f})")
    print(f"   • Данных в час: {recommended_coll_per_min_v3 * 60:.0f} семплов/символ")
    print()
    print("=" * 80 + "\n")


if __name__ == "__main__":
    analyze_with_real_cycle_time()
