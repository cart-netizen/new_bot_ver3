#!/usr/bin/env python3
"""
Анализ параметров сбора данных ML Data Collector
Проверка оптимальности настроек частоты сохранения, буферов и т.п.
"""


def analyze_data_collection_settings():
    """Анализ текущих настроек сбора данных."""

    print("\n" + "=" * 80)
    print("📊 АНАЛИЗ ПАРАМЕТРОВ СБОРА ДАННЫХ ML DATA COLLECTOR")
    print("=" * 80 + "\n")

    # === ТЕКУЩИЕ НАСТРОЙКИ ===
    print("🔧 ТЕКУЩИЕ НАСТРОЙКИ:")
    print("-" * 80)

    # Из backend/main.py:406-416
    max_samples_per_file = 300
    collection_interval = 10  # каждые N итераций
    max_buffer_memory_mb = 30  # МБ на символ

    # Из backend/config.py и .env
    analysis_interval = 0.5  # секунды (из backend/.env ANALYSIS_INTERVAL=0.5)
    # Из backend/.env TRADING_PAIRS (примерное количество)
    # TRUTHUSDT,BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT и т.д.
    symbols_count = 15  # типичное количество символов

    print(f"  • Analysis Interval:        {analysis_interval} сек")
    print(f"  • Collection Interval:      каждые {collection_interval} итераций")
    print(f"  • Max Samples per File:     {max_samples_per_file} семплов")
    print(f"  • Max Buffer Memory:        {max_buffer_memory_mb} МБ на символ")
    print(f"  • Торговых пар:             {symbols_count}")
    print()

    # === РАСЧЕТ ЧАСТОТЫ ===
    print("⏱️  РАСЧЕТ ЧАСТОТЫ СБОРА:")
    print("-" * 80)

    iterations_per_minute = 60 / analysis_interval
    iterations_per_hour = iterations_per_minute * 60

    collections_per_minute_per_symbol = iterations_per_minute / collection_interval
    collections_per_hour_per_symbol = collections_per_minute_per_symbol * 60

    print(f"  • Итераций анализа в минуту:     {iterations_per_minute:.1f}")
    print(f"  • Итераций анализа в час:        {iterations_per_hour:.0f}")
    print(f"  • Сборов данных в минуту (1 символ): {collections_per_minute_per_symbol:.1f}")
    print(f"  • Сборов данных в час (1 символ):    {collections_per_hour_per_symbol:.0f}")
    print(f"  • ВСЕГО сборов в час ({symbols_count} символов): {collections_per_hour_per_symbol * symbols_count:.0f}")
    print()

    # === РАСЧЕТ ВРЕМЕНИ ДО СОХРАНЕНИЯ ===
    print("💾 РАСЧЕТ ВРЕМЕНИ ДО СОХРАНЕНИЯ:")
    print("-" * 80)

    minutes_to_save = max_samples_per_file / collections_per_minute_per_symbol
    hours_to_save = minutes_to_save / 60

    print(f"  • Время до сохранения (1 символ):   {minutes_to_save:.1f} минут ({hours_to_save:.2f} часов)")
    print(f"  • Семплов в буфере перед сохранением: {max_samples_per_file}")
    print()

    # === РИСК ПОТЕРИ ДАННЫХ ===
    print("⚠️  РИСК ПОТЕРИ ДАННЫХ:")
    print("-" * 80)

    samples_at_risk_per_symbol = max_samples_per_file  # максимум в буфере
    samples_at_risk_total = samples_at_risk_per_symbol * symbols_count

    print(f"  • При крэше теряется до:         {minutes_to_save:.1f} минут данных")
    print(f"  • Семплов в риске (1 символ):    {samples_at_risk_per_symbol}")
    print(f"  • Семплов в риске ({symbols_count} символов): {samples_at_risk_total}")
    print()

    # === ИСПОЛЬЗОВАНИЕ ПАМЯТИ ===
    print("🧠 ИСПОЛЬЗОВАНИЕ ПАМЯТИ:")
    print("-" * 80)

    bytes_per_feature = 4  # float32
    features_count = 112
    bytes_per_feature_array = features_count * bytes_per_feature  # 448 bytes

    # Примерная оценка
    bytes_per_sample = bytes_per_feature_array + 200 + 300  # features + labels + metadata ≈ 950 bytes
    mb_per_sample = bytes_per_sample / (1024 * 1024)

    memory_per_symbol_at_save = max_samples_per_file * mb_per_sample
    memory_total_at_save = memory_per_symbol_at_save * symbols_count

    print(f"  • Байт на семпл (приблизительно):   ~{bytes_per_sample} bytes")
    print(f"  • МБ на семпл:                       ~{mb_per_sample:.4f} MB")
    print(f"  • Память на символ перед сохранением: ~{memory_per_symbol_at_save:.2f} MB")
    print(f"  • ОБЩАЯ память ({symbols_count} символов):    ~{memory_total_at_save:.2f} MB")
    print(f"  • Лимит памяти на символ (настройка): {max_buffer_memory_mb} MB")

    # Проверка превышения лимита
    if memory_per_symbol_at_save > max_buffer_memory_mb:
        print(f"  ⚠️  ВНИМАНИЕ: Память может превысить лимит на {memory_per_symbol_at_save - max_buffer_memory_mb:.2f} MB!")
    else:
        print(f"  ✅ Память в пределах лимита (запас: {max_buffer_memory_mb - memory_per_symbol_at_save:.2f} MB)")
    print()

    # === ОЦЕНКА КАЧЕСТВА НАСТРОЕК ===
    print("📈 ОЦЕНКА ТЕКУЩИХ НАСТРОЕК:")
    print("-" * 80)

    issues = []
    warnings = []
    good_points = []

    # Проверка времени до сохранения
    if minutes_to_save > 15:
        issues.append(f"Слишком редкое сохранение ({minutes_to_save:.1f} мин) - риск потери данных при крэше")
    elif minutes_to_save > 10:
        warnings.append(f"Сохранение каждые {minutes_to_save:.1f} минут - приемлемо, но можно чаще")
    else:
        good_points.append(f"Сохранение каждые {minutes_to_save:.1f} минут - хорошая частота")

    # Проверка частоты сбора
    if collections_per_minute_per_symbol < 5:
        warnings.append(f"Низкая частота сбора ({collections_per_minute_per_symbol:.1f}/мин) - мало данных")
    elif collections_per_minute_per_symbol > 20:
        warnings.append(f"Высокая частота сбора ({collections_per_minute_per_symbol:.1f}/мин) - возможна избыточность")
    else:
        good_points.append(f"Частота сбора {collections_per_minute_per_symbol:.1f}/мин - оптимально")

    # Проверка использования памяти
    memory_utilization = (memory_per_symbol_at_save / max_buffer_memory_mb) * 100
    if memory_utilization > 90:
        issues.append(f"Высокое использование памяти ({memory_utilization:.0f}%) - риск переполнения")
    elif memory_utilization > 70:
        warnings.append(f"Использование памяти {memory_utilization:.0f}% - близко к лимиту")
    else:
        good_points.append(f"Использование памяти {memory_utilization:.0f}% - в норме")

    # Вывод оценки
    if good_points:
        print("✅ ХОРОШО:")
        for point in good_points:
            print(f"   • {point}")
        print()

    if warnings:
        print("⚠️  ПРЕДУПРЕЖДЕНИЯ:")
        for warning in warnings:
            print(f"   • {warning}")
        print()

    if issues:
        print("❌ ПРОБЛЕМЫ:")
        for issue in issues:
            print(f"   • {issue}")
        print()

    # === РЕКОМЕНДАЦИИ ===
    print("💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:")
    print("-" * 80)

    # Рекомендуемые параметры
    recommended_save_minutes = 5  # Сохранять каждые 5 минут
    recommended_samples = int(collections_per_minute_per_symbol * recommended_save_minutes)

    # Альтернатива 1: Более частое сохранение
    alt1_samples = 150  # Половина от текущего
    alt1_minutes = alt1_samples / collections_per_minute_per_symbol

    # Альтернатива 2: Еще более частое
    alt2_samples = 100
    alt2_minutes = alt2_samples / collections_per_minute_per_symbol

    print("\n🎯 Вариант 1: БЕЗОПАСНЫЙ (рекомендуется)")
    print(f"   max_samples_per_file = {recommended_samples}")
    print(f"   └─ Сохранение каждые ~{recommended_save_minutes:.1f} минут")
    print(f"   └─ Риск потери данных: минимальный")
    print(f"   └─ Частота сохранений: ~{60/recommended_save_minutes:.0f} раз в час на символ")

    print("\n🎯 Вариант 2: СБАЛАНСИРОВАННЫЙ")
    print(f"   max_samples_per_file = {alt1_samples}")
    print(f"   └─ Сохранение каждые ~{alt1_minutes:.1f} минут")
    print(f"   └─ Риск потери данных: низкий")
    print(f"   └─ Частота сохранений: ~{60/alt1_minutes:.0f} раз в час на символ")

    print("\n🎯 Вариант 3: ЧАСТОЕ СОХРАНЕНИЕ")
    print(f"   max_samples_per_file = {alt2_samples}")
    print(f"   └─ Сохранение каждые ~{alt2_minutes:.1f} минут")
    print(f"   └─ Риск потери данных: очень низкий")
    print(f"   └─ Частота сохранений: ~{60/alt2_minutes:.0f} раз в час на символ")
    print(f"   └─ ⚠️  Больше файлов, больше операций I/O")

    print("\n🎯 Вариант 4: ТЕКУЩИЙ (для сравнения)")
    print(f"   max_samples_per_file = {max_samples_per_file}")
    print(f"   └─ Сохранение каждые ~{minutes_to_save:.1f} минут")
    print(f"   └─ Риск потери данных: {'⚠️  СРЕДНИЙ' if minutes_to_save > 15 else '✅ приемлемый'}")
    print(f"   └─ Частота сохранений: ~{60/minutes_to_save:.1f} раз в час на символ")

    print()

    # === ДОПОЛНИТЕЛЬНЫЕ РЕКОМЕНДАЦИИ ===
    print("🔧 ДОПОЛНИТЕЛЬНЫЕ РЕКОМЕНДАЦИИ:")
    print("-" * 80)

    print("\n1. ЧАСТОТА СБОРА (collection_interval):")
    print(f"   Текущее: каждые {collection_interval} итераций ({analysis_interval * collection_interval} сек)")
    print(f"   Рекомендация: ОСТАВИТЬ БЕЗ ИЗМЕНЕНИЙ")
    print(f"   Причина: {collections_per_minute_per_symbol:.1f} сборов/мин - оптимально для ML обучения")

    print("\n2. РАЗМЕР БУФЕРА ПАМЯТИ (max_buffer_memory_mb):")
    print(f"   Текущее: {max_buffer_memory_mb} МБ на символ")
    if memory_per_symbol_at_save < max_buffer_memory_mb * 0.5:
        print(f"   Рекомендация: Можно УМЕНЬШИТЬ до {int(memory_per_symbol_at_save * 1.5)} МБ")
        print(f"   Причина: Сейчас используется только {memory_utilization:.0f}%")
    else:
        print(f"   Рекомендация: ОСТАВИТЬ БЕЗ ИЗМЕНЕНИЙ")
        print(f"   Причина: Использование {memory_utilization:.0f}% - в норме")

    print("\n3. ИНТЕРВАЛ АНАЛИЗА (ANALYSIS_INTERVAL):")
    print(f"   Текущее: {analysis_interval} сек")
    print(f"   Рекомендация: ОСТАВИТЬ БЕЗ ИЗМЕНЕНИЙ")
    print(f"   Причина: Оптимальный баланс между частотой и производительностью")

    print()
    print("=" * 80)
    print("✅ АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 80 + "\n")

    # === ИТОГОВЫЕ ВЫВОДЫ ===
    print("📋 ИТОГОВЫЕ ВЫВОДЫ:")
    print("-" * 80)

    if minutes_to_save > 15:
        print("\n⚠️  ВНИМАНИЕ: Текущие настройки имеют ВЫСОКИЙ риск потери данных!")
        print(f"   При крэше бота можно потерять до {minutes_to_save:.1f} минут данных")
        print(f"   ({samples_at_risk_total} семплов для {symbols_count} символов)")
        print("\n   РЕКОМЕНДУЕТСЯ:")
        print(f"   1. Уменьшить max_samples_per_file до {recommended_samples} (сохранение каждые ~5 минут)")
        print(f"   2. Или до {alt1_samples} (сохранение каждые ~{alt1_minutes:.1f} минут)")
    elif minutes_to_save > 10:
        print("\n✅ Текущие настройки ПРИЕМЛЕМЫ, но можно улучшить:")
        print(f"   • Сохранение каждые {minutes_to_save:.1f} минут - работает, но не оптимально")
        print(f"\n   ОПЦИОНАЛЬНО (для большей безопасности):")
        print(f"   • Уменьшить max_samples_per_file до {recommended_samples} (каждые ~5 минут)")
    else:
        print("\n✅ Текущие настройки ОПТИМАЛЬНЫ!")
        print(f"   • Сохранение каждые {minutes_to_save:.1f} минут - отличный баланс")
        print(f"   • Риск потери данных: минимальный")
        print(f"   • Частота I/O операций: разумная")

    print()
    print("=" * 80 + "\n")


if __name__ == "__main__":
    analyze_data_collection_settings()
