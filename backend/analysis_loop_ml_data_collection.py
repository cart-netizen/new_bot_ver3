"""
Упрощенная версия analysis loop ТОЛЬКО для сбора данных для ML.

НАЗНАЧЕНИЕ:
Запуск на облачном сервере для экономии ресурсов.
Собирает данные для обучения ML модели БЕЗ торговли.

ЧТО ВКЛЮЧЕНО:
✅ Получение orderbook snapshots
✅ Получение candles
✅ Извлечение ML features (110+ признаков)
✅ Детекция манипуляций (spoofing/layering)
✅ S/R levels detection
✅ ML Data Collection для обучения

ЧТО УБРАНО:
❌ Генерация торговых сигналов (IntegratedEngine)
❌ ML Validation сигналов
❌ Risk & Quality Checks
❌ Execution Manager / размещение ордеров
❌ Real-time UI broadcasting

ИСПОЛЬЗОВАНИЕ:
Можно использовать как отдельную функцию, передав экземпляр bot_controller:

```python
from analysis_loop_ml_data_collection import ml_data_collection_loop

# В main.py вместо self._analysis_loop_ml_enhanced():
await ml_data_collection_loop(
    bot_controller=self,
    symbols=self.symbols,
    analysis_interval=settings.ANALYSIS_INTERVAL
)
```
"""

import asyncio
import time
import traceback
from typing import List, Dict, Optional, Any
from datetime import datetime

from core.logger import get_logger
from config import settings
from models.signal import SignalType

logger = get_logger(__name__)


async def ml_data_collection_loop(
    bot_controller: Any,  # BotController instance
    symbols: List[str],
    analysis_interval: int = 60
):
    """
    Упрощенный цикл анализа ТОЛЬКО для сбора ML данных.

    Args:
        bot_controller: Экземпляр BotController с доступом к:
            - orderbook_managers
            - candle_managers
            - ml_feature_pipeline
            - ml_data_collector
            - sr_detector
            - spoofing_detector (optional)
            - layering_detector (optional)
            - orderbook_analyzer
            - market_analyzer
            - websocket_manager
            - status
        symbols: Список торговых пар для анализа
        analysis_interval: Интервал анализа в секундах
    """

    logger.info("=" * 80)
    logger.info("🔬 ML DATA COLLECTION LOOP ЗАПУЩЕН (УПРОЩЕННАЯ ВЕРСИЯ)")
    logger.info("=" * 80)
    logger.info(f"📊 Режим: ML Data Collection ONLY (без торговли)")
    logger.info(f"⏱️ Интервал анализа: {analysis_interval}с")
    logger.info(f"📈 Торговые пары: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")

    # Проверка доступности компонентов
    has_ml_feature_pipeline = bot_controller.ml_feature_pipeline is not None
    has_ml_data_collector = bot_controller.ml_data_collector is not None
    has_sr_detector = bot_controller.sr_detector is not None
    has_spoofing_detector = hasattr(bot_controller, 'spoofing_detector') and bot_controller.spoofing_detector
    has_layering_detector = hasattr(bot_controller, 'layering_detector') and bot_controller.layering_detector
    has_orderbook_analyzer = bot_controller.orderbook_analyzer is not None
    has_market_analyzer = bot_controller.market_analyzer is not None

    logger.info("📦 Статус компонентов для сбора данных:")
    logger.info(f"   ├─ ML Feature Pipeline: {'✅' if has_ml_feature_pipeline else '❌ КРИТИЧНО'}")
    logger.info(f"   ├─ ML Data Collector: {'✅' if has_ml_data_collector else '❌ КРИТИЧНО'}")
    logger.info(f"   ├─ S/R Detector: {'✅' if has_sr_detector else '⚠️ Опционально'}")
    logger.info(f"   ├─ Spoofing Detector: {'✅' if has_spoofing_detector else '⚠️ Опционально'}")
    logger.info(f"   ├─ Layering Detector: {'✅' if has_layering_detector else '⚠️ Опционально'}")
    logger.info(f"   ├─ OrderBook Analyzer: {'✅' if has_orderbook_analyzer else '❌ КРИТИЧНО'}")
    logger.info(f"   └─ Market Analyzer: {'✅' if has_market_analyzer else '⚠️ Опционально'}")
    logger.info("=" * 80)

    # КРИТИЧЕСКАЯ ПРОВЕРКА
    if not has_ml_feature_pipeline or not has_ml_data_collector:
        logger.critical(
            "🚨 КРИТИЧЕСКАЯ ОШИБКА: ML Feature Pipeline или ML Data Collector не инициализированы!"
        )
        return

    # Инициализация счетчиков
    error_count = {}
    max_consecutive_errors = 5
    cycle_number = 0

    # Статистика
    stats = {
        'analysis_cycles': 0,
        'ml_data_collected': 0,
        'manipulations_detected': 0,
        'errors': 0,
        'warnings': 0
    }

    # Кеш предыдущих состояний
    prev_orderbook_snapshots = {}
    prev_candles = {}

    logger.info("✅ ML Data Collection Loop готов к работе")

    # ========================================================================
    # ГЛАВНЫЙ ЦИКЛ СБОРА ДАННЫХ
    # ========================================================================

    from models.enums import BotStatus

    while bot_controller.status == BotStatus.RUNNING:
        cycle_start = time.time()
        cycle_number += 1

        if not bot_controller.websocket_manager.is_all_connected():
            if cycle_number <= 5:
                logger.info(f"⏳ Цикл #{cycle_number}: WebSocket не подключен, ждём...")
            await asyncio.sleep(1)
            continue

        if cycle_number <= 5:
            logger.info(f"✅ Цикл #{cycle_number}: WebSocket подключен, сбор данных по {len(symbols)} символам")

        try:
            # ============================================================
            # ОБРАБОТКА КАЖДОГО СИМВОЛА
            # ============================================================

            for symbol in symbols:
                symbol_start = time.time()

                if cycle_number <= 5:
                    logger.info(f"  🔬 [{symbol}] Начало сбора данных в цикле #{cycle_number}")

                # Инициализация error counter
                if symbol not in error_count:
                    error_count[symbol] = 0

                # Пропуск при превышении ошибок
                if error_count[symbol] >= max_consecutive_errors:
                    if cycle_number % 10 == 0:
                        logger.warning(
                            f"⚠️ [{symbol}] Пропуск: {error_count[symbol]} "
                            f"последовательных ошибок (лимит: {max_consecutive_errors})"
                        )
                    continue

                try:
                    # ============================================================
                    # ШАГ 1: ПОЛУЧЕНИЕ MARKET DATA
                    # ============================================================

                    ob_manager = bot_controller.orderbook_managers[symbol]
                    candle_manager = bot_controller.candle_managers[symbol]

                    # Проверка данных
                    if not ob_manager.snapshot_received:
                        if cycle_number <= 5:
                            logger.info(f"  ⏭️  [{symbol}] OrderBook snapshot не получен")
                        continue

                    orderbook_snapshot = ob_manager.get_snapshot()
                    if not orderbook_snapshot:
                        if cycle_number <= 5:
                            logger.info(f"  ⏭️  [{symbol}] OrderBook не готов")
                        continue

                    candles = candle_manager.get_candles()
                    if not candles or len(candles) < 50:
                        if cycle_number <= 5:
                            logger.info(
                                f"  ⏭️  [{symbol}] Недостаточно свечей: "
                                f"{len(candles) if candles else 0}/50"
                            )
                        continue

                    current_price = orderbook_snapshot.mid_price
                    if current_price is None:
                        if cycle_number <= 5:
                            logger.info(f"  ⏭️  [{symbol}] Нет текущей цены")
                        continue

                    if cycle_number <= 5:
                        logger.info(
                            f"  ✅ [{symbol}] Данные готовы: "
                            f"price={current_price:.2f}, candles={len(candles)}"
                        )

                    # ============================================================
                    # ШАГ 2: АНАЛИЗ ORDERBOOK И MARKET METRICS
                    # ============================================================

                    orderbook_metrics = bot_controller.orderbook_analyzer.analyze(ob_manager)

                    market_metrics = None
                    if has_market_analyzer:
                        market_metrics = bot_controller.market_analyzer.analyze_symbol(
                            symbol,
                            ob_manager
                        )

                    market_volatility = None
                    if market_metrics and hasattr(market_metrics, 'volatility'):
                        market_volatility = market_metrics.volatility

                    logger.debug(
                        f"[{symbol}] Market Data: "
                        f"price={current_price:.2f}, "
                        f"spread={orderbook_metrics.spread:.2f}bps, "
                        f"imbalance={orderbook_metrics.imbalance:.3f}"
                    )

                    # ============================================================
                    # ШАГ 3: ПОЛУЧЕНИЕ ПРЕДЫДУЩИХ СОСТОЯНИЙ
                    # ============================================================

                    prev_orderbook = prev_orderbook_snapshots.get(symbol)
                    prev_candle = prev_candles.get(symbol)

                    # ============================================================
                    # ШАГ 4: ДЕТЕКЦИЯ МАНИПУЛЯЦИЙ (для меток данных)
                    # ============================================================

                    manipulation_detected = False
                    manipulation_types = []

                    # Spoofing Detection
                    if has_spoofing_detector:
                        try:
                            bot_controller.spoofing_detector.update(orderbook_snapshot)
                            has_spoofing = bot_controller.spoofing_detector.is_spoofing_active(
                                symbol,
                                time_window_seconds=60
                            )
                            if has_spoofing:
                                manipulation_detected = True
                                manipulation_types.append("spoofing")
                        except Exception as e:
                            logger.debug(f"[{symbol}] Spoofing Detector error: {e}")

                    # Layering Detection
                    if has_layering_detector:
                        try:
                            bot_controller.layering_detector.update(orderbook_snapshot)
                            has_layering = bot_controller.layering_detector.is_layering_active(
                                symbol,
                                time_window_seconds=60
                            )
                            if has_layering:
                                manipulation_detected = True
                                manipulation_types.append("layering")
                        except Exception as e:
                            logger.debug(f"[{symbol}] Layering Detector error: {e}")

                    if manipulation_detected:
                        logger.info(
                            f"⚠️ [{symbol}] МАНИПУЛЯЦИИ: {', '.join(manipulation_types).upper()}"
                        )
                        stats['manipulations_detected'] += 1

                    # ============================================================
                    # ШАГ 5: S/R LEVELS DETECTION
                    # ============================================================

                    sr_levels = None
                    if has_sr_detector:
                        try:
                            sr_levels = bot_controller.sr_detector.detect_levels(symbol)
                            if sr_levels and cycle_number <= 5:
                                supports = [lvl for lvl in sr_levels if lvl.level_type == "support"]
                                resistances = [lvl for lvl in sr_levels if lvl.level_type == "resistance"]
                                logger.debug(
                                    f"[{symbol}] S/R Levels: "
                                    f"{len(supports)} supports, {len(resistances)} resistances"
                                )
                        except Exception as e:
                            logger.debug(f"[{symbol}] S/R Detection error: {e}")

                    # ============================================================
                    # ШАГ 6: ML FEATURE EXTRACTION
                    # ============================================================

                    feature_vector = None
                    try:
                        feature_vector = await bot_controller.ml_feature_pipeline.extract_features_enhanced(
                            symbol=symbol,
                            orderbook_snapshot=orderbook_snapshot,
                            candles=candles,
                            orderbook_metrics=orderbook_metrics,
                            sr_levels=sr_levels if sr_levels else None,
                            prev_orderbook=prev_orderbook,
                            prev_candle=prev_candle
                        )

                        if feature_vector:
                            data_quality = feature_vector.metadata.get('data_quality', {})
                            logger.debug(
                                f"[{symbol}] Features извлечены: "
                                f"{feature_vector.feature_count} признаков, "
                                f"prev_snapshot={data_quality.get('has_prev_orderbook', False)}, "
                                f"prev_candle={data_quality.get('has_prev_candle', False)}"
                            )
                        else:
                            logger.warning(f"[{symbol}] Feature extraction вернул None")
                            continue

                    except Exception as e:
                        logger.error(f"[{symbol}] Ошибка ML Feature Extraction: {e}")
                        logger.debug(traceback.format_exc())
                        stats['errors'] += 1
                        continue

                    # ============================================================
                    # ШАГ 7: ML DATA COLLECTION (ГЛАВНАЯ ЦЕЛЬ)
                    # ============================================================

                    if feature_vector:
                        try:
                            # Проверяем нужно ли собирать данные
                            if bot_controller.ml_data_collector.should_collect():
                                # Подготовка sample
                                sample_data = {
                                    'symbol': symbol,
                                    'timestamp': int(time.time() * 1000),
                                    'features': feature_vector,
                                    'price': current_price,
                                    'orderbook_snapshot': {
                                        'best_bid': orderbook_snapshot.best_bid,
                                        'best_ask': orderbook_snapshot.best_ask,
                                        'mid_price': orderbook_snapshot.mid_price,
                                        'spread': orderbook_snapshot.spread,
                                        'imbalance': orderbook_metrics.imbalance
                                    },
                                    'market_metrics': {
                                        'volatility': market_volatility,
                                        'volume': candles[-1].volume if candles and len(candles) > 0 else None,
                                        'momentum': (
                                            ((candles[-1].close - candles[-2].close) / candles[-2].close) * 100
                                            if candles and len(candles) > 1 and candles[-2].close > 0
                                            else None
                                        )
                                    },
                                    'manipulations': {
                                        'detected': manipulation_detected,
                                        'types': manipulation_types
                                    }
                                }

                                # Сохранение sample
                                await bot_controller.ml_data_collector.collect_sample(
                                    symbol=symbol,
                                    feature_vector=feature_vector,
                                    orderbook_snapshot=orderbook_snapshot,
                                    market_metrics=market_metrics,
                                    executed_signal=None  # Нет сигнала в режиме сбора данных
                                )

                                stats['ml_data_collected'] += 1
                                logger.info(f"📊 [{symbol}] ML Data sample собран (total: {stats['ml_data_collected']})")

                        except Exception as e:
                            logger.error(f"[{symbol}] Ошибка ML Data Collection: {e}")
                            stats['errors'] += 1

                    # ============================================================
                    # ШАГ 8: СОХРАНЕНИЕ ТЕКУЩЕГО СОСТОЯНИЯ
                    # ============================================================

                    # Сохраняем текущий snapshot для следующей итерации
                    prev_orderbook_snapshots[symbol] = orderbook_snapshot

                    # Сохраняем текущую свечу
                    if candles and len(candles) > 0:
                        prev_candles[symbol] = candles[-1]

                    # ============================================================
                    # УСПЕШНОЕ ЗАВЕРШЕНИЕ
                    # ============================================================

                    # Сброс error counter
                    error_count[symbol] = 0

                    # Время выполнения
                    symbol_elapsed = time.time() - symbol_start
                    if symbol_elapsed > 5:
                        logger.warning(
                            f"⏱️  [{symbol}] Медленное выполнение: {symbol_elapsed:.2f}s"
                        )

                except Exception as e:
                    error_count[symbol] = error_count.get(symbol, 0) + 1
                    stats['errors'] += 1
                    logger.error(
                        f"❌ [{symbol}] Ошибка сбора данных: {e} "
                        f"(ошибка #{error_count[symbol]})"
                    )
                    logger.debug(traceback.format_exc())

            # ============================================================
            # ЗАВЕРШЕНИЕ ЦИКЛА
            # ============================================================

            stats['analysis_cycles'] += 1

            # Статистика каждые 10 циклов
            if cycle_number % 10 == 0:
                cycle_elapsed = time.time() - cycle_start
                logger.info("=" * 80)
                logger.info(f"📈 СТАТИСТИКА СБОРА ДАННЫХ (Цикл #{cycle_number})")
                logger.info(f"   ├─ Циклов анализа: {stats['analysis_cycles']}")
                logger.info(f"   ├─ ML данных собрано: {stats['ml_data_collected']}")
                logger.info(f"   ├─ Манипуляций обнаружено: {stats['manipulations_detected']}")
                logger.info(f"   ├─ Ошибок: {stats['errors']}")
                logger.info(f"   └─ Время цикла: {cycle_elapsed:.2f}s")
                logger.info("=" * 80)

            # Ожидание следующего цикла
            await asyncio.sleep(analysis_interval)

        except Exception as e:
            stats['errors'] += 1
            logger.error(f"❌ Критическая ошибка в цикле сбора данных: {e}", exc_info=True)
            await asyncio.sleep(analysis_interval)

    logger.info("=" * 80)
    logger.info("🛑 ML DATA COLLECTION LOOP ОСТАНОВЛЕН")
    logger.info(f"   Финальная статистика:")
    logger.info(f"   ├─ Циклов анализа: {stats['analysis_cycles']}")
    logger.info(f"   ├─ ML данных собрано: {stats['ml_data_collected']}")
    logger.info(f"   ├─ Манипуляций обнаружено: {stats['manipulations_detected']}")
    logger.info(f"   └─ Ошибок: {stats['errors']}")
    logger.info("=" * 80)
