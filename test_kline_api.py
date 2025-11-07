"""
Тестовый скрипт для проверки загрузки kline данных через Bybit API.
"""
import asyncio
from datetime import datetime, timedelta
from backend.exchange.rest_client import rest_client
from backend.core.logger import get_logger

logger = get_logger(__name__)


async def test_kline_loading():
    """Тест загрузки исторических свечей."""

    # Инициализация rest_client
    await rest_client.initialize()

    try:
        # Параметры теста
        symbol = "BTCUSDT"
        interval = "1"  # 1 минута

        # Период: последние 24 часа
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        logger.info("=" * 80)
        logger.info("ТЕСТ ЗАГРУЗКИ KLINE ДАННЫХ")
        logger.info("=" * 80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Interval: {interval}")
        logger.info(f"Start: {start_time.isoformat()} ({start_ms})")
        logger.info(f"End: {end_time.isoformat()} ({end_ms})")
        logger.info(f"Period: {(end_time - start_time).total_seconds() / 3600:.1f} hours")
        logger.info("=" * 80)

        # Запрос к API
        logger.info("📡 Запрос к Bybit API...")
        klines = await rest_client.get_kline(
            symbol=symbol,
            interval=interval,
            start=start_ms,
            end=end_ms,
            limit=1000
        )

        # Результаты
        logger.info("=" * 80)
        logger.info(f"✅ Получено {len(klines)} свечей")
        logger.info("=" * 80)

        if klines:
            first_kline = klines[0]
            last_kline = klines[-1]

            logger.info(f"Первая свеча:")
            logger.info(f"  Raw: {first_kline}")
            logger.info(f"  Time: {datetime.fromtimestamp(int(first_kline[0]) / 1000).isoformat()}")
            logger.info(f"  OHLC: O={first_kline[1]} H={first_kline[2]} L={first_kline[3]} C={first_kline[4]}")
            logger.info(f"  Volume: {first_kline[5]}")

            logger.info(f"\nПоследняя свеча:")
            logger.info(f"  Raw: {last_kline}")
            logger.info(f"  Time: {datetime.fromtimestamp(int(last_kline[0]) / 1000).isoformat()}")
            logger.info(f"  OHLC: O={last_kline[1]} H={last_kline[2]} L={last_kline[3]} C={last_kline[4]}")
            logger.info(f"  Volume: {last_kline[5]}")
        else:
            logger.error("⚠️ API вернул пустой список!")
            logger.error("Возможные причины:")
            logger.error("  1. Неверный формат временных меток")
            logger.error("  2. Неверный интервал")
            logger.error("  3. Период в будущем")
            logger.error("  4. Проблемы с доступом к API")

        logger.info("=" * 80)

        # Тест без временных меток (последние N свечей)
        logger.info("\n📡 Тест без временных меток (последние 100 свечей)...")
        klines_no_time = await rest_client.get_kline(
            symbol=symbol,
            interval=interval,
            limit=100
        )

        logger.info(f"✅ Получено {len(klines_no_time)} свечей без временных меток")

        if klines_no_time:
            first = klines_no_time[0]
            logger.info(f"Первая свеча: time={datetime.fromtimestamp(int(first[0]) / 1000).isoformat()}, close={first[4]}")

    finally:
        await rest_client.close()
        logger.info("\n✅ Тест завершен")


if __name__ == "__main__":
    asyncio.run(test_kline_loading())
