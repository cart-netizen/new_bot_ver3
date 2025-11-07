"""
Historical Data Handler - управление историческими данными для бэктестинга.

Функции:
- Загрузка исторических свечей (candles) из Bybit API
- Загрузка снимков orderbook (если доступны)
- Валидация данных (gaps, outliers, quality)
- Кэширование для ускорения повторных запусков
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import pickle
import hashlib

from backend.core.logger import get_logger
from backend.strategy.candle_manager import Candle
from backend.models.orderbook import OrderBookSnapshot, OrderBookMetrics
from backend.exchange.rest_client import rest_client
from backend.utils.helpers import get_timestamp_ms

logger = get_logger(__name__)


class DataQualityReport:
    """Отчет о качестве исторических данных."""

    def __init__(self):
        self.is_valid = True
        self.issues: List[str] = []
        self.quality_score = 100.0  # 0-100
        self.total_candles = 0
        self.gaps_count = 0
        self.outliers_count = 0
        self.invalid_ohlc_count = 0

    def add_issue(self, issue: str, severity: float = 1.0):
        """
        Добавить проблему качества.

        Args:
            issue: Описание проблемы
            severity: Серьезность (0-10), влияет на quality_score
        """
        self.issues.append(issue)
        self.quality_score -= severity
        if self.quality_score < 60:
            self.is_valid = False

    def __repr__(self):
        return (
            f"<DataQualityReport valid={self.is_valid} score={self.quality_score:.1f} "
            f"gaps={self.gaps_count} outliers={self.outliers_count}>"
        )


class HistoricalDataHandler:
    """
    Управление историческими данными для бэктестинга.

    Источники данных:
    1. Bybit API - основной источник (kline endpoint)
    2. Локальный кэш - для ускорения повторных запусков

    Best Practices:
    - Chunked loading для больших периодов
    - Валидация данных перед использованием
    - Кэширование часто используемых данных
    - Обработка rate limits API
    """

    def __init__(
        self,
        cache_dir: str = "data/backtest_cache",
        enable_cache: bool = True,
        chunk_size_days: int = 7,  # Загружать по 7 дней
        rate_limit_requests_per_second: int = 10
    ):
        """
        Инициализация data handler.

        Args:
            cache_dir: Директория для кэша данных
            enable_cache: Включить кэширование
            chunk_size_days: Размер чанка для загрузки (дни)
            rate_limit_requests_per_second: Лимит запросов к API
        """
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.chunk_size_days = chunk_size_days
        self.rate_limit = rate_limit_requests_per_second

        # Создать директорию кэша
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self._last_request_time = datetime.now()
        self._request_interval = 1.0 / rate_limit_requests_per_second

        logger.info(
            f"HistoricalDataHandler инициализирован: "
            f"cache={enable_cache}, chunk_size={chunk_size_days}d"
        )

    async def get_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1"
    ) -> List[Candle]:
        """
        Загрузить исторические свечи за период.

        Args:
            symbol: Торговая пара (BTCUSDT, ETHUSDT)
            start: Начальная дата
            end: Конечная дата
            interval: Интервал (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)

        Returns:
            Список свечей, отсортированных по времени
        """
        logger.info(
            f"📊 Загрузка свечей: {symbol} {interval}m "
            f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}"
        )

        # Проверка кэша
        cache_key = self._generate_cache_key(symbol, start, end, interval)
        cached_candles = self._load_from_cache(cache_key)

        if cached_candles is not None:
            logger.info(f"✅ Загружено из кэша: {len(cached_candles)} свечей")
            return cached_candles

        # Загрузка по чанкам
        all_candles = []
        current_start = start

        while current_start < end:
            current_end = min(
                current_start + timedelta(days=self.chunk_size_days),
                end
            )

            logger.debug(
                f"  Загрузка чанка: "
                f"{current_start.strftime('%Y-%m-%d')} → {current_end.strftime('%Y-%m-%d')}"
            )

            chunk_candles = await self._fetch_candles_chunk(
                symbol, current_start, current_end, interval
            )

            all_candles.extend(chunk_candles)
            current_start = current_end

            # Rate limiting
            await self._rate_limit_delay()

        # Сортировка по времени
        all_candles.sort(key=lambda c: c.timestamp)

        logger.info(f"✅ Загружено {len(all_candles)} свечей из Bybit API")

        # Сохранить в кэш
        if self.enable_cache:
            self._save_to_cache(cache_key, all_candles)

        return all_candles

    async def _fetch_candles_chunk(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> List[Candle]:
        """
        Загрузить чанк свечей через Bybit API.

        API endpoint: GET /v5/market/kline
        Limit: 1000 candles per request
        """
        try:
            # Конвертация временных меток
            start_ms = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)

            logger.debug(
                f"📡 Запрос к Bybit API: symbol={symbol}, interval={interval}, "
                f"start={start.isoformat()}, end={end.isoformat()}"
            )

            # Запрос к Bybit API
            response = await rest_client.get_kline(
                symbol=symbol,
                interval=interval,
                start=start_ms,
                end=end_ms,
                limit=1000
            )

            logger.debug(f"📥 Ответ API: {response.get('retCode') if response else 'None'}")

            if not response:
                logger.warning(f"Пустой ответ от API для {symbol}")
                return []

            # Проверка кода ответа
            if response.get('retCode') != 0:
                logger.error(
                    f"Ошибка API Bybit: code={response.get('retCode')}, "
                    f"msg={response.get('retMsg')}"
                )
                return []

            # Проверка наличия данных
            result = response.get('result', {})
            klines = result.get('list', [])

            if not klines:
                logger.warning(
                    f"API вернул пустой список свечей для {symbol} "
                    f"({start.isoformat()} - {end.isoformat()})"
                )
                return []

            # Парсинг свечей
            candles = []
            for kline in klines:
                # Bybit kline format: [startTime, open, high, low, close, volume, turnover]
                candle = Candle(
                    timestamp=int(kline[0]),
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5])
                    # Note: kline[6] is turnover/quote_volume, but Candle model doesn't have this field
                )
                candles.append(candle)

            logger.debug(f"✅ Получено {len(candles)} свечей")
            return candles

        except Exception as e:
            logger.error(f"Ошибка загрузки свечей {symbol}: {e}", exc_info=True)
            return []

    async def validate_data_quality(
        self,
        candles: List[Candle],
        interval_minutes: int = 1
    ) -> DataQualityReport:
        """
        Валидация качества исторических данных.

        Проверки:
        1. Gaps (пропуски во времени)
        2. Outliers (аномальные значения)
        3. Monotonic timestamps
        4. OHLC логика (high >= low, open/close в пределах [low, high])

        Args:
            candles: Список свечей
            interval_minutes: Интервал свечей (минуты)

        Returns:
            DataQualityReport с результатами валидации
        """
        report = DataQualityReport()
        report.total_candles = len(candles)

        if not candles:
            report.add_issue("Нет данных", severity=100)
            return report

        logger.info(f"🔍 Валидация качества данных: {len(candles)} свечей")

        # 1. Проверка gaps (пропусков)
        gaps = self._detect_gaps(candles, interval_minutes)
        report.gaps_count = len(gaps)
        if gaps:
            report.add_issue(
                f"Найдено {len(gaps)} пропусков в данных",
                severity=min(len(gaps) * 0.5, 20)
            )

        # 2. Проверка outliers
        outliers = self._detect_outliers(candles)
        report.outliers_count = len(outliers)
        if outliers:
            report.add_issue(
                f"Найдено {len(outliers)} аномальных значений",
                severity=min(len(outliers) * 0.1, 10)
            )

        # 3. Проверка монотонности timestamps
        if not self._is_monotonic(candles):
            report.add_issue("Не монотонные временные метки", severity=15)

        # 4. Проверка OHLC логики
        invalid_ohlc = self._validate_ohlc(candles)
        report.invalid_ohlc_count = len(invalid_ohlc)
        if invalid_ohlc:
            report.add_issue(
                f"{len(invalid_ohlc)} свечей с невалидным OHLC",
                severity=min(len(invalid_ohlc) * 0.2, 10)
            )

        logger.info(
            f"✅ Валидация завершена: quality_score={report.quality_score:.1f}, "
            f"gaps={report.gaps_count}, outliers={report.outliers_count}"
        )

        return report

    def _detect_gaps(
        self,
        candles: List[Candle],
        interval_minutes: int
    ) -> List[Tuple[datetime, datetime]]:
        """Поиск пропусков (gaps) в данных."""
        gaps = []
        expected_interval_ms = interval_minutes * 60 * 1000

        for i in range(1, len(candles)):
            prev_candle = candles[i - 1]
            current_candle = candles[i]

            time_diff = current_candle.timestamp - prev_candle.timestamp

            # Если разница больше ожидаемого интервала (с допуском 10%)
            if time_diff > expected_interval_ms * 1.1:
                gap_start = datetime.fromtimestamp(prev_candle.timestamp / 1000)
                gap_end = datetime.fromtimestamp(current_candle.timestamp / 1000)
                gaps.append((gap_start, gap_end))

        return gaps

    def _detect_outliers(self, candles: List[Candle]) -> List[int]:
        """Поиск аномальных значений."""
        if len(candles) < 50:
            return []

        outliers = []

        # Используем метод IQR (Interquartile Range)
        closes = [c.close for c in candles]

        # Вычисляем процентные изменения
        pct_changes = []
        for i in range(1, len(closes)):
            pct_change = abs((closes[i] - closes[i - 1]) / closes[i - 1]) * 100
            pct_changes.append(pct_change)

        if not pct_changes:
            return outliers

        # Сортировка для вычисления квартилей
        sorted_changes = sorted(pct_changes)
        q1_idx = len(sorted_changes) // 4
        q3_idx = 3 * len(sorted_changes) // 4

        q1 = sorted_changes[q1_idx]
        q3 = sorted_changes[q3_idx]
        iqr = q3 - q1

        # Outlier = за пределами [Q1 - 3*IQR, Q3 + 3*IQR]
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        for i, pct_change in enumerate(pct_changes):
            if pct_change < lower_bound or pct_change > upper_bound:
                outliers.append(i + 1)  # i+1 т.к. pct_changes на 1 короче

        return outliers

    def _is_monotonic(self, candles: List[Candle]) -> bool:
        """Проверка монотонности timestamps."""
        for i in range(1, len(candles)):
            if candles[i].timestamp <= candles[i - 1].timestamp:
                return False
        return True

    def _validate_ohlc(self, candles: List[Candle]) -> List[int]:
        """Валидация OHLC логики."""
        invalid = []

        for i, candle in enumerate(candles):
            # Проверка 1: high >= low
            if candle.high < candle.low:
                invalid.append(i)
                continue

            # Проверка 2: open в пределах [low, high]
            if not (candle.low <= candle.open <= candle.high):
                invalid.append(i)
                continue

            # Проверка 3: close в пределах [low, high]
            if not (candle.low <= candle.close <= candle.high):
                invalid.append(i)
                continue

        return invalid

    def _generate_cache_key(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> str:
        """Генерация ключа кэша."""
        key_str = f"{symbol}_{start.isoformat()}_{end.isoformat()}_{interval}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[List[Candle]]:
        """Загрузка из кэша."""
        if not self.enable_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'rb') as f:
                candles = pickle.load(f)
                logger.debug(f"Загружено из кэша: {len(candles)} свечей")
                return candles
        except Exception as e:
            logger.warning(f"Ошибка чтения кэша {cache_key}: {e}")
            return None

    def _save_to_cache(self, cache_key: str, candles: List[Candle]):
        """Сохранение в кэш."""
        if not self.enable_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(candles, f)
                logger.debug(f"Сохранено в кэш: {len(candles)} свечей")
        except Exception as e:
            logger.warning(f"Ошибка записи в кэш {cache_key}: {e}")

    async def _rate_limit_delay(self):
        """Задержка для соблюдения rate limit."""
        now = datetime.now()
        elapsed = (now - self._last_request_time).total_seconds()

        if elapsed < self._request_interval:
            delay = self._request_interval - elapsed
            await asyncio.sleep(delay)

        self._last_request_time = datetime.now()
