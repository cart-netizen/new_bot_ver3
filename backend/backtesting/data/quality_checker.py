"""
Data Quality Checker for Historical Data

Проверяет качество исторических данных перед бэктестом:
- Пропуски в данных (gaps)
- Аномальные значения (spikes)
- Нулевой объем
- Консистентность OHLC
- Дубликаты
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from backend.core.logger import get_logger
from backend.models.candle import Candle

logger = get_logger(__name__)


@dataclass
class QualityIssue:
    """Проблема качества данных"""
    severity: str  # "ERROR", "WARNING", "INFO"
    category: str  # "gaps", "spikes", "volume", "ohlc", "duplicates"
    description: str
    timestamp: Optional[datetime] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "details": self.details or {}
        }


@dataclass
class DataQualityReport:
    """Отчет о качестве данных"""
    total_candles: int
    time_range: Tuple[datetime, datetime]
    interval_minutes: int

    # Счетчики проблем
    errors_count: int
    warnings_count: int
    info_count: int

    # Детальные проблемы
    issues: List[QualityIssue]

    # Статистика
    gaps_count: int
    duplicates_count: int
    zero_volume_count: int
    ohlc_inconsistencies_count: int
    price_spikes_count: int

    # Рекомендации
    is_suitable_for_backtest: bool
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "total_candles": self.total_candles,
            "time_range": {
                "start": self.time_range[0].isoformat(),
                "end": self.time_range[1].isoformat()
            },
            "interval_minutes": self.interval_minutes,
            "summary": {
                "errors": self.errors_count,
                "warnings": self.warnings_count,
                "info": self.info_count
            },
            "statistics": {
                "gaps": self.gaps_count,
                "duplicates": self.duplicates_count,
                "zero_volume": self.zero_volume_count,
                "ohlc_inconsistencies": self.ohlc_inconsistencies_count,
                "price_spikes": self.price_spikes_count
            },
            "is_suitable_for_backtest": self.is_suitable_for_backtest,
            "recommendations": self.recommendations,
            "issues": [issue.to_dict() for issue in self.issues]
        }


class DataQualityChecker:
    """Проверка качества исторических данных"""

    def __init__(
        self,
        spike_threshold_std: float = 5.0,
        max_gap_tolerance_pct: float = 5.0,
        strict_mode: bool = False
    ):
        """
        Args:
            spike_threshold_std: Порог для определения аномальных спайков (кол-во std)
            max_gap_tolerance_pct: Максимальный допустимый процент пропусков
            strict_mode: Строгий режим (любая ошибка делает данные непригодными)
        """
        self.spike_threshold_std = spike_threshold_std
        self.max_gap_tolerance_pct = max_gap_tolerance_pct
        self.strict_mode = strict_mode

    def check_data_quality(
        self,
        candles: List[Candle],
        expected_interval_minutes: int
    ) -> DataQualityReport:
        """
        Проверить качество данных

        Args:
            candles: Список свечей для проверки
            expected_interval_minutes: Ожидаемый интервал между свечами (минуты)

        Returns:
            DataQualityReport с результатами проверки
        """
        if not candles:
            return DataQualityReport(
                total_candles=0,
                time_range=(datetime.now(), datetime.now()),
                interval_minutes=expected_interval_minutes,
                errors_count=1,
                warnings_count=0,
                info_count=0,
                issues=[QualityIssue(
                    severity="ERROR",
                    category="general",
                    description="No candles provided for quality check"
                )],
                gaps_count=0,
                duplicates_count=0,
                zero_volume_count=0,
                ohlc_inconsistencies_count=0,
                price_spikes_count=0,
                is_suitable_for_backtest=False,
                recommendations=["Obtain historical data before running backtest"]
            )

        # Сортировать свечи по времени
        sorted_candles = sorted(candles, key=lambda c: c.timestamp)

        issues = []

        # 1. Проверка дубликатов
        duplicates_issues = self._check_duplicates(sorted_candles)
        issues.extend(duplicates_issues)

        # 2. Проверка gaps
        gaps_issues = self._check_gaps(sorted_candles, expected_interval_minutes)
        issues.extend(gaps_issues)

        # 3. Проверка нулевого объема
        volume_issues = self._check_zero_volume(sorted_candles)
        issues.extend(volume_issues)

        # 4. Проверка OHLC консистентности
        ohlc_issues = self._check_ohlc_consistency(sorted_candles)
        issues.extend(ohlc_issues)

        # 5. Проверка аномальных спайков
        spike_issues = self._check_price_spikes(sorted_candles)
        issues.extend(spike_issues)

        # Подсчет проблем
        errors_count = sum(1 for i in issues if i.severity == "ERROR")
        warnings_count = sum(1 for i in issues if i.severity == "WARNING")
        info_count = sum(1 for i in issues if i.severity == "INFO")

        gaps_count = sum(1 for i in issues if i.category == "gaps")
        duplicates_count = sum(1 for i in issues if i.category == "duplicates")
        zero_volume_count = sum(1 for i in issues if i.category == "volume")
        ohlc_count = sum(1 for i in issues if i.category == "ohlc")
        spikes_count = sum(1 for i in issues if i.category == "spikes")

        # Определить пригодность для бэктеста
        is_suitable = self._determine_suitability(
            errors_count, warnings_count, gaps_count, duplicates_count
        )

        # Генерировать рекомендации
        recommendations = self._generate_recommendations(
            errors_count, warnings_count, gaps_count, zero_volume_count,
            ohlc_count, spikes_count
        )

        return DataQualityReport(
            total_candles=len(sorted_candles),
            time_range=(sorted_candles[0].timestamp, sorted_candles[-1].timestamp),
            interval_minutes=expected_interval_minutes,
            errors_count=errors_count,
            warnings_count=warnings_count,
            info_count=info_count,
            issues=issues,
            gaps_count=gaps_count,
            duplicates_count=duplicates_count,
            zero_volume_count=zero_volume_count,
            ohlc_inconsistencies_count=ohlc_count,
            price_spikes_count=spikes_count,
            is_suitable_for_backtest=is_suitable,
            recommendations=recommendations
        )

    def _check_duplicates(self, candles: List[Candle]) -> List[QualityIssue]:
        """Проверка на дубликаты по timestamp"""
        issues = []
        seen_timestamps = set()

        for candle in candles:
            if candle.timestamp in seen_timestamps:
                issues.append(QualityIssue(
                    severity="ERROR",
                    category="duplicates",
                    description="Duplicate candle detected",
                    timestamp=candle.timestamp,
                    details={"price": candle.close}
                ))
            seen_timestamps.add(candle.timestamp)

        return issues

    def _check_gaps(
        self,
        candles: List[Candle],
        expected_interval_minutes: int
    ) -> List[QualityIssue]:
        """Проверка на пропуски в данных"""
        issues = []
        expected_delta = timedelta(minutes=expected_interval_minutes)

        for i in range(1, len(candles)):
            prev_candle = candles[i - 1]
            curr_candle = candles[i]

            actual_delta = curr_candle.timestamp - prev_candle.timestamp

            # Разница больше ожидаемой (с небольшим допуском)
            if actual_delta > expected_delta * 1.5:
                missing_candles = int(actual_delta.total_seconds() / 60 / expected_interval_minutes) - 1

                severity = "ERROR" if missing_candles > 5 else "WARNING"

                issues.append(QualityIssue(
                    severity=severity,
                    category="gaps",
                    description=f"Gap detected: {missing_candles} candles missing",
                    timestamp=prev_candle.timestamp,
                    details={
                        "gap_start": prev_candle.timestamp.isoformat(),
                        "gap_end": curr_candle.timestamp.isoformat(),
                        "missing_candles": missing_candles,
                        "gap_duration_minutes": int(actual_delta.total_seconds() / 60)
                    }
                ))

        return issues

    def _check_zero_volume(self, candles: List[Candle]) -> List[QualityIssue]:
        """Проверка на нулевой объем"""
        issues = []

        for candle in candles:
            if candle.volume == 0:
                issues.append(QualityIssue(
                    severity="WARNING",
                    category="volume",
                    description="Zero volume detected",
                    timestamp=candle.timestamp,
                    details={"price": candle.close}
                ))

        return issues

    def _check_ohlc_consistency(self, candles: List[Candle]) -> List[QualityIssue]:
        """Проверка консистентности OHLC (High >= Low, etc.)"""
        issues = []

        for candle in candles:
            # High должен быть >= max(Open, Close)
            if candle.high < max(candle.open, candle.close):
                issues.append(QualityIssue(
                    severity="ERROR",
                    category="ohlc",
                    description="High is less than Open/Close",
                    timestamp=candle.timestamp,
                    details={
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close
                    }
                ))

            # Low должен быть <= min(Open, Close)
            if candle.low > min(candle.open, candle.close):
                issues.append(QualityIssue(
                    severity="ERROR",
                    category="ohlc",
                    description="Low is greater than Open/Close",
                    timestamp=candle.timestamp,
                    details={
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close
                    }
                ))

            # High должен быть >= Low
            if candle.high < candle.low:
                issues.append(QualityIssue(
                    severity="ERROR",
                    category="ohlc",
                    description="High is less than Low",
                    timestamp=candle.timestamp,
                    details={
                        "high": candle.high,
                        "low": candle.low
                    }
                ))

            # Отрицательные цены
            if any(p <= 0 for p in [candle.open, candle.high, candle.low, candle.close]):
                issues.append(QualityIssue(
                    severity="ERROR",
                    category="ohlc",
                    description="Negative or zero price detected",
                    timestamp=candle.timestamp,
                    details={
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close
                    }
                ))

        return issues

    def _check_price_spikes(self, candles: List[Candle]) -> List[QualityIssue]:
        """Проверка на аномальные скачки цен"""
        if len(candles) < 30:
            # Недостаточно данных для проверки
            return []

        issues = []

        # Вычислить price changes
        price_changes = []
        for i in range(1, len(candles)):
            prev_close = candles[i - 1].close
            curr_close = candles[i].close
            if prev_close > 0:
                pct_change = abs((curr_close - prev_close) / prev_close) * 100
                price_changes.append((i, pct_change))

        if not price_changes:
            return []

        # Вычислить среднее и std
        changes_values = [pc[1] for pc in price_changes]
        mean_change = np.mean(changes_values)
        std_change = np.std(changes_values)

        if std_change == 0:
            return []

        # Найти аномалии
        threshold = mean_change + (self.spike_threshold_std * std_change)

        for idx, pct_change in price_changes:
            if pct_change > threshold:
                candle = candles[idx]
                prev_candle = candles[idx - 1]

                issues.append(QualityIssue(
                    severity="WARNING",
                    category="spikes",
                    description=f"Abnormal price spike detected: {pct_change:.2f}%",
                    timestamp=candle.timestamp,
                    details={
                        "prev_price": prev_candle.close,
                        "curr_price": candle.close,
                        "percent_change": round(pct_change, 2),
                        "threshold": round(threshold, 2)
                    }
                ))

        return issues

    def _determine_suitability(
        self,
        errors_count: int,
        warnings_count: int,
        gaps_count: int,
        duplicates_count: int
    ) -> bool:
        """Определить пригодность данных для бэктеста"""

        if self.strict_mode:
            # Строгий режим: любая ошибка = непригодно
            return errors_count == 0

        # Нормальный режим: критичные проблемы
        if errors_count > 0:
            # Дубликаты и OHLC inconsistencies всегда критичны
            return False

        # Warnings допустимы в разумных пределах
        if warnings_count > 100:  # Более 100 warnings
            return False

        return True

    def _generate_recommendations(
        self,
        errors_count: int,
        warnings_count: int,
        gaps_count: int,
        zero_volume_count: int,
        ohlc_count: int,
        spikes_count: int
    ) -> List[str]:
        """Генерировать рекомендации на основе найденных проблем"""
        recommendations = []

        if errors_count == 0 and warnings_count == 0:
            recommendations.append("Data quality is excellent - ready for backtesting")
            return recommendations

        if gaps_count > 0:
            recommendations.append(
                f"Found {gaps_count} gaps in data. Consider using forward-fill "
                "or interpolation to fill missing values."
            )

        if zero_volume_count > 10:
            recommendations.append(
                f"Found {zero_volume_count} candles with zero volume. "
                "This may indicate illiquid periods or data collection issues."
            )

        if ohlc_count > 0:
            recommendations.append(
                f"Found {ohlc_count} OHLC inconsistencies. "
                "Data cleaning is required before backtesting."
            )

        if spikes_count > 5:
            recommendations.append(
                f"Found {spikes_count} abnormal price spikes. "
                "Review these candles manually to verify data accuracy."
            )

        if errors_count > 0:
            recommendations.append(
                "Critical errors detected - data cleaning is mandatory before backtesting."
            )
        elif warnings_count > 50:
            recommendations.append(
                "High number of warnings - consider data preprocessing before backtesting."
            )

        return recommendations
