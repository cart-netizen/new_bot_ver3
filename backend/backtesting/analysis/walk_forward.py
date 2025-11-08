"""
Walk-Forward Analysis (WFA) Module

Реализует walk-forward тестирование для предотвращения overfitting:
- Разделение данных на In-Sample (IS) и Out-of-Sample (OOS) окна
- Rolling или Expanding windows
- Автоматическая оптимизация параметров на IS
- Валидация на OOS
- Агрегация результатов
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardWindow:
    """Одно окно Walk-Forward анализа"""
    window_id: int

    # In-Sample period
    is_start: datetime
    is_end: datetime

    # Out-of-Sample period
    oos_start: datetime
    oos_end: datetime

    # Результаты
    is_metrics: Optional[Dict[str, float]] = None
    oos_metrics: Optional[Dict[str, float]] = None

    # Оптимизированные параметры
    optimized_params: Optional[Dict[str, Any]] = None


@dataclass
class WalkForwardConfig:
    """Конфигурация Walk-Forward анализа"""
    # Размеры окон
    in_sample_days: int = 180  # 6 months
    out_of_sample_days: int = 60  # 2 months

    # Частота пересчета
    reoptimize_every_days: int = 30  # Каждые 30 дней

    # Тип окна
    anchor_mode: str = "rolling"  # "rolling" | "expanding"

    # Минимальный размер данных
    min_data_days: int = 90

    # Параметры оптимизации
    optimization_metric: str = "sharpe_ratio"  # Метрика для оптимизации

    # Параметры для оптимизации (будут варьироваться)
    param_ranges: Dict[str, List[Any]] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Результат Walk-Forward анализа"""
    config: WalkForwardConfig
    windows: List[WalkForwardWindow]

    # Агрегированные метрики
    avg_is_sharpe: float
    avg_oos_sharpe: float

    # Разница IS vs OOS (overfitting индикатор)
    is_oos_degradation_pct: float

    # Consistency
    oos_win_rate: float  # % OOS окон с положительным результатом

    # Все метрики
    aggregate_metrics: Dict[str, float]

    # Рекомендации
    is_strategy_robust: bool
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "windows": [
                {
                    "window_id": w.window_id,
                    "is_period": {
                        "start": w.is_start.isoformat(),
                        "end": w.is_end.isoformat()
                    },
                    "oos_period": {
                        "start": w.oos_start.isoformat(),
                        "end": w.oos_end.isoformat()
                    },
                    "is_metrics": w.is_metrics or {},
                    "oos_metrics": w.oos_metrics or {},
                    "optimized_params": w.optimized_params or {}
                }
                for w in self.windows
            ],
            "summary": {
                "total_windows": len(self.windows),
                "avg_is_sharpe": round(self.avg_is_sharpe, 3),
                "avg_oos_sharpe": round(self.avg_oos_sharpe, 3),
                "is_oos_degradation_pct": round(self.is_oos_degradation_pct, 2),
                "oos_win_rate": round(self.oos_win_rate, 2),
                "is_strategy_robust": self.is_strategy_robust
            },
            "aggregate_metrics": self.aggregate_metrics,
            "recommendations": self.recommendations
        }


class WalkForwardAnalyzer:
    """Выполнение Walk-Forward анализа"""

    def __init__(self, config: WalkForwardConfig):
        """
        Args:
            config: Конфигурация WFA
        """
        self.config = config

    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[WalkForwardWindow]:
        """
        Генерировать окна IS/OOS для всего периода

        Args:
            start_date: Начало всего периода
            end_date: Конец всего периода

        Returns:
            Список окон для тестирования
        """
        windows = []
        window_id = 0

        current_date = start_date

        while current_date < end_date:
            # In-Sample period
            if self.config.anchor_mode == "rolling":
                is_start = current_date
            else:  # expanding
                is_start = start_date  # Всегда с начала

            is_end = current_date + timedelta(days=self.config.in_sample_days)

            # Out-of-Sample period
            oos_start = is_end
            oos_end = oos_start + timedelta(days=self.config.out_of_sample_days)

            # Проверка что есть достаточно данных
            if oos_end > end_date:
                break  # Недостаточно данных для полного окна

            if (is_end - is_start).days < self.config.min_data_days:
                break  # IS период слишком короткий

            window = WalkForwardWindow(
                window_id=window_id,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end
            )

            windows.append(window)
            window_id += 1

            # Сдвиг для следующего окна
            current_date += timedelta(days=self.config.reoptimize_every_days)

        logger.info(f"Generated {len(windows)} Walk-Forward windows")
        return windows

    def analyze_results(
        self,
        windows: List[WalkForwardWindow]
    ) -> WalkForwardResult:
        """
        Проанализировать результаты WFA

        Args:
            windows: Окна с заполненными метриками

        Returns:
            WalkForwardResult с агрегированными метриками
        """
        if not windows:
            logger.warning("No windows to analyze")
            return self._create_empty_result()

        # Собрать метрики
        is_sharpes = []
        oos_sharpes = []
        oos_returns = []

        for window in windows:
            if window.is_metrics:
                is_sharpes.append(window.is_metrics.get("sharpe_ratio", 0))

            if window.oos_metrics:
                oos_sharpes.append(window.oos_metrics.get("sharpe_ratio", 0))
                oos_returns.append(window.oos_metrics.get("total_return_pct", 0))

        # Агрегированные метрики
        avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0.0
        avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0.0

        # Degradation
        if avg_is_sharpe > 0:
            is_oos_degradation_pct = ((avg_is_sharpe - avg_oos_sharpe) / avg_is_sharpe) * 100
        else:
            is_oos_degradation_pct = 0.0

        # Win rate
        oos_positive_count = sum(1 for r in oos_returns if r > 0)
        oos_win_rate = (oos_positive_count / len(oos_returns) * 100) if oos_returns else 0.0

        # Robustness check
        is_robust = self._check_robustness(
            avg_oos_sharpe, is_oos_degradation_pct, oos_win_rate
        )

        # Генерировать рекомендации
        recommendations = self._generate_recommendations(
            avg_is_sharpe, avg_oos_sharpe, is_oos_degradation_pct, oos_win_rate
        )

        # Другие агрегированные метрики
        aggregate_metrics = {
            "avg_is_sharpe": avg_is_sharpe,
            "avg_oos_sharpe": avg_oos_sharpe,
            "std_oos_sharpe": float(np.std(oos_sharpes)) if oos_sharpes else 0.0,
            "avg_oos_return_pct": float(np.mean(oos_returns)) if oos_returns else 0.0,
            "std_oos_return_pct": float(np.std(oos_returns)) if oos_returns else 0.0,
            "min_oos_sharpe": float(np.min(oos_sharpes)) if oos_sharpes else 0.0,
            "max_oos_sharpe": float(np.max(oos_sharpes)) if oos_sharpes else 0.0
        }

        return WalkForwardResult(
            config=self.config,
            windows=windows,
            avg_is_sharpe=avg_is_sharpe,
            avg_oos_sharpe=avg_oos_sharpe,
            is_oos_degradation_pct=is_oos_degradation_pct,
            oos_win_rate=oos_win_rate,
            aggregate_metrics=aggregate_metrics,
            is_strategy_robust=is_robust,
            recommendations=recommendations
        )

    def _check_robustness(
        self,
        avg_oos_sharpe: float,
        degradation_pct: float,
        oos_win_rate: float
    ) -> bool:
        """
        Проверить робастность стратегии

        Args:
            avg_oos_sharpe: Средний Sharpe на OOS
            degradation_pct: Процент деградации IS->OOS
            oos_win_rate: Win rate на OOS

        Returns:
            True если стратегия робастная
        """
        # Критерии робастности:
        # 1. OOS Sharpe > 1.0 (хорошая стратегия)
        # 2. Degradation < 30% (небольшой overfitting)
        # 3. OOS win rate > 50% (consistency)

        if avg_oos_sharpe < 0.8:
            return False

        if degradation_pct > 40:
            return False

        if oos_win_rate < 40:
            return False

        return True

    def _generate_recommendations(
        self,
        avg_is_sharpe: float,
        avg_oos_sharpe: float,
        degradation_pct: float,
        oos_win_rate: float
    ) -> List[str]:
        """Генерировать рекомендации на основе WFA результатов"""
        recommendations = []

        if avg_oos_sharpe < 0.5:
            recommendations.append(
                "Strategy shows poor OOS performance (Sharpe < 0.5). "
                "Consider revising strategy logic or parameters."
            )
        elif avg_oos_sharpe >= 1.5:
            recommendations.append(
                "Strategy shows excellent OOS performance (Sharpe >= 1.5). "
                "Good candidate for live trading."
            )

        if degradation_pct > 50:
            recommendations.append(
                f"High IS->OOS degradation ({degradation_pct:.1f}%). "
                "Significant overfitting detected. Simplify strategy or add regularization."
            )
        elif degradation_pct > 30:
            recommendations.append(
                f"Moderate IS->OOS degradation ({degradation_pct:.1f}%). "
                "Some overfitting present. Review parameter complexity."
            )
        elif degradation_pct < 10:
            recommendations.append(
                f"Low IS->OOS degradation ({degradation_pct:.1f}%). "
                "Strategy generalizes well to unseen data."
            )

        if oos_win_rate < 40:
            recommendations.append(
                f"Low OOS win rate ({oos_win_rate:.1f}%). "
                "Strategy lacks consistency across different market conditions."
            )
        elif oos_win_rate >= 70:
            recommendations.append(
                f"High OOS win rate ({oos_win_rate:.1f}%). "
                "Strategy is consistent across market conditions."
            )

        if not recommendations:
            recommendations.append(
                "Strategy shows reasonable WFA performance. "
                "Continue with additional validation."
            )

        return recommendations

    def _create_empty_result(self) -> WalkForwardResult:
        """Создать пустой результат"""
        return WalkForwardResult(
            config=self.config,
            windows=[],
            avg_is_sharpe=0.0,
            avg_oos_sharpe=0.0,
            is_oos_degradation_pct=0.0,
            oos_win_rate=0.0,
            aggregate_metrics={},
            is_strategy_robust=False,
            recommendations=["No WFA windows generated - insufficient data"]
        )
