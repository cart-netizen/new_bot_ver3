"""
Strategy Performance Tracker - непрерывный мониторинг эффективности стратегий.

Функциональность:
- Отслеживание метрик каждой стратегии в реальном времени
- Win Rate, Sharpe Ratio, Profit Factor, Confidence Calibration
- Temporal windows (24h, 7d, 30d) с exponential decay
- Персистентное хранение в JSONL
- Ranking стратегий по performance score
- Детекция деградации производительности

Путь: backend/strategies/adaptive/strategy_performance_tracker.py
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

from backend.core.logger import get_logger
from backend.models.signal import TradingSignal, SignalType

logger = get_logger(__name__)


@dataclass
class SignalOutcome:
    """Результат исполнения сигнала."""
    timestamp: int
    strategy: str
    symbol: str
    signal_type: str  # "BUY" или "SELL"
    confidence: float
    
    # Параметры входа
    entry_price: float
    entry_timestamp: int
    
    # Параметры выхода
    exit_price: Optional[float]
    exit_timestamp: Optional[int]
    
    # Результаты
    return_pct: Optional[float]
    return_usdt: Optional[float]
    hold_duration_seconds: Optional[int]
    outcome: str  # "profit", "loss", "breakeven", "open"
    
    # Контекст
    market_regime: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Конвертация в словарь."""
        return asdict(self)  # type: ignore


@dataclass
class StrategyMetrics:
    """Метрики производительности стратегии."""
    strategy_name: str
    symbol: str
    time_window: str  # "24h", "7d", "30d"
    
    # Accuracy metrics
    total_signals: int
    closed_signals: int
    win_count: int
    loss_count: int
    breakeven_count: int
    win_rate: float
    
    # Financial metrics
    total_return_pct: float
    avg_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    
    sharpe_ratio: float
    profit_factor: float  # gross_profit / gross_loss
    max_drawdown_pct: float
    
    # Confidence calibration
    confidence_calibration_score: float  # Насколько predicted confidence соответствует actual
    avg_confidence: float
    
    # Timing metrics
    avg_hold_duration_seconds: float
    avg_time_to_profit_seconds: Optional[float]
    avg_time_to_loss_seconds: Optional[float]
    
    # Consistency
    consistency_score: float  # Низкая variance returns = высокая consistency
    signal_frequency: float  # Signals per day
    
    # Composite score
    performance_score: float  # 0-1, композитная метрика


@dataclass
class PerformanceTrackerConfig:
    """Конфигурация Performance Tracker."""
    # Storage
    data_dir: str = "data/strategy_performance"
    enable_persistence: bool = True
    
    # Temporal windows
    short_term_hours: int = 24
    medium_term_days: int = 7
    long_term_days: int = 30
    
    # Decay
    decay_factor: float = 0.95  # Exponential weight для старых сигналов
    
    # Minimum data requirements
    min_signals_for_metrics: int = 20
    min_closed_signals_for_metrics: int = 10
    
    # Performance score weights
    win_rate_weight: float = 0.30
    sharpe_weight: float = 0.30
    profit_factor_weight: float = 0.20
    calibration_weight: float = 0.20

    max_file_size_mb = 10  # Максимальный размер файла
    enable_compression = True  # Сжатие старых файлов

    # Degradation detection
    degradation_threshold: float = 0.3  # Performance score < 0.3 = деградация


class StrategyPerformanceTracker:
    """
    Трекер производительности стратегий.
    
    Отслеживает каждый сигнал от генерации до закрытия позиции,
    вычисляет метрики и выявляет деградацию.
    """

    def __init__(self, config: PerformanceTrackerConfig):
        """
        Инициализация трекера.

        Args:
            config: Конфигурация
        """
        self.config = config
        
        # История outcomes для каждой стратегии
        # strategy_name -> symbol -> List[SignalOutcome]
        self.outcomes: Dict[str, Dict[str, List[SignalOutcome]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Открытые позиции (не закрыты)
        # signal_id -> SignalOutcome
        self.open_positions: Dict[str, SignalOutcome] = {}
        
        # Кэш метрик (для производительности)
        # (strategy, symbol, window) -> StrategyMetrics
        self.metrics_cache: Dict[Tuple[str, str, str], StrategyMetrics] = {}
        self.cache_timestamp: Dict[Tuple[str, str, str], int] = {}
        self.cache_ttl_seconds = 300  # 5 минут
        
        # Создаем директорию для хранения
        if config.enable_persistence:
            Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        
        # Загружаем существующие данные
        self._load_historical_data()
        
        # Статистика
        self.total_signals_tracked = 0
        self.total_outcomes_recorded = 0

        # FIX: Counter for periodic cleanup to prevent memory leak
        self._cleanup_counter = 0
        self._cleanup_interval = 100  # Cleanup every 100 outcomes
        
        logger.info(
            f"Инициализирован StrategyPerformanceTracker: "
            f"persistence={config.enable_persistence}, "
            f"min_signals={config.min_signals_for_metrics}"
        )

    def record_signal_generated(
        self,
        strategy: str,
        symbol: str,
        signal: TradingSignal,
        signal_id: str
    ):
        """
        Записать сгенерированный сигнал (открытие позиции).

        Args:
            strategy: Имя стратегии
            symbol: Торговая пара
            signal: Сигнал
            signal_id: Уникальный ID сигнала
        """
        outcome = SignalOutcome(
            timestamp=signal.timestamp,
            strategy=strategy,
            symbol=symbol,
            signal_type=signal.signal_type.value,
            confidence=signal.confidence,
            entry_price=signal.price,
            entry_timestamp=signal.timestamp,
            exit_price=None,
            exit_timestamp=None,
            return_pct=None,
            return_usdt=None,
            hold_duration_seconds=None,
            outcome="open",
            market_regime=signal.metadata.get('market_regime') if signal.metadata else None
        )
        
        # Сохраняем в открытые позиции
        self.open_positions[signal_id] = outcome
        self.total_signals_tracked += 1
        
        logger.debug(
            f"[{strategy}] Записан сигнал: {symbol} {signal.signal_type.value}, "
            f"confidence={signal.confidence:.2f}, signal_id={signal_id}"
        )

    def record_signal_outcome(
        self,
        signal_id: str,
        exit_price: float,
        exit_timestamp: int,
        pnl_usdt: Optional[float] = None
    ):
        """
        Записать результат исполнения сигнала (закрытие позиции).

        Args:
            signal_id: Уникальный ID сигнала
            exit_price: Цена выхода
            exit_timestamp: Время выхода
            pnl_usdt: P&L в USDT (опционально)
        """
        if signal_id not in self.open_positions:
            logger.warning(f"Signal {signal_id} не найден в open_positions")
            return
        
        outcome = self.open_positions[signal_id]
        
        # Вычисляем результаты
        if outcome.signal_type == "BUY":
            return_pct = ((exit_price - outcome.entry_price) / outcome.entry_price) * 100
        else:  # SELL
            return_pct = ((outcome.entry_price - exit_price) / outcome.entry_price) * 100
        
        hold_duration = (exit_timestamp - outcome.entry_timestamp) / 1000  # в секундах
        
        # Определяем outcome
        if return_pct > 0.1:
            outcome_type = "profit"
        elif return_pct < -0.1:
            outcome_type = "loss"
        else:
            outcome_type = "breakeven"
        
        # Обновляем outcome
        outcome.exit_price = exit_price
        outcome.exit_timestamp = exit_timestamp
        outcome.return_pct = return_pct
        outcome.return_usdt = pnl_usdt
        outcome.hold_duration_seconds = int(hold_duration)
        outcome.outcome = outcome_type
        
        # Переносим в историю
        self.outcomes[outcome.strategy][outcome.symbol].append(outcome)
        
        # Удаляем из открытых
        del self.open_positions[signal_id]
        
        self.total_outcomes_recorded += 1

        # FIX: Periodic cleanup to prevent memory leak
        self._cleanup_counter += 1
        if self._cleanup_counter >= self._cleanup_interval:
            self._cleanup_old_outcomes()
            self._cleanup_counter = 0

        # Инвалидируем кэш метрик
        self._invalidate_cache(outcome.strategy, outcome.symbol)
        
        # Сохраняем в файл
        if self.config.enable_persistence:
            self._persist_outcome(outcome)
        
        logger.info(
            f"[{outcome.strategy}] Результат: {outcome.symbol} {outcome.signal_type}, "
            f"return={return_pct:+.2f}%, hold={hold_duration:.0f}s, outcome={outcome_type}"
        )

    def get_strategy_metrics(
        self,
        strategy: str,
        symbol: str,
        time_window: str = "7d"
    ) -> Optional[StrategyMetrics]:
        """
        Получить метрики стратегии для символа.

        Args:
            strategy: Имя стратегии
            symbol: Торговая пара
            time_window: "24h", "7d", "30d"

        Returns:
            StrategyMetrics или None если недостаточно данных
        """
        # Проверяем кэш
        cache_key = (strategy, symbol, time_window)
        
        if cache_key in self.metrics_cache:
            cached_time = self.cache_timestamp.get(cache_key, 0)
            current_time = int(datetime.now().timestamp())
            
            if (current_time - cached_time) < self.cache_ttl_seconds:
                return self.metrics_cache[cache_key]
        
        # Вычисляем метрики
        metrics = self._calculate_metrics(strategy, symbol, time_window)
        
        if metrics:
            # Обновляем кэш
            self.metrics_cache[cache_key] = metrics
            self.cache_timestamp[cache_key] = int(datetime.now().timestamp())
        
        return metrics

    def get_all_strategies_ranking(
        self,
        symbol: Optional[str] = None,
        time_window: str = "7d"
    ) -> List[Tuple[str, float]]:
        """
        Получить рейтинг всех стратегий по performance score.

        Args:
            symbol: Торговая пара (если None - агрегируем по всем)
            time_window: Временное окно

        Returns:
            Список кортежей (strategy_name, performance_score), отсортированный по убыванию
        """
        rankings = []
        
        for strategy_name in self.outcomes.keys():
            if symbol:
                metrics = self.get_strategy_metrics(strategy_name, symbol, time_window)
                if metrics:
                    rankings.append((strategy_name, metrics.performance_score))
            else:
                # Агрегируем по всем символам
                all_scores = []
                for sym in self.outcomes[strategy_name].keys():
                    metrics = self.get_strategy_metrics(strategy_name, sym, time_window)
                    if metrics:
                        all_scores.append(metrics.performance_score)
                
                if all_scores:
                    avg_score = np.mean(all_scores)
                    rankings.append((strategy_name, avg_score))
        
        # Сортируем по убыванию
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings

    def detect_degradation(
        self,
        strategy: str,
        symbol: str,
        time_window: str = "7d"
    ) -> Tuple[bool, Optional[str]]:
        """
        Детектировать деградацию производительности стратегии.

        Args:
            strategy: Имя стратегии
            symbol: Торговая пара
            time_window: Временное окно

        Returns:
            (is_degraded, reason)
        """
        metrics = self.get_strategy_metrics(strategy, symbol, time_window)
        
        if not metrics:
            return False, None
        
        # Проверка 1: Performance score ниже порога
        if metrics.performance_score < self.config.degradation_threshold:
            return True, f"Performance score критически низкий: {metrics.performance_score:.2f}"
        
        # Проверка 2: Win rate критически низкий
        if metrics.closed_signals >= 20 and metrics.win_rate < 0.35:
            return True, f"Win rate критически низкий: {metrics.win_rate:.2%}"
        
        # Проверка 3: Profit factor < 1 (больше потерь чем прибыли)
        if metrics.closed_signals >= 20 and metrics.profit_factor < 1.0:
            return True, f"Profit factor < 1.0: {metrics.profit_factor:.2f}"
        
        # Проверка 4: Sharpe ratio отрицательный
        if metrics.sharpe_ratio < -0.5:
            return True, f"Sharpe ratio отрицательный: {metrics.sharpe_ratio:.2f}"
        
        return False, None

    def _calculate_metrics(
        self,
        strategy: str,
        symbol: str,
        time_window: str
    ) -> Optional[StrategyMetrics]:
        """
        Вычислить метрики для стратегии.
        """
        if strategy not in self.outcomes or symbol not in self.outcomes[strategy]:
            return None
        
        all_outcomes = self.outcomes[strategy][symbol]
        
        if not all_outcomes:
            return None
        
        # Фильтруем по временному окну
        cutoff_time = self._get_cutoff_timestamp(time_window)
        filtered_outcomes = [o for o in all_outcomes if o.timestamp >= cutoff_time]
        
        if len(filtered_outcomes) < self.config.min_signals_for_metrics:
            return None
        
        # Разделяем на закрытые и открытые
        closed_outcomes = [o for o in filtered_outcomes if o.outcome != "open"]
        
        if len(closed_outcomes) < self.config.min_closed_signals_for_metrics:
            return None
        
        # ========== ACCURACY METRICS ==========
        total_signals = len(filtered_outcomes)
        closed_signals = len(closed_outcomes)
        
        win_count = len([o for o in closed_outcomes if o.outcome == "profit"])
        loss_count = len([o for o in closed_outcomes if o.outcome == "loss"])
        breakeven_count = len([o for o in closed_outcomes if o.outcome == "breakeven"])
        
        win_rate = win_count / closed_signals if closed_signals > 0 else 0.0
        
        # ========== FINANCIAL METRICS ==========
        returns = [o.return_pct for o in closed_outcomes if o.return_pct is not None]
        
        if not returns:
            return None
        
        total_return_pct = sum(returns)
        avg_return_pct = np.mean(returns)
        
        winning_returns = [r for r in returns if r > 0]
        losing_returns = [r for r in returns if r < 0]
        
        avg_win_pct = np.mean(winning_returns) if winning_returns else 0.0
        avg_loss_pct = np.mean(losing_returns) if losing_returns else 0.0
        
        # Sharpe Ratio
        if len(returns) > 1:
            sharpe_ratio = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Profit Factor
        gross_profit = sum(winning_returns) if winning_returns else 0.0
        gross_loss = abs(sum(losing_returns)) if losing_returns else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)
        
        # Max Drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown_pct = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # ========== CONFIDENCE CALIBRATION ==========
        # Проверяем насколько predicted confidence соответствует actual performance
        confidence_calibration_score = self._calculate_confidence_calibration(closed_outcomes)
        avg_confidence = np.mean([o.confidence for o in filtered_outcomes])
        
        # ========== TIMING METRICS ==========
        hold_durations = [o.hold_duration_seconds for o in closed_outcomes 
                         if o.hold_duration_seconds is not None]
        avg_hold_duration = np.mean(hold_durations) if hold_durations else 0.0
        
        # Time to profit/loss (для closed позиций)
        profit_durations = [o.hold_duration_seconds for o in closed_outcomes 
                           if o.outcome == "profit" and o.hold_duration_seconds is not None]
        loss_durations = [o.hold_duration_seconds for o in closed_outcomes 
                         if o.outcome == "loss" and o.hold_duration_seconds is not None]
        
        avg_time_to_profit = np.mean(profit_durations) if profit_durations else None
        avg_time_to_loss = np.mean(loss_durations) if loss_durations else None
        
        # ========== CONSISTENCY ==========
        # Низкая variance = высокая consistency
        returns_variance = np.var(returns) if len(returns) > 1 else 0.0
        consistency_score = 1.0 / (1.0 + returns_variance)  # Чем ниже variance, тем выше score
        
        # Signal frequency (signals per day)
        time_span_days = self._get_time_span_days(time_window)
        signal_frequency = total_signals / time_span_days if time_span_days > 0 else 0.0
        
        # ========== COMPOSITE PERFORMANCE SCORE ==========
        performance_score = self._calculate_performance_score(
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            confidence_calibration=confidence_calibration_score
        )
        
        # ========== СОЗДАНИЕ МЕТРИК ==========
        metrics = StrategyMetrics(
            strategy_name=strategy,
            symbol=symbol,
            time_window=time_window,
            total_signals=total_signals,
            closed_signals=closed_signals,
            win_count=win_count,
            loss_count=loss_count,
            breakeven_count=breakeven_count,
            win_rate=win_rate,
            total_return_pct=total_return_pct,
            avg_return_pct=float(avg_return_pct),
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown_pct,
            confidence_calibration_score=confidence_calibration_score,
            avg_confidence=float(avg_confidence),
            avg_hold_duration_seconds=avg_hold_duration,
            avg_time_to_profit_seconds=avg_time_to_profit,
            avg_time_to_loss_seconds=avg_time_to_loss,
            consistency_score=consistency_score,
            signal_frequency=signal_frequency,
            performance_score=performance_score
        )
        
        return metrics

    def _calculate_confidence_calibration(self, outcomes: List[SignalOutcome]) -> float:
        """
        Вычислить calibration score для confidence.
        
        Хорошая калибровка: сигналы с confidence 0.8 прибыльны в ~80% случаев.
        """
        if not outcomes:
            return 0.5
        
        # Группируем по confidence bins
        bins = np.linspace(0, 1, 11)  # 10 бинов: [0-0.1], [0.1-0.2], ..., [0.9-1.0]
        
        calibration_errors = []
        
        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]
            
            # Фильтруем outcomes в этом bin
            bin_outcomes = [
                o for o in outcomes
                if bin_start <= o.confidence < bin_end
            ]
            
            if len(bin_outcomes) < 5:  # Минимум 5 сигналов для статистики
                continue
            
            # Predicted confidence (средняя в bin)
            predicted_confidence = np.mean([o.confidence for o in bin_outcomes])
            
            # Actual accuracy (процент прибыльных)
            profitable_count = len([o for o in bin_outcomes if o.outcome == "profit"])
            actual_accuracy = profitable_count / len(bin_outcomes)
            
            # Calibration error
            error = abs(predicted_confidence - actual_accuracy)
            calibration_errors.append(error)
        
        if not calibration_errors:
            return 0.5
        
        # Calibration score: 1.0 - средняя ошибка
        avg_error = np.mean(calibration_errors)
        calibration_score = max(0.0, 1.0 - avg_error)
        
        return calibration_score

    def _calculate_performance_score(
        self,
        win_rate: float,
        sharpe_ratio: float,
        profit_factor: float,
        confidence_calibration: float
    ) -> float:
        """
        Вычислить композитный performance score (0-1).
        """
        # Нормализуем компоненты
        # Win rate: уже 0-1
        win_rate_normalized = win_rate
        
        # Sharpe ratio: нормализуем в диапазон 0-1
        # Sharpe > 2.0 считается отличным
        sharpe_normalized = min(max(sharpe_ratio, -1.0), 2.0) / 2.0
        sharpe_normalized = (sharpe_normalized + 0.5)  # Shift в 0-1
        
        # Profit factor: нормализуем
        # Profit factor > 2.0 = отлично, < 1.0 = плохо
        profit_factor_normalized = min(profit_factor, 2.0) / 2.0
        
        # Confidence calibration: уже 0-1
        calibration_normalized = confidence_calibration
        
        # Взвешенная сумма
        performance_score = (
            win_rate_normalized * self.config.win_rate_weight +
            sharpe_normalized * self.config.sharpe_weight +
            profit_factor_normalized * self.config.profit_factor_weight +
            calibration_normalized * self.config.calibration_weight
        )
        
        return min(max(performance_score, 0.0), 1.0)

    def _cleanup_old_outcomes(self):
        """
        Clean up outcomes older than 60 days to prevent memory leak.
        FIX: Memory leak prevention - remove stale data.
        """
        # Keep data for 60 days (2x long_term window for safety)
        cutoff_days = self.config.long_term_days * 2
        cutoff_time = datetime.now() - timedelta(days=cutoff_days)
        cutoff_timestamp = int(cutoff_time.timestamp() * 1000)

        total_removed = 0

        # Clean outcomes for each strategy/symbol
        for strategy in list(self.outcomes.keys()):
            for symbol in list(self.outcomes[strategy].keys()):
                original_count = len(self.outcomes[strategy][symbol])

                # Keep only recent outcomes
                self.outcomes[strategy][symbol] = [
                    outcome for outcome in self.outcomes[strategy][symbol]
                    if outcome.entry_timestamp >= cutoff_timestamp
                ]

                removed = original_count - len(self.outcomes[strategy][symbol])
                total_removed += removed

                # Remove empty entries
                if not self.outcomes[strategy][symbol]:
                    del self.outcomes[strategy][symbol]

            # Remove empty strategy entries
            if not self.outcomes[strategy]:
                del self.outcomes[strategy]

        if total_removed > 0:
            logger.info(
                f"🧹 Очистка старых outcomes: удалено {total_removed} записей "
                f"(старше {cutoff_days} дней)"
            )

    def _get_cutoff_timestamp(self, time_window: str) -> int:
        """Получить timestamp для фильтрации по временному окну."""
        now = datetime.now()
        
        if time_window == "24h":
            cutoff = now - timedelta(hours=self.config.short_term_hours)
        elif time_window == "7d":
            cutoff = now - timedelta(days=self.config.medium_term_days)
        elif time_window == "30d":
            cutoff = now - timedelta(days=self.config.long_term_days)
        else:
            cutoff = now - timedelta(days=self.config.medium_term_days)
        
        return int(cutoff.timestamp() * 1000)

    def _get_time_span_days(self, time_window: str) -> float:
        """Получить длительность временного окна в днях."""
        if time_window == "24h":
            return self.config.short_term_hours / 24.0
        elif time_window == "7d":
            return float(self.config.medium_term_days)
        elif time_window == "30d":
            return float(self.config.long_term_days)
        else:
            return float(self.config.medium_term_days)

    def _invalidate_cache(self, strategy: str, symbol: str):
        """Инвалидировать кэш метрик для стратегии."""
        keys_to_remove = [
            key for key in self.metrics_cache.keys()
            if key[0] == strategy and key[1] == symbol
        ]
        
        for key in keys_to_remove:
            del self.metrics_cache[key]
            if key in self.cache_timestamp:
                del self.cache_timestamp[key]

    def _persist_outcome(self, outcome: SignalOutcome):
        """Сохранить outcome в файл."""
        try:
            filename = f"{outcome.strategy}_{outcome.symbol}_outcomes.jsonl"
            filepath = os.path.join(self.config.data_dir, filename)
            
            with open(filepath, 'a') as f:
                f.write(json.dumps(outcome.to_dict()) + '\n')
        
        except Exception as e:
            logger.error(f"Ошибка сохранения outcome: {e}")

    def _load_historical_data(self):
        """Загрузить исторические данные из файлов."""
        if not self.config.enable_persistence:
            return
        
        data_dir = Path(self.config.data_dir)
        
        if not data_dir.exists():
            return
        
        try:
            for filepath in data_dir.glob("*_outcomes.jsonl"):
                with open(filepath, 'r') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        outcome = SignalOutcome(**data)
                        
                        # Добавляем в историю только закрытые outcomes
                        if outcome.outcome != "open":
                            self.outcomes[outcome.strategy][outcome.symbol].append(outcome)
            
            logger.info(
                f"Загружены исторические данные: "
                f"{len(self.outcomes)} стратегий"
            )
        
        except Exception as e:
            logger.error(f"Ошибка загрузки исторических данных: {e}")

    def get_statistics(self) -> Dict:
        """Получить общую статистику трекера."""
        total_outcomes_stored = sum(
            len(outcomes_by_symbol[symbol])
            for outcomes_by_symbol in self.outcomes.values()
            for symbol in outcomes_by_symbol
        )
        
        return {
            'total_signals_tracked': self.total_signals_tracked,
            'total_outcomes_recorded': self.total_outcomes_recorded,
            'open_positions': len(self.open_positions),
            'outcomes_stored': total_outcomes_stored,
            'strategies_tracked': len(self.outcomes),
            'cache_size': len(self.metrics_cache)
        }
