"""
Weight Optimizer - динамическая оптимизация весов стратегий.

Функциональность:
- Performance-based optimization (EWMA)
- Regime-adaptive optimization
- Bayesian optimization (Thompson Sampling)
- Constraints и safeguards
- Smooth transitions между весами
- Emergency rebalancing при деградации

Путь: backend/strategies/adaptive/weight_optimizer.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
from scipy.stats import beta

from core.logger import get_logger
from strategies.adaptive.strategy_performance_tracker import (
    StrategyPerformanceTracker,
    StrategyMetrics
)
from strategies.adaptive.market_regime_detector import (
    MarketRegimeDetector,
    MarketRegime
)

logger = get_logger(__name__)


class OptimizationMethod(Enum):
    """Методы оптимизации."""
    PERFORMANCE_BASED = "performance_based"  # На основе метрик
    REGIME_ADAPTIVE = "regime_adaptive"      # На основе режима рынка
    BAYESIAN = "bayesian"                    # Thompson Sampling
    HYBRID = "hybrid"                        # Комбинация всех


@dataclass
class WeightOptimizerConfig:
    """Конфигурация Weight Optimizer."""
    # Optimization method
    optimization_method: OptimizationMethod = OptimizationMethod.HYBRID
    
    # Constraints
    min_weight: float = 0.05  # Минимальный вес (5%)
    max_weight: float = 0.40  # Максимальный вес (40%)
    
    # Smoothing
    max_weight_change_per_update: float = 0.05  # Макс изменение за раз
    smoothing_factor: float = 0.3  # Для exponential smoothing
    
    # Update frequency
    update_frequency_seconds: int = 21600  # 6 часов
    micro_adjustment_seconds: int = 3600   # 1 час для мелких корректировок
    
    # Performance-based
    performance_window: str = "7d"
    min_signals_for_optimization: int = 30
    
    # Regime-adaptive
    regime_weight_blend: float = 0.6  # 60% performance, 40% regime
    
    # Bayesian (Thompson Sampling)
    bayesian_alpha_prior: float = 1.0
    bayesian_beta_prior: float = 1.0
    exploration_rate: float = 0.1  # 10% exploration
    
    # Safeguards
    enable_diversity_constraints: bool = True
    min_strategies_above_threshold: int = 2  # Минимум 2 стратегии > 0.15
    
    enable_outlier_protection: bool = True
    outlier_loss_threshold_pct: float = -5.0  # -5% потерь = outlier
    
    cooldown_period_hours: int = 24  # После аномальных потерь
    
    # Emergency rebalancing
    enable_emergency_rebalancing: bool = True
    emergency_performance_threshold: float = 0.25  # Performance < 0.25


@dataclass
class WeightUpdate:
    """Обновление весов."""
    timestamp: int
    symbol: str
    method: str
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    changes: Dict[str, float]
    reason: str
    market_regime: Optional[str] = None


class WeightOptimizer:
    """
    Оптимизатор весов стратегий.
    
    Динамически адаптирует веса на основе:
    - Текущей performance
    - Рыночного режима
    - Байесовской оптимизации
    """

    def __init__(
        self,
        config: WeightOptimizerConfig,
        performance_tracker: StrategyPerformanceTracker,
        regime_detector: MarketRegimeDetector
    ):
        """
        Инициализация оптимизатора.

        Args:
            config: Конфигурация
            performance_tracker: Трекер производительности
            regime_detector: Детектор режимов
        """
        self.config = config
        self.performance_tracker = performance_tracker
        self.regime_detector = regime_detector
        
        # Текущие веса для каждого символа
        self.current_weights: Dict[str, Dict[str, float]] = {}
        
        # История обновлений
        self.update_history: Dict[str, List[WeightUpdate]] = {}
        self.max_history_size = 100
        
        # Timestamp последнего обновления
        self.last_major_update: Dict[str, int] = {}
        self.last_micro_update: Dict[str, int] = {}
        
        # Bayesian parameters (Thompson Sampling)
        # strategy -> (alpha, beta) parameters
        self.bayesian_params: Dict[str, Dict[str, Tuple[float, float]]] = {}
        
        # Cooldown tracking (после аномальных потерь)
        self.strategy_cooldowns: Dict[str, Dict[str, int]] = {}  # symbol -> strategy -> timestamp
        
        # Статистика
        self.total_optimizations = 0
        self.emergency_rebalances = 0
        
        logger.info(
            f"Инициализирован WeightOptimizer: "
            f"method={config.optimization_method.value}, "
            f"update_freq={config.update_frequency_seconds}s"
        )

    def get_optimal_weights(
        self,
        symbol: str,
        strategy_names: List[str],
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Получить оптимальные веса для символа.

        Args:
            symbol: Торговая пара
            strategy_names: Список имен стратегий
            current_weights: Текущие веса (опционально)

        Returns:
            Dict[strategy_name, weight]
        """
        # Проверяем нужно ли обновление
        if not self._should_update(symbol):
            if symbol in self.current_weights:
                return self.current_weights[symbol]
        
        # Если нет текущих весов - используем равномерное распределение
        if current_weights is None:
            if symbol in self.current_weights:
                current_weights = self.current_weights[symbol]
            else:
                current_weights = {name: 1.0 / len(strategy_names) for name in strategy_names}
        
        # Получаем рыночный режим
        market_regime = self.regime_detector.get_current_regime(symbol)
        
        # Выбор метода оптимизации
        if self.config.optimization_method == OptimizationMethod.PERFORMANCE_BASED:
            optimized_weights = self._optimize_by_performance(
                symbol, strategy_names, current_weights
            )
        
        elif self.config.optimization_method == OptimizationMethod.REGIME_ADAPTIVE:
            optimized_weights = self._optimize_by_regime(
                symbol, strategy_names, current_weights, market_regime
            )
        
        elif self.config.optimization_method == OptimizationMethod.BAYESIAN:
            optimized_weights = self._optimize_bayesian(
                symbol, strategy_names, current_weights
            )
        
        else:  # HYBRID
            optimized_weights = self._optimize_hybrid(
                symbol, strategy_names, current_weights, market_regime
            )
        
        # Применяем constraints
        constrained_weights = self._apply_constraints(optimized_weights, strategy_names)
        
        # Smoothing (постепенные изменения)
        smoothed_weights = self._apply_smoothing(
            current_weights, constrained_weights
        )
        
        # Проверяем cooldowns
        final_weights = self._apply_cooldowns(symbol, smoothed_weights)
        
        # Проверяем emergency conditions
        if self.config.enable_emergency_rebalancing:
            emergency_weights = self._check_emergency_rebalance(
                symbol, strategy_names, final_weights
            )
            if emergency_weights:
                final_weights = emergency_weights
        
        # Нормализация (сумма = 1.0)
        final_weights = self._normalize_weights(final_weights)
        
        # Записываем обновление
        self._record_update(
            symbol, current_weights, final_weights, 
            self.config.optimization_method.value, market_regime
        )
        
        # Сохраняем
        self.current_weights[symbol] = final_weights
        self.last_major_update[symbol] = int(datetime.now().timestamp())
        self.total_optimizations += 1
        
        return final_weights

    def _optimize_by_performance(
        self,
        symbol: str,
        strategy_names: List[str],
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Оптимизация на основе performance метрик (EWMA).
        """
        performance_scores = {}
        
        for strategy in strategy_names:
            metrics = self.performance_tracker.get_strategy_metrics(
                strategy, symbol, self.config.performance_window
            )
            
            if metrics and metrics.closed_signals >= self.config.min_signals_for_optimization:
                performance_scores[strategy] = metrics.performance_score
            else:
                # Недостаточно данных - используем текущий вес
                performance_scores[strategy] = current_weights.get(strategy, 0.1)
        
        if not performance_scores:
            return current_weights
        
        # Нормализуем scores в веса
        total_score = sum(performance_scores.values())
        
        if total_score == 0:
            return current_weights
        
        optimized_weights = {
            strategy: score / total_score
            for strategy, score in performance_scores.items()
        }
        
        return optimized_weights

    def _optimize_by_regime(
        self,
        symbol: str,
        strategy_names: List[str],
        current_weights: Dict[str, float],
        market_regime: Optional[MarketRegime]
    ) -> Dict[str, float]:
        """
        Оптимизация на основе рыночного режима.
        """
        if not market_regime:
            return current_weights
        
        # Получаем рекомендуемые веса из regime
        recommended_weights = market_regime.recommended_strategy_weights
        
        # Фильтруем только наши стратегии
        optimized_weights = {}
        for strategy in strategy_names:
            if strategy in recommended_weights:
                optimized_weights[strategy] = recommended_weights[strategy]
            else:
                # Если стратегии нет в рекомендациях - используем минимальный вес
                optimized_weights[strategy] = self.config.min_weight
        
        return optimized_weights

    def _optimize_bayesian(
        self,
        symbol: str,
        strategy_names: List[str],
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Байесовская оптимизация через Thompson Sampling.
        
        Балансирует exploration (пробовать разные веса) и 
        exploitation (использовать лучшие веса).
        """
        # Инициализируем параметры если нужно
        if symbol not in self.bayesian_params:
            self.bayesian_params[symbol] = {}
        
        sampled_weights = {}
        
        for strategy in strategy_names:
            # Получаем или инициализируем Beta parameters
            if strategy not in self.bayesian_params[symbol]:
                # Prior: Beta(1, 1) = uniform
                alpha = self.config.bayesian_alpha_prior
                beta_param = self.config.bayesian_beta_prior
            else:
                alpha, beta_param = self.bayesian_params[symbol][strategy]
            
            # Обновляем параметры на основе недавних результатов
            metrics = self.performance_tracker.get_strategy_metrics(
                strategy, symbol, "24h"
            )
            
            if metrics and metrics.closed_signals > 0:
                # Обновляем: alpha += wins, beta += losses
                alpha += metrics.win_count
                beta_param += metrics.loss_count
                
                # Сохраняем обновленные параметры
                self.bayesian_params[symbol][strategy] = (alpha, beta_param)
            
            # Thompson Sampling: sample from Beta distribution
            # Exploration vs Exploitation
            if np.random.random() < self.config.exploration_rate:
                # Exploration: случайный вес
                sampled_value = np.random.uniform(0.05, 0.35)
            else:
                # Exploitation: sample from posterior
                sampled_value = np.random.beta(alpha, beta_param)
            
            sampled_weights[strategy] = sampled_value
        
        return sampled_weights

    def _optimize_hybrid(
        self,
        symbol: str,
        strategy_names: List[str],
        current_weights: Dict[str, float],
        market_regime: Optional[MarketRegime]
    ) -> Dict[str, float]:
        """
        Гибридная оптимизация: комбинация performance и regime.
        
        Formula: final = α * performance + (1-α) * regime
        где α = regime_weight_blend
        """
        # Performance weights
        performance_weights = self._optimize_by_performance(
            symbol, strategy_names, current_weights
        )
        
        # Regime weights
        regime_weights = self._optimize_by_regime(
            symbol, strategy_names, current_weights, market_regime
        )
        
        # Blend
        alpha = self.config.regime_weight_blend
        
        hybrid_weights = {}
        for strategy in strategy_names:
            perf_weight = performance_weights.get(strategy, current_weights.get(strategy, 0.1))
            regime_weight = regime_weights.get(strategy, current_weights.get(strategy, 0.1))
            
            blended = alpha * perf_weight + (1 - alpha) * regime_weight
            hybrid_weights[strategy] = blended
        
        return hybrid_weights

    def _apply_constraints(
        self,
        weights: Dict[str, float],
        strategy_names: List[str]
    ) -> Dict[str, float]:
        """
        Применить constraints на веса.
        
        - Min/Max weight per strategy
        - Diversity constraints
        """
        constrained = {}
        
        for strategy in strategy_names:
            weight = weights.get(strategy, 0.0)
            
            # Min/Max constraints
            weight = max(self.config.min_weight, min(weight, self.config.max_weight))
            
            constrained[strategy] = weight
        
        # Diversity constraint: минимум N стратегий должны иметь weight > threshold
        if self.config.enable_diversity_constraints:
            threshold = 0.15
            strategies_above = [s for s, w in constrained.items() if w >= threshold]
            
            if len(strategies_above) < self.config.min_strategies_above_threshold:
                # Boost слабых стратегий
                weak_strategies = [s for s in strategy_names if s not in strategies_above]
                
                for strategy in weak_strategies[:self.config.min_strategies_above_threshold]:
                    constrained[strategy] = threshold
        
        return constrained

    def _apply_smoothing(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Применить smoothing для постепенных изменений.
        
        Exponential smoothing + максимальное изменение за раз.
        """
        smoothed = {}
        
        for strategy, new_weight in new_weights.items():
            old_weight = old_weights.get(strategy, new_weight)
            
            # Exponential smoothing
            smoothing = self.config.smoothing_factor
            ema_weight = smoothing * new_weight + (1 - smoothing) * old_weight
            
            # Ограничение максимального изменения
            max_change = self.config.max_weight_change_per_update
            
            if abs(ema_weight - old_weight) > max_change:
                if ema_weight > old_weight:
                    smoothed_weight = old_weight + max_change
                else:
                    smoothed_weight = old_weight - max_change
            else:
                smoothed_weight = ema_weight
            
            smoothed[strategy] = smoothed_weight
        
        return smoothed

    def _apply_cooldowns(
        self,
        symbol: str,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Применить cooldowns после аномальных потерь.
        """
        if symbol not in self.strategy_cooldowns:
            return weights
        
        current_time = int(datetime.now().timestamp())
        cooldown_duration = self.config.cooldown_period_hours * 3600
        
        adjusted_weights = weights.copy()
        
        for strategy, cooldown_start in self.strategy_cooldowns[symbol].items():
            # Проверяем истек ли cooldown
            if (current_time - cooldown_start) < cooldown_duration:
                # Cooldown активен - снижаем вес
                adjusted_weights[strategy] = self.config.min_weight
                
                logger.debug(
                    f"[{symbol}] {strategy} в cooldown "
                    f"({(current_time - cooldown_start) / 3600:.1f}h)"
                )
            else:
                # Cooldown истек - удаляем
                del self.strategy_cooldowns[symbol][strategy]
        
        return adjusted_weights

    def _check_emergency_rebalance(
        self,
        symbol: str,
        strategy_names: List[str],
        current_weights: Dict[str, float]
    ) -> Optional[Dict[str, float]]:
        """
        Проверить необходимость emergency rebalancing.
        
        Срабатывает при критической деградации.
        """
        emergency_detected = False
        degraded_strategies = []
        
        for strategy in strategy_names:
            metrics = self.performance_tracker.get_strategy_metrics(
                strategy, symbol, "24h"
            )
            
            if not metrics:
                continue
            
            # Проверяем performance score
            if metrics.performance_score < self.config.emergency_performance_threshold:
                emergency_detected = True
                degraded_strategies.append(strategy)
                
                logger.warning(
                    f"⚠️  EMERGENCY [{symbol}] {strategy}: "
                    f"performance={metrics.performance_score:.2f} < "
                    f"{self.config.emergency_performance_threshold}"
                )
        
        if not emergency_detected:
            return None
        
        # Emergency rebalancing: снижаем веса деградировавших стратегий
        emergency_weights = current_weights.copy()
        
        for strategy in degraded_strategies:
            emergency_weights[strategy] = self.config.min_weight
            
            # Добавляем в cooldown
            if symbol not in self.strategy_cooldowns:
                self.strategy_cooldowns[symbol] = {}
            
            self.strategy_cooldowns[symbol][strategy] = int(datetime.now().timestamp())
        
        # Перераспределяем веса на здоровые стратегии
        healthy_strategies = [s for s in strategy_names if s not in degraded_strategies]
        
        if healthy_strategies:
            # Вычисляем доступный вес
            total_degraded_weight = sum(
                current_weights.get(s, 0.0) for s in degraded_strategies
            )
            
            # Распределяем пропорционально текущим весам
            healthy_weights_sum = sum(
                current_weights.get(s, 0.0) for s in healthy_strategies
            )
            
            if healthy_weights_sum > 0:
                for strategy in healthy_strategies:
                    proportion = current_weights.get(strategy, 0.0) / healthy_weights_sum
                    boost = total_degraded_weight * proportion
                    emergency_weights[strategy] = current_weights.get(strategy, 0.0) + boost
        
        self.emergency_rebalances += 1
        
        logger.warning(
            f"🚨 EMERGENCY REBALANCE [{symbol}]: "
            f"degraded={degraded_strategies}"
        )
        
        return emergency_weights

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Нормализовать веса чтобы сумма = 1.0."""
        total = sum(weights.values())
        
        if total == 0:
            # Равномерное распределение
            equal_weight = 1.0 / len(weights)
            return {s: equal_weight for s in weights.keys()}
        
        return {s: w / total for s, w in weights.items()}

    def _record_update(
        self,
        symbol: str,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
        method: str,
        market_regime: Optional[MarketRegime]
    ):
        """Записать обновление весов."""
        # Вычисляем изменения
        changes = {}
        for strategy in new_weights.keys():
            old = old_weights.get(strategy, 0.0)
            new = new_weights[strategy]
            changes[strategy] = new - old
        
        # Формируем reason
        significant_changes = [
            f"{s}: {old_weights.get(s, 0.0):.2f}→{new_weights[s]:.2f}"
            for s in changes.keys()
            if abs(changes[s]) > 0.02  # Изменение > 2%
        ]
        
        reason = f"Method: {method}"
        if significant_changes:
            reason += f" | Changes: {', '.join(significant_changes[:3])}"
        
        update = WeightUpdate(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            method=method,
            old_weights=old_weights.copy(),
            new_weights=new_weights.copy(),
            changes=changes,
            reason=reason,
            market_regime=market_regime.trend.value if market_regime else None
        )
        
        # Добавляем в историю
        if symbol not in self.update_history:
            self.update_history[symbol] = []
        
        self.update_history[symbol].append(update)
        
        if len(self.update_history[symbol]) > self.max_history_size:
            self.update_history[symbol].pop(0)
        
        # Логируем значительные изменения
        if significant_changes:
            logger.info(
                f"⚖️  WEIGHTS UPDATED [{symbol}]: {reason}"
            )

    def _should_update(self, symbol: str) -> bool:
        """Проверить нужно ли major обновление."""
        if symbol not in self.last_major_update:
            return True
        
        last_update = self.last_major_update[symbol]
        current_time = int(datetime.now().timestamp())
        
        return (current_time - last_update) >= self.config.update_frequency_seconds

    def get_statistics(self) -> Dict:
        """Получить статистику оптимизатора."""
        return {
            'total_optimizations': self.total_optimizations,
            'emergency_rebalances': self.emergency_rebalances,
            'symbols_tracked': len(self.current_weights),
            'active_cooldowns': sum(len(v) for v in self.strategy_cooldowns.values())
        }
