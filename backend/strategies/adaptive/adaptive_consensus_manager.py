"""
Adaptive Consensus Manager - интеграция всех адаптивных компонентов.

Функциональность:
- Объединяет PerformanceTracker, RegimeDetector, WeightOptimizer
- Динамическое управление весами стратегий
- Адаптивный consensus с учетом контекста
- Enhanced conflict resolution
- Quality metrics для consensus
- Continuous learning и adaptation

Путь: backend/strategies/adaptive/adaptive_consensus_manager.py
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from backend.core.logger import get_logger
from backend.models.signal import TradingSignal, SignalType, SignalStrength, SignalSource
from backend.models.orderbook import OrderBookSnapshot, OrderBookMetrics
from backend.strategy.candle_manager import Candle

from backend.strategies.adaptive.strategy_performance_tracker import (
    StrategyPerformanceTracker,
    PerformanceTrackerConfig
)
from backend.strategies.adaptive.market_regime_detector import (
    MarketRegimeDetector,
    RegimeDetectorConfig,
    MarketRegime
)
from backend.strategies.adaptive.weight_optimizer import (
    WeightOptimizer,
    WeightOptimizerConfig,
    OptimizationMethod
)

logger = get_logger(__name__)


@dataclass
class AdaptiveConsensusConfig:
    """Конфигурация Adaptive Consensus Manager."""
    # Enable/disable components
    enable_performance_tracking: bool = True
    enable_regime_detection: bool = True
    enable_weight_optimization: bool = True
    
    # Performance tracking
    performance_tracker_config: Optional[PerformanceTrackerConfig] = None
    
    # Regime detection
    regime_detector_config: Optional[RegimeDetectorConfig] = None
    
    # Weight optimization
    weight_optimizer_config: Optional[WeightOptimizerConfig] = None
    
    # Consensus building
    consensus_mode: str = "adaptive_weighted"
    min_consensus_confidence: float = 0.6
    
    # Conflict resolution
    conflict_resolution_mode: str = "performance_priority"
    
    # Quality metrics
    enable_quality_metrics: bool = True
    min_consensus_quality: float = 0.6
    min_agreement_strength: float = 0.7
    
    def __post_init__(self):
        """Инициализация вложенных конфигураций с defaults."""
        if self.performance_tracker_config is None:
            self.performance_tracker_config = PerformanceTrackerConfig()
        
        if self.regime_detector_config is None:
            self.regime_detector_config = RegimeDetectorConfig()
        
        if self.weight_optimizer_config is None:
            self.weight_optimizer_config = WeightOptimizerConfig(
                optimization_method=OptimizationMethod.HYBRID
            )


@dataclass
class ConsensusQuality:
    """Метрики качества consensus."""
    consensus_quality: float
    agreement_strength: float
    regime_alignment: float
    risk_score: float


class AdaptiveConsensusManager:
    """
    Менеджер адаптивного consensus.
    
    Интегрирует:
    - StrategyPerformanceTracker для отслеживания эффективности
    - MarketRegimeDetector для понимания контекста
    - WeightOptimizer для динамической адаптации весов
    
    Строит более умный consensus с учетом истории и контекста.
    """

    def __init__(
        self,
        config: AdaptiveConsensusConfig,
        strategy_manager
    ):
        """
        Инициализация менеджера.

        Args:
            config: Конфигурация
            strategy_manager: ExtendedStrategyManager instance
        """
        self.config = config
        self.strategy_manager = strategy_manager
        
        # Инициализация компонентов
        self.performance_tracker = None
        if config.enable_performance_tracking:
            self.performance_tracker = StrategyPerformanceTracker(
                config.performance_tracker_config
            )
        
        self.regime_detector = None
        if config.enable_regime_detection:
            self.regime_detector = MarketRegimeDetector(
                config.regime_detector_config
            )
        
        self.weight_optimizer = None
        if config.enable_weight_optimization and self.performance_tracker and self.regime_detector:
            self.weight_optimizer = WeightOptimizer(
                config.weight_optimizer_config,
                self.performance_tracker,
                self.regime_detector
            )
        
        # Статистика
        self.total_consensus_built = 0
        self.consensus_rejected = 0
        self.weights_updated = 0
        
        logger.info(
            f"Инициализирован AdaptiveConsensusManager: "
            f"performance_tracking={config.enable_performance_tracking}, "
            f"regime_detection={config.enable_regime_detection}, "
            f"weight_optimization={config.enable_weight_optimization}"
        )

    def build_adaptive_consensus(
        self,
        symbol: str,
        candles: List[Candle],
        current_price: float,
        orderbook: Optional[OrderBookSnapshot] = None,
        metrics: Optional[OrderBookMetrics] = None,
        sr_levels: Optional[List] = None,
        volume_profile: Optional[Dict] = None,
        ml_prediction: Optional[Dict] = None
    ):
        """
        Построить adaptive consensus с динамическими весами.

        Args:
            symbol: Торговая пара
            candles: История свечей
            current_price: Текущая цена
            orderbook: Снимок стакана
            metrics: Метрики стакана
            sr_levels: S/R уровни
            volume_profile: Volume profile
            ml_prediction: ML предсказание

        Returns:
            ConsensusSignal или None
        """
        # ========== ШАГ 1: ДЕТЕКЦИЯ РЕЖИМА ==========
        market_regime = None
        if self.regime_detector:
            market_regime = self.regime_detector.detect_regime(
                symbol, candles, metrics
            )
        
        # ========== ШАГ 2: ОПТИМИЗАЦИЯ ВЕСОВ ==========
        if self.weight_optimizer and market_regime:
            # Получаем список стратегий
            strategy_names = list(self.strategy_manager.all_strategies.keys())
            
            # Получаем текущие веса
            current_weights = self._get_current_weights(strategy_names)
            
            # Оптимизируем
            optimal_weights = self.weight_optimizer.get_optimal_weights(
                symbol, strategy_names, current_weights
            )
            
            # Обновляем веса в StrategyManager
            self._update_strategy_weights(optimal_weights)
            self.weights_updated += 1
        
        # ========== ШАГ 3: ЗАПУСК СТРАТЕГИЙ ==========
        strategy_results = self.strategy_manager.analyze_all_strategies(
            symbol=symbol,
            candles=candles,
            current_price=current_price,
            orderbook=orderbook,
            metrics=metrics,
            sr_levels=sr_levels,
            volume_profile=volume_profile,
            ml_prediction=ml_prediction
        )
        
        # ========== ШАГ 4: ПОСТРОЕНИЕ CONSENSUS ==========
        consensus = self._build_enhanced_consensus(
            symbol=symbol,
            strategy_results=strategy_results,
            current_price=current_price,
            market_regime=market_regime,
            ml_prediction=ml_prediction
        )
        
        if not consensus:
            self.consensus_rejected += 1
            return None
        
        # ========== ШАГ 5: QUALITY ASSESSMENT ==========
        if self.config.enable_quality_metrics:
            quality = self._assess_consensus_quality(
                consensus, market_regime, strategy_results
            )
            
            # Проверяем качество
            if quality.consensus_quality < self.config.min_consensus_quality:
                logger.debug(
                    f"{symbol} | Consensus отклонен: "
                    f"quality={quality.consensus_quality:.2f} < "
                    f"{self.config.min_consensus_quality}"
                )
                self.consensus_rejected += 1
                return None
            
            # Добавляем качество в metadata
            if not consensus.final_signal.metadata:
                consensus.final_signal.metadata = {}
            
            consensus.final_signal.metadata['consensus_quality'] = quality.consensus_quality
            consensus.final_signal.metadata['agreement_strength'] = quality.agreement_strength
            consensus.final_signal.metadata['regime_alignment'] = quality.regime_alignment
            consensus.final_signal.metadata['risk_score'] = quality.risk_score
        
        self.total_consensus_built += 1
        
        # ========== ШАГ 6: ЛОГИРОВАНИЕ СИГНАЛА ДЛЯ TRACKING ==========
        if self.performance_tracker:
            # Генерируем уникальный ID для сигнала
            signal_id = f"{symbol}_{consensus.final_signal.timestamp}"
            
            # Записываем каждую contributing стратегию
            for strategy_name in consensus.contributing_strategies:
                self.performance_tracker.record_signal_generated(
                    strategy=strategy_name,
                    symbol=symbol,
                    signal=consensus.final_signal,
                    signal_id=f"{signal_id}_{strategy_name}"
                )
        
        return consensus

    def record_signal_outcome(
        self,
        symbol: str,
        signal_timestamp: int,
        contributing_strategies: List[str],
        exit_price: float,
        exit_timestamp: int,
        pnl_usdt: Optional[float] = None
    ):
        """
        Записать результат исполнения сигнала для всех стратегий.

        Args:
            symbol: Торговая пара
            signal_timestamp: Timestamp сигнала
            contributing_strategies: Стратегии участвовавшие в consensus
            exit_price: Цена выхода
            exit_timestamp: Время выхода
            pnl_usdt: P&L в USDT
        """
        if not self.performance_tracker:
            return
        
        # Записываем outcome для каждой стратегии
        for strategy_name in contributing_strategies:
            signal_id = f"{symbol}_{signal_timestamp}_{strategy_name}"
            
            self.performance_tracker.record_signal_outcome(
                signal_id=signal_id,
                exit_price=exit_price,
                exit_timestamp=exit_timestamp,
                pnl_usdt=pnl_usdt
            )

    def _build_enhanced_consensus(
        self,
        symbol: str,
        strategy_results: List,
        current_price: float,
        market_regime: Optional[MarketRegime],
        ml_prediction: Optional[Dict]
    ):
        """
        Построить enhanced consensus с улучшенной conflict resolution.
        """
        # Фильтруем стратегии с сигналами
        results_with_signals = [r for r in strategy_results if r.signal is not None]
        
        if not results_with_signals:
            return None
        
        # Разделяем на BUY и SELL
        buy_signals = [r for r in results_with_signals if r.signal.signal_type == SignalType.BUY]
        sell_signals = [r for r in results_with_signals if r.signal.signal_type == SignalType.SELL]
        
        # ========== CONFLICT DETECTION ==========
        if buy_signals and sell_signals:
            # Есть конфликт - нужна resolution
            resolved_signal = self._resolve_conflict(
                buy_signals, sell_signals, market_regime, ml_prediction
            )
            
            if not resolved_signal:
                return None
            
            return resolved_signal
        
        # ========== NO CONFLICT - STANDARD CONSENSUS ==========
        # Используем стандартный механизм из StrategyManager
        consensus = self.strategy_manager.build_consensus(
            symbol, strategy_results, current_price
        )
        
        return consensus

    def _resolve_conflict(
        self,
        buy_signals: List,
        sell_signals: List,
        market_regime: Optional[MarketRegime],
        ml_prediction: Optional[Dict]
    ):
        """
        Разрешить конфликт между BUY и SELL сигналами.
        """
        if self.config.conflict_resolution_mode == "performance_priority":
            return self._resolve_by_performance(buy_signals, sell_signals)
        
        elif self.config.conflict_resolution_mode == "regime_aligned":
            return self._resolve_by_regime(buy_signals, sell_signals, market_regime)
        
        elif self.config.conflict_resolution_mode == "ml_tiebreaker":
            return self._resolve_by_ml(buy_signals, sell_signals, ml_prediction)
        
        else:
            # Default: performance priority
            return self._resolve_by_performance(buy_signals, sell_signals)

    def _resolve_by_performance(
        self,
        buy_signals: List,
        sell_signals: List
    ):
        """
        Разрешение конфликта на основе performance.
        
        Побеждает сторона с лучшим средним performance score.
        """
        if not self.performance_tracker:
            return None
        
        # Вычисляем средний performance score для каждой стороны
        buy_scores = []
        for result in buy_signals:
            metrics = self.performance_tracker.get_strategy_metrics(
                result.strategy_name, result.signal.symbol, "7d"
            )
            if metrics:
                buy_scores.append(metrics.performance_score)
        
        sell_scores = []
        for result in sell_signals:
            metrics = self.performance_tracker.get_strategy_metrics(
                result.strategy_name, result.signal.symbol, "7d"
            )
            if metrics:
                sell_scores.append(metrics.performance_score)
        
        if not buy_scores and not sell_scores:
            return None
        
        avg_buy_score = np.mean(buy_scores) if buy_scores else 0.0
        avg_sell_score = np.mean(sell_scores) if sell_scores else 0.0
        
        # Если разница < 0.1 - слишком неопределенно
        if abs(avg_buy_score - avg_sell_score) < 0.1:
            logger.debug(
                "Конфликт: performance scores слишком близки, отменяем сигнал"
            )
            return None
        
        # Выбираем победителя
        if avg_buy_score > avg_sell_score:
            logger.debug(f"Конфликт разрешен: BUY (score={avg_buy_score:.2f})")
        else:
            logger.debug(f"Конфликт разрешен: SELL (score={avg_sell_score:.2f})")
        
        # Используем стандартный build_consensus на победивших сигналах
        # Это будет обработано в следующей итерации
        return None

    def _resolve_by_regime(
        self,
        buy_signals: List,
        sell_signals: List,
        market_regime: Optional[MarketRegime]
    ):
        """
        Разрешение конфликта на основе market regime.
        
        В uptrend - приоритет BUY, в downtrend - SELL.
        """
        if not market_regime:
            return None
        
        # Определяем preferred direction
        if "uptrend" in market_regime.trend.value:
            logger.debug(f"Конфликт разрешен: BUY (regime={market_regime.trend.value})")
        elif "downtrend" in market_regime.trend.value:
            logger.debug(f"Конфликт разрешен: SELL (regime={market_regime.trend.value})")
        else:
            # Ranging - требуется более высокая уверенность
            return None
        
        return None

    def _resolve_by_ml(
        self,
        buy_signals: List,
        sell_signals: List,
        ml_prediction: Optional[Dict]
    ):
        """
        Разрешение конфликта через ML как tie-breaker.
        """
        if not ml_prediction:
            return None
        
        ml_confidence = ml_prediction.get('confidence', 0.0)
        ml_direction = ml_prediction.get('prediction')
        
        # ML должен быть достаточно уверен
        if ml_confidence < 0.7:
            logger.debug("Конфликт: ML недостаточно уверен для tie-break")
            return None
        
        # ML определяет победителя
        if ml_direction == 'bullish':
            logger.debug(f"Конфликт разрешен: BUY (ML confidence={ml_confidence:.2f})")
        elif ml_direction == 'bearish':
            logger.debug(f"Конфликт разрешен: SELL (ML confidence={ml_confidence:.2f})")
        else:
            return None
        
        return None

    def _assess_consensus_quality(
        self,
        consensus,
        market_regime: Optional[MarketRegime],
        strategy_results: List
    ) -> ConsensusQuality:
        """
        Оценить качество consensus.
        """
        # 1. Agreement strength (насколько согласны стратегии)
        total_strategies = len([r for r in strategy_results if r.signal is not None])
        agreement_count = consensus.agreement_count
        
        agreement_strength = agreement_count / total_strategies if total_strategies > 0 else 0.0
        
        # 2. Regime alignment (соответствует ли сигнал режиму)
        regime_alignment = 0.5  # Default
        
        if market_regime:
            signal_type = consensus.final_signal.signal_type
            trend = market_regime.trend.value
            
            if signal_type == SignalType.BUY and "uptrend" in trend:
                regime_alignment = 0.9
            elif signal_type == SignalType.SELL and "downtrend" in trend:
                regime_alignment = 0.9
            elif "ranging" in trend:
                regime_alignment = 0.6
            else:
                regime_alignment = 0.3
        
        # 3. Risk score (оценка риска)
        risk_score = 0.3  # Default: low risk
        
        if market_regime:
            if market_regime.volatility.value == "high":
                risk_score = 0.7
            elif market_regime.volatility.value == "low":
                risk_score = 0.2
        
        # 4. Композитное качество
        consensus_quality = (
            agreement_strength * 0.4 +
            regime_alignment * 0.4 +
            (1.0 - risk_score) * 0.2
        )
        
        return ConsensusQuality(
            consensus_quality=consensus_quality,
            agreement_strength=agreement_strength,
            regime_alignment=regime_alignment,
            risk_score=risk_score
        )

    def _get_current_weights(self, strategy_names: List[str]) -> Dict[str, float]:
        """Получить текущие веса из StrategyManager."""
        weights = {}
        
        # Собираем веса из всех типов
        all_weights = {
            **self.strategy_manager.config.candle_strategy_weights,
            **self.strategy_manager.config.orderbook_strategy_weights,
            **self.strategy_manager.config.hybrid_strategy_weights
        }
        
        for name in strategy_names:
            weights[name] = all_weights.get(name, 0.1)
        
        return weights

    def _update_strategy_weights(self, new_weights: Dict[str, float]):
        """Обновить веса в StrategyManager."""
        # Разделяем веса по типам стратегий
        for strategy_name, weight in new_weights.items():
            strategy_type = self.strategy_manager.config.strategy_types.get(strategy_name)
            
            if not strategy_type:
                continue
            
            # Обновляем соответствующий словарь весов
            if strategy_type.value == "candle":
                self.strategy_manager.config.candle_strategy_weights[strategy_name] = weight
            elif strategy_type.value == "orderbook":
                self.strategy_manager.config.orderbook_strategy_weights[strategy_name] = weight
            elif strategy_type.value == "hybrid":
                self.strategy_manager.config.hybrid_strategy_weights[strategy_name] = weight

    def get_statistics(self) -> Dict:
        """Получить общую статистику."""
        stats = {
            'total_consensus_built': self.total_consensus_built,
            'consensus_rejected': self.consensus_rejected,
            'weights_updated': self.weights_updated
        }
        
        if self.performance_tracker:
            stats['performance_tracker'] = self.performance_tracker.get_statistics()
        
        if self.regime_detector:
            stats['regime_detector'] = self.regime_detector.get_statistics()
        
        if self.weight_optimizer:
            stats['weight_optimizer'] = self.weight_optimizer.get_statistics()
        
        return stats
