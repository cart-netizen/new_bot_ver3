"""
Trade Reporter - детальные отчёты о реализованных сделках.

Записывает в trades.log полную информацию о каждой сделке:
- Индикаторы всех стратегий, участвовавших в решении
- ML прогнозы от ансамбля моделей (CNN-LSTM, MPD-Transformer, TLOB)
- MTF анализ (таймфреймы, согласование)
- Финальное решение и причины

Путь: backend/core/trade_reporter.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from backend.core.logger import get_trades_logger
from backend.models.signal import TradingSignal, SignalType


@dataclass
class StrategyIndicators:
    """Индикаторы от одной стратегии."""
    strategy_name: str
    signal_type: Optional[str]  # BUY/SELL/None
    confidence: Optional[float]
    indicators: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class MLModelPrediction:
    """Прогноз от одной ML модели."""
    model_name: str  # cnn_lstm, mpd_transformer, tlob
    direction: str  # BUY/SELL/HOLD
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)  # {BUY: 0.6, SELL: 0.2, HOLD: 0.2}
    expected_return: Optional[float] = None


@dataclass
class MTFAnalysis:
    """Результаты Multi-Timeframe анализа."""
    synthesis_mode: str
    timeframes_analyzed: List[str]
    timeframes_agreeing: List[str]
    alignment_score: float
    signal_quality: float
    reliability_score: float
    risk_level: str
    recommended_sl: Optional[float] = None
    recommended_tp: Optional[float] = None


@dataclass
class TradeReport:
    """Полный отчёт о сделке."""
    # Основная информация
    symbol: str
    direction: str  # BUY/SELL
    price: float
    timestamp: datetime

    # Итоговая уверенность и причина
    final_confidence: float
    final_reason: str

    # Стратегии
    strategies: List[StrategyIndicators] = field(default_factory=list)
    consensus_mode: str = "weighted"
    strategies_agreeing: int = 0
    strategies_disagreeing: int = 0

    # ML Ensemble
    ml_predictions: List[MLModelPrediction] = field(default_factory=list)
    ml_ensemble_direction: Optional[str] = None
    ml_ensemble_confidence: Optional[float] = None
    ml_validation_passed: bool = True
    ml_validation_reason: str = ""

    # MTF Analysis
    mtf_analysis: Optional[MTFAnalysis] = None

    # Risk Management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_usdt: Optional[float] = None
    risk_reward_ratio: Optional[float] = None

    # Additional context
    market_regime: Optional[str] = None
    orderbook_imbalance: Optional[float] = None
    volume_ratio: Optional[float] = None


class TradeReporter:
    """Формирование и запись детальных отчётов о сделках."""

    def __init__(self):
        self.trades_logger = get_trades_logger()
        self._reports_count = 0

    def log_trade(self, report: TradeReport) -> None:
        """
        Записать детальный отчёт о сделке в trades.log.

        Args:
            report: Полный отчёт о сделке
        """
        self._reports_count += 1

        # Формируем отчёт
        report_text = self._format_report(report)

        # Записываем в trades.log
        self.trades_logger.info(report_text)

    def _format_report(self, report: TradeReport) -> str:
        """Форматирование отчёта в читаемый текст."""
        lines = []

        # ===== ЗАГОЛОВОК =====
        lines.append("=" * 80)
        lines.append(f"TRADE #{self._reports_count}: {report.symbol} {report.direction}")
        lines.append("=" * 80)
        lines.append("")

        # ===== ОСНОВНАЯ ИНФОРМАЦИЯ =====
        lines.append("📊 ОСНОВНАЯ ИНФОРМАЦИЯ:")
        lines.append(f"  Символ:      {report.symbol}")
        lines.append(f"  Направление: {report.direction}")
        lines.append(f"  Цена входа:  ${report.price:.6f}")
        lines.append(f"  Время:       {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Confidence:  {report.final_confidence:.2%}")
        lines.append(f"  Причина:     {report.final_reason}")
        lines.append("")

        # ===== СТРАТЕГИИ =====
        lines.append("📈 АНАЛИЗ СТРАТЕГИЙ:")
        lines.append(f"  Режим консенсуса: {report.consensus_mode}")
        lines.append(f"  Согласились: {report.strategies_agreeing} | Не согласились: {report.strategies_disagreeing}")
        lines.append("")

        for strategy in report.strategies:
            signal_str = strategy.signal_type if strategy.signal_type else "NO SIGNAL"
            conf_str = f"{strategy.confidence:.2%}" if strategy.confidence else "N/A"

            lines.append(f"  [{strategy.strategy_name}]")
            lines.append(f"    Сигнал: {signal_str} | Confidence: {conf_str}")

            # Индикаторы
            if strategy.indicators:
                lines.append("    Индикаторы:")
                for key, value in strategy.indicators.items():
                    if isinstance(value, float):
                        lines.append(f"      - {key}: {value:.4f}")
                    else:
                        lines.append(f"      - {key}: {value}")

            if strategy.reason:
                lines.append(f"    Причина: {strategy.reason}")
            lines.append("")

        # ===== ML ENSEMBLE =====
        if report.ml_predictions:
            lines.append("🤖 ML ENSEMBLE ПРОГНОЗЫ:")
            lines.append(f"  Итоговое направление: {report.ml_ensemble_direction or 'N/A'}")
            lines.append(f"  Итоговая уверенность: {report.ml_ensemble_confidence:.2%}" if report.ml_ensemble_confidence else "  Итоговая уверенность: N/A")
            lines.append(f"  ML валидация: {'✅ PASSED' if report.ml_validation_passed else '❌ FAILED'}")
            if report.ml_validation_reason:
                lines.append(f"  Причина: {report.ml_validation_reason}")
            lines.append("")

            for pred in report.ml_predictions:
                lines.append(f"  [{pred.model_name}]")
                lines.append(f"    Направление: {pred.direction} | Confidence: {pred.confidence:.2%}")
                if pred.probabilities:
                    probs_str = " | ".join([f"{k}: {v:.2%}" for k, v in pred.probabilities.items()])
                    lines.append(f"    Вероятности: {probs_str}")
                if pred.expected_return is not None:
                    lines.append(f"    Expected Return: {pred.expected_return:.4f}")
                lines.append("")
        else:
            lines.append("🤖 ML ENSEMBLE: Данные недоступны")
            lines.append("")

        # ===== MTF ANALYSIS =====
        if report.mtf_analysis:
            mtf = report.mtf_analysis
            lines.append("📊 MULTI-TIMEFRAME АНАЛИЗ:")
            lines.append(f"  Режим синтеза:      {mtf.synthesis_mode}")
            lines.append(f"  Таймфреймы:         {', '.join(mtf.timeframes_analyzed)}")
            lines.append(f"  Согласованные ТФ:   {', '.join(mtf.timeframes_agreeing)}")
            lines.append(f"  Alignment Score:    {mtf.alignment_score:.2%}")
            lines.append(f"  Signal Quality:     {mtf.signal_quality:.2%}")
            lines.append(f"  Reliability:        {mtf.reliability_score:.2%}")
            lines.append(f"  Risk Level:         {mtf.risk_level}")
            if mtf.recommended_sl:
                lines.append(f"  Рекомендуемый SL:   ${mtf.recommended_sl:.6f}")
            if mtf.recommended_tp:
                lines.append(f"  Рекомендуемый TP:   ${mtf.recommended_tp:.6f}")
            lines.append("")

        # ===== RISK MANAGEMENT =====
        lines.append("⚠️ RISK MANAGEMENT:")
        if report.stop_loss:
            lines.append(f"  Stop Loss:     ${report.stop_loss:.6f}")
        if report.take_profit:
            lines.append(f"  Take Profit:   ${report.take_profit:.6f}")
        if report.position_size_usdt:
            lines.append(f"  Position Size: ${report.position_size_usdt:.2f} USDT")
        if report.risk_reward_ratio:
            lines.append(f"  R/R Ratio:     1:{report.risk_reward_ratio:.2f}")
        lines.append("")

        # ===== ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ =====
        if report.market_regime or report.orderbook_imbalance is not None or report.volume_ratio is not None:
            lines.append("📋 ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ:")
            if report.market_regime:
                lines.append(f"  Market Regime:       {report.market_regime}")
            if report.orderbook_imbalance is not None:
                lines.append(f"  OrderBook Imbalance: {report.orderbook_imbalance:.2%}")
            if report.volume_ratio is not None:
                lines.append(f"  Volume Ratio:        {report.volume_ratio:.2f}x")
            lines.append("")

        lines.append("=" * 80)
        lines.append("")

        return "\n".join(lines)

    def create_report_from_signal(
        self,
        signal: TradingSignal,
        strategy_results: Optional[List] = None,
        ml_validation_result: Optional[Any] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        position_size: Optional[float] = None
    ) -> TradeReport:
        """
        Создать отчёт из TradingSignal и дополнительных данных.

        Args:
            signal: Исполняемый торговый сигнал
            strategy_results: Результаты от стратегий (List[StrategyResult])
            ml_validation_result: Результат ML валидации (ValidationResult)
            sl_price: Цена Stop Loss
            tp_price: Цена Take Profit
            position_size: Размер позиции в USDT

        Returns:
            TradeReport готовый к записи
        """
        report = TradeReport(
            symbol=signal.symbol,
            direction=signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
            price=signal.price,
            timestamp=datetime.fromtimestamp(signal.timestamp / 1000),
            final_confidence=signal.confidence,
            final_reason=signal.reason or "No reason provided",
            stop_loss=sl_price,
            take_profit=tp_price,
            position_size_usdt=position_size
        )

        # Рассчитываем R/R ratio
        if sl_price and tp_price and signal.price:
            if signal.signal_type == SignalType.BUY:
                risk = abs(signal.price - sl_price)
                reward = abs(tp_price - signal.price)
            else:
                risk = abs(sl_price - signal.price)
                reward = abs(signal.price - tp_price)

            if risk > 0:
                report.risk_reward_ratio = reward / risk

        # Обрабатываем стратегии
        if strategy_results:
            report.strategies = self._extract_strategy_indicators(strategy_results)
            report.strategies_agreeing = len([s for s in report.strategies if s.signal_type])
            report.strategies_disagreeing = len(report.strategies) - report.strategies_agreeing

        # Обрабатываем ML результаты
        if ml_validation_result:
            self._extract_ml_data(report, ml_validation_result)

        # Обрабатываем metadata сигнала
        if signal.metadata:
            self._extract_metadata(report, signal.metadata)

        return report

    def _extract_strategy_indicators(self, strategy_results: List) -> List[StrategyIndicators]:
        """Извлечение индикаторов из результатов стратегий (сериализованных в dict)."""
        strategies = []

        for result in strategy_results:
            # Поддержка как dict, так и объектов
            if isinstance(result, dict):
                # Сериализованный формат (dict)
                strategy_name = result.get('strategy_name', 'unknown')
                signal_data = result.get('signal')

                if signal_data:
                    signal_type = signal_data.get('signal_type')
                    confidence = signal_data.get('confidence')
                    reason = signal_data.get('reason', '')
                    signal_metadata = signal_data.get('metadata', {})
                    indicators = {k: v for k, v in signal_metadata.items()
                                 if k not in ['strategy', 'consensus_mode']}
                else:
                    signal_type = None
                    confidence = None
                    reason = ""
                    indicators = {}
            else:
                # Объект StrategyResult (для обратной совместимости)
                strategy_name = result.strategy_name
                indicators = {}
                signal_type = None
                confidence = None
                reason = ""

                if result.signal:
                    signal_type = result.signal.signal_type.value if hasattr(result.signal.signal_type, 'value') else str(result.signal.signal_type)
                    confidence = result.signal.confidence
                    reason = result.signal.reason or ""

                    if result.signal.metadata:
                        indicators = {k: v for k, v in result.signal.metadata.items()
                                     if k not in ['strategy', 'consensus_mode']}

            strategies.append(StrategyIndicators(
                strategy_name=strategy_name,
                signal_type=signal_type,
                confidence=confidence,
                indicators=indicators,
                reason=reason
            ))

        return strategies

    def _extract_ml_data(self, report: TradeReport, validation_result: Any) -> None:
        """Извлечение данных ML валидации (поддержка dict и объектов)."""
        # Поддержка как dict, так и объектов
        if isinstance(validation_result, dict):
            # Сериализованный формат (dict)
            report.ml_ensemble_direction = validation_result.get('ml_direction')
            report.ml_ensemble_confidence = validation_result.get('ml_confidence')
            report.ml_validation_passed = validation_result.get('validated', True)
            report.ml_validation_reason = validation_result.get('reason', '')

            # Market regime из сериализованного результата
            if validation_result.get('market_regime'):
                report.market_regime = validation_result['market_regime']

        else:
            # Объект ValidationResult (для обратной совместимости)
            report.ml_ensemble_direction = validation_result.ml_direction
            report.ml_ensemble_confidence = validation_result.ml_confidence
            report.ml_validation_passed = validation_result.validated
            report.ml_validation_reason = validation_result.reason

            # Если есть детальные прогнозы в metadata
            if hasattr(validation_result, 'metadata') and validation_result.metadata:
                meta = validation_result.metadata

                # Извлекаем прогнозы отдельных моделей если они есть
                if 'model_predictions' in meta:
                    for model_name, pred_data in meta['model_predictions'].items():
                        report.ml_predictions.append(MLModelPrediction(
                            model_name=model_name,
                            direction=pred_data.get('direction', 'UNKNOWN'),
                            confidence=pred_data.get('confidence', 0.0),
                            probabilities=pred_data.get('probabilities', {}),
                            expected_return=pred_data.get('expected_return')
                        ))

                # Рыночный режим
                if 'market_regime' in meta:
                    report.market_regime = meta['market_regime']

    def _extract_metadata(self, report: TradeReport, metadata: Dict) -> None:
        """Извлечение данных из metadata сигнала."""
        # MTF данные
        if metadata.get('has_mtf_risk_params'):
            report.mtf_analysis = MTFAnalysis(
                synthesis_mode=metadata.get('synthesis_mode', 'unknown'),
                timeframes_analyzed=metadata.get('timeframes_analyzed', []),
                timeframes_agreeing=metadata.get('timeframes_agreeing', []),
                alignment_score=metadata.get('mtf_alignment_score', 0.0),
                signal_quality=metadata.get('mtf_signal_quality', 0.0),
                reliability_score=metadata.get('mtf_reliability_score', 0.0),
                risk_level=metadata.get('mtf_risk_level', 'UNKNOWN'),
                recommended_sl=metadata.get('mtf_recommended_stop_loss'),
                recommended_tp=metadata.get('mtf_recommended_take_profit')
            )

        # Consensus mode
        if 'consensus_mode' in metadata:
            report.consensus_mode = metadata['consensus_mode']

        # OrderBook данные
        if 'imbalance' in metadata:
            report.orderbook_imbalance = metadata['imbalance']

        # Volume данные
        if 'volume_ratio' in metadata:
            report.volume_ratio = metadata['volume_ratio']

    def create_report_from_dict(
        self,
        signal_dict: Dict,
        strategy_results: Optional[List] = None,
        ml_validation_result: Optional[Any] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        position_size: Optional[float] = None
    ) -> TradeReport:
        """
        Создать отчёт из словаря сигнала (entry_signal).

        Args:
            signal_dict: Словарь с данными сигнала
            strategy_results: Результаты от стратегий (List[dict])
            ml_validation_result: Результат ML валидации (dict)
            sl_price: Цена Stop Loss
            tp_price: Цена Take Profit
            position_size: Размер позиции в USDT

        Returns:
            TradeReport готовый к записи
        """
        # Извлекаем данные из словаря
        symbol = signal_dict.get('symbol', 'UNKNOWN')
        signal_type = signal_dict.get('signal_type', 'UNKNOWN')
        price = signal_dict.get('price', 0.0)
        timestamp_ms = signal_dict.get('timestamp', 0)
        confidence = signal_dict.get('confidence', 0.0)
        reason = signal_dict.get('reason', 'No reason provided')

        # Создаём базовый отчёт
        report = TradeReport(
            symbol=symbol,
            direction=signal_type,
            price=price,
            timestamp=datetime.fromtimestamp(timestamp_ms / 1000) if timestamp_ms else datetime.now(),
            final_confidence=confidence,
            final_reason=reason,
            stop_loss=sl_price,
            take_profit=tp_price,
            position_size_usdt=position_size
        )

        # Рассчитываем R/R ratio
        if sl_price and tp_price and price:
            if signal_type == 'BUY':
                risk = abs(price - sl_price)
                reward = abs(tp_price - price)
            else:
                risk = abs(sl_price - price)
                reward = abs(price - tp_price)

            if risk > 0:
                report.risk_reward_ratio = reward / risk

        # Обрабатываем стратегии
        if strategy_results:
            report.strategies = self._extract_strategy_indicators(strategy_results)
            report.strategies_agreeing = len([s for s in report.strategies if s.signal_type])
            report.strategies_disagreeing = len(report.strategies) - report.strategies_agreeing

        # Обрабатываем ML результаты
        if ml_validation_result:
            self._extract_ml_data(report, ml_validation_result)

        # Обрабатываем metadata из signal_dict (если есть вложенные данные)
        self._extract_metadata(report, signal_dict)

        return report


# Глобальный экземпляр репортера
trade_reporter = TradeReporter()
