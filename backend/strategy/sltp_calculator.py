"""
Унифицированный калькулятор Stop Loss и Take Profit.

АРХИТЕКТУРА:
    1. ML-based calculation (primary)
    2. ATR-based calculation (fallback)
    3. Fixed percentage (emergency fallback)

ОГРАНИЧЕНИЯ:
    - Max SL: 3% (configurable)
    - Min R/R: 2.0:1
    - SL всегда учитывает slippage buffer

Путь: backend/strategy/sltp_calculator.py
"""
from typing import Optional
import numpy as np

from core.logger import get_logger
from core.exceptions import RiskManagementError
from config import settings
from models.signal import TradingSignal, SignalType
from strategy.risk_models import SLTPCalculation, MarketRegime

logger = get_logger(__name__)


class UnifiedSLTPCalculator:
    """
    Единый калькулятор SL/TP с ML и fallback механизмами.

    Приоритеты:
    1. ML-based: Использует predicted_mae и predicted_return
    2. ATR-based: Адаптивный расчет на основе волатильности
    3. Fixed: Фиксированные проценты (emergency)
    """

    def __init__(self):
        """Инициализация калькулятора."""
        self.max_sl_percent = settings.SLTP_MAX_STOP_LOSS_PERCENT / 100
        self.min_rr_ratio = settings.SLTP_MIN_RISK_REWARD
        self.atr_sl_mult = settings.SLTP_ATR_MULTIPLIER_SL
        self.atr_tp_mult = settings.SLTP_ATR_MULTIPLIER_TP
        self.ml_fallback_enabled = settings.SLTP_ML_FALLBACK_ENABLED

        logger.info(
            f"UnifiedSLTPCalculator инициализирован: "
            f"max_sl={self.max_sl_percent:.1%}, "
            f"min_rr={self.min_rr_ratio:.1f}, "
            f"ml_fallback={self.ml_fallback_enabled}"
        )

    def calculate(
        self,
        signal: TradingSignal,
        entry_price: float,
        ml_result: Optional[dict] = None,
        atr: Optional[float] = None,
        market_regime: Optional[MarketRegime] = None
    ) -> SLTPCalculation:
        """
        Расчет SL/TP с автоматическим выбором метода.

        Args:
            signal: Торговый сигнал
            entry_price: Цена входа
            ml_result: ML предсказания (optional)
            atr: Average True Range (optional)
            market_regime: Режим рынка (optional)

        Returns:
            SLTPCalculation: Результат расчета

        Raises:
            RiskManagementError: Если signal_type не BUY или SELL
        """
        # Валидация signal_type
        if signal.signal_type not in [SignalType.BUY, SignalType.SELL]:
            error_msg = (
                f"{signal.symbol} | Invalid signal_type для расчета SL/TP: "
                f"{signal.signal_type}. Ожидается BUY или SELL."
            )
            logger.error(error_msg)
            raise RiskManagementError(error_msg)

        # Явное определение направления позиции
        if signal.signal_type == SignalType.BUY:
            position_side = "long"
        elif signal.signal_type == SignalType.SELL:
            position_side = "short"
        else:
            # Этот блок никогда не выполнится из-за проверки выше,
            # но добавлен для полноты и безопасности типов
            raise RiskManagementError(
                f"Неподдерживаемый signal_type: {signal.signal_type}"
            )

        # ПРИОРИТЕТ 1: ML-based calculation
        if ml_result and self._validate_ml_result(ml_result):
            try:
                result = self._calculate_ml_based(
                    entry_price, position_side, ml_result, market_regime
                )

                # Валидация результата
                if self._validate_sltp(result, entry_price, position_side):
                    logger.info(
                        f"{signal.symbol} | SL/TP calculated: ML-based | "
                        f"SL={result.stop_loss:.2f} ({self._calc_percent(entry_price, result.stop_loss):.2%}), "
                        f"TP={result.take_profit:.2f} ({self._calc_percent(entry_price, result.take_profit):.2%}), "
                        f"R/R={result.risk_reward_ratio:.2f}"
                    )
                    return result
                else:
                    logger.warning(
                        f"{signal.symbol} | ML-based SL/TP validation failed, "
                        f"falling back to ATR"
                    )
            except Exception as e:
                logger.error(f"{signal.symbol} | ML-based calculation error: {e}")

        # ПРИОРИТЕТ 2: ATR-based calculation
        if atr and self.ml_fallback_enabled:
            try:
                result = self._calculate_atr_based(
                    entry_price, position_side, atr, market_regime
                )

                if self._validate_sltp(result, entry_price, position_side):
                    logger.info(
                        f"{signal.symbol} | SL/TP calculated: ATR-based | "
                        f"SL={result.stop_loss:.2f}, TP={result.take_profit:.2f}"
                    )
                    return result
            except Exception as e:
                logger.error(f"{signal.symbol} | ATR-based calculation error: {e}")

        # ПРИОРИТЕТ 3: Fixed fallback
        logger.warning(
            f"{signal.symbol} | Using FIXED fallback for SL/TP calculation"
        )
        return self._calculate_fixed(entry_price, position_side)

    def _calculate_ml_based(
        self,
        entry_price: float,
        position_side: str,
        ml_result: dict,
        market_regime: Optional[MarketRegime]
    ) -> SLTPCalculation:
        """
        ML-based расчет SL/TP.

        Использует:
            - predicted_mae (Maximum Adverse Excursion)
            - predicted_return
            - confidence
            - market_regime
        """
        # Извлекаем ML предсказания
        predicted_mae = ml_result.get('predicted_mae', 0.012)  # Default -1.2%
        predicted_return = ml_result.get('predicted_return', 0.025)  # Default +2.5%
        confidence = ml_result.get('confidence', 0.75)

        # Корректировка на уверенность
        # Высокая уверенность → можем держать дольше
        confidence_multiplier = self._get_confidence_multiplier(confidence)

        # Корректировка на market regime
        regime_adjustment = self._get_regime_adjustment(market_regime, position_side)

        # Расчет SL
        sl_distance = predicted_mae * regime_adjustment['sl_mult']
        sl_distance = min(sl_distance, self.max_sl_percent)  # Cap at max

        # Добавляем buffer для slippage (0.2%)
        slippage_buffer = 0.002
        sl_distance += slippage_buffer

        if position_side == "long":
            stop_loss = entry_price * (1 - sl_distance)
        else:
            stop_loss = entry_price * (1 + sl_distance)

        # Расчет TP
        tp_distance = predicted_return * confidence_multiplier * regime_adjustment['tp_mult']

        if position_side == "long":
            take_profit = entry_price * (1 + tp_distance)
        else:
            take_profit = entry_price * (1 - tp_distance)

        # Проверка R/R
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk

        # Если R/R < min, корректируем TP
        if rr_ratio < self.min_rr_ratio:
            if position_side == "long":
                take_profit = entry_price + (risk * self.min_rr_ratio)
            else:
                take_profit = entry_price - (risk * self.min_rr_ratio)

            rr_ratio = self.min_rr_ratio
            logger.debug(
                f"R/R adjusted to minimum {self.min_rr_ratio:.1f} | "
                f"TP adjusted to {take_profit:.2f}"
            )

        # Trailing start при достижении прибыли
        trailing_start = tp_distance * 0.6  # 60% до TP

        return SLTPCalculation(
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr_ratio,
            trailing_start_profit=trailing_start,
            calculation_method="ml",
            reasoning={
                "predicted_mae": predicted_mae,
                "predicted_return": predicted_return,
                "confidence": confidence,
                "confidence_multiplier": confidence_multiplier,
                "regime_adjustment": regime_adjustment,
                "sl_distance_percent": sl_distance,
                "tp_distance_percent": tp_distance
            },
            confidence=confidence
        )

    def _calculate_atr_based(
        self,
        entry_price: float,
        position_side: str,
        atr: float,
        market_regime: Optional[MarketRegime]
    ) -> SLTPCalculation:
        """
        ATR-based расчет (адаптивный к волатильности).

        Args:
            entry_price: Цена входа
            position_side: Направление позиции
            atr: Average True Range
            market_regime: Режим рынка
        """
        # Корректировка множителей на основе режима
        regime_adjustment = self._get_regime_adjustment(market_regime, position_side)

        # Расчет дистанций
        sl_distance_price = atr * self.atr_sl_mult * regime_adjustment['sl_mult']
        tp_distance_price = atr * self.atr_tp_mult * regime_adjustment['tp_mult']

        # Проверка max SL
        sl_distance_percent = sl_distance_price / entry_price
        if sl_distance_percent > self.max_sl_percent:
            sl_distance_price = entry_price * self.max_sl_percent
            logger.warning(
                f"ATR-based SL exceeds max ({sl_distance_percent:.2%} > {self.max_sl_percent:.2%}), "
                f"capped at {self.max_sl_percent:.2%}"
            )

        # Расчет уровней
        if position_side == "long":
            stop_loss = entry_price - sl_distance_price
            take_profit = entry_price + tp_distance_price
        else:
            stop_loss = entry_price + sl_distance_price
            take_profit = entry_price - tp_distance_price

        # R/R ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk

        # Trailing start
        trailing_start = (tp_distance_price / entry_price) * 0.5

        return SLTPCalculation(
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr_ratio,
            trailing_start_profit=trailing_start,
            calculation_method="atr",
            reasoning={
                "atr": atr,
                "atr_sl_multiplier": self.atr_sl_mult * regime_adjustment['sl_mult'],
                "atr_tp_multiplier": self.atr_tp_mult * regime_adjustment['tp_mult'],
                "regime_adjustment": regime_adjustment
            },
            confidence=0.7  # Средняя уверенность для ATR
        )

    def _calculate_fixed(
        self,
        entry_price: float,
        position_side: str
    ) -> SLTPCalculation:
        """
        Fixed percentage расчет (emergency fallback).

        УЛУЧШЕНИЯ:
        - Адаптация для монет с очень маленькой ценой
        - Гарантия минимального расстояния SL/TP
        - Проверка на деление на ноль

        Args:
            entry_price: Цена входа
            position_side: "long" или "short"

        Returns:
            SLTPCalculation с гарантированным R/R
        """
        # ==========================================
        # ШАГ 1: ОПРЕДЕЛЕНИЕ БАЗОВЫХ ПРОЦЕНТОВ
        # ==========================================
        # Используем максимум между фиксированным процентом и долей от max_sl
        # Это важно для маленьких цен, где округление критично
        base_sl_pct = 0.02  # 2% базовый SL

        # Для очень маленьких цен (< $0.01) используем больший процент
        if entry_price < 0.01:
            base_sl_pct = 0.05  # 5% для микрокапов
            logger.debug(
                f"Очень маленькая цена ({entry_price:.8f}), "
                f"используем расширенный SL: {base_sl_pct:.1%}"
            )
        elif entry_price < 0.1:
            base_sl_pct = 0.03  # 3% для маленьких цен
            logger.debug(
                f"Маленькая цена ({entry_price:.8f}), "
                f"используем расширенный SL: {base_sl_pct:.1%}"
            )

        # Не превышаем максимальный SL
        sl_percent = min(base_sl_pct, self.max_sl_percent)

        # TP основан на минимальном R/R
        tp_percent = sl_percent * self.min_rr_ratio

        logger.debug(
            f"Fixed расчет: entry=${entry_price:.8f}, "
            f"sl_pct={sl_percent:.2%}, "
            f"tp_pct={tp_percent:.2%}, "
            f"min_rr={self.min_rr_ratio:.1f}"
        )

        # ==========================================
        # ШАГ 2: РАСЧЕТ АБСОЛЮТНЫХ УРОВНЕЙ
        # ==========================================
        if position_side == "long":
            stop_loss = entry_price * (1 - sl_percent)
            take_profit = entry_price * (1 + tp_percent)
        else:  # short
            stop_loss = entry_price * (1 + sl_percent)
            take_profit = entry_price * (1 - tp_percent)

        # ==========================================
        # ШАГ 3: КРИТИЧЕСКАЯ ВАЛИДАЦИЯ
        # ==========================================
        # Проверяем что SL реально отличается от entry
        sl_distance = abs(entry_price - stop_loss)
        min_distance = entry_price * 0.001  # Минимум 0.1% расстояние

        if sl_distance < min_distance:
            logger.warning(
                f"Fixed SL слишком близко к entry: "
                f"distance={sl_distance:.8f} ({sl_distance / entry_price:.3%}), "
                f"min_required={min_distance:.8f} ({min_distance / entry_price:.3%}). "
                f"Увеличиваем до 1%."
            )

            # Увеличиваем до 1% минимум
            sl_percent = 0.01
            tp_percent = sl_percent * self.min_rr_ratio

            if position_side == "long":
                stop_loss = entry_price * (1 - sl_percent)
                take_profit = entry_price * (1 + tp_percent)
            else:
                stop_loss = entry_price * (1 + sl_percent)
                take_profit = entry_price * (1 - tp_percent)

            # Пересчитываем расстояние
            sl_distance = abs(entry_price - stop_loss)

        # Проверяем что TP реально отличается от entry
        tp_distance = abs(take_profit - entry_price)

        if tp_distance < min_distance:
            logger.warning(
                f"Fixed TP слишком близко к entry: "
                f"distance={tp_distance:.8f}. "
                f"Корректируем на основе SL."
            )

            # TP = entry + (SL_distance * min_RR)
            if position_side == "long":
                take_profit = entry_price + (sl_distance * self.min_rr_ratio)
            else:
                take_profit = entry_price - (sl_distance * self.min_rr_ratio)

            tp_distance = abs(take_profit - entry_price)

        # ==========================================
        # ШАГ 4: ФИНАЛЬНАЯ ПРОВЕРКА R/R
        # ==========================================
        # Проверка на деление на ноль (не должно случиться после валидаций выше)
        if sl_distance == 0:
            logger.error(
                f"❌ КРИТИЧЕСКАЯ ОШИБКА в _calculate_fixed: "
                f"sl_distance = 0 после всех корректировок! "
                f"entry={entry_price:.8f}, SL={stop_loss:.8f}"
            )
            # Экстренная корректировка
            sl_distance = entry_price * 0.01  # 1% принудительно
            if position_side == "long":
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + (sl_distance * self.min_rr_ratio)
            else:
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - (sl_distance * self.min_rr_ratio)
            tp_distance = abs(take_profit - entry_price)

        # Вычисляем финальный R/R
        final_rr = tp_distance / sl_distance

        # ==========================================
        # ШАГ 5: ЛОГИРОВАНИЕ И ВОЗВРАТ
        # ==========================================
        logger.info(
            f"✓ Fixed SL/TP: "
            f"SL=${stop_loss:.8f} ({sl_distance / entry_price:.2%}), "
            f"TP=${take_profit:.8f} ({tp_distance / entry_price:.2%}), "
            f"R/R={final_rr:.2f}"
        )

        return SLTPCalculation(
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=final_rr,
            trailing_start_profit=tp_percent * 0.5,
            calculation_method="fixed",
            reasoning={
                "sl_percent": sl_distance / entry_price,
                "tp_percent": tp_distance / entry_price,
                "base_sl_percent": base_sl_pct,
                "adjustments_applied": sl_distance < min_distance or tp_distance < min_distance,
                "note": "Emergency fallback with price-adaptive parameters"
            },
            confidence=0.5  # Низкая уверенность для fixed
        )

    def _get_confidence_multiplier(self, confidence: float) -> float:
        """Множитель на основе уверенности ML."""
        if confidence > 0.9:
            return 1.5  # Очень уверены → можем держать дольше
        elif confidence > 0.8:
            return 1.2
        elif confidence > 0.7:
            return 1.0
        else:
            return 0.8  # Слабая уверенность → консервативнее

    def _get_regime_adjustment(
        self,
        regime: Optional[MarketRegime],
        position_side: str
    ) -> dict:
        """
        Корректировка SL/TP на основе market regime.
        """
        if not regime:
            return {'sl_mult': 1.0, 'tp_mult': 1.0}

        # STRONG TREND: Шире стопы, дальше цели
        if regime == MarketRegime.STRONG_TREND:
            return {'sl_mult': 1.3, 'tp_mult': 1.8}

        # MILD TREND: Норм
        elif regime == MarketRegime.MILD_TREND:
            return {'sl_mult': 1.0, 'tp_mult': 1.3}

        # RANGING: Ближе стопы и цели
        elif regime == MarketRegime.RANGING:
            return {'sl_mult': 0.8, 'tp_mult': 0.9}

        # HIGH VOLATILITY: Шире стопы
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return {'sl_mult': 1.5, 'tp_mult': 1.5}

        # DISTRIBUTION (продажа крупными): Не входить в long
        elif regime == MarketRegime.DISTRIBUTION:
            if position_side == "long":
                return {'sl_mult': 0.7, 'tp_mult': 0.7}  # Очень консервативно
            else:
                return {'sl_mult': 1.2, 'tp_mult': 1.6}  # Short выгоден

        # ACCUMULATION (покупка крупными): Не входить в short
        elif regime == MarketRegime.ACCUMULATION:
            if position_side == "short":
                return {'sl_mult': 0.7, 'tp_mult': 0.7}
            else:
                return {'sl_mult': 1.2, 'tp_mult': 1.6}

        return {'sl_mult': 1.0, 'tp_mult': 1.0}

    def _validate_ml_result(self, ml_result: dict) -> bool:
        """Проверка валидности ML результата."""
        required_keys = ['predicted_mae', 'predicted_return', 'confidence']
        return all(key in ml_result for key in required_keys)

    def _validate_sltp(
        self,
        result: SLTPCalculation,
        entry_price: float,
        position_side: str
    ) -> bool:
        """Валидация рассчитанных SL/TP."""
        # Проверка, что SL в правильном направлении
        if position_side == "long" and result.stop_loss >= entry_price:
            return False
        if position_side == "short" and result.stop_loss <= entry_price:
            return False

        # Проверка max SL
        sl_percent = abs((entry_price - result.stop_loss) / entry_price)
        if sl_percent > self.max_sl_percent:
            return False

        # Проверка R/R
        if result.risk_reward_ratio < self.min_rr_ratio:
            return False

        return True

    def _calc_percent(self, entry: float, level: float) -> float:
        """Расчет процентного изменения."""
        return (level - entry) / entry


# Глобальный экземпляр
sltp_calculator = UnifiedSLTPCalculator()