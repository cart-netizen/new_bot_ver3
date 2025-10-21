"""
ML-Enhanced Risk Manager с полной интеграцией ML предсказаний.

ИСПРАВЛЕНИЯ:
1. Исправлена работа с feature_vector (поддержка Dict и FeatureVector)
2. Исправлена инициализация ml_size_mult

4. Добавлена полная интеграция с расширенным ValidationResult

ИНТЕГРАЦИЯ С ML:
- Использует ML predictions для:
  * Dynamic position sizing
  * Optimal SL/TP placement
  * Market regime filtering
  * Manipulation detection

FALLBACK СИСТЕМА:
- При недоступности ML автоматически переключается на:
  * Adaptive Risk Calculator (sizing)
  * ATR-based SL/TP
  * Базовые проверки риска

Путь: backend/strategy/risk_manager_ml_enhanced.py
"""
from typing import Optional, Tuple, Dict, Union
from datetime import datetime

from core.logger import get_logger
from config import settings
from models.signal import TradingSignal, SignalType
from strategy.risk_manager import RiskManager
from strategy.sltp_calculator import sltp_calculator
from strategy.adaptive_risk_calculator import adaptive_risk_calculator
from strategy.correlation_manager import correlation_manager
from strategy.daily_loss_killer import daily_loss_killer
from strategy.risk_models import MLRiskAdjustments, MarketRegime, SLTPCalculation
from ml_engine.integration.ml_signal_validator import MLSignalValidator
from ml_engine.features import FeatureVector
from utils.helpers import safe_enum_value

logger = get_logger(__name__)


class RiskManagerMLEnhanced(RiskManager):
  """
  Расширенный Risk Manager с ML интеграцией.

  Наследует все базовые функции RiskManager и добавляет:
  - ML-enhanced валидацию сигналов
  - ML-based position sizing
  - ML-based SL/TP calculation
  - Manipulation detection
  - Market regime compatibility checks
  """

  def __init__(
      self,
      ml_validator: Optional[MLSignalValidator] = None,
      default_leverage: int = 10,
      initial_balance: Optional[float] = None
  ):
    """
    Инициализация ML-Enhanced Risk Manager.

    Args:
        ml_validator: MLSignalValidator instance (optional)
        default_leverage: Кредитное плечо по умолчанию
        initial_balance: Начальный баланс
    """
    super().__init__(default_leverage, initial_balance)

    self.ml_validator = ml_validator

    # ML конфигурация из settings
    self.ml_enabled = settings.ML_RISK_INTEGRATION_ENABLED
    self.ml_min_confidence = settings.ML_MIN_CONFIDENCE_THRESHOLD
    self.ml_require_agreement = settings.ML_REQUIRE_AGREEMENT
    self.ml_position_sizing = settings.ML_POSITION_SIZING
    self.ml_sltp_calculation = settings.ML_SLTP_CALCULATION
    self.ml_manipulation_check = settings.ML_MANIPULATION_CHECK
    self.ml_regime_check = settings.ML_REGIME_CHECK

    # Статистика ML использования
    self.ml_stats = {
      'total_validations': 0,
      'ml_available': 0,
      'ml_used': 0,
      'ml_rejected': 0,
      'fallback_used': 0
    }

    logger.info(
      f"RiskManagerMLEnhanced инициализирован: "
      f"ml_enabled={self.ml_enabled}, "
      f"ml_validator={'available' if ml_validator else 'not_set'}, "
      f"min_confidence={self.ml_min_confidence:.2f}"
    )

  async def validate_signal_ml_enhanced(
      self,
      signal: TradingSignal,
      balance: float,
      feature_vector: Optional[Union[Dict, FeatureVector]] = None
  ) -> Tuple[bool, Optional[str], Optional[MLRiskAdjustments]]:
    """
    ML-enhanced валидация сигнала с полной интеграцией ML данных.

    Pipeline:
    1. Daily Loss Killer check
    2. ML validation (если доступна)
    3. ML confidence check
    4. ML agreement check
    5. Manipulation detection
    6. Market regime compatibility
    7. SL/TP calculation (ML-based или fallback)
    8. Position sizing (ML-adjusted или fallback)
    9. Базовая валидация (лимиты, exposure)
    10. Correlation check

    Args:
        signal: Торговый сигнал
        balance: Доступный баланс
        feature_vector: ML признаки (Dict или FeatureVector, optional)

    Returns:
        Tuple[is_valid, rejection_reason, ml_adjustments]
    """
    self.ml_stats['total_validations'] += 1

    # ========================================
    # ШАГ 0: DAILY LOSS KILLER
    # ========================================
    is_allowed, shutdown_reason = daily_loss_killer.is_trading_allowed()
    if not is_allowed:
      logger.warning(
        f"{signal.symbol} | Trading blocked by Daily Loss Killer: {shutdown_reason}"
      )
      return False, shutdown_reason, None

    # ========================================
    # ШАГ 1: ПОЛУЧЕНИЕ ML РЕЗУЛЬТАТА
    # ========================================
    ml_result = None
    validation_result = None

    if self.ml_enabled and self.ml_validator and feature_vector:
      try:
        # ========================================
        # ИСПРАВЛЕНИЕ: Передаем feature_vector напрямую
        # MLSignalValidator теперь поддерживает оба типа
        # ========================================
        validation_result = await self.ml_validator.validate(
          signal,
          feature_vector
        )

        # Проверяем, что валидация успешна
        if validation_result and validation_result.validated:
          self.ml_stats['ml_available'] += 1

          # Формируем ml_result из ValidationResult
          ml_result = {
            'confidence': validation_result.ml_confidence or 0.0,
            'direction': validation_result.final_signal_type,
            'predicted_return': validation_result.ml_expected_return or 0.0,
            # ========================================
            # ИСПРАВЛЕНИЕ: Используем данные из ValidationResult
            # вместо заглушек
            # ========================================
            'predicted_mae': validation_result.predicted_mae or 0.015,
            'manipulation_risk': validation_result.manipulation_risk,
            'market_regime': validation_result.market_regime,
            'feature_quality': validation_result.feature_quality
          }

          logger.debug(
            f"{signal.symbol} | ML validation получена: "
            f"confidence={ml_result['confidence']:.2f}, "
            f"direction={ml_result['direction'].value}, "
            f"predicted_return={ml_result['predicted_return']:.3f}, "
            f"manipulation_risk={ml_result['manipulation_risk']:.3f}, "
            f"regime={ml_result['market_regime'].value if ml_result['market_regime'] else 'None'}"
          )

      except Exception as e:
        logger.warning(
          f"{signal.symbol} | ML validation failed: {e}",
          exc_info=True
        )
        ml_result = None
        validation_result = None

    # ========================================
    # ШАГ 2: ML ВАЛИДАЦИЯ (если доступна)
    # ========================================
    used_fallback = validation_result.used_fallback if validation_result else False

    if ml_result:
      # ========================================
      # ИСПРАВЛЕНИЕ: Проверяем used_fallback из ValidationResult
      # ========================================


      # 2.1 ML Confidence Check
      # ИСПРАВЛЕНИЕ: Пропускаем проверку для fallback режима
      if not used_fallback and ml_result['confidence'] < self.ml_min_confidence:
        reason = (
          f"ML confidence слишком низкий: "
          f"{ml_result['confidence']:.2f} < {self.ml_min_confidence:.2f}"
        )
        logger.warning(f"{signal.symbol} | {reason}")
        self.ml_stats['ml_rejected'] += 1
        return False, reason, None

      # Для fallback режима логируем но пропускаем
      if used_fallback:
        logger.info(
          f"{signal.symbol} | ML fallback режим: "
          f"пропускаем проверку ML confidence, "
          f"используем стратегию confidence={ml_result['confidence']:.2f}"
        )

      # 2.2 ML Agreement Check
      # ИСПРАВЛЕНИЕ: Пропускаем проверку для fallback режима
      if not used_fallback and self.ml_require_agreement:
        ml_direction = ml_result['direction']
        if ml_direction != signal.signal_type:
          reason = (
            f"ML не согласен со стратегией: "
            f"ML={ml_direction.value}, Strategy={signal.signal_type.value}"
          )
          logger.warning(f"{signal.symbol} | {reason}")
          self.ml_stats['ml_rejected'] += 1
          return False, reason, None

      # ========================================
      # 2.3 MANIPULATION DETECTION
      # ========================================
      # ИСПРАВЛЕНИЕ: Пропускаем для fallback режима
      if not used_fallback and self.ml_manipulation_check:
        manipulation_risk = ml_result['manipulation_risk']

        # Порог риска манипуляции из settings или дефолт
        max_manipulation_risk = getattr(
          settings,
          'ML_MAX_MANIPULATION_RISK',
          0.7
        )

        if manipulation_risk > max_manipulation_risk:
          reason = (
            f"Высокий риск манипуляции: "
            f"{manipulation_risk:.2f} > {max_manipulation_risk:.2f}"
          )
          logger.warning(f"{signal.symbol} | {reason}")
          self.ml_stats['ml_rejected'] += 1
          return False, reason, None

      # ========================================
      # 2.4 MARKET REGIME CHECK
      # ========================================
      # ИСПРАВЛЕНИЕ: Пропускаем для fallback режима
      if not used_fallback and self.ml_regime_check:
        regime = ml_result.get('market_regime')
        if regime:
          forbidden_regimes = self._get_forbidden_regimes(signal)
          if regime in forbidden_regimes:
            reason = (
              f"Несовместимый режим рынка: {regime.value}, "
              f"запрещенные: {[r.value for r in forbidden_regimes]}"
            )
            logger.warning(f"{signal.symbol} | {reason}")
            self.ml_stats['ml_rejected'] += 1
            return False, reason, None

      # Для fallback режима инкрементируем счетчик
      if used_fallback:
        self.ml_stats['fallback_used'] += 1
        logger.info(
          f"{signal.symbol} | ✅ ML fallback validation PASSED | "
          f"Using strategy signal with confidence={ml_result['confidence']:.2f}"
        )
      else:
        self.ml_stats['ml_used'] += 1
        logger.info(
          f"{signal.symbol} | ✅ ML validation PASSED | "
          f"confidence={ml_result['confidence']:.2f}"
        )

    # ========================================
    # ШАГ 3: SL/TP CALCULATION
    # ========================================
    if self.ml_sltp_calculation and ml_result and not used_fallback:
      # ML доступна - используем ML расчет
      try:
        sltp_calc = await self._calculate_ml_sltp(
          signal, ml_result, balance
        )
        logger.debug(
          f"{signal.symbol} | ML SL/TP использован: "
          f"SL={sltp_calc.stop_loss:.2f}, TP={sltp_calc.take_profit:.2f}"
        )
      except Exception as e:
        logger.warning(
          f"{signal.symbol} | ML SL/TP calculation error: {e}, "
          f"fallback to unified calculator"
        )
        # Fallback на unified calculator
        sltp_calc = sltp_calculator.calculate(
          signal=signal,
          entry_price=signal.price,
          ml_result=None,
          atr=None,
          market_regime=ml_result.get('market_regime') if ml_result else None
        )
    else:
      # Fallback режим или ML недоступна - используем ТОЛЬКО unified calculator
      logger.info(
        f"{signal.symbol} | Используем UnifiedSLTPCalculator "
        f"(fallback режим или ML недоступна)"
      )

      sltp_calc = sltp_calculator.calculate(
        signal=signal,
        entry_price=signal.price,
        ml_result=None,  # В fallback НЕ передаем ml_result
        atr=None,
        market_regime=ml_result.get('market_regime') if ml_result else None
      )

    logger.info(
      f"{signal.symbol} | SL/TP выбраны: "
      f"method={sltp_calc.calculation_method}, "
      f"SL={sltp_calc.stop_loss:.2f}, "
      f"TP={sltp_calc.take_profit:.2f}, "
      f"R/R={sltp_calc.risk_reward_ratio:.2f}"
    )

    # ========================================
    # ШАГ 4: POSITION SIZING
    # ========================================
    # ========================================
    # ИСПРАВЛЕНИЕ: Инициализируем ml_size_mult ПЕРЕД использованием
    # ========================================
    ml_size_mult = 1.0  # Дефолтное значение

    if ml_result and self.ml_position_sizing and not used_fallback:
      # ML-adjusted position sizing
      ml_size_mult = self._calculate_ml_size_multiplier(ml_result)

      logger.info(
        f"{signal.symbol} | ML size multiplier: {ml_size_mult:.2f}x, "
        f"factors: confidence={ml_result['confidence']:.2f}, "
        f"return={ml_result['predicted_return']:.3f}, "
        f"regime={ml_result['market_regime'].value if ml_result['market_regime'] else 'None'}"
      )
    else:
      # Fallback: Adaptive Risk Calculator с минимумом 1.0x
      from strategy.adaptive_risk_calculator import adaptive_risk_calculator

      adaptive_mult = adaptive_risk_calculator.calculate_risk_multiplier(
        signal.confidence
      )

      # КРИТИЧНО: Гарантируем минимум 1.0x в fallback режиме
      # чтобы избежать размера позиции ниже минимума
      ml_size_mult = max(1.0, adaptive_mult)

      logger.info(
        f"{signal.symbol} | Fallback adaptive size multiplier: "
        f"calculated={adaptive_mult:.2f}x, "
        f"final={ml_size_mult:.2f}x (минимум 1.0x для fallback)"
      )

      # Вычисляем базовый размер позиции
    base_position_size = self.calculate_position_size(
      signal=signal,
      available_balance=balance,
      stop_loss_price=sltp_calc.stop_loss,
      leverage=self.default_leverage,
      current_volatility=None,
      ml_confidence=ml_result['confidence'] if ml_result else None
    )

    # Применяем ML множитель
    position_size_usdt = base_position_size * ml_size_mult

    logger.info(
      f"{signal.symbol} | Position sizing: "
      f"base={base_position_size:.2f} USDT, "
      f"multiplier={ml_size_mult:.2f}x, "
      f"final={position_size_usdt:.2f} USDT"
    )

    # ========================================
    # ШАГ 5: БАЗОВАЯ ВАЛИДАЦИЯ
    # ========================================
    is_valid, reason = self.validate_signal(
      signal=signal,
      position_size_usdt=position_size_usdt,
      leverage=self.default_leverage
    )

    if not is_valid:
      logger.warning(
        f"{signal.symbol} | Базовая валидация failed: {reason}"
      )
      return False, reason, None

    # ========================================
    # ШАГ 6: CORRELATION CHECK
    # ========================================
    try:
      corr_allowed, corr_reason = correlation_manager.can_open_position(
        signal.symbol,
        position_size_usdt
      )

      if not corr_allowed:
        logger.warning(
          f"{signal.symbol} | Correlation check failed: {corr_reason}"
        )
        return False, corr_reason, None
    except Exception as e:
      logger.warning(
        f"{signal.symbol} | Correlation check error: {e}"
      )
      # Продолжаем без correlation check

    # ========================================
    # ШАГ 7: ФОРМИРОВАНИЕ ML ADJUSTMENTS
    # ========================================
    # ========================================
    # ИСПРАВЛЕНИЕ: ml_size_mult теперь всегда инициализирован
    # ========================================
    ml_adjustments = MLRiskAdjustments(
      position_size_multiplier=(
        ml_size_mult if ml_result and self.ml_position_sizing
        else 1.0
      ),
      stop_loss_price=sltp_calc.stop_loss,
      take_profit_price=sltp_calc.take_profit,
      ml_confidence=ml_result['confidence'] if ml_result else 0.0,
      expected_return=ml_result['predicted_return'] if ml_result else 0.0,
      market_regime=(
        ml_result.get('market_regime')
        if ml_result else MarketRegime.RANGING
      ),
      manipulation_risk_score=(
        ml_result.get('manipulation_risk', 0.0)
        if ml_result else 0.0
      ),
      feature_quality=(
        ml_result.get('feature_quality', 0.0)
        if ml_result else 0.0
      ),
      allow_entry=True,
      rejection_reason=None
    )

    if ml_result:
      self.ml_stats['ml_used'] += 1
    else:
      self.ml_stats['fallback_used'] += 1

    logger.info(
      f"{signal.symbol} | ✅ ML-enhanced validation PASSED | "
      f"ML={'used' if ml_result else 'fallback'}, "
      f"Confidence={ml_adjustments.ml_confidence:.2f}, "
      f"Size mult={ml_adjustments.position_size_multiplier:.2f}x, "
      f"Expected return={ml_adjustments.expected_return:.2%}, "
      f"Manipulation risk={ml_adjustments.manipulation_risk_score:.3f}, "
      f"Feature quality={ml_adjustments.feature_quality:.3f}"
    )

    return True, None, ml_adjustments

  def _calculate_ml_size_multiplier(self, ml_result: Dict) -> float:
    """
    Расчет множителя размера на основе ML факторов.

    Factors:
    - ML Confidence (0.7x - 2.0x)
    - Expected Return (0.8x - 1.5x)
    - Market Regime (0.7x - 1.3x)
    - Feature Quality (0.8x - 1.0x)
    - Manipulation Risk (0.5x - 1.0x)

    Args:
        ml_result: ML предсказание

    Returns:
        float: Множитель размера [0.5, 2.5]
    """
    confidence = ml_result['confidence']
    pred_return = ml_result['predicted_return']
    regime = ml_result.get('market_regime')
    quality = ml_result.get('feature_quality', 0.8)
    manipulation = ml_result.get('manipulation_risk', 0.0)

    # 1. Confidence multiplier (0.7x - 2.0x)
    if confidence > 0.9:
      conf_mult = 2.0
    elif confidence > 0.85:
      conf_mult = 1.7
    elif confidence > 0.8:
      conf_mult = 1.4
    elif confidence > 0.75:
      conf_mult = 1.2
    elif confidence > 0.7:
      conf_mult = 1.0
    else:
      conf_mult = 0.7

    # 2. Return multiplier (0.8x - 1.5x)
    if pred_return > 0.04:  # >4%
      ret_mult = 1.5
    elif pred_return > 0.03:  # >3%
      ret_mult = 1.3
    elif pred_return > 0.02:  # >2%
      ret_mult = 1.1
    elif pred_return > 0.01:  # >1%
      ret_mult = 1.0
    else:
      ret_mult = 0.8

    # 3. Regime multiplier (0.7x - 1.3x)
    if regime == MarketRegime.STRONG_TREND:
      regime_mult = 1.3
    elif regime == MarketRegime.MILD_TREND:
      regime_mult = 1.1
    elif regime == MarketRegime.RANGING:
      regime_mult = 0.8
    elif regime == MarketRegime.HIGH_VOLATILITY:
      regime_mult = 0.9
    elif regime == MarketRegime.DISTRIBUTION:
      regime_mult = 0.7
    elif regime == MarketRegime.ACCUMULATION:
      regime_mult = 1.2
    else:
      regime_mult = 1.0

    # 4. Feature quality multiplier (0.8x - 1.0x)
    if quality > 0.95:
      quality_mult = 1.0
    elif quality > 0.9:
      quality_mult = 0.98
    elif quality > 0.85:
      quality_mult = 0.95
    elif quality > 0.8:
      quality_mult = 0.9
    else:
      quality_mult = 0.8

    # 5. Manipulation risk multiplier (0.5x - 1.0x)
    if manipulation < 0.1:
      manip_mult = 1.0
    elif manipulation < 0.3:
      manip_mult = 0.9
    elif manipulation < 0.5:
      manip_mult = 0.7
    else:
      manip_mult = 0.5

    # Комбинируем все множители
    total_mult = conf_mult * ret_mult * regime_mult * quality_mult * manip_mult

    # Ограничиваем диапазон
    total_mult = max(0.5, min(2.5, total_mult))

    logger.debug(
      f"ML size multiplier breakdown: "
      f"conf={conf_mult:.2f}, ret={ret_mult:.2f}, "
      f"regime={regime_mult:.2f}, quality={quality_mult:.2f}, "
      f"manip={manip_mult:.2f}, total={total_mult:.2f}"
    )

    return total_mult

  async def _calculate_ml_sltp(
        self,
        signal: TradingSignal,
        ml_result: dict,
        balance: float
    ) -> SLTPCalculation:
      """
      ML-based расчет Stop Loss и Take Profit.

      Использует:
      - Predicted MAE для установки SL
      - Expected Return для установки TP
      - Market Regime для корректировки

      ИСПРАВЛЕНИЯ:
      - Проверка деления на ноль (sl_distance == 0)
      - Гарантия минимального R/R >= 2.0
      - Fallback на fixed расчет при ошибках

      Args:
          signal: Торговый сигнал
          ml_result: ML предсказание
          balance: Баланс

      Returns:
          SLTPCalculation

      Raises:
          ValueError: Если signal_type некорректен
      """
      # ==========================================
      # ШАГ 1: ИЗВЛЕЧЕНИЕ ML ДАННЫХ
      # ==========================================
      entry_price = signal.price
      predicted_mae = ml_result.get('predicted_mae', 0.015)  # Default 1.5%
      expected_return = ml_result.get('predicted_return', 0.02)  # Default 2%
      regime = ml_result.get('market_regime')
      confidence = ml_result.get('confidence', 0.75)

      logger.debug(
        f"{signal.symbol} | ML SL/TP расчет: "
        f"entry=${entry_price:.8f}, "
        f"mae={predicted_mae:.4f}, "
        f"return={expected_return:.4f}, "
        f"regime={regime.value if regime else 'None'}, "
        f"confidence={confidence:.2f}"
      )

      # ==========================================
      # ШАГ 2: ОПРЕДЕЛЕНИЕ НАПРАВЛЕНИЯ ПОЗИЦИИ
      # ==========================================
      if signal.signal_type == SignalType.BUY:
        position_side = "long"
      elif signal.signal_type == SignalType.SELL:
        position_side = "short"
      else:
        error_msg = (
          f"{signal.symbol} | Некорректный signal_type для ML SL/TP: "
          f"{signal.signal_type}. Используем fixed fallback."
        )
        logger.error(error_msg)
        # Fallback на fixed расчет
        return sltp_calculator._calculate_fixed(entry_price, "long")

      # ==========================================
      # ШАГ 3: БАЗОВЫЙ РАСЧЕТ SL ДИСТАНЦИИ
      # ==========================================
      # SL на основе predicted MAE с буфером
      sl_distance_pct = predicted_mae * 1.5  # 1.5x MAE для буфера

      # Регулируем на основе режима рынка
      if regime == MarketRegime.HIGH_VOLATILITY:
        sl_distance_pct *= 1.3  # Шире SL при волатильности
        logger.debug(f"{signal.symbol} | HIGH_VOLATILITY: SL расширен на 30%")
      elif regime == MarketRegime.STRONG_TREND:
        sl_distance_pct *= 0.9  # Уже SL в тренде
        logger.debug(f"{signal.symbol} | STRONG_TREND: SL сужен на 10%")
      elif regime == MarketRegime.RANGING:
        sl_distance_pct *= 0.8  # Ближе SL во флэте
        logger.debug(f"{signal.symbol} | RANGING: SL сужен на 20%")

      # Ограничиваем максимальным SL (3% по умолчанию)
      max_sl_pct = settings.SLTP_MAX_STOP_LOSS_PERCENT / 100
      if sl_distance_pct > max_sl_pct:
        logger.warning(
          f"{signal.symbol} | ML SL {sl_distance_pct:.2%} превышает max {max_sl_pct:.2%}, "
          f"ограничиваем"
        )
        sl_distance_pct = max_sl_pct

      # ==========================================
      # ШАГ 4: БАЗОВЫЙ РАСЧЕТ TP ДИСТАНЦИИ
      # ==========================================
      # TP на основе expected return
      tp_distance_pct = expected_return

      # Регулируем на основе режима рынка
      if regime == MarketRegime.RANGING:
        tp_distance_pct *= 0.8  # Ближе TP в флэте
        logger.debug(f"{signal.symbol} | RANGING: TP сужен на 20%")
      elif regime == MarketRegime.STRONG_TREND:
        tp_distance_pct *= 1.2  # Дальше TP в тренде
        logger.debug(f"{signal.symbol} | STRONG_TREND: TP расширен на 20%")
      elif regime == MarketRegime.HIGH_VOLATILITY:
        tp_distance_pct *= 1.1  # Немного дальше TP при волатильности
        logger.debug(f"{signal.symbol} | HIGH_VOLATILITY: TP расширен на 10%")

      # Регулируем на основе confidence
      if confidence > 0.85:
        tp_distance_pct *= 1.2  # Дальше TP при высокой уверенности
        logger.debug(f"{signal.symbol} | High confidence: TP расширен на 20%")
      elif confidence < 0.7:
        tp_distance_pct *= 0.9  # Ближе TP при низкой уверенности
        logger.debug(f"{signal.symbol} | Low confidence: TP сужен на 10%")

      # ==========================================
      # ШАГ 5: РАСЧЕТ АБСОЛЮТНЫХ УРОВНЕЙ SL/TP
      # ==========================================
      if position_side == "long":
        stop_loss = entry_price * (1 - sl_distance_pct)
        take_profit = entry_price * (1 + tp_distance_pct)
      else:  # short
        stop_loss = entry_price * (1 + sl_distance_pct)
        take_profit = entry_price * (1 - tp_distance_pct)

      # ==========================================
      # ШАГ 6: КРИТИЧЕСКАЯ ВАЛИДАЦИЯ И КОРРЕКЦИЯ R/R
      # ==========================================
      # Вычисляем дистанции
      sl_distance = abs(entry_price - stop_loss)
      tp_distance = abs(take_profit - entry_price)

      # КРИТИЧНО: Проверка на деление на ноль
      if sl_distance == 0 or sl_distance < entry_price * 0.0001:  # < 0.01%
        logger.error(
          f"{signal.symbol} | ❌ КРИТИЧЕСКАЯ ОШИБКА: SL дистанция слишком мала или = 0! "
          f"entry={entry_price:.8f}, SL={stop_loss:.8f}, distance={sl_distance:.8f}. "
          f"Используем FIXED fallback."
        )
        # Fallback на fixed расчет
        return sltp_calculator._calculate_fixed(entry_price, position_side)

      # Вычисляем R/R
      risk_reward = tp_distance / sl_distance

      # КРИТИЧНО: Гарантируем минимальный R/R
      min_rr = settings.SLTP_MIN_RISK_REWARD  # Обычно 2.0 или 4.0

      if risk_reward <= 0:
        logger.error(
          f"{signal.symbol} | ❌ КРИТИЧЕСКАЯ ОШИБКА: R/R <= 0! "
          f"risk_reward={risk_reward:.4f}, "
          f"sl_distance={sl_distance:.8f}, "
          f"tp_distance={tp_distance:.8f}. "
          f"Используем FIXED fallback."
        )
        # Fallback на fixed расчет
        return sltp_calculator._calculate_fixed(entry_price, position_side)

      if risk_reward < min_rr:
        logger.warning(
          f"{signal.symbol} | ML R/R слишком низкий: {risk_reward:.2f} < {min_rr:.2f}. "
          f"Корректируем TP для достижения min R/R."
        )

        # Корректируем TP для достижения минимального R/R
        if position_side == "long":
          take_profit = entry_price + (sl_distance * min_rr)
        else:  # short
          take_profit = entry_price - (sl_distance * min_rr)

        # Пересчитываем
        tp_distance = abs(take_profit - entry_price)
        risk_reward = min_rr

        logger.info(
          f"{signal.symbol} | ✓ TP скорректирован: "
          f"new_tp={take_profit:.8f}, "
          f"new_rr={risk_reward:.2f}"
        )

      # ==========================================
      # ШАГ 7: РАСЧЕТ TRAILING START
      # ==========================================
      # Начинаем trailing при достижении 40% от TP
      trailing_start_profit = 0.4

      # ==========================================
      # ШАГ 8: ФОРМИРОВАНИЕ РЕЗУЛЬТАТА
      # ==========================================
      logger.info(
        f"{signal.symbol} | ✓ ML SL/TP рассчитаны: "
        f"SL=${stop_loss:.8f} ({sl_distance_pct:.2%}), "
        f"TP=${take_profit:.8f} ({tp_distance_pct:.2%}), "
        f"R/R={risk_reward:.2f}, "
        f"confidence={confidence:.2f}"
      )

      return SLTPCalculation(
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward_ratio=risk_reward,  # ГАРАНТИРОВАННО >= min_rr
        trailing_start_profit=trailing_start_profit,
        calculation_method="ml",
        reasoning={
          "predicted_mae": predicted_mae,
          "expected_return": expected_return,
          "market_regime": regime.value if regime else None,
          "sl_distance_pct": sl_distance_pct,
          "tp_distance_pct": tp_distance_pct,
          "confidence": confidence,
          "risk_reward": risk_reward,
          "adjustments": {
            "regime_applied": regime is not None,
            "confidence_boost": confidence > 0.85,
            "rr_corrected": risk_reward == min_rr
          }
        },
        confidence=confidence
      )

  def _get_forbidden_regimes(self, signal: TradingSignal) -> list:
    """
    Определяет запрещенные режимы рынка для данной стратегии.

    Args:
        signal: Торговый сигнал

    Returns:
        list: Список запрещенных MarketRegime
    """
    # Можно расширять на основе типа стратегии
    # Пока базовые правила:

    forbidden = []

    # Высокая волатильность - рискованно для всех
    # (но не запрещаем полностью, просто снижаем размер через multiplier)

    # Можно добавить специфичные правила:
    # if signal.source == SignalSource.MOMENTUM:
    #     forbidden.append(MarketRegime.RANGING)

    return forbidden

  def get_ml_stats(self) -> Dict:
    """
    Получить статистику ML использования.

    Returns:
        Dict со статистикой ML валидаций
    """
    if self.ml_stats['total_validations'] == 0:
      return {
        'ml_enabled': self.ml_enabled,
        'total_validations': 0,
        'ml_usage_rate': 0.0,
        'ml_rejection_rate': 0.0,
        'fallback_rate': 0.0
      }

    total = self.ml_stats['total_validations']

    return {
      'ml_enabled': self.ml_enabled,
      'total_validations': total,
      'ml_available': self.ml_stats['ml_available'],
      'ml_used': self.ml_stats['ml_used'],
      'ml_rejected': self.ml_stats['ml_rejected'],
      'fallback_used': self.ml_stats['fallback_used'],
      'ml_usage_rate': (self.ml_stats['ml_used'] / total) * 100,
      'ml_rejection_rate': (self.ml_stats['ml_rejected'] / total) * 100,
      'fallback_rate': (self.ml_stats['fallback_used'] / total) * 100
    }


# Глобальный экземпляр (будет создан в main.py)
risk_manager_ml_enhanced: Optional[RiskManagerMLEnhanced] = None