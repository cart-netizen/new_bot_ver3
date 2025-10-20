"""
ML-Enhanced Risk Manager с полной интеграцией ML предсказаний.

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
from typing import Optional, Tuple, Dict
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
      feature_vector: Optional[Dict] = None
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
        feature_vector: ML признаки (optional)

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

    if self.ml_enabled and self.ml_validator and feature_vector:
      try:
        ml_result = await self._get_ml_prediction(signal, feature_vector)

        if ml_result:
          self.ml_stats['ml_available'] += 1
          logger.debug(
            f"{signal.symbol} | ML prediction получен: "
            f"confidence={ml_result['confidence']:.2f}, "
            f"direction={ml_result['direction'].value}, "
            f"predicted_return={ml_result['predicted_return']:.3f}"
          )

      except Exception as e:
        logger.warning(
          f"{signal.symbol} | ML prediction failed: {e}",
          exc_info=True
        )
        ml_result = None

    # ========================================
    # ШАГ 2: ML ВАЛИДАЦИЯ (если доступна)
    # ========================================
    if ml_result:
      # 2.1 ML Confidence Check
      if ml_result['confidence'] < self.ml_min_confidence:
        reason = (
          f"ML confidence слишком низкий: "
          f"{ml_result['confidence']:.2f} < {self.ml_min_confidence:.2f}"
        )
        logger.warning(f"{signal.symbol} | {reason}")
        self.ml_stats['ml_rejected'] += 1
        return False, reason, None

      # 2.2 ML Agreement Check
      if self.ml_require_agreement:
        ml_direction = ml_result['direction']
        if ml_direction != signal.signal_type:
          reason = (
            f"ML не согласна со стратегией: "
            f"Strategy={safe_enum_value(signal.signal_type)}, "
            f"ML={safe_enum_value(ml_direction)} "
            f"(confidence={ml_result['confidence']:.2f})"
          )
          logger.warning(f"{signal.symbol} | {reason}")
          self.ml_stats['ml_rejected'] += 1
          return False, reason, None

      # 2.3 Manipulation Check
      if self.ml_manipulation_check:
        manipulation_risk = ml_result.get('manipulation_risk', 0.0)
        if manipulation_risk > 0.8:
          reason = (
            f"Обнаружен высокий риск манипуляции: "
            f"{manipulation_risk:.2f}"
          )
          logger.warning(f"{signal.symbol} | {reason}")
          self.ml_stats['ml_rejected'] += 1
          return False, reason, None

      # 2.4 Market Regime Check
      if self.ml_regime_check:
        regime = ml_result.get('market_regime')
        regime_ok, regime_reason = self._check_regime_compatibility(
          signal, regime
        )
        if not regime_ok:
          logger.warning(f"{signal.symbol} | {regime_reason}")
          self.ml_stats['ml_rejected'] += 1
          return False, regime_reason, None

    # ========================================
    # ШАГ 3: РАСЧЕТ SL/TP
    # ========================================
    if ml_result and self.ml_sltp_calculation:
      # ML-based SL/TP
      sltp_calc = sltp_calculator.calculate(
        signal=signal,
        entry_price=signal.price,
        ml_result={
          'predicted_mae': ml_result.get('predicted_mae', 0.012),
          'predicted_return': ml_result['predicted_return'],
          'confidence': ml_result['confidence']
        },
        atr=signal.metadata.get('atr') if signal.metadata else None,
        market_regime=ml_result.get('market_regime')
      )

      logger.debug(
        f"{signal.symbol} | ML-based SL/TP: "
        f"SL={sltp_calc.stop_loss:.2f}, "
        f"TP={sltp_calc.take_profit:.2f}, "
        f"R/R={sltp_calc.risk_reward_ratio:.2f}"
      )
    else:
      # Fallback: ATR-based или fixed
      sltp_calc = sltp_calculator.calculate(
        signal=signal,
        entry_price=signal.price,
        ml_result=None,
        atr=signal.metadata.get('atr') if signal.metadata else None
      )

      if not ml_result:
        self.ml_stats['fallback_used'] += 1
        logger.debug(
          f"{signal.symbol} | Fallback SL/TP: "
          f"method={sltp_calc.calculation_method}"
        )

    # ========================================
    # ШАГ 4: РАСЧЕТ РАЗМЕРА ПОЗИЦИИ
    # ========================================
    if ml_result and self.ml_position_sizing:
      # ML-adjusted sizing
      base_size = balance * 0.02  # 2% базовый риск

      ml_size_mult = self._calculate_ml_size_multiplier(ml_result)

      position_size_usdt = base_size * ml_size_mult

      # Ограничиваем максимумом
      max_size = balance * 0.05  # 5% максимум
      position_size_usdt = min(position_size_usdt, max_size)

      logger.debug(
        f"{signal.symbol} | ML position sizing: "
        f"base=${base_size:.2f} × {ml_size_mult:.2f} = "
        f"${position_size_usdt:.2f}"
      )
    else:
      # Fallback: Adaptive Risk Calculator
      try:
        risk_params = adaptive_risk_calculator.calculate(
          signal=signal,
          balance=balance,
          stop_loss_price=sltp_calc.stop_loss,
          current_volatility=signal.metadata.get('volatility') if signal.metadata else None,
          ml_confidence=None
        )

        position_size_usdt = risk_params.max_position_usdt

        if not ml_result:
          logger.debug(
            f"{signal.symbol} | Fallback sizing: "
            f"${position_size_usdt:.2f}"
          )
      except Exception as e:
        logger.error(
          f"{signal.symbol} | Adaptive risk calculation failed: {e}",
          exc_info=True
        )
        # Emergency fallback
        position_size_usdt = balance * 0.02

    # ========================================
    # ШАГ 5: БАЗОВАЯ ВАЛИДАЦИЯ
    # ========================================
    is_valid, reason = self.validate_signal(signal, position_size_usdt)

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
    ml_adjustments = MLRiskAdjustments(
      position_size_multiplier=(
        ml_size_mult if ml_result and self.ml_position_sizing
        else 1.0
      ),
      stop_loss_price=sltp_calc.stop_loss,
      take_profit_price=sltp_calc.take_profit,
      ml_confidence=ml_result['confidence'] if ml_result else 0.0,
      expected_return=ml_result['predicted_return'] if ml_result else 0.0,
      market_regime=ml_result.get('market_regime') if ml_result else MarketRegime.RANGING,
      manipulation_risk_score=ml_result.get('manipulation_risk', 0.0) if ml_result else 0.0,
      feature_quality=ml_result.get('feature_quality', 0.0) if ml_result else 0.0,
      allow_entry=True,
      rejection_reason=None
    )

    if ml_result:
      self.ml_stats['ml_used'] += 1

    logger.info(
      f"{signal.symbol} | ✅ ML-enhanced validation PASSED | "
      f"ML={'used' if ml_result else 'fallback'}, "
      f"Confidence={ml_adjustments.ml_confidence:.2f}, "
      f"Size mult={ml_adjustments.position_size_multiplier:.2f}x, "
      f"Expected return={ml_adjustments.expected_return:.2%}"
    )

    return True, None, ml_adjustments

  async def _get_ml_prediction(
      self,
      signal: TradingSignal,
      feature_vector: Dict
  ) -> Optional[Dict]:
    """
    Получение ML предсказания через MLSignalValidator.

    Args:
        signal: Торговый сигнал
        feature_vector: ML признаки

    Returns:
        Dict с ML предсказаниями или None
    """
    if not self.ml_validator:
      return None

    try:
      # Валидация через ML Validator
      validation_result = await self.ml_validator.validate(
        signal,
        feature_vector
      )

      if not validation_result or not validation_result.validated:
        return None

      # Извлекаем ML данные из ValidationResult
      return {
        'confidence': validation_result.ml_confidence or 0.0,
        'direction': validation_result.final_signal_type,
        'predicted_return': validation_result.ml_expected_return or 0.0,
        'predicted_mae': 0.012,  # Default MAE, можно расширить ValidationResult
        'manipulation_risk': 0.0,  # TODO: добавить в ValidationResult
        'market_regime': None,  # TODO: добавить в ValidationResult
        'feature_quality': 0.8  # Default quality
      }

    except Exception as e:
      logger.error(
        f"{signal.symbol} | Error getting ML prediction: {e}",
        exc_info=True
      )
      return None

  def _calculate_ml_size_multiplier(self, ml_result: Dict) -> float:
    """
    Расчет множителя размера на основе ML факторов.

    Factors:
    - ML Confidence (0.7x - 2.0x)
    - Expected Return (0.8x - 1.5x)
    - Market Regime (0.7x - 1.3x)
    - Feature Quality (0.8x - 1.0x)

    Args:
        ml_result: ML предсказание

    Returns:
        float: Множитель размера [0.7, 2.5]
    """
    confidence = ml_result['confidence']
    pred_return = ml_result['predicted_return']
    regime = ml_result.get('market_regime')
    quality = ml_result.get('feature_quality', 0.8)

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
    else:
      regime_mult = 1.0

    # 4. Quality multiplier (0.8x - 1.0x)
    quality_mult = 0.8 + (quality * 0.2)

    # Total
    total_mult = conf_mult * ret_mult * regime_mult * quality_mult

    # Cap at [0.7, 2.5]
    total_mult = max(0.7, min(total_mult, 2.5))

    logger.debug(
      f"ML size multiplier: "
      f"conf={conf_mult:.2f}x, "
      f"ret={ret_mult:.2f}x, "
      f"regime={regime_mult:.2f}x, "
      f"quality={quality_mult:.2f}x → "
      f"total={total_mult:.2f}x"
    )

    return total_mult

  def _check_regime_compatibility(
      self,
      signal: TradingSignal,
      regime: Optional[MarketRegime]
  ) -> Tuple[bool, Optional[str]]:
    """
    Проверка совместимости сигнала с market regime.

    Rules:
    - DISTRIBUTION + BUY → REJECT (крупные продают)
    - ACCUMULATION + SELL → REJECT (крупные покупают)
    - HIGH_VOLATILITY → WARNING (но разрешаем)

    Args:
        signal: Торговый сигнал
        regime: Режим рынка

    Returns:
        Tuple[compatible, rejection_reason]
    """
    if not regime:
      return True, None

    signal_type_str = safe_enum_value(signal.signal_type)

    # DISTRIBUTION phase - не открываем long
    if regime == MarketRegime.DISTRIBUTION and signal_type_str == "BUY":
      return False, (
        f"Market regime: DISTRIBUTION detected - "
        f"институциональные игроки продают, избегаем long позиций"
      )

    # ACCUMULATION phase - не открываем short
    if regime == MarketRegime.ACCUMULATION and signal_type_str == "SELL":
      return False, (
        f"Market regime: ACCUMULATION detected - "
        f"институциональные игроки покупают, избегаем short позиций"
      )

    # HIGH_VOLATILITY - предупреждение, но разрешаем
    if regime == MarketRegime.HIGH_VOLATILITY:
      logger.warning(
        f"{signal.symbol} | Market regime: HIGH VOLATILITY - "
        f"reduced position size applied"
      )

    return True, None

  def get_ml_statistics(self) -> Dict:
    """
    Получение статистики ML использования.

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