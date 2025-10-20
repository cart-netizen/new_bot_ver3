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
            f"ML не согласен со стратегией: "
            f"ML={ml_direction.value}, Strategy={signal.signal_type.value}"
          )
          logger.warning(f"{signal.symbol} | {reason}")
          self.ml_stats['ml_rejected'] += 1
          return False, reason, None

      # ========================================
      # 2.3 MANIPULATION DETECTION
      # ========================================
      if self.ml_manipulation_check:
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
      if self.ml_regime_check and ml_result['market_regime']:
        regime = ml_result['market_regime']

        # Определяем запрещенные режимы для текущей стратегии
        forbidden_regimes = self._get_forbidden_regimes(signal)

        if regime in forbidden_regimes:
          reason = (
            f"Неподходящий режим рынка: {regime.value}, "
            f"стратегия не рекомендуется"
          )
          logger.warning(f"{signal.symbol} | {reason}")
          self.ml_stats['ml_rejected'] += 1
          return False, reason, None

    # ========================================
    # ШАГ 3: SL/TP CALCULATION
    # ========================================
    if self.ml_sltp_calculation and ml_result:
      # ML-based SL/TP
      sltp_calc = await self._calculate_ml_sltp(
        signal,
        ml_result,
        balance
      )
      logger.info(
        f"{signal.symbol} | ML-based SL/TP: "
        f"SL={sltp_calc.stop_loss:.2f}, "
        f"TP={sltp_calc.take_profit:.2f}, "
        f"R/R={sltp_calc.risk_reward_ratio:.2f}"
      )
    else:
      # Fallback: ATR-based SL/TP
      sltp_calc = await sltp_calculator.calculate_sltp(
        signal=signal,
        balance=balance,
        symbol_info=None  # Берется из кэша
      )
      logger.info(
        f"{signal.symbol} | Fallback ATR-based SL/TP: "
        f"SL={sltp_calc.stop_loss:.2f}, "
        f"TP={sltp_calc.take_profit:.2f}"
      )

    # ========================================
    # ШАГ 4: POSITION SIZING
    # ========================================
    # ========================================
    # ИСПРАВЛЕНИЕ: Инициализируем ml_size_mult ПЕРЕД использованием
    # ========================================
    ml_size_mult = 1.0  # Дефолтное значение

    if self.ml_position_sizing and ml_result:
      # ML-adjusted position sizing
      ml_size_mult = self._calculate_ml_size_multiplier(ml_result)

      logger.info(
        f"{signal.symbol} | ML size multiplier: {ml_size_mult:.2f}x, "
        f"factors: confidence={ml_result['confidence']:.2f}, "
        f"return={ml_result['predicted_return']:.3f}, "
        f"regime={ml_result['market_regime'].value if ml_result['market_regime'] else 'None'}"
      )
    else:
      # Fallback: Adaptive Risk Calculator
      adaptive_mult = adaptive_risk_calculator.calculate_risk_multiplier(
        signal.confidence
      )
      ml_size_mult = adaptive_mult

      logger.info(
        f"{signal.symbol} | Fallback adaptive size multiplier: "
        f"{ml_size_mult:.2f}x"
      )

    # Вычисляем базовый размер позиции
    base_position_size = self.calculate_position_size(
      signal=signal,  # ✅ Передаем весь сигнал
      available_balance=balance,  # ✅ Правильное имя параметра
      stop_loss_price=sltp_calc.stop_loss,  # ✅ Остается как есть
      leverage=self.default_leverage,  # ✅ Берем из self
      current_volatility=None,  # Optional
      ml_confidence=ml_result['confidence'] if ml_result else None  # ✅ ML confidence
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
      ml_result: Dict,
      balance: float
  ) -> SLTPCalculation:
    """
    ML-based расчет SL/TP.

    Использует:
    - Predicted MAE для установки SL
    - Expected Return для установки TP
    - Market Regime для корректировки

    Args:
        signal: Торговый сигнал
        ml_result: ML предсказание
        balance: Баланс

    Returns:
        SLTPCalculation
    """
    entry_price = signal.price
    predicted_mae = ml_result.get('predicted_mae', 0.015)
    expected_return = ml_result.get('predicted_return', 0.02)
    regime = ml_result.get('market_regime')

    # Корректируем SL на основе MAE и режима
    sl_distance_pct = predicted_mae * 1.5  # 1.5x MAE для буфера

    # Регулируем на основе режима
    if regime == MarketRegime.HIGH_VOLATILITY:
      sl_distance_pct *= 1.3  # Шире SL при волатильности
    elif regime == MarketRegime.STRONG_TREND:
      sl_distance_pct *= 0.9  # Уже SL в тренде

    # Корректируем TP на основе expected return и режима
    tp_distance_pct = expected_return

    # Регулируем на основе режима
    if regime == MarketRegime.RANGING:
      tp_distance_pct *= 0.8  # Ближе TP в флэте
    elif regime == MarketRegime.STRONG_TREND:
      tp_distance_pct *= 1.2  # Дальше TP в тренде

    # Вычисляем цены
    if signal.signal_type == SignalType.BUY:
      stop_loss = entry_price * (1 - sl_distance_pct)
      take_profit = entry_price * (1 + tp_distance_pct)
    else:  # SELL
      stop_loss = entry_price * (1 + sl_distance_pct)
      take_profit = entry_price * (1 - tp_distance_pct)

    # Вычисляем R/R
    sl_distance = abs(entry_price - stop_loss)
    tp_distance = abs(take_profit - entry_price)
    risk_reward = tp_distance / sl_distance if sl_distance > 0 else 1.0

    # Trailing start (начинаем trailing при 40% от TP)
    trailing_start_profit = 0.4

    return SLTPCalculation(
      stop_loss=stop_loss,
      take_profit=take_profit,
      risk_reward_ratio=risk_reward,
      trailing_start_profit=trailing_start_profit,
      calculation_method="ml",
      reasoning={
        "predicted_mae": predicted_mae,
        "expected_return": expected_return,
        "market_regime": regime.value if regime else None,
        "sl_distance_pct": sl_distance_pct,
        "tp_distance_pct": tp_distance_pct,
        "risk_reward": risk_reward
      },
      confidence=ml_result['confidence']
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