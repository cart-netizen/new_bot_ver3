"""
Signal Stabilizer - предотвращение мерцания сигналов.

Функциональность:
- Hysteresis для предотвращения частых смен направления
- Cooldown между сигналами
- История сигналов для стабилизации
- Консистентность проверка

Industry Standard:
- Предотвращает "whipsaw" эффект при боковом движении
- Обеспечивает стабильность сигналов при микро-волатильности
- Использует временное сглаживание для фильтрации шума

Путь: backend/strategies/signal_stabilizer.py
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np

from backend.core.logger import get_logger
from backend.models.signal import SignalType

logger = get_logger(__name__)


@dataclass
class StabilizedSignal:
    """Результат стабилизации сигнала."""
    should_emit: bool
    signal_type: Optional[SignalType]
    confidence: float
    original_confidence: float
    stability_score: float  # 0-1, насколько стабилен сигнал
    reason: str


@dataclass
class SignalHistoryEntry:
    """Запись в истории сигналов."""
    signal_type: SignalType
    confidence: float
    timestamp: int
    price: float


class SignalStabilizer:
    """
    Стабилизатор сигналов для предотвращения мерцания.

    Проблема: Стратегии могут генерировать противоположные сигналы
    при небольших изменениях цены (особенно около ключевых уровней).

    Решение:
    1. Hysteresis - требуем значительное изменение для смены направления
    2. Cooldown - минимальное время между сменами направления
    3. Consistency - проверяем стабильность сигнала в истории
    4. Confidence averaging - усредняем confidence по истории

    Использование:
        stabilizer = SignalStabilizer()

        # В методе analyze() стратегии:
        result = stabilizer.stabilize(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=50000.0
        )

        if result.should_emit:
            return create_signal(result.signal_type, result.confidence)
        else:
            return None
    """

    def __init__(
        self,
        cooldown_ms: int = 5000,
        history_size: int = 5,
        min_consistency: float = 0.6,
        hysteresis_pct: float = 0.1,
        max_history_age_ms: int = 30000
    ):
        """
        Инициализация стабилизатора.

        Args:
            cooldown_ms: Минимальное время между сменами направления (мс)
            history_size: Размер истории для анализа
            min_consistency: Минимальная консистентность (0-1) для эмиссии сигнала
            hysteresis_pct: Порог гистерезиса в % от цены для смены направления
            max_history_age_ms: Максимальный возраст записей в истории (мс)
        """
        self.cooldown_ms = cooldown_ms
        self.history_size = history_size
        self.min_consistency = min_consistency
        self.hysteresis_pct = hysteresis_pct
        self.max_history_age_ms = max_history_age_ms

        # История сигналов по символам
        self._history: Dict[str, deque] = {}

        # Последний эмитированный сигнал по символам
        self._last_emitted: Dict[str, Tuple[SignalType, int, float]] = {}  # symbol -> (type, ts, price)

        # Статистика
        self.total_signals_received = 0
        self.signals_emitted = 0
        self.signals_filtered = 0
        self.direction_changes_blocked = 0

        logger.info(
            f"SignalStabilizer initialized: cooldown={cooldown_ms}ms, "
            f"history_size={history_size}, min_consistency={min_consistency}, "
            f"hysteresis={hysteresis_pct}%"
        )

    def stabilize(
        self,
        symbol: str,
        signal_type: SignalType,
        confidence: float,
        price: float
    ) -> StabilizedSignal:
        """
        Стабилизировать сигнал.

        Args:
            symbol: Торговая пара
            signal_type: Тип сигнала (BUY/SELL)
            confidence: Уверенность сигнала (0-1)
            price: Текущая цена

        Returns:
            StabilizedSignal с решением об эмиссии
        """
        self.total_signals_received += 1
        current_time = self._current_time_ms()

        # Инициализация истории для символа
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self.history_size)

        # Добавляем в историю
        self._history[symbol].append(SignalHistoryEntry(
            signal_type=signal_type,
            confidence=confidence,
            timestamp=current_time,
            price=price
        ))

        # Очищаем устаревшие записи
        self._cleanup_old_entries(symbol, current_time)

        # Проверка 1: Cooldown при смене направления
        cooldown_result = self._check_cooldown(symbol, signal_type, current_time)
        if not cooldown_result['passed']:
            self.direction_changes_blocked += 1
            self.signals_filtered += 1
            return StabilizedSignal(
                should_emit=False,
                signal_type=None,
                confidence=0.0,
                original_confidence=confidence,
                stability_score=0.0,
                reason=cooldown_result['reason']
            )

        # Проверка 2: Hysteresis (требуем значительное изменение цены)
        hysteresis_result = self._check_hysteresis(symbol, signal_type, price)
        if not hysteresis_result['passed']:
            self.signals_filtered += 1
            return StabilizedSignal(
                should_emit=False,
                signal_type=None,
                confidence=0.0,
                original_confidence=confidence,
                stability_score=hysteresis_result['stability'],
                reason=hysteresis_result['reason']
            )

        # Проверка 3: Консистентность истории
        consistency_result = self._check_consistency(symbol, signal_type)
        if not consistency_result['passed']:
            self.signals_filtered += 1
            return StabilizedSignal(
                should_emit=False,
                signal_type=None,
                confidence=0.0,
                original_confidence=confidence,
                stability_score=consistency_result['consistency'],
                reason=consistency_result['reason']
            )

        # Все проверки пройдены - эмитируем сигнал
        # Усредняем confidence по консистентной истории
        avg_confidence = self._calculate_average_confidence(symbol, signal_type)
        final_confidence = (confidence * 0.6) + (avg_confidence * 0.4)  # 60% текущий, 40% история

        # Обновляем last_emitted
        self._last_emitted[symbol] = (signal_type, current_time, price)

        self.signals_emitted += 1

        return StabilizedSignal(
            should_emit=True,
            signal_type=signal_type,
            confidence=final_confidence,
            original_confidence=confidence,
            stability_score=consistency_result['consistency'],
            reason="Signal stable and consistent"
        )

    def _check_cooldown(
        self,
        symbol: str,
        signal_type: SignalType,
        current_time: int
    ) -> Dict:
        """Проверить cooldown при смене направления."""
        last = self._last_emitted.get(symbol)

        if not last:
            return {'passed': True, 'reason': 'No previous signal'}

        last_type, last_time, _ = last

        # Если направление то же - cooldown не применяется
        if last_type == signal_type:
            return {'passed': True, 'reason': 'Same direction'}

        # Смена направления - проверяем cooldown
        time_since_last = current_time - last_time

        if time_since_last < self.cooldown_ms:
            return {
                'passed': False,
                'reason': f"Direction change cooldown: {time_since_last}ms < {self.cooldown_ms}ms"
            }

        return {'passed': True, 'reason': 'Cooldown passed'}

    def _check_hysteresis(
        self,
        symbol: str,
        signal_type: SignalType,
        price: float
    ) -> Dict:
        """Проверить hysteresis (значительное изменение цены)."""
        last = self._last_emitted.get(symbol)

        if not last:
            return {'passed': True, 'stability': 1.0, 'reason': 'No previous signal'}

        last_type, _, last_price = last

        # Если направление то же - hysteresis не применяется
        if last_type == signal_type:
            return {'passed': True, 'stability': 1.0, 'reason': 'Same direction'}

        # Смена направления - требуем значительное изменение цены
        price_change_pct = abs(price - last_price) / last_price * 100

        if price_change_pct < self.hysteresis_pct:
            stability = price_change_pct / self.hysteresis_pct
            return {
                'passed': False,
                'stability': stability,
                'reason': f"Hysteresis: price change {price_change_pct:.3f}% < {self.hysteresis_pct}%"
            }

        return {'passed': True, 'stability': 1.0, 'reason': 'Hysteresis passed'}

    def _check_consistency(
        self,
        symbol: str,
        signal_type: SignalType
    ) -> Dict:
        """Проверить консистентность сигнала в истории."""
        history = self._history.get(symbol, [])

        if len(history) < 2:
            return {'passed': True, 'consistency': 1.0, 'reason': 'Insufficient history'}

        # Подсчитываем сигналы того же типа в истории
        same_type_count = sum(1 for h in history if h.signal_type == signal_type)
        consistency = same_type_count / len(history)

        if consistency < self.min_consistency:
            return {
                'passed': False,
                'consistency': consistency,
                'reason': f"Low consistency: {consistency:.2f} < {self.min_consistency}"
            }

        return {'passed': True, 'consistency': consistency, 'reason': 'Consistent'}

    def _calculate_average_confidence(
        self,
        symbol: str,
        signal_type: SignalType
    ) -> float:
        """Вычислить средний confidence для сигналов того же типа."""
        history = self._history.get(symbol, [])

        if not history:
            return 0.0

        # Фильтруем только сигналы того же типа
        same_type_confidences = [
            h.confidence for h in history
            if h.signal_type == signal_type
        ]

        if not same_type_confidences:
            return 0.0

        return float(np.mean(same_type_confidences))

    def _cleanup_old_entries(self, symbol: str, current_time: int):
        """Удалить устаревшие записи из истории."""
        if symbol not in self._history:
            return

        history = self._history[symbol]
        min_time = current_time - self.max_history_age_ms

        # Фильтруем только свежие записи
        fresh_entries = [h for h in history if h.timestamp >= min_time]

        self._history[symbol] = deque(fresh_entries, maxlen=self.history_size)

    def reset(self, symbol: Optional[str] = None):
        """
        Сбросить состояние стабилизатора.

        Args:
            symbol: Символ для сброса (None = сбросить все)
        """
        if symbol:
            self._history.pop(symbol, None)
            self._last_emitted.pop(symbol, None)
            logger.debug(f"SignalStabilizer reset for {symbol}")
        else:
            self._history.clear()
            self._last_emitted.clear()
            logger.info("SignalStabilizer fully reset")

    def get_statistics(self) -> Dict:
        """Получить статистику стабилизатора."""
        filter_rate = (
            self.signals_filtered / self.total_signals_received
            if self.total_signals_received > 0 else 0.0
        )

        return {
            'total_signals_received': self.total_signals_received,
            'signals_emitted': self.signals_emitted,
            'signals_filtered': self.signals_filtered,
            'direction_changes_blocked': self.direction_changes_blocked,
            'filter_rate': filter_rate,
            'symbols_tracked': len(self._history)
        }

    @staticmethod
    def _current_time_ms() -> int:
        """Получить текущее время в миллисекундах."""
        return int(datetime.now().timestamp() * 1000)


class DirectionStabilizer:
    """
    Упрощенный стабилизатор направления для стратегий.

    Предотвращает мерцание BUY/SELL при колебаниях около ключевых уровней.
    Использует взвешенное голосование по недавней истории.
    """

    def __init__(
        self,
        window_size: int = 5,
        dominance_threshold: float = 0.65,
        cooldown_signals: int = 2
    ):
        """
        Args:
            window_size: Размер окна истории
            dominance_threshold: Порог доминирования одного направления (0.5-1.0)
            cooldown_signals: Количество последовательных сигналов для смены направления
        """
        self.window_size = window_size
        self.dominance_threshold = dominance_threshold
        self.cooldown_signals = cooldown_signals

        self._history: Dict[str, deque] = {}
        self._current_direction: Dict[str, Optional[SignalType]] = {}

    def get_stable_direction(
        self,
        symbol: str,
        signal_type: SignalType,
        confidence: float
    ) -> Tuple[Optional[SignalType], float]:
        """
        Получить стабильное направление.

        Args:
            symbol: Торговая пара
            signal_type: Текущий сигнал
            confidence: Уверенность

        Returns:
            (stable_direction, adjusted_confidence) или (None, 0.0) если нестабильно
        """
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self.window_size)

        # Добавляем сигнал с весом = confidence
        self._history[symbol].append((signal_type, confidence))

        history = list(self._history[symbol])

        if len(history) < 2:
            self._current_direction[symbol] = signal_type
            return signal_type, confidence

        # Взвешенное голосование
        buy_weight = sum(c for t, c in history if t == SignalType.BUY)
        sell_weight = sum(c for t, c in history if t == SignalType.SELL)
        total_weight = buy_weight + sell_weight

        if total_weight == 0:
            return None, 0.0

        buy_ratio = buy_weight / total_weight

        # Определяем доминирующее направление
        if buy_ratio >= self.dominance_threshold:
            dominant = SignalType.BUY
            dominance = buy_ratio
        elif buy_ratio <= (1 - self.dominance_threshold):
            dominant = SignalType.SELL
            dominance = 1 - buy_ratio
        else:
            # Нет доминирования - возвращаем текущее направление или None
            current = self._current_direction.get(symbol)
            if current:
                # Проверяем cooldown - нужно N последовательных сигналов для смены
                recent = history[-self.cooldown_signals:]
                if all(t == signal_type for t, _ in recent) and signal_type != current:
                    self._current_direction[symbol] = signal_type
                    return signal_type, confidence * 0.8  # Пониженная уверенность
                return current, confidence * 0.7  # Держим текущее направление
            return None, 0.0

        self._current_direction[symbol] = dominant

        # Усредняем confidence для доминирующего направления
        dominant_confidences = [c for t, c in history if t == dominant]
        avg_confidence = np.mean(dominant_confidences) if dominant_confidences else confidence

        return dominant, avg_confidence * dominance

    def reset(self, symbol: Optional[str] = None):
        """Сбросить состояние."""
        if symbol:
            self._history.pop(symbol, None)
            self._current_direction.pop(symbol, None)
        else:
            self._history.clear()
            self._current_direction.clear()
