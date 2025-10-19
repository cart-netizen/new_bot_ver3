"""
Изолированные тесты для CorrelationManager.

Не требует полной инициализации бота и конфигурации.

Путь: backend/tests/test_correlation_isolated.py
"""
import pytest
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set


# ==================== MOCK DEPENDENCIES ====================

@dataclass
class CorrelationGroup:
    """Группа коррелирующих активов."""
    group_id: str
    symbols: List[str]
    avg_correlation: float
    active_positions: int = 0
    total_exposure_usdt: float = 0.0


class MockLogger:
    """Мок логгера для тестов."""

    def info(self, msg):
        pass

    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg, exc_info=None):
        pass


# ==================== CORRELATION CALCULATOR ====================

class CorrelationCalculator:
    """Расчет корреляций между торговыми парами."""

    @staticmethod
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """Расчет процентных изменений."""
        if len(prices) < 2:
            return np.array([])

        returns = np.diff(prices) / prices[:-1]
        return returns

    @staticmethod
    def calculate_correlation(returns_a: np.ndarray, returns_b: np.ndarray) -> float:
        """Расчет Pearson correlation coefficient."""
        if len(returns_a) < 2 or len(returns_b) < 2:
            return 0.0

        if len(returns_a) != len(returns_b):
            min_len = min(len(returns_a), len(returns_b))
            returns_a = returns_a[-min_len:]
            returns_b = returns_b[-min_len:]

        try:
            correlation_matrix = np.corrcoef(returns_a, returns_b)
            correlation = correlation_matrix[0, 1]

            if np.isnan(correlation):
                return 0.0

            return float(correlation)

        except Exception:
            return 0.0


# ==================== CORRELATION GROUP MANAGER ====================

class CorrelationGroupManager:
    """Управление группами коррелирующих активов."""

    def __init__(self, correlation_threshold: float = 0.7):
        self.correlation_threshold = correlation_threshold
        self.groups: Dict[str, CorrelationGroup] = {}
        self.symbol_to_group: Dict[str, str] = {}
        self.logger = MockLogger()

    def create_groups_from_matrix(
        self,
        symbols: List[str],
        correlation_matrix: Dict[Tuple[str, str], float]
    ):
        """Создание групп на основе correlation matrix."""
        self.groups.clear()
        self.symbol_to_group.clear()

        processed: Set[str] = set()
        group_counter = 0

        for symbol in symbols:
            if symbol in processed:
                continue

            group_id = f"group_{group_counter}"
            group_symbols = {symbol}

            for other_symbol in symbols:
                if other_symbol == symbol or other_symbol in processed:
                    continue

                key = (symbol, other_symbol) if symbol < other_symbol else (other_symbol, symbol)
                correlation = correlation_matrix.get(key, 0.0)

                if abs(correlation) >= self.correlation_threshold:
                    group_symbols.add(other_symbol)

            if len(group_symbols) > 1:
                correlations = []
                for s1 in group_symbols:
                    for s2 in group_symbols:
                        if s1 < s2:
                            key = (s1, s2)
                            if key in correlation_matrix:
                                correlations.append(abs(correlation_matrix[key]))

                avg_correlation = np.mean(correlations) if correlations else 0.0

                group = CorrelationGroup(
                    group_id=group_id,
                    symbols=sorted(list(group_symbols)),
                    avg_correlation=avg_correlation,
                    active_positions=0,
                    total_exposure_usdt=0.0
                )

                self.groups[group_id] = group

                for s in group_symbols:
                    self.symbol_to_group[s] = group_id
                    processed.add(s)

                group_counter += 1
            else:
                processed.add(symbol)

    def get_group_for_symbol(self, symbol: str) -> Optional[CorrelationGroup]:
        """Получить группу для символа."""
        group_id = self.symbol_to_group.get(symbol)
        if not group_id:
            return None
        return self.groups.get(group_id)

    def update_group_position_opened(self, symbol: str, exposure_usdt: float):
        """Обновление при открытии позиции."""
        group = self.get_group_for_symbol(symbol)
        if not group:
            return

        group.active_positions += 1
        group.total_exposure_usdt += exposure_usdt

    def update_group_position_closed(self, symbol: str, exposure_usdt: float):
        """Обновление при закрытии позиции."""
        group = self.get_group_for_symbol(symbol)
        if not group:
            return

        group.active_positions = max(0, group.active_positions - 1)
        group.total_exposure_usdt = max(0.0, group.total_exposure_usdt - exposure_usdt)


# ==================== CORRELATION MANAGER ====================

class CorrelationManager:
    """Главный менеджер корреляций."""

    def __init__(self):
        self.enabled = True
        self.max_threshold = 0.7
        self.max_positions_per_group = 1

        self.calculator = CorrelationCalculator()
        self.group_manager = CorrelationGroupManager(
            correlation_threshold=self.max_threshold
        )
        self.logger = MockLogger()

    def can_open_position(
        self,
        symbol: str,
        position_size_usdt: float
    ) -> Tuple[bool, Optional[str]]:
        """Проверка возможности открытия позиции."""
        if not self.enabled:
            return True, None

        group = self.group_manager.get_group_for_symbol(symbol)

        if not group:
            return True, None

        if group.active_positions >= self.max_positions_per_group:
            reason = (
                f"Достигнут лимит позиций в группе {group.group_id} "
                f"({group.active_positions}/{self.max_positions_per_group}). "
                f"Коррелирующие активы: {', '.join(group.symbols)}"
            )
            return False, reason

        return True, None

    def notify_position_opened(self, symbol: str, exposure_usdt: float):
        """Уведомление об открытии позиции."""
        if not self.enabled:
            return
        self.group_manager.update_group_position_opened(symbol, exposure_usdt)

    def notify_position_closed(self, symbol: str, exposure_usdt: float):
        """Уведомление о закрытии позиции."""
        if not self.enabled:
            return
        self.group_manager.update_group_position_closed(symbol, exposure_usdt)

    def get_statistics(self) -> Dict:
        """Получить статистику."""
        groups = list(self.group_manager.groups.values())

        total_active_positions = sum(g.active_positions for g in groups)
        total_exposure = sum(g.total_exposure_usdt for g in groups)
        groups_with_positions = [g for g in groups if g.active_positions > 0]

        return {
            "enabled": self.enabled,
            "total_groups": len(groups),
            "groups_with_positions": len(groups_with_positions),
            "total_active_positions": total_active_positions,
            "total_exposure_usdt": total_exposure,
            "max_positions_per_group": self.max_positions_per_group,
            "correlation_threshold": self.max_threshold
        }


# ==================== TESTS ====================

class TestCorrelationCalculator:
    """Тесты для CorrelationCalculator."""

    def setup_method(self):
        """Setup перед каждым тестом."""
        self.calculator = CorrelationCalculator()

    def test_calculate_returns(self):
        """Тест расчета returns."""
        prices = np.array([100.0, 102.0, 101.0, 105.0])
        returns = self.calculator.calculate_returns(prices)

        assert len(returns) == 3
        assert abs(returns[0] - 0.02) < 0.0001
        assert returns[1] < 0
        assert returns[2] > 0

    def test_calculate_returns_empty(self):
        """Тест с пустым массивом."""
        prices = np.array([])
        returns = self.calculator.calculate_returns(prices)
        assert len(returns) == 0

    def test_calculate_correlation_perfect_positive(self):
        """Тест идеальной положительной корреляции."""
        returns_a = np.array([0.01, 0.02, 0.03, 0.01])
        returns_b = np.array([0.01, 0.02, 0.03, 0.01])

        correlation = self.calculator.calculate_correlation(returns_a, returns_b)
        assert abs(correlation - 1.0) < 0.01

    def test_calculate_correlation_perfect_negative(self):
        """Тест идеальной отрицательной корреляции."""
        returns_a = np.array([0.01, 0.02, 0.03, 0.01])
        returns_b = np.array([-0.01, -0.02, -0.03, -0.01])

        correlation = self.calculator.calculate_correlation(returns_a, returns_b)
        assert abs(correlation - (-1.0)) < 0.01


class TestCorrelationGroupManager:
    """Тесты для CorrelationGroupManager."""

    def setup_method(self):
        """Setup."""
        self.manager = CorrelationGroupManager(correlation_threshold=0.7)

    def test_create_groups_high_correlation(self):
        """Тест создания групп с высокой корреляцией."""
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        correlation_matrix = {
            ("BTCUSDT", "ETHUSDT"): 0.85,
            ("BTCUSDT", "SOLUSDT"): 0.78,
            ("ETHUSDT", "SOLUSDT"): 0.80
        }

        self.manager.create_groups_from_matrix(symbols, correlation_matrix)

        assert len(self.manager.groups) == 1
        group = list(self.manager.groups.values())[0]
        assert len(group.symbols) == 3
        assert set(group.symbols) == set(symbols)

    def test_create_groups_no_correlation(self):
        """Тест когда нет корреляций."""
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        correlation_matrix = {
            ("BTCUSDT", "ETHUSDT"): 0.3,
            ("BTCUSDT", "SOLUSDT"): 0.2,
            ("ETHUSDT", "SOLUSDT"): 0.1
        }

        self.manager.create_groups_from_matrix(symbols, correlation_matrix)
        assert len(self.manager.groups) == 0

    def test_update_group_position_opened(self):
        """Тест обновления при открытии позиции."""
        symbols = ["BTCUSDT", "ETHUSDT"]
        correlation_matrix = {("BTCUSDT", "ETHUSDT"): 0.85}

        self.manager.create_groups_from_matrix(symbols, correlation_matrix)
        self.manager.update_group_position_opened("BTCUSDT", 1000.0)

        group = self.manager.get_group_for_symbol("BTCUSDT")
        assert group.active_positions == 1
        assert group.total_exposure_usdt == 1000.0


class TestCorrelationManager:
    """Тесты для CorrelationManager."""

    def setup_method(self):
        """Setup."""
        self.manager = CorrelationManager()
        self.manager.enabled = True

    def test_can_open_position_no_group(self):
        """Тест когда символ не в группе."""
        can_open, reason = self.manager.can_open_position("XRPUSDT", 1000.0)
        assert can_open is True
        assert reason is None

    def test_can_open_position_within_limit(self):
        """Тест открытия в пределах лимита."""
        group = CorrelationGroup(
            group_id="test_group",
            symbols=["BTCUSDT", "ETHUSDT"],
            avg_correlation=0.85,
            active_positions=0,
            total_exposure_usdt=0.0
        )

        self.manager.group_manager.groups["test_group"] = group
        self.manager.group_manager.symbol_to_group["BTCUSDT"] = "test_group"
        self.manager.max_positions_per_group = 1

        can_open, reason = self.manager.can_open_position("BTCUSDT", 1000.0)
        assert can_open is True
        assert reason is None

    def test_can_open_position_exceeds_limit(self):
        """Тест превышения лимита."""
        group = CorrelationGroup(
            group_id="test_group",
            symbols=["BTCUSDT", "ETHUSDT"],
            avg_correlation=0.85,
            active_positions=1,
            total_exposure_usdt=1000.0
        )

        self.manager.group_manager.groups["test_group"] = group
        self.manager.group_manager.symbol_to_group["ETHUSDT"] = "test_group"
        self.manager.max_positions_per_group = 1

        can_open, reason = self.manager.can_open_position("ETHUSDT", 500.0)
        assert can_open is False
        assert reason is not None
        assert "лимит" in reason.lower()

    def test_full_position_lifecycle(self):
        """Тест полного цикла: открытие -> проверка -> закрытие."""
        # 1. Создаем группу
        symbols = ["BTCUSDT", "ETHUSDT"]
        correlation_matrix = {("BTCUSDT", "ETHUSDT"): 0.85}

        self.manager.group_manager.create_groups_from_matrix(
            symbols, correlation_matrix
        )

        # 2. Открываем первую позицию
        can_open_1, _ = self.manager.can_open_position("BTCUSDT", 1000.0)
        assert can_open_1 is True

        self.manager.notify_position_opened("BTCUSDT", 1000.0)

        # 3. Пытаемся открыть вторую (должно быть отклонено)
        can_open_2, reason = self.manager.can_open_position("ETHUSDT", 500.0)
        assert can_open_2 is False
        assert "лимит" in reason.lower()

        # 4. Закрываем первую позицию
        self.manager.notify_position_closed("BTCUSDT", 1000.0)

        # 5. Теперь можем открыть вторую
        can_open_3, _ = self.manager.can_open_position("ETHUSDT", 500.0)
        assert can_open_3 is True

    def test_get_statistics(self):
        """Тест получения статистики."""
        group = CorrelationGroup(
            group_id="test_group",
            symbols=["BTCUSDT", "ETHUSDT"],
            avg_correlation=0.85,
            active_positions=2,
            total_exposure_usdt=1500.0
        )

        self.manager.group_manager.groups["test_group"] = group
        stats = self.manager.get_statistics()

        assert stats["enabled"] is True
        assert stats["total_groups"] == 1
        assert stats["total_active_positions"] == 2
        assert stats["total_exposure_usdt"] == 1500.0