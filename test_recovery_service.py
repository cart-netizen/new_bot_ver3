"""
Исправленные тесты для Recovery Service.

КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Правильный патчинг экземпляров repository
"""
import sys
from pathlib import Path

# Определяем путь к backend директории
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from infrastructure.resilience.recovery_service import recovery_service, RecoveryService
from domain.services.fsm_registry import fsm_registry, FSMRegistry
from domain.state_machines.order_fsm import OrderStateMachine
from domain.state_machines.position_fsm import PositionStateMachine
from database.models import (
    Order,
    Position,
    OrderStatus,
    PositionStatus,
    OrderSide,
    OrderType
)


# ==================== FIXTURES ====================

@pytest.fixture
def clean_fsm_registry():
    """Очистка FSM Registry перед каждым тестом."""
    fsm_registry.clear_all()
    yield fsm_registry
    fsm_registry.clear_all()


@pytest.fixture
def mock_order():
    """Мок-объект Order для тестов."""
    order = MagicMock(spec=Order)
    order.id = "test-order-id-123"
    order.client_order_id = "TEST_ORDER_001"
    order.exchange_order_id = "bybit-12345"
    order.symbol = "BTCUSDT"
    order.side = OrderSide.BUY
    order.order_type = OrderType.LIMIT
    order.status = OrderStatus.PLACED
    order.quantity = 0.001
    order.price = 50000.0
    order.created_at = datetime.utcnow() - timedelta(minutes=5)
    order.updated_at = datetime.utcnow() - timedelta(minutes=5)
    order.metadata_json = {}
    return order


@pytest.fixture
def mock_position():
    """Мок-объект Position для тестов."""
    position = MagicMock(spec=Position)
    position.id = "test-position-id-456"
    position.symbol = "BTCUSDT"
    position.side = OrderSide.BUY
    position.status = PositionStatus.OPEN
    position.quantity = 0.001
    position.entry_price = 50000.0
    position.current_price = 51000.0
    position.opened_at = datetime.utcnow() - timedelta(hours=1)
    position.metadata_json = {}
    return position


# ==================== FSM REGISTRY ТЕСТЫ ====================

class TestFSMRegistry:
    """Тесты для FSM Registry."""

    def test_registry_initialization(self, clean_fsm_registry):
        """Тест инициализации реестра."""
        assert clean_fsm_registry is not None
        stats = clean_fsm_registry.get_stats()
        assert stats["total_order_fsms"] == 0
        assert stats["total_position_fsms"] == 0

    def test_register_order_fsm(self, clean_fsm_registry):
        """Тест регистрации FSM ордера."""
        fsm = OrderStateMachine("TEST_ORDER_001", OrderStatus.PENDING)
        clean_fsm_registry.register_order_fsm("TEST_ORDER_001", fsm)

        retrieved_fsm = clean_fsm_registry.get_order_fsm("TEST_ORDER_001")
        assert retrieved_fsm is not None
        assert retrieved_fsm.order_id == "TEST_ORDER_001"
        assert retrieved_fsm.current_status == OrderStatus.PENDING

        stats = clean_fsm_registry.get_stats()
        assert stats["total_order_fsms"] == 1

    def test_register_position_fsm(self, clean_fsm_registry):
        """Тест регистрации FSM позиции."""
        fsm = PositionStateMachine("POS_001", PositionStatus.OPEN)
        clean_fsm_registry.register_position_fsm("POS_001", fsm)

        retrieved_fsm = clean_fsm_registry.get_position_fsm("POS_001")
        assert retrieved_fsm is not None
        assert retrieved_fsm.position_id == "POS_001"
        assert retrieved_fsm.current_status == PositionStatus.OPEN

        stats = clean_fsm_registry.get_stats()
        assert stats["total_position_fsms"] == 1

    def test_unregister_order_fsm(self, clean_fsm_registry):
        """Тест удаления FSM ордера."""
        fsm = OrderStateMachine("TEST_ORDER_001", OrderStatus.FILLED)
        clean_fsm_registry.register_order_fsm("TEST_ORDER_001", fsm)

        result = clean_fsm_registry.unregister_order_fsm("TEST_ORDER_001")
        assert result is True

        retrieved = clean_fsm_registry.get_order_fsm("TEST_ORDER_001")
        assert retrieved is None

    def test_get_order_ids_by_status(self, clean_fsm_registry):
        """Тест фильтрации ордеров по статусам."""
        fsm1 = OrderStateMachine("ORDER_1", OrderStatus.PENDING)
        fsm2 = OrderStateMachine("ORDER_2", OrderStatus.PLACED)
        fsm3 = OrderStateMachine("ORDER_3", OrderStatus.FILLED)

        clean_fsm_registry.register_order_fsm("ORDER_1", fsm1)
        clean_fsm_registry.register_order_fsm("ORDER_2", fsm2)
        clean_fsm_registry.register_order_fsm("ORDER_3", fsm3)

        active_ids = clean_fsm_registry.get_order_ids_by_status(
            ["Pending", "Placed"]
        )

        assert len(active_ids) == 2
        assert "ORDER_1" in active_ids
        assert "ORDER_2" in active_ids
        assert "ORDER_3" not in active_ids

    def test_get_active_position_ids(self, clean_fsm_registry):
        """Тест получения активных позиций."""
        fsm1 = PositionStateMachine("POS_1", PositionStatus.OPEN)
        fsm2 = PositionStateMachine("POS_2", PositionStatus.CLOSED)
        fsm3 = PositionStateMachine("POS_3", PositionStatus.OPEN)

        clean_fsm_registry.register_position_fsm("POS_1", fsm1)
        clean_fsm_registry.register_position_fsm("POS_2", fsm2)
        clean_fsm_registry.register_position_fsm("POS_3", fsm3)

        active_ids = clean_fsm_registry.get_active_position_ids()

        assert len(active_ids) == 2
        assert "POS_1" in active_ids
        assert "POS_3" in active_ids
        assert "POS_2" not in active_ids

    def test_clear_terminal_fsms(self, clean_fsm_registry):
        """Тест очистки терминальных FSM."""
        fsm1 = OrderStateMachine("ORDER_1", OrderStatus.PLACED)  # Активный
        fsm2 = OrderStateMachine("ORDER_2", OrderStatus.FILLED)  # Терминальный
        fsm3 = OrderStateMachine("ORDER_3", OrderStatus.CANCELLED)  # Терминальный

        clean_fsm_registry.register_order_fsm("ORDER_1", fsm1)
        clean_fsm_registry.register_order_fsm("ORDER_2", fsm2)
        clean_fsm_registry.register_order_fsm("ORDER_3", fsm3)

        result = clean_fsm_registry.clear_terminal_fsms()

        assert result["orders_cleared"] == 2

        assert clean_fsm_registry.get_order_fsm("ORDER_1") is not None
        assert clean_fsm_registry.get_order_fsm("ORDER_2") is None
        assert clean_fsm_registry.get_order_fsm("ORDER_3") is None


# ==================== RECOVERY SERVICE ТЕСТЫ ====================

class TestRecoveryService:
    """Тесты для Recovery Service."""

    @pytest.mark.asyncio
    async def test_check_hanging_orders_timeout(self, mock_order):
        """Тест обнаружения ордера зависшего по таймауту."""
        mock_order.status = OrderStatus.PLACED
        mock_order.updated_at = datetime.utcnow() - timedelta(minutes=45)

        # ✅ ИСПРАВЛЕНО: Патчим глобальный экземпляр order_repository
        with patch('infrastructure.repositories.order_repository.order_repository.get_active_orders',
                   new_callable=AsyncMock, return_value=[mock_order]):
            with patch('exchange.rest_client.rest_client.get_open_orders', new_callable=AsyncMock) as mock_exchange:
                mock_exchange.return_value = {
                    "result": {
                        "list": [{
                            "orderLinkId": mock_order.client_order_id,
                            "orderStatus": "New"
                        }]
                    }
                }

                hanging = await recovery_service._check_hanging_orders()

                assert len(hanging) == 1
                assert hanging[0]["client_order_id"] == mock_order.client_order_id
                assert hanging[0]["issue"]["type"] == "timeout_in_status"
                assert hanging[0]["issue"]["minutes_stuck"] > 30

    @pytest.mark.asyncio
    async def test_check_hanging_orders_not_found_on_exchange(self, mock_order):
        """Тест обнаружения ордера отсутствующего на бирже."""
        mock_order.status = OrderStatus.PLACED

        with patch('infrastructure.repositories.order_repository.order_repository.get_active_orders',
                   new_callable=AsyncMock, return_value=[mock_order]):
            with patch('exchange.rest_client.rest_client.get_open_orders', new_callable=AsyncMock) as mock_exchange:
                mock_exchange.return_value = {
                    "result": {"list": []}
                }

                with patch('exchange.rest_client.rest_client.get_order_info', new_callable=AsyncMock) as mock_order_info:
                    mock_order_info.return_value = None

                    hanging = await recovery_service._check_hanging_orders()

                    assert len(hanging) == 1
                    assert hanging[0]["issue"]["type"] == "not_found_on_exchange"

    @pytest.mark.asyncio
    async def test_check_hanging_orders_status_mismatch(self, mock_order):
        """Тест обнаружения расхождения статусов."""
        mock_order.status = OrderStatus.PLACED

        with patch('infrastructure.repositories.order_repository.order_repository.get_active_orders',
                   new_callable=AsyncMock, return_value=[mock_order]):
            with patch('exchange.rest_client.rest_client.get_open_orders', new_callable=AsyncMock) as mock_exchange:
                mock_exchange.return_value = {
                    "result": {"list": []}
                }

                with patch('exchange.rest_client.rest_client.get_order_info', new_callable=AsyncMock) as mock_order_info:
                    mock_order_info.return_value = {
                        "orderLinkId": mock_order.client_order_id,
                        "orderId": mock_order.exchange_order_id,
                        "orderStatus": "Filled",
                        "cumExecQty": "0.001",
                        "avgPrice": "50000.0"
                    }

                    hanging = await recovery_service._check_hanging_orders()

                    assert len(hanging) == 1
                    assert hanging[0]["issue"]["type"] == "status_mismatch"
                    assert hanging[0]["issue"]["local_status"] == "Placed"
                    assert hanging[0]["issue"]["exchange_status"] == "Filled"

    @pytest.mark.asyncio
    async def test_check_hanging_orders_no_issues(self, mock_order):
        """Тест отсутствия зависших ордеров."""
        mock_order.status = OrderStatus.PLACED
        mock_order.updated_at = datetime.utcnow() - timedelta(minutes=5)

        with patch('infrastructure.repositories.order_repository.order_repository.get_active_orders',
                   new_callable=AsyncMock, return_value=[mock_order]):
            with patch('exchange.rest_client.rest_client.get_open_orders', new_callable=AsyncMock) as mock_exchange:
                mock_exchange.return_value = {
                    "result": {
                        "list": [{
                            "orderLinkId": mock_order.client_order_id,
                            "orderStatus": "New"
                        }]
                    }
                }

                hanging = await recovery_service._check_hanging_orders()

                assert len(hanging) == 0

    @pytest.mark.asyncio
    async def test_restore_fsm_states_orders(self, mock_order, clean_fsm_registry):
        """Тест восстановления FSM для ордеров."""
        with patch('infrastructure.repositories.order_repository.order_repository.get_active_orders',
                   new_callable=AsyncMock, return_value=[mock_order]):
            with patch('infrastructure.repositories.position_repository.position_repository.get_active_positions',
                       new_callable=AsyncMock, return_value=[]):
                result = await recovery_service._restore_fsm_states()

                assert result["orders"] == 1
                assert result["positions"] == 0

                fsm = clean_fsm_registry.get_order_fsm(mock_order.client_order_id)
                assert fsm is not None
                assert fsm.current_status == mock_order.status

    @pytest.mark.asyncio
    async def test_restore_fsm_states_positions(self, mock_position, clean_fsm_registry):
        """Тест восстановления FSM для позиций."""
        with patch('infrastructure.repositories.order_repository.order_repository.get_active_orders',
                   new_callable=AsyncMock, return_value=[]):
            with patch('infrastructure.repositories.position_repository.position_repository.get_active_positions',
                       new_callable=AsyncMock, return_value=[mock_position]):
                result = await recovery_service._restore_fsm_states()

                assert result["orders"] == 0
                assert result["positions"] == 1

                fsm = clean_fsm_registry.get_position_fsm(str(mock_position.id))
                assert fsm is not None
                assert fsm.current_status == mock_position.status

    @pytest.mark.asyncio
    async def test_restore_fsm_states_with_history(self, mock_order, clean_fsm_registry):
        """Тест восстановления FSM с историей переходов."""
        mock_order.metadata_json = {
            "transition_history": [
                {"from": "Pending", "to": "Placed", "timestamp": "2024-01-01T10:00:00"}
            ]
        }

        with patch('infrastructure.repositories.order_repository.order_repository.get_active_orders',
                   new_callable=AsyncMock, return_value=[mock_order]):
            with patch('infrastructure.repositories.position_repository.position_repository.get_active_positions',
                       new_callable=AsyncMock, return_value=[]):
                result = await recovery_service._restore_fsm_states()

                fsm = clean_fsm_registry.get_order_fsm(mock_order.client_order_id)
                assert fsm is not None
                assert len(fsm.transition_history) == 1

    @pytest.mark.asyncio
    async def test_recover_from_crash_full_cycle(
        self,
        mock_order,
        mock_position,
        clean_fsm_registry
    ):
        """Тест полного цикла восстановления после краша."""
        with patch('infrastructure.repositories.order_repository.order_repository.get_active_orders',
                   new_callable=AsyncMock, return_value=[mock_order]):
            with patch('infrastructure.repositories.position_repository.position_repository.get_active_positions',
                       new_callable=AsyncMock, return_value=[mock_position]):
                with patch('exchange.rest_client.rest_client.get_open_orders', new_callable=AsyncMock) as mock_exchange_orders:
                    mock_exchange_orders.return_value = {
                        "result": {"list": []}
                    }

                    with patch('exchange.rest_client.rest_client.get_positions', new_callable=AsyncMock) as mock_exchange_positions:
                        mock_exchange_positions.return_value = {
                            "result": {"list": []}
                        }

                        with patch('infrastructure.repositories.audit_repository.audit_repository.log',
                                   new_callable=AsyncMock) as mock_audit:
                            mock_audit.return_value = None

                            result = await recovery_service.recover_from_crash()

                            assert result["recovered"] is True
                            assert "State reconciliation completed" in result["actions_taken"]
                            assert result["fsm_restored"]["orders"] == 1
                            assert result["fsm_restored"]["positions"] == 1

                            order_fsm = clean_fsm_registry.get_order_fsm(mock_order.client_order_id)
                            assert order_fsm is not None

                            position_fsm = clean_fsm_registry.get_position_fsm(str(mock_position.id))
                            assert position_fsm is not None


# ==================== INTEGRATION ТЕСТЫ ====================

class TestRecoveryIntegration:
    """Integration тесты для Recovery Service."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_recovery_workflow(self):
        """Интеграционный тест полного workflow восстановления."""
        # TODO: Реализовать после настройки тестовой БД
        pass

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_hanging_order_detection_real_scenario(self):
        """Интеграционный тест обнаружения зависших ордеров."""
        # TODO: Реализовать после настройки тестовой БД и мокирования API
        pass


# ==================== ЗАПУСК ТЕСТОВ ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])