"""
Тесты для компонентов Фазы 0.
"""

import pytest
from domain.state_machines.order_fsm import OrderStateMachine
from database.models import OrderStatus


@pytest.mark.asyncio
async def test_order_fsm_transitions():
  """Тест переходов Order FSM."""
  fsm = OrderStateMachine("test_order", OrderStatus.PENDING)

  # Корректный переход
  assert fsm.update_status(OrderStatus.PLACED)
  assert fsm.current_status == OrderStatus.PLACED

  # Некорректный переход
  assert not fsm.update_status(OrderStatus.REJECTED)
  assert fsm.current_status == OrderStatus.PLACED


@pytest.mark.asyncio
async def test_idempotency():
  """Тест идемпотентности."""
  from domain.services.idempotency_service import idempotency_service

  # Генерация client_order_id
  coid = idempotency_service.generate_client_order_id(
    symbol="BTCUSDT",
    side="Buy",
    quantity=0.001
  )

  assert "BTCUSDT" in coid
  assert "Buy" in coid
  assert len(coid) > 20