"""
Скрипт для ручного закрытия призрачных позиций.

Призрачная позиция - это позиция, которая:
- Есть в БД в статусе OPENING или OPEN
- Отсутствует на бирже (size = 0 или не найдена)

ИСПОЛЬЗОВАНИЕ:
    python close_ghost_positions.py --dry-run  # Только показать
    python close_ghost_positions.py --execute  # Закрыть призрачные позиции
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Добавляем путь к backend
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from core.logger import get_logger
from database.connection import db_manager
from database.models import PositionStatus, AuditAction
from infrastructure.repositories.position_repository import position_repository
from infrastructure.repositories.audit_repository import audit_repository
from domain.state_machines.position_fsm import PositionStateMachine
from domain.services.fsm_registry import fsm_registry
from exchange.rest_client import rest_client

logger = get_logger(__name__)


class GhostPositionCleaner:
  """Очистка призрачных позиций."""

  def __init__(self, dry_run: bool = True):
    """
    Инициализация.

    Args:
        dry_run: Если True, только показать, не закрывать
    """
    self.dry_run = dry_run
    self.ghost_positions = []

  async def find_ghost_positions(self):
    """Найти все призрачные позиции."""
    logger.info("=" * 80)
    logger.info("ПОИСК ПРИЗРАЧНЫХ ПОЗИЦИЙ")
    logger.info("=" * 80)

    # Получаем активные позиции из БД
    local_positions = await position_repository.get_active_positions()
    logger.info(f"Найдено {len(local_positions)} активных позиций в БД")

    if not local_positions:
      logger.info("✓ Активных позиций нет")
      return

    # Получаем позиции с биржи
    try:
      exchange_response = await rest_client.get_positions()
      exchange_positions_list = exchange_response.get("result", {}).get("list", [])
      logger.info(f"Получено {len(exchange_positions_list)} позиций с биржи")

    except Exception as e:
      logger.error(f"❌ Ошибка получения позиций с биржи: {e}")
      return

    # Создаем мапу позиций с биржи
    exchange_map = {}
    for pos in exchange_positions_list:
      symbol = pos.get("symbol")
      size = float(pos.get("size", 0))

      if symbol and size > 0:
        exchange_map[symbol] = pos

    # Ищем призрачные позиции
    for local_position in local_positions:
      symbol = local_position.symbol

      if symbol not in exchange_map:
        # Призрак найден!
        self.ghost_positions.append(local_position)

        logger.warning(
          f"👻 ПРИЗРАК #{len(self.ghost_positions)}: {symbol}\n"
          f"   Position ID: {local_position.id}\n"
          f"   Status: {local_position.status.value}\n"
          f"   Quantity: {local_position.quantity}\n"
          f"   Entry Price: {local_position.entry_price}\n"
          f"   Opened At: {local_position.opened_at}\n"
          f"   Age: {(datetime.utcnow() - local_position.opened_at).days} days"
        )

    logger.info("=" * 80)
    logger.info(f"ИТОГО: Найдено {len(self.ghost_positions)} призрачных позиций")
    logger.info("=" * 80)

  async def close_ghost_positions(self):
    """Закрыть все призрачные позиции."""
    if not self.ghost_positions:
      logger.info("✓ Призрачных позиций не найдено")
      return

    if self.dry_run:
      logger.info("=" * 80)
      logger.info("РЕЖИМ DRY-RUN: Позиции НЕ будут закрыты")
      logger.info("=" * 80)
      logger.info(f"Будет закрыто {len(self.ghost_positions)} позиций:")
      for i, pos in enumerate(self.ghost_positions, 1):
        print(f"  {i}. {pos.symbol} (ID: {pos.id})")
      return

    logger.info("=" * 80)
    logger.info(f"ЗАКРЫТИЕ {len(self.ghost_positions)} ПРИЗРАЧНЫХ ПОЗИЦИЙ")
    logger.info("=" * 80)

    closed_count = 0
    failed_count = 0

    for i, pos in enumerate(self.ghost_positions, 1):
      position_id = str(pos.id)
      symbol = pos.symbol

      logger.info(f"\n[{i}/{len(self.ghost_positions)}] Закрытие {symbol}...")

      try:
        # Получаем или создаем FSM
        position_fsm = fsm_registry.get_position_fsm(position_id)

        if not position_fsm:
          position_fsm = PositionStateMachine(
            position_id=position_id,
            initial_state=pos.status
          )
          fsm_registry.register_position_fsm(position_id, position_fsm)

        # Закрываем позицию
        if pos.status == PositionStatus.OPENING:
          # OPENING -> CLOSED (abort)
          position_fsm.abort()  # type: ignore[attr-defined]

          await position_repository.update_status(
            position_id=position_id,
            new_status=PositionStatus.CLOSED,
            exit_reason="Ghost position aborted (never opened on exchange)"
          )

          logger.info(f"✓ {symbol} прервана (OPENING -> CLOSED)")

        elif pos.status == PositionStatus.OPEN:
          # OPEN -> CLOSING -> CLOSED
          position_fsm.start_close()  # type: ignore[attr-defined]

          await position_repository.update_status(
            position_id=position_id,
            new_status=PositionStatus.CLOSING
          )

          position_fsm.confirm_close()  # type: ignore[attr-defined]

          await position_repository.update_status(
            position_id=position_id,
            new_status=PositionStatus.CLOSED,
            exit_price=pos.current_price or pos.entry_price,
            exit_reason="Ghost position closed (not found on exchange)"
          )

          logger.info(f"✓ {symbol} закрыта (OPEN -> CLOSED)")

        # Удаляем FSM
        fsm_registry.unregister_position_fsm(position_id)

        # Аудит
        await audit_repository.log(
          action=AuditAction.POSITION_CLOSE,
          entity_type="Position",
          entity_id=position_id,
          old_value={"status": pos.status.value},
          new_value={"status": "CLOSED"},
          reason="Ghost position closed by cleanup script",
          success=True
        )

        closed_count += 1

      except Exception as e:
        logger.error(f"❌ Ошибка закрытия {symbol}: {e}")
        failed_count += 1

    logger.info("=" * 80)
    logger.info(f"ЗАКРЫТИЕ ЗАВЕРШЕНО:")
    logger.info(f"  ✓ Закрыто: {closed_count}")
    logger.info(f"  ❌ Ошибок: {failed_count}")
    logger.info("=" * 80)

  async def run(self):
    """Запуск полного процесса."""
    try:
      # Инициализация
      await db_manager.initialize()
      await rest_client.initialize()

      # Поиск призраков
      await self.find_ghost_positions()

      # Закрытие
      await self.close_ghost_positions()

    finally:
      # Очистка
      await rest_client.close()
      await db_manager.close()


async def main():
  """Точка входа."""
  import argparse

  parser = argparse.ArgumentParser(description="Закрытие призрачных позиций")
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Только показать призрачные позиции, не закрывать"
  )
  parser.add_argument(
    "--execute",
    action="store_true",
    help="Закрыть призрачные позиции"
  )

  args = parser.parse_args()

  if not args.dry_run and not args.execute:
    print("❌ Укажите --dry-run или --execute")
    parser.print_help()
    return

  dry_run = args.dry_run

  cleaner = GhostPositionCleaner(dry_run=dry_run)
  await cleaner.run()


if __name__ == "__main__":
  asyncio.run(main())