"""
Database models package.
Re-exports all models from ../models.py (the file, not this package).
"""

# Import from the models.py FILE (not this directory)
# Using relative import from parent package
import sys
from pathlib import Path

# Add parent directory to path to import models.py file
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    # Import from models.py file in parent directory
    from models import (
        # Enums
        OrderSide,
        OrderType,
        OrderStatus,
        PositionStatus,
        AuditAction,
        BacktestStatus,

        # Models
        Order,
        Position,
        Trade,
        AuditLog,
        IdempotencyCache,
        MarketDataSnapshot,
        LayeringPattern,
        BacktestRun,
        BacktestTrade,
        BacktestEquity,
    )
finally:
    # Remove from path
    sys.path.pop(0)

__all__ = [
    # Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "PositionStatus",
    "AuditAction",
    "BacktestStatus",

    # Models
    "Order",
    "Position",
    "Trade",
    "AuditLog",
    "IdempotencyCache",
    "MarketDataSnapshot",
    "LayeringPattern",
    "BacktestRun",
    "BacktestTrade",
    "BacktestEquity",
]
