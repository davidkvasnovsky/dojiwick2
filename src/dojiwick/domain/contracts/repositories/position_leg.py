"""Position leg repository protocol."""

from datetime import datetime
from typing import Protocol

from dojiwick.domain.enums import PositionSide
from dojiwick.domain.models.value_objects.position_leg import PositionLeg
from dojiwick.domain.numerics import Price, Quantity


class PositionLegRepositoryPort(Protocol):
    """Hedge-native position leg persistence."""

    async def insert_leg(self, leg: PositionLeg) -> int:
        """Insert a position leg and return the DB-assigned id."""
        ...

    async def get_active_legs(self, account: str) -> tuple[PositionLeg, ...]:
        """Return all active (unclosed) legs for an account."""
        ...

    async def close_leg(self, leg_id: int, closed_at: datetime) -> None:
        """Mark a position leg as closed."""
        ...

    async def get_leg(self, leg_id: int) -> PositionLeg | None:
        """Return a position leg by id, or None."""
        ...

    async def update_leg(self, leg_id: int, quantity: Quantity, entry_price: Price) -> None:
        """Update quantity and entry price of an active leg."""
        ...

    async def get_active_leg(self, account: str, instrument_id: int, position_side: PositionSide) -> PositionLeg | None:
        """Return the active leg for (account, instrument, side), or None."""
        ...
