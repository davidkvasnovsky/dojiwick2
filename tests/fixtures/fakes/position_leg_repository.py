"""Fake position leg repository for tests."""

from dataclasses import dataclass, field, replace
from datetime import datetime

from dojiwick.domain.enums import PositionSide
from dojiwick.domain.models.value_objects.position_leg import PositionLeg
from dojiwick.domain.numerics import Price, Quantity


@dataclass(slots=True)
class FakePositionLegRepo:
    """In-memory position leg repository with update_leg and get_active_leg."""

    legs: dict[int, PositionLeg] = field(default_factory=dict)
    _next_id: int = 1

    async def insert_leg(self, leg: PositionLeg) -> int:
        db_id = self._next_id
        self._next_id += 1
        self.legs[db_id] = replace(leg, id=db_id)
        return db_id

    async def get_active_legs(self, account: str) -> tuple[PositionLeg, ...]:
        return tuple(leg for leg in self.legs.values() if leg.account == account and leg.closed_at is None)

    async def close_leg(self, leg_id: int, closed_at: datetime) -> None:
        leg = self.legs[leg_id]
        self.legs[leg_id] = replace(leg, closed_at=closed_at)

    async def get_leg(self, leg_id: int) -> PositionLeg | None:
        return self.legs.get(leg_id)

    async def update_leg(self, leg_id: int, quantity: Quantity, entry_price: Price) -> None:
        leg = self.legs[leg_id]
        self.legs[leg_id] = replace(leg, quantity=quantity, entry_price=entry_price)

    async def get_active_leg(self, account: str, instrument_id: int, position_side: PositionSide) -> PositionLeg | None:
        for leg in self.legs.values():
            if (
                leg.account == account
                and leg.instrument_id == instrument_id
                and leg.position_side == position_side
                and leg.closed_at is None
            ):
                return leg
        return None
