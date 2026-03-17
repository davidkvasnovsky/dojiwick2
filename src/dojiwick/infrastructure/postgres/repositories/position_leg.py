"""PostgreSQL position leg repository."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from dojiwick.domain.enums import PositionSide
from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.position_leg import PositionLeg

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO position_legs (
    account, instrument_id, position_side, quantity, entry_price,
    unrealized_pnl, leverage, liquidation_price, opened_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id
"""

_SELECT_ACTIVE_SQL = """
SELECT id, account, instrument_id, position_side, quantity, entry_price,
       unrealized_pnl, leverage, liquidation_price, opened_at, closed_at
FROM position_legs
WHERE account = %s AND closed_at IS NULL
ORDER BY opened_at
"""

_CLOSE_SQL = """
UPDATE position_legs SET closed_at = %s, updated_at = now()
WHERE id = %s AND closed_at IS NULL
"""

_SELECT_BY_ID_SQL = """
SELECT id, account, instrument_id, position_side, quantity, entry_price,
       unrealized_pnl, leverage, liquidation_price, opened_at, closed_at
FROM position_legs
WHERE id = %s
"""

_UPDATE_LEG_SQL = """
UPDATE position_legs SET quantity = %s, entry_price = %s, updated_at = now()
WHERE id = %s AND closed_at IS NULL
"""

_SELECT_ACTIVE_LEG_SQL = """
SELECT id, account, instrument_id, position_side, quantity, entry_price,
       unrealized_pnl, leverage, liquidation_price, opened_at, closed_at
FROM position_legs
WHERE account = %s AND instrument_id = %s AND position_side = %s AND closed_at IS NULL
LIMIT 1
"""


def _row_to_leg(row: tuple[object, ...]) -> PositionLeg:
    """Map a DB row to PositionLeg."""
    (
        db_id,
        account,
        instrument_id,
        position_side,
        quantity,
        entry_price,
        unrealized_pnl,
        leverage,
        liquidation_price,
        opened_at,
        closed_at,
    ) = row
    if isinstance(opened_at, str):
        opened_at = datetime.fromisoformat(opened_at)
    if isinstance(opened_at, datetime) and opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=UTC)
    if isinstance(closed_at, str):
        closed_at = datetime.fromisoformat(closed_at)
    if isinstance(closed_at, datetime) and closed_at.tzinfo is None:
        closed_at = closed_at.replace(tzinfo=UTC)
    return PositionLeg(
        id=int(str(db_id)),
        account=str(account),
        instrument_id=int(str(instrument_id)),
        position_side=PositionSide(str(position_side)),
        quantity=Decimal(str(quantity)),
        entry_price=Decimal(str(entry_price)),
        unrealized_pnl=Decimal(str(unrealized_pnl)),
        leverage=int(str(leverage)),
        liquidation_price=Decimal(str(liquidation_price)) if liquidation_price is not None else None,
        opened_at=opened_at if isinstance(opened_at, datetime) else None,
        closed_at=closed_at if isinstance(closed_at, datetime) else None,
    )


@dataclass(slots=True)
class PgPositionLegRepository:
    """Persists position legs into PostgreSQL."""

    connection: DbConnection

    async def insert_leg(self, leg: PositionLeg) -> int:
        """Insert a position leg and return the DB-assigned id."""
        row = (
            leg.account,
            leg.instrument_id,
            leg.position_side.value,
            leg.quantity,
            leg.entry_price,
            leg.unrealized_pnl,
            leg.leverage,
            leg.liquidation_price,
            leg.opened_at.isoformat() if leg.opened_at else None,
        )
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_INSERT_SQL, row)
                result = await cursor.fetchone()
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to insert position leg: {exc}") from exc
        if result is None:
            raise AdapterError("INSERT position_legs returned no id")
        return int(result[0])

    async def get_active_legs(self, account: str) -> tuple[PositionLeg, ...]:
        """Return all active (unclosed) legs for an account."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_ACTIVE_SQL, (account,))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get active legs: {exc}") from exc
        return tuple(_row_to_leg(r) for r in rows)

    async def close_leg(self, leg_id: int, closed_at: datetime) -> None:
        """Mark a position leg as closed."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_CLOSE_SQL, (closed_at.isoformat(), leg_id))
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to close position leg: {exc}") from exc

    async def get_leg(self, leg_id: int) -> PositionLeg | None:
        """Return a position leg by id, or None."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_ID_SQL, (leg_id,))
                row = await cursor.fetchone()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get position leg: {exc}") from exc
        if row is None:
            return None
        return _row_to_leg(row)

    async def update_leg(self, leg_id: int, quantity: Decimal, entry_price: Decimal) -> None:
        """Update quantity and entry price of an active leg."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_UPDATE_LEG_SQL, (quantity, entry_price, leg_id))
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to update position leg: {exc}") from exc

    async def get_active_leg(self, account: str, instrument_id: int, position_side: PositionSide) -> PositionLeg | None:
        """Return the active leg for (account, instrument, side), or None."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_ACTIVE_LEG_SQL, (account, instrument_id, position_side.value))
                row = await cursor.fetchone()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get active leg: {exc}") from exc
        if row is None:
            return None
        return _row_to_leg(row)
