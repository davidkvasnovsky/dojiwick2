"""PostgreSQL position event repository."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from dojiwick.domain.enums import PositionEventType
from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.position_leg import PositionEventRecord

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO position_events (position_leg_id, event_type, quantity, price, realized_pnl, occurred_at)
VALUES (%s, %s::position_event_type, %s, %s, %s, %s)
RETURNING id
"""

_SELECT_BY_LEG_SQL = """
SELECT id, position_leg_id, event_type, quantity, price, realized_pnl, occurred_at
FROM position_events
WHERE position_leg_id = %s
ORDER BY occurred_at
"""


def _row_to_event(row: tuple[object, ...]) -> PositionEventRecord:
    """Map a DB row to PositionEventRecord."""
    (db_id, position_leg_id, event_type, quantity, price, realized_pnl, occurred_at) = row
    if isinstance(occurred_at, str):
        occurred_at = datetime.fromisoformat(occurred_at)
    if isinstance(occurred_at, datetime) and occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=UTC)
    return PositionEventRecord(
        id=int(str(db_id)),
        position_leg_id=int(str(position_leg_id)),
        event_type=PositionEventType(str(event_type)),
        quantity=Decimal(str(quantity)),
        price=Decimal(str(price)),
        realized_pnl=Decimal(str(realized_pnl)) if realized_pnl is not None else None,
        occurred_at=occurred_at if isinstance(occurred_at, datetime) else None,
    )


@dataclass(slots=True)
class PgPositionEventRepository:
    """Persists position lifecycle events into PostgreSQL."""

    connection: DbConnection

    async def record_event(self, event: PositionEventRecord) -> int:
        """Persist a position event and return the DB-assigned id."""
        row = (
            event.position_leg_id,
            event.event_type,
            event.quantity,
            event.price,
            event.realized_pnl,
            event.occurred_at.isoformat() if event.occurred_at else None,
        )
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_INSERT_SQL, row)
                result = await cursor.fetchone()
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to record position event: {exc}") from exc
        if result is None:
            raise AdapterError("INSERT position_events returned no id")
        return int(result[0])

    async def get_events_for_leg(self, position_leg_id: int) -> tuple[PositionEventRecord, ...]:
        """Return all events for a position leg, ordered by occurred_at."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_LEG_SQL, (position_leg_id,))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get position events: {exc}") from exc
        return tuple(_row_to_event(r) for r in rows)
