"""PostgreSQL fill repository."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.order_request import Fill

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO fills (order_request_id, fill_id, price, quantity, commission, commission_asset, realized_pnl_exchange, filled_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (order_request_id, fill_id) WHERE fill_id != '' DO NOTHING
RETURNING id
"""

_SELECT_BY_ORDER_SQL = """
SELECT id, order_request_id, fill_id, price, quantity, commission, commission_asset, realized_pnl_exchange, filled_at
FROM fills
WHERE order_request_id = %s
ORDER BY filled_at
"""


def _row_to_fill(row: tuple[object, ...]) -> Fill:
    """Map a DB row to Fill."""
    (
        db_id,
        order_request_id,
        fill_id,
        price,
        quantity,
        commission,
        commission_asset,
        realized_pnl_exchange,
        filled_at,
    ) = row
    if isinstance(filled_at, str):
        filled_at = datetime.fromisoformat(filled_at)
    if isinstance(filled_at, datetime) and filled_at.tzinfo is None:
        filled_at = filled_at.replace(tzinfo=UTC)
    return Fill(
        id=int(str(db_id)),
        order_request_id=int(str(order_request_id)),
        fill_id=str(fill_id),
        price=Decimal(str(price)),
        quantity=Decimal(str(quantity)),
        commission=Decimal(str(commission)),
        commission_asset=str(commission_asset),
        realized_pnl_exchange=Decimal(str(realized_pnl_exchange)) if realized_pnl_exchange is not None else None,
        filled_at=filled_at if isinstance(filled_at, datetime) else None,
    )


@dataclass(slots=True)
class PgFillRepository:
    """Persists fill events into PostgreSQL."""

    connection: DbConnection

    async def insert_fill(self, fill: Fill) -> int | None:
        """Persist a fill event and return the DB-assigned id.

        Returns None if the fill was a duplicate (ON CONFLICT DO NOTHING).
        """
        row = (
            fill.order_request_id,
            fill.fill_id,
            fill.price,
            fill.quantity,
            fill.commission,
            fill.commission_asset,
            fill.realized_pnl_exchange,
            fill.filled_at.isoformat() if fill.filled_at else None,
        )
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_INSERT_SQL, row)
                result = await cursor.fetchone()
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to insert fill: {exc}") from exc
        if result is None:
            return None
        return int(result[0])

    async def get_fills_for_order(self, order_request_id: int) -> tuple[Fill, ...]:
        """Return all fills for an order request, ordered by filled_at."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_ORDER_SQL, (order_request_id,))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get fills: {exc}") from exc
        return tuple(_row_to_fill(r) for r in rows)
