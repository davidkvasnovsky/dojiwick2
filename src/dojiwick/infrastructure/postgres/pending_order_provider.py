"""PostgreSQL pending order provider — single JOIN query for in-flight quantities."""

from dataclasses import dataclass
from decimal import Decimal

from dojiwick.domain.enums import PositionSide
from dojiwick.domain.errors import AdapterError
from dojiwick.domain.numerics import Quantity

from dojiwick.infrastructure.postgres.connection import DbConnection

_PENDING_QTY_SQL = """
SELECT i.symbol, req.position_side,
       SUM(req.quantity - COALESCE(rpt.filled_qty, 0)) AS pending_qty
FROM order_requests req
JOIN order_reports rpt ON rpt.order_request_id = req.id
JOIN instruments i ON i.id = req.instrument_id
WHERE req.account = %s
  AND rpt.status IN ('new', 'partially_filled')
GROUP BY i.symbol, req.position_side
HAVING SUM(req.quantity - COALESCE(rpt.filled_qty, 0)) > 0
"""


@dataclass(slots=True)
class PgPendingOrderProvider:
    """Queries pending order quantities from PostgreSQL via a single JOIN."""

    connection: DbConnection

    async def get_pending_quantities(
        self,
        account: str,
    ) -> dict[tuple[str, PositionSide], Quantity]:
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_PENDING_QTY_SQL, (account,))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to query pending orders: {exc}") from exc

        result: dict[tuple[str, PositionSide], Quantity] = {}
        for row in rows:
            symbol = str(row[0])
            position_side = PositionSide(str(row[1]))
            pending_qty = Decimal(str(row[2]))
            result[(symbol, position_side)] = pending_qty
        return result
