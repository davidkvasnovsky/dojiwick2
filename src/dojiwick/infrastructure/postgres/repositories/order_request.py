"""PostgreSQL order request repository."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from dojiwick.domain.enums import OrderKind, OrderSide, OrderTimeInForce, OrderType, PositionSide, WorkingType
from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.order_request import OrderRequest

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO order_requests (
    venue, product,
    client_order_id, instrument_id, account, tick_id, side, order_type, quantity, price,
    position_side, reduce_only, close_position, time_in_force,
    working_type, price_protect, recv_window_ms, order_kind, position_leg_id
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id
"""

_SELECT_BY_CLIENT_ORDER_ID_SQL = """
SELECT id, venue, product, client_order_id, instrument_id, account, tick_id, side, order_type,
       quantity, price, position_side, reduce_only, close_position, time_in_force,
       working_type, price_protect, recv_window_ms, order_kind, position_leg_id, created_at
FROM order_requests
WHERE client_order_id = %s
"""


def _row_to_request(row: tuple[object, ...]) -> OrderRequest:
    """Map a DB row to OrderRequest."""
    (
        db_id,
        venue,
        product,
        client_order_id,
        instrument_id,
        account,
        tick_id,
        side,
        order_type,
        quantity,
        price,
        position_side,
        reduce_only,
        close_position,
        time_in_force,
        working_type,
        price_protect,
        recv_window_ms,
        order_kind,
        position_leg_id,
        created_at,
    ) = row
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    if isinstance(created_at, datetime) and created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return OrderRequest(
        id=int(str(db_id)),
        venue=str(venue),
        product=str(product),
        client_order_id=str(client_order_id),
        instrument_id=int(str(instrument_id)),
        account=str(account),
        tick_id=str(tick_id) if tick_id is not None else "",
        side=OrderSide(str(side)),
        order_type=OrderType(str(order_type)),
        quantity=Decimal(str(quantity)),
        price=Decimal(str(price)) if price is not None else None,
        position_side=PositionSide(str(position_side)),
        reduce_only=bool(reduce_only),
        close_position=bool(close_position),
        time_in_force=OrderTimeInForce(str(time_in_force)),
        working_type=WorkingType(str(working_type)),
        price_protect=bool(price_protect),
        recv_window_ms=int(str(recv_window_ms)) if recv_window_ms is not None else None,
        order_kind=OrderKind(str(order_kind)),
        position_leg_id=int(str(position_leg_id)) if position_leg_id is not None else None,
        created_at=created_at if isinstance(created_at, datetime) else None,
    )


@dataclass(slots=True)
class PgOrderRequestRepository:
    """Persists order requests into PostgreSQL."""

    connection: DbConnection

    async def insert_request(self, request: OrderRequest) -> int:
        """Persist an order request and return the DB-assigned id."""
        row = (
            request.venue,
            request.product,
            request.client_order_id,
            request.instrument_id,
            request.account,
            request.tick_id or None,
            request.side.value,
            request.order_type.value,
            request.quantity,
            request.price,
            request.position_side.value,
            request.reduce_only,
            request.close_position,
            request.time_in_force.value,
            request.working_type.value,
            request.price_protect,
            request.recv_window_ms,
            request.order_kind.value,
            request.position_leg_id,
        )
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_INSERT_SQL, row)
                result = await cursor.fetchone()
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to insert order request: {exc}") from exc
        if result is None:
            raise AdapterError("INSERT order_requests returned no id")
        return int(result[0])

    async def get_by_client_order_id(self, client_order_id: str) -> OrderRequest | None:
        """Return an order request by client_order_id, or None."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_CLIENT_ORDER_ID_SQL, (client_order_id,))
                row = await cursor.fetchone()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get order request: {exc}") from exc
        if row is None:
            return None
        return _row_to_request(row)

    async def advance_applied_qty(self, order_request_id: int, cumulative_qty: Decimal) -> Decimal:
        """Advance the position-applied high-water mark, returning the unapplied delta.

        FOR UPDATE serializes the tick (REST receipt) and consumer (WS event)
        paths: whichever arrives second sees the mark already advanced and
        gets a zero delta instead of double-applying the fill.
        """
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT position_applied_qty FROM order_requests WHERE id = %s FOR UPDATE",
                    (order_request_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise AdapterError(f"order request {order_request_id} not found")
                applied = Decimal(str(row[0]))
                delta = cumulative_qty - applied
                if delta > 0:
                    await cursor.execute(
                        "UPDATE order_requests SET position_applied_qty = %s WHERE id = %s",
                        (cumulative_qty, order_request_id),
                    )
            await self.connection.commit()
        except AdapterError:
            await self.connection.rollback()
            raise
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to advance applied qty: {exc}") from exc
        return delta if delta > 0 else Decimal(0)
