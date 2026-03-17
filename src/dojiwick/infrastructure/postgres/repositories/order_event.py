"""PostgreSQL order event repository."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from dojiwick.domain.enums import OrderEventType
from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.order_event import OrderEvent

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO order_events (order_id, event_type, exchange_order_id, filled_quantity, fees_usd, fee_asset, native_fee_amount, realized_pnl_exchange, detail, occurred_at)
VALUES (%s, %s::order_event_type, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_SELECT_BY_ORDER_SQL = """
SELECT order_id, event_type, occurred_at, exchange_order_id, filled_quantity, fees_usd, fee_asset, native_fee_amount, realized_pnl_exchange, detail
FROM order_events
WHERE order_id = %s
ORDER BY occurred_at
"""

_SELECT_SINCE_SQL = """
SELECT order_id, event_type, occurred_at, exchange_order_id, filled_quantity, fees_usd, fee_asset, native_fee_amount, realized_pnl_exchange, detail
FROM order_events
WHERE occurred_at >= %s
ORDER BY occurred_at
"""


def _row_to_event(row: tuple[object, ...]) -> OrderEvent:
    """Map a DB row to OrderEvent."""
    (
        order_id,
        event_type,
        occurred_at,
        exchange_order_id,
        filled_quantity,
        fees_usd,
        fee_asset,
        native_fee_amount,
        realized_pnl_exchange,
        detail,
    ) = row
    if isinstance(occurred_at, str):
        occurred_at = datetime.fromisoformat(occurred_at)
    if isinstance(occurred_at, datetime) and occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=UTC)
    if not isinstance(occurred_at, datetime):
        raise AdapterError("order_event.occurred_at is not a datetime")
    return OrderEvent(
        order_id=int(str(order_id)),
        event_type=OrderEventType(str(event_type)),
        occurred_at=occurred_at,
        exchange_order_id=str(exchange_order_id),
        filled_quantity=Decimal(str(filled_quantity)),
        fees_usd=Decimal(str(fees_usd)),
        fee_asset=str(fee_asset),
        native_fee_amount=Decimal(str(native_fee_amount)),
        realized_pnl_exchange=Decimal(str(realized_pnl_exchange)) if realized_pnl_exchange is not None else None,
        detail=str(detail),
    )


@dataclass(slots=True)
class PgOrderEventRepository:
    """Persists order events into PostgreSQL."""

    connection: DbConnection

    async def record_event(self, event: OrderEvent) -> None:
        """Persist an order event."""
        row = (
            event.order_id,
            event.event_type.value,
            event.exchange_order_id,
            event.filled_quantity,
            event.fees_usd,
            event.fee_asset,
            event.native_fee_amount,
            event.realized_pnl_exchange,
            event.detail,
            event.occurred_at.isoformat(),
        )
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_INSERT_SQL, row)
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to record order event: {exc}") from exc

    async def get_events_for_order(self, order_id: int) -> tuple[OrderEvent, ...]:
        """Return all events belonging to an order."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_ORDER_SQL, (order_id,))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get order events: {exc}") from exc
        return tuple(_row_to_event(r) for r in rows)

    async def get_events_since(self, since: datetime) -> tuple[OrderEvent, ...]:
        """Return all order events since the given timestamp."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_SINCE_SQL, (since.isoformat(),))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get order events: {exc}") from exc
        return tuple(_row_to_event(r) for r in rows)
