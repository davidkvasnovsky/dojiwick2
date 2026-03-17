"""PostgreSQL order report repository."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from dojiwick.domain.enums import OrderStatus
from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.order_request import OrderReport

from dojiwick.infrastructure.postgres.connection import DbConnection

_UPSERT_SQL = """
INSERT INTO order_reports (
    order_request_id, exchange_order_id, status, filled_qty,
    avg_price, cumulative_quote_qty, reported_at
) VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT ON CONSTRAINT order_reports_unique_exchange_order
DO UPDATE SET
    status = EXCLUDED.status,
    filled_qty = EXCLUDED.filled_qty,
    avg_price = EXCLUDED.avg_price,
    cumulative_quote_qty = EXCLUDED.cumulative_quote_qty,
    reported_at = EXCLUDED.reported_at,
    updated_at = now()
RETURNING id
"""

_SELECT_BY_EXCHANGE_ORDER_ID_SQL = """
SELECT id, order_request_id, exchange_order_id, status, filled_qty,
       avg_price, cumulative_quote_qty, reported_at
FROM order_reports
WHERE exchange_order_id = %s
"""

_SELECT_BY_EXCHANGE_ORDER_IDS_SQL = """
SELECT id, order_request_id, exchange_order_id, status, filled_qty,
       avg_price, cumulative_quote_qty, reported_at
FROM order_reports
WHERE exchange_order_id = ANY(%s)
"""

_SELECT_BY_REQUEST_SQL = """
SELECT id, order_request_id, exchange_order_id, status, filled_qty,
       avg_price, cumulative_quote_qty, reported_at
FROM order_reports
WHERE order_request_id = %s
ORDER BY reported_at
"""


def _row_to_report(row: tuple[object, ...]) -> OrderReport:
    """Map a DB row to OrderReport."""
    (db_id, order_request_id, exchange_order_id, status, filled_qty, avg_price, cumulative_quote_qty, reported_at) = row
    if isinstance(reported_at, str):
        reported_at = datetime.fromisoformat(reported_at)
    if isinstance(reported_at, datetime) and reported_at.tzinfo is None:
        reported_at = reported_at.replace(tzinfo=UTC)
    return OrderReport(
        id=int(str(db_id)),
        order_request_id=int(str(order_request_id)),
        exchange_order_id=str(exchange_order_id),
        status=OrderStatus(str(status)),
        filled_qty=Decimal(str(filled_qty)),
        avg_price=Decimal(str(avg_price)) if avg_price is not None else None,
        cumulative_quote_qty=Decimal(str(cumulative_quote_qty)) if cumulative_quote_qty is not None else None,
        reported_at=reported_at if isinstance(reported_at, datetime) else None,
    )


@dataclass(slots=True)
class PgOrderReportRepository:
    """Persists order reports into PostgreSQL."""

    connection: DbConnection

    async def upsert_report(self, report: OrderReport) -> int:
        """Insert or update an order report by exchange_order_id, returning the id."""
        row = (
            report.order_request_id,
            report.exchange_order_id,
            report.status.value,
            report.filled_qty,
            report.avg_price,
            report.cumulative_quote_qty,
            report.reported_at.isoformat() if report.reported_at else None,
        )
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_UPSERT_SQL, row)
                result = await cursor.fetchone()
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to upsert order report: {exc}") from exc
        if result is None:
            raise AdapterError("upsert order_reports returned no id")
        return int(result[0])

    async def get_by_exchange_order_id(self, exchange_order_id: str) -> OrderReport | None:
        """Return an order report by exchange_order_id, or None."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_EXCHANGE_ORDER_ID_SQL, (exchange_order_id,))
                row = await cursor.fetchone()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get order report: {exc}") from exc
        if row is None:
            return None
        return _row_to_report(row)

    async def get_by_exchange_order_ids(
        self,
        exchange_order_ids: Sequence[str],
    ) -> dict[str, OrderReport]:
        if not exchange_order_ids:
            return {}
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_EXCHANGE_ORDER_IDS_SQL, (list(exchange_order_ids),))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get order reports by ids: {exc}") from exc
        return {r.exchange_order_id: r for r in (_row_to_report(row) for row in rows)}

    async def get_reports_for_request(self, order_request_id: int) -> tuple[OrderReport, ...]:
        """Return all reports for an order request."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_REQUEST_SQL, (order_request_id,))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get order reports: {exc}") from exc
        return tuple(_row_to_report(r) for r in rows)
