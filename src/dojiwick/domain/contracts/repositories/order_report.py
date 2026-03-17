"""Order report repository protocol."""

from collections.abc import Sequence
from typing import Protocol

from dojiwick.domain.models.value_objects.order_request import OrderReport


class OrderReportRepositoryPort(Protocol):
    """Order report (exchange-acknowledged state) persistence."""

    async def upsert_report(self, report: OrderReport) -> int:
        """Insert or update an order report by exchange_order_id, returning the id."""
        ...

    async def get_by_exchange_order_id(self, exchange_order_id: str) -> OrderReport | None:
        """Return an order report by exchange_order_id, or None."""
        ...

    async def get_by_exchange_order_ids(
        self,
        exchange_order_ids: Sequence[str],
    ) -> dict[str, OrderReport]:
        """Return reports keyed by exchange_order_id. Missing IDs omitted."""
        ...

    async def get_reports_for_request(self, order_request_id: int) -> tuple[OrderReport, ...]:
        """Return all reports for an order request."""
        ...
