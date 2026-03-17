"""Fake order report repository for tests."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from dojiwick.domain.models.value_objects.order_request import OrderReport


@dataclass(slots=True)
class FakeOrderReportRepo:
    """In-memory order report repository with upsert by exchange_order_id."""

    reports: list[OrderReport] = field(default_factory=list)
    _next_id: int = 1

    async def upsert_report(self, report: OrderReport) -> int:
        for i, existing in enumerate(self.reports):
            if existing.exchange_order_id == report.exchange_order_id:
                db_id = existing.id or self._next_id
                self.reports[i] = replace(report, id=db_id)
                return db_id
        db_id = self._next_id
        self._next_id += 1
        self.reports.append(replace(report, id=db_id))
        return db_id

    async def get_by_exchange_order_id(self, exchange_order_id: str) -> OrderReport | None:
        for r in self.reports:
            if r.exchange_order_id == exchange_order_id:
                return r
        return None

    async def get_by_exchange_order_ids(
        self,
        exchange_order_ids: Sequence[str],
    ) -> dict[str, OrderReport]:
        ids = set(exchange_order_ids)
        return {r.exchange_order_id: r for r in self.reports if r.exchange_order_id in ids}

    async def get_reports_for_request(self, order_request_id: int) -> tuple[OrderReport, ...]:
        return tuple(r for r in self.reports if r.order_request_id == order_request_id)
