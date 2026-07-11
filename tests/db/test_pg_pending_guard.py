"""Pending-order guard SQL: protective/reduce-only rows must not count as pending entries."""

from decimal import Decimal
from typing import Any

import pytest

from dojiwick.domain.enums import OrderKind, OrderSide, OrderType, PositionSide
from dojiwick.domain.models.value_objects.order_request import OrderReport, OrderRequest

pytestmark = pytest.mark.db


@pytest.fixture
def request_repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.order_request import PgOrderRequestRepository

    return PgOrderRequestRepository(connection=db_connection)


@pytest.fixture
def report_repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.order_report import PgOrderReportRepository

    return PgOrderReportRepository(connection=db_connection)


@pytest.fixture
def provider(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.pending_order_provider import PgPendingOrderProvider

    return PgPendingOrderProvider(connection=db_connection)


def _request(
    instrument_id: int,
    client_order_id: str,
    *,
    order_kind: OrderKind = OrderKind.ENTRY,
    reduce_only: bool = False,
) -> OrderRequest:
    return OrderRequest(
        client_order_id=client_order_id,
        instrument_id=instrument_id,
        account="test-account",
        venue="binance",
        product="usd_c",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.020"),
        reduce_only=reduce_only,
        order_kind=order_kind,
    )


async def _seed_open_order(
    request_repo: Any,
    report_repo: Any,
    instrument_id: int,
    client_order_id: str,
    *,
    order_kind: OrderKind = OrderKind.ENTRY,
    reduce_only: bool = False,
) -> None:
    from datetime import UTC, datetime

    from dojiwick.domain.enums import OrderStatus

    db_id = await request_repo.insert_request(
        _request(instrument_id, client_order_id, order_kind=order_kind, reduce_only=reduce_only)
    )
    await report_repo.upsert_report(
        OrderReport(
            order_request_id=db_id,
            exchange_order_id=f"ex-{client_order_id}",
            status=OrderStatus.NEW,
            filled_qty=Decimal(0),
            avg_price=None,
            reported_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )


async def test_protective_and_reduce_only_rows_excluded(
    request_repo: Any, report_repo: Any, provider: Any, test_instrument_id: int
) -> None:
    await _seed_open_order(request_repo, report_repo, test_instrument_id, "entry-1")
    await _seed_open_order(
        request_repo,
        report_repo,
        test_instrument_id,
        "prot-stop-1",
        order_kind=OrderKind.PROTECTIVE_STOP,
        reduce_only=True,
    )
    await _seed_open_order(
        request_repo,
        report_repo,
        test_instrument_id,
        "prot-tp-1",
        order_kind=OrderKind.PROTECTIVE_TP,
        reduce_only=True,
    )

    pending = await provider.get_pending_quantities("test-account")

    assert pending == {("BTCUSDC", PositionSide.NET): Decimal("0.020")}, (
        "only the entry order counts toward pending quantities"
    )
