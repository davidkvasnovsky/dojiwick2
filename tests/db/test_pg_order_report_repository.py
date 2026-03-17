"""Integration tests for PgOrderReportRepository."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from dojiwick.domain.enums import OrderSide, OrderStatus, OrderType
from dojiwick.domain.models.value_objects.order_request import OrderReport, OrderRequest

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.order_report import PgOrderReportRepository

    return PgOrderReportRepository(connection=db_connection)


@pytest.fixture
def request_repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.order_request import PgOrderRequestRepository

    return PgOrderRequestRepository(connection=db_connection)


async def _insert_order_request(request_repo: Any, instrument_id: int) -> int:
    """Helper: insert an order request and return its id."""
    req = OrderRequest(
        client_order_id=f"report-test-{datetime.now(UTC).timestamp()}",
        instrument_id=instrument_id,
        account="test-account",
        venue="binance",
        product="usd_c",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.1"),
        price=Decimal("50000"),
    )
    return await request_repo.insert_request(req)


async def test_upsert_and_get_by_exchange_order_id(repo: Any, request_repo: Any, test_instrument_id: int) -> None:
    req_id = await _insert_order_request(request_repo, test_instrument_id)
    report = OrderReport(
        order_request_id=req_id,
        exchange_order_id="exc-order-1",
        status=OrderStatus.NEW,
        filled_qty=Decimal("0"),
        reported_at=datetime.now(UTC),
    )
    db_id = await repo.upsert_report(report)
    assert db_id > 0

    loaded = await repo.get_by_exchange_order_id("exc-order-1")
    assert loaded is not None
    assert loaded.status == OrderStatus.NEW


async def test_upsert_updates_existing(repo: Any, request_repo: Any, test_instrument_id: int) -> None:
    req_id = await _insert_order_request(request_repo, test_instrument_id)
    report = OrderReport(
        order_request_id=req_id,
        exchange_order_id="exc-upd-1",
        status=OrderStatus.NEW,
        filled_qty=Decimal("0"),
        reported_at=datetime.now(UTC),
    )
    await repo.upsert_report(report)

    updated = OrderReport(
        order_request_id=req_id,
        exchange_order_id="exc-upd-1",
        status=OrderStatus.FILLED,
        filled_qty=Decimal("0.1"),
        avg_price=Decimal("50000"),
        reported_at=datetime.now(UTC),
    )
    await repo.upsert_report(updated)

    loaded = await repo.get_by_exchange_order_id("exc-upd-1")
    assert loaded is not None
    assert loaded.status == OrderStatus.FILLED
    assert loaded.filled_qty == Decimal("0.1")


async def test_get_by_exchange_order_ids(repo: Any, request_repo: Any, test_instrument_id: int) -> None:
    req_id = await _insert_order_request(request_repo, test_instrument_id)
    await repo.upsert_report(OrderReport(order_request_id=req_id, exchange_order_id="bulk-1", status=OrderStatus.NEW))
    await repo.upsert_report(
        OrderReport(order_request_id=req_id, exchange_order_id="bulk-2", status=OrderStatus.FILLED)
    )

    result = await repo.get_by_exchange_order_ids(["bulk-1", "bulk-2", "missing"])

    assert len(result) == 2
    assert "bulk-1" in result
    assert "bulk-2" in result
    assert "missing" not in result
    assert result["bulk-1"].status == OrderStatus.NEW
    assert result["bulk-2"].status == OrderStatus.FILLED

    assert await repo.get_by_exchange_order_ids([]) == {}
