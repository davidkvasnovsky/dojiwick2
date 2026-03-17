"""Integration tests for PgFillRepository."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from dojiwick.domain.enums import OrderSide, OrderType
from dojiwick.domain.models.value_objects.order_request import Fill, OrderRequest

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.fill import PgFillRepository

    return PgFillRepository(connection=db_connection)


@pytest.fixture
def request_repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.order_request import PgOrderRequestRepository

    return PgOrderRequestRepository(connection=db_connection)


async def _insert_order_request(request_repo: Any, instrument_id: int) -> int:
    req = OrderRequest(
        client_order_id=f"fill-test-{datetime.now(UTC).timestamp()}",
        instrument_id=instrument_id,
        account="test-account",
        venue="binance",
        product="usd_c",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
    )
    return await request_repo.insert_request(req)


async def test_insert_and_get_fills(repo: Any, request_repo: Any, test_instrument_id: int) -> None:
    req_id = await _insert_order_request(request_repo, test_instrument_id)
    fill = Fill(
        order_request_id=req_id,
        fill_id="fill-1",
        price=Decimal("50000"),
        quantity=Decimal("0.05"),
        commission=Decimal("0.01"),
        commission_asset="USDC",
        filled_at=datetime.now(UTC),
    )
    fill_id = await repo.insert_fill(fill)
    assert fill_id > 0

    fills = await repo.get_fills_for_order(req_id)
    assert len(fills) == 1
    assert fills[0].price == Decimal("50000")
    assert fills[0].commission == Decimal("0.01")


async def test_get_fills_empty(repo: Any, test_instrument_id: int) -> None:
    fills = await repo.get_fills_for_order(999999)
    assert fills == ()
