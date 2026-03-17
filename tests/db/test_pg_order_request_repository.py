"""Integration tests for PgOrderRequestRepository."""

from decimal import Decimal
from typing import Any

import pytest

from dojiwick.domain.enums import OrderSide, OrderType
from dojiwick.domain.models.value_objects.order_request import OrderRequest

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.order_request import PgOrderRequestRepository

    return PgOrderRequestRepository(connection=db_connection)


def _make_request(instrument_id: int, client_order_id: str = "test-client-order-1") -> OrderRequest:
    return OrderRequest(
        client_order_id=client_order_id,
        instrument_id=instrument_id,
        account="test-account",
        venue="binance",
        product="usd_c",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.1"),
        price=Decimal("50000"),
    )


async def test_insert_and_get_by_client_order_id(repo: Any, test_instrument_id: int) -> None:
    req = _make_request(test_instrument_id)
    db_id = await repo.insert_request(req)
    assert db_id > 0

    loaded = await repo.get_by_client_order_id("test-client-order-1")
    assert loaded is not None
    assert loaded.side == OrderSide.BUY
    assert loaded.quantity == Decimal("0.1")


async def test_get_by_client_order_id_not_found(repo: Any, test_instrument_id: int) -> None:
    loaded = await repo.get_by_client_order_id("nonexistent")
    assert loaded is None
