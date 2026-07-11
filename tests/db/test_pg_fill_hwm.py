"""FOR UPDATE high-water-mark semantics of advance_applied_qty against real Postgres."""

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


def _request(instrument_id: int) -> OrderRequest:
    return OrderRequest(
        client_order_id="hwm-order-1",
        instrument_id=instrument_id,
        account="test-account",
        venue="binance",
        product="usd_c",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.010"),
    )


async def test_advance_applied_qty_returns_delta_once(repo: Any, test_instrument_id: int) -> None:
    """Sequential advances: first returns the delta, a smaller/equal cumulative returns zero."""
    order_id = await repo.insert_request(_request(test_instrument_id))

    first = await repo.advance_applied_qty(order_id, Decimal("0.005"))
    assert first == Decimal("0.005")

    duplicate = await repo.advance_applied_qty(order_id, Decimal("0.005"))
    assert duplicate == Decimal(0)

    stale_replay = await repo.advance_applied_qty(order_id, Decimal("0.003"))
    assert stale_replay == Decimal(0)

    progression = await repo.advance_applied_qty(order_id, Decimal("0.010"))
    assert progression == Decimal("0.005")


async def test_advance_applied_qty_unknown_order_raises(repo: Any, test_instrument_id: int) -> None:
    from dojiwick.domain.errors import AdapterError

    with pytest.raises(AdapterError, match="not found"):
        await repo.advance_applied_qty(999_999, Decimal("0.01"))
