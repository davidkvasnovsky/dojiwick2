"""Integration tests for PgPositionLegRepository."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from dojiwick.domain.enums import PositionSide
from dojiwick.domain.models.value_objects.position_leg import PositionLeg

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.position_leg import PgPositionLegRepository

    return PgPositionLegRepository(connection=db_connection)


def _make_leg(instrument_id: int, side: PositionSide = PositionSide.LONG) -> PositionLeg:
    return PositionLeg(
        account="test-account",
        instrument_id=instrument_id,
        position_side=side,
        quantity=Decimal("0.5"),
        entry_price=Decimal("50000"),
        unrealized_pnl=Decimal("100"),
        leverage=10,
        opened_at=datetime.now(UTC),
    )


async def test_insert_and_get_active(repo: Any, test_instrument_id: int) -> None:
    leg = _make_leg(test_instrument_id)
    leg_id = await repo.insert_leg(leg)
    assert leg_id > 0

    active = await repo.get_active_legs("test-account")
    assert len(active) >= 1
    assert any(a.id == leg_id for a in active)


async def test_close_leg(repo: Any, test_instrument_id: int) -> None:
    leg = _make_leg(test_instrument_id)
    leg_id = await repo.insert_leg(leg)

    await repo.close_leg(leg_id, datetime.now(UTC))

    active = await repo.get_active_legs("test-account")
    assert not any(a.id == leg_id for a in active)


async def test_get_leg_by_id(repo: Any, test_instrument_id: int) -> None:
    leg = _make_leg(test_instrument_id)
    leg_id = await repo.insert_leg(leg)

    loaded = await repo.get_leg(leg_id)
    assert loaded is not None
    assert loaded.account == "test-account"
    assert loaded.position_side == PositionSide.LONG
    assert loaded.quantity == Decimal("0.5")


async def test_get_leg_not_found(repo: Any, test_instrument_id: int) -> None:
    loaded = await repo.get_leg(999999)
    assert loaded is None
