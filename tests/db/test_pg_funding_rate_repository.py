"""Integration tests for PgFundingRateRepository."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from dojiwick.domain.models.value_objects.funding_rate import FundingRate

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.funding_rate import PgFundingRateRepository

    return PgFundingRateRepository(connection=db_connection)


def _rate(offset_hours: int = 0, rate: str = "0.0001") -> FundingRate:
    return FundingRate(
        symbol="BTCUSDT",
        funding_time=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=offset_hours),
        rate=Decimal(rate),
    )


async def test_upsert_and_get_rates(repo: Any, clean_tables: None) -> None:
    rates = (_rate(0), _rate(8, "-0.0002"), _rate(16, "0.0003"))
    count = await repo.upsert_rates("BTCUSDT", rates, venue="binance", product="usd_c")
    assert count == 3

    got = await repo.get_rates(
        "BTCUSDT",
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
        venue="binance",
        product="usd_c",
    )
    assert len(got) == 3
    assert got[0].rate == Decimal("0.0001")
    assert got[1].rate == Decimal("-0.0002")
    assert [r.funding_time for r in got] == sorted(r.funding_time for r in got)


async def test_reupsert_is_idempotent(repo: Any, clean_tables: None) -> None:
    rates = (_rate(0), _rate(8))
    await repo.upsert_rates("BTCUSDT", rates, venue="binance", product="usd_c")
    await repo.upsert_rates("BTCUSDT", rates, venue="binance", product="usd_c")

    got = await repo.get_rates(
        "BTCUSDT",
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
        venue="binance",
        product="usd_c",
    )
    assert len(got) == 2


async def test_venue_product_scoping(repo: Any, clean_tables: None) -> None:
    await repo.upsert_rates("BTCUSDT", (_rate(0),), venue="binance", product="usd_c")

    got = await repo.get_rates(
        "BTCUSDT",
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
        venue="other",
        product="usd_c",
    )
    assert got == ()


async def test_empty_venue_raises(repo: Any, clean_tables: None) -> None:
    from dojiwick.domain.errors import AdapterError

    with pytest.raises(AdapterError):
        await repo.upsert_rates("BTCUSDT", (_rate(0),), venue="", product="usd_c")
