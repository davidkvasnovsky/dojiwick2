"""Integration tests for PgCandleRepository."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.type_aliases import CandleInterval

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.candle import PgCandleRepository

    return PgCandleRepository(connection=db_connection)


def _make_candle(pair: str = "BTC/USDC", open_time: datetime | None = None) -> Candle:
    return Candle(
        pair=pair,
        interval=CandleInterval("1h"),
        open_time=open_time or datetime.now(UTC),
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("95"),
        close=Decimal("102"),
        volume=Decimal("1000"),
        quote_volume=Decimal("100000"),
    )


async def test_upsert_and_get_candles(repo: Any, clean_tables: None) -> None:
    now = datetime.now(UTC)
    candle = _make_candle(open_time=now)
    count = await repo.upsert_candles("BTC/USDC", CandleInterval("1h"), (candle,), venue="binance", product="usd_c")
    assert count == 1
    candles = await repo.get_candles(
        "BTC/USDC",
        CandleInterval("1h"),
        now - timedelta(seconds=1),
        now + timedelta(seconds=1),
        venue="binance",
        product="usd_c",
    )
    assert len(candles) >= 1
    assert candles[0].pair == "BTC/USDC"


async def test_get_candles_empty(repo: Any, clean_tables: None) -> None:
    now = datetime.now(UTC)
    candles = await repo.get_candles(
        "UNKNOWN/PAIR",
        CandleInterval("1h"),
        now - timedelta(seconds=1),
        now + timedelta(seconds=1),
        venue="binance",
        product="usd_c",
    )
    assert candles == ()
