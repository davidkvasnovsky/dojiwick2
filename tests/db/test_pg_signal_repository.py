"""Integration tests for PgSignalRepository."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from dojiwick.domain.models.value_objects.signal import Signal

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.signal import PgSignalRepository

    return PgSignalRepository(connection=db_connection)


def _make_signal(pair: str = "BTC/USDC") -> Signal:
    return Signal(
        pair=pair,
        target_id=pair,
        signal_type="breakout",
        priority=1,
        detected_at=datetime.now(UTC),
    )


async def test_record_signal_returns_id(repo: Any, clean_tables: None) -> None:
    signal = _make_signal()
    signal_id = await repo.record_signal(signal, venue="binance", product="usd_c")
    assert signal_id > 0


async def test_get_signals_for_tick(repo: Any, clean_tables: None) -> None:
    now = datetime.now(UTC)
    signal = Signal(
        pair="BTC/USDC",
        target_id="BTC/USDC",
        signal_type="breakout",
        priority=1,
        detected_at=now,
    )
    await repo.record_signal(signal, venue="binance", product="usd_c")
    signals = await repo.get_signals_for_tick("BTC/USDC", now - timedelta(seconds=1), now + timedelta(seconds=1))
    assert len(signals) >= 1
    assert signals[0].signal_type == "breakout"


async def test_get_signals_for_tick_empty(repo: Any, clean_tables: None) -> None:
    now = datetime.now(UTC)
    signals = await repo.get_signals_for_tick("UNKNOWN/PAIR", now - timedelta(seconds=1), now + timedelta(seconds=1))
    assert signals == ()
