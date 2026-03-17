"""Integration tests for PgOutcomeRepository."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from dojiwick.domain.enums import DecisionAuthority, DecisionStatus, MarketState, TradeAction
from dojiwick.domain.models.value_objects.outcome_models import DecisionOutcome

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.outcome import PgOutcomeRepository

    return PgOutcomeRepository(connection=db_connection)


def _make_outcome(pair: str = "BTC/USDC") -> DecisionOutcome:
    return DecisionOutcome(
        pair=pair,
        target_id=pair,
        observed_at=datetime.now(UTC),
        authority=DecisionAuthority.DETERMINISTIC_ONLY,
        status=DecisionStatus.HOLD,
        market_state=MarketState.RANGING,
        action=TradeAction.HOLD,
        strategy_name="trend_follow",
        strategy_variant="baseline",
        reason_code="strategy_hold",
        confidence=0.8,
        entry_price=Decimal(0),
        stop_price=Decimal(0),
        take_profit_price=Decimal(0),
        quantity=Decimal(0),
        notional_usd=Decimal(0),
        config_hash="test-config-hash",
    )


async def test_append_outcomes(repo: Any, db_cursor: Any) -> None:
    outcome = _make_outcome()
    await repo.append_outcomes((outcome,), venue="binance", product="usd_c")
    await db_cursor.execute(
        "SELECT pair, status, config_hash, venue, product FROM decision_outcomes ORDER BY id DESC LIMIT 1"
    )
    row = await db_cursor.fetchone()
    assert row is not None
    assert row[0] == "BTC/USDC"
    assert row[1] == "hold"
    assert row[2] == "test-config-hash"
    assert row[3] == "binance"
    assert row[4] == "usd_c"


async def test_append_multiple_outcomes(repo: Any, db_cursor: Any) -> None:
    outcomes = (
        _make_outcome(pair="BTC/USDC"),
        _make_outcome(pair="ETH/USDC"),
    )
    await repo.append_outcomes(outcomes, venue="binance", product="usd_c")
    await db_cursor.execute("SELECT COUNT(*) FROM decision_outcomes")
    row = await db_cursor.fetchone()
    assert row is not None
    assert row[0] >= 2
