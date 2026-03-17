"""Tests that verify schema CHECK and FK constraints reject bad data.

Covers all tables: decision_outcomes, regime_observations, position_legs,
order_requests, fills, instruments, reconciliation_runs, strategy_state.
"""

from typing import Any

import pytest

pytestmark = pytest.mark.db


async def test_decision_outcomes_confidence_range(db_cursor: Any) -> None:
    """confidence CHECK (confidence BETWEEN 0 AND 1) should reject out-of-range."""
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO decision_outcomes
            (pair, observed_at, status, authority, reason_code,
             action, strategy_name, strategy_variant, confidence,
             market_state, config_hash)
            VALUES ('BTC/USDC', now(), 'hold', 'deterministic_only', 'test',
                    'hold', 'test', 'test', 1.5,
                    'ranging', 'test-config-hash')"""
        )


async def test_regime_observations_coarse_state_range(db_cursor: Any) -> None:
    """coarse_state CHECK (coarse_state BETWEEN 1 AND 4) should reject out-of-range."""
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO regime_observations
            (pair, observed_at, coarse_state, confidence, valid)
            VALUES ('BTC/USDC', now(), 99, 0.5, true)"""
        )


# position_legs


async def test_position_legs_quantity_non_negative(db_cursor: Any, test_instrument_id: int) -> None:
    """CHECK (quantity >= 0) should reject negative quantity."""
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO position_legs
            (account, instrument_id, position_side, quantity, entry_price, leverage)
            VALUES ('test', %s, 'long', -1, 50000, 1)""",
            (test_instrument_id,),
        )


async def test_position_legs_leverage_minimum(db_cursor: Any, test_instrument_id: int) -> None:
    """CHECK (leverage >= 1) should reject zero leverage."""
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO position_legs
            (account, instrument_id, position_side, quantity, entry_price, leverage)
            VALUES ('test', %s, 'long', 1, 50000, 0)""",
            (test_instrument_id,),
        )


async def test_position_legs_instrument_fk(db_cursor: Any) -> None:
    """FK to instruments should reject non-existent instrument_id."""
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO position_legs
            (account, instrument_id, position_side, quantity, entry_price, leverage)
            VALUES ('test', 999999, 'long', 1, 50000, 1)"""
        )


# position_events


async def test_position_events_quantity_positive(db_cursor: Any) -> None:
    """CHECK (quantity > 0) should reject zero quantity."""
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO position_events
            (position_leg_id, event_type, quantity, price)
            VALUES (999999, 'open', 0, 50000)"""
        )


async def test_position_events_price_positive(db_cursor: Any) -> None:
    """CHECK (price > 0) should reject zero price."""
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO position_events
            (position_leg_id, event_type, quantity, price)
            VALUES (999999, 'open', 1, 0)"""
        )


# order_requests


async def test_order_requests_quantity_positive(db_cursor: Any) -> None:
    """CHECK (quantity > 0) should reject zero quantity."""
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO order_requests
            (client_order_id, instrument_id, account, side, order_type, quantity)
            VALUES ('test-ck-qty', 999999, 'test', 'buy', 'market', 0)"""
        )


async def test_order_requests_unique_client_order_id(db_cursor: Any, test_instrument_id: int) -> None:
    """UNIQUE constraint on client_order_id should reject duplicates."""
    await db_cursor.execute(
        """INSERT INTO order_requests
        (client_order_id, instrument_id, account, side, order_type, quantity)
        VALUES ('dup-test', %s, 'test', 'buy', 'market', 1)""",
        (test_instrument_id,),
    )
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO order_requests
            (client_order_id, instrument_id, account, side, order_type, quantity)
            VALUES ('dup-test', %s, 'test', 'buy', 'market', 1)""",
            (test_instrument_id,),
        )


# fills


async def test_fills_price_positive(db_cursor: Any) -> None:
    """CHECK (price > 0) should reject zero price."""
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO fills
            (order_request_id, price, quantity)
            VALUES (999999, 0, 1)"""
        )


async def test_fills_quantity_positive(db_cursor: Any) -> None:
    """CHECK (quantity > 0) should reject zero quantity."""
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO fills
            (order_request_id, price, quantity)
            VALUES (999999, 50000, 0)"""
        )


async def test_fills_commission_non_negative(db_cursor: Any) -> None:
    """CHECK (commission >= 0) should reject negative commission."""
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO fills
            (order_request_id, price, quantity, commission)
            VALUES (999999, 50000, 1, -0.01)"""
        )


# instruments


async def test_instruments_unique_symbol(db_cursor: Any, clean_tables: None) -> None:
    """UNIQUE constraint on (venue, product, symbol) should reject duplicates."""
    await db_cursor.execute(
        """INSERT INTO instruments
        (venue, product, symbol, base_asset, quote_asset, settle_asset,
         status, price_precision, quantity_precision,
         base_asset_precision, quote_asset_precision)
        VALUES ('binance', 'usd_c', 'ETHUSDC', 'ETH', 'USDC', 'USDC',
                'trading', 8, 3, 8, 8)"""
    )
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO instruments
            (venue, product, symbol, base_asset, quote_asset, settle_asset,
             status, price_precision, quantity_precision,
             base_asset_precision, quote_asset_precision)
            VALUES ('binance', 'usd_c', 'ETHUSDC', 'ETH', 'USDC', 'USDC',
                    'trading', 8, 3, 8, 8)"""
        )


# instrument_filters


async def test_instrument_filters_tick_size_positive(db_cursor: Any, test_instrument_id: int) -> None:
    """CHECK (tick_size > 0) should reject zero tick_size."""
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO instrument_filters
            (instrument_id, min_price, tick_size, min_qty, step_size, min_notional)
            VALUES (%s, 0.01, 0, 0.001, 0.001, 5)""",
            (test_instrument_id,),
        )


# reconciliation_runs


async def test_reconciliation_runs_type_check(db_cursor: Any) -> None:
    """CHECK constraint on run_type should reject invalid types."""
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO reconciliation_runs (run_type)
            VALUES ('invalid_type')"""
        )


# position_legs partial unique index


async def test_position_legs_partial_unique_active(db_cursor: Any, test_instrument_id: int) -> None:
    """Partial unique index allows only one active (unclosed) leg per key."""
    await db_cursor.execute(
        """INSERT INTO position_legs
        (account, instrument_id, position_side, quantity, entry_price, leverage)
        VALUES ('idx-test', %s, 'long', 1, 50000, 1)""",
        (test_instrument_id,),
    )
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO position_legs
            (account, instrument_id, position_side, quantity, entry_price, leverage)
            VALUES ('idx-test', %s, 'long', 2, 51000, 1)""",
            (test_instrument_id,),
        )


async def test_position_legs_partial_unique_allows_closed(db_cursor: Any, test_instrument_id: int) -> None:
    """Closed legs don't violate the partial unique index — a new active leg is allowed."""
    # Insert and close a leg
    await db_cursor.execute(
        """INSERT INTO position_legs
        (account, instrument_id, position_side, quantity, entry_price, leverage, closed_at)
        VALUES ('idx-closed-test', %s, 'long', 1, 50000, 1, now())""",
        (test_instrument_id,),
    )
    # A new active leg for the same key should succeed
    await db_cursor.execute(
        """INSERT INTO position_legs
        (account, instrument_id, position_side, quantity, entry_price, leverage)
        VALUES ('idx-closed-test', %s, 'long', 2, 51000, 1)""",
        (test_instrument_id,),
    )
    # Verify both rows exist
    await db_cursor.execute("SELECT COUNT(*) FROM position_legs WHERE account = 'idx-closed-test'")
    row: tuple[object, ...] | None = await db_cursor.fetchone()
    assert row is not None
    assert int(str(row[0])) == 2


async def test_position_legs_hedge_mode_independent_sides(db_cursor: Any, test_instrument_id: int) -> None:
    """Hedge mode: LONG and SHORT legs for the same instrument can coexist."""
    await db_cursor.execute(
        """INSERT INTO position_legs
        (account, instrument_id, position_side, quantity, entry_price, leverage)
        VALUES ('hedge-test', %s, 'long', 1, 50000, 1)""",
        (test_instrument_id,),
    )
    await db_cursor.execute(
        """INSERT INTO position_legs
        (account, instrument_id, position_side, quantity, entry_price, leverage)
        VALUES ('hedge-test', %s, 'short', 1, 50000, 1)""",
        (test_instrument_id,),
    )
    await db_cursor.execute("SELECT COUNT(*) FROM position_legs WHERE account = 'hedge-test' AND closed_at IS NULL")
    row: tuple[object, ...] | None = await db_cursor.fetchone()
    assert row is not None
    assert int(str(row[0])) == 2


# strategy_state composite PK


async def test_strategy_state_different_strategies_same_pair(db_cursor: Any) -> None:
    """Composite PK allows different (strategy, variant) rows for the same pair."""
    await db_cursor.execute(
        """INSERT INTO strategy_state (pair, active_strategy, variant, state_json)
        VALUES ('BTC/USDC', 'trend_follow', 'baseline', '{}')"""
    )
    await db_cursor.execute(
        """INSERT INTO strategy_state (pair, active_strategy, variant, state_json)
        VALUES ('BTC/USDC', 'mean_revert', 'baseline', '{}')"""
    )
    await db_cursor.execute("SELECT COUNT(*) FROM strategy_state WHERE pair = 'BTC/USDC'")
    row: tuple[object, ...] | None = await db_cursor.fetchone()
    assert row is not None
    assert int(str(row[0])) == 2


async def test_strategy_state_upsert_on_conflict(db_cursor: Any) -> None:
    """Duplicate (pair, strategy, variant) should conflict on PK."""
    await db_cursor.execute(
        """INSERT INTO strategy_state (pair, active_strategy, variant, state_json)
        VALUES ('BTC/USDC', 'trend_follow', 'v2', '{"a": 1}')"""
    )
    with pytest.raises(Exception):
        await db_cursor.execute(
            """INSERT INTO strategy_state (pair, active_strategy, variant, state_json)
            VALUES ('BTC/USDC', 'trend_follow', 'v2', '{"a": 2}')"""
        )
