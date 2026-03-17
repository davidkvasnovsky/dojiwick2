"""Database test fixtures — skip if DOJIWICK_TEST_DB_URL is unset."""

import os
from collections.abc import AsyncGenerator
from typing import cast

import pytest
import pytest_asyncio

from dojiwick.infrastructure.postgres.connection import DbConnection, DbCursor

_DB_URL = os.environ.get("DOJIWICK_TEST_DB_URL")

pytestmark = pytest.mark.db


@pytest.fixture(scope="session")
def db_dsn() -> str:
    """Return the test database DSN or skip the test."""
    if _DB_URL is None:
        pytest.skip("DOJIWICK_TEST_DB_URL not set — skipping DB tests")
    return _DB_URL


@pytest_asyncio.fixture
async def db_connection(db_dsn: str) -> AsyncGenerator[DbConnection, None]:
    """Yield an async psycopg connection with per-test transaction rollback."""
    try:
        import psycopg
    except ImportError:
        pytest.skip("psycopg not installed")

    conn = cast(DbConnection, await psycopg.AsyncConnection.connect(db_dsn))
    try:
        yield conn
    finally:
        await conn.rollback()
        await conn.close()


@pytest_asyncio.fixture
async def db_cursor(db_connection: DbConnection) -> AsyncGenerator[DbCursor, None]:
    """Convenience fixture for an async cursor within the test transaction."""
    async with db_connection.cursor() as cursor:
        yield cursor


@pytest_asyncio.fixture
async def clean_tables(db_cursor: DbCursor) -> None:
    """Truncate all tables for a clean test slate.

    Uses CASCADE to handle FK dependencies automatically.
    Order: leaf tables first, then parent tables.
    """
    await db_cursor.execute("""
        TRUNCATE
            adaptive_calibration_metrics,
            adaptive_outcomes,
            adaptive_selections,
            adaptive_posteriors,
            adaptive_configs,
            backtest_runs,
            model_costs,
            decision_traces,
            fills,
            order_reports,
            order_events,
            order_requests,
            position_events,
            position_legs,
            instrument_filters,
            instruments,
            reconciliation_runs,
            stream_cursors,
            system_event_log,
            bot_config_snapshots,
            bot_state_history,
            strategy_state,
            audit_log,
            performance_snapshots,
            bot_state,
            pair_trading_state,
            signals,
            candles,
            ticks,
            regime_observations,
            decision_outcomes
        CASCADE
    """)


@pytest_asyncio.fixture
async def test_instrument_id(db_cursor: DbCursor, clean_tables: None) -> int:
    """Insert a test instrument and return its DB-assigned id.

    Many tables have FK to instruments, so this provides a reusable parent row.
    """
    await db_cursor.execute(
        """INSERT INTO instruments (
            venue, product, symbol, base_asset, quote_asset, settle_asset,
            status, price_precision, quantity_precision,
            base_asset_precision, quote_asset_precision
        ) VALUES (
            'binance', 'usd_c', 'BTCUSDC', 'BTC', 'USDC', 'USDC',
            'trading', 8, 3, 8, 8
        ) RETURNING id"""
    )
    row: tuple[object, ...] | None = await db_cursor.fetchone()
    assert row is not None
    return int(str(row[0]))
