"""Tests that verify DB schema aligns with domain type expectations.

These tests query information_schema to confirm column types, nullable settings,
and table existence match the domain contracts.
"""

from typing import Any

import pytest

pytestmark = pytest.mark.db


async def _column_info(db_cursor: Any, table: str, column: str) -> tuple[str, str] | None:
    """Return (data_type, is_nullable) for a column, or None if missing."""
    await db_cursor.execute(
        """SELECT data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s AND column_name = %s""",
        (table, column),
    )
    row: tuple[object, ...] | None = await db_cursor.fetchone()
    if row is None:
        return None
    return (str(row[0]), str(row[1]))


async def _table_exists(db_cursor: Any, table: str) -> bool:
    await db_cursor.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s",
        (table,),
    )
    return await db_cursor.fetchone() is not None


# Table existence


@pytest.mark.parametrize(
    "table",
    [
        "instruments",
        "instrument_filters",
        "decision_outcomes",
        "regime_observations",
        "ticks",
        "decision_traces",
        "position_legs",
        "position_events",
        "order_requests",
        "order_reports",
        "fills",
        "order_events",
        "candles",
        "signals",
        "bot_state",
        "bot_state_history",
        "performance_snapshots",
        "audit_log",
        "pair_trading_state",
        "strategy_state",
        "stream_cursors",
        "reconciliation_runs",
        "system_event_log",
        "bot_config_snapshots",
        "adaptive_configs",
        "adaptive_posteriors",
        "adaptive_selections",
        "adaptive_outcomes",
        "adaptive_calibration_metrics",
        "model_costs",
        "backtest_runs",
    ],
)
async def test_table_exists(db_cursor: Any, table: str) -> None:
    """All tables must be present in the database."""
    assert await _table_exists(db_cursor, table), f"table {table!r} does not exist"


# Numeric columns use NUMERIC (not float) for price/money/quantity


@pytest.mark.parametrize(
    "table,column",
    [
        ("position_legs", "quantity"),
        ("position_legs", "entry_price"),
        ("position_legs", "unrealized_pnl"),
        ("position_events", "quantity"),
        ("position_events", "price"),
        ("order_requests", "quantity"),
        ("order_requests", "price"),
        ("order_reports", "filled_qty"),
        ("order_reports", "avg_price"),
        ("fills", "price"),
        ("fills", "quantity"),
        ("fills", "commission"),
        ("performance_snapshots", "equity_usd"),
        ("backtest_runs", "total_pnl_usd"),
        ("backtest_runs", "expectancy_usd"),
    ],
)
async def test_numeric_columns_use_numeric_type(db_cursor: Any, table: str, column: str) -> None:
    """Price/money/quantity columns must use NUMERIC, not float."""
    info = await _column_info(db_cursor, table, column)
    assert info is not None, f"{table}.{column} does not exist"
    data_type = info[0]
    assert data_type == "numeric", f"{table}.{column} uses {data_type!r}, expected 'numeric'"


# Timestamp columns use TIMESTAMPTZ


@pytest.mark.parametrize(
    "table,column",
    [
        ("instruments", "created_at"),
        ("position_legs", "opened_at"),
        ("position_legs", "closed_at"),
        ("position_events", "occurred_at"),
        ("order_requests", "created_at"),
        ("order_reports", "reported_at"),
        ("fills", "filled_at"),
        ("stream_cursors", "updated_at"),
        ("reconciliation_runs", "started_at"),
        ("system_event_log", "occurred_at"),
    ],
)
async def test_timestamp_columns_use_timestamptz(db_cursor: Any, table: str, column: str) -> None:
    """All timestamp columns must use TIMESTAMPTZ (timezone-aware)."""
    info = await _column_info(db_cursor, table, column)
    assert info is not None, f"{table}.{column} does not exist"
    data_type = info[0]
    assert data_type == "timestamp with time zone", (
        f"{table}.{column} uses {data_type!r}, expected 'timestamp with time zone'"
    )


# system_event_log column alignment


async def test_system_event_log_uses_component(db_cursor: Any) -> None:
    """Column is named 'component' (not 'category')."""
    info = await _column_info(db_cursor, "system_event_log", "component")
    assert info is not None, "system_event_log.component column missing"

    old = await _column_info(db_cursor, "system_event_log", "category")
    assert old is None, "old 'category' column should not exist"


async def test_system_event_log_uses_context(db_cursor: Any) -> None:
    """Column is named 'context' (not 'metadata')."""
    info = await _column_info(db_cursor, "system_event_log", "context")
    assert info is not None, "system_event_log.context column missing"

    old = await _column_info(db_cursor, "system_event_log", "metadata")
    assert old is None, "old 'metadata' column should not exist"


# order_events FK points to order_requests


async def test_order_events_fk_references_order_requests(db_cursor: Any) -> None:
    """order_events.order_id FK must reference order_requests(id)."""
    await db_cursor.execute(
        """SELECT ccu.table_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.constraint_column_usage ccu
            ON tc.constraint_name = ccu.constraint_name
        WHERE tc.table_name = 'order_events'
            AND tc.constraint_type = 'FOREIGN KEY'
            AND tc.constraint_name = 'fk_order_events_order_request'"""
    )
    row: tuple[object, ...] | None = await db_cursor.fetchone()
    assert row is not None, "FK constraint fk_order_events_order_request not found"
    assert str(row[0]) == "order_requests"


async def test_decision_outcomes_has_config_hash(db_cursor: Any) -> None:
    """decision_outcomes must persist the resolved configuration hash."""
    info = await _column_info(db_cursor, "decision_outcomes", "config_hash")
    assert info is not None, "decision_outcomes.config_hash column missing"
    data_type, nullable = info
    assert data_type == "text"
    assert nullable == "NO"
