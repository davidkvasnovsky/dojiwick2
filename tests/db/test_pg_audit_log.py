"""Integration tests for PgAuditLog."""

from typing import Any

import pytest

from dojiwick.domain.enums import AuditSeverity

pytestmark = pytest.mark.db


@pytest.fixture
def audit_log(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.gateways.audit_log import PgAuditLog

    return PgAuditLog(connection=db_connection)


async def test_log_event(audit_log: Any, db_cursor: Any, clean_tables: None) -> None:
    await audit_log.log_event(
        severity=AuditSeverity.INFO,
        event_type="test_event",
        message="test message",
        context={"key": "value"},
    )
    await db_cursor.execute("SELECT event_type, message FROM audit_log ORDER BY id DESC LIMIT 1")
    row = await db_cursor.fetchone()
    assert row is not None
    assert row[0] == "test_event"
    assert row[1] == "test message"


async def test_log_event_no_context(audit_log: Any, db_cursor: Any, clean_tables: None) -> None:
    await audit_log.log_event(
        severity=AuditSeverity.WARNING,
        event_type="no_context",
        message="warning message",
    )
    await db_cursor.execute("SELECT event_type FROM audit_log WHERE event_type = 'no_context'")
    row = await db_cursor.fetchone()
    assert row is not None
