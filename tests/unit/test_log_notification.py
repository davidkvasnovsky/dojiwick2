"""Tests for LogNotification adapter."""

import logging

import pytest

from dojiwick.domain.enums import AuditSeverity
from dojiwick.infrastructure.observability.log_notification import LogNotification


class TestLogNotification:
    @pytest.mark.asyncio
    async def test_protocol_conformance(self) -> None:
        """LogNotification has the send_alert method matching NotificationPort."""
        n = LogNotification()
        # Should not raise
        await n.send_alert(AuditSeverity.INFO, "test message")

    @pytest.mark.asyncio
    async def test_severity_mapping_info(self, caplog: pytest.LogCaptureFixture) -> None:
        n = LogNotification()
        with caplog.at_level(logging.DEBUG):
            await n.send_alert(AuditSeverity.INFO, "info msg")
        assert any(r.levelno == logging.INFO and "info msg" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_severity_mapping_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        n = LogNotification()
        with caplog.at_level(logging.DEBUG):
            await n.send_alert(AuditSeverity.WARNING, "warn msg")
        assert any(r.levelno == logging.WARNING and "warn msg" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_severity_mapping_critical(self, caplog: pytest.LogCaptureFixture) -> None:
        n = LogNotification()
        with caplog.at_level(logging.DEBUG):
            await n.send_alert(AuditSeverity.CRITICAL, "crit msg")
        assert any(r.levelno == logging.CRITICAL and "crit msg" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_context_included(self, caplog: pytest.LogCaptureFixture) -> None:
        n = LogNotification()
        with caplog.at_level(logging.DEBUG):
            await n.send_alert(AuditSeverity.INFO, "with ctx", context={"key": "val"})
        assert any("key" in r.message for r in caplog.records)
