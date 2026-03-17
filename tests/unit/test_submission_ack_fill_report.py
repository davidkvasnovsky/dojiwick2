"""Tests for SubmissionAck and FillReport value objects."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from dojiwick.domain.enums import SubmissionStatus
from dojiwick.domain.models.value_objects.fill_report import FillReport
from dojiwick.domain.models.value_objects.submission_ack import SubmissionAck


def test_submission_ack_accepted() -> None:
    ack = SubmissionAck(
        status=SubmissionStatus.ACCEPTED,
        order_id="12345",
        exchange_timestamp=datetime.now(UTC),
    )
    assert ack.status is SubmissionStatus.ACCEPTED
    assert ack.order_id == "12345"


def test_submission_ack_rejected() -> None:
    ack = SubmissionAck(status=SubmissionStatus.REJECTED, reason="insufficient_margin")
    assert ack.status is SubmissionStatus.REJECTED
    assert ack.reason == "insufficient_margin"


def test_fill_report_valid() -> None:
    fill = FillReport(
        fill_price=Decimal("50000"),
        filled_quantity=Decimal("0.01"),
        fees_usd=Decimal("0.50"),
        fee_asset="BNB",
        native_fee_amount=Decimal("0.001"),
        exchange_timestamp=datetime.now(UTC),
    )
    assert fill.fill_price == Decimal("50000")
    assert fill.fee_asset == "BNB"


def test_fill_report_rejects_zero_price() -> None:
    with pytest.raises(ValueError, match="fill_price must be positive"):
        FillReport(fill_price=Decimal(0), filled_quantity=Decimal("0.01"))


def test_fill_report_rejects_negative_fee() -> None:
    with pytest.raises(ValueError, match="native_fee_amount must be non-negative"):
        FillReport(
            fill_price=Decimal("100"),
            filled_quantity=Decimal("0.01"),
            native_fee_amount=Decimal("-1"),
        )


def test_submission_ack_vs_fill_report_types_are_distinct() -> None:
    """SubmissionAck and FillReport are separate types, not interchangeable."""
    ack = SubmissionAck(status=SubmissionStatus.ACCEPTED)
    fill = FillReport(fill_price=Decimal("100"), filled_quantity=Decimal("0.01"))
    assert type(ack) is not type(fill)
