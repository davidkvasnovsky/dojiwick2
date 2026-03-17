"""Tests for native fee tracking on ExecutionReceipt."""

from decimal import Decimal

import pytest

from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt
from dojiwick.domain.enums import ExecutionStatus
from fixtures.factories.compute import ExecutionReceiptBuilder


def test_receipt_tracks_native_fee_and_asset() -> None:
    """ExecutionReceipt accepts and validates fee_asset + native_fee_amount."""
    receipt = (
        ExecutionReceiptBuilder()
        .filled(price="50000", quantity="0.01")
        .with_fees("0.50")
        .with_fee_asset("BNB")
        .with_native_fee("0.001")
        .build()
    )

    assert receipt.fee_asset == "BNB"
    assert receipt.native_fee_amount == Decimal("0.001")
    assert receipt.fees_usd == Decimal("0.50")
    assert receipt.status is ExecutionStatus.FILLED


def test_receipt_native_fee_defaults() -> None:
    """Default fee_asset is empty and native_fee_amount is zero."""
    receipt = ExecutionReceiptBuilder().skipped().build()
    assert receipt.fee_asset == ""
    assert receipt.native_fee_amount == Decimal(0)


def test_receipt_rejects_negative_native_fee() -> None:
    """native_fee_amount must be non-negative."""
    with pytest.raises(ValueError, match="native_fee_amount must be non-negative"):
        ExecutionReceipt(
            status=ExecutionStatus.SKIPPED,
            reason="test",
            native_fee_amount=Decimal("-1"),
        )
