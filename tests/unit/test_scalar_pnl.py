"""Characterization tests for the scalar PnL and liquidation kernels."""

import pytest

from dojiwick.compute.kernels.pnl.liquidation import liquidation_price
from dojiwick.compute.kernels.pnl.pnl import scalar_net_pnl


def test_fees_charged_on_leveraged_notional() -> None:
    # Flat price move, zero slippage: PnL is pure fees on notional × leverage.
    pnl = scalar_net_pnl(
        is_long=True,
        entry_price=100.0,
        exit_price=100.0,
        quantity=1.0,
        notional=100.0,
        slippage_bps=0.0,
        fee_bps=10.0,
        fee_multiplier=2.0,
        leverage=3.0,
    )
    assert pnl == pytest.approx(-100.0 * 3.0 * 0.001 * 2.0)  # pyright: ignore[reportUnknownMemberType]


def test_slippage_applied_on_both_sides() -> None:
    pnl = scalar_net_pnl(
        is_long=True,
        entry_price=100.0,
        exit_price=100.0,
        quantity=2.0,
        notional=200.0,
        slippage_bps=10.0,
        fee_bps=0.0,
        leverage=1.0,
    )
    # entry 100.1, exit 99.9 → -0.2 per unit × qty 2
    assert pnl == pytest.approx(-0.4)  # pyright: ignore[reportUnknownMemberType]


def test_short_mirrors_long() -> None:
    long_pnl = scalar_net_pnl(
        is_long=True,
        entry_price=100.0,
        exit_price=110.0,
        quantity=1.0,
        notional=100.0,
        slippage_bps=5.0,
        fee_bps=4.0,
        leverage=2.0,
    )
    short_pnl = scalar_net_pnl(
        is_long=False,
        entry_price=100.0,
        exit_price=90.0,
        quantity=1.0,
        notional=100.0,
        slippage_bps=5.0,
        fee_bps=4.0,
        leverage=2.0,
    )
    # Symmetric 10-point favorable moves differ only via price-proportional
    # exit slippage (110 vs 90 base): (110 + 90 - 2×100) × slip × leverage
    slip_rate = 5.0 / 10_000.0
    assert long_pnl - short_pnl == pytest.approx(-20.0 * slip_rate * 2.0, abs=1e-9)  # pyright: ignore[reportUnknownMemberType]
    assert long_pnl == pytest.approx((110.0 * (1 - slip_rate) - 100.0 * (1 + slip_rate)) * 2.0 - 100.0 * 2 * 0.0004 * 2)  # pyright: ignore[reportUnknownMemberType]


def test_funding_usd_is_signed_passthrough() -> None:
    base = scalar_net_pnl(
        is_long=True,
        entry_price=100.0,
        exit_price=105.0,
        quantity=1.0,
        notional=100.0,
        slippage_bps=0.0,
        fee_bps=0.0,
    )
    paid = scalar_net_pnl(
        is_long=True,
        entry_price=100.0,
        exit_price=105.0,
        quantity=1.0,
        notional=100.0,
        slippage_bps=0.0,
        fee_bps=0.0,
        funding_usd=1.5,
    )
    received = scalar_net_pnl(
        is_long=True,
        entry_price=100.0,
        exit_price=105.0,
        quantity=1.0,
        notional=100.0,
        slippage_bps=0.0,
        fee_bps=0.0,
        funding_usd=-1.5,
    )
    assert paid == pytest.approx(base - 1.5)  # pyright: ignore[reportUnknownMemberType]
    assert received == pytest.approx(base + 1.5)  # pyright: ignore[reportUnknownMemberType]


def test_liquidation_price_math() -> None:
    # margin_distance = 1/2 - 0.01 = 0.49
    assert liquidation_price(100.0, 2.0, 0.01, is_long=True) == pytest.approx(51.0)  # pyright: ignore[reportUnknownMemberType]
    assert liquidation_price(100.0, 2.0, 0.01, is_long=False) == pytest.approx(149.0)  # pyright: ignore[reportUnknownMemberType]


def test_liquidation_disabled_when_unleveraged() -> None:
    assert liquidation_price(100.0, 1.0, 0.01, is_long=True) == 0.0
