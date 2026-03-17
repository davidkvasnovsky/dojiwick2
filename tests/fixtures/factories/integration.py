"""Shared helpers for integration tests."""

from decimal import Decimal

from dojiwick.domain.models.value_objects.account_state import AccountBalance, AccountSnapshot

from fixtures.factories.domain import ContextBuilder


def signal_triggering_context_builder() -> ContextBuilder:
    return (
        ContextBuilder()
        .with_prices([96.0, 98.0])
        .with_rsi([30.0, 32.0])
        .with_adx([15.0, 14.0])
        .with_atr([0.5, 0.6])
        .with_indicator("ema_fast", [97.0, 99.0])
        .with_indicator("ema_slow", [99.0, 101.0])
        .with_indicator("ema_base", [98.0, 100.0])
        .with_indicator("ema_trend", [96.0, 98.0])
        .with_indicator("bb_upper", [104.0, 106.0])
        .with_indicator("bb_lower", [96.0, 98.0])
    )


def empty_snapshot(account: str = "default") -> AccountSnapshot:
    return AccountSnapshot(
        account=account,
        balances=(AccountBalance(asset="USDC", wallet_balance=Decimal(10_000), available_balance=Decimal(5_000)),),
        positions=(),
        total_wallet_balance=Decimal(10_000),
        available_balance=Decimal(5_000),
    )
