"""Unit tests for target resolution from vectorized intents."""

from decimal import Decimal

import numpy as np

from dojiwick.application.orchestration.target_resolver import pair_to_instrument_id, resolve_targets
from dojiwick.domain.enums import PositionMode, PositionSide, TradeAction
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.domain.models.value_objects.batch_models import BatchExecutionIntent
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId


def _intent(
    pairs: tuple[str, ...],
    actions: list[int],
    quantities: list[float],
    active: list[bool],
) -> BatchExecutionIntent:
    size = len(pairs)
    return BatchExecutionIntent(
        pairs=pairs,
        action=np.array(actions, dtype=np.int64),
        quantity=np.array(quantities, dtype=np.float64),
        notional_usd=np.array(quantities, dtype=np.float64),
        entry_price=np.full(size, 100.0, dtype=np.float64),
        stop_price=np.full(size, 95.0, dtype=np.float64),
        take_profit_price=np.full(size, 110.0, dtype=np.float64),
        strategy_name=tuple("test" for _ in range(size)),
        strategy_variant=tuple("baseline" for _ in range(size)),
        active_mask=np.array(active, dtype=np.bool_),
    )


def _build_instrument_map(pairs: tuple[str, ...]) -> dict[str, InstrumentId]:
    """Build instrument_map for test pairs using pair_to_instrument_id."""
    return {
        pair: pair_to_instrument_id(pair, venue=BINANCE_VENUE, product=BINANCE_USD_C, quote_asset="USDC")
        for pair in pairs
    }


def test_pair_to_instrument_id() -> None:
    iid = pair_to_instrument_id("BTC/USDC", venue=BINANCE_VENUE, product=BINANCE_USD_C, quote_asset="USDC")
    assert iid.venue == BINANCE_VENUE
    assert iid.product == BINANCE_USD_C
    assert iid.symbol == "BTCUSDC"
    assert iid.base_asset == "BTC"
    assert iid.quote_asset == "USDC"
    assert iid.settle_asset == "USDC"


def test_resolve_targets_buy() -> None:
    pairs = ("BTC/USDC", "ETH/USDC")
    intents = _intent(
        pairs=pairs,
        actions=[TradeAction.BUY, TradeAction.HOLD],
        quantities=[0.5, 0.0],
        active=[True, False],
    )

    resolved = resolve_targets(
        intents,
        account="default",
        venue=BINANCE_VENUE,
        product=BINANCE_USD_C,
        quote_asset="USDC",
        instrument_map=_build_instrument_map(pairs),
    )

    assert len(resolved.targets) == 1
    assert resolved.batch_indices == (0,)
    target = resolved.targets[0]
    assert target.position_side == PositionSide.NET
    assert target.target_qty == Decimal("0.5")
    assert target.instrument_id.symbol == "BTCUSDC"


def test_resolve_targets_short_hedge_mode() -> None:
    pairs = ("BTC/USDC",)
    intents = _intent(
        pairs=pairs,
        actions=[TradeAction.SHORT],
        quantities=[0.3],
        active=[True],
    )

    resolved = resolve_targets(
        intents,
        account="default",
        venue=BINANCE_VENUE,
        product=BINANCE_USD_C,
        quote_asset="USDC",
        position_mode=PositionMode.HEDGE,
        instrument_map=_build_instrument_map(pairs),
    )

    assert len(resolved.targets) == 1
    target = resolved.targets[0]
    assert target.position_side == PositionSide.SHORT
    assert target.target_qty == Decimal("0.3")


def test_resolve_targets_short_one_way_uses_negative_net_qty() -> None:
    pairs = ("BTC/USDC",)
    intents = _intent(
        pairs=pairs,
        actions=[TradeAction.SHORT],
        quantities=[0.3],
        active=[True],
    )

    resolved = resolve_targets(
        intents,
        account="default",
        venue=BINANCE_VENUE,
        product=BINANCE_USD_C,
        quote_asset="USDC",
        position_mode=PositionMode.ONE_WAY,
        instrument_map=_build_instrument_map(pairs),
    )

    assert len(resolved.targets) == 1
    target = resolved.targets[0]
    assert target.position_side == PositionSide.NET
    assert target.target_qty == Decimal("-0.3")


def test_resolve_targets_skips_inactive() -> None:
    pairs = ("BTC/USDC", "ETH/USDC")
    intents = _intent(
        pairs=pairs,
        actions=[TradeAction.BUY, TradeAction.BUY],
        quantities=[0.5, 0.3],
        active=[True, False],
    )

    resolved = resolve_targets(
        intents,
        account="default",
        venue=BINANCE_VENUE,
        product=BINANCE_USD_C,
        quote_asset="USDC",
        instrument_map=_build_instrument_map(pairs),
    )

    assert len(resolved.targets) == 1
    assert resolved.batch_indices == (0,)


def test_resolve_targets_multiple_active() -> None:
    pairs = ("BTC/USDC", "ETH/USDC")
    intents = _intent(
        pairs=pairs,
        actions=[TradeAction.BUY, TradeAction.SHORT],
        quantities=[0.5, 0.3],
        active=[True, True],
    )

    resolved = resolve_targets(
        intents,
        account="default",
        venue=BINANCE_VENUE,
        product=BINANCE_USD_C,
        quote_asset="USDC",
        instrument_map=_build_instrument_map(pairs),
    )

    assert len(resolved.targets) == 2
    assert resolved.batch_indices == (0, 1)
    assert resolved.targets[0].target_qty == Decimal("0.5")
    assert resolved.targets[1].target_qty == Decimal("-0.3")


def test_resolve_targets_empty_when_all_hold() -> None:
    pairs = ("BTC/USDC",)
    intents = _intent(
        pairs=pairs,
        actions=[TradeAction.HOLD],
        quantities=[0.0],
        active=[False],
    )

    resolved = resolve_targets(
        intents,
        account="default",
        venue=BINANCE_VENUE,
        product=BINANCE_USD_C,
        quote_asset="USDC",
        instrument_map=_build_instrument_map(pairs),
    )

    assert len(resolved.targets) == 0
    assert len(resolved.batch_indices) == 0


def test_pair_to_instrument_id_accepts_exchange_symbol() -> None:
    iid = pair_to_instrument_id("BTCUSDC", venue=BINANCE_VENUE, product=BINANCE_USD_C, quote_asset="USDC")
    assert iid.symbol == "BTCUSDC"
    assert iid.base_asset == "BTC"
    assert iid.quote_asset == "USDC"
