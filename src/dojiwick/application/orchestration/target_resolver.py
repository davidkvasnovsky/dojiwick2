"""Resolve vectorized execution intents into typed target leg positions."""

from dataclasses import dataclass

from dojiwick.domain.enums import PositionMode, PositionSide, TradeAction
from dojiwick.domain.type_aliases import ProductCode, VenueCode
from dojiwick.domain.models.value_objects.batch_models import BatchExecutionIntent
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId, TargetLegPosition
from dojiwick.domain.numerics import ZERO, to_quantity
from dojiwick.domain.symbols import pair_to_symbol, split_symbol


@dataclass(slots=True, frozen=True)
class ResolvedTargets:
    """Target leg positions together with their originating batch indices."""

    targets: tuple[TargetLegPosition, ...]
    batch_indices: tuple[int, ...]


def pair_to_instrument_id(
    pair: str,
    venue: VenueCode,
    product: ProductCode,
    *,
    quote_asset: str,
    settle_asset: str = "",
    pair_separator: str = "/",
) -> InstrumentId:
    """Convert pair/symbol input to an InstrumentId.

    Supports both:
    - pair format: ``BASE/QUOTE`` (e.g., ``BTC/USDT``)
    - exchange symbol format: ``BTCUSDT``
    """
    if pair_separator and pair_separator in pair:
        parts = pair.split(pair_separator, 1)
        base = parts[0]
        quote = parts[1]
    else:
        symbol = pair_to_symbol(pair, pair_separator)
        base, quote = split_symbol(symbol, quote_asset)

    symbol = f"{base}{quote}"
    return InstrumentId(
        venue=venue,
        product=product,
        symbol=symbol,
        base_asset=base,
        quote_asset=quote,
        settle_asset=settle_asset or quote,
    )


def resolve_targets(
    intents: BatchExecutionIntent,
    *,
    account: str,
    position_mode: PositionMode = PositionMode.ONE_WAY,
    venue: VenueCode,
    product: ProductCode,
    quote_asset: str,
    settle_asset: str = "",
    pair_separator: str = "/",
    instrument_map: dict[str, InstrumentId],
) -> ResolvedTargets:
    """Convert active execution intents into target leg positions for the planner.

    Returns both the targets and the batch indices they originated from,
    enabling receipt alignment after plan execution.

    For each active row in the intent batch:
    - BUY action → LONG target (or NET in one-way mode)
    - SHORT action → SHORT target (or NET with negative qty in one-way mode)
    """
    targets: list[TargetLegPosition] = []
    indices: list[int] = []

    for index in range(len(intents.pairs)):
        if not intents.active_mask[index]:
            continue

        action = TradeAction(int(intents.action[index]))
        if action == TradeAction.HOLD:
            continue

        pair = intents.pairs[index]
        quantity = to_quantity(float(intents.quantity[index]))
        if quantity <= ZERO:
            continue

        mapped = instrument_map.get(pair)
        if mapped is None:
            raise ValueError(f"pair '{pair}' not found in instrument_map — all active pairs must have a target mapping")
        instrument_id = mapped

        if position_mode == PositionMode.HEDGE:
            position_side = PositionSide.LONG if action == TradeAction.BUY else PositionSide.SHORT
            target_qty = quantity
        else:
            position_side = PositionSide.NET
            target_qty = quantity if action == TradeAction.BUY else -quantity

        targets.append(
            TargetLegPosition(
                account=account,
                instrument_id=instrument_id,
                position_side=position_side,
                target_qty=target_qty,
            )
        )
        indices.append(index)

    return ResolvedTargets(targets=tuple(targets), batch_indices=tuple(indices))
