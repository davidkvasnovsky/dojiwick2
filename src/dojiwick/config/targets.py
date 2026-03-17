"""Target-resolution helpers for instrument mapping.

All functions require non-empty ``settings.universe.targets``.
"""

from dojiwick.config.schema import Settings
from dojiwick.domain.errors import ConfigurationError
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId


def resolve_symbols(settings: Settings) -> tuple[str, ...]:
    """Return exchange symbols for candle fetch — market_data_instrument from targets."""
    return tuple(t.market_data_instrument for t in settings.universe.targets)


def resolve_execution_symbols(settings: Settings) -> tuple[str, ...]:
    """Return exchange symbols for live orders/subscriptions."""
    return tuple(t.execution_instrument for t in settings.universe.targets)


def resolve_target_ids(settings: Settings) -> tuple[str, ...]:
    """Return target_ids from configured targets."""
    return tuple(t.target_id for t in settings.universe.targets)


def resolve_instrument_map(settings: Settings) -> dict[str, InstrumentId]:
    """Build display_pair → InstrumentId mapping from targets.

    Parses each execution_instrument into base/quote by splitting on
    the configured quote_asset suffix. Raises ConfigurationError if the
    execution_instrument does not end with quote_asset.
    """
    result: dict[str, InstrumentId] = {}
    quote = settings.universe.quote_asset
    settle = settings.universe.settle_asset
    venue = settings.exchange.venue
    product = settings.exchange.product
    for t in settings.universe.targets:
        sym = t.execution_instrument
        if not sym.endswith(quote):
            raise ConfigurationError(
                f"execution_instrument '{sym}' for target '{t.target_id}' does not end with quote_asset '{quote}'"
            )
        base = sym[: -len(quote)]
        result[t.display_pair] = InstrumentId(
            venue=venue,
            product=product,
            symbol=sym,
            base_asset=base,
            quote_asset=quote,
            settle_asset=settle or quote,
        )
    return result
