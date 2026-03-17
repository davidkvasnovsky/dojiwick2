"""Symbol and pair normalization helpers."""


def pair_to_symbol(pair_or_symbol: str, pair_separator: str = "/") -> str:
    """Normalize a configured pair or exchange symbol to exchange symbol form.

    Examples:
    - ``BTC/USDT`` -> ``BTCUSDT``
    - ``BTCUSDT`` -> ``BTCUSDT``
    """
    if not pair_or_symbol:
        raise ValueError("pair_or_symbol must not be empty")

    if pair_separator and pair_separator in pair_or_symbol:
        base, quote = pair_or_symbol.split(pair_separator, 1)
        if not base or not quote:
            raise ValueError(f"invalid pair format: {pair_or_symbol}")
        return f"{base}{quote}"

    return pair_or_symbol


def split_symbol(symbol: str, quote_asset: str) -> tuple[str, str]:
    """Split exchange symbol into base/quote using the configured quote asset."""
    if not symbol:
        raise ValueError("symbol must not be empty")
    if not quote_asset:
        raise ValueError("quote_asset must not be empty")

    if symbol.endswith(quote_asset) and len(symbol) > len(quote_asset):
        return symbol[: -len(quote_asset)], quote_asset

    return symbol, quote_asset
