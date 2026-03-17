"""Canonical indicator ordering for batch tensors."""

INDICATOR_NAMES: tuple[str, ...] = (
    "rsi",
    "adx",
    "atr",
    "ema_fast",
    "ema_slow",
    "ema_base",
    "bb_upper",
    "bb_lower",
    "ema_trend",
    "volume_ema_ratio",
    "bb_mid",
    "macd_histogram",
    "macd_signal",
)
INDICATOR_INDEX: dict[str, int] = {name: index for index, name in enumerate(INDICATOR_NAMES)}
INDICATOR_COUNT = len(INDICATOR_NAMES)
