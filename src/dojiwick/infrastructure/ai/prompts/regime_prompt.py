"""Regime classification prompt builder for LLM ensemble."""

from functools import cache

from dojiwick.domain.indicator_schema import INDICATOR_NAMES
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchRegimeProfile,
)

_MARKET_STATE_LABELS: dict[int, str] = {
    1: "TRENDING_UP",
    2: "TRENDING_DOWN",
    3: "RANGING",
    4: "VOLATILE",
}


@cache
def build_regime_system_prompt() -> str:
    """Return the static regime classification system prompt (cached)."""
    return (
        "You are a market regime classifier. Given market indicators, classify "
        "the current regime into exactly one of these 4 states:\n\n"
        "1. TRENDING_UP - Sustained upward price movement with strong momentum\n"
        "2. TRENDING_DOWN - Sustained downward price movement with strong momentum\n"
        "3. RANGING - Sideways price action within a band, low directional conviction\n"
        "4. VOLATILE - High variance, erratic moves, breakout or breakdown risk\n\n"
        "A deterministic classifier has already produced a baseline. Your job is to "
        "confirm or disagree based on contextual nuance the deterministic model may miss.\n\n"
        "Respond with JSON only: "
        '{"state": "<STATE>", "confidence": <0.0-1.0>}\n'
        "state must be one of: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE."
    )


def build_regime_user_prompt(
    context: BatchDecisionContext,
    deterministic_regime: BatchRegimeProfile,
    pair_index: int,
) -> str:
    """Build a per-pair user prompt for regime classification."""
    pair = context.market.pairs[pair_index]
    price = float(context.market.price[pair_index])

    indicators_row = context.market.indicators[pair_index]
    indicator_lines = "\n".join(f"  {name}: {float(indicators_row[i]):.4f}" for i, name in enumerate(INDICATOR_NAMES))

    det_state_code = int(deterministic_regime.coarse_state[pair_index])
    det_label = _MARKET_STATE_LABELS.get(det_state_code, "UNKNOWN")
    det_conf = float(deterministic_regime.confidence[pair_index])

    return (
        "<data>\n"
        f"Pair: {pair}\n"
        f"Price: {price:.4f}\n"
        f"Indicators:\n{indicator_lines}\n"
        f"\nDeterministic baseline: {det_label} (confidence: {det_conf:.2f})\n"
        f"Classify the current regime.\n"
        "</data>"
    )
