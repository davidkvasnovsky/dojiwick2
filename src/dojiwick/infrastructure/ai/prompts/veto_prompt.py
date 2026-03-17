"""Veto prompt builder for LLM-as-filter."""

from functools import cache

from dojiwick.domain.indicator_schema import INDICATOR_NAMES
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchRegimeProfile,
    BatchTradeCandidate,
)

_MARKET_STATE_LABELS: dict[int, str] = {
    1: "TRENDING_UP",
    2: "TRENDING_DOWN",
    3: "RANGING",
    4: "VOLATILE",
}


@cache
def build_veto_system_prompt() -> str:
    """Return the static veto system prompt (cached)."""
    return (
        "You are a trade risk filter. A deterministic trading engine has generated "
        "a trade signal. Your role is to identify ONLY critical contextual "
        "disqualifiers that the deterministic system cannot detect.\n\n"
        "DEFAULT ACTION: APPROVE. You must justify any BLOCK with specific evidence.\n\n"
        "You may ONLY block for these 4 reasons:\n"
        "1. CONFLICTING_REGIME - Market structure contradicts signal direction\n"
        "2. EXTREME_VOLATILITY - Abnormal price action suggesting cascade risk\n"
        "3. CORRELATION_RISK - Multiple correlated positions amplify exposure\n"
        "4. STALE_SIGNAL - Indicators delayed or inconsistent\n\n"
        'Respond with JSON only: {"approved": true/false, "reason": "<reason_code>"}\n'
        'If approved, reason must be "approved".'
    )


def build_veto_user_prompt(
    context: BatchDecisionContext,
    candidates: BatchTradeCandidate,
    pair_index: int,
    regimes: BatchRegimeProfile | None = None,
) -> str:
    """Build a per-pair user prompt with market context and candidate details."""
    pair = context.market.pairs[pair_index]
    price = float(context.market.price[pair_index])

    indicators_row = context.market.indicators[pair_index]
    indicator_lines = "\n".join(f"  {name}: {float(indicators_row[i]):.4f}" for i, name in enumerate(INDICATOR_NAMES))

    action_code = int(candidates.action[pair_index])
    action_label = {0: "HOLD", 1: "BUY", -1: "SHORT"}.get(action_code, "UNKNOWN")
    entry = float(candidates.entry_price[pair_index])
    stop = float(candidates.stop_price[pair_index])
    tp = float(candidates.take_profit_price[pair_index])

    regime_section = ""
    if regimes is not None:
        state_code = int(regimes.coarse_state[pair_index])
        state_label = _MARKET_STATE_LABELS.get(state_code, "UNKNOWN")
        conf = float(regimes.confidence[pair_index])
        regime_section = f"\nRegime: {state_label} (confidence: {conf:.2f})"

    equity = float(context.portfolio.equity_usd[pair_index])
    open_positions = int(context.portfolio.open_positions_total[pair_index])
    has_open = bool(context.portfolio.has_open_position[pair_index])

    return (
        "<data>\n"
        f"Pair: {pair}\n"
        f"Price: {price:.4f}\n"
        f"Indicators:\n{indicator_lines}\n"
        f"{regime_section}\n"
        f"Candidate: {action_label} entry={entry:.4f} stop={stop:.4f} tp={tp:.4f}\n"
        f"Portfolio: equity=${equity:.2f} open_positions={open_positions} has_open={has_open}\n"
        "</data>"
    )
