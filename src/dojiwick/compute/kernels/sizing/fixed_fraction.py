"""Vectorized position sizing kernel."""

import logging

import numpy as np

from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchExecutionIntent,
    BatchRiskAssessment,
    BatchTradeCandidate,
)
from dojiwick.compute.kernels.math import safe_divide
from dojiwick.domain.enums import TradeAction
from dojiwick.domain.models.value_objects.params import RiskParams

log = logging.getLogger(__name__)


def size_intents(
    *,
    context: BatchDecisionContext,
    candidate: BatchTradeCandidate,
    assessment: BatchRiskAssessment,
    risk_params: tuple[RiskParams, ...],
    leverage: float = 1.0,
) -> BatchExecutionIntent:
    """Build execution intents with deterministic fixed-fraction sizing."""

    size = context.size
    prices = candidate.entry_price

    active = assessment.allowed_mask & (candidate.action != TradeAction.HOLD.value)
    quantity = np.zeros(size, dtype=np.float64)
    notional = np.zeros(size, dtype=np.float64)

    equity = context.portfolio.equity_usd

    _fields = np.array(
        [
            (
                rp.risk_per_trade_pct,
                rp.max_notional_pct_of_equity,
                rp.min_notional_usd,
                rp.max_risk_inflation_mult,
                rp.max_notional_usd,
            )
            for rp in risk_params
        ],
        dtype=np.float64,
    )
    risk_pct = _fields[:, 0]
    max_notional_pct = _fields[:, 1]
    min_notional_arr = _fields[:, 2]
    max_inflation = _fields[:, 3]
    max_notional_usd_arr = _fields[:, 4]

    risk_usd = equity * risk_pct / 100.0
    stop_distance = np.abs(candidate.entry_price - candidate.stop_price)

    # Defense-in-depth: deactivate rows with zero stop distance or a
    # non-positive price before any division uses them
    zero_stop = stop_distance == 0.0
    active = active & ~zero_stop & (prices > 0.0)

    raw_quantity = safe_divide(risk_usd, stop_distance)
    raw_notional = raw_quantity * prices
    max_notional_pct_cap = equity * max_notional_pct / 100.0 * leverage
    max_notional = np.minimum(max_notional_pct_cap, max_notional_usd_arr)

    clipped_notional = np.clip(raw_notional, min_notional_arr, max_notional)
    clipped_notional[~active] = 0.0
    quantity[active] = clipped_notional[active] / prices[active]
    notional[active] = clipped_notional[active]

    # Guard: min-notional clip must not inflate risk beyond policy
    effective_risk_pct = (
        np.divide(
            clipped_notional * stop_distance,
            prices * equity,
            out=np.zeros(size, dtype=np.float64),
            where=(prices > 0) & (equity > 0),
        )
        * 100.0
    )
    oversized = active & (effective_risk_pct > risk_pct * max_inflation)
    if np.any(oversized):
        oversized_pairs = [context.market.pairs[i] for i in np.flatnonzero(oversized)]
        log.warning("deactivated oversized rows: %s", oversized_pairs)
    quantity[oversized] = 0.0
    notional[oversized] = 0.0
    active[oversized] = False

    invalid = active & (~np.isfinite(quantity) | ~np.isfinite(notional))
    if np.any(invalid):
        invalid_pairs = [context.market.pairs[i] for i in np.flatnonzero(invalid)]
        log.warning("deactivated rows with invalid sizing: %s", invalid_pairs)
    quantity[invalid] = 0.0
    notional[invalid] = 0.0
    active[invalid] = False

    return BatchExecutionIntent(
        pairs=context.market.pairs,
        action=candidate.action,
        quantity=quantity,
        notional_usd=notional,
        entry_price=candidate.entry_price,
        stop_price=candidate.stop_price,
        take_profit_price=candidate.take_profit_price,
        strategy_name=candidate.strategy_name,
        strategy_variant=candidate.strategy_variant,
        active_mask=active.astype(np.bool_),
    )
