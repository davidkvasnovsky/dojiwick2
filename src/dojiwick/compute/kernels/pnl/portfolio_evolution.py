"""Portfolio state evolution between bars for sequential backtest replay."""

from dataclasses import replace
from datetime import datetime

import numpy as np

from dojiwick.compute.kernels.pnl.liquidation import cap_pnl_at_margin, check_liquidation
from dojiwick.compute.kernels.pnl.pnl import apply_slippage, gross_pnl, net_pnl
from dojiwick.domain.models.value_objects.batch_models import (
    BatchExecutionIntent,
    BatchPortfolioSnapshot,
)
from dojiwick.domain.models.value_objects.cost_model import CostModel
from dojiwick.domain.type_aliases import FloatVector


def compute_bar_net_pnl(
    intents: BatchExecutionIntent,
    next_prices: np.ndarray,
    cost: CostModel,
) -> FloatVector:
    """Compute per-pair net P&L for a single bar (slippage → gross → net → liquidation)."""
    slipped = apply_slippage(intents.entry_price, intents.action, cost.slippage_bps)
    gross = gross_pnl(intents.action, slipped, next_prices, intents.quantity, cost.leverage)
    net = net_pnl(
        gross,
        intents.notional_usd,
        cost.fee_bps,
        cost.fee_multiplier,
        funding_rate_per_bar=cost.funding_rate_per_bar,
        action=intents.action,
    )
    if cost.leverage > 1.0:
        liquidated = check_liquidation(
            slipped, next_prices, cost.leverage, intents.action, cost.maintenance_margin_rate
        )
        net = cap_pnl_at_margin(net, intents.notional_usd, cost.leverage, liquidated)
    return net


def evolve_portfolio(
    portfolio: BatchPortfolioSnapshot,
    bar_pnl: FloatVector,
    current_time: datetime,
    prev_time: datetime | None,
    *,
    has_open_position: np.ndarray | None = None,
    open_positions_total: np.ndarray | None = None,
) -> BatchPortfolioSnapshot:
    """Update portfolio state after a bar completes.

    - Adds bar P&L to equity
    - Floors equity at 0
    - Resets day_start_equity on day boundary
    - Optionally updates open position tracking vectors
    """
    new_equity = np.maximum(portfolio.equity_usd + bar_pnl, 0.0)

    day_changed = prev_time is not None and current_time.date() != prev_time.date()
    if day_changed:
        new_day_start = new_equity.copy()
    else:
        new_day_start = portfolio.day_start_equity_usd

    return replace(
        portfolio,
        equity_usd=new_equity,
        day_start_equity_usd=new_day_start,
        has_open_position=has_open_position if has_open_position is not None else portfolio.has_open_position,
        open_positions_total=open_positions_total
        if open_positions_total is not None
        else portfolio.open_positions_total,
    )
