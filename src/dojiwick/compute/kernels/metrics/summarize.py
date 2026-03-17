"""Vectorized summary metrics kernel."""

import numpy as np

from dojiwick.domain.models.value_objects.outcome_models import BacktestSummary
from dojiwick.domain.type_aliases import FloatVector


_INTERVAL_BARS: dict[str, float] = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
}


def interval_to_bars_per_year(interval: str) -> float:
    """Convert a candle interval string to the number of bars per year."""
    if interval not in _INTERVAL_BARS:
        raise ValueError(f"unsupported candle_interval: {interval!r}")
    return _INTERVAL_BARS[interval]


class SummarizeResult:
    """Return type for ``summarize``: summary + trade-level returns."""

    __slots__ = ("summary", "trade_returns")

    def __init__(self, summary: BacktestSummary, trade_returns: FloatVector) -> None:
        self.summary = summary
        self.trade_returns = trade_returns


def scalar_profit_factor(gross_wins: float, gross_losses: float) -> float:
    """Profit factor from pre-aggregated win/loss totals.

    Canonical scalar version of the formula used in ``summarize()``.
    """
    if gross_losses > 0:
        return gross_wins / gross_losses
    return float("inf") if gross_wins > 0 else 0.0


def compute_daily_sharpe(
    bar_pnls_total: FloatVector,
    portfolio_equity: FloatVector,
    initial_equity: float,
    bars_per_day: int,
) -> float:
    """Daily-return Sharpe ratio annualized by sqrt(365).

    Parameters
    ----------
    bar_pnls_total:
        Total PnL per bar summed across all pairs, shape ``(n_bars,)``.
    portfolio_equity:
        End-of-bar portfolio equity, shape ``(n_bars,)``.
    initial_equity:
        Portfolio equity before bar 0.
    bars_per_day:
        Number of bars in one calendar day.
    """
    n_bars = len(bar_pnls_total)
    n_days = n_bars // bars_per_day
    if n_days <= 1 or bars_per_day <= 0:
        return 0.0
    daily_pnl = bar_pnls_total[: n_days * bars_per_day].reshape(n_days, bars_per_day).sum(axis=1)
    daily_equity = np.empty(n_days)
    daily_equity[0] = initial_equity
    daily_equity[1:] = portfolio_equity[bars_per_day - 1 :: bars_per_day][: n_days - 1]
    daily_returns = np.divide(daily_pnl, daily_equity, out=np.zeros(n_days), where=daily_equity > 0)
    daily_mean = float(np.mean(daily_returns))
    daily_std = float(np.std(daily_returns))
    return (daily_mean / daily_std * float(np.sqrt(365))) if daily_std > 0 else 0.0


def quick_sharpe(
    net_pnl: FloatVector,
    notional: FloatVector,
    *,
    n_bars: int,
    bars_per_year: float = 8760.0,
) -> float:
    """Lightweight Sharpe-like score for pruning checkpoints.

    Computes only mean/std of trade returns and annualizes — skips equity
    curves, drawdowns, and all other summary fields.
    """
    trade_rows = notional > 0.0
    if not np.any(trade_rows):
        return 0.0

    pnl = net_pnl[trade_rows]
    notional_active = notional[trade_rows]
    returns = pnl / notional_active * 100.0
    trades = len(pnl)

    mean_return = float(np.mean(returns))
    stdev = float(np.std(returns))
    sharpe = 0.0 if stdev == 0.0 else mean_return / stdev

    if n_bars > 0 and trades > 1:
        trades_per_year = trades * (bars_per_year / n_bars)
        sharpe *= float(np.sqrt(trades_per_year))

    return sharpe


def summarize(
    net_pnl: FloatVector,
    notional: FloatVector,
    *,
    compute_curves: bool = True,
    n_bars: int = 0,
    bars_per_year: float = 8760.0,
) -> SummarizeResult:
    """Compute summary metrics from trade-level vectors."""

    trade_rows = notional > 0.0
    if not np.any(trade_rows):
        return SummarizeResult(
            BacktestSummary(
                trades=0,
                total_pnl_usd=0.0,
                win_rate=0.0,
                expectancy_usd=0.0,
                sharpe_like=0.0,
                max_drawdown_pct=0.0,
            ),
            np.array([], dtype=np.float64),
        )

    pnl = net_pnl[trade_rows]
    notional_active = notional[trade_rows]
    returns = pnl / notional_active * 100.0

    total_pnl = float(np.sum(pnl))
    trades = int(len(pnl))

    win_mask = pnl > 0.0
    loss_mask = pnl < 0.0
    wins = pnl[win_mask]
    losses = pnl[loss_mask]

    win_rate = float(np.mean(win_mask))
    expectancy = float(total_pnl / trades)
    avg_notional = float(np.mean(notional_active))

    mean_return = float(np.mean(returns))
    stdev = float(np.std(returns))
    sharpe_like = 0.0 if stdev == 0.0 else mean_return / stdev

    eq = np.cumprod(np.maximum(1.0 + returns / 100.0, 0.0))
    peaks = np.maximum.accumulate(eq)
    dd_arr = np.where(peaks > 0.0, (peaks - eq) / peaks * 100.0, 0.0)
    max_drawdown = float(np.max(dd_arr)) if len(dd_arr) else 0.0
    equity_curve = eq if compute_curves else None
    drawdowns = dd_arr if compute_curves else None

    # Sortino: mean return / downside std
    downside = returns[returns < 0.0]
    downside_std = float(np.std(downside)) if len(downside) > 0 else 0.0
    sortino = 0.0 if downside_std == 0.0 else mean_return / downside_std

    # Annualize Sharpe and Sortino when bar count is known
    if n_bars > 0 and trades > 1:
        trades_per_year = trades * (bars_per_year / n_bars)
        ann = float(np.sqrt(trades_per_year))
        sharpe_like = sharpe_like * ann
        sortino = sortino * ann

    # Calmar: annualized return / max drawdown (use actual bar-based annualization)
    if n_bars > 0 and trades > 0:
        total_return_pct = mean_return * trades
        year_fraction = n_bars / bars_per_year
        annualized_return = total_return_pct / year_fraction if year_fraction > 0 else 0.0
    else:
        annualized_return = mean_return * np.sqrt(252.0) if trades > 0 else 0.0
    calmar = 0.0 if max_drawdown == 0.0 else float(annualized_return) / max_drawdown

    # Profit factor: sum(wins) / sum(losses)
    sum_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = float(np.sum(wins)) / sum_losses if sum_losses > 0.0 else float("inf") if len(wins) > 0 else 0.0

    # Max consecutive losses
    max_consecutive_losses = _max_consecutive(loss_mask.astype(np.int64))

    # Payoff ratio: avg win / avg loss
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 0.0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0.0 else 0.0

    return SummarizeResult(
        BacktestSummary(
            trades=trades,
            total_pnl_usd=total_pnl,
            win_rate=win_rate,
            expectancy_usd=expectancy,
            sharpe_like=sharpe_like,
            max_drawdown_pct=max_drawdown,
            sortino=sortino,
            calmar=calmar,
            profit_factor=profit_factor,
            max_consecutive_losses=max_consecutive_losses,
            payoff_ratio=payoff_ratio,
            avg_notional_usd=avg_notional,
            equity_curve=equity_curve,
            drawdown_curve=drawdowns,
        ),
        returns,
    )


def _max_consecutive(flags: np.ndarray) -> int:
    """Compute the longest run of 1s in a binary array."""
    if len(flags) == 0:
        return 0
    padded = np.concatenate(([0], flags, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return 0
    return int(np.max(ends - starts))
