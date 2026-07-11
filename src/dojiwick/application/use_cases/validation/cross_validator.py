"""Embargoed K-fold cross-validation use case.

Slices a ``BacktestTimeSeries`` into contiguous folds and runs a backtest on
each to estimate the out-of-sample Sharpe distribution. Every fold after the
first drops its leading ``embargo_bars``: consecutive folds share a boundary,
and without the gap a position or warm hysteresis state from the end of one
fold leaks into the start of the next.
"""

from dataclasses import dataclass

import numpy as np

from dojiwick.application.use_cases.run_backtest import BacktestService, BacktestTimeSeries
from dojiwick.domain.type_aliases import FloatVector


@dataclass(slots=True, frozen=True, kw_only=True)
class CVResult:
    """Cross-validation result."""

    fold_sharpes: FloatVector
    mean_sharpe: float
    std_sharpe: float
    min_sharpe: float
    trade_returns: np.ndarray | None = None


async def cross_validate(
    *,
    backtest_service: BacktestService,
    series: BacktestTimeSeries,
    n_folds: int,
    embargo_bars: int,
) -> CVResult:
    """Run embargoed K-fold cross-validation over a backtest time series."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    fold_sharpes: list[float] = []
    all_trade_returns: list[np.ndarray] = []

    for fold_idx, fold in enumerate(np.array_split(np.arange(series.n_bars), n_folds)):
        indices = fold[embargo_bars:] if fold_idx > 0 else fold
        if len(indices) < 2:
            raise ValueError(f"fold {fold_idx} has {len(indices)} bars after embargo — reduce n_folds or embargo_bars")
        sub_series = series.slice_by_indices([int(i) for i in indices])

        summary, trade_rets = await backtest_service.run_with_hysteresis_summary_only(sub_series)
        fold_sharpes.append(summary.sharpe_like)
        if len(trade_rets) > 0:
            all_trade_returns.append(trade_rets)

    sharpes = np.array(fold_sharpes, dtype=np.float64)
    combined_returns = np.concatenate(all_trade_returns) if all_trade_returns else None
    return CVResult(
        fold_sharpes=sharpes,
        mean_sharpe=float(np.mean(sharpes)),
        std_sharpe=float(np.std(sharpes)),
        min_sharpe=float(np.min(sharpes)),
        trade_returns=combined_returns,
    )
