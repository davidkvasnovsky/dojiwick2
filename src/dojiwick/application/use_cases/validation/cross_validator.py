"""Purged K-fold cross-validation use case.

Slices a ``BacktestTimeSeries`` into purged folds and runs backtests on
each test fold to estimate out-of-sample Sharpe distribution.
"""

from dataclasses import dataclass

import numpy as np

from dojiwick.application.use_cases.run_backtest import BacktestService, BacktestTimeSeries
from dojiwick.compute.kernels.validation.purged_kfold import purged_kfold_splits
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
    purge_bars: int,
    embargo_bars: int,
) -> CVResult:
    """Run purged K-fold cross-validation over a backtest time series.

    For each fold, the test indices are used to slice a sub-series from
    ``series``, and a full backtest-with-hysteresis is run on that slice.
    The resulting Sharpe from each fold is collected.
    """

    splits = purged_kfold_splits(
        n_samples=series.n_bars,
        n_folds=n_folds,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )

    fold_sharpes: list[float] = []
    all_trade_returns: list[np.ndarray] = []

    for _train_idx, test_idx in splits:
        sub_series = series.slice_by_indices([int(i) for i in test_idx])

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
