"""Walk-forward validation use case.

Runs backtests on both in-sample and out-of-sample windows produced by
the walk-forward splitter, then computes degradation metrics.
"""

from dataclasses import dataclass

import numpy as np

from dojiwick.application.use_cases.run_backtest import BacktestService, BacktestTimeSeries
from dojiwick.compute.kernels.validation.walk_forward import walk_forward_splits


@dataclass(slots=True, frozen=True, kw_only=True)
class WindowResult:
    """Single walk-forward window result."""

    is_sharpe: float
    oos_sharpe: float
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int
    is_trades: int = 0
    oos_trades: int = 0


@dataclass(slots=True, frozen=True, kw_only=True)
class WalkForwardResult:
    """Aggregate walk-forward validation result."""

    windows: tuple[WindowResult, ...]
    aggregate_oos_sharpe: float
    oos_is_ratio: float
    min_oos_sharpe: float = 0.0


async def walk_forward_validate(
    *,
    backtest_service: BacktestService,
    series: BacktestTimeSeries,
    train_size: int,
    test_size: int,
    expanding: bool = False,
    min_trades: int = 0,
) -> WalkForwardResult:
    """Run walk-forward validation over a backtest time series.

    The OOS/IS Sharpe ratio measures degradation.
    """

    splits = walk_forward_splits(
        n_samples=series.n_bars,
        train_size=train_size,
        test_size=test_size,
        expanding=expanding,
    )

    windows: list[WindowResult] = []
    oos_sharpes: list[float] = []
    is_sharpes: list[float] = []

    for train_idx, test_idx in splits:
        train_list = [int(i) for i in train_idx]
        test_list = [int(i) for i in test_idx]

        is_series = series.slice_by_indices(train_list)
        oos_series = series.slice_by_indices(test_list)

        is_result, _ = await backtest_service.run_with_hysteresis_summary_only(is_series)
        oos_result, _ = await backtest_service.run_with_hysteresis_summary_only(oos_series)

        windows.append(
            WindowResult(
                is_sharpe=is_result.sharpe_like,
                oos_sharpe=oos_result.sharpe_like,
                is_start=train_list[0],
                is_end=train_list[-1],
                oos_start=test_list[0],
                oos_end=test_list[-1],
                is_trades=is_result.trades,
                oos_trades=oos_result.trades,
            )
        )
        if min_trades <= 0 or oos_result.trades >= min_trades:
            is_sharpes.append(is_result.sharpe_like)
            oos_sharpes.append(oos_result.sharpe_like)

    mean_is = float(np.mean(is_sharpes)) if is_sharpes else 0.0
    mean_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    oos_is_ratio = mean_oos / mean_is if mean_is != 0.0 else 0.0
    min_oos = float(np.min(oos_sharpes)) if oos_sharpes else 0.0

    return WalkForwardResult(
        windows=tuple(windows),
        aggregate_oos_sharpe=mean_oos,
        oos_is_ratio=oos_is_ratio,
        min_oos_sharpe=min_oos,
    )
