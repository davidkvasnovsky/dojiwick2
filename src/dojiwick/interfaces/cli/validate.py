"""CLI entrypoint for walk-forward validation, cross-validation, and research gate.

Usage::

    python -m dojiwick.interfaces.cli.validate \
        --config config.toml --start 2025-01-01 --end 2025-06-01 --mode full-gate
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dojiwick.application.use_cases.run_backtest import BacktestService, BacktestTimeSeries

log = logging.getLogger(__name__)


async def _compute_pbo(service: BacktestService, series: BacktestTimeSeries) -> float:
    """Run a full backtest and compute PBO from the equity curve."""
    import numpy as np
    from dojiwick.compute.kernels.validation.cscv import compute_pbo

    result = await service.run_with_hysteresis(series)
    flat_pnl = result.summary.equity_curve
    if flat_pnl is not None:
        return float(compute_pbo(np.array(flat_pnl, dtype=np.float64)))
    return 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation checks on a backtest time series.")
    from dojiwick.interfaces.cli._shared import add_common_args

    add_common_args(parser)
    parser.add_argument(
        "--mode",
        type=str,
        choices=("walk-forward", "cross-validate", "full-gate"),
        default="full-gate",
        help="Validation mode (default: full-gate)",
    )
    return parser.parse_args()


async def _run_walk_forward(
    args: argparse.Namespace,
) -> None:
    """Run walk-forward validation and print per-window results."""
    from dojiwick.interfaces.cli._shared import build_service, load_settings_and_series
    from dojiwick.application.use_cases.validation.walk_forward_validator import walk_forward_validate

    settings, series, cleanup = await load_settings_and_series(args)
    try:
        service = build_service(settings)

        result = await walk_forward_validate(
            backtest_service=service,
            series=series,
            train_size=settings.research.wf_train_size,
            test_size=settings.research.wf_test_size,
            expanding=settings.research.wf_expanding,
            min_trades=settings.research.wf_min_trades,
        )

        print(f"\n{'=' * 76}")
        print("WALK-FORWARD VALIDATION")
        print(f"{'=' * 76}")
        print(
            f"  {'Window':<8} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'OOS/IS':>8} {'IS Trades':>10} {'OOS Trades':>11}"
        )
        print(f"  {'-' * 60}")
        for i, w in enumerate(result.windows):
            ratio = w.oos_sharpe / w.is_sharpe if w.is_sharpe != 0 else 0.0
            print(
                f"  {i + 1:<8} {w.is_sharpe:>10.4f} {w.oos_sharpe:>11.4f} {ratio:>8.4f}"
                f" {w.is_trades:>10} {w.oos_trades:>11}"
            )
        print(f"  {'-' * 60}")
        print(f"  Aggregate OOS Sharpe: {result.aggregate_oos_sharpe:.4f}")
        print(f"  Min OOS Sharpe:       {result.min_oos_sharpe:.4f}")
        print(f"  OOS/IS Ratio:         {result.oos_is_ratio:.4f}")
        print(f"{'=' * 76}\n")
    finally:
        await cleanup()


async def _run_cross_validate(
    args: argparse.Namespace,
) -> None:
    """Run cross-validation + PBO and print per-fold results."""
    from dojiwick.interfaces.cli._shared import build_service, load_settings_and_series
    from dojiwick.application.use_cases.validation.cross_validator import cross_validate

    settings, series, cleanup = await load_settings_and_series(args)
    try:
        service = build_service(settings)

        cv_result = await cross_validate(
            backtest_service=service,
            series=series,
            n_folds=settings.research.cv_folds,
            purge_bars=settings.research.purge_bars,
            embargo_bars=settings.research.embargo_bars,
        )

        pbo = await _compute_pbo(service, series)

        print(f"\n{'=' * 60}")
        print("CROSS-VALIDATION")
        print(f"{'=' * 60}")
        print(f"  {'Fold':<8} {'Sharpe':>12}")
        print(f"  {'-' * 22}")
        for i, s in enumerate(cv_result.fold_sharpes):
            print(f"  {i + 1:<8} {float(s):>12.4f}")
        print(f"  {'-' * 22}")
        print(f"  Mean Sharpe: {cv_result.mean_sharpe:.4f}")
        print(f"  Std Sharpe:  {cv_result.std_sharpe:.4f}")
        print(f"  Min Sharpe:  {cv_result.min_sharpe:.4f}")
        print(f"  PBO:         {pbo:.4f}")
        print(f"{'=' * 60}\n")
    finally:
        await cleanup()


async def _run_full_gate(
    args: argparse.Namespace,
) -> None:
    """Run full research gate: CV + PBO + walk-forward + all 9 criteria."""
    from dojiwick.application.use_cases.validation.gate_evaluator import DefaultGateEvaluator
    from dojiwick.config.targets import resolve_target_ids
    from dojiwick.interfaces.cli._shared import load_settings_and_series, print_gate_result, print_wf_windows

    settings, series, cleanup = await load_settings_and_series(args)
    try:
        target_ids = resolve_target_ids(settings)
        venue = str(settings.exchange.venue)
        product = str(settings.exchange.product)

        evaluator = DefaultGateEvaluator(
            settings=settings,
            series=series,
            target_ids=target_ids,
            venue=venue,
            product=product,
        )

        log.info("running full research gate evaluation")
        gate = await evaluator.evaluate(best_params={})

        verdict = "PASS" if gate.passed else "FAIL"
        print(f"\n{'=' * 60}")
        print(f"RESEARCH GATE: {verdict}")
        print(f"{'=' * 60}")
        print_gate_result(gate)
        print_wf_windows(gate.wf_windows)
    finally:
        await cleanup()


async def _run() -> None:
    args = _parse_args()
    mode = args.mode

    if mode == "walk-forward":
        await _run_walk_forward(args)
    elif mode == "cross-validate":
        await _run_cross_validate(args)
    else:
        await _run_full_gate(args)


def main() -> None:
    from dojiwick.interfaces.cli._shared import setup_env

    setup_env()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
