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
    pass

log = logging.getLogger(__name__)


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
    from dojiwick.application.use_cases.validation.walk_forward_validator import walk_forward_validate
    from dojiwick.interfaces.cli._shared import build_service, load_settings_and_series

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
    from dojiwick.application.use_cases.validation.cross_validator import cross_validate
    from dojiwick.application.use_cases.validation.gate_evaluator import compute_pbo_from_cv
    from dojiwick.interfaces.cli._shared import build_service, load_settings_and_series

    settings, series, cleanup = await load_settings_and_series(args)
    try:
        service = build_service(settings)

        cv_result = await cross_validate(
            backtest_service=service,
            series=series,
            n_folds=settings.research.cv_folds,
            embargo_bars=settings.research.embargo_bars,
        )

        # PBO needs a returns vector; the equity curve is a monotone level
        # series whose block Sharpes are ~always positive (PBO trivially 0)
        pbo = compute_pbo_from_cv(
            cv_result,
            pbo_min_trade_returns=settings.research.pbo_min_trade_returns,
            pbo_max_partitions=settings.research.pbo_max_partitions,
        )

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
) -> int:
    """Run full research gate: CV + PBO + walk-forward + all 9 criteria."""
    from dojiwick.interfaces.cli._shared import build_gate_evaluator, load_settings_and_series, print_gate_block

    settings, series, cleanup = await load_settings_and_series(args)
    try:
        evaluator = build_gate_evaluator(settings, series)

        log.info("running full research gate evaluation")
        # No apply_tuned_from and an empty param set: the evaluator gates the
        # loaded config exactly as promoted, with no re-tuning applied.
        gate = await evaluator.evaluate(best_params={})

        return print_gate_block(gate)
    finally:
        await cleanup()


async def _run() -> int:
    args = _parse_args()
    mode = args.mode

    if mode == "walk-forward":
        await _run_walk_forward(args)
        return 0
    if mode == "cross-validate":
        await _run_cross_validate(args)
        return 0
    return await _run_full_gate(args)


def main() -> int:
    from dojiwick.interfaces.cli._shared import setup_env

    setup_env()
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
