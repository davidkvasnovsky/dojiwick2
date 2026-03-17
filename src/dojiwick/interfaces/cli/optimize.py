"""CLI entrypoint for strategy optimization on historical data.

Usage::

    python -m dojiwick.interfaces.cli.optimize \
        --config config.toml --start 2025-01-01 --end 2025-06-01 --trials 50
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from dojiwick.application.use_cases.optimization.objective import HysteresisObjective, WalkForwardObjective
    from dojiwick.application.use_cases.optimization.runner import OptimizationRunSpec
    from dojiwick.application.use_cases.optimization.search_space import ParamSet
    from dojiwick.application.use_cases.run_backtest import BacktestTimeSeries
    from dojiwick.config.schema import Settings
    from dojiwick.domain.enums import ObjectiveMode

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize strategy parameters on historical data.")
    from dojiwick.interfaces.cli._shared import add_common_args

    add_common_args(parser)
    parser.add_argument("--gate", action="store_true", help="Run research gate after optimization")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes (default: 1)")
    return parser.parse_args()


def _build_apply_tuned(settings: Settings) -> Callable[[ParamSet], Settings]:
    """Build the apply_tuned callable for objectives."""
    from dojiwick.config.param_tuning import apply_params

    return lambda params: apply_params(settings, params, baseline=settings)


def _build_objective(
    settings: Settings,
    series: BacktestTimeSeries,
    mode: ObjectiveMode,
    train_fraction: float,
    target_ids: tuple[str, ...],
    venue: str,
    product: str,
) -> HysteresisObjective | WalkForwardObjective:
    """Build the right objective for the resolved mode."""
    from dojiwick.application.use_cases.optimization.objective import HysteresisObjective, WalkForwardObjective
    from dojiwick.domain.enums import ObjectiveMode as _OM

    apply_tuned = _build_apply_tuned(settings)
    if mode == _OM.WALK_FORWARD:
        return WalkForwardObjective(
            settings=settings,
            apply_tuned=apply_tuned,
            series=series,
            n_folds=settings.optimization.objective_cv_folds,
            consistency_penalty=settings.optimization.objective_consistency_penalty,
            target_ids=target_ids,
            venue=venue,
            product=product,
        )
    return HysteresisObjective(
        settings=settings,
        apply_tuned=apply_tuned,
        series=series,
        train_fraction=train_fraction,
        target_ids=target_ids,
        venue=venue,
        product=product,
    )


def _resolve_optuna_storage(settings: Settings) -> str:
    """Derive Optuna storage URL from database DSN if not explicitly configured."""
    url = settings.optimization.storage_url
    if url:
        return url
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(settings.database.dsn)
    if parsed.scheme in ("postgresql", "postgres"):
        parsed = parsed._replace(scheme="postgresql+psycopg")
    # Rewrite Docker service hostname to localhost for host-side CLI usage
    if parsed.hostname == "postgres":
        parsed = parsed._replace(netloc=parsed.netloc.replace("@postgres:", "@localhost:", 1))
    return urlunparse(parsed)


def _worker_fn(
    settings: Settings,
    series: BacktestTimeSeries,
    spec: OptimizationRunSpec,
    n_trials: int,
    train_fraction: float,
    objective_mode: str,
    target_ids: tuple[str, ...],
    venue: str,
    product: str,
) -> None:
    """Run optimization trials in a forked worker process."""
    from dojiwick.interfaces.cli._shared import setup_env

    setup_env()
    logging.getLogger("dojiwick.application.use_cases.run_backtest").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)

    from dataclasses import replace

    from dojiwick.domain.enums import ObjectiveMode
    from dojiwick.application.use_cases.optimization.runner import OptunaRunner
    from dojiwick.application.use_cases.optimization.search_space import SearchSpace

    # Workers load the existing study (warm-start already enqueued by parent)
    worker_spec = replace(spec, trials=n_trials)
    objective = _build_objective(
        settings, series, ObjectiveMode(objective_mode), train_fraction, target_ids, venue, product
    )
    asyncio.run(
        OptunaRunner().run(
            spec=worker_spec,
            search_space=SearchSpace(
                partial_tp_enabled=settings.strategy.partial_tp_enabled,
                confluence_filter_enabled=settings.strategy.confluence_filter_enabled,
                enabled_strategies=settings.trading.enabled_strategies,
            ),
            objective=objective,
        )
    )
    gc.collect()


def _generate_warm_start(settings: Settings) -> tuple[ParamSet, ...]:
    """Generate warm-start trials: baseline + 4 perturbations at +/-10%."""
    from dojiwick.config.param_tuning import generate_warm_start_params

    trials = generate_warm_start_params(settings)
    log.info("generated %d warm-start trials from baseline", len(trials))
    return trials


async def _run() -> None:
    args = _parse_args()

    from dojiwick.application.use_cases.optimization.runner import (
        OptunaRunner,
        OptimizationResult,
        OptimizationRunSpec,
    )
    from dojiwick.application.use_cases.optimization.search_space import SearchSpace
    from dojiwick.domain.enums import ObjectiveMode
    from dojiwick.interfaces.cli._shared import load_settings_and_series

    settings, series, cleanup = await load_settings_and_series(args)

    try:
        from dojiwick.config.targets import resolve_target_ids

        target_ids = resolve_target_ids(settings)
        venue = str(settings.exchange.venue)
        product = str(settings.exchange.product)

        resolved_mode = settings.optimization.objective_mode

        warm_start = _generate_warm_start(settings)
        optuna_storage = _resolve_optuna_storage(settings)

        spec = OptimizationRunSpec(
            study_name=settings.optimization.study_name,
            storage_url=optuna_storage,
            trials=settings.optimization.trials,
            trial_timeout_sec=settings.optimization.trial_timeout_sec,
            warm_start_params=warm_start,
            multivariate_sampler=settings.optimization.multivariate_sampler,
            constant_liar=settings.optimization.constant_liar,
            sampler_seed=settings.optimization.sampler_seed,
            pruning_enabled=settings.optimization.pruning_enabled,
            pruning_percentile=settings.optimization.pruning_percentile,
            pruning_startup_trials=settings.optimization.pruning_startup_trials,
        )

        train_fraction: float = settings.optimization.train_fraction
        workers: int = args.workers
        if workers > 1:
            import multiprocessing as mp
            import optuna

            from dojiwick.application.use_cases.optimization.runner import create_study_from_spec

            # Close DB/HTTP connections before forking -- candle data is already
            # loaded into ``series``; keeping the connections open would leak
            # file descriptors into forked workers (shared socket corruption).
            await cleanup()

            async def _noop() -> None:
                pass

            cleanup = _noop  # prevent double-close in finally block

            # Create study and enqueue warm-start trials BEFORE forking so
            # workers don't duplicate warm-start enqueue
            study = create_study_from_spec(spec)

            # Release parent's SQLAlchemy connections before forking --
            # pooled connections must NOT be shared across fork() boundaries.
            del study
            gc.collect()

            # Workers load the existing study -- no warm-start re-enqueue
            from dataclasses import replace

            worker_spec = replace(spec, trials=0, warm_start_params=(), load_existing=True)

            ctx = mp.get_context("spawn")
            trials_per_worker = spec.trials // workers
            remainder = spec.trials % workers
            log.info(
                "starting optimization: %d trials across %d workers (train_fraction=%.2f)",
                spec.trials,
                workers,
                train_fraction,
            )
            _Process = ctx.Process  # noqa: N806  # multiprocessing convention
            procs: list[mp.Process] = []
            for i in range(workers):
                n = trials_per_worker + (1 if i < remainder else 0)
                p: mp.Process = _Process(
                    target=_worker_fn,
                    args=(
                        settings,
                        series,
                        worker_spec,
                        n,
                        train_fraction,
                        resolved_mode.value,
                        target_ids,
                        venue,
                        product,
                    ),
                )  # pyright: ignore[reportAssignmentType]  # ctx.Process returns SpawnProcess, not mp.Process
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
            failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
            if failed:
                log.error("worker(s) %s failed", failed)
                sys.exit(1)

            from dojiwick.application.use_cases.optimization.runner import build_storage

            study = optuna.load_study(study_name=spec.study_name, storage=build_storage(spec.storage_url))
            result = OptimizationResult(
                best_value=float(study.best_value),
                best_params=dict(study.best_params),
                study_name=spec.study_name,
            )
        else:
            objective = _build_objective(settings, series, resolved_mode, train_fraction, target_ids, venue, product)
            if resolved_mode == ObjectiveMode.WALK_FORWARD:
                log.info(
                    "starting optimization: %d trials (walk_forward, %d folds)",
                    spec.trials,
                    settings.optimization.objective_cv_folds,
                )
            else:
                log.info("starting optimization: %d trials (train_fraction=%.2f)", spec.trials, train_fraction)
            logging.getLogger("optuna").setLevel(logging.WARNING)
            result = await OptunaRunner().run(
                spec=spec,
                search_space=SearchSpace(
                    partial_tp_enabled=settings.strategy.partial_tp_enabled,
                    confluence_filter_enabled=settings.strategy.confluence_filter_enabled,
                    enabled_strategies=settings.trading.enabled_strategies,
                ),
                objective=objective,
            )

        print(f"\n{'=' * 50}")
        print("OPTIMIZATION RESULTS")
        print(f"{'=' * 50}")
        print(f"  Best Score:  {result.best_value:.4f}")
        print(f"  Best Params: {result.best_params}")
        print(f"{'=' * 50}\n")

        if args.gate:
            from dojiwick.config.param_tuning import apply_params
            from dojiwick.application.use_cases.validation.gate_evaluator import DefaultGateEvaluator

            evaluator = DefaultGateEvaluator(
                settings=settings,
                series=series,
                target_ids=target_ids,
                venue=venue,
                product=product,
                apply_tuned=lambda params: apply_params(settings, params, baseline=settings),
            )
            gate_result = await evaluator.evaluate(result.best_params, workers=workers)
            from dojiwick.interfaces.cli._shared import print_gate_result, print_wf_windows

            print(f"\n{'=' * 50}")
            print("RESEARCH GATE")
            print(f"{'=' * 50}")
            print_gate_result(gate_result)
            print(f"{'=' * 50}")
            print_wf_windows(gate_result.wf_windows)

    finally:
        await cleanup()


def main() -> None:
    from dojiwick.interfaces.cli._shared import setup_env

    setup_env()
    # Suppress noisy per-bar backtest logs during optimization
    logging.getLogger("dojiwick.application.use_cases.run_backtest").setLevel(logging.WARNING)
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
