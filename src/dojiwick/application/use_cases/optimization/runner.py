"""Optuna runner boundary for vectorized optimization objective."""

from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from dojiwick.application.use_cases.optimization.pruning import PrunedError, PruningCallback
from dojiwick.application.use_cases.optimization.search_space import ParamSet, TrialPort

if TYPE_CHECKING:
    import optuna

    from dojiwick.application.use_cases.validation.research_gate import GateResult

logger = logging.getLogger(__name__)


class OptimizationObjective(Protocol):
    """Protocol for optimization objective evaluation."""

    async def evaluate(self, params: ParamSet, *, pruning_callback: PruningCallback | None = None) -> float:
        """Return objective score."""
        ...


class SearchSpacePort(Protocol):
    """Protocol for trial sampling."""

    def sample(self, trial: TrialPort) -> ParamSet:
        """Return sampled params for one trial."""
        ...


@dataclass(slots=True, frozen=True, kw_only=True)
class OptimizationRunSpec:
    """Optimization run settings."""

    study_name: str
    storage_url: str
    trials: int
    direction: str = "maximize"
    trial_timeout_sec: float = 60.0
    warm_start_params: tuple[ParamSet, ...] = ()
    load_existing: bool = False
    multivariate_sampler: bool = True
    constant_liar: bool = True
    sampler_seed: int | None = None
    pruning_enabled: bool = True
    pruning_percentile: int = 25
    pruning_startup_trials: int = 30


@dataclass(slots=True, frozen=True, kw_only=True)
class OptimizationResult:
    """Normalized optimization result."""

    best_value: float
    best_params: ParamSet
    study_name: str
    gate_result: GateResult | None = None


class _OptunaTrialAdapter:
    """Adapts an Optuna Trial to the ``PruningCallback`` protocol."""

    __slots__ = ("_report", "_should_prune")

    def __init__(self, trial: optuna.trial.Trial) -> None:
        self._report: Callable[[float, int], None] = trial.report
        self._should_prune: Callable[[], bool] = trial.should_prune

    def report(self, value: float, step: int) -> None:
        self._report(value, step)

    def should_prune(self) -> bool:
        return self._should_prune()


def build_sampler(spec: OptimizationRunSpec) -> optuna.samplers.BaseSampler:
    """Build an Optuna TPE sampler from run spec."""
    import optuna

    return optuna.samplers.TPESampler(
        multivariate=spec.multivariate_sampler,
        group=spec.multivariate_sampler,
        constant_liar=spec.constant_liar,
        seed=spec.sampler_seed,
    )


def build_pruner(spec: OptimizationRunSpec) -> optuna.pruners.BasePruner | None:
    """Build an Optuna pruner from run spec, or None if pruning disabled."""
    if not spec.pruning_enabled:
        return None
    import optuna

    return optuna.pruners.PercentilePruner(
        percentile=spec.pruning_percentile,
        n_startup_trials=spec.pruning_startup_trials,
        n_warmup_steps=0,
    )


def build_storage(url: str) -> optuna.storages.RDBStorage:
    """Build an Optuna RDBStorage with pool_pre_ping enabled."""
    import optuna

    return optuna.storages.RDBStorage(url=url, engine_kwargs={"pool_pre_ping": True})


def create_study_from_spec(spec: OptimizationRunSpec, storage_url: str | None = None) -> optuna.study.Study:
    """Create (or recreate) an Optuna study from a run spec.

    Deletes any existing study with the same name, creates a fresh one,
    and enqueues warm-start trials.  Returns the ``optuna.Study`` object.
    """
    import optuna

    url = storage_url or spec.storage_url
    storage = build_storage(url)

    try:
        optuna.delete_study(study_name=spec.study_name, storage=storage)
        logger.info("deleted existing study '%s'", spec.study_name)
    except KeyError:
        pass

    sampler = build_sampler(spec)
    pruner = build_pruner(spec)
    study = optuna.create_study(
        study_name=spec.study_name,
        storage=storage,
        direction=spec.direction,
        sampler=sampler,
        pruner=pruner,
    )

    for warm_params in spec.warm_start_params:
        study.enqueue_trial(warm_params)
        logger.info("enqueued warm-start trial: %s", warm_params)

    return study


class OptunaRunner:
    """Runs Optuna optimization with optional import boundary."""

    async def run(
        self,
        *,
        spec: OptimizationRunSpec,
        search_space: SearchSpacePort,
        objective: OptimizationObjective,
    ) -> OptimizationResult:
        """Execute optimization and return best result."""

        import optuna

        if spec.load_existing:
            storage = build_storage(spec.storage_url)
            study = optuna.load_study(study_name=spec.study_name, storage=storage)
        else:
            study = create_study_from_spec(spec)

        loop = asyncio.get_running_loop()

        def _objective(trial: optuna.trial.Trial) -> float:
            pruning_cb: PruningCallback | None = _OptunaTrialAdapter(trial) if spec.pruning_enabled else None

            params = search_space.sample(trial)
            future = asyncio.run_coroutine_threadsafe(objective.evaluate(params, pruning_callback=pruning_cb), loop)
            try:
                result = future.result(timeout=spec.trial_timeout_sec)
                if trial.number % 10 == 0:
                    gc.collect()
                return result
            except PrunedError:
                raise optuna.TrialPruned()
            except TimeoutError:
                logger.warning("Trial %d timed out after %.1fs", trial.number, spec.trial_timeout_sec)
                gc.collect()
                raise

        await loop.run_in_executor(
            None,
            lambda: study.optimize(
                _objective,
                n_trials=spec.trials,
                catch=(TimeoutError,),
            ),
        )
        return OptimizationResult(
            best_value=float(study.best_value),
            best_params=dict(study.best_params),
            study_name=spec.study_name,
        )
