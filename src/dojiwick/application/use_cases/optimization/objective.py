"""Vectorized optimization objective built on shared backtest service."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from dojiwick.application.models.pipeline_settings import OptimizationSettingsPort, PipelineSettings
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.policies.risk.engine import RiskPolicyEngine
from dojiwick.application.registry.strategy_registry import StrategyRegistry, build_default_strategy_registry
from dojiwick.application.use_cases.optimization.pruning import PruningCallback
from dojiwick.application.use_cases.optimization.search_space import (
    ParamSet,
    REGIME_PARAMS,
    SearchSpace,
    extract_regularization_baseline,
)
from dojiwick.application.use_cases.run_backtest import BacktestService, BacktestTimeSeries, PruningConfig
from dojiwick.domain.models.value_objects.outcome_models import BacktestSummary
from dojiwick.domain.models.value_objects.batch_models import BatchDecisionContext


def _init_cached_fields(obj: object, settings: PipelineSettings) -> None:
    """Set cached init=False fields on a frozen dataclass from settings."""
    object.__setattr__(
        obj,
        "_strategy_registry",
        build_default_strategy_registry(enabled=settings.trading.enabled_strategies),
    )
    object.__setattr__(
        obj,
        "_risk_engine",
        build_default_risk_engine(settings.risk),
    )


@dataclass(slots=True, frozen=True, kw_only=True)
class VectorObjective:
    """Optimization objective for deterministic strategy tuning."""

    settings: PipelineSettings
    apply_tuned: Callable[[ParamSet], PipelineSettings]
    base_context: BatchDecisionContext
    next_prices: np.ndarray
    target_ids: tuple[str, ...]
    venue: str
    product: str
    _strategy_registry: StrategyRegistry = field(init=False, repr=False)
    _risk_engine: RiskPolicyEngine = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _init_cached_fields(self, self.settings)

    async def evaluate(self, params: ParamSet, *, pruning_callback: PruningCallback | None = None) -> float:
        """Score one sampled parameter set."""

        tuned = self.apply_tuned(params)

        summary = await BacktestService(
            settings=tuned,
            strategy_registry=self._strategy_registry,
            risk_engine=self._risk_engine,
            target_ids=self.target_ids,
            venue=self.venue,
            product=self.product,
        ).run(self.base_context, self.next_prices)
        equity_start = float(self.base_context.portfolio.equity_usd[0])
        base = _base_score(tuned.optimization, summary, equity_start, n_bars=len(self.next_prices))
        # Regularization skipped — VectorObjective has no stored baseline
        return base


@dataclass(slots=True, frozen=True, kw_only=True)
class HysteresisObjective:
    """Optimization objective using BacktestTimeSeries with hysteresis replay.

    When ``train_fraction < 1.0``, each trial is scored on an OOS portion
    (last ``1 - train_fraction`` bars) blended with IS, directly
    incentivizing generalisable parameters during optimisation.
    """

    settings: PipelineSettings
    apply_tuned: Callable[[ParamSet], PipelineSettings]
    series: BacktestTimeSeries
    train_fraction: float = 1.0
    oos_weight: float = 0.7
    target_ids: tuple[str, ...]
    venue: str
    product: str
    _strategy_registry: StrategyRegistry = field(init=False, repr=False)
    _risk_engine: RiskPolicyEngine = field(init=False, repr=False)
    _baseline: dict[str, float] = field(init=False, repr=False)
    _search_names: frozenset[str] = field(init=False, repr=False)
    _is_series: BacktestTimeSeries = field(init=False, repr=False)
    _oos_series: BacktestTimeSeries = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _init_cached_fields(self, self.settings)
        object.__setattr__(
            self,
            "_baseline",
            extract_regularization_baseline(self.settings),
        )
        object.__setattr__(
            self,
            "_search_names",
            SearchSpace(
                partial_tp_enabled=self.settings.strategy.partial_tp_enabled,
                confluence_filter_enabled=self.settings.strategy.confluence_filter_enabled,
                enabled_strategies=self.settings.trading.enabled_strategies,
            ).strategy_param_names()
            | REGIME_PARAMS,
        )
        if self.train_fraction < 1.0:
            n = self.series.n_bars
            split = int(n * self.train_fraction)
            split = max(1, min(split, n - 1))
            object.__setattr__(self, "_is_series", self.series.slice_by_indices(range(split)))
            object.__setattr__(self, "_oos_series", self.series.slice_by_indices(range(split, n)))
        else:
            object.__setattr__(self, "_is_series", self.series)
            object.__setattr__(self, "_oos_series", self.series)  # unused sentinel

    async def evaluate(self, params: ParamSet, *, pruning_callback: PruningCallback | None = None) -> float:
        """Score one sampled parameter set using time-series replay."""
        tuned = self.apply_tuned(params)
        service = BacktestService(
            settings=tuned,
            strategy_registry=self._strategy_registry,
            risk_engine=self._risk_engine,
            target_ids=self.target_ids,
            venue=self.venue,
            product=self.product,
        )

        min_trades = tuned.optimization.objective_min_trades

        if self.train_fraction < 1.0:
            is_equity = float(self._is_series.contexts[0].portfolio.equity_usd[0])
            is_n_bars = self._is_series.n_bars
            pruning = (
                PruningConfig(
                    callback=pruning_callback,
                    min_trades=min_trades,
                    score_fn=lambda s: _base_score(tuned.optimization, s, is_equity, n_bars=is_n_bars),
                )
                if pruning_callback is not None
                else None
            )
            is_result, _ = await service.run_with_hysteresis_summary_only(
                self._is_series,
                skip_benchmark=True,
                pruning=pruning,
            )
            oos_result, _ = await service.run_with_hysteresis_summary_only(
                self._oos_series,
                skip_benchmark=True,
            )
            oos_equity = float(self._oos_series.contexts[0].portfolio.equity_usd[0])
            oos_n_bars = self._oos_series.n_bars
            is_base = _base_score(tuned.optimization, is_result, is_equity, n_bars=is_n_bars)
            oos_base = _base_score(tuned.optimization, oos_result, oos_equity, n_bars=oos_n_bars)
            base = (1 - self.oos_weight) * is_base + self.oos_weight * oos_base
        else:
            equity_start = float(self.series.contexts[0].portfolio.equity_usd[0])
            full_n_bars = self.series.n_bars
            pruning = (
                PruningConfig(
                    callback=pruning_callback,
                    min_trades=min_trades,
                    score_fn=lambda s: _base_score(tuned.optimization, s, equity_start, n_bars=full_n_bars),
                )
                if pruning_callback is not None
                else None
            )
            result, _ = await service.run_with_hysteresis_summary_only(
                self.series,
                skip_benchmark=True,
                pruning=pruning,
            )
            base = _base_score(tuned.optimization, result, equity_start, n_bars=full_n_bars)
        reg_strength = tuned.optimization.objective_regularization_strength
        return base - reg_strength * _regularization(params, self._baseline)


@dataclass(slots=True, frozen=True, kw_only=True)
class WalkForwardObjective:
    """Walk-forward optimization objective that trains across K contiguous folds.

    Incentivizes cross-temporal robustness during optimization by scoring
    parameters on multiple non-overlapping time windows and penalizing
    score variance across folds.
    """

    settings: PipelineSettings
    apply_tuned: Callable[[ParamSet], PipelineSettings]
    series: BacktestTimeSeries
    n_folds: int = 5
    consistency_penalty: float = 0.5
    target_ids: tuple[str, ...]
    venue: str
    product: str
    _strategy_registry: StrategyRegistry = field(init=False, repr=False)
    _risk_engine: RiskPolicyEngine = field(init=False, repr=False)
    _baseline: dict[str, float] = field(init=False, repr=False)
    _search_names: frozenset[str] = field(init=False, repr=False)
    _fold_series: tuple[BacktestTimeSeries, ...] = field(init=False, repr=False)
    _per_fold_min_trades: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _init_cached_fields(self, self.settings)
        object.__setattr__(
            self,
            "_baseline",
            extract_regularization_baseline(self.settings),
        )
        object.__setattr__(
            self,
            "_search_names",
            SearchSpace(
                partial_tp_enabled=self.settings.strategy.partial_tp_enabled,
                confluence_filter_enabled=self.settings.strategy.confluence_filter_enabled,
                enabled_strategies=self.settings.trading.enabled_strategies,
            ).strategy_param_names()
            | REGIME_PARAMS,
        )
        n = self.series.n_bars
        fold_size = n // self.n_folds
        folds: list[BacktestTimeSeries] = []
        for i in range(self.n_folds):
            start = i * fold_size
            end = start + fold_size if i < self.n_folds - 1 else n
            folds.append(self.series.slice_by_indices(range(start, end)))
        object.__setattr__(self, "_fold_series", tuple(folds))
        object.__setattr__(
            self,
            "_per_fold_min_trades",
            max(1, self.settings.optimization.objective_min_trades // self.n_folds),
        )

    async def evaluate(self, params: ParamSet, *, pruning_callback: PruningCallback | None = None) -> float:
        """Score one sampled parameter set across K walk-forward folds."""
        tuned = self.apply_tuned(params)
        service = BacktestService(
            settings=tuned,
            strategy_registry=self._strategy_registry,
            risk_engine=self._risk_engine,
            target_ids=self.target_ids,
            venue=self.venue,
            product=self.product,
        )

        scores: list[float] = []
        for i, fold in enumerate(self._fold_series):
            fold_equity = float(fold.contexts[0].portfolio.equity_usd[0])
            fold_n_bars = fold.n_bars
            pruning = (
                PruningConfig(
                    callback=pruning_callback,
                    min_trades=self._per_fold_min_trades,
                    score_fn=lambda s, fe=fold_equity, fn=fold_n_bars: _base_score(
                        tuned.optimization, s, fe, n_bars=fn, min_trades_override=self._per_fold_min_trades
                    ),
                )
                if pruning_callback is not None and i == 0
                else None
            )
            result, _ = await service.run_with_hysteresis_summary_only(
                fold,
                skip_benchmark=True,
                pruning=pruning,
            )
            score = _base_score(
                tuned.optimization,
                result,
                fold_equity,
                n_bars=fold_n_bars,
                min_trades_override=self._per_fold_min_trades,
            )
            if score <= tuned.optimization.objective_min_trades_penalty:
                return tuned.optimization.objective_min_trades_penalty
            scores.append(score)

        scores_arr = np.array(scores)
        reg_strength = tuned.optimization.objective_regularization_strength
        return float(
            np.mean(scores_arr)
            - self.consistency_penalty * np.std(scores_arr)
            - reg_strength * _regularization(params, self._baseline)
        )


def _base_score(
    opt: OptimizationSettingsPort,
    summary: BacktestSummary,
    equity_start: float,
    *,
    n_bars: int = 0,
    min_trades_override: int | None = None,
) -> float:
    """Composite score without regularisation penalty."""
    effective_min_trades = min_trades_override if min_trades_override is not None else opt.objective_min_trades

    # Graduated min-trades penalty instead of binary cliff
    if effective_min_trades > 0 and summary.trades < effective_min_trades:
        shortfall = (effective_min_trades - summary.trades) / effective_min_trades
        start = opt.objective_min_trades_penalty_start
        slope = -opt.objective_min_trades_penalty + start
        return max(opt.objective_min_trades_penalty, start - slope * shortfall)

    dd_for_penalty = summary.effective_max_drawdown_pct
    drawdown_ratio = dd_for_penalty / 100.0
    score = (
        opt.objective_return_weight * summary.sortino
        + opt.objective_sharpe_weight * summary.sharpe_like
        + opt.objective_win_rate_weight * summary.win_rate
        + opt.objective_profit_factor_weight * min(summary.profit_factor, opt.objective_profit_factor_cap)
        - opt.objective_drawdown_penalty * drawdown_ratio
    )

    # Progressive drawdown cliff -- exponential penalty above threshold
    if dd_for_penalty > opt.objective_max_drawdown_threshold:
        excess = (dd_for_penalty - opt.objective_max_drawdown_threshold) / 100.0
        score -= opt.objective_drawdown_cliff_penalty * excess**2

    # Trade frequency bonus (logarithmic — diminishing returns, never capped hard)
    if n_bars > 0 and summary.trades > 0:
        trades_per_1000_bars = summary.trades / n_bars * 1000
        score += opt.objective_trade_freq_weight * math.log2(1 + trades_per_1000_bars)

    # Trade density penalty — multiplicative reduction for very sparse strategies
    if n_bars > 0 and opt.objective_min_density_threshold > 0:
        density = summary.trades / n_bars
        if density < opt.objective_min_density_threshold:
            score *= density / opt.objective_min_density_threshold

    # High win-rate bonus
    if summary.win_rate > opt.objective_high_winrate_threshold:
        score += opt.objective_high_winrate_bonus * (summary.win_rate - opt.objective_high_winrate_threshold)

    # Expectancy component (normalized by avg notional)
    if opt.objective_expectancy_weight > 0 and summary.avg_notional_usd > 0:
        norm_expectancy = (summary.expectancy_usd / summary.avg_notional_usd) * 100.0
        score += opt.objective_expectancy_weight * norm_expectancy

    # Consecutive loss penalty
    if (
        opt.objective_consecutive_loss_penalty > 0
        and summary.max_consecutive_losses > opt.objective_consecutive_loss_threshold
    ):
        score -= (
            summary.max_consecutive_losses - opt.objective_consecutive_loss_threshold
        ) * opt.objective_consecutive_loss_penalty

    # Payoff ratio reward (avg_win / avg_loss) — penalizes loss asymmetry
    if opt.objective_payoff_ratio_weight > 0 and summary.payoff_ratio > 0:
        score += opt.objective_payoff_ratio_weight * min(summary.payoff_ratio, opt.objective_payoff_ratio_cap)

    # Total PnL: log-scaled return on starting capital (distinguishes 250x from 33x)
    if equity_start > 0:
        raw_return = summary.total_pnl_usd / equity_start
        score += opt.objective_pnl_weight * math.log1p(max(-0.9999, min(raw_return, opt.objective_pnl_cap)))

    return score


def _regularization(params: ParamSet, baseline: dict[str, float]) -> float:
    """L2 deviation penalty normalised by baseline values."""
    total = 0.0
    for k, v in baseline.items():
        if k not in params or abs(v) < 1e-9:
            continue
        total += (float(params[k]) - v) ** 2 / v**2
    return total
