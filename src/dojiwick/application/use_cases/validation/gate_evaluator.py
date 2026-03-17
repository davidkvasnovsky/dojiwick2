"""Concrete research gate evaluator wiring all validation checks."""

import asyncio
import logging
import multiprocessing as mp
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from dojiwick.application.models.pipeline_settings import PipelineSettings
from dojiwick.application.use_cases.optimization.search_space import ParamSet
from dojiwick.application.use_cases.run_backtest import BacktestService, BacktestTimeSeries, build_backtest_service
from dojiwick.application.use_cases.validation.cross_validator import CVResult, cross_validate
from dojiwick.application.use_cases.validation.research_gate import (
    ConcentrationResult,
    ContinuousBacktestResult,
    GateCheckResults,
    GateResult,
    GateThresholds,
    PairRobustnessResult,
    RegimePFResult,
    ShockTestResult,
    evaluate_research_gate,
)
from dojiwick.application.use_cases.validation.walk_forward_validator import WalkForwardResult, walk_forward_validate
from dojiwick.compute.kernels.metrics.summarize import scalar_profit_factor
from dojiwick.compute.kernels.validation.cscv import compute_pbo
from dojiwick.domain.enums import MarketState
from dojiwick.domain.models.value_objects.outcome_models import TradeDetail

log = logging.getLogger(__name__)


def _worker_init(
    settings: PipelineSettings,
    target_ids: tuple[str, ...],
    venue: str,
    product: str,
) -> BacktestService:
    """Shared setup for multiprocessing gate workers."""
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("dojiwick.application.use_cases.run_backtest").setLevel(logging.WARNING)
    return build_backtest_service(settings, target_ids=target_ids, venue=venue, product=product)


def _compute_pbo(cv_result: CVResult, *, pbo_min_trade_returns: int = 8, pbo_max_partitions: int = 16) -> float:
    """Compute PBO from CV result, preferring trade-level returns over fold Sharpes."""
    if cv_result.trade_returns is not None and len(cv_result.trade_returns) >= pbo_min_trade_returns:
        n_partitions = min(pbo_max_partitions, max(4, len(cv_result.trade_returns) // 10 * 2))
        return compute_pbo(cv_result.trade_returns, n_partitions=n_partitions)
    n_f = len(cv_result.fold_sharpes)
    n_partitions = max(4, n_f // 2 * 2)
    if n_f < n_partitions:
        return 0.0
    log.warning("PBO fallback: using fold Sharpes (%d folds), insufficient trade returns", n_f)
    return compute_pbo(cv_result.fold_sharpes, n_partitions=n_partitions)


class _CVWorkerResult(NamedTuple):
    fold_sharpes: list[float]
    trade_returns: list[float]
    mean_sharpe: float
    std_sharpe: float
    min_sharpe: float
    pbo: float


def _gate_cv_worker(
    settings: PipelineSettings,
    series: BacktestTimeSeries,
    n_folds: int,
    purge_bars: int,
    embargo_bars: int,
    target_ids: tuple[str, ...],
    venue: str,
    product: str,
    pbo_min_trade_returns: int = 8,
    pbo_max_partitions: int = 16,
) -> _CVWorkerResult:
    """Run cross-validation + PBO in a separate process. Returns serializable results."""
    service = _worker_init(settings, target_ids, venue, product)

    cv_result = asyncio.run(
        cross_validate(
            backtest_service=service,
            series=series,
            n_folds=n_folds,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )
    )

    pbo = _compute_pbo(cv_result, pbo_min_trade_returns=pbo_min_trade_returns, pbo_max_partitions=pbo_max_partitions)

    return _CVWorkerResult(
        fold_sharpes=cv_result.fold_sharpes.tolist(),
        trade_returns=[] if cv_result.trade_returns is None else cv_result.trade_returns.tolist(),
        mean_sharpe=cv_result.mean_sharpe,
        std_sharpe=cv_result.std_sharpe,
        min_sharpe=cv_result.min_sharpe,
        pbo=pbo,
    )


def _gate_wf_worker(
    settings: PipelineSettings,
    series: BacktestTimeSeries,
    train_size: int,
    test_size: int,
    expanding: bool,
    min_trades: int,
    target_ids: tuple[str, ...],
    venue: str,
    product: str,
) -> WalkForwardResult:
    """Run walk-forward validation in a separate process."""
    service = _worker_init(settings, target_ids, venue, product)

    return asyncio.run(
        walk_forward_validate(
            backtest_service=service,
            series=series,
            train_size=train_size,
            test_size=test_size,
            expanding=expanding,
            min_trades=min_trades,
        )
    )


@dataclass(slots=True, frozen=True, kw_only=True)
class DefaultGateEvaluator:
    """Concrete implementation that wires CV, PBO, and walk-forward checks."""

    settings: PipelineSettings
    series: BacktestTimeSeries
    target_ids: tuple[str, ...]
    venue: str
    product: str
    apply_tuned: Callable[[ParamSet], PipelineSettings] | None = None

    async def evaluate(self, best_params: ParamSet, workers: int = 1) -> GateResult:
        """Run all three validation checks against the best params."""
        if self.apply_tuned is not None:
            tuned = self.apply_tuned(best_params)
        else:
            tuned = self.settings

        if workers > 1:
            return await self._evaluate_parallel(tuned, workers)

        service = build_backtest_service(tuned, target_ids=self.target_ids, venue=self.venue, product=self.product)

        cv_result = await cross_validate(
            backtest_service=service,
            series=self.series,
            n_folds=self.settings.research.cv_folds,
            purge_bars=self.settings.research.purge_bars,
            embargo_bars=self.settings.research.embargo_bars,
        )

        pbo = _compute_pbo(
            cv_result,
            pbo_min_trade_returns=self.settings.research.pbo_min_trade_returns,
            pbo_max_partitions=self.settings.research.pbo_max_partitions,
        )

        wf_result = await walk_forward_validate(
            backtest_service=service,
            series=self.series,
            train_size=self.settings.research.wf_train_size,
            test_size=self.settings.research.wf_test_size,
            expanding=self.settings.research.wf_expanding,
            min_trades=self.settings.research.wf_min_trades,
        )

        continuous = await self._run_continuous_check(service)

        return self._build_gate_result(cv_result, pbo, wf_result, continuous)

    async def _evaluate_parallel(self, tuned: PipelineSettings, workers: int) -> GateResult:
        """Run CV and WF in parallel processes."""
        loop = asyncio.get_running_loop()
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=min(workers, 2))

        try:
            cv_future = pool.apply_async(
                _gate_cv_worker,
                (
                    tuned,
                    self.series,
                    self.settings.research.cv_folds,
                    self.settings.research.purge_bars,
                    self.settings.research.embargo_bars,
                    self.target_ids,
                    self.venue,
                    self.product,
                    self.settings.research.pbo_min_trade_returns,
                    self.settings.research.pbo_max_partitions,
                ),
            )
            wf_future = pool.apply_async(
                _gate_wf_worker,
                (
                    tuned,
                    self.series,
                    self.settings.research.wf_train_size,
                    self.settings.research.wf_test_size,
                    self.settings.research.wf_expanding,
                    self.settings.research.wf_min_trades,
                    self.target_ids,
                    self.venue,
                    self.product,
                ),
            )
            pool.close()

            cv_raw, wf_result = await asyncio.gather(
                loop.run_in_executor(None, lambda: cv_future.get(timeout=3600)),
                loop.run_in_executor(None, lambda: wf_future.get(timeout=3600)),
            )
        finally:
            pool.terminate()
            pool.join()

        cv_result = CVResult(
            fold_sharpes=np.array(cv_raw.fold_sharpes, dtype=np.float64),
            mean_sharpe=cv_raw.mean_sharpe,
            std_sharpe=cv_raw.std_sharpe,
            min_sharpe=cv_raw.min_sharpe,
            trade_returns=np.array(cv_raw.trade_returns, dtype=np.float64) if cv_raw.trade_returns else None,
        )

        service = build_backtest_service(tuned, target_ids=self.target_ids, venue=self.venue, product=self.product)
        continuous = await self._run_continuous_check(service)

        return self._build_gate_result(cv_result, cv_raw.pbo, wf_result, continuous)

    def _build_gate_result(
        self,
        cv_result: CVResult,
        pbo: float,
        wf_result: WalkForwardResult,
        continuous: ContinuousBacktestResult | None,
    ) -> GateResult:
        """Build GateResult with all gate checks — shared by serial and parallel paths."""
        rs = self.settings.research
        min_ct = int(self.series.n_bars * rs.min_continuous_trades_per_1000_bars / 1000)

        shock_test: ShockTestResult | None = None
        regime_pf: RegimePFResult | None = None
        concentration: ConcentrationResult | None = None
        pair_robustness: PairRobustnessResult | None = None

        need_analysis = (
            rs.shock_test_enabled
            or rs.per_regime_pf_enabled
            or rs.concentration_check_enabled
            or rs.pair_robustness_enabled
        )
        if continuous is not None and continuous.trade_details and need_analysis:
            trades = continuous.trade_details
            analysis = _analyze_trades(
                trades,
                tp_shift_pct=rs.shock_test_tp_shift_pct,
                sl_shift_pct=rs.shock_test_sl_shift_pct,
                regime_min_trades=rs.per_regime_min_trades,
            )
            if rs.shock_test_enabled:
                shock_test = ShockTestResult(
                    profit_factor=scalar_profit_factor(analysis.shock_win, analysis.shock_loss)
                )
            if rs.concentration_check_enabled and continuous.total_pnl_usd > 0:
                max_month = (
                    max(continuous.monthly_pnl.values()) / continuous.total_pnl_usd * 100
                    if continuous.monthly_pnl
                    else 0.0
                )
                max_trade = analysis.max_trade_pnl / continuous.total_pnl_usd * 100
                concentration = ConcentrationResult(max_month_pct=max_month, max_trade_pct=max_trade)
            if rs.per_regime_pf_enabled:
                regime_pf = RegimePFResult(regime_pfs=analysis.regime_pfs)
            if rs.pair_robustness_enabled:
                above = sum(1 for pf_val in analysis.pair_pfs.values() if pf_val >= rs.pair_robustness_min_pf_threshold)
                pair_robustness = PairRobustnessResult(pairs_above_threshold=above, total_pairs=len(analysis.pair_pfs))

        gate_checks = GateCheckResults(
            continuous=continuous,
            shock_test=shock_test,
            regime_pf=regime_pf,
            concentration=concentration,
            pair_robustness=pair_robustness,
        )

        return evaluate_research_gate(
            cv_result,
            pbo,
            wf_result,
            thresholds=GateThresholds(
                min_cv_sharpe=rs.min_cv_sharpe,
                max_pbo=rs.max_pbo,
                wf_mode=rs.wf_mode,
                min_oos_degradation_ratio=rs.min_oos_degradation_ratio,
                min_wf_oos_sharpe=rs.min_wf_oos_sharpe,
                min_continuous_trades=min_ct,
                max_continuous_drawdown_pct=rs.max_continuous_drawdown_pct,
                shock_test_min_pf=rs.shock_test_min_pf,
                per_regime_min_pf=rs.per_regime_min_pf,
                concentration_max_month_pct=rs.concentration_max_month_pct,
                concentration_max_trade_pct=rs.concentration_max_trade_pct,
                pair_robustness_min_pairs=rs.pair_robustness_min_pairs,
            ),
            checks=gate_checks,
        )

    async def _run_continuous_check(self, service: BacktestService) -> ContinuousBacktestResult | None:
        """Run a full continuous backtest to detect state-dependent failures."""
        if not self.settings.research.continuous_validation_enabled:
            return None
        result = await service.run_with_hysteresis(self.series)
        summary = result.summary
        return ContinuousBacktestResult(
            trades=summary.trades,
            total_pnl_usd=summary.total_pnl_usd,
            max_drawdown_pct=summary.max_drawdown_pct,
            max_portfolio_drawdown_pct=summary.max_portfolio_drawdown_pct,
            trade_details=result.trade_details,
            monthly_pnl=result.monthly_pnl,
        )


# --- Single-pass trade analysis ---


def _regime_key(regime: MarketState | None) -> str:
    """Map MarketState to the 3-regime model key."""
    if regime is None:
        return "unknown"
    if regime in (MarketState.TRENDING_UP, MarketState.TRENDING_DOWN):
        return "trending"
    return regime.name.lower()


class _TradeAnalysis(NamedTuple):
    """Single-pass aggregates over all trades."""

    shock_win: float
    shock_loss: float
    max_trade_pnl: float
    regime_pfs: dict[str, float]
    pair_pfs: dict[str, float]


def _analyze_trades(
    trades: tuple[TradeDetail, ...],
    tp_shift_pct: float,
    sl_shift_pct: float,
    regime_min_trades: int,
) -> _TradeAnalysis:
    """Single pass over trades to compute all gate check inputs."""
    tp_mult = 1 + tp_shift_pct / 100
    sl_mult = 1 + sl_shift_pct / 100
    shock_win = 0.0
    shock_loss = 0.0
    max_trade_pnl = 0.0

    # Per-regime accumulators
    regime_wins: dict[str, float] = defaultdict(float)
    regime_losses: dict[str, float] = defaultdict(float)
    regime_counts: dict[str, int] = defaultdict(int)

    # Per-pair accumulators
    pair_wins: dict[str, float] = defaultdict(float)
    pair_losses: dict[str, float] = defaultdict(float)

    for t in trades:
        pnl = t.pnl_usd
        rk = _regime_key(t.regime)
        pair = t.pair

        # Shock test aggregation
        if pnl > 0:
            shock_win += pnl * tp_mult
            regime_wins[rk] += pnl
            pair_wins[pair] += pnl
        elif pnl < 0:
            shock_loss += abs(pnl) * sl_mult
            regime_losses[rk] += abs(pnl)
            pair_losses[pair] += abs(pnl)

        regime_counts[rk] += 1

        if pnl > max_trade_pnl:
            max_trade_pnl = pnl

    # Compute per-regime PFs (only groups with enough trades)
    regime_pfs = {
        key: scalar_profit_factor(regime_wins.get(key, 0.0), regime_losses.get(key, 0.0))
        for key, count in regime_counts.items()
        if count >= regime_min_trades
    }

    # Compute per-pair PFs (all pairs)
    all_pairs = set(pair_wins) | set(pair_losses)
    pair_pfs = {pair: scalar_profit_factor(pair_wins.get(pair, 0.0), pair_losses.get(pair, 0.0)) for pair in all_pairs}

    return _TradeAnalysis(
        shock_win=shock_win,
        shock_loss=shock_loss,
        max_trade_pnl=max_trade_pnl,
        regime_pfs=regime_pfs,
        pair_pfs=pair_pfs,
    )
