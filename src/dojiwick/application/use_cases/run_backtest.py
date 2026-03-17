"""Vectorized backtest and regime evaluation service.

Uses the same canonical pipeline as the live tick loop via
``run_decision_pipeline``: regime -> variant selection -> strategy -> veto
-> risk -> sizing -> (vectorized P&L instead of execution).  The sizing
output represents target-position intents -- the vectorized P&L kernel is
the offline-equivalent of the planner -> gateway execution path,
preserving full semantic parity.

The ``run_with_hysteresis`` method adds sequential replay: bars are fed
one-at-a-time through the shared pipeline with a shared
``RegimeHysteresis`` instance, so regime state accumulates identically
to live ticks.  P&L computation stays vectorized after collection.

Multi-bar position holds: trades remain open until stop or take-profit is
hit (checked against next bar's high/low), making ``stop_atr_mult`` and
``rr_ratio`` meaningful exit parameters.
"""

import logging
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from dojiwick.application.models.pipeline_settings import PipelineSettings
from dojiwick.application.orchestration.decision_pipeline import run_decision_pipeline, run_decision_pipeline_sync
from dojiwick.application.orchestration.regime_hysteresis import RegimeHysteresis
from dojiwick.application.policies.risk.engine import RiskPolicyEngine
from dojiwick.application.registry.strategy_registry import STRATEGY_MEAN_REVERT, StrategyRegistry
from dojiwick.compute.kernels.metrics.summarize import (
    compute_daily_sharpe,
    interval_to_bars_per_year,
    quick_sharpe,
    summarize,
)
from dojiwick.compute.kernels.pnl.entry_price import resolve_entry_price
from dojiwick.compute.kernels.pnl.partial_fill import apply_fill_ratio, compute_fill_ratio
from dojiwick.compute.kernels.pnl.pnl import scalar_net_pnl
from dojiwick.compute.kernels.pnl.portfolio_evolution import (
    compute_bar_net_pnl,
    evolve_portfolio,
)
from dojiwick.compute.kernels.regime.classify import classify_regime_batch
from dojiwick.compute.kernels.regime.evaluate import evaluate_regimes
from dojiwick.domain.models.value_objects.batch_models import BatchDecisionContext, BatchExecutionIntent
from dojiwick.domain.models.value_objects.cost_model import CostModel
from dojiwick.domain.enums import (
    CloseReason,
    EntryPriceModel,
    MarketState,
    RegimeExitProfile,
    TradeAction,
    safe_market_state,
)
from dojiwick.domain.indicator_schema import INDICATOR_INDEX
from dojiwick.domain.models.value_objects.outcome_models import (
    BacktestResult,
    BacktestSummary,
    RegimeEvaluationReport,
    TradeDetail,
)
from dojiwick.application.use_cases.optimization.pruning import PrunedError, PruningCallback
from dojiwick.domain.models.value_objects.params import RiskParams, StrategyParams

log = logging.getLogger(__name__)
_CHECKPOINT_FRACTIONS = (0.25, 0.35, 0.50, 0.70, 0.85)


@dataclass(slots=True, frozen=True, kw_only=True)
class PruningConfig:
    """Bundled pruning configuration for bar collection."""

    callback: PruningCallback
    min_trades: int
    score_fn: Callable[[BacktestSummary], float] | None = None


def _checkpoint_score(
    bar_pnls: np.ndarray,
    bar_notionals: np.ndarray,
    bar_idx: int,
    candle_interval: str,
    pruning: PruningConfig,
) -> float | None:
    """Compute a checkpoint score for pruning, or None if below min_trades."""
    partial_net = bar_pnls[: bar_idx + 1].ravel()
    partial_notional = bar_notionals[: bar_idx + 1].ravel()
    bpy = interval_to_bars_per_year(candle_interval)

    if pruning.score_fn is not None:
        partial = summarize(partial_net, partial_notional, compute_curves=False, n_bars=bar_idx + 1, bars_per_year=bpy)
        if partial.summary.trades < pruning.min_trades:
            return None
        return pruning.score_fn(partial.summary)

    n_trades = int(np.count_nonzero(partial_notional > 0.0))
    if n_trades < pruning.min_trades:
        return None
    return quick_sharpe(partial_net, partial_notional, n_bars=bar_idx + 1, bars_per_year=bpy)


def build_backtest_service(
    settings: PipelineSettings,
    *,
    config_hash: str = "",
    target_ids: tuple[str, ...],
    venue: str,
    product: str,
) -> "BacktestService":
    """Build a BacktestService from settings using default registry and risk engine."""
    from dojiwick.application.policies.risk.defaults import build_default_risk_engine
    from dojiwick.application.registry.strategy_registry import build_default_strategy_registry

    from dojiwick.domain.errors import ConfigurationError

    simulated = getattr(settings.backtest, "simulated_execution", False)
    if simulated:
        raise ConfigurationError("simulated_execution is not yet implemented — set to false or remove")
    if not target_ids:
        raise ConfigurationError("build_backtest_service requires non-empty target_ids")
    if not venue or not product:
        raise ConfigurationError("build_backtest_service requires non-empty venue and product")

    return BacktestService(
        settings=settings,
        strategy_registry=build_default_strategy_registry(enabled=settings.trading.enabled_strategies),
        risk_engine=build_default_risk_engine(settings.risk),
        config_hash=config_hash,
        target_ids=target_ids,
        venue=venue,
        product=product,
    )


@dataclass(slots=True)
class _OpenPosition:
    """Tracks an open multi-bar position for stop/TP exit logic."""

    entry_price: float
    entry_bar: int
    quantity: float
    action: TradeAction
    stop_price: float
    take_profit_price: float
    notional_usd: float
    strategy_name: str
    pair: str
    # Trailing stop state
    extreme_price: float = 0.0
    trailing_activation_price: float = 0.0  # 0.0 = no trailing
    trailing_distance: float = 0.0
    breakeven_price: float = 0.0  # 0.0 = no breakeven
    original_stop: float = 0.0
    # Time exit
    max_hold_bars: int = 0  # 0 = no limit
    # Diagnostics
    regime: MarketState | None = None
    regime_confidence: float = 0.0
    atr_at_entry: float = 0.0
    strategy_variant: str = ""
    # Partial take profit
    partial_tp_enabled: bool = False
    tp1_price: float = 0.0
    tp1_fraction: float = 0.5
    tp1_filled: bool = False
    entry_quantity: float = 0.0
    partial_tp_stop_ratio: float = 1.0


def _full_notional(pos: _OpenPosition) -> float:
    """Original notional USD, accounting for partial TP quantity reduction."""
    return pos.entry_quantity * pos.entry_price if pos.partial_tp_enabled else pos.notional_usd


@dataclass(slots=True, frozen=True)
class _BarCollectionResult:
    trade_details: list[TradeDetail]
    bar_pnls: np.ndarray  # shape (n_bars, n_pairs)
    bar_notionals: np.ndarray  # shape (n_bars, n_pairs)
    config_hash: str
    max_portfolio_drawdown_pct: float
    portfolio_equity: np.ndarray  # shape (n_bars,)
    initial_equity: float


@dataclass(slots=True, frozen=True, kw_only=True)
class BacktestTimeSeries:
    """Multi-bar time series for sequential replay with hysteresis.

    Each bar contains a ``BatchDecisionContext`` (N pairs) and a
    ``next_prices`` vector used for P&L computation after intent
    collection.  Optional OHLC vectors enable alternative entry price
    models (next_open, vwap_proxy, worst_case).
    """

    contexts: tuple[BatchDecisionContext, ...]
    next_prices: tuple[np.ndarray, ...]
    next_open: tuple[np.ndarray, ...] | None = None
    next_high: tuple[np.ndarray, ...] | None = None
    next_low: tuple[np.ndarray, ...] | None = None

    def __post_init__(self) -> None:
        if not self.contexts:
            raise ValueError("contexts must not be empty")
        if len(self.contexts) != len(self.next_prices):
            raise ValueError("contexts and next_prices length mismatch")
        n_pairs = self.contexts[0].size
        for i, ctx in enumerate(self.contexts):
            if ctx.size != n_pairs:
                raise ValueError(f"bar {i} has {ctx.size} pairs, expected {n_pairs}")
        for name, arr_tuple in (
            ("next_open", self.next_open),
            ("next_high", self.next_high),
            ("next_low", self.next_low),
        ):
            if arr_tuple is not None and len(arr_tuple) != len(self.contexts):
                raise ValueError(f"{name} length mismatch")

    @property
    def n_bars(self) -> int:
        return len(self.contexts)

    @property
    def n_pairs(self) -> int:
        return self.contexts[0].size

    def slice_by_indices(self, indices: Sequence[int]) -> "BacktestTimeSeries":
        """Return a sub-series containing only the bars at *indices*."""
        return BacktestTimeSeries(
            contexts=tuple(self.contexts[i] for i in indices),
            next_prices=tuple(self.next_prices[i] for i in indices),
            next_open=tuple(self.next_open[i] for i in indices) if self.next_open is not None else None,
            next_high=tuple(self.next_high[i] for i in indices) if self.next_high is not None else None,
            next_low=tuple(self.next_low[i] for i in indices) if self.next_low is not None else None,
        )


@dataclass(slots=True, frozen=True, kw_only=True)
class BacktestService:
    """Runs offline simulation with live-equivalent kernels.

    Uses the shared ``run_decision_pipeline`` to ensure decision-level
    parity with the live tick loop, followed by vectorized P&L
    computation as the offline-equivalent of the planner -> gateway
    execution path.
    """

    settings: PipelineSettings
    strategy_registry: StrategyRegistry
    risk_engine: RiskPolicyEngine
    config_hash: str = ""
    target_ids: tuple[str, ...] = ()
    venue: str = ""
    product: str = ""

    async def run(self, context: BatchDecisionContext, next_prices: np.ndarray) -> BacktestSummary:
        """Run deterministic backtest for aligned sample rows.

        Pipeline: shared decision pipeline (regime -> variants -> strategy -> risk -> sizing)
        -> vectorized P&L (planner-equivalent).
        """

        # 1-5. Shared decision pipeline (no hysteresis, no veto, no adaptive)
        pipeline = await run_decision_pipeline(
            context=context,
            settings=self.settings,
            strategy_registry=self.strategy_registry,
            risk_engine=self.risk_engine,
        )

        # 6. Vectorized P&L (offline equivalent of planner -> execution)
        net = compute_bar_net_pnl(pipeline.intents, next_prices, self.settings.backtest.cost_model)
        result = summarize(
            net,
            pipeline.intents.notional_usd,
            bars_per_year=interval_to_bars_per_year(self.settings.trading.candle_interval),
        )
        return replace(result.summary, config_hash=self.config_hash)

    async def run_with_hysteresis(
        self,
        series: BacktestTimeSeries,
        *,
        hysteresis_bars: int | None = None,
    ) -> BacktestResult:
        """Run sequential replay with shared regime hysteresis.

        Feeds bars one-at-a-time through ``run_decision_pipeline`` so regime
        state accumulates identically to the live tick loop.  Positions are
        held across multiple bars until stop or take-profit is hit.

        Parameters
        ----------
        series:
            Multi-bar time series (T bars x N pairs).
        hysteresis_bars:
            Override ``settings.regime.hysteresis_bars`` for this run.
            ``None`` uses the value from ``self.settings``.
        """
        collected = await self._collect_bars(series, hysteresis_bars)
        net = collected.bar_pnls.ravel()
        flat_notional = collected.bar_notionals.ravel()
        bpy = interval_to_bars_per_year(self.settings.trading.candle_interval)
        result = summarize(net, flat_notional, compute_curves=True, n_bars=series.n_bars, bars_per_year=bpy)
        summary = self._apply_benchmark(result.summary, series, collected.config_hash)
        # Portfolio equity curve + drawdown curve
        peq = collected.portfolio_equity
        peaks = np.maximum.accumulate(peq)
        pdd = np.where(peaks > 0, (peaks - peq) / peaks * 100.0, 0.0)

        # Daily Sharpe (industry-standard)
        bar_totals = collected.bar_pnls.sum(axis=1)
        bars_per_day = int(bpy / 365)
        daily_sharpe = compute_daily_sharpe(bar_totals, peq, collected.initial_equity, bars_per_day)

        summary = replace(
            summary,
            max_portfolio_drawdown_pct=collected.max_portfolio_drawdown_pct,
            portfolio_equity_curve=peq,
            portfolio_drawdown_curve=pdd,
            daily_sharpe=daily_sharpe,
        )

        # Monthly PnL breakdown (bar_totals already computed above)
        monthly: dict[str, float] = {}
        for ctx, pnl in zip(series.contexts, bar_totals.tolist()):
            key = ctx.market.observed_at.strftime("%Y-%m")
            monthly[key] = monthly.get(key, 0.0) + pnl

        return BacktestResult(
            summary=summary,
            trade_details=tuple(collected.trade_details),
            monthly_pnl=monthly,
        )

    async def run_with_hysteresis_summary_only(
        self,
        series: BacktestTimeSeries,
        *,
        hysteresis_bars: int | None = None,
        skip_benchmark: bool = False,
        pruning: PruningConfig | None = None,
    ) -> tuple[BacktestSummary, np.ndarray]:
        """Return summary + trade returns (for validators). Skips curves and trade details."""
        collected = await self._collect_bars(
            series,
            hysteresis_bars,
            collect_details=False,
            collect_equity=False,
            pruning=pruning,
        )
        net = collected.bar_pnls.ravel()
        flat_notional = collected.bar_notionals.ravel()
        bpy = interval_to_bars_per_year(self.settings.trading.candle_interval)
        result = summarize(net, flat_notional, compute_curves=False, n_bars=series.n_bars, bars_per_year=bpy)
        if skip_benchmark:
            summary = result.summary
        else:
            summary = self._apply_benchmark(result.summary, series, collected.config_hash)
        summary = replace(summary, max_portfolio_drawdown_pct=collected.max_portfolio_drawdown_pct)
        return summary, result.trade_returns

    async def _collect_bars(
        self,
        series: BacktestTimeSeries,
        hysteresis_bars: int | None,
        collect_details: bool = True,
        collect_equity: bool = True,
        pruning: PruningConfig | None = None,
    ) -> _BarCollectionResult:
        """Run the bar-collection loop with multi-bar position holds.

        Positions remain open until stop or take-profit is hit (checked
        against next bar's high/low). Pairs with open positions are not
        eligible for new entries.
        """
        settings = self.settings

        # Pruning checkpoint setup: map bar_idx -> step for O(1) lookup
        # Skip pruning on tiny series where checkpoint fractions collide
        checkpoint_map: dict[int, int] = {}
        if pruning is not None and series.n_bars >= 20:
            checkpoint_map = {int(series.n_bars * f): step for step, f in enumerate(_CHECKPOINT_FRACTIONS)}

        hyst = RegimeHysteresis()
        bar_pnls = np.zeros((series.n_bars, series.n_pairs), dtype=np.float64)
        bar_notionals = np.zeros((series.n_bars, series.n_pairs), dtype=np.float64)
        trade_details: list[TradeDetail] = []
        cost = settings.backtest.cost_model

        portfolio = series.contexts[0].portfolio
        prev_time = None
        n_pairs = series.n_pairs

        entry_model = settings.backtest.entry_price_model
        use_alt_entry = entry_model != EntryPriceModel.CLOSE
        use_partial = settings.backtest.partial_fill_enabled

        has_ohlc = series.next_high is not None and series.next_low is not None
        positions: dict[int, _OpenPosition] = {}

        # Scope resolution caches -- persisted across bars
        strategy_scope_cache: dict[tuple[str, int | None], StrategyParams] = {}
        risk_scope_cache: dict[tuple[str, int | None], RiskParams] = {}
        phase2_scope_cache: dict[tuple[str, int | None, str | None], StrategyParams] = {}
        has_strategy_rules = settings.strategy_scope.has_strategy_rules

        bar_has_open = np.zeros(n_pairs, dtype=np.bool_)
        bar_open_count = np.zeros(n_pairs, dtype=np.int64)
        bar_pnl_buf = np.zeros(n_pairs)
        bar_notional_buf = np.zeros(n_pairs)

        # Cache frequently accessed risk settings (avoid attribute chain per bar)
        _ecf_enabled = settings.risk.equity_curve_filter_enabled
        _dd_scale_enabled = settings.risk.drawdown_risk_scale_enabled
        _dd_scale_max = settings.risk.drawdown_risk_scale_max_dd
        _dd_scale_floor = settings.risk.drawdown_risk_scale_floor
        _max_loss_frac = settings.risk.max_loss_per_trade_pct / 100.0
        _baseline_pairs = settings.risk.portfolio_risk_baseline_pairs
        _pair_scale = max(1.0, n_pairs / _baseline_pairs)
        _max_portfolio_frac = settings.risk.max_portfolio_risk_pct / 100.0 * _pair_scale

        # Drawdown protection state + portfolio equity tracking
        initial_equity: float = float(np.mean(portfolio.equity_usd))
        peak_equity: float = initial_equity
        max_portfolio_dd: float = 0.0
        portfolio_equity = np.zeros(series.n_bars, dtype=np.float64) if collect_equity else np.empty(0)
        ecf_period = settings.risk.equity_curve_filter_period
        equity_window: deque[float] = deque(maxlen=ecf_period)
        equity_running_sum: float = 0.0
        ecf_scale: float = 1.0

        # Consecutive loss tracking (per-pair list -- scalar access only)
        consecutive_losses: list[int] = [0] * n_pairs

        _any_drawdown_active = _dd_scale_enabled or _ecf_enabled or settings.risk.drawdown_halt_pct < 100.0

        pending_entries: list[tuple[int, _OpenPosition, float]] = []
        pending_risk_total: float = 0.0

        for bar_idx in range(series.n_bars):
            # Fast context construction -- bypass __post_init__ validation.
            # Data is already validated at series construction time.
            _port = replace(portfolio, has_open_position=bar_has_open, open_positions_total=bar_open_count)
            ctx = replace(series.contexts[bar_idx], portfolio=_port)

            # Skip pipeline when all positions open
            intents = None
            bar_regimes: np.ndarray | None = None
            bar_regime_conf: np.ndarray | None = None
            per_pair_strategy_params: tuple[StrategyParams, ...] | None = None
            if len(positions) < n_pairs:
                pipeline = run_decision_pipeline_sync(
                    context=ctx,
                    settings=settings,
                    strategy_registry=self.strategy_registry,
                    risk_engine=self.risk_engine,
                    hysteresis=hyst,
                    strategy_scope_cache=strategy_scope_cache,
                    risk_scope_cache=risk_scope_cache,
                    has_strategy_rules=has_strategy_rules,
                    phase2_scope_cache=phase2_scope_cache,
                    hysteresis_bars_override=hysteresis_bars,
                )
                n = ctx.size
                if pipeline.intents.action.shape[0] != n:
                    raise RuntimeError(
                        f"bar {bar_idx}: intent size {pipeline.intents.action.shape[0]} != context size {n}"
                    )

                intents = pipeline.intents
                bar_regimes = pipeline.regimes.coarse_state
                bar_regime_conf = pipeline.regimes.confidence

                # Entry price resolution + partial fill simulation
                entry_price = intents.entry_price
                quantity = intents.quantity
                notional_usd = intents.notional_usd
                modified = False

                if (
                    use_alt_entry
                    and series.next_open is not None
                    and series.next_high is not None
                    and series.next_low is not None
                ):
                    entry_price = resolve_entry_price(
                        entry_model,
                        close=entry_price,
                        next_open=series.next_open[bar_idx],
                        next_high=series.next_high[bar_idx],
                        next_low=series.next_low[bar_idx],
                        next_close=series.next_prices[bar_idx],
                        action=intents.action,
                    )
                    modified = True

                if use_partial and ctx.market.volume is not None:
                    fill_ratio = compute_fill_ratio(
                        notional_usd=notional_usd,
                        bar_volume=ctx.market.volume,
                        entry_price=entry_price,
                        action=intents.action,
                        threshold_pct=settings.backtest.partial_fill_threshold_pct,
                        min_ratio=settings.backtest.partial_fill_min_ratio,
                    )
                    quantity, notional_usd = apply_fill_ratio(
                        quantity=quantity,
                        notional_usd=notional_usd,
                        fill_ratio=fill_ratio,
                    )
                    modified = True

                if modified:
                    intents = replace(intents, entry_price=entry_price, quantity=quantity, notional_usd=notional_usd)

                per_pair_strategy_params = pipeline.per_pair_params

            bar_pnl_buf.fill(0.0)
            bar_notional_buf.fill(0.0)

            # Pre-compute bar-level OHLC
            if has_ohlc:
                assert series.next_high is not None and series.next_low is not None
                bar_highs = series.next_high[bar_idx]
                bar_lows = series.next_low[bar_idx]
            else:
                bar_highs = bar_lows = series.next_prices[bar_idx]

            avg_equity = float(np.mean(portfolio.equity_usd))
            # Always track portfolio drawdown (used by objective function)
            peak_equity = max(peak_equity, avg_equity)
            drawdown_pct = (peak_equity - avg_equity) / peak_equity * 100.0 if peak_equity > 0 else 0.0
            max_portfolio_dd = max(max_portfolio_dd, drawdown_pct)

            if _any_drawdown_active:
                # Maintain equity window with O(1) running sum
                if _ecf_enabled:
                    if len(equity_window) == ecf_period:
                        equity_running_sum -= equity_window[0]
                    equity_window.append(avg_equity)
                    equity_running_sum += avg_equity

                # Equity curve filter -- proportional scaling (never blocks)
                ecf_scale = 1.0
                if _ecf_enabled and len(equity_window) == ecf_period:
                    sma = equity_running_sum / ecf_period
                    if sma > 0 and avg_equity < sma:
                        ecf_scale = max(avg_equity / sma, _dd_scale_floor)

            pending_entries.clear()
            pending_risk_total = 0.0
            for pair_idx in range(n_pairs):
                if pair_idx in positions:
                    # Check exit for existing position
                    pos = positions[pair_idx]
                    next_h = float(bar_highs[pair_idx])
                    next_l = float(bar_lows[pair_idx])

                    # Update trailing stop before checking exit
                    _update_trailing_stop(pos, next_h, next_l)

                    # Check partial TP1 before full exit
                    if pos.partial_tp_enabled and not pos.tp1_filled and pos.tp1_price > 0:
                        is_long = pos.action == TradeAction.BUY
                        favorable = next_h if is_long else next_l
                        tp1_hit = (favorable >= pos.tp1_price) if is_long else (favorable <= pos.tp1_price)
                        if tp1_hit:
                            tp1_qty = pos.entry_quantity * pos.tp1_fraction
                            hold_bars_tp1 = bar_idx - pos.entry_bar
                            tp1_pnl = _realized_pnl(pos, pos.tp1_price, cost, hold_bars_tp1, quantity=tp1_qty)
                            bar_pnl_buf[pair_idx] += tp1_pnl
                            bar_notional_buf[pair_idx] = _full_notional(pos)
                            if collect_details:
                                trade_details.append(
                                    TradeDetail(
                                        bar_index=pos.entry_bar,
                                        exit_bar_index=bar_idx,
                                        hold_bars=bar_idx - pos.entry_bar,
                                        close_reason=CloseReason.PARTIAL_TP,
                                        pair=pos.pair,
                                        strategy_name=pos.strategy_name,
                                        action=pos.action,
                                        entry_price=pos.entry_price,
                                        exit_price=pos.tp1_price,
                                        quantity=tp1_qty,
                                        notional_usd=tp1_qty * pos.entry_price,
                                        pnl_usd=tp1_pnl,
                                        regime=pos.regime,
                                        regime_confidence=pos.regime_confidence,
                                        atr_at_entry=pos.atr_at_entry,
                                        stop_price=pos.original_stop,
                                        take_profit_price=pos.take_profit_price,
                                        strategy_variant=pos.strategy_variant,
                                    )
                                )
                            pos.tp1_filled = True
                            remaining_qty = pos.entry_quantity - tp1_qty
                            pos.quantity = remaining_qty
                            pos.notional_usd = remaining_qty * pos.entry_price
                            pos.stop_price = (
                                pos.original_stop + (pos.entry_price - pos.original_stop) * pos.partial_tp_stop_ratio
                            )
                            # Partial TP counts as a win
                            if tp1_pnl > 0:
                                consecutive_losses[pair_idx] = 0
                            continue  # Defer exit check to next bar -- protective stop at entry activates next bar

                    result = _check_exit(pos, next_h, next_l, bar_idx)
                    if result is not None:
                        exit_price, reason = result
                        hold_bars = bar_idx - pos.entry_bar
                        pnl = _realized_pnl(pos, exit_price, cost, hold_bars)
                        bar_pnl_buf[pair_idx] += pnl
                        bar_notional_buf[pair_idx] = _full_notional(pos)
                        if collect_details:
                            trade_details.append(_build_trade_detail(pos, bar_idx, exit_price, reason, pnl))
                        # Track consecutive losses
                        if pnl < 0:
                            consecutive_losses[pair_idx] += 1
                        else:
                            consecutive_losses[pair_idx] = 0
                        del positions[pair_idx]
                        bar_has_open[pair_idx] = False
                        bar_open_count[pair_idx] = 0
                    # else: position stays open, no PnL this bar
                elif intents is not None:
                    # No open position -- collect candidate entry (pass 1 of fair allocation)
                    pair_params: StrategyParams = (
                        per_pair_strategy_params[pair_idx]
                        if per_pair_strategy_params is not None
                        else settings.strategy
                    )

                    # Block entries after consecutive losses (respects risk scope)
                    pair_consec = consecutive_losses[pair_idx]
                    regime_key = int(bar_regimes[pair_idx]) if bar_regimes is not None else None
                    pair_name = ctx.market.pairs[pair_idx]
                    pair_risk = risk_scope_cache.get((pair_name, regime_key))
                    max_consec = (
                        pair_risk.max_consecutive_losses
                        if pair_risk is not None
                        else settings.risk.max_consecutive_losses
                    )
                    if pair_consec >= max_consec:
                        continue

                    regime_state: MarketState | None = None
                    regime_conf = 0.0
                    if bar_regimes is not None and bar_regime_conf is not None:
                        regime_state = safe_market_state(int(bar_regimes[pair_idx]))
                        regime_conf = float(bar_regime_conf[pair_idx])
                    new_pos = _open_position(
                        intents,
                        pair_idx,
                        bar_idx,
                        ctx,
                        pair_params,
                        regime=regime_state,
                        regime_confidence=regime_conf,
                    )
                    if new_pos is not None:
                        # Progressive risk scaling: drawdown + ECF combined (never blocks)
                        dd_scale = 1.0
                        if _dd_scale_enabled and drawdown_pct > 0:
                            max_dd = _dd_scale_max
                            raw = max(1.0 - drawdown_pct / max_dd, 0.0)
                            dd_scale = max(raw**0.5, _dd_scale_floor)

                        combined_scale = min(dd_scale, ecf_scale)
                        if combined_scale < 1.0:
                            new_pos.quantity *= combined_scale
                            new_pos.notional_usd *= combined_scale
                            if new_pos.partial_tp_enabled:
                                new_pos.entry_quantity *= combined_scale

                        # Cap leveraged risk per trade and collect for fair allocation
                        if new_pos.entry_price > 0:
                            stop_dist_pct = abs(new_pos.entry_price - new_pos.stop_price) / new_pos.entry_price
                            leveraged_risk = new_pos.notional_usd * stop_dist_pct * cost.leverage
                            max_loss_usd = avg_equity * _max_loss_frac
                            if leveraged_risk > max_loss_usd and leveraged_risk > 0:
                                cap_scale = max_loss_usd / leveraged_risk
                                new_pos.quantity *= cap_scale
                                new_pos.notional_usd *= cap_scale
                                if new_pos.partial_tp_enabled:
                                    new_pos.entry_quantity *= cap_scale
                                leveraged_risk = max_loss_usd

                            pending_entries.append((pair_idx, new_pos, leveraged_risk))
                            pending_risk_total += leveraged_risk

            # Pass 2: Fair portfolio-level risk allocation across ALL pending entries
            if pending_entries:
                existing_risk_usd = 0.0
                for existing_pos in positions.values():
                    pos_stop_dist = abs(existing_pos.entry_price - existing_pos.stop_price) / existing_pos.entry_price
                    existing_risk_usd += existing_pos.notional_usd * pos_stop_dist * cost.leverage

                max_portfolio_risk_usd = avg_equity * _max_portfolio_frac
                available = max_portfolio_risk_usd - existing_risk_usd

                if available > 0:
                    if pending_risk_total > available:
                        port_scale = available / pending_risk_total
                        for _, pe_pos, _ in pending_entries:
                            pe_pos.quantity *= port_scale
                            pe_pos.notional_usd *= port_scale
                            if pe_pos.partial_tp_enabled:
                                pe_pos.entry_quantity *= port_scale

                    for pe_idx, pe_pos, _ in pending_entries:
                        positions[pe_idx] = pe_pos
                        bar_has_open[pe_idx] = True
                        bar_open_count[pe_idx] = 1

            bar_pnls[bar_idx] = bar_pnl_buf
            bar_notionals[bar_idx] = bar_notional_buf
            current_time = ctx.market.observed_at
            portfolio = evolve_portfolio(
                portfolio,
                bar_pnl_buf,
                current_time,
                prev_time,
                has_open_position=bar_has_open,
                open_positions_total=bar_open_count,
            )
            prev_time = current_time
            if collect_equity:
                portfolio_equity[bar_idx] = float(np.mean(portfolio.equity_usd))

            if bar_idx > 0 and bar_idx % 100 == 0 and log.isEnabledFor(logging.INFO):
                log.info(
                    "bar %d/%d | open positions: %d | avg equity: $%.2f",
                    bar_idx,
                    series.n_bars,
                    len(positions),
                    float(np.mean(portfolio.equity_usd)),
                )

            # Pruning checkpoints: report intermediate value at 50%/70%/85%
            if checkpoint_map and pruning is not None:
                step = checkpoint_map.get(bar_idx)
                if step is not None:
                    score = _checkpoint_score(
                        bar_pnls, bar_notionals, bar_idx, settings.trading.candle_interval, pruning
                    )
                    if score is not None:
                        pruning.callback.report(score, step)
                        if pruning.callback.should_prune():
                            raise PrunedError(f"trial pruned at bar {bar_idx}/{series.n_bars} (step {step})")

        # Force-close remaining positions at last bar's next_prices
        if positions:
            last_bar = series.n_bars - 1
            for pair_idx, pos in positions.items():
                exit_price = float(series.next_prices[last_bar][pair_idx])
                hold_bars = last_bar - pos.entry_bar  # exclusive semantics (Fix 5)
                pnl = _realized_pnl(pos, exit_price, cost, hold_bars)
                if collect_details:
                    trade_details.append(
                        _build_trade_detail(pos, last_bar, exit_price, CloseReason.END_OF_BACKTEST, pnl)
                    )
                # Add to last bar's PnL
                bar_pnls[-1][pair_idx] += pnl
                bar_notionals[-1][pair_idx] += _full_notional(pos)

        return _BarCollectionResult(
            trade_details=trade_details,
            bar_pnls=bar_pnls,
            bar_notionals=bar_notionals,
            config_hash=self.config_hash,
            max_portfolio_drawdown_pct=max_portfolio_dd,
            portfolio_equity=portfolio_equity,
            initial_equity=initial_equity,
        )

    @staticmethod
    def _apply_benchmark(
        summary: BacktestSummary,
        series: BacktestTimeSeries,
        config_hash: str,
    ) -> BacktestSummary:
        initial_prices = series.contexts[0].market.price
        final_prices = series.next_prices[-1]
        benchmark_returns = (final_prices / initial_prices) - 1.0
        equity_usd = float(series.contexts[0].portfolio.equity_usd[0])
        benchmark_pnl = float(np.mean(benchmark_returns) * equity_usd)
        return replace(
            summary,
            config_hash=config_hash,
            benchmark_pnl_usd=benchmark_pnl,
        )

    def evaluate_regimes(
        self,
        *,
        prices: np.ndarray,
        context: BatchDecisionContext,
        horizons: tuple[int, ...] = (1, 3, 6),
    ) -> RegimeEvaluationReport:
        """Evaluate classifier state quality against deterministic truth labels."""

        predicted = classify_regime_batch(context.market, self.settings.regime.params)
        return evaluate_regimes(
            prices=prices,
            predicted_states=predicted.coarse_state,
            settings=self.settings.regime.params,
            horizons=horizons,
        )


def _update_trailing_stop(pos: _OpenPosition, bar_high: float, bar_low: float) -> None:
    """Update extreme price and trailing stop in-place."""
    if pos.trailing_activation_price == 0.0 and pos.breakeven_price == 0.0:
        return
    is_long = pos.action == TradeAction.BUY
    new_extreme = max(pos.extreme_price, bar_high) if is_long else min(pos.extreme_price, bar_low)
    new_stop = pos.stop_price

    # Breakeven: once price exceeds breakeven threshold, move stop to entry
    be_hit = (new_extreme >= pos.breakeven_price) if is_long else (new_extreme <= pos.breakeven_price)
    stop_not_at_entry = (pos.stop_price < pos.entry_price) if is_long else (pos.stop_price > pos.entry_price)
    if pos.breakeven_price > 0.0 and be_hit and stop_not_at_entry:
        new_stop = pos.entry_price

    # Trailing: once price exceeds activation, trail stop behind extreme
    act_hit = (
        (new_extreme >= pos.trailing_activation_price) if is_long else (new_extreme <= pos.trailing_activation_price)
    )
    if pos.trailing_activation_price > 0.0 and act_hit:
        trail_stop = (new_extreme - pos.trailing_distance) if is_long else (new_extreme + pos.trailing_distance)
        new_stop = max(new_stop, trail_stop) if is_long else min(new_stop, trail_stop)

    pos.extreme_price = new_extreme
    pos.stop_price = new_stop


def _check_exit(
    pos: _OpenPosition,
    bar_high: float,
    bar_low: float,
    current_bar: int,
) -> tuple[float, CloseReason] | None:
    """Return (exit_price, reason) if stop/TP/time hit, else None. Time exit checked first."""
    # Time exit -- checked before stop/TP
    if pos.max_hold_bars > 0 and (current_bar - pos.entry_bar) >= pos.max_hold_bars:
        exit_price = (bar_high + bar_low) / 2.0  # midpoint approximation
        return exit_price, CloseReason.TIME_EXIT

    is_long = pos.action == TradeAction.BUY
    adverse = bar_low if is_long else bar_high
    favorable = bar_high if is_long else bar_low
    stop_trailed = (pos.stop_price > pos.original_stop) if is_long else (pos.stop_price < pos.original_stop)

    if (adverse <= pos.stop_price) if is_long else (adverse >= pos.stop_price):
        reason = (
            CloseReason.TRAILING_STOP
            if pos.trailing_activation_price != 0.0 and stop_trailed
            else CloseReason.STOP_LOSS
        )
        return pos.stop_price, reason
    if (favorable >= pos.take_profit_price) if is_long else (favorable <= pos.take_profit_price):
        return pos.take_profit_price, CloseReason.TAKE_PROFIT
    return None


def _build_trade_detail(
    pos: _OpenPosition,
    exit_bar: int,
    exit_price: float,
    close_reason: CloseReason,
    pnl: float,
) -> TradeDetail:
    """Build a TradeDetail with consistent hold_bars = exit_bar - entry_bar."""
    return TradeDetail(
        bar_index=pos.entry_bar,
        exit_bar_index=exit_bar,
        hold_bars=exit_bar - pos.entry_bar,
        close_reason=close_reason,
        pair=pos.pair,
        strategy_name=pos.strategy_name,
        action=pos.action,
        entry_price=pos.entry_price,
        exit_price=exit_price,
        quantity=pos.quantity,
        notional_usd=pos.notional_usd,
        pnl_usd=pnl,
        regime=pos.regime,
        regime_confidence=pos.regime_confidence,
        atr_at_entry=pos.atr_at_entry,
        stop_price=pos.original_stop,
        take_profit_price=pos.take_profit_price,
        strategy_variant=pos.strategy_variant,
    )


@dataclass(slots=True)
class _ExitProfile:
    """Mutable exit configuration built during position opening."""

    stop: float
    take_profit: float
    trailing_activation_price: float
    trailing_distance: float
    breakeven_price: float
    max_hold_bars: int


@dataclass(slots=True, frozen=True)
class _ExitContext:
    """Immutable context passed to exit modifiers."""

    entry: float
    stop_distance: float
    direction: int
    bb_mid: float
    params: StrategyParams


def _apply_mean_revert_overrides(profile: _ExitProfile, ctx: _ExitContext) -> None:
    """Override TP to BB midline and optionally disable breakeven."""
    if ctx.params.mean_revert_use_bb_mid_tp and ctx.bb_mid > 0:
        profile.take_profit = ctx.bb_mid
    if ctx.params.mean_revert_disable_breakeven:
        profile.breakeven_price = 0.0


def _apply_volatile_exits(profile: _ExitProfile, ctx: _ExitContext) -> None:
    """Tighten stops, adjust RR, cap hold time for volatile regimes."""
    stop_distance_new = ctx.stop_distance * ctx.params.adaptive_volatile_stop_scale
    if ctx.direction > 0:
        profile.stop = ctx.entry - stop_distance_new
        profile.take_profit = ctx.entry + stop_distance_new * ctx.params.adaptive_volatile_rr_mult
    else:
        profile.stop = ctx.entry + stop_distance_new
        profile.take_profit = ctx.entry - stop_distance_new * ctx.params.adaptive_volatile_rr_mult
    v_max = ctx.params.adaptive_volatile_max_bars
    profile.max_hold_bars = min(profile.max_hold_bars, v_max) if profile.max_hold_bars > 0 else v_max


def _apply_trending_exits(profile: _ExitProfile, ctx: _ExitContext) -> None:
    """Widen trailing distance for trending regimes."""
    if profile.trailing_distance > 0:
        profile.trailing_distance *= ctx.params.adaptive_trending_trail_scale


def _apply_ranging_exits(profile: _ExitProfile, ctx: _ExitContext) -> None:
    """Target BB midline, disable breakeven, cap hold time for ranging."""
    if ctx.bb_mid > 0:
        profile.take_profit = ctx.bb_mid
    profile.breakeven_price = 0.0
    r_max = ctx.params.adaptive_ranging_max_bars
    profile.max_hold_bars = min(profile.max_hold_bars, r_max) if profile.max_hold_bars > 0 else r_max


_REGIME_EXIT_MODIFIERS: dict[MarketState, Callable[[_ExitProfile, _ExitContext], None]] = {
    MarketState.VOLATILE: _apply_volatile_exits,
    MarketState.TRENDING_UP: _apply_trending_exits,
    MarketState.TRENDING_DOWN: _apply_trending_exits,
    MarketState.RANGING: _apply_ranging_exits,
}


def _open_position(
    intents: BatchExecutionIntent,
    pair_idx: int,
    bar_idx: int,
    ctx: BatchDecisionContext,
    pair_params: StrategyParams,
    *,
    regime: MarketState | None = None,
    regime_confidence: float = 0.0,
) -> _OpenPosition | None:
    """Create an open position from entry intents, or None if no valid entry."""
    act = TradeAction(int(intents.action[pair_idx]))
    if act == TradeAction.HOLD:
        return None
    entry = float(intents.entry_price[pair_idx])
    stop = float(intents.stop_price[pair_idx])
    tp = float(intents.take_profit_price[pair_idx])
    if stop <= 0.0 or tp <= 0.0:
        return None

    stop_distance = abs(entry - stop)
    atr_at_entry = float(ctx.market.indicators[pair_idx, INDICATOR_INDEX["atr"]])
    direction = 1 if act == TradeAction.BUY else -1

    trailing_activation_price = 0.0
    trailing_distance = 0.0
    if pair_params.trailing_stop_activation_rr is not None and pair_params.trailing_stop_atr_mult is not None:
        trailing_activation_price = entry + direction * stop_distance * pair_params.trailing_stop_activation_rr
        trailing_distance = atr_at_entry * pair_params.trailing_stop_atr_mult

    breakeven_price = 0.0
    if pair_params.breakeven_after_rr is not None:
        breakeven_price = entry + direction * stop_distance * pair_params.breakeven_after_rr

    max_hold_bars = pair_params.max_hold_bars if pair_params.max_hold_bars is not None else 0
    strategy_name = intents.strategy_name[pair_idx]
    bb_mid_val = float(ctx.market.indicators[pair_idx, INDICATOR_INDEX["bb_mid"]])

    profile = _ExitProfile(
        stop=stop,
        take_profit=tp,
        trailing_activation_price=trailing_activation_price,
        trailing_distance=trailing_distance,
        breakeven_price=breakeven_price,
        max_hold_bars=max_hold_bars,
    )
    exit_ctx = _ExitContext(
        entry=entry,
        stop_distance=stop_distance,
        direction=direction,
        bb_mid=bb_mid_val,
        params=pair_params,
    )

    if strategy_name == STRATEGY_MEAN_REVERT:
        _apply_mean_revert_overrides(profile, exit_ctx)

    if pair_params.regime_exit_profile == RegimeExitProfile.ADAPTIVE and regime is not None:
        modifier = _REGIME_EXIT_MODIFIERS.get(regime)
        if modifier is not None:
            modifier(profile, exit_ctx)

    # Partial take profit setup (uses original stop_distance, not post-modifier)
    quantity = float(intents.quantity[pair_idx])
    partial_tp_enabled = pair_params.partial_tp_enabled
    tp1_price = 0.0
    if partial_tp_enabled and pair_params.partial_tp1_rr > 0:
        tp1_price = entry + direction * stop_distance * pair_params.partial_tp1_rr

    return _OpenPosition(
        entry_price=entry,
        entry_bar=bar_idx,
        quantity=quantity,
        action=act,
        stop_price=profile.stop,
        take_profit_price=profile.take_profit,
        notional_usd=float(intents.notional_usd[pair_idx]),
        strategy_name=strategy_name,
        pair=intents.pairs[pair_idx],
        extreme_price=entry,
        trailing_activation_price=profile.trailing_activation_price,
        trailing_distance=profile.trailing_distance,
        breakeven_price=profile.breakeven_price,
        original_stop=profile.stop,
        max_hold_bars=profile.max_hold_bars,
        regime=regime,
        regime_confidence=regime_confidence,
        atr_at_entry=atr_at_entry,
        strategy_variant=intents.strategy_variant[pair_idx],
        partial_tp_enabled=partial_tp_enabled,
        tp1_price=tp1_price,
        tp1_fraction=pair_params.partial_tp1_fraction,
        tp1_filled=False,
        entry_quantity=quantity,
        partial_tp_stop_ratio=pair_params.partial_tp_stop_ratio,
    )


def _realized_pnl(
    pos: _OpenPosition,
    exit_price: float,
    cost: CostModel,
    hold_bars: int,
    *,
    quantity: float | None = None,
) -> float:
    """Compute net PnL for a closed position -- delegates to scalar_net_pnl.

    When *quantity* is given, computes PnL for that tranche instead of the
    full position (used for partial take-profit).
    """
    qty = quantity if quantity is not None else pos.quantity
    notional = (qty * pos.entry_price) if quantity is not None else pos.notional_usd
    return scalar_net_pnl(
        is_long=pos.action == TradeAction.BUY,
        entry_price=pos.entry_price,
        exit_price=exit_price,
        quantity=qty,
        notional=notional,
        slippage_bps=cost.slippage_bps,
        fee_bps=cost.fee_bps,
        fee_multiplier=cost.fee_multiplier,
        leverage=cost.leverage,
        funding_rate_per_bar=cost.funding_rate_per_bar,
        hold_bars=hold_bars,
    )
