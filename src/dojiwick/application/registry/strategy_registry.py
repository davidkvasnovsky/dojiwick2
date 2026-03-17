"""Strategy plugin registry -- registration-driven signal composition."""

import logging

import numpy as np

from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchRegimeProfile,
    BatchSignalFragment,
    BatchTradeCandidate,
)
from dojiwick.domain.models.value_objects.params import StrategyParams, extract_param_vector
from dojiwick.domain.enums import TradeAction
from dojiwick.domain.indicator_schema import INDICATOR_INDEX
from dojiwick.domain.reason_codes import STRATEGY_HOLD, STRATEGY_SIGNAL

from dojiwick.compute.kernels.strategy.confluence import compute_confluence_score
from dojiwick.compute.kernels.strategy.plugin import StrategyPlugin

log = logging.getLogger(__name__)

STRATEGY_TREND_FOLLOW = "trend_follow"
STRATEGY_MEAN_REVERT = "mean_revert"
STRATEGY_VOL_REVERT = "volatility_revert"

# Required float fields only — optional (float | None) fields use resolve_optional_param_vector
# in individual kernels because they need null-awareness for the 0.0 sentinel.
_EXTRACTABLE_FIELDS = tuple(name for name, info in StrategyParams.model_fields.items() if info.annotation is float)


class StrategyRegistry:
    """Register plugins, OR-merge signals, priority-based name assignment, ATR stop/TP."""

    def __init__(self) -> None:
        self._plugins: list[StrategyPlugin] = []
        self._names: set[str] = set()

    def register(self, plugin: StrategyPlugin) -> None:
        if plugin.name in self._names:
            raise ValueError(f"duplicate strategy name: {plugin.name}")
        self._names.add(plugin.name)
        self._plugins.append(plugin)

    def propose_candidates(
        self,
        *,
        context: BatchDecisionContext,
        regime: BatchRegimeProfile,
        settings: StrategyParams,
        variants: tuple[str, ...],
        per_pair_params: tuple[StrategyParams, ...] | None = None,
    ) -> BatchTradeCandidate:
        """Compose all registered strategy signals into trade candidates."""

        size = context.size
        prices = context.market.price
        indicators = context.market.indicators

        # 0. Pre-extract all param vectors once (avoids repeated extraction
        #    in each kernel and in the stop/TP block below).
        pre_extracted: dict[str, np.ndarray] | None = None
        if per_pair_params is not None:
            pre_extracted = {f: extract_param_vector(per_pair_params, f) for f in _EXTRACTABLE_FIELDS}

        # 1. Collect BatchSignalFragment from each registered plugin
        fragments: list[BatchSignalFragment] = [
            plugin.signal(
                states=regime.coarse_state,
                indicators=indicators,
                prices=prices,
                settings=settings,
                per_pair_settings=per_pair_params,
                pre_extracted=pre_extracted,
                regime_confidence=regime.confidence,
            )
            for plugin in self._plugins
        ]

        # 2. OR-merge buy/short masks across all fragments
        #    Per-strategy deconfliction: within each strategy, short is
        #    suppressed where buy is also True. But different strategies
        #    can independently emit buy and short (e.g. trend_follow SHORT
        #    + mean_revert BUY can coexist). Risk engine's max_open_positions
        #    handles position limits.
        buy_mask = np.zeros(size, dtype=np.bool_)
        short_mask = np.zeros(size, dtype=np.bool_)

        for frag in fragments:
            # Per-strategy: short only where this strategy didn't also buy
            frag_short = frag.short_mask & ~frag.buy_mask
            buy_mask |= frag.buy_mask
            short_mask |= frag_short

        if not regime.valid_mask.all():
            suppressed_count = int(np.sum((buy_mask | short_mask) & ~regime.valid_mask))
            if suppressed_count:
                log.debug("suppressed %d signals due to invalid regime data", suppressed_count)

        buy_mask &= regime.valid_mask
        short_mask &= regime.valid_mask
        valid = buy_mask | short_mask

        # 3. Priority-based strategy name assignment
        #    First registered plugin whose buy or short mask is True for a
        #    given row wins the name. Iteration order = registration order,
        #    which mirrors the old router: last writer wins -> reversed here
        #    so that *first* registered has highest priority (matches old
        #    trend > mean > vol ordering where trend overwrote earlier names).
        strategy_name = np.full(size, "none", dtype=object)
        reason_codes = np.full(size, STRATEGY_HOLD, dtype=object)

        # Walk fragments in *reverse* registration order so that the
        # first-registered plugin (highest priority) writes last and wins.
        for frag in reversed(fragments):
            active = frag.buy_mask | frag.short_mask
            strategy_name[active] = frag.strategy_name

        reason_codes[valid] = STRATEGY_SIGNAL

        # 4. Build action / entry / stop / take-profit vectors
        #    ATR-based stop and take-profit logic.
        #    NOTE: keep formula in sync with the scalar version in
        #    ``decision_pipeline.py:_compute_stop_tp_scalar``.
        action = np.full(size, TradeAction.HOLD.value, dtype=np.int64)
        entry = prices.copy()
        stop = np.zeros(size, dtype=np.float64)
        take_profit = np.zeros(size, dtype=np.float64)

        action[buy_mask] = TradeAction.BUY.value
        action[short_mask] = TradeAction.SHORT.value

        atr = indicators[:, INDICATOR_INDEX["atr"]]

        if pre_extracted is not None:
            stop_atr_mult = pre_extracted["stop_atr_mult"]
            rr_ratio = pre_extracted["rr_ratio"]
            min_stop_pct = pre_extracted["min_stop_distance_pct"]
        else:
            stop_atr_mult = np.full(size, settings.stop_atr_mult, dtype=np.float64)
            rr_ratio = np.full(size, settings.rr_ratio, dtype=np.float64)
            min_stop_pct = np.full(size, settings.min_stop_distance_pct, dtype=np.float64)

        base_distance = np.maximum(
            atr * stop_atr_mult,
            prices * min_stop_pct / 100.0,
        )

        buy_rows = action == TradeAction.BUY.value
        short_rows = action == TradeAction.SHORT.value

        stop[buy_rows] = prices[buy_rows] - base_distance[buy_rows]
        take_profit[buy_rows] = prices[buy_rows] + base_distance[buy_rows] * rr_ratio[buy_rows]

        stop[short_rows] = prices[short_rows] + base_distance[short_rows]
        take_profit[short_rows] = prices[short_rows] - base_distance[short_rows] * rr_ratio[short_rows]

        np.maximum(stop, 0.0, out=stop)
        np.maximum(take_profit, 0.0, out=take_profit)

        # 5. Confluence filter — gate low-quality entries
        if settings.confluence_filter_enabled:
            scores = compute_confluence_score(indicators, prices, action, settings)
            low_score = scores < settings.min_confluence_score
            action[low_score] = TradeAction.HOLD.value
            stop[low_score] = 0.0
            take_profit[low_score] = 0.0
            valid = valid & ~low_score
            strategy_name[low_score] = "none"
            reason_codes[low_score] = STRATEGY_HOLD

        # 6. Return BatchTradeCandidate
        return BatchTradeCandidate(
            action=action,
            entry_price=entry,
            stop_price=stop,
            take_profit_price=take_profit,
            strategy_name=tuple(strategy_name),
            strategy_variant=variants,
            reason_codes=tuple(reason_codes),
            valid_mask=valid.astype(np.bool_),
        )


def build_default_strategy_registry(
    enabled: tuple[str, ...] | None = None,
) -> StrategyRegistry:
    """Build registry with default plugins: trend_follow > mean_revert > vol_revert.

    If *enabled* is given, only plugins whose ``name`` appears in it are registered.
    Raises ``ValueError`` if any name in *enabled* doesn't match a known plugin.
    """
    from dojiwick.compute.kernels.strategy.plugin import StrategyPluginAdapter
    from dojiwick.compute.kernels.strategy.trend_follow import trend_follow_signal
    from dojiwick.compute.kernels.strategy.mean_revert import mean_revert_signal
    from dojiwick.compute.kernels.strategy.vol_revert import vol_revert_signal

    all_plugins = [
        StrategyPluginAdapter(trend_follow_signal, STRATEGY_TREND_FOLLOW),
        StrategyPluginAdapter(mean_revert_signal, STRATEGY_MEAN_REVERT),
        StrategyPluginAdapter(vol_revert_signal, STRATEGY_VOL_REVERT),
    ]

    if enabled is not None:
        known_names = {p.name for p in all_plugins}
        unknown = set(enabled) - known_names
        if unknown:
            raise ValueError(
                f"unknown strategy names: {', '.join(sorted(unknown))}; known: {', '.join(sorted(known_names))}"
            )
        enabled_set = set(enabled)
        all_plugins = [p for p in all_plugins if p.name in enabled_set]

    registry = StrategyRegistry()
    for plugin in all_plugins:
        registry.register(plugin)
    return registry
