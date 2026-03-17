"""Strategy registry kernel tests."""

import numpy as np

from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.compute.kernels.regime.classify import classify_regime_batch
from fixtures.factories.infrastructure import default_settings
from dojiwick.domain.enums import TradeAction
from dojiwick.domain.reason_codes import STRATEGY_HOLD, STRATEGY_SIGNAL
from fixtures.factories.domain import ContextBuilder


def test_routing_priority_trend_over_mean_revert() -> None:
    """When both trend-follow and mean-revert fire, trend-follow wins."""
    ctx = ContextBuilder().trending_up().build()
    settings = default_settings()
    regime = classify_regime_batch(ctx.market, settings.regime.params)
    variants = tuple(settings.strategy.default_variant for _ in range(ctx.size))
    registry = build_default_strategy_registry()
    candidates = registry.propose_candidates(context=ctx, regime=regime, settings=settings.strategy, variants=variants)

    for i in range(ctx.size):
        if candidates.valid_mask[i]:
            assert candidates.strategy_name[i] == "trend_follow"


def test_stop_and_tp_computed_for_buy() -> None:
    ctx = ContextBuilder().trending_up().build()
    settings = default_settings()
    regime = classify_regime_batch(ctx.market, settings.regime.params)
    variants = tuple(settings.strategy.default_variant for _ in range(ctx.size))
    registry = build_default_strategy_registry()
    candidates = registry.propose_candidates(context=ctx, regime=regime, settings=settings.strategy, variants=variants)

    buy_rows = candidates.action == TradeAction.BUY.value
    if np.any(buy_rows):
        assert np.all(candidates.stop_price[buy_rows] < candidates.entry_price[buy_rows])
        assert np.all(candidates.take_profit_price[buy_rows] > candidates.entry_price[buy_rows])


def test_stop_and_tp_computed_for_short() -> None:
    ctx = ContextBuilder().trending_up().build()
    settings = default_settings()
    regime = classify_regime_batch(ctx.market, settings.regime.params)
    variants = tuple(settings.strategy.default_variant for _ in range(ctx.size))
    registry = build_default_strategy_registry()
    candidates = registry.propose_candidates(context=ctx, regime=regime, settings=settings.strategy, variants=variants)

    short_rows = candidates.action == TradeAction.SHORT.value
    if np.any(short_rows):
        assert np.all(candidates.stop_price[short_rows] > candidates.entry_price[short_rows])
        assert np.all(candidates.take_profit_price[short_rows] < candidates.entry_price[short_rows])


def test_hold_rows_have_strategy_hold_reason() -> None:
    ctx = ContextBuilder().ranging().build()
    settings = default_settings()
    regime = classify_regime_batch(ctx.market, settings.regime.params)
    variants = tuple(settings.strategy.default_variant for _ in range(ctx.size))
    registry = build_default_strategy_registry()
    candidates = registry.propose_candidates(context=ctx, regime=regime, settings=settings.strategy, variants=variants)

    for i in range(ctx.size):
        if not candidates.valid_mask[i]:
            assert candidates.reason_codes[i] == STRATEGY_HOLD


def test_valid_rows_have_strategy_signal_reason() -> None:
    ctx = ContextBuilder().trending_up().build()
    settings = default_settings()
    regime = classify_regime_batch(ctx.market, settings.regime.params)
    variants = tuple(settings.strategy.default_variant for _ in range(ctx.size))
    registry = build_default_strategy_registry()
    candidates = registry.propose_candidates(context=ctx, regime=regime, settings=settings.strategy, variants=variants)

    for i in range(ctx.size):
        if candidates.valid_mask[i]:
            assert candidates.reason_codes[i] == STRATEGY_SIGNAL
