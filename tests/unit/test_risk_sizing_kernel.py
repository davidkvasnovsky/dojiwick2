"""Risk sizing kernel edge case tests."""

import numpy as np

from dojiwick.compute.kernels.sizing.fixed_fraction import size_intents
from fixtures.factories.infrastructure import default_risk_params, default_settings
from dojiwick.domain.models.value_objects.batch_models import BatchRiskAssessment, BatchTradeCandidate
from dojiwick.domain.enums import TradeAction
from fixtures.factories.domain import ContextBuilder


def _make_candidate(size: int, action: int = TradeAction.BUY.value) -> BatchTradeCandidate:
    return BatchTradeCandidate(
        action=np.full(size, action, dtype=np.int64),
        entry_price=np.full(size, 100.0, dtype=np.float64),
        stop_price=np.full(size, 99.0, dtype=np.float64),
        take_profit_price=np.full(size, 102.0, dtype=np.float64),
        strategy_name=tuple("trend_follow" for _ in range(size)),
        strategy_variant=tuple("baseline" for _ in range(size)),
        reason_codes=tuple("strategy_signal" for _ in range(size)),
        valid_mask=np.ones(size, dtype=np.bool_),
    )


def _make_risk(size: int, allowed: bool = True) -> BatchRiskAssessment:
    return BatchRiskAssessment(
        allowed_mask=np.full(size, allowed, dtype=np.bool_),
        reason_codes=tuple("risk_ok" for _ in range(size)),
        risk_score=np.full(size, 0.5, dtype=np.float64),
    )


def test_zero_equity_produces_zero_sizing() -> None:
    context = ContextBuilder().with_equity([0.0, 0.0]).build()
    candidate = _make_candidate(2)
    risk = _make_risk(2)
    settings = default_settings()

    rp = settings.risk.params
    intent = size_intents(context=context, candidate=candidate, assessment=risk, risk_params=(rp, rp))

    assert np.all(intent.quantity == 0.0)
    assert np.all(intent.notional_usd == 0.0)


def test_zero_stop_distance_clips_to_min_notional() -> None:
    """When stop_distance=0, raw_quantity=0 but min_notional clips up."""
    context = ContextBuilder().build()
    candidate = BatchTradeCandidate(
        action=np.array([TradeAction.BUY.value, TradeAction.BUY.value], dtype=np.int64),
        entry_price=np.array([100.0, 100.0], dtype=np.float64),
        stop_price=np.array([100.0, 100.0], dtype=np.float64),
        take_profit_price=np.array([102.0, 102.0], dtype=np.float64),
        strategy_name=("trend_follow", "trend_follow"),
        strategy_variant=("baseline", "baseline"),
        reason_codes=("strategy_signal", "strategy_signal"),
        valid_mask=np.array([True, True], dtype=np.bool_),
    )
    risk = _make_risk(2)
    settings = default_settings()

    rp = settings.risk.params
    intent = size_intents(context=context, candidate=candidate, assessment=risk, risk_params=(rp, rp))

    # zero stop distance is caught by the risk gate upstream;
    # the sizing kernel applies min_notional clipping to the zero raw_quantity
    assert np.all(intent.notional_usd[intent.active_mask] >= settings.risk.min_notional_usd)


def test_max_notional_cap() -> None:
    context = ContextBuilder().with_equity([100.0, 100.0]).build()
    candidate = _make_candidate(2)
    risk = _make_risk(2)
    settings = default_settings()

    rp = settings.risk.params
    intent = size_intents(context=context, candidate=candidate, assessment=risk, risk_params=(rp, rp))

    max_notional = 100.0 * settings.risk.max_notional_pct_of_equity / 100.0
    assert np.all(intent.notional_usd[intent.active_mask] <= max_notional)


def test_per_pair_risk_different_risk_pct() -> None:
    """Two pairs with different risk_per_trade_pct produce different notional sizes."""
    context = ContextBuilder().with_equity([10000.0, 10000.0]).build()
    # Use larger stop distance (entry=100, stop=90) to avoid hitting max_notional cap
    candidate = BatchTradeCandidate(
        action=np.full(2, TradeAction.BUY.value, dtype=np.int64),
        entry_price=np.full(2, 100.0, dtype=np.float64),
        stop_price=np.full(2, 90.0, dtype=np.float64),
        take_profit_price=np.full(2, 120.0, dtype=np.float64),
        strategy_name=("trend_follow", "trend_follow"),
        strategy_variant=("baseline", "baseline"),
        reason_codes=("strategy_signal", "strategy_signal"),
        valid_mask=np.ones(2, dtype=np.bool_),
    )
    risk = _make_risk(2)

    low_risk = default_risk_params(risk_per_trade_pct=0.5)
    high_risk = default_risk_params(risk_per_trade_pct=2.0)

    intent = size_intents(
        context=context,
        candidate=candidate,
        assessment=risk,
        risk_params=(low_risk, high_risk),
    )

    # Both rows should be active; row 1 should have larger notional than row 0
    assert intent.active_mask[0]
    assert intent.active_mask[1]
    assert intent.notional_usd[1] > intent.notional_usd[0]
