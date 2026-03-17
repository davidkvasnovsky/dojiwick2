"""Tests for regime-adaptive exit profile modifiers in run_backtest.py."""

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false

import pytest

from dojiwick.application.use_cases.run_backtest import (
    _ExitContext,
    _ExitProfile,
    _apply_mean_revert_overrides,
    _apply_ranging_exits,
    _apply_trending_exits,
    _apply_volatile_exits,
)
from fixtures.factories.infrastructure import default_strategy_params


def _base_profile() -> _ExitProfile:
    return _ExitProfile(
        stop=95.0,
        take_profit=110.0,
        trailing_activation_price=105.0,
        trailing_distance=2.0,
        breakeven_price=102.0,
        max_hold_bars=20,
    )


def _base_context(**overrides: object) -> _ExitContext:
    defaults: dict[str, object] = {
        "entry": 100.0,
        "stop_distance": 5.0,
        "direction": 1,
        "bb_mid": 103.0,
        "params": default_strategy_params(),
    }
    defaults.update(overrides)
    return _ExitContext(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
class TestVolatileModifier:
    def test_scales_stop_and_tp(self) -> None:
        profile = _base_profile()
        params = default_strategy_params(adaptive_volatile_stop_scale=1.5, adaptive_volatile_rr_mult=1.2)
        ctx = _base_context(params=params)
        _apply_volatile_exits(profile, ctx)

        new_dist = 5.0 * 1.5  # 7.5
        assert profile.stop == pytest.approx(100.0 - new_dist)
        assert profile.take_profit == pytest.approx(100.0 + new_dist * 1.2)

    def test_preserves_trailing(self) -> None:
        profile = _base_profile()
        orig_activation = profile.trailing_activation_price
        orig_distance = profile.trailing_distance
        ctx = _base_context()
        _apply_volatile_exits(profile, ctx)
        assert profile.trailing_activation_price == orig_activation
        assert profile.trailing_distance == orig_distance

    def test_caps_max_hold_bars(self) -> None:
        profile = _base_profile()
        params = default_strategy_params(adaptive_volatile_max_bars=10)
        ctx = _base_context(params=params)
        _apply_volatile_exits(profile, ctx)
        assert profile.max_hold_bars == 10

    def test_caps_max_hold_bars_zero_means_set(self) -> None:
        profile = _base_profile()
        profile.max_hold_bars = 0
        params = default_strategy_params(adaptive_volatile_max_bars=15)
        ctx = _base_context(params=params)
        _apply_volatile_exits(profile, ctx)
        assert profile.max_hold_bars == 15


@pytest.mark.unit
class TestTrendingModifier:
    def test_scales_trailing(self) -> None:
        profile = _base_profile()
        params = default_strategy_params(adaptive_trending_trail_scale=2.0)
        ctx = _base_context(params=params)
        _apply_trending_exits(profile, ctx)
        assert profile.trailing_distance == pytest.approx(4.0)

    def test_noop_when_no_trailing(self) -> None:
        profile = _base_profile()
        profile.trailing_distance = 0.0
        ctx = _base_context()
        _apply_trending_exits(profile, ctx)
        assert profile.trailing_distance == 0.0


@pytest.mark.unit
class TestRangingModifier:
    def test_uses_bb_mid(self) -> None:
        profile = _base_profile()
        ctx = _base_context(bb_mid=103.5)
        _apply_ranging_exits(profile, ctx)
        assert profile.take_profit == 103.5

    def test_disables_breakeven(self) -> None:
        profile = _base_profile()
        ctx = _base_context()
        _apply_ranging_exits(profile, ctx)
        assert profile.breakeven_price == 0.0

    def test_caps_max_hold_bars(self) -> None:
        profile = _base_profile()
        params = default_strategy_params(adaptive_ranging_max_bars=8)
        ctx = _base_context(params=params)
        _apply_ranging_exits(profile, ctx)
        assert profile.max_hold_bars == 8


@pytest.mark.unit
class TestMeanRevertOverrides:
    def test_tp_to_bb_mid(self) -> None:
        profile = _base_profile()
        params = default_strategy_params(mean_revert_use_bb_mid_tp=True)
        ctx = _base_context(bb_mid=104.0, params=params)
        _apply_mean_revert_overrides(profile, ctx)
        assert profile.take_profit == 104.0

    def test_mean_revert_plus_ranging_idempotent(self) -> None:
        profile = _base_profile()
        params = default_strategy_params(mean_revert_use_bb_mid_tp=True, adaptive_ranging_max_bars=12)
        ctx = _base_context(bb_mid=103.0, params=params)
        _apply_mean_revert_overrides(profile, ctx)
        _apply_ranging_exits(profile, ctx)
        # Both set TP to bb_mid — idempotent
        assert profile.take_profit == 103.0
        assert profile.breakeven_price == 0.0
