"""Search-space emission of per-regime volatile entry-gate params."""

from fixtures.factories.infrastructure import default_settings

from dojiwick.application.use_cases.optimization.search_space import SearchSpace
from dojiwick.config.param_tuning import apply_params


def test_volatile_entry_gates_in_bounds() -> None:
    space = SearchSpace(enabled_strategies=("trend_follow", "volatility_revert", "mean_revert"))
    bounds = space.bounds()
    assert bounds["scope_volatile__trend_breakout_adx_min"] == (25.0, 60.0)
    assert bounds["scope_volatile__trend_pullback_adx_min"] == (15.0, 55.0)
    # Existing exit params still present
    assert "scope_volatile__stop_atr_mult" in bounds


def test_volatile_entry_gates_absent_when_volatile_inactive() -> None:
    space = SearchSpace(enabled_strategies=("mean_revert",))
    bounds = space.bounds()
    assert "scope_volatile__trend_breakout_adx_min" not in bounds
    assert "scope_volatile__trend_pullback_adx_min" not in bounds


def test_apply_params_routes_volatile_entry_gates_to_auto_rule() -> None:
    settings = default_settings()
    tuned = apply_params(
        settings,
        {
            "scope_volatile__trend_breakout_adx_min": 42.0,
            "scope_volatile__trend_pullback_adx_min": 33.0,
            "scope_volatile__stop_atr_mult": 2.0,
        },
    )
    rules = {r.id: r for r in tuned.strategy_scope.rules}
    auto = rules["auto_volatile"]
    assert auto.values.trend_breakout_adx_min == 42.0
    assert auto.values.trend_pullback_adx_min == 33.0
    assert auto.values.stop_atr_mult == 2.0
    assert auto.selector.strategy is None  # regime-only rule: binds in phase-1 entry resolution
