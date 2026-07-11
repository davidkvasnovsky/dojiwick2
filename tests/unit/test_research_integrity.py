"""Characterization tests for research-gate and optimizer integrity fixes."""

import numpy as np
import pytest

from dojiwick.application.use_cases.validation.research_gate import (
    GateThresholds,
    evaluate_research_gate,
)
from dojiwick.application.use_cases.validation.walk_forward_validator import WalkForwardResult, WindowResult
from dojiwick.config.param_tuning import perturb_exit_geometry
from dojiwick.config.risk_scope import RiskOverrideValues, RiskScopeRule
from dojiwick.config.scope import ScopeSelector, StrategyOverrideValues, StrategyScopeResolver, StrategyScopeRule
from dojiwick.config.risk_scope import RiskScopeResolver
from dojiwick.config.schema import Settings
from dojiwick.domain.enums import MarketState, WFMode
from dojiwick.application.use_cases.validation.cross_validator import CVResult
from fixtures.factories.infrastructure import default_settings


def _wf(windows: list[tuple[float, float]], ratio: float, min_oos: float | None = None) -> WalkForwardResult:
    ws = tuple(
        WindowResult(
            is_sharpe=i_s,
            oos_sharpe=o_s,
            is_start=0,
            is_end=1,
            oos_start=2,
            oos_end=3,
            is_trades=50,
            oos_trades=50,
        )
        for i_s, o_s in windows
    )
    oos = [o for _, o in windows]
    return WalkForwardResult(
        windows=ws,
        aggregate_oos_sharpe=float(np.mean(oos)),
        oos_is_ratio=ratio,
        min_oos_sharpe=min_oos if min_oos is not None else float(np.min(oos)),
    )


def _cv(mean: float = 1.5) -> CVResult:
    sharpes = np.array([mean, mean, mean], dtype=np.float64)
    return CVResult(fold_sharpes=sharpes, mean_sharpe=mean, std_sharpe=0.0, min_sharpe=mean)


def test_negative_is_and_oos_cannot_pass_ratio_gate() -> None:
    """Two negatives no longer produce a passing positive ratio."""
    # The validator's derivation: mean_is <= 0 → ratio 0.0
    mean_is, mean_oos = -2.0, -1.0
    ratio = mean_oos / mean_is if mean_is > 0.0 else 0.0
    assert ratio == 0.0

    result = _wf([(-2.0, -1.0)], ratio=ratio)
    gate = evaluate_research_gate(
        _cv(),
        pbo=0.1,
        walk_forward_result=result,
        thresholds=GateThresholds(
            min_cv_sharpe=0.5,
            max_pbo=0.5,
            wf_mode=WFMode.RATIO,
            min_oos_degradation_ratio=0.4,
            min_wf_oos_sharpe=-10.0,
        ),
    )
    assert not gate.passed


def test_worst_window_floor_rejects_single_catastrophic_window() -> None:
    result = _wf([(2.0, 2.0), (2.0, 1.5), (2.0, -3.0)], ratio=0.9)
    thresholds = GateThresholds(
        min_cv_sharpe=0.5,
        max_pbo=0.5,
        wf_mode=WFMode.RATIO,
        min_oos_degradation_ratio=0.4,
        min_wf_oos_sharpe=0.0,
        min_wf_worst_window_sharpe=-1.0,
    )
    gate = evaluate_research_gate(_cv(), pbo=0.1, walk_forward_result=result, thresholds=thresholds)
    assert not gate.passed
    assert any("worst window" in r for r in gate.rejection_reasons)


def test_trade_freq_bonus_is_capped() -> None:
    from dojiwick.application.use_cases.optimization.objective import _base_score  # pyright: ignore[reportPrivateUsage]
    from dojiwick.domain.models.value_objects.outcome_models import BacktestSummary
    from fixtures.factories.infrastructure import default_optimization_settings

    opt = default_optimization_settings(
        objective_min_trades=1, objective_trade_freq_weight=1.0, objective_trade_freq_cap=50.0
    )

    def score_at(trades: int) -> float:
        summary = BacktestSummary(
            trades=trades,
            total_pnl_usd=0.0,
            win_rate=0.5,
            expectancy_usd=0.0,
            sharpe_like=1.0,
            max_drawdown_pct=5.0,
        )
        return _base_score(opt, summary, equity_start=0.0, n_bars=1000)

    # 50/1000 bars hits the cap; 500/1000 must not score any higher freq bonus
    assert score_at(500) == pytest.approx(score_at(50), abs=1e-9)  # pyright: ignore[reportUnknownMemberType]


def test_risk_scope_scaling_is_clamped_to_search_bounds() -> None:
    from dojiwick.config.param_tuning import apply_params

    rule = RiskScopeRule(
        id="volatile_risk",
        priority=100,
        selector=ScopeSelector(regime=MarketState.VOLATILE),
        values=RiskOverrideValues(risk_per_trade_pct=3.5),
    )
    settings = default_settings().model_copy(update={"risk_scope": RiskScopeResolver(rules=(rule,))})
    # Optimizer triples global risk (1.0 → 3.0): unclamped scaling would push
    # the scoped 3.5 to 10.5 — far past the explored search space (max 4.0)
    tuned = apply_params(settings, {"risk_per_trade_pct": 3.0}, baseline=settings)
    scaled = tuned.risk_scope.rules[0].values.risk_per_trade_pct
    assert scaled == pytest.approx(4.0)  # pyright: ignore[reportUnknownMemberType]


def test_perturb_exit_geometry_scales_global_and_scope_rules() -> None:
    scope_rule = StrategyScopeRule(
        id="auto_volatile",
        priority=10,
        selector=ScopeSelector(regime=MarketState.VOLATILE),
        values=StrategyOverrideValues(stop_atr_mult=2.0, rr_ratio=3.0),
    )
    settings = default_settings().model_copy(update={"strategy_scope": StrategyScopeResolver(rules=(scope_rule,))})

    shocked = perturb_exit_geometry(settings, tp_shift_pct=-10.0, sl_shift_pct=10.0)
    assert isinstance(shocked, Settings)

    assert shocked.strategy.rr_ratio == pytest.approx(settings.strategy.rr_ratio * 0.9)  # pyright: ignore[reportUnknownMemberType]
    assert shocked.strategy.stop_atr_mult == pytest.approx(settings.strategy.stop_atr_mult * 1.1)  # pyright: ignore[reportUnknownMemberType]
    values = shocked.strategy_scope.rules[0].values
    assert values.rr_ratio == pytest.approx(2.7)  # pyright: ignore[reportUnknownMemberType]
    assert values.stop_atr_mult == pytest.approx(2.2)  # pyright: ignore[reportUnknownMemberType]
