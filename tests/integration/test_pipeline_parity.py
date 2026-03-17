"""Pipeline parity test: identical context → identical decisions.

Verifies that backtest-mode and live-mode (with veto disabled) produce
identical intents through the shared ``run_decision_pipeline``.
"""

import numpy as np

from dojiwick.application.orchestration.decision_pipeline import run_decision_pipeline
from dojiwick.application.orchestration.regime_hysteresis import RegimeHysteresis
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from fixtures.factories.infrastructure import default_risk_settings, default_settings
from dojiwick.infrastructure.ai.llm_filter import NullVetoService
from fixtures.factories.domain import ContextBuilder


async def test_pipeline_parity_backtest_vs_live() -> None:
    """Same context, same settings → identical intents from shared pipeline.

    Backtest mode: no hysteresis, no veto, no adaptive.
    Live mode (veto disabled): no hysteresis, NullVetoService, no adaptive.
    Both must produce identical intents.
    """
    context = ContextBuilder().trending_up().build()
    settings = default_settings()
    registry = build_default_strategy_registry()
    engine = build_default_risk_engine(default_risk_settings())

    # Backtest-equivalent: no optional deps
    result_backtest = await run_decision_pipeline(
        context=context,
        settings=settings,
        strategy_registry=registry,
        risk_engine=engine,
    )

    # Live-equivalent (veto disabled): NullVetoService (all-approved)
    result_live = await run_decision_pipeline(
        context=context,
        settings=settings,
        strategy_registry=registry,
        risk_engine=engine,
        veto_service=NullVetoService(),
    )

    # Intents must be identical
    np.testing.assert_array_equal(result_backtest.intents.action, result_live.intents.action)
    np.testing.assert_array_equal(result_backtest.intents.quantity, result_live.intents.quantity)
    np.testing.assert_array_equal(result_backtest.intents.entry_price, result_live.intents.entry_price)
    np.testing.assert_array_equal(result_backtest.intents.active_mask, result_live.intents.active_mask)
    assert result_backtest.intents.strategy_name == result_live.intents.strategy_name
    assert result_backtest.intents.strategy_variant == result_live.intents.strategy_variant

    # Regime classifications must be identical (no hysteresis in either)
    np.testing.assert_array_equal(result_backtest.regimes.coarse_state, result_live.regimes.coarse_state)
    np.testing.assert_array_equal(result_backtest.regimes.confidence, result_live.regimes.confidence)

    # Variants must be identical
    assert result_backtest.variants == result_live.variants


async def test_hysteresis_replay_matches_sequential_ticks() -> None:
    """Two fresh RegimeHysteresis instances fed identical bars produce identical results.

    Proves that replay through ``run_decision_pipeline`` with hysteresis is
    deterministic: same contexts, same hysteresis state → same per-bar
    actions and regime classifications.
    """
    from dojiwick.application.orchestration.decision_pipeline import PipelineResult

    contexts = [ContextBuilder().trending_up().build() for _ in range(3)]
    settings = default_settings()
    registry = build_default_strategy_registry()
    engine = build_default_risk_engine(default_risk_settings())

    async def _run_sequence() -> list[PipelineResult]:
        hyst = RegimeHysteresis()
        results: list[PipelineResult] = []
        for ctx in contexts:
            result = await run_decision_pipeline(
                context=ctx,
                settings=settings,
                strategy_registry=registry,
                risk_engine=engine,
                hysteresis=hyst,
            )
            results.append(result)
        return results

    first_run = await _run_sequence()
    second_run = await _run_sequence()

    for bar_idx, (a, b) in enumerate(zip(first_run, second_run, strict=True)):
        np.testing.assert_array_equal(
            a.intents.action,
            b.intents.action,
            err_msg=f"bar {bar_idx}: action mismatch",
        )
        np.testing.assert_array_equal(
            a.regimes.coarse_state,
            b.regimes.coarse_state,
            err_msg=f"bar {bar_idx}: coarse_state mismatch",
        )
