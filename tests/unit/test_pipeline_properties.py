"""Property-based tests for pipeline determinism and invariants."""

# pyright: reportMissingImports=false, reportUnknownVariableType=false
# pyright: reportUntypedFunctionDecorator=false, reportUnknownParameterType=false
# pyright: reportMissingParameterType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false

import numpy as np
from hypothesis import given, settings

from dojiwick.application.orchestration.decision_pipeline import run_decision_pipeline
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from fixtures.factories.infrastructure import default_risk_settings, default_settings
from dojiwick.domain.enums import MarketState
from fixtures.strategies import st_batch_decision_context


@given(context=st_batch_decision_context())
@settings(max_examples=50, deadline=5000)
async def test_pipeline_same_inputs_same_outputs(context):
    """Two runs with identical context produce bitwise-equal outputs."""
    s = default_settings()
    reg = build_default_strategy_registry()
    eng = build_default_risk_engine(default_risk_settings())

    r1 = await run_decision_pipeline(context=context, settings=s, strategy_registry=reg, risk_engine=eng)
    r2 = await run_decision_pipeline(context=context, settings=s, strategy_registry=reg, risk_engine=eng)

    np.testing.assert_array_equal(r1.regimes.coarse_state, r2.regimes.coarse_state)
    np.testing.assert_array_equal(r1.regimes.confidence, r2.regimes.confidence)
    np.testing.assert_array_equal(r1.intents.action, r2.intents.action)
    np.testing.assert_array_equal(r1.confidence_raw, r2.confidence_raw)
    assert r1.variants == r2.variants
    assert r1.authority == r2.authority


@given(context=st_batch_decision_context())
@settings(max_examples=50, deadline=5000)
async def test_pipeline_output_shapes_match_input(context):
    """All output vectors have length == context.size."""
    result = await run_decision_pipeline(
        context=context,
        settings=default_settings(),
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
    )
    size = context.size
    assert len(result.regimes.coarse_state) == size
    assert len(result.regimes.confidence) == size
    assert len(result.intents.action) == size
    assert len(result.confidence_raw) == size
    assert len(result.variants) == size
    assert len(result.candidates.action) == size
    assert len(result.veto.approved_mask) == size
    assert len(result.risk.allowed_mask) == size


@given(context=st_batch_decision_context())
@settings(max_examples=50, deadline=5000)
async def test_pipeline_confidence_raw_equals_regime_without_ai(context):
    """Without AI, confidence_raw equals deterministic regime confidence."""
    result = await run_decision_pipeline(
        context=context,
        settings=default_settings(),
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
    )
    np.testing.assert_array_equal(result.confidence_raw, result.regimes.confidence)


@given(context=st_batch_decision_context())
@settings(max_examples=50, deadline=5000)
async def test_pipeline_regime_confidence_bounded(context):
    """Regime confidence always in [0, 1]."""
    result = await run_decision_pipeline(
        context=context,
        settings=default_settings(),
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
    )
    assert np.all(result.regimes.confidence >= 0.0)
    assert np.all(result.regimes.confidence <= 1.0)


@given(context=st_batch_decision_context())
@settings(max_examples=50, deadline=5000)
async def test_pipeline_regime_states_are_valid_enum_values(context):
    """All coarse_state values are valid MarketState members."""
    result = await run_decision_pipeline(
        context=context,
        settings=default_settings(),
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
    )
    valid_values = {m.value for m in MarketState}
    for state in result.regimes.coarse_state:
        assert int(state) in valid_values
