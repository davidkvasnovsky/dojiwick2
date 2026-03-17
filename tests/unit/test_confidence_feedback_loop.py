"""Tests for confidence_raw capture before AI ensemble."""

# pyright: reportMissingImports=false, reportUnknownVariableType=false
# pyright: reportUntypedFunctionDecorator=false, reportUnknownParameterType=false
# pyright: reportMissingParameterType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false

import numpy as np
from hypothesis import given, settings

from dojiwick.application.orchestration.decision_pipeline import run_decision_pipeline
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.compute.kernels.regime.classify import classify_regime_batch
from dojiwick.compute.kernels.regime.ensemble import combine_regime_ensemble
from fixtures.factories.infrastructure import default_risk_settings, default_settings
from dojiwick.domain.enums import DecisionAuthority
from fixtures.factories.domain import ContextBuilder
from fixtures.strategies import st_batch_decision_context


class TestConfidenceFeedbackLoop:
    async def test_confidence_raw_captured_before_ai(self) -> None:
        """Without AI, confidence_raw equals deterministic regime confidence."""
        ctx = ContextBuilder().trending_up().build()
        result = await run_decision_pipeline(
            context=ctx,
            settings=default_settings(),
            strategy_registry=build_default_strategy_registry(),
            risk_engine=build_default_risk_engine(default_risk_settings()),
        )
        # confidence_raw was captured from regimes.confidence before AI step
        for i in range(ctx.size):
            assert float(result.confidence_raw[i]) == float(result.regimes.confidence[i])

    async def test_confidence_raw_wired_to_outcomes(self) -> None:
        """Pipeline result exposes confidence_raw for outcome threading."""
        ctx = ContextBuilder().trending_up().build()
        result = await run_decision_pipeline(
            context=ctx,
            settings=default_settings(),
            strategy_registry=build_default_strategy_registry(),
            risk_engine=build_default_risk_engine(default_risk_settings()),
        )
        assert result.authority == DecisionAuthority.DETERMINISTIC_ONLY
        assert len(result.confidence_raw) == ctx.size
        assert all(0.0 <= float(c) <= 1.0 for c in result.confidence_raw)

    @given(context=st_batch_decision_context())
    @settings(max_examples=50, deadline=5000)
    async def test_confidence_always_bounded(self, context) -> None:
        """After ensemble adjustment, confidence stays in [0, 1]."""
        s = default_settings()
        det_regimes = classify_regime_batch(context.market, s.regime.params)

        # Simulate AI regime with various boost/penalty values
        for boost, penalty in [(1.0, 1.0), (1.5, 0.3), (2.0, 0.1)]:
            ensemble = combine_regime_ensemble(
                deterministic=det_regimes,
                ai=det_regimes,
                boost=boost,
                penalty=penalty,
            )
            assert np.all(ensemble.confidence >= 0.0), f"confidence < 0 with boost={boost}, penalty={penalty}"
            assert np.all(ensemble.confidence <= 1.0), f"confidence > 1 with boost={boost}, penalty={penalty}"

    @given(context=st_batch_decision_context(min_size=1, max_size=2))
    @settings(max_examples=30, deadline=5000)
    async def test_repeated_disagreement_monotonically_decreases(self, context) -> None:
        """N repeated penalties → confidence sequence is non-increasing."""
        s = default_settings()
        det_regimes = classify_regime_batch(context.market, s.regime.params)

        penalty = 0.6
        prev_confidence = det_regimes.confidence.copy()

        from dojiwick.domain.models.value_objects.batch_models import BatchRegimeProfile

        for _ in range(5):
            # Simulate disagreement: AI gives opposite states
            flipped = det_regimes.coarse_state.copy()
            flipped[:] = (flipped + 1) % 4 + 1  # shift to different MarketState

            ai_regimes = BatchRegimeProfile(
                coarse_state=flipped,
                confidence=np.ones(context.size, dtype=np.float64),
                valid_mask=np.ones(context.size, dtype=np.bool_),
            )
            ensemble = combine_regime_ensemble(
                deterministic=BatchRegimeProfile(
                    coarse_state=det_regimes.coarse_state.copy(),
                    confidence=prev_confidence.copy(),
                    valid_mask=det_regimes.valid_mask.copy(),
                ),
                ai=ai_regimes,
                boost=1.25,
                penalty=penalty,
            )
            assert np.all(ensemble.confidence <= prev_confidence + 1e-12)
            prev_confidence = ensemble.confidence.copy()
