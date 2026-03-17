"""Strategy registry and risk engine tests."""

import numpy as np

from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.compute.kernels.regime.classify import classify_regime_batch
from dojiwick.compute.kernels.sizing.fixed_fraction import size_intents
from fixtures.factories.infrastructure import default_risk_settings, default_settings
from dojiwick.domain.models.value_objects.batch_models import BatchDecisionContext


def test_strategy_and_risk_flow_produces_active_intents(sample_context: BatchDecisionContext) -> None:
    settings = default_settings()
    regime = classify_regime_batch(sample_context.market, settings.regime.params)
    variants = tuple(settings.strategy.default_variant for _ in range(sample_context.size))

    registry = build_default_strategy_registry()
    candidate = registry.propose_candidates(
        context=sample_context,
        regime=regime,
        settings=settings.strategy,
        variants=variants,
    )
    engine = build_default_risk_engine(default_risk_settings())
    rp = settings.risk.params
    risk_params = (rp,) * sample_context.size
    risk = engine.assess_risk(context=sample_context, candidate=candidate, risk_params=risk_params)
    intent = size_intents(context=sample_context, candidate=candidate, assessment=risk, risk_params=risk_params)

    assert np.any(candidate.valid_mask)
    assert np.any(risk.allowed_mask)
    assert np.any(intent.active_mask)
    assert np.all(intent.notional_usd[intent.active_mask] > 0.0)
