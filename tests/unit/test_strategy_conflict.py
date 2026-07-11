"""Cross-strategy signal conflict resolution: first-registered plugin wins."""

import numpy as np
from fixtures.factories.domain import ContextBuilder
from fixtures.factories.infrastructure import default_strategy_params

from dojiwick.application.registry.strategy_registry import StrategyRegistry
from dojiwick.compute.kernels.strategy.plugin import StrategyPluginAdapter
from dojiwick.domain.enums import TradeAction
from dojiwick.domain.models.value_objects.params import StrategyParams
from dojiwick.domain.type_aliases import BoolVector, FloatMatrix, FloatVector, IntVector


def _fixed_signal(buy: list[bool], short: list[bool]):
    def signal(
        *,
        states: IntVector,
        indicators: FloatMatrix,
        prices: FloatVector,
        settings: StrategyParams,
        per_pair_settings: tuple[StrategyParams, ...] | None = None,
        pre_extracted: dict[str, np.ndarray] | None = None,
        regime_confidence: FloatVector | None = None,
    ) -> tuple[BoolVector, BoolVector]:
        return np.array(buy, dtype=np.bool_), np.array(short, dtype=np.bool_)

    return signal


def test_first_registered_plugin_wins_action_and_attribution() -> None:
    registry = StrategyRegistry()
    # Row 0: A says BUY, B says SHORT → A must win both action and name.
    # Row 1: only B signals SHORT → B wins.
    registry.register(StrategyPluginAdapter(_fixed_signal([True, False], [False, False]), "alpha"))
    registry.register(StrategyPluginAdapter(_fixed_signal([False, False], [True, True]), "beta"))

    ctx = ContextBuilder(pairs=("BTC/USDC", "ETH/USDC")).build()
    from dojiwick.domain.models.value_objects.batch_models import BatchRegimeProfile

    regime = BatchRegimeProfile(
        coarse_state=np.zeros(2, dtype=np.int64),
        confidence=np.ones(2, dtype=np.float64),
        valid_mask=np.ones(2, dtype=np.bool_),
    )
    params = default_strategy_params(confluence_filter_enabled=False)
    candidates = registry.propose_candidates(
        context=ctx,
        regime=regime,
        settings=params,
        variants=("baseline", "baseline"),
        per_pair_params=None,
    )

    assert candidates.action[0] == TradeAction.BUY.value
    assert candidates.strategy_name[0] == "alpha"
    assert candidates.action[1] == TradeAction.SHORT.value
    assert candidates.strategy_name[1] == "beta"
