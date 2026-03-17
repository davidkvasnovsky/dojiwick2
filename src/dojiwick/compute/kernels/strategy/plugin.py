"""Strategy plugin protocol and generic adapter for composable signal generation."""

from typing import Protocol

import numpy as np

from dojiwick.domain.models.value_objects.batch_models import BatchSignalFragment
from dojiwick.domain.models.value_objects.params import StrategyParams
from dojiwick.domain.type_aliases import BoolVector, FloatMatrix, FloatVector, IntVector


class SignalFunction(Protocol):
    """Callable protocol for kernel signal functions."""

    def __call__(
        self,
        *,
        states: IntVector,
        indicators: FloatMatrix,
        prices: FloatVector,
        settings: StrategyParams,
        per_pair_settings: tuple[StrategyParams, ...] | None = None,
        pre_extracted: dict[str, np.ndarray] | None = None,
        regime_confidence: FloatVector | None = None,
    ) -> tuple[BoolVector, BoolVector]: ...


class StrategyPlugin(Protocol):
    """A pluggable strategy signal generator."""

    @property
    def name(self) -> str: ...

    def signal(
        self,
        *,
        states: IntVector,
        indicators: FloatMatrix,
        prices: FloatVector,
        settings: StrategyParams,
        per_pair_settings: tuple[StrategyParams, ...] | None = None,
        pre_extracted: dict[str, np.ndarray] | None = None,
        regime_confidence: FloatVector | None = None,
    ) -> BatchSignalFragment: ...


class StrategyPluginAdapter:
    """Generic adapter that wraps a kernel signal function as a StrategyPlugin."""

    def __init__(self, signal_fn: SignalFunction, strategy_name: str) -> None:
        self._signal_fn = signal_fn
        self._name = strategy_name

    @property
    def name(self) -> str:
        return self._name

    def signal(
        self,
        *,
        states: IntVector,
        indicators: FloatMatrix,
        prices: FloatVector,
        settings: StrategyParams,
        per_pair_settings: tuple[StrategyParams, ...] | None = None,
        pre_extracted: dict[str, np.ndarray] | None = None,
        regime_confidence: FloatVector | None = None,
    ) -> BatchSignalFragment:
        buy_mask, short_mask = self._signal_fn(
            states=states,
            indicators=indicators,
            prices=prices,
            settings=settings,
            per_pair_settings=per_pair_settings,
            pre_extracted=pre_extracted,
            regime_confidence=regime_confidence,
        )
        return BatchSignalFragment(
            strategy_name=self._name,
            buy_mask=buy_mask,
            short_mask=short_mask,
        )
