"""Pair state repository test doubles."""

from dataclasses import dataclass, field

from dojiwick.domain.models.entities.pair_state import PairTradingState


@dataclass(slots=True)
class InMemoryPairStateRepo:
    """In-memory per-pair state for tests."""

    _states: dict[str, PairTradingState] = field(default_factory=dict)

    async def get_state(self, pair: str) -> PairTradingState | None:
        return self._states.get(pair)

    async def get_all(self) -> tuple[PairTradingState, ...]:
        return tuple(self._states.values())

    async def upsert(self, state: PairTradingState) -> None:
        self._states[state.pair] = state


class FailingPairStateRepo:
    """Raises on all operations."""

    async def get_state(self, pair: str) -> PairTradingState | None:
        del pair
        raise RuntimeError("pair state repo failure")

    async def get_all(self) -> tuple[PairTradingState, ...]:
        raise RuntimeError("pair state repo failure")

    async def upsert(self, state: PairTradingState) -> None:
        del state
        raise RuntimeError("pair state repo failure")
