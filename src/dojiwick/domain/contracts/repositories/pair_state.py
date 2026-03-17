"""Pair trading state repository protocol."""

from typing import Protocol

from dojiwick.domain.models.entities.pair_state import PairTradingState


class PairStateRepositoryPort(Protocol):
    """Per-pair performance state persistence."""

    async def get_state(self, pair: str) -> PairTradingState | None:
        """Return the trading state for a pair, or None if not tracked."""
        ...

    async def get_all(self) -> tuple[PairTradingState, ...]:
        """Return all tracked pair states."""
        ...

    async def upsert(self, state: PairTradingState) -> None:
        """Insert or update a pair's trading state."""
        ...
