"""Strategy state repository protocol."""

from typing import Protocol


class StrategyStateRepositoryPort(Protocol):
    """Per-pair strategy state persistence."""

    async def get_state(self, pair: str, strategy_name: str, variant: str) -> dict[str, object] | None:
        """Return the stored state for a pair/strategy/variant, or None if absent."""
        ...

    async def upsert_state(
        self,
        pair: str,
        strategy_name: str,
        variant: str,
        state: dict[str, object],
        *,
        target_id: str,
        venue: str,
        product: str,
    ) -> None:
        """Insert or update the state for a pair/strategy/variant."""
        ...
