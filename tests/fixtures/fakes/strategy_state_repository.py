"""Fake strategy state repository for tests."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class FakeStrategyStateRepository:
    """In-memory strategy state repository for test assertions."""

    _storage: dict[tuple[str, str, str], dict[str, object]] = field(default_factory=dict)

    async def get_state(self, pair: str, strategy_name: str, variant: str) -> dict[str, object] | None:
        return self._storage.get((pair, strategy_name, variant))

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
        del target_id, venue, product
        self._storage[(pair, strategy_name, variant)] = state
