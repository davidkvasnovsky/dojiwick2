"""Adaptive config repository protocol."""

from typing import Protocol


class AdaptiveConfigRepositoryPort(Protocol):
    """Adaptive configuration persistence."""

    async def get_config(self, config_idx: int) -> dict[str, object] | None:
        """Return a configuration by index, or None if absent."""
        ...

    async def get_all_configs(self) -> tuple[tuple[int, dict[str, object]], ...]:
        """Return all configurations as (index, config) tuples."""
        ...
