"""Bot config snapshot repository protocol."""

from typing import Protocol


class BotConfigSnapshotRepositoryPort(Protocol):
    """Configuration snapshot persistence for audit and rollback."""

    async def record_snapshot(self, config_hash: str, config_json: str) -> None:
        """Persist a configuration snapshot."""
        ...

    async def get_latest(self) -> tuple[str, str] | None:
        """Return the latest snapshot as (hash, json), or None if absent."""
        ...
