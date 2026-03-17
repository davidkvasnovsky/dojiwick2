"""Tick repository protocol."""

from typing import Protocol

from dojiwick.domain.enums import TickStatus
from dojiwick.domain.models.value_objects.tick_record import TickRecord


class TickRepositoryPort(Protocol):
    """Persists tick lifecycle records for deduplication and audit."""

    async def try_insert(self, record: TickRecord) -> bool:
        """Atomically insert a tick record if no completed tick with the same ID exists.

        Returns True if the row was inserted, False if a completed tick already exists (dedup).
        """
        ...

    async def update_status(
        self,
        tick_id: str,
        status: TickStatus,
        *,
        intent_hash: str = "",
        ops_hash: str = "",
        duration_ms: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update an existing tick's status and optional hash fields."""
        ...

    async def recover_stale_started(self, stale_threshold_sec: int = 300) -> int:
        """Mark stale STARTED ticks as FAILED. Returns count of recovered ticks."""
        ...
