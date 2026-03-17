"""Tick repository test doubles."""

import time
from dataclasses import dataclass, field, replace

from dojiwick.domain.enums import TickStatus
from dojiwick.domain.models.value_objects.tick_record import TickRecord


@dataclass(slots=True)
class NoOpTickRepository:
    """Null object — all operations succeed silently."""

    async def try_insert(self, record: TickRecord) -> bool:
        del record
        return True

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
        pass

    async def recover_stale_started(self, stale_threshold_sec: int = 300) -> int:
        return 0


@dataclass(slots=True)
class InMemoryTickRepo:
    """Dict-based tick storage for tests."""

    _records: dict[str, TickRecord] = field(default_factory=dict)
    _insert_times: dict[str, float] = field(default_factory=dict)

    async def try_insert(self, record: TickRecord) -> bool:
        existing = self._records.get(record.tick_id)
        if existing is not None and existing.status == TickStatus.COMPLETED:
            return False
        self._records[record.tick_id] = record
        self._insert_times[record.tick_id] = time.monotonic()
        return True

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
        existing = self._records[tick_id]
        self._records[tick_id] = replace(
            existing,
            status=status,
            intent_hash=intent_hash or existing.intent_hash,
            ops_hash=ops_hash or existing.ops_hash,
            duration_ms=duration_ms,
            error_message=error_message,
        )

    async def recover_stale_started(self, stale_threshold_sec: int = 300) -> int:
        now = time.monotonic()
        count = 0
        for tick_id, record in list(self._records.items()):
            if record.status == TickStatus.STARTED:
                inserted_at = self._insert_times.get(tick_id, now)
                if now - inserted_at >= stale_threshold_sec:
                    self._records[tick_id] = replace(
                        record,
                        status=TickStatus.FAILED,
                        error_message="recovered_stale_started",
                    )
                    count += 1
        return count

    def get(self, tick_id: str) -> TickRecord | None:
        return self._records.get(tick_id)


class FailingTickRepo:
    """Raises on all operations."""

    async def try_insert(self, record: TickRecord) -> bool:
        del record
        raise RuntimeError("tick repo failure")

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
        del tick_id, status, intent_hash, ops_hash, duration_ms, error_message
        raise RuntimeError("tick repo failure")

    async def recover_stale_started(self, stale_threshold_sec: int = 300) -> int:
        del stale_threshold_sec
        raise RuntimeError("tick repo failure")
