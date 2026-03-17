"""Fake bot config snapshot repository for tests."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class FakeBotConfigSnapshotRepository:
    """In-memory config snapshot repository for test assertions."""

    snapshots: list[tuple[str, str]] = field(default_factory=list)

    async def record_snapshot(self, config_hash: str, config_json: str) -> None:
        self.snapshots.append((config_hash, config_json))

    async def get_latest(self) -> tuple[str, str] | None:
        return self.snapshots[-1] if self.snapshots else None
